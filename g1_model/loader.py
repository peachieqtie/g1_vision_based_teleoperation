"""The dataset loader: episodes on disk -> (observation window, action chunk).

WHAT THIS IS FOR
----------------
All three policies - BC, ACT and ACT-LSTM - read the same samples through this
one class, because a comparison in which the conditions saw different data is
not a comparison. BC is the K=1 case of the same chunk, not a second loader.

EVERYTHING IS IN MEMORY
-----------------------
40 episodes is 30,256 ticks: ~5.7 MB of state and ~2.7 MB of action in float32.
`Episode.load()` re-opens and decompresses a whole .npz per call (g1_data/
dataset.py), which is fine for scoring a dataset once and absurd for chunk
access - every tick is read by up to K samples as a member of some action chunk
and by W_o samples as a member of some observation window, so a per-`__getitem__`
load would decompress each episode thousands of times per epoch. Everything is
read once at construction and indexed from RAM afterwards.

NORMALIZATION IS APPLIED ONCE, AT LOAD
--------------------------------------
Vectorized over whole episodes at construction, not per `__getitem__`. Same
reason: a tick appears in ~K chunks, so per-sample normalization repeats the
identical arithmetic K times. It is stored float32 - what a torch model consumes
anyway - which costs about 6e-8 relative precision against the float64 that
`NormStats.normalize` returns, far below the 1e-3-ish scale of anything the
policy is asked to reproduce.

The loader NEVER fits normalization and never modifies it. It takes `NormStats`
objects that were fitted elsewhere, from the training split only, and it checks
their seed provenance (`assert_norm_stats_match`) so that a normalizer belonging
to another split or another dataset raises instead of silently shifting every
input by a constant.

WHAT THE LOADER DELIBERATELY DOES NOT TOUCH
-------------------------------------------
  `gait_phase`    EXCLUDED from the state by the 2026-09-11 schema freeze. It is
                  a clock, and handing every policy a clock gives BC the temporal
                  capability ACT-LSTM is meant to supply, which collapses the
                  RQ2/RQ3 gap this thesis is measuring. It is not read here.
  `phase_labels`  the failure taxonomy, never a model input (spec.py). Exposed
                  per episode for analysis; never part of a sample.
  velocity ranges nothing here hardcodes one. `spec.VELOCITY_CLIP` is the only
                  definition and D15 records that it is still unsettled.
  `ctrl`          never indexed. Routing an action back to the model goes through
                  `spec.upper_ctrl_from_action`, in the deployment harness.

The action DIMENSION mask (`spec.ACTION_MASK`, 16 of 22 trainable) is exposed as
an attribute rather than copied into every sample: it is constant across the
dataset, and a per-sample copy invites a training loop to build its own.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from g1_data import dataset as DS
from g1_data import recorder as REC
from g1_data import spec


class LoaderError(AssertionError):
    """A dataset that would train a policy on something other than it claims."""


def _rel(path: str) -> str:
    """`path` relative to the repo, or the path itself when that is impossible.

    `os.path.relpath` RAISES across Windows drive letters, so a loader pointed at
    a directory on another drive - a scratch directory, an external disk holding
    a backup of the collection - died while formatting a message rather than
    while doing anything. Cosmetic input, real crash.
    """
    try:
        return os.path.relpath(path, DS.ROOT)
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class TrackingPolicy:
    """What to do about ticks whose arm command was not backed by a live frame.

    O28: a tracking dropout leaves the arms holding their last pose while the
    recorded action keeps claiming to be a command. Such a tick is not a
    demonstration of anything - the operator was not driving - but it is also not
    corrupt, and on the 40 scripted episodes there are none of them at all.

    BOTH POLICIES ARE OFF BY DEFAULT, so today's behaviour is unchanged and
    turning one on is a decision someone made. They are independent:

      `exclude_overlapping_chunks`  drop the SAMPLE if its action chunk covers a
                                    degraded tick. Per-sample surgery: it keeps
                                    the episode and removes the windows that
                                    would train on a held pose.
      `max_degraded_fraction`       drop the EPISODE if more than this fraction
                                    of its ticks are degraded. Per-episode: an
                                    episode that lost tracking throughout is a
                                    recording of an occlusion, not of the task,
                                    and excluding its chunks one at a time would
                                    leave a shredded trajectory behind.

    The chunk test covers the ACTION chunk, not the observation window: the
    action is what a degraded tick misrepresents. A degraded tick in the
    observation window is still a true record of where the robot was.
    """

    exclude_overlapping_chunks: bool = False
    max_degraded_fraction: Optional[float] = None

    def __post_init__(self):
        f = self.max_degraded_fraction
        if f is not None and not (0.0 <= float(f) <= 1.0):
            raise LoaderError("max_degraded_fraction must be a fraction in "
                              "[0, 1] or None, got %r" % (f,))

    @property
    def active(self) -> bool:
        return bool(self.exclude_overlapping_chunks
                    or self.max_degraded_fraction is not None)


@dataclass
class LoaderConfig:
    """Everything about a `ChunkDataset` that is not the data itself.

    `chunk_size` and `obs_window` have NO DEFAULTS. `obs_window` especially:
    it is the hyperparameter that can invalidate the whole ACT vs ACT-LSTM
    comparison, because an observation window long enough to carry the task's
    temporal structure hands BC the capability the LSTM is supposed to supply.
    Inheriting it silently from a default is exactly how that happens without
    anyone choosing it, so every call site states it.
    """

    chunk_size: int
    obs_window: int
    tracking: TrackingPolicy = field(default_factory=TrackingPolicy)

    def __post_init__(self):
        if int(self.chunk_size) < 1:
            raise LoaderError("chunk_size must be >= 1, got %r" % (self.chunk_size,))
        if int(self.obs_window) < 1:
            raise LoaderError("obs_window must be >= 1, got %r" % (self.obs_window,))
        self.chunk_size = int(self.chunk_size)
        self.obs_window = int(self.obs_window)


class ChunkDataset(Dataset):
    """Observation windows and action chunks, from staged episode files.

    A sample is `(episode, start_tick)`: the observation window ENDING at
    `start_tick` and the action chunk BEGINNING at it. That alignment is the
    timing convention in spec.py - `action[t]` is the command applied FROM
    `state[t]` - and it is the one thing here that must not be adjusted to make a
    shape work out.

    THE SAMPLE INDEX IS EXPLICIT AND DETERMINISTIC. It is built once, at
    construction, as a plain `(episode_index, start_tick)` table, and
    `__getitem__` contains no RNG: which sample comes next is the DataLoader's
    business. A deterministic index is what makes "overfit 10 samples and watch
    the loss hit zero" reproducible, and that run is the first thing that will be
    asked of the model code.
    """

    def __init__(self, episodes: Sequence[DS.Episode], cfg: LoaderConfig,
                 state_norm: spec.NormStats, action_norm: spec.NormStats,
                 norm_meta: Optional[dict],
                 norm_fit_seeds: Optional[Sequence[int]] = None,
                 where: str = ""):
        """Prefer `ChunkDataset.from_directory`; this takes episodes already scanned.

        `norm_meta` is REQUIRED and may be explicitly None. Passing the metadata
        makes the seed provenance check run (B8/A3); passing None skips it and is
        what tests with identity statistics do. There is no default, because
        "nobody passed it" and "somebody decided it did not apply" must not look
        the same at the call site.

        `norm_fit_seeds` is the episode set the statistics are EXPECTED to have
        been fitted on. Omit it for a training loader, where that set is this
        dataset. A VALIDATION loader must pass the TRAINING seeds: its own
        episodes are deliberately not the ones the normalizer saw, so checking
        the stats against itself would fire on a correct setup.
        """
        super().__init__()
        t0 = time.perf_counter()
        self.cfg = cfg
        self.where = where
        for st, kind in ((state_norm, "state"), (action_norm, "action")):
            if not isinstance(st, spec.NormStats) or st.kind != kind:
                raise LoaderError(
                    "expected a spec.NormStats of kind %r, got %r. The loader "
                    "never fits statistics and never loads them itself - they "
                    "are fitted from the training split and handed in."
                    % (kind, st))
        self.state_norm, self.action_norm = state_norm, action_norm
        self.norm_meta = norm_meta

        episodes = sorted(episodes, key=lambda e: int(e.seed))
        if not episodes:
            raise LoaderError("no episodes to load%s"
                              % ((" in " + where) if where else ""))

        # B8: the three dataset refusals still apply. They are cheap and they
        # are the ones that make a result meaningless rather than wrong.
        DS.assert_uniform(episodes)
        DS.assert_no_leak(episodes)

        # A3: a normalizer from another split loads without complaint and shifts
        # every input by a constant. This is the only place that can notice.
        if norm_meta is not None:
            DS.assert_norm_stats_match(
                norm_meta,
                list(norm_fit_seeds) if norm_fit_seeds is not None else episodes,
                where=where or "norm stats")

        # ─── B1: read everything, once ───────────────────────────────────────
        self.seeds: List[int] = []
        self.paths: List[str] = []
        self.states: List[np.ndarray] = []       # normalized, float32
        self.actions: List[np.ndarray] = []      # normalized, float32
        self.phase_labels: List[np.ndarray] = []  # taxonomy only, never a sample
        self.tracking: List[np.ndarray] = []
        self.tracking_is_measured: List[bool] = []
        self.lengths: List[int] = []
        dropped_episodes: List[Tuple[int, float]] = []

        for e in episodes:
            arrays, meta = e.load()              # raises on a length mismatch (A1)
            n = int(np.asarray(arrays["states"]).shape[0])
            ok = REC.tracking_ok_of(arrays, n_ticks=n)   # raises on a bad length (A2)
            degraded = float((~ok).sum()) / float(n) if n else 0.0
            lim = cfg.tracking.max_degraded_fraction
            if lim is not None and degraded > float(lim):
                dropped_episodes.append((int(e.seed), degraded))
                continue
            self.seeds.append(int(e.seed))
            self.paths.append(e.path)
            self.states.append(
                self.state_norm.normalize(
                    np.asarray(arrays["states"], dtype=np.float64)
                ).astype(np.float32))
            self.actions.append(
                self.action_norm.normalize(
                    np.asarray(arrays["actions"], dtype=np.float64)
                ).astype(np.float32))
            self.phase_labels.append(np.asarray(arrays["phase_labels"],
                                                dtype=np.int8))
            self.tracking.append(ok)
            self.tracking_is_measured.append(REC.tracking_measured(arrays))
            self.lengths.append(n)

        self.dropped_episodes = dropped_episodes
        if not self.lengths:
            raise LoaderError(
                "every episode was dropped by the tracking policy "
                "(max_degraded_fraction=%r). %d episode(s) were considered."
                % (cfg.tracking.max_degraded_fraction, len(episodes)))

        # ─── B2: the sample index ────────────────────────────────────────────
        self.index, self.excluded_samples = self._build_index()
        self.build_seconds = time.perf_counter() - t0

    # ---- construction helpers -------------------------------------------
    def _build_index(self) -> Tuple[np.ndarray, int]:
        """Every (episode, start tick) pair, in a fixed order.

        Every tick is a valid start. The chunks near the end of an episode are
        short and get padded (B3) rather than dropped: dropping them would delete
        the final K ticks of every episode, which is where RELEASE and VERIFY
        live - the end of the task would be the part of it with the least data.
        """
        K = self.cfg.chunk_size
        rows, excluded = [], 0
        for ei, n in enumerate(self.lengths):
            ok = self.tracking[ei]
            for t in range(n):
                if self.cfg.tracking.exclude_overlapping_chunks:
                    if not bool(ok[t:t + K].all()):
                        excluded += 1
                        continue
                rows.append((ei, t))
        if not rows:
            raise LoaderError(
                "the tracking policy excluded every sample: %d chunk(s) all "
                "overlap a degraded tick." % excluded)
        return np.asarray(rows, dtype=np.int64), excluded

    @classmethod
    def from_directory(cls, directory: str, cfg: LoaderConfig,
                       state_norm: spec.NormStats, action_norm: spec.NormStats,
                       norm_meta: Optional[dict],
                       seeds: Optional[Sequence[int]] = None,
                       norm_fit_seeds: Optional[Sequence[int]] = None):
        """Scan a staged directory, optionally restricted to `seeds` (a split).

        Goes through `dataset.scan`, so the mixed-SPEC_VERSION and
        mixed-contact-contract refusals run on the directory as a whole - before
        the split narrows it - which is where a half-re-recorded dataset is
        visible and after which it is not.
        """
        eps = DS.scan(directory)
        if seeds is not None:
            want = {int(s) for s in seeds}
            have = {int(e.seed) for e in eps}
            missing = sorted(want - have)
            if missing:
                raise LoaderError(
                    "%d seed(s) in the requested split are not staged in %s: %s. "
                    "A split that silently loses episodes makes the train/val "
                    "proportions something other than what they say."
                    % (len(missing), _rel(directory), missing))
            eps = [e for e in eps if int(e.seed) in want]
        return cls(eps, cfg, state_norm, action_norm, norm_meta,
                   norm_fit_seeds=norm_fit_seeds,
                   where=_rel(directory))

    # ---- the constant dimension mask -------------------------------------
    @property
    def action_dim_mask(self) -> torch.Tensor:
        """(22,) bool, True on the 16 TRAINABLE action dims (spec.ACTION_MASK).

        Constant across the dataset, so it is an attribute rather than a column
        in every sample - and it is exposed at all so that a loss never defines
        its own. Masked dims carry mean 0 / std 1 in `NormStats`, so they pass
        through normalization unchanged and are already exactly what was
        recorded.
        """
        return torch.from_numpy(np.asarray(spec.ACTION_MASK, dtype=bool).copy())

    # ---- the samples ------------------------------------------------------
    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ei, t = (int(v) for v in self.index[int(i)])
        K, W = self.cfg.chunk_size, self.cfg.obs_window
        S, A = self.states[ei], self.actions[ei]
        n = self.lengths[ei]

        # B4. Observation window ENDING at t, left-padded at the episode start.
        obs = np.zeros((W, spec.STATE_DIM), dtype=np.float32)
        obs_mask = np.zeros(W, dtype=bool)
        lo = max(0, t - W + 1)
        got = S[lo:t + 1]
        obs[W - len(got):] = got
        obs_mask[W - len(got):] = True

        # B3. Action chunk BEGINNING at t, right-padded at the episode end.
        # Padded values are ZERO, not NaN and not the last action repeated:
        # finite, so a loss that forgot the mask produces a wrong number rather
        # than a NaN that silently kills every gradient in the batch; and zero
        # rather than a repeat so that a forgotten mask is visibly wrong instead
        # of plausibly wrong.
        act = np.zeros((K, spec.ACTION_DIM), dtype=np.float32)
        act_mask = np.zeros(K, dtype=bool)
        chunk = A[t:min(n, t + K)]
        act[:len(chunk)] = chunk
        act_mask[:len(chunk)] = True

        return dict(
            obs=torch.from_numpy(obs),
            obs_mask=torch.from_numpy(obs_mask),
            action=torch.from_numpy(act),
            action_mask=torch.from_numpy(act_mask),
            episode_index=torch.tensor(ei, dtype=torch.int64),
            start_tick=torch.tensor(t, dtype=torch.int64),
            seed=torch.tensor(self.seeds[ei], dtype=torch.int64),
        )

    # ---- what it loaded ---------------------------------------------------
    def summary(self) -> dict:
        """Everything a training run should print before it starts."""
        deg = [int((~ok).sum()) for ok in self.tracking]
        return dict(
            episodes=len(self.lengths), seeds=list(self.seeds),
            ticks=int(sum(self.lengths)), samples=len(self),
            chunk_size=self.cfg.chunk_size, obs_window=self.cfg.obs_window,
            episode_len=dict(min=int(min(self.lengths)),
                             median=int(np.median(self.lengths)),
                             max=int(max(self.lengths))),
            tracking_measured=int(sum(self.tracking_is_measured)),
            degraded_ticks=int(sum(deg)),
            excluded_samples=int(self.excluded_samples),
            dropped_episodes=list(self.dropped_episodes),
            norm_split=self.state_norm.split,
            norm_source=(self.norm_meta or {}).get("source", "unverified"),
            bytes_in_ram=int(sum(a.nbytes for a in self.states)
                             + sum(a.nbytes for a in self.actions)),
            build_seconds=round(self.build_seconds, 3),
        )


def collate_chunks(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack samples into a batch. Every sample is already exactly K and W_o
    long, so this is a plain stack - the padding happened per sample, where the
    episode boundary is known, rather than per batch, where it is not."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
