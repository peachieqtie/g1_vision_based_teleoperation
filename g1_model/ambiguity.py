"""Neighbour ambiguity: how much of the action the OBSERVATION cannot determine.

WHAT THIS MEASURES, AND WHY THE GATE NEEDS IT
----------------------------------------------
"Train until the loss is near zero" is a valid gate only when the input
determines the output. Here it does not. Two ticks 40 ms apart have nearly
identical 47-D states and genuinely different commanded actions, and no
deterministic function of the state can produce both. A criterion of "near zero"
therefore asks for something the data does not contain, and failing it says
nothing about the model.

So: for every sample, find its NEAREST NEIGHBOUR IN THE SPACE THE MODEL SEES,
and measure how far apart their actions are. That difference is action variation
the observation cannot account for - if the model's own input cannot tell the two
samples apart, no architecture can give them different outputs. The mean of that
difference is the reference a training error is judged against.

MEASURED IN THE MODEL'S OWN SPACE, PER CONFIGURATION
-----------------------------------------------------
The neighbour search runs on the flattened observation window, exactly as
`ChunkDataset.__getitem__` assembles it, padding included. So the reference moves
with `obs_window` - a longer window can separate samples a shorter one confuses,
and the reference falls accordingly - and with `chunk_size`, because predicting
100 actions from one observation is a harder question than predicting 1. A single
global number would silently favour whichever stage happened to match it.

Every gate therefore computes its OWN reference from its OWN loader config, and
reports both numbers and the ratio.

=========================================================================
WHAT THIS IS NOT: LIMITS OF THE ESTIMATOR
=========================================================================
1. IT IS AN EMPIRICAL PROXY, NOT A BOUND. Nothing here proves a model cannot do
   better. Two neighbours differing by d do not force an error of d: the best
   deterministic L1 predictor at a point sits at the conditional median, which
   for a coincident pair costs about d/2 each. The reference is therefore
   LENIENT by roughly a factor of two, and a model beating it has not reached
   any theoretical floor - it has reached the resolution the data supports at
   this density.

2. IT DEPENDS ON DATA DENSITY, so it FALLS AS EPISODES ARE ADDED. More samples
   means closer neighbours means smaller action differences. A reference computed
   on 10 episodes is larger than one computed on 150, and a model that passes
   against the first may fail against the second WITHOUT HAVING CHANGED.

3. IT IS THEREFORE NOT COMPARABLE ACROSS DATASETS OF DIFFERENT SIZE. Citing it
   without saying which dataset and how many episodes produced it is citing a
   number that cannot be checked. `AmbiguityResult.cite()` exists so that the
   provenance travels with the figure; use it rather than quoting `.mean`.

4. IT INHERITS THE DATA'S CHARACTER. Computed on SCRIPTED episodes it describes
   the scripted demonstrator, whose action distribution is smoother and more
   repeatable than piloted teleoperation. It must be recomputed on piloted data
   before any result that rests on it is reported.

5. EUCLIDEAN DISTANCE ON NORMALIZED STATE is the metric, which weights all 47
   dims equally. That is a choice, not a fact about the task - a metric that
   weighted, say, palm position more heavily would find different neighbours.
   It is the same space the MLP's first layer sees, which is the defensible
   reason for it here.

6. ACROSS `obs_window` IT IS CONFOUNDED BY DIMENSIONALITY, and this one bites
   the W_o instrument directly. Growing W_o multiplies the search space by W_o
   while the sample count stays fixed, so neighbours get relatively farther and
   their actions differ more - for reasons that have nothing to do with how much
   the window explains. MEASURED on the scripted episodes, K=1:

       W_o      1        2        4        8       16
       ambig  .002077  .001975  .001982  .002067  .002131     (32 episodes)
       nn-d    .0130    .0191    .0294    .0461    .0681

   Ambiguity RISES past W_o=2 while the median neighbour distance grows 5.2x
   over a 16x larger space. A longer window cannot destroy information, so the
   rise is the estimator, not the data. Consequences:

     * a FALLING region of the curve is informative - the window is explaining
       more than the sparsity costs;
     * a RISING region means only that the dataset cannot populate a space that
       big. That is still useful - it says data density, not architecture, caps
       the usable W_o - but it is NOT evidence that a longer window is worse;
     * the minimum is NOT automatically the right W_o.

   `neighbour_distance` is reported alongside every point precisely so this
   confound is visible rather than inferred, and the same three-size sweep above
   is reproducible with `ambiguity_curve`. A density-matched estimator - compare
   at equal neighbour distance rather than equal sample count - is the fix, and
   is not built here.
=========================================================================
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from g1_data import spec
from g1_model.loader import ChunkDataset


class AmbiguityError(AssertionError):
    """A reference that would be quoted without the provenance to check it."""


# ─── units: the gate compares like with like, or refuses ─────────────────────
#: The reference's unit: mean |action difference| over trainable dims and real
#: timesteps, in NORMALIZED action units - exactly what `train.masked_l1`
#: computes. A model's reconstruction error is in these units.
RECON_UNITS = "masked_l1_reconstruction"
#: A model's TOTAL training loss. For BC it happens to equal the reconstruction;
#: for ACT it adds beta*KL, which is in nats times a weight, not action units.
TOTAL_LOSS_UNITS = "total_training_loss"

# ─── how the number was MEASURED: the second axis the gate checks ─────────────
#: The ONLY measurement the gate accepts: the deployed function, scored once on
#: fixed weights - eval mode (no dropout), `model(obs)` alone (the target is never
#: passed, so a CVAE runs its prior, z = 0, and its encoder cannot run), no
#: gradient. Produced by `train.score_deployment` and by nothing else.
MEASURED_AT_DEPLOYMENT = "deployment: eval mode, model(obs) only, fixed weights"
#: The loop's running mean of per-step TRAINING losses over a window: dropout on,
#: ACT's z sampled from a posterior that read the target chunk, weights moving.
#: Same units as the reference - which is exactly why the units check alone let it
#: through. It scored converged ACT 0.050886 (FAIL, 1.140) where the deployed
#: function scores 0.041081 (PASS, 0.920): the whole gap was dropout.
MEASURED_TRAIN_MODE_WINDOW = "train mode: running mean over a window of optimizer steps"


class UnitsError(AmbiguityError):
    """A quantity whose units differ from the reference's (TR29)."""


class MeasurementError(AmbiguityError):
    """A quantity in the right units, measured the wrong way (the eval-mode fix,
    2026-09-22): not the deployed function on fixed weights."""


@dataclass(frozen=True)
class Quantity:
    """A number that carries what it measures.

    TR29, made structural. The Stage-4 ACT gate compared TOTAL loss (0.289,
    reconstruction plus beta*KL) against the ambiguity reference, a
    reconstruction quantity, and reported a ratio of 6.47 where the right
    comparison gives 3.09. It was silent, and it was right by accident twice -
    for BC and chunked BC, which have no KL term, and for ACT at lr 1e-3, whose
    KL had collapsed to zero. A bare float cannot say which it is, so the gate
    accepts only a `Quantity`, labelled where it is COMPUTED (`train.train`), and
    refuses one whose units are not the reference's.

    `measured` is the second axis, added after the same family struck again:
    units right, measurement wrong (a train-mode window mean, dropout on). The gate
    refuses anything but `MEASURED_AT_DEPLOYMENT`. None means "not stated", and is
    refused too - a number that cannot say how it was taken is not gated.
    """

    value: float
    units: str
    measured: Optional[str] = None


@dataclass
class AmbiguityResult:
    """The reference, and everything needed to cite it honestly."""

    mean: float                  # element-weighted; the gate reference
    median: float                # per-sample, unweighted
    p90: float
    samples: int
    elements: int
    obs_window: int
    chunk_size: int
    episodes: int
    seeds: List[int]
    sources: List[str]
    neighbour_distance: Dict[str, float]
    exclude_same_episode: bool
    subsampled_to: Optional[int] = None
    metric: str = "euclidean on normalized observation window"
    #: What `mean` measures. A gate refuses any train error not in these units.
    units: str = RECON_UNITS

    def cite(self) -> str:
        """The figure WITH its provenance. Quote this, never `.mean` alone."""
        return ("neighbour ambiguity %.6f (normalized L1, %d trainable dims; "
                "W_o=%d, K=%d; %d episodes / %d samples from %s%s)"
                % (self.mean, int(spec.ACTION_MASK.sum()), self.obs_window,
                   self.chunk_size, self.episodes, self.samples,
                   "+".join(self.sources) or "unknown",
                   "; subsampled to %d" % self.subsampled_to
                   if self.subsampled_to else ""))

    def as_metadata(self) -> dict:
        d = asdict(self)
        d["citation"] = self.cite()
        return d


# ─── materializing the model's view ───────────────────────────────────────────
def observation_matrix(ds: ChunkDataset) -> np.ndarray:
    """(N, W_o * 47): every sample's observation window, flattened.

    Built vectorized per episode rather than by calling `ds[i]` N times, which
    matters for the W_o sweep. `test_observation_matrix_matches_getitem` asserts
    the two agree EXACTLY - a fast path that quietly disagrees with the loader
    would make the reference describe a space no model ever sees.
    """
    W = ds.cfg.obs_window
    out = []
    for S in ds.states:
        n = S.shape[0]
        padded = np.concatenate(
            [np.zeros((W - 1, spec.STATE_DIM), dtype=np.float32), S], axis=0)
        win = np.lib.stride_tricks.sliding_window_view(
            padded, (W, spec.STATE_DIM))          # (n, 1, W, 47)
        out.append(win.reshape(n, W * spec.STATE_DIM))
    return np.ascontiguousarray(np.concatenate(out, axis=0))


def action_chunks(ds: ChunkDataset):
    """((N, K, 22) chunks, (N, K) bool padding mask), as `__getitem__` builds them."""
    K = ds.cfg.chunk_size
    chunks, masks = [], []
    for A in ds.actions:
        n = A.shape[0]
        padded = np.concatenate(
            [A, np.zeros((K - 1, spec.ACTION_DIM), dtype=np.float32)], axis=0)
        win = np.lib.stride_tricks.sliding_window_view(
            padded, (K, spec.ACTION_DIM))         # (n, 1, K, 22)
        chunks.append(win.reshape(n, K, spec.ACTION_DIM))
        t = np.arange(n)[:, None] + np.arange(K)[None, :]
        masks.append(t < n)
    return (np.ascontiguousarray(np.concatenate(chunks, axis=0)),
            np.ascontiguousarray(np.concatenate(masks, axis=0)))


# ─── the measurement ──────────────────────────────────────────────────────────
def neighbour_ambiguity(ds: ChunkDataset,
                        device: Optional[torch.device] = None,
                        batch: int = 256,
                        max_samples: Optional[int] = None,
                        seed: int = 0,
                        exclude_same_episode: bool = False) -> AmbiguityResult:
    """The gate reference for THIS dataset under THIS loader configuration.

    For each sample i, find the nearest OTHER sample j in observation space, then
    average |action_i - action_j| over the elements the loss would score: real
    timesteps in BOTH chunks, and trainable dims only. The reduction is
    element-weighted, exactly like `train.masked_l1`, so the two numbers are in
    the same units and dividing one by the other is meaningful.

    `exclude_same_episode` defaults FALSE, deliberately. The nearest neighbour of
    a sample is very often the adjacent tick of its own episode, and excluding it
    would be assuming the answer: if two consecutive observations really are
    nearly identical and their actions really are nearly identical, then the
    input DOES determine the output there, and that is a fact about the data, not
    an artifact to be removed. The option exists because it is the first question
    anyone asks, and it should be answerable by measurement rather than argument.

    `max_samples` subsamples for speed; it makes the reference LARGER (fewer
    candidates means more distant neighbours), so it is conservative for a
    density sweep and is recorded in the result either way.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))
    X = observation_matrix(ds)
    A, M = action_chunks(ds)
    N = X.shape[0]
    if N < 2:
        raise AmbiguityError(
            "neighbour ambiguity needs at least 2 samples, got %d. With one "
            "sample there is no neighbour and the reference is undefined." % N)

    epi = np.concatenate([np.full(n, i, dtype=np.int64)
                          for i, n in enumerate(ds.lengths)])

    sub = None
    if max_samples is not None and max_samples < N:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(N, size=int(max_samples), replace=False))
        X, A, M, epi = X[keep], A[keep], M[keep], epi[keep]
        N, sub = X.shape[0], int(max_samples)

    Xg = torch.from_numpy(X).to(device)
    Ag = torch.from_numpy(A).to(device)
    Mg = torch.from_numpy(M).to(device)
    Eg = torch.from_numpy(epi).to(device)
    dim = torch.from_numpy(
        np.asarray(spec.ACTION_MASK, dtype=bool).copy()).to(device)
    n_dims = int(dim.sum())

    num_total, den_total = 0.0, 0
    per_sample, dists = [], []
    for i in range(0, N, batch):
        hi = min(i + batch, N)
        d = torch.cdist(Xg[i:hi], Xg)                       # (b, N)
        rows = torch.arange(hi - i, device=device)
        d[rows, torch.arange(i, hi, device=device)] = float("inf")   # not itself
        if exclude_same_episode:
            d[Eg[i:hi].unsqueeze(1) == Eg.unsqueeze(0)] = float("inf")
        dist, j = d.min(dim=1)
        dists.append(dist.cpu().numpy())

        valid = (Mg[i:hi] & Mg[j]).unsqueeze(-1) & dim.view(1, 1, -1)
        diff = (Ag[i:hi] - Ag[j]).abs() * valid
        num = diff.sum(dim=(1, 2))
        den = valid.sum(dim=(1, 2))
        num_total += float(num.sum())
        den_total += int(den.sum())
        per_sample.append((num / den.clamp(min=1)).cpu().numpy())

    per_sample = np.concatenate(per_sample)
    dists = np.concatenate(dists)
    return AmbiguityResult(
        mean=num_total / max(den_total, 1),
        median=float(np.median(per_sample)),
        p90=float(np.percentile(per_sample, 90)),
        samples=int(N), elements=int(den_total),
        obs_window=ds.cfg.obs_window, chunk_size=ds.cfg.chunk_size,
        episodes=len(ds.lengths), seeds=list(ds.seeds),
        sources=sorted(set(ds.sources)),
        neighbour_distance=dict(
            p50=float(np.percentile(dists, 50)),
            p90=float(np.percentile(dists, 90)),
            max=float(dists.max())),
        exclude_same_episode=bool(exclude_same_episode),
        subsampled_to=sub, metric="euclidean on normalized observation window "
                                  "(%d dims)" % (ds.cfg.obs_window * spec.STATE_DIM))


# ─── the gate verdict ─────────────────────────────────────────────────────────
@dataclass
class GateVerdict:
    passed: bool
    train_error: float
    reference: float
    ratio: float
    detail: AmbiguityResult

    def render(self) -> str:
        return (
            "  train error       %.6f\n"
            "  ambiguity ref     %.6f   (W_o=%d, K=%d, %d episodes)\n"
            "  ratio             %.4f   (< 1.0 passes)\n"
            "  VERDICT: %s\n"
            "  reference: %s"
            % (self.train_error, self.reference, self.detail.obs_window,
               self.detail.chunk_size, self.detail.episodes, self.ratio,
               "PASS - training error is below what the observation can "
               "determine." if self.passed else
               "FAIL - training error exceeds the neighbour-ambiguity "
               "reference. Something is wrong with the model or the plumbing; "
               "do NOT add capacity or train longer to force it.",
               self.detail.cite()))


def gate(train_error: Quantity, reference: AmbiguityResult) -> GateVerdict:
    """A stage passes when its training error on 10 episodes falls BELOW the
    neighbour-ambiguity reference computed for its own loader configuration.

    Replaces "drive the loss to near zero", which was measured against nothing
    and which the Stage 2 BC gate failed while being at the resolution the input
    supports (NOTES.md, 2026-09-21). Both numbers and the ratio are reported at
    every gate, always: a verdict without them is not checkable.

    `train_error` MUST be a `Quantity` in the reference's units. A bare float is
    refused, because a bare float is how TR29 happened: nothing about 0.289
    said it was reconstruction plus beta*KL.
    """
    if not isinstance(train_error, Quantity):
        raise UnitsError(
            "gate() needs a Quantity, not a bare %s (%r). A bare number cannot say "
            "what it measures; that is how ACT's total loss was once scored against "
            "a reconstruction reference (TR29)." % (type(train_error).__name__,
                                                   train_error))
    if train_error.units != reference.units:
        raise UnitsError(
            "gate() refuses to compare %r (%s) against a reference in %s. Score the "
            "model's reconstruction error - train() reports it as "
            "result['quantities']['deployment_recon'] - not its total loss (TR29)."
            % (train_error.value, train_error.units, reference.units))
    if train_error.measured != MEASURED_AT_DEPLOYMENT:
        raise MeasurementError(
            "gate() refuses %r measured as %r. The reference describes the data, "
            "with no dropout and no sampling; the policy must be scored the same way - "
            "the deployed function on fixed weights, `train.score_deployment`, which "
            "train() reports as result['quantities']['deployment_recon']. A train-mode "
            "window mean failed converged ACT (1.140) that deploys at 0.920."
            % (train_error.value, train_error.measured))
    train_error = float(train_error.value)
    ref = float(reference.mean)
    if ref <= 0:
        raise AmbiguityError(
            "the ambiguity reference is %r, so every error 'passes'. That means "
            "every observation determines its action exactly, which on real "
            "recorded data means the measurement is broken." % ref)
    return GateVerdict(passed=bool(train_error < ref), train_error=float(train_error),
                       reference=ref, ratio=float(train_error) / ref,
                       detail=reference)


# ─── Part D: the W_o selection instrument ─────────────────────────────────────
def ambiguity_curve(directory: str, seeds: Sequence[int],
                    windows: Sequence[int], chunk_size: int,
                    state_norm, action_norm, norm_meta=None,
                    norm_fit_seeds=None, device=None,
                    max_samples: Optional[int] = None,
                    tracking=None) -> List[AmbiguityResult]:
    """Neighbour ambiguity as a function of `obs_window`. NO MODEL IS INVOLVED.

    This is the instrument for choosing W_o before either ACT variant trains.
    W_o has to be identical across ACT and ACT-LSTM or the comparison is
    confounded, and it has to be justifiable in Chapter 3 by something other than
    "we swept it and this won" - a sweep chooses the window that suits the model,
    which is exactly the confound. This measures a property of the DATA: how much
    of the action's unpredictability a longer window removes. Being model-free,
    it cannot be confounded by architecture.

    READ THE FALLING REGION, NOT THE MINIMUM. Limit 6 in the module docstring:
    the estimator is confounded by dimensionality across W_o, so the curve turns
    upward once the dataset can no longer populate the larger space. Where it
    FALLS, the window is explaining more than the added sparsity costs, and that
    is the informative part. Where it RISES, the reading is "data density caps
    the usable window here", not "a longer window is worse". Check
    `neighbour_distance` at every point: if it is growing fast, the rise is the
    estimator.

    See the module docstring for the rest of the limits - in particular that the
    curve is a property of the DATASET IT WAS RUN ON, so a curve from scripted
    episodes cannot choose W_o for piloted ones.
    """
    from g1_model.loader import ChunkDataset, LoaderConfig, TrackingPolicy
    if tracking is None:
        raise AmbiguityError(
            "state the tracking policy explicitly (A2); pass TrackingPolicy() "
            "for today's behaviour rather than letting it default")
    out = []
    for w in windows:
        ds = ChunkDataset.from_directory(
            directory, LoaderConfig(chunk_size=int(chunk_size), obs_window=int(w),
                                    tracking=tracking),
            state_norm, action_norm, norm_meta, seeds=seeds,
            norm_fit_seeds=norm_fit_seeds)
        out.append(neighbour_ambiguity(ds, device=device,
                                       max_samples=max_samples))
    return out
