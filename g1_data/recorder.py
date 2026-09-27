"""Episode recorder: one .npz per demonstration, built through `spec.SpecLayout`.

OBSERVATION ONLY
----------------
The recorder never writes to the model, never changes a control, and never
touches `scripted_demo.py`. It attaches by wrapping `mujoco.mj_step` and reading
the demonstrator's own frame locals - the way the 2026-09-10 constant-dimension
audit was instrumented - because the gates are only meaningful if recording a
run and not recording it produce the same run. `tools/record_episodes.py
--invariance` measures exactly that.

TIMING (spec.py, TIMING CONVENTION)
-----------------------------------
A tick is taken when the demonstrator's loop index is a multiple of
`spec.PHYSICS_STEPS_PER_TICK` (20 steps = 25 Hz, D4), at `mj_step` ENTRY. That
is the one instant where `data.ctrl` holds the command about to be applied and
the state is the one it is applied from, so `state[t]` and `action[t]` are the
pair "what the demonstrator saw, what it did about it". Both come from
`SpecLayout.build`; nothing here hand-assembles a vector.

WHAT IS LOGGED THAT THE 47-D STATE DOES NOT CARRY, AND WHY
----------------------------------------------------------
RAW `qpos` and `qvel` per tick. They are INSURANCE, not a policy input, and they
are the reason the state design is the one dataset decision that cannot be
regretted: almost anything one later wishes were in the 47-D vector is derivable
from raw state plus an observation window (velocity is a finite difference), but
LEG JOINT ANGLES are not - proposal 3.4 excludes them, and once 150 episodes are
collected without them they are gone.

`gait_phase` is logged as metadata and is NEVER a policy input: it is a clock,
and handing every policy a clock gives BC the temporal capability ACT-LSTM is
meant to supply (CLAUDE.md section 8, schema freeze).

`contact_contract` records whether the D18 hand/pickup-platform exclusion was
active, read from the compiled model rather than from a flag. If collection and
evaluation disagree on it, every policy is evaluated in physics it never
learned, and nothing else in the file would reveal that.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Optional, Tuple

import numpy as np
import mujoco

from g1_data import spec
from g1_data.paths import repo_relpath
from g1_data.reset import state_fingerprint
from g1_teleop.contact_contract import contract_of
from g1_teleop.indices import ModelIndex

TICK = spec.PHYSICS_STEPS_PER_TICK
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── the source registry ──────────────────────────────────────────────────────
# `meta["source"]` is written by the recorder, not by an operator flag, so it is
# the only provenance field that cannot be got wrong by forgetting an argument.
# Every entry point that can produce an episode is registered here with the
# label the ledger should carry and whether its episodes are REAL demonstrations.
#
# `real=False` is not a judgement about quality - it means no human produced the
# trajectory, so the episode is not a demonstration of the task being learned and
# must never reach a training split. The scripted demonstrator and the headless
# synthetic teleop fixture are both in that class.
#
# An unregistered source is a new entry point that nobody declared. Staging
# REFUSES it rather than guessing: guessing wrong routes non-demonstration data
# into training, which is the failure this registry exists to prevent.
SOURCE_SCRIPTED = "scripted_demo.run_episode"
SOURCE_TELEOP_FIXTURE = "run_integrated_combined --synthetic grasp"

SOURCES = {
    SOURCE_SCRIPTED:       dict(label="scripted", real=False),
    SOURCE_TELEOP_FIXTURE: dict(label="teleop-fixture", real=False),
}


# ─── the recording namespaces ─────────────────────────────────────────────────
# A namespace is a directory AND the ledger that belongs to it. They are one
# thing, never two arguments, because `EpisodeLedger.pending` skips any seed
# that already has an ACCEPT row: point the real collection at the scripted
# ledger and it silently skips seeds 0-44 - 40 of the 200 partitioned training
# seeds - with no warning, because skipping accepted seeds is the intended
# resume behaviour. A separate directory alone does NOT fix that; only a
# separate ledger does. Measured 2026-09-16, hence this registry.
LEDGER_NAME = "ledger.jsonl"
NS_SCRIPTED = os.path.join(ROOT, "recordings", "episodes")
NS_COLLECTION = os.path.join(ROOT, "recordings", "demonstrations")

NAMESPACES = {
    NS_SCRIPTED:   dict(real=False, what="scripted-demonstrator episodes"),
    NS_COLLECTION: dict(real=True,  what="real teleoperated demonstrations"),
}


def ledger_for(directory: str) -> str:
    """The ledger that BELONGS to an output directory. Never a free argument."""
    return os.path.join(os.path.abspath(directory), LEDGER_NAME)


def namespace_info(directory: str) -> Optional[dict]:
    """The registry row for a directory, or None if it is not a declared
    namespace. Ad-hoc directories are allowed - staging routes on source, so an
    unregistered directory cannot smuggle anything into a training split."""
    d = os.path.abspath(directory)
    for path, info in NAMESPACES.items():
        if os.path.abspath(path) == d:
            return info
    return None


class NamespaceMismatch(ValueError):
    """A source writing into a namespace declared for the other kind."""


def assert_namespace(directory: str, source: str, where: str = "") -> None:
    """Refuse a writer whose source does not belong in this namespace."""
    ns = namespace_info(directory)
    if ns is None:
        return
    src = source_info(source, where)
    if bool(ns["real"]) != bool(src["real"]):
        raise NamespaceMismatch(
            "refusing to write %s into %s.\n"
            "  directory holds : %s\n"
            "  this source is  : %r (real demonstrations: %s)\n"
            "  The two namespaces are separate so that a real collection cannot "
            "inherit the scripted ledger, which would silently skip every seed "
            "already accepted there. Write to %s instead."
            % ("a real demonstration" if src["real"] else "scripted output",
               repo_relpath(directory), ns["what"], source, src["real"],
               repo_relpath(NS_COLLECTION if src["real"] else NS_SCRIPTED)))


class UnknownSource(ValueError):
    """An episode whose `meta["source"]` is not in `SOURCES`."""


def source_info(source: Optional[str], where: str = "") -> dict:
    """The registry row for `source`, or raise. Never returns a default."""
    if source in SOURCES:
        return SOURCES[source]
    raise UnknownSource(
        "unregistered episode source %r%s.\n"
        "  registered: %s\n"
        "  An unrecognised source is an entry point that was never declared, "
        "so nothing here knows whether its episodes are real demonstrations. "
        "Add it to g1_data.recorder.SOURCES with the correct `real` flag "
        "rather than letting it be routed on a guess."
        % (source, " in " + where if where else "",
           ", ".join(sorted(map(repr, SOURCES)))))


def label_of(source: str, where: str = "") -> str:
    """The ledger label, DERIVED from the recorder-set source (A2)."""
    return source_info(source, where)["label"]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=ROOT, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:                                       # noqa: BLE001
        return "unknown"


class EpisodeBuffer:
    """Per-tick rows held in memory; written once, at the end.

    A rejected episode must cost nothing, so nothing reaches the disk until
    `save` is called - which the operator only reaches after ACCEPT.
    """

    def __init__(self, model, ix: Optional[ModelIndex] = None,
                 sp: Optional[spec.SpecLayout] = None,
                 measures_tracking: bool = False):
        self.ix = ix or ModelIndex.resolve(model)
        self.sp = sp or spec.SpecLayout.resolve(model, self.ix)
        self.states, self.actions, self.phases = [], [], []
        self.gait, self.qpos, self.qvel, self.ticks = [], [], [], []
        # O28. Per tick: was the arm command backed by a LIVE tracked frame?
        # An auxiliary array, not a state or action dim - the 47/22 layout and
        # SPEC_VERSION are untouched (see `tracking_ok_of`). Written ONLY when
        # the source has a tracking path: a stored array means MEASURED
        # (`tracking_measured`), and the scripted demonstrator measures nothing.
        self.measures_tracking = bool(measures_tracking)
        self.tracking = []
        self.t_wall0 = time.perf_counter()

    def tick(self, model, data, act, cmd, phase_label: int, gait_counter: int,
             step_index: int, dt: float, gait_period: float,
             tracking_ok: bool = True) -> None:
        """One recorded tick. Call at `mj_step` entry, never after it."""
        s, a = self.sp.build(model, data, self.ix, act=act, cmd=cmd, sync=True)
        # D15, at the moment it would be written. ASSERT, never clip: clipping
        # here would make the recorded action differ from the command that
        # actually drove the robot, so the state/action pair would stop being a
        # record of what happened - and it would hide the one condition worth
        # seeing. The limits are the demonstrator's own `hold_max` /
        # `hold_max_yaw`, so a violation means a NEW command path is writing
        # velocities, which is a recorder-level fact, not a tuning question.
        spec.assert_velocity_within_clip(
            a[spec.VEL_A], where="recorded tick %d" % len(self.states))
        self.states.append(s)
        self.actions.append(a)
        self.phases.append(int(phase_label))
        phz = ((gait_counter * dt) % gait_period) / gait_period
        self.gait.append((np.sin(2 * np.pi * phz), np.cos(2 * np.pi * phz)))
        self.qpos.append(np.array(data.qpos, dtype=np.float64))
        self.qvel.append(np.array(data.qvel, dtype=np.float64))
        self.ticks.append(int(step_index))
        # Defaults True: the scripted demonstrator has no camera and no tracking
        # path, so every one of its ticks is genuinely backed by a real command.
        self.tracking.append(bool(tracking_ok))

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: str, meta: dict) -> str:
        """Write the episode. Float32 arrays, JSON metadata."""
        if not self.states:
            raise ValueError("refusing to write an episode with no ticks")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        m = dict(meta)
        m.setdefault("spec_version", spec.SPEC_VERSION)
        m.setdefault("git_commit", _git_commit())
        m.setdefault("mujoco_version", mujoco.__version__)
        m.setdefault("python_version", sys.version.split()[0])
        m.setdefault("n_ticks", len(self.states))
        arrays = dict(
            states=np.asarray(self.states, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.float32),
            phase_labels=np.asarray(self.phases, dtype=np.int8),
            gait_phase=np.asarray(self.gait, dtype=np.float32),
            qpos=np.asarray(self.qpos, dtype=np.float32),
            qvel=np.asarray(self.qvel, dtype=np.float32),
            step_index=np.asarray(self.ticks, dtype=np.int64),
            meta=np.array(json.dumps(m)))
        if self.measures_tracking:
            arrays["tracking_ok"] = np.asarray(self.tracking, dtype=np.uint8)
        np.savez_compressed(path, **arrays)
        return path


class EpisodeLengthError(AssertionError):
    """Per-tick arrays that do not agree on how many ticks there are."""


def tracking_measured(arrays: dict) -> bool:
    """Was `tracking_ok` MEASURED for this episode, or is it the default?

    `tracking_ok_of` returns an all-True array either way, and the two cases mean
    different things: a measured all-True episode is evidence that tracking held,
    while a defaulted one is only the absence of the field. Nothing downstream
    can tell them apart from the array, so the question is asked here instead of
    being inferred from `.all()` - which would read a real teleop episode that
    happened to hold tracking as though it had never been measured.
    """
    return arrays.get("tracking_ok") is not None


def tracking_ok_of(arrays: dict, n_ticks: Optional[int] = None) -> np.ndarray:
    """Per-tick "was the arm command backed by a live tracked frame" (O28).

    BACKWARD COMPATIBLE BY CONSTRUCTION. `load_episode` builds its dict from
    `z.files`, so an episode written before this array existed simply does not
    have the key - it does not fail to load, and nothing in `assert_uniform`
    (which reads metadata only) or the loader inspects the array set. No
    SPEC_VERSION bump: the 47-D state and 22-D action layout, the masks, the
    clips and the normalization contract are all untouched, and `fit_norm_stats`
    reads only `states` and `actions`.

    A missing array means "recorded before the field existed". It is reported as
    all-True, which is CORRECT for every episode that exists today: all 45 came
    from the scripted demonstrator, which has no camera and no tracking path.
    Do not read that default as a measurement on a teleop episode - a teleop
    episode written after this change always carries the real array. Ask
    `tracking_measured` which case you are in; do not infer it from `.all()`.

    A STORED array of the wrong length RAISES. It used to be returned as it
    stood, at whatever length it had, so a caller masking `states` with it either
    failed far downstream or - the case that motivated this - silently masked the
    wrong ticks, because numpy is happy to index 3 positions of a 706-tick
    episode. There is no length at which a partial tracking record is meaningful.
    """
    got = arrays.get("tracking_ok")
    n = (len(arrays["states"]) if n_ticks is None else int(n_ticks))
    if got is not None:
        got = np.asarray(got, dtype=bool)
        if got.ndim != 1 or got.shape[0] != n:
            raise EpisodeLengthError(
                "tracking_ok has length %s but the episode is %d tick(s) long. "
                "A per-tick record that does not cover every tick cannot be "
                "aligned to one: masking with it would silently mark the wrong "
                "ticks." % (list(got.shape), n))
        return got
    return np.ones(n, dtype=bool)


# ─── the per-tick contract ────────────────────────────────────────────────────
#: Every array in an episode file that carries ONE ROW PER TICK. They are equal
#: in length by construction in `EpisodeBuffer` - one append per array per tick -
#: and that construction was the only thing guaranteeing it. A file written by an
#: older build, a hand-edited file, or an external converter has no such
#: guarantee, and an off-by-one between `states` and `actions` is precisely the
#: corruption that does NOT look like corruption: it silently reindexes the
#: state/action pairing that the whole dataset means (spec.py, TIMING CONVENTION).
#:
#: `tracking_ok` is checked ONLY WHEN PRESENT: the 45 episodes recorded before
#: O28 do not carry it and must keep loading (see `tracking_ok_of`).
PER_TICK_ARRAYS: Tuple[str, ...] = (
    "states", "actions", "phase_labels", "gait_phase", "qpos", "qvel",
    "step_index",
)
OPTIONAL_PER_TICK_ARRAYS: Tuple[str, ...] = ("tracking_ok",)


def assert_tick_lengths(arrays: dict, meta: Optional[dict] = None,
                        where: str = "") -> int:
    """Every per-tick array agrees on axis 0, and `meta["n_ticks"]` agrees too.

    Returns the tick count. Raises `EpisodeLengthError` naming the offending key
    and both lengths - the point is to say WHICH array disagrees, because "the
    arrays are inconsistent" does not tell you whether to re-record or to fix a
    reader.

    `n_ticks` is required rather than optional-when-absent. `load_episode`
    already requires `meta["spec_version"]`, every writer in the repo sets
    `n_ticks` via `EpisodeBuffer.save`, and a per-tick contract with no declared
    length leaves nothing to check the arrays AGAINST: if every array is short by
    the same amount they agree with each other and the episode is still truncated.
    """
    tag = (" in " + where) if where else ""
    if "states" not in arrays:
        raise EpisodeLengthError(
            "episode%s has no `states` array, so it has no ticks to check" % tag)
    n = int(np.asarray(arrays["states"]).shape[0])
    for key in PER_TICK_ARRAYS + OPTIONAL_PER_TICK_ARRAYS:
        if key not in arrays:
            if key in OPTIONAL_PER_TICK_ARRAYS:
                continue
            raise EpisodeLengthError(
                "episode%s is missing the per-tick array %r. The per-tick "
                "arrays are %s; a file without one of them is not an episode "
                "this build can read." % (tag, key, ", ".join(PER_TICK_ARRAYS)))
        got = int(np.asarray(arrays[key]).shape[0])
        if got != n:
            raise EpisodeLengthError(
                "per-tick array length mismatch%s: %r has %d row(s) but "
                "`states` has %d. Every per-tick array must cover the same "
                "ticks - otherwise state[t] and action[t] are no longer the "
                "pair the dataset is built from." % (tag, key, got, n))
    if meta is not None:
        if "n_ticks" not in meta:
            raise EpisodeLengthError(
                "episode%s carries no meta[\"n_ticks\"], so the arrays have "
                "nothing to be checked against: arrays that are all short by "
                "the same amount agree with each other and the episode is "
                "still truncated." % tag)
        declared = int(meta["n_ticks"])
        if declared != n:
            raise EpisodeLengthError(
                "episode%s declares meta[\"n_ticks\"] = %d but its per-tick "
                "arrays are %d tick(s) long. The file was truncated, or was "
                "written by a path that set the count and the arrays "
                "separately." % (tag, declared, n))
    return n


def load_episode(path: str):
    """(arrays, meta). Refuses a file written under another spec version, or one
    whose per-tick arrays do not agree on their length."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        spec.assert_spec_version(meta["spec_version"], where=os.path.basename(path))
        arrays = {k: z[k].copy() for k in z.files if k != "meta"}
    assert_tick_lengths(arrays, meta, where=os.path.basename(path))
    return arrays, meta


class ScriptedRecorder:
    """Records `scripted_demo.run_episode` without modifying it.

    A context manager around the call. Two things are wrapped, both read-only:

      `mujoco.mj_step`              the tick hook. It fires only for frames
                                    belonging to `run_episode`, so a step taken
                                    anywhere else could never be mistaken for a
                                    demonstrator tick.
      `scripted_demo.reset_episode` captures the post-reset fingerprint, which is
                                    what the replay test asserts against BEFORE it
                                    blames the actions for a divergence.

    `act` is the velocity command in force for the step about to be taken. The
    demonstrator computes it AFTER `mj_step`, for the next step, so at tick 0 it
    does not exist yet and zeros are logged - which is the truth: no locomotion
    command had been issued. While the base lock is engaged `build_action` zeroes
    those dims anyway (D17), reading the lock from the model's own equality
    rather than from anything passed in here.
    """

    #: No camera, no tracking path: `tracking_ok` is not written (O28).
    MEASURES_TRACKING = False

    def __init__(self):
        self.buf: Optional[EpisodeBuffer] = None
        self.reset_fingerprint = ""
        self.weld_engage_tick = -1
        self.weld_release_tick = -1
        self.lock_engage_tick = -1
        self.lock_release_tick = -1
        self.box_pre_grasp_disturb_mm = 0.0
        self._box0 = None
        self._prev_weld = False
        self._prev_lock = False

    # ---- attachment ----------------------------------------------------
    def __enter__(self) -> "ScriptedRecorder":
        from g1_data import scripted_demo as SD
        self._sd = SD
        self._real_step = mujoco.mj_step
        self._real_reset = SD.reset_episode
        rec = self

        def reset_wrapper(model, data, index, cfg, seed, *a, **k):
            start = rec._real_reset(model, data, index, cfg, seed, *a, **k)
            rec.reset_fingerprint = state_fingerprint(data, index)
            return start

        def step_wrapper(m, d, *a, **k):
            f = sys._getframe(1)
            if f.f_code.co_name == "run_episode":
                # A recorder fault must not corrupt a collection session
                # silently: let it raise, and fail the episode loudly.
                rec._observe(m, d, f.f_locals)
            return rec._real_step(m, d, *a, **k)

        mujoco.mj_step = step_wrapper
        SD.reset_episode = reset_wrapper
        return self

    def __exit__(self, *exc) -> None:
        mujoco.mj_step = self._real_step
        self._sd.reset_episode = self._real_reset

    # ---- the tick ------------------------------------------------------
    def _observe(self, m, d, L: dict) -> None:
        i, ix, weld = L["i"], L["ix"], L["weld"]
        if self.buf is None:
            self.buf = EpisodeBuffer(m, ix,
                                     measures_tracking=self.MEASURES_TRACKING)
            # The model the episode actually ran on: the contact contract in the
            # metadata is read back from THIS model, never from a config flag.
            self.model = m
            self._weld_eq = weld.eq_id
            self._lock_eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY,
                                              "base_lock")
            self._box0 = np.array(d.qpos[ix.box_qpos][:3], dtype=np.float64)
        welded = bool(d.eq_active[self._weld_eq])
        locked = bool(d.eq_active[self._lock_eq])
        tick = i // TICK
        if welded and not self._prev_weld and self.weld_engage_tick < 0:
            self.weld_engage_tick = tick
        if self._prev_weld and not welded and self.weld_release_tick < 0:
            self.weld_release_tick = tick
        if locked and not self._prev_lock and self.lock_engage_tick < 0:
            self.lock_engage_tick = tick
        if self._prev_lock and not locked and self.lock_release_tick < 0:
            self.lock_release_tick = tick
        self._prev_weld, self._prev_lock = welded, locked
        if self.weld_engage_tick < 0:
            # How far the box moved BEFORE THE FIRST GRASP. Bounded by the
            # first engage, not by "not welded": after RELEASE at the goal the
            # box is legitimately 1.4 m from its spawn, and counting that read
            # 1411 mm on a healthy episode. A demonstration that shoves the box
            # before grasping it is a demonstration of a different task, and O26
            # plus the D18 teleop finding both make that a live risk.
            moved = np.linalg.norm(np.asarray(d.qpos[ix.box_qpos][:3]) - self._box0)
            self.box_pre_grasp_disturb_mm = max(self.box_pre_grasp_disturb_mm,
                                                1000.0 * float(moved))
        if i % TICK:
            return
        act = L.get("act")
        act = np.zeros(3) if act is None else np.asarray(act, dtype=np.float64)
        loco = L["cfg"].loco
        # O28: `arm_stale` is a local of the TELEOP loop only. The scripted
        # demonstrator has no such local and no tracking path, so `.get` falls
        # back to False; its buffer does not write the array at all
        # (`MEASURES_TRACKING`), so the episode reads as NOT measured.
        self.buf.tick(m, d, act=act, cmd=float(L.get("cmd", 0.0)),
                      tracking_ok=not bool(L.get("arm_stale", False)),
                      phase_label=int(L["phase"]),
                      gait_counter=int(L["carry"].counter),
                      step_index=i, dt=loco.sim_dt, gait_period=loco.gait_period)

    # ---- the file ------------------------------------------------------
    def metadata(self, model, result: dict, cfg, demo, seed: int,
                 extra: Optional[dict] = None) -> dict:
        """Episode metadata. Every value measured or read back, none asserted."""
        wall = time.perf_counter() - self.buf.t_wall0
        sim_s = float(result["n_steps"]) * cfg.loco.sim_dt
        box0 = np.asarray(result["box_start"], dtype=float)
        m = dict(
            seed=int(seed),
            box_spawn_xy=[float(box0[0]), float(box0[1])],
            heldout=bool(result["heldout"]),
            standoff_cmd=float(demo.standoff),
            lateral_cmd=float(result.get("lock_lateral", float("nan"))),
            # Outcome as the DEMONSTRATOR reports it. The four canonical
            # per-phase success flags are the next chunk (they need the phase
            # labeller and the success module); everything they are computed
            # from is already in this file.
            outcome=dict(
                ok=bool(result["ok"]), fail_phase=result["fail_phase"],
                engaged=bool(result["engaged"]), resting=bool(result["resting"]),
                fell=bool(result["fell"]),
                placement_error=float(result["place_err_m"]),
                tilt_deg=float(result["tilt_deg"]),
                palm_err_mm=float(result["palm_err_mm"]),
                carry_drift_mm=float(result["carry_drift_mm"]),
                max_pitch_deg=float(result["max_pitch_deg"])),
            weld_engage_tick=self.weld_engage_tick,
            weld_release_tick=self.weld_release_tick,
            lock_engage_tick=self.lock_engage_tick,
            lock_release_tick=self.lock_release_tick,
            wrist_dev_rad=float(result["wrist_dev_rad"]),
            wrist_dev_joint=result["wrist_dev_joint"],
            # TR19: a POSITIVE clearance cannot be measured in this build, so
            # this is penetration DEPTH - negative while the hand is inside the
            # slab, 0.0 when it never touched. It is not "clearance = 0".
            min_hand_platform_clearance_mm=float(result["hand_plat_depth_mm"]),
            hand_platform_contacts=int(result["hand_plat_contacts"]),
            min_box_platform_clearance_m=float(result["min_clearance_m"]),
            box_pre_grasp_disturb_mm=float(self.box_pre_grasp_disturb_mm),
            sim_to_wall_ratio=float(sim_s / wall) if wall > 0 else float("nan"),
            sim_seconds=sim_s, wall_seconds=float(wall),
            n_steps=int(result["n_steps"]),
            reset_fingerprint=self.reset_fingerprint,
            contact_contract=contract_of(model),
            lock_predicate=bool(demo.lock_predicate),
            source=SOURCE_SCRIPTED,
            n_ticks=len(self.buf),
        )
        if extra:
            m.update(extra)
        return m


class TeleopRecorder(ScriptedRecorder):
    """The same recorder, attached to `run_integrated_combined.main` instead.

    Teleop episodes have NO ground-truth phase column - the operator does not run
    a phase machine - so `phase_labels` is written as `Phase.UNKNOWN` and the
    labels are derived offline by `g1_data.phase_label`. That is the whole reason
    the labeller is offline (2026-09-08).

    The velocity command logged is `active_cmd`: the command actually fed to the
    locomotion policy after the station-keeping and heading hold, not the raw key
    state. While the base lock is engaged `build_action` zeroes it anyway (D17).
    `reset_episode` is not wrapped: that entry point does not call it, so there is
    no reset fingerprint to capture and the field is left empty rather than faked.
    """

    FRAME = "main"
    #: The teleop loop's `arm_stale` is a real measurement (O28).
    MEASURES_TRACKING = True

    def __enter__(self) -> "TeleopRecorder":
        self._real_step = mujoco.mj_step
        rec = self

        def step_wrapper(m, d, *a, **k):
            f = sys._getframe(1)
            if f.f_code.co_name == rec.FRAME:
                rec._observe_teleop(m, d, f.f_locals)
            return rec._real_step(m, d, *a, **k)

        mujoco.mj_step = step_wrapper
        return self

    def __exit__(self, *exc) -> None:
        mujoco.mj_step = self._real_step

    def _observe_teleop(self, m, d, L: dict) -> None:
        ix = L.get("ix")
        if ix is None or "weld" not in L:
            return
        i = int(L.get("counter", 0))
        shim = dict(L)
        shim["i"] = i
        shim["ix"] = ix
        shim["cmd"] = float(L.get("grasp_cmd", 0.0))
        shim["act"] = L.get("active_cmd")
        shim["phase"] = int(spec.Phase.UNKNOWN)
        shim["carry"] = _GaitClock(i)
        shim["cfg"] = L.get("cfg")
        self._observe(m, d, shim)


class _GaitClock:
    """`carry.counter` is all `_observe` wants from the demonstrator's carryover;
    the teleop loop keeps the same clock in its own `counter`."""

    def __init__(self, counter: int):
        self.counter = counter
