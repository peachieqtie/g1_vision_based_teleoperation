"""The 47-D state and the 22-D action, resolved from names. One definition.

After this module lands, **no other file may hardcode a dimension index, a
mask, a clip limit or a normalization rule.** Recorder, dataset, the three
policies, the deployment loop and the evaluation harness all import from here.
Built the way `g1_teleop/indices.py` is built: names in, indices out,
assertions at load, loud failure.

WHY THIS EXISTS AT ALL
----------------------
Every number below is measured, not assumed - see NOTES.md "2026-09-10 Phase 2
constant-dimension audit", 12 episodes of the walking demonstrator sampled at
25 Hz. The audit exists because three of the choices here fail SILENTLY if
guessed:

  1. the permutation (below),
  2. the gripper dims, which are one scalar written twice,
  3. the velocity clip, which the demonstrator does not take from the place
     everything else takes it from.

THE PERMUTATION - the trap this module exists to close
------------------------------------------------------
`config.UPPER_BODY_JOINTS = WAIST + LEFT_ARM + RIGHT_ARM`, so
`data.ctrl[ix.upper_ctrl]` comes out ordered **[waist 3, left 7, right 7]**.
The action vector is proposal Table 3.4 order, **[left 7, right 7, waist 3]**.

    action[0:17] = data.ctrl[ix.upper_ctrl]      # WRONG

does not raise, does not change how the robot behaves, and mislabels 17 of 22
action dimensions in every episode ever recorded. It would surface as a policy
that drives its waist with shoulder commands, months later, with the whole
dataset already collected. The permutation is therefore DERIVED from the joint
name lists (never written out), applied through `ACTION_FROM_UPPER`, and
asserted to be a bijection whose round trip is the identity, at load.

TIMING CONVENTION - read this before writing the recorder
---------------------------------------------------------
At tick t:

    state[t]  is read from the stepped model
    action[t] is the command written to `data.ctrl` for the 20 physics steps
              that follow (25 Hz record rate, 500 Hz physics, D4)

so **action[t] is the command applied FROM state[t]**, and the pair (s_t, a_t)
is "what the demonstrator did in this situation". `state[t+1]` is the result.
An off-by-one here does not crash and does not look like an off-by-one: the
open-loop replay test diverges, which looks exactly like having logged the
wrong quantity (twin instead of stepped, or qpos instead of ctrl). Build the
pair at ONE point in the loop, before `mj_step`, via `SpecLayout.build`.

STEPPED MODEL ONLY
------------------
Everything is read from the stepped `MjModel`/`MjData`. The `PoseBook` twin is
never a source: `PoseBook.solve` returns an IK residual on the kinematic twin
and TR16 measured that understating the achieved palm distance by an order of
magnitude (11-43 mm reported against 227-264 mm achieved).

One consequence has bitten the audit already: `site_xpos` is only refreshed by
`mj_forward`/`mj_step`, so reading it BEFORE `mj_step` gives kinematics one
step stale relative to the `qpos` being logged in the same vector. `build_state`
calls `mj_kinematics` first by default. That cannot perturb the simulation -
`mj_step` recomputes the same quantities from `qpos` at the top of the step.

WHAT CHANGED IN 1.1.0 (from 1.0.0)
----------------------------------
Two things, and both alter what a file contains, which is why the version moves:

1. **Velocity dims 19-21 are ZERO whenever the base lock is engaged.** 1.0.0 left
   this to the caller and `build_action` refused a default, which was right at
   the time but never resolved. Measured 2026-09-11 over the 12 gate seeds: the
   base is welded for **36.4%** of every episode, and through all of it the
   demonstrator's station-keeping block is **skipped** - `locked` and
   "legs PD-held" agreed on 15336 of 15336 ticks - so `act` is neither
   recomputed (it changed on 24 of 5580 locked ticks, and those 24 are the lock
   transitions themselves) nor consumed (`carry.obs[6:9] = act * cmd_scale` sits
   inside the skipped block) nor applied (the legs are held at DEFAULT_ANGLES).
   It is a stale value from the last free tick. Logging it would record
   **vy = -0.732 held constant through REACH->LIFT** as though it were a command,
   and teach a policy to emit locomotion commands with no consequence. Zero is
   the semantically true value: no locomotion is being requested.
   The channel's phase structure becomes legible - live during the walk-in,
   zero through REACH->LIFT, live during MOVE, zero through LOWER->RELEASE,
   live again in VERIFY.
2. **The phase-label vocabulary is defined here** (`Phase`, `ScoredPhase`,
   `SCORED_OF`), with explicit stable integers. 1.0.0 had no opinion, so the
   recorder would have defined the contract by writing the column.
3. **`ScoredPhase.WALK_IN`**, a fifth failure-attribution bucket, with
   `SETTLE` mapped to it instead of to `GRASP`. Amended into 1.1.0 rather than
   versioned again because no episode file has ever been written at 1.1.0 -
   checked: no `.npz` exists in the project, and no recorder exists yet.
   **This does not change the metric set.** The four success rates in proposal
   3.8.2 / 3.8.3 are episode-level OUTCOME criteria on the final state and are
   untouched; `SCORED_OF` decides only which bucket a failed episode is charged
   to. See its comment.

A loader meeting a 1.0.0 file should assume: dims 19-21 may contain a stale
station-keeper output during locked ticks, and the phase column - if present at
all - has undocumented integers.

PHASE LABELS ARE NOT A POLICY INPUT
-----------------------------------
Decided 2026-09-08: the recorder logs a phase label every timestep, and **no
policy is conditioned on it**. It exists for the per-phase failure taxonomy and
for the data-scaling analysis - cheap to record now, expensive to add later.

That has a consequence worth stating so nobody builds the wrong thing:
**teleoperated episodes do not need a live phase classifier.** Labels can be
derived OFFLINE from the recorded trajectory, because every boundary the
taxonomy needs is already a function of the 47-D state - the weld bit
(`grip_left`) gives grasp and release, box height above resting gives lift and
lower, and base-to-box distance gives the approach and the transport. Nothing in
collection has to know the phase while it is happening.

DEVIATIONS RECORDED HERE (for CLAUDE.md section 7, not yet folded in)
--------------------------------------------------------------------
D13 (proposed). Proposal Table 3.3 sources the two gripper STATE dims from
"MuJoCo actuator state" - i.e. the palm-pad joints. This spec sources BOTH from
the weld bit (D11). Measured, 2026-09-10: the pad channels carry no grasp
signal at all - point-biserial correlation with weld state -0.072 (left) and
+0.038 (right), std collapsing from 9e-4 while released to 4e-5 while welded,
because the pads are commanded to zero and their contact is disabled whenever
the weld carries the box. The channel is passive slide-joint deflection at the
+-10 mm scale; z-score normalization would amplify it to unit variance and hand
every policy a pure-noise input. `g_L` and `g_R` are consequently identical -
retained as two dims so the vector still matches Table 3.3's 47.

D14 (proposed). The velocity ACTION clip is (+-0.80, +-0.80, +-0.60), not the
(+-0.80, +-0.40, +-0.80) that CLAUDE.md section 6 documents. See VELOCITY_CLIP.

KNOWN DEGENERACIES IN THE DATA THIS SPEC DESCRIBES
--------------------------------------------------
Not defects in the spec; facts about the demonstrator, all measured 2026-09-10,
all of which belong in the thesis rather than being silently normalized away:

  * 5 action dims are HARD-CONSTANT at exactly 0 and 1 is a duplicate ->
    CONSTANT_ACTION_DIMS, 16 trainable dims remain.
  * 4 more (wrist roll/pitch) are bit-identical time-series in every episode.
    They are NOT masked: they vary in time, so a policy must reproduce them.
  * `g_L`/`g_R` and both gripper action dims are bit-identical across episodes
    - the weld engages at the same tick every time, and it was never once
    commanded and refused. The grasp bit is a clock in this dataset.
  * no state dim is constant; the three waist STATE dims are live (servo
    deflection up to 3.37 deg) even though the three waist ACTION dims are
    exactly 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np
import mujoco

from g1_teleop import config as C
from g1_teleop.base_lock import BaseLockConfig
from g1_teleop.grasp import GraspConfig
from g1_teleop.indices import ModelIndex


#: Bump on ANY layout change: a dimension moved, a mask changed, a source
#: changed, a clip changed. Every episode file carries it and the dataset
#: loader must refuse to mix versions in one dataset - normalization stats
#: fitted under one layout are silently wrong under another.
SPEC_VERSION: str = "g1-spec-1.1.0"

STATE_DIM: int = 47
ACTION_DIM: int = 22

#: Record/policy rate (D4). Physics runs at 500 Hz; one recorded tick is
#: `PHYSICS_STEPS_PER_TICK` steps of it.
CONTROL_HZ: float = 25.0
PHYSICS_STEPS_PER_TICK: int = 20


# ─── phase vocabulary ─────────────────────────────────────────────────────────
# The episode file carries one label per tick. These integers ARE the file
# format: they are written out explicitly and must never be derived from
# declaration order, because reordering the enum would silently relabel every
# episode already recorded.
#
# The values below are the ones the demonstrator has always used, and they are
# frozen. REPOSITION and APPROACH sit out of sequence because they were added
# after VERIFY already existed; tidying them into execution order would be
# exactly the silent relabelling this comment exists to prevent.
#
# Labels are NOT a policy input (2026-09-08) - see the module docstring. They
# exist for the per-phase failure taxonomy, and for teleoperated episodes they
# can be derived offline from the recorded trajectory.


class Phase(IntEnum):
    """The demonstrator's 10 phases. The integers are the file format."""

    UNKNOWN = -1        # a tick whose phase could not be determined
    SETTLE = 0
    REACH = 1
    GRASP = 2
    LIFT = 3
    MOVE = 4
    LOWER = 5
    RELEASE = 6
    VERIFY = 7
    REPOSITION = 8      # added later than VERIFY - do NOT renumber
    APPROACH = 9        # likewise


class ScoredPhase(IntEnum):
    """Buckets for FAILURE ATTRIBUTION - which part of the task a failed
    episode broke in. See SCORED_OF for what this is and is not.

    WALK_IN is APPENDED at 4, not slotted in front of GRASP where it belongs
    chronologically, for the same reason REPOSITION is 8 and APPROACH is 9:
    these integers are written into files, and renumbering silently relabels
    everything already recorded.

    It is called WALK_IN and not APPROACH deliberately. `Phase.APPROACH`
    already exists in this module and means the ARM descending beside the box -
    it maps to GRASP, not to this. Two different things called APPROACH in one
    module is a trap.
    """

    GRASP = 0
    LIFT = 1
    TRANSPORT = 2
    PLACE = 3
    WALK_IN = 4         # appended - do NOT renumber the four above


#: The 10 -> 5 mapping, as DATA. A dict rather than a chain of `if`s inside a
#: function, so it can be asserted over, printed, and stored beside a dataset
#: instead of living in a body someone can quietly edit.
#:
#: WHAT THIS IS FOR, AND WHAT IT IS NOT
#: -----------------------------------
#: This is **failure attribution only**: which bucket a failed episode lands in.
#: It does **not** change the metric set. The four success rates in proposal
#: sections 3.8.2 / 3.8.3 are EPISODE-LEVEL OUTCOME criteria evaluated on the
#: final state - a robot that falls during the walk-in fails the grasp criterion
#: regardless of any per-tick label, and the reported success rates stay exactly
#: as the proposal specifies. Adding WALK_IN adds a category to the taxonomy,
#: nothing else.
#:
#: SETTLE -> WALK_IN, and this is the reason it exists: the spatial variation
#: Objective 4 is scored on lives almost entirely in the walk-in. The base
#: absorbs the spawn position and the arms then execute a near-identical motion
#: (measured 2026-09-10: arm-command variance ACROSS episodes is 3% of the
#: variance WITHIN one). So a policy that fails on a held-out box position will
#: most likely fail by walking to the wrong place, and charging that to "grasp
#: failure" would hide the exact effect the thesis exists to measure.
#:
#: REACH/REPOSITION/APPROACH/GRASP -> GRASP: the approach phases are how the
#: grasp is achieved, and a failure in any of them is a failed grasp.
#: REPOSITION is a judgement call and is recorded as one: it is the base closing
#: the standoff gap, so it is arguably locomotion and arguably WALK_IN. It stays
#: under GRASP because it happens AFTER arrival - the robot is already at the
#: box, and what it is fixing is the recession the reach itself caused, not
#: where it walked to. A WALK_IN failure should mean "went to the wrong place",
#: and REPOSITION failures do not mean that.
#:
#: LOWER/RELEASE/VERIFY -> PLACE: the box is not placed until it has been let go
#: and stayed put, which is what VERIFY measures.
SCORED_OF: Dict["Phase", "ScoredPhase"] = {
    Phase.SETTLE: ScoredPhase.WALK_IN,
    Phase.REACH: ScoredPhase.GRASP,
    Phase.REPOSITION: ScoredPhase.GRASP,
    Phase.APPROACH: ScoredPhase.GRASP,
    Phase.GRASP: ScoredPhase.GRASP,
    Phase.LIFT: ScoredPhase.LIFT,
    Phase.MOVE: ScoredPhase.TRANSPORT,
    Phase.LOWER: ScoredPhase.PLACE,
    Phase.RELEASE: ScoredPhase.PLACE,
    Phase.VERIFY: ScoredPhase.PLACE,
}

#: The 10 real phases, in execution order. UNKNOWN is deliberately absent: it is
#: a value the column may hold, not a phase the demonstrator performs.
PHASES: Tuple["Phase", ...] = (
    Phase.SETTLE, Phase.REACH, Phase.REPOSITION, Phase.APPROACH, Phase.GRASP,
    Phase.LIFT, Phase.MOVE, Phase.LOWER, Phase.RELEASE, Phase.VERIFY)

#: dtype for the phase column. int8 holds -1..9 with room to spare.
PHASE_DTYPE = np.int8


def scored_phase(label):
    """ScoredPhase for a Phase or its integer; vectorises over an array.

    Callers write `spec.scored_phase(labels)` and never their own lookup. On an
    array, ticks that are UNKNOWN (or any value not in the vocabulary) come back
    as -1 rather than raising, because a dataset may legitimately contain them.
    """
    if isinstance(label, np.ndarray):
        out = np.full(label.shape, -1, dtype=PHASE_DTYPE)
        for ph, sc in SCORED_OF.items():
            out[label == int(ph)] = int(sc)
        return out
    ph = Phase(int(label))
    if ph is Phase.UNKNOWN:
        raise ValueError(
            "UNKNOWN has no scored phase - it is not a phase the demonstrator "
            "performs. Filter those ticks out before scoring.")
    return SCORED_OF[ph]


def phase_name(label) -> str:
    """Name for a phase integer, for logs and failure taxonomies."""
    return Phase(int(label)).name


def _validate_phases() -> None:
    """Load-time, in the ModelIndex spirit: fail here, not at training time."""
    ints = [int(p) for p in Phase]
    if len(set(ints)) != len(ints):
        raise AssertionError("Phase integers are not unique: %s" % ints)
    if len(PHASES) != 10 or len(set(PHASES)) != 10:
        raise AssertionError("PHASES must list the 10 real phases exactly once")
    if Phase.UNKNOWN in PHASES:
        raise AssertionError("UNKNOWN is not a phase the demonstrator performs")
    missing = [p.name for p in PHASES if p not in SCORED_OF]
    if missing:
        raise AssertionError("SCORED_OF is not total: %s unmapped" % missing)
    extra = [p.name for p in SCORED_OF if p not in PHASES]
    if extra:
        raise AssertionError("SCORED_OF maps a non-phase: %s" % extra)
    # The image must be EXACTLY the four scored phases - neither short of one
    # nor inventing one. A scored phase with no source phase reports 0/0 for
    # the whole dataset and nobody notices.
    image = set(SCORED_OF.values())
    if len(ScoredPhase) != 5:
        raise AssertionError(
            "ScoredPhase should have 5 members (4 proposal phases + WALK_IN); "
            "found %d. Adding one means every bucket needs a source phase."
            % len(ScoredPhase))
    if image != set(ScoredPhase):
        raise AssertionError(
            "SCORED_OF image is %s, expected exactly %s"
            % (sorted(x.name for x in image), sorted(x.name for x in ScoredPhase)))
    lo, hi = min(ints), max(ints)
    info = np.iinfo(PHASE_DTYPE)
    if lo < info.min or hi > info.max:
        raise AssertionError("phase integers %d..%d do not fit %s"
                             % (lo, hi, PHASE_DTYPE))


_validate_phases()


# ─── joint name order ─────────────────────────────────────────────────────────
# The two orders that must never be confused. Both come from config.py, which
# is the source of truth for model order (see g1_teleop/indices.py).

#: Model order: what `data.ctrl[ix.upper_ctrl]` returns. Waist first.
UPPER_JOINTS: Tuple[str, ...] = tuple(C.UPPER_BODY_JOINTS)

#: Action order: proposal Table 3.4. Arms first, waist last.
ACTION_JOINTS: Tuple[str, ...] = tuple(
    list(C.LEFT_ARM_JOINTS) + list(C.RIGHT_ARM_JOINTS) + list(C.WAIST_JOINTS))

#: Gather index: `upper_vec[ACTION_FROM_UPPER]` is in action order.
ACTION_FROM_UPPER: np.ndarray = np.array(
    [UPPER_JOINTS.index(n) for n in ACTION_JOINTS], dtype=np.int32)

#: Gather index: `action_joint_vec[UPPER_FROM_ACTION]` is back in model order,
#: i.e. ready to write to `data.ctrl[ix.upper_ctrl]`.
UPPER_FROM_ACTION: np.ndarray = np.array(
    [ACTION_JOINTS.index(n) for n in UPPER_JOINTS], dtype=np.int32)

N_UPPER: int = len(UPPER_JOINTS)


def _short(joint: str) -> str:
    return joint[:-6] if joint.endswith("_joint") else joint


# ─── state layout ─────────────────────────────────────────────────────────────
BOX_POS = slice(0, 3)
BOX_QUAT = slice(3, 7)
BASE_POS = slice(7, 10)
BASE_QUAT = slice(10, 14)
PALM_L_POS = slice(14, 17)
PALM_L_QUAT = slice(17, 21)
PALM_R_POS = slice(21, 24)
PALM_R_QUAT = slice(24, 28)
GRIP_L = 28
GRIP_R = 29
ARM_L_Q = slice(30, 37)
ARM_R_Q = slice(37, 44)
WAIST_Q = slice(44, 47)

#: (name, slice) in vector order. Used by the load-time coverage assertion and
#: by anything that wants to report per-group statistics.
STATE_GROUPS: Tuple[Tuple[str, slice], ...] = (
    ("box_pos", BOX_POS), ("box_quat", BOX_QUAT),
    ("base_pos", BASE_POS), ("base_quat", BASE_QUAT),
    ("palmL_pos", PALM_L_POS), ("palmL_quat", PALM_L_QUAT),
    ("palmR_pos", PALM_R_POS), ("palmR_quat", PALM_R_QUAT),
    ("gripper", slice(GRIP_L, GRIP_R + 1)),
    ("arm_left_q", ARM_L_Q), ("arm_right_q", ARM_R_Q), ("waist_q", WAIST_Q),
)

_XYZ = ("x", "y", "z")
_WXYZ = ("w", "x", "y", "z")

STATE_NAMES: Tuple[str, ...] = tuple(
    [f"box_pos_{a}" for a in _XYZ] + [f"box_quat_{a}" for a in _WXYZ]
    + [f"base_pos_{a}" for a in _XYZ] + [f"base_quat_{a}" for a in _WXYZ]
    + [f"palmL_pos_{a}" for a in _XYZ] + [f"palmL_quat_{a}" for a in _WXYZ]
    + [f"palmR_pos_{a}" for a in _XYZ] + [f"palmR_quat_{a}" for a in _WXYZ]
    + ["g_L", "g_R"]
    + [f"q_{_short(n)}" for n in C.LEFT_ARM_JOINTS]
    + [f"q_{_short(n)}" for n in C.RIGHT_ARM_JOINTS]
    + [f"q_{_short(n)}" for n in C.WAIST_JOINTS]
)


# ─── action layout ────────────────────────────────────────────────────────────
ARM_L_A = slice(0, 7)
ARM_R_A = slice(7, 14)
WAIST_A = slice(14, 17)
UPPER_A = slice(0, 17)          # the 17 joint-target dims, in ACTION order
GRIP_L_A = 17
GRIP_R_A = 18
VEL_A = slice(19, 22)

ACTION_GROUPS: Tuple[Tuple[str, slice], ...] = (
    ("arm_left", ARM_L_A), ("arm_right", ARM_R_A), ("waist", WAIST_A),
    ("gripper", slice(GRIP_L_A, GRIP_R_A + 1)), ("velocity", VEL_A),
)

ACTION_NAMES: Tuple[str, ...] = tuple(
    [f"a_{_short(n)}" for n in ACTION_JOINTS] + ["a_gL", "a_gR"]
    + ["a_vx", "a_vy", "a_wz"])


# ─── constant-dimension mask ──────────────────────────────────────────────────
# Derived by NAME, then checked against the audited literals at load. Deriving
# it means a reordering of the config joint lists moves the mask with the dims
# instead of silently masking the wrong ones; checking it against the literals
# means such a reordering still fails loudly, because the audit's numbers were
# measured for these positions.
#
# Excluded from loss and from normalization (measured, 2026-09-10):
#   a_left_wrist_yaw, a_right_wrist_yaw   HARD-CONSTANT at exactly 0.0
#   a_waist_yaw/roll/pitch                HARD-CONSTANT at exactly 0.0 (D10)
#   a_gR                                  bit-identical to a_gL at every tick
#                                         of every episode; one scalar feeds
#                                         both through GraspWeld.update, so the
#                                         correlation is 1.0 by construction
#
# NOT excluded, deliberately: a_left/right_wrist_roll and _pitch. They are the
# same time-series in every episode (cross-episode spread exactly 0.0), but
# they VARY IN TIME - 0 -> WRIST_NATURAL over the reach - so a policy has to
# reproduce them. Masking them would leave the wrists at zero on deployment.
_MASKED_ACTION_NAMES: Tuple[str, ...] = (
    "a_left_wrist_yaw", "a_right_wrist_yaw",
    "a_waist_yaw", "a_waist_roll", "a_waist_pitch",
    "a_gR",
)

CONSTANT_ACTION_DIMS: Tuple[int, ...] = tuple(
    sorted(ACTION_NAMES.index(n) for n in _MASKED_ACTION_NAMES))

#: Audited positions. If the derived tuple stops matching this, the joint-name
#: lists moved and every number in the 2026-09-10 audit needs re-measuring.
_AUDITED_CONSTANT_DIMS: Tuple[int, ...] = (6, 13, 14, 15, 16, 18)

TRAINABLE_ACTION_DIMS: Tuple[int, ...] = tuple(
    d for d in range(ACTION_DIM) if d not in CONSTANT_ACTION_DIMS)

#: True where the dim is trainable. Loss and normalization use this; nothing
#: else may define its own.
ACTION_MASK: np.ndarray = np.ones(ACTION_DIM, dtype=bool)
ACTION_MASK[list(CONSTANT_ACTION_DIMS)] = False
ACTION_MASK.flags.writeable = False

#: The audit found ZERO constant state dimensions - not one of the 47 has zero
#: variance, including the three waist dims that CLAUDE.md section 6 describes
#: as "constant 0" (they are constant in the ACTION, live in the STATE). There
#: is therefore no state mask, and this is the assertion that says so.
STATE_MASK: np.ndarray = np.ones(STATE_DIM, dtype=bool)
STATE_MASK.flags.writeable = False


# ─── clip limits ──────────────────────────────────────────────────────────────
#: Body-frame velocity command limits, (low, high) per dim, for (vx, vy, wz).
#
# MEASURED from the demonstrator, not inherited. CLAUDE.md section 6 and the
# NOTES action-vector detail both document (+-0.80, +-0.40, +-0.80), which is
# `locomotion_input._clip_cmd` (MAX_FORWARD/MAX_LATERAL/MAX_TURN). The scripted
# demonstrator never calls that function: it clips act[0:2] JOINTLY with its own
# `DemoConfig.hold_max = 0.80` and act[2] with `hold_max_yaw = 0.60`. Measured
# over the 12-seed gate: vy reaches +-0.800 and exceeds the documented +-0.40 on
# 16.5% of the ticks where the policy is queried; wz never exceeds +-0.60.
#
# Per the maintenance protocol (CLAUDE.md section 14) the code is authoritative
# and the doc is what needs fixing, so the spec follows the code. Recorded as
# proposed deviation D14 in the module docstring.
VELOCITY_CLIP: np.ndarray = np.array(
    [[-0.80, 0.80], [-0.80, 0.80], [-0.60, 0.60]], dtype=np.float64)
VELOCITY_CLIP.flags.writeable = False

#: The gripper dims are the WELD COMMAND (D11): 0 = released, 1 = engaged,
#: gated on the geometric preconditions in g1_teleop/grasp.py. Commanding 1
#: with the hands in the wrong place does nothing.
GRIP_CLIP: Tuple[float, float] = (0.0, 1.0)

#: Tolerance for CHECKING recorded data against VELOCITY_CLIP. Not slack in the
#: limit - slack in the comparison.
#
# The command is computed in float32 (`act = np.zeros(3, dtype=np.float32)` in
# the demonstrator, `KeyboardCommand.cmd` likewise), so a command clipped to the
# rail is stored as float32(0.80) = 0.80000001192... and float32(0.60) =
# 0.60000002384..., both of which are ABOVE the limit once widened to float64.
# Measured over the 12-seed gate: 7.3% of ticks sit above the limit by at most
# 2.4e-8, which is one float32 ulp and not a real excursion. A Phase 3 loader
# asserting `abs(vy) <= 0.80` exactly would reject a fourteenth of the dataset
# for a rounding artifact, so validate through `assert_velocity_within_clip`.
VELOCITY_CLIP_TOL: float = 1e-6


def clip_velocity(velocity) -> np.ndarray:
    """Clip a (..., 3) body-frame velocity command to VELOCITY_CLIP.

    For EMITTING a command (deployment). To check recorded data, use
    `assert_velocity_within_clip` - see VELOCITY_CLIP_TOL for why they differ.
    """
    v = np.asarray(velocity, dtype=np.float64)
    return np.clip(v, VELOCITY_CLIP[:, 0], VELOCITY_CLIP[:, 1])


def assert_velocity_within_clip(velocity, tol: float = VELOCITY_CLIP_TOL,
                                where: str = "") -> None:
    """Check a (..., 3) recorded command against VELOCITY_CLIP, within tol."""
    v = np.asarray(velocity, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"velocity must have last dimension 3, got {v.shape}")
    excess = (np.abs(v.reshape(-1, 3)) - VELOCITY_CLIP[:, 1] - tol).max(axis=0)
    if np.any(excess > 0):
        bad = {n: float(e) for n, e in zip(("vx", "vy", "wz"), excess) if e > 0}
        raise AssertionError(
            f"velocity command outside VELOCITY_CLIP{' in ' + where if where else ''} "
            f"by more than {tol:g}: {bad}. Either the source clips somewhere "
            f"other than DemoConfig.hold_max / hold_max_yaw, or the limits here "
            f"are stale.")


# ─── named accessors ──────────────────────────────────────────────────────────
# Callers write spec.box_pos(state), never state[0:3]. Every accessor takes
# (..., STATE_DIM) or (..., ACTION_DIM), so a single vector and a batch of them
# behave identically.

def _get(vec, sl, dim: int, what: str):
    v = np.asarray(vec)
    if v.shape[-1] != dim:
        raise ValueError(f"{what} must have last dimension {dim}, got "
                         f"{v.shape}")
    return v[..., sl]


def box_pos(state): return _get(state, BOX_POS, STATE_DIM, "state")
def box_quat(state): return _get(state, BOX_QUAT, STATE_DIM, "state")
def base_pos(state): return _get(state, BASE_POS, STATE_DIM, "state")
def base_quat(state): return _get(state, BASE_QUAT, STATE_DIM, "state")
def palm_left_pos(state): return _get(state, PALM_L_POS, STATE_DIM, "state")
def palm_left_quat(state): return _get(state, PALM_L_QUAT, STATE_DIM, "state")
def palm_right_pos(state): return _get(state, PALM_R_POS, STATE_DIM, "state")
def palm_right_quat(state): return _get(state, PALM_R_QUAT, STATE_DIM, "state")
def grip_left(state): return _get(state, GRIP_L, STATE_DIM, "state")
def grip_right(state): return _get(state, GRIP_R, STATE_DIM, "state")
def arm_left_q(state): return _get(state, ARM_L_Q, STATE_DIM, "state")
def arm_right_q(state): return _get(state, ARM_R_Q, STATE_DIM, "state")
def waist_q(state): return _get(state, WAIST_Q, STATE_DIM, "state")


def arm_left_cmd(action): return _get(action, ARM_L_A, ACTION_DIM, "action")
def arm_right_cmd(action): return _get(action, ARM_R_A, ACTION_DIM, "action")
def waist_cmd(action): return _get(action, WAIST_A, ACTION_DIM, "action")
def upper_cmd(action): return _get(action, UPPER_A, ACTION_DIM, "action")
def grip_left_cmd(action): return _get(action, GRIP_L_A, ACTION_DIM, "action")
def grip_right_cmd(action): return _get(action, GRIP_R_A, ACTION_DIM, "action")
def velocity_cmd(action): return _get(action, VEL_A, ACTION_DIM, "action")


def upper_ctrl_from_action(action) -> np.ndarray:
    """The 17 joint targets in MODEL order, ready for `ctrl[ix.upper_ctrl]`.

    The inverse of the permutation `build_action` applies. Deployment writes a
    predicted action back to the model through this and nothing else; writing
    `action[0:17]` straight to `ctrl[ix.upper_ctrl]` is the failure this module
    exists to prevent, and it is silent in that direction too.
    """
    return upper_cmd(action)[..., UPPER_FROM_ACTION]


def action_from_upper_ctrl(upper) -> np.ndarray:
    """The 17 joint targets in ACTION order, from a MODEL-order vector."""
    u = np.asarray(upper)
    if u.shape[-1] != N_UPPER:
        raise ValueError(f"upper vector must have last dimension {N_UPPER}, "
                         f"got {u.shape}")
    return u[..., ACTION_FROM_UPPER]


# ─── normalization contract ───────────────────────────────────────────────────
@dataclass(frozen=True)
class NormStats:
    """Per-dimension mean/std. **The contract; the estimator is Phase 3.**

    THE CONTRACT, in full:

      1. Statistics are computed from the **training split only**. Not from
         validation, not from the held-out patch (Objective 4), not from the
         evaluation rollouts. Fitting on anything else leaks the test set into
         the normalizer, and the leak is invisible in every metric.
      2. They are computed **once**, stored beside the dataset, and loaded
         UNCHANGED at validation, at both experiments, and at deployment. A
         policy that was trained under one normalizer and deployed under
         another is simply a different policy.
      3. **Masked dims are never touched**: mean 0, std 1, so normalize and
         denormalize are the identity there. A constant dim has std 0, and
         dividing by it produces inf/NaN that propagates into the loss - which
         is the real reason the mask exists, over and above wasting capacity.
      4. `spec_version` is stored and checked. Stats fitted under one layout
         are silently wrong under another; that is what versioning is for.

    `std` is floored at `STD_FLOOR` for unmasked dims: a dim that is nearly
    constant in the training split but not exactly constant (the audit found
    several at 1e-4 to 1e-3) would otherwise get an enormous scale factor and
    amplify sampling noise into the model's input.
    """

    kind: str                      # "state" | "action"
    mean: np.ndarray
    std: np.ndarray
    split: str = "train"
    spec_version: str = SPEC_VERSION

    STD_FLOOR = 1e-6

    def __post_init__(self):
        if self.kind not in ("state", "action"):
            raise ValueError(f"kind must be 'state' or 'action', got "
                             f"{self.kind!r}")
        dim = STATE_DIM if self.kind == "state" else ACTION_DIM
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if mean.shape != (dim,) or std.shape != (dim,):
            raise ValueError(f"{self.kind} stats must be ({dim},); got "
                             f"mean {mean.shape}, std {std.shape}")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
            raise ValueError(f"{self.kind} stats contain non-finite values")
        assert_spec_version(self.spec_version, f"{self.kind} NormStats")
        mask = STATE_MASK if self.kind == "state" else ACTION_MASK
        if np.any(std[mask] < self.STD_FLOOR):
            bad = [i for i in np.flatnonzero(mask) if std[i] < self.STD_FLOOR]
            names = STATE_NAMES if self.kind == "state" else ACTION_NAMES
            raise ValueError(
                "std below STD_FLOOR on unmasked dims "
                + ", ".join(f"{i}:{names[i]}={std[i]:.3e}" for i in bad)
                + ". Either the dim belongs in the mask or the training split "
                  "is degenerate - do not floor it silently.")
        if np.any(mean[~mask] != 0.0) or np.any(std[~mask] != 1.0):
            raise ValueError(
                "masked dims must carry mean 0 / std 1 exactly, so normalize "
                "and denormalize are the identity there")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def identity(cls, kind: str, split: str = "train") -> "NormStats":
        """A no-op normalizer. For tests and for ablating normalization."""
        dim = STATE_DIM if kind == "state" else ACTION_DIM
        return cls(kind=kind, mean=np.zeros(dim), std=np.ones(dim), split=split)

    def normalize(self, x) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def denormalize(self, z) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.std + self.mean


def assert_spec_version(version: str, where: str = "") -> None:
    """Refuse anything written under a different layout."""
    if version != SPEC_VERSION:
        raise AssertionError(
            f"spec version mismatch{' in ' + where if where else ''}: "
            f"data says {version!r}, this build is {SPEC_VERSION!r}. Episodes, "
            f"normalization stats and checkpoints from different spec versions "
            f"must not be mixed - the dimension layout may differ.")


# ─── model-resolved layout ────────────────────────────────────────────────────
@dataclass(frozen=True)
class SpecLayout:
    """Everything the builders need, resolved from names against one model.

    Resolve once at load, like `ModelIndex`. `validate()` runs automatically
    and is where a renamed site, a reordered joint list or a lost equality
    turns into an exception instead of a silently wrong dataset.
    """

    ix: ModelIndex
    site_left: int
    site_right: int
    weld_id: int
    #: The world<->pelvis weld (D12). Read, never written, and only to decide
    #: whether a locomotion command is being requested at all - see
    #: `build_action`. Resolved from the model rather than taken on the caller's
    #: word, so the logged semantics cannot disagree with the physics.
    base_lock_id: int
    #: The 17 upper-body actuator ids in ACTION order. Resolved independently
    #: of `ix.upper_ctrl` (by name, from ACTION_JOINTS) so that the permutation
    #: has two derivations that must agree.
    action_ctrl: np.ndarray

    @classmethod
    def resolve(cls, model, ix: Optional[ModelIndex] = None,
                grasp: Optional[GraspConfig] = None) -> "SpecLayout":
        g = grasp or GraspConfig()
        ix = ix or ModelIndex.resolve(model)

        def site(name: str) -> int:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"palm site not found in model: {name}")
            return int(sid)

        lk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                               BaseLockConfig().eq_name)
        if lk < 0:
            raise ValueError(
                "base lock equality not found: %s. The velocity action dims are "
                "zero while the base is welded (D12, spec 1.1.0), so the spec "
                "cannot describe a scene without it."
                % BaseLockConfig().eq_name)
        eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, g.weld_name)
        if eq < 0:
            raise ValueError(
                f"weld equality not found: {g.weld_name}. The grasp state dims "
                f"are the weld bit (D11); it lives in scene.xml.")

        ids = []
        for name in ACTION_JOINTS:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"actuator not found in model: {name}")
            ids.append(int(aid))

        out = cls(ix=ix, site_left=site(g.left_site), site_right=site(g.right_site),
                  weld_id=int(eq), base_lock_id=int(lk),
                  action_ctrl=np.asarray(ids, dtype=np.int32))
        out.validate(model)
        return out

    # ---- assertions -----------------------------------------------------
    def validate(self, model) -> None:
        """Fail loudly, at load, on every way this can go quietly wrong."""
        # 1. Names and dimensions agree with the declared sizes.
        if len(STATE_NAMES) != STATE_DIM:
            raise AssertionError(f"STATE_NAMES has {len(STATE_NAMES)} entries, "
                                 f"STATE_DIM is {STATE_DIM}")
        if len(ACTION_NAMES) != ACTION_DIM:
            raise AssertionError(f"ACTION_NAMES has {len(ACTION_NAMES)} "
                                 f"entries, ACTION_DIM is {ACTION_DIM}")
        if len(ACTION_JOINTS) != N_UPPER or len(UPPER_JOINTS) != N_UPPER:
            raise AssertionError("upper-body joint lists disagree on length")
        if sorted(ACTION_JOINTS) != sorted(UPPER_JOINTS):
            raise AssertionError(
                "ACTION_JOINTS and UPPER_JOINTS are not the same set of "
                "joints; one of the config name lists changed")

        # 2. Groups tile each vector exactly once, with no gap and no overlap.
        for label, groups, dim in (("state", STATE_GROUPS, STATE_DIM),
                                   ("action", ACTION_GROUPS, ACTION_DIM)):
            covered = []
            for _, sl in groups:
                covered.extend(range(*sl.indices(dim)))
            if sorted(covered) != list(range(dim)):
                raise AssertionError(
                    f"{label} groups do not tile 0..{dim - 1} exactly: "
                    f"{sorted(covered)}")

        # 3. THE PERMUTATION. A bijection, and its round trip is the identity.
        for name, perm in (("ACTION_FROM_UPPER", ACTION_FROM_UPPER),
                           ("UPPER_FROM_ACTION", UPPER_FROM_ACTION)):
            if sorted(perm.tolist()) != list(range(N_UPPER)):
                raise AssertionError(
                    f"{name} is not a permutation of 0..{N_UPPER - 1}: "
                    f"{perm.tolist()}")
        probe = np.arange(N_UPPER, dtype=np.float64)
        if not np.array_equal(probe[ACTION_FROM_UPPER][UPPER_FROM_ACTION], probe):
            raise AssertionError(
                "permutation round trip is not the identity: "
                "upper -> action -> upper does not return the input")
        if not np.array_equal(probe[UPPER_FROM_ACTION][ACTION_FROM_UPPER], probe):
            raise AssertionError(
                "permutation round trip is not the identity: "
                "action -> upper -> action does not return the input")
        # ...and it maps the right NAMES, not merely some bijection.
        permuted = [UPPER_JOINTS[k] for k in ACTION_FROM_UPPER]
        if tuple(permuted) != ACTION_JOINTS:
            raise AssertionError(
                "ACTION_FROM_UPPER does not reorder model order into action "
                f"order: got {permuted}, want {list(ACTION_JOINTS)}")

        # 4. Every action joint dim maps to exactly one actuator, and the
        #    name-resolved ids agree with permuting ix.upper_ctrl.
        if self.action_ctrl.shape != (N_UPPER,):
            raise AssertionError(f"action_ctrl must hold {N_UPPER} actuator "
                                 f"ids, got {self.action_ctrl.shape}")
        if len(set(self.action_ctrl.tolist())) != N_UPPER:
            raise AssertionError(
                f"action_ctrl maps two action dims onto one actuator: "
                f"{self.action_ctrl.tolist()}")
        upper_ids = np.r_[self.ix.upper_ctrl] if isinstance(
            self.ix.upper_ctrl, slice) else np.asarray(self.ix.upper_ctrl)
        if not np.array_equal(upper_ids[ACTION_FROM_UPPER], self.action_ctrl):
            raise AssertionError(
                "the two derivations of the action actuator order disagree: "
                f"ix.upper_ctrl permuted is {upper_ids[ACTION_FROM_UPPER].tolist()}, "
                f"name resolution gives {self.action_ctrl.tolist()}. One of "
                f"config.UPPER_BODY_JOINTS or the model's actuator order moved.")
        if int(self.action_ctrl.max()) >= int(model.nu):
            raise AssertionError("action actuator id past model.nu")

        # 5. The mask is the audited one, at the audited positions.
        if CONSTANT_ACTION_DIMS != _AUDITED_CONSTANT_DIMS:
            raise AssertionError(
                f"constant-dim mask resolved to {CONSTANT_ACTION_DIMS}, but "
                f"the 2026-09-10 audit measured {_AUDITED_CONSTANT_DIMS}. The "
                f"joint-name lists moved; re-measure before trusting the mask.")
        got = tuple(ACTION_NAMES[d] for d in CONSTANT_ACTION_DIMS)
        if sorted(got) != sorted(_MASKED_ACTION_NAMES):
            raise AssertionError(
                f"masked dims name {got}, expected {_MASKED_ACTION_NAMES}")
        if int(ACTION_MASK.sum()) != ACTION_DIM - len(_MASKED_ACTION_NAMES):
            raise AssertionError("ACTION_MASK disagrees with "
                                 "CONSTANT_ACTION_DIMS")
        if not bool(STATE_MASK.all()):
            raise AssertionError(
                "STATE_MASK must be all-True: the audit found no constant "
                "state dimension. Masking one needs a new measurement.")

        # 6. The state sources exist and have the sizes the layout assumes.
        for label, idx, want in (
                ("box_qpos", self.ix.box_qpos, 7),
                ("base_qpos", self.ix.base_qpos, 7),
                ("left_arm_qpos", self.ix.left_arm_qpos, 7),
                ("right_arm_qpos", self.ix.right_arm_qpos, 7),
                ("waist_qpos", self.ix.waist_qpos, 3)):
            n = ((idx.stop - idx.start) if isinstance(idx, slice)
                 else int(np.asarray(idx).size))
            if n != want:
                raise AssertionError(f"ix.{label} resolved {n} indices, "
                                     f"expected {want}")
        if self.weld_id >= int(model.neq):
            raise AssertionError("weld equality id past model.neq")
        if self.base_lock_id >= int(model.neq):
            raise AssertionError("base lock equality id past model.neq")
        if self.base_lock_id == self.weld_id:
            raise AssertionError(
                "the base lock and the grasp weld resolved to the same "
                "equality; one of the names in scene.xml has changed")

    # ---- builders -------------------------------------------------------
    def build_state(self, model, data, ix: Optional[ModelIndex] = None,
                    sync: bool = True) -> np.ndarray:
        """The 47-D state from the STEPPED model. See module docstring.

        `sync` refreshes forward kinematics first. Leave it on when building
        before `mj_step` (which is where the pair must be built): `site_xpos`
        is otherwise one step stale relative to the `qpos` in the same vector,
        so the palm poses would describe a different instant from the joint
        angles. It cannot perturb the simulation - `mj_step` recomputes the
        same arrays from `qpos` at the top of the step.
        """
        ix = self._check_ix(ix)
        if sync:
            mujoco.mj_kinematics(model, data)
        s = np.empty(STATE_DIM, dtype=np.float64)
        s[BOX_POS] = data.qpos[ix.box_qpos][0:3]
        s[BOX_QUAT] = data.qpos[ix.box_qpos][3:7]
        s[BASE_POS] = data.qpos[ix.base_qpos][0:3]
        s[BASE_QUAT] = data.qpos[ix.base_qpos][3:7]
        s[PALM_L_POS] = data.site_xpos[self.site_left]
        mujoco.mju_mat2Quat(s[PALM_L_QUAT], data.site_xmat[self.site_left])
        s[PALM_R_POS] = data.site_xpos[self.site_right]
        mujoco.mju_mat2Quat(s[PALM_R_QUAT], data.site_xmat[self.site_right])
        # Both gripper dims are the same weld bit (D11, and D13 in the module
        # docstring): the weld is left-hand only and the pads carry no grasp
        # signal, so there is no second channel to read.
        weld = float(data.eq_active[self.weld_id])
        s[GRIP_L] = weld
        s[GRIP_R] = weld
        s[ARM_L_Q] = data.qpos[ix.left_arm_qpos]
        s[ARM_R_Q] = data.qpos[ix.right_arm_qpos]
        s[WAIST_Q] = data.qpos[ix.waist_qpos]
        return s

    def build_action(self, data, ix: Optional[ModelIndex] = None, act=None,
                     cmd: float = 0.0) -> np.ndarray:
        """The 22-D action from `data.ctrl` on the STEPPED model.

        `act` is the body-frame velocity command (vx, vy, wz) **pre-cmd_scale**
        - the raw command, before `LocomotionConfig.cmd_scale` multiplies it
        into the policy's observation. Log the command, not the scaled
        observation: the scale is a property of the pre-trained policy, not of
        the demonstration.

        `cmd` is the weld command (D11), 0 or 1, the same scalar passed to
        `GraspWeld.update` this tick. Both gripper dims carry it.

        **While the base lock is engaged the velocity dims are logged as ZERO**,
        whatever `act` contains (spec 1.1.0). That is the semantically true
        command: no locomotion is being requested. Measured 2026-09-11, the
        demonstrator skips its station-keeping block entirely while locked, so
        `act` there is a stale value from the last free tick that is neither
        recomputed, nor fed to the policy, nor applied to the legs - logging it
        would teach a policy to emit commands with no consequence, including
        vy = -0.732 held constant through REACH->LIFT.

        The lock state is read from the model's own equality, not taken on the
        caller's word, so the logged semantics cannot drift from the physics.

        `act` is still REQUIRED even when it will be zeroed: the caller must say
        what the command was, so there is one code path and the value can be
        cross-checked. At tick 0 nothing has been computed yet; pass zeros
        explicitly if that is the intent. Passing None raises.
        """
        ix = self._check_ix(ix)
        if act is None:
            raise ValueError(
                "act (vx, vy, wz) is required. There is no default: while the "
                "base is locked the locomotion policy is never queried, and "
                "what to log then is a recorder decision. Pass zeros "
                "explicitly if that is the intent.")
        v = np.asarray(act, dtype=np.float64).reshape(-1)
        if v.shape != (3,):
            raise ValueError(f"act must be 3 values (vx, vy, wz), got "
                             f"{np.asarray(act).shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"act is not finite: {v}")
        c = float(cmd)
        if not (GRIP_CLIP[0] - 1e-9 <= c <= GRIP_CLIP[1] + 1e-9):
            raise ValueError(f"gripper command must be in [0, 1] (D11), got {c}")
        self._check_gripper_source(data, ix, c)

        a = np.empty(ACTION_DIM, dtype=np.float64)
        # The permutation. Model order in, action order out - never a raw slice.
        a[UPPER_A] = np.asarray(data.ctrl[ix.upper_ctrl],
                                dtype=np.float64)[ACTION_FROM_UPPER]
        a[GRIP_L_A] = c
        a[GRIP_R_A] = c
        a[VEL_A] = 0.0 if bool(data.eq_active[self.base_lock_id]) else v
        return a

    def build(self, model, data, ix: Optional[ModelIndex] = None, act=None,
              cmd: float = 0.0, sync: bool = True):
        """(state, action) for one tick. The recorder's only entry point.

        Call once per recorded tick, BEFORE `mj_step`, so that `data.ctrl`
        holds the command about to be applied and the state is the one it is
        applied from (see TIMING CONVENTION in the module docstring).
        """
        return (self.build_state(model, data, ix, sync=sync),
                self.build_action(data, ix, act=act, cmd=cmd))

    # ---- internals ------------------------------------------------------
    def _check_ix(self, ix: Optional[ModelIndex]) -> ModelIndex:
        """Accept the caller's ModelIndex, but not a different model's."""
        if ix is None or ix is self.ix:
            return self.ix
        mine = np.r_[self.ix.upper_ctrl] if isinstance(
            self.ix.upper_ctrl, slice) else np.asarray(self.ix.upper_ctrl)
        theirs = np.r_[ix.upper_ctrl] if isinstance(
            ix.upper_ctrl, slice) else np.asarray(ix.upper_ctrl)
        if not np.array_equal(mine, theirs):
            raise AssertionError(
                "the ModelIndex passed to the builder was resolved against a "
                "different model than this SpecLayout. Resolve both from the "
                "same MjModel.")
        return ix

    def _check_gripper_source(self, data, ix: ModelIndex, cmd: float) -> None:
        """Catch a rig that presses the pads with something other than `cmd`.

        The two collection paths write the gripper command to DIFFERENT places:
        the scripted demonstrator holds `ctrl[ix.pad_ctrl] = 0` and passes the
        command to `GraspWeld.update` (TR17: driving the pads while welded
        buries them in the box and the release ejects it), while
        `run_integrated_combined.py` writes the same command to
        `ctrl[ix.pad_ctrl]`. The action dims are the WELD COMMAND either way
        (D11, CLAUDE.md section 6), which is why `cmd` is an argument rather
        than something read off the model. This check only asserts the two do
        not DISAGREE: if the pads are being driven at all, they must be driven
        with the command being logged.
        """
        if ix.pad_ctrl is None:
            return
        pads = np.asarray(data.ctrl[ix.pad_ctrl], dtype=np.float64)
        if np.any(pads != 0.0) and np.any(np.abs(pads - cmd) > 1e-9):
            raise AssertionError(
                f"pad actuators are driven at {pads.tolist()} while the logged "
                f"gripper command is {cmd}. The action's gripper dims are the "
                f"weld command (D11); a rig pressing the pads with a different "
                f"value would record a command the robot never received.")


def describe() -> str:
    """One-screen summary, for a log header or a session sanity check."""
    lines = [f"{SPEC_VERSION}: state {STATE_DIM}-D, action {ACTION_DIM}-D, "
             f"{CONTROL_HZ:.0f} Hz ({PHYSICS_STEPS_PER_TICK} physics steps)",
             "  action order  : " + ", ".join(ACTION_JOINTS[:3]) + ", ... "
             "(arms, then waist)",
             "  model order   : " + ", ".join(UPPER_JOINTS[:3]) + ", ... "
             "(waist, then arms)",
             f"  masked action : {CONSTANT_ACTION_DIMS} = "
             + ", ".join(_MASKED_ACTION_NAMES),
             f"  trainable     : {len(TRAINABLE_ACTION_DIMS)} of {ACTION_DIM} "
             f"action dims, {STATE_DIM} of {STATE_DIM} state dims",
             "  velocity clip : vx +-%.2f, vy +-%.2f, wz +-%.2f"
             % (VELOCITY_CLIP[0, 1], VELOCITY_CLIP[1, 1], VELOCITY_CLIP[2, 1])]
    return "\n".join(lines)
