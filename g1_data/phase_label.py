"""Phase labels derived OFFLINE from recorded state. There is no live labeller.

WHY OFFLINE, AND WHY NOBODY SHOULD BUILD A LIVE ONE
---------------------------------------------------
Decided 2026-09-08 and restated in spec.py: the recorder logs a phase label every
timestep and **no policy is conditioned on it**. Labels exist for the per-phase
failure taxonomy and the data-scaling analysis, so they can be derived after the
fact from the trajectory - every boundary the taxonomy needs is a function of
quantities already in the 47-D state. Nothing in collection has to know the phase
while it is happening, and a live classifier would be a second thing to keep
correct for no benefit.

The vocabulary is `spec.Phase` and the buckets are `spec.SCORED_OF`. This module
defines neither.

WHAT THIS CAN AND CANNOT SEPARATE
---------------------------------
`REACH`, `REPOSITION` and `APPROACH` are not separable from state: they differ by
which pose the demonstrator commanded, not by anything the box, the base or the
weld does. All three are labelled `REACH` here, and all three map to
`ScoredPhase.GRASP` in `SCORED_OF`, so the taxonomy is unaffected - the confusion
matrix reports it rather than hiding it.

MONOTONE BY CONSTRUCTION
------------------------
The task is a sequence: once the box is welded the episode is not back in SETTLE.
Raw per-tick rules do flicker at boundaries (measured, below), so the labels are
forced non-decreasing in execution order. That is the hysteresis, applied once,
offline, where it is inspectable - and `label_episode` returns the number of
ticks it had to correct so a caller can see how much work it did.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from g1_data import spec
from g1_data.success import geometry

_PLAT = {}


def _lock_ticks(states) -> list:
    """The ticks the base lock fires at, recomputed OFFLINE from recorded state.

    `LockPredicate` is a pure function of the 47-D vector (that is the whole
    point of D12 being state-driven), so running it over a recording reproduces
    the demonstrator's own lock instants exactly - including the settle window
    and the 5-tick debounce. The SECOND lock is the MOVE -> LOWER boundary, and
    nothing else in the recording marks it: the recorder stores only the first
    engage and release.
    """
    from g1_data.phases import LockConfig, LockPredicate, PlatformGeometry
    if "geo" not in _PLAT:
        from g1_teleop.config import TeleopConfig
        from g1_teleop.contact_contract import load_model
        _PLAT["geo"] = PlatformGeometry.resolve(load_model(TeleopConfig()))
    pred = LockPredicate(_PLAT["geo"], LockConfig())
    out = []
    for t, st in enumerate(states):
        if pred.update(np.asarray(st, dtype=np.float64)) == "lock":
            out.append(t)
    return out

#: Execution order. REPOSITION and APPROACH sit where they RUN, not where their
#: integers fall - the integers are a file format, this is a sequence (spec.py).
ORDER = (spec.Phase.SETTLE, spec.Phase.REACH, spec.Phase.REPOSITION,
         spec.Phase.APPROACH, spec.Phase.GRASP, spec.Phase.LIFT, spec.Phase.MOVE,
         spec.Phase.LOWER, spec.Phase.RELEASE, spec.Phase.VERIFY)
RANK = {int(p): i for i, p in enumerate(ORDER)}

LIFT_MIN = 0.055          # m above resting: "carrying" (LockConfig.lift_min)
# Both thresholds are CALIBRATED on the scripted demonstrator, whose per-tick
# phase is ground truth (29 episodes): at the true LIFT->MOVE boundary the box is
# 0.049-0.077 m from its spawn, and at the true MOVE->LOWER boundary it is
# 0.045-0.096 m from the goal - by then the arm has already done most of the
# lowering, which is why a height rule reads LOWER ~84 ticks early.
LEFT_PICKUP = 0.045       # m from its spawn: the box is under way, not being lifted
NEAR_GOAL = 0.10          # m from the goal: arrived, and what remains is setting down
WITHDRAW = 0.25           # m: palms clear of the box (LockConfig.withdraw)


def label_episode(arrays: dict, meta: Optional[dict] = None, geo=None):
    """(labels (T,) int8, diagnostics). Pure function of recorded state.

    BOUNDARIES, NOT PER-TICK RULES. The first version classified each tick
    independently and scored 50.9% against the demonstrator: it never emitted
    LIFT at all, and read 2575 MOVE ticks as LOWER because the carried box's
    height hovers either side of a threshold. The task is a sequence, so the
    boundaries are found once, from the trajectory, and the segments between them
    are filled - which makes the labels monotone by construction rather than by
    correction afterwards.
    """
    g = geo or geometry()
    s = np.asarray(arrays["states"], dtype=np.float64)
    T = len(s)
    box = s[:, spec.BOX_POS]
    welded = s[:, spec.GRIP_L] >= 0.5
    lift = box[:, 2] - g["rest_z"]
    to_goal = np.linalg.norm(box[:, :2] - g["goal_xy"], axis=1)
    palm_far = np.maximum(
        np.linalg.norm(s[:, spec.PALM_L_POS] - box, axis=1),
        np.linalg.norm(s[:, spec.PALM_R_POS] - box, axis=1))
    meta = meta or {}
    spawn = np.asarray(meta.get("box_spawn_xy", box[0, :2]), dtype=np.float64)
    d_pickup = np.linalg.norm(box[:, :2] - spawn, axis=1)
    locks = _lock_ticks(s)

    def first(mask, lo=0, default=T):
        idx = np.flatnonzero(mask[lo:])
        return int(lo + idx[0]) if len(idx) else default

    def last(mask, hi, default=T):
        idx = np.flatnonzero(mask[:hi])
        return int(idx[-1]) if len(idx) else default

    ever = bool(welded.any())
    t_lock = locks[0] if locks else int(meta.get("lock_engage_tick", 0) or 0)
    t_eng = first(welded) if ever else T
    t_rel = first(~welded, lo=t_eng) if ever else T
    t_lift = first(lift >= LIFT_MIN, lo=t_eng)
    t_move = first(d_pickup >= LEFT_PICKUP, lo=t_lift)
    # The SECOND lock: the base has arrived at the goal and settled, which is
    # exactly where the demonstrator ends MOVE. A box-height rule reads this ~49
    # ticks early (the arm lowers while the robot is still station-keeping) and a
    # distance-to-goal rule ~9 early (the box arrives before the base settles).
    t_lower = (locks[1] if len(locks) > 1
               else first(to_goal <= NEAR_GOAL, lo=t_move))
    t_verify = first(palm_far >= WITHDRAW, lo=t_rel)

    # Order them. A boundary that cannot be found sits at T, which collapses its
    # segment to nothing rather than reordering the sequence.
    bounds = [t_lock, t_eng, t_lift, t_move, t_lower, t_rel, t_verify]
    for k in range(1, len(bounds)):
        bounds[k] = max(bounds[k], bounds[k - 1])
    t_lock, t_eng, t_lift, t_move, t_lower, t_rel, t_verify = bounds

    labels = np.full(T, int(spec.Phase.SETTLE), dtype=spec.PHASE_DTYPE)
    for lo, hi, ph in ((t_lock, t_eng, spec.Phase.REACH),
                       (t_eng, t_lift, spec.Phase.GRASP),
                       (t_lift, t_move, spec.Phase.LIFT),
                       (t_move, t_lower, spec.Phase.MOVE),
                       (t_lower, t_rel, spec.Phase.LOWER),
                       (t_rel, t_verify, spec.Phase.RELEASE),
                       (t_verify, T, spec.Phase.VERIFY)):
        if hi > lo:
            labels[lo:hi] = int(ph)

    labels, fixed = _monotone(labels)
    diag = dict(ticks=T, corrected_by_monotone=int(fixed),
                boundaries=dict(SETTLE=0, REACH=t_lock, GRASP=t_eng, LIFT=t_lift,
                                MOVE=t_move, LOWER=t_lower, RELEASE=t_rel,
                                VERIFY=t_verify),
                engage_tick=t_eng if ever else -1,
                release_tick=t_rel if ever and t_rel < T else -1,
                transitions=int((np.diff(labels.astype(int)) != 0).sum()))
    return labels, diag


def _monotone(raw):
    """Force non-decreasing execution order; return (labels, ticks corrected)."""
    out = raw.copy()
    best = -1
    fixed = 0
    for t in range(len(out)):
        r = RANK.get(int(out[t]), None)
        if r is None:
            out[t] = out[t - 1] if t else int(spec.Phase.SETTLE)
            continue
        if r < best:
            out[t] = ORDER[best]
            fixed += 1
        else:
            best = r
    return out, fixed


def confusion(truth, derived):
    """(matrix, labels) over the phases that appear in either sequence."""
    truth = np.asarray(truth, dtype=int)
    derived = np.asarray(derived, dtype=int)
    present = sorted(set(truth.tolist()) | set(derived.tolist()),
                     key=lambda v: RANK.get(v, -1))
    idx = {v: i for i, v in enumerate(present)}
    m = np.zeros((len(present), len(present)), dtype=int)
    for a, b in zip(truth, derived):
        m[idx[int(a)], idx[int(b)]] += 1
    return m, present


def boundary_offsets(truth, derived):
    """{phase: derived_first_tick - truth_first_tick} for phases in both.

    A boundary that is consistently a few ticks LATE is a property of a
    state-based rule (the state has to move before it can be seen) and is fine.
    One that is early on some episodes and late on others by a lot is not, and
    this is where that shows.
    """
    truth = np.asarray(truth, dtype=int)
    derived = np.asarray(derived, dtype=int)
    out = {}
    for p in ORDER:
        a = np.flatnonzero(truth == int(p))
        b = np.flatnonzero(derived == int(p))
        if len(a) and len(b):
            out[p.name] = int(b[0] - a[0])
    return out
