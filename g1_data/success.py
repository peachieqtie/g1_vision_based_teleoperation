"""The four phase-level success criteria (proposal 3.8.2), from RECORDED data.

Everything here is a pure function of one episode .npz - `states`, `qvel` and the
metadata - plus scene geometry that Q6/TR13 fix. Nothing reads a live model, so
an episode recorded today can be re-scored months from now, and a scoring bug can
be fixed without re-collecting.

TWO OF THE PROPOSAL'S FOUR CRITERIA NEEDED WORK (D19, D20)
----------------------------------------------------------
3.8.2 scores a grasp as "both gripper states closed AND box above the platform
surface". The replay negative control (NOTES.md 2026-09-16) produced a
counter-example from real physics: with the box displaced 235 mm, the arm SHOVED
it 185 mm into its own palms, the weld gate conditions were then genuinely met,
and the episode wrecked - the box ended 3.36 m away. Both of the proposal's
conditions held. A policy that bulldozes the box into its hands would score a
grasp success, and "bulldoze until the gate fires" is exactly the kind of
degenerate strategy behavioural cloning can learn from a demonstrator it only
half-imitates.

So a third condition is added: at the instant of engage, the box must still be
near WHERE IT SPAWNED. `box_spawn_xy` is in every episode file.

    measured, 40 scripted episodes: pre-grasp box disturbance 0.1 - 2.4 mm
    threshold: 50 mm  (>20x the measured healthy maximum, and 4x below the
                       185 mm shove that motivated it)

The margin is deliberately enormous in both directions: it cannot fire on a
healthy episode, and it cannot miss a bulldoze. It is a fault detector, not a
quality metric.

THE LIFT CRITERION IS VACUOUS AS WRITTEN (D20)
----------------------------------------------
3.8.2 scores a lift as "box height exceeds platform + 0.05 m". Read as the box
CENTRE - the obvious reading, and the one the 47-D state hands you - a 0.18 m box
resting untouched on the platform is already 0.09 m above it, so the criterion is
satisfied before the robot does anything. A deliberately injected "welded but
never raised" episode passed it. Scored here on the box BOTTOM, which is
equivalent to 0.05 m above its resting height and consistent with
`LockConfig.lift_min` (0.055 m, the demonstrator's own "carrying" test).

WHAT IS NOT DECIDED HERE
------------------------
These are EPISODE-LEVEL OUTCOME criteria, exactly as 3.8.2 specifies. Which
BUCKET a failed episode is charged to is failure attribution and lives in
`spec.SCORED_OF` (see `attribute`); adding WALK_IN there changed the taxonomy and
not the metric set, and the same is true here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from g1_data import spec

# ─── scene geometry. Fixed by Q6/TR13; resolved from the model by name so a
# scene edit cannot leave a stale number behind (O3), cached because scoring a
# dataset should not load the model once per episode.
_GEO = {}


def geometry(model=None) -> dict:
    if not _GEO or model is not None:
        import mujoco
        from g1_teleop.config import TeleopConfig
        from g1_teleop.contact_contract import load_model
        m = model if model is not None else load_model(TeleopConfig())
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "platform_pickup_geom")
        bgid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box1_geom")
        ggid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "platform_goal_geom")
        bid = int(m.geom_bodyid[gid])
        _GEO.update(
            top_z=float(m.body_pos[bid][2] + m.geom_size[gid][2]),
            box_half=float(m.geom_size[bgid][2]),
            goal_xy=np.array(m.body_pos[int(m.geom_bodyid[ggid])][:2], dtype=float))
        _GEO["rest_z"] = _GEO["top_z"] + _GEO["box_half"]
    return dict(_GEO)


@dataclass(frozen=True)
class SuccessConfig:
    """Thresholds, each with its provenance. Q4 and 3.8.2 fix the first three."""
    lift_above_platform: float = 0.05     # 3.8.2: "platform + 0.05 m" (D20: of the
                                          # box BOTTOM - see the module docstring)
    place_radius: float = 0.10            # Q4 / d_place
    upright_deg: float = 15.0             # Q4 "roughly upright"; measured max 10.9
    # "Resting" is a CONTACT test in the demonstrator (scripted_demo.py:1243), and
    # a recorded file has no contacts - so it is reconstructed geometrically here.
    # Both thresholds come from the 40-episode healthy population: final
    # |z - rest| max 18.3 mm, final box speed max 0.060 m/s.
    resting_tol: float = 0.03             # m from resting height (1.6x measured max)
    settled_speed: float = 0.25           # m/s (4x measured max)
    floor_clear_z: float = 0.30           # m: healthy carry >= 0.84, floor rest 0.09
    grasp_spawn_max: float = 0.05         # D19, see the module docstring
    engage_above_platform: float = -0.01  # 3.8.2 "above the platform surface"


def _tilt_deg(quat) -> np.ndarray:
    """Angle between the box z-axis and world up, per tick."""
    q = np.asarray(quat, dtype=np.float64)
    w, x, y = q[..., 0], q[..., 1], q[..., 2]
    zz = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(zz, -1.0, 1.0)))


def evaluate(arrays: dict, meta: dict, cfg: Optional[SuccessConfig] = None,
             geo: Optional[dict] = None) -> dict:
    """The four criteria for one recorded episode, with the numbers behind them."""
    c = cfg or SuccessConfig()
    g = geo or geometry()
    s = np.asarray(arrays["states"], dtype=np.float64)
    box = s[:, spec.BOX_POS]
    welded = s[:, spec.GRIP_L] >= 0.5
    spawn = np.asarray(meta["box_spawn_xy"], dtype=np.float64)
    T = len(s)

    eng = int(np.argmax(welded)) if welded.any() else -1
    out = dict(engage_tick=eng, n_ticks=T)

    # ---- 1. GRASP (3.8.2 + D19) ----------------------------------------
    if eng < 0:
        out["grasp"] = False
        out["grasp_detail"] = dict(reason="the weld never engaged")
    else:
        above = float(box[eng, 2] - g["rest_z"]) >= c.engage_above_platform
        moved = float(np.linalg.norm(box[eng, :2] - spawn))
        near_spawn = moved <= c.grasp_spawn_max
        out["grasp"] = bool(above and near_spawn)
        out["grasp_detail"] = dict(
            gripper_closed=True, box_above_platform=bool(above),
            box_moved_from_spawn_m=moved, box_near_spawn=bool(near_spawn),
            reason="" if (above and near_spawn) else
            ("the box was not on the platform at engage" if not above else
             "the box had been pushed %.3f m from its spawn before the weld fired "
             "(D19: the proposal's two conditions both held)" % moved))

    # ---- 2. LIFT -------------------------------------------------------
    after = slice(eng, T) if eng >= 0 else slice(0, 0)
    # D20: the box BOTTOM, not its centre. Read as the centre, "box height
    # exceeds platform + 0.05 m" is VACUOUS - a 0.18 m box resting untouched on
    # the platform already has its centre 0.09 m above it, so every episode
    # scores a lift, including one where the box never moved. Demonstrated by the
    # `never_lifted` fault injection, which the centre reading did not catch.
    lift_peak = (float(box[after, 2].max() - g["box_half"] - g["top_z"])
                 if eng >= 0 else float("nan"))
    out["lift"] = bool(out["grasp"] and lift_peak >= c.lift_above_platform)
    out["lift_detail"] = dict(peak_above_platform_m=lift_peak,
                              required_m=c.lift_above_platform)

    # ---- 3. TRANSPORT --------------------------------------------------
    # "the box remains above floor level throughout" - between the grasp and the
    # end. A dropped box rests at box_half = 0.09 m; a carried one is at 0.84+.
    min_z = float(box[after, 2].min()) if eng >= 0 else float("nan")
    out["transport"] = bool(out["lift"] and min_z >= c.floor_clear_z)
    out["transport_detail"] = dict(min_box_z_m=min_z, floor_clear_z=c.floor_clear_z)

    # ---- 4. PLACE (Q4) -------------------------------------------------
    final = box[-1]
    place_err = float(np.linalg.norm(final[:2] - g["goal_xy"]))
    tilt = float(_tilt_deg(s[-1, spec.BOX_QUAT]))
    resting = abs(float(final[2]) - g["rest_z"]) <= c.resting_tol
    speed = float("nan")
    if "qvel" in arrays:
        # The box free joint leads qvel; 3 linear dims. "Resting" means it has
        # stopped, not that it happens to pass through the right height.
        qv = np.asarray(arrays["qvel"], dtype=np.float64)
        speed = float(np.linalg.norm(qv[-1, -6:-3]))
        resting = resting and speed <= c.settled_speed
    out["place"] = bool(out["transport"] and place_err <= c.place_radius
                        and resting and tilt <= c.upright_deg)
    out["place_detail"] = dict(placement_error_m=place_err, tilt_deg=tilt,
                               resting=bool(resting), final_box_speed=speed,
                               limit_m=c.place_radius, upright_limit_deg=c.upright_deg)
    out["episode_success"] = bool(out["place"])
    out["attributed_to"] = attribute(out, arrays)
    return out


def attribute(result: dict, arrays: dict) -> str:
    """Which `spec.ScoredPhase` a FAILED episode is charged to.

    The first criterion that fails names the bucket, except that a failure while
    the robot was still walking in is charged to WALK_IN - which is the whole
    reason that bucket exists (CLAUDE.md section 8). The phase column decides
    that, not a guess about the cause.
    """
    if result["episode_success"]:
        return ""
    labels = np.asarray(arrays.get("phase_labels", []), dtype=int)
    if not result["grasp"]:
        if len(labels) and result["engage_tick"] < 0:
            # It never grasped: charge it where it spent its last tick.
            last = int(labels[-1])
            if last in (int(spec.Phase.SETTLE),):
                return spec.ScoredPhase.WALK_IN.name
        return spec.ScoredPhase.GRASP.name
    for name, bucket in (("lift", spec.ScoredPhase.LIFT),
                         ("transport", spec.ScoredPhase.TRANSPORT),
                         ("place", spec.ScoredPhase.PLACE)):
        if not result[name]:
            return bucket.name
    return ""


def summarize(results) -> dict:
    """Rates over a set of episodes, in the proposal's four columns."""
    n = len(results)
    out = {k: (sum(bool(r[k]) for r in results), n)
           for k in ("grasp", "lift", "transport", "place")}
    buckets = {}
    for r in results:
        if r["attributed_to"]:
            buckets[r["attributed_to"]] = buckets.get(r["attributed_to"], 0) + 1
    out["failures_by_bucket"] = buckets
    return out


# ─── deliberate fault injection, for validating the detector ─────────────────
# These edit a LOADED episode in memory and never touch a file. They exist
# because a success detector that has only ever seen successes is untested: the
# 40 scripted episodes are all known-good, so the failures have to be made.
def sabotage(arrays: dict, meta: dict, kind: str):
    a = {k: v.copy() for k, v in arrays.items()}
    m = dict(meta)
    s = a["states"]
    geo = geometry()
    if kind == "no_weld":                       # the weld never fires
        s[:, spec.GRIP_L] = 0.0
        s[:, spec.GRIP_R] = 0.0
    elif kind == "box_dropped":                 # dropped to the floor mid-carry
        half = len(s) // 2
        s[half:, spec.BOX_POS.start + 2] = geo["box_half"]
    elif kind == "placement_out":               # placed 0.35 m off the goal
        s[-1, spec.BOX_POS.start + 1] += 0.35
    elif kind == "tipped":                      # placed on its side
        s[-1, spec.BOX_QUAT] = np.array([0.7071, 0.7071, 0.0, 0.0])
    elif kind == "bulldozed":                   # D19: box shoved into the palms
        eng = int(np.argmax(s[:, spec.GRIP_L] >= 0.5))
        s[:eng + 1, spec.BOX_POS.start + 1] += 0.185
    elif kind == "never_lifted":                # welded but never raised
        eng = int(np.argmax(s[:, spec.GRIP_L] >= 0.5))
        s[eng:, spec.BOX_POS.start + 2] = geo["rest_z"]
    else:
        raise ValueError("unknown sabotage %r" % kind)
    return a, m


SABOTAGE_KINDS = ("no_weld", "box_dropped", "placement_out", "tipped",
                  "bulldozed", "never_lifted")
#: What each injected fault must cost, so the test asserts the RIGHT failure and
#: not merely "something failed".
SABOTAGE_EXPECT = dict(no_weld="grasp", box_dropped="transport",
                       placement_out="place", tipped="place",
                       bulldozed="grasp", never_lifted="lift")
