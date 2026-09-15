"""Event-driven base lock (D12), as an observable predicate over the 47-D state.

WHY THE LOCK CANNOT BE A PHASE TRANSITION
-----------------------------------------
The scripted demonstrator used to lock and release the base on membership of a
phase set (`LOCKED_PHASES_WALKING`). Deployment has no phase machine, and
neither does teleoperated collection - the human supplies the trajectory, not a
schedule. If the lock were keypress-triggered during collection and
predicate-triggered at deployment, the demonstrations would show lock timings
the policy never sees.

So the lock fires from quantities that are IN the state vector, on the same
principle as D11's geometric weld gate: something the policy causes and can
observe. Every term below is a `g1_data.spec` accessor plus two fixed platform
constants (Q6/TR13: the platforms never move).

MEASURED, NOT GUESSED
---------------------
Thresholds come from NOTES.md "2026-09-10 Base-lock predicate (D12)", 12
episodes of the walking demonstrator at 25 Hz, all quantities read from the
STEPPED model (TR14). The margins measured at the four transitions were:

    disp        +9.8 mm     fwd (pickup)  +66.7 mm    fwd (goal)  +19.1 mm
    lat        +37.0 mm     head          +2.71 deg   to_goal    +1.278 m
    box_lift    +6.4 mm     |box_lift|    +19.9 mm    palm_far   +77.8 mm

THE SETTLED TEST IS A GAIT-PERIOD TEST
--------------------------------------
"Is the base still?" cannot be answered over an arbitrary window. TR2: the
locomotion policy IS the balance controller and its in-place march never stops.
Measured over 0.52 s (0.65 of a gait period) the march's net displacement stayed
under 10 mm on only ~30% of settled ticks, with runs of up to 19 consecutive
ticks above it. Over one whole `GAIT_PERIOD` the limit cycle cancels and the
same test holds on 98.7%. `GAIT_PERIOD = 0.8` is fixed inside the network (TR4),
so `settle_window = 20` ticks at 25 Hz is a constant, not a tuned value.

Even then the term is not monotone, and TR8 applies (a single-frame stillness
threshold is not a stillness test): the predicate is debounced over 5 consecutive
ticks, and the lock LATCHES. LOCK is only tested while free, RELEASE only while
locked. It is an edge trigger, never a continuously re-evaluated condition.

WHY THE "TASK ALREADY DONE" GUARD IS NOT OPTIONAL
-------------------------------------------------
At the first lock the robot is settled at a standoff from a box at rest with the
hands far away - which is *also* exactly what RELEASE's placed-and-withdrawn
branch describes. And after the final release the box sits at the goal standoff
in front of a settled robot, which is what LOCK's geometry describes. Without
`to_goal` each predicate fires in the other's situation. `to_goal <= 0.10` is the
Q4 placement tolerance, i.e. "is the box where the task wants it" - the one
distinction that separates them. It costs one scene constant.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import mujoco

from g1_data import spec


@dataclass(frozen=True)
class PlatformGeometry:
    """The two platforms and the box, resolved from the model by name.

    Fixed by Q6/TR13 - only the box POSITION is randomised, never the platforms -
    so these are legitimate constants for a predicate to reference. Resolved
    rather than hardcoded for the reason ModelIndex exists: a scene edit must
    not silently leave a stale number behind (O3, O22).
    """

    pickup_xy: np.ndarray
    pickup_half: np.ndarray
    goal_xy: np.ndarray
    goal_half: np.ndarray
    top_z: float
    box_half: float

    @property
    def rest_z(self) -> float:
        """Box centre height when the box rests on a platform."""
        return self.top_z + self.box_half

    @classmethod
    def resolve(cls, model, box_geom: str = "box1_geom") -> "PlatformGeometry":
        def plat(geom: str):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            if gid < 0:
                raise ValueError(f"platform geom not found: {geom}")
            bid = int(model.geom_bodyid[gid])
            return (np.array(model.body_pos[bid][:2], dtype=np.float64),
                    np.array(model.geom_size[gid][:2], dtype=np.float64),
                    float(model.body_pos[bid][2] + model.geom_size[gid][2]))

        p_xy, p_half, p_top = plat("platform_pickup_geom")
        g_xy, g_half, g_top = plat("platform_goal_geom")
        if abs(p_top - g_top) > 1e-9:
            raise AssertionError(
                f"the two platform tops differ ({p_top} vs {g_top}); the lock "
                f"predicate and the clearance metric assume one resting height")
        bgid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, box_geom)
        if bgid < 0:
            raise ValueError(f"box geom not found: {box_geom}")
        return cls(pickup_xy=p_xy, pickup_half=p_half, goal_xy=g_xy,
                   goal_half=g_half, top_z=p_top,
                   box_half=float(model.geom_size[bgid][2]))

    def clearance(self, box_pos) -> float:
        """Box bottom above a platform top, or NaN when over neither.

        NaN, not a large number: between the platforms the box is over the
        floor and there is nothing to scrape, which is a different statement
        from "it is high above the platform". A minimum taken over an episode
        must skip those ticks, not be dominated by them.
        """
        b = np.asarray(box_pos, dtype=np.float64)
        for ctr, half in ((self.pickup_xy, self.pickup_half),
                          (self.goal_xy, self.goal_half)):
            if (abs(b[0] - ctr[0]) <= half[0] + self.box_half
                    and abs(b[1] - ctr[1]) <= half[1] + self.box_half):
                return float(b[2] - self.box_half - self.top_z)
        return float("nan")


@dataclass(frozen=True)
class LockConfig:
    """Thresholds, exactly as measured 2026-09-10. Do not re-tune casually.

    Each one has a measured margin at the transition it governs (module
    docstring). `lift_min` is the only one tied to another config value: it is
    a fraction of what the arm ACHIEVES of `DemoConfig.lift_h`, so changing the
    commanded lift means re-measuring the achieved lift and re-checking this.
    """

    settle_window: int = 20        # ticks: 0.80 s = one GAIT_PERIOD (TR4)
    debounce: int = 5              # ticks: 0.20 s (TR8)
    disp_max: float = 0.010        # m of base XY travel per gait period
    fwd_lo: float = 0.26           # m, standoff band the arms can serve (O14)
    fwd_hi: float = 0.40
    lat_max: float = 0.05          # m, the head-on band grasping works in (O18)
    head_max_deg: float = 5.0
    lift_min: float = 0.055        # m above resting: carrying, hand to the legs
    rest_tol: float = 0.020        # m: the box is down
    withdraw: float = 0.25         # m, worst palm to box centre: hands clear
    done_radius: float = 0.10      # m, the Q4 placement tolerance


class LockPredicate:
    """Two-state latch driven by the 47-D state. One call per 25 Hz tick.

    Usage, from the control loop::

        pred = LockPredicate(geo)
        ...
        event = pred.update(state)          # "lock" | "release" | None
        if event == "lock":   base_lock.lock(model, data)
        elif event == "release": base_lock.release(model, data, policy)

    `state` must be the spec 47-D vector built from the STEPPED model. The
    predicate never sees a phase, a tick index or an elapsed time, by
    construction: nothing in this class has access to one.
    """

    def __init__(self, geo: PlatformGeometry, cfg: Optional[LockConfig] = None):
        self.geo = geo
        self.cfg = cfg or LockConfig()
        self.reset()

    def reset(self) -> None:
        """Per-episode state. Every field here leaks into the next episode if
        forgotten - see g1_data/reset.py's docstring for why that matters."""
        self.locked = False
        # maxlen == the window, so _hist[0] is exactly `settle_window`
        # ticks back at the moment `terms` reads it (the current tick is
        # appended after). One gait period, not one plus a tick.
        self._hist = deque(maxlen=self.cfg.settle_window)
        self._run = 0
        self.n_lock = 0
        self.n_release = 0

    # ---- terms ---------------------------------------------------------
    def terms(self, state) -> dict:
        """Every quantity the predicate tests, for diagnostics and logging."""
        c, g = self.cfg, self.geo
        box = spec.box_pos(state)
        base = spec.base_pos(state)
        q = spec.base_quat(state)
        yaw = float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                               1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])))
        welded = float(spec.grip_left(state)) >= 0.5
        target = g.goal_xy if welded else box[:2]
        dx, dy = target[0] - base[0], target[1] - base[1]
        cy, sy = np.cos(yaw), np.sin(yaw)
        palm_far = max(
            float(np.linalg.norm(spec.palm_left_pos(state) - box)),
            float(np.linalg.norm(spec.palm_right_pos(state) - box)))
        bearing = float(np.arctan2(dy, dx))
        return dict(
            welded=welded,
            fwd=float(cy * dx + sy * dy),
            lat=float(-sy * dx + cy * dy),
            head=float(np.degrees(np.arctan2(np.sin(bearing - yaw),
                                             np.cos(bearing - yaw)))),
            disp=self._disp(base[:2]),
            box_lift=float(box[2] - g.rest_z),
            to_goal=float(np.linalg.norm(box[:2] - g.goal_xy)),
            palm_far=palm_far,
        )

    def _disp(self, base_xy) -> float:
        """Net base travel over one gait period, or inf before the window fills.

        Infinite, not zero: a "settled" test that is trivially true on the first
        frame of an episode is a bug, and this one would fire before the robot
        had taken a single step.
        """
        if len(self._hist) < self._hist.maxlen:
            return float("inf")
        return float(np.linalg.norm(np.asarray(base_xy) - self._hist[0]))

    def lock_ok(self, t: dict) -> bool:
        c = self.cfg
        if (not t["welded"]) and t["to_goal"] <= c.done_radius:
            return False                    # the task is already finished
        return bool(t["disp"] <= c.disp_max
                    and c.fwd_lo <= t["fwd"] <= c.fwd_hi
                    and abs(t["lat"]) <= c.lat_max
                    and abs(t["head"]) <= c.head_max_deg)

    def release_ok(self, t: dict) -> bool:
        c = self.cfg
        if t["welded"]:
            return bool(t["box_lift"] >= c.lift_min)      # carrying: legs take over
        return bool(t["to_goal"] <= c.done_radius
                    and abs(t["box_lift"]) <= c.rest_tol
                    and t["palm_far"] >= c.withdraw)      # placed and withdrawn

    # ---- the one call the control loop makes ---------------------------
    def update(self, state) -> Optional[str]:
        """Advance one 25 Hz tick. Returns "lock", "release" or None."""
        t = self.terms(state)
        self._hist.append(np.asarray(spec.base_pos(state)[:2], dtype=np.float64))
        want = self.release_ok(t) if self.locked else self.lock_ok(t)
        self._run = self._run + 1 if want else 0
        if self._run < self.cfg.debounce:
            return None
        self._run = 0
        self.locked = not self.locked
        if self.locked:
            self.n_lock += 1
            return "lock"
        self.n_release += 1
        return "release"
