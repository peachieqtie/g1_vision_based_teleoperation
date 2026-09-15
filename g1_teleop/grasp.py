"""Weld-based grasp (D11), replacing the friction pinch.

WHY A WELD
----------
Friction grasping was characterised as infeasible under this project's
constraints, and that characterisation is a reportable negative result rather
than a workaround (CLAUDE.md O2, NOTES.md 2026-09-08). The short version: on a
rig that reproduces the real standing configuration, the pinch holds at 1 of 9
standoffs across `GRASP_MIN..GRASP_MAX`; press force is flat across a 20x range
and pad area across 4x; and holding couples to eight *unrelated* foot-floor
contacts, changing slip eight-fold at constant normal force.

WHY THE TRIGGER IS GEOMETRIC, NOT A FLAG
----------------------------------------
The weld must be something a policy could learn to produce. If it engaged on an
internal flag the policy could not observe or cause, the learning problem would
be a fiction: the policy would appear to grasp without ever having to bring the
hands to the box. So engagement requires the hands to actually be in a grasping
configuration, measured from quantities that are in the state vector (palm site
poses and box pose) and that the policy controls through its arm commands.

The gripper action dimension is the *command*; geometry is the *precondition*.
Commanding 1 with the hands in the wrong place does nothing, exactly as closing
a real gripper on empty air does nothing.

THE CONDITIONS, IN FULL
-----------------------
Engage when ALL of:
  1. command >= `engage_command` (0.5)
  2. each palm site is within `palm_radius` of the box centre
  3. the palms are on OPPOSITE sides: the unit vectors from the box centre to
     each palm have dot product <= `opposed_dot` (negative = opposed)
  4. palm separation lies in [`sep_min`, `sep_max`], i.e. the hands straddle the
     box rather than being collapsed together or spread past it
Release when command < `engage_command`. Release is unconditional: a policy must
always be able to let go.

Thresholds are measured, not guessed - see `NOTES.md` for the distributions they
were taken from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import mujoco


@dataclass(frozen=True)
class GraspConfig:
    """Weld trigger thresholds. See module docstring for what each gates."""
    weld_name: str = "box_grasp"
    left_site: str = "left_palm_site"
    right_site: str = "right_palm_site"
    engage_command: float = 0.5      # gripper action dim above this = "close"
    palm_radius: float = 0.16        # m, palm site to box centre
    opposed_dot: float = -0.50       # palms must be on opposite sides
    sep_min: float = 0.12            # m, hands straddle rather than collapse
    sep_max: float = 0.30            # m, hands straddle rather than overshoot


class GraspWeld:
    """Runtime handle for the box weld on one model/data pair."""

    def __init__(self, model, box_body: str = "box1",
                 cfg: Optional[GraspConfig] = None):
        self.cfg = cfg or GraspConfig()
        self.eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                       self.cfg.weld_name)
        if self.eq_id < 0:
            raise ValueError(f"equality not found: {self.cfg.weld_name}. The "
                             f"weld lives in scene.xml (D11).")
        self.hand_bid = int(model.eq_obj1id[self.eq_id])
        self.box_bid = int(model.eq_obj2id[self.eq_id])
        self.pad_geoms = [g for g in (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("left_pad_geom", "right_pad_geom")) if g >= 0]
        self._pad_contype = [int(model.geom_contype[g]) for g in self.pad_geoms]
        self._pad_conaff = [int(model.geom_conaffinity[g]) for g in self.pad_geoms]
        self.site_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                        self.cfg.left_site)
        self.site_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                        self.cfg.right_site)
        if self.site_l < 0 or self.site_r < 0:
            raise ValueError("palm sites not found; expected "
                             f"{self.cfg.left_site} and {self.cfg.right_site}")

    # ---- geometry -----------------------------------------------------------
    def conditions(self, model, data) -> Tuple[bool, dict]:
        """Are the hands in a grasping configuration? Returns (ok, measurements)."""
        c = self.cfg
        box = data.xpos[self.box_bid]
        pl, pr = data.site_xpos[self.site_l], data.site_xpos[self.site_r]
        dl, dr = float(np.linalg.norm(pl - box)), float(np.linalg.norm(pr - box))
        sep = float(np.linalg.norm(pl - pr))
        ul = (pl - box) / max(dl, 1e-9)
        ur = (pr - box) / max(dr, 1e-9)
        dot = float(ul @ ur)
        m = dict(d_left=dl, d_right=dr, sep=sep, opposed=dot)
        m["near"] = dl <= c.palm_radius and dr <= c.palm_radius
        m["straddle"] = c.sep_min <= sep <= c.sep_max
        m["opposite"] = dot <= c.opposed_dot
        return bool(m["near"] and m["straddle"] and m["opposite"]), m

    # ---- state --------------------------------------------------------------
    def engaged(self, data) -> bool:
        return bool(data.eq_active[self.eq_id])

    def engage(self, model, data) -> None:
        """Attach the box to the hand at its CURRENT relative pose.

        Writing relpose live is what makes engagement jump-free: the constraint
        is already satisfied at the instant it turns on, so the solver has
        nothing to correct.
        """
        hand_p = data.xpos[self.hand_bid]
        hand_q = data.xquat[self.hand_bid]
        box_p = data.xpos[self.box_bid]
        box_q = data.xquat[self.box_bid]

        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, hand_q)                 # inverse of hand rotation
        dp = np.zeros(3)
        mujoco.mju_rotVecQuat(dp, box_p - hand_p, neg)  # box origin in hand frame
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, neg, box_q)           # box orientation in hand frame

        # anchor is the weld point in **body2's own frame** - the box's origin,
        # so zero. Writing `dp` here (the box origin expressed in the HAND's
        # frame) names a point ~160 mm outside the box, and the solver yanks the
        # box to satisfy it: measured as a one-off ~147 mm jump at engage, after
        # which the weld holds rigidly at the displaced offset. That is the
        # residue behind the D11 "drift <= 29 mm" figure. The same convention was
        # measured independently for the base lock (g1_teleop/base_lock.py).
        model.eq_data[self.eq_id][0:3] = 0.0            # anchor, box frame
        model.eq_data[self.eq_id][3:6] = dp             # relpose position
        model.eq_data[self.eq_id][6:10] = rel_q         # relpose orientation
        model.eq_data[self.eq_id][10] = 1.0             # torquescale
        data.eq_active[self.eq_id] = 1
        # Stand the pads down while the weld carries the load. A pad pressing
        # into a rigidly welded box cannot move it, so the contact just
        # accumulates force (measured 123.6 N, insensitive to the pad command
        # because the servo cannot retract within one step). Releasing the weld
        # with that force stored launches the box at ~0.44 m/s. Disabling the
        # pad contact while welded removes the conflict entirely; detection is
        # restored on release, and the trigger only needs it before engagement.
        for g in self.pad_geoms:
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0

    def release(self, model, data) -> None:
        """Let go. The box keeps the velocity it had, so it falls or rests
        naturally — removing a satisfied constraint injects no impulse."""
        data.eq_active[self.eq_id] = 0
        for g, ct, ca in zip(self.pad_geoms, self._pad_contype, self._pad_conaff):
            model.geom_contype[g] = ct
            model.geom_conaffinity[g] = ca

    # ---- the one call the control loop makes --------------------------------
    def update(self, model, data, command: float) -> Tuple[bool, dict]:
        """Apply the gripper command for this tick. Returns (engaged, measurements).

        `command` is the gripper action dimension: 0 = released, 1 = engaged.
        """
        ok, m = self.conditions(model, data)
        want = float(command) >= self.cfg.engage_command
        if want and ok and not self.engaged(data):
            self.engage(model, data)
        elif not want and self.engaged(data):
            self.release(model, data)
        m["command"] = float(command)
        m["gated"] = bool(want and not ok and not self.engaged(data))
        return self.engaged(data), m
