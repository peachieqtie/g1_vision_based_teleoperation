"""Measurements for the B-prime contact exclusion (2026-09-15). Disclosure numbers,
not gates.

PASS-THROUGH DEPTH - replaces the withdrawn palm-site-height proxy
------------------------------------------------------------------
The withdrawn measure read the palm SITE height below the platform top while over
its footprint. It reads -130 mm in baseline too, because a point under the slab
and a point inside it look the same. The replacement uses the real geoms:

  PRIMARY  `ShadowProbe`: a second copy of the scene compiled WITHOUT the
           exclusion is posed at the live qpos (mj_kinematics + mj_collision -
           no dynamics, no integration), and MuJoCo's own narrow phase reports
           the hand <-> pickup-slab contacts that the exclusion suppressed.
           Depth = -contact.dist, the minimum translation separating the hand
           geom's collision hull from the slab: exactly the contact the physics
           would have generated. A hand UNDER the slab has no overlap and reads
           0. Gated on d.ncon of the shadow data, never mj_geomDistance (TR19).
  CHECK    `vertex_depth`: every vertex of each hand geom (mesh vertices, or a
           box's 8 corners) transformed to world and tested against the slab box.
           Depth = the largest distance from an inside vertex to the nearest slab
           face. Independent of the collision engine; bounded by the slab's
           half-thickness (20 mm) by construction, so it saturates where the
           primary does not. It cannot see a slab edge poking through a hull face
           with no vertex inside, so it may read 0 where the primary reads >0.
           The reverse (vertex inside, no shadow contact) was expected to be
           impossible and is NOT: 0-2 of ~860 probes per teleop run, 2026-09-15,
           magnitude not recorded - presumably grazing contacts inside the
           narrow phase's tolerance. Counted as `shadow_vs_vertex_disagree`.

VALIDATED where the contact is live (exclusion stripped): the shadow depth
equals the live contact depth to 0.000 mm on every compared probe, and
reproduces O25's -7.6 to -19.1 mm hand-slab penetration.

Duration is counted in physics steps at the probe interval and reported in
seconds, both total and longest continuous run.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import mujoco

from g1_teleop.config import ContactConfig
from g1_teleop.contact_contract import HAND_BODIES, load_model

PICKUP_GEOM = "platform_pickup_geom"
GOAL_GEOM = "platform_goal_geom"
BOX_GEOM = "box1_geom"


def _body_geoms(m, names):
    out = {}
    for g in range(m.ngeom):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        if b in names and (m.geom_contype[g] or m.geom_conaffinity[g]):
            out[g] = b
    return out


def hand_geoms(m, side=None):
    """Collision geoms on hand bodies: wrist roll/pitch/yaw links and pads."""
    keys = ("wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link", "pad")
    out = {}
    for g in range(m.ngeom):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            if not b.endswith("pad"):
                continue
        if any(b.endswith(k) for k in keys) and (side is None or b.startswith(side)):
            out[g] = b
    return out


class ShadowProbe:
    def __init__(self, cfg):
        off = dataclasses.replace(cfg, contact=ContactConfig(hand_pickup_exclusion=False))
        self.m = load_model(off)
        self.d = mujoco.MjData(self.m)
        self.plat = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, PICKUP_GEOM)
        # Only the EXCLUDED bodies: anything else collides live and is not hidden.
        self.hands = _body_geoms(self.m, set(HAND_BODIES))
        self.pads = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)
                     for n in ("left_pad_geom", "right_pad_geom")]

    def probe(self, m, d):
        """(max depth m, {body: depth}) of suppressed hand<->pickup overlap at d.qpos."""
        sd = self.d
        sd.qpos[:] = d.qpos
        for g in self.pads:                       # GraspWeld toggles these live
            self.m.geom_contype[g] = m.geom_contype[g]
            self.m.geom_conaffinity[g] = m.geom_conaffinity[g]
        mujoco.mj_kinematics(self.m, sd)
        mujoco.mj_collision(self.m, sd)
        depth, per = 0.0, {}
        for c in range(sd.ncon):
            con = sd.contact[c]
            if self.plat not in (con.geom1, con.geom2):
                continue
            o = con.geom2 if con.geom1 == self.plat else con.geom1
            if o in self.hands:
                dep = max(0.0, -float(con.dist))
                per[self.hands[o]] = max(per.get(self.hands[o], 0.0), dep)
                depth = max(depth, dep)
        return depth, per


class VertexCheck:
    def __init__(self, m):
        self.plat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, PICKUP_GEOM)
        self.local = {}
        for g in _body_geoms(m, set(HAND_BODIES)):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(m.geom_dataid[g])
                a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                v = np.array(m.mesh_vert[a:a + n], dtype=np.float64)
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                s = m.geom_size[g]
                v = np.array([[sx * s[0], sy * s[1], sz * s[2]] for sx in (-1, 1)
                              for sy in (-1, 1) for sz in (-1, 1)])
            else:
                continue
            # Vertices must sit inside the geom's own local AABB; if MuJoCo ever
            # stored them in another frame this fails instead of mis-measuring.
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            assert np.all(np.abs(v - c) <= h + 1e-6), "mesh vertex frame assumption broken"
            self.local[g] = v

    def depth(self, m, d):
        p, R = d.geom_xpos[self.plat], d.geom_xmat[self.plat].reshape(3, 3)
        h = m.geom_size[self.plat]
        best = 0.0
        for g, v in self.local.items():
            w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
            loc = (w - p) @ R
            inside = np.all(np.abs(loc) < h, axis=1)
            if inside.any():
                fd = np.min(h - np.abs(loc[inside]), axis=1)
                best = max(best, float(fd.max()))
        return best


class Duration:
    def __init__(self, dt):
        self.dt, self.total, self.run, self.longest, self.max = dt, 0.0, 0.0, 0.0, 0.0

    def add(self, active, value=0.0):
        if active:
            self.total += self.dt
            self.run += self.dt
            self.longest = max(self.longest, self.run)
            self.max = max(self.max, value)
        else:
            self.run = 0.0


def live_pairs(m, d, a_geoms, b_geoms):
    """(count, max normal force N) of live contacts between two geom sets."""
    n, fmax, ff = 0, 0.0, np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        if (con.geom1 in a_geoms and con.geom2 in b_geoms) or \
           (con.geom2 in a_geoms and con.geom1 in b_geoms):
            n += 1
            mujoco.mj_contactForce(m, d, c, ff)
            fmax = max(fmax, abs(float(ff[0])))
    return n, fmax


def box_bottom_gap(m, d, box_qpos):
    """Sinking into a slab: over the box-geom corners that lie INSIDE a platform's
    xy footprint, the lowest corner z minus that platform's top. None when no
    corner is over a platform. Negative = a corner below the top surface.

    Corners outside the footprint are ignored: a box overhanging the edge, or
    tilted past it, dips a corner below the top without being in the slab (the
    first version counted those and read -12 mm in baseline)."""
    bg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, BOX_GEOM)
    s = m.geom_size[bg]
    corners = np.array([[sx * s[0], sy * s[1], sz * s[2]] for sx in (-1, 1)
                        for sy in (-1, 1) for sz in (-1, 1)])
    w = d.geom_xpos[bg] + corners @ d.geom_xmat[bg].reshape(3, 3).T
    best = None
    for name in (PICKUP_GEOM, GOAL_GEOM):
        pg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        pc, ph = d.geom_xpos[pg], m.geom_size[pg]
        over = (np.abs(w[:, 0] - pc[0]) <= ph[0]) & (np.abs(w[:, 1] - pc[1]) <= ph[1])             & (w[:, 2] > pc[2] - ph[2] - 0.05)
        if over.any():
            gap = float(w[over, 2].min() - (pc[2] + ph[2]))
            best = gap if best is None else min(best, gap)
    return best


def tilt_deg(quat):
    w, x, y, z = quat
    zz = 1.0 - 2.0 * (x * x + y * y)             # body z-axis . world z
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))
