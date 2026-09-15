"""O25 PARTIAL: run the teleop path under STEPPED physics, with no camera.

    python tools/teleop_physics_check.py            # everything
    python tools/teleop_physics_check.py human      # (b) only
    python tools/teleop_physics_check.py replay     # (a) continuous (see caveat)
    python tools/teleop_physics_check.py replay2    # (a) static per-pose + pre-MOVE

`replay` is kept for completeness but its numbers are NOT citable: its "settled"
window is one pose measured 122 times (TR18), and it replays the LOWER/RELEASE
poses - executed by the demonstrator at the GOAL - against the pickup platform.
`replay2` is the measurement.

WHAT THIS RUNS, AND HOW IT DIFFERS FROM run_integrated_combined.py
------------------------------------------------------------------
The stack below the camera is the real one, unmodified: `SyntheticSource` ->
`TeleopController` (One-Euro, depth low-pass, stillness lock, coast, arm_alpha)
-> `G1Robot` twin -> `solve_arm_ik` -> twin qpos copied into
`ctrl[ix.upper_ctrl]` on the STEPPED model, physics at 500 Hz, gravity on, the
weld available, and - in the free-base runs - the pre-trained locomotion policy
at 50 Hz.

It is a HARNESS, not that entry point, and the differences are stated (TR14).
`run_integrated_combined.py` is not at parity with collection either. This:

  * parks the base at the working standoff from the box. The entry point leaves
    it at the keyframe ~1.2 m away, where the hands can never reach the platform
    and the question this run exists to answer cannot arise;
  * runs each motion twice - base LOCKED (D12, how collection will run) and
    base FREE with the locomotion policy live (how the entry point runs today,
    and the only honest test of TR15);
  * holds the pads at zero (D11) instead of driving them with the grasp command
    (the TR17 bug at run_integrated_combined.py:296);
  * feeds one frame per 17 physics steps (29.4 Hz), matching the `euro_freq =
    30` the One-Euro filter is tuned for, instead of an async camera thread;
  * has no UI, renderer or keyboard.

MEASUREMENT RULES
-----------------
Everything is read from the stepped model. No twin IK residual is reported
anywhere (TR16a) - "is the pose reachable" is answered by comparing the
retargeted TARGET the IK was asked for against where the stepped arm actually
ended up, both expressed in their own pelvis frame. Contacts are gated on
`d.ncon` + `mj_contactForce`, never `mj_geomDistance` (TR19). The wedge detector
is the wrist PITCH pair against 0.8 rad (TR23); a wedge that persists after the
stream holds its last pose for 1.5 s is the permanent kind that destroys data.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g1_teleop import config as C
from g1_teleop import teleop as TP
from g1_teleop.base_lock import BaseLock
from g1_teleop.box_reset import reset_box
from g1_teleop.config import TeleopConfig
from g1_teleop.grasp import GraspWeld
from g1_teleop.indices import ModelIndex
from g1_teleop.robot import G1Robot
from g1_teleop.synthetic_source import (SyntheticSource, arm_dirs_from_hand,
                                        keypoints_from_arm_dirs, replay_frames,
                                        with_dropout)
from g1_teleop.teleop import TeleopController

WEDGE_RAD = 0.8                  # TR23, set from measured populations
STEPS_PER_FRAME = 17             # 500 Hz / 17 = 29.4 Hz, matches euro_freq=30
HOLD_FRAMES = 44                 # ~1.5 s holding the last pose, to tell a
                                 # transient wedge from a permanent one
PITCH = ("left_wrist_pitch_joint", "right_wrist_pitch_joint")
ALL_WRIST = ("left_wrist_roll_joint", "left_wrist_pitch_joint",
             "left_wrist_yaw_joint", "right_wrist_roll_joint",
             "right_wrist_pitch_joint", "right_wrist_yaw_joint")


# ─── the IK-target tap: what the IK was ASKED for, never what it claims ───────
_TARGETS = {}
_real_solve = TP.solve_arm_ik


def _tap_solve(model, data, el_body, wr_body, el_target, wr_target, *a, **k):
    side = "left" if wr_body == _TARGETS.get("_left_wr_body") else "right"
    _TARGETS[side] = (np.array(el_target, float), np.array(wr_target, float))
    return _real_solve(model, data, el_body, wr_body, el_target, wr_target,
                       *a, **k)


TP.solve_arm_ik = _tap_solve


def _yaw(quat):
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _in_pelvis(vec_world, pel_world, yaw):
    c, s = np.cos(-yaw), np.sin(-yaw)
    v = np.asarray(vec_world, float) - np.asarray(pel_world, float)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


class Rig:
    """Stepped scene, base parked at the working standoff from the box."""

    def __init__(self, cfg: TeleopConfig, seed: int = 0, standoff: float = 0.32,
                 filter_hands=False):
        self.cfg = cfg
        m = self.m = mujoco.MjModel.from_xml_path(cfg.model_path)
        m.opt.timestep = cfg.loco.sim_dt
        d = self.d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        ix = self.ix = ModelIndex.resolve(m)
        self.DEF = np.asarray(cfg.loco.default_angles, dtype=np.float64)
        self.KPS = np.asarray(cfg.loco.kps, dtype=np.float32)
        self.KDS = np.asarray(cfg.loco.kds, dtype=np.float32)
        d.qpos[ix.leg_qpos] = self.DEF
        d.qvel[ix.leg_qvel] = 0.0
        self.box0 = reset_box(m, d, ix, cfg.box, seed=seed)
        d.qpos[ix.base_qpos][0] = min(self.box0[0] - standoff, cfg.grasp.max_base_x)
        d.qpos[ix.base_qpos][1] = float(self.box0[1])
        mujoco.mj_forward(m, d)
        if filter_hands:
            # Candidate B, BEFORE GraspWeld saves the pad bitmasks. `True` means
            # both platforms; "pickup" means the pickup platform only (B-prime).
            from g1_teleop.contact_filter import apply_hand_platform_filter
            if filter_hands == "pickup":
                apply_hand_platform_filter(m, ("platform_pickup_geom",))
            else:
                apply_hand_platform_filter(m)
        self.weld = GraspWeld(m)
        self.lock = BaseLock(m)
        self.pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.plat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                      "platform_pickup_geom")
        self.box_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box1_geom")
        self.hand_g, self.arm_g, self.body_g = set(), set(), set()
        for g in range(m.ngeom):
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
            if not b or int(m.geom_contype[g]) == 0:
                continue
            if "wrist" in b or "pad" in b or "elbow" in b:
                self.hand_g.add(g)
            if "shoulder" in b or "elbow" in b or "wrist" in b or "pad" in b:
                self.arm_g.add(g)
            if b in ("torso_link", "pelvis"):
                self.body_g.add(g)
        self.jnt = {}
        for n in ALL_WRIST:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            self.jnt[n] = (int(m.jnt_qposadr[j]), int(a), int(m.jnt_dofadr[j]),
                           float(m.jnt_actfrcrange[j][1]))
        self.bodies = {s: (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                             getattr(C, s.upper() + "_ELBOW_BODY")),
                           mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                             getattr(C, s.upper() + "_WRIST_BODY")))
                       for s in ("left", "right")}

    def shoulder_frame(self):
        """Robot shoulders, box, and limb lengths, all in the pelvis frame."""
        m, d = self.m, self.d
        yaw = _yaw(d.qpos[self.ix.base_quat_qpos])
        pel = d.xpos[self.pel]
        out = {}
        for s in ("left", "right"):
            sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                   getattr(C, s.upper() + "_SHOULDER_BODY"))
            eb, wb = self.bodies[s]
            out[s] = dict(
                shoulder=_in_pelvis(d.xpos[sb], pel, yaw),
                upper=float(np.linalg.norm(d.xpos[eb] - d.xpos[sb])),
                fore=float(np.linalg.norm(d.xpos[wb] - d.xpos[eb])))
        for s_ in ("left", "right"):
            out[s_]["home_hand"] = _in_pelvis(d.xpos[self.bodies[s_][1]], pel, yaw)
        out["box"] = _in_pelvis(d.qpos[self.ix.box_qpos][:3], pel, yaw)
        out["plat_edge_fwd"] = float(
            (self.m.body_pos[self.m.geom_bodyid[self.plat]][0]
             - self.m.geom_size[self.plat][0]) - pel[0])
        return out


def run_stream(rig: Rig, frames, policy=None, locked=True):
    """Feed a keypoint stream through the full stack under stepped physics."""
    cfg, m, d, ix = rig.cfg, rig.m, rig.d, rig.ix
    twin = G1Robot(cfg)
    _TARGETS.clear()
    _TARGETS["_left_wr_body"] = twin.left_wrist_body
    ctl = TeleopController(twin, cfg)
    ctl.reset()
    twin_ix = ModelIndex.resolve(twin.model)
    twin_pel = mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    frames = list(frames) + [frames[-1]] * HOLD_FRAMES
    src = SyntheticSource(frames)
    n_stream = len(frames) - HOLD_FRAMES

    if locked:
        rig.lock.lock(m, d)
    L = cfg.loco
    import torch
    import walk_test as W
    action = np.zeros(L.num_actions, dtype=np.float32)
    target_leg = rig.DEF.copy()
    obs = np.zeros(L.num_obs, dtype=np.float32)
    arm_targets = np.array(d.ctrl[ix.upper_ctrl], dtype=np.float64)
    ff = np.zeros(6)
    r = dict(applied=0, frozen=0, plat_steps=0, plat_force=0.0, plat_depth=0.0,
             box_steps=0, torso_steps=0, torso_force=0.0,
             pitch_max=0.0, pitch_joint="", pitch_end=0.0, wrist_any=0.0,
             sat_frames=0, fell=False, min_base_z=9.9, max_pitch_deg=0.0,
             base_travel=0.0, target_err=[], targets=[], frame_i=0,
             plat_per_frame=[], pitch_per_frame=[], torso_per_frame=[])
    base0 = np.array(d.qpos[ix.base_qpos][:2], float)
    counter = 0
    fi = 0
    while True:
        frame = src.grab()
        if frame is None:
            break
        out = ctl.step(frame)
        r["applied" if out.applied else "frozen"] += 1
        arm_targets = np.array(twin.data.qpos[twin_ix.upper_qpos], dtype=np.float64)
        r["targets"].append(arm_targets.copy())
        tw_tgt = {s: _TARGETS.get(s) for s in ("left", "right")}
        twin_pel_w = np.array(twin.data.xpos[twin_pel], float)
        for _ in range(STEPS_PER_FRAME):
            lq, ldq = d.qpos[ix.leg_qpos], d.qvel[ix.leg_qvel]
            tgt = rig.DEF if locked else target_leg
            d.ctrl[ix.leg_ctrl] = (tgt - lq) * rig.KPS + (0.0 - ldq) * rig.KDS
            d.ctrl[ix.upper_ctrl] = arm_targets
            d.ctrl[ix.pad_ctrl] = 0.0
            mujoco.mj_step(m, d)
            counter += 1
            for c in range(d.ncon):
                g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                hand = g1 in rig.hand_g or g2 in rig.hand_g
                arm = g1 in rig.arm_g or g2 in rig.arm_g
                if hand and rig.plat in (g1, g2):
                    r["plat_steps"] += 1
                    r["plat_depth"] = min(r["plat_depth"], float(d.contact[c].dist))
                    mujoco.mj_contactForce(m, d, c, ff)
                    r["plat_force"] = max(r["plat_force"], abs(float(ff[0])))
                elif hand and rig.box_g in (g1, g2):
                    r["box_steps"] += 1
                elif arm and (g1 in rig.body_g or g2 in rig.body_g):
                    r["torso_steps"] += 1
                    mujoco.mj_contactForce(m, d, c, ff)
                    r["torso_force"] = max(r["torso_force"], abs(float(ff[0])))
            if not locked and policy is not None and counter % L.control_decimation == 0:
                n = L.num_actions
                phz = ((counter * L.sim_dt) % L.gait_period) / L.gait_period
                obs[:3] = d.qvel[ix.base_angvel_qvel] * L.ang_vel_scale
                obs[3:6] = W.get_gravity_orientation(d.qpos[ix.base_quat_qpos])
                obs[6:9] = 0.0
                obs[9:9 + n] = (d.qpos[ix.leg_qpos] - rig.DEF) * L.dof_pos_scale
                obs[9 + n:9 + 2 * n] = d.qvel[ix.leg_qvel] * L.dof_vel_scale
                obs[9 + 2 * n:9 + 3 * n] = action
                obs[9 + 3 * n:9 + 3 * n + 2] = [np.sin(2 * np.pi * phz),
                                                np.cos(2 * np.pi * phz)]
                action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
                target_leg = action * L.action_scale + rig.DEF

        # ---- per-frame, stepped state ------------------------------------
        r["plat_per_frame"].append(r["plat_steps"])
        r["torso_per_frame"].append(r["torso_steps"])
        r["pitch_per_frame"].append(max(abs(float(d.qpos[rig.jnt[q][0]]
                                                  - d.ctrl[rig.jnt[q][1]]))
                                        for q in PITCH))
        for nm, (qa, aa, dof, lim) in rig.jnt.items():
            e = abs(float(d.qpos[qa] - d.ctrl[aa]))
            r["wrist_any"] = max(r["wrist_any"], e)
            if nm in PITCH and e > r["pitch_max"]:
                r["pitch_max"], r["pitch_joint"] = e, nm
        if any(abs(float(d.qfrc_actuator[v[2]])) >= 0.999 * v[3]
               for v in rig.jnt.values()):
            r["sat_frames"] += 1
        yaw = _yaw(d.qpos[ix.base_quat_qpos])
        pel = d.xpos[rig.pel]
        if fi < n_stream and all(tw_tgt.values()):
            errs = []
            for s in ("left", "right"):
                el_t, wr_t = tw_tgt[s]
                eb, wb = rig.bodies[s]
                # the target, in the TWIN's pelvis frame (twin yaw is 0) ...
                wr_rel_t = wr_t - twin_pel_w
                # ... against where the STEPPED wrist actually is, in its own
                wr_rel_s = _in_pelvis(d.xpos[wb], pel, yaw)
                errs.append(float(np.linalg.norm(wr_rel_s - wr_rel_t)))
            r["target_err"].append(max(errs))
        bz = float(d.qpos[ix.base_qpos][2])
        r["min_base_z"] = min(r["min_base_z"], bz)
        M = np.zeros(9)
        mujoco.mju_quat2Mat(M, d.qpos[ix.base_quat_qpos])
        r["max_pitch_deg"] = max(r["max_pitch_deg"], abs(float(np.degrees(
            np.arcsin(np.clip(M.reshape(3, 3)[2, 0], -1, 1))))))
        r["fell"] = r["fell"] or bz < 0.35
        fi += 1
    r["pitch_end"] = max(abs(float(d.qpos[rig.jnt[n][0]] - d.ctrl[rig.jnt[n][1]]))
                         for n in PITCH)
    r["base_travel"] = float(np.linalg.norm(np.array(d.qpos[ix.base_qpos][:2]) - base0))
    r["wedged_end"] = r["pitch_end"] > WEDGE_RAD
    r["wedged_any"] = r["pitch_max"] > WEDGE_RAD
    return r


# ─── generator (b): scripted human, authored in the ROBOT's shoulder frame ────
def scripted_human(geo, n=150, ramp=45, home=15, approach="direct"):
    """Hand paths a demonstrator would make, placed where they matter FOR THE ROBOT.

    The retargeting keeps only segment DIRECTIONS and rescales them by the
    robot's own limb lengths, so a path authored with human limb lengths lands
    the robot's hand somewhere else. These use the robot's measured shoulders and
    limb lengths, which is what makes "at box height" and "near the chest" true.

    Every stream STARTS AT HOME, arms hanging, and ramps in. A first version
    began with the arms already out in front: frame 0 was then a step change in
    every joint target, the arm slammed through the platform on frame 1, and the
    whole contact table measured the start-up jolt rather than the motion. A real
    operator starts at their sides, and so do these.

    `approach` picks how the hands get from home to the motion's first point:
      "direct"  - a straight line, which is what a person does unprompted
      "raised"  - up the body first, then out, then down: the staged raise that
                  removed the scripted wedge (2026-09-11), as an operator would
                  perform it if TRAINED to
    Returns {name: (frames, segment_lengths)} where segments are
    (home, approach, motion).
    """
    box = geo["box"]
    bz, bx = float(box[2]), float(box[0])
    chest = bz + 0.18
    above_slab = bz + 0.14

    def arm(side, hand):
        sh = geo[side]["shoulder"]
        sgn = 1.0 if side == "left" else -1.0
        return arm_dirs_from_hand(np.asarray(hand, float) - sh,
                                  np.array([0.0, sgn * 0.35, -1.0]),
                                  upper=geo[side]["upper"], fore=geo[side]["fore"])

    def to_frames(hands_l, hands_r):
        out = []
        for hl, hr in zip(hands_l, hands_r):
            ul, fl = arm("left", hl)
            ur, fr = arm("right", hr)
            out.append(keypoints_from_arm_dirs(ul, fl, ur, fr))
        return out

    def lerp(a, b, k):
        return [np.asarray(a) + (np.asarray(b) - np.asarray(a)) * (i / max(k - 1, 1))
                for i in range(k)]

    def build(motion):
        """motion(side, t) -> hand position in the pelvis frame."""
        seqs = {}
        for side in ("left", "right"):
            h0 = geo[side]["home_hand"]
            m0 = motion(side, 0.0)
            seq = [h0] * home
            if approach == "direct":
                seq += lerp(h0, m0, ramp)
            else:                                  # "raised"
                k = ramp // 3
                up = np.array([h0[0], m0[1], above_slab])
                out_ = np.array([m0[0], m0[1], above_slab])
                seq += lerp(h0, up, k) + lerp(up, out_, k) + lerp(out_, m0, ramp - 2 * k)
            seq += [motion(side, i / max(n - 1, 1)) for i in range(n)]
            seqs[side] = seq
        return to_frames(seqs["left"], seqs["right"]), (home, ramp, n)

    def face_y(side, t=0.0, inward=0.0):
        sgn = 1.0 if side == "left" else -1.0
        return box[1] + sgn * (0.09 - inward * t)

    motions = {
        # the ordinary reach: out to the box faces at box height
        "reach_out_boxh": lambda s_, t: np.array(
            [0.10 + (bx - 0.10) * t, face_y(s_), bz]),
        # O19: hands IN toward the chest at box height - torso AND platform
        "hands_in_boxh": lambda s_, t: np.array(
            [bx - (bx - 0.02) * t, face_y(s_, t, 0.03), bz]),
        # hands in toward the chest ABOVE the platform - isolates the torso
        "hands_in_chest": lambda s_, t: np.array(
            [bx - (bx - 0.02) * t, face_y(s_, t, 0.03), chest]),
        # fast lateral sweep above the box: One-Euro + stillness lock
        "lateral_sweep": lambda s_, t: np.array(
            [bx - 0.06, face_y(s_) + 0.15 * np.sin(10 * np.pi * t), bz + 0.12]),
    }
    out = {name: build(fn) for name, fn in motions.items()}
    frames, seg = build(lambda s_, t: np.array(
        [0.10 + (bx - 0.10) * t, face_y(s_), bz + 0.12]))
    off = home + ramp
    out["dropout_3_5_8"] = (with_dropout(frames, [(off + 40, 3), (off + 80, 5),
                                                  (off + 120, 8)]), seg)
    return out


# ─── generator (a): replay the scripted demonstrator ──────────────────────────
def demonstrator_replay(seed=0):
    """Arm directions and commands the demonstrator ACHIEVED, per 25 Hz tick.

    Directions come from stepped body positions expressed in the pelvis frame,
    which is the frame the twin works in. Nothing from the twin or an IK
    residual is used.
    """
    import torch
    import walk_test as W
    from g1_data import scripted_demo as SD
    rows = []
    real = mujoco.mj_step

    def hook(m, d, *a, **k):
        f = sys._getframe(1)
        if f.f_code.co_name == "run_episode":
            L = f.f_locals
            i = L.get("i")
            if i is not None and i % 20 == 0:
                ix = L["ix"]
                mujoco.mj_kinematics(m, d)
                pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                yaw = _yaw(d.qpos[ix.base_quat_qpos])
                dirs = []
                for s in ("left", "right"):
                    sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                           getattr(C, s.upper() + "_SHOULDER_BODY"))
                    eb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                           getattr(C, s.upper() + "_ELBOW_BODY"))
                    wb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                           getattr(C, s.upper() + "_WRIST_BODY"))
                    sh = _in_pelvis(d.xpos[sb], d.xpos[pel], yaw)
                    el = _in_pelvis(d.xpos[eb], d.xpos[pel], yaw)
                    wr = _in_pelvis(d.xpos[wb], d.xpos[pel], yaw)
                    dirs += [(el - sh) / np.linalg.norm(el - sh),
                             (wr - el) / np.linalg.norm(wr - el)]
                rows.append((tuple(dirs), np.array(d.qpos[ix.upper_qpos], float),
                             L["phase"].name))
        return real(m, d, *a, **k)

    mujoco.mj_step = hook
    try:
        cfg = TeleopConfig()
        demo = SD.DemoConfig(walk_place=True, start_xy=(0.60, 0.00), settle_s=14.0)
        SD.run_episode(seed, cfg=cfg, demo=demo, book=SD.PoseBook(cfg, demo),
                       policy=torch.jit.load(W.POLICY_PATH))
    finally:
        mujoco.mj_step = real
    return rows


def _row(tag, mode, r):
    te = np.array(r["target_err"]) if r["target_err"] else np.array([np.nan])
    print("| %-16s | %-6s | %4d/%-4d | %6.0f | %6.0f | %5d | %+6.1f | %4.0f | %5d | %4.0f | %.3f | %.3f | %-5s | %3d | %-4s | %.3f | %4.1f | %.3f |"
          % (tag, mode, r["applied"], r["applied"] + r["frozen"],
             1000 * np.nanmedian(te), 1000 * np.nanmax(te),
             r["plat_steps"], 1000 * r["plat_depth"], r["plat_force"],
             r["torso_steps"], r["torso_force"],
             r["pitch_max"], r["pitch_end"],
             "WEDGE" if r["wedged_end"] else ("trans" if r["wedged_any"] else "ok"),
             r["sat_frames"], "FELL" if r["fell"] else "up",
             r["min_base_z"], r["max_pitch_deg"], r["base_travel"]))


HDR = ("| motion           | base   | applied   | tgt p50 | tgt max | platS | depth  | plN  "
       "| torS  | toN  | pitchMx | pitchEnd | wedge | sat | fall | min z | pitch | travel |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    import torch
    import walk_test as W
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = TeleopConfig()
    policy = torch.jit.load(W.POLICY_PATH)

    geo = Rig(cfg).shoulder_frame()
    print("robot geometry, pelvis frame (standing, base at the standoff):")
    for s in ("left", "right"):
        print("  %-5s shoulder %s  upper %.3f  fore %.3f"
              % (s, np.round(geo[s]["shoulder"], 3), geo[s]["upper"], geo[s]["fore"]))
    print("  box centre %s   platform near edge %.3f m ahead of pelvis\n"
          % (np.round(geo["box"], 3), geo["plat_edge_fwd"]))

    if which in ("all", "human"):
        print("=== (b) SCRIPTED HUMAN - starts at home; contacts split by segment ===")
        print("| motion | approach | base | platS home/ramp/motion/hold | depth mm | plN | torS | pitch max | pitch END | verdict | sat | fall | min z | pitch deg | travel m | tgt p50/max mm |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for approach in ("direct", "raised"):
            for name, (frames, seg) in scripted_human(geo, approach=approach).items():
                for locked, mode in ((True, "locked"), (False, "free")):
                    r = run_stream(Rig(cfg), frames, policy=policy, locked=locked)
                    pc = np.diff(np.r_[0, r["plat_per_frame"]])
                    h, rp, mo = seg
                    parts = (pc[:h].sum(), pc[h:h + rp].sum(),
                             pc[h + rp:h + rp + mo].sum(), pc[h + rp + mo:].sum())
                    te = np.array(r["target_err"]) if r["target_err"] else np.array([np.nan])
                    verdict = ("WEDGE" if r["wedged_end"] else
                               ("transient" if r["wedged_any"] else "ok"))
                    print("| %s | %s | %s | %d/%d/%d/%d | %+.1f | %.0f | %d | %.3f | %.3f | %s | %d | %s | %.3f | %.1f | %.3f | %.0f/%.0f |"
                          % (name, approach, mode, *parts, 1000 * r["plat_depth"],
                             r["plat_force"], r["torso_steps"], r["pitch_max"],
                             r["pitch_end"], verdict, r["sat_frames"],
                             "FELL" if r["fell"] else "up", r["min_base_z"],
                             r["max_pitch_deg"], r["base_travel"],
                             1000 * np.nanmedian(te), 1000 * np.nanmax(te)), flush=True)
        print()

    if which in ("all", "replay"):
        print("=== (a) REPLAY ROUND TRIP ===")
        rows = demonstrator_replay()
        dirs = [x[0] for x in rows]
        src_q = np.array([x[1] for x in rows])
        phases = [x[2] for x in rows]
        print("captured %d ticks from the scripted demonstrator" % len(rows))
        r = run_stream(Rig(cfg), replay_frames(dirs), policy=policy, locked=True)
        print(HDR)
        _row("replay", "locked", r)
        tg = np.array(r["targets"])[:len(src_q)]
        names = list(C.UPPER_BODY_JOINTS)
        ik = [i for i, nme in enumerate(names) if "shoulder" in nme or "elbow" in nme]
        # settled = the demonstrator's own command has not moved for 25 ticks
        moving = np.r_[True, np.any(np.abs(np.diff(src_q[:, ik], axis=0)) > 1e-4, axis=1)]
        settled = np.zeros(len(src_q), bool)
        run = 0
        for k in range(len(src_q)):
            run = 0 if moving[k] else run + 1
            settled[k] = run >= 25
        print("\n  per-joint round-trip error, rad: teleop output (twin qpos sent to "
              "ctrl) vs the demonstrator's ACHIEVED stepped joint angles")
        print("  settled ticks: %d of %d" % (int(settled.sum()), len(src_q)))
        print("  | joint | all mean | all p95 | all max | settled mean | settled max |")
        print("  |---|---|---|---|---|---|")
        for j in ik:
            e = np.abs(tg[:, j] - src_q[:, j])
            es = e[settled] if settled.any() else np.array([np.nan])
            print("  | %-26s | %.4f | %.4f | %.4f | %.4f | %.4f |"
                  % (names[j], e.mean(), np.percentile(e, 95), e.max(),
                     np.nanmean(es), np.nanmax(es)))
        e_all = np.abs(tg[:, ik] - src_q[:, ik])
        es = e_all[settled]
        print("  | %-26s | %.4f | %.4f | %.4f | %.4f | %.4f |"
              % ("ALL 8 IK-driven joints", e_all.mean(), np.percentile(e_all, 95),
                 e_all.max(), es.mean(), es.max()))
        # where is it worst?
        worst = np.argmax(e_all.max(axis=1))
        print("  worst tick %d, phase %s, error %.3f rad on %s"
              % (worst, phases[worst], e_all[worst].max(),
                 names[ik[int(np.argmax(e_all[worst]))]]))
        te = np.array(r["target_err"])
        print("  wrist: retargeted target vs stepped achieved, mm - p50 %.0f  p95 %.0f  max %.0f"
              % (1000 * np.median(te), 1000 * np.percentile(te, 95), 1000 * te.max()))


if __name__ == "__main__":
    main()


# ─── (a) done properly: static per-pose round trip, and a pre-MOVE replay ─────
def replay_static(rows, cfg, n_poses=40, hold=60):
    """Round trip per pose, with no lag and no scene confound.

    The continuous replay mixes three things: pipeline LAG (One-Euro, arm_alpha,
    depth_alpha), the stillness lock, and a scene confound - it replays the
    LOWER/RELEASE poses, which the demonstrator executed at the GOAL, against the
    pickup platform. So here each pose is held for `hold` frames (0.4**60 ~ 0 for
    arm_alpha) from a rig whose arm already STARTS at that pose, and only the
    pre-MOVE part of the task is sampled.

    Reports three separable errors:
      mapping   - teleop joint output vs the source joints   (retarget + IK)
      execution - stepped achieved joints vs the output       (servo + gravity)
      position  - stepped elbow/wrist BODY positions vs the source, pelvis frame
    """
    names = list(C.UPPER_BODY_JOINTS)
    ik = [i for i, n in enumerate(names) if "shoulder" in n or "elbow" in n]
    pre = [k for k, r in enumerate(rows) if r[2] in
           ("SETTLE", "REACH", "REPOSITION", "APPROACH", "GRASP", "LIFT")]
    # evenly across the pre-MOVE window, skipping duplicate consecutive poses
    pick, last = [], None
    for k in np.linspace(0, len(pre) - 1, n_poses * 3).astype(int):
        q = rows[pre[k]][1]
        if last is None or np.abs(q[ik] - last[ik]).max() > 0.02:
            pick.append(pre[k]); last = q
        if len(pick) >= n_poses:
            break
    res = []
    for k in pick:
        dirs, src_q, phase = rows[k]
        rig = Rig(cfg)
        m, d, ix = rig.m, rig.d, rig.ix
        d.qpos[ix.upper_qpos] = src_q          # arm STARTS at the source pose
        d.ctrl[ix.upper_ctrl] = src_q
        mujoco.mj_forward(m, d)
        # source body positions, pelvis frame, from the rig at that pose
        yaw = _yaw(d.qpos[ix.base_quat_qpos]); pel = d.xpos[rig.pel]
        src_pos = {s: (_in_pelvis(d.xpos[rig.bodies[s][0]], pel, yaw),
                       _in_pelvis(d.xpos[rig.bodies[s][1]], pel, yaw))
                   for s in ("left", "right")}
        frames = replay_frames([dirs]) * hold
        twin = G1Robot(cfg)
        twin_ix = ModelIndex.resolve(twin.model)
        # seed the twin at the source pose so the IK starts on the right branch,
        # exactly as a continuous teleop session would be by the time it got here
        twin.data.qpos[twin_ix.upper_qpos] = src_q
        mujoco.mj_forward(twin.model, twin.data)
        _TARGETS.clear(); _TARGETS["_left_wr_body"] = twin.left_wrist_body
        ctl = TeleopController(twin, cfg)
        ctl.reset()
        ctl.prev_left = np.array(twin.data.qpos[twin.ik_left_qpos], float)
        ctl.prev_right = np.array(twin.data.qpos[twin.ik_right_qpos], float)
        rig.lock.lock(m, d)
        src = SyntheticSource(frames)
        while True:
            fr = src.grab()
            if fr is None:
                break
            ctl.step(fr)
            out_q = np.array(twin.data.qpos[twin_ix.upper_qpos], float)
            for _ in range(STEPS_PER_FRAME):
                lq, ldq = d.qpos[ix.leg_qpos], d.qvel[ix.leg_qvel]
                d.ctrl[ix.leg_ctrl] = (rig.DEF - lq) * rig.KPS + (0.0 - ldq) * rig.KDS
                d.ctrl[ix.upper_ctrl] = out_q
                d.ctrl[ix.pad_ctrl] = 0.0
                mujoco.mj_step(m, d)
        ach_q = np.array(d.qpos[ix.upper_qpos], float)
        yaw = _yaw(d.qpos[ix.base_quat_qpos]); pel = d.xpos[rig.pel]
        pos_err = max(max(float(np.linalg.norm(_in_pelvis(d.xpos[rig.bodies[s][0]], pel, yaw) - src_pos[s][0])),
                          float(np.linalg.norm(_in_pelvis(d.xpos[rig.bodies[s][1]], pel, yaw) - src_pos[s][1])))
                      for s in ("left", "right"))
        res.append(dict(k=k, phase=phase, map=np.abs(out_q[ik] - src_q[ik]),
                        exe=np.abs(ach_q[ik] - out_q[ik]), pos=pos_err))
    return res, [names[i] for i in ik]


def replay_premove(rows, cfg, policy):
    """The continuous replay, truncated at the first MOVE tick - no scene confound."""
    cut = next((k for k, r in enumerate(rows) if r[2] == "MOVE"), len(rows))
    dirs = [r[0] for r in rows[:cut]]
    r = run_stream(Rig(cfg), replay_frames(dirs), policy=policy, locked=True)
    return r, cut


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "replay2":
    import torch, walk_test as W
    cfg = TeleopConfig()
    pol = torch.jit.load(W.POLICY_PATH)
    rows = demonstrator_replay()
    print("captured %d ticks" % len(rows))
    res, jn = replay_static(rows, cfg)
    M = np.array([x["map"] for x in res]); E = np.array([x["exe"] for x in res])
    P = np.array([x["pos"] for x in res])
    print("\n=== (a) STATIC per-pose round trip, %d distinct poses, pre-MOVE ===" % len(res))
    print("| joint | mapping mean | mapping p95 | mapping max | execution mean | execution max |")
    print("|---|---|---|---|---|---|")
    for j, n in enumerate(jn):
        print("| %s | %.4f | %.4f | %.4f | %.4f | %.4f |"
              % (n, M[:, j].mean(), np.percentile(M[:, j], 95), M[:, j].max(),
                 E[:, j].mean(), E[:, j].max()))
    print("| ALL | %.4f | %.4f | %.4f | %.4f | %.4f |"
          % (M.mean(), np.percentile(M, 95), M.max(), E.mean(), E.max()))
    print("elbow/wrist BODY position, stepped vs source, mm: p50 %.1f  p95 %.1f  max %.1f"
          % (1000 * np.median(P), 1000 * np.percentile(P, 95), 1000 * P.max()))
    print("distinct source poses: %d | distinct mapping-error rows: %d (TR18 check)"
          % (len(res), len({tuple(np.round(x['map'], 5)) for x in res})))
    by = {}
    for x in res:
        by.setdefault(x["phase"], []).append(float(x["map"].max()))
    print("worst mapping error by phase:", {k: round(max(v), 3) for k, v in by.items()})
    r, cut = replay_premove(rows, cfg, pol)
    print("\n=== (a) CONTINUOUS replay, truncated before MOVE (%d ticks) ===" % cut)
    pc = np.diff(np.r_[0, r["plat_per_frame"]])
    print("platform contact-steps %d (in the hold tail %d) | depth %.1f mm | %.0f N"
          % (r["plat_steps"], pc[cut:].sum(), 1000 * r["plat_depth"], r["plat_force"]))
    print("wrist pitch max %.3f | END %.3f | verdict %s | torso contact-steps %d | sat %d"
          % (r["pitch_max"], r["pitch_end"],
             "WEDGE" if r["wedged_end"] else ("transient" if r["wedged_any"] else "ok"),
             r["torso_steps"], r["sat_frames"]))
