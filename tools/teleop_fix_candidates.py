"""Measure the two candidate teleop-hazard fixes against the same motions and gates.

    python tools/teleop_fix_candidates.py teleop  A|B          # synthetic motion set
    python tools/teleop_fix_candidates.py gates   base|A|B|Bns  N  lock|pred
    python tools/teleop_fix_candidates.py corridor base|A       # close-in reach probe

MEASUREMENT ONLY - neither candidate is adopted by running this.

  base  : adopted configuration, both flags off
  A     : IKConfig.free_wrists = True    (IK drives 7 joints/arm, palm-site task)
  B     : DemoConfig.hand_platform_filter (no hand<->platform contact)
  Bns   : B with the staged raise switched OFF - is the mitigation still needed?

Rules (CLAUDE.md): stepped model only, never twin residuals (TR16a); contacts
gate on d.ncon + mj_contactForce, never mj_geomDistance (TR19); every synthetic
motion starts at home (the stream-starts-extended artifact, 2026-09-15).

WRIST-TWIST ARTIFACT - defined BEFORE looking, and not moved afterwards
----------------------------------------------------------------------
D2 was adopted because wrist twist made the grasp pose unusable. Operationally,
at the GRASP instant (the first tick the weld is commanded):
  * the palm-plane normal (thin axis of the hand mesh, via
    tools/check_palm_orientation.hand_mesh_axes) makes an angle with the
    palm->box-centre direction; the sign of the normal that faces the box is
    used, so a palm presenting squarely reads ~0 deg and edge-on reads 90 deg;
  * ARTIFACT if that angle exceeds 45 deg on any seed, OR any wrist joint sits
    more than 1.0 rad from WRIST_NATURAL.
Baseline is measured with the same metric, so "normal" is not assumed.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from g1_teleop import config as C
from g1_teleop.config import TeleopConfig

TWIST_ANGLE_DEG = 45.0
TWIST_JOINT_RAD = 1.0
OUT = os.path.join(ROOT, "docs", "measurements")


def cfg_for(tag):
    cfg = TeleopConfig()
    if tag == "A":
        cfg = dataclasses.replace(cfg, ik=dataclasses.replace(cfg.ik, free_wrists=True))
    return cfg


def demo_for(tag, lock):
    from g1_data import scripted_demo as SD
    kw = dict(walk_place=True, start_xy=(0.60, 0.00), settle_s=14.0,
              lock_predicate=(lock == "pred"))
    if tag in ("B", "Bns"):
        kw["hand_platform_filter"] = True
    if tag in ("Bp", "Bpns"):
        kw["hand_platform_filter"] = "pickup"
    if tag in ("Bns", "Bpns"):
        kw["staged_reach"] = False
    return SD.DemoConfig(**kw)


# ─── gates, instrumented for twist, box behaviour and hand-through-platform ───
def run_gates(tag, n, lock):
    import torch
    import walk_test as W
    from g1_data import scripted_demo as SD
    from check_palm_orientation import hand_geom_id, hand_mesh_axes

    cfg, demo = cfg_for(tag), demo_for(tag, lock)
    book = SD.PoseBook(cfg, demo)
    pol = torch.jit.load(W.POLICY_PATH)
    real = mujoco.mj_step
    rows = []

    for seed in range(n):
        st = dict(grasp_seen=False, angle=[], wdev=0.0, box_below=0.0,
                  hand_below=0.0, box_rest_contact_end=False)

        def hook(m, d, *a, **k):
            f = sys._getframe(1)
            if f.f_code.co_name != "run_episode":
                return real(m, d, *a, **k)
            L = f.f_locals
            ph = L["phase"].name
            weld = L["weld"]
            if "ids" not in st:
                st["ids"] = {}
                for s in ("left", "right"):
                    wb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                           getattr(C, s.upper() + "_WRIST_BODY"))
                    st["ids"][s] = (wb, hand_geom_id(m, wb))
                st["plat"] = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                              for g in ("platform_pickup_geom", "platform_goal_geom")]
                st["box_g"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box1_geom")
                st["wj"] = [(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)], v)
                            for j, v in C.WRIST_NATURAL.items()]
            # twist, at the first tick the weld is commanded
            if ph == "GRASP" and float(L.get("cmd", 0.0)) >= 0.5 and not st["grasp_seen"]:
                st["grasp_seen"] = True
                mujoco.mj_kinematics(m, d)
                box = d.xpos[weld.box_bid]
                for s in ("left", "right"):
                    wb, hg = st["ids"][s]
                    ax = hand_mesh_axes(m, d, hg, wb)
                    site = weld.site_l if s == "left" else weld.site_r
                    to_box = box - d.site_xpos[site]
                    to_box = to_box / max(np.linalg.norm(to_box), 1e-9)
                    nrm = ax["normal_world"]
                    c = abs(float(np.dot(nrm, to_box)))
                    st["angle"].append(float(np.degrees(np.arccos(np.clip(c, 0, 1)))))
                st["wdev"] = max(abs(float(d.qpos[q]) - v) for q, v in st["wj"])
            # box tunnelling and hand-through-platform, by POINT proxies while
            # over a platform footprint (no contacts exist for hands under B,
            # and mj_geomDistance is unreliable here - TR19)
            if L["i"] % 20 == 0:
                bp = d.qpos[L["ix"].box_qpos][:3]
                for pg in st["plat"]:
                    bid = m.geom_bodyid[pg]
                    c_ = m.body_pos[bid]; h_ = m.geom_size[pg]
                    top = c_[2] + h_[2]
                    if abs(bp[0] - c_[0]) <= h_[0] and abs(bp[1] - c_[1]) <= h_[1]:
                        st["box_below"] = min(st["box_below"], float(bp[2] - 0.09 - top))
                    for site in (weld.site_l, weld.site_r):
                        sp = d.site_xpos[site]
                        if abs(sp[0] - c_[0]) <= h_[0] and abs(sp[1] - c_[1]) <= h_[1]:
                            st["hand_below"] = min(st["hand_below"], float(sp[2] - top))
            return real(m, d, *a, **k)

        mujoco.mj_step = hook
        try:
            r = SD.run_episode(seed, cfg=cfg, demo=demo, book=book, policy=pol)
        finally:
            mujoco.mj_step = real
        rows.append(dict(
            seed=seed, ok=bool(r["ok"]), fail=r["fail_phase"],
            engaged=bool(r["engaged"]), palm=float(r["palm_goal_err_mm"]),
            place=float(r["place_err_m"]), tilt=float(r["tilt_deg"]),
            resting=bool(r["resting"]), fell=bool(r["fell"]),
            pitch=float(r["max_pitch_deg"]), n=int(r["n_samples"]),
            wedged=bool(r["wedged"]), wristP=float(r["wrist_dev_rad"]),
            hits=int(r["hand_plat_contacts"]), depth=float(r["hand_plat_depth_mm"]),
            twist_angle=max(st["angle"]) if st["angle"] else float("nan"),
            twist_joint=float(st["wdev"]),
            box_below_mm=1000 * st["box_below"], palm_below_top_mm=1000 * st["hand_below"]))
        x = rows[-1]
        print("  %s seed %2d ok=%-5s %-12s palm %5.1f place %.4f tilt %4.1f rest %-5s "
              "twist %5.1f deg / %.2f rad | handC %5d %+6.1fmm | box_below %+6.1f "
              "palm_below_top %+6.1f | wedge %s n %d"
              % (tag, seed, x["ok"], x["fail"] or "-", x["palm"], x["place"], x["tilt"],
                 x["resting"], x["twist_angle"], x["twist_joint"], x["hits"], x["depth"],
                 x["box_below_mm"], x["palm_below_top_mm"], x["wedged"], x["n"]),
              flush=True)
    return rows


# ─── the synthetic teleop motion set, base locked ─────────────────────────────
def run_teleop(tag):
    import torch
    import walk_test as W
    import teleop_physics_check as T
    cfg = cfg_for(tag)
    pol = torch.jit.load(W.POLICY_PATH)
    filt = True if tag == "B" else ("pickup" if tag == "Bp" else False)
    geo = T.Rig(cfg, filter_hands=filt).shoulder_frame()
    rows = []
    for approach in ("direct", "raised"):
        for name, (frames, seg) in T.scripted_human(geo, approach=approach).items():
            r = T.run_stream(T.Rig(cfg, filter_hands=filt), frames, policy=pol, locked=True)
            te = np.array(r["target_err"]) if r["target_err"] else np.array([np.nan])
            v = "WEDGE" if r["wedged_end"] else ("transient" if r["wedged_any"] else "ok")
            rows.append(dict(motion=name, approach=approach, plat=r["plat_steps"],
                             depth=1000 * r["plat_depth"], torso=r["torso_steps"],
                             pitch_max=r["pitch_max"], pitch_end=r["pitch_end"],
                             verdict=v, sat=r["sat_frames"],
                             tgt_p50=1000 * float(np.nanmedian(te))))
            x = rows[-1]
            print("  %s | %-15s | %-6s | platS %6d | depth %+6.1f | torS %5d | pitch max %.3f "
                  "END %.3f | %-9s | sat %3d | tgt p50 %.0f mm"
                  % (tag, name, approach, x["plat"], x["depth"], x["torso"],
                     x["pitch_max"], x["pitch_end"], v, x["sat"], x["tgt_p50"]), flush=True)
    return rows


# ─── close-in reach: achieved palm error on stepped state ─────────────────────
def run_corridor(tag):
    """Palm targets at forward 0.08-0.26, z 0.86-0.92, grasp separation.

    Same design as the 2026-09-10 probe: solved joint angles applied to the
    STEPPED model at a captured standing pose with the base welded, held 0.5 s by
    the real servos, then the ACHIEVED palm sites are measured. No residual.
    """
    import torch
    import walk_test as W
    from g1_data import scripted_demo as SD
    from g1_teleop.base_lock import BaseLock
    from g1_teleop.indices import ModelIndex
    from g1_teleop.box_reset import reset_box

    cfg = cfg_for(tag)
    demo = SD.DemoConfig(walk_place=True, start_xy=(0.60, 0.00), settle_s=14.0)
    book = SD.PoseBook(cfg, demo)
    m = mujoco.MjModel.from_xml_path(cfg.model_path)
    m.opt.timestep = cfg.loco.sim_dt
    d = mujoco.MjData(m)
    ix = ModelIndex.resolve(m)
    DEF = np.asarray(cfg.loco.default_angles)
    KPS = np.asarray(cfg.loco.kps, np.float32)
    KDS = np.asarray(cfg.loco.kds, np.float32)
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    sid = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s + "_palm_site")
           for s in ("left", "right")}

    def fresh():
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[ix.leg_qpos] = DEF
        box = reset_box(m, d, ix, cfg.box, seed=0)
        d.qpos[ix.base_qpos][0] = min(box[0] - 0.32, cfg.grasp.max_base_x)
        d.qpos[ix.base_qpos][1] = box[1]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        BaseLock(m).lock(m, d)

    fresh()
    pel_z = float(d.xpos[pel][2])
    fwds = np.round(np.arange(0.08, 0.261, 0.02), 3)
    zs = np.round(np.arange(0.86, 0.921, 0.02), 3)
    err = np.zeros((len(zs), len(fwds)))
    for iz, zw in enumerate(zs):
        for jf, fw in enumerate(fwds):
            off = np.array([float(fw), 0.0, float(zw - pel_z)])
            pose, _ = book.solve(off)          # joint angles only
            fresh()
            d.qpos[ix.upper_qpos] = pose
            d.ctrl[ix.upper_ctrl] = pose
            mujoco.mj_forward(m, d)
            for _ in range(250):
                lq, ldq = d.qpos[ix.leg_qpos], d.qvel[ix.leg_qvel]
                d.ctrl[ix.leg_ctrl] = (DEF - lq) * KPS + (0.0 - ldq) * KDS
                d.ctrl[ix.pad_ctrl] = 0.0
                mujoco.mj_step(m, d)
            worst = 0.0
            for s, sgn in (("left", 1.0), ("right", -1.0)):
                goal = d.xpos[pel] + off + np.array([0.0, sgn * demo.sep / 2, 0.0])
                worst = max(worst, float(np.linalg.norm(d.site_xpos[sid[s]] - goal)))
            err[iz, jf] = worst
    print("  %s corridor: achieved palm error, mm (guard 45)" % tag)
    print("  z \\ fwd | " + " ".join("%5.2f" % f for f in fwds))
    for iz, zw in enumerate(zs):
        print("  %.2f    | " % zw + " ".join("%5.0f" % (1000 * v) for v in err[iz]))
    inside = (err <= 0.045)
    print("  cells within the 45 mm guard: %d of %d | min %.0f  median %.0f  max %.0f mm"
          % (inside.sum(), inside.size, 1000 * err.min(), 1000 * np.median(err),
             1000 * err.max()))
    return dict(fwds=fwds.tolist(), zs=zs.tolist(), err=err.tolist())


def main():
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1]
    if what == "teleop":
        tag = sys.argv[2]
        res = run_teleop(tag)
        name = "teleop_%s" % tag
    elif what == "gates":
        tag, n, lock = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        res = run_gates(tag, n, lock)
        ok = sum(r["ok"] for r in res)
        print("GATE %s %s: %d/%d" % (tag, lock, ok, n))
        name = "gates_%s_%s_%d" % (tag, lock, n)
    elif what == "corridor":
        tag = sys.argv[2]
        res = run_corridor(tag)
        name = "corridor_%s" % tag
    else:
        raise SystemExit(__doc__)
    with open(os.path.join(OUT, name + ".json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print("DONE", name)


if __name__ == "__main__":
    main()
