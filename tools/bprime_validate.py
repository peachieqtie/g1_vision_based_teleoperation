"""Validate the ADOPTED B-prime pair exclusion (2026-09-15).

    python tools/bprime_validate.py teleop  adopted|base|Bp
    python tools/bprime_validate.py gates   adopted|adoptedns|base  N  lock|pred

  adopted   : default config - scene.xml pair exclusion ON
  adoptedns : adopted, staged raise OFF (is the raise still required?)
  base      : exclusion stripped (cfg.contact.hand_pickup_exclusion=False) =
              the pre-adoption configuration, bit-identical to the checkpoint
  Bp        : base + the old contype/conaffinity pickup-only filter, the
              measured candidate - a CONTROL that must show hand<->hand = 0,
              so the crossing check is shown able to fail

Rules: stepped model only, no twin residual (TR16a); every contact gates on
d.ncon + mj_contactForce, never mj_geomDistance (TR19); motions start at home.
Measures are defined in tools/contact_measures.py.
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
from g1_teleop.config import TeleopConfig, ContactConfig
from g1_teleop.contact_contract import contract_of
import contact_measures as CM

OUT = os.path.join(ROOT, "docs", "measurements")
PROBE_EVERY = 5          # physics steps between shadow/vertex probes (10 ms)


def cfg_for(tag):
    cfg = TeleopConfig()
    if tag in ("base", "Bp"):
        cfg = dataclasses.replace(cfg, contact=ContactConfig(hand_pickup_exclusion=False))
    return cfg


class Monitor:
    """Everything measured per step of a stepped model, installed as an mj_step hook."""

    def __init__(self, m, cfg, box_qpos, dt):
        self.m = m
        # shadow-vs-live cross-check only means something where the contact is live
        self.compare_live = not cfg.contact.hand_pickup_exclusion
        self.shadow = CM.ShadowProbe(cfg)
        self.vert = CM.VertexCheck(m)
        self.plat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, CM.PICKUP_GEOM)
        self.goal = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, CM.GOAL_GEOM)
        self.box_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, CM.BOX_GEOM)
        self.hl = set(CM.hand_geoms(m, "left"))
        self.hr = set(CM.hand_geoms(m, "right"))
        self.hands = self.hl | self.hr
        self.robot = {g for g in range(m.ngeom)
                      if (m.geom_contype[g] or m.geom_conaffinity[g])
                      and g not in (self.plat, self.goal, self.box_g)
                      and m.geom_bodyid[g] != 0}
        self.box_qpos = box_qpos
        self.dt = dt
        self.k = 0
        self.pass_dur = CM.Duration(PROBE_EVERY * dt)
        self.r = dict(hand_hand_steps=0, hand_hand_N=0.0, hand_goal_steps=0, hand_goal_N=0.0,
                      hand_box_steps=0, robot_pickup_steps=0, robot_pickup_bodies={},
                      pass_depth_mm=0.0, pass_bodies={}, vertex_depth_mm=0.0,
                      shadow_vs_vertex_disagree=0, probes=0,
                      shadow_vs_live_maxdiff_mm=0.0, shadow_live_compared=0,
                      box_sink_mm=0.0, box_disp_mm=0.0, box_disp_xy_mm=0.0,
                      box_tilt_max_deg=0.0, box_off_pickup=False,
                      box_sink_unwelded_mm=0.0, box_sink_phase="")
        self.box0 = None
        self.box_track = True
        self.phase, self.welded = "", False

    def before(self, m, d):
        if self.box0 is None:
            self.box0 = np.array(d.qpos[self.box_qpos][:3])
        self.k += 1
        self._pre = None
        if self.k % PROBE_EVERY == 0:
            dep, per = self.shadow.probe(m, d)
            vd = self.vert.depth(m, d)
            r = self.r
            r["probes"] += 1
            r["pass_depth_mm"] = max(r["pass_depth_mm"], 1000 * dep)
            r["vertex_depth_mm"] = max(r["vertex_depth_mm"], 1000 * vd)
            if vd > 0 and dep == 0:
                r["shadow_vs_vertex_disagree"] += 1
            for b, v in per.items():
                r["pass_bodies"][b] = max(r["pass_bodies"].get(b, 0.0), 1000 * v)
            self.pass_dur.add(dep > 0, dep)
            self._pre = dep

    def after(self, m, d):
        r = self.r
        n, f = CM.live_pairs(m, d, self.hl, self.hr)
        if n:
            r["hand_hand_steps"] += 1
            r["hand_hand_N"] = max(r["hand_hand_N"], f)
        n, f = CM.live_pairs(m, d, self.hands, {self.goal})
        if n:
            r["hand_goal_steps"] += 1
            r["hand_goal_N"] = max(r["hand_goal_N"], f)
        n, _ = CM.live_pairs(m, d, self.hands, {self.box_g})
        if n:
            r["hand_box_steps"] += 1
        live_dep = 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            if self.plat in (con.geom1, con.geom2):
                o = con.geom2 if con.geom1 == self.plat else con.geom1
                if o in self.robot:
                    r["robot_pickup_steps"] += 1
                    b = m.body(m.geom_bodyid[o]).name
                    r["robot_pickup_bodies"][b] = r["robot_pickup_bodies"].get(b, 0) + 1
                    if o in self.shadow.hands:
                        live_dep = max(live_dep, -float(con.dist))
        # Cross-validation of the shadow measure: where hand<->pickup contact is
        # LIVE (exclusion stripped), the shadow probe at the pre-step qpos must
        # reproduce the depth the physics just used. Meaningless for `Bp`, whose
        # bitmask filter suppresses the live contact the comparison needs.
        if self.compare_live and self._pre is not None and (live_dep > 0 or self._pre > 0):
            r["shadow_live_compared"] += 1
            r["shadow_vs_live_maxdiff_mm"] = max(r["shadow_vs_live_maxdiff_mm"],
                                                 1000 * abs(live_dep - self._pre))
        if self.k % PROBE_EVERY == 0:
            gap = CM.box_bottom_gap(m, d, self.box_qpos)
            if gap is not None:
                if 1000 * gap < r["box_sink_mm"]:
                    r["box_sink_mm"] = 1000 * gap
                    r["box_sink_phase"] = self.phase + ("/welded" if self.welded else "")
                if not self.welded:
                    r["box_sink_unwelded_mm"] = min(r["box_sink_unwelded_mm"], 1000 * gap)
            if self.box_track:
                bp = np.array(d.qpos[self.box_qpos][:3])
                r["box_disp_mm"] = max(r["box_disp_mm"], 1000 * float(np.linalg.norm(bp - self.box0)))
                r["box_disp_xy_mm"] = max(r["box_disp_xy_mm"],
                                          1000 * float(np.linalg.norm(bp[:2] - self.box0[:2])))
                r["box_tilt_max_deg"] = max(r["box_tilt_max_deg"],
                                            CM.tilt_deg(d.qpos[self.box_qpos][3:7]))
                self.box_last = (bp, np.array(d.qpos[self.box_qpos][3:7]))
                pg = self.plat
                pc, ph = d.geom_xpos[pg], m.geom_size[pg]
                if (abs(bp[0] - pc[0]) > ph[0] or abs(bp[1] - pc[1]) > ph[1]
                        or bp[2] < pc[2] + ph[2] + 0.09 - 0.03):
                    r["box_off_pickup"] = True

    def finish(self):
        if getattr(self, "box_last", None) is not None:
            bp, q = self.box_last
            self.r["box_end_disp_mm"] = 1000 * float(np.linalg.norm(bp - self.box0))
            self.r["box_end_tilt_deg"] = CM.tilt_deg(q)
        self.r["pass_s"] = self.pass_dur.total
        self.r["pass_longest_s"] = self.pass_dur.longest
        return self.r


# ─── synthetic teleop motion set, base locked ─────────────────────────────────
def run_teleop(tag):
    import torch
    import walk_test as W
    import teleop_physics_check as T
    cfg = cfg_for(tag)
    pol = torch.jit.load(W.POLICY_PATH)
    filt = "pickup" if tag == "Bp" else False
    geo = T.Rig(cfg, filter_hands=filt).shoulder_frame()
    real = mujoco.mj_step
    rows = []
    for approach in ("direct", "raised"):
        for name, (frames, seg) in T.scripted_human(geo, approach=approach, extra=True).items():
            rig = T.Rig(cfg, filter_hands=filt)
            mon = Monitor(rig.m, cfg, rig.ix.box_qpos, rig.m.opt.timestep)

            def hook(m, d, *a, **k):
                mon.before(m, d)
                real(m, d, *a, **k)
                mon.after(m, d)

            mujoco.mj_step = hook
            try:
                r = T.run_stream(rig, frames, policy=pol, locked=True)
            finally:
                mujoco.mj_step = real
            x = mon.finish()
            v = "WEDGE" if r["wedged_end"] else ("transient" if r["wedged_any"] else "ok")
            te = np.array(r["target_err"]) if r["target_err"] else np.array([np.nan])
            x.update(motion=name, approach=approach, verdict=v, pitch_end=r["pitch_end"],
                     pitch_max=r["pitch_max"], sat=r["sat_frames"], torso=r["torso_steps"],
                     tgt_p50=1000 * float(np.nanmedian(te)), contract=contract_of(rig.m))
            rows.append(x)
            print("  %s | %-15s | %-6s | %-9s END %.3f | pickup %5d %s | hh %5d %4.0fN | "
                  "box disp %6.1f mm xy %6.1f tilt %5.1f off %s end %6.1f/%4.1f | pass %5.1f mm (vtx %4.1f) "
                  "%.2fs long %.2fs | shadow-live %.3f mm/%d | tgt %.0f"
                  % (tag, name, approach, v, r["pitch_end"], x["robot_pickup_steps"],
                     x["robot_pickup_bodies"] or "", x["hand_hand_steps"], x["hand_hand_N"],
                     x["box_disp_mm"], x["box_disp_xy_mm"], x["box_tilt_max_deg"],
                     x["box_off_pickup"], x.get("box_end_disp_mm", -1), x.get("box_end_tilt_deg", -1),
                     x["pass_depth_mm"], x["vertex_depth_mm"],
                     x["pass_s"], x["pass_longest_s"], x["shadow_vs_live_maxdiff_mm"],
                     x["shadow_live_compared"], x["tgt_p50"]), flush=True)
    return rows


# ─── both demonstrator gates, fully instrumented ──────────────────────────────
def run_gates(tag, n, lock):
    import torch
    import walk_test as W
    from g1_data import scripted_demo as SD
    cfg = cfg_for(tag)
    demo = SD.DemoConfig(walk_place=True, start_xy=(0.60, 0.00), settle_s=14.0,
                         lock_predicate=(lock == "pred"),
                         staged_reach=(tag != "adoptedns"))
    book = SD.PoseBook(cfg, demo)
    pol = torch.jit.load(W.POLICY_PATH)
    real = mujoco.mj_step
    rows = []
    for seed in range(n):
        st = {}

        def hook(m, d, *a, **k):
            f = sys._getframe(1)
            if f.f_code.co_name != "run_episode":
                return real(m, d, *a, **k)
            L = f.f_locals
            if "mon" not in st:
                st["mon"] = Monitor(m, cfg, L["ix"].box_qpos, m.opt.timestep)
                st["phases"] = {}
            mon = st["mon"]
            ph = L["phase"].name
            # box displacement is a PRE-GRASP measure: stop at the first weld tick
            mon.phase, mon.welded = ph, bool(d.eq_active[L["weld"].eq_id])
            if mon.welded:
                mon.box_track = False
            mon.before(m, d)
            real(m, d, *a, **k)
            mon.after(m, d)
            if (mon._pre or 0.0) > 0.0:
                st["phases"][ph] = st["phases"].get(ph, 0) + 1

        mujoco.mj_step = hook
        try:
            r = SD.run_episode(seed, cfg=cfg, demo=demo, book=book, policy=pol)
        finally:
            mujoco.mj_step = real
        x = st["mon"].finish()
        x.update(seed=seed, ok=bool(r["ok"]), fail=r["fail_phase"], engaged=bool(r["engaged"]),
                 palm=float(r["palm_goal_err_mm"]), place=float(r["place_err_m"]),
                 drift=float(r["carry_drift_mm"]), pitch=float(r["max_pitch_deg"]),
                 tilt=float(r["tilt_deg"]), resting=bool(r["resting"]), fell=bool(r["fell"]),
                 n=int(r["n_samples"]), wedged=bool(r["wedged"]),
                 wristP=float(r["wrist_dev_rad"]), fallbacks=len(r["lock_fallbacks"]),
                 hits=int(r["hand_plat_contacts"]), pass_phases=st["phases"])
        rows.append(x)
        print("  %s seed %2d ok=%-5s %-10s palm %6.2f place %.4f drift %.2f pitch %5.2f tilt %4.1f "
              "rest %-5s n %4d fb %d | pickup %4d hh %3d goal %4d | sink %+5.1f %s unwelded %+5.1f | predisp %5.1f | "
              "pass %5.1f mm %.2fs %s"
              % (tag, seed, x["ok"], x["fail"] or "-", x["palm"], x["place"], x["drift"],
                 x["pitch"], x["tilt"], x["resting"], x["n"], x["fallbacks"],
                 x["robot_pickup_steps"], x["hand_hand_steps"], x["hand_goal_steps"],
                 x["box_sink_mm"], x["box_sink_phase"], x["box_sink_unwelded_mm"], x["box_disp_mm"], x["pass_depth_mm"], x["pass_s"],
                 x["pass_phases"]), flush=True)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1]
    if what == "teleop":
        tag = sys.argv[2]
        res = run_teleop(tag)
        name = "bprime_teleop_%s" % tag
    elif what == "gates":
        tag, n, lock = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        res = run_gates(tag, n, lock)
        print("GATE %s %s: %d/%d" % (tag, lock, sum(r["ok"] for r in res), n))
        name = "bprime_gates_%s_%s_%d" % (tag, lock, n)
    else:
        raise SystemExit(__doc__)
    with open(os.path.join(OUT, name + ".json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print("DONE", name)


if __name__ == "__main__":
    main()
