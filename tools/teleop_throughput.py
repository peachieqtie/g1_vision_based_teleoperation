"""Teleop throughput: where the loop's time goes, and proof the speed fix is a
speed fix (2026-09-15).

    python tools/teleop_throughput.py verify      # mj_jac identical, kinematics vs forward
    python tools/teleop_throughput.py profile     # inside one controller.step
    python tools/teleop_throughput.py replay FILE.npz [--tol 1e-9]
                                                  # arm trajectory unchanged?
    python tools/teleop_throughput.py iters       # IK iterations actually used

The sim-to-wall RATIO is measured by the entry point itself
(`run_integrated_combined.py --profile`), because a replica of the loop would
measure the replica. This tool measures what happens INSIDE one
`controller.step`, which the loop cannot see, and holds the evidence that the
change does not move the arm.

`replay` is the behaviour check: it drives the real `TeleopController` with a
recorded or fabricated keypoint stream and stores the arm joint trajectory, so
two revisions can be compared joint by joint, tick by tick. Run it BEFORE and
AFTER a change (`--save`/`--cmp`).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g1_teleop import config as C
from g1_teleop import ik as IK
from g1_teleop import synthetic_source as SS
from g1_teleop.config import TeleopConfig
from g1_teleop.robot import G1Robot
from g1_teleop.teleop import TeleopController


def _stream(n=400):
    """A repeatable two-handed reach in the twin's pelvis frame."""
    tw = G1Robot(TeleopConfig())
    mujoco.mj_forward(tw.model, tw.data)
    pel = mujoco.mj_name2id(tw.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    p = tw.data.xpos[pel].copy()
    sh = {"left": tw.left_shoulder_world() - p, "right": tw.right_shoulder_world() - p}
    ln = {"left": (tw.upper_arm_left, tw.forearm_left),
          "right": (tw.upper_arm_right, tw.forearm_right)}
    home = {"left": tw.data.xpos[tw.left_wrist_body] - p,
            "right": tw.data.xpos[tw.right_wrist_body] - p}
    box = np.array([0.32, 0.0, 0.05])
    frames, _, _ = SS.grasp_lift_frames(sh, ln, home, box, 29.4, t_home=0.5, t_end=0.5)
    return frames[:n]


def _run(frames, cfg=None):
    """Drive the real controller; return the arm joint trajectory (ticks, 17)."""
    cfg = cfg or TeleopConfig()
    tw = G1Robot(cfg)
    ctl = TeleopController(tw, cfg)
    from g1_teleop.indices import ModelIndex
    ix = ModelIndex.resolve(tw.model)
    src = SS.SyntheticSource(frames)
    traj = []
    with contextlib.redirect_stdout(io.StringIO()):
        while True:
            f = src.grab()
            if f is None:
                break
            ctl.step(f)
            traj.append(np.array(tw.data.qpos[ix.upper_qpos], dtype=np.float64))
    return np.array(traj)


def cmd_verify(_):
    """mj_jac after mj_forward vs after mj_kinematics+mj_comPos, and the whole
    solve, at poses spread over the arm's range."""
    cfg = TeleopConfig()
    tw = G1Robot(cfg)
    m, d = tw.model, tw.data
    rng = np.random.default_rng(0)
    jacp_f, jacp_k, jacr = (np.zeros((3, m.nv)) for _ in range(3))
    worst_j, worst_x, worst_s = 0.0, 0.0, 0.0
    for trial in range(25):
        for qid, lim in zip(tw.ik_left_qpos, tw.ik_left_lim):
            d.qpos[qid] = rng.uniform(lim[0], lim[1])
        for qid, lim in zip(tw.ik_right_qpos, tw.ik_right_lim):
            d.qpos[qid] = rng.uniform(lim[0], lim[1])
        mujoco.mj_forward(m, d)
        xpos_f = d.xpos.copy()
        site_f = d.site_xpos.copy()
        mujoco.mj_jac(m, d, jacp_f, jacr, d.xpos[tw.left_wrist_body], tw.left_wrist_body)
        jac_f = jacp_f.copy()
        # same qpos, cheap refresh
        IK._kinematics(m, d)
        mujoco.mj_jac(m, d, jacp_k, jacr, d.xpos[tw.left_wrist_body], tw.left_wrist_body)
        worst_j = max(worst_j, float(np.abs(jac_f - jacp_k).max()))
        worst_x = max(worst_x, float(np.abs(xpos_f - d.xpos).max()))
        worst_s = max(worst_s, float(np.abs(site_f - d.site_xpos).max()))
    print("over 25 random arm poses:")
    print("  max |mj_jac(forward) - mj_jac(kinematics+comPos)| = %.3e" % worst_j)
    print("  max |xpos difference|                            = %.3e" % worst_x)
    print("  max |site_xpos difference|                       = %.3e" % worst_s)
    ok = worst_j == 0.0 and worst_x == 0.0 and worst_s == 0.0
    print("  IDENTICAL" if ok else "  DIFFERENT - mj_comPos missing or ordered wrong")
    return 0 if ok else 1


def cmd_profile(a):
    """Where one controller.step goes: retargeting, IK, smoothing, robot.forward."""
    import g1_teleop.teleop as T
    if getattr(a, "forward", False):
        IK._kinematics = lambda m, d: mujoco.mj_forward(m, d)
    frames = _stream(a.n)
    cfg = TeleopConfig()
    tw = G1Robot(cfg)
    ctl = TeleopController(tw, cfg)
    acc = dict(retarget=0.0, ik=0.0, forward=0.0, step=0.0)
    real_ik, real_rt, real_fw = T.solve_arm_ik, T.compute_arm_targets, tw.forward

    def timed(key, fn):
        def wrap(*args, **kw):
            t0 = time.perf_counter()
            out = fn(*args, **kw)
            acc[key] += time.perf_counter() - t0
            return out
        return wrap

    T.solve_arm_ik = timed("ik", real_ik)
    T.compute_arm_targets = timed("retarget", real_rt)
    tw.forward = timed("forward", real_fw)
    IK.COLLECT_STATS, IK.ITERATIONS[:] = True, []
    src = SS.SyntheticSource(frames)
    n = 0
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            while True:
                f = src.grab()
                if f is None:
                    break
                t0 = time.perf_counter()
                ctl.step(f)
                acc["step"] += time.perf_counter() - t0
                n += 1
    finally:
        T.solve_arm_ik, T.compute_arm_targets = real_ik, real_rt
        IK.COLLECT_STATS = False
    other = acc["step"] - acc["ik"] - acc["retarget"] - acc["forward"]
    print("one controller.step, mean over %d frames (%s):" % (n, a.label or "current"))
    for k in ("ik", "retarget", "forward"):
        print("  %-10s %7.3f ms  %5.1f%%" % (k, 1000 * acc[k] / n, 100 * acc[k] / acc["step"]))
    print("  %-10s %7.3f ms  %5.1f%%   (smoothing, One-Euro, bookkeeping)"
          % ("other", 1000 * other / n, 100 * other / acc["step"]))
    print("  %-10s %7.3f ms" % ("TOTAL", 1000 * acc["step"] / n))
    it = np.array(IK.ITERATIONS)
    print("IK iterations per arm-solve: mean %.2f, median %d, p95 %d, max %d, "
          "hit max_iter(%d) %.1f%% of solves"
          % (it.mean(), int(np.median(it)), int(np.percentile(it, 95)), it.max(),
             cfg.ik.max_iter, 100.0 * np.mean(it >= cfg.ik.max_iter)))
    return 0


def cmd_iters(a):
    return cmd_profile(a)


def cmd_sweep(a):
    """How far the COMMANDED arm joints move as max_iter is cut, against the
    max_iter=30 command. Twin-space on purpose: this compares one command with
    another, it is not a claim about achieved accuracy (TR16a) - the stepped
    check for that is run_integrated_combined.py --ik-max-iter."""
    import dataclasses
    frames = _stream(a.n)
    base = TeleopConfig()
    ref = _run(frames, base)
    print("max_iter   max |q - q(30)| rad   mean rad   ms per controller.step")
    for k in a.iters:
        cfg = dataclasses.replace(base, ik=dataclasses.replace(base.ik, max_iter=k))
        t0 = time.perf_counter()
        traj = _run(frames, cfg)
        ms = 1000 * (time.perf_counter() - t0) / len(traj)
        d = np.abs(traj - ref)
        print("   %3d           %.3e        %.3e     %6.2f" % (k, d.max(), d.mean(), ms))
    return 0


def cmd_stepped(a):
    """ACHIEVED error on the STEPPED model vs max_iter (TR16a: no twin residual).

    Uses the O25 harness (tools/teleop_physics_check.Rig + run_stream), base
    locked as collection runs, and reports the IK TARGET against the STEPPED
    wrist, each in its own pelvis frame - the same measure the O25 table uses -
    against the 45 mm palm guard.
    """
    import dataclasses
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import torch
    import walk_test as W
    import teleop_physics_check as TP
    frames = _stream(a.n)
    pol = torch.jit.load(W.POLICY_PATH)
    base = TeleopConfig()
    print("max_iter   target error mm: p50   p95    max   |  contacts  wrist END")
    for k in a.iters:
        cfg = dataclasses.replace(base, ik=dataclasses.replace(base.ik, max_iter=k))
        with contextlib.redirect_stdout(io.StringIO()):
            r = TP.run_stream(TP.Rig(cfg), frames, policy=pol, locked=True)
        te = 1000 * np.array(r["target_err"])
        print("   %3d              %6.1f %6.1f %6.1f   |  %6d   %.3f"
              % (k, np.nanmedian(te), np.nanpercentile(te, 95), np.nanmax(te),
                 r["plat_steps"], r["pitch_end"]))
    return 0


def cmd_replay(a):
    frames = (SS.load_recording(a.path).frames("raw")[0] if a.path else _stream(a.n))
    if a.forward:
        # Reproduce the PRE-2026-09-15 solver exactly: the only change was
        # _kinematics, so pointing it back at mj_forward is the old code path.
        IK._kinematics = lambda m, d: mujoco.mj_forward(m, d)
        print("(solver refresh forced back to mj_forward - the 'before' path)")
    traj = _run(frames)
    print("replayed %d ticks, %d arm dims" % traj.shape)
    if a.save:
        np.save(a.save, traj)
        print("saved %s" % a.save)
    if a.cmp:
        ref = np.load(a.cmp)
        if ref.shape != traj.shape:
            print("SHAPE CHANGED %s -> %s" % (ref.shape, traj.shape))
            return 1
        d = np.abs(ref - traj)
        print("max |after - before| = %.3e rad (tol %.1e), mean %.3e, ticks differing: %d of %d"
              % (d.max(), a.tol, d.mean(), int((d.max(axis=1) > a.tol).sum()), len(d)))
        worst = int(np.argmax(d.max(axis=0)))
        print("worst joint index %d, max %.3e rad" % (worst, d[:, worst].max()))
        print("TRAJECTORY UNCHANGED" if d.max() <= a.tol else "TRAJECTORY MOVED")
        return 0 if d.max() <= a.tol else 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify"); v.set_defaults(fn=cmd_verify)
    for name in ("profile", "iters"):
        p_ = sub.add_parser(name)
        p_.add_argument("-n", type=int, default=400)
        p_.add_argument("--label", default="")
        p_.add_argument("--forward", action="store_true",
                        help="profile the OLD path (mj_forward inside the solver)")
        p_.set_defaults(fn=cmd_profile)
    sw = sub.add_parser("sweep")
    sw.add_argument("-n", type=int, default=400)
    sw.add_argument("--iters", type=int, nargs="+", default=[30, 20, 16, 12, 10, 8, 6, 4, 3, 2, 1])
    sw.set_defaults(fn=cmd_sweep)
    st = sub.add_parser("stepped")
    st.add_argument("-n", type=int, default=400)
    st.add_argument("--iters", type=int, nargs="+", default=[30, 16, 12, 10, 8, 6, 4])
    st.set_defaults(fn=cmd_stepped)
    r = sub.add_parser("replay")
    r.add_argument("path", nargs="?", default=None, help="a .npz keypoint recording; omit for the fabricated stream")
    r.add_argument("-n", type=int, default=400)
    r.add_argument("--save", default=None)
    r.add_argument("--cmp", default=None)
    r.add_argument("--tol", type=float, default=1e-9)
    r.add_argument("--forward", action="store_true", help="use mj_forward in the solver (the old path)")
    r.set_defaults(fn=cmd_replay)
    a = ap.parse_args()
    raise SystemExit(a.fn(a))


if __name__ == "__main__":
    main()
