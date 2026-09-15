"""OPEN-LOOP REPLAY: does the recorded action reproduce the recorded episode?

    python tools/replay_check.py recordings/episodes/ep_seed0000.npz
    python tools/replay_check.py "recordings/episodes/ep_seed*.npz" --negative

The most important test in the project. It is the only thing that proves the
recorder logged the quantity that ACTUALLY drove the robot. If it logged the
wrong one - twin qpos instead of `data.ctrl`, or the action vector written
straight into `ctrl[upper_ctrl]` without the permutation - behavioural cloning
trains on a signal that never moved anything, every policy fails, and no amount
of architecture work fixes it.

WHAT "OPEN LOOP" MEANS HERE
---------------------------
Nothing is driven from live state. Each tick takes its 22-D action from the FILE
and routes it exactly as deployment will:

    17 joint targets -> ctrl[upper_ctrl] VIA spec.upper_ctrl_from_action (the
                        inverse permutation; ctrl order is [waist, left, right]
                        while the action is [left, right, waist])
    2 gripper dims   -> GraspWeld.update (the weld COMMAND, D11/D14)
    3 velocity dims  -> the locomotion policy observation

The base lock is the one thing NOT replayed from the file: it is a state-driven
predicate (D12), so replay runs the predicate and the tick it fires at is a
RESULT, not an input. A lock that fires at a different tick means the state had
already diverged, which is why it is reported.

ORDER OF OPERATIONS mirrors `run_episode` step for step: predicate at the 25 Hz
tick, lock sync, leg PD, upper ctrl, pads at zero (TR17), `weld.update`,
`mj_step`, counter, then the policy query at the control decimation while free.

WHAT A DIVERGENCE MEANS, AND WHAT IT DOES NOT
----------------------------------------------
The demonstrator drives `ctrl` at 500 Hz along a smooth interpolation; the
recording samples that command at 25 Hz (D4) and replay holds each sample for 20
steps. A zero-order hold at 40 ms is a REAL difference in the commanded
trajectory, and it is also exactly what a 25 Hz policy will do at deployment. So
this test reports two different things and never conflates them: numerical
divergence (expected, bounded, reported per dim group) and whether the TASK still
succeeds (weld engages, box lifts, placement inside 0.10 m).

The negative control is not optional: the same actions replayed against a
DIFFERENT box spawn must FAIL. If they still succeed, the replay is not actually
depending on the recorded actions and proves nothing.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g1_data import spec
from g1_data.phases import LockConfig, LockPredicate, PlatformGeometry
from g1_data.recorder import load_episode
from g1_data.reset import LocomotionCarryover, reset_episode, state_fingerprint
from g1_teleop.base_lock import BaseLock
from g1_teleop.box_reset import sample_box_pose
from g1_teleop.config import TeleopConfig
from g1_teleop.contact_contract import contract_of, load_model
from g1_teleop.grasp import GraspWeld
from g1_teleop.indices import ModelIndex
import walk_test as W

GROUPS = (("box", spec.BOX_POS), ("box_quat", spec.BOX_QUAT),
          ("base", spec.BASE_POS), ("base_quat", spec.BASE_QUAT),
          ("palm_L", spec.PALM_L_POS), ("palm_R", spec.PALM_R_POS),
          ("gripper", slice(spec.GRIP_L, spec.GRIP_R + 1)),
          ("arm_L", spec.ARM_L_Q), ("arm_R", spec.ARM_R_Q),
          ("waist", spec.WAIST_Q))
POS_GROUPS = ("box", "base", "palm_L", "palm_R")
GOAL_XY = np.array([1.5, -1.5])          # DemoConfig.goal_xy, scene.xml
LIFT_M = 0.055                           # LockConfig.lift_min: "carrying"
PLACE_LIMIT = 0.10                       # Q4
# What open-loop replay can actually assert (2026-09-16 diagnosis, below).
# Set from the measured 5-episode population, not chosen to make a run pass.
LOCKED_BOX_M = 0.025                     # box divergence while the base is welded
TICK_TOL = 2                             # weld / lock firing tick, at 25 Hz


def replay(path: str, threshold: float = 0.01, negative_seed=None, verbose=True,
           wrong_permutation: bool = False, vel_at_step=None,
           negative_shift: float = 0.0):
    """`wrong_permutation` is the POSITIVE control: it writes action[0:17]
    straight into ctrl[upper_ctrl], the exact mistake spec.py exists to prevent.
    A replay test that cannot tell that apart from correct routing is not a test.

    `vel_at_step` is a diagnostic hook: {step_index: (vx, vy, wz)} sampled at the
    demonstrator's own 50 Hz control rate, used once to separate "the 25 Hz
    recording under-samples the command" from "an open-loop walk cannot be
    reproduced at any rate" (NOTES 2026-09-16)."""
    import torch
    arrays, meta = load_episode(path)
    states, actions = arrays["states"], arrays["actions"]
    seed = int(meta["seed"])
    T = len(actions)

    cfg = TeleopConfig()
    model = load_model(cfg)
    model.opt.timestep = cfg.loco.sim_dt
    if contract_of(model) != meta["contact_contract"]:
        raise AssertionError(
            "contact contract mismatch (D18): the episode was recorded under\n  %s\n"
            "and this model is\n  %s\nReplaying across that would test physics the "
            "episode was never produced in." % (meta["contact_contract"],
                                                contract_of(model)))
    data = mujoco.MjData(model)
    ix = ModelIndex.resolve(model)
    sp = spec.SpecLayout.resolve(model, ix)
    carry = LocomotionCarryover()
    policy = torch.jit.load(W.POLICY_PATH)

    # 1. Reset determinism FIRST. If this differs, the reset is the problem and
    #    blaming the actions for what follows would be wrong (reset.py docstring:
    #    LSTM buffers, qacc_warmstart, teleop filters).
    reset_episode(model, data, ix, cfg, seed, carry=carry, policy=policy)
    fp = state_fingerprint(data, ix)
    fp_ok = (fp == meta["reset_fingerprint"])
    report = dict(path=path, seed=seed, ticks=T, reset_fingerprint_ok=fp_ok,
                  negative=negative_seed is not None)
    if verbose:
        print("\n=== %s  seed %d, %d ticks ===" % (os.path.basename(path), seed, T))
        print("reset fingerprint: %s" % ("MATCH" if fp_ok else "MISMATCH - stop here"))
    if not fp_ok:
        return report

    # 2. Restore the episode's OWN initial state. `reset_episode` does not place
    #    the base at the walk-in start - that lives in `run_episode` - so row 0 of
    #    the recorded raw qpos/qvel is the honest starting point, and using it
    #    keeps this a test of the ACTIONS rather than of setup code.
    data.qpos[:] = arrays["qpos"][0]
    data.qvel[:] = arrays["qvel"][0]
    if negative_seed is not None:
        pos, quat = sample_box_pose(cfg.box, negative_seed)
        data.qpos[ix.box_qpos] = np.concatenate([pos, quat])
        report["negative_seed"] = int(negative_seed)
        report["negative_spawn_shift_mm"] = float(
            1000 * np.linalg.norm(pos[:2] - np.asarray(meta["box_spawn_xy"])))
    elif negative_shift:
        # A seed swap moves the spawn by whatever the sample region gives - 36 mm
        # on some pairs, which the weld gate ACCEPTS (palm radius 0.16 m), so it
        # cannot discriminate. This variant displaces the box by a stated amount
        # well outside that tolerance, staying on the platform (|y| <= 0.23 with
        # half-extent 0.32 and box half-width 0.09), so "the weld must not fire"
        # becomes a real test of whether the hand trajectory depends on the box.
        y = float(data.qpos[ix.box_qpos][1])
        new_y = float(np.clip(y + negative_shift if y <= 0 else y - negative_shift,
                              -0.23, 0.23))
        report["negative_shift_mm"] = float(1000 * abs(new_y - y))
        data.qpos[ix.box_qpos][1] = new_y
    mujoco.mj_forward(model, data)

    box_init = np.array(data.qpos[ix.box_qpos][:3])
    shove_at_weld_mm = float("nan")
    weld = GraspWeld(model)
    lock = BaseLock(model)
    pred = LockPredicate(PlatformGeometry.resolve(model), LockConfig())
    L = cfg.loco
    DEF = np.asarray(L.default_angles, dtype=np.float64)
    KPS = np.asarray(L.kps, dtype=np.float32)
    KDS = np.asarray(L.kds, dtype=np.float32)
    CMDS = np.asarray(L.cmd_scale, dtype=np.float32)
    locked_leg_pos = None
    lock_tick = release_tick = -1
    weld_tick = -1
    div = np.zeros((T, spec.STATE_DIM))
    box_lift_max = -9.9

    for t in range(T):
        a = actions[t].astype(np.float64)
        # ---- the predicate, on LIVE state (a result, not an input) ----------
        live = sp.build_state(model, data, ix, sync=False)
        pred.update(live)
        if pred.locked and not lock.locked(data):
            lock.lock(model, data)
            locked_leg_pos = DEF.copy()
            lock_tick = t if lock_tick < 0 else lock_tick
        elif not pred.locked and lock.locked(data):
            lock.release(model, data, policy=policy)
            locked_leg_pos = None
            carry.action = np.zeros_like(carry.action)
            carry.target_leg_pos = DEF.copy()
            release_tick = t if release_tick < 0 else release_tick
        # ---- divergence at the instant the recording sampled ---------------
        full = sp.build_state(model, data, ix, sync=True)
        div[t] = np.abs(full - states[t].astype(np.float64))
        geo_lift = float(full[spec.BOX_POS][2]) - (0.75 + 0.09)
        box_lift_max = max(box_lift_max, geo_lift)
        # ---- route the RECORDED action -------------------------------------
        upper = (a[spec.UPPER_A] if wrong_permutation
                 else spec.upper_ctrl_from_action(a))
        vel = a[spec.VEL_A]
        grip = float(a[spec.GRIP_L_A])
        for k in range(spec.PHYSICS_STEPS_PER_TICK):
            lq, ldq = data.qpos[ix.leg_qpos], data.qvel[ix.leg_qvel]
            leg_target = carry.target_leg_pos if locked_leg_pos is None else locked_leg_pos
            data.ctrl[ix.leg_ctrl] = (leg_target - lq) * KPS + (0.0 - ldq) * KDS
            data.ctrl[ix.upper_ctrl] = upper
            data.ctrl[ix.pad_ctrl] = 0.0            # TR17
            eng = weld.update(model, data, grip)[0]
            if eng and weld_tick < 0:
                weld_tick = t
                shove_at_weld_mm = 1000 * float(np.linalg.norm(
                    np.asarray(data.qpos[ix.box_qpos][:3]) - box_init))
            mujoco.mj_step(model, data)
            carry.counter += 1
            if carry.counter % L.control_decimation == 0 and locked_leg_pos is None:
                n = L.num_actions
                phz = ((carry.counter * L.sim_dt) % L.gait_period) / L.gait_period
                carry.obs[:3] = data.qvel[ix.base_angvel_qvel] * L.ang_vel_scale
                carry.obs[3:6] = W.get_gravity_orientation(data.qpos[ix.base_quat_qpos])
                live_vel = vel if vel_at_step is None else vel_at_step.get(
                    t * spec.PHYSICS_STEPS_PER_TICK + k, vel)
                carry.obs[6:9] = live_vel * CMDS
                carry.obs[9:9 + n] = (data.qpos[ix.leg_qpos] - DEF) * L.dof_pos_scale
                carry.obs[9 + n:9 + 2 * n] = data.qvel[ix.leg_qvel] * L.dof_vel_scale
                carry.obs[9 + 2 * n:9 + 3 * n] = carry.action
                carry.obs[9 + 3 * n:9 + 3 * n + 2] = [np.sin(2 * np.pi * phz),
                                                      np.cos(2 * np.pi * phz)]
                carry.action = policy(torch.from_numpy(carry.obs).unsqueeze(0)
                                      ).detach().numpy().squeeze()
                carry.target_leg_pos = carry.action * L.action_scale + DEF

    # ---- outcome -----------------------------------------------------------
    box = np.asarray(data.qpos[ix.box_qpos][:3])
    place_err = float(np.linalg.norm(box[:2] - GOAL_XY))
    report.update(
        weld_engaged=weld_tick >= 0, weld_tick=weld_tick,
        recorded_weld_tick=meta["weld_engage_tick"],
        box_lift_max_m=float(box_lift_max), lifted=bool(box_lift_max >= LIFT_M),
        placement_error_m=place_err, placed=bool(place_err <= PLACE_LIMIT),
        recorded_placement_m=float(meta["outcome"]["placement_error"]),
        lock_tick=lock_tick, recorded_lock_tick=meta["lock_engage_tick"],
        release_tick=release_tick, recorded_release_tick=meta["lock_release_tick"],
        groups={}, first_over=None, div=div, meta=meta)
    report["task_success"] = bool(report["weld_engaged"] and report["lifted"]
                                  and report["placed"])
    for name, sl in GROUPS:
        g = div[:, sl]
        report["groups"][name] = dict(max=float(g.max()), rms=float(np.sqrt((g ** 2).mean())))
    pos_div = np.max(np.stack([div[:, dict(GROUPS)[n]].max(axis=1)
                               for n in POS_GROUPS]), axis=0)
    over = np.flatnonzero(pos_div > threshold)
    report["first_over"] = int(over[0]) if len(over) else None
    report["threshold_m"] = threshold
    report["final_pos_div_m"] = float(pos_div[-1])

    # ---- the two claims, kept apart ---------------------------------------
    # MANIPULATION: while the base is welded the locomotion channel is inert
    # (D17 zeroes it, the policy is not queried), so this window isolates the 17
    # joint targets and the gripper command - the part open-loop replay CAN
    # assert. WALK: the other windows, reported as a measurement, not a gate.
    lo = meta["lock_engage_tick"]
    hi = meta["lock_release_tick"] if meta["lock_release_tick"] > 0 else T
    gi = dict(GROUPS)
    win = lambda sl, a_, b_: float(div[a_:b_, sl].max()) if b_ > a_ else float("nan")
    report["locked"] = dict(
        ticks=[int(lo), int(hi)],
        box=win(gi["box"], lo, hi), palm_L=win(gi["palm_L"], lo, hi),
        palm_R=win(gi["palm_R"], lo, hi), arm_L=win(gi["arm_L"], lo, hi),
        arm_R=win(gi["arm_R"], lo, hi), base=win(gi["base"], lo, hi))
    report["walk"] = dict(
        walk_in_base=win(gi["base"], 0, lo), after_release_base=win(gi["base"], hi, T))
    dw = (abs(weld_tick - meta["weld_engage_tick"])
          if weld_tick >= 0 and meta["weld_engage_tick"] >= 0 else 10 ** 6)
    dl = (abs(lock_tick - meta["lock_engage_tick"])
          if lock_tick >= 0 and meta["lock_engage_tick"] >= 0 else 10 ** 6)
    report["weld_tick_delta"], report["lock_tick_delta"] = dw, dl
    report["box_shove_at_weld_mm"] = shove_at_weld_mm
    report["box_total_move_mm"] = 1000 * float(np.linalg.norm(
        np.asarray(data.qpos[ix.box_qpos][:3]) - box_init))
    # The lock tick is deliberately NOT part of this criterion. It is a function
    # of the BASE settling (the predicate tests base displacement over a gait
    # period), so it belongs to the locomotion channel that open-loop replay
    # provably cannot reproduce - measured 2026-09-16: dLock 0,8,1,1,2 ticks
    # across five episodes, tracking the 21-26 mm walk-in divergence. It is
    # reported as evidence either way, because a large shift is still the signal
    # that state diverged before the grasp.
    report["manipulation_ok"] = bool(
        report["weld_engaged"] and dw <= TICK_TOL
        and report["locked"]["box"] <= LOCKED_BOX_M)

    if verbose:
        print("divergence per dim group (|replay - recorded|, metres / rad / unit):")
        print("  %-10s %10s %10s" % ("group", "max", "rms"))
        for name, _ in GROUPS:
            g = report["groups"][name]
            print("  %-10s %10.5f %10.5f" % (name, g["max"], g["rms"]))
        print("first tick with any position group over %.0f mm: %s (of %d)"
              % (1000 * threshold,
                 report["first_over"] if report["first_over"] is not None else "never", T))
        print("task: weld %s (tick %s vs recorded %s) | lift %.3f m %s | placement "
              "%.4f m %s (recorded %.4f)"
              % ("ENGAGED" if report["weld_engaged"] else "NEVER FIRED",
                 weld_tick, meta["weld_engage_tick"], box_lift_max,
                 "OK" if report["lifted"] else "FAIL", place_err,
                 "OK" if report["placed"] else "FAIL",
                 meta["outcome"]["placement_error"]))
        print("base lock: fired tick %s (recorded %s), released %s (recorded %s)"
              % (lock_tick, meta["lock_engage_tick"], release_tick,
                 meta["lock_release_tick"]))
        lk = report["locked"]
        print("LOCKED window %s: box %.4f m, palms %.4f/%.4f m, arms %.4f/%.4f rad"
              % (lk["ticks"], lk["box"], lk["palm_L"], lk["palm_R"],
                 lk["arm_L"], lk["arm_R"]))
        print("WALK windows: base %.3f m walking in, %.3f m after release"
              % (report["walk"]["walk_in_base"], report["walk"]["after_release_base"]))
        print("MANIPULATION reproduced: %s | whole task: %s%s"
              % (report["manipulation_ok"], report["task_success"],
                 "  [negative control]" if negative_seed is not None else ""))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.01, help="m, position divergence")
    ap.add_argument("--negative", action="store_true",
                    help="also replay each episode against a DIFFERENT box spawn")
    ap.add_argument("--wrong-permutation", action="store_true",
                    help="positive control: route action[0:17] straight to ctrl")
    ap.add_argument("--negative-shift", type=float, default=0.25,
                    help="m, control A: displace the box beyond the 0.16 m weld gate")
    ap.add_argument("--negative-offset", type=int, default=500,
                    help="seed offset used for the negative control spawn")
    a = ap.parse_args()
    files = sorted({p for g in a.paths for p in glob.glob(g)})
    if not files:
        raise SystemExit("no episodes match %s" % a.paths)
    reps, negs, shifted = [], [], []
    for p in files:
        r = replay(p, threshold=a.threshold, wrong_permutation=a.wrong_permutation)
        reps.append(r)
        if a.negative:
            negs.append(replay(p, threshold=a.threshold,
                               negative_seed=int(r["seed"]) + a.negative_offset))
            shifted.append(replay(p, threshold=a.threshold,
                                  negative_shift=a.negative_shift))
    good = [r for r in reps if r.get("manipulation_ok")]
    bad = [r for r in reps if not r.get("manipulation_ok")]

    print("\n================ SUMMARY over %d episode(s) ================" % len(files))
    print("reset fingerprint matched      : %d/%d"
          % (sum(r["reset_fingerprint_ok"] for r in reps), len(files)))
    print("MANIPULATION reproduced        : %d/%d  (weld and lock within %d ticks, "
          "box within %.0f mm while welded)"
          % (len(good), len(files), TICK_TOL, 1000 * LOCKED_BOX_M))
    print("whole task reproduced          : %d/%d  (see the walk numbers below)"
          % (sum(bool(r.get("task_success")) for r in reps), len(files)))
    if reps:
        print("\nper episode:")
        print("  %-16s %8s %8s %6s %6s %9s %9s"
              % ("episode", "boxLOCK", "armLOCK", "dWeld", "dLock", "walkIn", "afterRel"))
        for r in reps:
            lk = r.get("locked", {})
            print("  %-16s %8.4f %8.4f %6s %6s %9.3f %9.3f"
                  % (os.path.basename(r["path"])[:16], lk.get("box", float("nan")),
                     max(lk.get("arm_L", 0.0), lk.get("arm_R", 0.0)),
                     r.get("weld_tick_delta"), r.get("lock_tick_delta"),
                     r["walk"]["walk_in_base"], r["walk"]["after_release_base"]))
        print("\nunits: metres except armLOCK (rad); dWeld/dLock are 25 Hz ticks.")
    for r in bad:
        print("MANIPULATION NOT REPRODUCED: %s (weld %s, dWeld %s, dLock %s, boxLOCK %.4f)"
              % (os.path.basename(r["path"]), r.get("weld_engaged"),
                 r.get("weld_tick_delta"), r.get("lock_tick_delta"),
                 r.get("locked", {}).get("box", float("nan"))))
    ctrl_ok = True
    if shifted:
        succeeded = [n for n in shifted if n.get("task_success")]
        fired = [n for n in shifted if n.get("weld_engaged")]
        print("\nnegative control A - box displaced beyond the 0.16 m weld gate:")
        for n in shifted:
            print("    seed %d, box moved %3.0f mm: weld %s, box ended %.0f mm away, "
                  "task %s"
                  % (n["seed"], n.get("negative_shift_mm", 0.0),
                     ("FIRED after the arm shoved it %.0f mm"
                      % n["box_shove_at_weld_mm"]) if n["weld_engaged"] else "refused",
                     n["box_total_move_mm"], "SUCCEEDS" if n["task_success"] else "fails"))
        print("  task failed in %d/%d; weld refused in %d/%d."
              % (len(shifted) - len(succeeded), len(shifted),
                 len(shifted) - len(fired), len(shifted)))
        print("  A weld that fires anyway is not the test failing: the box is a free "
              "body the arm can reach, so the recorded hand path SHOVES it into the "
              "gate - measured, and the episode is wrecked either way.")
        ctrl_ok = ctrl_ok and not succeeded
    if negs:
        succeeded = [n for n in negs if n.get("task_success")]
        print("\nnegative control B - actions replayed against ANOTHER seed's spawn:")
        for n in negs:
            print("    seed %d -> %d, spawn moved %3.0f mm: weld %s, placement %.3f m, "
                  "task %s"
                  % (n["seed"], n["negative_seed"], n["negative_spawn_shift_mm"],
                     "FIRED" if n["weld_engaged"] else "refused",
                     n["placement_error_m"],
                     "SUCCEEDS" if n["task_success"] else "fails"))
        print("  the task failed in %d/%d, as it must. A weld that still fires on a "
              "small shift is the 0.16 m gate tolerance (D11), not a broken test - "
              "control A is the one that isolates that."
              % (len(negs) - len(succeeded), len(negs)))
        ctrl_ok = ctrl_ok and not succeeded
    ok = (len(good) == len(files)
          and all(r["reset_fingerprint_ok"] for r in reps) and ctrl_ok)
    print("\nVERDICT")
    print("  the recorder logged the quantity that drove the robot: %s" % ok)
    print("  what that rests on: reset fingerprints match, the weld fires within "
          "%d tick(s) of the recorded tick, the box tracks within %.0f mm while the "
          "base is welded, and both negative controls fail as they must."
          % (TICK_TOL, 1000 * LOCKED_BOX_M))
    print("  what it does NOT claim: that open-loop replay reproduces the WALK. It "
          "does not, and no recording rate fixes that - see the walk columns and "
          "NOTES.md 2026-09-16.")
    print("REPLAY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
