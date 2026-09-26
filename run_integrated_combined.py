"""Vision-driven locomotion evaluation — SINGLE WINDOW, for presentation.

Same simulation as run_integrated_vision.py (walking policy on legs, ZED arm IK
on arms, locomotion command from a vision strategy) but rendered into ONE
window: camera feed on the left, MuJoCo on the right. Intended for showing an
adviser the failure modes rather than describing them.

    python run_integrated_combined.py pelvis   # thesis 3.3.4 as written
    python run_integrated_combined.py lean     # mode-switched joystick
    python run_integrated_combined.py keyboard # decoupled fallback

The overlay carries the evidence:

    TRAVELLED   distance the robot has moved from where it started. Under the
                pelvis strategy this climbs while the demonstrator stands
                still, which is the whole argument in one number.
    LOCO cmd    the live velocity command driving the legs.
    diagnostics strategy internals — pelvis visibility, trigger state, lean
                magnitudes — so the cause is visible alongside the effect.

An UNCOMMANDED banner appears whenever the robot is moving while the
demonstrator is not deliberately commanding motion. That is the moment worth
capturing on video.

Camera (click the window first):
  a/d rotate   w/s tilt   +/- zoom   r reset
  space  emergency stop and re-anchor
  q      quit

TELEOPERATED GRASP (2026-09-15) - what makes it physically possible
-------------------------------------------------------------------
A live camera test confirmed O17: free-standing, the base settles at 0.47-0.59 m
from the box while the arms serve 0.28-0.36 m, so the palms never reach the box
and GraspWeld (correctly) refuses. Four changes, matching the scripted
demonstrator rather than inventing a second mechanism:

  BASE LOCK (D12)  g1_data.phases.LockPredicate on the 47-D state at 25 Hz, the
                   same predicate and thresholds the demonstrator uses. Walk the
                   robot to the box and stand still: it LOCKS when settled at a
                   0.26-0.40 m standoff, head-on. While locked the locomotion
                   policy is NOT queried and the legs are PD-held at
                   DEFAULT_ANGLES; walking keys do nothing. It RELEASES itself
                   once the welded box is lifted 55 mm (carry), or once the box
                   is placed at the goal and the hands withdrawn.
  u                manual ABORT of the lock, logged. Re-locking is inhibited until
                   the robot leaves the standoff/lateral/heading band (walk or turn).
  g                grasp command. The weld still engages only when all three
                   geometric conditions hold - the overlay shows which one fails.
  b                B-prime hand<->pickup-platform contact exclusion on/off
                   (default ON; g1_teleop/contact_contract.py).
  o                overlay: MINIMAL (default, 4 lines, for piloting) <-> FULL
                   (every number, for debugging and reports). Remembered for the
                   session.
  pads             held at ZERO; the command goes to GraspWeld.update only (TR17).

Test fixtures, no camera:
  python run_integrated_combined.py keyboard 0 --synthetic grasp --headless --start-standoff 0.32
"""
import argparse
import dataclasses
import os
import sys
import threading
import time

import cv2
import numpy as np
import mujoco
import torch

from g1_teleop.config import TeleopConfig
from g1_teleop.robot import G1Robot
from g1_teleop.teleop import TeleopController
from g1_teleop.zed_source import ZEDSource
from g1_teleop.overlay import draw_skeleton
from g1_teleop import config as C
from g1_teleop.indices import ModelIndex
from g1_teleop.box_reset import reset_box
from g1_teleop.grasp import GraspWeld
from g1_teleop.base_lock import BaseLock
from g1_teleop import contact_contract as CC
from g1_data import spec
from g1_data.phases import LockConfig, LockPredicate, PlatformGeometry
from locomotion_input import PelvisVelocity, LeanJoystick, KeyboardCommand

# ── Paths (EDIT THESE) ────────────────────────────────────────────────────────
# G1_POLICY_PATH overrides the policy path per machine; unset keeps the original.
POLICY_PATH = os.environ.get(
    "G1_POLICY_PATH",
    r"D:\Charles_Aninon\Thesis Project\unitree_rl_gym\deploy\pre_train\g1\motion.pt")
# G1_SCENE_PATH overrides the scene path per machine; unset keeps the original.
SCENE_PATH = os.environ.get(
    "G1_SCENE_PATH", r"D:\Charles_Aninon\Thesis Project\scene.xml")

# ── Display ───────────────────────────────────────────────────────────────────
PANEL_H = 620
ZED_PANEL_W = 760
MJ_PANEL_W = 760
MUJOCO_W, MUJOCO_H = 960, 720      # offscreen render size (<= scene.xml offwidth)
WINDOW = "Vision locomotion evaluation — ZED | MuJoCo"

CAM_LOOKAT = [1.2, 0.0, 0.8]
CAM_DISTANCE = 4.5
CAM_AZIMUTH = 0.0
CAM_ELEVATION = -20.0

# ── Walking policy config ────────────────────────────────────────────────────
SIM_DT = 0.002
CONTROL_DECIMATION = 10
KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)
ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.25, 1.0, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
NUM_ACTIONS, NUM_OBS = 12, 47

IDLE_THRESHOLD, HOLD_KP, HOLD_MAX, HOLD_DEADBAND = 0.05, 0.8, 0.25, 0.02

# ── Grasp range ───────────────────────────────────────────────────────────────
# Moved into g1_teleop/config.py (GraspConfig). As module-level literals here
# they were invisible to every assertion, so nothing could check the spawn
# region against them — the same failure class as O3. `assert_reach_fits` now
# does, at model load.
GRASP_MIN, GRASP_MAX = TeleopConfig().grasp.grasp_min, TeleopConfig().grasp.grasp_max

# ── Heading hold ──────────────────────────────────────────────────────────────
# The position-hold loop corrects x and y but nothing corrected yaw, so while
# the robot marched in place its facing drifted freely and never came back.
# These close that loop: capture the heading when the operator stops commanding
# motion, and feed a small corrective turn to hold it.
HOLD_YAW_KP = 1.2         # rad/s per radian of heading error
HOLD_YAW_MAX = 0.25       # cap on corrective turn rate
HOLD_YAW_DEADBAND = 0.03  # ignore errors under ~1.7 degrees

# ── O28: the arm hold during a tracking dropout ──────────────────────────────
# A frame with no usable body never reaches `controller.step` (the branch at
# `if frame.keypoints_3d:` below), so `SmoothingConfig.max_coast_frames` - which
# lives INSIDE `TeleopController._coast` - is never consulted on this path.
# Measured 2026-09-16 with the synthetic fixture: at gaps of 5, 50 and 500
# frames `controller.step` was called 0 times and `arm_targets` was byte-for-
# byte constant for the whole window, because the loop re-applies the same latch
# to `data.ctrl[ix.upper_ctrl]` on every physics step. A 500-frame gap held the
# arms for 17.0 s with no bound and no signal beyond a status string.
#
# Two harms: the operator sees a robot that has stopped responding, and the
# RECORDER writes those ticks as a deliberate hold - 39 of 39 consecutive ticks
# at exactly zero arm delta in the measurement, indistinguishable from the
# demonstrator choosing to stay still.
#
# This bounds the hold's VALIDITY. The arms keep their last pose past the bound
# (commanding motion nobody asked for is strictly worse while a box is welded to
# the hands), but the ticks stop counting as demonstration data and the operator
# gets a loud signal. The per-tick `tracking_ok` array records which ticks were
# backed by a live tracked frame; see g1_data/recorder.py.
NO_BODY_HOLD_S = 0.40     # seconds of stale arm command before ticks are degraded

# ── Why the robot always marches in place ────────────────────────────────────
# The policy's observation includes a gait phase as sin/cos of a clock that runs
# unconditionally. It was trained with that clock always cycling, including at
# zero velocity command, so it learned that "no velocity" means "step in place".
# There is no standstill behaviour inside the network, because balance IS the
# stepping — the gait continuously repositions the support polygon under the
# centre of mass.
#
# Two mitigations were implemented and rejected:
#   - Freezing the leg targets removes the balance controller entirely; the
#     robot topples within seconds.
#   - Freezing the phase input while still querying the policy preserves balance
#     but puts the network outside its training distribution, producing an
#     unnatural stance and unstable walk/stop transitions.
# Both are documented as a limitation rather than worked around further.

# Gait cadence. This is the value the policy was trained with. It is not
# adjustable from here: attempts to slow it (constant slower period, and an
# idle/moving blend) destabilised the gait, and the measured leg cadence did
# not follow the commanded clock. Stepping rate is fixed inside the network.
GAIT_PERIOD = 0.8

# qpos/qvel/ctrl offsets are resolved by joint and actuator NAME at model load
# (g1_teleop/indices.py), never hardcoded — the palm-pad gripper inserts a slide
# joint inside each wrist_yaw_link, shifting right-arm indices while leaving the
# legs alone, so a stale literal would break silently on one half of the robot.
# The upper-body joint list lives in config.UPPER_BODY_JOINTS.


def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat
    return np.array([2 * (-qz * qx + qw * qy),
                     -2 * (qz * qy + qw * qx),
                     1 - 2 * (qw * qw + qz * qz)])


def yaw_from_quat(quat):
    qw, qx, qy, qz = quat
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def world_to_body(v, yaw):
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def fit_to_box(img, w, h):
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    canvas = np.zeros((h, w, 3), dtype=img.dtype)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh))
    return canvas


def text(img, s, org, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)



GREEN, RED, AMBER, GREY, CYAN = (0, 220, 0), (0, 0, 255), (0, 170, 255), (200, 200, 200), (255, 255, 0)
WRIST_JOINTS = ("left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")
WEDGE_RAD = 0.8          # TR23: wrist-pitch deviation from command that means a jam
FRAME_STEPS = 17         # headless synthetic: one frame per 17 steps = 29.4 Hz
ABORT_MARGIN_FWD = 0.05  # m past the predicate's standoff band   } how far the robot must
ABORT_MARGIN_LAT = 0.05  # m past its lateral limit                } leave the lock geometry
ABORT_MARGIN_HEAD = 10.0 # deg past its heading limit              } before an abort expires
NEAR_BOX_M = 1.0         # minimal overlay: show a "how to lock" action inside this range
HINT_SECONDS = 8.0       # how long the key hints stay up before the FULL view owns them


def chips(img, x, y, items, scale=0.5):
    """Draw [(text, colour), ...] left to right on one line."""
    for s_, col in items:
        text(img, s_, (x, y), scale, col)
        x += cv2.getTextSize(s_, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] + 12
    return x


def mark(ok):
    return ("[OK]" if ok else "[X] "), (GREEN if ok else RED)


# Display refresh rate. Rendering and key polling are decoupled from both the
# physics rate and the camera rate so neither can stall the other.
# Must be a multiple of CONTROL_DECIMATION, because the render check is only
# reached on control ticks. 20 steps x 0.002 s = 40 ms => 25 Hz display.
RENDER_DECIMATION = 20


def lock_geometry_ok(t, c):
    """The lock predicate's GEOMETRY terms only (standoff band, lateral, heading).

    The abort inhibit clears when these fail, not when `lock_ok` does: `lock_ok`
    also tests stillness, and the release jolt plus the resumed march break
    stillness within ~0.1 s (measured), which cleared the inhibit immediately and
    let the predicate re-lock a robot the operator had just unlocked. Walking or
    turning out of the band is a deliberate act; marching in place is not.

    With MARGINS: the exact band is not enough either. Measured on the headless
    abort test, the release transient alone swung heading to -6.1 deg and lateral
    to -35 mm within 0.24 s, against 5 deg / 50 mm limits. The margins are sized to
    take a deliberate step or turn, well past that transient; they gate only the
    operator-abort inhibit, never the lock itself."""
    return bool(c.fwd_lo - ABORT_MARGIN_FWD <= t["fwd"] <= c.fwd_hi + ABORT_MARGIN_FWD
                and abs(t["lat"]) <= c.lat_max + ABORT_MARGIN_LAT
                and abs(t["head"]) <= c.head_max_deg + ABORT_MARGIN_HEAD)


def wrist_deviation(data, wrist_ids, pitch_only=False):
    """(max |achieved - commanded| rad, joint) over the wrist joints."""
    best, name = 0.0, ""
    for qa, act, n in wrist_ids:
        if pitch_only and "pitch" not in n:
            continue
        e = abs(float(data.qpos[qa] - data.ctrl[act]))
        if e > best:
            best, name = e, n
    return best, name


def grasp_action(diag, g, weld_on, standoff, band):
    """(line, colour) for the ONE grasp line of the minimal overlay.

    An ACTION, not a measurement, and only the single most-blocking condition.
    Priority is CAUSAL rather than by violation size: palms that are not at the
    box make the other two conditions meaningless - a wide stance a metre away
    reads as "hands apart" when the real problem is that nothing is near the box.
    So: near -> straddle -> opposition.
    """
    if weld_on:
        return "HELD", GREEN
    if not diag:
        return "NO HANDS TRACKED", AMBER
    near = diag["d_left"] <= g.palm_radius and diag["d_right"] <= g.palm_radius
    straddle = g.sep_min <= diag["sep"] <= g.sep_max
    opposite = diag["opposed"] <= g.opposed_dot
    if near and straddle and opposite:
        return "READY - press g", GREEN
    if not near:
        # Distinguish the two ways the palms can be far from the box: the robot
        # is not there yet, or it is and the arms are not out.
        return ("STEP CLOSER" if standoff > band.grasp_max else "REACH TO BOX"), AMBER
    if not straddle:
        return ("HANDS TOGETHER" if diag["sep"] > g.sep_max else "HANDS APART"), AMBER
    return "FACE PALMS IN", AMBER


def lock_action(t, c):
    """The one thing that would lock the base, or None. Same causal ordering:
    stand in the right place, face the box, then hold still."""
    if not t:
        return None
    if t["fwd"] > c.fwd_hi:
        return "STEP CLOSER"
    if t["fwd"] < c.fwd_lo:
        return "STEP BACK"
    if abs(t["lat"]) > c.lat_max:
        return "STEP LEFT" if t["lat"] > 0 else "STEP RIGHT"     # lat > 0: box is to the left
    if abs(t["head"]) > c.head_max_deg:
        return "TURN LEFT" if t["head"] > 0 else "TURN RIGHT"
    if t["disp"] > c.disp_max:
        return "STAND STILL"
    return "LOCKING..."                                          # debouncing


def draw_minimal_panel(img, st):
    """The piloting view: at most FOUR lines, large, top-left. Nothing else.

    A live operator has both hands up in front of the camera and cannot read a
    wall of text. Every line answers a question they have in the moment, and the
    numbers behind them live in the FULL view (key o).
    """
    x, y, dy, sc = 16, 46, 46, 1.0
    if not st["tracking"]:
        # Grasp advice is meaningless when the arms are not following the
        # operator, so the fault replaces the grasp line rather than adding one.
        text(img, "TRACKING LOST", (x, y), sc, RED)
    else:
        line, col = grasp_action(st["diag"], st["weld"], st["weld_on"], st["standoff"], st["gcfg"])
        text(img, line, (x, y), sc, col)
    y += dy
    if st["locked"]:
        text(img, "BASE LOCKED", (x, y), sc, GREEN)
    else:
        act = lock_action(st["terms"], st["lcfg"]) if st["standoff"] < NEAR_BOX_M else None
        text(img, "BASE FREE" + ("   " + act if act else ""), (x, y), sc, AMBER)
    y += dy
    lo, hi = st["gcfg"].grasp_min, st["gcfg"].grasp_max
    text(img, "BOX %.2f m" % st["standoff"], (x, y), sc,
         GREEN if lo <= st["standoff"] <= hi else GREY)
    y += dy
    flags = []
    if st["precision"]:
        flags.append("PRECISION")
    if not st["bprime"]:
        flags.append("B-PRIME OFF")
    if flags:
        text(img, "   ".join(flags), (x, y), 0.8, AMBER)


def draw_operator_panel(img, x, y, st):
    """Lock state, lock / release terms, standoff, the three weld conditions
    SEPARATELY, what is blocking, wrist deviation and B-prime - colour coded, so
    the operator can see which condition is holding things up."""
    dy = 24
    t, lc, g, d = st["terms"], st["lcfg"], st["weld"], st["diag"]
    if st["locked"]:
        chips(img, x, y, [("BASE LOCKED", GREEN), ("legs held, policy off  |  u = abort", GREY)], 0.6)
    else:
        items = [("BASE FREE", AMBER)]
        if st["inhibit"]:
            items.append(("re-lock inhibited: walk/turn away", RED))
        chips(img, x, y, items, 0.6)
    y += dy
    if t:
        if not st["locked"]:
            chips(img, x, y, [
                ("lock needs:", GREY),
                ("fwd %.2f in [%.2f,%.2f]" % (t["fwd"], lc.fwd_lo, lc.fwd_hi), GREEN if lc.fwd_lo <= t["fwd"] <= lc.fwd_hi else RED),
                ("lat %+.2f" % t["lat"], GREEN if abs(t["lat"]) <= lc.lat_max else RED),
                ("head %+.1f" % t["head"], GREEN if abs(t["head"]) <= lc.head_max_deg else RED),
                ("still %s" % ("--" if t["disp"] == float("inf") else "%.0fmm" % (1000 * t["disp"])),
                 GREEN if t["disp"] <= lc.disp_max else RED)], 0.45)
        elif t["welded"]:
            chips(img, x, y, [("releases when box lifted:", GREY),
                              ("%.3f >= %.3f m" % (t["box_lift"], lc.lift_min),
                               GREEN if t["box_lift"] >= lc.lift_min else AMBER)], 0.45)
        else:
            chips(img, x, y, [("releases when box placed at goal and hands withdrawn", GREY)], 0.45)
    y += dy
    lo, hi = st["gcfg"].grasp_min, st["gcfg"].grasp_max
    ok = lo <= st["standoff"] <= hi
    chips(img, x, y, [("STANDOFF %.3f m" % st["standoff"], GREEN if ok else RED),
                      ("[%.2f, %.2f] %s" % (lo, hi, "IN BAND" if ok else "OUT"), GREEN if ok else RED)], 0.6)
    y += dy + 6
    gstate = "WELDED" if st["weld_on"] else ("GATED" if st["grasp_cmd"] >= 0.5 else "open")
    chips(img, x, y, [("GRASP cmd %.0f" % st["grasp_cmd"], GREY),
                      (gstate, GREEN if st["weld_on"] else (RED if gstate == "GATED" else GREY))], 0.6)
    y += dy
    blocking = []
    if d:
        rows = [("L palm-box %.3f m <= %.2f" % (d["d_left"], g.palm_radius), d["d_left"] <= g.palm_radius, "L palm-box"),
                ("R palm-box %.3f m <= %.2f" % (d["d_right"], g.palm_radius), d["d_right"] <= g.palm_radius, "R palm-box"),
                ("opposition %+.2f <= %.2f" % (d["opposed"], g.opposed_dot), d["opposed"] <= g.opposed_dot, "opposition"),
                ("separation %.3f m in [%.2f, %.2f]" % (d["sep"], g.sep_min, g.sep_max),
                 g.sep_min <= d["sep"] <= g.sep_max, "separation")]
        for label, good, short in rows:
            m_, col = mark(good)
            chips(img, x + 10, y, [(m_, col), (label, col)], 0.5)
            if not good:
                blocking.append(short)
            y += dy - 2
    if st["weld_on"]:
        text(img, "weld engaged", (x, y), 0.55, GREEN)
    elif blocking:
        text(img, "BLOCKING: " + ", ".join(blocking), (x, y), 0.55, RED)
    else:
        text(img, "GATE OPEN - press g", (x, y), 0.55, GREEN)
    y += dy + 6
    pc = RED if st["pitch"] >= WEDGE_RAD else (AMBER if st["wrist"] >= 0.5 else GREEN)
    text(img, "WRIST dev max %.2f rad (%s)  pitch %.2f%s" % (
        st["wrist"], st["wrist_joint"].replace("_joint", ""), st["pitch"],
        "  WEDGE" if st["pitch"] >= WEDGE_RAD else ""), (x, y), 0.5, pc)
    y += dy
    text(img, "B-PRIME %s  (b)" % ("ON" if st["bprime"] else "OFF"), (x, y), 0.55,
         GREEN if st["bprime"] else AMBER)
    return y + dy


class SimClockGrabber:
    """Headless stand-in for FrameGrabber: one frame per FRAME_STEPS physics
    steps, pulled synchronously, so a synthetic stream plays at 29.4 Hz of
    SIMULATED time however fast the loop runs."""

    def __init__(self, source):
        self.source, self._frame, self._seq = source, None, 0

    def poll(self, counter):
        if counter % FRAME_STEPS == 0:
            f = self.source.grab()
            if f is not None:
                self._frame, self._seq = f, self._seq + 1

    def latest(self):
        return self._frame, self._seq

    def stop(self):
        pass

    def join(self, timeout=None):
        pass


class FrameGrabber(threading.Thread):
    """Pulls ZED frames on a background thread.

    zed.grab() blocks for 35-50 ms. Calling it inside the physics loop meant ten
    physics steps (20 ms of simulated time) cost ~50 ms of wall time, so the
    simulation ran at roughly 40% of real time and every control input felt
    delayed. Grabbing off-thread lets physics run at full rate; the main loop
    simply reads whatever the most recent frame is.
    """

    def __init__(self, zed):
        super().__init__(daemon=True)
        self.zed = zed
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._running = True

    def run(self):
        while self._running:
            frame = self.zed.grab()
            if frame is not None:
                with self._lock:
                    self._frame = frame
                    self._seq += 1

    def latest(self):
        with self._lock:
            return self._frame, self._seq

    def stop(self):
        self._running = False


def main():
    ap = argparse.ArgumentParser(description="ZED | MuJoCo teleoperation")
    ap.add_argument("which", nargs="?", default="pelvis", help="pelvis | lean | keyboard")
    # Episode seed for box placement (O4). Same seed -> same box pose, so a
    # demonstration can be re-run; different seeds -> different poses, which is
    # what Objectives 3 and 4 need and what the keyframe pose never gave.
    ap.add_argument("seed", nargs="?", type=int, default=0)
    ap.add_argument("--synthetic", nargs="?", const="reach", choices=("reach", "grasp", "none"),
                    help="no camera: fabricated BODY_38 stream (test fixture). "
                         "'none' delivers frames with NO body, the cheap path")
    ap.add_argument("--headless", action="store_true",
                    help="no window, no renderer, no wall-clock pacing; needs --synthetic")
    ap.add_argument("--seconds", type=float, default=30.0, help="headless: simulated seconds")
    ap.add_argument("--start-standoff", type=float, default=None,
                    help="start the base this far behind the box, head-on, instead of the keyframe")
    ap.add_argument("--bprime-off", action="store_true", help="start with B-prime OFF (toggle: b)")
    ap.add_argument("--ik-max-iter", type=int, default=None,
                    help="override IKConfig.max_iter (throughput measurement)")
    ap.add_argument("--ik-mj-forward", action="store_true",
                    help="throughput measurement: put mj_forward back inside the IK loop "
                         "(the pre-2026-09-15 solver), for before/after comparison")
    ap.add_argument("--profile", action="store_true",
                    help="throughput measurement: render and draw as usual but no window, "
                         "no pacing; prints the sim-to-wall ratio and where the time went")
    ap.add_argument("--abort-at", type=float, default=None,
                    help="headless test fixture: press 'u' (abort the base lock) at this sim time")
    args = ap.parse_args()
    if args.headless and not args.synthetic:
        ap.error("--headless needs --synthetic (there is no operator to watch a camera)")
    which, episode_seed, headless = args.which, args.seed, args.headless
    profile = args.profile
    if args.ik_mj_forward:
        from g1_teleop import ik as _ik
        _ik._kinematics = lambda m_, d_: mujoco.mj_forward(m_, d_)
        print("IK solver refresh forced to mj_forward (the OLD path)")
    # --profile keeps the renderer and the overlay (they are part of what the
    # operator pays for) but drops the window, the key polling and the pacing,
    # so the loop runs as fast as the machine allows and the ratio is a capacity
    # measurement rather than a measurement of the sleep.
    prof = dict(step=0.0, ctrl=0.0, policy=0.0, render=0.0, overlay=0.0, frames=0, ticks=0)
    if which.startswith("p"):
        loco, loco_name = PelvisVelocity(), "PELVIS VELOCITY (thesis 3.3.4)"
    elif which.startswith("l"):
        loco, loco_name = LeanJoystick(), "LEAN JOYSTICK (mode-switched)"
    else:
        loco, loco_name = KeyboardCommand(), "KEYBOARD (decoupled)"

    cfg_box_for_reset = TeleopConfig().box
    # Contact contract (2026-09-15): the collection entry point must load the
    # same contact model evaluation will; load_model raises on a mismatch.
    from g1_teleop.contact_contract import load_model
    model = load_model(TeleopConfig(), SCENE_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ix = ModelIndex.resolve(model)
    # Weld grasp (D11). `grasp_cmd` is the gripper action dimension: 0 released,
    # 1 engaged. Engagement is gated on the geometric preconditions in
    # g1_teleop/grasp.py, so pressing the key with the hands away from the box
    # does nothing — exactly what the policy will have to learn.
    weld = GraspWeld(model)
    grasp_cmd = 0.0
    data.qpos[ix.leg_qpos] = DEFAULT_ANGLES
    data.qvel[ix.leg_qvel] = 0.0
    # Randomise the box on the STEPPED model, not just the kinematic twin (O4).
    box_xy = reset_box(model, data, ix, cfg_box_for_reset, seed=episode_seed)
    if args.start_standoff is not None:
        data.qpos[ix.base_qpos][0] = float(box_xy[0]) - args.start_standoff
        data.qpos[ix.base_qpos][1] = float(box_xy[1])
    mujoco.mj_forward(model, data)
    # B-prime (adopted 2026-09-15): hands do not collide with the PICKUP slab.
    # Compiled in by scene.xml; toggled in place with 'b'.
    CC.set_hand_pickup_exclusion(model, not args.bprime_off)
    print("B-prime hand<->pickup exclusion: %s" % ("ON" if CC.hand_pickup_exclusion_active(model) else "OFF"))
    # Base lock (D12), driven by the demonstrator's state predicate.
    base_lock = BaseLock(model)
    sp = spec.SpecLayout.resolve(model, ix)
    pred = LockPredicate(PlatformGeometry.resolve(model), LockConfig())
    locked_leg_pos = None          # DEFAULT_ANGLES while locked, else None
    relock_inhibit = False         # set by a manual abort
    lock_terms, n_abort = {}, 0
    wrist_ids = [(int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]),
                  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n), n) for n in WRIST_JOINTS]
    weld_diag = {}
    pel_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    print(f"episode seed {episode_seed}: box at "
          f"({box_xy[0]:.3f}, {box_xy[1]:.3f}, {box_xy[2]:.3f})")
    start_xy = np.array(data.qpos[ix.base_xy_qpos], dtype=np.float64)

    cfg = TeleopConfig()
    if args.ik_max_iter is not None:
        cfg = dataclasses.replace(cfg, ik=dataclasses.replace(cfg.ik, max_iter=args.ik_max_iter))
        print("IK max_iter overridden to %d" % cfg.ik.max_iter)
    twin = G1Robot(cfg)
    # Keep the twin's box on the same seed as the physics model. Nothing reads
    # the twin's box today, but letting the two models hold different box poses
    # is exactly the divergence that produced O4 in the first place.
    twin.reset_box(seed=episode_seed)
    controller = TeleopController(twin, cfg)
    # O25 (2026-09-11): `--synthetic` swaps the camera for a fabricated
    # BODY_38 stream so this path can run with no hardware. It changes NOTHING
    # below the source - the same TeleopController, the same twin, the same
    # copy into ctrl. It is a flag and not a default because a synthetic source
    # is a test fixture, not a collection mode.
    grasp_on_s = None
    frame_hz = 1.0 / (FRAME_STEPS * SIM_DT)
    if args.synthetic == "reach":
        from g1_teleop.synthetic_source import SyntheticSource, hand_path_frames
        print("[O25] SYNTHETIC keypoint source - no camera, reach-out motion")
        zed = SyntheticSource(hand_path_frames(
            lambda t: np.array([0.10 + 0.22 * t, +0.10, -0.10]),
            lambda t: np.array([0.10 + 0.22 * t, -0.10, -0.10]), 300),
            fps=None if (headless or profile) else frame_hz)
    elif args.synthetic == "none":
        from g1_teleop.synthetic_source import SyntheticSource
        print("[profile] SYNTHETIC source with NO body - the cheap path")
        zed = SyntheticSource([[] for _ in range(100000)], fps=None if (headless or profile) else frame_hz)
    elif args.synthetic == "grasp":
        from g1_teleop.synthetic_source import SyntheticSource, grasp_lift_frames
        mujoco.mj_forward(twin.model, twin.data)
        tp = twin.data.xpos[mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")].copy()
        shoulders = {"left": twin.left_shoulder_world() - tp, "right": twin.right_shoulder_world() - tp}
        home = {"left": twin.data.xpos[twin.left_wrist_body] - tp,
                "right": twin.data.xpos[twin.right_wrist_body] - tp}
        lengths = {"left": (twin.upper_arm_left, twin.forearm_left),
                   "right": (twin.upper_arm_right, twin.forearm_right)}
        box_rel = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box1")] - data.xpos[pel_bid]
        frames_g, grasp_on_s, lift_s = grasp_lift_frames(shoulders, lengths, home, box_rel, frame_hz)
        print("[synthetic] grasp-and-lift: %d frames, grasp command at %.1f s, lift at %.1f s, "
              "box in pelvis frame %s" % (len(frames_g), grasp_on_s, lift_s, np.round(box_rel, 3)))
        zed = SyntheticSource(frames_g, fps=None if (headless or profile) else frame_hz)
    else:
        zed = ZEDSource(cfg.zed)
    if (headless or profile) and args.synthetic:
        # Only a SYNTHETIC stream may be pulled on the simulated clock. The real
        # camera must keep its thread: zed.grab() blocks 35-50 ms and calling it
        # inline would put the camera back inside the physics loop (TR10).
        grabber = SimClockGrabber(zed)
    else:
        grabber = FrameGrabber(zed)
        grabber.start()

    # The twin is a separate MjModel; resolve its indices independently rather
    # than assuming the two models number their joints identically.
    twin_ix = ModelIndex.resolve(twin.model)
    twin_upper_qpos = twin_ix.upper_qpos

    policy = torch.jit.load(POLICY_PATH)
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box1")

    renderer = None
    try:
        if not headless or profile:
            renderer = mujoco.Renderer(model, height=MUJOCO_H, width=MUJOCO_W)
    except ValueError as e:
        print("Renderer init failed:", e)
        print("Increase <global offwidth/offheight> in scene.xml, or lower "
              "MUJOCO_W/MUJOCO_H here.")
        zed.close()
        return

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = CAM_LOOKAT
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    scene_option = mujoco.MjvOption()

    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_leg_pos = DEFAULT_ANGLES.copy()
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    cmd = np.zeros(3, dtype=np.float32)
    hold_target = np.array(data.qpos[ix.base_xy_qpos], dtype=np.float64)
    hold_yaw = yaw_from_quat(data.qpos[ix.base_quat_qpos])
    arm_targets = np.array(data.ctrl[ix.upper_ctrl], dtype=np.float32)
    counter = 0
    status_text, status_color = "waiting for ZED", (0, 100, 255)
    loco_diag = {}
    last_disp = None
    last_seq = -1
    frame = None
    wall_start = time.time()

    if not headless and not profile:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, ZED_PANEL_W + MJ_PANEL_W, PANEL_H)
    events = []                        # (sim s, what) - printed at exit
    # Overlay mode. MINIMAL is the default because it is the piloting view; FULL
    # is for debugging and for reports. The choice is remembered for the rest of
    # the session (it is not written to disk, so a restart is MINIMAL again).
    overlay_full = False
    tracking_ok = False
    # O28: sim time of the last controller.step that actually applied a pose.
    # Time, not a frame count, because the other dropout mode is "no new frame
    # arrives at all", which produces no frame to count at all.
    last_applied_s = 0.0
    arm_stale = False
    stale_reported = False
    degraded_ticks = 0

    def abort_lock(source):
        """Manual ABORT of the base lock - the only operator input that touches
        it; there is deliberately no way to force a lock. Logged. Re-locking is
        inhibited until the robot leaves the lock GEOMETRY (see lock_geometry_ok),
        otherwise the predicate would re-lock a still-settled robot a second later."""
        nonlocal relock_inhibit, n_abort
        if not base_lock.locked(data):
            print("%s: base is not locked - nothing to abort" % source)
            return
        pred.locked = False
        pred._run = 0
        relock_inhibit = True
        n_abort += 1
        events.append((counter * SIM_DT, "MANUAL ABORT of base lock (%s) terms %s"
                       % (source, {k: (round(v, 3) if isinstance(v, float) else v)
                                   for k, v in lock_terms.items()})))
        print("[%.2f s] %s" % events[-1])

    print(f"=== Vision locomotion evaluation ===")
    print(f"locomotion command source: {loco_name}")
    print("Click the window. a/d rotate, w/s tilt, +/- zoom, space stop, q quit.\n")

    try:
        while True:
            t_sim = counter * SIM_DT
            if (headless or profile) and args.synthetic:
                grabber.poll(counter)
                if grasp_on_s is not None and grasp_cmd < 0.5 and t_sim >= grasp_on_s:
                    grasp_cmd = 1.0
                    events.append((t_sim, "grasp command -> 1 (synthetic schedule)"))
                if args.abort_at is not None and t_sim >= args.abort_at:
                    abort_lock("--abort-at fixture")
                    args.abort_at = None
            # The run limit is NOT synthetic-only: --profile with the real camera
            # would otherwise never stop (measured, 2026-09-15).
            if (headless or profile) and t_sim >= args.seconds:
                break

            # ── Base lock (D12): the demonstrator's predicate, at 25 Hz on the
            # 47-D state from the STEPPED model (sync=False, as scripted_demo:
            # weld.update below must see what it saw before). A tick index is
            # not an input to the predicate - only the sampling rate is.
            if counter % spec.PHYSICS_STEPS_PER_TICK == 0:
                state = sp.build_state(model, data, ix, sync=False)
                lock_terms = pred.terms(state)
                if relock_inhibit and not pred.locked and not lock_geometry_ok(lock_terms, pred.cfg):
                    relock_inhibit = False
                    events.append((t_sim, "re-lock allowed again (left the lock geometry) fwd %.3f lat %+.3f head %+.1f" % (lock_terms["fwd"], lock_terms["lat"], lock_terms["head"])))
                ev = pred.update(state)
                if ev == "lock" and relock_inhibit:
                    # Operator aborted and has not moved out of the lock
                    # condition: undo the latch. The predicate has no notion of
                    # an abort, so this is the only place one is honoured.
                    pred.locked = False
                    pred.n_lock -= 1
            want_lock = pred.locked
            if want_lock and not base_lock.locked(data):
                base_lock.lock(model, data)
                locked_leg_pos = DEFAULT_ANGLES.copy()
                sto = float(np.linalg.norm(data.xpos[box_body_id][:2] - data.qpos[ix.base_xy_qpos]))
                events.append((t_sim, "BASE LOCKED (predicate) standoff %.3f fwd %.3f lat %+.3f head %+.1f"
                               % (sto, lock_terms.get("fwd", np.nan), lock_terms.get("lat", np.nan),
                                  lock_terms.get("head", np.nan))))
                print("[%.2f s] %s" % events[-1])
            elif not want_lock and base_lock.locked(data):
                base_lock.release(model, data, policy=policy)   # zeroes the LSTM state
                locked_leg_pos = None
                action = np.zeros(NUM_ACTIONS, dtype=np.float32)
                target_leg_pos = DEFAULT_ANGLES.copy()
                hold_target[:] = data.qpos[ix.base_xy_qpos]
                hold_yaw = yaw_from_quat(data.qpos[ix.base_quat_qpos])
                events.append((t_sim, "BASE RELEASED (%s) box_lift %.3f welded %s"
                               % ("manual abort" if relock_inhibit else "predicate",
                                  lock_terms.get("box_lift", np.nan), lock_terms.get("welded"))))
                print("[%.2f s] %s" % events[-1])

            leg_q, leg_dq = data.qpos[ix.leg_qpos], data.qvel[ix.leg_qvel]
            leg_target = target_leg_pos if locked_leg_pos is None else locked_leg_pos
            data.ctrl[ix.leg_ctrl] = ((leg_target - leg_q) * KPS
                                      + (0.0 - leg_dq) * KDS)
            data.ctrl[ix.upper_ctrl] = arm_targets
            # TR17: pads stay RETRACTED. The gripper command is the WELD command;
            # driving the pads with it buries them in a box whose contact the weld
            # disables, and release() then ejects the box (~145 mm, measured).
            data.ctrl[ix.pad_ctrl] = 0.0
            was_on = weld.engaged(data)
            weld_on, weld_diag = weld.update(model, data, grasp_cmd)
            if weld_on != was_on:
                events.append((t_sim, "WELD %s  L %.3f R %.3f opp %+.2f sep %.3f  "
                               "palmL %s palmR %s"
                               % ("ENGAGED" if weld_on else "RELEASED", weld_diag["d_left"],
                                  weld_diag["d_right"], weld_diag["opposed"], weld_diag["sep"],
                                  np.round(data.site_xpos[weld.site_l], 4),
                                  np.round(data.site_xpos[weld.site_r], 4))))
                print("[%.2f s] %s" % events[-1])
            if profile:
                _t = time.perf_counter()
                mujoco.mj_step(model, data)
                prof["step"] += time.perf_counter() - _t
            else:
                mujoco.mj_step(model, data)
            counter += 1

            if counter % CONTROL_DECIMATION:
                continue

            # Pace the simulation to wall time. Without the blocking camera call
            # physics would free-run and the robot would move faster than real
            # time, which is unusable for teleoperation.
            sim_elapsed = counter * SIM_DT
            wall_elapsed = time.time() - wall_start
            if sim_elapsed > wall_elapsed and not headless and not profile:
                time.sleep(sim_elapsed - wall_elapsed)

            # ── Velocity command: hold station when nothing is commanded ────
            # While LOCKED (D12) none of this runs and the policy is not queried:
            # the legs are PD-held at DEFAULT_ANGLES in the physics step above.
            if locked_leg_pos is not None:
                active_cmd = np.zeros(3, dtype=np.float32)
            elif np.linalg.norm(cmd) < IDLE_THRESHOLD:
                active_cmd = np.zeros(3, dtype=np.float32)
                yaw_now = yaw_from_quat(data.qpos[ix.base_quat_qpos])

                # Position hold.
                err_world = hold_target - data.qpos[ix.base_xy_qpos]
                if np.linalg.norm(err_world) >= HOLD_DEADBAND:
                    err_body = world_to_body(err_world, yaw_now)
                    active_cmd[0:2] = np.clip(HOLD_KP * err_body,
                                              -HOLD_MAX, HOLD_MAX)

                # Heading hold, so the facing does not drift while marching.
                yaw_err = (hold_yaw - yaw_now + np.pi) % (2 * np.pi) - np.pi
                if abs(yaw_err) >= HOLD_YAW_DEADBAND:
                    active_cmd[2] = float(np.clip(HOLD_YAW_KP * yaw_err,
                                                  -HOLD_YAW_MAX, HOLD_YAW_MAX))
            else:
                active_cmd = cmd.astype(np.float32)
                hold_target[:] = data.qpos[ix.base_xy_qpos]
                hold_yaw = yaw_from_quat(data.qpos[ix.base_quat_qpos])

            prof["ticks"] += 1
            _t = time.perf_counter() if profile else 0.0
            if locked_leg_pos is None:
                qj = (data.qpos[ix.leg_qpos] - DEFAULT_ANGLES) * DOF_POS_SCALE
                dqj = data.qvel[ix.leg_qvel] * DOF_VEL_SCALE
                t = counter * SIM_DT
                phase = (t % GAIT_PERIOD) / GAIT_PERIOD

                obs[:3] = data.qvel[ix.base_angvel_qvel] * ANG_VEL_SCALE
                obs[3:6] = get_gravity_orientation(data.qpos[ix.base_quat_qpos])
                obs[6:9] = active_cmd * CMD_SCALE
                obs[9:9 + NUM_ACTIONS] = qj
                obs[9 + NUM_ACTIONS:9 + 2 * NUM_ACTIONS] = dqj
                obs[9 + 2 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS] = action
                obs[9 + 3 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS + 2] = [
                    np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
                action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
                target_leg_pos = action * ACTION_SCALE + DEFAULT_ANGLES

            if profile:
                prof["policy"] += time.perf_counter() - _t

            # ── ZED: non-blocking read of the newest frame ──────────────────
            # The grabber thread owns zed.grab(). We only process a frame when
            # its sequence number changes, so the physics loop never waits on
            # the camera.
            new_frame, seq = grabber.latest()
            if new_frame is not None and seq != last_seq:
                last_seq = seq
                frame = new_frame
                if frame.keypoints_3d:
                    prof["frames"] += 1
                    _t = time.perf_counter() if profile else 0.0
                    outcome = controller.step(frame)
                    if profile:
                        prof["ctrl"] += time.perf_counter() - _t
                    tracking_ok = bool(outcome.applied)
                    if outcome.applied:
                        last_applied_s = t_sim
                    status_text = ("IK OK" if outcome.applied
                                   else f"HOLD: {outcome.reason.value}")
                    status_color = (0, 255, 0) if outcome.applied else (0, 100, 255)
                    arm_targets = np.array(twin.data.qpos[twin_upper_qpos],
                                           dtype=np.float32)
                    draw_skeleton(frame.image, frame.keypoints_2d,
                                  frame.confidences, C.REQUIRED_KEYPOINTS)
                    if not isinstance(loco, KeyboardCommand):
                        cmd[:] = loco(frame.keypoints_3d)
                        loco_diag = getattr(loco, "diag", {})
                else:
                    tracking_ok = False
                    status_text, status_color = "no body detected", (0, 100, 255)

            # ── O28: bound the arm hold ────────────────────────────────────
            # `tracking_ok` alone is not enough: it is only assigned inside the
            # "a new frame arrived" block, so if the camera stops delivering
            # frames ENTIRELY it keeps its last value - which may be True - and
            # the stale arms would go unmarked. Elapsed time since the last
            # APPLIED pose covers both dropout modes with one test.
            arm_stale = (t_sim - last_applied_s) > NO_BODY_HOLD_S
            if arm_stale:
                degraded_ticks += 1
                status_text = ("TRACKING LOST %.1fs - ARMS HELD, TICKS DEGRADED"
                               % (t_sim - last_applied_s))
                status_color = (0, 0, 255)
                if not stale_reported:
                    stale_reported = True
                    events.append((t_sim, "TRACKING LOST: arm command stale > "
                                          "%.2f s, ticks marked degraded"
                                          % NO_BODY_HOLD_S))
                    print("[O28] tracking lost at t %.2f s - arms holding, ticks "
                          "from here are marked NOT tracking_ok" % t_sim, flush=True)
            elif stale_reported:
                stale_reported = False
                events.append((t_sim, "tracking recovered"))
                print("[O28] tracking recovered at t %.2f s" % t_sim, flush=True)

            # KeyboardCommand ignores keypoints, so it updates every control
            # tick rather than only when the camera delivers a frame.
            if isinstance(loco, KeyboardCommand):
                cmd[:] = loco()
                loco_diag = getattr(loco, "diag", {})

            if headless and not profile:
                if counter % 500 == 0:
                    wd, wj = wrist_deviation(data, wrist_ids)
                    print("  t %5.1f  base %-6s z %.3f  standoff %.3f  weld %-3s L %.3f R %.3f "
                          "opp %+.2f sep %.3f  box_lift %+.3f  wrist %.3f"
                          % (t_sim, "LOCKED" if base_lock.locked(data) else "free",
                             data.qpos[ix.base_qpos][2],
                             float(np.linalg.norm(data.xpos[box_body_id][:2] - data.qpos[ix.base_xy_qpos])),
                             "ON" if weld_on else "off", weld_diag.get("d_left", np.nan),
                             weld_diag.get("d_right", np.nan), weld_diag.get("opposed", np.nan),
                             weld_diag.get("sep", np.nan), lock_terms.get("box_lift", np.nan), wd))
                continue

            # ── Render and input on their own cadence ───────────────────────
            if counter % RENDER_DECIMATION or frame is None:
                continue

            # ── Left panel: camera ──────────────────────────────────────────
            left = fit_to_box(frame.image, ZED_PANEL_W, PANEL_H)
            if overlay_full:
                text(left, f"Status: {status_text}", (10, 28), 0.6, status_color)
                y = 56
                for k, v in loco_diag.items():
                    if k == "strategy":
                        continue
                    s = f"{k} = {v:+.4f}" if isinstance(v, float) else f"{k} = {v}"
                    text(left, s, (10, y), 0.5, (200, 220, 255))
                    y += 24

            # ── Right panel: simulation ─────────────────────────────────────
            _t = time.perf_counter() if profile else 0.0
            renderer.update_scene(data, camera=cam, scene_option=scene_option)
            right = fit_to_box(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR),
                               MJ_PANEL_W, PANEL_H)
            if profile:
                prof["render"] += time.perf_counter() - _t
            _t = time.perf_counter() if profile else 0.0

            travelled = float(np.linalg.norm(data.qpos[ix.base_xy_qpos] - start_xy))
            wd, wj = wrist_deviation(data, wrist_ids)
            st = dict(
                locked=base_lock.locked(data), inhibit=relock_inhibit, n_lock=pred.n_lock,
                n_release=pred.n_release, n_abort=n_abort, terms=lock_terms, lcfg=pred.cfg,
                standoff=float(np.linalg.norm(data.xpos[box_body_id][:2] - data.qpos[ix.base_xy_qpos])),
                gcfg=cfg.grasp, weld=weld.cfg, diag=weld_diag, weld_on=weld_on, grasp_cmd=grasp_cmd,
                wrist=wd, wrist_joint=wj, pitch=wrist_deviation(data, wrist_ids, pitch_only=True)[0],
                bprime=CC.hand_pickup_exclusion_active(model), tracking=tracking_ok,
                precision=bool(isinstance(loco, KeyboardCommand) and loco.precision))

            # Flag motion that the demonstrator did not ask for.
            moving = last_disp is not None and abs(travelled - last_disp) > 0.004
            commanded = np.linalg.norm(cmd) > IDLE_THRESHOLD
            last_disp = travelled

            if not overlay_full:
                draw_minimal_panel(right, st)
            else:
                text(right, loco_name, (10, 24), 0.55, CYAN)
                text(right, f"TRAVELLED {travelled:.2f} m   LOCO cmd fwd {cmd[0]:+.2f}"
                            f"  turn {cmd[2]:+.2f}", (10, 46), 0.5)
                draw_operator_panel(right, 10, 76, st)
                if st["precision"]:
                    text(right, "PRECISION", (MJ_PANEL_W - 130, 24), 0.55, (255, 200, 0))
                yaw_now_disp = yaw_from_quat(data.qpos[ix.base_quat_qpos])
                hd = np.degrees((hold_yaw - yaw_now_disp + np.pi)
                                % (2 * np.pi) - np.pi)
                text(right, f"HEADING  now {np.degrees(yaw_now_disp):+6.1f}"
                            f"   target {np.degrees(hold_yaw):+6.1f}   err {hd:+5.1f}",
                     (10, PANEL_H - 76), 0.45, GREY)
                if moving and not commanded:
                    text(right, "UNCOMMANDED MOTION", (10, PANEL_H - 52), 0.6, (0, 80, 255))

            # The hints do not earn permanent screen space while piloting: they
            # show for the first HINT_SECONDS and then live in the FULL view.
            if overlay_full or counter * SIM_DT < HINT_SECONDS:
                # Two lines: one does not fit the panel width at a readable scale.
                hints = ["g grasp   u unlock   b B-prime   o overlay (minimal/full)",
                         "a/d rot  w/s tilt  +/- zoom  r reset  space stop  q quit"]
                if isinstance(loco, KeyboardCommand):
                    hints[0] = "arrows walk/turn (hold)   p precision   " + hints[0]
                for k_, h_ in enumerate(hints):
                    text(right, h_, (10, PANEL_H - 30 + 16 * k_), 0.45, (200, 200, 200))

            if profile:
                prof["overlay"] += time.perf_counter() - _t
                continue                      # no window, no keys: this is a measurement
            cv2.imshow(WINDOW, np.hstack([left, right]))

            # DRAIN the key queue rather than taking one event per poll.
            #
            # Holding a key triggers keyboard auto-repeat at roughly 30 events
            # per second. OpenCV queues them. Consuming one per render tick
            # (~25 Hz) means the queue grows while a key is held, so input is
            # processed seconds after it happened and keeps arriving after
            # release. Tapping never builds a backlog, which is why tapping
            # felt responsive and holding did not.
            #
            # Draining to empty each tick keeps latency bounded: we always act
            # on the freshest input and the queue can never accumulate.
            # waitKeyEx also reports extended codes, so arrow keys survive
            # (plain waitKey masks them off).
            drained = []
            while True:
                k = cv2.waitKeyEx(1)
                if k == -1:
                    break
                drained.append(k)
                if len(drained) > 64:      # pathological flood guard
                    break

            # Arrow key codes differ by platform; accept the common ones plus
            # i/j/k/l as a guaranteed fallback.
            ARROW_UP = {2490368, 65362, 82}
            ARROW_DOWN = {2621440, 65364, 84}
            ARROW_LEFT = {2424832, 65361, 81}
            ARROW_RIGHT = {2555904, 65363, 83}

            # Feed EVERY drained event to the walking handler, not just the
            # last one, so pressing forward and turn together registers both.
            if isinstance(loco, KeyboardCommand):
                for ev in drained:
                    low = ev & 0xFF
                    if ev in ARROW_UP or low == ord("i"):
                        loco.on_key(265)
                    elif ev in ARROW_DOWN or low == ord("k"):
                        loco.on_key(264)
                    elif ev in ARROW_LEFT or low == ord("j"):
                        loco.on_key(263)
                    elif ev in ARROW_RIGHT or low == ord("l"):
                        loco.on_key(262)

            # Single-shot keys act on the most recent event only.
            raw = drained[-1] if drained else -1
            key = raw & 0xFF if raw != -1 else 255

            if key == ord("q"):
                break
            elif key == ord("a"):
                cam.azimuth = (cam.azimuth - 5) % 360
            elif key == ord("d"):
                cam.azimuth = (cam.azimuth + 5) % 360
            elif key == ord("w"):
                cam.elevation = min(cam.elevation + 5, 89)
            elif key == ord("s"):
                cam.elevation = max(cam.elevation - 5, -89)
            elif key in (ord("+"), ord("=")):
                cam.distance = max(cam.distance - 0.3, 0.5)
            elif key in (ord("-"), ord("_")):
                cam.distance += 0.3
            elif key == ord("r"):
                cam.lookat[:] = CAM_LOOKAT
                cam.distance, cam.azimuth, cam.elevation = (
                    CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION)
            elif key == ord("g"):
                grasp_cmd = 0.0 if grasp_cmd > 0.5 else 1.0
                print("grasp command -> %.0f%s" % (
                    grasp_cmd, "" if weld_on or grasp_cmd == 0 else
                    "  (gated: hands not in a grasp configuration)"))
            elif key == ord("o"):
                overlay_full = not overlay_full
                print("overlay -> %s" % ("FULL" if overlay_full else "MINIMAL"))
            elif key == ord("u"):
                abort_lock("key u")
            elif key == ord("b"):
                on = not CC.hand_pickup_exclusion_active(model)
                CC.set_hand_pickup_exclusion(model, on)
                events.append((counter * SIM_DT, "B-prime -> %s (key b)" % ("ON" if on else "OFF")))
                print("[%.2f s] %s" % events[-1])
            elif key == ord("p") and isinstance(loco, KeyboardCommand):
                loco.toggle_precision()
            elif key == 32:
                cmd[:] = 0.0
                hold_target[:] = data.qpos[ix.base_xy_qpos]
                hold_yaw = yaw_from_quat(data.qpos[ix.base_quat_qpos])
                if isinstance(loco, KeyboardCommand):
                    loco.cmd[:] = 0.0
                print("stopped and anchored")
    finally:
        grabber.stop()
        grabber.join(timeout=1.0)
        if renderer is not None:
            renderer.close()
        if not headless:
            cv2.destroyAllWindows()
        zed.close()
        if profile:
            wall = time.time() - wall_start
            sim = counter * SIM_DT
            print("\nTHROUGHPUT  sim %.2f s / wall %.2f s = %.1f%% of real time"
                  % (sim, wall, 100.0 * sim / max(wall, 1e-9)))
            other = wall - sum(prof[k] for k in ("step", "ctrl", "policy", "render", "overlay"))
            for k in ("step", "ctrl", "policy", "render", "overlay"):
                per = prof[k] / max(prof["frames"] if k in ("ctrl",) else prof["ticks"], 1)
                print("  %-8s %7.2f s  %5.1f%% of wall   %6.2f ms per %s"
                      % (k, prof[k], 100.0 * prof[k] / max(wall, 1e-9), 1000 * per,
                         "frame" if k == "ctrl" else "tick"))
            print("  %-8s %7.2f s  %5.1f%% of wall" % ("other", other, 100.0 * other / max(wall, 1e-9)))
            print("  camera frames processed %d, control ticks %d" % (prof["frames"], prof["ticks"]))
        print("\nEVENTS (sim s):")
        for t_, what in events:
            print("  %7.2f  %s" % (t_, what))
        print("locks %d, releases %d (predicate), manual aborts %d"
              % (pred.n_lock, pred.n_release, n_abort))
        print("Done.")


if __name__ == "__main__":
    main()
