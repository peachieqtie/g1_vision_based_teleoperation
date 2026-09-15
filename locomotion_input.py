"""Vision-based locomotion command strategies — for empirical evaluation.

Two candidate approaches for deriving the walking velocity command from ZED
body tracking, plus diagnostic logging so their failure modes can be measured
rather than argued about.

STRATEGY A — PelvisVelocity (the literal thesis proposal, section 3.3.4)
    Track the pelvis keypoint, compute translational velocity over a sliding
    window, threshold it, and emit a velocity command when it is exceeded.

    Known concerns to measure:
      - The pelvis is often outside the camera frame in an upper-body setup,
        producing NaN. Logged as pelvis_visible.
      - Section 3.3.8 specifies "walking in space", but a pelvis marching in
        place oscillates about a fixed point, so its NET velocity over a window
        longer than one step is near zero. Logged as vel_mag.
      - Leaning to reach for the box also translates the pelvis forward, which
        is indistinguishable from intent to walk. Logged as would_trigger.

STRATEGY B — LeanJoystick (mode-switched, after CHILD)
    Locomotion is OFF by default and outputs zero velocity. Raising both wrists
    above shoulder height for a dwell period toggles locomotion mode. While
    active, shoulder-midpoint displacement from a neutral captured at activation
    acts as a joystick: lean forward to walk, return to neutral to stop, rotate
    the shoulder line to turn.

    Rationale: CHILD gates leg-as-joystick behind an explicit gripper gesture
    and defaults to zero velocity otherwise. Declaring intent removes the
    lean-to-reach ambiguity that Strategy A cannot resolve.

Both return (vx, vy, wz) in the robot body frame, already clipped to the
validated stable envelope.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

# ZED BODY_38 keypoint indices
PELVIS = 0
LEFT_SHOULDER = 12
RIGHT_SHOULDER = 13
LEFT_WRIST = 16
RIGHT_WRIST = 17

# Validated stable command envelope (measured on this robot)
MAX_FORWARD = 0.80
MAX_LATERAL = 0.40
MAX_TURN = 0.80


def _clip_cmd(vx, vy, wz):
    return np.array([
        np.clip(vx, -MAX_FORWARD, MAX_FORWARD),
        np.clip(vy, -MAX_LATERAL, MAX_LATERAL),
        np.clip(wz, -MAX_TURN, MAX_TURN),
    ], dtype=np.float32)


class PelvisVelocity:
    """Strategy A — the literal proposal. Pelvis translational velocity."""

    def __init__(self, window_sec=0.3, speed_threshold=0.15,
                 min_duration_sec=0.2, gain=2.0):
        self.window_sec = window_sec
        self.speed_threshold = speed_threshold   # m/s to count as walking
        self.min_duration_sec = min_duration_sec  # sustain before triggering
        self.gain = gain                          # human m/s -> robot m/s
        self._hist = deque()                      # (timestamp, pelvis_xyz)
        self._above_since = None
        self.diag = {}

    def reset(self):
        """Clear the velocity window and trigger latch between episodes."""
        self._hist.clear()
        self._above_since = None
        self.diag = {}

    def __call__(self, keypoints):
        now = time.time()
        pelvis = np.asarray(keypoints[PELVIS], dtype=float)
        visible = not np.any(np.isnan(pelvis))

        self.diag = {"strategy": "pelvis_velocity", "pelvis_visible": visible,
                     "vel_mag": 0.0, "sustained": 0.0, "would_trigger": False}

        if not visible:
            self._hist.clear()
            self._above_since = None
            return _clip_cmd(0, 0, 0)

        self._hist.append((now, pelvis))
        while self._hist and now - self._hist[0][0] > self.window_sec:
            self._hist.popleft()

        if len(self._hist) < 2:
            return _clip_cmd(0, 0, 0)

        t0, p0 = self._hist[0]
        dt = now - t0
        if dt < 1e-3:
            return _clip_cmd(0, 0, 0)

        # NET displacement over the window. Marching in place oscillates about
        # a fixed point, so this tends toward zero however vigorous the motion.
        vel = (pelvis - p0) / dt
        # Camera frame: x=right, y=down, z=forward. Ignore vertical (y).
        speed = float(np.linalg.norm([vel[0], vel[2]]))
        self.diag["vel_mag"] = speed

        if speed > self.speed_threshold:
            if self._above_since is None:
                self._above_since = now
            sustained = now - self._above_since
        else:
            self._above_since = None
            sustained = 0.0
        self.diag["sustained"] = sustained

        if self._above_since is None or sustained < self.min_duration_sec:
            return _clip_cmd(0, 0, 0)

        self.diag["would_trigger"] = True
        # Camera -> body: approaching the camera (-z) is forward; +x is right.
        forward = -vel[2] * self.gain
        lateral = -vel[0] * self.gain
        return _clip_cmd(forward, lateral, 0.0)


class LeanJoystick:
    """Strategy B — mode-switched lean joystick. Zero velocity unless active.

    MAPPING
        lean forward/back  -> forward velocity
        lean left/right    -> TURN rate
        (lateral strafe is not exposed)

    Sideways lean drives turning rather than strafing for two reasons. The box
    task needs turning — walk to a platform, turn, walk to the other — and
    almost never needs lateral translation. And the obvious alternative for
    turn, rotating the shoulder line, requires twisting the torso away from the
    camera, which is exactly what the arm-tracking pipeline assumes you will
    not do.
    """

    def __init__(self, dwell_sec=0.8, lean_deadband=0.04, lean_gain=4.0,
                 turn_gain=3.0):
        self.dwell_sec = dwell_sec            # hold gesture this long to toggle
        self.lean_deadband = lean_deadband    # metres of lean ignored
        self.lean_gain = lean_gain            # metres of forward lean -> m/s
        self.turn_gain = turn_gain            # metres of side lean -> rad/s
        self.active = False
        self._gesture_since = None
        self._toggle_latch = False
        self._neutral = None                  # shoulder midpoint at activation
        self.diag = {}

    def reset(self):
        """Deactivate and clear the gesture latch and neutral pose."""
        self.active = False
        self._gesture_since = None
        self._toggle_latch = False
        self._neutral = None
        self.diag = {}

    @staticmethod
    def _shoulder_mid(kp):
        return 0.5 * (np.asarray(kp[LEFT_SHOULDER], float)
                      + np.asarray(kp[RIGHT_SHOULDER], float))

    def _gesture_held(self, kp):
        """Both wrists above shoulder height. In camera frame y points DOWN, so
        'above' means a smaller y value. This pose does not occur naturally
        while reaching for, carrying, or placing the box."""
        ls, rs = np.asarray(kp[LEFT_SHOULDER], float), np.asarray(kp[RIGHT_SHOULDER], float)
        lw, rw = np.asarray(kp[LEFT_WRIST], float), np.asarray(kp[RIGHT_WRIST], float)
        if np.any(np.isnan([ls, rs, lw, rw])):
            return False
        return (lw[1] < ls[1]) and (rw[1] < rs[1])

    def __call__(self, keypoints):
        now = time.time()
        required = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST]
        if any(np.any(np.isnan(np.asarray(keypoints[i], float))) for i in required):
            self.diag = {"strategy": "lean_joystick", "active": self.active,
                         "tracking": False, "lean_fwd": 0.0, "lean_side": 0.0}
            return _clip_cmd(0, 0, 0)

        # ── Mode toggle on sustained gesture ──────────────────────────────
        if self._gesture_held(keypoints):
            if self._gesture_since is None:
                self._gesture_since = now
            elif (now - self._gesture_since > self.dwell_sec
                  and not self._toggle_latch):
                self.active = not self.active
                self._toggle_latch = True
                if self.active:
                    self._neutral = self._shoulder_mid(keypoints)
                    print("[loco] locomotion mode ON — neutral captured")
                else:
                    print("[loco] locomotion mode OFF")
        else:
            self._gesture_since = None
            self._toggle_latch = False

        self.diag = {"strategy": "lean_joystick", "active": self.active,
                     "tracking": True, "lean_fwd": 0.0, "lean_side": 0.0}

        if not self.active or self._neutral is None:
            return _clip_cmd(0, 0, 0)

        # ── Lean as joystick displacement from the captured neutral ───────
        mid = self._shoulder_mid(keypoints)
        delta = mid - self._neutral
        # Camera frame: -z is toward the camera (forward), +x is right.
        lean_fwd = -delta[2]
        lean_side = -delta[0]
        self.diag["lean_fwd"] = float(lean_fwd)
        self.diag["lean_side"] = float(lean_side)

        def deadband(v):
            return 0.0 if abs(v) < self.lean_deadband else v - np.sign(v) * self.lean_deadband

        forward = deadband(lean_fwd) * self.lean_gain
        turn = deadband(lean_side) * self.turn_gain
        return _clip_cmd(forward, 0.0, turn)


class KeyboardCommand:
    """Fallback — operator-commanded velocity, matching the decoupled control
    used by Mobile-TeleVision (pedals), CHIP/SONIC (VR joysticks) and others.

    HOLD TO MOVE, RELEASE TO STOP
    An accumulating command (press eight times to reach full speed, eight more
    to stop) makes precise approach almost impossible, because stopping is a
    separate deliberate act from not-moving. Instead each keypress refreshes a
    velocity that decays to zero shortly after you stop pressing. Hold a key to
    walk, release it to halt, tap it to inch forward.

    OpenCV reports key-down but never key-up, so "release" is inferred from the
    absence of a repeat within release_sec. Key auto-repeat makes holding a key
    produce a steady stream of events, which reads as continuous motion.

    PRECISION MODE
    The final approach to the platform needs finer control than crossing the
    room. Press 'p' to halve the speeds for accurate positioning.

    TURNING RADIUS
    Radius is forward_speed / turn_speed, so a gentler turn_speed at the same
    forward_speed sweeps a wider arc. At 0.35 m/s and 0.25 rad/s the radius is
    about 1.4 m, which is controllable; the previous 0.5 rad/s gave 1.1 m and
    felt sharp. Raising forward_speed also widens the radius.
    """

    def __init__(self, forward_speed=0.35, turn_speed=0.25, release_sec=0.25,
                 precision_scale=0.5):
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        self.release_sec = release_sec
        self.precision_scale = precision_scale
        self.precision = False
        self.cmd = np.zeros(3, dtype=np.float32)
        self._pressed = {}          # direction -> timestamp of last press
        self.diag = {"strategy": "keyboard", "precision": False}

    def reset(self, keep_precision: bool = True):
        """Clear the command and key latches between episodes.

        `precision` is an operator preference, not episode state, so it
        persists by default — pass keep_precision=False for a fully
        deterministic reset (the evaluation harness should).
        """
        self.cmd[:] = 0.0
        self._pressed.clear()
        if not keep_precision:
            self.precision = False
        self.diag = {"strategy": "keyboard", "precision": self.precision}

    def toggle_precision(self):
        self.precision = not self.precision
        print(f"[kbd] precision mode {'ON' if self.precision else 'OFF'}")

    def on_key(self, keycode):
        now = time.time()
        if keycode == 265:      # up
            self._pressed["fwd"] = now
        elif keycode == 264:    # down
            self._pressed["back"] = now
        elif keycode == 263:    # left
            self._pressed["left"] = now
        elif keycode == 262:    # right
            self._pressed["right"] = now
        elif keycode == 32:     # space
            self._pressed.clear()
            self.cmd[:] = 0.0

    def __call__(self, keypoints=None):
        now = time.time()
        scale = self.precision_scale if self.precision else 1.0
        held = {d for d, t in self._pressed.items()
                if now - t < self.release_sec}

        fwd = 0.0
        if "fwd" in held:
            fwd += self.forward_speed * scale
        if "back" in held:
            fwd -= self.forward_speed * scale

        turn = 0.0
        if "left" in held:
            turn += self.turn_speed * scale
        if "right" in held:
            turn -= self.turn_speed * scale

        self.cmd[:] = _clip_cmd(fwd, 0.0, turn)
        self.diag = {"strategy": "keyboard", "precision": self.precision,
                     "held": ",".join(sorted(held)) if held else "-"}
        return self.cmd.copy()