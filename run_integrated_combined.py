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
"""
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
from locomotion_input import PelvisVelocity, LeanJoystick, KeyboardCommand

# ── Paths (EDIT THESE) ────────────────────────────────────────────────────────
POLICY_PATH = r"D:\Charles_Aninon\Thesis Project\unitree_rl_gym\deploy\pre_train\g1\motion.pt"
SCENE_PATH = r"D:\Charles_Aninon\Thesis Project\scene.xml"

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



# Display refresh rate. Rendering and key polling are decoupled from both the
# physics rate and the camera rate so neither can stall the other.
# Must be a multiple of CONTROL_DECIMATION, because the render check is only
# reached on control ticks. 20 steps x 0.002 s = 40 ms => 25 Hz display.
RENDER_DECIMATION = 20


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
    which = sys.argv[1] if len(sys.argv) > 1 else "pelvis"
    # Episode seed for box placement (O4). Same seed -> same box pose, so a
    # demonstration can be re-run; different seeds -> different poses, which is
    # what Objectives 3 and 4 need and what the keyframe pose never gave.
    episode_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
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
    mujoco.mj_forward(model, data)
    print(f"episode seed {episode_seed}: box at "
          f"({box_xy[0]:.3f}, {box_xy[1]:.3f}, {box_xy[2]:.3f})")
    start_xy = np.array(data.qpos[ix.base_xy_qpos], dtype=np.float64)

    cfg = TeleopConfig()
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
    if "--synthetic" in sys.argv:
        from g1_teleop.synthetic_source import SyntheticSource, hand_path_frames
        print("[O25] SYNTHETIC keypoint source - no camera, reach-out motion")
        zed = SyntheticSource(hand_path_frames(
            lambda t: np.array([0.10 + 0.22 * t, +0.10, -0.10]),
            lambda t: np.array([0.10 + 0.22 * t, -0.10, -0.10]), 300))
    else:
        zed = ZEDSource(cfg.zed)
    grabber = FrameGrabber(zed)
    grabber.start()

    # The twin is a separate MjModel; resolve its indices independently rather
    # than assuming the two models number their joints identically.
    twin_ix = ModelIndex.resolve(twin.model)
    twin_upper_qpos = twin_ix.upper_qpos

    policy = torch.jit.load(POLICY_PATH)
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box1")

    try:
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

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, ZED_PANEL_W + MJ_PANEL_W, PANEL_H)

    print(f"=== Vision locomotion evaluation ===")
    print(f"locomotion command source: {loco_name}")
    print("Click the window. a/d rotate, w/s tilt, +/- zoom, space stop, q quit.\n")

    try:
        while True:
            leg_q, leg_dq = data.qpos[ix.leg_qpos], data.qvel[ix.leg_qvel]
            data.ctrl[ix.leg_ctrl] = ((target_leg_pos - leg_q) * KPS
                                      + (0.0 - leg_dq) * KDS)
            data.ctrl[ix.upper_ctrl] = arm_targets
            data.ctrl[ix.pad_ctrl] = grasp_cmd
            weld_on, weld_diag = weld.update(model, data, grasp_cmd)
            mujoco.mj_step(model, data)
            counter += 1

            if counter % CONTROL_DECIMATION:
                continue

            # Pace the simulation to wall time. Without the blocking camera call
            # physics would free-run and the robot would move faster than real
            # time, which is unusable for teleoperation.
            sim_elapsed = counter * SIM_DT
            wall_elapsed = time.time() - wall_start
            if sim_elapsed > wall_elapsed:
                time.sleep(sim_elapsed - wall_elapsed)

            # ── Velocity command: hold station when nothing is commanded ────
            if np.linalg.norm(cmd) < IDLE_THRESHOLD:
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

            # ── ZED: non-blocking read of the newest frame ──────────────────
            # The grabber thread owns zed.grab(). We only process a frame when
            # its sequence number changes, so the physics loop never waits on
            # the camera.
            new_frame, seq = grabber.latest()
            if new_frame is not None and seq != last_seq:
                last_seq = seq
                frame = new_frame
                if frame.keypoints_3d:
                    outcome = controller.step(frame)
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
                    status_text, status_color = "no body detected", (0, 100, 255)

            # KeyboardCommand ignores keypoints, so it updates every control
            # tick rather than only when the camera delivers a frame.
            if isinstance(loco, KeyboardCommand):
                cmd[:] = loco()
                loco_diag = getattr(loco, "diag", {})

            # ── Render and input on their own cadence ───────────────────────
            if counter % RENDER_DECIMATION or frame is None:
                continue

            # ── Left panel: camera ──────────────────────────────────────────
            left = fit_to_box(frame.image, ZED_PANEL_W, PANEL_H)
            text(left, f"Status: {status_text}", (10, 28), 0.6, status_color)
            y = 56
            for k, v in loco_diag.items():
                if k == "strategy":
                    continue
                s = f"{k} = {v:+.4f}" if isinstance(v, float) else f"{k} = {v}"
                text(left, s, (10, y), 0.5, (200, 220, 255))
                y += 24

            # ── Right panel: simulation ─────────────────────────────────────
            renderer.update_scene(data, camera=cam, scene_option=scene_option)
            right = fit_to_box(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR),
                               MJ_PANEL_W, PANEL_H)

            travelled = float(np.linalg.norm(data.qpos[ix.base_xy_qpos] - start_xy))
            text(right, loco_name, (10, 28), 0.6, (255, 255, 0))
            text(right, f"TRAVELLED  {travelled:.2f} m", (10, 56), 0.7,
                 (0, 255, 255))
            text(right, f"LOCO cmd  fwd {cmd[0]:+.2f}  turn {cmd[2]:+.2f}",
                 (10, 84), 0.55)

            # ── Approach feedback ───────────────────────────────────────────
            # Distance from the robot's shoulders to the box, so the approach
            # can be judged from a number instead of timed by feel. The arm can
            # only reach about ARM_REACH, so this is the difference between a
            # graspable stop and one that is 10 cm short.
            box_xy = data.xpos[box_body_id][:2]
            base_xy = data.qpos[ix.base_xy_qpos]
            gap = float(np.linalg.norm(box_xy - base_xy))
            in_range = GRASP_MIN <= gap <= GRASP_MAX
            colour = (0, 255, 0) if in_range else (0, 200, 255)
            text(right, f"BOX DISTANCE  {gap:.2f} m", (10, 118), 0.7, colour)
            gcol = (0, 255, 0) if weld_on else ((0, 200, 255) if grasp_cmd > 0.5 else (200, 200, 200))
            text(right, f"GRASP  cmd {grasp_cmd:.0f}  " +
                 ("WELDED" if weld_on else ("gated" if grasp_cmd > 0.5 else "open")),
                 (10, 246), 0.6, gcol)
            if in_range:
                text(right, "IN GRASP RANGE", (10, 150), 0.7, (0, 255, 0))
            elif gap > GRASP_MAX:
                text(right, f"too far  (walk {gap - GRASP_MAX:.2f} m closer)",
                     (10, 150), 0.55, (0, 200, 255))
            else:
                text(right, "too close  (back up)", (10, 150), 0.55, (0, 200, 255))

            if isinstance(loco, KeyboardCommand) and loco.precision:
                text(right, "PRECISION", (10, 182), 0.6, (255, 200, 0))
            yaw_now_disp = yaw_from_quat(data.qpos[ix.base_quat_qpos])
            hd = np.degrees((hold_yaw - yaw_now_disp + np.pi)
                            % (2 * np.pi) - np.pi)
            text(right, f"HEADING  now {np.degrees(yaw_now_disp):+6.1f}"
                        f"   target {np.degrees(hold_yaw):+6.1f}   err {hd:+5.1f}",
                 (10, 210), 0.5, (255, 255, 255))

            # Flag motion that the demonstrator did not ask for.
            moving = last_disp is not None and abs(travelled - last_disp) > 0.004
            commanded = np.linalg.norm(cmd) > IDLE_THRESHOLD
            if moving and not commanded:
                text(right, "UNCOMMANDED MOTION", (10, 242), 0.7, (0, 80, 255))
            last_disp = travelled

            hint = "a/d rot  w/s tilt  +/- zoom  r reset  space stop  q quit"
            if isinstance(loco, KeyboardCommand):
                hint = ("arrows = walk/turn (hold)   p = precision   |   " + hint)
            text(right, hint, (10, PANEL_H - 14), 0.45, (200, 200, 200))

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
        renderer.close()
        cv2.destroyAllWindows()
        zed.close()
        print("Done.")


if __name__ == "__main__":
    main()
