"""Standalone walking test — pre-trained locomotion policy in YOUR full scene.

Runs the Unitree pre-trained G1 walking policy on the 12 leg joints of your
full 29-DOF robot (with box + platforms), holding the arms/waist at a fixed
pose via their position actuators, under stepped physics.

Two things this script demonstrates:

1. The policy IS the balance controller. It must run every control step, even
   at zero velocity command — the small in-place march is how a biped keeps its
   centre of mass over its support polygon. Freezing the legs at a fixed stance
   removes all balance feedback and the robot falls. (Do not "fix" the stepping
   by bypassing the policy.)

2. Because this robot carries arm/torso mass the 12-DOF training model did not,
   it drifts slowly at zero command (~2 cm/s) and never corrects back. A
   position-hold outer loop closes that gap: measure how far the base has moved
   from where it should be, and feed a small corrective velocity back into the
   policy. The policy already knows how to walk — we just tell it where to go.

Controls (focus the MuJoCo window):
  arrow up/down    : forward speed +/-
  arrow left/right : turn rate +/-
  space            : stop and hold current position
  h                : re-anchor the hold target to the current position
  (close window to quit)

Setup: edit POLICY_PATH and SCENE_PATH below.
Run:   python walk_test.py
"""
import time
import numpy as np
import mujoco
import mujoco.viewer
import torch

from g1_teleop.indices import ModelIndex

# ── Paths (EDIT THESE) ────────────────────────────────────────────────────────
POLICY_PATH = r"D:\Charles_Aninon\Thesis Project\unitree_rl_gym\deploy\pre_train\g1\motion.pt"
SCENE_PATH = r"D:\Charles_Aninon\Thesis Project\scene.xml"

# ── Policy config (from unitree_rl_gym deploy/deploy_mujoco/configs/g1.yaml) ──
SIM_DT = 0.002
CONTROL_DECIMATION = 10          # policy runs every 10 sim steps => 50 Hz
KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
NUM_ACTIONS = 12
NUM_OBS = 47
GAIT_PERIOD = 0.8

# ── Command limits (empirically validated as stable on this robot) ────────────
MAX_FORWARD = 0.80      # m/s
MAX_LATERAL = 0.40      # m/s
MAX_TURN = 0.80         # rad/s

# ── Position hold (station keeping at zero command) ───────────────────────────
# Below IDLE_THRESHOLD the operator is not commanding motion, so instead of
# sending zero (which lets the robot drift) we send a small corrective velocity
# proportional to how far it has strayed from the hold target.
IDLE_THRESHOLD = 0.05   # command magnitude below which hold takes over
HOLD_KP = 0.8           # corrective m/s per metre of position error
HOLD_MAX = 0.25         # cap on corrective speed (keep well under MAX_FORWARD)
HOLD_DEADBAND = 0.02    # ignore errors under 2 cm to avoid twitchy corrections

# ── Layout of YOUR scene's qpos/qvel/ctrl ─────────────────────────────────────
# Resolved by joint/actuator NAME at model load — see g1_teleop/indices.py.
# These offsets used to be literals (legs qpos[7:19], waist+arms ctrl[12:29]
# and so on). They must not be: the palm-pad gripper adds a slide joint inside
# each wrist_yaw_link, which shifts right-arm indices while leaving leg indices
# alone, so a literal slice would stay correct for the legs and silently
# address the wrong joints for the arms.

# Fixed pose for waist+arms while testing walking. Matches the keyframe ctrl
# block so the arms do not jolt at startup (a jolt shoves the torso and can
# topple the walker).
ARM_HOLD_TARGETS = np.array([
    0.0, 0.0, 0.0,                        # waist yaw, roll, pitch
    0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,   # left arm
    0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0,  # right arm
], dtype=np.float32)


def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def yaw_from_quat(quat):
    qw, qx, qy, qz = quat
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def world_to_body(vec_world, yaw):
    """Rotate a world-frame XY vector into the robot's body frame.

    The velocity command the policy consumes is body-relative, but drift is
    measured in world coordinates. Without this rotation the correction would
    push the wrong way once the robot has turned.
    """
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([c * vec_world[0] - s * vec_world[1],
                     s * vec_world[0] + c * vec_world[1]])


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    mujoco.mj_resetDataKeyframe(model, data, 0)

    ix = ModelIndex.resolve(model)

    # The policy expects the legs to START at DEFAULT_ANGLES (a slight knee-bent
    # crouch), not the teleop keyframe's straight-leg pose. Starting elsewhere
    # puts the policy in a state it never trained on and it cannot recover.
    data.qpos[ix.leg_qpos] = DEFAULT_ANGLES
    data.qvel[ix.leg_qvel] = 0.0
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(POLICY_PATH)

    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_leg_pos = DEFAULT_ANGLES.copy()
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)   # operator command
    hold_target = np.array(data.qpos[ix.base_xy_qpos], dtype=np.float64)  # world XY
    counter = 0

    def key_callback(keycode):
        nonlocal cmd
        if keycode == 265:      # up arrow
            cmd[0] += 0.1
        elif keycode == 264:    # down arrow
            cmd[0] -= 0.1
        elif keycode == 263:    # left arrow
            cmd[2] += 0.1
        elif keycode == 262:    # right arrow
            cmd[2] -= 0.1
        elif keycode == 32:     # space — stop and hold here
            cmd[:] = 0.0
            hold_target[:] = data.qpos[ix.base_xy_qpos]
            print(f"holding at x={hold_target[0]:+.3f} y={hold_target[1]:+.3f}")
        elif keycode == ord('H'):   # re-anchor hold target
            hold_target[:] = data.qpos[ix.base_xy_qpos]
            print(f"hold re-anchored at x={hold_target[0]:+.3f} y={hold_target[1]:+.3f}")
        cmd[0] = np.clip(cmd[0], -MAX_FORWARD, MAX_FORWARD)
        cmd[1] = np.clip(cmd[1], -MAX_LATERAL, MAX_LATERAL)
        cmd[2] = np.clip(cmd[2], -MAX_TURN, MAX_TURN)
        print(f"cmd = forward {cmd[0]:.2f}  lateral {cmd[1]:.2f}  turn {cmd[2]:.2f}")

    print("=== Standalone walking test ===")
    print("Focus the MuJoCo window. Arrows: forward/turn. Space: stop+hold. H: re-anchor.")
    print("At zero command the robot marches in place — that is the policy balancing.")
    print("Position hold keeps it from drifting away while it does so.\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # PD torque control on the LEG actuators (motor type).
            leg_q = data.qpos[ix.leg_qpos]
            leg_dq = data.qvel[ix.leg_qvel]
            tau = (target_leg_pos - leg_q) * KPS + (0.0 - leg_dq) * KDS
            data.ctrl[ix.leg_ctrl] = tau

            # Hold waist+arms at a fixed pose (position actuators).
            data.ctrl[ix.upper_ctrl] = ARM_HOLD_TARGETS

            mujoco.mj_step(model, data)
            counter += 1

            if counter % 500 == 0:   # ~1 Hz drift readout
                err = np.linalg.norm(hold_target - data.qpos[ix.base_xy_qpos])
                bx, by, bz = data.qpos[ix.base_qpos][:3]
                print(f"base xyz = {bx:+.3f} {by:+.3f} "
                      f"{bz:.3f}   hold_err = {err:.3f} m")

            if counter % CONTROL_DECIMATION == 0:
                # ── Choose the command the policy actually receives ─────────
                # Idle: substitute a corrective velocity to hold station.
                # Active: pass the operator command straight through, and keep
                # the hold target following the robot so releasing the keys
                # holds wherever it ended up.
                if np.linalg.norm(cmd) < IDLE_THRESHOLD:
                    err_world = hold_target - data.qpos[ix.base_xy_qpos]
                    if np.linalg.norm(err_world) < HOLD_DEADBAND:
                        active_cmd = np.zeros(3, dtype=np.float32)
                    else:
                        yaw = yaw_from_quat(data.qpos[ix.base_quat_qpos])
                        err_body = world_to_body(err_world, yaw)
                        active_cmd = np.zeros(3, dtype=np.float32)
                        active_cmd[0:2] = np.clip(HOLD_KP * err_body,
                                                  -HOLD_MAX, HOLD_MAX)
                else:
                    active_cmd = cmd.astype(np.float32)
                    hold_target[:] = data.qpos[ix.base_xy_qpos]

                qj = (data.qpos[ix.leg_qpos] - DEFAULT_ANGLES) * DOF_POS_SCALE
                dqj = data.qvel[ix.leg_qvel] * DOF_VEL_SCALE
                omega = data.qvel[ix.base_angvel_qvel] * ANG_VEL_SCALE
                grav = get_gravity_orientation(data.qpos[ix.base_quat_qpos])

                t = counter * SIM_DT
                phase = (t % GAIT_PERIOD) / GAIT_PERIOD
                sin_p, cos_p = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = grav
                obs[6:9] = active_cmd * CMD_SCALE
                obs[9:9 + NUM_ACTIONS] = qj
                obs[9 + NUM_ACTIONS:9 + 2 * NUM_ACTIONS] = dqj
                obs[9 + 2 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS] = action
                obs[9 + 3 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS + 2] = [sin_p, cos_p]

                action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
                target_leg_pos = action * ACTION_SCALE + DEFAULT_ANGLES

            viewer.sync()
            dt_left = model.opt.timestep - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)


if __name__ == "__main__":
    main()
