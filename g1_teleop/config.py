"""Central configuration for the G1 vision-based teleoperation pipeline.

All tunable constants live here so experiments don't require touching logic.
Grouped by subsystem: paths, coordinate transform, smoothing, IK, workspace
limits, torso yaw, validity gating, and box spawning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ─── Paths ────────────────────────────────────────────────────────────────────
# scene.xml sits one level above this package, next to run_teleop.py. Resolving
# it relative to this file means the path does not break when the project is
# moved or run from a different working directory.
_PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_PATH: str = str(_PACKAGE_DIR.parent / "scene.xml")


# ─── ZED BODY_38 keypoint indices ─────────────────────────────────────────────
PELVIS: int = 0
LEFT_SHOULDER: int = 12
RIGHT_SHOULDER: int = 13
LEFT_ELBOW: int = 14
RIGHT_ELBOW: int = 15
LEFT_WRIST: int = 16
RIGHT_WRIST: int = 17

REQUIRED_ARM_KEYPOINTS: List[int] = [
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
]
# Pelvis is only needed for locomotion (not yet built). Keeping it out of the
# arm-teleop gate avoids freezing when the lower body is out of frame.
REQUIRED_KEYPOINTS: List[int] = REQUIRED_ARM_KEYPOINTS

SKELETON_PAIRS = [
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (PELVIS, LEFT_SHOULDER),
    (PELVIS, RIGHT_SHOULDER),
]


# ─── Coordinate transform (camera → robot frame) ──────────────────────────────
# ZED camera:  X=right, Y=down, Z=forward-into-scene
# G1 robot:    X=forward, Y=left, Z=up
# Empirically verified: Y_rob=+X_cam, Z_rob=-Y_cam, X_rob=-DEPTH_SCALE*Z_cam.
#
# DEPTH_SCALE controls how much forward/back reach is preserved. It was 0.3
# (crushing forward reach, causing arms to drift sideways and jitter). Noise is
# now handled by the depth_alpha temporal filter, so this can be near 1.0 to
# keep real reach. Lower slightly if forward reach becomes too noisy.
DEPTH_SCALE: float = 0.6


# ─── Joint names ──────────────────────────────────────────────────────────────
# These lists are the single source of truth for model indexing. Nothing may
# hardcode a qpos/qvel/ctrl offset: adding joints anywhere in the body tree
# shifts every index downstream of the insertion, and MuJoCo gives no warning
# when a stale slice silently addresses the wrong joint. Resolve through
# g1_teleop.indices.ModelIndex instead.
FLOATING_BASE_JOINT: str = "floating_base_joint"

# Model order — the pre-trained locomotion policy's 12 actions, its qj/dqj
# observation block, and the KPS/KDS/DEFAULT_ANGLES gain vectors are all in
# THIS order. Do not reorder.
LEG_JOINTS: List[str] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
WAIST_JOINTS: List[str] = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

LEFT_ARM_JOINTS: List[str] = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS: List[str] = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# The 17 position-actuated upper-body joints, in model order. This is the order
# of the 17 joint-target dims of the 22-D action vector, and of walk_test's
# ARM_HOLD_TARGETS.
UPPER_BODY_JOINTS: List[str] = WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

# Palm-pad gripper (decided 2026-08-23 Q1, not yet in g1.xml). Named here so the
# resolver can pick them up the moment they exist; ModelIndex treats them as
# optional and leaves the fields None until then.
LEFT_PAD_JOINT: str = "left_pad_slide_joint"
RIGHT_PAD_JOINT: str = "right_pad_slide_joint"
PAD_JOINTS: List[str] = [LEFT_PAD_JOINT, RIGHT_PAD_JOINT]

# IK drives shoulder(3) + elbow(1); wrists are held at a natural pose.
N_IK_JOINTS: int = 4
WAIST_YAW_JOINT: str = "waist_yaw_joint"

# Seed pose for the IK nullspace: the natural joint configuration for reaching
# forward, per side, for the 4 IK joints [shoulder_pitch, shoulder_roll,
# shoulder_yaw, elbow]. The nullspace pulls the redundant DOF toward this so the
# solver settles on a consistent elbow-out/down pose instead of flipping between
# elbow-in and elbow-out (the folding-across-chest artifact). Tune live:
#   shoulder_pitch < 0 raises the arm forward; elbow > 0 bends it; roll spreads.
IK_SEED_LEFT  = [-0.3,  0.2, 0.0, 0.6]   # left arm: slight outward roll (+)
IK_SEED_RIGHT = [-0.3, -0.2, 0.0, 0.6]   # right arm: mirror roll (-)

WRIST_NATURAL: Dict[str, float] = {
    # Task-smart fixed wrist pose for bimanual grasping: palms angled inward
    # and slightly down so the hands are oriented to press a box in front.
    # Fixed (not IK-driven) to avoid the wrist-twist artifact. Tune these live:
    # roll rotates the palm, pitch angles the hand down, yaw turns it in/out.
    "left_wrist_roll_joint":   0.3,
    "left_wrist_pitch_joint":  0.2,
    "left_wrist_yaw_joint":    0.0,
    "right_wrist_roll_joint":  -0.3,
    "right_wrist_pitch_joint":  0.2,
    "right_wrist_yaw_joint":    0.0,
}

# Body names used to read live link positions.
LEFT_SHOULDER_BODY = "left_shoulder_pitch_link"
RIGHT_SHOULDER_BODY = "right_shoulder_pitch_link"
LEFT_ELBOW_BODY = "left_elbow_link"
RIGHT_ELBOW_BODY = "right_elbow_link"
LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"


@dataclass(frozen=True)
class SmoothingConfig:
    """Low-pass filter factors (thesis Eq 3.8: q = (1-a)*prev + a*new)."""
    arm_alpha: float = 0.6      # was 0.8 — a bit steadier, still responsive
    yaw_alpha: float = 0.3      # yaw is noisier, smooth harder
    depth_alpha: float = 0.5    # extra low-pass on the noisy forward/back axis
    max_coast_frames: int = 5   # hold last good pose through short tracking dropouts
    # One-Euro keypoint filter (Casiez et al. 2012): min_cutoff lower = smoother
    # when still; beta higher = more responsive during motion (less lag).
    euro_enabled: bool = True
    euro_min_cutoff: float = 2.0
    euro_beta: float = 0.08
    euro_freq: float = 30.0     # ZED body tracking runs ~30 Hz


@dataclass(frozen=True)
class IKConfig:
    """Damped least-squares IK solver parameters.

    Tuned for stability over raw responsiveness: higher damping and a stronger
    neutral bias make the solver settle to consistent joint solutions instead
    of churning between equivalent ones (the source of frame-to-frame jitter).
    """
    max_iter: int = 30
    tol: float = 1e-3
    step_size: float = 0.5
    damping: float = 0.12          # more damping = smoother, less churn
    neutral_weight: float = 0.03   # (legacy, unused by nullspace solver)
    nullspace_weight: float = 0.5  # pull toward seed pose in the nullspace (elbow-out)
    # CANDIDATE A (2026-09-15, measurement only, default OFF). D2 pins the
    # wrists because wrist twist made the FRICTION grasp pose unusable; the
    # friction grasp is gone (D11), so that justification may no longer hold.
    # When True the IK drives all 7 arm joints and its second task point is the
    # PALM SITE rather than the wrist body. Driving the wrists with the old
    # task would be a no-op: measured, wrist yaw moves the wrist-body origin by
    # 0.000 m/rad and wrist roll by 0.009, so they would sit in the nullspace;
    # on the palm site they move it 0.106 and 0.152 m/rad. Adopting this is a
    # SPEC_VERSION bump - action dims 6 and 13 leave the constant mask.
    free_wrists: bool = False
    target_deadzone: float = 0.008  # (legacy, unused by stillness lock)
    still_enter: float = 0.020     # per-frame motion below 20mm counts as "still"
    still_break: float = 0.040     # exceed 40mm to unlock (One-Euro lowers source noise)
    still_frames: int = 6          # this many consecutive still frames -> relock


@dataclass(frozen=True)
class WorkspaceConfig:
    """Elbow workspace clamps that keep each arm on its own side (robot frame)."""
    elbow_y_min_left: float = 0.02    # left elbow stays >= 2cm left of center
    elbow_y_max_right: float = -0.02  # right elbow stays >= 2cm right of center
    elbow_x_min: float = -0.05        # elbows can't go >5cm behind shoulder


@dataclass(frozen=True)
class TorsoYawConfig:
    """Maps demonstrator torso yaw onto the G1 waist_yaw joint.

    sign flips turn direction (camera mirrors the demonstrator); if you turn
    left and the robot turns right, set sign = -1.0. scale dampens the motion.
    """
    sign: float = 1.0
    scale: float = 1.0


@dataclass(frozen=True)
class GatingConfig:
    """Tracking-validity gating thresholds.

    A frame is rejected (arms/yaw freeze at last good pose) when tracking is
    unreliable. This prevents self-occlusion and out-of-range poses from
    corrupting demonstrations.

    Thresholds are deliberately loose: gating should catch clearly broken
    frames (NaN, tracking dropout, limbs collapsed/exploded), not normal
    teleoperation. Tighten only if bad demos slip through.
    """
    confidence_min: float = 30.0        # per-keypoint ZED confidence floor
    max_segment_len: float = 1.5        # implausible if a limb segment exceeds this (m)
    min_segment_len: float = 0.03       # implausible if a segment collapses below this (m)
    max_facing_yaw: float = 2.5         # reject only near-full turn-around (rad, ~143deg)
    enable_yaw_gate: bool = False       # off until yaw sign is verified live


@dataclass(frozen=True)
class BoxConfig:
    """Box spawn region and the Objective 4 held-out patch (Q6, closed 2026-09-08).

    `pickup_half` is PER-AXIS. The x and y bounds are not symmetric and never
    were: `max_safe_half` has always returned an array, and a single float
    silently took the min. x is capped near 0.08 by the pelvis/platform standoff
    (O13), while y is free.

        pickup_half[i] + box_half + edge_margin <= platform_half[i]
        x: 0.08 + 0.09 + 0.02 = 0.19    y: 0.21 + 0.09 + 0.02 = 0.32

    Held-out region for Objective 4 is a 2-D INTERIOR PATCH, not a band on one
    axis. Both marginals stay in distribution — every held-out x appears in
    training at some other y, and vice versa — so only the *combination* is
    unseen. That makes it compositional generalization strictly inside the
    convex hull of the training data. A y-only band would risk being solved by
    squaring up to the box, ceilinging all three policies and destroying
    discrimination just as surely as extrapolation would floor them.
    """
    pickup_center: tuple = (1.5, 0.0)      # platform_pickup x, y
    # x trimmed 0.08 -> 0.06 for O22. The walk-in drives the base to
    # box_x - standoff, so the far sample edge must leave the base clear of
    # max_base_x (1.255) with margin, not merely under it. Measured cliff:
    # target base x <= 1.250 converges (14.6 mm position error, heading -0.01
    # deg, palm 35 mm); 1.260 jams (46 mm, +5.5 deg) and the grasp fails. At
    # 1.250 one of three y values still jammed, so 1.240 is the last edge clean
    # at every y tried -> far box edge 1.56 at the nominal 0.32 standoff.
    # Trimming the region keeps the standoff band uniform across demonstrations,
    # which widening the standoff for far spawns would not.
    pickup_half: tuple = (0.06, 0.21)      # per-axis uniform sampling half-range
    edge_margin: float = 0.02              # footprint corner to platform edge
    spawn_z: float = 0.84                  # platform top 0.75 + box half-height 0.09
    body_name: str = "box1"
    box_geom: str = "box1_geom"
    platform_geom: str = "platform_pickup_geom"

    # Objective 4 held-out patch, world XY. Training draws from the sample
    # region MINUS this patch; evaluation draws only from inside it.
    heldout_x: tuple = (1.47, 1.53)
    heldout_y: tuple = (0.04, 0.16)


@dataclass(frozen=True)
class LocomotionConfig:
    """Pre-trained walking policy constants (unitree_rl_gym deploy/configs/g1.yaml).

    These were duplicated verbatim in `walk_test.py` and
    `run_integrated_combined.py`, and episode reset needs them too — a third
    copy is how the twin and the physics model drifted apart in O4. Verified
    byte-identical across both entry points before consolidating here.

    None of these are free parameters. `default_angles` is the crouch the policy
    was trained from (D5, TR5); `gait_period` is baked into the network (TR4);
    `sim_dt` and `control_decimation` set the 500/50 Hz split (D4).
    """
    sim_dt: float = 0.002
    control_decimation: int = 10        # policy at 50 Hz
    num_actions: int = 12
    num_obs: int = 47
    gait_period: float = 0.8            # NOT tunable — see TR4
    action_scale: float = 0.25
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    default_angles: tuple = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                             -0.1, 0.0, 0.0, 0.3, -0.2, 0.0)
    kps: tuple = (100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40)
    kds: tuple = (2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2)
    cmd_scale: tuple = (2.0, 2.0, 0.25)
    # Station keeping at zero command.
    idle_threshold: float = 0.05
    hold_kp: float = 0.8
    hold_max: float = 0.25
    hold_deadband: float = 0.02


@dataclass(frozen=True)
class GraspConfig:
    """Where the base must stop to grasp, and how far forward it may legally go.

    These were module-level constants in `run_integrated_combined.py`, invisible
    to every assertion — the same failure class as O3. They live here so
    `box_reset.assert_reach_fits` can check that the far edge of the spawn
    region is actually reachable.

    `max_base_x` is MEASURED, not derived: the pelvis cannot pass under the
    0.75 m platform top, and the limit is 1.255 essentially regardless of
    platform half-extent (the pelvis meets the slab from beneath once the base
    is inside the footprint). Re-measure if the platform height, the slab
    thickness, or `DEFAULT_ANGLES` changes — none of those are derivable here.
    """
    grasp_min: float = 0.28     # MEASURED (O14), not the old optimistic 0.20
    grasp_max: float = 0.36     # measured usable far end
    max_base_x: float = 1.255   # measured 2026-09-08; see CLAUDE.md O13

    def assert_standoff(self, gap: float, who: str = "caller") -> None:
        """Refuse a standoff outside the measured usable band (O14).

        `grasp_min` was 0.20 on an arm-reach estimate. Measured on rig v5 with a
        45 mm palm guard, the 4-DOF IK (D2) CANNOT put the palms on the box below
        0.28 m: palm error runs 69/63/56/49 mm at 0.20/0.22/0.24/0.26. Those are
        not weak grasps, they are arm poses that never reach the box. A
        demonstrator or evaluation harness allowed to stop there would record
        episodes that could not succeed for reasons nothing downstream can see.
        """
        if not (self.grasp_min <= gap <= self.grasp_max):
            raise AssertionError(
                f"{who} requested standoff {gap:.3f} m, outside the measured "
                f"usable band [{self.grasp_min}, {self.grasp_max}] (O14). Below "
                f"{self.grasp_min} the 4-DOF IK cannot place the palms on the box.")


@dataclass(frozen=True)
class ZEDConfig:
    """ZED camera / body-tracking runtime parameters."""
    resolution: str = "HD720"
    depth_mode: str = "NEURAL"
    confidence_threshold: int = 50
    body_model: str = "HUMAN_BODY_ACCURATE"
    body_fitting: bool = True
    camera_fps: int = 30


@dataclass(frozen=True)
class ContactConfig:
    """Which contacts the physics excludes. See g1_teleop/contact_contract.py.

    `hand_pickup_exclusion` (B-prime, adopted 2026-09-15): the hands pass
    through the PICKUP platform only. The pairs themselves live in scene.xml;
    this flag is what `reset_episode` checks the compiled model against, so
    recording and evaluation cannot silently run different contact models.
    Set False ONLY for A/B measurement, and load with `contact_contract.load_model`.
    """
    hand_pickup_exclusion: bool = True


@dataclass(frozen=True)
class TeleopConfig:
    """Top-level config aggregating all subsystems."""
    model_path: str = MODEL_PATH
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    torso_yaw: TorsoYawConfig = field(default_factory=TorsoYawConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    box: BoxConfig = field(default_factory=BoxConfig)
    grasp: GraspConfig = field(default_factory=GraspConfig)
    loco: LocomotionConfig = field(default_factory=LocomotionConfig)
    zed: ZEDConfig = field(default_factory=ZEDConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)