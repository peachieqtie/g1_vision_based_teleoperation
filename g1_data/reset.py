"""Episode reset for the stepped physics model.

The last piece before the 10/10 grasp gate is runnable, and the foundation of
the data harness: every demonstration and every evaluation rollout starts here.

WHY THIS IS DELICATE
--------------------
Nothing in this file raises when it is wrong. A forgotten piece of state does
not crash — it silently leaks the previous episode into the next one, and the
symptom is a dataset with correlated episodes that nobody notices until the
policies train badly. `counter` is the sharpest example: gait phase is derived
from it, so leaving it set means every episode begins at a different point in
the stride, and the locomotion policy sees a different world on frame 0 of each
demonstration.

So the state is enumerated explicitly below rather than reset by habit, and
every holder that owns per-episode state now has its own `reset()` — reaching
into another module's private attributes from here would break silently the
first time that module changed.

WHAT IS RESET
-------------
Physics (`MjData`), via `mj_resetDataKeyframe`, which clears qpos, qvel, qacc,
**qacc_warmstart** (the solver's warm start — persists across steps and breaks
bit-reproducibility if only qpos/qvel are overwritten by hand), time, ctrl,
qfrc_applied, xfrc_applied and the contact list. `na == 0` and `nmocap == 0` on
this model, so there is no actuator activation or mocap state to clear.

Then, on top of the keyframe:
  legs      -> DEFAULT_ANGLES, NOT the keyframe pose. The walking policy cannot
               recover from straight legs (D5, TR5).
  waist+arms-> keyframe ctrl targets, so the arms do not jolt on frame 0. A jolt
               shoves the torso and can topple the walker.
  pads      -> retracted (qpos 0, ctrl 0).
  box       -> seeded pose from `sample_box_pose`.

Locomotion carryover (`LocomotionCarryover`): counter, action, target_leg_pos,
obs, hold_target, hold_yaw, cmd, wall_start.

Locomotion policy: `hidden_state` and `cell_state`. **The pre-trained policy is
an LSTM** — `motion.pt` mutates those two buffers in place on every forward
pass. This was not previously known or documented, and neither entry point ever
reset them.

Teleop stack, when present: `TeleopController.reset()` (arm smoothing, depth
filter, stillness lock, coast counter, One-Euro history) and the locomotion
input strategy's `reset()`.

WHAT IS DELIBERATELY PERSISTENT
-------------------------------
  MjModel                 immutable scene
  the torch policy WEIGHTS  immutable; its recurrent BUFFERS are not — see above
  ZED camera + grabber    hardware; re-opening per episode would cost seconds
  KeyboardCommand.precision   an operator preference, not episode state. Pass
                          `keep_precision=False` for a fully deterministic
                          reset — the evaluation harness should.
  wall_start              re-stamped, not zeroed: it is a wall-clock reference
                          for sim pacing, so zeroing it would make the loop
                          think it was hours behind and skip all pacing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco

from g1_teleop.box_reset import in_heldout, sample_box_pose
from g1_teleop.config import LocomotionConfig, TeleopConfig


# Buffer names that carry recurrent state on the pre-trained locomotion policy.
# It is an LSTM: `motion.pt` holds `hidden_state` and `cell_state` (1, 1, 64) and
# mutates them IN PLACE on every forward pass. Nothing reset them before, because
# both entry points load the policy fresh per process and run one continuous
# episode -- an episode loop makes it matter on episode 2.
RECURRENT_BUFFERS = ("hidden_state", "cell_state")


def reset_policy_state(policy) -> list:
    """Zero the locomotion policy's recurrent state. Returns what it cleared.

    Without this, episode N begins with the LSTM still remembering episode
    N-1's gait: the first actions differ, the robot starts from a different
    effective controller state, and nothing raises. Verified: zeroing these two
    buffers is what makes repeated runs bit-identical.
    """
    cleared = []
    for name in RECURRENT_BUFFERS:
        buf = getattr(policy, name, None)
        if buf is not None:
            buf.zero_()
            cleared.append(name)
    return cleared


def _yaw_from_quat(quat) -> float:
    qw, qx, qy, qz = quat
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


@dataclass
class LocomotionCarryover:
    """The locomotion loop's state between control ticks.

    These were loose locals in the entry points. Loose locals are exactly what
    gets forgotten at reset, so they get a home with a `reset` that takes all of
    them together.
    """
    counter: int = 0
    action: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    target_leg_pos: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    obs: np.ndarray = field(default_factory=lambda: np.zeros(47, dtype=np.float32))
    hold_target: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    hold_yaw: float = 0.0
    cmd: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    wall_start: float = 0.0

    def reset(self, data, index, loco: LocomotionConfig) -> None:
        default_angles = np.asarray(loco.default_angles, dtype=np.float32)
        # counter drives gait phase: (counter*dt % gait_period)/gait_period.
        # Leaving it set starts every episode at a different point in the stride.
        self.counter = 0
        self.action = np.zeros(loco.num_actions, dtype=np.float32)
        self.target_leg_pos = default_angles.copy()
        self.obs = np.zeros(loco.num_obs, dtype=np.float32)
        # Anchor station-keeping to where the robot actually is AFTER the reset,
        # not to a stale target from the previous episode.
        self.hold_target = np.array(data.qpos[index.base_xy_qpos], dtype=np.float64)
        self.hold_yaw = _yaw_from_quat(data.qpos[index.base_quat_qpos])
        self.cmd = np.zeros(3, dtype=np.float32)
        self.wall_start = time.time()


@dataclass(frozen=True)
class EpisodeStart:
    """What the recorder needs to label an episode."""
    seed: int
    box_pos: np.ndarray
    heldout: bool          # in the Objective 4 held-out patch?

    @property
    def split(self) -> str:
        return "heldout" if self.heldout else "train"


def reset_episode(model, data, index, cfg: TeleopConfig, seed: int,
                  carry: Optional[LocomotionCarryover] = None,
                  controller=None, loco_input=None, twin=None, policy=None,
                  keep_precision: bool = True) -> EpisodeStart:
    """Restore the stepped model to a clean episode start for `seed`.

    Every argument after `seed` is optional so the same function serves the
    grasp gate (physics only, no ZED) and the recorder (full stack). Whatever is
    passed gets reset; whatever is not passed is the caller's responsibility.

    Returns an `EpisodeStart` carrying the held-out flag, computed with the same
    `in_heldout` predicate the end-of-collection leak check uses — the rule is
    defined once so a relabelling can never disagree with the audit.
    """
    box_cfg, loco = cfg.box, cfg.loco

    # 1. Full MjData reset. Do NOT hand-clear fields instead: qacc_warmstart and
    #    the contact list would survive and break bit-reproducibility.
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # 2. Legs to the crouch the policy was trained from (D5/TR5). The keyframe's
    #    straight-leg pose is out of distribution and topples on the first step.
    data.qpos[index.leg_qpos] = np.asarray(loco.default_angles, dtype=np.float64)
    data.qvel[index.leg_qvel] = 0.0

    # 3. Pads retracted. The keyframe already carries 0, but state this here so
    #    it survives a keyframe edit.
    if index.pad_qpos is not None:
        data.qpos[index.pad_qpos] = 0.0
    if index.pad_ctrl is not None:
        data.ctrl[index.pad_ctrl] = 0.0

    # 4. Hold waist+arms at the keyframe targets so frame 0 has zero tracking
    #    error and the arms do not jolt the torso.
    data.ctrl[index.upper_ctrl] = data.qpos[index.upper_qpos]

    # 5. Seeded box pose (O4).
    pos, quat = sample_box_pose(box_cfg, seed)
    data.qpos[index.box_qpos] = np.concatenate([pos, quat])
    data.qvel[index.box_qvel] = 0.0

    mujoco.mj_forward(model, data)

    # 6. Locomotion carryover — after mj_forward, so hold_target/hold_yaw anchor
    #    to the reset pose rather than to whatever the base held before.
    if carry is not None:
        carry.reset(data, index, loco)

    # 7. Teleop smoothing / IK / filter state.
    if controller is not None:
        controller.reset()

    # 8. Locomotion command latches.
    if loco_input is not None and hasattr(loco_input, "reset"):
        try:
            loco_input.reset(keep_precision=keep_precision)
        except TypeError:
            loco_input.reset()

    # 9. Locomotion policy recurrent state. THE POLICY IS AN LSTM -- see
    #    reset_policy_state. Skipping this is invisible: the robot still walks,
    #    it just starts each episode with the previous episode's memory.
    if policy is not None:
        reset_policy_state(policy)

    # 10. Kinematic twin, same seed, so the two models cannot disagree about
    #     where the box is (the O4 failure mode).
    if twin is not None:
        twin.reset_box(seed=seed)

    return EpisodeStart(seed=seed, box_pos=pos.copy(),
                        heldout=in_heldout(pos[:2], box_cfg))


def state_fingerprint(data, index) -> str:
    """SHA-256 of the state that must be identical across equal-seed resets.

    Deliberately includes qacc_warmstart: it is invisible, it persists across
    steps, and it is the field a hand-rolled reset forgets.
    """
    import hashlib
    parts = [np.asarray(data.qpos, dtype=np.float64),
             np.asarray(data.qvel, dtype=np.float64),
             np.asarray(data.ctrl, dtype=np.float64),
             np.asarray(data.qacc_warmstart, dtype=np.float64),
             np.array([data.time], dtype=np.float64)]
    h = hashlib.sha256()
    for p in parts:
        h.update(p.tobytes())
    return h.hexdigest()[:32]
