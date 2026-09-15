"""Per-frame teleoperation controller.

Orchestrates one tracking frame into robot motion: arm target computation, IK,
and low-pass smoothing. The only input guard is a NaN check on the arm
keypoints — the ZED tracking is clean enough that confidence/yaw/segment
gating was removed as it caused more freezes than it prevented.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from .config import TeleopConfig
from .gating import RejectReason
from .ik import solve_arm_ik
from .onceuro import KeypointFilter
from .retargeting import compute_arm_targets
from .robot import G1Robot
from .zed_source import BodyFrame


@dataclass
class FrameOutcome:
    applied: bool
    reason: RejectReason


class TeleopController:
    def __init__(self, robot: G1Robot, cfg: TeleopConfig, debug_gate: bool = False):
        self.robot = robot
        self.cfg = cfg
        self.prev_left = robot.neutral_left.copy()
        self.prev_right = robot.neutral_right.copy()
        self._coast_count = 0        # consecutive dropout frames currently coasted
        self._have_good_pose = False  # becomes True after the first applied frame
        self._depth_state = {}        # per-side previous depth (X_rob) of el/wr targets
        self._still_state = {}        # per-side stillness-lock anchor + locked flag
        sm = cfg.smoothing
        self._kp_filter = KeypointFilter(
            (C.LEFT_SHOULDER, C.LEFT_ELBOW, C.LEFT_WRIST,
             C.RIGHT_SHOULDER, C.RIGHT_ELBOW, C.RIGHT_WRIST),
            freq=sm.euro_freq, min_cutoff=sm.euro_min_cutoff, beta=sm.euro_beta,
        ) if sm.euro_enabled else None

    def reset(self) -> None:
        """Clear every piece of per-episode state this controller carries.

        Enumerated deliberately — none of these raise if forgotten, they just
        leak the previous episode into the next one:
          prev_left/right   last smoothed joint outputs (the arm_alpha filter)
          _coast_count      frames currently coasted through a tracking dropout
          _have_good_pose   whether a good pose was ever applied
          _depth_state      per-side previous depth for the depth_alpha filter
          _still_state      per-side stillness-lock anchors, locked flag, counter
          _shoulder_snapshot per-frame shoulder positions
          _kp_filter        One-Euro history per keypoint
          _counts/_frame_i  logging counters (cosmetic, reset for tidiness)
        """
        self.prev_left = self.robot.neutral_left.copy()
        self.prev_right = self.robot.neutral_right.copy()
        self._coast_count = 0
        self._have_good_pose = False
        self._depth_state = {}
        self._still_state = {}
        self._shoulder_snapshot = {}
        if self._kp_filter is not None:
            self._kp_filter.reset()
        self._counts = {"applied": 0, "frozen": 0}
        self._last_reason = None
        self._frame_i = 0

    def _arms_have_nan(self, kp) -> bool:
        """Only guard we keep: never feed a NaN arm keypoint into IK, since
        that produces a broken pose. Everything else (confidence, yaw, segment
        gating) is removed — the ZED tracking is clean enough not to need it."""
        for idx in (C.LEFT_SHOULDER, C.LEFT_ELBOW, C.LEFT_WRIST,
                    C.RIGHT_SHOULDER, C.RIGHT_ELBOW, C.RIGHT_WRIST):
            if np.any(np.isnan(kp[idx])):
                return True
        return False

    def _solve_arm(self, kp, side: str) -> np.ndarray | None:
        r, w = self.robot, self.cfg.workspace
        if side == "left":
            sh, el, wr = C.LEFT_SHOULDER, C.LEFT_ELBOW, C.LEFT_WRIST
            ua, fa = r.upper_arm_left, r.forearm_left
            shoulder_world = self._shoulder_snapshot["left"]
            el_body, wr_body = r.left_elbow_body, r.left_wrist_body
            qpos, dof, lim = r.ik_left_qpos, r.ik_left_dof, r.ik_left_lim
            neutral = r.neutral_left
            kw = {"elbow_y_min": w.elbow_y_min_left}
        else:
            sh, el, wr = C.RIGHT_SHOULDER, C.RIGHT_ELBOW, C.RIGHT_WRIST
            ua, fa = r.upper_arm_right, r.forearm_right
            shoulder_world = self._shoulder_snapshot["right"]
            el_body, wr_body = r.right_elbow_body, r.right_wrist_body
            qpos, dof, lim = r.ik_right_qpos, r.ik_right_dof, r.ik_right_lim
            neutral = r.neutral_right
            kw = {"elbow_y_max": w.elbow_y_max_right}

        el_target, wr_target = compute_arm_targets(
            np.asarray(kp[sh], float), np.asarray(kp[el], float),
            np.asarray(kp[wr], float), ua, fa, shoulder_world,
            C.DEPTH_SCALE, w, **kw,
        )
        if el_target is None:
            return None

        el_target, wr_target = self._smooth_depth(side, el_target, wr_target)
        el_target, wr_target = self._apply_deadzone(side, el_target, wr_target)

        if getattr(self.cfg.ik, "free_wrists", False):
            # Candidate A: the second task point is the PALM. The six keypoints
            # this reads carry no hand-orientation signal, so the palm target
            # continues the forearm direction by the hand length - roughly the
            # palm a pinned wrist would have given - and the wrist joints are
            # left to the IK and its nullspace.
            sid = r.palm_site_left if side == "left" else r.palm_site_right
            hand = r.hand_len_left if side == "left" else r.hand_len_right
            u = wr_target - el_target
            nu_ = float(np.linalg.norm(u))
            palm_target = wr_target + (u / nu_ if nu_ > 1e-9 else u) * hand
            return solve_arm_ik(r.model, r.data, el_body, wr_body,
                                el_target, palm_target, qpos, dof, lim,
                                neutral, self.cfg.ik, task_site_id=sid)
        return solve_arm_ik(r.model, r.data, el_body, wr_body,
                            el_target, wr_target, qpos, dof, lim,
                            neutral, self.cfg.ik)

    def _apply_deadzone(self, side, el_target, wr_target):
        """Stillness lock (independent per arm).

        Holds a fixed anchor pose while the arm is still (zero recorded jitter),
        and releases only on a clear intentional move. The re-lock requires
        SUSTAINED stillness — several consecutive low-movement frames — rather
        than a single frame under threshold. Without that, ZED tracking noise
        keeps every single frame above the enter threshold, the lock never
        re-engages, and the arm follows the noise forever (the 'automove').
        """
        enter = self.cfg.ik.still_enter
        brk = self.cfg.ik.still_break
        need = self.cfg.ik.still_frames
        st = self._still_state.get(side)

        if st is None:
            self._still_state[side] = {
                "anchor_el": el_target.copy(), "anchor_wr": wr_target.copy(),
                "locked": True, "cand_el": el_target.copy(),
                "cand_wr": wr_target.copy(), "still_count": 0,
            }
            return el_target, wr_target

        if st["locked"]:
            # Distance from the locked anchor. Only a clear move unlocks.
            d = max(np.linalg.norm(el_target - st["anchor_el"]),
                    np.linalg.norm(wr_target - st["anchor_wr"]))
            if d > brk:
                st["locked"] = False
                st["still_count"] = 0
                st["cand_el"] = el_target.copy()
                st["cand_wr"] = wr_target.copy()
                return el_target, wr_target
            return st["anchor_el"], st["anchor_wr"]

        # Unlocked: track the target, but watch for sustained stillness to relock.
        # Movement is measured against a slowly-updated CANDIDATE anchor, not the
        # live target, so noise around a held pose accumulates as "still" frames.
        d_cand = max(np.linalg.norm(el_target - st["cand_el"]),
                     np.linalg.norm(wr_target - st["cand_wr"]))
        if d_cand < enter:
            st["still_count"] += 1
        else:
            st["still_count"] = 0
            st["cand_el"] = el_target.copy()
            st["cand_wr"] = wr_target.copy()

        if st["still_count"] >= need:
            # Sustained stillness -> relock at the candidate pose.
            st["locked"] = True
            st["anchor_el"] = st["cand_el"].copy()
            st["anchor_wr"] = st["cand_wr"].copy()
            return st["anchor_el"], st["anchor_wr"]

        return el_target, wr_target

    def _smooth_depth(self, side, el_target, wr_target):
        """Low-pass only the forward/back axis (X_rob, index 0) of the targets.

        Depth is the noisiest axis of the ZED estimate, and it is exactly the
        reach axis for a front-facing demonstrator. Smoothing it over time
        steadies the reach signal that gets recorded into demonstrations,
        without lagging the well-tracked left/right and up/down axes.
        """
        a = self.cfg.smoothing.depth_alpha
        prev = self._depth_state.get(side)
        el = el_target.copy()
        wr = wr_target.copy()
        if prev is not None:
            el[0] = (1 - a) * prev[0] + a * el[0]
            wr[0] = (1 - a) * prev[1] + a * wr[0]
        self._depth_state[side] = (el[0], wr[0])
        return el, wr

    def step(self, frame: BodyFrame) -> FrameOutcome:
        """Process one tracking frame. Freezes the robot if the frame is gated."""
        outcome = self._step_inner(frame)
        self._log_outcome(outcome)
        return outcome

    def _log_outcome(self, outcome: FrameOutcome) -> None:
        """Print a periodic summary of applied vs frozen frames and the last
        freeze reason, so the terminal always shows why teleop is stalling."""
        if not hasattr(self, "_counts"):
            self._counts = {"applied": 0, "frozen": 0}
            self._last_reason = None
            self._frame_i = 0
        self._frame_i += 1
        self._counts["applied" if outcome.applied else "frozen"] += 1
        if not outcome.applied:
            self._last_reason = outcome.reason.value
        if self._frame_i % 30 == 0:
            a, f = self._counts["applied"], self._counts["frozen"]
            print(f"[teleop] applied={a} frozen={f}  last_freeze={self._last_reason}")

    def _step_inner(self, frame: BodyFrame) -> FrameOutcome:
        dropped = (not frame.keypoints_3d) or self._arms_have_nan(frame.keypoints_3d)

        if dropped:
            return self._coast()

        kp = frame.keypoints_3d
        if self._kp_filter is not None:
            kp = self._kp_filter(kp)
        # Snapshot both shoulders BEFORE any IK runs. Each arm's IK calls
        # mj_forward internally, which would otherwise shift the other arm's
        # shoulder reading mid-frame and make one arm drift when the other moves.
        self._shoulder_snapshot = {
            "left": self.robot.left_shoulder_world(),
            "right": self.robot.right_shoulder_world(),
        }
        raw_left = self._solve_arm(kp, "left")
        raw_right = self._solve_arm(kp, "right")
        if raw_left is None or raw_right is None:
            return self._coast()

        a = self.cfg.smoothing.arm_alpha
        smooth_left = (1 - a) * self.prev_left + a * raw_left
        smooth_right = (1 - a) * self.prev_right + a * raw_right

        self.robot.set_arm_qpos(self.robot.ik_left_qpos, smooth_left)
        self.robot.set_arm_qpos(self.robot.ik_right_qpos, smooth_right)
        self.prev_left = smooth_left.copy()
        self.prev_right = smooth_right.copy()
        self._coast_count = 0
        self._have_good_pose = True

        self.robot.forward()
        return FrameOutcome(True, RejectReason.OK)

    def _coast(self) -> FrameOutcome:
        """Handle a dropped frame. For up to max_coast_frames, re-apply the last
        good pose so a brief tracking blink is invisible in the demonstration.
        Beyond that, hold in place and report the freeze."""
        if not self._have_good_pose:
            return FrameOutcome(False, RejectReason.NAN)
        if self._coast_count < self.cfg.smoothing.max_coast_frames:
            self._coast_count += 1
            self.robot.set_arm_qpos(self.robot.ik_left_qpos, self.prev_left)
            self.robot.set_arm_qpos(self.robot.ik_right_qpos, self.prev_right)
            self.robot.forward()
            return FrameOutcome(True, RejectReason.OK)
        return FrameOutcome(False, RejectReason.NAN)