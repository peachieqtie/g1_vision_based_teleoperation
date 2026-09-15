"""Synthetic BODY_38 keypoint source - a drop-in for `ZEDSource`, no hardware.

WHY THIS EXISTS
---------------
O25: neither teleop entry point has ever run under stepped physics, so the
retargeting -> IK -> arm path is proven to produce POSES and never to produce
MOTION A STANDING ROBOT CAN EXECUTE. The ZED is not available. It does not need
to be: the pipeline consumes BODY_38 keypoints and does not care where they came
from, so a source that fabricates them exercises every stage below the camera.

WHAT IT DOES AND DOES NOT REPLACE
---------------------------------
It replaces the camera and the body tracker. It does NOT replace their error
model. Real ZED noise, dropout statistics, body-fitting bias, and true human
motion statistics are all absent, and O25 stays open until a camera is on the
desk. What this closes is the part that needs no camera: whether the poses this
pipeline produces are executable under gravity by a robot that is standing up.

THE INVERSE TRANSFORM
---------------------
`transforms.apply_camera_rotation(v_cam, s) = (-s*v_cam.z, v_cam.x, -v_cam.y)`,
so the inverse is `(v_rob.y, -v_rob.z, -v_rob.x / s)`. Note the 1/s on the depth
axis: `DEPTH_SCALE` is not a rotation, it is a deliberate gain on the noisiest
ZED axis (D1), and inverting it means a synthetic "human" has to reach 1/0.6 =
1.67x further in depth than the robot does. That asymmetry is real and is worth
seeing in the numbers rather than hidden.

Only DIRECTIONS survive the pipeline: `retargeting.compute_arm_targets` takes
the shoulder->elbow and elbow->wrist unit vectors and rescales them by the
ROBOT's own limb lengths. So absolute keypoint positions and human limb lengths
are free parameters; they are set to plausible values here purely so the numbers
look like a person.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from . import config as C
from .zed_source import BodyFrame

#: BODY_38 has 38 keypoints; only the six arm indices are ever read (D2/O25).
N_KEYPOINTS = 38
#: Plausible human segment lengths, metres. Free parameters - see module
#: docstring - kept realistic so the fabricated skeleton reads as a person.
HUMAN_UPPER_ARM = 0.30
HUMAN_FOREARM = 0.26
#: Where the synthetic person stands in the camera frame. Also free.
CAM_ORIGIN = np.array([0.0, 0.0, 2.0])
SHOULDER_HALF_WIDTH = 0.20


def inv_camera_rotation(v_rob: np.ndarray, depth_scale: float) -> np.ndarray:
    """Camera-frame vector that `apply_camera_rotation` maps to `v_rob`."""
    v = np.asarray(v_rob, dtype=float)
    return np.array([v[1], -v[2], -v[0] / depth_scale])


def _nan_keypoints() -> List[np.ndarray]:
    return [np.full(3, np.nan) for _ in range(N_KEYPOINTS)]


def keypoints_from_arm_dirs(u_upper_l, u_fore_l, u_upper_r, u_fore_r,
                            depth_scale: float = C.DEPTH_SCALE
                            ) -> List[np.ndarray]:
    """Build a BODY_38 keypoint list from four robot-frame unit directions.

    Every keypoint the pipeline does not read stays NaN, so a future change that
    starts reading one fails loudly instead of consuming a fabricated zero.
    """
    kp = _nan_keypoints()
    for side, sgn, u_up, u_fo in (("left", +1.0, u_upper_l, u_fore_l),
                                  ("right", -1.0, u_upper_r, u_fore_r)):
        sh_idx = C.LEFT_SHOULDER if side == "left" else C.RIGHT_SHOULDER
        el_idx = C.LEFT_ELBOW if side == "left" else C.RIGHT_ELBOW
        wr_idx = C.LEFT_WRIST if side == "left" else C.RIGHT_WRIST
        # The robot's +y is the human's left, so a left shoulder sits at +y_rob.
        sh_cam = CAM_ORIGIN + inv_camera_rotation(
            np.array([0.0, sgn * SHOULDER_HALF_WIDTH, 0.0]), depth_scale)
        el_cam = sh_cam + HUMAN_UPPER_ARM * inv_camera_rotation(u_up, depth_scale)
        wr_cam = el_cam + HUMAN_FOREARM * inv_camera_rotation(u_fo, depth_scale)
        kp[sh_idx], kp[el_idx], kp[wr_idx] = sh_cam, el_cam, wr_cam
    return kp


def arm_dirs_from_hand(hand_rel_shoulder: np.ndarray, swivel: np.ndarray,
                       upper: float = HUMAN_UPPER_ARM,
                       fore: float = HUMAN_FOREARM):
    """(u_upper, u_fore) for a hand at `hand_rel_shoulder`, elbow on `swivel`.

    Two-link placement, the same geometry the scripted demonstrator uses to pick
    an elbow. If the hand is out of reach the arm is straightened toward it
    rather than the solve failing - which is what a person's arm does too.
    """
    v = np.asarray(hand_rel_shoulder, dtype=float)
    d = float(np.linalg.norm(v))
    span = upper + fore
    if d > span * 0.995:
        v = v * (span * 0.995 / d)
        d = span * 0.995
    if d < 1e-6:
        v = np.array([0.0, 0.0, -1e-3])
        d = 1e-3
    u = v / d
    a = np.arccos(np.clip((upper ** 2 + d ** 2 - fore ** 2) / (2 * upper * d),
                          -1.0, 1.0))
    p = np.asarray(swivel, dtype=float) - np.dot(swivel, u) * u
    n = float(np.linalg.norm(p))
    p = p / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])
    elbow = upper * (np.cos(a) * u + np.sin(a) * p)
    fo = v - elbow
    nf = float(np.linalg.norm(fo))
    return elbow / upper, (fo / nf if nf > 1e-9 else u)


class SyntheticSource:
    """Drop-in for `ZEDSource`: `.grab()` returns a `BodyFrame` or None.

    `frames` is any sequence of BODY_38 keypoint lists; a list containing NaN
    arm keypoints exercises the controller's coast path exactly as a real
    tracking dropout does. `grab()` returns None past the end, which is how
    `ZEDSource` reports a failed grab.
    """

    def __init__(self, frames: Sequence[List[np.ndarray]], image_hw=(8, 8)):
        self.frames = list(frames)
        self.i = 0
        self._image = np.zeros((image_hw[0], image_hw[1], 3), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.frames)

    def grab(self) -> Optional[BodyFrame]:
        if self.i >= len(self.frames):
            return None
        kp = self.frames[self.i]
        self.i += 1
        return BodyFrame(keypoints_3d=kp,
                         keypoints_2d=[None] * N_KEYPOINTS,
                         confidences=[1.0] * N_KEYPOINTS,
                         image=self._image)

    def close(self) -> None:
        pass


# ─── generator (a): replay ────────────────────────────────────────────────────
def replay_frames(arm_dirs: Sequence[tuple],
                  depth_scale: float = C.DEPTH_SCALE) -> List[List[np.ndarray]]:
    """Keypoints that would have produced these robot-frame arm directions.

    `arm_dirs` is a sequence of (u_upper_l, u_fore_l, u_upper_r, u_fore_r), each
    a unit vector in the robot's PELVIS frame. Feed the result back through the
    pipeline and the arm should return to where the directions came from: a
    round trip through retargeting and IK, which nothing has ever measured.
    """
    return [keypoints_from_arm_dirs(*d, depth_scale=depth_scale)
            for d in arm_dirs]


# ─── generator (b): scripted human ────────────────────────────────────────────
def hand_path_frames(path_l: Callable[[float], np.ndarray],
                     path_r: Callable[[float], np.ndarray],
                     n: int,
                     swivel_l=(0.35, 0.35, -1.0),
                     swivel_r=(0.35, -0.35, -1.0),
                     depth_scale: float = C.DEPTH_SCALE
                     ) -> List[List[np.ndarray]]:
    """Frames for hand paths given as functions of phase in [0, 1].

    Each path returns the hand position RELATIVE TO THAT SHOULDER, in the
    robot's pelvis frame (x forward, y left, z up) - the frame the retargeting
    works in, so a motion can be authored in the terms the hazard is stated in
    ("in toward the chest", "out at box height") without a mental transform.
    """
    out = []
    for i in range(n):
        t = i / max(n - 1, 1)
        ul, fl = arm_dirs_from_hand(path_l(t), np.asarray(swivel_l, float))
        ur, fr = arm_dirs_from_hand(path_r(t), np.asarray(swivel_r, float))
        out.append(keypoints_from_arm_dirs(ul, fl, ur, fr, depth_scale))
    return out


def with_dropout(frames: List[List[np.ndarray]],
                 gaps: Sequence[tuple]) -> List[List[np.ndarray]]:
    """Replace frames with NaN keypoints. `gaps` is [(start, length), ...].

    A NaN arm keypoint is the ONLY input guard the controller keeps (TR6), and
    it drives the coast path: up to `max_coast_frames` the last good pose is
    re-applied, beyond that the arm freezes and the frame is reported.
    """
    out = [list(f) for f in frames]
    for start, length in gaps:
        for k in range(start, min(start + length, len(out))):
            out[k] = _nan_keypoints()
    return out
