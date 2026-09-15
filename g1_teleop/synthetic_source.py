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

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

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
    tracking dropout does, and an EMPTY list is what `ZEDSource.grab` returns
    when it has no usable body. `grab()` returns None past the end, which is
    how `ZEDSource` reports a failed grab.

    `confidences`, if given, is one 38-list per frame and is returned verbatim;
    otherwise every keypoint reports 1.0 (the authored generators have no
    confidence). `SyntheticSource.from_recording` replays a real ZED take.
    """

    def __init__(self, frames: Sequence[List[np.ndarray]], image_hw=(8, 8),
                 confidences: Optional[Sequence[List[float]]] = None,
                 fps: Optional[float] = None):
        # `fps`: pace grab() to wall clock, like a camera. Without it a grabber
        # thread drains every frame in milliseconds and the control loop sees
        # only the last one. Leave None when the caller pulls frames on its own
        # (simulated) clock, as run_integrated_combined.py --headless does.
        self.fps = fps
        self._t0 = None
        self.frames = list(frames)
        self.confidences = None if confidences is None else list(confidences)
        if self.confidences is not None:
            assert len(self.confidences) == len(self.frames)
        self.i = 0
        self._image = np.zeros((image_hw[0], image_hw[1], 3), dtype=np.uint8)

    @classmethod
    def from_recording(cls, path_or_rec, mode: str = "zed") -> "SyntheticSource":
        """Replay a raw keypoint recording (tools/record_keypoints.py).

        mode="zed": exactly the frame sequence `ZEDSource.grab` would have handed
                    the controller live - rejected / no-body / stale frames become
                    EMPTY keypoint lists, failed grabs are skipped (the grabber
                    never passes a None on). Use this to test the pipeline.
        mode="raw": every recorded row, including the raw keypoints of a body the
                    selector REJECTED (NaN arm keypoints preserved) and all-NaN rows
                    where no body existed. Use this to test alternative selection.
        """
        rec = path_or_rec if isinstance(path_or_rec, Recording) else load_recording(path_or_rec)
        kp, conf = rec.frames(mode)
        return cls(kp, confidences=conf)

    def __len__(self) -> int:
        return len(self.frames)

    def grab(self) -> Optional[BodyFrame]:
        if self.i >= len(self.frames):
            return None
        if self.fps:
            import time
            if self._t0 is None:
                self._t0 = time.monotonic()
            wait = self._t0 + self.i / self.fps - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        kp = self.frames[self.i]
        self.i += 1
        conf = ([1.0] * N_KEYPOINTS if self.confidences is None
                else self.confidences[self.i - 1])
        return BodyFrame(keypoints_3d=kp,
                         keypoints_2d=[None] * N_KEYPOINTS,
                         confidences=conf,
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


# ─── generator (c): grasp and lift, for the base-lock / weld path ─────────────
def grasp_lift_frames(shoulders, lengths, home, box, fps: float,
                      hand_dx: float = -0.09, face_y: float = 0.13, hand_dz: float = 0.04,
                      clear_dz: float = 0.14, lift: float = 0.20,
                      t_home: float = 5.0, t_up: float = 1.5, t_out: float = 1.5,
                      t_down: float = 1.5, t_hold: float = 2.0, t_lift: float = 2.0,
                      t_end: float = 4.0):
    """A two-handed grasp-and-lift in the robot's PELVIS frame. Returns
    (frames, grasp_on_s, lift_start_s).

    `shoulders`, `home` and `box` are pelvis-frame positions (dicts per side for
    the first two); `lengths[side] = (upper, fore)` are the ROBOT's limb lengths,
    because retargeting keeps only directions and rescales by them. Hand targets
    are wrist points.

    Path per hand: hold home (the base-lock predicate needs a settled base first)
    -> rise straight up beside the body to `clear_dz` above the box centre -> out
    to the box -> down to the grasp point -> hold (grasp command goes on at the
    start of the hold) -> lift by `lift` -> hold.

    The default grasp point (hand 0.09 m behind the box centre, 0.13 m to the
    side, 0.04 m up) came from a probe on the kinematic TWIN, which only picks a
    candidate (TR16a); whether it passes the weld gate is decided on the stepped
    model by the run that uses it. Palms at 0.13 m lateral sit ~37 mm outside the
    box faces, so the hands close the gate without pushing the box.
    """
    box = np.asarray(box, float)
    segs = [("home", t_home), ("up", t_up), ("out", t_out), ("down", t_down),
            ("hold", t_hold), ("lift", t_lift), ("end", t_end)]
    paths = {}
    for side, sg in (("left", 1.0), ("right", -1.0)):
        h0 = np.asarray(home[side], float)
        grasp = box + np.array([hand_dx, sg * face_y, hand_dz])
        up = np.array([h0[0], grasp[1], box[2] + clear_dz])
        out = np.array([grasp[0], grasp[1], box[2] + clear_dz])
        lifted = grasp + np.array([0.0, 0.0, lift])
        paths[side] = {"home": (h0, h0), "up": (h0, up), "out": (up, out),
                       "down": (out, grasp), "hold": (grasp, grasp),
                       "lift": (grasp, lifted), "end": (lifted, lifted)}
    frames, t = [], 0.0
    marks = {}
    for name, dur in segs:
        marks[name] = t
        n = max(1, int(round(dur * fps)))
        for k in range(n):
            a = 0.5 - 0.5 * np.cos(np.pi * k / max(n - 1, 1))     # ease in/out
            dirs = []
            for side, sg in (("left", 1.0), ("right", -1.0)):
                p0, p1 = paths[side][name]
                hand = p0 + a * (p1 - p0) - np.asarray(shoulders[side], float)
                u, f = arm_dirs_from_hand(hand, np.array([0.0, sg * 0.35, -1.0]),
                                          *lengths[side])
                dirs += [u, f]
            frames.append(keypoints_from_arm_dirs(*dirs))
        t += n / fps
    return frames, marks["hold"], marks["lift"]


# ─── recorded real motion: the on-disk form of the `frames` sequence ─────────
# The replay format above is a sequence of BODY_38 keypoint lists in the CAMERA
# frame. A recording is that same sequence stored as arrays - no second format:
# `Recording.frames()` rebuilds exactly the list-of-38-arrays `SyntheticSource`
# takes. One `.npz` per take, written by tools/record_keypoints.py.
#
#   keypoints     (N, 38, 3) float64  camera frame, metres, as the SDK reported
#                                     them - NaN preserved, nothing filtered
#   confidence    (N, 38)    float64  SDK keypoint_confidence (0-100, NaN kept)
#   status        (N,)       int8     STATUS_* below: why ZEDSource would or would
#                                     not have used this frame
#   n_bodies      (N,)       int16    bodies in the SDK list this frame
#   body_id       (N,)       int32    tracker id of the stored body, -1 if none
#   frame_index   (N,)       int64    grab-loop index, 0-based, includes failures
#   ts_image_ns   (N,)       int64    SDK image timestamp (0 when the grab failed)
#   ts_host_ns    (N,)       int64    host monotonic clock after the grab
#   dropped_total (N,)       int64    SDK get_frame_dropped_count() after the grab
#   meta          ()         str      JSON: session + take metadata
#
# STATUS mirrors the control flow of ZEDSource.grab exactly:
STATUS_OK = 0            # body selected; ZEDSource returns its keypoints
STATUS_GRAB_FAILED = 1   # camera.grab() != SUCCESS; ZEDSource returns None
STATUS_NOT_NEW = 2       # bodies.is_new False; ZEDSource returns an EMPTY frame
STATUS_NO_BODY = 3       # body_list empty; EMPTY frame
STATUS_ARM_NAN = 4       # bodies present, _select_best_body rejected every one
                         # for a NaN arm KEYPOINT; EMPTY frame. The stored
                         # keypoints are the best REJECTED body, so which arm
                         # point went NaN stays visible.
STATUS_ARM_CONF_NAN = 5  # a body's arm keypoints are all finite but one of its
                         # arm CONFIDENCES is NaN: _select_best_body's mean score
                         # is NaN, `NaN > best_score` is False, and the body is
                         # silently never selected; EMPTY frame. Found on the
                         # first real take (2026-09-15): with body fitting on,
                         # SDK 5.4 fills occluded keypoints and reports NaN
                         # confidence for them. Stored body = best such body.
                         # FIXED the same day (nanmean score): the current
                         # selector cannot produce it. Kept so recordings made
                         # before the fix still load; their meta has no
                         # `selector` key, and mode="zed" replays the OLD
                         # selector's decision recorded in `status`.
STATUS_NAMES = {STATUS_OK: "ok", STATUS_GRAB_FAILED: "grab_failed",
                STATUS_NOT_NEW: "not_new", STATUS_NO_BODY: "no_body",
                STATUS_ARM_NAN: "arm_nan", STATUS_ARM_CONF_NAN: "arm_conf_nan"}
RECORDING_VERSION = "g1-kp-rec-2"
_REC_ARRAYS = ("keypoints", "confidence", "status", "n_bodies", "body_id",
               "frame_index", "ts_image_ns", "ts_host_ns", "dropped_total")


@dataclass
class Recording:
    keypoints: np.ndarray
    confidence: np.ndarray
    status: np.ndarray
    n_bodies: np.ndarray
    body_id: np.ndarray
    frame_index: np.ndarray
    ts_image_ns: np.ndarray
    ts_host_ns: np.ndarray
    dropped_total: np.ndarray
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.keypoints.shape[0])

    def delivered(self) -> np.ndarray:
        """Rows ZEDSource.grab returns a BodyFrame for (all but grab failures)."""
        return self.status != STATUS_GRAB_FAILED

    def frames(self, mode: str = "zed"):
        """(keypoint lists, confidence lists) in the SyntheticSource shape."""
        if mode not in ("zed", "raw"):
            raise ValueError(mode)
        kps, confs = [], []
        for i in range(len(self)):
            st = int(self.status[i])
            if mode == "zed":
                if st == STATUS_GRAB_FAILED:
                    continue
                if st != STATUS_OK:       # ZEDSource: BodyFrame([], [None]*38, [0.0]*38)
                    kps.append([])
                    confs.append([0.0] * N_KEYPOINTS)
                    continue
            kps.append([self.keypoints[i, k].copy() for k in range(N_KEYPOINTS)])
            confs.append([float(c) for c in self.confidence[i]])
        return kps, confs


def save_recording(path: str, rec: Recording) -> None:
    n = len(rec)
    assert rec.keypoints.shape == (n, N_KEYPOINTS, 3), rec.keypoints.shape
    assert rec.confidence.shape == (n, N_KEYPOINTS), rec.confidence.shape
    for name in _REC_ARRAYS[2:]:
        assert getattr(rec, name).shape == (n,), name
    meta = dict(rec.meta, recording_version=RECORDING_VERSION)
    np.savez_compressed(
        path,
        keypoints=np.asarray(rec.keypoints, np.float64),
        confidence=np.asarray(rec.confidence, np.float64),
        status=np.asarray(rec.status, np.int8),
        n_bodies=np.asarray(rec.n_bodies, np.int16),
        body_id=np.asarray(rec.body_id, np.int32),
        frame_index=np.asarray(rec.frame_index, np.int64),
        ts_image_ns=np.asarray(rec.ts_image_ns, np.int64),
        ts_host_ns=np.asarray(rec.ts_host_ns, np.int64),
        dropped_total=np.asarray(rec.dropped_total, np.int64),
        meta=np.array(json.dumps(meta)))


def load_recording(path: str) -> Recording:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta.get("recording_version") != RECORDING_VERSION:
            raise ValueError("%s: recording version %r, this loader reads %r"
                             % (path, meta.get("recording_version"), RECORDING_VERSION))
        return Recording(**{k: z[k].copy() for k in _REC_ARRAYS}, meta=meta)
