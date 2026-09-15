"""Raw ZED BODY_38 keypoint recorder. No physics, no retargeting, no filtering.

    python tools/record_keypoints.py --label reach_box            # preview window
        SPACE / r  start or stop a take      q / ESC  quit (saves an open take)
    python tools/record_keypoints.py --label idle --headless --duration 30
    python tools/record_keypoints.py --label reach --still-s 3    # 3 s stand-still lead-in

Each take is written to recordings/keypoints/<label>_<NN>_<YYYYmmdd-HHMMSS>.npz in the
format g1_teleop/synthetic_source.py replays (`SyntheticSource.from_recording`), and
a summary is printed. Analyse with tools/analyze_keypoints.py.

WHAT IS RECORDED, AND WHAT IS NOT
---------------------------------
RAW camera-frame keypoints exactly as the SDK reports them - before One-Euro, depth
low-pass, geometric scaling, IK and coast logic, all of which must stay replayable
offline. NaNs are kept; nothing is interpolated, smoothed or dropped. A failed grab
is a row too. Dropout behaviour IS the measurement.

The camera is opened through `ZEDSource` itself, so init and body-tracking
parameters are the live pipeline's by construction, and the parameters are read
BACK from the camera into the session metadata rather than copied from code.
`grab_raw` follows `ZEDSource.grab` step for step - including retrieving and
copying the image, so the loop costs what the live grab costs and the frame rate
(and therefore dropout run lengths in frames) matches - and calls
`ZEDSource._select_best_body` itself, so the ok / arm_nan decision is the live one.
The grab runs on a background thread, as `run_integrated_combined.FrameGrabber`
does; the preview window only reads the latest frame.

Status per row (synthetic_source.STATUS_*): ok, grab_failed, not_new, no_body,
arm_nan, arm_conf_nan - "no body detected", "body detected with NaN arm
keypoints" and "body with finite arm keypoints but NaN arm confidence" are
different rows, and for the rejections the rejected body's keypoints are stored.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import platform
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from g1_teleop import config as C
from g1_teleop import synthetic_source as SS
from g1_teleop.zed_source import ZEDSource

OUT_DIR = os.path.join(ROOT, "recordings", "keypoints")
ARM_IDS = (12, 13, 14, 15, 16, 17)


# ─── pure logic, testable without a camera ───────────────────────────────────
def classify(grab_ok: bool, is_new: bool, body_list):
    """(status, body_to_store) following ZEDSource.grab's control flow exactly."""
    if not grab_ok:
        return SS.STATUS_GRAB_FAILED, None
    if not (is_new and body_list):
        # ZEDSource collapses both into one empty frame; split them here.
        status = SS.STATUS_NOT_NEW if not is_new else SS.STATUS_NO_BODY
        return status, (_best_raw(body_list) if body_list else None)
    body = ZEDSource._select_best_body(body_list)
    if body is not None:
        return SS.STATUS_OK, body
    # The DECISION above is ZEDSource's. Below only explains it: a body with all
    # six arm keypoints finite can only have been skipped because its mean arm
    # confidence was NaN (NaN > best_score is False).
    finite = [b for b in body_list
              if not any(np.any(np.isnan(np.asarray(b.keypoint, float)[i])) for i in ARM_IDS)]
    if finite:
        for b in finite:
            score = float(np.mean(np.asarray(b.keypoint_confidence, float)[list(ARM_IDS)]))
            assert np.isnan(score), ("ZEDSource._select_best_body rejected a body with finite "
                                     "arm keypoints and score %r - its logic has changed; "
                                     "update classify()" % score)
        return SS.STATUS_ARM_CONF_NAN, _best_raw(finite)
    return SS.STATUS_ARM_NAN, _best_raw(body_list)


def _best_raw(body_list):
    """Body to store when none is selected: highest NaN-tolerant mean arm confidence."""
    best, score = None, -np.inf
    for b in body_list:
        c = np.asarray(b.keypoint_confidence, float)[list(ARM_IDS)]
        s = np.nanmean(c) if np.isfinite(c).any() else -1.0
        if s > score:
            best, score = b, s
    return best


class TakeBuffer:
    def __init__(self):
        self.rows = []

    def add(self, status, body, n_bodies, ts_image_ns, ts_host_ns, dropped):
        if body is None:
            kp = np.full((SS.N_KEYPOINTS, 3), np.nan)
            cf = np.full(SS.N_KEYPOINTS, np.nan)
            bid = -1
        else:
            kp = np.array(body.keypoint, dtype=np.float64).reshape(SS.N_KEYPOINTS, 3)
            cf = np.array(body.keypoint_confidence, dtype=np.float64).reshape(SS.N_KEYPOINTS)
            bid = int(body.id)
        self.rows.append((kp, cf, status, n_bodies, bid, ts_image_ns, ts_host_ns, dropped))

    def to_recording(self, meta) -> SS.Recording:
        n = len(self.rows)
        col = list(zip(*self.rows)) if n else [[]] * 8
        return SS.Recording(
            keypoints=np.array(col[0], np.float64).reshape(n, SS.N_KEYPOINTS, 3),
            confidence=np.array(col[1], np.float64).reshape(n, SS.N_KEYPOINTS),
            status=np.array(col[2], np.int8), n_bodies=np.array(col[3], np.int16),
            body_id=np.array(col[4], np.int32), frame_index=np.arange(n, dtype=np.int64),
            ts_image_ns=np.array(col[5], np.int64), ts_host_ns=np.array(col[6], np.int64),
            dropped_total=np.array(col[7], np.int64), meta=meta)


# ─── camera ──────────────────────────────────────────────────────────────────
class RawGrabber(threading.Thread):
    def __init__(self, zed: ZEDSource):
        super().__init__(daemon=True)
        self.zed = zed
        self.lock = threading.Lock()
        self.running = True
        self.take = None                 # TakeBuffer while recording
        self.latest = None               # (image, kp2d, status, n_bodies) for preview
        self.error = None

    def grab_raw(self):
        """One iteration of ZEDSource.grab, returning everything it discards."""
        zed, sl = self.zed, self.zed._sl
        cam = zed.camera
        ok = cam.grab() == sl.ERROR_CODE.SUCCESS
        t_host = time.monotonic_ns()
        dropped = int(cam.get_frame_dropped_count())
        if not ok:
            return SS.STATUS_GRAB_FAILED, None, 0, 0, t_host, dropped, None, None
        cam.retrieve_image(zed._image, sl.VIEW.LEFT)
        image = zed._image.get_data()[:, :, :3].copy()        # same cost as live grab
        ts_img = int(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        cam.retrieve_bodies(zed._bodies, zed._runtime)
        bl = list(zed._bodies.body_list)
        status, body = classify(True, bool(zed._bodies.is_new), bl)
        kp2d = None if body is None else np.array(body.keypoint_2d, dtype=np.float64)
        return status, body, len(bl), ts_img, t_host, dropped, image, kp2d

    def run(self):
        try:
            while self.running:
                status, body, nb, ts_img, t_host, dropped, image, kp2d = self.grab_raw()
                with self.lock:
                    if self.take is not None:
                        self.take.add(status, body, nb, ts_img, t_host, dropped)
                    if image is not None:
                        self.latest = (image, kp2d, status, nb)
        except Exception as e:                                  # noqa: BLE001
            self.error = e
            raise

    def start_take(self):
        with self.lock:
            self.take = TakeBuffer()

    def stop_take(self):
        with self.lock:
            t, self.take = self.take, None
        return t


def session_meta(zed: ZEDSource, args) -> dict:
    sl, cam = zed._sl, zed.camera
    info = cam.get_camera_information()
    cc = info.camera_configuration
    init = cam.get_init_parameters()
    meta = dict(
        label=args.label, note=args.note, host=platform.node(),
        sdk_version=str(sl.Camera.get_sdk_version()),
        camera_model=str(info.camera_model), serial_number=int(info.serial_number),
        firmware_version=int(cc.firmware_version),
        resolution=dict(actual_wh=[int(cc.resolution.width), int(cc.resolution.height)],
                        init=str(init.camera_resolution), zed_config=zed.cfg.resolution),
        camera_fps=dict(actual=float(cc.fps), init=int(init.camera_fps),
                        zed_config=zed.cfg.camera_fps,
                        note="ZEDSource never applies ZEDConfig.camera_fps; `actual` is what ran"),
        depth_mode=str(init.depth_mode), coordinate_system=str(init.coordinate_system),
        coordinate_units=str(init.coordinate_units),
        runtime=dict(detection_confidence_threshold=float(zed._runtime.detection_confidence_threshold),
                     minimum_keypoints_threshold=int(zed._runtime.minimum_keypoints_threshold),
                     skeleton_smoothing=float(zed._runtime.skeleton_smoothing)),
        zed_config=dataclasses.asdict(zed.cfg),
        max_coast_frames_at_record=C.SmoothingConfig().max_coast_frames,
        preview=not args.headless, still_s=float(args.still_s),
    )
    try:
        bp = cam.get_body_tracking_parameters()
        meta["body_tracking"] = dict(
            detection_model=str(bp.detection_model), body_format=str(bp.body_format),
            enable_body_fitting=bool(bp.enable_body_fitting), enable_tracking=bool(bp.enable_tracking),
            max_range=float(bp.max_range), prediction_timeout_s=float(bp.prediction_timeout_s))
    except Exception as e:                                      # noqa: BLE001
        meta["body_tracking"] = dict(error="get_body_tracking_parameters failed: %r" % e,
                                     zed_source_hardcodes="HUMAN_BODY_ACCURATE, BODY_38, fitting+tracking on")
    return meta


def save_take(buf: TakeBuffer, base_meta: dict, take_no: int, started: datetime.datetime) -> str:
    import analyze_keypoints as AK
    meta = dict(base_meta, take=take_no, started_local=started.isoformat(timespec="seconds"))
    rec = buf.to_recording(meta)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "%s_%02d_%s.npz" % (base_meta["label"], take_no,
                                                     started.strftime("%Y%m%d-%H%M%S")))
    SS.save_recording(path, rec)
    print("\nsaved %s" % path)
    AK.summary(rec)
    return path


def _draw(cv2, image, kp2d, status, nb, rec_on, t_take, still_s):
    img = np.ascontiguousarray(image)
    if kp2d is not None:
        for k, p in enumerate(kp2d):
            if np.all(np.isfinite(p)):
                cv2.circle(img, (int(p[0]), int(p[1])), 5 if k in ARM_IDS else 3,
                           (0, 255, 255) if k in ARM_IDS else (200, 200, 200), -1)
    col = (0, 200, 0) if status == SS.STATUS_OK else (0, 0, 255)
    cv2.putText(img, "%s  bodies=%d" % (SS.STATUS_NAMES[status], nb), (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
    if rec_on:
        cv2.circle(img, (img.shape[1] - 40, 40), 15, (0, 0, 255), -1)
        msg = "REC %.1fs" % t_take
        if t_take < still_s:
            msg += "  STAND STILL %.1f" % (still_s - t_take)
        cv2.putText(img, msg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(img, "SPACE: record   q: quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="name of the take, e.g. reach_box_lamp_on")
    ap.add_argument("--note", default="", help="free text: lighting, clothing, distance...")
    ap.add_argument("--headless", action="store_true", help="no preview window")
    ap.add_argument("--duration", type=float, default=None, help="seconds per take (required headless)")
    ap.add_argument("--takes", type=int, default=1, help="headless: number of takes")
    ap.add_argument("--still-s", type=float, default=0.0,
                    help="seconds to stand still at the start of each take (marked in meta; "
                         "analyze_keypoints uses it as the jitter window)")
    args = ap.parse_args()
    if args.headless and not args.duration:
        ap.error("--headless needs --duration")

    print("opening ZED through ZEDSource (body-tracking model load can take a while)...")
    zed = ZEDSource(C.ZEDConfig())
    if not zed.camera.is_opened():
        raise SystemExit("ZED did not open - ZEDSource does not check camera.open(); check the cable/SDK")
    base = session_meta(zed, args)
    print(json.dumps({k: base[k] for k in ("sdk_version", "camera_model", "resolution", "camera_fps",
                                           "depth_mode", "body_tracking")}, indent=1))
    grabber = RawGrabber(zed)
    grabber.start()
    take_no, saved = 0, []
    started = None
    try:
        if args.headless:
            for _ in range(args.takes):
                take_no += 1
                for c in (3, 2, 1):
                    print("take %d starts in %d..." % (take_no, c))
                    time.sleep(1.0)
                started = datetime.datetime.now()
                grabber.start_take()
                if args.still_s:
                    print("STAND STILL for %.1f s" % args.still_s)
                t0 = time.monotonic()
                while time.monotonic() - t0 < args.duration:
                    if grabber.error:
                        raise grabber.error
                    time.sleep(0.05)
                saved.append(save_take(grabber.stop_take(), base, take_no, started))
                started = None
        else:
            import cv2
            win = "ZED raw keypoint recorder"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            t_take = None
            while True:
                if grabber.error:
                    raise grabber.error
                with grabber.lock:
                    latest = grabber.latest
                if latest is not None:
                    el = time.monotonic() - t_take if t_take else 0.0
                    cv2.imshow(win, _draw(cv2, *latest, t_take is not None, el, args.still_s))
                key = cv2.waitKey(15) & 0xFF
                if key in (ord(" "), ord("r")):
                    if t_take is None:
                        take_no += 1
                        started = datetime.datetime.now()
                        grabber.start_take()
                        t_take = time.monotonic()
                        print("take %d recording..." % take_no)
                    else:
                        saved.append(save_take(grabber.stop_take(), base, take_no, started))
                        t_take, started = None, None
                elif key in (ord("q"), 27):
                    break
            cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        buf = grabber.stop_take()
        if buf is not None and buf.rows and started is not None:
            saved.append(save_take(buf, base, take_no, started))
        grabber.running = False
        grabber.join(timeout=2.0)
        zed.close()
    print("\n%d take(s) saved:" % len(saved))
    for p in saved:
        print("  " + p)
    if saved:
        print("analyse:  python tools/analyze_keypoints.py %s" % " ".join('"%s"' % p for p in saved))


if __name__ == "__main__":
    main()
