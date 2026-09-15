"""Camera-free test of the keypoint recorder, format and analysis.

    python tools/test_keypoint_recording.py

Fake SDK bodies drive the recorder's own `classify` + `TakeBuffer` through every
status (ok, grab_failed, not_new, no_body, arm_nan, arm_conf_nan - including an arm_nan body
whose rejected keypoints must survive, and a NaN in a NON-arm keypoint on an ok
frame, which ZEDSource accepts). The take is saved, loaded, and:
  1. replayed through SyntheticSource in both modes and compared bit-for-bit;
  2. mode="zed" compared with what ZEDSource.grab's own logic returns for the
     same fake bodies;
  3. replayed through the real TeleopController, whose frozen-frame count must
     equal the analysis script's prediction - so "fraction of runs exceeding
     max_coast_frames" is measured against the controller, not a re-implementation.
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from g1_teleop import synthetic_source as SS
from g1_teleop.zed_source import ZEDSource
import record_keypoints as RK
import analyze_keypoints as AK


class FakeBody:
    def __init__(self, kp, conf, bid=7):
        self.keypoint, self.keypoint_confidence, self.id = kp, conf, bid
        self.keypoint_2d = np.zeros((38, 2))


def arm_pose(t):
    """A plausible arm skeleton in the camera frame, moving slowly."""
    kp = np.full((38, 3), 0.1) + np.array([0.0, 0.0, 2.0])
    for side, sx in ((0, 1.0), (1, -1.0)):
        sh = np.array([-sx * 0.2, -0.3, 2.0])
        el = sh + np.array([-sx * 0.05, 0.28, -0.05 * np.sin(t)])
        wr = el + np.array([0.0, 0.05, -0.25 - 0.05 * np.cos(t)])
        kp[12 + side], kp[14 + side], kp[16 + side] = sh, el, wr
    return kp


def main():
    rng = np.random.default_rng(0)
    # schedule: (status wanted, count). Runs of 1..9 dropouts of mixed cause, a
    # leading dropout before any good frame, grab failures inside a run.
    plan = [("no_body", 3), ("ok", 20), ("arm_nan", 2), ("ok", 10), ("not_new", 5),
            ("ok", 8), ("no_body", 6), ("ok", 5), ("arm_nan", 4), ("grab_failed", 2),
            ("arm_nan", 3), ("ok", 12), ("ok_nonarm_nan", 4), ("not_new", 1), ("ok", 6),
            ("no_body", 9), ("ok", 15), ("conf_nan", 7), ("ok", 5)]
    buf = RK.TakeBuffer()
    expected_zed = []            # what ZEDSource.grab returns, None = not delivered
    t_ns, i = 1_000_000_000, 0
    for what, n in plan:
        for _ in range(n):
            i += 1
            t_ns += 16_666_667
            kp = arm_pose(i * 0.05) + rng.normal(0, 0.002, (38, 3))
            conf = rng.uniform(40, 95, 38)
            if what == "grab_failed":
                grab_ok, is_new, bl = False, True, []
            elif what == "no_body":
                grab_ok, is_new, bl = True, True, []
            elif what == "not_new":
                grab_ok, is_new, bl = True, False, [FakeBody(kp, conf)]
            elif what == "arm_nan":
                k2 = kp.copy()
                k2[16] = np.nan                              # left wrist lost
                c2 = conf.copy()
                c2[16] = np.nan
                bl = [FakeBody(k2, c2), FakeBody(k2 + 1.0, c2 * 0.5, bid=9)]
                grab_ok, is_new = True, True
            elif what == "conf_nan":
                c2 = conf.copy()
                c2[13] = np.nan                              # keypoint finite, confidence NaN
                grab_ok, is_new, bl = True, True, [FakeBody(kp, c2, bid=11)]
            elif what == "ok_nonarm_nan":
                k2 = kp.copy()
                k2[30] = np.nan                              # a non-arm point
                grab_ok, is_new, bl = True, True, [FakeBody(k2, conf)]
            else:
                grab_ok, is_new, bl = True, True, [FakeBody(kp, conf), FakeBody(kp + 0.5, conf * 0.3, 8)]
            status, body = RK.classify(grab_ok, is_new, bl)
            buf.add(status, body, len(bl), t_ns if grab_ok else 0, t_ns, 0)
            # ZEDSource.grab's own branches, written from zed_source.py:
            if not grab_ok:
                expected_zed.append(None)
            elif not (is_new and bl):
                expected_zed.append([])
            else:
                sel = ZEDSource._select_best_body(bl)
                expected_zed.append([] if sel is None else [np.array(k, float) for k in sel.keypoint])
    rec = buf.to_recording(dict(label="selftest", still_s=0.0))
    counts = {SS.STATUS_NAMES[c]: int((rec.status == c).sum()) for c in SS.STATUS_NAMES}
    print("statuses:", counts)
    assert all(v > 0 for v in counts.values()), counts
    arm_nan_rows = np.flatnonzero(rec.status == SS.STATUS_ARM_NAN)
    assert np.isnan(rec.keypoints[arm_nan_rows, 16]).all() and not np.isnan(rec.keypoints[arm_nan_rows, 12]).any(), \
        "arm_nan rows must keep the rejected body's keypoints, NaN wrist included"
    assert (rec.body_id[arm_nan_rows] == 7).all(), "stored body must be the best rejected one"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "selftest.npz")
        SS.save_recording(path, rec)
        ok1 = AK.roundtrip(path)
        src = SS.SyntheticSource.from_recording(path, mode="zed")
        got = []
        while True:
            f = src.grab()
            if f is None:
                break
            got.append(f.keypoints_3d)
        exp = [e for e in expected_zed if e is not None]
        ok2 = len(got) == len(exp) and all(
            (g == [] and e == []) or (len(g) == len(e) == 38 and np.array_equal(np.stack(g), np.stack(e), equal_nan=True))
            for g, e in zip(got, exp))
        print("  mode=zed equals ZEDSource.grab logic on the same fake bodies:", ok2)
        ok3 = AK.controller_check(path)
        AK.summary(SS.load_recording(path))
        AK.dropout_report([SS.load_recording(path)], 5)
    ok = ok1 and ok2 and ok3
    print("\nSELFTEST %s" % ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
