"""Analyse raw ZED keypoint recordings (tools/record_keypoints.py).

    python tools/analyze_keypoints.py recordings/keypoints/*.npz
    python tools/analyze_keypoints.py take.npz --still 0:3      # still window, seconds
    python tools/analyze_keypoints.py take.npz --roundtrip      # prove the format

WHAT A "DROPOUT" IS HERE - the controller's definition, not the SDK's
--------------------------------------------------------------------
`TeleopController._step_inner` coasts a frame when its keypoint list is EMPTY or
an arm keypoint is NaN. `ZEDSource.grab` returns an EMPTY list for status
not_new / no_body / arm_nan / arm_conf_nan, and returns None for grab_failed - which the
grabber thread never passes on, so a failed grab is NOT a controller frame. So:

    dropout frame = a delivered frame (status != grab_failed) with status != ok
    dropout run   = maximal run of consecutive dropout frames

A run longer than `SmoothingConfig.max_coast_frames` is where the arm stops
coasting and FREEZES: frames 1..max_coast re-apply the last pose, the rest are
frozen. Run length in ms is measured on the host clock from the first dropout
frame to the first frame after the run - the time the pipeline had no fresh pose.

ASSUMPTION stated, not hidden: one controller step per camera frame, as in
SyntheticSource replay and run_teleop.py. run_integrated_combined.py processes
only the LATEST frame per physics tick, so if its main loop runs slower than the
camera it sees a subsample, and a run measured here in camera frames spans fewer
controller steps there.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g1_teleop import config as C
from g1_teleop import synthetic_source as SS

ARM = {12: "L shoulder", 13: "R shoulder", 14: "L elbow", 15: "R elbow",
       16: "L wrist", 17: "R wrist"}
LIMBS = (("L upper", 12, 14), ("L fore", 14, 16), ("R upper", 13, 15), ("R fore", 15, 17))


# ─── primitives, shared with the recorder's per-take printout ────────────────
def dropout_runs(rec):
    """[(start_row, n_frames, ms)] over DELIVERED frames, in row order."""
    rows = np.flatnonzero(rec.delivered())
    drop = rec.status[rows] != SS.STATUS_OK
    t = rec.ts_host_ns[rows]
    runs, i = [], 0
    while i < len(rows):
        if not drop[i]:
            i += 1
            continue
        j = i
        while j < len(rows) and drop[j]:
            j += 1
        end_t = t[j] if j < len(rows) else t[-1]
        runs.append((int(rows[i]), j - i, (end_t - t[i]) / 1e6))
        i = j
    return runs


def predicted_frozen(rec, coast):
    """Frames the controller would FREEZE, replaying TeleopController._coast:
    a dropout coasts while fewer than `coast` consecutive frames have been
    coasted and a good pose exists; otherwise it freezes. Dropouts before the
    first good frame all freeze (no pose to coast on). Assumes the IK never
    returns None on an ok frame - checked against the real controller by
    `--controller`."""
    have_good, count, frozen = False, 0, 0
    for st in rec.status[rec.delivered()]:
        if st == SS.STATUS_OK:
            have_good, count = True, 0
        elif have_good and count < coast:
            count += 1
        else:
            frozen += 1
    return frozen


def duration_s(rec):
    if len(rec) < 2:
        return 0.0
    return float(rec.ts_host_ns[-1] - rec.ts_host_ns[0]) / 1e9


def summary(rec, printer=print):
    """The per-take printout the brief asks for."""
    n = len(rec)
    dur = duration_s(rec)
    deliv = int(rec.delivered().sum())
    runs = dropout_runs(rec)
    st = {name: int((rec.status == code).sum()) for code, name in SS.STATUS_NAMES.items()}
    img = rec.ts_image_ns[rec.delivered()]
    img_hz = (len(img) - 1) / ((img[-1] - img[0]) / 1e9) if len(img) > 1 and img[-1] > img[0] else float("nan")
    printer("take %r: %d frames (%d delivered), %.2f s, effective %.2f Hz host / %.2f Hz image-clock"
            % (rec.meta.get("label"), n, deliv, dur, (deliv - 1) / dur if dur > 0 else float("nan"), img_hz))
    printer("  status: " + ", ".join("%s %d" % kv for kv in st.items())
            + " | SDK dropped-frame counter +%d" % int(rec.dropped_total[-1] - rec.dropped_total[0] if n else 0))
    conf = rec.confidence[rec.status == SS.STATUS_OK]
    if len(conf):
        with np.errstate(all="ignore"):
            mean, mn = np.nanmean(conf, axis=0), np.nanmin(conf, axis=0)
        cells = []
        for k in range(SS.N_KEYPOINTS):
            tag = "*" if k in ARM else " "
            cells.append("%s%2d %5.1f/%5.1f" % (tag, k, mean[k], mn[k]))
        printer("  confidence mean/min per keypoint over ok frames (* = arm, gates the pipeline):")
        for r in range(0, SS.N_KEYPOINTS, 6):
            printer("    " + "  ".join(cells[r:r + 6]))
    else:
        printer("  confidence: no ok frames")
    longest = max(runs, key=lambda r: r[1]) if runs else None
    printer("  dropouts: %d runs, %d frames | longest %s"
            % (len(runs), sum(r[1] for r in runs),
               "%d frames = %.0f ms" % (longest[1], longest[2]) if longest else "-"))
    return dict(frames=n, delivered=deliv, duration_s=dur, runs=runs, status=st)


# ─── analysis ────────────────────────────────────────────────────────────────
def conf_table(recs):
    conf = np.concatenate([r.confidence[r.status == SS.STATUS_OK] for r in recs]) \
        if recs else np.zeros((0, SS.N_KEYPOINTS))
    print("\n== confidence per keypoint (ok frames, n=%d) ==" % len(conf))
    if not len(conf):
        return
    print("  kp  name          mean    p5   p50   min  NaN%")
    order = sorted(range(SS.N_KEYPOINTS), key=lambda k: (k not in ARM, k))
    with np.errstate(all="ignore"):
        for k in order:
            c = conf[:, k]
            print("  %s%2d  %-11s %5.1f %5.1f %5.1f %5.1f %5.1f"
                  % ("*" if k in ARM else " ", k, ARM.get(k, ""), np.nanmean(c),
                     np.nanpercentile(c, 5), np.nanpercentile(c, 50), np.nanmin(c),
                     100 * np.mean(np.isnan(c))))
            if k == 17:
                print("  -- the six arm keypoints above gate the pipeline; the rest are not read --")


def dropout_report(recs, coast):
    print("\n== dropouts (controller definition; see module docstring) ==")
    all_runs, minutes, deliv, dframes = [], 0.0, 0, 0
    reason_frames = {}
    for r in recs:
        runs = dropout_runs(r)
        all_runs += runs
        minutes += duration_s(r) / 60.0
        d = r.delivered()
        deliv += int(d.sum())
        dframes += sum(x[1] for x in runs)
        for code, name in SS.STATUS_NAMES.items():
            if code not in (SS.STATUS_OK, SS.STATUS_GRAB_FAILED):
                reason_frames[name] = reason_frames.get(name, 0) + int((r.status == code).sum())
        per = np.diff(r.ts_host_ns[d]) / 1e6
        if len(per):
            print("  %-28s frame period p50 %.1f ms, p95 %.1f, max %.1f | grab failures %d"
                  % (os.path.basename(r.meta.get("path", r.meta.get("label", "?")))[:28],
                     np.median(per), np.percentile(per, 95), per.max(),
                     int((r.status == SS.STATUS_GRAB_FAILED).sum())))
    print("  total: %.2f min, %d delivered frames, %d dropout frames (%.2f%%), %d runs = %.1f runs/min"
          % (minutes, deliv, dframes, 100.0 * dframes / max(deliv, 1), len(all_runs),
             len(all_runs) / minutes if minutes else float("nan")))
    print("  dropout frames by cause: %s" % reason_frames)
    if not all_runs:
        print("  no dropout runs - nothing to say about max_coast_frames from this data")
        return
    L = np.array([x[1] for x in all_runs])
    ms = np.array([x[2] for x in all_runs])
    edges = [1, 2, 3, 4, 5, 6, 8, 11, 16, 31, 61, 10 ** 9]
    print("  run-length histogram (frames):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = int(((L >= lo) & (L < hi)).sum())
        lab = "%d" % lo if hi == lo + 1 else ("%d-%d" % (lo, hi - 1) if hi < 10 ** 9 else ">=%d" % lo)
        print("    %-7s %5d  %s%s" % (lab, k, "#" * min(k, 60), " <- coast limit" if lo <= coast < hi else ""))
    over = L > coast
    frozen = sum(predicted_frozen(r, coast) for r in recs)
    print("\n  THE KEY NUMBER: %d of %d dropout runs (%.1f%%) exceed max_coast_frames = %d"
          % (int(over.sum()), len(L), 100.0 * over.mean(), coast))
    print("  frozen frames (beyond the coast window): %d of %d dropout frames (%.1f%%), "
          "%.2f%% of all delivered frames"
          % (frozen, int(L.sum()), 100.0 * frozen / L.sum(), 100.0 * frozen / max(deliv, 1)))
    per_ms = float(np.median(np.concatenate([np.diff(r.ts_host_ns[r.delivered()]) for r in recs]) / 1e6))
    print("  coast window = %d frames = %.0f ms at the measured %.1f ms frame period"
          % (coast, coast * per_ms, per_ms))
    for q in (50, 90, 95, 99):
        print("  run length p%d: %.1f frames, %.0f ms" % (q, np.percentile(L, q), np.percentile(ms, q)))
    print("  longest run: %d frames, %.0f ms" % (L.max(), ms.max()))
    need95 = int(np.ceil(np.percentile(L, 95)))
    print("\n  VERDICT INPUTS: covering 95%% of runs needs max_coast_frames >= %d (%.0f ms)."
          % (need95, need95 * per_ms))
    if over.mean() <= 0.05:
        verdict = "5 covers >=95% of runs - adequate for this demonstrator and lighting"
    else:
        verdict = ("5 is TOO SHORT here: %.1f%% of runs freeze the arms" % (100.0 * over.mean()))
    print("  VERDICT: %s." % verdict)
    print("  Caveat: a longer coast is not free - it replays a stale pose for longer, so "
          "runs near the top of the histogram (long occlusions) should freeze. Judge the "
          "tail, not only the 95% figure. One demonstrator, one room: not general.")


def _still_windows(rec, still, win_s=1.0, max_windows=5):
    """(start, end) rows. From --still, else meta['still_s'], else the lowest-motion
    1 s windows with no dropout (auto - flagged as such in the output)."""
    t = (rec.ts_host_ns - rec.ts_host_ns[0]) / 1e9
    ok = rec.status == SS.STATUS_OK
    if still:
        a, b = still
        return [(int(np.searchsorted(t, a)), int(np.searchsorted(t, b)))], "given"
    if rec.meta.get("still_s"):
        return [(0, int(np.searchsorted(t, float(rec.meta["still_s"]))))], "recorded still segment"
    arm = rec.keypoints[:, [12, 13, 14, 15, 16, 17], :]
    rows = np.flatnonzero(ok)
    if len(rows) < 10:
        return [], "none"
    per = np.median(np.diff(t[rows]))
    w = max(5, int(round(win_s / per)))
    cands = []
    for s in range(0, len(rec) - w, max(1, w // 2)):
        seg = slice(s, s + w)
        if not ok[seg].all():
            continue
        cands.append((float(np.nanmean(np.nanstd(arm[seg], axis=0))), s, s + w))
    cands.sort()
    chosen, used = [], []
    for sc, s, e in cands:
        if all(e <= a or s >= b for a, b in used):
            chosen.append((s, e))
            used.append((s, e))
        if len(chosen) >= max_windows:
            break
    return chosen, "AUTO (lowest-motion 1 s windows; not verified to be still - use --still)"


def jitter_report(recs, still):
    print("\n== jitter while still: the noise floor One-Euro has to handle ==")
    print("  camera frame: x right, y down, z = depth (the axis D1 scales by DEPTH_SCALE)")
    for r in recs:
        wins, how = _still_windows(r, still)
        print("  %s: windows from %s: %s" % (r.meta.get("label"), how,
                                           ["%.1f-%.1fs" % ((r.ts_host_ns[a] - r.ts_host_ns[0]) / 1e9,
                                                            (r.ts_host_ns[min(b, len(r) - 1)] - r.ts_host_ns[0]) / 1e9)
                                            for a, b in wins]))
        if not wins:
            continue
        print("    kp  name         std x   std y   std z (mm)   frame-to-frame RMS (mm)")
        for k, name in ARM.items():
            sds, f2f = [], []
            for a, b in wins:
                seg = r.keypoints[a:b, k, :]
                seg = seg[r.status[a:b] == SS.STATUS_OK]
                seg = seg[~np.isnan(seg).any(axis=1)]
                if len(seg) < 3:
                    continue
                sds.append(np.std(seg, axis=0))
                f2f.append(np.sqrt(np.mean(np.sum(np.diff(seg, axis=0) ** 2, axis=1))))
            if sds:
                s = 1000 * np.mean(sds, axis=0)
                print("    %2d  %-10s  %6.1f  %6.1f  %6.1f        %6.1f" % (k, name, *s, 1000 * np.mean(f2f)))


def limb_report(recs):
    print("\n== limb-length stability (does the tracker rescale the skeleton?) ==")
    for r in recs:
        ok = r.status == SS.STATUS_OK
        t = (r.ts_host_ns - r.ts_host_ns[0]) / 1e9
        print("  %s:" % r.meta.get("label"))
        print("    limb       mean mm   std   CV%   p5..p95        first10% -> last10% median   slope mm/min")
        for name, a, b in LIMBS:
            L = np.linalg.norm(r.keypoints[:, a, :] - r.keypoints[:, b, :], axis=1)
            m = ok & ~np.isnan(L)
            if m.sum() < 10:
                print("    %-8s  too few frames" % name)
                continue
            v, tt = 1000 * L[m], t[m]
            k = max(1, len(v) // 10)
            slope = np.polyfit(tt / 60.0, v, 1)[0] if tt[-1] > tt[0] else float("nan")
            print("    %-8s  %7.1f %6.2f %5.2f  %6.1f..%6.1f   %6.1f -> %6.1f            %+7.2f%s"
                  % (name, v.mean(), v.std(), 100 * v.std() / v.mean(), np.percentile(v, 5),
                     np.percentile(v, 95), np.median(v[:k]), np.median(v[-k:]), slope,
                     "   <- identical every frame: body fitting fixes the skeleton (TR18: expected, not a rig fault)"
                     if v.std() < 1e-6 else ""))


def controller_check(path):
    """Replay mode="zed" through the REAL TeleopController (twin only, no
    physics) and compare its frozen-frame count with predicted_frozen."""
    from g1_teleop.config import TeleopConfig
    from g1_teleop.robot import G1Robot
    from g1_teleop.teleop import TeleopController
    import contextlib, io
    rec = SS.load_recording(path)
    cfg = TeleopConfig()
    ctl = TeleopController(G1Robot(cfg), cfg)
    src = SS.SyntheticSource.from_recording(rec, mode="zed")
    frozen = 0
    with contextlib.redirect_stdout(io.StringIO()):
        while True:
            f = src.grab()
            if f is None:
                break
            frozen += 0 if ctl.step(f).applied else 1
    pred = predicted_frozen(rec, cfg.smoothing.max_coast_frames)
    print("  controller replay: %d frozen frames, analysis predicted %d -> %s"
          % (frozen, pred, "MATCH" if frozen == pred else "MISMATCH"))
    return frozen == pred


def roundtrip(path):
    """Replay through SyntheticSource with no camera; compare to what was recorded."""
    rec = SS.load_recording(path)
    fails = 0
    for mode in ("raw", "zed"):
        src = SS.SyntheticSource.from_recording(path, mode=mode)
        got_kp, got_cf = [], []
        while True:
            f = src.grab()
            if f is None:
                break
            got_kp.append(f.keypoints_3d)
            got_cf.append(f.confidences)
        rows = np.arange(len(rec)) if mode == "raw" else np.flatnonzero(rec.delivered())
        ok = len(got_kp) == len(rows)
        for g, i in zip(got_kp, rows):
            st = int(rec.status[i])
            if mode == "zed" and st != SS.STATUS_OK:
                ok &= (g == [])
            else:
                ok &= len(g) == SS.N_KEYPOINTS and np.array_equal(np.stack(g), rec.keypoints[i], equal_nan=True)
        for g, i in zip(got_cf, rows):
            st = int(rec.status[i])
            exp = [0.0] * SS.N_KEYPOINTS if (mode == "zed" and st != SS.STATUS_OK) else rec.confidence[i]
            ok &= np.array_equal(np.asarray(g, float), np.asarray(exp, float), equal_nan=True)
        nan_rec = int(np.isnan(rec.keypoints[rows]).sum())
        nan_got = int(sum(np.isnan(np.stack(g)).sum() for g in got_kp if len(g)))
        print("  round trip %-3s: %d frames replayed, %d expected, keypoints+confidence identical: %s "
              "| NaN recorded %d, replayed %d"
              % (mode, len(got_kp), len(rows), bool(ok), nan_rec if mode == "raw" else
                 int(np.isnan(rec.keypoints[rows][rec.status[rows] == SS.STATUS_OK]).sum()), nan_got))
        fails += 0 if ok else 1
    return fails == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--still", default=None, help="START:END seconds of a still window")
    ap.add_argument("--roundtrip", action="store_true")
    ap.add_argument("--controller", action="store_true",
                    help="with --roundtrip: also replay through TeleopController")
    a = ap.parse_args()
    paths = sorted({p for g in a.paths for p in glob.glob(g)})
    if not paths:
        raise SystemExit("no recordings match %s" % a.paths)
    if a.roundtrip:
        ok = all([roundtrip(p) and (controller_check(p) if a.controller else True) for p in paths])
        print("ROUND TRIP %s" % ("PASS" if ok else "FAIL"))
        raise SystemExit(0 if ok else 1)
    recs = []
    for p in paths:
        r = SS.load_recording(p)
        r.meta["path"] = p
        recs.append(r)
        summary(r)
    coast = C.SmoothingConfig().max_coast_frames
    still = tuple(float(x) for x in a.still.split(":")) if a.still else None
    conf_table(recs)
    dropout_report(recs, coast)
    jitter_report(recs, still)
    limb_report(recs)


if __name__ == "__main__":
    main()
