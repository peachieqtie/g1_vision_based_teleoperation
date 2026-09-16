"""O28: tracking dropout - the defect, the bound, and the per-tick marking.

Run:  python -m g1_data.test_dropout

These drive the REAL loop in `run_integrated_combined.main()` through the
headless synthetic fixture, because the defect is a property of that loop's
control flow, not of a helper that can be tested in isolation. A no-body frame
(`with_no_body`, an EMPTY keypoint list) never reaches `TeleopController.step`,
so `SmoothingConfig.max_coast_frames` - which lives inside `_coast`, on the
other side of that call - never applies. Testing the coast path instead, with
`with_dropout`'s NaN frames, is what let O28 go unnoticed: that path works.

The audit's finding was mechanisms that exist and are never called. These
tests call them.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco

import g1_teleop.synthetic_source as SS
from g1_data import recorder as REC
from g1_data.recorder import TeleopRecorder
from g1_teleop.teleop import TeleopController

FRAME_HZ = 500.0 / 17.0          # run_integrated_combined.FRAME_STEPS = 17
GAP_START_S = 3.0                # after the lock, inside the reach


def _run(gap_frames, seconds=8.0):
    """Run the fixture with a no-body gap; return (episode arrays, step calls)."""
    import run_integrated_combined as RIC

    real_frames = SS.grasp_lift_frames
    start = int(GAP_START_S * FRAME_HZ)

    def patched(*a, **k):
        frames, g_on, lift = real_frames(*a, **k)
        return SS.with_no_body(frames, [(start, gap_frames)]), g_on, lift

    calls = {"n": 0}
    real_step = TeleopController.step

    def spy(self, frame):
        calls["n"] += 1
        return real_step(self, frame)

    SS.grasp_lift_frames = patched
    TeleopController.step = spy
    argv = sys.argv
    sys.argv = ["run_integrated_combined.py", "keyboard", "0", "--synthetic",
                "grasp", "--headless", "--start-standoff", "0.32",
                "--seconds", str(seconds)]
    try:
        with TeleopRecorder() as rec:
            try:
                RIC.main()
            except SystemExit:
                pass
    finally:
        sys.argv = argv
        TeleopController.step = real_step
        SS.grasp_lift_frames = real_frames

    track = np.asarray(rec.buf.tracking, dtype=bool)
    actions = np.asarray(rec.buf.actions, dtype=np.float64)
    return track, actions, calls["n"], start


def _gap_ticks(start_frame, gap_frames):
    """Recorded tick indices (25 Hz) covered by the no-body window."""
    t0 = start_frame / FRAME_HZ
    t1 = (start_frame + gap_frames) / FRAME_HZ
    return int(t0 * 25.0), int(t1 * 25.0)


def test_no_body_frames_freeze_the_arms_and_are_marked():
    """The defect, and that the marking now catches it.

    A 50-frame gap is 1.70 s, far past NO_BODY_HOLD_S, so every tick inside it
    but for the first few must be marked NOT tracking_ok, and the recorded arm
    command must not move at all while it is held.
    """
    import run_integrated_combined as RIC
    track, actions, _, start = _run(50)
    lo, hi = _gap_ticks(start, 50)
    bound_ticks = int(RIC.NO_BODY_HOLD_S * 25.0)

    inside = track[lo + bound_ticks + 2:hi - 1]
    assert len(inside) > 5, "the window must cover enough ticks to judge"
    assert not inside.any(), (
        "ticks past the bound must be marked NOT tracking_ok, got %d/%d still "
        "marked tracked" % (int(inside.sum()), len(inside)))

    # and the arms really were frozen: zero delta across all 14 arm dims
    d = np.abs(np.diff(actions[lo + 2:hi - 1, :14], axis=0)).max(axis=1)
    assert (d == 0).all(), (
        "arm command moved during a no-body window: max delta %.2e" % d.max())


def test_ticks_outside_the_dropout_stay_tracked():
    """The guard must not mark healthy ticks - otherwise it is useless."""
    track, _, _, start = _run(50)
    lo, _ = _gap_ticks(start, 50)
    before = track[8:lo - 2]
    assert before.all(), (
        "%d/%d ticks BEFORE the dropout were wrongly marked degraded"
        % (int((~before).sum()), len(before)))


def test_a_blink_shorter_than_the_bound_is_not_marked():
    """A 3-frame blink is 0.10 s, inside NO_BODY_HOLD_S = 0.40 s.

    This is the behaviour `max_coast_frames` was meant to give and never did on
    this path: a brief tracking blink stays a valid demonstration.
    """
    import run_integrated_combined as RIC
    assert 3 / FRAME_HZ < RIC.NO_BODY_HOLD_S, "fixture must be inside the bound"
    track, _, _, start = _run(3)
    lo, hi = _gap_ticks(start, 3)
    window = track[max(lo - 2, 0):hi + 2]
    assert window.all(), (
        "a %.2f s blink was marked degraded, but the bound is %.2f s"
        % (3 / FRAME_HZ, RIC.NO_BODY_HOLD_S))


def test_controller_step_is_never_called_on_a_no_body_frame():
    """Confirms max_coast_frames cannot apply here: step is not reached.

    Compared against a clean run, a 50-frame gap must cost ~50 step calls.
    """
    _, _, calls_gap, _ = _run(50)
    _, _, calls_clean, _ = _run(0)
    missing = calls_clean - calls_gap
    assert missing >= 40, (
        "expected ~50 fewer controller.step calls with a 50-frame no-body gap, "
        "got %d (clean %d, gap %d)" % (missing, calls_clean, calls_gap))


def test_missing_array_reads_as_all_tracked():
    """Backward compatibility with the 45 episodes recorded before O28."""
    arrays = dict(states=np.zeros((7, 47), dtype=np.float32))
    got = REC.tracking_ok_of(arrays)
    assert got.shape == (7,) and got.all() and got.dtype == bool
    assert REC.tracking_ok_of(arrays, n_ticks=3).shape == (3,)
    # and a present array is returned as it stands
    arrays["tracking_ok"] = np.array([1, 0, 1], dtype=np.uint8)
    assert list(REC.tracking_ok_of(arrays)) == [True, False, True]


def test_scripted_recorder_marks_every_tick_tracked():
    """The scripted demonstrator has no camera, so `arm_stale` is not in its
    locals and every tick must record as tracked."""
    import inspect
    from g1_data.recorder import EpisodeBuffer
    sig = inspect.signature(EpisodeBuffer.tick).parameters["tracking_ok"]
    assert sig.default is True, "tracking_ok must default True for the scripted path"
    from g1_data import dataset as DS
    eps = DS.scan(DS.SYNTHETIC)
    if not eps:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    arrays, _ = eps[0].load()
    assert REC.tracking_ok_of(arrays).all()


def _tests():
    return [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]


if __name__ == "__main__":
    failed = 0
    for name, fn in _tests():
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            failed += 1
            print("FAIL  %s\n        %s: %s" % (name, type(e).__name__, e))
        else:
            print("ok    %s" % name)
    print("\n%d/%d passed" % (len(_tests()) - failed, len(_tests())))
    raise SystemExit(1 if failed else 0)
