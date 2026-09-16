"""Record ONE teleop episode from the headless synthetic fixture, for the labeller.

    python tools/record_teleop_episode.py --seconds 22 --out recordings/teleop_ep.npz

Phase 3, second chunk. This exists for one reason: the phase labeller has to be
run on a teleop episode, where there is NO ground-truth phase column, to see
whether the derived sequence is plausible and monotone. The scripted
demonstrator cannot answer that - it has a phase machine, which teleop does not.

`run_integrated_combined.py` is not modified: `TeleopRecorder` wraps `mj_step` and
reads `main`'s frame, exactly as the scripted recorder reads `run_episode`'s. The
fixture is the same headless synthetic grasp-and-lift used since 2026-09-15, so
this needs no camera and no operator.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g1_data import spec
from g1_data.ledger import EpisodeLedger
from g1_data.recorder import (SOURCE_TELEOP_FIXTURE, TeleopRecorder,
                             assert_namespace, label_of, ledger_for)


def _rel(path: str) -> str:
    """Repo-relative like the scripted ledger writes, absolute if that is not
    expressible - `--out` may legitimately sit on another drive."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return os.path.abspath(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=22.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--standoff", type=float, default=0.32)
    ap.add_argument("--out", default=os.path.join(ROOT, "recordings", "teleop_ep.npz"))
    a = ap.parse_args()

    # A3: the ledger, in the SAME schema the scripted recorder writes. Before
    # this, a piloted episode left no ledger row at all, so `coverage` could not
    # see it and nothing recorded that the session had happened. The field names
    # below are copied from tools/record_episodes.py, not invented: session
    # (label, seeds, auto, spec_version), issue (seed, label), accept (seed,
    # path, heldout, label, checks, placement_error).
    label = label_of(SOURCE_TELEOP_FIXTURE, where="record_teleop_episode")
    # A2: derived from --out, never a separate argument (see recorder.ledger_for).
    assert_namespace(os.path.dirname(os.path.abspath(a.out)),
                     SOURCE_TELEOP_FIXTURE, where="record_teleop_episode --out")
    ledger = EpisodeLedger(ledger_for(os.path.dirname(os.path.abspath(a.out))))
    ledger.append("session", label=label, seeds=[a.seed], auto="pass",
                  spec_version=spec.SPEC_VERSION)
    ledger.append("issue", seed=a.seed, label=label)

    import run_integrated_combined as RIC
    argv = ["run_integrated_combined.py", "keyboard", str(a.seed), "--synthetic",
            "grasp", "--headless", "--start-standoff", str(a.standoff),
            "--seconds", str(a.seconds)]
    old = sys.argv
    sys.argv = argv
    try:
        with TeleopRecorder() as rec:
            try:
                RIC.main()
            except SystemExit:
                pass
    finally:
        sys.argv = old

    if rec.buf is None or not len(rec.buf):
        ledger.append("error", seed=a.seed, reason="no ticks recorded")
        raise SystemExit("nothing was recorded - did the loop run?")
    meta = dict(
        seed=a.seed, source="run_integrated_combined --synthetic grasp",
        box_spawn_xy=[float(v) for v in rec.buf.states[0][0:2]],
        heldout=False, standoff_cmd=a.standoff, lateral_cmd=float("nan"),
        weld_engage_tick=rec.weld_engage_tick, weld_release_tick=rec.weld_release_tick,
        lock_engage_tick=rec.lock_engage_tick, lock_release_tick=rec.lock_release_tick,
        box_pre_grasp_disturb_mm=rec.box_pre_grasp_disturb_mm,
        reset_fingerprint="",            # this entry point does not reset_episode
        contact_contract=__import__("g1_teleop.contact_contract",
                                    fromlist=["contract_of"]).contract_of(rec.model),
        # O28: how many ticks were NOT backed by a live tracked frame. The
        # per-tick detail is the `tracking_ok` array; this is the summary an
        # operator reads at ACCEPT time.
        degraded_ticks=sum(1 for v in rec.buf.tracking if not v),
        note="teleop: phase_labels are UNKNOWN by construction - derive them offline")
    rec.buf.save(a.out, meta)
    # `checks` mirrors the scripted recorder's shape (name -> bool). The fixture
    # runs no placement, so `placement_error` is null rather than a number that
    # would read as a measured 0.0 m placement.
    ledger.append("accept", seed=a.seed, path=_rel(a.out),
                  heldout=bool(meta["heldout"]), label=label,
                  checks={"weld fired": rec.weld_engage_tick >= 0,
                          "ticks recorded": len(rec.buf) > 0,
                          "tracking never lost": meta["degraded_ticks"] == 0},
                  placement_error=None)
    print("saved %s: %d ticks, weld tick %s, lock tick %s"
          % (a.out, len(rec.buf), rec.weld_engage_tick, rec.lock_engage_tick))
    print("ledger %s: session/issue/accept appended, label %r"
          % (ledger.path, label))


if __name__ == "__main__":
    main()
