"""Regression verdict: adopted B-prime vs the stripped (pre-adoption) baseline.

    python tools/bprime_compare.py lock 12
    python tools/bprime_compare.py pred 40

CRITERIA, fixed before the adopted results were read (2026-09-15). Comparisons
are STRICT - no tolerance - because the instruction is to revert on a regression
in ANY measure. A REGRESSION is any of:

  R1  fewer passing seeds
  R2  any seed that passes in baseline and fails adopted
  R3  a failure category appears that baseline does not have: fall, wedge,
      lock fallback, placement > 0.10 m, palm error > 45 mm, not resting
  R4  an ENVELOPE worsens past baseline: max palm error, max placement error,
      max carry drift, max base pitch, max box tilt, max samples, mean samples
  R5  box sinks deeper than baseline: min unwelded corner gap, min gap overall
  R6  pre-grasp box displacement exceeds baseline's max
  R7  hand <-> GOAL platform contact missing in any seed (load-bearing at release)

Per-seed deltas are printed for transparency. A seed moving inside the
baseline envelope is not by itself a regression: the exclusion is expected to
change the episodes in which the hands touched the slab.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "docs", "measurements")


def load(tag, lock, n):
    return json.load(open(os.path.join(M, "bprime_gates_%s_%s_%d.json" % (tag, lock, n))))


def main():
    lock, n = sys.argv[1], int(sys.argv[2])
    b, a = load("base", lock, n), load("adopted", lock, n)
    assert [x["seed"] for x in b] == [x["seed"] for x in a] == list(range(n))
    reg = []

    ok_b, ok_a = sum(x["ok"] for x in b), sum(x["ok"] for x in a)
    if ok_a < ok_b:
        reg.append("R1 pass count %d < %d" % (ok_a, ok_b))
    for xb, xa in zip(b, a):
        if xb["ok"] and not xa["ok"]:
            reg.append("R2 seed %d passes in baseline, fails adopted (%s)" % (xb["seed"], xa["fail"]))

    def count(rows, f):
        return sum(1 for x in rows if f(x))
    cats = {"fall": lambda x: x["fell"], "wedge": lambda x: x["wedged"],
            "fallback": lambda x: x["fallbacks"] > 0,
            "place>0.10": lambda x: x["place"] > 0.10,
            "palm>45": lambda x: x["palm"] > 45.0,
            "not resting": lambda x: not x["resting"]}
    for k, f in cats.items():
        cb, ca = count(b, f), count(a, f)
        if ca > cb:
            reg.append("R3 %s: %d adopted vs %d baseline" % (k, ca, cb))

    env = [("palm", max), ("place", max), ("drift", max), ("pitch", max), ("tilt", max), ("n", max)]
    print("%-22s %-26s %-26s" % ("measure", "baseline min..max", "adopted min..max"))
    for k, worst in env:
        vb, va = [x[k] for x in b], [x[k] for x in a]
        print("%-22s %10.4f .. %10.4f   %10.4f .. %10.4f" % (k, min(vb), max(vb), min(va), max(va)))
        if max(va) > max(vb):
            reg.append("R4 max %s %.6f > baseline %.6f" % (k, max(va), max(vb)))
    mb, ma = np.mean([x["n"] for x in b]), np.mean([x["n"] for x in a])
    print("%-22s %10.2f                 %10.2f" % ("mean samples", mb, ma))
    if ma > mb:
        reg.append("R4 mean samples %.2f > %.2f" % (ma, mb))

    for k in ("box_sink_unwelded_mm", "box_sink_mm"):
        vb, va = [x[k] for x in b], [x[k] for x in a]
        print("%-22s %10.4f .. %10.4f   %10.4f .. %10.4f" % (k, min(vb), max(vb), min(va), max(va)))
        if min(va) < min(vb):
            reg.append("R5 min %s %.4f < baseline %.4f" % (k, min(va), min(vb)))
    vb, va = [x["box_disp_mm"] for x in b], [x["box_disp_mm"] for x in a]
    print("%-22s %10.4f .. %10.4f   %10.4f .. %10.4f" % ("pre-grasp box disp mm", min(vb), max(vb), min(va), max(va)))
    if max(va) > max(vb):
        reg.append("R6 pre-grasp box displacement %.4f > baseline %.4f mm" % (max(va), max(vb)))
    miss = [x["seed"] for x in a if x["hand_goal_steps"] == 0]
    if miss:
        reg.append("R7 no hand<->goal contact in seeds %s" % miss)

    print("\ncount: baseline %d/%d, adopted %d/%d" % (ok_b, n, ok_a, n))
    print("\nper-seed (adopted - baseline):")
    print("seed  d_palm_mm  d_place_mm  d_drift  d_pitch  d_tilt  d_n  | hand-pickup steps b->a | "
          "hand-goal b/a | hand-hand b/a | pass mm (a) s | sink_unw b/a")
    for xb, xa in zip(b, a):
        hb = sum(v for k, v in xb["robot_pickup_bodies"].items() if "wrist" in k or "pad" in k)
        ha = sum(v for k, v in xa["robot_pickup_bodies"].items() if "wrist" in k or "pad" in k)
        print("%4d  %+9.4f  %+10.2f  %+7.3f  %+7.3f  %+6.2f  %+4d | %5d -> %5d | %4d/%4d | %3d/%3d | %5.1f %.2f | %+.2f/%+.2f"
              % (xb["seed"], xa["palm"] - xb["palm"], 1000 * (xa["place"] - xb["place"]),
                 xa["drift"] - xb["drift"], xa["pitch"] - xb["pitch"], xa["tilt"] - xb["tilt"],
                 xa["n"] - xb["n"], hb, ha, xb["hand_goal_steps"], xa["hand_goal_steps"],
                 xb["hand_hand_steps"], xa["hand_hand_steps"], xa["pass_depth_mm"], xa["pass_s"],
                 xb["box_sink_unwelded_mm"], xa["box_sink_unwelded_mm"]))
    other = {}
    for x in a:
        for k, v in x["robot_pickup_bodies"].items():
            other[k] = other.get(k, 0) + v
    print("\nadopted robot<->pickup contact bodies (all seeds):", other)
    print("\nREGRESSIONS:" if reg else "\nREGRESSIONS: none")
    for r in reg:
        print("  " + r)
    return 1 if reg else 0


if __name__ == "__main__":
    raise SystemExit(main())
