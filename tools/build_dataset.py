"""Stage episodes, score them, label their phases, split, and fit normalization.

    python tools/build_dataset.py layout
    python tools/build_dataset.py stage --from recordings/episodes --synthetic
    python tools/build_dataset.py score    [--sabotage]
    python tools/build_dataset.py phases   [--teleop FILE.npz]
    python tools/build_dataset.py split
    python tools/build_dataset.py norm --allow-synthetic
    python tools/build_dataset.py check    # everything, in order

Phase 3, second chunk. Nothing here trains anything: the model codebase, the
training loop, chunking and the deployment harness are Phase 4.

`--synthetic` stages into data/synthetic/, which is where the scripted
demonstrator's episodes belong and where they stay: they are never mixed with
real demonstrations and never enter normalization statistics without an explicit
flag that is then written into the stats file.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g1_data import dataset as DS
from g1_data import phase_label as PL
from g1_data import recorder as REC
from g1_data import spec
from g1_data import success as SU
from g1_data.ledger import EpisodeLedger
from g1_data.paths import repo_relpath


def _episodes(a):
    eps = DS.scan(DS.SYNTHETIC if a.synthetic else DS.RAW_TRAIN)
    if not eps:
        raise SystemExit("no episodes in %s - run `stage` first"
                         % (DS.SYNTHETIC if a.synthetic else DS.RAW_TRAIN))
    return eps


def cmd_layout(a):
    for d in DS.ensure_layout():
        print("  %s%s" % (repo_relpath(d),
                          "   (must stay empty during collection)"
                          if d == DS.RAW_HELDOUT else ""))
    print("SPEC_VERSION %s" % spec.SPEC_VERSION)
    return 0


def cmd_stage(a):
    """Stage, routing on `meta["source"]` and EXCLUDING held-out spawns.

    TWO doors, and both were open before:

    1. PROVENANCE. The destination used to come from the operator's
       `--synthetic` flag, so forgetting it put scripted episodes into the
       training set. It now comes from `recorder.SOURCES`, which is keyed on a
       field the RECORDER writes. An episode whose source is not registered is
       REFUSED, not routed on a guess - an unknown source is an entry point
       nobody declared, so nothing here knows whether it is a demonstration.

    2. HELD-OUT SPAWN. `assert_no_leak` is an audit, and an audit that fires
       after a 150-episode run has already cost the collection. Seeds 0-44 were
       recorded before the partitioner existed and 5 spawn in the held-out
       patch; those are quarantined here, at the door.
    """
    DS.ensure_layout()
    n, held, by_dst = 0, [], {}
    for f in sorted(os.listdir(a.source)):
        if not f.endswith(".npz"):
            continue
        src = os.path.join(a.source, f)
        meta = DS.read_meta(src)
        # Raises UnknownSource, naming the file and the source string.
        info = REC.source_info(meta.get("source"), where=f)
        dst = DS.RAW_TRAIN if info["real"] else DS.SYNTHETIC
        ep = DS.Episode(src, meta)
        if DS.heldout_leak([ep]):
            held.append(ep.seed)
            continue
        shutil.copy2(src, os.path.join(dst, f))
        by_dst.setdefault(dst, []).append(info["label"])
        n += 1
    print("staged %d episode(s), routed by meta[\"source\"]:" % n)
    for dst, labels in sorted(by_dst.items()):
        counts = {L: labels.count(L) for L in sorted(set(labels))}
        print("  %-20s %s" % (repo_relpath(dst), counts))
    if a.synthetic and DS.RAW_TRAIN in by_dst:
        raise SystemExit(
            "--synthetic was passed but %d episode(s) are registered as REAL "
            "demonstrations and went to data/raw/train. The flag does not "
            "route any more; fix the invocation or the registry."
            % len(by_dst[DS.RAW_TRAIN]))
    if held:
        print("  QUARANTINED %d episode(s) whose spawn is in the HELD-OUT patch: "
              "seeds %s" % (len(held), held))
        print("  they were recorded from an unpartitioned seed stream; "
              "`partition` now prevents this before collection, not after.")
    for dst in sorted(by_dst):
        eps = DS.scan(dst)
        print("uniform spec version and contact contract: %d episode(s) pass in %s"
              % (len(eps), repo_relpath(dst)))
    return 0


def cmd_score(a):
    eps = _episodes(a)
    results, disagree = [], []
    for e in eps:
        arrays, meta = e.load()
        r = SU.evaluate(arrays, meta)
        results.append(r)
        # The demonstrator's own verdict is the cross-check: it is computed live,
        # by different code, from the stepped model.
        if bool(meta["outcome"]["ok"]) != bool(r["episode_success"]):
            disagree.append((e.seed, meta["outcome"]["ok"], r))
    s = SU.summarize(results)
    print("scored %d episodes from %s" % (len(eps), "data/synthetic" if a.synthetic
                                          else "data/raw/train"))
    for k in ("grasp", "lift", "transport", "place"):
        ok, n = s[k]
        print("  %-10s %3d/%-3d" % (k, ok, n))
    print("  failures by bucket: %s" % (s["failures_by_bucket"] or "none"))
    print("agreement with the demonstrator's own result: %d/%d"
          % (len(eps) - len(disagree), len(eps)))
    for seed, ok, r in disagree:
        print("   seed %d: demonstrator ok=%s, detector %s"
              % (seed, ok, {k: r[k] for k in ("grasp", "lift", "transport", "place")}))
    if a.sabotage:
        print("\ndeliberately sabotaged episodes (fault injected into seed %d):"
              % eps[0].seed)
        arrays, meta = eps[0].load()
        bad = 0
        for kind in SU.SABOTAGE_KINDS:
            sa, sm = SU.sabotage(arrays, meta, kind)
            r = SU.evaluate(sa, sm)
            want = SU.SABOTAGE_EXPECT[kind]
            caught = (not r[want]) and not r["episode_success"]
            bad += 0 if caught else 1
            print("  %-14s -> grasp %-5s lift %-5s transport %-5s place %-5s | "
                  "expected %-9s to fail: %s"
                  % (kind, r["grasp"], r["lift"], r["transport"], r["place"],
                     want, "CAUGHT" if caught else "MISSED"))
        print("  %d/%d injected faults caught" % (len(SU.SABOTAGE_KINDS) - bad,
                                                  len(SU.SABOTAGE_KINDS)))
        if bad:
            return 1
    return 1 if disagree else 0


def cmd_phases(a):
    if a.teleop:
        arrays, meta = DS.load_episode(a.teleop) if hasattr(DS, "load_episode") else (
            __import__("g1_data.recorder", fromlist=["load_episode"]).load_episode(a.teleop))
        labels, diag = PL.label_episode(arrays, meta)
        names = [spec.Phase(int(v)).name for v in labels]
        seq, run = [], None
        for nm in names:
            if nm != run:
                seq.append(nm)
                run = nm
        print("teleop episode %s" % os.path.basename(a.teleop))
        print("  derived sequence : %s" % " -> ".join(seq))
        print("  monotone?        : %s (no phase is entered twice)"
              % (len(seq) == len(set(seq))))
        print("  transitions      : %d, hysteresis corrected %d tick(s)"
              % (diag["transitions"], diag["corrected_by_monotone"]))
        print("  boundaries (tick): %s" % diag["boundaries"])
        print("  weld engage %s, release %s of %d ticks"
              % (diag["engage_tick"], diag["release_tick"], diag["ticks"]))
        return 0
    eps = _episodes(a)
    total = None
    offsets, present = {}, None
    agree = bagree = tot = 0
    for e in eps:
        arrays, meta = e.load()
        truth = np.asarray(arrays["phase_labels"], dtype=int)
        derived, diag = PL.label_episode(arrays, meta)
        m, present = PL.confusion(truth, derived)
        total = m if total is None else total + m
        agree += int((truth == np.asarray(derived, dtype=int)).sum())
        bagree += int((spec.scored_phase(truth.astype(np.int8))
                       == spec.scored_phase(np.asarray(derived, dtype=np.int8))).sum())
        tot += len(truth)
        for k, v in PL.boundary_offsets(truth, derived).items():
            offsets.setdefault(k, []).append(v)
    names = [spec.Phase(v).name[:6] for v in present]
    print("confusion over %d episodes, %d ticks (rows = demonstrator truth, "
          "columns = derived)" % (len(eps), tot))
    print("            " + " ".join("%7s" % n for n in names))
    for i, n in enumerate(names):
        print("  %-10s" % n + " ".join("%7d" % v for v in total[i]))
    print("\nper-tick agreement, all 10 phases      : %d/%d = %.1f%%"
          % (agree, tot, 100.0 * agree / tot))
    print("per-tick agreement, the 5 SCORED buckets: %d/%d = %.1f%%"
          % (bagree, tot, 100.0 * bagree / tot))
    print("  the gap is REACH / REPOSITION / APPROACH, which are not separable "
          "from state and all map to ScoredPhase.GRASP - the taxonomy, which is "
          "what labels are for, is unaffected.")
    print("boundary offset, derived minus truth, in ticks (25 Hz):")
    for k, v in offsets.items():
        print("  %-10s n=%2d  min %+d  median %+d  max %+d"
              % (k, len(v), min(v), int(np.median(v)), max(v)))
    return 0


def cmd_partition(a):
    part = DS.partition_seeds()
    paths = DS.save_partition(part)
    leaked = set(part["train"]) & set(part["exp2"])
    print("pre-partitioned seed streams (before any collection):")
    print("  train %d seeds, e.g. %s" % (len(part["train"]), part["train"][:8]))
    print("  exp1  %d seeds, e.g. %s  (disjoint from train, D16)"
          % (len(part["exp1"]), part["exp1"][:8]))
    print("  exp2  %d seeds, e.g. %s  (the held-out patch)"
          % (len(part["exp2"]), part["exp2"][:8]))
    print("  train n exp2 = %s" % (leaked or "empty - the guarantee, by construction"))
    for p in paths:
        print("  wrote %s" % repo_relpath(p))
    return 0


def cmd_split(a):
    eps = _episodes(a)
    DS.assert_uniform(eps)
    DS.assert_no_leak(eps)
    led = EpisodeLedger(a.ledger)
    cov = DS.coverage(led, eps)
    print("coverage: %d accepted in the ledger, %d on disk" %
          (cov["accepted_in_ledger"], cov["on_disk"]))
    print("  missing from disk : %s" % (cov["missing_from_disk"] or "none"))
    print("  not in the ledger : %s" % (cov["not_in_ledger"] or "none"))
    print("  rejected          : %s" % (cov["rejected"] or "none"))
    print("  spawn x %s  y %s" % (cov["spawn_x"], cov["spawn_y"]))
    print("  held-out leak     : %s" % (cov["heldout_accepted"] or "NONE - good"))
    sp = DS.stratified_split(eps, val_frac=a.val_frac, grid=(a.grid, a.grid))
    DS.save_splits(sp)
    occupied = {k: len(v) for k, v in sp["bins"].items()}
    print("split: %d train / %d val over %d occupied bins of %d"
          % (len(sp["train"]), len(sp["val"]), len(occupied), a.grid ** 2))
    print("  per-bin counts: %s" % occupied)
    print("  bins with no val episode: %s  (8 val cannot cover 9 bins; at 150 "
          "episodes every bin draws ~3)" % (sp["bins_without_val"] or "none"))
    print("  val seeds: %s" % sp["val"])
    print("  saved %s" % repo_relpath(DS.SPLITS))
    return 0


def cmd_norm(a):
    eps = _episodes(a)
    sp = DS.load_splits() if os.path.exists(DS.SPLITS) else DS.stratified_split(eps)
    train = [e for e in eps if e.seed in set(sp["train"])]
    st, ac, rep = DS.fit_norm_stats(train, "train",
                                    allow_synthetic=a.allow_synthetic)
    DS.save_norm_stats(st, ac, source=rep["source"], n_episodes=rep["n_episodes"],
                       seeds=rep["seeds"], split=rep["split"])
    print("fitted on %d TRAIN episodes (%d ticks), source=%s"
          % (rep["n_episodes"], rep["n_ticks"], rep["source"]))
    print("saved %s" % repo_relpath(DS.NORM_STATS))

    st2, ac2, meta = DS.load_norm_stats()
    print("reloaded: spec %s, split %s, source %s" % (meta["spec_version"],
                                                      meta["split"], meta["source"]))
    print("provenance: %d seed(s) %s%s, fitted %s, commit %s"
          % (len(meta["seeds"]), meta["seeds"][:8],
             " ..." if len(meta["seeds"]) > 8 else "",
             meta["fitted_utc"], meta["git_commit"][:12]))
    # The provenance is only worth writing if something reads it back.
    DS.assert_norm_stats_match(meta, train, where=os.path.basename(DS.NORM_STATS))
    print("  seed provenance matches the %d episode(s) just fitted on" % len(train))
    # round trip
    arrays, _ = train[0].load()
    S = np.asarray(arrays["states"], dtype=np.float64)
    A = np.asarray(arrays["actions"], dtype=np.float64)
    ds = float(np.abs(st2.denormalize(st2.normalize(S)) - S).max())
    da = float(np.abs(ac2.denormalize(ac2.normalize(A)) - A).max())
    print("round trip normalize->denormalize: max |error| state %.3e, action %.3e"
          % (ds, da))

    print("\nmasked ACTION dims (never touched): %s" % list(spec.CONSTANT_ACTION_DIMS))
    bad = [i for i in spec.CONSTANT_ACTION_DIMS
           if ac2.mean[i] != 0.0 or ac2.std[i] != 1.0]
    print("  mean 0 / std 1 exactly: %s" % (not bad))

    print("\nsmallest unmasked standard deviations (the z-scoring risk):")
    for kind, stats, raw, names, mask in (
            ("state", st2, rep["state_raw_std"], spec.STATE_NAMES, spec.STATE_MASK),
            ("action", ac2, rep["action_raw_std"], spec.ACTION_NAMES, spec.ACTION_MASK)):
        idx = [i for i in np.argsort(raw) if mask[i]][:5]
        for i in idx:
            print("  %-6s dim %2d %-24s raw std %.3e -> used %.3e"
                  % (kind, i, names[i], raw[i], stats.std[i]))
    print("  floored to STD_FLOOR: state %s, action %s"
          % (rep["state_floored"] or "none", rep["action_floored"] or "none"))

    # The weld bit is binary (D14), so say what normalizing it does.
    for i in (spec.GRIP_L, spec.GRIP_R):
        print("  state dim %d (%s) is the BINARY weld bit: mean %.3f, std %.3f -> "
              "z-scores %.2f (released) and %.2f (welded)"
              % (i, spec.STATE_NAMES[i], st2.mean[i], st2.std[i],
                 float(st2.normalize(np.zeros(spec.STATE_DIM))[i]),
                 float(st2.normalize(np.eye(spec.STATE_DIM)[i])[i])))
    return 0


def cmd_check(a):
    rc = 0
    for name, fn in (("layout", cmd_layout), ("partition", cmd_partition),
                     ("score", cmd_score), ("phases", cmd_phases),
                     ("split", cmd_split), ("norm", cmd_norm)):
        print("\n================ %s ================" % name)
        rc |= fn(a)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("layout", cmd_layout), ("partition", cmd_partition),
                     ("stage", cmd_stage), ("score", cmd_score),
                     ("phases", cmd_phases), ("split", cmd_split), ("norm", cmd_norm),
                     ("check", cmd_check)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--synthetic", action="store_true",
                       help="operate on data/synthetic (the scripted demonstrator)")
        p.add_argument("--ledger", default=os.path.join(ROOT, "recordings", "episodes",
                                                        "ledger.jsonl"))
        if name in ("stage",):
            p.add_argument("--from", dest="source",
                           default=os.path.join(ROOT, "recordings", "episodes"))
        if name in ("score", "check"):
            p.add_argument("--sabotage", action="store_true",
                           help="also inject faults and check the detector catches them")
        if name in ("phases", "check"):
            p.add_argument("--teleop", default=None,
                           help="label one teleop episode (no ground truth exists)")
        if name in ("split", "check"):
            p.add_argument("--val-frac", type=float, default=0.2)
            p.add_argument("--grid", type=int, default=3)
        if name in ("norm", "check"):
            p.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args()
    for k, v in (("sabotage", False), ("teleop", None), ("val_frac", 0.2),
                 ("grid", 3), ("allow_synthetic", False), ("source", None)):
        if not hasattr(a, k):
            setattr(a, k, v)
    raise SystemExit(a.fn(a))


if __name__ == "__main__":
    main()
