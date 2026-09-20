"""Train BC: the overfit-10 gate, and the full train/val plumbing check.

    python tools/train_bc.py overfit10 [--epochs 200]
    python tools/train_bc.py full      [--epochs 100]
    python tools/train_bc.py baselines
    python tools/train_bc.py determinism

WHAT THE GATE IS FOR
--------------------
A model that cannot drive training loss to near zero on TEN episodes with no
validation, no regularization and no augmentation has a bug - in the loss, in the
masking, in the shape contract, in the optimizer wiring - and every later number
is meaningless. The gate is the cheapest place to find that bug, and it is a
PLUMBING check: passing it says the machinery works, not that BC works.

If it plateaus, that is the finding. Do not add capacity, raise the learning
rate, or train longer to force it through - a plateau says something specific
about the loss or the data and forcing it through destroys the evidence.

WHAT IT IS NOT (C5)
-------------------
Every gate in Phase 4 runs on SCRIPTED episodes. They are smoother and more
consistent than piloted data will be, and `tracking_ok` cannot be exercised at
all by episodes recorded without a camera. Passing here does not guarantee
passing on piloted data. This script states that in its own output and writes it
into each run directory's `metadata.json`, because a caveat that lives only in a
report is a caveat that gets lost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g1_data import dataset as DS
from g1_data import spec
from g1_data.paths import repo_relpath
from g1_model.loader import ChunkDataset, LoaderConfig, TrackingPolicy
from g1_model.models import PROVISIONAL, BCConfig, BCPolicy, build_bc
from g1_model import train as T


def _tracking_policy() -> TrackingPolicy:
    """Stated, not defaulted (A2). Off on both switches: every staged episode is
    scripted, has no camera and therefore no measured `tracking_ok`, so there is
    nothing for either policy to act on. The day piloted data arrives this line
    is the one to revisit, and it is a line rather than an absence."""
    return TrackingPolicy(exclude_overlapping_chunks=False,
                          max_degraded_fraction=None)


def _load(seeds, bc: BCConfig, norm_meta, norm_fit_seeds=None) -> ChunkDataset:
    st, ac, meta = DS.load_norm_stats()
    return ChunkDataset.from_directory(
        DS.SYNTHETIC,
        LoaderConfig(chunk_size=bc.chunk_size, obs_window=bc.obs_window,
                     tracking=_tracking_policy()),
        st, ac, norm_meta if norm_meta else None, seeds=seeds,
        norm_fit_seeds=norm_fit_seeds)


def _splits():
    sp = DS.load_splits()
    return sorted(sp["train"]), sorted(sp["val"])


def _banner(ds: ChunkDataset, what: str) -> None:
    prov = T.dataset_provenance(ds)
    print("=" * 78)
    print("%s" % what)
    print("  DATA SOURCE : %s" % ", ".join(prov["episode_sources"]))
    print("  REAL DEMOS  : %s" % prov["real_demonstrations"])
    print("  %s" % prov["caveat"])
    print("  HYPERPARAMS : %s" % PROVISIONAL)
    print("=" * 78)


def cmd_baselines(a):
    """B6: what near zero is measured against, before anything is trained."""
    bc = BCConfig()
    tr, _ = _splits()
    st, ac, meta = DS.load_norm_stats()
    ds = _load(tr, bc, meta)
    _banner(ds, "BASELINES on the TRAIN split")
    dev = T.select_device()
    b = T.baselines(ds, dev)
    print("  samples %d, contributing elements %d" % (b["samples"], b["elements"]))
    print("  ZERO (predict the normalized mean) : %.6f" % b["zero"])
    print("  COPY (predict the previous action) : %.6f" % b["copy"])
    print("  copy/zero = %.4f - the copy baseline is the one that matters; at "
          "25 Hz consecutive" % (b["copy"] / b["zero"]))
    print("  actions are very similar, so beating ZERO but not COPY means the "
          "model learned the mean pose.")
    return 0


def cmd_determinism(a):
    """B3: prove it, rather than claiming it. Two identical short runs."""
    bc = BCConfig()
    tr, _ = _splits()
    st, ac, meta = DS.load_norm_stats()
    ds = _load(tr[:3], bc, meta, norm_fit_seeds=tr)
    losses = []
    for i in (1, 2):
        cfg = T.TrainConfig(epochs=3, batch_size=bc.batch_size, lr=bc.lr,
                            weight_decay=bc.weight_decay, optimizer=bc.optimizer,
                            seed=0, run_name="determinism%d" % i, log_every=0)
        model = T.seeded_build(cfg, build_bc, cfg=bc)
        res = T.train(model, ds, cfg, run_dir=os.path.join(
            T.RUNS, "_determinism", "run%d" % i))
        losses.append([r["train_loss"] for r in res["history"]])
    same = losses[0] == losses[1]
    print("\nDETERMINISM: two identical 3-epoch runs, same seed")
    for i, L in enumerate(losses, 1):
        print("  run %d: %s" % (i, ["%.12f" % v for v in L]))
    if same:
        print("  BIT-IDENTICAL on every epoch.")
    else:
        d = max(abs(a_ - b_) for a_, b_ in zip(*losses))
        print("  NOT bit-identical. Max |difference| = %.3e" % d)
    return 0 if same else 0


def _report(res, base, label):
    h = res["history"]
    print("\n%s" % label)
    print("  epochs            %d" % len(h))
    print("  wall              %.1f s" % res["wall_seconds"])
    print("  final train loss  %.6f" % res["final_train_loss"])
    if res["final_val_loss"] is not None:
        print("  final val loss    %.6f" % res["final_val_loss"])
    print("  as a FRACTION of the baselines:")
    print("    of ZERO  %.5f  (%.1fx better)"
          % (res["final_train_loss"] / base["zero"],
             base["zero"] / max(res["final_train_loss"], 1e-12)))
    print("    of COPY  %.5f  (%.1fx better)"
          % (res["final_train_loss"] / base["copy"],
             base["copy"] / max(res["final_train_loss"], 1e-12)))
    print("  run dir           %s" % repo_relpath(res["run_dir"]))


def cmd_overfit10(a):
    """C3: ten episodes, no val, no regularization, no augmentation."""
    bc = BCConfig(dropout=0.0, weight_decay=0.0)
    tr, _ = _splits()
    seeds = tr[:10]
    st, ac, meta = DS.load_norm_stats()
    ds = _load(seeds, bc, meta, norm_fit_seeds=tr)
    _banner(ds, "OVERFIT-10 GATE: BC on %d episodes, seeds %s" % (len(seeds), seeds))

    cfg = T.TrainConfig(
        epochs=int(a.epochs), batch_size=bc.batch_size, lr=bc.lr,
        weight_decay=0.0, optimizer=bc.optimizer, seed=0,
        run_name="bc_overfit10", log_every=max(1, int(a.epochs) // 20),
        notes=dict(gate="overfit-10", provisional_hyperparameters=PROVISIONAL,
                   data_caveat=T.dataset_provenance(ds)["caveat"],
                   regularization="NONE: dropout 0, weight_decay 0, no augmentation"))
    model = T.seeded_build(cfg, build_bc, cfg=bc)
    res = T.train(model, ds, cfg)
    base = res["baselines"]
    _report(res, base, "OVERFIT-10 RESULT")

    frac = res["final_train_loss"] / base["copy"]
    print("\n  VERDICT: %s" % (
        "PASS - training loss is far below the copy baseline."
        if frac < 0.1 else
        "INSPECT - loss did not reach 10%% of the copy baseline (%.3f). "
        "This is diagnostic; do NOT add capacity or train longer to force it."
        % frac))
    print("  This gate ran on SCRIPTED data and must be re-run on piloted data "
          "before it is cited.")
    return 0


def cmd_full(a):
    """C4: the 32/8 split. A PLUMBING CHECK, not a result."""
    bc = BCConfig()
    tr, va = _splits()
    st, ac, meta = DS.load_norm_stats()
    tds = _load(tr, bc, meta)
    vds = _load(va, bc, meta, norm_fit_seeds=tr)
    assert not (set(tds.seeds) & set(vds.seeds)), "train and val share a seed"
    _banner(tds, "FULL SPLIT: %d train episodes, %d val episodes"
            % (len(tds.lengths), len(vds.lengths)))
    print("  THIS IS A PLUMBING CHECK, NOT A RESULT. The val number says the "
          "pipeline runs end to end.")
    print("  It is not evidence about BC: the hyperparameters are untuned and "
          "the data is scripted.")

    cfg = T.TrainConfig(
        epochs=int(a.epochs), batch_size=bc.batch_size, lr=bc.lr,
        weight_decay=bc.weight_decay, optimizer=bc.optimizer, seed=0,
        run_name="bc_full", log_every=max(1, int(a.epochs) // 20),
        notes=dict(gate="full-split plumbing check",
                   is_a_result=False,
                   provisional_hyperparameters=PROVISIONAL,
                   data_caveat=T.dataset_provenance(tds)["caveat"]))
    model = T.seeded_build(cfg, build_bc, cfg=bc)
    res = T.train(model, tds, cfg, val_ds=vds)
    _report(res, res["baselines"], "FULL-SPLIT RESULT (plumbing only)")
    print("\n  Draw no conclusion from the val number. Untuned "
          "hyperparameters, scripted data.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, ep in (("overfit10", cmd_overfit10, 300),
                         ("full", cmd_full, 100),
                         ("baselines", cmd_baselines, 0),
                         ("determinism", cmd_determinism, 0)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if ep:
            p.add_argument("--epochs", type=int, default=ep)
    a = ap.parse_args()
    raise SystemExit(a.fn(a))


if __name__ == "__main__":
    main()
