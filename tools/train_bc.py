"""Train BC: the overfit-10 gate, and the full train/val plumbing check.

    python tools/train_bc.py overfit10   [--epochs 300] [--chunk 1]
    python tools/train_bc.py full        [--epochs 100] [--chunk 1]
    python tools/train_bc.py baselines
    python tools/train_bc.py determinism
    python tools/train_bc.py wo-curve    [--windows 1,2,4,8,16,32]

THE GATE CRITERION (revised 2026-09-21, NOTES.md)
--------------------------------------------------
A stage passes when its training error on 10 episodes falls BELOW the
NEIGHBOUR-AMBIGUITY REFERENCE computed for that stage's own loader configuration.

The criterion used to be "drive the loss to near zero". That is valid only when
the input determines the output, and here it does not: nearest neighbours in
observation space differ in action by more than the trained model's own error,
so "near zero" asked for something the data does not contain. The Stage 2 BC gate
failed it while sitting at the resolution the input supports. Both numbers and
the ratio are reported at every gate, always - a verdict without them cannot be
checked. See `g1_model/ambiguity.py` for what the reference does NOT mean.

If a stage fails, that is the finding. Do not add capacity, raise the learning
rate, or train longer to force it through - a failure says something specific
about the loss or the plumbing and forcing it through destroys the evidence.

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
import math
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
from g1_model.models import (K_PROVISIONAL, PROVISIONAL, BCConfig,
                             BCPolicy, build_bc)
from g1_model import ambiguity as AMB
from g1_model.act import ACTConfig, build_act
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


def _model_config(kind, K, batch, gate):
    """Each model STATES its own optimizer settings (TR28).

    The runner shares the training LOOP. It never shares a hyperparameter source:
    the Stage-4 ACT gate ran at lr 1e-3 because this function used to build ACT
    from a `BCConfig`, and `lr` came along silently. Now each branch constructs
    the model's own config, and `train.assert_optimizer_source` raises if the
    TrainConfig disagrees with what the model declares.

    `gate=True` means the overfit-10 gate, which is UNREGULARISED BY DESIGN for
    every model: weight decay 0 and dropout 0 for BC, weight decay 0 for ACT.
    That is stated here for each model rather than forced from outside.
    """
    if kind == "bc":
        return BCConfig(obs_window=1, chunk_size=K, batch_size=int(batch),
                        **(dict(dropout=0.0, weight_decay=0.0) if gate else {}))
    if kind == "act":
        from g1_model.act import LR as ACT_LR, WEIGHT_DECAY as ACT_WD
        # lr is ACT's published 1e-5 (README.md:77). weight_decay is the
        # reference's 1e-4 (main.py:17) EXCEPT at the gate, which is unregularised
        # for every model. Dropout stays at ACT's 0.1 (main.py:43): it is part of
        # the architecture and was 0.1 in the lr-1e-3 run too, so changing it
        # would change a second variable.
        return ACTConfig(obs_window=1, chunk_size=K, lr=ACT_LR,
                         weight_decay=0.0 if gate else ACT_WD)
    raise SystemExit("unknown --model %r (bc|act)" % kind)


def _train_config(mcfg, epochs, batch, **kw):
    """A TrainConfig whose optimizer values come from THE MODEL'S OWN config.

    `_budget`, if given, is the stated epoch budget and is merged into `notes`
    so that it is written into the run's metadata.json beside everything else.
    """
    budget = kw.pop("_budget", None)
    if budget:
        kw["notes"] = dict(kw.get("notes") or {}, **budget)
    if isinstance(mcfg, ACTConfig):
        opt = mcfg.optimizer_config()
    else:
        opt = dict(lr=mcfg.lr, weight_decay=mcfg.weight_decay,
                   optimizer=mcfg.optimizer)
    return T.TrainConfig(epochs=None if epochs is None else int(epochs),
                         batch_size=int(batch), **opt, **kw)


def _make_model(mcfg, tcfg):
    """Build through `seeded_build` (D10): seeding inside `train()` is already
    too late to control weight initialisation."""
    if isinstance(mcfg, ACTConfig):
        return T.seeded_build(tcfg, build_act, cfg=mcfg)
    return T.seeded_build(tcfg, build_bc, cfg=mcfg)


def _budget(a, ds):
    """The epoch budget, stated: epochs, the optimizer steps they amount to, and
    why. Steps are recorded because they - not epochs - are what compares across
    batch sizes: at batch 8 an epoch of 7,628 samples is 954 steps, at batch 256
    it is 30, so "300 epochs" and "12 epochs" are 9,000 and 11,448 steps."""
    spe = int(math.ceil(len(ds) / float(a.batch)))
    if getattr(a, "max_steps", None):
        # Step-budgeted: the comparable unit. The epoch count is reported only as
        # a derived, dataset-dependent figure.
        return dict(budget_unit="optimizer_steps", max_steps=int(a.max_steps),
                    steps_per_epoch=spe,
                    max_steps_in_epochs=round(int(a.max_steps) / spe, 2),
                    budget_reason=str(a.budget_reason))
    return dict(epoch_budget=int(a.epochs), steps_per_epoch=spe,
                optimizer_steps=spe * int(a.epochs),
                epoch_budget_reason=str(a.budget_reason))


def _step_kwargs(a):
    """TrainConfig fields for a step-budgeted run, or {} for an epoch run."""
    if not getattr(a, "max_steps", None):
        return {}
    rule = None
    if a.stop_patience:
        rule = T.StopRule(patience_windows=int(a.stop_patience),
                          min_rel_improvement=float(a.stop_rel))
    return dict(max_steps=int(a.max_steps), window_steps=int(a.window_steps),
                stop_rule=rule)


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
    bc = BCConfig(obs_window=1, chunk_size=1)
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
    bc = BCConfig(obs_window=1, chunk_size=1)
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
    rec = res["final_recon_l1"]
    print("\n%s" % label)
    print("  optimizer steps   %d   (%.2f steps/s)" % (res["optimizer_steps"],
                                                      res["steps_per_second"] or 0))
    print("  wall              %.1f s" % res["wall_seconds"])
    if res.get("stop"):
        print("  stopped by        %s at step %d" % (res["stop"]["reason"],
                                                     res["stop"]["step"]))
    print("  final recon_l1    %.6f   (the gated quantity)" % rec)
    print("  final total loss  %.6f   (recon + any KL term; NOT gated)"
          % res["final_train_loss"])
    if res["final_val_loss"] is not None:
        print("  final val loss    %.6f" % res["final_val_loss"])
    # TR29: the baselines are reconstruction quantities, so they are compared
    # against the reconstruction error, never the total loss.
    print("  reconstruction as a FRACTION of the baselines:")
    print("    of ZERO  %.5f  (%.1fx better)"
          % (rec / base["zero"], base["zero"] / max(rec, 1e-12)))
    print("    of COPY  %.5f  (%.1fx better)"
          % (rec / base["copy"], base["copy"] / max(rec, 1e-12)))
    print("  run dir           %s" % repo_relpath(res["run_dir"]))


def cmd_overfit10(a):
    """A2/C4: ten episodes, no val, no regularization, no augmentation.

    K=1 is BC; K>1 is chunked BC. SAME class, same loss, same loop, same
    optimizer - the only difference is how many actions come out of one
    observation, which is what Stage 3 isolates. The reference is recomputed for
    whichever configuration is gated, because both the observation space and the
    prediction target changed.
    """
    K = int(a.chunk)
    mcfg = _model_config(a.model, K, a.batch, gate=True)
    bc = BCConfig(obs_window=1, chunk_size=K)       # loader shape only; W_o, K
    tr, _ = _splits()
    seeds = tr[:10]
    st, ac, meta = DS.load_norm_stats()
    ds = _load(seeds, bc, meta, norm_fit_seeds=tr)
    name = ("ACT (K=%d)" % K if a.model == "act"
            else "BC" if K == 1 else "CHUNKED BC (K=%d)" % K)
    _banner(ds, "OVERFIT-10 GATE: %s on %d episodes, seeds %s"
            % (name, len(seeds), seeds))
    if K > 1:
        print("  K=%d is PROVISIONAL and was NOT swept: %d ticks = %.1f s at "
              "25 Hz, the value ACT uses." % (K, K, K / 25.0))
        print("  No temporal ensembling (C2) - that is Stage 4. Deployment "
              "takes the chunk's first action.")

    print("")
    print("  computing the neighbour-ambiguity reference for W_o=%d, K=%d ..."
          % (bc.obs_window, K))
    ref = AMB.neighbour_ambiguity(ds, device=T.select_device())
    print("  %s" % ref.cite())

    cfg = _train_config(
        mcfg, a.epochs, a.batch, seed=0,
        run_name="%s_overfit10_K%d" % (a.model, K),
        log_every=max(1, int(a.epochs) // 20) if a.epochs else 1,
        _budget=_budget(a, ds), **_step_kwargs(a),
        notes=dict(gate="overfit-10", chunk_size=K, model=a.model,
                   **(dict(
                       kl_weight_used=10.0,
                       kl_weight_reference="reference/act README.md:76 --kl_weight 10; "
                                           "paper Table III beta 10",
                       kl_weight_balance_matched=14.70,
                       kl_weight_note=(
                           "Our masked_l1 reduces over CONTRIBUTING elements (16 of 22 "
                           "dims, padding excluded); the reference divides by the full "
                           "element count (policy.py:31). MEASURED 2026-09-21 on 7,628 "
                           "real samples at K=100: our reconstruction term is 1.4704x "
                           "the reference's scale, so preserving the reference's L1:KL "
                           "balance needs beta'=14.70. beta=10 is used here because the "
                           "overfit-10 gate cannot discriminate between them; the choice "
                           "is DEFERRED to a validation signal. NOT rescaled silently."),
                   ) if a.model == "act" else {}),
                   criterion="train error < neighbour-ambiguity reference for "
                             "this loader configuration (NOTES.md 2026-09-21)",
                   ambiguity_reference=ref.as_metadata(),
                   provisional_hyperparameters=PROVISIONAL,
                   data_caveat=T.dataset_provenance(ds)["caveat"],
                   regularization="NONE: dropout 0, weight_decay 0, no augmentation"))
    model = _make_model(mcfg, cfg)
    res = T.train(model, ds, cfg)
    _report(res, res["baselines"], "OVERFIT-10 RESULT (%s)" % name)

    # TR29: the reference is a RECONSTRUCTION quantity, so the gate scores the
    # reconstruction error train() labelled at the source - never the total loss,
    # which for ACT includes beta*KL. gate() refuses anything else.
    verdict = AMB.gate(res["quantities"]["recon_l1"], ref)
    print("")
    print("GATE (neighbour-ambiguity criterion)")
    print(verdict.render())
    with open(os.path.join(res["run_dir"], "gate.json"), "w",
              encoding="utf-8") as fh:
        json.dump(dict(passed=verdict.passed, train_error=verdict.train_error,
                       train_error_units=res["quantities"]["recon_l1"].units,
                       total_loss=res["final_train_loss"],
                       optimizer_steps=res["optimizer_steps"],
                       steps_per_second=res["steps_per_second"],
                       stop=res["stop"],
                       reference=verdict.reference, ratio=verdict.ratio,
                       detail=ref.as_metadata()), fh, indent=1, default=str)
    print("  This gate ran on SCRIPTED data and must be re-run on piloted data "
          "before it is cited.")
    return 0 if verdict.passed else 1


def cmd_full(a):
    """Part B: the 32/8 split, deferred from Stage 2. A PLUMBING CHECK."""
    K = int(a.chunk)
    mcfg = _model_config(a.model, K, a.batch, gate=False)
    bc = BCConfig(obs_window=1, chunk_size=K)       # loader shape only; W_o, K
    tr, va = _splits()
    st, ac, meta = DS.load_norm_stats()
    tds = _load(tr, bc, meta)
    vds = _load(va, bc, meta, norm_fit_seeds=tr)
    assert not (set(tds.seeds) & set(vds.seeds)), "train and val share a seed"
    _banner(tds, "FULL SPLIT: %d train episodes, %d val episodes (K=%d)"
            % (len(tds.lengths), len(vds.lengths), K))
    print("  THIS IS A PLUMBING CHECK, NOT A RESULT. The val number says the "
          "pipeline runs end to end.")
    print("  It is not evidence about BC: the hyperparameters are untuned and "
          "the data is scripted.")

    print("")
    print("  neighbour-ambiguity reference, computed on each split separately:")
    dev = T.select_device()
    ref_tr = AMB.neighbour_ambiguity(tds, device=dev)
    ref_va = AMB.neighbour_ambiguity(vds, device=dev)
    print("    train %s" % ref_tr.cite())
    print("    val   %s" % ref_va.cite())
    print("    (the two differ because the reference depends on data DENSITY - "
          "32 episodes vs 8 - so they are not interchangeable.)")

    cfg = _train_config(
        mcfg, a.epochs, a.batch, seed=0,
        run_name="%s_full_K%d" % (a.model, K),
        log_every=max(1, int(a.epochs) // 20) if a.epochs else 1,
        _budget=_budget(a, tds), **_step_kwargs(a),
        notes=dict(gate="full-split plumbing check",
                   is_a_result=False,
                   chunk_size=K,
                   what_this_is="A PLUMBING CHECK ON SCRIPTED DATA. It shows "
                                "the pipeline carries water end to end. It is "
                                "NOT evidence about BC and no conclusion may be "
                                "drawn from the validation number.",
                   ambiguity_reference_train=ref_tr.as_metadata(),
                   ambiguity_reference_val=ref_va.as_metadata(),
                   provisional_hyperparameters=PROVISIONAL,
                   data_caveat=T.dataset_provenance(tds)["caveat"]))
    model = _make_model(mcfg, cfg)
    res = T.train(model, tds, cfg, val_ds=vds)
    _report(res, res["baselines"], "FULL-SPLIT RESULT (plumbing only)")
    print("  against the neighbour-ambiguity references:")
    print("    train %.6f / %.6f = %.4f   (reconstruction; TR29)"
          % (res["final_recon_l1"], ref_tr.mean,
             res["final_recon_l1"] / ref_tr.mean))
    print("    val   %.6f / %.6f = %.4f"
          % (res["final_val_loss"], ref_va.mean,
             res["final_val_loss"] / ref_va.mean))
    print("")
    print("  Draw no conclusion from the val number. Untuned hyperparameters, "
          "scripted data.")
    return 0


def cmd_wo_curve(a):
    """Part D: neighbour ambiguity as a function of obs_window. NO MODEL."""
    windows = [int(w) for w in str(a.windows).split(",") if w.strip()]
    tr, _ = _splits()
    st, ac, meta = DS.load_norm_stats()
    print("=" * 78)
    print("W_o SELECTION INSTRUMENT: neighbour ambiguity vs observation window")
    print("  NO MODEL IS INVOLVED. This is a measurement of the DATA, so it "
          "cannot be")
    print("  confounded by architecture - which is the point: W_o must be "
          "chosen once,")
    print("  identically for ACT and ACT-LSTM, and justified by something "
          "other than a sweep.")
    print("  DATA: %d SCRIPTED episodes. THE CURVE BELOW CANNOT CHOOSE W_o - "
          "see the report." % len(tr))
    print("=" * 78)
    rows = AMB.ambiguity_curve(
        DS.SYNTHETIC, tr, windows, int(a.chunk), st, ac, meta,
        device=T.select_device(), tracking=_tracking_policy())
    print("  %4s %12s %10s %12s %8s" % ("W_o", "ambiguity", "vs W=1",
                                        "nn-dist p50", "dims"))
    first = rows[0].mean
    for r in rows:
        print("  %4d %12.6f %9.3fx %12.4f %8d"
              % (r.obs_window, r.mean, r.mean / first,
                 r.neighbour_distance["p50"], r.obs_window * spec.STATE_DIM))
    out = os.path.join(T.RUNS, "wo_curve.json")
    os.makedirs(T.RUNS, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(chunk_size=int(a.chunk), episodes=len(tr), seeds=tr,
                       source="scripted", rows=[r.as_metadata() for r in rows],
                       caveat="SCRIPTED data: the piloted action distribution "
                              "differs, so this curve validates the INSTRUMENT "
                              "and cannot choose W_o. Also read limit 6 in "
                              "g1_model/ambiguity.py before reading a knee."),
                  fh, indent=1, default=str)
    print("  wrote %s" % repo_relpath(out))
    print("")
    print("  READING IT: a FALLING region means the window explains more than "
          "the added")
    print("  sparsity costs. A RISING region means the dataset cannot populate "
          "a space")
    print("  that big - watch nn-dist p50 grow - and is NOT evidence that a "
          "longer window")
    print("  is worse. See limit 6 in g1_model/ambiguity.py.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, ep in (("overfit10", cmd_overfit10, 300),
                         ("full", cmd_full, 100),
                         ("baselines", cmd_baselines, 0),
                         ("determinism", cmd_determinism, 0),
                         ("wo-curve", cmd_wo_curve, 0)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if ep:
            # TR28's family: an epoch budget that differs across stages for an
            # UNSTATED reason. BC's gate ran 300 because 300 was the default here;
            # ACT's ran 12 because a wall-clock limit was typed on the command
            # line; neither run recorded why. So the budget has no default and
            # must come with a reason, and both go into the run's metadata.
            grp = p.add_mutually_exclusive_group(required=True)
            grp.add_argument("--epochs", type=int,
                             help="an epoch budget (dataset-dependent unit)")
            grp.add_argument("--max-steps", type=int,
                             help="an OPTIMIZER-STEP budget: the hard cap")
            p.add_argument("--window-steps", type=int, default=1000,
                           help="optimizer steps per logged window")
            p.add_argument("--stop-patience", type=int, default=0,
                           help="windows in the trailing span of the stop rule "
                                "(0 = no stop rule; run to the cap)")
            p.add_argument("--stop-rel", type=float, default=0.01,
                           help="minimum relative improvement over the span")
            p.add_argument("--budget-reason", required=True,
                           help="REQUIRED. Why this many epochs - e.g. 'identical "
                                "to the run being compared' or a wall-clock cap. "
                                "Written into metadata.json.")
        if name in ("overfit10", "full"):
            p.add_argument("--chunk", type=int, default=1,
                           help="K. 1 = BC, >1 = chunked BC (same class)")
            p.add_argument("--model", default="bc", choices=("bc", "act"))
            p.add_argument("--batch", type=int, default=256,
                           help="batch size. ACT's 4 GB ceiling is 32 (measured)")
        if name == "wo-curve":
            p.add_argument("--windows", default="1,2,4,8,16,32")
            p.add_argument("--chunk", type=int, default=1)
    a = ap.parse_args()
    raise SystemExit(a.fn(a))


if __name__ == "__main__":
    main()
