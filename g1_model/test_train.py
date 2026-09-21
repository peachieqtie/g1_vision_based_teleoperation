"""Tests for the shared training machinery and the BC policy.

    python -m g1_model.test_train

The loss tests are the important ones. A masking bug does not crash and does not
look wrong: it produces a loss that is a little too big or a little too small,
uniformly, so every model trained under it is consistently mis-scored and the
BC/ACT/ACT-LSTM comparison inherits the error without anyone seeing it. Each
property is therefore tested against a case constructed so that the CORRECT
answer is known exactly - zero - rather than against a number from a previous run.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_data import dataset as DS
from g1_data import spec
from g1_model import train as T
from g1_model.loader import ChunkDataset, LoaderConfig, TrackingPolicy
from g1_model.models import (K_PROVISIONAL, BCConfig, BCPolicy, build_bc,
                             first_action)


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn),
                                                  exc.__name__))


def _tiny(n_eps=2, n_ticks=40, tmp=None):
    """A small real-format dataset, for tests that must actually train."""
    src = sorted(p for p in DS._npz(DS.SYNTHETIC))
    if not src:
        return None
    tmp = tmp or tempfile.mkdtemp()
    with np.load(src[0], allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    for s in range(n_eps):
        rng = np.random.default_rng(s)
        m = dict(meta, seed=s, n_ticks=n_ticks,
                 box_spawn_xy=[1.50, -0.10 + 0.02 * s])
        np.savez_compressed(
            os.path.join(tmp, "ep_seed%04d.npz" % s),
            states=rng.normal(size=(n_ticks, spec.STATE_DIM)).astype(np.float32),
            actions=rng.normal(size=(n_ticks, spec.ACTION_DIM)).astype(np.float32),
            phase_labels=np.zeros(n_ticks, dtype=np.int8),
            gait_phase=np.zeros((n_ticks, 2), dtype=np.float32),
            qpos=np.zeros((n_ticks, 45), dtype=np.float32),
            qvel=np.zeros((n_ticks, 43), dtype=np.float32),
            step_index=(np.arange(n_ticks) * 20).astype(np.int64),
            meta=np.array(json.dumps(m)))
    st, ac = (spec.NormStats.identity("state"), spec.NormStats.identity("action"))
    ds = ChunkDataset.from_directory(
        tmp, LoaderConfig(chunk_size=1, obs_window=1, tracking=TrackingPolicy()),
        st, ac, None)
    return tmp, ds


# ─── B4: the loss ─────────────────────────────────────────────────────────────
def test_loss_ignores_padded_timesteps_entirely():
    """A fully padded sample contributes EXACTLY zero, not approximately."""
    B, K, A = 2, 5, spec.ACTION_DIM
    pred = torch.randn(B, K, A)
    target = torch.randn(B, K, A)
    pad = torch.zeros(B, K, dtype=torch.bool)          # nothing is real
    loss, n = T.masked_l1(pred, target, pad)
    assert float(loss) == 0.0, float(loss)
    assert int(n) == 0, "a fully padded batch has no contributing elements"

    # and a half-padded sample equals the same batch truncated to its real part
    pad = torch.zeros(1, K, dtype=torch.bool)
    pad[0, :3] = True
    full, _ = T.masked_l1(pred[:1], target[:1], pad)
    trunc, _ = T.masked_l1(pred[:1, :3], target[:1, :3],
                           torch.ones(1, 3, dtype=torch.bool))
    assert torch.allclose(full, trunc), (float(full), float(trunc))


def test_loss_is_exactly_zero_when_only_masked_dims_are_wrong():
    """The 6 constant dims are excluded by measurement, so being wrong on them
    costs nothing - if it costs something, the dim mask is not being applied."""
    B, K = 4, 3
    target = torch.randn(B, K, spec.ACTION_DIM)
    pred = target.clone()
    for i in spec.CONSTANT_ACTION_DIMS:
        pred[:, :, i] += 17.0                          # wildly wrong, masked
    pad = torch.ones(B, K, dtype=torch.bool)
    loss, n = T.masked_l1(pred, target, pad)
    assert float(loss) == 0.0, float(loss)
    assert int(n) == B * K * int(spec.ACTION_MASK.sum())

    # being wrong on ONE trainable dim does cost something
    pred[:, :, int(np.flatnonzero(spec.ACTION_MASK)[0])] += 1.0
    loss2, _ = T.masked_l1(pred, target, pad)
    assert float(loss2) > 0.0


def test_loss_is_normalized_by_contributing_elements_not_by_shape():
    """Otherwise the same model scores differently at K=1 and K=100 and the
    scaling comparison measures the padding."""
    A = spec.ACTION_DIM
    target = torch.zeros(1, 10, A)
    pred = torch.ones(1, 10, A)                        # error of 1 everywhere
    full = torch.ones(1, 10, dtype=torch.bool)
    half = torch.zeros(1, 10, dtype=torch.bool)
    half[0, :5] = True
    a, na = T.masked_l1(pred, target, full)
    b, nb = T.masked_l1(pred, target, half)
    assert float(a) == 1.0 and float(b) == 1.0, (float(a), float(b))
    assert int(na) == 2 * int(nb)


def test_loss_rejects_a_shape_mismatch():
    pad = torch.ones(2, 3, dtype=torch.bool)
    _raises(T.TrainError, T.masked_l1, torch.zeros(2, 3, spec.ACTION_DIM),
            torch.zeros(2, 4, spec.ACTION_DIM), pad)
    _raises(T.TrainError, T.masked_l1, torch.zeros(2, 3, 5),
            torch.zeros(2, 3, 5), pad)


# ─── B6: the baselines ────────────────────────────────────────────────────────
def test_baselines_are_computed_through_the_same_loss():
    tmp = _tiny()
    if tmp is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = tmp
    try:
        b = T.baselines(ds)
        assert b["samples"] == len(ds)
        assert b["elements"] == len(ds) * int(spec.ACTION_MASK.sum())
        # zero baseline == mean |target| over trainable dims, by definition
        A = np.concatenate(ds.actions)[:, np.asarray(spec.ACTION_MASK, dtype=bool)]
        assert abs(b["zero"] - float(np.abs(A).mean())) < 1e-5
        assert b["copy"] > 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── B3: determinism ──────────────────────────────────────────────────────────
def test_two_identical_runs_are_bit_identical():
    """B3, proved rather than claimed. This caught a real bug: seeding inside
    `train()` happens AFTER the caller built the model, so weight init was
    unseeded and two 'identical' runs differed by 6.0e-3. Hence `seeded_build`."""
    tmp = _tiny(n_eps=2, n_ticks=40)
    if tmp is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = tmp
    out = tempfile.mkdtemp()
    try:
        losses = []
        for i in (1, 2):
            cfg = T.TrainConfig(epochs=2, batch_size=16, lr=1e-3, seed=7,
                                run_name="det%d" % i, log_every=0)
            model = T.seeded_build(cfg, build_bc, cfg=BCConfig(obs_window=1, chunk_size=1))
            res = T.train(model, ds, cfg, run_dir=os.path.join(out, "r%d" % i))
            losses.append([r["train_loss"] for r in res["history"]])
        assert losses[0] == losses[1], (
            "two runs at the same seed diverged: %s vs %s" % (losses[0], losses[1]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_seeded_build_is_what_makes_it_deterministic():
    """The mechanism, not just the outcome: building WITHOUT seeding first gives
    different initial weights, which is the bug `seeded_build` exists to stop."""
    cfg = T.TrainConfig(epochs=1, batch_size=8, lr=1e-3, seed=3)
    a = T.seeded_build(cfg, build_bc, cfg=BCConfig(obs_window=1, chunk_size=1))
    b = T.seeded_build(cfg, build_bc, cfg=BCConfig(obs_window=1, chunk_size=1))
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb), "seeded_build must give identical weights"
    torch.manual_seed(999)                       # some other RNG state
    c = build_bc(BCConfig(obs_window=1, chunk_size=1))
    assert not all(torch.equal(pa, pc)
                   for pa, pc in zip(a.parameters(), c.parameters())), \
        "an unseeded build must differ - otherwise this test proves nothing"


def test_determinism_report_is_honest_about_what_it_achieved():
    d = T.set_determinism(0, strict=True)
    assert d["seed"] == 0 and d["cudnn_benchmark"] is False and d["tf32"] is False
    assert isinstance(d["deterministic_algorithms"], bool)
    assert isinstance(d["notes"], list)
    if not d["deterministic_algorithms"]:
        assert d["notes"], "a refusal must be explained, not left silent"


# ─── splits ───────────────────────────────────────────────────────────────────
def test_train_and_val_splits_share_no_seed():
    if not os.path.exists(DS.SPLITS):
        print("      (skipped: no splits_v1.json)")
        return
    sp = DS.load_splits()
    tr, va = set(sp["train"]), set(sp["val"])
    assert tr and va
    assert not (tr & va), "train and val share seeds %s" % sorted(tr & va)
    for cell, members in sp["bins"].items():
        assert set(members) <= (tr | va), cell
    assert len(tr) + len(va) == sum(len(v) for v in sp["bins"].values())


# ─── checkpoints ──────────────────────────────────────────────────────────────
def test_checkpoint_round_trips_to_identical_predictions():
    """A checkpoint that reloads to slightly different weights is the worst
    possible failure: evaluation silently scores a model nobody trained."""
    tmp = _tiny(n_eps=2, n_ticks=30)
    if tmp is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = tmp
    out = tempfile.mkdtemp()
    try:
        cfg = T.TrainConfig(epochs=1, batch_size=16, lr=1e-3, seed=1,
                            run_name="ckpt", log_every=0)
        model = T.seeded_build(cfg, build_bc, cfg=BCConfig(obs_window=1, chunk_size=1))
        res = T.train(model, ds, cfg, run_dir=out)
        obs = torch.stack([ds[i]["obs"] for i in range(16)])
        model = model.cpu().eval()
        with torch.no_grad():
            before = model(obs).clone()
        back, ck = T.load_checkpoint(os.path.join(out, "last.pt"), build_bc)
        with torch.no_grad():
            after = back(obs)
        assert torch.equal(before, after), (
            "max |difference| %.3e" % float((before - after).abs().max()))
        assert ck["spec_version"] == spec.SPEC_VERSION
        assert ck["model_cls"] == "BCPolicy"
        assert ck["data"]["episode_sources"] == ["scripted"]
        # a checkpoint from another spec version is refused, not loaded
        import copy
        raw = torch.load(os.path.join(out, "last.pt"), map_location="cpu",
                         weights_only=False)
        raw["spec_version"] = "g1-spec-0.9.0"
        torch.save(raw, os.path.join(out, "bad.pt"))
        _raises(AssertionError, T.load_checkpoint,
                os.path.join(out, "bad.pt"), build_bc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


# ─── C6: the BC model never sees the clock or the labels ──────────────────────
def test_bc_never_receives_gait_phase_or_phase_labels():
    """The schema freeze excludes the gait clock: handing every policy a clock
    gives BC the temporal capability ACT-LSTM is meant to supply."""
    import ast
    import inspect
    from g1_model import models
    tree = ast.parse(inspect.getsource(models))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for bad in ("gait_phase", "phase_labels", "gait", "Phase"):
        assert bad not in names and bad not in attrs, bad

    tmp = _tiny(n_eps=1, n_ticks=20)
    if tmp is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = tmp
    try:
        # the model is only ever called on `obs`, whose width is the state dim
        model = build_bc(BCConfig(obs_window=1, chunk_size=1))
        s = ds[0]
        assert s["obs"].shape == (1, spec.STATE_DIM)
        out = model(s["obs"].unsqueeze(0))
        assert out.shape == (1, 1, spec.ACTION_DIM)
        # 47 is the whole input: a clock would have to widen it
        first = next(m for m in model.net if isinstance(m, torch.nn.Linear))
        assert first.in_features == spec.STATE_DIM * model.obs_window
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chunked_bc_is_the_same_class_at_k_greater_than_one():
    """C1. Stage 2 REFUSED K>1 here so chunked BC would need its own class; that
    was wrong. If it were a separate class the measured effect of chunking would
    include every incidental difference between two implementations, and the
    isolation Stage 3 exists to provide would be gone. The ONLY difference is the
    number of actions predicted from one observation."""
    bc = BCPolicy(obs_window=1, chunk_size=1)
    ch = BCPolicy(obs_window=1, chunk_size=K_PROVISIONAL)
    assert type(bc) is type(ch) is BCPolicy
    assert bc(torch.zeros(3, 1, spec.STATE_DIM)).shape == (3, 1, spec.ACTION_DIM)
    assert ch(torch.zeros(3, 1, spec.STATE_DIM)).shape ==         (3, K_PROVISIONAL, spec.ACTION_DIM)
    # everything except the output head is identical in shape
    bl = [m for m in bc.net if isinstance(m, torch.nn.Linear)]
    cl = [m for m in ch.net if isinstance(m, torch.nn.Linear)]
    assert len(bl) == len(cl)
    for x, y in zip(bl[:-1], cl[:-1]):
        assert (x.in_features, x.out_features) == (y.in_features, y.out_features)
    assert bl[-1].out_features == spec.ACTION_DIM
    assert cl[-1].out_features == K_PROVISIONAL * spec.ACTION_DIM
    assert ch.hparams["chunk_size"] == K_PROVISIONAL
    _raises(ValueError, BCPolicy, obs_window=1, chunk_size=0)
    _raises(ValueError, BCPolicy, obs_window=0, chunk_size=1)
    _raises(ValueError, BCPolicy, obs_window=1, chunk_size=1,
            activation="banana")


def test_chunked_bc_trains_through_the_unchanged_loop():
    """Same loop, same loss, same optimizer - the point of C1."""
    tmp = _tiny(n_eps=2, n_ticks=60)
    if tmp is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, _ = tmp
    out = tempfile.mkdtemp()
    try:
        st, ac = (spec.NormStats.identity("state"),
                  spec.NormStats.identity("action"))
        K = 8
        ds = ChunkDataset.from_directory(
            tmp, LoaderConfig(chunk_size=K, obs_window=1,
                              tracking=TrackingPolicy()), st, ac, None)
        cfg = T.TrainConfig(epochs=2, batch_size=16, lr=1e-3, seed=0,
                            run_name="chunk", log_every=0)
        model = T.seeded_build(cfg, build_bc,
                               cfg=BCConfig(obs_window=1, chunk_size=K))
        res = T.train(model, ds, cfg, run_dir=out)
        assert len(res["history"]) == 2
        assert res["history"][-1]["train_loss"] < res["history"][0]["train_loss"]
        assert res["metadata"]["loader"]["chunk_size"] == K
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_first_action_is_the_no_ensembling_rule():
    """C2. Temporal ensembling is an ACT mechanism and belongs to Stage 4;
    folding it in here would mean Stage 3 measured chunking AND ensembling."""
    import ast
    import inspect
    from g1_model import models
    pred = torch.arange(2 * 4 * spec.ACTION_DIM, dtype=torch.float32).view(
        2, 4, spec.ACTION_DIM)
    got = first_action(pred)
    assert got.shape == (2, spec.ACTION_DIM)
    assert torch.equal(got, pred[:, 0, :])
    _raises(ValueError, first_action, torch.zeros(2, 4))
    # and nothing in the module implements ensembling
    src = inspect.getsource(models)
    names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    for bad in ("ensemble", "temporal_ensemble", "ensembling"):
        assert bad not in names and bad not in attrs, bad


def test_k_is_required_and_marked_provisional():
    """C3: K has no default, like obs_window, and its provenance is in the code."""
    import inspect
    sig = inspect.signature(BCConfig).parameters
    for name in ("obs_window", "chunk_size"):
        assert sig[name].default is inspect.Parameter.empty, name
    _raises(TypeError, BCConfig)
    _raises(TypeError, BCConfig, obs_window=1)
    assert K_PROVISIONAL == 100
    from g1_model import models
    doc = models.__dict__["__doc__"] or ""
    src = inspect.getsource(models)
    assert "PROVISIONAL" in src and "NOT SWEPT" in src
    assert "SETTLE IT" in src and "PILOTED" in src.upper()


def test_bc_refuses_an_observation_window_it_was_not_built_for():
    m = BCPolicy(obs_window=1, chunk_size=1)
    m(torch.zeros(2, 1, spec.STATE_DIM))                    # fine
    e = _raises(ValueError, m, torch.zeros(2, 4, spec.STATE_DIM))
    assert "obs_window" in str(e)
    _raises(ValueError, m, torch.zeros(2, 1, 46))


def test_hyperparameters_are_marked_provisional_in_the_code():
    """C2: the statement has to live in the file, not only in a report."""
    from g1_model.models import PROVISIONAL, BCConfig
    assert "PROVISIONAL" in PROVISIONAL and "SCRIPTED" in PROVISIONAL
    assert "not tuned" in PROVISIONAL.lower()
    assert BCConfig(obs_window=1, chunk_size=1).provisional == PROVISIONAL
    assert PROVISIONAL in BCConfig(obs_window=1, chunk_size=1).as_metadata()["provisional"]


# ─── C5: provenance travels with the run ──────────────────────────────────────
def test_run_metadata_states_which_data_it_ran_on():
    """C5: a caveat that lives only in a report is a caveat that gets lost."""
    tmp = _tiny(n_eps=2, n_ticks=20)
    if tmp is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = tmp
    out = tempfile.mkdtemp()
    try:
        prov = T.dataset_provenance(ds)
        assert prov["episode_sources"] == ["scripted"]
        assert prov["real_demonstrations"] is False
        assert "piloted" in prov["caveat"] and "CITED" in prov["caveat"].upper()

        cfg = T.TrainConfig(epochs=1, batch_size=16, lr=1e-3, seed=0,
                            run_name="prov", log_every=0)
        model = T.seeded_build(cfg, build_bc, cfg=BCConfig(obs_window=1, chunk_size=1))
        T.train(model, ds, cfg, run_dir=out)
        with open(os.path.join(out, "metadata.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta["data"]["real_demonstrations"] is False
        assert "piloted" in meta["data"]["caveat"]
        assert meta["spec_version"] == spec.SPEC_VERSION
        assert meta["baselines"]["zero"] > 0
        assert meta["loader"]["tracking"] == dict(
            exclude_overlapping_chunks=False, max_degraded_fraction=None)
        # and the per-epoch metrics file exists and is one JSON object per line
        rows = [json.loads(L) for L in
                open(os.path.join(out, "metrics.jsonl"), encoding="utf-8")
                if L.strip()]
        assert len(rows) == 1 and "train_loss" in rows[0]
        assert "git_commit" in rows[0] and "lr" in rows[0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


# ═══ step budgets and the stopping rule (Stage 4, part C) ═════════════════════
def _w(recon, total=None):
    return dict(recon_l1=recon, train_loss=recon if total is None else total)


def test_stop_rule_fires_only_when_every_monitored_quantity_stalls():
    rule = T.StopRule(patience_windows=3, min_rel_improvement=0.01)
    improving = [_w(1.0 - 0.05 * i) for i in range(8)]
    assert rule.check(improving) is None, "a steadily improving run must not stop"
    flat = [_w(1.0)] * 3 + [_w(0.999)] * 5
    fired = rule.check(flat)
    assert fired and all(v < 0.01 for v in fired["rel_improvement_over_span"].values())
    # recon flat but total still falling (a CVAE spending its KL): keep going
    mixed = [_w(0.5, 1.0 - 0.05 * i) for i in range(8)]
    assert rule.check(mixed) is None, "stop only when EVERY monitored quantity stalls"
    assert rule.check(improving[:3]) is None, "needs more windows than the span"


def test_stop_rule_uses_best_of_span_so_one_noisy_window_cannot_decide():
    rule = T.StopRule(patience_windows=2, min_rel_improvement=0.01)
    # a noisy spike in the recent span must not look like a stall...
    noisy = [_w(1.0), _w(0.9), _w(0.8), _w(0.95), _w(0.70)]
    assert rule.check(noisy) is None
    # ...and one lucky window must not keep a flat run alive forever
    flat = [_w(0.5)] * 4 + [_w(0.499)] * 2
    assert rule.check(flat) is not None


def test_budget_must_be_exactly_one_of_epochs_or_steps():
    _raises(T.TrainError, T.TrainConfig, epochs=None, batch_size=8, lr=1e-3)
    _raises(T.TrainError, T.TrainConfig, epochs=3, batch_size=8, lr=1e-3, max_steps=10)
    rule = T.StopRule(patience_windows=2, min_rel_improvement=0.01)
    _raises(T.TrainError, T.TrainConfig, epochs=3, batch_size=8, lr=1e-3, stop_rule=rule)
    T.TrainConfig(epochs=None, batch_size=8, lr=1e-3, max_steps=10, stop_rule=rule)
    _raises(T.TrainError, T.StopRule, patience_windows=0, min_rel_improvement=0.01)
    _raises(T.TrainError, T.StopRule, patience_windows=2, min_rel_improvement=1.5)


def test_step_budgeted_run_logs_windows_records_budget_and_labels_quantities():
    tmp = _tiny(n_eps=2, n_ticks=40)
    if tmp is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = tmp
    out = tempfile.mkdtemp()
    try:
        from g1_model.ambiguity import RECON_UNITS, TOTAL_LOSS_UNITS
        cfg = T.TrainConfig(epochs=None, batch_size=8, lr=1e-3, seed=0,
                            max_steps=25, window_steps=5, log_every=0)
        m = T.seeded_build(cfg, build_bc, cfg=BCConfig(obs_window=1, chunk_size=1))
        r = T.train(m, ds, cfg, run_dir=out)
        assert r["optimizer_steps"] == 25, r["optimizer_steps"]
        assert r["stop"] == dict(reason="hard_cap", step=25), r["stop"]
        assert [h["step"] for h in r["history"]] == [5, 10, 15, 20, 25]
        assert r["quantities"]["recon_l1"].units == RECON_UNITS
        assert r["quantities"]["train_loss"].units == TOTAL_LOSS_UNITS
        assert r["steps_per_second"] > 0
        meta = json.load(open(os.path.join(out, "metadata.json"), encoding="utf-8"))
        assert meta["budget"]["unit"] == "optimizer_steps"
        assert meta["budget"]["max_steps"] == 25
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_stop_rule_ends_a_run_before_the_cap_and_says_where():
    tmp = _tiny(n_eps=2, n_ticks=40)
    if tmp is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = tmp
    out = tempfile.mkdtemp()
    try:
        # lr 0 freezes the weights. The window is ONE EPOCH (80 samples / batch 8
        # = 10 steps), so every window scores the whole dataset and its mean is
        # the same number - which makes the firing step exact. With a window
        # SMALLER than the dataset the mean moves with which samples happened to
        # land in it even when nothing is learning (measured: a 5-step window of
        # 40 samples moved 0.49% on frozen weights and delayed the rule by one
        # window). The real run's 1000-step window is 8,000 samples, ~1.05 epochs
        # of the 10-episode gate set, for exactly this reason.
        rule = T.StopRule(patience_windows=2, min_rel_improvement=0.01)
        spe = (len(ds) + 7) // 8
        cfg = T.TrainConfig(epochs=None, batch_size=8, lr=0.0, seed=0,
                            max_steps=1000, window_steps=spe, stop_rule=rule,
                            log_every=0)
        m = T.seeded_build(cfg, build_bc, cfg=BCConfig(obs_window=1, chunk_size=1))
        r = T.train(m, ds, cfg, run_dir=out)
        assert r["stop"]["reason"] == "stop_rule", r["stop"]
        assert r["stop"]["step"] == 3 * spe,             "a 2-window span needs 3 windows: %s" % r["stop"]
        assert r["optimizer_steps"] == 3 * spe
        meta = json.load(open(os.path.join(out, "metadata.json"), encoding="utf-8"))
        assert "1.00% relative" in meta["budget"]["stop_rule"], meta["budget"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


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
