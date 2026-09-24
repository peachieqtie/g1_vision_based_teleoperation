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
            cfg = T.TrainConfig(epochs=2, batch_size=16, seed=7, **_BC_OPT,
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
    cfg = T.TrainConfig(epochs=1, batch_size=8, seed=3, **_BC_OPT)
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
        cfg = T.TrainConfig(epochs=1, batch_size=16, seed=1, **_BC_OPT,
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
        cfg = T.TrainConfig(epochs=2, batch_size=16, seed=0, **_BC_OPT,
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

        cfg = T.TrainConfig(epochs=1, batch_size=16, seed=0, **_BC_OPT,
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
    kw = dict(batch_size=8, seed=0, **_BC_OPT)
    _raises(T.TrainError, T.TrainConfig, epochs=None, **kw)
    _raises(T.TrainError, T.TrainConfig, epochs=3, max_steps=10, **kw)
    rule = T.StopRule(patience_windows=2, min_rel_improvement=0.01)
    _raises(T.TrainError, T.TrainConfig, epochs=3, stop_rule=rule, **kw)
    T.TrainConfig(epochs=None, max_steps=10, stop_rule=rule, window_steps=5, **kw)
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
        cfg = T.TrainConfig(epochs=None, batch_size=8, seed=0, **_BC_OPT,
                            max_steps=25, window_steps=5, stop_rule=None, log_every=0)
        m = T.seeded_build(cfg, build_bc, cfg=BCConfig(obs_window=1, chunk_size=1))
        r = T.train(m, ds, cfg, run_dir=out)
        assert r["optimizer_steps"] == 25, r["optimizer_steps"]
        assert r["stop"] == dict(reason="hard_cap", step=25), r["stop"]
        assert [h["step"] for h in r["history"]] == [5, 10, 15, 20, 25]
        from g1_model.ambiguity import MEASURED_AT_DEPLOYMENT, MEASURED_TRAIN_MODE_WINDOW
        assert r["quantities"]["recon_l1"].units == RECON_UNITS
        assert r["quantities"]["recon_l1"].measured == MEASURED_TRAIN_MODE_WINDOW
        assert r["quantities"]["train_loss"].units == TOTAL_LOSS_UNITS
        dq = r["quantities"]["deployment_recon"]
        assert dq.units == RECON_UNITS and dq.measured == MEASURED_AT_DEPLOYMENT
        assert all(h["eval_recon_l1"] is not None for h in r["history"])
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
        frozen = BCConfig(obs_window=1, chunk_size=1, lr=0.0)
        cfg = T.TrainConfig(epochs=None, batch_size=8, seed=0,
                            **frozen.optimizer_config(),
                            max_steps=1000, window_steps=spe, stop_rule=rule,
                            log_every=0)
        m = T.seeded_build(cfg, build_bc, cfg=frozen)
        r = T.train(m, ds, cfg, run_dir=out)
        assert r["stop"]["reason"] == "stop_rule", r["stop"]
        assert r["stop"]["step"] == 3 * spe,             "a 2-window span needs 3 windows: %s" % r["stop"]
        assert r["optimizer_steps"] == 3 * spe
        meta = json.load(open(os.path.join(out, "metadata.json"), encoding="utf-8"))
        assert "1.00% relative" in meta["budget"]["stop_rule"], meta["budget"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


# ═══ Stage 4 A/D: resume, and a run that survives being killed ════════════════
#: BC's own declared training hyperparameters (TR28): what every BC TrainConfig
#: in this file is built from, instead of restating lr and leaving the rest to
#: loop defaults.
_BC_OPT = BCConfig(obs_window=1, chunk_size=1).optimizer_config()


def _tiny_act(K=5):
    """A small ACT WITH dropout, so resuming must restore the dropout RNG too."""
    from g1_model.act import ACTConfig, build_act
    return (ACTConfig(obs_window=1, chunk_size=K, lr=1e-3, weight_decay=0.0,
                      hidden_dim=32, dim_feedforward=64, nheads=4, enc_layers=1,
                      dec_layers=1, dropout=0.1), build_act)


def _tiny_k(K, n_eps=2, n_ticks=40):
    tmp = _tiny(n_eps=n_eps, n_ticks=n_ticks)
    if tmp is None:
        return None
    tmp, _ = tmp
    st, ac = (spec.NormStats.identity("state"), spec.NormStats.identity("action"))
    ds = ChunkDataset.from_directory(
        tmp, LoaderConfig(chunk_size=K, obs_window=1, tracking=TrackingPolicy()),
        st, ac, None)
    return tmp, ds


def _act_run(ds, out, max_steps, window, resume=None, gate_reference=None):
    mcfg, build = _tiny_act()
    cfg = T.TrainConfig(epochs=None, batch_size=8, seed=3, max_steps=max_steps,
                        window_steps=window, stop_rule=None, log_every=0,
                        **mcfg.optimizer_config())
    m = T.seeded_build(cfg, build, cfg=mcfg)
    return T.train(m, ds, cfg, run_dir=out, resume=resume,
                   gate_reference=gate_reference)


def _losses(history):
    return [(h["step"], h["train_loss"], h["recon_l1"], h.get("train_kl"))
            for h in history]


def test_full_state_resume_is_bit_exact():
    """A: a resumed run CONTINUES the curve - it reproduces, to the bit, the
    numbers the uninterrupted run produced after the resume point.

    Built to be hard: ACT WITH dropout (so the dropout RNG must be restored), and
    a window of 7 steps against a 10-step epoch (80 samples / batch 8), so the
    kill lands MID-PASS and the resumed run must reproduce that pass's shuffled
    order and skip the batches already trained on.
    """
    got = _tiny_k(K=5)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    a, b, c = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        whole = _act_run(ds, a, max_steps=28, window=7)          # never interrupted
        first = _act_run(ds, b, max_steps=14, window=7)          # "killed" at 14
        assert os.path.isfile(os.path.join(b, "state.pt"))
        ck = torch.load(os.path.join(b, "state.pt"), map_location="cpu",
                        weights_only=False)
        assert ck["step"] == 14 and ck["batches_done"] == 4, (ck["step"], ck["batches_done"])
        assert "optimizer" in ck and ck["optimizer"]["state"], "optimizer state missing"
        assert ck["scheduler"] is None      # no scheduler exists; recorded as such
        resumed = _act_run(ds, c, max_steps=28, window=7,
                           resume=T.Resume(os.path.join(b, "state.pt")))
        assert _losses(resumed["history"]) == _losses(whole["history"]), (
            "resumed curve differs from the uninterrupted one:\n  whole   %s\n  resumed %s"
            % (_losses(whole["history"]), _losses(resumed["history"])))
        assert [h["segment"] for h in resumed["history"]] == [1, 1, 2, 2]
        assert resumed["metadata"]["resume"]["kind"] == "full_state"
        assert resumed["metadata"]["resume"]["not_restored"] == []
        assert resumed["optimizer_steps"] == 28 and resumed["segment_steps"] == 14
        rows = [json.loads(L) for L in open(os.path.join(c, "metrics.jsonl"),
                                            encoding="utf-8") if L.strip()]
        assert [r["step"] for r in rows] == [7, 14, 21, 28], "curve not continuous on disk"
    finally:
        for d in (tmp, a, b, c):
            shutil.rmtree(d, ignore_errors=True)


def test_resume_actually_depends_on_the_restored_state():
    """The bit-exact test would pass vacuously if resuming ignored the file and
    re-trained from scratch to the same numbers. Break the restored optimizer
    state and the curve must change."""
    got = _tiny_k(K=5)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    a, b, c = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        whole = _act_run(ds, a, max_steps=28, window=7)
        _act_run(ds, b, max_steps=14, window=7)
        p = os.path.join(b, "state.pt")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        ck["optimizer"]["state"] = {}                 # wipe AdamW's moments
        torch.save(ck, p)
        broken = _act_run(ds, c, max_steps=28, window=7, resume=T.Resume(p))
        assert _losses(broken["history"])[:2] == _losses(whole["history"])[:2]
        assert _losses(broken["history"])[2:] != _losses(whole["history"])[2:], \
            "wiping the optimizer state changed nothing: resume is not using it"
    finally:
        for d in (tmp, a, b, c):
            shutil.rmtree(d, ignore_errors=True)


def test_weights_only_resume_continues_the_curve_and_says_what_it_lost():
    """A: best.pt carries weights and provenance but no optimizer, RNG or step.
    Resuming from it must identify the step from the prior run's own records,
    carry the history over, CONTINUE (not restart) the curve, and record exactly
    what was not restored."""
    got = _tiny_k(K=5)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    a, c = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        prior = _act_run(ds, a, max_steps=40, window=10)
        ck = torch.load(os.path.join(a, "best.pt"), map_location="cpu",
                        weights_only=False)
        assert "optimizer" not in ck and "rng" not in ck
        best_row = min(prior["history"], key=lambda h: h["eval_recon_l1"])
        res = _act_run(ds, c, max_steps=60, window=10,
                       resume=T.Resume(os.path.join(a, "best.pt")))
        info = res["metadata"]["resume"]
        assert info["kind"] == "weights_only"
        assert info["resumed_at_step"] == best_row["step"], info
        assert any("AdamW" in s for s in info["not_restored"])
        assert any("RNG" in s for s in info["not_restored"])
        n1 = sum(1 for h in res["history"] if h["segment"] == 1)
        assert n1 == len([h for h in prior["history"] if h["step"] <= best_row["step"]])
        first_new = [h for h in res["history"] if h["segment"] == 2][0]
        first_old = prior["history"][0]
        assert first_new["step"] == best_row["step"] + 10
        # CONTINUES: the first new window sits near where the prior run left off,
        # far below where a fresh model starts
        assert first_new["recon_l1"] < first_old["recon_l1"], (first_new, first_old)
    finally:
        for d in (tmp, a, c):
            shutil.rmtree(d, ignore_errors=True)


def test_resume_refuses_a_different_training_config():
    got = _tiny_k(K=5)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    a, c = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        _act_run(ds, a, max_steps=10, window=5)
        from g1_model.act import ACTConfig, build_act
        mcfg = ACTConfig(obs_window=1, chunk_size=5, lr=5e-4, weight_decay=0.0,
                         hidden_dim=32, dim_feedforward=64, nheads=4, enc_layers=1,
                         dec_layers=1, dropout=0.1)                 # lr differs
        cfg = T.TrainConfig(epochs=None, batch_size=8, seed=3, max_steps=20,
                            window_steps=5, stop_rule=None, log_every=0,
                            **mcfg.optimizer_config())
        m = T.seeded_build(cfg, build_act, cfg=mcfg)
        e = _raises(T.TrainError, T.train, m, ds, cfg, run_dir=c,
                    resume=T.Resume(os.path.join(a, "state.pt")))
        assert "lr" in str(e)
    finally:
        for d in (tmp, a, c):
            shutil.rmtree(d, ignore_errors=True)


def test_a_crash_leaves_the_verdict_and_a_resumable_state():
    """D: the 2 h 19 min run died with no result.json and no gate.json because
    both were written only at the end. Now: progress files every window, the
    crash reason recorded, and a state.pt to resume from.

    The crash is injected INSIDE a forward pass, the way the real out-of-memory
    error struck (inside clip_grad_norm_, mid-step). Before raising, the injector
    reads result.json from disk, proving the files existed WHILE the run was
    alive and said so."""
    got = _tiny_k(K=5)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    from g1_model import ambiguity as AMB
    ref = AMB.neighbour_ambiguity(ds)
    a, c = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        mcfg, build = _tiny_act()
        cfg = T.TrainConfig(epochs=None, batch_size=8, seed=3, max_steps=40,
                            window_steps=5, stop_rule=None, log_every=0,
                            **mcfg.optimizer_config())
        m = T.seeded_build(cfg, build, cfg=mcfg)
        seen, calls = {}, {"n": 0}
        real_terms = m.loss_terms

        def dying(*args, **kw):
            calls["n"] += 1
            if calls["n"] == 13:                   # mid-window, after 2 windows
                with open(os.path.join(a, "result.json"), encoding="utf-8") as fh:
                    seen["while_alive"] = json.load(fh)
                raise RuntimeError("CUDA error: out of memory (simulated)")
            return real_terms(*args, **kw)

        m.loss_terms = dying
        _raises(RuntimeError, T.train, m, ds, cfg, run_dir=a, gate_reference=ref)

        alive = seen["while_alive"]
        assert alive["status"] == "running" and alive["optimizer_steps"] == 10
        res = json.load(open(os.path.join(a, "result.json"), encoding="utf-8"))
        assert res["status"] == "crashed" and "out of memory" in res["error"]
        assert [h["step"] for h in res["history"]] == [5, 10]
        gate = json.load(open(os.path.join(a, "gate.json"), encoding="utf-8"))
        assert gate["status"] == "crashed" and gate["at_step"] == 12, gate["at_step"]
        assert gate["train_error_units"] == AMB.RECON_UNITS
        assert gate["train_error_measured"] == AMB.MEASURED_AT_DEPLOYMENT
        assert gate["final"] is False and "PROVISIONAL" in gate["note"]
        # the provisional verdict scores the weights at the last completed window
        assert gate["scored_at_step"] == 10
        assert abs(gate["train_error"] - res["history"][-1]["eval_recon_l1"]) < 1e-12

        # and the state it left behind finishes the run
        done = _act_run(ds, c, max_steps=40, window=5,
                        resume=T.Resume(os.path.join(a, "state.pt")),
                        gate_reference=ref)
        assert done["optimizer_steps"] == 40
        final = json.load(open(os.path.join(c, "gate.json"), encoding="utf-8"))
        assert final["final"] is True and final["status"] == "hard_cap"
        assert final["note"] is None
    finally:
        for d in (tmp, a, c):
            shutil.rmtree(d, ignore_errors=True)


def test_progress_files_are_written_atomically():
    """A kill DURING a write must leave the previous complete file, not a
    truncated one: write-then-rename, never write-in-place."""
    import inspect
    src = inspect.getsource(T._atomic_json)
    assert "os.replace" in src and ".tmp" in src
    assert "os.replace" in inspect.getsource(T._atomic_torch_save)
    loop = inspect.getsource(T.train)
    assert '_atomic_json(os.path.join(run_dir, "result.json")' in loop
    assert '_atomic_json(os.path.join(run_dir, "gate.json")' in loop
    assert "open(os.path.join(run_dir, \"result.json\"), \"w\"" not in loop



# ═══ Stage 4: the gate scores the DEPLOYED function (eval-mode fix, 2026-09-22) ═
# Converged ACT failed the gate at 0.050886 (ratio 1.140), scored as a train-mode
# window mean with dropout 0.1 on. Its deployed function scores 0.041081 (0.920,
# PASS). TR29 was the first of this family (total loss scored as reconstruction);
# these tests make a third impossible: the gate cannot score a model in train
# mode, cannot reach a CVAE encoder, and cannot be handed a number measured any
# other way.
def _dropout_bc(p=0.5):
    cfg = BCConfig(obs_window=1, chunk_size=1, dropout=p)
    torch.manual_seed(0)
    return build_bc(cfg=cfg)


def _manual_eval_l1(model, ds):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(len(ds)):
            b = T.collate_chunks([ds[i]])
            l1, c = T.masked_l1(model(b["obs"]), b["action"], b["action_mask"])
            tot += float(l1) * int(c)
            n += int(c)
    return tot / n


def test_scoring_is_eval_mode_whatever_mode_the_model_is_in():
    got = _tiny(n_eps=2, n_ticks=40)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    try:
        m = _dropout_bc()
        m.train()                                   # handed over in TRAIN mode
        a = T.score_deployment(m, ds, torch.device("cpu")).value
        b = T.score_deployment(m, ds, torch.device("cpu")).value
        assert a == b, "dropout leaked into scoring: two passes differ (%r, %r)" % (a, b)
        assert m.training, "the caller's mode was not restored"
        assert abs(a - _manual_eval_l1(m, ds)) < 1e-6
        # and the guard is not vacuous: in train mode this model DOES score differently
        m.train()
        with torch.no_grad():
            bt = T.collate_chunks([ds[i] for i in range(len(ds))])
            tr_l1, _ = T.masked_l1(m(bt["obs"]), bt["action"], bt["action_mask"])
        assert abs(float(tr_l1) - a) > 1e-4, "dropout 0.5 changed nothing - test is blind"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scoring_refuses_a_forward_that_switches_back_to_train_mode():
    got = _tiny(n_eps=1, n_ticks=20)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    try:
        class Sneaky(BCPolicy):
            def forward(self, obs):
                self.train()                        # flips dropout back on
                return super().forward(obs)
        m = Sneaky(obs_window=1, chunk_size=1, dropout=0.5)
        e = _raises(T.ScoringError, T.score_deployment, m, ds, torch.device("cpu"))
        assert "TRAIN mode" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scoring_never_hands_the_model_the_target():
    got = _tiny(n_eps=1, n_ticks=20)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    try:
        calls = []

        class Spy(BCPolicy):
            def forward(self, *args, **kw):
                calls.append((len(args), sorted(kw), tuple(args[0].shape)))
                return super().forward(args[0])
        m = Spy(obs_window=1, chunk_size=1)
        T.score_deployment(m, ds, torch.device("cpu"))
        assert calls and all(c[0] == 1 and c[1] == [] for c in calls), calls[:3]
        assert all(c[2][1:] == (1, spec.STATE_DIM) for c in calls), calls[:3]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scoring_cannot_reach_the_act_encoder():
    got = _tiny_k(K=5, n_eps=1, n_ticks=20)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    try:
        from g1_model.act import ACTPolicy
        mcfg, build = _tiny_act()
        torch.manual_seed(0)
        m = build(cfg=mcfg)
        # a normal ACT scores through its prior (z = 0) ...
        q = T.score_deployment(m, ds, torch.device("cpu")).value
        assert abs(q - _manual_eval_l1(m, ds)) < 1e-6
        # ... which differs from what eval-mode loss_terms reports, because that
        # path runs the encoder ON THE TARGET - the leak this guard exists for
        m.eval()
        with torch.no_grad():
            bt = T.collate_chunks([ds[i] for i in range(len(ds))])
            _, post, _, _ = m.loss_terms(bt["obs"], bt["action"], bt["action_mask"])
        assert abs(float(post) - q) > 1e-6, "posterior path equals prior - test is blind"

        class Leaky(ACTPolicy):
            def forward(self, obs, actions=None, action_mask=None):
                if actions is None:                 # "inference" that peeks
                    fake = torch.zeros(obs.shape[0], self.cfg.chunk_size,
                                       self.cfg.action_dim)
                    self.encode(obs, fake)
                return super().forward(obs, actions, action_mask)
        leaky = Leaky(mcfg)
        e = _raises(T.ScoringError, T.score_deployment, leaky, ds, torch.device("cpu"))
        assert "reads the target" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_act_declares_exactly_the_modules_encode_uses():
    """The hook list is only as good as the declaration. Run `encode` and
    `forward(obs)` with every child hooked, and compare."""
    mcfg, build = _tiny_act()
    torch.manual_seed(0)
    m = build(cfg=mcfg).eval()
    fired = set()
    hs = [mod.register_forward_pre_hook(lambda _m, _i, n=n: fired.add(n))
          for n, mod in m.named_children()]
    obs = torch.zeros(2, 1, spec.STATE_DIM)
    acts = torch.zeros(2, mcfg.chunk_size, spec.ACTION_DIM)
    with torch.no_grad():
        m.encode(obs, acts)
        by_encode = set(fired)
        fired.clear()
        m(obs)
        by_inference = set(fired)
    for h in hs:
        h.remove()
    declared = set(m.TARGET_READING_MODULES)
    # every module encode CALLS is declared (and hooked) ...
    assert by_encode <= declared, (by_encode, declared)
    # ... the one declared-but-uncalled is cls_embed, read via .weight (no hook can
    # fire on it); the four that ARE called fire on every encode, so it is covered
    assert declared - by_encode == {"cls_embed"}, declared - by_encode
    assert len(by_encode) == 4
    assert not (by_inference & declared), by_inference


def test_a_model_with_an_encoder_it_does_not_declare_is_refused():
    got = _tiny(n_eps=1, n_ticks=20)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    try:
        class Undeclared(BCPolicy):
            def encode(self, obs, actions):
                return obs
        e = _raises(T.ScoringError, T.score_deployment,
                    Undeclared(obs_window=1, chunk_size=1), ds, torch.device("cpu"))
        assert "TARGET_READING_MODULES" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scoring_does_not_touch_the_training_rng():
    """Scored every window, so any RNG draw would shift the run's dropout and
    shuffle streams. It cuts batches by index - no DataLoader base-seed draw."""
    got = _tiny(n_eps=1, n_ticks=20)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    try:
        m = _dropout_bc()
        before = (torch.get_rng_state().clone(), np.random.get_state()[1].copy())
        T.score_deployment(m, ds, torch.device("cpu"))
        assert torch.equal(before[0], torch.get_rng_state())
        assert np.array_equal(before[1], np.random.get_state()[1])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_selection_and_the_gate_use_one_quantity():
    """B and A together, on an ACT WITH dropout: best.pt is the window with the
    lowest DEPLOYMENT score, the gate scores last.pt's weights the same way, and
    both are what score_deployment returns when re-run from the files."""
    got = _tiny_k(K=5)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    from g1_model import ambiguity as AMB
    from g1_model.act import build_act
    ref = AMB.neighbour_ambiguity(ds)
    out = tempfile.mkdtemp()
    try:
        res = _act_run(ds, out, max_steps=30, window=5, gate_reference=ref)
        rows = res["history"]
        best_row = min(rows, key=lambda h: h["eval_recon_l1"])
        best, _ = T.load_checkpoint(os.path.join(out, "best.pt"), build_act)
        ck = torch.load(os.path.join(out, "best.pt"), map_location="cpu",
                        weights_only=False)
        assert ck["step"] == best_row["step"], (ck["step"], best_row["step"])
        assert ck["selection"]["criterion"] == T.SELECTION_CRITERION
        rescored = T.score_deployment(best, ds, torch.device("cpu")).value
        assert abs(rescored - best_row["eval_recon_l1"]) < 1e-5
        assert abs(ck["selection"]["value"] - best_row["eval_recon_l1"]) < 1e-12
        # the selection criterion is NOT the train-mode number: they differ here
        assert any(abs(h["eval_recon_l1"] - h["recon_l1"]) > 1e-4 for h in rows)

        gate = json.load(open(os.path.join(out, "gate.json"), encoding="utf-8"))
        last, _ = T.load_checkpoint(os.path.join(out, "last.pt"), build_act)
        final = T.score_deployment(last, ds, torch.device("cpu")).value
        assert gate["final"] is True and gate["scored_weights"] == "final weights"
        assert gate["train_error_measured"] == AMB.MEASURED_AT_DEPLOYMENT
        assert abs(gate["train_error"] - final) < 1e-5, (gate["train_error"], final)
        assert abs(res["quantities"]["deployment_recon"].value - gate["train_error"]) < 1e-12
        assert gate["train_mode_window_recon"]["value"] == rows[-1]["recon_l1"]
    finally:
        for d in (tmp, out):
            shutil.rmtree(d, ignore_errors=True)


def test_resume_from_a_state_chosen_on_the_old_criterion_restarts_selection():
    """A state.pt written before this fix holds a `best` chosen on the train-mode
    window mean. Comparing the new quantity against it would compare two
    different measurements, so the resumed run restarts selection and says so."""
    got = _tiny_k(K=5)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    a, c = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        _act_run(ds, a, max_steps=10, window=5)
        p = os.path.join(a, "state.pt")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        del ck["selection"]
        ck["best"] = -1.0            # a value no new window could ever beat
        torch.save(ck, p)
        res = _act_run(ds, c, max_steps=20, window=5, resume=T.Resume(p))
        assert any("best value" in x for x in res["metadata"]["resume"]["not_restored"])
        assert os.path.isfile(os.path.join(c, "best.pt")), "selection did not restart"
    finally:
        for d in (tmp, a, c):
            shutil.rmtree(d, ignore_errors=True)


def test_the_scoring_path_is_structurally_separate_from_the_training_loss():
    import inspect
    import ast, textwrap
    fn = ast.parse(textwrap.dedent(inspect.getsource(T.score_deployment))).body[0]
    fn.body = fn.body[1:]                          # the code, not the docstring
    src = ast.unparse(fn)
    for banned in ("loss_terms", "forward_loss", "actions=", "DataLoader("):
        assert banned not in src, banned
    assert "model(obs)" in src and "_DeploymentGuard(model)" in src, src
    assert "guard.finish(" in src, "the end-of-pass checks must run on every score"
    assert "score_deployment" in inspect.getsource(T.evaluate)
    loop = inspect.getsource(T.train)
    assert loop.count("forward_loss(") == 1, "forward_loss must serve training only"
    assert "_gate(q, gate_reference)" in loop


# ═══ The hardened deployment guard (docs/ACT_AUDIT_REPORT.md item 5) ═════════
# The audit defeated 4 of 6 attacks on the first guard, and measured that a leak
# moves the gated number by 1.0e-7 - an implausible score would never reveal one.
# Each test below is the audit's own attack (m5_gate.py), and each FAILS against
# the pre-hardening guard: it raised on none of these.
def _guarded_score(model, ds):
    return T.score_deployment(model, ds, torch.device("cpu"))


def _attack_act(cls):
    mcfg, _ = _tiny_act()
    torch.manual_seed(0)
    return cls(mcfg)


def _expect_refused(model, ds, *needles):
    e = _raises(T.ScoringError, _guarded_score, model, ds)
    for s in needles:
        assert s in str(e), (s, str(e))
    return e


def _with_act_data(fn):
    got = _tiny_k(K=5, n_eps=1, n_ticks=20)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    try:
        fn(ds)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_guard_refuses_a_latent_that_is_not_the_prior_mean():
    """Audit attack 1: z drawn at inference. Also a DETERMINISTIC non-zero z, so
    the refusal is the z check itself, not only the RNG check."""
    from g1_model.act import ACTPolicy

    class SamplePrior(ACTPolicy):
        def forward(self, obs, actions=None, action_mask=None):
            if actions is None:
                z = torch.randn(obs.shape[0], self.cfg.latent_dim) * 50
                return self.action_head(self._decode(obs, self.latent_out_proj(z)))
            return super().forward(obs, actions, action_mask)

    class FixedNonzero(ACTPolicy):
        def forward(self, obs, actions=None, action_mask=None):
            if actions is None:
                z = torch.full((obs.shape[0], self.cfg.latent_dim), 0.5)
                return self.action_head(self._decode(obs, self.latent_out_proj(z)))
            return super().forward(obs, actions, action_mask)

    def run(ds):
        _expect_refused(_attack_act(SamplePrior), ds, "not the prior mean")
        _expect_refused(_attack_act(FixedNonzero), ds, "not the prior mean", "5.000e-01")
    _with_act_data(run)


def test_guard_refuses_the_encoder_called_through_forward():
    """Audit attack 2: `.forward()` skips pre-hooks. The guard now replaces the
    forward of every target-reading module AND its submodules on the instance."""
    from g1_model.act import ACTPolicy

    class BypassHook(ACTPolicy):
        def forward(self, obs, actions=None, action_mask=None):
            if actions is None:
                src = torch.zeros(self.cfg.chunk_size + 2, obs.shape[0], self.cfg.hidden_dim)
                self.encoder.forward(src)
            return super().forward(obs, actions, action_mask)

    class SubmoduleBypass(ACTPolicy):
        def forward(self, obs, actions=None, action_mask=None):
            if actions is None:
                src = torch.zeros(self.cfg.chunk_size + 2, obs.shape[0], self.cfg.hidden_dim)
                self.encoder.layers[0](src)
            return super().forward(obs, actions, action_mask)

    def run(ds):
        _expect_refused(_attack_act(BypassHook), ds, "reads the target")
        _expect_refused(_attack_act(SubmoduleBypass), ds, "reads the target")
    _with_act_data(run)


def test_guard_refuses_functional_use_of_the_encoder_weights():
    """No module call at all: an encoder WEIGHT read directly into the output.
    The NaN probe catches it because the output then depends on it."""
    from g1_model.act import ACTPolicy

    class FunctionalLeak(ACTPolicy):
        def forward(self, obs, actions=None, action_mask=None):
            out = super().forward(obs, actions, action_mask)
            if actions is None:                         # weights, no module call
                out = out + 1e-3 * self.encoder_action_proj.weight.mean()
            return out

    _with_act_data(lambda ds: _expect_refused(
        _attack_act(FunctionalLeak), ds, "depends on the weights"))


def test_guard_refuses_functional_dropout():
    """Audit attack 3: F.dropout(training=True) has no module whose mode could be
    checked. It draws randomness, and the probe batch no longer reproduces."""
    from g1_model.act import ACTPolicy
    import torch.nn.functional as F

    class FunctionalDropout(ACTPolicy):
        def forward(self, obs, actions=None, action_mask=None):
            out = super().forward(obs, actions, action_mask)
            return F.dropout(out, 0.5, training=True) if actions is None else out

    _with_act_data(lambda ds: _expect_refused(_attack_act(FunctionalDropout), ds))


def test_guard_refuses_a_target_stashed_during_training():
    """Audit attack 4: `loss_terms` caches the target, `forward(obs)` returns it -
    it scored 0.000000 against the old guard. A tensor living on a module outside
    its parameters and buffers is refused before scoring starts."""
    from g1_model.act import ACTPolicy

    class Stash(ACTPolicy):
        def loss_terms(self, obs, target, pad_mask, dim_mask=None):
            self._stash = target.detach()
            return super().loss_terms(obs, target, pad_mask, dim_mask)

        def forward(self, obs, actions=None, action_mask=None):
            st = getattr(self, "_stash", None)
            if actions is None and st is not None and st.shape[0] == obs.shape[0]:
                return st
            return super().forward(obs, actions, action_mask)

    def run(ds):
        m = _attack_act(Stash)
        b = T.collate_chunks([ds[i] for i in range(len(ds))])
        with torch.no_grad():
            m.loss_terms(b["obs"], b["action"], b["action_mask"])
        _expect_refused(m, ds, "_stash")
    _with_act_data(run)


def test_guard_requires_every_forward_to_pass_a_verified_latent():
    """A forward that builds its latent token without the declared module (so z
    could be anything) cannot have z = 0 verified, and is refused."""
    from g1_model.act import ACTPolicy

    class Unverifiable(ACTPolicy):
        def forward(self, obs, actions=None, action_mask=None):
            if actions is None:
                z = torch.zeros(obs.shape[0], self.cfg.latent_dim)
                lat = torch.nn.functional.linear(z, self.latent_out_proj.weight,
                                                 self.latent_out_proj.bias)
                return self.action_head(self._decode(obs, lat))
            return super().forward(obs, actions, action_mask)

    _with_act_data(lambda ds: _expect_refused(
        _attack_act(Unverifiable), ds, "could not be verified"))


def test_a_cvae_that_does_not_declare_its_latent_entry_is_refused():
    from g1_model.act import ACTPolicy

    class NoPrior(ACTPolicy):
        PRIOR_LATENT_MODULES = ()

    _with_act_data(lambda ds: _expect_refused(
        _attack_act(NoPrior), ds, "PRIOR_LATENT_MODULES"))


# ── state carried between calls: built BEFORE ACT-LSTM, which could do this ──
class _RecurrentBC(BCPolicy):
    """The un-reset ACT-LSTM bug in miniature: an LSTM whose (h, c) is kept on
    the module and carried from call to call, never reset per episode."""

    def __init__(self, **kw):
        super().__init__(obs_window=1, chunk_size=1, **kw)
        self.lstm = torch.nn.LSTM(spec.STATE_DIM, spec.STATE_DIM)
        self.hc = None

    def forward(self, obs):
        x = obs.reshape(1, obs.shape[0], -1)
        y, (h, c) = self.lstm(x, self.hc) if self.hc is not None else self.lstm(x)
        self.hc = (h.detach(), c.detach())
        return super().forward(y.reshape(obs.shape[0], 1, -1))


def test_guard_refuses_a_recurrent_state_left_over_from_training():
    """The case the user named: state carried INTO scoring. A hidden state left
    by the last training batch is refused before a single prediction."""
    def run(ds):
        torch.manual_seed(0)
        m = _RecurrentBC()
        m(torch.zeros(4, 1, spec.STATE_DIM))          # "training" leaves (h, c)
        _expect_refused(m, ds, "holds state outside its parameters", "hc")
    _with_bc_data(run)


def test_guard_refuses_a_recurrent_state_created_during_scoring():
    """Starts clean, carries state across the scoring batches: the first batch no
    longer reproduces after the pass."""
    def run(ds):
        torch.manual_seed(0)
        _expect_refused(_RecurrentBC(), ds, "carried state between calls")
    _with_bc_data(run)


def test_guard_refuses_state_kept_in_a_registered_buffer():
    """State hidden in a registered buffer, updated in place: the weights
    fingerprint moves. (The probe batch catches it too; either is enough.)"""
    class BufferState(BCPolicy):
        def __init__(self):
            super().__init__(obs_window=1, chunk_size=1)
            self.register_buffer("h", torch.zeros(spec.STATE_DIM))

        def forward(self, obs):
            out = super().forward(obs + self.h)
            self.h.add_(obs.mean(dim=(0, 1)))
            return out

    def run(ds):
        torch.manual_seed(0)
        _expect_refused(BufferState(), ds)
    _with_bc_data(run)


def test_guard_refuses_state_kept_outside_the_module():
    """State in a closure, invisible to any scan of the module: only behaviour
    can show it, and the probe batch does."""
    memory = {"n": 0}

    class ClosureState(BCPolicy):
        def forward(self, obs):
            memory["n"] += 1
            return super().forward(obs) + 1e-3 * memory["n"]

    def run(ds):
        torch.manual_seed(0)
        _expect_refused(ClosureState(obs_window=1, chunk_size=1), ds,
                        "carried state between calls")
    _with_bc_data(run)


def _with_bc_data(fn):
    got = _tiny(n_eps=1, n_ticks=20)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    try:
        fn(ds)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_honest_models_pass_the_hardened_guard_unchanged():
    """The guard must cost an honest model nothing: same score as a manual
    eval-mode pass, weights bit-identical afterwards (the NaN probe restores
    them), no RNG drawn, and the caller's train/eval mode restored."""
    from g1_model.act import build_act

    def check(m, ds):
        before = {k: v.clone() for k, v in m.state_dict().items()}
        m.train()
        rng = (torch.get_rng_state().clone(), np.random.get_state()[1].copy())
        q = _guarded_score(m, ds).value
        assert m.training, "the caller's mode must be restored"
        assert torch.equal(rng[0], torch.get_rng_state())
        assert np.array_equal(rng[1], np.random.get_state()[1])
        for k, v in m.state_dict().items():
            assert torch.equal(v, before[k]), "weight %s changed by scoring" % k
        assert abs(q - _manual_eval_l1(m, ds)) < 1e-6

    def act(ds):
        mcfg, _ = _tiny_act()
        torch.manual_seed(0)
        check(build_act(cfg=mcfg), ds)
    _with_act_data(act)
    _with_bc_data(lambda ds: check(_dropout_bc(), ds))


# ═══ TR28, third and last: the WHOLE loop config has a stated source ═════════
def test_every_train_config_field_is_classified():
    import dataclasses
    names = {f.name for f in dataclasses.fields(T.TrainConfig)}
    assert set(T.TRAIN_CONFIG_FIELDS) == names, (
        "unclassified: %s; stale: %s" % (sorted(names - set(T.TRAIN_CONFIG_FIELDS)),
                                         sorted(set(T.TRAIN_CONFIG_FIELDS) - names)))
    for f in dataclasses.fields(T.TrainConfig):
        cat, why = T.TRAIN_CONFIG_FIELDS[f.name]
        assert cat in ("model", "run", "environment", "plumbing") and why, f.name
        if cat == "model":
            assert f.default is dataclasses.MISSING, "%s is a model field with a default" % f.name
        if cat == "run":
            assert f.default in (dataclasses.MISSING, None, T.UNSTATED), (
                "%s is a run field with a real default %r" % (f.name, f.default))
            if f.default is None:        # the budget pair: exactly one must be given
                assert f.name in ("epochs", "max_steps"), f.name


def test_every_model_declares_exactly_the_model_fields():
    from g1_model.act import ACTConfig, build_act
    want = set(T.MODEL_DECLARED_FIELDS)
    assert want == {"lr", "weight_decay", "optimizer", "grad_clip"}
    bc = BCConfig(obs_window=1, chunk_size=1)
    ac = ACTConfig(obs_window=1, chunk_size=4, lr=1e-5)
    assert set(bc.optimizer_config()) == want == set(ac.optimizer_config())
    assert build_bc(cfg=bc).optimizer_config() == bc.optimizer_config()
    assert build_act(cfg=ac).optimizer_config() == ac.optimizer_config()


def test_grad_clip_is_one_value_for_every_model():
    """Identical across the ladder, from ONE constant, and a model that declares
    anything else is refused even when its TrainConfig agrees with it."""
    from g1_model.models import LADDER_GRAD_CLIP
    from g1_model.act import ACTConfig
    assert LADDER_GRAD_CLIP == 1.0
    assert BCConfig(obs_window=1, chunk_size=1).grad_clip == LADDER_GRAD_CLIP
    assert ACTConfig(obs_window=1, chunk_size=4, lr=1e-5).grad_clip == LADDER_GRAD_CLIP
    odd = BCConfig(obs_window=1, chunk_size=1, grad_clip=0.5)
    m = build_bc(cfg=odd)
    e = _raises(T.OptimizerSourceError, T.make_optimizer, m,
                T.TrainConfig(epochs=1, batch_size=8, seed=0, **odd.optimizer_config()))
    assert "ladder invariant" in str(e)


def test_grad_clip_mismatch_between_model_and_loop_is_refused():
    m = build_bc(cfg=BCConfig(obs_window=1, chunk_size=1))
    cfg = dict(_BC_OPT, grad_clip=0.25)
    e = _raises(T.OptimizerSourceError, T.make_optimizer, m,
                T.TrainConfig(epochs=1, batch_size=8, seed=0, **cfg))
    assert "grad_clip" in str(e)


def test_an_undeclared_model_is_refused_not_unchecked():
    """BC trained a whole stage with no stated source for its optimizer settings,
    because a model that declared nothing was simply not checked."""
    bare = BCPolicy(obs_window=1, chunk_size=1)
    e = _raises(T.OptimizerSourceError, T.make_optimizer, bare,
                T.TrainConfig(epochs=1, batch_size=8, seed=0, **_BC_OPT))
    assert "declares no training hyperparameters" in str(e)

    class Partial(BCPolicy):
        def optimizer_config(self):
            return dict(lr=1e-3, weight_decay=0.0, optimizer="adamw")
    e = _raises(T.OptimizerSourceError, T.make_optimizer,
                Partial(obs_window=1, chunk_size=1),
                T.TrainConfig(epochs=1, batch_size=8, seed=0, **_BC_OPT))
    assert "grad_clip" in str(e) and "incomplete" in str(e)


def test_step_budget_must_state_its_stop_rule_and_window():
    kw = dict(epochs=None, batch_size=8, seed=0, max_steps=10, **_BC_OPT)
    e = _raises(T.TrainError, T.TrainConfig, window_steps=5, **kw)
    assert "stop_rule" in str(e)
    e = _raises(T.TrainError, T.TrainConfig, stop_rule=None, **kw)
    assert "window_steps" in str(e)
    ok = T.TrainConfig(stop_rule=None, window_steps=5, **kw)
    assert ok.stop_rule is None and ok.window_steps == 5
    ep = T.TrainConfig(epochs=1, batch_size=8, seed=0, **_BC_OPT)
    assert ep.stop_rule is None and ep.window_steps is None
    _raises(T.TrainError, T.TrainConfig, epochs=1, batch_size=8, seed=0,
            window_steps=5, **_BC_OPT)


def test_every_run_records_where_its_hyperparameters_came_from():
    """The grad-clip deviation is written into the run's own metadata, with the
    declared values and the whole field policy."""
    got = _tiny(n_eps=1, n_ticks=20)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, ds = got
    out = tempfile.mkdtemp()
    try:
        cfg = T.TrainConfig(epochs=1, batch_size=16, seed=0, **_BC_OPT,
                            run_name="hp", log_every=0)
        model = T.seeded_build(cfg, build_bc, cfg=BCConfig(obs_window=1, chunk_size=1))
        T.train(model, ds, cfg, run_dir=out)
        meta = json.load(open(os.path.join(out, "metadata.json"), encoding="utf-8"))
        hp = meta["training_hyperparameters"]
        assert hp["declared"] == _BC_OPT
        assert hp["ladder_invariants"] == dict(grad_clip=1.0)
        assert "main.py:20" in hp["disclosed_deviations"]["grad_clip"]
        assert hp["field_policy"]["grad_clip"] == "model"
        assert set(hp["field_policy"]) == set(T.TRAIN_CONFIG_FIELDS)
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
