"""Tests for state-only ACT — the ones the overfit-10 gate cannot see.

    python -m g1_model.test_act

The gate measures training loss on ten episodes. It cannot see whether the latent
reaches the decoder, whether the CVAE encoder is skipped at inference, whether z
was computed partly from padding, or whether the KL is on the right scale. Every
one of those produces a model that trains beautifully and is wrong, so each gets
a test that fails loudly if the property breaks.

`test_latent_reaches_the_decoder` is the load-bearing one. It is the direct guard
against copying the reference's unreachable `backbones is None` branch
(`detr_vae.py:132-136`), which never passes `latent_input` and would give us
chunked BC wearing a CVAE costume.
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
from g1_model import act as A
from g1_model import train as T
from g1_model.act import (ACTConfig, ACTPolicy, TemporalEnsembler, act_loss,
                          build_act, kl_divergence, reparametrize)
from g1_model.loader import ChunkDataset, LoaderConfig, TrackingPolicy


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn),
                                                  exc.__name__))


#: The small test model's optimizer settings, STATED rather than inherited
#: (TR28). They match the TrainConfigs the training tests below construct, which
#: is exactly what `train.assert_optimizer_source` now enforces.
TEST_LR = 1e-4
TEST_WD = 0.0


def _cfg(K=8, W=1, **kw):
    kw.setdefault("lr", TEST_LR)
    kw.setdefault("weight_decay", TEST_WD)
    return ACTConfig(obs_window=W, chunk_size=K, hidden_dim=64,
                     dim_feedforward=128, nheads=4, enc_layers=2, dec_layers=3,
                     **kw)


def _model(K=8, W=1, **kw):
    torch.manual_seed(0)
    return ACTPolicy(_cfg(K, W, **kw)).eval()


def _batch(B=4, K=8, W=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(B, W, spec.STATE_DIM, generator=g)
    act = torch.randn(B, K, spec.ACTION_DIM, generator=g)
    pad = torch.ones(B, K, dtype=torch.bool)
    return obs, act, pad


# ═══ B1: THE LATENT REACHES THE DECODER ══════════════════════════════════════
def test_latent_reaches_the_decoder():
    """THE most important test in the session.

    Two different z, one identical observation -> different action chunks. If
    this passes trivially the CVAE is decorative and ACT is chunked BC: exactly
    what copying the reference's dead state-only branch would produce, and it
    would pass the overfit-10 gate unchanged.
    """
    m = _model()
    obs = torch.randn(3, 1, spec.STATE_DIM)
    z1 = torch.zeros(3, m.cfg.latent_dim)
    z2 = torch.full((3, m.cfg.latent_dim), 2.0)
    with torch.no_grad():
        a1 = m.action_head(m._decode(obs, m.latent_out_proj(z1)))
        a2 = m.action_head(m._decode(obs, m.latent_out_proj(z2)))
    d = float((a1 - a2).abs().max())
    assert d > 1e-4, ("the latent does NOT reach the decoder: identical output "
                      "for two different z (max diff %.3e). The model is chunked "
                      "BC with an unused encoder." % d)

    # and it must reach it through a GRADIENT path, not just numerically
    z = torch.zeros(3, m.cfg.latent_dim, requires_grad=True)
    out = m.action_head(m._decode(obs, m.latent_out_proj(z)))
    out.sum().backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0.0, \
        "no gradient flows from the action chunk back to z"


def test_the_dead_branch_signature_is_not_reproduced():
    """The reference's dead branch drops `latent_input` by passing 4 args where
    the vision path passes 7. Our `_decode` cannot express that: the latent is a
    positional argument, so a call that omits it is a TypeError, not a silently
    latent-free model."""
    import inspect
    sig = inspect.signature(ACTPolicy._decode).parameters
    assert "latent_input" in sig, sig
    assert sig["latent_input"].default is inspect.Parameter.empty, \
        "latent_input must be required, so it cannot be forgotten"
    m = _model()
    _raises(TypeError, m._decode, torch.randn(2, 1, spec.STATE_DIM))
    # the latent is the FIRST token of the encoder sequence (transformer.py:62)
    src = inspect.getsource(ACTPolicy._decode)
    assert "torch.stack([latent_input, proprio]" in src, src


# ═══ B2: the KL term ═════════════════════════════════════════════════════════
def test_kl_is_nonzero_and_responds_to_beta():
    m = _model()
    obs, act, pad = _batch()
    torch.manual_seed(1)
    _, mu, logvar = m(obs, act, pad)
    kl = kl_divergence(mu, logvar)
    assert float(kl) > 0.0, "KL collapsed to zero at initialisation"

    # ONE prediction, scored twice. Re-running the model would redraw z through
    # `reparametrize`, so the two reconstruction terms would differ and the
    # comparison would be measuring the sample, not beta.
    pred = m(obs, act, pad)[0].detach()
    l_a, l1_a, kl_a, _ = act_loss(pred, act, pad, mu, logvar, 10.0)
    l_b, l1_b, kl_b, _ = act_loss(pred, act, pad, mu, logvar, 20.0)
    assert torch.allclose(l1_a, l1_b), "same prediction must give the same L1"
    assert abs(float(kl_a) - float(kl_b)) < 1e-6, "KL itself must not depend on beta"
    assert float(l_b) > float(l_a), "doubling beta must increase the total loss"
    assert abs((float(l_b) - float(l_a)) - 10.0 * float(kl_a)) < 1e-3


def test_kl_matches_the_reference_formula():
    """reference/act/policy.py:79-80: summed over latent dims, averaged over batch.
    `mean_kld` instead of `total_kld` would rescale by the latent dimension - 32x -
    and would still train."""
    mu = torch.tensor([[0.5, -0.25], [1.0, 0.0]])
    logvar = torch.tensor([[0.1, -0.2], [0.0, 0.3]])
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    want_total = float(klds.sum(1).mean(0))
    want_mean = float(klds.mean(1).mean(0))
    got = float(kl_divergence(mu, logvar))
    assert abs(got - want_total) < 1e-6, (got, want_total)
    assert abs(got - want_mean) > 1e-6, "kl_divergence returned mean_kld, not total_kld"


def test_reparametrize_uses_half_logvar():
    """A missing `.div(2)` trains fine and halves the effective KL scale."""
    mu = torch.zeros(4096, 8)
    logvar = torch.full((4096, 8), 2.0)                # std = exp(1) = 2.718
    torch.manual_seed(0)
    s = reparametrize(mu, logvar)
    got = float(s.std())
    assert abs(got - float(np.exp(1.0))) < 0.15, (got, np.exp(1.0))


# ═══ B3: the encoder is not invoked at inference ═════════════════════════════
def test_encoder_is_not_invoked_at_inference():
    """Instrumented, not reasoned about. The reference discards the CVAE encoder
    at test time (paper §IV-B) and sets z to the prior mean, zero
    (`detr_vae.py:113`). Running it at inference would leak the ground-truth
    action chunk into the prediction - an oracle - and the gate, which only
    measures training loss, would never notice."""
    m = _model()
    obs, act, pad = _batch()
    calls = {"n": 0}
    real = m.encode

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    m.encode = counting
    with torch.no_grad():
        m(obs)                                   # inference
    assert calls["n"] == 0, "the CVAE encoder ran at inference (%d calls)" % calls["n"]
    with torch.no_grad():
        m(obs, act, pad)                         # training
    assert calls["n"] == 1, "the CVAE encoder did NOT run at training"


def test_inference_latent_is_exactly_the_prior_mean():
    m = _model()
    obs = torch.randn(2, 1, spec.STATE_DIM)
    seen = {}
    real = m.latent_out_proj.forward
    m.latent_out_proj.forward = lambda z: seen.setdefault("z", z.clone()) if False else (
        seen.update(z=z.clone()) or real(z))
    with torch.no_grad():
        m(obs)
    assert "z" in seen
    assert float(seen["z"].abs().max()) == 0.0, "inference z must be exactly zeros"
    assert seen["z"].shape == (2, m.cfg.latent_dim)


def test_inference_mode_is_structural_not_a_caller_flag():
    """A2: which mode is in force is decided by whether the action chunk was
    supplied (`detr_vae.py:85`), so a caller cannot run the encoder by mistake."""
    import inspect
    sig = inspect.signature(ACTPolicy.forward).parameters
    assert "actions" in sig and sig["actions"].default is None
    for bad in ("is_training", "training_mode", "use_encoder"):
        assert bad not in sig, "mode must not be a caller-set flag: %r" % bad
    m = _model()
    obs = torch.randn(2, 1, spec.STATE_DIM)
    assert isinstance(m(obs), torch.Tensor)                     # 1 output
    assert len(m(obs, *_batch(2)[1:])) == 3                     # 3 outputs


# ═══ B8 / D9: padding must be masked out of the latent ═══════════════════════
def test_padding_does_not_leak_into_the_latent():
    """D9, gate-blind. Same real content, different padding -> identical mu and
    logvar. Without `src_key_padding_mask` (`detr_vae.py:98-99, 104`) z is
    computed partly from zeros that are not data, and the model trains fine.

    The case is CONSTRUCTED: at K=100 only ~6.5% of elements are padded, so a
    random batch is very unlikely to contain a boundary sample.
    """
    m = _model(K=8)
    obs = torch.randn(1, 1, spec.STATE_DIM)
    real = torch.randn(1, 5, spec.ACTION_DIM)

    a1 = torch.zeros(1, 8, spec.ACTION_DIM)
    a1[:, :5] = real                                    # 5 real, 3 zero pads
    p1 = torch.zeros(1, 8, dtype=torch.bool); p1[:, :5] = True

    a2 = torch.randn(1, 8, spec.ACTION_DIM)             # same 5 real...
    a2[:, :5] = real                                    # ...different garbage after
    p2 = p1.clone()

    m.eval()
    with torch.no_grad():
        mu1, lv1 = m.encode(obs, a1, p1)
        mu2, lv2 = m.encode(obs, a2, p2)
    assert torch.allclose(mu1, mu2, atol=1e-6), \
        "padding leaked into mu: max diff %.3e" % float((mu1 - mu2).abs().max())
    assert torch.allclose(lv1, lv2, atol=1e-6), \
        "padding leaked into logvar: max diff %.3e" % float((lv1 - lv2).abs().max())

    # ...and the test is not vacuous: WITHOUT the mask the two differ
    with torch.no_grad():
        mu3, _ = m.encode(obs, a1, None)
        mu4, _ = m.encode(obs, a2, None)
    assert not torch.allclose(mu3, mu4, atol=1e-6), \
        "unmasked encoding gave the same mu, so this test proves nothing"


def test_cls_and_state_tokens_are_never_masked():
    """`detr_vae.py:98-99` prepends two False. Masking [CLS] destroys the latent."""
    import inspect
    src = inspect.getsource(ACTPolicy.encode)
    assert "torch.zeros((B, 2)" in src and "dtype=torch.bool" in src, src
    m = _model(K=4)
    obs = torch.randn(1, 1, spec.STATE_DIM)
    a = torch.randn(1, 4, spec.ACTION_DIM)
    allpad = torch.zeros(1, 4, dtype=torch.bool)        # every action padded
    with torch.no_grad():
        mu, lv = m.encode(obs, a, allpad)               # must not NaN
    assert torch.isfinite(mu).all() and torch.isfinite(lv).all()


# ═══ architecture fidelity ═══════════════════════════════════════════════════
def test_encoder_sequence_is_cls_state_then_chunk():
    """`detr_vae.py:95`, length K+2 (paper §IV-C)."""
    import inspect
    src = inspect.getsource(ACTPolicy.encode)
    assert "torch.cat([cls, qpos_embed, action_embed], dim=1)" in src, src
    m = _model(K=8)
    assert m.pos_table.shape == (1, 8 + 2, m.cfg.hidden_dim), m.pos_table.shape


def test_positional_table_is_a_fixed_buffer_not_learned():
    """`register_buffer` at `detr_vae.py:72`, `.clone().detach()` at `:101`."""
    m = _model(K=8)
    names = {n for n, _ in m.named_parameters()}
    assert not any("pos_table" in n for n in names), "pos_table is learned"
    assert "pos_table" in dict(m.named_buffers())
    before = m.pos_table.clone()
    obs, a, p = _batch(2, 8)
    m(obs, a, p)[0].sum().backward()
    assert torch.equal(before, m.pos_table), "pos_table changed during a step"


def test_latent_is_read_from_the_cls_token_only():
    """`encoder_output[0]` at `detr_vae.py:105`. Taking the mean over tokens, or
    the last token, still trains."""
    import inspect
    src = inspect.getsource(ACTPolicy.encode)
    assert "self.latent_proj(out[0])" in src, src


def test_queries_are_learned_and_number_k():
    """`nn.Embedding(num_queries, hidden)` at `detr_vae.py:54`; `num_queries =
    chunk_size` at `imitate_episodes.py:58`. The PAPER (§IV-C) calls them a
    "fixed position embedding" - code and paper disagree and we follow the code."""
    m = _model(K=13)
    assert isinstance(m.query_embed, torch.nn.Embedding)
    assert m.query_embed.weight.shape == (13, m.cfg.hidden_dim)
    assert m.query_embed.weight.requires_grad
    import inspect
    assert "torch.zeros_like(query)" in inspect.getsource(ACTPolicy._decode)


def test_two_encoder_tokens_and_a_two_entry_position_embedding():
    """Correspondence row 4, option A: the non-image part of the reference's
    sequence is exactly [latent, proprio], evidenced by
    `additional_pos_embed = nn.Embedding(2, hidden)` (`detr_vae.py:76`)."""
    m = _model()
    assert m.additional_pos_embed.weight.shape == (2, m.cfg.hidden_dim)
    seen = {}
    real = m.t_encoder.forward
    m.t_encoder.forward = lambda src, **k: seen.update(shape=tuple(src.shape)) or real(src, **k)
    with torch.no_grad():
        m(torch.randn(5, 1, spec.STATE_DIM))
    assert seen["shape"] == (2, 5, m.cfg.hidden_dim), seen


def test_decoder_reads_index_zero_and_deeper_layers_get_no_gradient():
    """The reference builds seven decoder layers and reads `hs[0]`
    (`detr_vae.py:131`). We build ONE (superseding D4) because layers 2..n get no
    gradient - reproduced here on a multi-layer model, measured not assumed - and
    removing them is bit-identical (next test)."""
    assert A.REF_DEC_LAYERS == 7, "the reference's own depth stays on record"
    assert A.DEC_LAYERS == 1 and A.ENC_LAYERS == 4
    assert A.DECODER_LAYER_READ == 0
    m = _model(K=4)
    assert len(m.t_decoder.layers) == m.cfg.dec_layers
    assert len(m.t_encoder.layers) == m.cfg.enc_layers
    assert len(m.encoder.layers) == m.cfg.enc_layers
    import inspect
    assert "hs[DECODER_LAYER_READ]" in inspect.getsource(ACTPolicy._decode)
    # and the reference's own consequence is reproduced, measured not assumed
    m.zero_grad()
    out = m.action_head(m._decode(torch.randn(2, 1, spec.STATE_DIM),
                                  torch.randn(2, m.cfg.hidden_dim)))
    out.sum().backward()
    g = [float(l.linear1.weight.grad.abs().sum()) if l.linear1.weight.grad is not None
         else None for l in m.t_decoder.layers]
    assert g[0] and g[0] > 0.0, g
    assert all((x == 0.0) for x in g[1:]), \
        "expected the reference's hs[0] behaviour: only layer 1 trains, got %s" % g


def test_post_norm_and_relu_by_default():
    assert A.PRE_NORM is False and A.ACTIVATION == "relu"
    m = _model()
    assert m.encoder.norm is None, "post-norm encoder must have no final norm"
    assert m.t_decoder.norm is not None, "decoder always has a final LayerNorm"


def test_reference_constants_match_the_pinned_commit():
    assert A.REF_COMMIT == "742c753c0d4a5d87076c8f69e5628c79a8cc5488"
    assert (A.LATENT_DIM, A.HIDDEN_DIM, A.DIM_FEEDFORWARD, A.NHEADS) == (32, 512, 3200, 8)
    assert (A.DROPOUT, A.KL_WEIGHT, A.TEMPORAL_ENSEMBLE_M) == (0.1, 10.0, 0.01)


def test_is_pad_head_is_dropped():
    """Row 33: the reference computes it (`detr_vae.py:138`) and never uses it."""
    m = _model()
    assert not hasattr(m, "is_pad_head")


def test_shape_contract_and_rejections():
    m = _model(K=8, W=1)
    assert m(torch.randn(3, 1, spec.STATE_DIM)).shape == (3, 8, spec.ACTION_DIM)
    _raises(ValueError, m, torch.randn(3, 2, spec.STATE_DIM))       # wrong W_o
    _raises(ValueError, m, torch.randn(3, 1, 46))                   # wrong state dim
    obs, a, p = _batch(2, K=7)
    _raises(ValueError, m, obs, a, p)                               # wrong K
    _raises(ValueError, ACTConfig, obs_window=0, chunk_size=1, lr=TEST_LR)
    _raises(ValueError, ACTConfig, obs_window=1, chunk_size=0, lr=TEST_LR)
    _raises(ValueError, ACTConfig, obs_window=1, chunk_size=1, hidden_dim=10, nheads=4,
            lr=TEST_LR)


# ═══ B5: the loss reuses Stage 2's, masks apply ══════════════════════════════
def test_reconstruction_term_is_the_shared_masked_l1():
    """D6: one reconstruction loss across every stage. A second implementation
    would put an unknown offset inside every cross-model comparison."""
    import inspect
    src = inspect.getsource(A.act_loss)
    assert "from g1_model.train import masked_l1" in src and "masked_l1(" in src
    m = _model(K=6)
    obs, a, p = _batch(3, 6)
    pred, mu, lv = m(obs, a, p)
    loss, l1, kl, n = act_loss(pred, a, p, mu, lv, 10.0)
    want, wn = T.masked_l1(pred, a, p)
    assert torch.allclose(l1, want) and int(n) == int(wn)
    assert abs(float(loss) - (float(l1) + 10.0 * float(kl))) < 1e-4


def test_masked_dims_and_padding_are_ignored_by_the_reconstruction_term():
    """B5, reusing Stage 2's guarantees rather than duplicating their tests."""
    m = _model(K=5)
    obs, a, p = _batch(2, 5)
    pred, mu, lv = m(obs, a, p)
    exact = a.clone()
    for i in spec.CONSTANT_ACTION_DIMS:
        exact[:, :, i] = pred[:, :, i].detach() + 9.0     # wrong only where masked
    _, l1, _, _ = act_loss(pred.detach(), exact, p, mu, lv, 10.0)
    _, l1_ref, _, _ = act_loss(pred.detach(), a, p, mu, lv, 10.0)
    trainable = np.flatnonzero(np.asarray(spec.ACTION_MASK))
    assert torch.allclose(exact[:, :, trainable], a[:, :, trainable])
    assert abs(float(l1) - float(l1_ref)) < 1e-6, "masked dims changed the loss"

    nopad = torch.zeros(2, 5, dtype=torch.bool)
    _, l1z, _, nz = act_loss(pred.detach(), a, nopad, mu, lv, 10.0)
    assert float(l1z) == 0.0 and int(nz) == 0


# ═══ B4 / A3: temporal ensembling ════════════════════════════════════════════
def test_temporal_ensemble_weights_match_the_reference_expression():
    """`imitate_episodes.py:256-257`: exp(-m*arange(n)), then normalised."""
    for n in (1, 3, 10):
        w = TemporalEnsembler.weights(n, 0.01)
        ref = np.exp(-0.01 * np.arange(n)); ref = ref / ref.sum()
        assert np.allclose(w, ref), (w, ref)
        assert abs(w.sum() - 1.0) < 1e-12
    w = TemporalEnsembler.weights(5, 0.01)
    assert w[0] > w[-1], "index 0 must weight the OLDEST prediction most"
    assert A.TEMPORAL_ENSEMBLE_M == 0.01


def test_temporal_ensemble_changes_inference_and_never_training():
    e = TemporalEnsembler(chunk_size=4, action_dim=spec.ACTION_DIM)
    g = torch.Generator().manual_seed(0)
    chunks = [torch.randn(4, spec.ACTION_DIM, generator=g) for _ in range(3)]
    outs = [e.step(c) for c in chunks]
    assert all(o.shape == (spec.ACTION_DIM,) for o in outs)
    assert torch.allclose(outs[0], chunks[0][0]), "first tick has one prediction"
    assert not torch.allclose(outs[2], chunks[2][0]), \
        "with three overlapping chunks the ensemble must differ from the newest"

    e.reset()
    assert e.t == 0 and not e._buf, "reset must clear the buffer between episodes"

    # it is inference-only: nothing in the training path mentions it
    import inspect
    assert "TemporalEnsembler" not in inspect.getsource(ACTPolicy.loss_terms)
    assert "TemporalEnsembler" not in inspect.getsource(T.train)
    assert not any(p.requires_grad for p in []) and \
        not hasattr(TemporalEnsembler, "parameters"), "ensembler holds no parameters"


# ═══ B6 / D10: determinism ═══════════════════════════════════════════════════
def test_seeded_build_makes_act_deterministic():
    """D10. Stage 2 measured that seeding AFTER construction leaves weight init
    unseeded (6.0e-3 divergence); this is a new class, so it is re-checked."""
    cfg = T.TrainConfig(epochs=1, batch_size=4, lr=1e-4, seed=5)
    a = T.seeded_build(cfg, build_act, cfg=_cfg())
    b = T.seeded_build(cfg, build_act, cfg=_cfg())
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    torch.manual_seed(999)
    c = build_act(_cfg())
    assert not all(torch.equal(pa, pc) for pa, pc in zip(a.parameters(),
                                                          c.parameters())), \
        "an unseeded build must differ, or this test proves nothing"


def test_two_identical_act_runs_are_bit_identical():
    from g1_model.test_train import _tiny
    tmp = _tiny(n_eps=2, n_ticks=40)
    if tmp is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, _ = tmp
    out = tempfile.mkdtemp()
    try:
        st, ac = (spec.NormStats.identity("state"), spec.NormStats.identity("action"))
        ds = ChunkDataset.from_directory(
            tmp, LoaderConfig(chunk_size=6, obs_window=1, tracking=TrackingPolicy()),
            st, ac, None)
        losses = []
        for i in (1, 2):
            cfg = T.TrainConfig(epochs=2, batch_size=8, lr=1e-4, seed=3,
                                run_name="act%d" % i, log_every=0)
            m = T.seeded_build(cfg, build_act, cfg=_cfg(K=6))
            r = T.train(m, ds, cfg, run_dir=os.path.join(out, "r%d" % i))
            losses.append([x["train_loss"] for x in r["history"]])
        assert losses[0] == losses[1], losses
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_act_trains_through_the_unchanged_shared_loop():
    """The loop is model-agnostic: ACT plugs in through `loss_terms` and nothing
    in `train.py` imports ACT."""
    # The loop must not know ACT exists: check the IMPORTS, not the prose - the
    # hook's own docstring names ACT as an example and a word-search hits it.
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(T))
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add(n.module or "")
    assert not any("act" in m.split(".")[-1] for m in imported), imported
    assert "loss_terms" in inspect.getsource(T.forward_loss)
    from g1_model.test_train import _tiny
    tmp = _tiny(n_eps=2, n_ticks=40)
    if tmp is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, _ = tmp
    out = tempfile.mkdtemp()
    try:
        st, ac = (spec.NormStats.identity("state"), spec.NormStats.identity("action"))
        ds = ChunkDataset.from_directory(
            tmp, LoaderConfig(chunk_size=6, obs_window=1, tracking=TrackingPolicy()),
            st, ac, None)
        cfg = T.TrainConfig(epochs=3, batch_size=8, lr=1e-4, seed=0,
                            run_name="actloop", log_every=0)
        m = T.seeded_build(cfg, build_act, cfg=_cfg(K=6))
        r = T.train(m, ds, cfg, run_dir=out)
        assert r["history"][-1]["train_loss"] < r["history"][0]["train_loss"]
        # the KL is logged separately, so a collapsed encoder is visible
        assert "train_kl" in r["history"][0], r["history"][0].keys()
        assert r["history"][0]["train_kl"] > 0.0
        assert "recon_l1" in r["history"][0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_checkpoint_round_trips_to_identical_predictions():
    from g1_model.test_train import _tiny
    tmp = _tiny(n_eps=2, n_ticks=30)
    if tmp is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, _ = tmp
    out = tempfile.mkdtemp()
    try:
        st, ac = (spec.NormStats.identity("state"), spec.NormStats.identity("action"))
        ds = ChunkDataset.from_directory(
            tmp, LoaderConfig(chunk_size=6, obs_window=1, tracking=TrackingPolicy()),
            st, ac, None)
        cfg = T.TrainConfig(epochs=1, batch_size=8, lr=1e-4, seed=1,
                            run_name="actck", log_every=0)
        m = T.seeded_build(cfg, build_act, cfg=_cfg(K=6))
        T.train(m, ds, cfg, run_dir=out)
        obs = torch.stack([ds[i]["obs"] for i in range(4)])
        m = m.cpu().eval()
        with torch.no_grad():
            before = m(obs).clone()
        back, ck = T.load_checkpoint(os.path.join(out, "last.pt"), build_act)
        with torch.no_grad():
            after = back(obs)
        assert torch.equal(before, after), float((before - after).abs().max())
        assert ck["model_cls"] == "ACTPolicy"
        assert ck["spec_version"] == spec.SPEC_VERSION
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


# ═══ D8: no dead flag ════════════════════════════════════════════════════════
def test_there_is_no_dead_use_lstm_flag():
    """D8: a flag that does nothing is this repo's standing failure. ACT-LSTM
    must live in the SAME class when it is built - not as an inert switch today."""
    import inspect
    src = inspect.getsource(A)
    tree_names = {n.id for n in __import__("ast").walk(__import__("ast").parse(src))
                  if isinstance(n, __import__("ast").Name)}
    assert "use_lstm" not in tree_names
    assert "use_lstm" not in {f for f in ACTConfig.__dataclass_fields__}
    assert not any(isinstance(mod, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN))
                   for mod in _model().modules())


# ═══ TR28: ACT states its own optimizer config, and a borrowed one RAISES ═══════
def test_unstated_learning_rate_is_an_error_not_an_inheritance():
    """TR28. The Stage-4 gate ran ACT at lr 1e-3 because the shared runner drew
    `lr` from `BCConfig`. An ACTConfig with no lr must not construct at all."""
    import inspect
    p = inspect.signature(ACTConfig).parameters["lr"]
    assert p.default is inspect.Parameter.empty, "ACTConfig.lr must have no default"
    _raises(TypeError, ACTConfig, obs_window=1, chunk_size=4)


def test_act_config_cannot_be_satisfied_by_bc_config_values():
    """The exact leak, reproduced and refused: a TrainConfig assembled from
    BCConfig's optimizer values must be rejected for an ACT model."""
    from g1_model.models import BCConfig
    bc = BCConfig(obs_window=1, chunk_size=4)
    m = ACTPolicy(_cfg(K=4, lr=A.LR, weight_decay=A.WEIGHT_DECAY))
    leaked = T.TrainConfig(epochs=1, batch_size=8, lr=bc.lr,
                           weight_decay=bc.weight_decay, optimizer=bc.optimizer)
    assert bc.lr != A.LR, "the test is vacuous if the two lrs happen to agree"
    e = _raises(T.OptimizerSourceError, T.make_optimizer, m, leaked)
    assert "lr" in str(e) and "TR28" in str(e)
    ok = T.TrainConfig(epochs=1, batch_size=8, **m.optimizer_config())
    T.make_optimizer(m, ok)                           # its own values: accepted


def test_every_declared_optimizer_value_is_checked_not_just_lr():
    """If lr leaked, others could. weight_decay DID - the old runner handed ACT
    0.0 while the reference is 1e-4 - so the guard compares every declared value."""
    m = ACTPolicy(_cfg(K=4, lr=A.LR, weight_decay=A.WEIGHT_DECAY))
    good = m.optimizer_config()
    for field_, bad in (("lr", 1e-3), ("weight_decay", 0.0), ("optimizer", "sgd")):
        cfg = dict(good); cfg[field_] = bad
        e = _raises(T.OptimizerSourceError, T.make_optimizer, m,
                    T.TrainConfig(epochs=1, batch_size=8, **cfg))
        assert field_ in str(e), (field_, str(e))


def test_reference_optimizer_values_are_cited_constants():
    """README.md:77 --lr 1e-5; main.py:17 --weight_decay 1e-4; main.py:87 AdamW."""
    assert A.LR == 1e-5 and A.WEIGHT_DECAY == 1e-4 and A.OPTIMIZER == "adamw"
    c = ACTConfig(obs_window=1, chunk_size=4, lr=A.LR)
    assert c.optimizer_config() == dict(lr=1e-5, weight_decay=1e-4, optimizer="adamw")


def test_the_runner_takes_act_lr_from_act_not_from_bc():
    """The runner is where the leak happened, so the runner is tested directly."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tools"))
    import importlib
    tb = importlib.import_module("train_bc")
    gate = tb._model_config("act", 100, 8, gate=True)
    assert isinstance(gate, ACTConfig)
    assert gate.lr == A.LR, "the runner must use ACT's own lr"
    assert gate.weight_decay == 0.0, "the gate is unregularised for every model"
    full = tb._model_config("act", 100, 8, gate=False)
    assert full.weight_decay == A.WEIGHT_DECAY, "outside the gate: the reference's"
    tc = tb._train_config(gate, 12, 8, seed=0)
    assert (tc.lr, tc.weight_decay, tc.optimizer) == (A.LR, 0.0, "adamw")
    m = ACTPolicy(_cfg(K=4, lr=gate.lr, weight_decay=gate.weight_decay))
    T.assert_optimizer_source(m, tc)                  # consistent: no raise


def test_epoch_budget_must_be_stated_with_a_reason():
    """TR28's family (A2). BC's gate ran 300 epochs because 300 was the argparse
    default; ACT's ran 12 because a wall-clock cap was typed on the command line.
    Neither run recorded why. Both --epochs and --budget-reason are now REQUIRED
    and go into metadata beside the optimizer steps they amount to."""
    import subprocess
    tool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tools", "train_bc.py")
    for extra in ([], ["--epochs", "3"]):
        r = subprocess.run([sys.executable, tool, "overfit10", "--model", "act"] + extra,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode != 0, extra
        assert "--budget-reason" in r.stderr, r.stderr
    sys.path.insert(0, os.path.dirname(tool))
    import importlib, types
    tb = importlib.import_module("train_bc")
    ns = types.SimpleNamespace(epochs=12, batch=8, budget_reason="identical to the run compared")
    b = tb._budget(ns, list(range(7628)))
    assert b == dict(epoch_budget=12, steps_per_epoch=954, optimizer_steps=11448,
                     epoch_budget_reason="identical to the run compared"), b


# ═══ B: one decoder layer is the reference's seven, minus dead computation ════
def _pair(dropout):
    """7-layer and 1-layer models built from the SAME seed, NO weight copying."""
    kw = dict(obs_window=1, chunk_size=10, lr=TEST_LR, weight_decay=TEST_WD,
              hidden_dim=64, dim_feedforward=128, nheads=4, enc_layers=2,
              dropout=dropout)
    torch.manual_seed(0)
    m7 = ACTPolicy(ACTConfig(dec_layers=7, **kw))
    torch.manual_seed(0)
    m1 = ACTPolicy(ACTConfig(dec_layers=1, **kw))
    return m7, m1


def test_one_decoder_layer_is_bit_identical_to_the_references_seven():
    """What lets Chapter 3 state the equivalence as MEASURED.

    Same seed, no weight copying. Every parameter the 1-layer model has is
    identical to its namesake in the 7-layer model - construction consumes the
    RNG identically up to the point the dead layers begin - and then the forward
    output, the loss and every gradient over the shared parameters are equal to
    the bit. This runs in TRAIN mode with ACT's real dropout (0.1), not in a
    sanitised setting: for a given random state, layers 2-7 change nothing.
    """
    m7, m1 = _pair(dropout=A.DROPOUT)
    sd7 = m7.state_dict()
    for k, v in m1.state_dict().items():
        assert torch.equal(v, sd7[k]), "shared tensor %s differs at construction" % k
    extra = set(sd7) - set(m1.state_dict())
    assert extra and all(".layers." in k and not k.split(".layers.")[1].startswith("0.")
                         for k in extra if k.startswith("t_decoder.")),         "the 7-layer model may differ ONLY by decoder layers 2-7"

    obs = torch.randn(4, 1, spec.STATE_DIM)
    tgt = torch.randn(4, 10, spec.ACTION_DIM)
    pad = torch.ones(4, 10, dtype=torch.bool)
    pad[1, 6:] = False
    m7.train(); m1.train()
    torch.manual_seed(7); p7, mu7, lv7 = m7(obs, tgt, pad)
    torch.manual_seed(7); p1, mu1, lv1 = m1(obs, tgt, pad)
    assert torch.equal(p7, p1), "forward output differs"
    assert torch.equal(mu7, mu1) and torch.equal(lv7, lv1), "latent differs"
    l7 = act_loss(p7, tgt, pad, mu7, lv7, A.KL_WEIGHT)[0]
    l1 = act_loss(p1, tgt, pad, mu1, lv1, A.KL_WEIGHT)[0]
    assert torch.equal(l7, l1), "loss differs"
    l7.backward(); l1.backward()
    g7 = dict(m7.named_parameters())
    for name, p in m1.named_parameters():
        a, b = p.grad, g7[name].grad
        assert (a is None) == (b is None), name
        if a is not None:
            assert torch.equal(a, b), "gradient differs for %s" % name
    m7.eval(); m1.eval()
    with torch.no_grad():
        assert torch.equal(m7(obs), m1(obs)), "inference output differs"


def test_dropout_rng_makes_multistep_training_differ_not_the_function():
    """The honest limit of the claim above - pinned so Chapter 3 cannot overstate it.

    Layers 2-7 compute nothing the output reads, but they still RUN, and their
    dropout draws random numbers. That shifts every LATER dropout mask. So:
      dropout 0.0 -> training is bit-identical at every step;
      dropout 0.1 -> identical at step 1, different from step 2 on.
    The two models compute the same function; training them with dropout does
    not produce the same weights step for step. Statistically equivalent, not
    bit-identical over training - and that is the sentence the thesis may use.
    """
    def losses(dropout, dec, steps=4):
        m7, m1 = _pair(dropout)
        m = m7 if dec == 7 else m1
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(1)
        torch.manual_seed(42)
        out = []
        for _ in range(steps):
            obs = torch.randn(4, 1, spec.STATE_DIM, generator=g)
            tgt = torch.randn(4, 10, spec.ACTION_DIM, generator=g)
            pad = torch.ones(4, 10, dtype=torch.bool)
            p, mu, lv = m(obs, tgt, pad)
            loss = act_loss(p, tgt, pad, mu, lv, A.KL_WEIGHT)[0]
            opt.zero_grad(); loss.backward(); opt.step()
            out.append(float(loss.detach()))
        return out
    assert losses(0.0, 7) == losses(0.0, 1), "without dropout, every step must match"
    a, b = losses(0.1, 7), losses(0.1, 1)
    assert a[0] == b[0], "step 1 must match even with dropout"
    assert a[1:] != b[1:], ("with dropout the runs must diverge after step 1 - if "
                            "they do not, this test no longer describes the code")


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
