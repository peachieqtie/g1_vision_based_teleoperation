"""Tests for the neighbour-ambiguity reference and the gate built on it.

    python -m g1_model.test_ambiguity

The load-bearing test is `test_observation_matrix_matches_getitem`. The
measurement builds its own vectorized view of the data for speed; if that view
disagrees with what the loader hands the model, the reference describes a space
no model ever sees, the gate compares two different things, and nothing about
either number is wrong-looking.

The rest pin the properties the gate's meaning depends on: a dataset where the
observation determines the action exactly must measure ~0, one where identical
observations carry different actions must measure the difference, and the
reference must MOVE with the loader configuration - a fixed number would silently
favour whichever stage happened to match it.
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
from g1_model import ambiguity as AMB
from g1_model.loader import ChunkDataset, LoaderConfig, TrackingPolicy


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn),
                                                  exc.__name__))


def _write(tmp, seed, states, actions):
    """A real .npz with real metadata and the arrays we specify."""
    src = sorted(p for p in DS._npz(DS.SYNTHETIC))
    if not src:
        return None
    with np.load(src[0], allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    n = states.shape[0]
    meta.update(seed=int(seed), n_ticks=n,
                box_spawn_xy=[1.50, -0.10 + 0.01 * seed])
    p = os.path.join(tmp, "ep_seed%04d.npz" % seed)
    np.savez_compressed(
        p, states=states.astype(np.float32), actions=actions.astype(np.float32),
        phase_labels=np.zeros(n, dtype=np.int8),
        gait_phase=np.zeros((n, 2), dtype=np.float32),
        qpos=np.zeros((n, 45), dtype=np.float32),
        qvel=np.zeros((n, 43), dtype=np.float32),
        step_index=(np.arange(n) * 20).astype(np.int64),
        meta=np.array(json.dumps(meta)))
    return p


def _ds(tmp, K, W):
    st, ac = (spec.NormStats.identity("state"), spec.NormStats.identity("action"))
    return ChunkDataset.from_directory(
        tmp, LoaderConfig(chunk_size=K, obs_window=W, tracking=TrackingPolicy()),
        st, ac, None)


def _staged(K, W, seeds=None):
    st, ac = (spec.NormStats.identity("state"), spec.NormStats.identity("action"))
    eps = DS.scan(DS.SYNTHETIC)
    if not eps:
        return None
    if seeds is not None:
        eps = [e for e in eps if e.seed in set(seeds)]
    return ChunkDataset(eps, LoaderConfig(chunk_size=K, obs_window=W,
                                          tracking=TrackingPolicy()),
                        st, ac, None)


# ─── the fast path must equal the loader ──────────────────────────────────────
def test_observation_matrix_matches_getitem():
    """If these disagree the reference measures a space no model ever sees."""
    for W in (1, 3, 8):
        ds = _staged(K=5, W=W, seeds=[0, 1])
        if ds is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        X = AMB.observation_matrix(ds)
        assert X.shape == (len(ds), W * spec.STATE_DIM), X.shape
        idx = [0, 1, W, 17, len(ds) // 2, len(ds) - 1]
        for i in idx:
            want = ds[i]["obs"].numpy().reshape(-1)
            assert np.array_equal(X[i], want), (W, i)


def test_action_chunks_match_getitem():
    for K in (1, 4, 30):
        ds = _staged(K=K, W=1, seeds=[0, 1])
        if ds is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        A, M = AMB.action_chunks(ds)
        assert A.shape == (len(ds), K, spec.ACTION_DIM)
        assert M.shape == (len(ds), K)
        for i in (0, 1, len(ds) // 2, len(ds) - 1, len(ds) - 2):
            s = ds[i]
            assert np.array_equal(A[i], s["action"].numpy()), (K, i)
            assert np.array_equal(M[i], s["action_mask"].numpy()), (K, i)


# ─── the measurement means what it says ───────────────────────────────────────
def test_a_deterministic_dataset_measures_near_zero():
    """Observation determines action exactly -> the reference must be ~0, which
    is the case where "train to near zero" WOULD have been the right gate.

    The samples lie on a DENSE 1-D path, not a random cloud. In a sparse 47-D
    cloud even a perfectly deterministic map measures large, because the nearest
    neighbour is genuinely far away - which is limit 2 of the estimator, not a
    property of the map, and is why this test has to control density to mean
    what it claims.
    """
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(0)
        n = 200
        t = np.linspace(0.0, 1.0, n)[:, None]
        S = np.repeat(t, spec.STATE_DIM, axis=1)          # dense, closely spaced
        Wm = rng.normal(size=(spec.STATE_DIM, spec.ACTION_DIM)) * 0.1
        A = S @ Wm                                        # deterministic in S
        if _write(tmp, 0, S, A) is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        r = AMB.neighbour_ambiguity(_ds(tmp, K=1, W=1))
        scale = float(np.abs(A).mean())
        assert r.mean < 0.05 * scale, (r.mean, scale)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_identical_observations_with_different_actions_measure_the_difference():
    """The case the gate exists for: the input cannot tell the samples apart, so
    the difference in their actions is irreducible for ANY architecture.

    Exactly TWO copies of each state, carrying different actions. A third copy
    would let a sample's nearest neighbour be another copy with the SAME action
    - the distance-0 tie is broken by index - and the measurement would come out
    as a blend of 0.0 and 0.5 rather than the difference being tested.
    """
    tmp = tempfile.mkdtemp()
    try:
        half = 20
        base = np.repeat(np.arange(half, dtype=float)[:, None],
                         spec.STATE_DIM, axis=1)
        S = np.vstack([base, base])                       # each state twice
        A = np.zeros((2 * half, spec.ACTION_DIM))
        A[half:, :] = 0.5                                 # the twin differs by 0.5
        if _write(tmp, 0, S, A) is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        r = AMB.neighbour_ambiguity(_ds(tmp, K=1, W=1))
        assert abs(r.mean - 0.5) < 1e-5, r.mean
        assert r.neighbour_distance["p50"] == 0.0, "duplicates must be distance 0"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reduction_is_element_weighted_like_the_loss():
    """The reference and the training loss must be in the same units, or the
    ratio between them means nothing."""
    ds = _staged(K=20, W=1, seeds=[0, 1])
    if ds is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    r = AMB.neighbour_ambiguity(ds)
    # padded steps are excluded on BOTH sides, so the element count is below the
    # naive N*K*16
    assert r.elements < len(ds) * ds.cfg.chunk_size * int(spec.ACTION_MASK.sum())
    assert r.elements > 0 and r.samples == len(ds)
    assert 0.0 < r.mean < 100.0


def test_masked_action_dims_never_contribute():
    """The 6 constant dims are excluded by measurement; if they leaked in, the
    reference would be diluted by dims no model is scored on."""
    tmp = tempfile.mkdtemp()
    try:
        n = 30
        rng = np.random.default_rng(3)
        S = rng.normal(size=(n, spec.STATE_DIM))
        A = np.zeros((n, spec.ACTION_DIM))
        for i in spec.CONSTANT_ACTION_DIMS:
            A[:, i] = rng.normal(size=n) * 50.0      # enormous, and masked out
        if _write(tmp, 0, S, A) is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        _write(tmp, 1, rng.normal(size=(n, spec.STATE_DIM)), A.copy())
        r = AMB.neighbour_ambiguity(_ds(tmp, K=1, W=1))
        assert r.mean == 0.0, ("masked dims leaked into the reference", r.mean)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── it must move with the configuration ──────────────────────────────────────
def test_reference_moves_with_obs_window():
    """The reference must be computed PER CONFIGURATION. A fixed number would
    silently favour whichever stage happened to match it."""
    vals = []
    for W in (1, 2, 4, 8):
        ds = _staged(K=1, W=W, seeds=[0, 1, 2])
        if ds is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        r = AMB.neighbour_ambiguity(ds)
        assert r.obs_window == W
        vals.append(r.mean)
    assert vals[0] > 0
    assert len(set(vals)) == len(vals), "the reference did not move with W_o"


def test_the_obs_window_dimensionality_confound_is_real_and_visible():
    """Limit 6, pinned by measurement rather than left as prose.

    A longer window cannot destroy information, so if ambiguity RISES with W_o
    the cause is the estimator: the space grows by a factor of W_o while the
    sample count does not, neighbours get relatively farther, and their actions
    differ more for reasons unrelated to what the window explains. The rise and
    the growing neighbour distance must both be observable, because the curve is
    only readable by someone who can see the confound alongside it.
    """
    dist, amb = [], []
    for W in (1, 4, 16):
        ds = _staged(K=1, W=W, seeds=[0, 1, 2])
        if ds is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        r = AMB.neighbour_ambiguity(ds)
        dist.append(r.neighbour_distance["p50"])
        amb.append(r.mean)
    assert dist[0] < dist[1] < dist[2], (
        "neighbour distance must grow with the window's dimensionality: %s" % dist)
    assert dist[2] > 3.0 * dist[0], dist
    # and on this data the confound wins past the minimum
    assert amb[2] > amb[0], (
        "expected the documented upward turn on scripted data: %s" % amb)


def test_reference_changes_with_chunk_size():
    """Predicting K actions from one observation is a harder question than
    predicting 1, so the reference must be computed per configuration."""
    a = _staged(K=1, W=1, seeds=[0, 1, 2])
    if a is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    b = _staged(K=50, W=1, seeds=[0, 1, 2])
    ra, rb = AMB.neighbour_ambiguity(a), AMB.neighbour_ambiguity(b)
    assert rb.mean > ra.mean, (ra.mean, rb.mean)
    assert ra.chunk_size == 1 and rb.chunk_size == 50


def test_reference_falls_as_episodes_are_added():
    """Limit 2 in the module docstring, measured rather than asserted: the
    reference is a property of data DENSITY, so it is not comparable across
    datasets of different size."""
    few = _staged(K=1, W=1, seeds=[0, 1, 2])
    if few is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    many = _staged(K=1, W=1)
    if len(many.lengths) <= len(few.lengths):
        print("      (skipped: need more staged episodes)")
        return
    rf, rm = AMB.neighbour_ambiguity(few), AMB.neighbour_ambiguity(many)
    assert rm.mean < rf.mean, (rf.mean, rm.mean)


def test_excluding_the_same_episode_is_available_and_raises_the_reference():
    """The first question anyone asks - is the neighbour just the next tick? -
    is answerable by measurement rather than argument."""
    ds = _staged(K=1, W=1, seeds=[0, 1, 2])
    if ds is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    a = AMB.neighbour_ambiguity(ds, exclude_same_episode=False)
    b = AMB.neighbour_ambiguity(ds, exclude_same_episode=True)
    assert b.mean > a.mean, (a.mean, b.mean)
    assert a.exclude_same_episode is False and b.exclude_same_episode is True


# ─── provenance and the verdict ───────────────────────────────────────────────
def test_citation_carries_the_provenance():
    """Limit 3: a figure quoted without its dataset and episode count cannot be
    checked, so `cite()` is what gets quoted."""
    ds = _staged(K=1, W=1, seeds=[0, 1])
    if ds is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    r = AMB.neighbour_ambiguity(ds)
    c = r.cite()
    for must in ("W_o=1", "K=1", "episodes", "scripted", "%.6f" % r.mean):
        assert must in c, (must, c)
    assert r.as_metadata()["citation"] == c
    assert set(r.as_metadata()) >= {"mean", "samples", "seeds", "sources",
                                    "obs_window", "chunk_size", "episodes"}


def test_gate_reports_both_numbers_and_the_ratio():
    ds = _staged(K=1, W=1, seeds=[0, 1])
    if ds is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    r = AMB.neighbour_ambiguity(ds)
    good = AMB.gate(AMB.Quantity(r.mean * 0.5, AMB.RECON_UNITS), r)
    bad = AMB.gate(AMB.Quantity(r.mean * 2.0, AMB.RECON_UNITS), r)
    assert good.passed is True and bad.passed is False
    assert abs(good.ratio - 0.5) < 1e-9 and abs(bad.ratio - 2.0) < 1e-9
    for v in (good, bad):
        txt = v.render()
        assert "train error" in txt and "ambiguity ref" in txt and "ratio" in txt
        assert v.detail.cite() in txt
    assert "PASS" in good.render() and "FAIL" in bad.render()
    # a boundary case is a fail, not a pass: strictly below
    assert AMB.gate(AMB.Quantity(r.mean, AMB.RECON_UNITS), r).passed is False


def test_gate_refuses_a_degenerate_reference():
    ds = _staged(K=1, W=1, seeds=[0, 1])
    if ds is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    r = AMB.neighbour_ambiguity(ds)
    r.mean = 0.0
    _raises(AMB.AmbiguityError, AMB.gate, AMB.Quantity(0.001, AMB.RECON_UNITS), r)


def test_one_sample_has_no_neighbour_and_says_so():
    tmp = tempfile.mkdtemp()
    try:
        if _write(tmp, 0, np.zeros((1, spec.STATE_DIM)),
                  np.zeros((1, spec.ACTION_DIM))) is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        _raises(AMB.AmbiguityError, AMB.neighbour_ambiguity, _ds(tmp, K=1, W=1))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_subsampling_is_recorded_and_is_conservative():
    ds = _staged(K=1, W=1, seeds=[0, 1, 2])
    if ds is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    full = AMB.neighbour_ambiguity(ds)
    sub = AMB.neighbour_ambiguity(ds, max_samples=300, seed=0)
    assert sub.subsampled_to == 300 and full.subsampled_to is None
    assert sub.samples == 300
    assert sub.mean >= full.mean, "fewer candidates must not LOWER the reference"
    assert "subsampled to 300" in sub.cite()


def test_curve_requires_an_explicit_tracking_policy():
    """A2 again: the policy is never inherited silently, not even here."""
    st, ac = (spec.NormStats.identity("state"), spec.NormStats.identity("action"))
    _raises(AMB.AmbiguityError, AMB.ambiguity_curve, DS.SYNTHETIC, [0, 1],
            [1, 2], 1, st, ac)


def test_gate_refuses_a_quantity_in_other_units():
    """TR29. The gate compared ACT's TOTAL loss (reconstruction + beta*KL, 0.289)
    against a reconstruction reference and reported ratio 6.47 where the right
    answer was 3.09. It was silent, and right by accident twice (no KL for BC;
    KL collapsed to 0 for ACT at lr 1e-3). A bare float and a total loss are both
    refused now; only a reconstruction-unit Quantity is scored."""
    ds = _staged(K=1, W=1, seeds=[0, 1])
    if ds is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    r = AMB.neighbour_ambiguity(ds)
    assert r.units == AMB.RECON_UNITS
    e = _raises(AMB.UnitsError, AMB.gate, r.mean * 0.5, r)          # bare float
    assert "Quantity" in str(e) and "TR29" in str(e)
    e = _raises(AMB.UnitsError, AMB.gate,
                AMB.Quantity(r.mean * 0.5, AMB.TOTAL_LOSS_UNITS), r)  # total loss
    assert AMB.TOTAL_LOSS_UNITS in str(e) and AMB.RECON_UNITS in str(e)
    ok = AMB.gate(AMB.Quantity(r.mean * 0.5, AMB.RECON_UNITS), r)
    assert ok.passed and abs(ok.ratio - 0.5) < 1e-9


def test_train_labels_its_quantities_at_the_source():
    """The labels are only as honest as where they are attached, so they are
    attached in train(), the one place that knows which number is which."""
    import inspect
    from g1_model import train as T
    src = inspect.getsource(T.train)
    assert "Quantity(float(history[-1][\"recon_l1\"]), RECON_UNITS)" in src, src
    assert "TOTAL_LOSS_UNITS" in src


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
