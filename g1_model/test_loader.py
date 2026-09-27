"""Tests for the Phase 4 dataset loader.

    python -m g1_model.test_loader

Two kinds. The SHAPE tests run on the real staged episodes, because padding and
windowing bugs live at the episode boundaries and a synthetic episode of round
length would not have the boundaries the real ones do. The TRACKING tests run on
a synthetic episode built with degraded ticks on purpose: the 40 staged episodes
are all on the all-True default (O28 postdates them), so the real data cannot
exercise either policy, and a policy that has never fired is the repo's standing
finding rather than a feature.
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
from g1_data import recorder as REC
from g1_data import spec
from g1_model.loader import ChunkDataset, LoaderConfig, LoaderError, TrackingPolicy


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn),
                                                  exc.__name__))


def _identity():
    return (spec.NormStats.identity("state"), spec.NormStats.identity("action"))


def _staged():
    """The real staged episodes, or None when nothing is staged."""
    eps = DS.scan(DS.SYNTHETIC)
    return eps or None


def _synthetic_episode(tmp, seed, n_ticks, degraded=(), spawn=(1.50, -0.10)):
    """A real .npz with real metadata, whose tracking_ok is what we say it is.

    Built from a staged episode so that the metadata - spec version, contact
    contract, source - is genuine and the dataset refusals behave as they do on
    real files; only the arrays are replaced.
    """
    src = sorted(p for p in DS._npz(DS.SYNTHETIC))
    if not src:
        return None
    with np.load(src[0], allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    ok = np.ones(n_ticks, dtype=np.uint8)
    for t in degraded:
        ok[t] = 0
    meta.update(seed=int(seed), n_ticks=int(n_ticks),
                box_spawn_xy=[float(spawn[0]), float(spawn[1])])
    path = os.path.join(tmp, "ep_seed%04d.npz" % seed)
    rng = np.random.default_rng(seed)
    np.savez_compressed(
        path,
        states=rng.normal(size=(n_ticks, spec.STATE_DIM)).astype(np.float32),
        actions=rng.normal(size=(n_ticks, spec.ACTION_DIM)).astype(np.float32),
        phase_labels=np.zeros(n_ticks, dtype=np.int8),
        gait_phase=np.zeros((n_ticks, 2), dtype=np.float32),
        qpos=np.zeros((n_ticks, 45), dtype=np.float32),
        qvel=np.zeros((n_ticks, 43), dtype=np.float32),
        step_index=(np.arange(n_ticks) * 20).astype(np.int64),
        tracking_ok=ok,
        meta=np.array(json.dumps(meta)))
    return path


# ─── B2: the sample index ─────────────────────────────────────────────────────
def test_sample_index_is_deterministic_across_constructions():
    """Two constructions give byte-identical indices, and __getitem__ has no RNG,
    so `overfit 10 samples` is reproducible."""
    eps = _staged()
    if eps is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    st, ac = _identity()
    cfg = LoaderConfig(chunk_size=100, obs_window=1, tracking=TrackingPolicy())
    a = ChunkDataset(eps, cfg, st, ac, None)
    b = ChunkDataset(list(reversed(eps)), cfg, st, ac, None)
    assert np.array_equal(a.index, b.index)
    assert a.seeds == b.seeds, "episode order must not depend on scan order"
    assert len(a) == a.index.shape[0] == sum(a.lengths)
    for i in (0, 7, len(a) // 2, len(a) - 1):
        for k in ("obs", "action", "start_tick"):
            assert torch.equal(a[i][k], b[i][k])
        assert torch.equal(a[i]["obs"], a[i]["obs"])     # no RNG on re-read


def test_every_tick_is_a_start_and_the_index_is_in_order():
    eps = _staged()
    if eps is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    st, ac = _identity()
    d = ChunkDataset(eps, LoaderConfig(chunk_size=50, obs_window=1, tracking=TrackingPolicy()), st, ac, None)
    assert len(d) == sum(d.lengths)
    for ei, n in enumerate(d.lengths):
        rows = d.index[d.index[:, 0] == ei]
        assert np.array_equal(rows[:, 1], np.arange(n))


# ─── B3: chunk padding ────────────────────────────────────────────────────────
def test_chunk_is_padded_and_masked_at_the_end_of_an_episode():
    eps = _staged()
    if eps is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    st, ac = _identity()
    K = 100
    d = ChunkDataset(eps, LoaderConfig(chunk_size=K, obs_window=1, tracking=TrackingPolicy()), st, ac, None)
    n = d.lengths[0]

    # the LAST tick: one real action, K-1 padded
    last = int(np.flatnonzero((d.index[:, 0] == 0) & (d.index[:, 1] == n - 1))[0])
    s = d[last]
    assert s["action"].shape == (K, spec.ACTION_DIM)
    assert s["action_mask"].shape == (K,)
    assert int(s["action_mask"].sum()) == 1
    assert bool(s["action_mask"][0]) and not bool(s["action_mask"][1:].any())
    assert torch.isfinite(s["action"]).all(), "padding must be finite (B3)"
    assert float(s["action"][1:].abs().max()) == 0.0, "padding must be zero"
    assert torch.allclose(s["action"][0],
                          torch.from_numpy(d.actions[0][n - 1]))

    # a tick exactly K from the end: full chunk, no padding
    full = int(np.flatnonzero((d.index[:, 0] == 0) & (d.index[:, 1] == n - K))[0])
    s = d[full]
    assert bool(s["action_mask"].all())
    assert torch.allclose(s["action"], torch.from_numpy(d.actions[0][n - K:n]))

    # one tick later: exactly one padded step
    s = d[full + 1]
    assert int(s["action_mask"].sum()) == K - 1
    assert not bool(s["action_mask"][-1])


def test_chunk_longer_than_the_episode_is_all_padding_after_the_end():
    """K larger than an episode must not index out of bounds or wrap."""
    tmp = tempfile.mkdtemp()
    try:
        if _synthetic_episode(tmp, 0, 12) is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        st, ac = _identity()
        eps = DS.scan(tmp)
        d = ChunkDataset(eps, LoaderConfig(chunk_size=40, obs_window=1, tracking=TrackingPolicy()),
                         st, ac, None)
        s = d[0]
        assert s["action"].shape == (40, spec.ACTION_DIM)
        assert int(s["action_mask"].sum()) == 12
        assert torch.isfinite(s["action"]).all()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── B4: the observation window ───────────────────────────────────────────────
def test_obs_window_of_one_is_the_single_state():
    eps = _staged()
    if eps is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    st, ac = _identity()
    d = ChunkDataset(eps, LoaderConfig(chunk_size=10, obs_window=1, tracking=TrackingPolicy()), st, ac, None)
    s = d[0]
    assert s["obs"].shape == (1, spec.STATE_DIM)
    assert bool(s["obs_mask"].all())
    assert torch.allclose(s["obs"][0], torch.from_numpy(d.states[0][0]))


def test_obs_window_is_left_padded_at_tick_zero():
    eps = _staged()
    if eps is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    st, ac = _identity()
    W = 5
    d = ChunkDataset(eps, LoaderConfig(chunk_size=10, obs_window=W, tracking=TrackingPolicy()), st, ac, None)

    s = d[0]                                    # tick 0: 4 pads then the state
    assert s["obs"].shape == (W, spec.STATE_DIM)
    assert list(s["obs_mask"]) == [False] * (W - 1) + [True]
    assert float(s["obs"][:W - 1].abs().max()) == 0.0
    assert torch.allclose(s["obs"][-1], torch.from_numpy(d.states[0][0]))

    s = d[2]                                    # tick 2: 2 pads then 3 states
    assert list(s["obs_mask"]) == [False, False, True, True, True]
    assert torch.allclose(s["obs"][2:], torch.from_numpy(d.states[0][0:3]))

    s = d[W - 1]                                # tick 4: full window, no pad
    assert bool(s["obs_mask"].all())
    assert torch.allclose(s["obs"], torch.from_numpy(d.states[0][0:W]))
    # the window ENDS at the start tick: obs[-1] pairs with action[0] (spec.py)
    assert torch.allclose(s["obs"][-1], torch.from_numpy(d.states[0][W - 1]))
    assert torch.allclose(s["action"][0], torch.from_numpy(d.actions[0][W - 1]))


def test_no_loader_config_field_has_a_default():
    """B4 + A2. `obs_window` can invalidate the ACT vs ACT-LSTM comparison;
    `tracking` defaulting to off is how `tracking_ok` came to be written and
    read by nothing. Neither may be inherited silently."""
    import inspect
    sig = inspect.signature(LoaderConfig).parameters
    for name in ("chunk_size", "obs_window", "tracking"):
        assert sig[name].default is inspect.Parameter.empty, \
            "%s must have no default" % name
    # omitting the policy is a TypeError, not a quiet "off"
    _raises(TypeError, LoaderConfig, chunk_size=1, obs_window=1)
    tp = TrackingPolicy()
    _raises(LoaderError, LoaderConfig, chunk_size=0, obs_window=1, tracking=tp)
    _raises(LoaderError, LoaderConfig, chunk_size=1, obs_window=0, tracking=tp)
    # the switches inside it still default off: today's behaviour, written down
    assert tp.exclude_overlapping_chunks is False
    assert tp.max_degraded_fraction is None and tp.active is False


# ─── B5: normalization ────────────────────────────────────────────────────────
def test_normalization_round_trips_and_masked_dims_are_untouched():
    eps = _staged()
    if eps is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    if not os.path.exists(DS.NORM_STATS):
        print("      (skipped: no norm_stats_v1.npz)")
        return
    st, ac, meta = DS.load_norm_stats()
    sp = DS.load_splits()
    train = [e for e in eps if e.seed in set(sp["train"])]
    d = ChunkDataset(train, LoaderConfig(chunk_size=8, obs_window=1, tracking=TrackingPolicy()),
                     st, ac, meta)

    raw_arrays, _ = train[0].load()
    raw_s = np.asarray(raw_arrays["states"], dtype=np.float64)
    raw_a = np.asarray(raw_arrays["actions"], dtype=np.float64)

    # stored normalized; denormalizing returns the recorded values
    back_s = st.denormalize(d.states[0].astype(np.float64))
    back_a = ac.denormalize(d.actions[0].astype(np.float64))
    assert np.abs(back_s - raw_s).max() < 1e-3, np.abs(back_s - raw_s).max()
    assert np.abs(back_a - raw_a).max() < 1e-3, np.abs(back_a - raw_a).max()

    # and it actually normalized - identity would leave these equal
    assert not np.allclose(d.states[0], raw_s.astype(np.float32))

    # masked action dims carry mean 0 / std 1, so they pass through UNCHANGED
    for i in spec.CONSTANT_ACTION_DIMS:
        assert ac.mean[i] == 0.0 and ac.std[i] == 1.0
        assert np.allclose(d.actions[0][:, i], raw_a[:, i].astype(np.float32))
    mask = d.action_dim_mask
    assert mask.shape == (spec.ACTION_DIM,) and int(mask.sum()) == 16
    assert not bool(mask[list(spec.CONSTANT_ACTION_DIMS)].any())


def test_loader_refuses_to_be_handed_anything_but_norm_stats():
    eps = _staged()
    if eps is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    st, ac = _identity()
    cfg = LoaderConfig(chunk_size=4, obs_window=1, tracking=TrackingPolicy())
    _raises(LoaderError, ChunkDataset, eps, cfg, None, ac, None)
    _raises(LoaderError, ChunkDataset, eps, cfg, ac, ac, None)   # kinds swapped
    assert not hasattr(ChunkDataset, "fit_norm_stats")


# ─── B8: inherited refusals ───────────────────────────────────────────────────
def test_normalizer_from_another_split_is_refused():
    """A3 + B8. This is the failure that changes every number and no message."""
    eps = _staged()
    if eps is None or not os.path.exists(DS.NORM_STATS):
        print("      (skipped: needs staged episodes and norm stats)")
        return
    st, ac, meta = DS.load_norm_stats()
    sp = DS.load_splits()
    train = [e for e in eps if e.seed in set(sp["train"])]
    val = [e for e in eps if e.seed in set(sp["val"])]
    cfg = LoaderConfig(chunk_size=8, obs_window=1, tracking=TrackingPolicy())

    ChunkDataset(train, cfg, st, ac, meta)                  # fitted on these
    # a val loader checked against ITSELF must fire: it is not the fitted set
    e = _raises(DS.DatasetError, ChunkDataset, val, cfg, st, ac, meta)
    assert "different episode set" in str(e)
    # ...and passes when checked against the TRAINING seeds, which is correct
    ChunkDataset(val, cfg, st, ac, meta, norm_fit_seeds=sp["train"])

    # stats with no seed provenance are refused, not waved through
    old = {k: v for k, v in meta.items() if k != "seeds"}
    e = _raises(DS.DatasetError, ChunkDataset, train, cfg, st, ac, old)
    assert "provenance" in str(e)


def test_mixed_spec_version_is_refused_through_the_loader():
    """B8: the dataset refusals are not bypassed by going through g1_model."""
    tmp = tempfile.mkdtemp()
    try:
        if _synthetic_episode(tmp, 0, 30) is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        _synthetic_episode(tmp, 1, 30)
        st, ac = _identity()
        cfg = LoaderConfig(chunk_size=5, obs_window=1, tracking=TrackingPolicy())
        ChunkDataset.from_directory(tmp, cfg, st, ac, None)      # uniform: fine

        p = os.path.join(tmp, "ep_seed0001.npz")
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads(str(arrays["meta"]))
        meta["contact_contract"] = "full-contact-no-bprime"
        arrays["meta"] = np.asarray(json.dumps(meta))
        np.savez(p, **arrays)
        e = _raises(DS.DatasetError, ChunkDataset.from_directory,
                    tmp, cfg, st, ac, None)
        assert "contact contract" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_split_seed_that_is_not_staged_is_refused():
    tmp = tempfile.mkdtemp()
    try:
        if _synthetic_episode(tmp, 0, 20) is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        st, ac = _identity()
        cfg = LoaderConfig(chunk_size=5, obs_window=1, tracking=TrackingPolicy())
        e = _raises(LoaderError, ChunkDataset.from_directory, tmp, cfg, st, ac,
                    None, [0, 4242])
        assert "4242" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── B6: the tracking policies ────────────────────────────────────────────────
def test_tracking_policies_are_off_by_default_and_change_nothing_today():
    eps = _staged()
    if eps is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    assert TrackingPolicy().active is False
    st, ac = _identity()
    d = ChunkDataset(eps, LoaderConfig(chunk_size=100, obs_window=1, tracking=TrackingPolicy()), st, ac, None)
    s = d.summary()
    assert s["degraded_ticks"] == 0
    assert s["excluded_samples"] == 0 and s["dropped_episodes"] == []
    assert s["tracking_measured"] == 0, (
        "scripted episodes must carry NO tracking_ok array (recorder "
        "MEASURES_TRACKING); %d do - re-record them" % s["tracking_measured"])


def test_chunk_exclusion_policy_fires_on_a_degraded_episode():
    """The 40 staged episodes cannot exercise this, so build one that can."""
    tmp = tempfile.mkdtemp()
    try:
        # 40 ticks, one degraded at t=20
        if _synthetic_episode(tmp, 0, 40, degraded=(20,)) is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        st, ac = _identity()
        K = 5
        off = ChunkDataset.from_directory(
            tmp, LoaderConfig(chunk_size=K, obs_window=1, tracking=TrackingPolicy()), st, ac, None)
        assert len(off) == 40 and off.summary()["degraded_ticks"] == 1

        on = ChunkDataset.from_directory(
            tmp, LoaderConfig(chunk_size=K, obs_window=1,
                              tracking=TrackingPolicy(exclude_overlapping_chunks=True)),
            st, ac, None)
        # chunks starting at 16..20 all cover tick 20
        assert on.excluded_samples == K, on.excluded_samples
        assert len(on) == 40 - K
        starts = set(on.index[:, 1].tolist())
        assert starts.isdisjoint({16, 17, 18, 19, 20})
        assert 15 in starts and 21 in starts
        assert off.summary()["tracking_measured"] == 1    # measured, not default
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_episode_fraction_policy_drops_the_whole_episode():
    tmp = tempfile.mkdtemp()
    try:
        # seed 0: 4 of 40 degraded (10%).  seed 1: 20 of 40 (50%).
        if _synthetic_episode(tmp, 0, 40, degraded=(1, 2, 3, 4)) is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        _synthetic_episode(tmp, 1, 40, degraded=tuple(range(20)),
                           spawn=(1.46, -0.18))
        st, ac = _identity()
        cfg = dict(chunk_size=5, obs_window=1)

        both = ChunkDataset.from_directory(
            tmp, LoaderConfig(**cfg, tracking=TrackingPolicy()), st, ac, None)
        assert both.seeds == [0, 1] and len(both) == 80

        # threshold between the two: the 50% episode goes, the 10% one stays
        one = ChunkDataset.from_directory(
            tmp, LoaderConfig(**cfg, tracking=TrackingPolicy(
                max_degraded_fraction=0.25)), st, ac, None)
        assert one.seeds == [0], one.seeds
        assert len(one) == 40
        assert one.dropped_episodes == [(1, 0.5)], one.dropped_episodes

        # a threshold below both leaves nothing, and says so rather than
        # returning an empty dataset a training run would divide by
        e = _raises(LoaderError, ChunkDataset.from_directory, tmp,
                    LoaderConfig(**cfg, tracking=TrackingPolicy(
                        max_degraded_fraction=0.01)), st, ac, None)
        assert "every episode was dropped" in str(e).lower()

        # the two policies compose
        both_on = ChunkDataset.from_directory(
            tmp, LoaderConfig(**cfg, tracking=TrackingPolicy(
                exclude_overlapping_chunks=True, max_degraded_fraction=0.25)),
            st, ac, None)
        assert both_on.seeds == [0] and len(both_on) < 40
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tracking_policy_rejects_a_nonsense_threshold():
    _raises(LoaderError, TrackingPolicy, max_degraded_fraction=1.5)
    _raises(LoaderError, TrackingPolicy, max_degraded_fraction=-0.1)


# ─── B7: what the loader must not do ──────────────────────────────────────────
def test_loader_never_reads_gait_phase_or_feeds_phase_labels():
    """The schema freeze excludes the gait clock from the state; a sample that
    carried it would hand BC the temporal capability ACT-LSTM must supply."""
    import inspect
    src = inspect.getsource(ChunkDataset.__getitem__)
    assert "gait" not in src and "phase_label" not in src
    eps = _staged()
    if eps is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    st, ac = _identity()
    d = ChunkDataset(eps, LoaderConfig(chunk_size=4, obs_window=1, tracking=TrackingPolicy()), st, ac, None)
    assert not hasattr(d, "gait_phase")
    keys = set(d[0])
    assert "gait_phase" not in keys and "phase_labels" not in keys
    assert keys == {"obs", "obs_mask", "action", "action_mask",
                    "episode_index", "start_tick", "seed"}


def test_loader_hardcodes_no_velocity_range_and_no_ctrl_indexing():
    """B7. Parsed, not grepped: the module DOCUMENTS that spec.VELOCITY_CLIP is
    the only definition, and a text search cannot tell that sentence from a use.
    The AST sees executable code only, so docstrings and comments drop out."""
    import ast
    import inspect
    from g1_model import loader
    tree = ast.parse(inspect.getsource(loader))
    attrs, names, numbers = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attrs.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, float):
            numbers.add(float(node.value))
    for bad in ("VELOCITY_CLIP", "upper_ctrl", "pad_ctrl", "ctrl",
                "clip_velocity", "gait_phase"):
        assert bad not in attrs and bad not in names, \
            "loader must not reference %r in executable code" % bad
    # no float literal that could be a velocity limit in disguise
    assert not numbers & {0.8, 0.6, 0.4, -0.8, -0.6, -0.4}, numbers
    # and it does not define its own normalization anywhere
    for bad in ("fit_norm_stats", "save_norm_stats", "load_norm_stats"):
        assert bad not in attrs and bad not in names, bad


# ─── the batch ────────────────────────────────────────────────────────────────
def test_dataloader_batches_and_shapes_are_what_a_model_expects():
    from torch.utils.data import DataLoader
    from g1_model.loader import collate_chunks
    eps = _staged()
    if eps is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    st, ac = _identity()
    K, W, B = 20, 3, 8
    d = ChunkDataset(eps, LoaderConfig(chunk_size=K, obs_window=W, tracking=TrackingPolicy()), st, ac, None)
    dl = DataLoader(d, batch_size=B, shuffle=True, collate_fn=collate_chunks,
                    num_workers=0)
    b = next(iter(dl))
    assert b["obs"].shape == (B, W, spec.STATE_DIM) and b["obs"].dtype == torch.float32
    assert b["action"].shape == (B, K, spec.ACTION_DIM)
    assert b["action"].dtype == torch.float32
    assert b["obs_mask"].shape == (B, W) and b["obs_mask"].dtype == torch.bool
    assert b["action_mask"].shape == (B, K) and b["action_mask"].dtype == torch.bool
    assert b["seed"].shape == (B,) and b["seed"].dtype == torch.int64
    assert torch.isfinite(b["obs"]).all() and torch.isfinite(b["action"]).all()


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
