"""What the dataset loader must REFUSE, proved by building bad datasets on purpose.

The three refusals are the ones that would otherwise produce a result that looks
fine and is wrong: mixed SPEC_VERSIONs (the arrays stop meaning the same thing),
mixed contact contracts (D18 - half the episodes lived in different physics), and
a held-out spawn in the training set (the Objective 4 guarantee). Each RAISES.
A warning here would be read once, on the day it was written, and never again.

    python g1_data/test_dataset.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_data import dataset as DS
from g1_data import spec


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn),
                                                  exc.__name__))


def _ep(seed, version=spec.SPEC_VERSION, contract="bprime", spawn=(1.50, -0.10)):
    """An Episode with only the metadata the checks read - no arrays needed."""
    return DS.Episode("mem://seed%d.npz" % seed,
                      dict(seed=seed, spec_version=version,
                           contact_contract=contract, box_spawn_xy=list(spawn)))


def test_uniform_dataset_passes():
    DS.assert_uniform([_ep(0), _ep(1), _ep(2)])


def test_mixed_spec_version_raises():
    e = _raises(DS.DatasetError, DS.assert_uniform,
                [_ep(0), _ep(1, version="g1-spec-1.0.0")])
    assert "SPEC_VERSION" in str(e)
    assert "g1-spec-1.0.0" in str(e) and "g1-spec-1.1.0" in str(e)


def test_uniform_but_wrong_spec_version_raises():
    # All agree with each other and all disagree with the code: still refused.
    _raises(AssertionError, DS.assert_uniform,
            [_ep(0, version="g1-spec-1.0.0"), _ep(1, version="g1-spec-1.0.0")])


def test_mixed_contact_contract_raises():
    e = _raises(DS.DatasetError, DS.assert_uniform,
                [_ep(0, contract="bprime"), _ep(1, contract="full")])
    assert "contact contract" in str(e) and "D18" in str(e)


def test_missing_contract_counts_as_a_different_contract():
    a, b = _ep(0), _ep(1)
    b.meta.pop("contact_contract")
    _raises(DS.DatasetError, DS.assert_uniform, [a, b])


def test_heldout_spawn_in_training_raises():
    from g1_teleop.config import TeleopConfig
    cfg = TeleopConfig().box
    inside = (float(np.mean(cfg.heldout_x)), float(np.mean(cfg.heldout_y)))
    e = _raises(DS.DatasetError, DS.assert_no_leak, [_ep(0), _ep(7, spawn=inside)])
    assert "HELD-OUT" in str(e) and "7" in str(e)


def test_clean_training_set_passes_the_leak_check():
    DS.assert_no_leak([_ep(0), _ep(1, spawn=(1.46, -0.18))])


def test_raw_heldout_directory_must_be_empty():
    """The emptiness of data/raw/heldout IS the leak check, so a file in it fails."""
    real = DS.RAW_HELDOUT
    tmp = tempfile.mkdtemp()
    try:
        DS.RAW_HELDOUT = tmp
        DS.assert_no_leak([_ep(0)])                    # empty: passes
        np.savez(os.path.join(tmp, "leaked.npz"), x=np.zeros(1))
        e = _raises(DS.DatasetError, DS.assert_no_leak, [_ep(0)])
        assert "not empty" in str(e)
    finally:
        DS.RAW_HELDOUT = real
        shutil.rmtree(tmp, ignore_errors=True)


def test_partition_streams_are_disjoint_and_correctly_sided():
    from g1_teleop.box_reset import in_heldout, sample_box_pose
    from g1_teleop.config import TeleopConfig
    cfg = TeleopConfig().box
    p = DS.partition_seeds(n_train=30, n_exp1=10, n_exp2=10)
    tr, e1, e2 = set(p["train"]), set(p["exp1"]), set(p["exp2"])
    assert len(tr) == 30 and len(e1) == 10 and len(e2) == 10
    assert not (tr & e2), "collection stream overlaps the held-out patch"
    assert not (tr & e1), "D16: Exp 1 seeds must be FRESH, not collection spawns"
    assert not (e1 & e2)
    for s in tr | e1:
        assert not in_heldout(sample_box_pose(cfg, s)[0][:2], cfg)
    for s in e2:
        assert in_heldout(sample_box_pose(cfg, s)[0][:2], cfg)


def test_split_is_exactly_80_20_and_covers_every_bin():
    eps = []
    for i in range(150):                       # the planned collection size
        x = 1.44 + 0.12 * ((i % 10) / 9.0)
        y = -0.21 + 0.42 * (((i // 10) % 15) / 14.0)
        eps.append(_ep(i, spawn=(x, y)))
    sp = DS.stratified_split(eps)
    assert len(sp["train"]) + len(sp["val"]) == 150
    assert len(sp["val"]) == 30, sp["val"]      # exactly 20%, not 4% (the carry)
    assert not set(sp["train"]) & set(sp["val"])
    assert len(sp["bins"]) == 9 and not sp["bins_without_val"]


def test_split_is_deterministic():
    eps = [_ep(i, spawn=(1.44 + 0.012 * (i % 10), -0.2 + 0.04 * (i // 10)))
           for i in range(40)]
    assert DS.stratified_split(eps) == DS.stratified_split(list(reversed(eps)))


def test_norm_stats_refuse_synthetic_by_default():
    """Fitting statistics on the demonstrator has to be a deliberate act."""
    e = _ep(0)
    e.meta["source"] = "scripted_demo"
    err = _raises(DS.DatasetError, DS.fit_norm_stats, [e])
    assert "scripted" in str(err) or "synthetic" in str(err)
    _raises(DS.DatasetError, DS.fit_norm_stats, [])


def test_real_files_on_disk_are_refused_when_mixed():
    """The same two refusals, end to end through `scan`, on real episode files.

    The in-memory tests above call the checks directly; this one goes through the
    path a collection run actually takes - read the .npz, parse its metadata,
    scan the directory - so a refusal that only works on hand-built dicts would
    show up here.
    """
    src = sorted(p for p in DS._npz(DS.SYNTHETIC))
    if len(src) < 2:
        print("      (skipped: needs 2 staged episodes in data/synthetic)")
        return
    tmp = tempfile.mkdtemp()
    try:
        good = []
        for i, p in enumerate(src[:2]):
            dst = os.path.join(tmp, "ep%d.npz" % i)
            shutil.copy2(p, dst)
            good.append(dst)
        assert len(DS.scan(tmp)) == 2                       # uniform: accepted

        def _patch(path, **fields):
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            meta = json.loads(str(arrays["meta"]))
            meta.update(fields)
            arrays["meta"] = np.asarray(json.dumps(meta))
            np.savez(path, **arrays)

        _patch(good[1], spec_version="g1-spec-1.0.0")
        e = _raises(DS.DatasetError, DS.scan, tmp)
        assert "SPEC_VERSION" in str(e)

        _patch(good[1], spec_version=spec.SPEC_VERSION,
               contact_contract="full-contact-no-bprime")
        e = _raises(DS.DatasetError, DS.scan, tmp)
        assert "contact contract" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── the provenance and velocity guards (decision audit, 2026-09-16) ──────────
def test_every_registered_source_declares_whether_it_is_real():
    from g1_data import recorder as REC
    assert REC.SOURCES, "the registry must not be empty"
    for src, info in REC.SOURCES.items():
        assert isinstance(src, str) and src
        assert set(info) == {"label", "real"}, info
        assert isinstance(info["real"], bool)
    # Neither entry point that exists today produces real demonstrations, so
    # nothing can currently route into data/raw/train. That is correct: no human
    # demonstration has been collected yet.
    assert not any(i["real"] for i in REC.SOURCES.values())


def test_unknown_source_is_refused_not_guessed():
    from g1_data import recorder as REC
    e = _raises(REC.UnknownSource, REC.source_info, "brand_new_tool.py",
                where="ep_seed0007.npz")
    assert "brand_new_tool.py" in str(e) and "ep_seed0007.npz" in str(e)
    _raises(REC.UnknownSource, REC.source_info, None)
    _raises(REC.UnknownSource, REC.label_of, "")


def test_label_is_derived_from_source():
    from g1_data import recorder as REC
    assert REC.label_of(REC.SOURCE_SCRIPTED) == "scripted"
    assert REC.label_of(REC.SOURCE_TELEOP_FIXTURE) == "teleop-fixture"


def test_velocity_guard_refuses_an_out_of_clip_command():
    """The D15 guard the recorder now calls on every tick."""
    rail = np.array([0.80, 0.80, 0.60], dtype=np.float32).astype(np.float64)
    spec.assert_velocity_within_clip(rail)                    # must not raise
    for dim, over in ((0, [0.81, 0.0, 0.0]), (1, [0.0, 0.81, 0.0]),
                      (2, [0.0, 0.0, 0.61])):
        e = _raises(AssertionError, spec.assert_velocity_within_clip,
                    np.asarray(over), where="tick 42")
        assert "tick 42" in str(e)


def test_staged_episodes_all_pass_both_guards():
    """The guards must be no-ops on the data already collected."""
    from g1_data import recorder as REC
    eps = DS.scan(DS.SYNTHETIC)
    if not eps:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    for e in eps:
        REC.source_info(e.meta.get("source"), where=os.path.basename(e.path))
        arrays, _ = e.load()
        spec.assert_velocity_within_clip(
            np.asarray(arrays["actions"][:, spec.VEL_A], dtype=np.float64),
            where=os.path.basename(e.path))


def test_namespaces_are_separate_and_each_owns_its_ledger():
    from g1_data import recorder as REC
    assert REC.NS_SCRIPTED != REC.NS_COLLECTION
    assert REC.NAMESPACES[REC.NS_SCRIPTED]["real"] is False
    assert REC.NAMESPACES[REC.NS_COLLECTION]["real"] is True
    # The ledger is derived from the directory, so the two can never be paired
    # wrongly - that pairing is what silently skipped 40 of 200 training seeds.
    assert REC.ledger_for(REC.NS_SCRIPTED) != REC.ledger_for(REC.NS_COLLECTION)
    for ns in (REC.NS_SCRIPTED, REC.NS_COLLECTION):
        assert os.path.dirname(REC.ledger_for(ns)) == os.path.abspath(ns)


def test_scripted_source_refused_in_the_collection_namespace():
    from g1_data import recorder as REC
    e = _raises(REC.NamespaceMismatch, REC.assert_namespace,
                REC.NS_COLLECTION, REC.SOURCE_SCRIPTED)
    assert "demonstrations" in str(e) and REC.SOURCE_SCRIPTED in str(e)
    # and the reverse: a real demonstration may not land in the scripted tree
    REC.SOURCES["__test_real__"] = dict(label="test-real", real=True)
    try:
        e = _raises(REC.NamespaceMismatch, REC.assert_namespace,
                    REC.NS_SCRIPTED, "__test_real__")
        assert "episodes" in str(e)
    finally:
        del REC.SOURCES["__test_real__"]


def test_unregistered_directory_is_allowed():
    """Ad-hoc output dirs are fine: staging routes on source, so an
    unregistered directory cannot smuggle anything into a training split."""
    from g1_data import recorder as REC
    assert REC.namespace_info(tempfile.gettempdir()) is None
    REC.assert_namespace(tempfile.gettempdir(), REC.SOURCE_SCRIPTED)


def test_collection_ledger_does_not_inherit_the_scripted_accepts():
    """The A4 failure case, as a test."""
    from g1_data.ledger import EpisodeLedger
    tmp = tempfile.mkdtemp()
    try:
        old = EpisodeLedger(os.path.join(tmp, "scripted", "ledger.jsonl"))
        for seed in range(5):
            old.append("accept", seed=seed, path="x.npz", heldout=False,
                       label="scripted", checks={}, placement_error=0.0)
        new = EpisodeLedger(os.path.join(tmp, "demonstrations", "ledger.jsonl"))
        stream = list(range(8))
        assert old.pending(stream) == [5, 6, 7], "shared ledger skips 0-4"
        assert new.pending(stream) == stream, "separate ledger issues every seed"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
