"""Tests for the B-prime contact contract (g1_teleop/contact_contract.py).

    python -m g1_data.test_contact_contract
    pytest g1_data/test_contact_contract.py

Two kinds of test. CONTRACT tests prove a mismatch between the configured and
the compiled contact model RAISES - at load, at reset, and for a recorded value.
COLLISION tests prove which pairs actually collide by putting geoms on top of
each other on the compiled model and reading `d.ncon` (TR19: never
mj_geomDistance). Geometry is moved by editing `body_pos` of a static platform
or of a hand's root body, so the tests do not depend on IK reaching a pose.
Each collision test also runs a configuration where the pair must NOT collide,
so a check that cannot fail would be caught.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import mujoco

from g1_teleop import contact_contract as CC
from g1_teleop.config import ContactConfig, TeleopConfig
from g1_teleop.contact_filter import apply_hand_platform_filter
from g1_teleop.indices import ModelIndex


def _cfg(on=True):
    return dataclasses.replace(TeleopConfig(), contact=ContactConfig(hand_pickup_exclusion=on))


def _fresh(on=True):
    m = CC.load_model(_cfg(on))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def _geoms(m, body_names):
    ids = {m.body(b).id for b in body_names}
    return {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in ids
            and (m.geom_contype[g] or m.geom_conaffinity[g])}


def _count(m, d, a, b):
    n = 0
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if (g1 in a and g2 in b) or (g2 in a and g1 in b):
            n += 1
    return n


def _move_static(m, d, body, to_body, offset=(0.0, 0.0, 0.0)):
    m.body_pos[m.body(body).id] = d.xpos[m.body(to_body).id] + np.asarray(offset)
    mujoco.mj_forward(m, d)


HAND_L = ("left_wrist_pitch_link", "left_wrist_yaw_link", "left_pad")
HAND_R = ("right_wrist_pitch_link", "right_wrist_yaw_link", "right_pad")


# ─── contract ────────────────────────────────────────────────────────────────
def test_scene_excludes_exactly_the_contract_pairs():
    m = mujoco.MjModel.from_xml_path(TeleopConfig().model_path)
    assert CC.excluded_pairs(m) == CC.expected_pairs(True), CC.excluded_pairs(m)
    assert len(CC.excluded_pairs(m)) == 6
    names = {n for p in CC.excluded_pairs(m) for n in p}
    assert "platform_goal" not in names, "the GOAL platform must never be excluded"
    assert not any("roll" in n for n in names)


def test_default_is_on():
    assert TeleopConfig().contact.hand_pickup_exclusion is True


def test_disabled_loader_strips_every_exclude():
    m, _ = _fresh(False)
    assert m.nexclude == 0
    assert CC.contract_of(m) == CC.expected_contract(ContactConfig(False))


def test_load_mismatch_raises():
    m_off, _ = _fresh(False)
    try:
        CC.assert_contract(m_off, ContactConfig(True))
    except CC.ContactContractMismatch:
        pass
    else:
        raise AssertionError("an unexcluded model passed an exclusion-on config")
    m_on, _ = _fresh(True)
    try:
        CC.assert_contract(m_on, ContactConfig(False))
    except CC.ContactContractMismatch:
        pass
    else:
        raise AssertionError("an excluded model passed an exclusion-off config")


def test_reset_episode_raises_on_mismatch_and_returns_contract():
    from g1_data.reset import reset_episode
    for model_on, cfg_on in ((False, True), (True, False)):
        m, d = _fresh(model_on)
        try:
            reset_episode(m, d, ModelIndex.resolve(m), _cfg(cfg_on), seed=0)
        except CC.ContactContractMismatch:
            pass
        else:
            raise AssertionError("reset_episode accepted model=%s cfg=%s" % (model_on, cfg_on))
    m, d = _fresh(True)
    start = reset_episode(m, d, ModelIndex.resolve(m), _cfg(True), seed=0)
    assert start.contact_contract == CC.contract_of(m) != ""


def test_recorded_contract_mismatch_raises():
    m_on, _ = _fresh(True)
    m_off, _ = _fresh(False)
    CC.assert_recorded_contract(CC.contract_of(m_on), m_on)          # same: passes
    for recorded, model in ((CC.contract_of(m_on), m_off),
                            (CC.contract_of(m_off), m_on),
                            (None, m_on), ("", m_on)):
        try:
            CC.assert_recorded_contract(recorded, model)
        except CC.ContactContractMismatch:
            continue
        raise AssertionError("recorded %r accepted against %s" % (recorded, CC.contract_of(model)))


def test_bitmask_filter_layered_on_top_is_caught():
    m, _ = _fresh(True)
    apply_hand_platform_filter(m, ("platform_pickup_geom",))
    assert "bitmask=custom" in CC.contract_of(m)
    try:
        CC.assert_contract(m, ContactConfig(True))
    except CC.ContactContractMismatch:
        return
    raise AssertionError("a contype/conaffinity filter slipped past the contract")


# ─── collision: what is excluded and what survives ───────────────────────────
def test_hand_pickup_excluded_when_on_collides_when_off():
    for on, want in ((True, False), (False, True)):
        m, d = _fresh(on)
        _move_static(m, d, "platform_pickup", "left_wrist_yaw_link")
        n = _count(m, d, _geoms(m, HAND_L), _geoms(m, ("platform_pickup",)))
        assert (n > 0) == want, "exclusion=%s: %d hand<->pickup contacts" % (on, n)


def test_wrist_roll_link_still_collides_with_pickup():
    m, d = _fresh(True)
    _move_static(m, d, "platform_pickup", "left_wrist_roll_link")
    n = _count(m, d, _geoms(m, ("left_wrist_roll_link",)), _geoms(m, ("platform_pickup",)))
    assert n > 0, "wrist_roll_link is not in the exclusion set and must collide"


def test_hand_goal_platform_survives():
    m, d = _fresh(True)
    _move_static(m, d, "platform_goal", "left_wrist_yaw_link")
    n = _count(m, d, _geoms(m, HAND_L), _geoms(m, ("platform_goal",)))
    assert n > 0, "hand<->GOAL platform contact is load-bearing at release"


def test_hand_box_survives():
    m, d = _fresh(True)
    ix = ModelIndex.resolve(m)
    d.qpos[ix.box_qpos][:3] = d.xpos[m.body("left_wrist_yaw_link").id]
    mujoco.mj_forward(m, d)
    n = _count(m, d, _geoms(m, HAND_L), _geoms(m, ("box1",)))
    assert n > 0, "hand<->box contact must survive (grasp approach, staged raise)"


def _overlap_hands(m, d):
    """Shift the right hand's root body so its yaw link sits on the left one."""
    rb, lb = m.body("right_wrist_roll_link").id, m.body("left_wrist_yaw_link").id
    ry = m.body("right_wrist_yaw_link").id
    parent = int(m.body_parentid[rb])
    delta = d.xpos[lb] - d.xpos[ry]
    m.body_pos[rb] = m.body_pos[rb] + d.xmat[parent].reshape(3, 3).T @ delta
    mujoco.mj_forward(m, d)
    assert np.linalg.norm(d.xpos[ry] - d.xpos[lb]) < 1e-9


def test_hand_hand_survives_exclusion():
    m, d = _fresh(True)
    _overlap_hands(m, d)
    n = _count(m, d, _geoms(m, HAND_L + ("left_wrist_roll_link",)),
               _geoms(m, HAND_R + ("right_wrist_roll_link",)))
    assert n > 0, "hand<->hand contact lost under the pair exclusion"


def test_hand_hand_check_can_fail():
    """The old contype/conaffinity candidate DID lose hand<->hand; this test's
    measure must see that, or the survival test above proves nothing."""
    m, d = _fresh(False)
    apply_hand_platform_filter(m, ("platform_pickup_geom",))
    _overlap_hands(m, d)
    lh = {g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith("left_")
          and any(k in m.body(int(m.geom_bodyid[g])).name for k in ("wrist", "pad"))}
    rh = {g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith("right_")
          and any(k in m.body(int(m.geom_bodyid[g])).name for k in ("wrist", "pad"))}
    assert _count(m, d, lh, rh) == 0


def _all_tests():
    return [(k, v) for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    failed = 0
    tests = _all_tests()
    for name, fn in tests:
        try:
            fn()
            print("ok   ", name)
        except Exception as e:                    # noqa: BLE001
            failed += 1
            print("FAIL ", name, "-", repr(e))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    raise SystemExit(1 if failed else 0)
