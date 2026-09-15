"""Tests for g1_data/spec.py. No recorded data needed - a live model load.

Run either way:

    python -m g1_data.test_spec        # plain, no test runner needed
    pytest g1_data/test_spec.py        # if pytest is installed

The permutation test is the one that matters. It does NOT check that the
permutation round-trips (a wrong permutation can round-trip perfectly); it
writes a DIFFERENT known value into each of the 17 upper-body actuators by
name, and checks that each value lands on the action dimension whose name says
it should. That is the only form of the test that catches
`action[0:17] = ctrl[ix.upper_ctrl]`.
"""
from __future__ import annotations

import numpy as np
import mujoco

from g1_teleop import config as C
from g1_teleop.grasp import GraspConfig
from g1_teleop.indices import ModelIndex
from g1_data import spec


# ─── fixtures, hand-rolled so pytest is not a dependency ──────────────────────
_CACHE = {}


def _model():
    if "m" not in _CACHE:
        m = mujoco.MjModel.from_xml_path(C.MODEL_PATH)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        _CACHE["m"], _CACHE["d"] = m, d
        _CACHE["ix"] = ModelIndex.resolve(m)
        _CACHE["spec"] = spec.SpecLayout.resolve(m, _CACHE["ix"])
    return _CACHE["m"], _CACHE["d"], _CACHE["ix"], _CACHE["spec"]


def _fresh_data():
    """A second MjData so a test that scribbles on ctrl cannot leak."""
    m, _, ix, sp = _model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d, ix, sp


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception as e:                                  # noqa: BLE001
        raise AssertionError(f"expected {exc.__name__}, got "
                             f"{type(e).__name__}: {e}") from e
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


# ─── layout ───────────────────────────────────────────────────────────────────
def test_resolves_against_the_real_model():
    m, d, ix, sp = _model()
    sp.validate(m)                       # idempotent; resolve() already ran it
    assert sp.site_left >= 0 and sp.site_right >= 0
    assert sp.weld_id >= 0
    assert sp.action_ctrl.shape == (17,)


def test_dimensions_and_names():
    assert spec.STATE_DIM == 47 and spec.ACTION_DIM == 22
    assert len(spec.STATE_NAMES) == 47
    assert len(spec.ACTION_NAMES) == 22
    assert len(set(spec.STATE_NAMES)) == 47, "duplicate state name"
    assert len(set(spec.ACTION_NAMES)) == 22, "duplicate action name"
    # the layout the brief fixes, spot-checked at every boundary
    assert spec.STATE_NAMES[0] == "box_pos_x"
    assert spec.STATE_NAMES[3] == "box_quat_w"
    assert spec.STATE_NAMES[7] == "base_pos_x"
    assert spec.STATE_NAMES[14] == "palmL_pos_x"
    assert spec.STATE_NAMES[21] == "palmR_pos_x"
    assert (spec.STATE_NAMES[28], spec.STATE_NAMES[29]) == ("g_L", "g_R")
    assert spec.STATE_NAMES[30] == "q_left_shoulder_pitch"
    assert spec.STATE_NAMES[37] == "q_right_shoulder_pitch"
    assert spec.STATE_NAMES[44] == "q_waist_yaw"
    assert spec.ACTION_NAMES[0] == "a_left_shoulder_pitch"
    assert spec.ACTION_NAMES[7] == "a_right_shoulder_pitch"
    assert spec.ACTION_NAMES[14] == "a_waist_yaw"
    assert (spec.ACTION_NAMES[17], spec.ACTION_NAMES[18]) == ("a_gL", "a_gR")
    assert spec.ACTION_NAMES[19:22] == ("a_vx", "a_vy", "a_wz")


# ─── the permutation ──────────────────────────────────────────────────────────
def test_permutation_puts_each_joint_on_the_dimension_its_name_claims():
    """The test that catches `action[0:17] = ctrl[ix.upper_ctrl]`."""
    m, d, ix, sp = _fresh_data()
    # A distinct, recognisable value per joint, written BY NAME so the test
    # does not reuse the mapping it is testing.
    code = {}
    for k, name in enumerate(spec.UPPER_JOINTS):
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert aid >= 0, name
        code[name] = 0.01 * (k + 1)
        d.ctrl[aid] = code[name]

    a = sp.build_action(d, ix, act=np.zeros(3), cmd=0.0)
    for k, name in enumerate(spec.ACTION_JOINTS):
        assert abs(a[k] - code[name]) < 1e-12, (
            f"action dim {k} ({spec.ACTION_NAMES[k]}) carries {a[k]}, which is "
            f"joint {[n for n, v in code.items() if abs(v - a[k]) < 1e-12]}, "
            f"not {name}")

    # and the naive version really would have been wrong, so the test bites
    naive = np.asarray(d.ctrl[ix.upper_ctrl], dtype=np.float64)
    assert not np.allclose(naive, a[spec.UPPER_A]), (
        "model order and action order came out identical - the permutation is "
        "no longer exercised and this test proves nothing")


def test_permutation_round_trip_is_the_identity():
    probe = np.arange(17, dtype=np.float64) + 0.5
    there = spec.action_from_upper_ctrl(probe)
    assert np.array_equal(spec.upper_ctrl_from_action(
        np.r_[there, np.zeros(5)]), probe)
    # and both directions, on a full action vector built from a model read
    m, d, ix, sp = _fresh_data()
    for k, name in enumerate(spec.UPPER_JOINTS):
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        d.ctrl[aid] = 0.1 * (k + 1)
    a = sp.build_action(d, ix, act=np.zeros(3), cmd=0.0)
    back = spec.upper_ctrl_from_action(a)
    assert np.allclose(back, np.asarray(d.ctrl[ix.upper_ctrl]))


def test_permutation_is_a_bijection():
    assert sorted(spec.ACTION_FROM_UPPER.tolist()) == list(range(17))
    assert sorted(spec.UPPER_FROM_ACTION.tolist()) == list(range(17))
    assert set(spec.ACTION_JOINTS) == set(spec.UPPER_JOINTS)


def test_action_actuator_ids_agree_with_two_derivations():
    m, d, ix, sp = _model()
    by_name = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
               for n in spec.ACTION_JOINTS]
    upper = np.r_[ix.upper_ctrl] if isinstance(ix.upper_ctrl, slice) else \
        np.asarray(ix.upper_ctrl)
    assert np.array_equal(sp.action_ctrl, np.asarray(by_name))
    assert np.array_equal(sp.action_ctrl, upper[spec.ACTION_FROM_UPPER])


# ─── state ────────────────────────────────────────────────────────────────────
def test_build_state_shape_and_sources():
    m, d, ix, sp = _model()
    s = sp.build_state(m, d, ix)
    assert s.shape == (spec.STATE_DIM,) and s.dtype == np.float64
    assert np.all(np.isfinite(s))
    assert np.allclose(spec.box_pos(s), d.qpos[ix.box_qpos][0:3])
    assert np.allclose(spec.box_quat(s), d.qpos[ix.box_qpos][3:7])
    assert np.allclose(spec.base_pos(s), d.qpos[ix.base_qpos][0:3])
    assert np.allclose(spec.base_quat(s), d.qpos[ix.base_qpos][3:7])
    assert np.allclose(spec.palm_left_pos(s), d.site_xpos[sp.site_left])
    assert np.allclose(spec.palm_right_pos(s), d.site_xpos[sp.site_right])
    assert np.allclose(spec.arm_left_q(s), d.qpos[ix.left_arm_qpos])
    assert np.allclose(spec.arm_right_q(s), d.qpos[ix.right_arm_qpos])
    assert np.allclose(spec.waist_q(s), d.qpos[ix.waist_qpos])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, d.site_xmat[sp.site_left])
    assert np.allclose(spec.palm_left_quat(s), q)
    assert abs(np.linalg.norm(spec.palm_left_quat(s)) - 1.0) < 1e-9
    assert abs(np.linalg.norm(spec.palm_right_quat(s)) - 1.0) < 1e-9


def test_palm_sites_are_the_hand_sites_from_graspconfig():
    m, d, ix, sp = _model()
    g = GraspConfig()
    assert sp.site_left == mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_SITE, g.left_site)
    assert sp.site_right == mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_SITE, g.right_site)


def test_both_gripper_state_dims_are_the_weld_bit():
    m, d, ix, sp = _fresh_data()
    d.eq_active[sp.weld_id] = 0
    mujoco.mj_forward(m, d)
    s = sp.build_state(m, d, ix)
    assert spec.grip_left(s) == 0.0 and spec.grip_right(s) == 0.0
    d.eq_active[sp.weld_id] = 1
    mujoco.mj_forward(m, d)
    s = sp.build_state(m, d, ix)
    assert spec.grip_left(s) == 1.0 and spec.grip_right(s) == 1.0
    assert spec.grip_left(s) == spec.grip_right(s)
    d.eq_active[sp.weld_id] = 0


def test_build_state_does_not_perturb_the_simulation():
    """`sync=True` calls mj_kinematics; that must not move the model."""
    m, d, ix, sp = _fresh_data()
    q0, v0, c0 = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()
    sp.build_state(m, d, ix, sync=True)
    assert np.array_equal(d.qpos, q0)
    assert np.array_equal(d.qvel, v0)
    assert np.array_equal(d.ctrl, c0)


# ─── action ───────────────────────────────────────────────────────────────────
def test_build_action_shape_and_gripper_and_velocity():
    m, d, ix, sp = _fresh_data()
    a = sp.build_action(d, ix, act=[0.1, -0.2, 0.3], cmd=1.0)
    assert a.shape == (spec.ACTION_DIM,) and a.dtype == np.float64
    assert spec.grip_left_cmd(a) == 1.0 and spec.grip_right_cmd(a) == 1.0
    assert np.allclose(spec.velocity_cmd(a), [0.1, -0.2, 0.3])
    assert np.allclose(spec.upper_cmd(a), a[0:17])


def test_build_returns_the_pair():
    m, d, ix, sp = _fresh_data()
    s, a = sp.build(m, d, ix, act=np.zeros(3), cmd=0.0)
    assert s.shape == (47,) and a.shape == (22,)


def test_act_has_no_default():
    m, d, ix, sp = _fresh_data()
    _raises(ValueError, sp.build_action, d, ix, None, 0.0)
    _raises(ValueError, sp.build_action, d, ix, [0.0, 0.0], 0.0)
    _raises(ValueError, sp.build_action, d, ix, [0.0, np.nan, 0.0], 0.0)


def test_gripper_command_range_is_checked():
    m, d, ix, sp = _fresh_data()
    _raises(ValueError, sp.build_action, d, ix, np.zeros(3), 1.5)
    _raises(ValueError, sp.build_action, d, ix, np.zeros(3), -0.1)


def test_pads_driven_against_the_logged_command_is_caught():
    """The two collection paths write the gripper command to different places."""
    m, d, ix, sp = _fresh_data()
    if ix.pad_ctrl is None:
        return
    d.ctrl[ix.pad_ctrl] = 1.0
    _raises(AssertionError, sp.build_action, d, ix, np.zeros(3), 0.0)
    a = sp.build_action(d, ix, act=np.zeros(3), cmd=1.0)   # agreeing is fine
    assert spec.grip_left_cmd(a) == 1.0
    d.ctrl[ix.pad_ctrl] = 0.0


# ─── mask ─────────────────────────────────────────────────────────────────────
def test_constant_action_mask_is_the_audited_one():
    assert spec.CONSTANT_ACTION_DIMS == (6, 13, 14, 15, 16, 18)
    assert [spec.ACTION_NAMES[d] for d in spec.CONSTANT_ACTION_DIMS] == [
        "a_left_wrist_yaw", "a_right_wrist_yaw",
        "a_waist_yaw", "a_waist_roll", "a_waist_pitch", "a_gR"]
    assert len(spec.TRAINABLE_ACTION_DIMS) == 16
    assert int(spec.ACTION_MASK.sum()) == 16
    assert not spec.ACTION_MASK[list(spec.CONSTANT_ACTION_DIMS)].any()
    assert spec.ACTION_MASK[list(spec.TRAINABLE_ACTION_DIMS)].all()


def test_wrist_roll_and_pitch_stay_trainable():
    """Constant across episodes but time-varying, so a policy must learn them."""
    for name in ("a_left_wrist_roll", "a_left_wrist_pitch",
                 "a_right_wrist_roll", "a_right_wrist_pitch"):
        d = spec.ACTION_NAMES.index(name)
        assert d in spec.TRAINABLE_ACTION_DIMS, name
        assert spec.ACTION_MASK[d], name


def test_no_state_mask():
    assert spec.STATE_MASK.all() and spec.STATE_MASK.shape == (47,)


def test_masks_are_read_only():
    for mask in (spec.ACTION_MASK, spec.STATE_MASK, spec.VELOCITY_CLIP):
        assert not mask.flags.writeable


# ─── accessors ────────────────────────────────────────────────────────────────
def test_accessors_work_batched_and_reject_wrong_width():
    s = np.arange(47, dtype=np.float64)
    batch = np.tile(s, (5, 1))
    assert np.array_equal(spec.box_pos(s), [0, 1, 2])
    assert spec.box_pos(batch).shape == (5, 3)
    assert np.array_equal(spec.waist_q(s), [44, 45, 46])
    assert spec.grip_left(s) == 28 and spec.grip_right(s) == 29
    a = np.arange(22, dtype=np.float64)
    assert np.array_equal(spec.arm_left_cmd(a), np.arange(7))
    assert np.array_equal(spec.arm_right_cmd(a), np.arange(7, 14))
    assert np.array_equal(spec.waist_cmd(a), [14, 15, 16])
    assert np.array_equal(spec.velocity_cmd(a), [19, 20, 21])
    assert spec.velocity_cmd(np.tile(a, (3, 1))).shape == (3, 3)
    _raises(ValueError, spec.box_pos, np.zeros(46))
    _raises(ValueError, spec.velocity_cmd, np.zeros(21))


def test_state_and_action_groups_tile_the_vectors():
    for groups, dim in ((spec.STATE_GROUPS, 47), (spec.ACTION_GROUPS, 22)):
        seen = []
        for _, sl in groups:
            seen.extend(range(*sl.indices(dim)))
        assert sorted(seen) == list(range(dim))


# ─── clip ─────────────────────────────────────────────────────────────────────
def test_velocity_clip_limits():
    assert np.allclose(spec.VELOCITY_CLIP[:, 1], [0.80, 0.80, 0.60])
    assert np.allclose(spec.VELOCITY_CLIP[:, 0], [-0.80, -0.80, -0.60])
    out = spec.clip_velocity([2.0, -2.0, 2.0])
    assert np.allclose(out, [0.80, -0.80, 0.60])
    assert np.allclose(spec.clip_velocity([0.5, -0.5, 0.5]), [0.5, -0.5, 0.5])
    # the measured demonstrator range must survive the clip unchanged: vy
    # reaches +-0.800 (CLAUDE.md section 6 says +-0.40; the code wins)
    assert np.allclose(spec.clip_velocity([0.0, -0.80, 0.0]), [0.0, -0.80, 0.0])


def test_recorded_float32_rails_pass_the_clip_check():
    """The command is computed in float32; the rail lands one ulp over."""
    rail = np.array([np.float32(0.80), np.float32(-0.80), np.float32(0.60)],
                    dtype=np.float64)
    assert np.any(np.abs(rail) > spec.VELOCITY_CLIP[:, 1]), (
        "float32 rails no longer exceed the limit - this test is inert")
    assert float(np.max(np.abs(rail) - spec.VELOCITY_CLIP[:, 1])) < 1e-7
    spec.assert_velocity_within_clip(rail)                  # must not raise
    spec.assert_velocity_within_clip(np.tile(rail, (5, 1)))
    _raises(AssertionError, spec.assert_velocity_within_clip,
            [0.0, 0.0, 0.65])
    _raises(AssertionError, spec.assert_velocity_within_clip,
            [0.9, 0.0, 0.0], 1e-6, "unit test")
    _raises(ValueError, spec.assert_velocity_within_clip, np.zeros(2))


# ─── version and normalization contract ───────────────────────────────────────
def test_spec_version_is_checked():
    assert isinstance(spec.SPEC_VERSION, str) and spec.SPEC_VERSION
    spec.assert_spec_version(spec.SPEC_VERSION)
    _raises(AssertionError, spec.assert_spec_version, "g1-spec-0.0.0", "test")


def test_norm_stats_identity_and_shapes():
    for kind, dim in (("state", 47), ("action", 22)):
        st = spec.NormStats.identity(kind)
        x = np.arange(dim, dtype=np.float64)
        assert np.allclose(st.normalize(x), x)
        assert np.allclose(st.denormalize(st.normalize(x)), x)
    _raises(ValueError, spec.NormStats, "action", np.zeros(21), np.ones(21))
    _raises(ValueError, spec.NormStats, "banana", np.zeros(22), np.ones(22))


def test_norm_stats_masked_dims_must_be_untouched():
    mean = np.zeros(22)
    std = np.ones(22)
    mean[6] = 0.3                       # a masked dim given a real mean
    _raises(ValueError, spec.NormStats, "action", mean, std)
    mean[6], std[18] = 0.0, 2.0         # a masked dim given a real std
    _raises(ValueError, spec.NormStats, "action", mean, std)


def test_norm_stats_rejects_a_degenerate_unmasked_dim():
    std = np.ones(22)
    std[0] = 0.0                        # trainable dim with zero variance
    _raises(ValueError, spec.NormStats, "action", np.zeros(22), std)


def test_norm_stats_round_trip():
    rng = np.random.default_rng(0)
    mean = rng.normal(size=22)
    std = rng.uniform(0.5, 2.0, size=22)
    mean[list(spec.CONSTANT_ACTION_DIMS)] = 0.0
    std[list(spec.CONSTANT_ACTION_DIMS)] = 1.0
    st = spec.NormStats("action", mean, std)
    x = rng.normal(size=(4, 22))
    assert np.allclose(st.denormalize(st.normalize(x)), x)
    z = st.normalize(x)
    for d in spec.CONSTANT_ACTION_DIMS:
        assert np.allclose(z[:, d], x[:, d]), "masked dim was rescaled"


def test_norm_stats_carries_and_checks_the_spec_version():
    _raises(AssertionError, spec.NormStats, "action", np.zeros(22),
            np.ones(22), "train", "g1-spec-0.0.0")


# ─── load-time assertions actually fire ───────────────────────────────────────
def test_validate_catches_a_broken_permutation():
    m, d, ix, sp = _model()
    good = spec.ACTION_FROM_UPPER.copy()
    try:
        spec.ACTION_FROM_UPPER[0], spec.ACTION_FROM_UPPER[1] = good[1], good[0]
        _raises(AssertionError, sp.validate, m)
    finally:
        spec.ACTION_FROM_UPPER[:] = good
    sp.validate(m)                       # and it passes again once restored


def test_builder_rejects_a_foreign_model_index():
    m, d, ix, sp = _model()
    import dataclasses
    other = dataclasses.replace(ix, upper_ctrl=np.array([99, 98, 97],
                                                        dtype=np.int32))
    _raises(AssertionError, sp.build_state, m, d, other)


# ─── phase vocabulary (spec 1.1.0) ────────────────────────────────────────────
def test_phase_integers_are_the_frozen_ones():
    """These integers are the file format. A change here relabels every episode."""
    want = {"UNKNOWN": -1, "SETTLE": 0, "REACH": 1, "GRASP": 2, "LIFT": 3,
            "MOVE": 4, "LOWER": 5, "RELEASE": 6, "VERIFY": 7,
            "REPOSITION": 8, "APPROACH": 9}
    assert {p.name: int(p) for p in spec.Phase} == want
    # Declaration order is SETTLE..VERIFY then REPOSITION, APPROACH, so the
    # integers are not positional and must not be made so.
    assert [p.name for p in spec.Phase][-2:] == ["REPOSITION", "APPROACH"]
    assert int(spec.Phase.REPOSITION) == 8 and int(spec.Phase.APPROACH) == 9


def test_phase_list_and_unknown():
    assert len(spec.PHASES) == 10
    assert spec.Phase.UNKNOWN not in spec.PHASES
    assert len(set(spec.PHASES)) == 10
    assert spec.PHASE_DTYPE(np.int8(-1)) == -1


def test_scored_mapping_is_total_and_its_image_is_exact():
    for p in spec.PHASES:
        assert p in spec.SCORED_OF, p
    assert set(spec.SCORED_OF) == set(spec.PHASES)
    assert set(spec.SCORED_OF.values()) == set(spec.ScoredPhase)
    assert len(spec.ScoredPhase) == 5
    # WALK_IN is APPENDED; the original four keep their integers
    assert {p.name: int(p) for p in spec.ScoredPhase} == {
        "GRASP": 0, "LIFT": 1, "TRANSPORT": 2, "PLACE": 3, "WALK_IN": 4}


def test_scored_phase_accessor():
    # SETTLE is the walk-in, NOT the grasp - that is the whole point of the
    # fifth bucket. Phase.APPROACH is the ARM approaching and stays under GRASP.
    assert spec.scored_phase(spec.Phase.SETTLE) is spec.ScoredPhase.WALK_IN
    assert spec.scored_phase(spec.Phase.APPROACH) is spec.ScoredPhase.GRASP
    assert spec.scored_phase(spec.Phase.REPOSITION) is spec.ScoredPhase.GRASP
    assert spec.scored_phase(4) is spec.ScoredPhase.TRANSPORT
    assert spec.scored_phase(spec.Phase.VERIFY) is spec.ScoredPhase.PLACE
    _raises(ValueError, spec.scored_phase, spec.Phase.UNKNOWN)
    got = spec.scored_phase(np.array([0, 1, 3, 4, 7, -1], dtype=np.int8))
    assert list(got) == [4, 0, 1, 2, 3, -1]
    assert got.dtype == spec.PHASE_DTYPE
    assert spec.phase_name(8) == "REPOSITION"


def test_walk_in_is_not_phase_approach():
    """Two things called APPROACH in one module would be a trap."""
    assert "WALK_IN" in spec.ScoredPhase.__members__
    assert "APPROACH" not in spec.ScoredPhase.__members__
    assert spec.Phase.APPROACH.name == "APPROACH"
    assert spec.scored_phase(spec.Phase.APPROACH) is not spec.ScoredPhase.WALK_IN


def test_only_settle_maps_to_walk_in():
    src = [p for p in spec.PHASES if spec.SCORED_OF[p] is spec.ScoredPhase.WALK_IN]
    assert src == [spec.Phase.SETTLE]


def test_phase_validation_catches_a_broken_mapping():
    good = dict(spec.SCORED_OF)
    try:
        del spec.SCORED_OF[spec.Phase.MOVE]          # no longer total
        _raises(AssertionError, spec._validate_phases)
        spec.SCORED_OF.clear(); spec.SCORED_OF.update(good)

        # total, but the image loses TRANSPORT
        spec.SCORED_OF[spec.Phase.MOVE] = spec.ScoredPhase.PLACE
        _raises(AssertionError, spec._validate_phases)
        spec.SCORED_OF.clear(); spec.SCORED_OF.update(good)

        # total, but the image loses WALK_IN - the regression this change
        # would suffer if someone "simplified" SETTLE back to GRASP
        spec.SCORED_OF[spec.Phase.SETTLE] = spec.ScoredPhase.GRASP
        _raises(AssertionError, spec._validate_phases)
        spec.SCORED_OF.clear(); spec.SCORED_OF.update(good)

        # total, image complete, but maps something that is not a phase
        spec.SCORED_OF[spec.Phase.UNKNOWN] = spec.ScoredPhase.GRASP
        _raises(AssertionError, spec._validate_phases)
    finally:
        spec.SCORED_OF.clear()
        spec.SCORED_OF.update(good)
    spec._validate_phases()


def test_the_demonstrator_uses_this_enum_and_no_other():
    from g1_data import scripted_demo as sd
    assert sd.Phase is spec.Phase
    assert all(p in spec.PHASES for p in sd.LOCKED_PHASES_WALKING)
    assert all(p in spec.PHASES for p in sd.EVENT_ENDED_PHASES)


# ─── velocity zeroing while the base is locked (spec 1.1.0) ───────────────────
def test_velocity_is_zeroed_while_the_base_lock_is_engaged():
    m, d, ix, sp = _fresh_data()
    act = [0.4, -0.7, 0.5]
    d.eq_active[sp.base_lock_id] = 0
    a = sp.build_action(d, ix, act=act, cmd=0.0)
    assert np.allclose(spec.velocity_cmd(a), act), "free: the command is logged"
    d.eq_active[sp.base_lock_id] = 1
    a = sp.build_action(d, ix, act=act, cmd=0.0)
    assert np.allclose(spec.velocity_cmd(a), 0.0), "locked: zeros, whatever act is"
    # and nothing else moved
    b = sp.build_action(d, ix, act=np.zeros(3), cmd=0.0)
    assert np.allclose(a, b), "only dims 19-21 may depend on the lock state"
    d.eq_active[sp.base_lock_id] = 0


def test_act_is_still_required_when_locked():
    m, d, ix, sp = _fresh_data()
    d.eq_active[sp.base_lock_id] = 1
    _raises(ValueError, sp.build_action, d, ix, None, 0.0)
    _raises(ValueError, sp.build_action, d, ix, [0.0, 0.0], 0.0)
    d.eq_active[sp.base_lock_id] = 0


def test_base_lock_resolves_and_is_not_the_grasp_weld():
    m, d, ix, sp = _model()
    assert sp.base_lock_id >= 0
    assert sp.base_lock_id != sp.weld_id
    from g1_teleop.base_lock import BaseLockConfig
    assert sp.base_lock_id == mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_EQUALITY, BaseLockConfig().eq_name)


def test_spec_version_is_1_1_0():
    assert spec.SPEC_VERSION == "g1-spec-1.1.0"
    _raises(AssertionError, spec.assert_spec_version, "g1-spec-1.0.0", "old file")


def _tests():
    return [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]


if __name__ == "__main__":
    print(spec.describe())
    print()
    failed = 0
    for name, fn in _tests():
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}\n        {type(e).__name__}: {e}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(_tests()) - failed}/{len(_tests())} passed")
    raise SystemExit(1 if failed else 0)
