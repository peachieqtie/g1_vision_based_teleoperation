"""The contact contract: which body pairs the physics is NOT allowed to collide.

ADOPTED 2026-09-15 (B-prime): the hands do not collide with the PICKUP platform.

WHY
---
Teleoperated reaches near the pickup platform hook the hand under the slab edge
and jam the wrist permanently: 6 of 10 synthetic teleop motions under the base
lock (O25). Disabling hand <-> pickup-platform contact removes every jam and
every contact while the scripted demonstrator still passes both gates. It is a
SIMULATION-ONLY RELAXATION: the hands pass through the pickup slab, which is
physically false and is disclosed as a limitation (depth and duration in
NOTES.md, 2026-09-15).

WHAT IS EXCLUDED, AND WHAT IS NOT
---------------------------------
`<contact><exclude>` pairs in `scene.xml`, one per hand body against
`platform_pickup`, both sides:

    wrist_pitch_link, wrist_yaw_link, pad

`wrist_pitch_link` is in the set because excluding only yaw + pad left pitch-
link contact with the slab in 10 of 10 teleop motions (4-2414 contact-steps,
measured 2026-09-15). `wrist_roll_link` is not: it touched the slab in no run
under any exclusion set, so excluding it would lose fidelity for nothing.

Everything else keeps colliding. Three of those are LOAD-BEARING and verified
by `test_contact_contract.py`, not assumed:
  * hand <-> hand    - an exclude is per body pair, so unlike the earlier
                       contype/conaffinity candidate the hands still meet.
  * hand <-> GOAL platform - the hand rests on the goal slab at release and
                       that is what lets the box settle. Filtering it (plain
                       candidate B) scored 27/40.
  * hand <-> box     - the grasp approach and the staged raise depend on it.

WHY A CONTRACT AND NOT JUST A FLAG
----------------------------------
Recording and evaluation MUST run the same contact model. Demonstrations
collected with the exclusion and policies evaluated without it would be tested
in physics they never learned, and Experiments 1 and 2 would be invalid with no
error anywhere. So:

  1. The setting lives in `TeleopConfig.contact` - the config that the recorder,
     the evaluator, the demonstrator and `reset_episode` all take.
  2. The contract is read back from the COMPILED model (`contract_of`), never
     from the flag, so a model compiled from a different scene file, or edited
     after load, is caught by what it actually contains.
  3. `reset_episode` - the one place an episode starts, for recording and
     evaluation alike - raises if the model does not match `cfg.contact`.
  4. `EpisodeStart.contact_contract` carries the string. The recorder must store
     it in every episode; evaluation must call `assert_recorded_contract` with
     the dataset's value against its own model. A mismatch raises.

(3) is live today. (4) is the hook the recorder and evaluator (not yet built)
must use; `test_contact_contract.py` proves that it raises.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import mujoco

PICKUP_BODY = "platform_pickup"

#: Hand bodies excluded against the pickup platform. Order is irrelevant; the
#: contract compares sorted name pairs.
HAND_BODIES: Tuple[str, ...] = (
    "left_wrist_pitch_link", "left_wrist_yaw_link", "left_pad",
    "right_wrist_pitch_link", "right_wrist_yaw_link", "right_pad",
)

#: Geoms GraspWeld toggles to (0,0) while welded and restores on release. They
#: are excluded from the bitmask check because their state is the weld's.
_WELD_TOGGLED_GEOMS = ("left_pad_geom", "right_pad_geom")

CONTRACT_PREFIX = "contact-v1"


def expected_pairs(hand_pickup_exclusion: bool) -> Tuple[Tuple[str, str], ...]:
    if not hand_pickup_exclusion:
        return ()
    return tuple(sorted(tuple(sorted((b, PICKUP_BODY))) for b in HAND_BODIES))


def excluded_pairs(model) -> Tuple[Tuple[str, str], ...]:
    """The exclude table of the COMPILED model, as sorted body-name pairs.

    MuJoCo stores each pair as `(min_body_id << 16) + max_body_id` (measured on
    this build, 2026-09-15, not assumed).
    """
    out = []
    for sig in model.exclude_signature[:model.nexclude]:
        b1, b2 = int(sig) >> 16, int(sig) & 0xFFFF
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or "#%d" % b1
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or "#%d" % b2
        out.append(tuple(sorted((n1, n2))))
    return tuple(sorted(out))


def _bitmask_tag(model) -> str:
    """'uniform' if every colliding geom (weld-toggled pads aside) is (1,1).

    Catches a contype/conaffinity filter layered on top - the earlier candidate
    B mechanism, which silently disabled hand <-> hand contact too.
    """
    skip = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in _WELD_TOGGLED_GEOMS}
    odd = []
    for g in range(model.ngeom):
        if g in skip:
            continue
        ct, ca = int(model.geom_contype[g]), int(model.geom_conaffinity[g])
        if (ct, ca) not in ((0, 0), (1, 1)):
            odd.append("%d:%d/%d" % (g, ct, ca))
    return "uniform" if not odd else "custom[" + ",".join(odd) + "]"


def contract_of(model) -> str:
    pairs = excluded_pairs(model)
    body = ";".join("%s|%s" % p for p in pairs) if pairs else "none"
    return "%s exclude={%s} bitmask=%s" % (CONTRACT_PREFIX, body, _bitmask_tag(model))


def expected_contract(contact_cfg) -> str:
    pairs = expected_pairs(contact_cfg.hand_pickup_exclusion)
    body = ";".join("%s|%s" % p for p in pairs) if pairs else "none"
    return "%s exclude={%s} bitmask=uniform" % (CONTRACT_PREFIX, body)


class ContactContractMismatch(AssertionError):
    pass


def assert_contract(model, contact_cfg, where: str = "") -> str:
    """Raise unless the compiled model matches the configured contact setting."""
    got, want = contract_of(model), expected_contract(contact_cfg)
    if got != want:
        raise ContactContractMismatch(
            "contact contract mismatch%s:\n  config expects: %s\n  model has:      %s\n"
            "Recording and evaluation must run the same contact model. Load the "
            "model with g1_teleop.contact_contract.load_model(cfg)."
            % (" in " + where if where else "", want, got))
    return got


def assert_recorded_contract(recorded: Optional[str], model, where: str = "") -> None:
    """Evaluation-side check: the dataset's contract must equal this model's.

    A missing value raises too - an episode without a contract cannot be
    shown to have been recorded under the physics it is evaluated in.
    """
    got = contract_of(model)
    if recorded != got:
        raise ContactContractMismatch(
            "recorded contact contract does not match the evaluation model%s:\n"
            "  recorded: %s\n  model:    %s\nEvery policy would be tested in "
            "physics it never learned." % (" in " + where if where else "",
                                           recorded, got))


def load_model(cfg, path: Optional[str] = None):
    """Load the scene with the configured contact setting and verify it.

    `scene.xml` carries the exclusion, so the default loads directly. With
    `cfg.contact.hand_pickup_exclusion = False` (A/B work only) the exclude
    table is stripped through MjSpec before compiling.
    """
    path = path or cfg.model_path
    if cfg.contact.hand_pickup_exclusion:
        model = mujoco.MjModel.from_xml_path(path)
    else:
        spec = mujoco.MjSpec.from_file(path)
        for e in list(spec.excludes):
            spec.delete(e)
        model = spec.compile()
    assert_contract(model, cfg.contact, where="load_model(%s)" % path)
    return model
