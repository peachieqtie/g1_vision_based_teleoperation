"""Name-based resolution of every qpos / qvel / ctrl index in the model.

WHY THIS EXISTS
---------------
MuJoCo numbers joints by a depth-first walk of the body tree and actuators by
their order of declaration. Insert one joint anywhere and every index after it
shifts by one -- silently. Nothing raises; a stale slice just starts addressing
the wrong joint, and the only symptom is a robot that behaves subtly wrong.

The palm-pad gripper (Q1) puts a slide joint inside each wrist_yaw_link. The
left pad's joint lands between the left and right arm chains, so right-arm
indices shift while leg indices do not: half of any hardcoded slice stays
correct and half breaks. That is the worst possible failure mode, because the
robot still walks.

So: no module may hardcode an offset. Resolve names through ModelIndex once at
load, and let the assertions below fail loudly if the model stops matching what
the code expects.

USAGE
    ix = ModelIndex.resolve(model)
    data.ctrl[ix.leg_ctrl] = tau
    leg_q = data.qpos[ix.leg_qpos]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import mujoco

from . import config as C

# Contiguous index ranges stay slices so that data.qpos[idx] returns a VIEW,
# exactly as the hardcoded slices did. A fancy-index array would return a copy;
# nothing currently writes through such a view, but preserving it keeps this a
# behaviour-free refactor rather than one that merely looks equivalent.
Index = Union[slice, np.ndarray]


def _joint_id(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"joint not found in model: {name}")
    return jid


def _actuator_id(model, name: str) -> int:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"actuator not found in model: {name}")
    return aid


def _pack(ids: Sequence[int]) -> Index:
    """slice() when the ids are ascending and contiguous, else an index array.

    Both index and assign identically. The slice form is preferred only so the
    read semantics (view, not copy) match what the replaced literals did.
    """
    arr = np.asarray(list(ids), dtype=np.int32)
    if arr.size and np.array_equal(arr, np.arange(arr[0], arr[0] + arr.size)):
        return slice(int(arr[0]), int(arr[0]) + int(arr.size))
    return arr


def _count(idx: Index) -> int:
    return (idx.stop - idx.start) if isinstance(idx, slice) else int(idx.size)


@dataclass(frozen=True)
class ModelIndex:
    """Every index the control code needs, resolved from joint/actuator names."""

    # Floating base (root freejoint). Structurally always first, but resolved
    # anyway so that no offset in the codebase is a literal.
    base_qpos: Index            # 7: xyz + wxyz quat
    base_xy_qpos: Index         # 2: world XY, for the position-hold loop
    base_quat_qpos: Index       # 4: orientation quat
    base_qvel: Index            # 6: linear + angular
    base_angvel_qvel: Index     # 3: angular velocity, for the policy obs

    # Legs: 12 torque-controlled joints driven by the locomotion policy.
    leg_qpos: Index
    leg_qvel: Index
    leg_ctrl: Index

    # Upper body: 17 position-controlled joints (waist 3 + arms 7 + 7).
    upper_qpos: Index
    upper_ctrl: Index
    waist_qpos: Index
    left_arm_qpos: Index
    right_arm_qpos: Index

    # Box freejoint.
    box_qpos: Index
    box_qvel: Index

    # Palm-pad gripper. None until the pads exist in g1.xml (Q1).
    pad_qpos: Optional[Index]
    pad_ctrl: Optional[Index]

    @classmethod
    def resolve(cls, model, box_body: str = "box1") -> "ModelIndex":
        def qpos_of(names: List[str]) -> Index:
            return _pack([model.jnt_qposadr[_joint_id(model, n)] for n in names])

        def qvel_of(names: List[str]) -> Index:
            return _pack([model.jnt_dofadr[_joint_id(model, n)] for n in names])

        def ctrl_of(names: List[str]) -> Index:
            return _pack([_actuator_id(model, n) for n in names])

        base_jid = _joint_id(model, C.FLOATING_BASE_JOINT)
        if model.jnt_type[base_jid] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"{C.FLOATING_BASE_JOINT} is not a free joint")
        bq = int(model.jnt_qposadr[base_jid])
        bv = int(model.jnt_dofadr[base_jid])

        box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, box_body)
        if box_bid < 0:
            raise ValueError(f"body not found in model: {box_body}")
        box_jid = int(model.body_jntadr[box_bid])
        boxq = int(model.jnt_qposadr[box_jid])
        boxv = int(model.jnt_dofadr[box_jid])

        # Pads are optional: present only after the Q1 gripper lands in g1.xml.
        have_pads = all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0
            for n in C.PAD_JOINTS
        )

        ix = cls(
            base_qpos=slice(bq, bq + 7),
            base_xy_qpos=slice(bq, bq + 2),
            base_quat_qpos=slice(bq + 3, bq + 7),
            base_qvel=slice(bv, bv + 6),
            base_angvel_qvel=slice(bv + 3, bv + 6),
            leg_qpos=qpos_of(C.LEG_JOINTS),
            leg_qvel=qvel_of(C.LEG_JOINTS),
            leg_ctrl=ctrl_of(C.LEG_JOINTS),
            upper_qpos=qpos_of(C.UPPER_BODY_JOINTS),
            upper_ctrl=ctrl_of(C.UPPER_BODY_JOINTS),
            waist_qpos=qpos_of(C.WAIST_JOINTS),
            left_arm_qpos=qpos_of(C.LEFT_ARM_JOINTS),
            right_arm_qpos=qpos_of(C.RIGHT_ARM_JOINTS),
            box_qpos=slice(boxq, boxq + 7),
            box_qvel=slice(boxv, boxv + 6),
            pad_qpos=qpos_of(C.PAD_JOINTS) if have_pads else None,
            pad_ctrl=ctrl_of(C.PAD_JOINTS) if have_pads else None,
        )
        ix.validate(model)
        return ix

    def validate(self, model) -> None:
        """Fail loudly if the model stopped matching the code's expectations.

        Catches the three ways the gripper work could go wrong quietly: a joint
        or actuator renamed or dropped (count checks), an actuator added that no
        config name list knows about (the nu check), and a name list reordered
        relative to the model so positional gain vectors mis-map (order check).
        """
        expect = [
            ("leg_qpos", self.leg_qpos, len(C.LEG_JOINTS)),
            ("leg_qvel", self.leg_qvel, len(C.LEG_JOINTS)),
            ("leg_ctrl", self.leg_ctrl, len(C.LEG_JOINTS)),
            ("upper_qpos", self.upper_qpos, len(C.UPPER_BODY_JOINTS)),
            ("upper_ctrl", self.upper_ctrl, len(C.UPPER_BODY_JOINTS)),
            ("waist_qpos", self.waist_qpos, len(C.WAIST_JOINTS)),
            ("left_arm_qpos", self.left_arm_qpos, len(C.LEFT_ARM_JOINTS)),
            ("right_arm_qpos", self.right_arm_qpos, len(C.RIGHT_ARM_JOINTS)),
        ]
        for name, idx, n in expect:
            got = _count(idx)
            if got != n:
                raise AssertionError(
                    f"{name}: resolved {got} indices, expected {n}")

        # Every actuator in the model must be accounted for, or the 22-D action
        # vector is addressing a control channel nobody knows about.
        n_known = _count(self.leg_ctrl) + _count(self.upper_ctrl)
        if self.pad_ctrl is not None:
            n_known += _count(self.pad_ctrl)
        if n_known != model.nu:
            raise AssertionError(
                f"model has nu={model.nu} actuators but ModelIndex accounts for "
                f"{n_known}. An actuator was added without updating "
                f"g1_teleop/config.py joint-name lists.")

        # ORDER is the invariant that matters, not contiguity. The locomotion
        # policy's 12 actions, the qj/dqj observation block and the KPS / KDS /
        # DEFAULT_ANGLES gain vectors are all positional: element i means leg
        # joint i in model order. If a config name list were ever reordered
        # relative to the model, every gain would map to the wrong joint and the
        # robot would still walk -- badly, and for no visible reason.
        #
        # Gaps are fine. Pad joints and actuators may be declared anywhere in
        # the XML; the resolved index simply stops being a slice and becomes an
        # index array, which reads and assigns identically.
        for name in ("leg_qpos", "leg_qvel", "leg_ctrl",
                     "upper_qpos", "upper_ctrl"):
            idx = getattr(self, name)
            ids = np.r_[idx] if isinstance(idx, slice) else idx
            if ids.size > 1 and not np.all(np.diff(ids) > 0):
                raise AssertionError(
                    f"{name} resolved to non-ascending indices {list(ids)}. The "
                    f"joint-name list in g1_teleop/config.py is no longer in "
                    f"model order, so positional gain and action vectors would "
                    f"map to the wrong joints.")

        # Legs and upper body must not overlap.
        leg = set(np.r_[self.leg_ctrl] if isinstance(self.leg_ctrl, slice)
                  else self.leg_ctrl)
        upper = set(np.r_[self.upper_ctrl] if isinstance(self.upper_ctrl, slice)
                    else self.upper_ctrl)
        if leg & upper:
            raise AssertionError(
                f"leg_ctrl and upper_ctrl overlap at {sorted(leg & upper)}")

    def describe(self) -> str:
        pads = "absent" if self.pad_qpos is None else str(self.pad_qpos)
        return (f"ModelIndex(base_qpos={self.base_qpos}, "
                f"leg_qpos={self.leg_qpos}, leg_qvel={self.leg_qvel}, "
                f"leg_ctrl={self.leg_ctrl}, upper_qpos={self.upper_qpos}, "
                f"upper_ctrl={self.upper_ctrl}, box_qpos={self.box_qpos}, "
                f"pads={pads})")
