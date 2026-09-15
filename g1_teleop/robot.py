"""G1 robot model wrapper."""

from __future__ import annotations
from typing import List
import numpy as np
import mujoco

from . import config as C
from .indices import ModelIndex
from .box_reset import (assert_spawn_fits, assert_reach_fits,
                        assert_heldout_inside_region, reset_box)


def _joint_id(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"joint not found: {name}")
    return jid


def _body_id(model, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"body not found: {name}")
    return bid


class G1Robot:
    def __init__(self, cfg: C.TeleopConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self._apply_wrist_natural()
        mujoco.mj_forward(self.model, self.data)

        # Name-resolved qpos/qvel/ctrl blocks, validated against the expected
        # joint and actuator counts. Nothing here consumes it yet — the arm and
        # box indices below are resolved the same way, per-joint — but building
        # it means the twin fails loudly at load if the model stops matching
        # what the code expects, instead of silently addressing wrong joints
        # once the gripper adds joints mid-chain.
        self.index = ModelIndex.resolve(self.model, box_body=cfg.box.body_name)

        # O3 guard: refuse to load if the configured spawn region could put a
        # box corner off the platform edge. Re-derived from the model, so a Q6
        # platform resize is picked up instead of silently invalidating it.
        assert_spawn_fits(self.model, cfg.box)
        # O13 guard: the far edge of the spawn region must be reachable given
        # where the pelvis lets the base stop. And the held-out patch must be
        # strictly interior, or Objective 4 silently becomes extrapolation.
        assert_reach_fits(self.model, cfg.box, cfg.grasp)
        assert_heldout_inside_region(cfg.box)

        self._resolve_bodies()
        self._resolve_arm_indices()
        self._resolve_waist_yaw()
        self._resolve_box()
        self._measure_link_lengths()

        self.reset_box(randomize=False)

    def _apply_wrist_natural(self) -> None:
        for name, angle in C.WRIST_NATURAL.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = angle

    def _resolve_bodies(self) -> None:
        m = self.model
        self.left_shoulder_body = _body_id(m, C.LEFT_SHOULDER_BODY)
        self.right_shoulder_body = _body_id(m, C.RIGHT_SHOULDER_BODY)
        self.left_elbow_body = _body_id(m, C.LEFT_ELBOW_BODY)
        self.right_elbow_body = _body_id(m, C.RIGHT_ELBOW_BODY)
        self.left_wrist_body = _body_id(m, C.LEFT_WRIST_BODY)
        self.right_wrist_body = _body_id(m, C.RIGHT_WRIST_BODY)

    def _arm_arrays(self, joint_names: List[str]):
        m = self.model
        dof = [m.jnt_dofadr[_joint_id(m, n)] for n in joint_names]
        qpos = [m.jnt_qposadr[_joint_id(m, n)] for n in joint_names]
        lim = [m.jnt_range[_joint_id(m, n)] for n in joint_names]
        return dof, qpos, lim

    def _resolve_arm_indices(self) -> None:
        free = bool(getattr(self.cfg.ik, "free_wrists", False))
        n = len(C.LEFT_ARM_JOINTS) if free else C.N_IK_JOINTS
        ldof, lqpos, llim = self._arm_arrays(C.LEFT_ARM_JOINTS)
        rdof, rqpos, rlim = self._arm_arrays(C.RIGHT_ARM_JOINTS)
        self.ik_left_dof, self.ik_left_qpos, self.ik_left_lim = ldof[:n], lqpos[:n], llim[:n]
        self.ik_right_dof, self.ik_right_qpos, self.ik_right_lim = rdof[:n], rqpos[:n], rlim[:n]
        seed_l, seed_r = list(C.IK_SEED_LEFT), list(C.IK_SEED_RIGHT)
        if free:
            # Candidate A: the wrists join the IK, seeded at the pose D2 pinned
            # them to, so the nullspace pulls toward that pose rather than zero.
            seed_l += [C.WRIST_NATURAL[j] for j in C.LEFT_ARM_JOINTS[4:]]
            seed_r += [C.WRIST_NATURAL[j] for j in C.RIGHT_ARM_JOINTS[4:]]
        self.neutral_left = np.clip(np.array(seed_l),
                                    [l[0] for l in self.ik_left_lim],
                                    [l[1] for l in self.ik_left_lim])
        self.neutral_right = np.clip(np.array(seed_r),
                                     [l[0] for l in self.ik_right_lim],
                                     [l[1] for l in self.ik_right_lim])

    def _resolve_waist_yaw(self) -> None:
        jid = _joint_id(self.model, C.WAIST_YAW_JOINT)
        self.waist_yaw_qpos = self.model.jnt_qposadr[jid]
        self.waist_yaw_limit = self.model.jnt_range[jid]

    def _resolve_box(self) -> None:
        bid = _body_id(self.model, self.cfg.box.body_name)
        jid = self.model.body_jntadr[bid]
        self.box_qadr = self.model.jnt_qposadr[jid]
        self.box_dofadr = self.model.jnt_dofadr[jid]

    def _measure_link_lengths(self) -> None:
        x = self.data.xpos
        self.palm_site_left = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_palm_site")
        self.palm_site_right = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_palm_site")
        self.hand_len_left = float(np.linalg.norm(self.data.site_xpos[self.palm_site_left] - x[self.left_wrist_body]))
        self.hand_len_right = float(np.linalg.norm(self.data.site_xpos[self.palm_site_right] - x[self.right_wrist_body]))
        self.upper_arm_left = np.linalg.norm(x[self.left_elbow_body] - x[self.left_shoulder_body])
        self.forearm_left = np.linalg.norm(x[self.left_wrist_body] - x[self.left_elbow_body])
        self.upper_arm_right = np.linalg.norm(x[self.right_elbow_body] - x[self.right_shoulder_body])
        self.forearm_right = np.linalg.norm(x[self.right_wrist_body] - x[self.right_elbow_body])

    def left_shoulder_world(self) -> np.ndarray:
        return self.data.xpos[self.left_shoulder_body].copy()

    def right_shoulder_world(self) -> np.ndarray:
        return self.data.xpos[self.right_shoulder_body].copy()

    def set_arm_qpos(self, qpos_ids, values) -> None:
        for qid, v in zip(qpos_ids, values):
            self.data.qpos[qid] = v

    def set_waist_yaw(self, angle: float) -> float:
        clamped = float(np.clip(angle, self.waist_yaw_limit[0], self.waist_yaw_limit[1]))
        self.data.qpos[self.waist_yaw_qpos] = clamped
        return clamped

    def reset_box(self, randomize: bool = True, seed: int | None = None) -> np.ndarray:
        """Place the box, optionally randomised (O4).

        Delegates to `box_reset.reset_box` rather than sampling here. The old
        version sampled inline on the twin only, which is precisely why the
        stepped physics model never varied: two code paths, one of them
        forgotten. There is now one implementation and both callers use it.

        `seed` makes the pose reproducible. `randomize=True` with `seed=None`
        draws a fresh seed, so interactive use still varies run to run; pass an
        explicit seed for a repeatable episode.
        """
        if not randomize:
            return reset_box(self.model, self.data, self.index, self.cfg.box, seed=None)
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2 ** 32))
        return reset_box(self.model, self.data, self.index, self.cfg.box, seed=seed)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)
