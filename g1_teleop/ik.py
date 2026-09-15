"""Damped least-squares inverse kinematics for the G1 arms."""

from __future__ import annotations
from typing import Sequence
import numpy as np
import mujoco

from .config import IKConfig


def solve_arm_ik(
    model,
    data,
    elbow_body_id: int,
    wrist_body_id: int,
    elbow_target: np.ndarray,
    wrist_target: np.ndarray,
    qpos_ids: Sequence[int],
    dof_ids: Sequence[int],
    joint_limits: Sequence[np.ndarray],
    neutral_q: np.ndarray,
    cfg: IKConfig,
    task_site_id=None,
) -> np.ndarray:
    """`task_site_id`, when given, replaces the wrist BODY as the second task
    point with that SITE (Candidate A: the palm). Default None is unchanged."""
    jacp_el = np.zeros((3, model.nv))
    jacp_wr = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    n = len(dof_ids)
    eye_n = np.eye(n)

    for _ in range(cfg.max_iter):
        mujoco.mj_forward(model, data)
        el_pos = data.xpos[elbow_body_id].copy()
        wr_pos = (data.site_xpos[task_site_id].copy() if task_site_id is not None
                  else data.xpos[wrist_body_id].copy())

        err_el = elbow_target - el_pos
        err_wr = wrist_target - wr_pos
        if np.linalg.norm(err_el) < cfg.tol and np.linalg.norm(err_wr) < cfg.tol:
            break

        mujoco.mj_jac(model, data, jacp_el, jacr, el_pos, elbow_body_id)
        if task_site_id is not None:
            mujoco.mj_jacSite(model, data, jacp_wr, jacr, task_site_id)
        else:
            mujoco.mj_jac(model, data, jacp_wr, jacr, wr_pos, wrist_body_id)

        jac = np.vstack([jacp_el[:, dof_ids], jacp_wr[:, dof_ids]])
        err = np.concatenate([err_el, err_wr])

        jjt = jac @ jac.T + cfg.damping ** 2 * np.eye(6)
        jac_pinv = jac.T @ np.linalg.solve(jjt, np.eye(6))
        dq_task = jac_pinv @ err

        current_q = np.array([data.qpos[qid] for qid in qpos_ids])
        seed_pull = cfg.nullspace_weight * (neutral_q - current_q)
        nullspace = (eye_n - jac_pinv @ jac) @ seed_pull
        dq = cfg.step_size * dq_task + nullspace

        for i, (qid, lim) in enumerate(zip(qpos_ids, joint_limits)):
            data.qpos[qid] = np.clip(data.qpos[qid] + dq[i], lim[0], lim[1])

    mujoco.mj_forward(model, data)
    return np.array([data.qpos[qid] for qid in qpos_ids])
