"""Damped least-squares inverse kinematics for the G1 arms."""

from __future__ import annotations
from typing import Sequence
import numpy as np
import mujoco

from .config import IKConfig

#: Iteration diagnostics, OFF by default (tools/teleop_throughput.py turns it on).
#: The solver warm-starts from the twin's current qpos, so after the first frame
#: it converges in far fewer than `max_iter` - measuring that is what decides
#: whether the iteration count is worth tuning.
COLLECT_STATS = False
ITERATIONS: list = []


def _kinematics(model, data) -> None:
    """Everything this solver reads, and nothing else.

    It needs body poses (`xpos`/`xquat`), site poses for the palm-site variant,
    and the quantities `mj_jac` derives a Jacobian from - `subtree_com` and
    `cdof`, both produced by `mj_comPos`. `mj_forward` additionally runs
    collision detection over the whole robot, builds and solves the constraint
    system, and evaluates actuation and the full dynamics pipeline, none of
    which any line below looks at. Measured 2026-09-15: `mj_jac` returns
    bit-identical Jacobians either way (tools/teleop_throughput.py --verify).
    """
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


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

    used = cfg.max_iter
    for it in range(cfg.max_iter):
        _kinematics(model, data)
        el_pos = data.xpos[elbow_body_id].copy()
        wr_pos = (data.site_xpos[task_site_id].copy() if task_site_id is not None
                  else data.xpos[wrist_body_id].copy())

        err_el = elbow_target - el_pos
        err_wr = wrist_target - wr_pos
        if np.linalg.norm(err_el) < cfg.tol and np.linalg.norm(err_wr) < cfg.tol:
            used = it
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

    # The trailing refresh leaves the twin consistent with the qpos this solve
    # just wrote, for the second arm's solve and for whoever reads `xpos` /
    # `site_xpos` next. `TeleopController._step_inner` still calls
    # `robot.forward()` (a full mj_forward) once per frame after the smoothing,
    # so anything that wants the full pipeline still gets it - see NOTES.
    _kinematics(model, data)
    if COLLECT_STATS:
        ITERATIONS.append(used)
    return np.array([data.qpos[qid] for qid in qpos_ids])
