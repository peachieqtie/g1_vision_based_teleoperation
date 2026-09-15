"""Seeded box-pose randomisation for episode reset (O4), with a spawn region
that keeps the whole box footprint on the platform (O3).

WHY THIS EXISTS
---------------
Randomisation used to live inside `G1Robot.__init__`, which builds the
*kinematic twin*. The stepped physics model never saw it, so every episode on
the real model started from the keyframe pose. Ten "runs" of the 10/10 gate were
one run repeated, and Objectives 3 and 4 -- which are entirely about box
position -- had no source of variation at all.

The fix is not to bolt randomisation onto the physics path as well. Two copies
of the sampling logic is exactly how the twin and the physics model drifted
apart in the first place. This module is the single implementation; `G1Robot`
delegates to it, and any episode loop on the stepped model calls the same
function with the same config.

SEEDING
-------
`sample_box_pose(cfg, seed)` is a pure function of the seed: the same seed always
gives the same pose, and different seeds give different poses. It is NOT a
stateful stream, so reproducing episode 7 does not require replaying episodes
0-6 -- you just pass 7. That matters for the held-out evaluation (Objective 4),
where specific box positions have to be re-runnable on demand.

ORIENTATION
-----------
Position only. Box orientation is fixed by the thesis constraints (CLAUDE.md
section 1), so the quaternion is always identity.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import mujoco

from .config import BoxConfig, GraspConfig


def _geom_id(model, name: str) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        raise ValueError(f"geom not found in model: {name}")
    return gid


def platform_extent(model, cfg: BoxConfig) -> Tuple[np.ndarray, np.ndarray]:
    """(centre_xy, half_extent_xy) of the pickup platform, read from the model.

    Read by name rather than trusted from config, so that a Q6 platform resize
    is picked up automatically instead of silently invalidating the spawn range.
    """
    gid = _geom_id(model, cfg.platform_geom)
    bid = model.geom_bodyid[gid]
    centre = model.body_pos[bid][:2] + model.geom_pos[gid][:2]
    return np.asarray(centre, dtype=float), np.asarray(model.geom_size[gid][:2], dtype=float)


def box_half_extent(model, cfg: BoxConfig) -> np.ndarray:
    return np.asarray(_geom_size(model, cfg.box_geom)[:2], dtype=float)


def _geom_size(model, name: str) -> np.ndarray:
    return model.geom_size[_geom_id(model, name)]


def max_safe_half(model, cfg: BoxConfig) -> np.ndarray:
    """Largest per-axis sampling half-range that keeps the whole footprint on the
    platform with `cfg.edge_margin` to spare.

    The box is axis-aligned and stays that way, so this is a per-axis bound, not
    a radial one: a corner spawn sits at (half_x + box_x, half_y + box_y).
    """
    _, plat_half = platform_extent(model, cfg)
    return plat_half - box_half_extent(model, cfg) - cfg.edge_margin


def assert_spawn_fits(model, cfg: BoxConfig) -> None:
    """O3: fail loudly if the spawn region can put a box corner off the edge.

    Per-axis. x and y have genuinely different bounds -- x is capped near 0.08
    by the pelvis standoff (O13), y is free -- so collapsing this to a single
    scalar would silently over-constrain y or under-constrain x.
    """
    safe = max_safe_half(model, cfg)
    half = np.asarray(cfg.pickup_half, dtype=float)
    if half.shape != (2,):
        raise AssertionError(
            f"BoxConfig.pickup_half must be a 2-tuple (x, y); got {cfg.pickup_half}")
    over = half > safe + 1e-9
    if np.any(over):
        _, plat_half = platform_extent(model, cfg)
        box_half = box_half_extent(model, cfg)
        axis = "xy"[int(np.argmax(half - safe))]
        raise AssertionError(
            f"BoxConfig.pickup_half={cfg.pickup_half} exceeds the safe range "
            f"{tuple(np.round(safe, 4))} on axis {axis}. Platform half-extent "
            f"{plat_half}, box half-width {box_half}, margin {cfg.edge_margin}. "
            f"A corner spawn would reach {tuple(np.round(half + box_half, 4))} "
            f"against platform {tuple(plat_half)} and could topple (O3).")

    cfg_centre = np.asarray(cfg.pickup_center, dtype=float)
    model_centre, _ = platform_extent(model, cfg)
    if not np.allclose(cfg_centre, model_centre, atol=1e-6):
        raise AssertionError(
            f"BoxConfig.pickup_center={tuple(cfg_centre)} does not match the "
            f"platform in the model at {tuple(model_centre)}.")


def sample_box_pose(cfg: BoxConfig, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic box pose for `seed`. seed=None returns the platform centre.

    Pure function of the seed -- same seed, same pose; different seeds, different
    poses; no dependence on call order or global RNG state.

    Draws from the WHOLE sample region, held-out patch included. Splitting train
    from held-out is the recorder's job (it has to record which side an episode
    came from), so this stays a plain sampler; `in_heldout` is the predicate to
    filter with, and the end-of-collection leak check uses the same function.
    """
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    cx, cy = cfg.pickup_center
    if seed is None:
        return np.array([cx, cy, cfg.spawn_z]), quat
    rng = np.random.default_rng(seed)
    hx, hy = cfg.pickup_half
    return (np.array([rng.uniform(cx - hx, cx + hx),
                      rng.uniform(cy - hy, cy + hy),
                      cfg.spawn_z]), quat)


def in_heldout(pos_xy, cfg: BoxConfig) -> bool:
    """Is this box position inside the Objective 4 held-out patch?

    A 2-D interior patch, so BOTH coordinates must be inside. Every held-out x
    also occurs in training at some other y and vice versa: only the combination
    is unseen, which is what makes this compositional generalization inside the
    convex hull rather than extrapolation.
    """
    (x0, x1), (y0, y1) = cfg.heldout_x, cfg.heldout_y
    return bool(x0 <= pos_xy[0] <= x1 and y0 <= pos_xy[1] <= y1)


def assert_heldout_inside_region(cfg: BoxConfig) -> None:
    """The held-out patch must sit strictly inside the sample region, or it is
    not an interior patch and the test is extrapolation after all."""
    cx, cy = cfg.pickup_center
    hx, hy = cfg.pickup_half
    for name, (lo, hi), c, h in (("x", cfg.heldout_x, cx, hx),
                                 ("y", cfg.heldout_y, cy, hy)):
        if not (c - h < lo < hi < c + h):
            raise AssertionError(
                f"held-out {name} range {(lo, hi)} is not strictly inside the "
                f"sample range {(c - h, c + h)} — that would make Objective 4 "
                f"an extrapolation test, which floors every policy (Q6).")


def reset_box(model, data, index, cfg: BoxConfig,
              seed: Optional[int] = None, forward: bool = True) -> np.ndarray:
    """Write a (seeded) box pose into ANY model's box freejoint and zero its velocity.

    Works on the kinematic twin and the stepped physics model alike -- that is
    the whole point of O4. `index` is a ModelIndex, so the freejoint is located
    by name and the pad joints cannot shift it out from under us.
    """
    pos, quat = sample_box_pose(cfg, seed)
    data.qpos[index.box_qpos] = np.concatenate([pos, quat])
    data.qvel[index.box_qvel] = 0.0
    if forward:
        mujoco.mj_forward(model, data)
    return pos


def footprint_corners(pos_xy, model, cfg: BoxConfig) -> np.ndarray:
    """The four XY corners of the box footprint, for on-platform checks."""
    bh = box_half_extent(model, cfg)
    return np.array([[pos_xy[0] + sx * bh[0], pos_xy[1] + sy * bh[1]]
                     for sx in (-1, 1) for sy in (-1, 1)])


def corners_on_platform(pos_xy, model, cfg: BoxConfig) -> bool:
    centre, half = platform_extent(model, cfg)
    return bool(np.all(np.abs(footprint_corners(pos_xy, model, cfg) - centre)
                       <= half + 1e-9))


def reach_limit(grasp: GraspConfig) -> float:
    """Furthest box x the robot can grasp: it cannot stand past `max_base_x`."""
    return grasp.max_base_x + grasp.grasp_max


def assert_reach_fits(model, cfg: BoxConfig, grasp: GraspConfig) -> None:
    """O13: the far edge of the spawn region must be physically reachable.

    Same failure class as O3 -- a constant no assertion could see. GRASP_MIN and
    GRASP_MAX used to live as module-level literals in the entry point, so
    nothing could check them against the spawn region. If this fires, either
    shrink `pickup_half[0]` or shrink the platform: growing the platform does
    NOT help, because its near edge advances on the robot as fast as its far
    edge retreats (CLAUDE.md O13).
    """
    cx, _ = cfg.pickup_center
    far_edge = cx + cfg.pickup_half[0]
    limit = reach_limit(grasp)
    if far_edge > limit + 1e-9:
        raise AssertionError(
            f"far sample edge x={far_edge:.3f} exceeds the reach limit "
            f"{limit:.3f} (max_base_x {grasp.max_base_x} + grasp_max "
            f"{grasp.grasp_max}). Boxes spawned there cannot be grasped: the "
            f"pelvis stops the base at x={grasp.max_base_x}.")
