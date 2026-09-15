"""Base support during manipulation (D12).

WHY THE BASE IS CONSTRAINED
---------------------------
O17: the robot's arms can place the palms only within 0.28-0.40 m of the
pelvis, but an arm-extended robot standing freely settles at an *attractor*
standoff of 0.47-0.59 m. The two never coincide. TR16 records every lever that
was swept to close that gap - grasp depth, lift height, commanded reach, box
height, platform height, waist pitch, feedforward pre-positioning - and none is
within an order of magnitude. The gap is not a tuning problem.

So the base is held while the arms work, and released for locomotion. This is a
method deviation, not a fix: it changes what the demonstrations show, and the
thesis must say so (D12).

WHY THE LOCOMOTION POLICY STOPS WHILE LOCKED
--------------------------------------------
The two cannot both run. The policy IS the balance controller (TR2) and its
in-place march exists to keep the CoM over the support polygon; with the pelvis
welded there is nothing to balance, and the march would only scrub the feet
against a base that cannot respond. Worse, the policy would be driven by
observations - near-zero base angular velocity, a perfectly upright gravity
vector - that it never saw in training, which is TR3's failure mode.

The alternative, holding the base "softly" so the policy tolerates it, was
rejected: any compliance in the hold reappears as standoff drift, and standoff
drift is precisely the thing the lock exists to remove.

So while locked: the policy is not queried, and the legs are PD-held at
`DEFAULT_ANGLES`. TR2 does not apply, because balance is being supplied
externally. Holding at DEFAULT_ANGLES rather than at whatever mid-march pose
the legs happened to be in is what makes the *release* safe - TR5 - since that
is the configuration the policy expects to be handed back.

CLEAN LOCK AND CLEAN RELEASE
----------------------------
`lock()` writes relpose from the live pelvis pose, so the constraint is already
satisfied at the instant it activates and the solver has nothing to correct -
the same trick `GraspWeld.engage` uses. `release()` deactivates it and zeroes
the policy's recurrent state, because the LSTM's hidden/cell buffers are stale
after seconds of not being stepped (2026-09-08: they are mutated in place every
forward pass). Neither touches the box or the grasp weld.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import mujoco


@dataclass(frozen=True)
class BaseLockConfig:
    eq_name: str = "base_lock"
    body: str = "pelvis"


class BaseLock:
    """Runtime handle for the world<->pelvis weld on one model/data pair."""

    def __init__(self, model, cfg: Optional[BaseLockConfig] = None):
        self.cfg = cfg or BaseLockConfig()
        self.eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                       self.cfg.eq_name)
        if self.eq_id < 0:
            raise ValueError(f"equality not found: {self.cfg.eq_name}. The base "
                             f"lock lives in scene.xml (D12).")
        self.body_bid = int(model.eq_obj2id[self.eq_id])
        expect = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.body)
        if self.body_bid != expect:
            raise ValueError(f"{self.cfg.eq_name} must weld body2={self.cfg.body}")

    def locked(self, data) -> bool:
        return bool(data.eq_active[self.eq_id])

    def lock(self, model, data) -> None:
        """Pin the pelvis at its CURRENT world pose. Jump-free by construction.

        The layout is measured, not assumed - the obvious reading of the docs
        diverges. `anchor` is the weld point in **body2's own frame**, so it is
        the origin; `relpose` is body2's pose relative to body1, and body1 is
        the world, so that is just the pelvis's world pose. Verified by holding
        the pelvis for 500 steps: 0.004 mm drift, base_z 0.7900 throughout.
        Putting the anchor in world coordinates instead (or leaving relpose at
        its default) drives the pelvis into the floor - 789 mm - because the
        anchor is then a point roughly 1.5 m outside the body, and any residual
        rotation is levered into a large position error.
        """
        q = np.array(data.xquat[self.body_bid], dtype=np.float64)
        model.eq_data[self.eq_id][0:3] = 0.0                    # anchor, body frame
        model.eq_data[self.eq_id][3:6] = data.xpos[self.body_bid]   # relpose pos
        model.eq_data[self.eq_id][6:10] = q                     # relpose quat
        model.eq_data[self.eq_id][10] = 1.0                     # torquescale
        data.eq_active[self.eq_id] = 1

    def release(self, model, data, policy=None) -> None:
        """Hand the base back to the locomotion policy.

        Removing a satisfied constraint injects no impulse, so the base keeps
        the (near-zero) velocity it had. The policy's recurrent state is zeroed
        because it has not been stepped while locked; that is exactly what
        `reset_episode` does at the start of an episode and is verified.
        """
        data.eq_active[self.eq_id] = 0
        if policy is not None:
            from g1_data.reset import reset_policy_state
            reset_policy_state(policy)

    # ---- what the lock is costing -------------------------------------------
    def reaction(self, model, data) -> Tuple[float, float]:
        """(force N, torque N*m) the weld is carrying right now.

        This is the load the legs would have had to supply unaided, and it is
        the honest measure of how much the lock is doing. A few newtons of
        horizontal restraint means the pose is nearly self-supporting and the
        lock is only cancelling the slow recession; a number near body weight
        would mean the robot is being held up.
        """
        f = np.zeros(3)
        t = np.zeros(3)
        for r in range(data.nefc):
            if (int(data.efc_type[r]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                    and int(data.efc_id[r]) == self.eq_id):
                # weld rows are ordered: 3 translational then 3 rotational
                k = r - self._first_row(data)
                if 0 <= k < 3:
                    f[k] = data.efc_force[r]
                elif 3 <= k < 6:
                    t[k - 3] = data.efc_force[r]
        return float(np.linalg.norm(f)), float(np.linalg.norm(t))

    def _first_row(self, data) -> int:
        for r in range(data.nefc):
            if (int(data.efc_type[r]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                    and int(data.efc_id[r]) == self.eq_id):
                return r
        return 0
