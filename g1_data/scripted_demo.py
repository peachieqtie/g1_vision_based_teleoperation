"""Scripted demonstrator — the Phase 1 exit gate.

Performs the full manipulation with no camera and no human input, so the whole
stack below the recorder can be exercised and scored.

TARGET ANCHORING — read TR15 before changing this
--------------------------------------------------
Arm targets are **precomputed joint-space poses**, solved once per episode from
a stable standing configuration, then ramped between. There is NO world-anchored
target and no closed loop from base state into arm commands at any point.

That is deliberate. TR15: re-solving IK against a fixed world point while the
base is free to pitch creates positive feedback — the base tips, the shoulder
moves, holding the palm at that world point demands a more extreme pose, which
tips it further. Measured at standoff 0.30: fixed pose 6.2 deg pitch and stands;
re-solved 27.1 deg; re-solved with the base at the standoff 70.9 deg and falls.

The two options the brief offered were (a) anchor at the live shoulder as
`compute_arm_targets` does, or (b) rate-limit target motion. This uses neither,
because a third option is strictly safer: **open-loop joint space**. Live-shoulder
anchoring is what makes the *teleop* path safe, but there the target comes from
the demonstrator's own arm; a scripted grasp has to reach a specific box, so
anchoring at the shoulder would not put the palms anywhere in particular.
Rate-limiting only slows the feedback, it does not remove it. Commanding a fixed
joint pose removes the loop entirely — and it is the configuration the A1 sweep
measured as stable at every standoff (max pitch 3.9-10.4 deg, no falls).

The cost is that the grasp pose is computed for the box's position at reset and
does not adapt if the box moves afterwards. Nothing moves it before the grasp,
so that is fine here; an autonomous policy would need the loop and must inherit
TR15's warning.

THE TASK: WALK IN, GRASP, CARRY, PLACE  (`walk_place=True`, the default)
-----------------------------------------------------------------------
The robot walks to the pickup platform, grasps the box, CARRIES it 1.5 m to the
goal platform at (1.5, -1.5) with its arms static, then lowers and releases. This
is the task the proposal describes.

The base is welded to the world while the arms work and handed back to the
locomotion policy for transport - `LOCKED_PHASES_WALKING` drops MOVE from the
locked set (D12). Nothing from the pickup solve is reused at the goal: the place
chain is re-solved from the pelvis the robot actually walked to (TR14(d)).

RETIRED: the stationary lateral place (`walk_place=False`)
----------------------------------------------------------
An earlier variant could not walk, so it stood still and swept the box sideways
by `place_dy` = 0.20 m instead. That is **retired** and kept only as a control -
it still passes 10/10, and it is the baseline the scope correction is measured
against.

It was retired because the sweep drove the trailing arm into the torso (O24):
the shoulder-roll servo saturates at ~5x its +-25 N*m limit against `torso_link`,
and the achieved displacement asymptotes at ~0.105 m however hard it is
commanded. The clean envelope is only +-0.06 m - SMALLER than the 0.10 m Q4
tolerance - so no lateral sweep both clears the collision and leaves the success
criterion able to discriminate "placed it" from "did not move it". Walking
transport removes the motion rather than engineering around it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import mujoco

from g1_teleop.base_lock import BaseLock
from g1_teleop.config import TeleopConfig
from g1_teleop.grasp import GraspWeld
from g1_teleop.ik import solve_arm_ik
from g1_teleop.indices import ModelIndex
from g1_teleop.robot import G1Robot
from g1_data import spec
from g1_data.spec import Phase
from g1_data.phases import LockConfig, LockPredicate, PlatformGeometry
from g1_data.reset import LocomotionCarryover, reset_episode, reset_policy_state


# `Phase` is imported from g1_data/spec.py, which owns the vocabulary and the
# integers that go into the episode file (spec 1.1.0). It is re-exported here so
# `from g1_data.scripted_demo import Phase` keeps working, but there is exactly
# one definition and it is not this one.
#
# Two phases exist for reasons that belong with the demonstrator rather than
# with the schema, and they are recorded here:
#
# REPOSITION sits between REACH and GRASP. The reach pushes the robot backwards
# harder than the policy can counteract WHILE reaching (the command saturates at
# full forward and the base still recedes), but with the arms already frozen the
# base can close the gap: measured 121 mm in 3 s at cmd 0.8 with the arms
# extended. Separating the two motions is what makes the standoff recoverable.
#
# APPROACH sits between the staged reach and the grasp: the palms descend beside
# the box, wider than it, and only then close. Without it the reach interpolates
# in joint space straight from the home pose to the grasp pose and sweeps the
# hands through the 40 mm platform slab - measured, the wrists wedge under the
# near edge at 100 N and the shoulder position servos saturate at their +-25 N*m
# limit trying to pull free, leaving the palms 183 mm below the box.


#: Phases during which the base is welded to the world (D12). The lock goes on
#: entering the first of these and comes off entering the first phase that is
#: not in the set, so the transitions are declared here rather than scattered
#: through the loop. MOVE is in the set for the stationary variant; the walking
#: variant removes it, which is the whole reason this is a set and not a pair of
#: phase names. VERIFY is deliberately OUTSIDE it: the episode ends with the
#: base handed back to the locomotion policy, so a pose the robot could not hold
#: unaided shows up as a fall in scoring instead of being hidden by the weld.
LOCKED_PHASES = frozenset({
    Phase.REACH, Phase.REPOSITION, Phase.APPROACH, Phase.GRASP, Phase.LIFT,
    Phase.MOVE, Phase.LOWER, Phase.RELEASE,
})

#: Walking place (the proposal's actual task, adopted 2026-09-09). MOVE leaves
#: the set, so the base is handed back to the locomotion policy and the robot
#: CARRIES the box to the goal platform with its arms static. This is exactly
#: what the phase-set was built to allow. The 0.20 m lateral arm sweep it
#: replaces was an artifact of a stationary variant that could not walk: it drove
#: the trailing arm into the torso (O24), whose clean envelope is only +-0.06 m,
#: smaller than the 0.10 m Q4 tolerance -- so no lateral sweep both clears the
#: collision and leaves the success criterion able to discriminate.
LOCKED_PHASES_WALKING = LOCKED_PHASES - {Phase.MOVE}

#: Phases whose duration is an UPPER BOUND, not a schedule: they end when the
#: lock predicate fires, or at the bound if it never does (2026-09-10). Both are
#: station-keeping phases, so ending them early removes dead time and nothing
#: else - the arm poses either side are identical (SETTLE holds `home`, the
#: walking MOVE holds `P["lift"]`), so no interpolation is truncated.
EVENT_ENDED_PHASES = frozenset({Phase.SETTLE, Phase.MOVE})


@dataclass(frozen=True)
class DemoConfig:
    standoff: float = 0.32      # middle of the O14 band [0.28, 0.40]
    waist_pitch: float = 0.0    # Q5/D10 pinned; -0.26 is the measured fallback
    waist_yaw: float = 0.0      # Q5/D10 pinned. Non-zero is a MEASUREMENT knob
                                # only (O18 lateral coverage); unpinning the waist
                                # is a decision D10 owns, not a default.
    # How far to raise the box off the platform. 0.08 was measured (2026-09-10)
    # to ACHIEVE only 61.2 mm - the arm sags under the box - which left the box
    # bottom 1.4 mm BELOW the platform top on the way out (it scraped) and only
    # 5.7 mm of clearance at the goal on 10 of 12 seeds. Swept on stepped state,
    # 12 seeds per value, achieved lift / clearance at the second lock:
    #   0.08 -> 61.2 mm / 8.8-17.8 mm (min over transport -1.4 mm, SCRAPES)
    #   0.10 -> 78.4 mm / 12.8-27.5 mm
    #   0.12 -> 95.9 mm / 21.4 mm          <- smallest with >=80 mm and >=20 mm
    #   0.14 -> 113.7 mm / 35.9-60.2 mm
    #   0.16 -> 131.8 mm / 50.4-76.6 mm
    # Achieved tracks commanded linearly with a ~19 mm constant deficit. Raising
    # it does NOT reopen a reach or collision problem: peak torso_link force is
    # 215.6 N at 0.12, the LOWEST of the five, and 12/12 at every value.
    lift_h: float = 0.12
    place_dy: float = 0.20      # lateral place offset; > the 0.10 m Q4 tolerance
    # O23: DISABLED (0) after measurement. Correcting the commanded offset works
    # in the free direction (~30 mm constant) but CANNOT work in the blocked one:
    # the achieved displacement asymptotes at ~0.105 m (cmd 0.10/0.14/0.18/0.20 ->
    # 0.084/0.096/0.102/0.105), because the trailing arm jams into the torso
    # (torso_link vs *_shoulder_yaw_link, 208-278 N at EVERY magnitude, including
    # 0.10) and its shoulder-roll servo saturates at ~5x its +-25 N*m limit.
    # Commanding more just presses harder. Enabling it would make the behaviour
    # direction-dependent, which is worse for the dataset than the present
    # roughly-symmetric shortfall. Kept for when the collision is fixed.
    place_correct_iters: int = 0
    # Walking place (scope correction, 2026-09-09): carry the box to the goal
    # platform instead of sweeping it sideways. The arms hold the lift pose
    # through MOVE; the legs do the transport. The 0.20 m lateral sweep this
    # replaces was an artifact of a stationary variant that could not walk.
    walk_place: bool = True
    goal_xy: tuple = (1.5, -1.5)   # goal platform centre (scene.xml)
    move_s_walk: float = 16.0      # transport leg; now a MAX duration, not a
                                   # schedule - MOVE ends when the lock predicate
                                   # fires at the goal. Measured arrival is
                                   # 30.4-33.5 s, so 16 s left 10-13 s of marching
                                   # in place at the goal (2026-09-10).
    # Event-driven base lock (D12, 2026-09-10). The lock fires from an observable
    # predicate over the 47-D state instead of from phase-set membership, because
    # deployment and teleoperated collection have no phase machine. SETTLE and
    # MOVE then end on the lock event, with their durations as upper bounds.
    # Only for `walk_place`: the retired stationary variant needs the base locked
    # THROUGH its lateral MOVE sweep, which is exactly what the predicate's
    # release condition (box lifted and carried) would undo.
    #
    # ENABLED 2026-09-11. It was off because it dropped the gate to 8/12 by
    # exposing O26 - locking ~10 s earlier sampled the gait freely and the reach
    # sweep hooked a pad under the platform. The staged raise removes the wedge,
    # and with it the predicate scores **40/40 over 40 seeds**, against a pre-fix
    # baseline of 24/40, with all four transitions firing on every seed and the
    # max-duration fallback never used. Episodes go from a constant 1278 samples
    # to 694-846 (mean 757), a 41% reduction.
    lock_predicate: bool = True
    # O22's lesson applied at the goal: the walk stops SHORT of the commanded
    # standoff, so aiming at exactly `standoff` landed 2 of 12 arrivals at
    # 0.279 m, outside the [0.28, 0.36] band `assert_standoff` enforces at the
    # pickup. Measured deficit 21-41 mm; this biases the navigation target back
    # by the middle of it so arrivals land inside the band. Moving the target is
    # preferred over asserting on arrival: an assertion would reject episodes
    # that place perfectly well, and the standoff is an OUTCOME here, not a
    # precondition anything could have checked (O3, O22).
    goal_standoff_bias: float = 0.03
    # Rejection criterion (2026-09-10): the arm sags 38-58 mm under the box
    # during transport, and at lift_h = 0.08 the box bottom passed 1.4 mm BELOW
    # the platform top on the way out - it scraped. A demonstration where the box
    # drags the platform edge must not enter the dataset silently.
    min_clearance: float = 0.005
    # O26 (2026-09-11): a wrist held this far off its command is a WEDGE, and
    # D2 makes it permanent. This replaces "zero contacts" as the reach's gate
    # criterion - the staged raise does not remove the collision, it removes
    # the wedge, so the wedge is what is scored.
    #
    # 0.8, not the 0.5 first proposed: measured post-REACH wrist-pitch deviation
    # is 0.298-0.486 rad in episodes that pass, and 1.418 rad in the diagnosed
    # wedge. A 0.5 threshold leaves 14 mrad of margin against healthy runs and
    # would manufacture failures; 0.8 sits between the two populations with
    # 0.31 rad below and 0.62 rad above.
    wedge_rad: float = 0.8
    # CANDIDATE B (2026-09-15, measurement only, default OFF): no hand<->platform
    # contact. See g1_teleop/contact_filter.py. A modelling choice to disclose,
    # not a fix - the hand passes through the table.
    hand_platform_filter: object = False   # False | True/"both" | "pickup"
    sep: float = 0.18           # palm separation = box width
    settle_s: float = 1.5
    reach_s: float = 3.0
    reposition_s: float = 6.0
    # 0.6 s was 15 samples at 25 Hz, ~1% of the episode - the scored phase with
    # the tightest tolerance was also the rarest thing in the dataset
    # (2026-09-10 constant-dimension audit). The weld engages in the second half
    # of GRASP either way; the extra time is dwell at the grasp pose.
    grasp_s: float = 1.6
    lift_s: float = 2.0
    move_s: float = 3.0
    lower_s: float = 2.0
    release_s: float = 0.5
    verify_s: float = 2.5
    max_pitch_deg: float = 25.0
    fall_z: float = 0.35
    # Base station-keeping, local to the demonstrator. LocomotionConfig's values
    # (deadband 0.02, kp 0.8) are tuned for idle drift of a few mm/s; the reach
    # drives 151 mm of drift, so they are far too loose here. Not changed
    # globally — walk_test and the teleop entry point keep their own tuning.
    hold_deadband: float = 0.004
    hold_kp: float = 2.5
    hold_max: float = 0.80
    # Heading control. The station-keeper drove act[0:2] only; act[2] (wz) was
    # never commanded even though the policy accepts it (cmd_scale wz 0.25) and
    # the action vector carries it. Heading then accumulated the policy's
    # lateral-walking bias -- a systematic -15.4 deg -- and 0.328 m of standoff
    # swung through that is 87 mm of lateral offset in the frame the arms work
    # in, which is what actually broke the walk-in grasp (O21, correlation
    # 0.969 between |effective lateral| and palm error).
    hold_kp_yaw: float = 2.0
    hold_max_yaw: float = 0.60      # policy tolerates +-0.80
    hold_deadband_yaw: float = 0.01  # rad
    # ONE discrete arm re-solve at a settled state just before the grasp. This
    # is a base-state -> arm-command path, so it is TR15-adjacent and is
    # deliberately fired exactly once and then frozen: the isolation table showed
    # a single re-solve at the origin reached 27 deg but STOOD; the fall needed a
    # moving base AND continuous re-solving.
    discrete_resolve: bool = True
    # Base support during manipulation (D12). See g1_teleop/base_lock.py for why
    # the locomotion policy stops while locked rather than being fought.
    base_lock: bool = True
    # Staged approach, all measured against the platform slab at 0.73-0.75 m.
    # ---- staged raise (O26 MITIGATION, 2026-09-11) ----------------------
    # The old reach lerped home -> P["up"] in joint space, and that arc swings
    # the hands FORWARD to 0.150 m while still at z 0.663 - past the platform's
    # near edge (measured 0.124 m in front of the pelvis) and under the slab.
    # The pads then hook and the wrist jams 1.42 rad off command for the rest of
    # the episode, unrecoverably (D2: the wrist command never changes again).
    #
    # This raises the hands up the BODY first, at lateral 0 and at the home palm
    # separation, and only then carries them out to P["high"]. It is a
    # MITIGATION, not a fix: it does not remove the collision, it removes the
    # WEDGE. Measured over four lock states, 3 of 3 wedging states stop wedging
    # (wrist error 1.418 -> 0.001 rad) and contacts fall 96-98%.
    #
    # Interpolation is CARTESIAN - the sub-waypoints are solved along a straight
    # palm path - because a joint-space lerp between two clear poses is exactly
    # what arcs through the slab. The whole chain is still solved ONCE at the
    # lock instant and played back open loop, so TR15 does not apply.
    staged_reach: bool = True
    raise_sub: int = 10          # sub-waypoints up the body
    out_sub: int = 6             # sub-waypoints out to P["high"]
    raise_top: float = 0.02      # top of the raise, above the box centre
    raise_sep: float = 0.30      # palm separation at the top of the raise
    stage_x: float = 0.10        # palms this far forward while raising - the
                                 # platform's near edge is ~0.12 m ahead of the
                                 # pelvis at the working standoff
    approach_h: float = 0.14     # stage this far above the box centre. 0.20 gives
                                 # more clearance over the box top and fixes the
                                 # cells where the approach knocks it off, but it
                                 # drops the stationary gate to 6/10 - the staging
                                 # waypoint is unreachable either way (O19), so
                                 # more height just picks a different bad pose.
    approach_sep: float = 0.06   # extra palm separation while approaching
    approach_s: float = 2.0
    retract_s: float = 1.5       # widen and withdraw before the base is freed
    seat_depth: float = 0.03     # lower this far past the pickup height, so the
                                 # box is resting on the platform - not hanging
                                 # from the weld - when the hands let go
    trace_every: int = 0         # >0 logs a diagnostic sample every N steps
    # Fixed WORLD base position (x, y). Default None places the base at
    # `standoff` from the sampled box, which cancels the spawn randomisation
    # (O18). Pinning it in world coordinates instead lets the box vary relative
    # to the robot, which is what Objective 4 needs.
    fixed_base: Optional[tuple] = None
    # Override the seeded spawn with an explicit world (x, y). For the coverage
    # sweep only - episodes that feed the recorder must keep the seeded spawn so
    # the held-out patch means something.
    box_xy: Optional[tuple] = None
    # Force the place direction instead of deriving it from the box's y sign.
    # Measurement only (O23): lets the direction be separated from the spawn
    # position, which the automatic rule confounds.
    place_shift_override: Optional[float] = None
    # Start the base HERE and let the locomotion policy walk it to the standoff
    # during SETTLE, instead of teleporting it into place. Measurement of walking
    # positioning precision (O18/O19): if the walk stops within a few cm
    # laterally, only head-on grasps are ever needed.
    start_xy: Optional[tuple] = None
    # Truncate the schedule after this phase name. Used by the coverage sweep:
    # whether a spawn is graspable is decided by GRASP, and running the transport
    # as well would quadruple the sweep cost for no extra information.
    stop_after: Optional[str] = None
    release_sep: float = 0.14    # extra separation for letting go and withdrawing.
                                 # Wider than the approach: the approach only has
                                 # to miss a box sitting still, the withdrawal has
                                 # to miss one it has just placed, and 30 mm of
                                 # clearance per side was measured nudging it 55 mm
    # Seconds to hold after the base is released, before scoring. Long enough
    # for an unholdable pose to actually fall (fall_z is 0.35 m from 0.79 m).
    verify_s_free: float = 2.5


def _pitch_deg(quat) -> float:
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, quat)
    return float(np.degrees(-np.arcsin(np.clip(M.reshape(3, 3)[2, 0], -1, 1))))


def _elbow(sh, wr, upper, fore, swivel):
    v = wr - sh
    d = np.linalg.norm(v)
    span = upper + fore
    if d > span * 0.995:
        wr = sh + v * (span * 0.995 / d)
        v = wr - sh
        d = np.linalg.norm(v)
    u = v / d
    a = np.arccos(np.clip((upper**2 + d**2 - fore**2) / (2 * upper * d), -1, 1))
    p = swivel - np.dot(swivel, u) * u
    n = np.linalg.norm(p)
    p = p / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])
    return sh + upper * (np.cos(a) * u + np.sin(a) * p), wr


class PoseBook:
    """Solves the joint-space poses the script commands, once per episode.

    Every pose is solved on a twin held in the STANDING configuration (upright
    pelvis, DEFAULT_ANGLES legs) and expressed purely as joint angles. Nothing
    here reads the live physics state, which is what keeps TR15's loop closed.
    """

    def __init__(self, cfg: TeleopConfig, demo: DemoConfig):
        self.cfg, self.demo = cfg, demo
        self.twin = G1Robot(cfg)
        self.ix = ModelIndex.resolve(self.twin.model)
        self.default = np.asarray(cfg.loco.default_angles, dtype=np.float64)

    def palm_mid(self):
        """Midpoint of the two palm sites in the twin's CURRENT solved state.

        Valid immediately after a `solve`; it is how the place correction reads
        what a commanded pose actually achieves without stepping physics.
        """
        tm, tw = self.twin.model, self.twin.data
        l = tw.site_xpos[mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_SITE, "left_palm_site")]
        r = tw.site_xpos[mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_SITE, "right_palm_site")]
        return 0.5 * (np.asarray(l) + np.asarray(r))

    def solve(self, offset, sep: Optional[float] = None,
              seed: Optional[np.ndarray] = None,
              sep_dir: Optional[np.ndarray] = None) -> tuple:
        """Joint pose putting the palms on a box at `offset` from the pelvis.

        `offset` is (forward, LATERAL, vertical) in the PELVIS frame.

        `sep_dir` is the axis the palms straddle along, also in the PELVIS frame,
        and it is what makes the solve correct under a non-zero base heading
        (O20). The box's faces are fixed in the WORLD, so the palms must
        straddle along world y; expressing that in the pelvis frame gives
        `R(-yaw) . y_world = (sin yaw, cos yaw, 0)`. Rotating only the offset and
        leaving the separation along the pelvis's own y is what made the naive
        fix worse than no fix (10/10 -> 3/10): the palms swung onto the box's
        corners instead of its faces. Default None keeps the old behaviour,
        which is correct exactly when the heading is zero.

        `sep` overrides the palm separation for this pose only, which is what
        lets the approach come down beside the box wider than the box and then
        close on it, rather than descending onto its top edges.

        `seed` starts the IK from a given upper-body pose instead of from the
        keyframe. Waypoints MUST be solved in path order, each seeded from the
        previous: the IK has several branches (elbow in/out, mirrored) and
        solving every waypoint from the same keyframe seed lets consecutive
        waypoints land on different ones. Interpolating between two branches
        sweeps the arm through the space between them, which knocks the box off
        the platform. Measured without seeding: palm error erratic and
        non-monotonic in lateral offset (67 / 303 / 42 mm for one cell across
        three clearance settings).
        """
        tm, tw = self.twin.model, self.twin.data
        mujoco.mj_resetDataKeyframe(tm, tw, 0)
        tw.qpos[self.ix.leg_qpos] = self.default
        from g1_teleop import config as C
        for nm, ang in C.WRIST_NATURAL.items():
            j = mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_JOINT, nm)
            tw.qpos[tm.jnt_qposadr[j]] = ang
        wp = mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")
        tw.qpos[tm.jnt_qposadr[wp]] = self.demo.waist_pitch
        wy = mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw_joint")
        tw.qpos[tm.jnt_qposadr[wy]] = self.demo.waist_yaw
        free = bool(getattr(self.cfg.ik, "free_wrists", False))
        if seed is not None:
            # Continuity seed, then re-pin every joint the IK does not drive so
            # the seed cannot drag the wrists or waist off their fixed values.
            # Under Candidate A the IK drives the wrists, so they keep the seed.
            tw.qpos[self.ix.upper_qpos] = seed
            for nm, ang in ([] if free else C.WRIST_NATURAL.items()):
                j = mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_JOINT, nm)
                tw.qpos[tm.jnt_qposadr[j]] = ang
            tw.qpos[tm.jnt_qposadr[wp]] = self.demo.waist_pitch
            tw.qpos[tm.jnt_qposadr[wy]] = self.demo.waist_yaw
        mujoco.mj_forward(tm, tw)
        pel = tw.xpos[mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_BODY, "pelvis")].copy()
        goal_c = pel + np.asarray(offset)
        worst = 0.0
        for side, sgn in (("left", +1.0), ("right", -1.0)):
            sh = (self.twin.left_shoulder_world() if side == "left"
                  else self.twin.right_shoulder_world())
            up = self.twin.upper_arm_left if side == "left" else self.twin.upper_arm_right
            fo = self.twin.forearm_left if side == "left" else self.twin.forearm_right
            eb = self.twin.left_elbow_body if side == "left" else self.twin.right_elbow_body
            wb = self.twin.left_wrist_body if side == "left" else self.twin.right_wrist_body
            qp = self.twin.ik_left_qpos if side == "left" else self.twin.ik_right_qpos
            df = self.twin.ik_left_dof if side == "left" else self.twin.ik_right_dof
            lm = self.twin.ik_left_lim if side == "left" else self.twin.ik_right_lim
            nu = self.twin.neutral_left if side == "left" else self.twin.neutral_right
            sid = mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_SITE, side + "_palm_site")
            half = (self.demo.sep if sep is None else sep) / 2.0
            if sep_dir is None:
                dhat = np.array([0.0, 1.0, 0.0])
            else:
                dhat = np.asarray(sep_dir, dtype=np.float64).copy()
                dhat[2] = 0.0
                nrm = np.linalg.norm(dhat)
                dhat = dhat / nrm if nrm > 1e-9 else np.array([0.0, 1.0, 0.0])
            goal = goal_c + sgn * half * dhat
            # The elbow swivel is a lateral preference, so it follows the same
            # axis; leaving it on the pelvis y would fight the rotated target.
            swivel = np.array([sgn * 0.35 * dhat[0], sgn * 0.35 * dhat[1], -1.0])
            wr_goal = goal.copy()
            for _ in range(6):
                et, wt = _elbow(sh, wr_goal, up, fo, swivel)
                if free:
                    # Candidate A: aim the PALM at the goal directly and let
                    # the wrist joints help place it.
                    solve_arm_ik(tm, tw, eb, wb, et, goal, qp, df, lm, nu,
                                 self.cfg.ik, task_site_id=sid)
                else:
                    solve_arm_ik(tm, tw, eb, wb, et, wt, qp, df, lm, nu, self.cfg.ik)
                wr_goal = goal - (tw.site_xpos[sid] - tw.xpos[wb])
            worst = max(worst, float(np.linalg.norm(tw.site_xpos[sid] - goal)))
        pose = np.array(tw.qpos[self.ix.upper_qpos], dtype=np.float64)
        pose[0] = self.demo.waist_yaw
        pose[2] = self.demo.waist_pitch
        return pose, worst


def build_poses(book, demo: "DemoConfig", offset_xyz, place_shift: float,
                sep_dir: Optional[np.ndarray] = None, home_palm=None):
    """Every pose the script commands, for a box at `offset_xyz` from the pelvis.

    Module-level so diagnostics exercise the real path rather than a copy of it.

    `offset_xyz` is (forward, LATERAL, vertical) from the pelvis. The lateral term
    used to be pinned at 0, which was harmless only while the base was placed at
    the box's own y. With a fixed base (O18) it is the whole point.

    The path is staged so the hands never transit the platform slab: raise close
    to the body, extend forward above the box, descend beside it wider than it,
    close, and reverse the same way on the release.
    """
    x, y, z = offset_xyz
    wide = demo.sep + demo.approach_sep
    free = demo.sep + demo.release_sep
    h = demo.approach_h
    # The place shift is a displacement along the box's face axis (world y), so
    # it rotates with `sep_dir` exactly as the separation does.
    dh = (np.array([0.0, 1.0, 0.0]) if sep_dir is None
          else np.asarray(sep_dir, dtype=np.float64))
    dh = np.array([dh[0], dh[1], 0.0])
    nrm = np.linalg.norm(dh)
    dh = dh / nrm if nrm > 1e-9 else np.array([0.0, 1.0, 0.0])
    o = np.array([x, y, z], dtype=np.float64)
    pv = place_shift * dh
    up_v = np.array([0.0, 0.0, 1.0])
    sx = np.array([demo.stage_x, y, z], dtype=np.float64)
    # Solve the GRASP pose first, unseeded, then propagate OUTWARD from it in
    # both directions, each waypoint seeded from its neighbour. Anchoring on the
    # grasp matters: seeding forward from the tucked `up` pose instead makes the
    # IK converge to a poor branch for every later waypoint (measured: palm error
    # ~300 mm at cells that otherwise pass, and the stationary gate breaks).
    P = {}
    P["grasp"], e = book.solve(o, sep_dir=sep_dir)
    grasp_mid = book.palm_mid().copy()
    back = [("side",  o,                                   wide),
            ("high",  o + h * up_v,                        wide),
            ("up",    sx + h * up_v,                       wide)]
    seed = P["grasp"]
    for name, off, sp in back:
        P[name], _ = book.solve(off, sep=sp, seed=seed, sep_dir=sep_dir)
        seed = P[name]

    # O23: the arm does not achieve the displacement it is commanded. Measured
    # undershoot is exactly repeatable (spread 0.1-0.2 mm over the whole sample
    # region) and independent of where the box sits, but it is direction- and
    # pose-dependent - 41.5 mm moving one way against 93.1 mm the other, and it
    # shifts 13.7 mm for a 1.9 mm change in standoff. A hardcoded constant
    # therefore does not hold, and baking the achieved value into `place_dy`
    # would put a systematic bias in every demonstration for the policy to copy.
    #
    # Instead: correct the COMMANDED offset until the pose the IK returns
    # actually displaces the palms by `place_shift`. The twin predicts this
    # exactly - the arm tracks commanded joints to within 0.03 rad with nothing
    # saturating - so the correction needs no physics and no fitted constant.
    # Both horizontal components are corrected: the uncommanded forward drift
    # (+45 mm one direction) is as much of the placement error as the lateral
    # shortfall.
    want = place_shift * dh
    pv = want.copy()
    for it in range(demo.place_correct_iters + 1):
        seed = P["grasp"]
        for name, off, sp in (("lift",  o + demo.lift_h * up_v,          None),
                              ("move",  o + pv + demo.lift_h * up_v,     None),
                              ("lower", o + pv - demo.seat_depth * up_v, None)):
            P[name], _ = book.solve(off, sep=sp, seed=seed, sep_dir=sep_dir)
            seed = P[name]
        if it == demo.place_correct_iters:
            break                      # chain always solved at least once
        got = book.palm_mid() - grasp_mid
        errv = want - got
        errv[2] = 0.0
        if float(np.linalg.norm(errv)) < 5e-5:
            break
        pv = pv + errv

    seed = P["lower"]
    for name, off, sp in (("open",  o + pv,                free),
                          ("clear", o + pv + h * up_v,     free),
                          ("out",   sx + h * up_v,         free)):
        P[name], _ = book.solve(off, sep=sp, seed=seed, sep_dir=sep_dir)
        seed = P[name]
    # ---- staged raise (O26 mitigation) ----------------------------------
    # A straight palm path up the body and then out, solved sub-waypoint by
    # sub-waypoint. `home_palm` is (height above the pelvis, separation) of the
    # palms at the home pose, MEASURED on the stepped model by the caller - the
    # raise has to start where the hands actually are, not where a constant says
    # they are. Without it the old two-segment reach is kept, which is what the
    # retired stationary variant and any diagnostic caller get.
    if demo.staged_reach and home_palm is not None:
        hz, hsep = float(home_palm[0]), float(home_palm[1])
        p0 = (0.0, 0.0, hz, hsep)                       # at the body, home height
        p1 = (0.0, 0.0, z + demo.raise_top, demo.raise_sep)   # above the slab
        p2 = (x, y, z + h, wide)                        # == P["high"]
        pts, secs = [], []
        for i in range(1, demo.raise_sub + 1):
            t = i / demo.raise_sub
            pts.append(tuple(a + t * (b - a) for a, b in zip(p0, p1)))
            secs.append(demo.reach_s * 0.5 / demo.raise_sub)
        for i in range(1, demo.out_sub + 1):
            t = i / demo.out_sub
            pts.append(tuple(a + t * (b - a) for a, b in zip(p1, p2)))
            secs.append(demo.reach_s * 0.5 / demo.out_sub)
        chain, seed = [], None
        for fx, fy, fz, fsep in pts[:-1]:
            q, _ = book.solve(np.array([fx, fy, fz]), sep=fsep, seed=seed,
                              sep_dir=sep_dir)
            chain.append(q)
            seed = q
        # The last waypoint IS P["high"], solved the existing way (anchored back
        # at the grasp), so REPOSITION and APPROACH start from exactly the pose
        # they always did and nothing downstream of REACH moves.
        chain.append(P["high"])
        P["reach"] = chain
        P["reach_secs"] = secs
    return P, e


def schedule_of(demo: "DemoConfig", P: dict, home):
    """(phase, duration, from-pose, to-pose). A phase may appear more than once:
    the recorder's label is the phase, not the waypoint."""
    if P.get("reach"):
        reach, prev = [], home
        for q, dur in zip(P["reach"], P["reach_secs"]):
            reach.append((Phase.REACH, dur, prev, q))
            prev = q
    else:
        reach = [(Phase.REACH, demo.reach_s * 0.5, home, P["up"]),
                 (Phase.REACH, demo.reach_s * 0.5, P["up"], P["high"])]
    return [(Phase.SETTLE, demo.settle_s, home, home)] + reach + [
            (Phase.REPOSITION, demo.reposition_s, P["high"], P["high"]),
            (Phase.APPROACH, demo.approach_s, P["high"], P["side"]),
            (Phase.GRASP, demo.grasp_s, P["side"], P["grasp"]),
            (Phase.LIFT, demo.lift_s, P["grasp"], P["lift"]),
            # Walking place: MOVE holds the lift pose while the LEGS carry the
            # box to the goal. Stationary place: MOVE sweeps the arms sideways.
            (Phase.MOVE,
             demo.move_s_walk if demo.walk_place else demo.move_s,
             P["lift"], P["lift"] if demo.walk_place else P["move"]),
            (Phase.LOWER, demo.lower_s,
             P["lift"] if demo.walk_place else P["move"], P["lower"]),
            (Phase.RELEASE, demo.release_s, P["lower"], P["open"]),
            (Phase.RELEASE, demo.retract_s * 0.5, P["open"], P["clear"]),
            (Phase.RELEASE, demo.retract_s * 0.5, P["clear"], P["out"]),
            (Phase.VERIFY, demo.verify_s_free if demo.base_lock else demo.verify_s,
             P["out"], P["out"])]


def assert_recordable(demo: "DemoConfig") -> None:
    """Precondition the recorder must check before logging an episode.

    `box_xy` and `stop_after` are sweep hooks that fail SILENTLY rather than
    loudly: `box_xy` overrides the seeded spawn, so every episode would record
    the same box position and Objective 4 would have no spatial variation at all;
    `stop_after` truncates the schedule, so the episode would be logged without
    its place and release. Neither raises on its own, and neither is visible in
    the recorded state vector - the dataset would simply be wrong.
    """
    bad = [n for n in ("box_xy", "stop_after") if getattr(demo, n) is not None]
    if bad:
        raise AssertionError(
            "refusing to record with sweep hooks set: %s. These are diagnostic "
            "overrides (O18/O22 sweeps); leave them None for collection."
            % ", ".join("%s=%r" % (n, getattr(demo, n)) for n in bad))


def _ease(a):
    return 0.5 - 0.5 * np.cos(np.pi * float(np.clip(a, 0.0, 1.0)))


def run_episode(seed: int, cfg: Optional[TeleopConfig] = None,
                demo: Optional[DemoConfig] = None, book: Optional[PoseBook] = None,
                policy=None, verbose: bool = False) -> dict:
    """One scripted stationary episode. Returns a scored result dict."""
    import torch
    import walk_test as W

    cfg = cfg or TeleopConfig()
    demo = demo or DemoConfig()
    book = book or PoseBook(cfg, demo)
    if policy is None:
        policy = torch.jit.load(W.POLICY_PATH)

    L = cfg.loco
    dt = L.sim_dt
    DEF = np.asarray(L.default_angles, dtype=np.float64)
    KPS = np.asarray(L.kps, dtype=np.float32)
    KDS = np.asarray(L.kds, dtype=np.float32)
    CMDS = np.asarray(L.cmd_scale, dtype=np.float32)

    if demo.fixed_base is None:
        cfg.grasp.assert_standoff(demo.standoff, "scripted demonstrator")
    if demo.start_xy is not None and demo.box_xy is None:
        # O22: the walk-in drives the base to box_x - standoff, so the far
        # sample edge has to leave it clear of max_base_x with margin. The
        # load-time `assert_reach_fits` cannot see this - it checks grasp_max
        # (limit 1.615), while the walk-in commits to the NOMINAL standoff.
        far = cfg.box.pickup_center[0] + cfg.box.pickup_half[0]
        need = cfg.grasp.max_base_x - 0.015
        assert far - demo.standoff <= need + 1e-9, (
            "far sample edge x=%.3f at standoff %.2f puts the base at %.3f, "
            "past the usable limit %.3f (max_base_x %.3f less 15 mm margin). "
            "Trim BoxConfig.pickup_half[0]." % (far, demo.standoff,
                                                far - demo.standoff, need,
                                                cfg.grasp.max_base_x))
    # With a fixed base the standoff is whatever the spawn makes it - measuring
    # which spawns work is the point, so the band is reported, not asserted.

    m = mujoco.MjModel.from_xml_path(cfg.model_path)
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    ix = ModelIndex.resolve(m)
    carry = LocomotionCarryover()
    start = reset_episode(m, d, ix, cfg, seed, carry=carry, policy=policy)
    reset_policy_state(policy)
    if demo.hand_platform_filter:
        # BEFORE GraspWeld: it saves the pad bitmasks now and restores them on
        # release, so it must save the filtered values, not the originals.
        from g1_teleop.contact_filter import apply_hand_platform_filter
        if demo.hand_platform_filter == "pickup":
            apply_hand_platform_filter(m, ("platform_pickup_geom",))
        else:
            apply_hand_platform_filter(m)
    weld = GraspWeld(m)

    # Stationary variant: place the base at a valid standoff and leave it there.
    if demo.box_xy is not None:
        d.qpos[ix.box_qpos][0] = float(demo.box_xy[0])
        d.qpos[ix.box_qpos][1] = float(demo.box_xy[1])
        mujoco.mj_forward(m, d)
    box0 = d.qpos[ix.box_qpos][:3].copy()
    if demo.start_xy is not None:
        # Walk-in: the base starts away from the box and the station-keeping in
        # SETTLE drives it to the standoff.
        base_x, base_y = float(demo.start_xy[0]), float(demo.start_xy[1])
    elif demo.fixed_base is not None:
        base_x, base_y = float(demo.fixed_base[0]), float(demo.fixed_base[1])
    else:
        base_x = min(box0[0] - demo.standoff, cfg.grasp.max_base_x)
        base_y = float(box0[1])
    d.qpos[ix.base_qpos][0] = base_x
    d.qpos[ix.base_qpos][1] = base_y
    mujoco.mj_forward(m, d)
    carry.reset(d, ix, L)

    pel_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    dz = float(box0[2] - d.xpos[pel_bid][2])

    # Place TOWARDS the platform's centre line, not always in +y. A fixed +0.20 m
    # shift carries the box off the far edge for any spawn with y >~ 0.12 (the
    # platform's y half-extent is 0.32 and the box's own half-width is 0.09), and
    # it lands on the floor - measured, 3 of 10 seeds. Moving toward the centre
    # keeps every sampled spawn on the platform while still displacing the box by
    # `place_dy`, which is twice the 0.10 m Q4 tolerance, so "did not move it"
    # still fails.
    if demo.walk_place:
        place_shift = 0.0          # no lateral arm sweep at all; the legs move it
    else:
        place_shift = (demo.place_shift_override if demo.place_shift_override is not None
                       else (-demo.place_dy if box0[1] > 0.0 else demo.place_dy))
    if not demo.walk_place:
        plat_half_y = 0.32 - 0.09
        assert abs(box0[1] + place_shift) <= plat_half_y, (
            "place target y=%.3f leaves the platform" % (box0[1] + place_shift))

    def build(offset_xyz, sep_dir=None):
        # Where the palms ACTUALLY are right now, on the stepped model - the
        # staged raise has to start from the hands' real position, not from a
        # constant. Every caller of `build` is at the home pose (SETTLE holds
        # it, and the lock-instant re-solve happens before REACH begins).
        hp = None
        if demo.staged_reach:
            pl = d.site_xpos[weld.site_l]
            pr = d.site_xpos[weld.site_r]
            hp = (float(0.5 * (pl[2] + pr[2]) - d.xpos[pel_bid][2]),
                  float(np.linalg.norm(pl - pr)))
        return build_poses(book, demo, offset_xyz, place_shift, sep_dir=sep_dir,
                           home_palm=hp)

    def schedule(P):
        return schedule_of(demo, P, home)

    def lay_out(segs):
        """[phase, duration in steps, from-pose, to-pose], in order.

        Was a list of absolute (lo, hi) step bounds. It is a duration list now
        because two of the phases end on an EVENT, so no boundary after them has
        a fixed step index any more. The loop walks it with a cursor instead.
        Durations are unchanged and remain the upper bound for every phase.
        """
        return [[ph, max(int(dur / dt), 1), a_, b_] for ph, dur, a_, b_ in segs]

    home = np.array(d.qpos[ix.upper_qpos], dtype=np.float64)
    poses, err_grasp = build([demo.standoff, 0.0, dz])
    lock = BaseLock(m) if demo.base_lock else None
    locked_set = LOCKED_PHASES_WALKING if demo.walk_place else LOCKED_PHASES
    # Event-driven lock (D12, 2026-09-10). The stationary variant keeps the
    # phase-set trigger: it needs the base held THROUGH its lateral MOVE sweep,
    # and the predicate's release condition (box lifted and carried) fires right
    # at the start of that sweep.
    geo = PlatformGeometry.resolve(m)
    sp = spec.SpecLayout.resolve(m, ix)
    pred = (LockPredicate(geo, LockConfig())
            if (demo.lock_predicate and demo.walk_place and lock is not None)
            else None)
    solved_at_goal = False
    # With the base locked the poses are solved once, at the lock instant, from
    # the pelvis pose they will actually be executed against. That closes
    # TR14(d) - precomputed arm poses applied to a differently-posed base - and
    # is still TR15-safe: one solve at a static state, then open loop forever.
    solved_at_lock = False
    # Base station-keeping target: hold the STANDOFF to the box, not a fixed
    # world point. Closing the loop on the BASE leaves arm commands open-loop,
    # so TR15's base-state -> arm-command path stays closed.
    base_goal = box0[:2] - np.array([demo.standoff, 0.0])
    hold_frozen = False
    yaw_goal = float("nan")
    resolved = False

    def trim(segs):
        if not demo.stop_after:
            return segs
        keep, hit = [], False
        for seg in segs:
            if hit and seg[0].name != demo.stop_after:
                break
            keep.append(seg)
            hit = hit or seg[0].name == demo.stop_after
        return keep

    segs = lay_out(trim(schedule(poses)))
    # With two phases ending on an event, the episode length is no longer a
    # constant. `total` is now the WORST CASE - every phase running to its full
    # duration - so the loop is still bounded and a predicate that never fires
    # can neither hang the episode nor make it unbounded.
    total = sum(s[1] for s in segs)
    seg_i, seg_start = 0, 0
    lock_ticks, release_ticks, fallbacks = [], [], []
    min_clear, clear_at_lock2 = float("inf"), float("nan")
    goal_standoff = float("nan")
    phase_steps = {}

    place_target =(np.array([demo.goal_xy[0], demo.goal_xy[1], box0[2]])
                    if demo.walk_place else box0 + np.array([0.0, place_shift, 0.0]))
    worst_pitch, fell, engaged_at, gated = 0.0, False, None, False
    standoff_at_grasp = float("nan")
    # Diagnostics the brief asks for: what the lock costs, and proof that no
    # number here was produced by the robot propping on the platform (TR16).
    lock_f = lock_t = 0.0
    lock_hold = []
    walk_trace = []
    lock_standoff = lock_lateral = float("nan")
    locked_leg_pos = None
    grasp_meas = None
    palm_goal_err = float("inf")
    waist_ach = float("nan")
    support = {}            # body name -> peak normal force from a NON-foot contact
    foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    # ---- O26 metrics (2026-09-11) ---------------------------------------
    # Hand-vs-pickup-platform contact, and wrist deviation from command.
    # "Clearance" is measured as contact PENETRATION DEPTH (`contact.dist`,
    # negative when interpenetrating), not with `mj_geomDistance`: that call
    # returns +0.0 for box-vs-mesh pairs at real positive separation in this
    # build (TR19). Depth 0.0 therefore means "never touched", and a positive
    # clearance is deliberately not claimed - it cannot be measured reliably.
    _plat_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                "platform_pickup_geom")
    _hand_g = set()
    for _g in range(m.ngeom):
        _b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                               int(m.geom_bodyid[_g]))
        if _b and ("wrist" in _b or "pad" in _b or "elbow" in _b)                 and int(m.geom_contype[_g]) != 0:
            _hand_g.add(_g)
    _wrist_ja = []
    for _nm in ("left_wrist_roll_joint", "left_wrist_pitch_joint",
                "left_wrist_yaw_joint", "right_wrist_roll_joint",
                "right_wrist_pitch_joint", "right_wrist_yaw_joint"):
        _j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, _nm)
        _a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, _nm)
        _wrist_ja.append((int(m.jnt_qposadr[_j]), int(_a), _nm,
                          "pitch" in _nm))
    hand_hits, hand_depth, hand_force = 0, 0.0, 0.0
    # Two different quantities, and conflating them scores every passing
    # episode as a failure. The 2026-09-10 audit measured ALL 12 seeds wedging
    # a wrist transiently DURING the reach, up to 1.42 rad, and recovering.
    # What loses an episode is the wedge that does NOT recover, and D2 makes
    # that permanent: the wrist command never changes after REACH, so a jam
    # that survives into REPOSITION survives to the end. So:
    #   wrist_dev_reach  - the transient, reported only
    #   wrist_dev        - after REACH, the permanent one, and the gate
    #
    # And the gate watches the wrist PITCH joints only. Measured: after REACH
    # the left wrist YAW sits 0.55-0.57 rad off command in the configuration
    # that passes 12/12, because the welded box hangs off that link and the
    # joint is rated 5 N*m. Gating on all six at 0.5 rad would score every
    # passing episode as a wedge. The joints that actually jam are the pitch
    # pair (1.42 rad in the diagnosed failure, ~0.3 rad when healthy).
    wrist_dev_reach = 0.0
    wrist_dev_any = 0.0
    wrist_dev, wrist_dev_joint, wrist_dev_phase = 0.0, "", ""
    released_pitch, released_z, released_fell = 0.0, float("nan"), False
    _ff = np.zeros(6)
    rel0, drift = None, np.zeros(3)
    carry_drift, drift_by_phase, trace = 0.0, {}, []
    box_at_lower = box_at_release = None
    fail_phase = None

    for i in range(total):
        if seg_i >= len(segs):
            break                       # every phase ended early; episode done
        phase, seg_dur, pa, pb = segs[seg_i]
        elapsed = i - seg_start
        # Interpolation stays DURATION-based even in the two event-ended phases:
        # the event moves the boundary, it does not rescale the ramp. Both of
        # those phases hold one pose anyway, so nothing is truncated.
        frac = min(elapsed / max(seg_dur, 1), 1.0)
        cmd = 1.0 if (phase in (Phase.LIFT, Phase.MOVE, Phase.LOWER)
                      or (phase is Phase.GRASP and frac >= 0.5)) else 0.0

        # ---- the lock predicate, evaluated on the 47-D state at 25 Hz -------
        # `sync=False`: building the state does NOT refresh forward kinematics,
        # so `weld.update` and the contact audit below still see exactly what
        # they saw before this change. The cost is that the palm sites lag qpos
        # by one 2 ms step, which moves `palm_far` by well under a millimetre
        # against a 78 mm margin.
        event = None
        if pred is not None and i % spec.PHYSICS_STEPS_PER_TICK == 0:
            event = pred.update(sp.build_state(m, d, ix, sync=False))
            if event == "lock":
                lock_ticks.append(i // spec.PHYSICS_STEPS_PER_TICK)
            elif event == "release":
                release_ticks.append(i // spec.PHYSICS_STEPS_PER_TICK)

        # ONE discrete re-solve at a settled state, free-base path only. With
        # the base locked the solve already happens at the lock instant, which
        # is strictly better: the base cannot move afterwards at all.
        if (demo.discrete_resolve and lock is None and not resolved
                and phase is Phase.GRASP):
            live = d.qpos[ix.box_qpos][:3] - d.xpos[pel_bid]
            poses, _ = build([float(live[0]), 0.0, float(live[2])])
            segs = lay_out(schedule(poses))
            resolved = True
            phase, seg_dur, pa, pb = segs[seg_i]

        # ---- base support (D12). The edges come from the lock PREDICATE over
        # the 47-D state (2026-09-10), or from phase-set membership on the
        # retired stationary path. Never from a timer, either way.
        if lock is not None:
            want_lock = pred.locked if pred is not None else phase in locked_set
            if want_lock and not lock.locked(d):
                lock.lock(m, d)
                locked_leg_pos = DEF.copy()
                if not solved_at_lock:
                    # Re-solve from the pelvis we are actually locked at, using
                    # the MEASURED pelvis->box offset rather than the nominal
                    # standoff: the base has been station-keeping through SETTLE
                    # and is not exactly where it was placed. Using the nominal
                    # number here would reintroduce TR14(d) verbatim.
                    live = d.qpos[ix.box_qpos][:3] - d.xpos[pel_bid]
                    # O20 fixed: rotate the WORLD box delta into the pelvis frame
                    # AND carry the separation axis with it. Both or neither -
                    # rotating the offset alone put the palms on the box's
                    # corners and scored worse than doing nothing (10/10 -> 3/10).
                    # The box's faces are world-fixed, so the palms must straddle
                    # along world y, which in the pelvis frame is
                    # R(-yaw) . y_world = (sin yaw, cos yaw, 0).
                    yaw_b = float(W.yaw_from_quat(d.qpos[ix.base_quat_qpos]))
                    cy, sy = np.cos(yaw_b), np.sin(yaw_b)
                    fwd_o = float(cy * live[0] + sy * live[1])
                    lat_o = float(-sy * live[0] + cy * live[1])
                    sep_dir = np.array([sy, cy, 0.0])
                    x_off = fwd_o
                    lock_standoff = x_off
                    lock_lateral = lat_o
                    poses, err_grasp = build([x_off, lat_o, float(live[2])],
                                             sep_dir=sep_dir)
                    # Swap the poses in, keep the cursor: the durations did not
                    # change, and re-laying out from step 0 would throw away how
                    # far into the current phase we are.
                    segs = lay_out(trim(schedule(poses)))
                    phase, seg_dur, pa, pb = segs[seg_i]
                    solved_at_lock = True
                elif demo.walk_place and not solved_at_goal:
                    # Keyed on "this is the SECOND lock", not on `phase is LOWER`.
                    # With the predicate driving, the base re-locks on the last
                    # tick of MOVE - the arrival at the goal is what fires it -
                    # so a phase test here would never match and the place chain
                    # would silently keep the pickup solve (TR14(d)).
                    # Second lock, at the goal platform. Re-solve the place chain
                    # from the pelvis we are actually standing at now - the base
                    # walked here, so nothing from the pickup solve applies. The
                    # box is welded to the hand, so the target is the goal
                    # platform's centre at resting height.
                    tgt = np.array([demo.goal_xy[0], demo.goal_xy[1], box0[2]])
                    rel = tgt - d.xpos[pel_bid]
                    yaw_g = float(W.yaw_from_quat(d.qpos[ix.base_quat_qpos]))
                    cg, sg = np.cos(yaw_g), np.sin(yaw_g)
                    fwd_g = float(cg * rel[0] + sg * rel[1])
                    lat_g = float(-sg * rel[0] + cg * rel[1])
                    sdir_g = np.array([sg, cg, 0.0])
                    poses_g, _ = build([fwd_g, lat_g, float(rel[2])], sep_dir=sdir_g)
                    goal_standoff = fwd_g
                    clear_at_lock2 = geo.clearance(d.qpos[ix.box_qpos][:3])
                    newb, seen_rel = [], 0
                    for ph_, dur_, a_, b_ in segs:
                        if ph_ is Phase.LOWER:
                            newb.append([ph_, dur_, poses["lift"], poses_g["lower"]])
                        elif ph_ is Phase.RELEASE:
                            pair = ((poses_g["lower"], poses_g["open"]),
                                    (poses_g["open"], poses_g["clear"]),
                                    (poses_g["clear"], poses_g["out"]))[seen_rel]
                            seen_rel += 1
                            newb.append([ph_, dur_, pair[0], pair[1]])
                        elif ph_ is Phase.VERIFY:
                            newb.append([ph_, dur_, poses_g["out"], poses_g["out"]])
                        else:
                            newb.append([ph_, dur_, a_, b_])
                    segs = newb
                    phase, seg_dur, pa, pb = segs[seg_i]
                    solved_at_goal = True
            elif not want_lock and lock.locked(d):
                lock.release(m, d, policy=policy)
                locked_leg_pos = None
                carry.action = np.zeros_like(carry.action)
                carry.target_leg_pos = DEF.copy()

        lq, ldq = d.qpos[ix.leg_qpos], d.qvel[ix.leg_qvel]
        # While locked the policy is not queried; the legs are PD-held at
        # DEFAULT_ANGLES so the release hands back the configuration the policy
        # expects (TR5). TR2 does not apply - balance is supplied by the weld.
        leg_target = carry.target_leg_pos if locked_leg_pos is None else locked_leg_pos
        d.ctrl[ix.leg_ctrl] = (leg_target - lq) * KPS + (0.0 - ldq) * KDS
        d.ctrl[ix.upper_ctrl] = pa + _ease(frac) * (pb - pa)
        # Pads stay retracted (D11). The gripper action dim is the WELD command;
        # the pads exist only for contact detection and the trigger is purely
        # geometric, so they are never needed to press. Driving them with `cmd`
        # extends them 0.10 m into a box whose contact is disabled while welded,
        # and `GraspWeld.release` then restores that contact with the pads buried
        # inside it - measured, this ejects the box ~145 mm and tips it off the
        # platform.
        d.ctrl[ix.pad_ctrl] = 0.0
        eng, mm = weld.update(m, d, cmd)
        if eng and engaged_at is None:
            engaged_at = i
        if phase is Phase.GRASP:
            gated = gated or mm["gated"]
            if np.isnan(standoff_at_grasp):
                standoff_at_grasp = float(np.linalg.norm(
                    d.qpos[ix.box_qpos][:2] - d.qpos[ix.base_xy_qpos]))
            # TR16 trap 1: `err_grasp` is the IK residual on the kinematic twin.
            # What decides the grasp is the ACHIEVED palm geometry on stepped
            # state, so keep the best moment GraspWeld itself sees.
            if grasp_meas is None or max(mm["d_left"], mm["d_right"]) < max(
                    grasp_meas["d_left"], grasp_meas["d_right"]):
                grasp_meas = dict(mm)
            # The ACHIEVED analogue of the 45 mm O14 palm guard: distance from
            # each palm site to the point it was aimed at (box centre +- sep/2),
            # on stepped state. The guard was originally read off the kinematic
            # twin's IK residual, which TR16 showed can understate this by an
            # order of magnitude.
            bc = d.xpos[weld.box_bid]
            half = np.array([0.0, demo.sep / 2.0, 0.0])
            e_ = max(float(np.linalg.norm(d.site_xpos[weld.site_l] - (bc + half))),
                     float(np.linalg.norm(d.site_xpos[weld.site_r] - (bc - half))))
            palm_goal_err = min(palm_goal_err, e_)
            # Waist tracking. The IK is solved on a twin holding `waist_yaw`
            # exactly; if the physical waist undershoots, every arm pose is
            # rotated wrongly and the palm error grows with the commanded yaw.
            waist_ach = float(d.qpos[ix.upper_qpos][0])

        # TR16 trap 2: a number produced by the robot propping on the platform
        # is not a result. Audit every non-foot contact on the robot.
        # Audit every phase, not just the ones around the grasp. Limiting this to
        # APPROACH/GRASP/LIFT hid pad-vs-platform contact during REACH entirely.
        if phase is not Phase.SETTLE:
            for c_ in range(d.ncon):
                g1_, g2_ = d.contact[c_].geom1, d.contact[c_].geom2
                for gg, other in ((g1_, g2_), (g2_, g1_)):
                    nb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                           int(m.geom_bodyid[int(gg)]))
                    ob = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                           int(m.geom_bodyid[int(other)]))
                    if nb in (None, "world", "box1") or nb in foot_names:
                        continue
                    if ob in ("box1",) or ob is None:
                        continue          # hand-on-box is the task, not support
                    mujoco.mj_contactForce(m, d, c_, _ff)
                    f_ = abs(float(_ff[0]))
                    if f_ > support.get(nb, 0.0):
                        support[nb] = f_

        # ---- O26 metrics, every step (contacts) -------------------------
        for c_ in range(d.ncon):
            g1_, g2_ = d.contact[c_].geom1, d.contact[c_].geom2
            if _plat_g in (g1_, g2_) and (g1_ in _hand_g or g2_ in _hand_g):
                hand_hits += 1
                hand_depth = min(hand_depth, float(d.contact[c_].dist))
                mujoco.mj_contactForce(m, d, c_, _ff)
                hand_force = max(hand_force, abs(float(_ff[0])))
        if i % 20 == 0:
            _post = phase not in (Phase.SETTLE, Phase.REACH)
            for _qa, _aa, _nm, _is_pitch in _wrist_ja:
                _e = abs(float(d.qpos[_qa] - d.ctrl[_aa]))
                if not _post:
                    if _is_pitch:
                        wrist_dev_reach = max(wrist_dev_reach, _e)
                    continue
                wrist_dev_any = max(wrist_dev_any, _e)
                if _is_pitch and _e > wrist_dev:
                    wrist_dev, wrist_dev_joint = _e, _nm
                    wrist_dev_phase = phase.name

        mujoco.mj_step(m, d)
        carry.counter += 1
        if lock is not None and lock.locked(d):
            f, tq = lock.reaction(m, d)
            lock_f, lock_t = max(lock_f, f), max(lock_t, tq)
            if phase in (Phase.GRASP, Phase.LIFT, Phase.MOVE):
                lock_hold.append(f)
        if carry.counter % L.control_decimation == 0 and locked_leg_pos is None:
            # Track the standoff to the box until the grasp, then freeze: once
            # welded the box moves with the hand, so tracking it would chase.
            if demo.walk_place and phase is Phase.MOVE:
                # Transport leg: navigate to the goal platform at the same
                # standoff and facing used for the pickup.
                # `goal_standoff_bias` back-offs the target because the walk
                # stops SHORT of it: measured deficit 21-41 mm, which put 2 of
                # 12 arrivals at 0.279 m, under the [0.28, 0.36] band the pickup
                # asserts. Same class of miss as O22 - a constant nothing could
                # assert on, because the standoff here is an outcome.
                carry.hold_target[:] = (
                    demo.goal_xy[0] - demo.standoff - demo.goal_standoff_bias,
                    demo.goal_xy[1])
                to_goal = np.asarray(demo.goal_xy) - d.qpos[ix.base_xy_qpos]
                if np.linalg.norm(to_goal) > 1e-6:
                    yaw_goal = float(np.arctan2(to_goal[1], to_goal[0]))
                hold_frozen = False
            elif demo.fixed_base is not None:
                # Hold the WORLD position, not a standoff to the box. Tracking
                # the box here silently undoes `fixed_base`: the base walks to
                # the nominal standoff during SETTLE, before the lock, and every
                # base position then reports the same 0.33 m standoff.
                carry.hold_target[:] = (base_x, base_y)
            elif phase in (Phase.SETTLE, Phase.REACH, Phase.REPOSITION,
                           Phase.GRASP) and not hold_frozen:
                carry.hold_target[:] = d.qpos[ix.box_qpos][:2] - np.array(
                    [demo.standoff, 0.0])
            elif not hold_frozen:
                hold_frozen = True
            err = carry.hold_target - d.qpos[ix.base_xy_qpos]
            if demo.start_xy is not None and phase is Phase.SETTLE:
                walk_trace.append((i * dt,
                                   float(d.qpos[ix.base_qpos][0]),
                                   float(d.qpos[ix.base_qpos][1]),
                                   float(W.yaw_from_quat(d.qpos[ix.base_quat_qpos])),
                                   float(np.linalg.norm(err))))
            act = np.zeros(3, dtype=np.float32)
            yaw = W.yaw_from_quat(d.qpos[ix.base_quat_qpos])
            if np.linalg.norm(err) >= demo.hold_deadband:
                act[0:2] = np.clip(demo.hold_kp * W.world_to_body(err, yaw),
                                   -demo.hold_max, demo.hold_max)
            # Close the loop on HEADING too: face the box. Frozen with the rest
            # of the hold once the grasp is underway, so the welded box moving
            # with the hand cannot drag the heading target around.
            # During the walking transport the heading target is the GOAL, set
            # in the navigation block above. It must not be recomputed from the
            # box: the box is welded to the hand and travels with the robot, so
            # its bearing is meaningless here.
            if not hold_frozen and not (demo.walk_place and phase is Phase.MOVE):
                to_box = d.qpos[ix.box_qpos][:2] - d.qpos[ix.base_xy_qpos]
                if np.linalg.norm(to_box) > 1e-6:
                    yaw_goal = float(np.arctan2(to_box[1], to_box[0]))
            if yaw_goal == yaw_goal:
                e_yaw = float(np.arctan2(np.sin(yaw_goal - yaw), np.cos(yaw_goal - yaw)))
                if abs(e_yaw) >= demo.hold_deadband_yaw:
                    act[2] = float(np.clip(demo.hold_kp_yaw * e_yaw,
                                           -demo.hold_max_yaw, demo.hold_max_yaw))
            n = L.num_actions
            phz = ((carry.counter * dt) % L.gait_period) / L.gait_period
            carry.obs[:3] = d.qvel[ix.base_angvel_qvel] * L.ang_vel_scale
            carry.obs[3:6] = W.get_gravity_orientation(d.qpos[ix.base_quat_qpos])
            carry.obs[6:9] = act * CMDS
            carry.obs[9:9 + n] = (d.qpos[ix.leg_qpos] - DEF) * L.dof_pos_scale
            carry.obs[9 + n:9 + 2 * n] = d.qvel[ix.leg_qvel] * L.dof_vel_scale
            carry.obs[9 + 2 * n:9 + 3 * n] = carry.action
            carry.obs[9 + 3 * n:9 + 3 * n + 2] = [np.sin(2 * np.pi * phz), np.cos(2 * np.pi * phz)]
            carry.action = policy(torch.from_numpy(carry.obs).unsqueeze(0)).detach().numpy().squeeze()
            carry.target_leg_pos = carry.action * L.action_scale + DEF

        worst_pitch = max(worst_pitch, abs(_pitch_deg(d.qpos[ix.base_quat_qpos])))
        if d.qpos[ix.base_qpos][2] < demo.fall_z and not fell:
            fell, fail_phase = True, phase.name
        if phase is Phase.LOWER:
            box_at_lower = d.qpos[ix.box_qpos][:3].copy()
        elif phase is Phase.RELEASE:
            box_at_release = d.qpos[ix.box_qpos][:3].copy()
        if lock is not None and phase is Phase.VERIFY:
            released_pitch = max(released_pitch, abs(_pitch_deg(d.qpos[ix.base_quat_qpos])))
            released_z = float(d.qpos[ix.base_qpos][2])
            released_fell = released_fell or released_z < demo.fall_z

        if eng:
            neg = np.zeros(4)
            mujoco.mju_negQuat(neg, d.xquat[weld.hand_bid])
            rel = np.zeros(3)
            mujoco.mju_rotVecQuat(
                rel, d.qpos[ix.box_qpos][:3] - d.xpos[weld.hand_bid], neg)
            if rel0 is None:
                rel0 = rel.copy()
            drift = rel - rel0
            if phase in (Phase.LIFT, Phase.MOVE):
                carry_drift = max(carry_drift, float(np.linalg.norm(drift)))
            drift_by_phase[phase.name] = float(np.linalg.norm(drift))
        if demo.trace_every and i % demo.trace_every == 0:
            trace.append((i * dt, phase.name, int(eng),
                          float(d.xpos[weld.hand_bid][2]),
                          float(d.qpos[ix.box_qpos][2]),
                          float(d.xpos[weld.box_bid][2]),
                          float(np.linalg.norm(drift))))
        if verbose and i % 1000 == 0:
            print("    %-8s t=%5.2f eng=%d base_z=%.3f pitch=%+5.1f box=%s"
                  % (phase.name, i * dt, int(eng), d.qpos[ix.base_qpos][2],
                     _pitch_deg(d.qpos[ix.base_quat_qpos]),
                     np.round(d.qpos[ix.box_qpos][:3], 3)))

        # ---- transport clearance (rejection criterion, 2026-09-10) ----------
        # Only during MOVE, and only while the box is actually over a platform
        # footprint: LIFT starts with the box resting and LOWER ends with it
        # resting, so a minimum taken across those would always read zero and
        # say nothing. Between the platforms the box is over the floor and
        # `clearance` returns NaN, which is skipped.
        if phase is Phase.MOVE:
            c_ = geo.clearance(d.qpos[ix.box_qpos][:3])
            if c_ == c_:
                min_clear = min(min_clear, c_)

        # ---- phase boundary ------------------------------------------------
        # SETTLE and MOVE end on the lock event; every other phase runs its
        # duration. The duration is still the upper bound for the two event
        # phases, so a predicate that never fires costs a flagged episode, not
        # a hung or unbounded one.
        phase_steps[phase.name] = phase_steps.get(phase.name, 0) + 1
        by_event = (event == "lock" and phase in EVENT_ENDED_PHASES)
        by_time = elapsed + 1 >= seg_dur
        if by_event or by_time:
            if pred is not None and phase in EVENT_ENDED_PHASES and not by_event:
                fallbacks.append(phase.name)
            seg_i += 1
            seg_start = i + 1

    # ---- score -------------------------------------------------------------
    box_f = d.qpos[ix.box_qpos][:3].copy()
    M = np.zeros(9); mujoco.mju_quat2Mat(M, d.qpos[ix.box_qpos][3:7])
    tilt = float(np.degrees(np.arccos(np.clip(M.reshape(3, 3)[2, 2], -1, 1))))
    plat_g = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_GEOM,
        "platform_goal_geom" if demo.walk_place else "platform_pickup_geom")
    box_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box1_geom")
    resting = any({d.contact[c].geom1, d.contact[c].geom2} == {plat_g, box_g}
                  for c in range(d.ncon))
    place_err = float(np.linalg.norm(box_f[:2] - place_target[:2]))

    # Transport clearance: a demonstration where the box drags the platform edge
    # must not enter the dataset silently. `inf` means the box was never over a
    # platform during MOVE, which is not a pass - it means the metric never ran.
    n_steps = sum(phase_steps.values())
    scraped = bool(np.isfinite(min_clear) and min_clear < demo.min_clearance)
    # O26: a wedge is a wrist driven far off its command and held there. D2
    # makes it permanent - the wrist command never changes after REACH - so any
    # large deviation is unrecoverable and the episode is lost. This is the
    # gate criterion the staged raise is judged on.
    wedged = bool(wrist_dev > demo.wedge_rad)
    goal_standoff_ok = bool(
        not demo.walk_place or not np.isfinite(goal_standoff)
        or cfg.grasp.grasp_min <= goal_standoff <= cfg.grasp.grasp_max)

    if wedged:
        fail_phase = fail_phase or ("WEDGE(%s)" % wrist_dev_joint)
    if engaged_at is None:
        fail_phase = fail_phase or "GRASP"
    elif scraped:
        fail_phase = fail_phase or "TRANSPORT(clearance)"
    elif place_err > 0.10:
        fail_phase = fail_phase or "PLACE(xy)"
    elif not resting:
        fail_phase = fail_phase or "PLACE(resting)"
    elif tilt > 15.0:
        fail_phase = fail_phase or "PLACE(upright)"

    ok = (engaged_at is not None and place_err <= 0.10 and resting
          and tilt <= 15.0 and not fell and not scraped and not wedged)
    return dict(seed=seed, ok=ok, fail_phase=fail_phase, engaged=engaged_at is not None,
                gated=gated, palm_err_mm=err_grasp * 1000, place_err_m=place_err,
                tilt_deg=tilt, resting=resting, drift_z_mm=float(drift[2]) * 1000,
                drift_xy_mm=float(np.linalg.norm(drift[:2])) * 1000,
                max_pitch_deg=worst_pitch, fell=fell, base_z=float(d.qpos[ix.base_qpos][2]),
                standoff_at_grasp=standoff_at_grasp,
                lock_force_N=lock_f, lock_torque_Nm=lock_t,
                lock_hold_N=float(np.median(lock_hold)) if lock_hold else float("nan"),
                carry_drift_mm=carry_drift * 1000.0,
                box_at_lower=box_at_lower, box_at_release=box_at_release,
                place_shift=place_shift,
                # How far the box ended up from where the weld says it should
                # be. Zero means the attachment is exactly rigid; it is the
                # check that caught the eq_data anchor-frame bug in grasp.py.
                weld_violation_mm=float("nan") if rel0 is None else float(
                    np.linalg.norm(rel0 + drift - m.eq_data[weld.eq_id][3:6])) * 1000.0,
                trace=trace,
                drift_by_phase={k: v * 1000.0 for k, v in drift_by_phase.items()},
                lock_standoff=lock_standoff, lock_lateral=lock_lateral,
                walk_trace=walk_trace,
                lock_yaw=float("nan") if not walk_trace else walk_trace[-1][3],
                grasp_meas=grasp_meas, support=support,
                palm_goal_err_mm=palm_goal_err * 1000.0,
                waist_yaw_cmd=demo.waist_yaw, waist_yaw_achieved=waist_ach,
                released_pitch_deg=released_pitch, released_base_z=released_z,
                released_fell=released_fell,
                start_standoff=float(box0[0] - base_x),
                heldout=start.heldout, box_start=box0, box_end=box_f,
                place_target=place_target,
                # Episode length is a DISTRIBUTION now, not a constant: two
                # phases end on the lock event. `episode_s` is what this episode
                # actually ran; `episode_s_max` is the schedule's upper bound.
                episode_s=n_steps * dt, episode_s_max=total * dt,
                n_steps=n_steps,
                # ceil, not floor: a recorder sampling every
                # PHYSICS_STEPS_PER_TICK steps takes one at step 0, so a
                # 14135-step episode yields 707 samples, not 706. The event
                # boundaries land one step after a sample tick, so `n_steps` is
                # no longer a multiple of the decimation.
                n_samples=-(-n_steps // spec.PHYSICS_STEPS_PER_TICK),
                phase_steps=dict(phase_steps),
                # Lock predicate, closed loop: which 25 Hz ticks fired, and
                # whether either event phase fell back to its max duration.
                lock_ticks=list(lock_ticks), release_ticks=list(release_ticks),
                lock_fallbacks=list(fallbacks),
                predicate_driven=pred is not None,
                # Transport clearance and the goal standoff (2026-09-10).
                min_clearance_m=float(min_clear),
                clear_at_lock2_m=float(clear_at_lock2),
                scraped=scraped, goal_standoff=float(goal_standoff),
                goal_standoff_ok=goal_standoff_ok,
                # O26 (2026-09-11). `hand_plat_depth_mm` is NEGATIVE when the
                # hand interpenetrates the pickup slab and 0.0 when it never
                # touched; a positive clearance is not claimed because it
                # cannot be measured reliably here (TR19).
                hand_plat_contacts=int(hand_hits),
                hand_plat_depth_mm=float(hand_depth) * 1000.0,
                hand_plat_force_N=float(hand_force),
                wrist_dev_rad=float(wrist_dev), wrist_dev_joint=wrist_dev_joint,
                wrist_dev_phase=wrist_dev_phase, wedged=wedged,
                wrist_dev_reach_rad=float(wrist_dev_reach),
                wrist_dev_any_rad=float(wrist_dev_any))
