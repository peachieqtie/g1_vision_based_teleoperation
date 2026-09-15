# CLAUDE.md

## 1. Thesis, objectives, constraints, authority model

**Title:** A State-Based Imitation Learning Pipeline for Bimanual Box Pickup on the Unitree G1
Humanoid Robot Using ACT-LSTM in MuJoCo Simulation. BSCS, Caraga State University, April 2026.

**Objectives (FIXED — never negotiate these away):** (1) teleop demonstration-collection system:
ZED 2i captures human bimanual motion, retargeted to G1 joint commands in MuJoCo; (2) train a
hybrid ACT-LSTM policy plus a standard BC baseline for bimanual box pickup/placement; (3) evaluate
both on box positions seen during collection (phase-level success rate); (4) and on positions NOT
seen during collection (spatial generalization).

**Constraints:** simulation-only, no hardware. Policy consumes privileged MuJoCo state, not camera
input. Fixed box size/orientation. Fingers unused. RL and locomotion-learning-from-demonstration
out of scope. **Generalization is measured over MANIPULATION targets (box position), never
NAVIGATION targets (platform position)** — Q6, TR13. Grasp is a weld (D11) and the base is held
during manipulation (D12), so grasp robustness and unaided whole-body reaching are out of scope.

**Authority model — read before changing anything:** `docs/Thesis_Proposal.pdf` is authoritative
for OBJECTIVES, scope and evaluation intent; **this file is authoritative for METHODS.** The
proposal's methods are a direction, not a spec (§7). Never "fix" code toward the proposal —
record the divergence and ask Charles. A drift from an OBJECTIVE, not a method, is what to flag.

## 2. CURRENT BLOCKER

**None. Phase 1's exit gate is MET.** The demonstrator performs the proposal's actual task —
walk to the pickup, grasp, **carry the box 1.5 m to the goal platform**, lower and release —
and passes **12/12 at both 14 s and 20 s settle**: weld gate 12/12, palm error 32–35 mm against
the 45 mm guard, box drift during transport 2.5–2.8 mm, placement error 0.029–0.049 m against
the 0.10 m Q4 limit (0/12 over), resting and upright every time, **zero falls**.

**Scope correction (2026-09-09):** the 0.20 m lateral arm sweep is **retired**. It was an
artifact of a stationary variant that could not walk, and it drove the trailing arm into the
torso (O24) — whose clean envelope is only ±0.06 m, *smaller* than the 0.10 m tolerance, so no
lateral sweep could both clear the collision and keep the criterion discriminating. Walking
transport removes the motion rather than engineering around it. `LOCKED_PHASES_WALKING` drops
MOVE from the locked set, exactly as D12 anticipated.

**Disclose:** placement carries a systematic **+28 to +31 mm bias in x** (±0.1 mm at 20 s), the
residual IK/servo offset that the collision fix does not remove — every demonstration places the
box ~30 mm past the nominal target, and a policy will learn that. Well inside the tolerance, but
success is scored against a point the demonstrator never exactly hits.

**Next: the recorder.** Note **O10 must be reset** — the episode is now 50.1 s = **1253 samples
at 25 Hz**, against a 500-step cap and a "~750 defensible" figure that are both far short.

## 3. Full pipeline (end to end)

```
[EXISTS] ZED (30 Hz, BODY_38) -> One-Euro -> scaling -> DLS IK -> arms; waist pinned
[EXISTS] KeyboardCommand -> rl_gym LSTM policy -> leg torques; physics @500 Hz
[EXISTS] WELD grasp (D11) + base lock (D12) + seeded spawn + reset, on the stepped model
[EXISTS] scripted demonstrator (g1_data/scripted_demo.py), exit gate 10/10
[MISSING] recorder @25 Hz -> success detection -> dataset -> BC | ACT | ACT-LSTM
[MISSING] autonomous deployment -> Exp 1 in-distribution, Exp 2 held-out patch
```

## 4. Current state (verified against files)

- **Kinematic path** — `teleop.py` / `run_teleop*.py` never `mj_step`; preview tools only. **Actuated path** — `run_integrated_combined.py` steps physics on a *separate* model and copies 17 twin qpos into the upper-body ctrl block; the only full-stack loop. `walk_test.py` is the same path, arms pinned, no ZED.
- **IK** — DLS with nullspace seed-pull, **4 joints per arm**, wrists pinned at `WRIST_NATURAL`,
  6-D elbow+wrist task (D2, D3). Retargeting: geometric scaling, One-Euro, per-arm stillness lock,
  5-frame dropout coast.
- **Locomotion** — `motion.pt` is an **LSTM** (`hidden_state`/`cell_state`, mutated in place
  every forward pass — must be reset per episode). 47-D obs, 12 actions, decimation 10 → 50 Hz,
  `GAIT_PERIOD=0.8`. Legs `<motor>` + PD; waist+arms `<position>` kp=500. `KeyboardCommand` (D6).
- **Scene** — tops at 0.75 m: pickup (1.5, 0) half-extent **0.19 × 0.32** (Q6), goal (1.5, −1.5)
  half 0.18; `box1` 0.18 m cube, 0.4 kg. **nq=45, nv=43, nu=31**, one keyframe `stand`.
- **Gripper** — **weld** (D11), `scene.xml` `<equality><weld box_grasp>` + `g1_teleop/grasp.py`.
  Geometric trigger gates the 0–1 command. Carry drift ≤0.26 mm since the TR17 anchor fix (the old
  ≤29 mm figure was that bug). Pads are detection-only and stay retracted. Friction pinch 0/60.
- **Box spawn** — `g1_teleop/box_reset.py`: seeded, on the **stepped** model. Sample x [1.44,
  1.56] × y [−0.21, 0.21] (x trimmed for O22); held-out patch x [1.47, 1.53] × y [0.04, 0.16],
  14.2% of seeds. Three guards, 2000 seeds. **Episode reset** — `g1_data/reset.py`: physics + locomotion carryover + policy LSTM +
  teleop filters. Audit in `NOTES.md`.
- **Scripted demonstrator** — `g1_data/scripted_demo.py`: 10-phase machine (APPROACH added), base
  lock by phase set (D12), staged reach clearing the platform slab, poses solved at the lock
  instant. **Exit gate 10/10** (§2). `fixed_base`/`box_xy`/`stop_after`/`waist_yaw` are sweep-only
  knobs (O18); the seeded-spawn path is untouched. `g1_teleop/base_lock.py` holds the pelvis.
- **Absent entirely:** demonstration logging, success detection, dataset, training, policy, evaluation code.

## 5. Planned components (none built)

Learning code goes at the **repo root**, siblings to `g1_teleop/`, so the recorder imports the
teleop stack unchanged. Tree in `NOTES.md`. Hard requirement: **ACT and ACT-LSTM are ONE model
class behind a flag** — two implementations would stop the gap isolating the LSTM.
`Phase` and the phase labels already exist in `g1_data/scripted_demo.py`.

## 6. State and action vectors

**State — 47-D (proposal Table 3.3).** box pos+quat 7, base pos+quat 7, L/R palm-site pos+quat
14, L/R gripper 2 (weld 0/1 + `pad_qpos`), arm joints 14, waist 3 (constant 0, D10). Every source
is resolved and listed per-group in `NOTES.md`; palm poses come from `*_palm_site` on the hand,
not the pad.

**Action — 22-D (proposal Table 3.4):** arms 7+7, waist 3, grippers 1+1, walking velocity 3.
Joint targets go to `ctrl[ModelIndex.upper_ctrl]`; **log `data.ctrl`, not twin qpos**. The 2
gripper dims are the **weld command** (D11), gated on `g1_teleop/grasp.py`'s geometric
preconditions, so the command alone does nothing. Velocity is body-frame, from `KeyboardCommand`
(D6). Only 8 of 14 arm dims vary and the 3 waist dims are constant (D10) — disclose in the
thesis. Per-dim detail and clip limits in `NOTES.md`.

## 7. Deviations from the proposal

| # | Proposal says | Code does | Why | Chapter to update |
|---|---|---|---|---|
| D1 | §3.3.3 pure rotation matrix camera→robot | rotation + `DEPTH_SCALE=0.6` on the depth axis | depth is the noisiest ZED axis and also the reach axis | 3.3.3 |
| D2 | §3.3.5 IK over all arm joints | IK drives 4 joints/arm; wrists pinned | wrist-twist artifact made grasp pose unusable | 3.3.5 |
| D3 | §3.3.5 IK targets end-effector position | IK targets elbow AND wrist (6-D task) | elbow-free IK produced mirrored/folded poses | 3.3.5 |
| D4 | §3.3.1 single 25 Hz control rate | recording + policy at 25 Hz, locomotion inner loop at 50 Hz | the two rates are deliberately decoupled; 50 Hz is the pre-trained policy's native rate and is not free to change | 3.3.1, 3.3.5 |
| D5 | §3.3.1 robot starts with all joints at zero | legs start at `DEFAULT_ANGLES` crouch | the walking policy cannot recover from straight legs — verified | 3.3.1, 3.8.1 |
| D6 | §3.3.4 pelvis-velocity walking trigger | **abandoned.** `KeyboardCommand` is the collection interface | §3.3.4 and §3.3.8 "walking in space" are mutually incompatible (TR1). Pelvis method is written up as a negative result | 3.3.4, 3.3.8 |
| D7 | §3.3.6 grippers press inward via arm IK error | actuated palm pads press; arm holds a clean pose | hands have no collision geometry, and IK-error pressing corrupts the arm joint dims it is recorded into | 3.3.6 |
| D10 | §3.4 waist joint angles as live DOF | waist pinned at 0; 3 state + 3 action dims constant | torso-yaw sign never verified, and waist motion perturbs a locomotion policy trained without it; turning is done by `wz` instead | 3.4 |
| D11 | §3.3.6 friction grasp via pressing grippers | **weld** equality box↔left hand, gated on palm proximity/opposition/separation | a friction pinch holds 0/60 on a rig reproducing the real standing configuration, and holding couples to unrelated foot-floor contacts (8× slip at constant normal force). Reportable negative result | 3.3.6 + limitations |
| D8 | two policies (ACT-LSTM, BC) | three (BC, plain ACT, ACT-LSTM) | isolates the LSTM's contribution rather than confounding it with chunking | 3.6, 3.7, 3.8.4, and RQ2 |
| D9 | §3.9 Ubuntu 22.04 + ROS2 Jazzy | Windows 11, no ROS | ROS adds no value for a single-process sim pipeline | 3.9 |
| D12 | §3.3 robot stands freely throughout the pickup | **pelvis welded to the world** through REACH→RELEASE, released for locomotion and before scoring | O17: arms reach 0.28–0.40 m but a free-standing arm-extended robot settles at an attractor standoff of 0.47–0.59 m, and every lever is swept and closed (TR16). The lock carries ~44% of body weight — genuine external support, and a limitation on every result downstream | 3.3, 3.8, + limitations |

D6, D7 and **D12** touch **objectives**, not just methods. D6/D7 have decided resolutions; D12 is
decided but is the most consequential deviation in the project and belongs in the limitations,
not buried in the methods.

## 8. Decision log

- 2026-08-23 — Q1–Q5 settled: palm-pad gripper; `KeyboardCommand` over the pelvis trigger (TR1);
  25 Hz recording with a 50 Hz locomotion loop; `d_place = 0.10 m` plus a resting clause; waist
  pinned (D10). Also: learning code at the repo root, recorder logs `data.ctrl`, plain ACT added
  as a third condition (D8). Rationale in `NOTES.md`.
- 2026-09-08 (D11) — **Weld replaces the friction pinch.** Friction 0/60 on rig v5, weld 60/60.
  Trigger is geometric so a policy must bring the hands to the box. Pads stand down while welded
  (a pad on a welded box stores 123.6 N and launches it at release). `NOTES.md`.
- 2026-09-08 (Q6 CLOSED) — Randomize **box position only**; platforms fixed. Objective 4 held out
  as a **2-D interior patch** — both marginals stay in-distribution, so only the *combination* is
  unseen (numbers in §4). Evaluation also gains a **data-scaling curve** (25/50/100/150 demos) and
  a **per-phase failure taxonomy**, and the recorder **logs a phase label every timestep** even
  though no policy is conditioned on it — cheap now, expensive later. `NOTES.md`.
- 2026-09-08 — Box spawn moved to `g1_teleop/box_reset.py`, seeded, on the **stepped** model
  (O3/O4). The old code sampled the kinematic twin only, so the physics box never moved and the
  10/10 gate would have been one run repeated ten times. Same session: **the locomotion policy is
  an LSTM** (`hidden_state`/`cell_state`, mutated in place, undocumented); `reset_episode(policy=…)`
  zeroes it — verified load-bearing, and relevant to D8. `NOTES.md`.
- 2026-09-08 — Episode reset built (`g1_data/reset.py`); every state holder gained a `reset()`.
  Model indexing is now name-based (`g1_teleop/indices.py`), verified bit-identical: pad joints
  shift arm and box indices but not leg indices, so literals stay half-correct and the robot
  still walks with only the arms wrong. Pad-geometry decisions superseded by D11. `NOTES.md`.
- 2026-09-09 (SCOPE CORRECTION) — **Walking place adopted; the 0.20 m lateral sweep retired.**
  Not a fix: the motion that caused O24 no longer exists. The demonstrator now carries the box to
  the goal platform with the arms static, which is the task the proposal describes; the sweep only
  ever existed because the stationary variant could not walk. Exit gate **12/12 at both settle
  durations**, placement 0.029–0.049 m vs the 0.10 m limit, zero falls. Retired path kept behind
  `walk_place=False`. Consequences: **O23 closed**, **O24 closed as not-applicable**, and **O10's
  episode cap must be reset** (50.1 s = 1253 samples at 25 Hz). `NOTES.md`.
- 2026-09-09 (O22 CLOSED) — **Sample region trimmed, not the standoff.** `pickup_half[0]`
  0.08 → 0.06 so the far edge leaves 15 mm under `max_base_x` at the nominal standoff; measured
  cliff is target base x 1.250 (converges, 14.6 mm) vs 1.260 (jams, 46 mm, +5.5°). Keeps the
  standoff band uniform across demonstrations. 25% of x span lost, held-out patch still inside
  (30 mm margin), walk-in 12/12 on every measure. `assert_reach_fits` could not have caught this
  (it checks `grasp_max`); a walk-in guard now sits where the standoff is known. `NOTES.md`.
- 2026-09-09 (O21 CLOSED, O20 CLOSED) — **Heading control added and the frame fix done properly.**
  `act[2]` (wz) is now driven toward the box bearing; heading −15.4° → +0.76° mean (a 2.4°
  peak-to-peak limit cycle, stationary from t=4 s), effective lateral 97 mm → 13.2 mm. `solve`
  gained a separation **direction** so offset and straddle axis rotate together — the pair is what
  the earlier 3/10 attempt was missing. Walk-in grasp 11/12 full task, gate 10/12, guard 10/12;
  stationary gate still 10/10. **O18 closes as not-required** (the robot repositions per spawn, so
  only head-on grasps are needed, ≤15 mm lateral). **O19 does not close** — still 1/12. `NOTES.md`.
- 2026-09-09 (O21) — **Walking positioning measured: position excellent, heading uncontrolled.**
  Standoff ±4 mm inside the band, world lateral 4–12 mm, 12/12 settle in ≤3.1 s — but heading is
  a systematic −11.5° to −18.8° because the station-keeper commands only `act[0:2]` and never the
  `wz` channel the policy accepts. That converts to 97 mm of effective lateral offset and the
  grasp lands 3/12 inside the guard (correlation 0.969). Reframes O18/O19 from a coverage problem
  into one uncontrolled DOF. Nothing proposed or built. `NOTES.md`.
- 2026-09-09 (O19) — **Lateral-aware approach path attempted and NOT achieved.** Coverage stayed
  at 15%/13% against a 58% static ceiling. Diagnosis overturned the premise: no cell fails by
  self-collision along the path, and contact force does not separate passes from failures. The
  staging waypoint is unreachable by construction (O19). **D10 was NOT reversed** — waist yaw makes
  the final pose feasible but changes nothing about the path, so the stated justification for
  unpinning it does not hold. `waist_yaw` stays a measurement knob defaulting to 0. `NOTES.md`.
- 2026-09-09 (O18) — **Fixed-base coverage measured and rejected as sufficient.** One world-fixed
  base position grasps only 15% of the sample region (band |y| ≲ 0.05 m) and misses the held-out
  patch entirely; the limit is lateral self-collision, not reach, so no base position helps. Waist
  yaw clears the self-collision statically (24% → 58%) but not in stepped physics (13%). Objective
  4 therefore still needs walking transport or a new approach path. `NOTES.md`.
- 2026-09-09 (D12) — **Base support adopted.** Pelvis welded to the world through REACH→RELEASE,
  released for locomotion; lock/release are phase transitions (`LOCKED_PHASES`), not timers. The
  locomotion policy is **not queried while locked** and the legs are PD-held at `DEFAULT_ANGLES`:
  it is the balance controller and has nothing to balance, its march would scrub the feet, and a
  soft hold would reappear as the standoff drift the lock exists to remove. Exit gate 10/10. Also
  fixed a live `grasp.py` bug — §9 TR17. `NOTES.md`.
- 2026-09-09 (O17) — **The stationary grasp is unreachable by tuning.** The base settles at an
  attractor `S*(R)` that tracks arm extension but saturates 0.13 m short of what the IK can serve;
  the gate fires at no reach and no platform height 0.75–0.90. Recorded as TR16. `NOTES.md`.
- (earlier, from code) — `DEPTH_SCALE` 0.3 → 0.6 once One-Euro handled noise (TR7); IK gained a
  nullspace seed-pull to stop elbow flipping; tracking gating removed (TR6).

## 9. Tried and rejected — NEVER retry these

- **TR1. Pelvis-velocity locomotion trigger as specified in §3.3.4.** Marching in place oscillates the pelvis about a fixed point, so *net* velocity over any window longer than one step is ~zero. The pelvis is also often out of frame, and leaning to reach translates it forward indistinguishably from intent to walk. §3.3.4 and §3.3.8 cannot both stand. Diagnostics in `locomotion_input.py`. Abandoned 2026-08-23.
- **TR2. Freezing leg targets to stop the in-place march.** The locomotion policy *is* the balance controller — the gait continuously repositions the support polygon. Freezing removes balance entirely and the robot topples within seconds. (Exception: under D12 the pelvis is welded, so balance is supplied externally and the legs are deliberately held.)
- **TR3. Freezing the gait-phase input while still querying the policy.** Preserves balance but puts the network out of distribution: unnatural stance, unstable walk/stop transitions.
- **TR4. Slowing the gait cadence** (slower period, and an idle/moving blend). Both destabilised the gait and measured cadence did not follow the commanded clock. `GAIT_PERIOD=0.8` is fixed inside the network, not a tunable.
- **TR5. Starting legs at the teleop keyframe's straight-leg pose.** Out of distribution; the policy cannot recover. Must start at `DEFAULT_ANGLES`.
- **TR6. Confidence / yaw / segment-length gating on tracking frames.** Froze the robot far more often than it caught bad frames. Only the NaN guard survives.
- **TR7. `DEPTH_SCALE = 0.3`.** Crushed forward reach; arms drifted sideways and jittered.
- **TR8. Single-frame stillness threshold for the arm lock.** ZED noise keeps every frame above threshold, so the lock never re-engaged and the arm followed noise forever ("automove"). Re-lock needs sustained stillness across `still_frames` consecutive frames.
- **TR9. One OpenCV key event per render tick.** Auto-repeat (~30 Hz) outruns the render loop; the queue grows and input arrives seconds late. Drain to empty each tick.
- **TR14. Trusting a grip number without stating the whole-body configuration and checking the palms are on the box.** Four rigs in one session each gave confident wrong numbers, every one because something other than the grasp carried the load: the platform, a pinned base, PD-held legs, or a stale arm pose. Withdrawn and **not citable**: the **64 mm seating slip**, the **40 N contact force**, the "force flat across 20×, area flat across 4×" sweeps, and the "1 of 9". Each was caught by a result insensitive to something it should depend on — the tell that later caught TR16's leaning artifact. `NOTES.md`.
- **TR15. Re-solving arm IK against a WORLD-ANCHORED target while the base is free to pitch.** Creates positive feedback: the base pitches, the shoulder moves, holding the palm at a fixed world point demands a more extreme pose, which pitches it further. Measured at standoff 0.30: fixed pose 6.2° pitch and stands; re-solved 27.1°; re-solved with the base moved to the standoff 70.9° and **falls**. This is what produced the withdrawn "reaching topples the robot" claim — the POSE is fine (6–15°). The live teleop path does NOT have this loop (`compute_arm_targets` anchors at the *live* shoulder, so the target moves with the robot), but a scripted grasp or an autonomous policy aiming at a world-frame box pose does. `NOTES.md`.
- **TR16. Tuning the stationary grasp into reach** — grasp depth, lift height, commanded arm reach, box height (0.74–0.94), **platform height (0.75–0.90)**, waist pitch (±0.52), and feedforward pre-positioning. All swept, none sufficient; resolved by D12 instead. The recession is an **attractor**, not a displacement: the base settles at `S*(R)`, so pre-positioning is rejected by the same dynamics that caused it, and extension saturates at the ~0.46 m arm limit while `S*−R` is still 0.13 m. Tables in `NOTES.md`. **Two traps it exposed.** (a) `PoseBook.solve` returns the IK residual on the kinematic TWIN; the ACHIEVED palm distance is 227–264 mm, not the 11–43 mm the residual reports — always gate on `GraspWeld.conditions()` against stepped state. (b) A raised platform lets the robot prop its forearms on it, so `S*` reads *inside* its own measured collision limit: a number carried by the table, TR14 again.
- **TR17. Assuming MuJoCo's weld `eq_data` layout instead of measuring it.** `anchor` is the weld point in **body2's OWN frame**; `relpose` is body2's pose relative to body1. `GraspWeld.engage` wrote the anchor as the box origin expressed in the *hand's* frame — a point ~160 mm outside the box — and the solver yanked the box to satisfy it: a one-off **147 mm jump at engage**, after which the weld held rigidly at the wrong offset. This was live and silent; it is the residue behind D11's "drift ≤29 mm". Same mistake pinned the pelvis 789 mm into the floor. Fixed with `anchor = 0`; carry drift went 147.2 mm → **0.26 mm** and achieved-vs-`relpose` to **0.000 mm**. **Verify any constraint against its own stored `relpose`, not against a plausible-looking number.** Two further path bugs found the same way: a joint-space reach interpolation sweeps the hands *through* the 40 mm platform slab (wrists wedge at 100 N, shoulder servos saturate at ±25 N·m demanding 222, palms end 183 mm low), and driving the pad servos while welded buries them in the box so `release()` ejects it. `NOTES.md`.
- **TR18. Trusting a fixed base position without checking the standoff actually varied.** The demonstrator's base station-keeping tracks `box_xy − [standoff, 0]`, so it walked the robot to the nominal standoff during SETTLE and silently undid `fixed_base`: every base position reported the same **0.33 m** standoff across a grid whose standoff must span 0.16 m. Caught only because a quantity that had to vary did not. Any fixed-base sweep must assert the standoff span. Related: the twin IK **never checks collision**, so its reachability map (73–75%) overstates feasibility by 3× — self-collision, not reach, is what actually limits lateral coverage (O18). `NOTES.md`.
- **TR12. Full-scene randomization** (platform positions, box position, robot start, together). Rejected without implementing: it inflates the sampling space to 4–6 dimensions, and 100–150 demonstrations cannot cover that densely enough for three policies to separate. A comparison that cannot discriminate is worse than a narrower one that can.
- **TR13. Randomizing the pickup-platform position** (a weaker TR12). Three reasons: it converts Objective 4 from *manipulation* generalization into *navigation* generalization, which is not what the thesis claims to measure; it puts the unreliable locomotion path on the critical path of every episode, so locomotion failures score as manipulation failures; and it still inflates the sampling space past the demonstration budget. Box position is the only randomized quantity.
- **TR11. Pinning the floating base by overwriting `qpos[base]` after `mj_step`.** With limp legs the robot sags each step and is teleported back up while the box is not, ratcheting it down at ~20 mm/s that looks exactly like grasp slip. Invariant to press force, mu, pad area, `solimp`, `solref`, `impratio`, cone and `noslip` — that insensitivity is the tell. Use an equality weld (D12) instead; if pinning, `body_gravcomp=1` on robot bodies AND base pinning. `NOTES.md`.
- **TR10. `zed.grab()` inside the physics loop.** Blocks 35–50 ms, so the sim ran at ~40% of real time. Grab on a background thread; the main loop reads the newest frame.

## 10. Known open issues

- **O14/O15/O16.** RESOLVED or WITHDRAWN. **Two standoff numbers, both correct:** A1 measured both criteria passing across **0.28–0.40 m**; `GraspConfig` ships the conservative **[0.28, 0.36]** and `assert_standoff` enforces it. Weld is wired into `run_integrated_combined.py` (`g` toggles); the "reaching topples the robot" fall was an artifact (TR15).
- **O17.** ~~Standoff unreachable~~ — RESOLVED by **D12** (base support), not by tuning. The underlying fact is permanent and shapes the thesis: a free-standing robot's equilibrium standoff never coincides with the standoff its arms can serve, at any commanded reach, box height, platform height or waist pitch. **Do not retry any tuning** — §9 TR16.
- **O18.** ~~No spatial variation~~ — CLOSES as **not-required**, superseded rather than solved: with walking transport the robot repositions per spawn, so the demonstrator only performs the head-on grasp it already does at 10/10 and need absorb ≤15 mm of lateral offset, not the ±210 mm span. Retained below because the fixed-base numbers stay valid if walking is ever dropped. **O18 (historical).** The stationary gate places the base at a fixed standoff *relative to the sampled box*, cancelling the spawn randomisation. The cheap alternative — pin the base in WORLD coordinates and let the box vary — was measured 2026-09-09 and **is not sufficient**: best coverage **15% of the sample region** (8/55 cells, base x 1.14), a narrow band at **|y| ≲ 0.05 m**, and the Q6 held-out patch is **0/3 graspable** at every base position tried. The limit is **lateral self-collision**, not reach: a fixed 0.18 m palm separation makes the far hand cross the centreline and the arm folds into the torso (152–212 N, `torso_link` vs `right_shoulder_yaw_link`) — 28 of 55 cells. Base x only trades which x rows are in range; `base_y = 0` is optimal by symmetry. Static collision-aware feasibility is 24% (or 58% with waist yaw, which does **not** survive stepped physics — still 13%), so headroom exists but needs a lateral-aware approach path. `NOTES.md`.
- **O19. OPEN — the reach stage commands an unreachable pose.** `P["up"]` (palms `stage_x` in front of the pelvis, above the box, at the wide approach separation) has an IK residual of **117–218 mm** against the 45 mm guard, at every `stage_x` 0.06–0.26 and separation 0.24–0.48. The arm cannot place the hands near the body: 4 joints/arm, wrists pinned (D2), 6-D elbow+wrist task (D3), and the home palms sit at (−0.004, ±0.239, −0.173) from the pelvis. Only waypoints out at box height are reachable (`side` 16–20 mm, `grasp` 14–21 mm). The stationary gate passes anyway because the resulting sweep is benign at y ≈ 0. **Fixing the approach path requires changing the IK formulation, not the waypoints.** `NOTES.md`.
- **O20/O21.** RESOLVED. Heading is now closed-loop on `act[2]` (−15.4° → +0.76° mean, a 2.4° peak-to-peak limit cycle), and `PoseBook.solve` takes a separation **direction** so offset and straddle axis rotate together. **Both are needed: rotating the offset alone scored 10/10 → 3/10** by putting the palms on the box's corners. Effective lateral 97 → 13.2 mm. `NOTES.md`.
- **O22.** RESOLVED — region trimmed to x [1.44, 1.56] (`pickup_half[0]` 0.06); walk-in 12/12. Historical detail: **O22 (was)** Box x ≳ 1.57 puts the target base x at 1.251–1.253 against `max_base_x` **1.255**, so the robot presses into the pelvis/platform limit, never converges (33 mm position error, +5.3° heading bias, both stationary) and the grasp fails. Standing further back frees it — standoff 0.36 gives heading −0.17° and a clean grasp. This is Q6 geometry biting in the walking variant; decide whether to widen the standoff for far spawns or trim the x sample range. `NOTES.md`.
- **O23/O24.** CLOSED by the walking place, not by tuning. O23: placement no longer sits on the tolerance — 0.029–0.049 m achieved against 0.10 m, 12/12 at both settle durations, criterion no longer flips. O24: the trailing arm never adducts across the body, so the torso collision does not arise. **Residual disclosed:** a systematic **+28 to +31 mm x bias** in where the box lands (±0.1 mm at 20 s) — the IK/servo term that survives, since it was present with zero contact and no saturation. Historical, retained as the evidence no lateral sweep could work: **the clean lateral envelope was ±0.06 m, SMALLER than the 0.10 m tolerance.** Measured during MOVE+LOWER only (the earlier 208–278 N figure was a whole-episode max and may have been O19's approach): ±0.04 and ±0.06 are **clean** — zero torso contact, roll 15.6–18.4 N·m — while ±0.08 is not (torso 74–88 N, roll 26–27 N·m past the ±25 limit). Onset is sharp; no margin to trade. Achieved inside the clean envelope is 0.037–0.060 m, so **placing inside it is NOT viable**: the commanded move would be smaller than the tolerance and "did not move the box" would pass — the exact failure `place_dy = 0.20` exists to prevent. The + direction still undershoots ~20 mm with zero contact and no saturation, so that part is IK residual, not collision. **Third option:** the lateral sweep exists only because the stationary variant could not walk; walking is now 12/12, so the place could be the proposal's real one — carry the box to the goal platform at (1.5, −1.5), arms static — sidestepping O24 instead of engineering around it. `NOTES.md`.
- **O24 (mechanism).** The arms press into the torso during large lateral places. `torso_link` against `*_shoulder_yaw_link` at **208–278 N at every place magnitude**, including 0.10 m. The trailing arm's shoulder roll cannot track (cmd ±0.385 → achieved ±0.12, err 0.26–0.27 rad) and its servo saturates at **±131–135 N·m against a ±25 N·m limit**; neither joint is near its range limit. The commanded pose asks the arm to pass through the torso, because the twin IK never checks collision — the same root cause as O19, in the place phase. Consequence: the blocked direction's achieved displacement **asymptotes at ~0.105 m** (cmd 0.10/0.14/0.18/0.20 → 0.084/0.096/0.102/0.105) while the free direction tracks with a ~30 mm offset. Fixes: waist yaw during the place (reverses D10; O21 shows the policy tolerates torso rotation) or a collision-aware place solve. `NOTES.md`.
- **O23. OPEN — placement undershoot; route-1 correction infeasible, see O24.** Diagnosed, not fixed. Post-release box movement is **0.0 mm** in all 24 runs and carry drift ≤0.26 mm, so it is neither drift nor release disturbance: the arm places short. Commanded 0.20 m lateral → achieved **0.158 m (+y) / 0.107 m (−y)**. Commanded poses are symmetric (place-pose IK residuals 34/40/25 mm both directions; palm lateral sweep symmetric to the mm), so this is execution, not target: the box is welded to `left_wrist_yaw_link`, so carrying it toward −y adducts the left arm across the body, and `left_shoulder_roll_joint` is asymmetric ([−1.588, +2.252] rad). No joint saturates. Settle duration does not change placement quality — it shifts the base pose (standoff 0.3268→0.3287, heading −0.01°→+2.27°) and moves the −y undershoot 93→106 mm, straddling the 0.10 m limit. **Do not loosen the tolerance**; either correct the commanded offset for the undershoot or set `place_dy` from the achieved displacement. `NOTES.md`.
- **O6.** Dead code: `gating.py`, `GatingConfig`, `TorsoYawConfig`, `IKConfig.neutral_weight`/`.target_deadzone`, `set_waist_yaw`. `RejectReason` survives only for `NAN`. **O11.** Sim-vs-wall-clock pacing only in the integrated entry point.
- **O7.** Stale docs (2026-09-08): README claims torso-yaw following and active gating; `run_integrated_combined.py` cites a nonexistent `run_integrated_vision.py`; `config.py` says locomotion is "not yet built"; `test/*.py` is stale.
- **O25. OPEN — Objective 1 is validated KINEMATICALLY ONLY, and must be validated under physics before collection.** Neither teleop entry point has ever run under stepped physics: `mj_step` count is 0 in `run_teleop.py` and `run_teleop_combined.py`, there is no gravity, no locomotion policy, and the base is never integrated (§4, verified 2026-09-08). No recorded session data exists — searched for `*.npz`, `*.hdf5`, `*.csv`, `*.json`, `demonstrations/`, `data/` and found none. So the retargeting → IK → arm-command path is proven to produce *poses*, never to produce *motion a standing robot can execute*. **O19 applies directly:** the 4-DOF pinned-wrist IK cannot place the palms anywhere near the body (residual 117–218 mm at every `stage_x` and separation tried), so a demonstrator whose hands pass near their own torso — which human bimanual motion does constantly, e.g. bringing the box in toward the chest — maps to unreachable poses. Under `mj_forward` that degrades silently into a bad pose; under physics it is what saturated the servos and jammed the arm in O19/O24. **This must be validated under stepped physics ahead of the Phase 4 pilot, before any collection begins.** Blocked on hardware: needs a ZED (§13). `NOTES.md`.
- **O10. OPEN and now urgent.** Rate decided (D4); the **episode cap is far too small**. The walking-place episode is **50.1 s = 1253 samples at 25 Hz**, against a 500-step cap and a "~750 defensible" figure. Reset it from this before the recorder is built — it sets dataset size, and cost per demonstration is ~2.5× what the stationary variant implied.
- **O12.** Q7 (walking during demos?) still open — forced by O18. PLAN.md.

## 11. File and module structure

Full listing in `NOTES.md`. Only the entries that carry a decision live here:

- `locomotion_input.py` — **do not delete**: `PelvisVelocity` is the evidence for the §3.3.4
  negative result (D6). `g1_teleop/gating.py` — DEAD, kept for `RejectReason` (O6).
- `g1_teleop/grasp.py` — **its thresholds are the grasp contract**; `g1_teleop/base_lock.py` (D12)
  — its docstring holds the measured `eq_data` layout (TR17).
- `g1_data/reset.py` — the one place an episode starts. **Read its docstring before adding any
  per-episode state**; a forgotten field leaks silently into the next episode.
- `../docs/Thesis_Proposal.pdf` — Ch.3 at PDF pages 33–56. `../NOTES.md` — overflow for this
  file. `../test/*.py` — stale scratch (O7).

## 12. Conventions

- **Frames:** ZED X right, Y down, Z forward; G1 X forward, Y left, Z up. Mapping in `transforms.apply_camera_rotation`. **BODY_38:** pelvis 0, shoulder 12/13, elbow 14/15, wrist 16/17.
- **Rates:** physics 500 Hz; locomotion 50 Hz (decimation 10); recording + policy 25 Hz (D4).
- **MuJoCo indexing:** **never hardcode a qpos/qvel/ctrl offset.** `ModelIndex.resolve(model)` resolves every block by name and asserts counts, coverage and ordering at load; the joint-name lists in `config.py` are the source of truth.
- **MuJoCo constraints:** **never assume an `eq_data` layout — measure it** (TR17).
- **Config:** tunables are frozen dataclasses in `config.py` under `TeleopConfig`; entry-point scripts violate this with module-level constants.

## 13. Verified vs assumed

Everything in §4, §6 and §12 is verified by loading the model or reading the file; full list in
`NOTES.md`. **Still assumed:** that `run_integrated_combined.py` runs end to end (never executed —
needs a ZED); that the ZED is a 2i; that box friction/solref were tuned, not inherited.

## 14. Maintenance protocol — instructions to future sessions

**At the end of every working session, update this file:** (1) rewrite §2 to whatever actually
blocks now — a fresh session must start from §2 alone; (2) add a dated §8 entry for every decision
with its reason, a §9 entry for anything tried that did not work, and a §7 row for any new
proposal divergence with the chapter affected; (3) move completed items from §5 into §4, citing
the file that proves it; (4) update §10 — resolved issues are the *only* deletable thing here.

**Rules:**
- **Never delete a §9 entry** — the most expensive knowledge here to rediscover, even when it
  later looks obvious. Never solve overflow by cutting §9.
- **Keep this file under 300 lines.** Past that, move detail to `NOTES.md` and leave a pointer.
- **If code and this file disagree, the code is correct.** Fix the file, note the drift in §8.
- **If code and the proposal disagree, neither is automatically correct.** Do not silently
  reconcile — surface it to Charles and record the outcome in §7.
- **No routine implementation detail here** — only decisions, state, and what a fresh session
  could not work out from the code in five minutes.
