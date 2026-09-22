# CLAUDE.md

## 1. Thesis, objectives, constraints, authority model

**Title:** A State-Based Imitation Learning Pipeline for Bimanual Box Pickup on the Unitree G1 Humanoid Robot Using ACT-LSTM in MuJoCo Simulation. BSCS, Caraga State University, April 2026.

**Objectives (FIXED — never negotiate these away):** (1) teleop demonstration-collection system: ZED 2i captures human bimanual motion,
retargeted to G1 joint commands in MuJoCo; (2) train a hybrid ACT-LSTM policy plus a standard BC baseline for bimanual box pickup/placement;
(3) evaluate both on box positions seen during collection (phase-level success rate); (4) and on positions NOT seen during collection
(spatial generalization).

**Constraints:** simulation-only, no hardware. Policy consumes privileged MuJoCo state, not camera input. Fixed box size/orientation. Fingers
unused. RL and locomotion-learning-from-demonstration out of scope. **Generalization is measured over MANIPULATION targets (box position),
never NAVIGATION targets (platform position)** — Q6, TR13. Grasp is a weld (D11) and the base is held during manipulation (D12), so grasp
robustness and unaided whole-body reaching are out of scope.

**Authority model — read before changing anything:** `docs/Thesis_Proposal.pdf` is authoritative for OBJECTIVES, scope and evaluation intent;
**this file is authoritative for METHODS.** The proposal's methods are a direction, not a spec (§7). Never "fix" code toward the proposal —
record the divergence and ask Charles. A drift from an OBJECTIVE, not a method, is what to flag.

## 2. CURRENT BLOCKER

**None. Phase numbers are PLAN.md's** (P3 recorder, P4 pilot, P5 model code); commits and `NOTES.md` since 2026-09-21 call the P5 work
"Phase 4 stages". **P1–P3 CLOSED.** P3's open-loop replay reproduces the manipulation channel on 5 scripted episodes; the locomotion channel
cannot be replayed open loop by any recorder (`NOTES.md` 2026-09-16). **P5 in progress:** BC, chunked BC and state-only ACT all pass the
overfit-10 gate (§8 2026-09-22). **Next: ACT-LSTM, the `use_lstm` flag on the SAME class** (`g1_model/act.py`). **Every model result so far
is on SCRIPTED data — no piloted episode exists** (`data/raw/` is empty), so O31 and O32 stay open until P4.

**P4 pilot has not started.** It must settle D12: the grasp worked free-based on 2026-09-15, so D12 may bind only the scripted demonstrator
(§8). D18 is enforced at every reset (`g1_data/reset.py` raises on a contract mismatch; the recorder stores it per episode) — the
evaluation harness, not yet built, must start its episodes through the same reset.

## 3. Full pipeline (end to end)

```
[EXISTS] ZED (30 Hz, BODY_38) -> One-Euro -> scaling -> DLS IK -> arms; waist pinned | KeyboardCommand -> LSTM policy -> legs; 500 Hz physics
[EXISTS] WELD grasp (D11) + base lock (D12) + B-prime exclusion (D18) + seeded spawn; teleop under stepped physics, full task by a live operator
[EXISTS] scripted demonstrator (g1_data/scripted_demo.py) 12/12 and 40/40; g1_data/spec.py frozen at g1-spec-1.1.0
[EXISTS] recorder @25 Hz -> success detection -> offline phase labels -> dataset -> loader -> shared train loop + masked-L1 loss -> neighbour-ambiguity gate -> BC and chunked BC (ONE class, K=1 vs K>1): g1_data/{recorder,success,phase_label,dataset}.py + g1_model/{loader,train,ambiguity,models}.py
[EXISTS] ACT, state-only, one decoder layer; overfit-10 PASS 0.920 on scripted data: g1_model/act.py
[MISSING] ACT-LSTM (the `use_lstm` flag on the SAME class) -> autonomous deployment -> Exp 1 in-distribution, Exp 2 held-out patch
```

## 4. Current state — the measurements a fresh session cannot re-derive

**Re-derive status from artifacts; never inherit it from a checklist** (§14). Anything readable off the code, a passing test or a gate output
is deliberately NOT repeated here — `NOTES.md`, the suites and the run directories carry it. Only what cost an instrumented run survives:

- **The arm IK NEVER converges**: mean **30.00** iterations, `max_iter` hit on **100%** of solves, and 26 of those 30 buy 0.1 mm — 4 joints/arm
  cannot satisfy the 6-D elbow+wrist task (D2, D3; the achieved error is retargeting loss, not iteration count — O27). **`IKConfig` is SHARED
  with the scripted demonstrator, so lowering iterations globally breaks the byte-identical gates** — teleop needs a separate config.
- **`motion.pt` is an LSTM** whose `hidden_state`/`cell_state` are mutated IN PLACE on every forward pass and MUST be reset per episode
  (`g1_data/reset.py`) — the one locomotion fact not readable from the code itself.
- **The held-out patch is 14.25% of the spawn region**, measured over 2000 draws (`g1_teleop/box_reset.py`) — the Objective 4 split.

## 5. Not yet built

**ACT-LSTM — the `use_lstm` flag on the SAME class as ACT** (`g1_model/act.py`; two implementations would stop the gap isolating the LSTM), plus autonomous deployment and the Exp 1 / Exp 2 evaluation harness. Everything else in the learning stack is built (§3, `g1_model/`).
Dataset and evaluation design — 150 episodes, splits, fixed eval seeds, nested scaling subsets, the statistical limit — is in PLAN.md and `NOTES.md` "PHASE 2 CLOSEOUT".

## 6. State and action vectors

**Defined once, in `g1_data/spec.py`.** Summary only; that module is authoritative.

**State — 47-D (Table 3.3):** box pos+quat 7, base pos+quat 7, L/R palm-site pos+quat 14, L/R gripper 2 (both = the weld bit, D14), arm joints
14, waist 3 (constant 0, D10). **No state dim is constant.** Palm poses come from `*_palm_site`, not the pad.

**Action — 22-D (Table 3.4):** arms 7+7, waist 3, grippers 1+1, walking velocity 3. **Log `data.ctrl`, not twin qpos.** 6 dims are excluded
from loss and normalization — 6, 13 (wrist yaw), 14, 15, 16 (waist), 18 (`a_gR`, correlation exactly 1.0 with 17) — leaving **16 trainable**.
Velocity dims 19–21 are logged **ZERO while the base lock is engaged** (D17). Clips ±0.80 / ±0.80 / ±0.60 (D15). Timing: `action[t]` is the
command applied FROM `state[t]`; build the pair at ONE point in the loop, before `mj_step`.

**Phase labels** are logged every timestep and **no policy is conditioned on them**; they can be derived offline from the trajectory, so teleop needs no live phase classifier.

## 7. Deviations from the proposal

| # | Proposal says | Code does | Why | Chapter |
|---|---|---|---|---|
| D1 | §3.3.3 pure rotation camera→robot | rotation + `DEPTH_SCALE=0.6` on depth | depth is the noisiest ZED axis and also the reach axis | 3.3.3 |
| D2 | §3.3.5 IK over all arm joints | 4 joints/arm; wrists pinned | **The recorded reason was wrong.** Measured 2026-09-14: unpinning does NOT reintroduce twist at the grasp (palm angles match baseline, palm error improves) — it fails at RELEASE, pitching to −0.69/−0.82 rad and tipping the box. **D2 protects the PLACE, not the grasp pose.** It also makes a wedge UNRECOVERABLE: the wrist command never changes after REACH, so a jammed arm cannot back out | 3.3.5 |
| D3 | §3.3.5 IK targets end-effector | targets elbow AND wrist (6-D) | elbow-free IK produced mirrored/folded poses | 3.3.5 |
| D4 | §3.3.1 single 25 Hz rate | record+policy 25 Hz, locomotion 50 Hz | 50 Hz is the pre-trained policy's native rate, not free to change | 3.3.1, 3.3.5 |
| D5 | §3.3.1 joints start at zero | legs start at `DEFAULT_ANGLES` | the walking policy cannot recover from straight legs | 3.3.1, 3.8.1 |
| D6 | §3.3.4 pelvis-velocity trigger | **abandoned**; `KeyboardCommand` | §3.3.4 and §3.3.8 are incompatible (TR1); written up as a negative result | 3.3.4, 3.3.8 |
| D7 | §3.3.6 grippers press via IK error | pads press; arm holds a clean pose | IK-error pressing corrupts the arm dims it is recorded into | 3.3.6 |
| D8 | two policies | three (BC, ACT, ACT-LSTM) | isolates the LSTM instead of confounding it with chunking | 3.6–3.8.4, RQ2 |
| D9 | §3.9 Ubuntu + ROS2 | Windows 11, no ROS | ROS adds no value for a single-process sim | 3.9 |
| D10 | §3.4 waist as live DOF | waist pinned at 0 | torso-yaw sign unverified; waist motion perturbs the locomotion policy | 3.4 |
| D11 | §3.3.6 friction grasp | **weld**, gated on palm proximity/opposition/separation | friction holds 1 of 9 standoffs and couples to unrelated foot contacts; reportable negative result | 3.3.6 + limits |
| D12 | §3.3 robot stands freely | **pelvis welded to the world** through REACH→RELEASE | O17: the free attractor standoff never meets the servable band, every lever swept (TR16); carries ~44% of body weight. **But a live operator grasped free-based (§8, 2026-09-15) — D12 may bind only the scripted demonstrator** | 3.3, 3.8 + limits |
| D13 | §3.8.2 cap 500 timesteps | cap from the measured 694–846 | the schedule-derived number was never the episode length | 3.8 |
| D14 | Table 3.3 gripper state from "MuJoCo actuator state" | both dims carry the **weld bit** | pads are detection-only (D11) and retracted, so that source is constant 0. Measured: pad_qpos point-biserial with weld −0.072/+0.038, std 9e-4 → 4e-5 when welded — no grasp signal, and z-scoring would amplify pure noise to unit variance | 3.4 |
| D15 | vy ±0.40 (this file, previously) | spec clips ±0.80/±0.80/±0.60 | the demonstrator uses its own `hold_max` and exceeds ±0.40 on 16.5% of ticks; code is authoritative (§14) | 3.4 |
| D16 | Exp 1 reads as replaying collection spawns | fresh seeds from the same region | reproducing 150 memorised spawns is not in-distribution execution | 3.8 |
| D17 | velocity dims always logged | dims 19–21 **ZERO while locked** | the station-keeping block does not run while locked: `act` is not consumed (`carry.obs[6:9]` sits inside the skipped block) and not applied (legs PD-held); locked vs PD-held agreed on 15336/15336 ticks. Logging it writes a phantom — vy = −0.732 held constant through REACH→LIFT, 36.4% of every episode. Lock state read from the model's own equality | 3.4 |
| D18 | full contact everywhere | **hand ↔ `platform_pickup` contact excluded** (B-prime): `wrist_pitch_link`, `wrist_yaw_link`, pads, both sides; hand↔hand, hand↔goal-platform, hand↔box all survive | a SIMULATION-ONLY RELAXATION that must apply to collection AND evaluation identically or policies are tested in physics they never learned. Without it the teleoperated grasp does not happen at all: the wrist hooks the slab, jams at 1.42 rad, the weld never engages | 3.3 + limits |

D6, D7, **D12** and **D18** touch **objectives**, not just methods. D6/D7 are resolved; D12 and D18 belong in the limitations.

## 8. Decision log

Older entries are one line; detail is in `NOTES.md` under the same date. Retired entries (§14) keep a one-line pointer.

- 2026-09-22 — **ACT PASSES the overfit-10 gate: 0.041081 / 0.044657 = 0.920** (converged, 195k steps, lr 1e-5, beta 10, one decoder
  layer). It had "failed" at 1.140 because the gate scored a TRAIN-MODE window mean, with dropout 0.1 on and the weights moving; ALL of the
  gap is dropout (z = posterior vs 0 moves < 2e-4). **The gate AND best.pt now use ONE quantity, `train.score_deployment`**: final weights,
  eval mode enforced by hooks, `model(obs)` only, so the CVAE encoder is unreachable by construction. Every gate.json rewritten; no other
  verdict changed (TR30). `NOTES.md` 2026-09-22.
- 2026-09-22 — **ACT's latent COLLAPSED (KL 1e-05; z = 0 equals the posterior mean) — EXPECTED on scripted data**: a deterministic
  demonstrator has no style variation for z to encode, so on scripted data ACT vs chunked BC cannot isolate the CVAE (RQ2). A PREDICTION to
  test on piloted data (O32), NOT a defect: do not tune beta for it.
- 2026-09-21 — **THE GATE CRITERION IS THE NEIGHBOUR-AMBIGUITY REFERENCE**, computed per loader configuration, not "loss near
  zero" (`g1_model/ambiguity.py`). A stage passes when 10-episode training error falls below it; report both numbers and the ratio,
  always. BC 0.009512/0.012764 = 0.745 PASS; chunked BC 0.042570/0.044657 = 0.953 PASS (deployed function, 2026-09-22). Quote `AmbiguityResult.cite()`, never
  `.mean`. Revision record, and why the old criterion failed a working model: `NOTES.md` 2026-09-21.
- 2026-09-21 — **BC's residual is PARTIAL OBSERVABILITY, not capacity.** The 3 velocity dims are 18.8% of trainable dims and 48.5%
  of the loss, 11.5× worse while the base lock is DISENGAGED — and the lock state is deliberately not in the 47-D state (2026-09-11),
  so no policy can see what drives its largest error. Fixing it is a schema question and a `SPEC_VERSION` bump. `NOTES.md` 2026-09-21.
- 2026-09-21 — **Chunked BC is the SAME CLASS as BC** (`BCPolicy` at K>1); a separate class would hide incidental differences inside
  the measured effect of chunking. `K_PROVISIONAL = 100` (4.0 s), ACT's published value, NOT swept — settle on piloted data. **No
  temporal ensembling until ACT** (Stage 4); deployment takes `models.first_action`. `NOTES.md` 2026-09-21.
- 2026-09-21 — **The W_o instrument cannot choose W_o** (`ambiguity_curve`, `train_bc.py wo-curve`). Model-free by design, but on
  scripted data it RISES with W_o (0.009608 at W=1 → 0.011850 at W=32) because the space grows faster than the data fills it — a
  dimensionality confound, NOT evidence against longer windows. Real run is on piloted data. `NOTES.md` 2026-09-21.
- 2026-09-15 — Objective 1 demonstrated: live operator, full task, stepped physics. Archived: `NOTES.md` "2026-09-22 — CLAUDE.md §8 ARCHIVE".
- 2026-09-15 — **THE GRASP WORKED WITHOUT THE BASE LOCK.** O17's attractor was measured with the scripted station-keeper, whose corrective
  speed is capped; a human at full forward command pushes against the recession. The box welded, lifted, stayed held and the task completed
  free-based. **D12 may therefore bind only the SCRIPTED demonstrator.** Unmeasured: standoff spread at grasp without the lock, and whether a
  learned policy can fight the attractor as a human does. Worth resolving before collection — it removes a major limitation AND gives
  Objective 4 natural standoff variation.
- 2026-09-15 — IK throughput: `mj_kinematics` + `mj_comPos` in the solver, 37.4% → 128.2% of real time. Archived: `NOTES.md` "2026-09-22 — CLAUDE.md §8 ARCHIVE".
- 2026-09-15 — D18 adopted as `<contact><exclude>` pairs. Archived: `NOTES.md` "2026-09-22 — CLAUDE.md §8 ARCHIVE".
- 2026-09-11 — **Schema frozen** at `g1-spec-1.1.0`. Four decisions: (a) box position stored WORLD frame, base-relative derived in the loader;
  (b) **gait phase EXCLUDED** from the state, logged as per-timestep metadata — it is a clock, and giving every policy a clock hands BC the
  temporal capability ACT-LSTM is meant to supply, collapsing the RQ2/RQ3 gap; (c) the base lock is neither a state nor an action dim —
  episode metadata plus an observable predicate; (d) the O10 cap comes from the measured distribution.
- 2026-09-11 — **Phase vocabulary frozen in `spec.py`.** `Phase` (10) and `ScoredPhase` (5) are IntEnums with EXPLICIT integers;
  **`REPOSITION=8` and `APPROACH=9` are out of execution order and must NEVER be renumbered** — tidying them silently relabels every recorded
  episode. `SCORED_OF` is data, not a function body. `scripted_demo.py` imports `Phase` from `spec.py`.
- 2026-09-11 — **`ScoredPhase.WALK_IN`**, a fifth failure bucket; `SETTLE` maps there, not to `GRASP`, so walk-in failures are not charged to
  the grasp — and the spatial variation Objective 4 is scored on lives almost entirely in the walk-in. **THE FOUR PROPOSAL SUCCESS RATES ARE
  UNCHANGED**: they are episode-level outcome criteria; `SCORED_OF` only decides which bucket a failed episode is charged to. Named WALK_IN
  because `Phase.APPROACH` is the arm descending beside the box and maps to GRASP. **Chapter 4 caveat:** WALK_IN is 10–13% of a
  predicate-lock episode (27.4% under the phase lock) — the null baseline its failure share is read against.
- 2026-09-11 — **Dataset design.** 150 episodes; train/val 80/20 stratified by binned spawn; Exp 1 = 100 fresh in-region seeds, Exp 2 = 100
  held-out-patch seeds, **evaluation seeds FIXED across all three policies**; scaling curve on **NESTED** subsets 25 ⊂ 50 ⊂ 100 ⊂ 150; seed
  streams **pre-partitioned before collection** so leakage is provable, not audited; normalization from the training split only, masked dims
  pinned to mean 0 / std 1; ~50 MB. **Collect 25 FIRST and train BC on them** — if BC saturates at 25 the task does not discriminate and the
  comparison cannot answer RQ2/RQ3. **State up front: 100 evaluation episodes give SE ≈ 5% near a 50% rate, so policies are separable at
  roughly 15 points.** Episode-file contents: `NOTES.md` "PHASE 2 CLOSEOUT".
- 2026-09-11 — O26 mitigated by the staged raise. Archived: `NOTES.md` "2026-09-22 — CLAUDE.md §8 ARCHIVE".
- 2026-09-10 — Constant-dim mask measured; base-lock predicate enabled. Archived: `NOTES.md` "2026-09-22 — CLAUDE.md §8 ARCHIVE".
- 2026-09-09 — **Walking place adopted; the 0.20 m lateral sweep retired** (O23/O24 closed), 12/12 at both settle durations. **Disclose:
  placement carries a systematic +28 to +31 mm x bias.** O22 closed by trimming the sample region (`pickup_half[0]` 0.06), not the standoff.
- 2026-09-09 — O20/O21 closed: heading closed-loop on `act[2]` (−15.4° → +0.76°) and `PoseBook.solve` takes a separation DIRECTION so offset
  and straddle axis rotate together — rotating the offset alone scored 10/10 → 3/10. **D12 base support adopted**: policy not queried while
  locked, legs PD-held at `DEFAULT_ANGLES`; O17 recorded as TR16.
- 2026-09-08 — **D11 weld replaces the friction pinch** (friction 0/60, weld 60/60); the trigger is geometric, so a policy must bring the hands
  to the box. **Q6 closed:** randomize box position only; Objective 4 held out as a 2-D interior patch, so both marginals stay
  in-distribution and only the COMBINATION is unseen; adds the data-scaling curve and the per-phase taxonomy.
- 2026-08-23 — Q1–Q5 answered. Archived: `NOTES.md` "2026-09-22 — CLAUDE.md §8 ARCHIVE".

## 9. Tried and rejected — NEVER retry these

**Never delete an entry here.** Measurements are in `NOTES.md`.

- **TR1.** Pelvis-velocity locomotion trigger (§3.3.4). Net pelvis velocity while marching is ~0, the pelvis is often out of frame, and leaning
  to reach is indistinguishable from intent to walk. **TR2.** Freezing leg targets to stop the march — the policy IS the balance controller and
  it topples (under D12 balance comes from the weld, so the legs are held deliberately). **TR3.** Freezing the gait-phase input while still
  querying the policy: out of distribution. **TR4.** Slowing the gait cadence — `GAIT_PERIOD=0.8` is fixed inside the network.
- **TR5.** Starting legs at the straight-leg keyframe pose; the policy cannot recover. **TR6.** Confidence / yaw / segment-length gating on
  tracking frames — only the NaN guard survives. **TR7.** `DEPTH_SCALE = 0.3`: crushed forward reach, arms drifted sideways and jittered.
  **TR8.** Single-frame stillness threshold for the arm lock; needs sustained stillness. **TR9.** One OpenCV key event per render tick —
  auto-repeat outruns the loop, so drain the queue. **TR10.** `zed.grab()` inside the physics loop: blocks 35–50 ms; grab on a thread.
- **TR11.** Pinning the base by overwriting `qpos[base]` after `mj_step` — ratchets the box down ~20 mm/s, looks exactly like grasp slip, and is
  invariant to every friction lever (that insensitivity is the tell). Use an equality weld (D12).
- **TR12.** Full-scene randomization — inflates the sampling space past the demonstration budget. **TR13.** Randomizing the pickup-platform
  position — converts Objective 4 from manipulation into navigation generalization and puts the unreliable locomotion path on every episode.
- **TR14.** Trusting a grip number without stating the whole-body configuration and checking the palms are on the box. Four rigs, four confident
  wrong numbers, each because something else carried the load. The 64 mm seating slip, the 40 N contact force, the flat sweeps and the "1 of 9"
  are **withdrawn and not citable**.
- **TR15.** Re-solving arm IK against a WORLD-ANCHORED target while the base is free to pitch: positive feedback, 6.2° → 70.9°, a fall.
  **Mechanism corrected 2026-09-15:** the live path cannot form the loop because `compute_arm_targets` anchors at the TWIN's shoulder and the
  twin's base is never synced — not because "the target moves with the robot". So the target does NOT follow the robot: a receding base simply
  leaves the box further away.
- **TR16.** Tuning the stationary grasp into reach — grasp depth, lift height, commanded reach, box height, platform height, waist pitch,
  feedforward pre-positioning. The recession is an ATTRACTOR, not a displacement. **TR16a:** `PoseBook.solve` returns a TWIN residual — achieved
  palm distance was 227–264 mm where it said 11–43 mm. **Never quote a twin residual.**
- **TR17.** Assuming MuJoCo's weld `eq_data` layout instead of measuring it. `anchor` is in body2's OWN frame, `relpose` is body2 relative to
  body1; getting it wrong jumped the box 147 mm at engage and pinned the pelvis 789 mm into the floor. **Verify any constraint against its own
  stored `relpose`.** Also: driving the pads while welded buries them in the box and `release()` ejects it.
- **TR18.** Trusting a fixed base position without checking the standoff actually varied — every base position reported the same 0.33 m across a
  grid that had to span 0.16 m. **If a quantity that must vary comes out identical, treat it as a rig artifact until proven otherwise.**
- **TR19.** `mj_geomDistance` for clearance in this build: it returns exactly **+0.0** for box-vs-mesh pairs at real positive separation (every
  +0 cell had zero contacts and a wrist mesh as nearest geom; pad primitives were sane). Negative values are trustworthy; **+0.0 means NOT
  MEASURED**. Gate clearance work on `d.ncon` + `mj_contactForce`.
- **TR20.** Gating the base lock on pelvis height. The "sharp cliff at ~0.774 m with no overlap" came from 12 samples of two confounded
  variables (corr(base_z, lateral) = −0.93); a controlled sweep over 18 mm — 3.5× the claimed cliff — moved clearance 0.7 mm and never changed
  the outcome; at 40 seeds the distributions overlap. **No `base_z` term anywhere.**
- **TR21.** Reading `data.actuator_force` as delivered torque. It is the UNCLAMPED demand — the −692 N·m came from there. Delivered torque is
  `data.qfrc_actuator`, clamped by `model.jnt_actfrcrange` (where the XML `actuatorfrcrange` lands); `model.actuator_forcerange` is [0,0] =
  unlimited and is the wrong field. **Wrist pitch/yaw are ±5 N·m, not ±25 — the joints that jam are the weakest in the chain.**
- **TR22.** Standing further back for clearance. Standoff 0.36 cuts contact-steps 63% without reaching zero and breaks the grasp (palm error
  50.4 / 79.9 mm against a 45 mm guard that 0.32 was inside); the corridor map's 25 mm was a settled single-pose figure and the walk-in arrives
  0.125–0.200 m from the edge across seeds. TR14's pattern — keep 0.32.
- **TR23.** Gating all six wrist joints at 0.5 rad as a wedge detector — it fails EVERY healthy episode: post-REACH the left wrist yaw sits
  0.55–0.57 rad off command in the 12/12 configuration, because the welded box hangs off that link on a 5 N·m joint. Gate the **pitch pair only,
  post-REACH only, at 0.8 rad** (healthy 0.297–0.490, wedge 1.418).
- **TR24.** Routing a contact-free path through the close-in region: ~290 paths over 4 lock states, non-monotone in subdivision count
  (101/98/217/66). The corridor is REACHABLE but not CONTROLLABLE — steering through a 4 mm margin needs placement better than 4 mm and the IK
  gives 56–131 mm, with the error changing direction cell to cell. Extra waypoints re-roll the error rather than reduce it.
- **TR25.** A "lift has finished" settling test on RELEASE 1: fires +33 to +495 ticks late — the cosine ease is asymptotic and the transport walk
  bounces the carried box past any threshold.
- **TR26.** Unpinning the wrists (candidate A, 2026-09-14). No twist at the grasp, but it fails at RELEASE — pitching to −0.69/−0.82 rad and
  tipping the released box, 9/12 and 36/40 — and it still hooks the slab in 10/10 teleop runs, so it does **not** make D18 unnecessary. Adoption
  would also need a `SPEC_VERSION` bump (dims 6, 13 leave the mask).
- **TR27.** Gating a learning stage on "training loss near zero" without first measuring whether the observation DETERMINES the
  action. BC was called a failure while sitting BELOW the 1-NN action spread. Two related traps: the COPY baseline is handed
  `action[t-1]`, which a W_o=1 model does not have, so it is an oracle not a peer; and neighbour ambiguity itself rises with
  `obs_window` for dimensionality reasons (O31), so a rising region is the estimator, never a finding. `NOTES.md` 2026-09-21.
- **TR28.** Letting a model inherit a hyperparameter from ANOTHER model's config. ACT ran at BC's lr 1e-3 (the reference's is 1e-5) and
  its latent collapsed; the guard then caught `weight_decay` leaking too. Each model STATES its optimizer config (`ACTConfig.lr` has no
  default) and `train.assert_optimizer_source` raises on a mismatch. Same family: a training budget with no recorded reason. `NOTES.md` 2026-09-21.
- **TR29.** Gating on a number in the wrong UNITS. ACT's TOTAL loss (reconstruction + beta*KL, 0.289) was scored against a reconstruction
  reference: ratio 6.47 where the right answer was 3.09 — silent, and right by accident for BC (no KL term). `gate()` takes only a
  `Quantity` and refuses other units. `NOTES.md` 2026-09-21.
- **TR30.** Gating on a TRAIN-MODE number. The window mean of per-step losses carries dropout, a posterior z that read the target, and
  moving weights: right units, wrong measurement (TR29's family). Converged ACT read 1.140 FAIL, deployed 0.920 PASS. Score only through
  `train.score_deployment`; `gate()` refuses anything else. Eval-mode `loss_terms` was NOT enough: it still fed the encoder the target.

## 10. Known open issues

- **O26. MITIGATED, NOT CLOSED.** The wedge is gone; **the collision is not**. The hand enters the pickup platform 7.5–15.7 mm in every
  scripted episode, up to 70 mm for 5.6 s under teleop. The A/B is **mixed**: contact duration falls 72–87% but depth and peak force RISE
  (−8.5 → −10.9 mm, 75 → 187 N). Shorter and harder, not gentler.
- **O25. PARTIALLY CLOSED.** Teleop runs under stepped physics and a live operator completed the full task. Not covered: ZED noise beyond one
  session, human motion statistics, operator reaction, more than one demonstrator.
- **O19/O24. OPEN.** Achieved close-in placement is **56–131 mm on STEPPED state**; the old "117–218 mm" was a twin residual and is **not
  citable** (TR16a). Root cause: collision-blind IK plus D2.
- **O27. NEW — teleop fidelity.** p50 achieved palm error **56 mm**, from retargeting loss and unreachable close-in targets, not iteration
  count. The 45 mm guard is the SCRIPTED demonstrator's criterion and does not apply to teleop (the weld gate allows 160 mm), but **56 mm is
  the number to quote as teleoperation fidelity** in the limitations.
- **O28. NEW — dropouts are silent live.** A frame with no body means `controller.step` is never called, so the arms hold their last pose
  **indefinitely** and `max_coast_frames = 5` never applies on this path. The recorder's dropout analysis describes a code path that is not
  running live.
- **O29. NEW — One-Euro is tuned for a rate that does not occur.** `euro_freq = 30` but frames arrive at ~20 Hz (TR10; 77 SDK drops in 8 s
  measured). Overstating the rate shrinks alpha, so the filter OVER-smooths and adds lag. Set it from the measured rate or feed per-frame dt.
  `ZEDConfig.camera_fps` is never applied — dead config, same class as O6.
- **O30. NEW, FIXED, kept as a caveat.** `ZEDSource._select_best_body` scored by `mean(arm confidences)` against −1.0, so ONE NaN confidence
  discarded a body whose KEYPOINTS were valid — found on the first real take, 160/160 frames. Fixed with `nanmean`. Kept because it is a caveat
  for any earlier session, and because confidence can be NaN at range.
- **O31. NEW — the ambiguity reference is density-dependent and dimensionality-confounded.** It FALLS as episodes are added
  (3/10/32 episodes → 0.002779/0.002386/0.002077), so a stage can fail a denser reference without having changed; across
  `obs_window` it RISES for geometric reasons. A density-matched estimator is the fix and is **NOT built**. Every Phase 4 gate is
  provisional on scripted data until re-run on piloted. `NOTES.md` 2026-09-21.
- **The Phase 1 exit gate was INCOMPLETE.** It scored grasp, placement, resting, tilt and falls and never scored CLEARANCE, so the box scraped
  the pickup platform (−1.4 mm at the shipping `lift_h`) and the hand penetrated it, every episode, invisibly. "Phase 1 closed 12/12" is true on
  the criteria as written, and the criteria had a hole. Under free lock timing the honest figure was **24/40 before the mitigation, 40/40 after**.
- **O17.** Permanent fact: a free-standing robot's equilibrium standoff never coincides with the standoff its arms can serve, at any lever swept
  (TR16). **Qualified 2026-09-15:** measured with the capped scripted station-keeper; a human at full command overcame it (§8).
- **O32. NEW — prediction: the ACT latent is ACTIVE on piloted data.** Scripted: KL 1e-05, z inert (§8 2026-09-22). Predicted on teleop:
  KL stays above zero and the z = 0 vs posterior-mean gap opens. If it collapses there too, beta (10; balance-matched 14.70) is the suspect.
- **O6.** Dead code: `gating.py`, `GatingConfig`, `TorsoYawConfig`, `IKConfig.neutral_weight`/`.target_deadzone`, `set_waist_yaw`,
  `ZEDConfig.camera_fps` (O29); `RejectReason` survives for `NAN`. **O7.** Stale docs: README claims torso-yaw following and active gating;
  `config.py` says locomotion is "not yet built"; `test/*.py` is stale. Resolved issues (O10, O12, O14–O16, O18, O20–O24): `NOTES.md` "2026-09-22 — CLAUDE.md §8 ARCHIVE".

## 11. File and module structure

Full listing in `NOTES.md`. Only the entries that carry a decision live here:

- `g1_data/spec.py` — **the contract** (§6). `g1_data/phases.py` — lock predicate and platform geometry. `g1_data/reset.py` — the one place an
  episode starts; **read its docstring before adding per-episode state**; it also enforces the D18 contact contract.
- `g1_teleop/grasp.py` — **its thresholds are the grasp contract**. `g1_teleop/base_lock.py` (D12) — its docstring holds the measured `eq_data`
  layout (TR17). `g1_teleop/contact_contract.py` (D18) — the setting, the compiled-model check and the runtime toggle.
- `locomotion_input.py` — **do not delete**: `PelvisVelocity` is the evidence for D6. `g1_teleop/gating.py` — DEAD, kept for `RejectReason` (O6).
- `tools/` — `record_keypoints.py`, `analyze_keypoints.py`, `teleop_physics_check.py`, `teleop_throughput.py`, `bprime_validate.py`,
  `plot_platform_regions.py`. `../docs/Thesis_Proposal.pdf` — Ch.3 at PDF pages 33–56. `../NOTES.md` — overflow for this file.

## 12. Conventions

- **Frames:** ZED X right, Y down, Z forward; G1 X forward, Y left, Z up. **BODY_38:** pelvis 0, shoulder 12/13, elbow 14/15, wrist 16/17.
  **Rates:** physics 500 Hz; locomotion 50 Hz (decimation 10); recording + policy 25 Hz (D4).
- **MuJoCo indexing:** **never hardcode a qpos/qvel/ctrl offset** — `ModelIndex.resolve(model)`. Dimensions, masks and clips come from
  `g1_data/spec.py` and nowhere else. **Constraints:** never assume an `eq_data` layout — measure it (TR17). **Contacts:** gate on `d.ncon` +
  `mj_contactForce`, never `mj_geomDistance` (TR19). **Measurement:** stepped model only, never a twin residual (TR16a); if a quantity that
  must vary comes out identical, treat it as a rig artifact (TR18).
- **Config:** tunables are frozen dataclasses in `config.py`; entry-point scripts violate this with module-level constants.

## 13. Verified vs assumed

Everything in §4, §6 and §12 is verified by loading the model or running the file; full list in `NOTES.md`. The camera's identity and
settings are read back FROM the camera and stored in every take's metadata (`tools/record_keypoints.py`); read them there. **Still assumed:**
box friction/solref were tuned, not inherited.

## 14. Maintenance protocol — instructions to future sessions

**At the end of every working session, update this file:** (1) rewrite §2 to whatever actually blocks now — a fresh session must start from §2
alone; (2) add a dated §8 entry for every decision with its reason, a §9 entry for anything tried that did not work, and a §7 row for any new
proposal divergence with the chapter affected; (3) move completed items from §5 into §4, citing the file that proves it; (4) update §10 —
resolved issues are the *only* deletable thing here.

**Rules:** **never delete a §9 entry** — the most expensive knowledge here to rediscover, even when it later looks obvious, and never solve
overflow by cutting §9. **Keep this file under 400 lines** — raised from 300 on 2026-09-22: that cap was set when the project was small,
the model work ("Phase 4" in commits, PLAN.md P5) alone added three TR entries and several decisions, and landing at 299 left no room
for the next session; a cap that forces a retirement exercise every session costs more than it saves. **If code and this file disagree,
the code is correct** — fix the file and note the drift in §8. **If code and the proposal disagree, neither is automatically
correct** — do not silently reconcile; surface it to Charles and record the outcome in §7. **No routine implementation detail here** — only
decisions, state, and what a fresh session could not work out from the code in five minutes.

**Retirement policy — how this file sheds weight; do not invent another trim.** A §8 decision retires to a dated `NOTES.md` archive section,
verbatim, leaving a one-line pointer, when ALL hold: (1) its phase is CLOSED (PLAN.md numbering); (2) everything it decided is carried
elsewhere — a §6/§7/§9/§10 entry or the code's own docstring, cited in the archive; (3) it guards nothing. It STAYS if it is a "never /
do not re-open" item, an open question, a disclosure still owed to the write-up, or a result with no other home in this file. Resolved
§10 issues move to the same archive. §9 never retires. Report the stay/go list before moving anything; never cut a decision to fit.
