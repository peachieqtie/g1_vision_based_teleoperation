# PLAN.md — Current state to thesis completion

Companion to `CLAUDE.md`. CLAUDE.md holds *state and decisions*; this file holds *sequence*. Read CLAUDE.md §2 first — it is
always more current than this file. Detail and measurements live in `NOTES.md`.

**Phase numbering.** The numbering below is **authoritative** and CLAUDE.md §2 uses it: P1 grasp physics, P2 freeze specs,
P3 recorder, P4 pilot, P5 model code, P6 collection, P7 train+eval, P8 writing. One wrinkle to know about: **Phase 2 grew well
beyond its original task list.** It absorbed the base-lock predicate, the O26 mitigation and the whole teleop-parity effort
(base lock in the entry point, the TR17 pad fix, D18, the operator overlay, the keypoint recorder, the IK speed fix). Some of
that is really Phase-4 pilot preparation done early, because a live camera was available. Nothing was skipped; the work simply
did not arrive in the order this file predicted.

**Status: Phases 1 and 2 are CLOSED.** Phase 5 can start immediately and in parallel. Phase 3 is next.

```
P1 grasp physics ─┐
   (CLOSED)       ├─> P3 recorder ──> P4 pilot ──> P6 collection ──> P7 train+eval ──> P8 writing
P2 freeze specs ──┘                                                     ^
   (CLOSED)                                                             │
P5 model codebase ──────────────────────────────────────────────────────┘  (parallel, no P1 dependency)
```

---

## Phase 1 — Stabilize box-grasp physics — CLOSED 2026-09-09

**Closed by** the walking place: the demonstrator walks in, grasps, carries the box 1.5 m to the goal platform, lowers and
releases — **12/12 at both 14 s and 20 s settle**, palm error 32–35 mm against the 45 mm guard, placement 0.029–0.049 m against
the 0.10 m limit, zero falls. The gripper is a **weld** (D11), not the pressing pad pair this file originally specified; the
friction pinch is a reportable negative result.

**Two things to carry forward rather than forget.**

1. **The exit gate was incomplete.** It scored grasp, placement, resting, tilt and falls and never scored CLEARANCE, so the box
   scraped the pickup platform and the hand penetrated it in every episode, invisibly (CLAUDE.md §10). Under free lock timing
   the honest reliability figure was 24/40 before the O26 mitigation, 40/40 after. **Any future exit gate must score the thing
   that is allowed to go wrong silently**, not only the thing being demonstrated.
2. **Placement carries a systematic +28 to +31 mm x bias.** Every demonstration places the box ~30 mm past the nominal target,
   well inside tolerance, and a policy will learn it.

---

## Phase 2 — Freeze state vector, action vector, spatial split — CLOSED 2026-09-11

**Exit criteria, all three met:** `g1_data/spec.py` exists and both vectors are dimension-complete with no `TODO`;
the normalization contract is written (training-split statistics only, masked dims pinned to mean 0 / std 1); and the platform
diagram is saved for Chapter 3 — **`docs/figures/platform_regions.pdf`**, generated from the model by
`tools/plot_platform_regions.py`.

**What was frozen:** `SPEC_VERSION = g1-spec-1.1.0`, the name-derived permutation, the constant-dim mask (6 of 22 action dims
excluded, 16 trainable), the phase vocabulary with explicit stable integers, clip limits, and the timing convention. 43 tests.
The four schema decisions and their reasons are in CLAUDE.md §8 and `NOTES.md` "PHASE 2 CLOSEOUT".

**Also delivered under this banner** (see the numbering note above): the event-driven base-lock predicate, the O26 staged-raise
mitigation, teleop parity in `run_integrated_combined.py`, D18, the raw keypoint recorder and analysis, and the IK speed fix
that took the live loop from 37.4% to 128.2% of real time.

**Objective 1 is demonstrated end to end:** a live operator completed the full task with the ZED under stepped physics.

---

## Phase 3 — Data logging pipeline — NEXT

**Decided already, do not re-open:** one `.npz` per episode; phase labels **auto-derived**, and derivable OFFLINE from the
recorded trajectory (the weld bit gives grasp and release, box height gives lift and lower, base-to-box distance gives approach
and transport), so collection needs no live phase classifier.

**Tasks:**
1. `g1_data/recorder.py` — log the pair at 25 Hz, built at ONE point in the loop before `mj_step` via `spec.SpecLayout.build`.
   **Action is `data.ctrl`, never twin qpos.** Buffer in memory, write once at episode end, so a rejected episode costs nothing.
2. Write the per-episode file exactly as agreed: `states`, `actions`, `phase_labels`, `gait_phase`, `seed`, `box_spawn_xy`,
   `standoff_cmd`, `lateral_cmd`, `heldout`, four per-phase success flags, `placement_error`, `tilt`, weld and lock
   engage/release ticks, `wrist_dev_rad`, **sim-to-wall ratio**, `reset_fingerprint`, `SPEC_VERSION`, git commit, mujoco
   version. ~50 MB for 150 episodes.
3. **Carry the contact contract.** Record `contact_contract` from the compiled model and refuse to evaluate a policy against a
   different one (`g1_teleop/contact_contract.py` already raises in `reset_episode`; the hook for dataset-vs-model exists and
   is unused). D18 is only safe if collection and evaluation share it.
4. `g1_data/success.py` — the four criteria from §3.8.2, using the spec accessors: grasp, lift (box ≥ platform + 0.05 m),
   transport (box above floor throughout), place (‖box − goal‖_xy < 0.10 m **and** resting **and** roughly upright, Q4).
5. Operator controls: start / discard-and-retry / accept (proposal §3.3.8).
6. **Log the sim-to-wall ratio per episode** (CLAUDE.md §8, 2026-09-15). Below real time the operator's motion is
   time-compressed in sim time and the compression varies with whether tracking is holding; at ≥100% the pacing sleep absorbs
   it and the ratio should read ~1.0. An episode that drops below is the one to inspect.

**Exit criterion:** 5 hand-driven episodes recorded, reloaded, and replayed **open-loop** into the simulator, reproducing the
original trajectory within a stated tolerance. If replay does not reproduce, the logged action is the wrong quantity — find that
out now, not after 150 episodes.

---

## Phase 4 — Pilot: full loop end to end — HARD GATE before scaling

**Partly de-risked already:** a live operator has completed the full task, and the teleop path runs at 128% of real time with
tracking active, so the pilot is no longer discovering whether the loop runs at all. It is discovering whether the DATA is right.

**Tasks:**
1. Record **10 complete episodes** with the real ZED, real demonstrator, real walking, real grasp, real placement — training
   region only. Measure per-episode wall-clock time and the reject rate; that turns Phase 6's schedule into a number.
2. Build the dataset from those 10, run normalization and action-chunk construction, and train BC to **deliberately overfit**
   them. A model that cannot overfit 10 episodes has a data-format bug, not a capacity problem.
3. Deploy that overfit policy autonomously and confirm the loop closes: state in, 22-D action out, arm/waist/gripper dims reach
   the actuators, velocity dims reach the locomotion policy.
4. Confirm phase labels and success detection fire correctly on known-good AND deliberately sabotaged episodes.
5. **Settle the D12 question here if it is not settled earlier** (CLAUDE.md §2): measure the standoff spread at grasp without
   the base lock, over seeds, and whether a learned policy can hold against the O17 attractor the way the operator did.

**Exit criterion:** 10 clean episodes on disk; BC overfits them; the autonomous loop runs a full episode without crashing; the
success detector agrees with human judgement on all 10.

---

## Phase 5 — Shared model codebase — RUNS IN PARALLEL, can start now

No dependency on Phase 1 or 3, only on the Phase-2 dimensions, which are frozen.

**Decide before starting:** observation window `W_o`, action chunk size `K`, LSTM hidden size and layer count (proposal: 2
stacked, dropout 0.3), transformer width/depth, KL weight β. The proposal defers all of these.

**Tasks:**
1. `g1_policy/base.py` — one interface all three conditions satisfy, so deployment never branches on policy type.
2. `g1_policy/bc.py` — feedforward, single-step, L1 only, no CVAE.
3. `g1_policy/act.py` — **one class**: transformer encoder/decoder, action chunking, CVAE objective (L1 + β·KL), and a
   `use_lstm` flag adding the stacked LSTM encoder whose hidden state is projected and concatenated as an input token.
4. `g1_policy/ensemble.py` — temporal ensembling of overlapping chunks at deployment.
5. `g1_train/train.py` — Adam, cosine annealing, early stop after 10 epochs without validation improvement, best-checkpoint
   save. Identical across conditions.
6. Shape-and-gradient unit tests on synthetic data at the frozen dimensions. **Mask the 6 constant action dims out of the loss.**

**Exit criterion:** all three train to convergence on synthetic data, and toggling `use_lstm` is the *only* difference between
ACT and ACT-LSTM — verified by diffing the two config objects.

---

## Phase 6 — Demonstration collection

**Decided:** **150 episodes**; seed streams **pre-partitioned before collection** so held-out leakage is provable by
construction rather than audited afterwards; no two consecutive episodes share a spawn.

**Tasks:**
1. **Collect 25 FIRST and train BC on them before collecting the rest.** If BC saturates at 25 the task does not discriminate
   and the comparison cannot answer RQ2/RQ3 — that is a cheap check against an expensive mistake.
2. Collect the remaining episodes to 150, inspecting and re-recording failures (§3.3.8).
3. Plot spawn coverage of the training region — a Chapter 3 figure — and verify zero held-out leakage.
4. Back up `data/raw/` off-machine. Irreplaceable human-generated data; note that `*.npz` is gitignored.
5. Watch the recorded **sim-to-wall ratio** and reject episodes that ran badly below real time.

**Exit criterion:** 150 accepted episodes, coverage plot generated, zero held-out leakage, backup verified restorable.

---

## Phase 7 — Training and evaluation

**Decided:** train/val **80/20 stratified by binned spawn position**; normalization from the training split only;
**Experiment 1 = 100 FRESH in-region seeds** (not the collection spawns, D16), **Experiment 2 = 100 held-out-patch seeds**;
**evaluation seeds FIXED across all three policies**; data-scaling curve on **NESTED** subsets 25 ⊂ 50 ⊂ 100 ⊂ 150.

**Tasks:**
1. Train all three conditions on the identical dataset with identical shared hyperparameters.
2. Experiment 1 and Experiment 2, four phase-level success rates each (grasp, lift, transport, place).
3. Generalization gap per phase per policy: `Δ = SR_in − SR_out`.
4. Three pairwise comparisons: BC vs ACT (does chunking help?), ACT vs ACT-LSTM (the actual research question), BC vs ACT-LSTM
   (the headline).
5. Per-phase failure taxonomy over the five buckets — **WALK_IN included**, and report its share against the fact that WALK_IN
   is only 10–13% of a predicate-lock episode (CLAUDE.md §8).
6. Record failure-mode video: the table says *where* each policy fails, video says *how*.

**State up front, in the write-up and not only in the discussion:** 100 evaluation episodes give SE ≈ 5% on a rate near 50%, so
two policies are separable at roughly **15 percentage points**. Smaller differences are not measurable with this design.

**Exit criterion:** a 3-policy × 4-phase × 2-experiment table plus the gap table, every number reproducible from a saved
checkpoint and a fixed seed list.

---

## Phase 8 — Thesis writing

Can start during Phase 6 — collection is mostly waiting.

| Chapter | Fed by | Status |
|---|---|---|
| 1 Introduction | — | proposal text largely reusable; RQ2 must fold in the plain-ACT condition (D8) |
| 2 RRL | — | reusable; add a chunking-ablation framing |
| 3.1–3.2 Design | — | reusable |
| 3.3.1 Scene | P1, P2 | **rewrite**: init pose (D5), control rate (D4), spawn region + held-out patch |
| 3.3.3 Coordinate transform | code | **rewrite**: `DEPTH_SCALE` is not a pure rotation (D1) |
| 3.3.4 Walking detection | P2 | **rewrite**: the literal pelvis-velocity method does not work (D6, TR1) — a documented negative result on a proposed method is a contribution, not a retreat |
| 3.3.5 Arm teleoperation | code | **rewrite**: 4-joint IK, pinned wrists, elbow+wrist 6-D task (D2, D3), and D2's REAL justification — it protects the place, not the grasp pose |
| 3.3.6 Grasping trigger | P1 | **full rewrite**: weld grasp (D11), geometric gate, friction pinch as a negative result (D7) |
| 3.3 + limitations | P2 | **new**: base support (D12) and the hand/platform contact exclusion (D18), both simulation-only relaxations that shape every downstream result |
| 3.3.8 Protocol | P4, P6 | update with pilot-measured timings, the real reject rate, and the operator's reported difficulties |
| 3.4 State/Action | P2 | **rewrite** from `spec.py`: gripper dims from the weld bit (D14), velocity clips (D15), velocity dims zero while locked (D17), the constant-dim mask |
| 3.5 Preprocessing | P5 | mostly reusable; normalization contract is fixed |
| 3.6 Architecture | P5 | **extend**: plain ACT as a third condition (D8) |
| 3.7 Training | P5, P7 | fill in the deferred hyperparameters |
| 3.8 Evaluation | P7 | extend to three policies; `d_place` = 0.10 m; episode cap from the measured 694–846 (D13); fresh Exp-1 seeds (D16); the 15-point separability limit |
| 3.9 Hardware/Software | — | **rewrite**: Windows 11, no ROS2 (D9) |
| 4 Results | P7 | new; report WALK_IN's share against its 10–13% base rate |
| 5 Conclusion | P7 | new; teleoperation fidelity (p50 56 mm palm error, O27) belongs in the limitations |

**Exit criterion:** defence-ready draft in which every deviation in CLAUDE.md §7 — now D1–D18 — has a written justification in
its chapter.

---

## Open questions

**Q1–Q6 answered** (CLAUDE.md §8, 2026-08-23 and 2026-09-08). Q6 in its final form: randomize the box position only; Objective
4 is a 2-D interior held-out patch, so both marginals stay in-distribution and only the COMBINATION is unseen.

**Q7 — RESOLVED 2026-09-09/15. The robot does walk during demonstrations.** Walking transport is the adopted task: the scripted
demonstrator carries the box 1.5 m at 12/12, and a live operator walked the robot in, grasped, transported and placed. The
fallback of putting both platforms within arm's reach is off the table, and loco-manipulation stays in the contribution.

**Q8 — OPEN, and worth answering before collection. Is D12 (the base lock) needed at all outside the scripted demonstrator?**
A live operator grasped, lifted and placed free-based, because a human at full forward command pushes against the O17 attractor
in a way the capped scripted station-keeper cannot. Unmeasured: standoff spread at grasp without the lock, and whether a learned
policy can hold against the attractor. Dropping D12 would remove the project's largest limitation AND give Objective 4 natural
standoff variation. Settle it in Phase 4 at the latest.

**Q9 — OPEN. How much of the teleop fidelity gap reaches the data?** p50 achieved palm error is 56 mm (O27), dropouts hold the
arms indefinitely rather than freezing (O28), and One-Euro is tuned for a frame rate that does not occur (O29). None of these
blocks collection, but all three shape what a policy learns from, and the recorder is where they become measurable per episode.
