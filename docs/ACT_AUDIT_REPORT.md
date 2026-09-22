# ACT implementation audit — report

**Date:** 2026-09-23. **Audited:** `g1_model/act.py`, `g1_model/train.py` (loss, scoring, gate), `tools/train_bc.py` (gate
runner), and run `runs/20260922-153630_act_overfit10_K100`. **Reference:** `reference/act/` at
`742c753c0d4a5d87076c8f69e5628c79a8cc5488` (verified with `git rev-parse HEAD`; `git status` clean before and after).
**Paper:** arXiv 2304.13705v1, fetched and text-extracted; "p.N" means PDF page N.

**Reading order kept.** `docs/ACT_CORRESPONDENCE.md` was opened only after every measurement below was finished, and only to
answer §I (whether deviations were disclosed). **Nothing in the repo or the clone was modified.** The reference was loaded with
`sys.dont_write_bytecode = True`, so no `__pycache__` was written into it. The measurement scripts are in the session scratchpad,
not the repo, per the no-new-files rule: `audit_common.py`, `m3_decoder.py`, `m124.py`, `m5_gate.py`, `m_extra.py`, `m2b.py`.

---

## Verdict

**Architecturally this is ACT, and that was measured, not reasoned.** Run on the same inputs as the reference's own
`transformer.py`, the implementation gives bit-identical outputs, latents, losses and gradients at every parameter. That holds in
train mode (dropout on) and eval mode. The reference side was built as its vision path with zero image tokens and 7 decoder layers
read at `hs[0]`.

**Its training is not the reference's.** Three undisclosed or wrongly disclosed differences change what gets learned: gradient
clipping binds on almost every step, the CVAE encoder is initialised differently, and 4 state dims are normalised up to 4.9×
harder than the reference would. Two correspondence rows marked **IDENTICAL** are false (27 and 44).

**On the only data that exists, the trained model is functionally chunked BC.** The posterior mean has |mu| ≤ 1.07e-3. Feeding
the encoder the TARGET changes the gated number by 1.0e-7. Nothing the gate measures depends on the CVAE.

---

## The five measured items

### 1. The latent reaches the decoder — **PASS**

- Published config (h 512, ff 3200, K 100). Same observation, different z, through `action_head(_decode(obs, latent_out_proj(z)))`:
  max|a(z=0) − a(z~N(0,I))| = **1.210**, max|a(z=0) − a(z=2)| = **1.672**. The output scale is max|a| = 1.434.
- Through the public training forward (eval mode, same ε, two different action chunks): max|Δmu| = **0.376**, max|Δa| = **0.168**.
- Gradient path: L1 of d(Σa)/dz = **6.66e3**.
- **Does the check fire?** Two mutants, patched in memory:
  - (a) latent token zeroed: my measurement Δ = **0.000**, and the repo's `test_act.py::test_latent_reaches_the_decoder` **FAILED**.
  - (b) latent token removed (the reference's dead `detr_vae.py:133-136` shape): Δ = **0.000**, and the test **FAILED**.
  - Unmutated, the test PASSED. The test is not trivial.
- Wiring: the latent becomes a transformer-encoder token at `act.py:539` (`torch.stack([latent_input, proprio])`), mirroring
  `transformer.py:62`. That is the reference's vision path (`detr_vae.py:131`), not its state-only path (`:136`).
- **Caveat on the trained checkpoint:** z~N(0,I) moves the outputs by a mean of **3.8e-3** (max 1.38). That is 9% of the gated
  error. The latent reaches the decoder; the trained decoder mostly ignores it (see item 5).

### 2. Encoder not invoked at inference, z = prior mean — **PASS**

I hooked every named submodule and ran `model(obs)`.

- In **eval and train mode alike**: 54 modules ran, **zero** CVAE-encoder modules ran, `torch.randn_like` was called **0**
  times, and z has shape (4, 32) with max|z| = **0.0**. The control (training forward) ran all 40 encoder-side modules and made 1
  `randn_like` call.
- **Weight-scramble test:** after adding N(0, 3²) to every CVAE-encoder parameter, inference output is **bit-identical**.
- The inference latent is built at `act.py:569` (`torch.zeros(B, latent_dim)`). The action chunk is not an argument on that path:
  `forward(obs)` has `actions=None`, and `encode` is reached only at `act.py:575`. The paper states z = 0 at test time in §IV-B
  (p.5) and Appendix C (p.15): "We simply set z to a zero vector". This matches `detr_vae.py:112-113`.
- **Mutation results for the repo's tests:**
  - `encode()` called at inference: `test_encoder_is_not_invoked_at_inference` **FAILED** (fires).
  - z sampled at inference: `test_inference_latent_is_exactly_the_prior_mean` **FAILED** (fires).
  - Encoder stack run *directly*, bypassing `encode()`: `test_act.py`'s test **PASSED (blind)**, because it counts calls to the
    `encode` method, not module executions. That gap is covered elsewhere:
    `test_train.py::test_act_declares_exactly_the_modules_encode_uses` **FAILED** on the same mutant.

### 3. One decoder layer = the reference's seven — **PASS (bit-identical)**

**Method.** This is independent of the implementer's own test (`test_act.py:681`), which compares the implementer's 1-layer
`_Decoder` with the implementer's 7-layer `_Decoder`, so it cannot detect a porting error in `_Decoder` itself.

- The reference side is the reference's `Transformer` from `reference/act/detr/models/transformer.py`, with 7 decoder layers,
  post-norm and `return_intermediate_dec=True` (`transformer.py:293-303`). It is called with the vision-path signature
  `detr_vae.py:131` and a zero-width image tensor, so `transformer.py:59-63` reduces to `[latent, proprio]`, and it is read at
  `[0]`.
- The encoder, decoder layer 1 and decoder norm were copied in. Decoder layers 2–7 kept their own independent xavier weights
  (|w| mean 0.0201 each), so they genuinely compute.

**Results.** The table shows max |Δ| between the two sides:

| | inference (z=0) | train mode, CVAE forward | eval mode, CVAE forward |
|---|---|---|---|
| action chunk | 0.0 (3 seeds) | 0.0 (3 seeds) | 0.0 (3 seeds) |
| mu | — | 0.0 | 0.0 |
| loss, ours (`act_loss`) | — | 0.0 (e.g. 90.6015549 both) | 0.0 |
| loss, reference formula (`policy.py:30-34`) | — | 0.0 | 0.0 |
| every gradient, all params | — | **0.0, 0 params differ** | **0.0, 0 params differ** |

- The reference's decoder layers 2–7 receive gradient of exactly **0.0**. The grad is a zero tensor, not None.
- **5 AdamW steps** (lr 1e-5, weight decay 1e-4, the reference's optimizer holding all 7 layers):
  - Without clipping: parameters are bit-identical.
  - With the gate's clip of 1.0: max|Δparam| = **5.96e-8**. This is float reduction order: the grad norm is 678.01947 against
    678.01953, because the zero tensors enter the norm sum. It is not a model difference.
- One consequence: the norm at initialisation is **678**, against a clip of 1.0 (see finding R1).

### 4. Padded timesteps masked in the CVAE encoder — **PASS**

**Constructed case**, published config, K = 100, three samples with the loader's zero padding (`loader.py:376-382`):

| sample | real steps | encoder len | positions masked | [CLS, state] masked | mask[2:] == ~action_mask |
|---|---|---|---|---|---|
| 0 | 100 | 102 | 0 | False, False | True |
| 1 (last tick) | 1 | 102 | 99 | False, False | True |
| 2 | 37 | 102 | 63 | False, False | True |

- **Garbage test:** padded rows replaced by N(0, 100²) give Δmu = Δlogvar = **0.000** in eval mode, and also in train mode with
  the same dropout seed.
- **Controls (the check fires):**
  - Changing sample 1's single real step by +1 gives Δmu = **0.107**.
  - With the mask removed, the same garbage gives Δmu = **1.61**.
  - Mutant with the mask dropped inside `encode`: `test_padding_does_not_leak_into_the_latent` **FAILED**.
- **Training path:** `loss_terms` delivers the identical mask to the encoder. With garbage padded targets the total loss is
  unchanged (95.991729736 both).
- **Real loader sample** (#705, last tick of an episode, gate dataset): 1 real step, **99 of 102** positions masked, [CLS, state]
  unmasked.
- The implementation matches `detr_vae.py:98-99, 104` and `utils.py:53-54`.

### 5. The gate scores the deployed function — **PASS on the real run; the guards are partly defeatable**

**On the real run:**

- **Final weights.** `last.pt` equals `state.pt` bit for bit, both at step 195000, which equals `result.json`
  `optimizer_steps`. `gate.json` says `scored_weights: final weights (last.pt)`.
- **Reproduced.** `score_deployment(last.pt)` = **0.041081068614**, against `gate.json` 0.041081068614 (Δ 0.0). My own
  independent loop, which does not use the repo's guard, gives 0.041081068970: 11,412,800 elements, Δ 3.6e-10 from summation
  precision.
- **Instrumented during scoring:** no CVAE-encoder module ran, no module was in train mode, and max|z| = 0.0.

**Attacks on `_deployment_guard` (`train.py:498-537`).** The honest score on a 64-sample subset is 0.078569.

| attack | result |
|---|---|
| inference samples z ~ N(0, 50²) instead of z = 0 | **NOT CAUGHT** (scored 1.349) |
| CVAE encoder run via `.forward()` (pre-hooks bypassed) | **NOT CAUGHT** |
| functional `F.dropout(..., training=True)` in the deployed forward | **NOT CAUGHT** (scored 1.238) |
| target stashed in `loss_terms`, returned by `forward(obs)` | **NOT CAUGHT** (scored **0.000000**) |
| submodule flipped to `.train()` inside forward | caught |
| `encode()` called at inference | caught |
| model handed over in train mode | scored in eval (0.078569), and train mode was restored afterwards |

**What this means:**

- The guard enforces **module mode and a hand-maintained name list**. It does **not** enforce z = 0; only `ACTPolicy.forward`
  and a unit test do.
- It cannot see state carried between calls. The stash attack is exactly the shape of an ACT-LSTM hidden state that is not reset
  before scoring (G5). `score_deployment` has no reset hook.
- None of these defeats is present in the current code. The instrumented real pass above shows that.

**The audit file's A5 premise is false on this checkpoint.** Scoring the *oracle* path (encoder fed the target, z = posterior
mean) gives **0.041081172**, against 0.041081069 for the deployed path: a difference of **1.0e-7**, or 2.5e-6 relative. A leak
would not "score implausibly well"; it would score the same. The guard is the **only** defence, and no score-based test can
substitute for it.

---

## Discrepancies against the reference, ranked by whether they change what the model computes

### R — change what is learned (the function class is the same, the trained function is not)

**R1. Gradient clipping at 1.0, binding on nearly every step.**

- *Reference:* none. `--clip_max_norm` is marked `# not used` (`detr/main.py:20`), and `grep clip_grad` finds nothing anywhere
  in the clone.
- *Ours:* `TrainConfig.grad_clip = 1.0` (`train.py:362`), applied at `train.py:1040-1041`, and 1.0 in the gate run
  (`metadata.json:8`).
- *Measured:* norm **678** at initialisation. At the *converged* gate weights, the norm over 20 batches of 8 is median
  **1.35** (range 0.99–2.49) and exceeds the clip on **19/20** batches, so the clip is active throughout training.
- *Disclosure:* row 40 discloses it as ADAPTED.
- *Why it matters:* this is TR28's own pattern. The value is inherited from the shared loop's default, not stated by ACT.
  `assert_optimizer_source` checks only lr, weight decay and optimizer (`act.py:382-383`, `train.py:419-424`), so
  **`grad_clip` can drift between ACT and ACT-LSTM with nothing raising**. That confounds RQ3 directly (G7).

**R2. CVAE encoder initialisation.**

- *Reference:* `build_encoder` (`detr_vae.py:212-226`) never re-initialises, and `TransformerEncoder` deep-copies one layer
  (`transformer.py:83, 289-290`). `_reset_parameters` belongs to `Transformer` only (`transformer.py:39, 44-47`).
  - *Measured on the reference's code:* its 4 CVAE-encoder layers are **identical at init**, with PyTorch default init
    (linear1 std **0.02553**).
- *Ours:* xavier is applied to `self.encoder` as well (`act.py:490-493`), giving 4 distinct layers (linear1 std **0.02321**).
- *Disclosure:* the docstring at `act.py:486-489` ("Applied to the transformer stacks only, as in the reference") is wrong about
  exactly this module. **Correspondence row 27 "Weight init — IDENTICAL" is false.**

**R3. Normalisation std floor.**

- *Reference:* clips std to `[1e-2, ∞)` for both qpos and action (`utils.py:97, 102`).
- *Ours:* `STD_FLOOR = 1e-6` (`spec.py:639`).
- *Measured:* on the train-split stats, **4 state dims** fall below 1e-2:

  | dim | what it is | std |
  |---|---|---|
  | 9 | base z (`BASE_POS`, `spec.py:379`) | 0.0087 |
  | 43 | last joint of `ARM_R_Q`, pinned wrist (`spec.py:388`) | 0.0020 |
  | 44 | waist (`WAIST_Q`, `spec.py:389`, pinned per D10) | 0.0030 |
  | 45 | waist | 0.0069 |

- *Effect:* pinned-joint jitter is scaled up to **4.9×** more than the reference would scale it. This is the same "z-scoring
  amplifies noise" hazard D14 was written against.
- *Disclosure:* row 45 quotes the reference's clip but does not list the floor as a divergence. No action dims are affected.

**R4. L1:KL balance.**

- *Disclosed* at `act.py:628-636`: reconstruction is 1.4704× the reference's scale, so β = 10 here is about β = 6.8 in the
  reference's units.
- *Measured D2:* ours is K-invariant (a uniform error of 0.5 scores 0.5 at K = 1 and K = 100). The reference's reduction scores
  **0.3325** at K = 100 with padding, so its scale depends on padding.
- *Measured D6:* at the first logged window β·KL = **4.86** against reconstruction **0.372**. The KL is **13×** the
  reconstruction and dominates the loss, then decays monotonically to 5.2e-5 by step 113k
  (`runs/20260921-161407_act_overfit10_K100/metrics.jsonl`). The collapse is visible within the first 20k steps.

**R5. Model-selection criterion.**

- *Reference:* validation `L1 + β·KL` with the encoder fed the target, in eval mode (`imitate_episodes.py:342-354` →
  `policy.py:23-34`).
- *Ours:* deployment reconstruction, z = 0, no KL (`train.py:584-589`).
- Ours is arguably the better criterion, but **correspondence row 44 "IDENTICAL" is false**. It is moot at the gate (`"val":
  null`) and binding for the Exp 1/2 models.

**R6. Rollout rule.**

- *Ours:* `first_action`, re-planning every tick (`models.py:131, 182`).
- *Reference default:* open-loop for K steps (`imitate_episodes.py:191, 248, 261`).
- *Paper:* queries every step with ensembling (§IV-A and its inference algorithm, p.5).
- Ours is neither of them. It is disclosed (row 50), but **not measurable: no rollout exists**.

**R7. Horizon.** K = 100 at 25 Hz is **4.0 s**. The reference's 100 at `DT = 0.02` (`constants.py:36`) is **2.0 s**. It is
disclosed as UNRESOLVED (row 52).

### S — present in the model, but do not change what it computes (measured or trivially true)

- **S1.** 1 decoder layer instead of 7: bit-identical (item 3). Correspondence rows 18/22 still say **UNRESOLVED**, while the code
  has decided (`act.py:93`), so the document is stale.
- **S2.** Learned queries (`detr_vae.py:54`), where the paper says "fixed position embedding, with dimensions k×512" (§IV-C,
  p.6) and "fixed sinusoidal embeddings for the first layer" (App. C, p.14). Following the code is right, and it is disclosed at
  `act.py:470-472`.
- **S3.** The action head is `Linear` in the code and ours (`detr_vae.py:52`, `act.py:474`); the paper says "down-projected with
  an MLP into k×14" (§IV-C, p.6). **This paper/code disagreement is recorded nowhere I found**, and the audit rule says to record
  both.
- **S4.** The ensembler does not reproduce the reference's all-zero-row filter (`imitate_episodes.py:253`). It differs only if a
  predicted action is exactly all zeros.
- **S5.** `is_pad_head` is dropped. It is unused in the reference's loss (`policy.py:27-35`). Fine, and disclosed.

### T — documentation and metadata errors (no computation changes; they would put false statements in the thesis)

- **T1.** `act.py:656-657`: "a smaller `m` incorporates new observations more slowly". **The paper says the opposite**: "a
  smaller m means faster incorporation" (§IV-A, p.5). This is an I4 hit, a paper citation that inverts the paper. Correspondence
  row 49 quotes the paper correctly, so the code docstring contradicts its own companion document. On "no numeric m": confirmed
  **NOT FOUND IN SOURCE** (§IV-A and the inference algorithm's line 7, p.5, give only `w_i = exp(−m∗i)`). "w₀ is the weight for the oldest action"
  is confirmed (§IV-A, p.5).
- **T2.** `tools/train_bc.py:328` writes `regularization: "NONE: dropout 0, weight_decay 0, no augmentation"` into **every**
  overfit-10 run. ACT keeps dropout 0.1 (`train_bc.py:96-102`), and the same file's model repr shows `Dropout(p=0.1)`.
  `runs/20260922-153630_act_overfit10_K100/metadata.json:62` is false. Dropout was also the whole of the 1.140 → 0.920 gap
  (CLAUDE.md §8, 2026-09-22), so a metadata line denying dropout is precisely the one that must not be wrong.
- **T3.** Correspondence **row 27 (init) and row 44 (selection) say IDENTICAL; both are false** (R2, R5). Row 36 still says
  "decision required", although `act.py:115` sets 1e-5.
- **T4.** `act.py:408` says the LSTM attaches "between `_encode_obs` and the transformer". There is no `_encode_obs`.

### Verified as matching (with evidence)

- Latent dim 32 (`act.py:68` ↔ `detr_vae.py:67`).
- `reparametrize` with `div(2)` (`act.py:172` ↔ `detr_vae.py:18`).
- Encoder sequence `[CLS, state, chunk]`, length K+2 (`act.py:512` ↔ `detr_vae.py:95`; paper App. C p.14 "(k + 2)").
- `pos_table` is a buffer, not a parameter (measured: shape (1, 102, 512), absent from `named_parameters`).
- CLS-only readout (`act.py:523` ↔ `detr_vae.py:105`).
- `tgt = 0`, with queries entering as `query_pos` (`act.py:542, 545` ↔ `transformer.py:72, 74-75`).
- KL is `total_kld`: ours 1.459705472 against 1.459705424 hand-computed (`policy.py:79-80`).
- Constant-dim error gives masked_l1 = **0.0** exactly.
- AdamW, and no scheduler (`train.py:962`; the reference constructs none).
- Post-norm, ReLU, dropout 0.1, and CVAE/policy encoders as separate modules.
- Table III (p.18): lr 1e-5, batch 8, encoder layers 4, decoder layers 7, feedforward 3200, hidden 512, heads 8, chunk 100, β 10,
  dropout 0.1. Every one matches the cited README / imitate_episodes values.
- Paper Algorithm 1 line 9 says `MSE` (p.5), while §IV-C says L1 (p.6) and the code uses L1 (`policy.py:30`). Ours is L1.

---

## Not verified

- **A2 as specified** (a short training run with `latent_input` forced to zero, comparing losses). Not run. The proxies are item
  5's oracle gap (1.0e-7) and the trained model's z-sensitivity (3.8e-3 mean).
- **All of G (ACT-LSTM).** It does not exist, and `use_lstm` is absent (`test_act.py:571` asserts that). G4 (RNG shift from
  constructing an LSTM) cannot be measured until it does. The parameter count exists for ACT only: **40,226,006**.
- **F1/F2 behaviour.** There is no rollout or evaluation harness.
- **I1.** There is no Chapter 3 methods table in the repo to trace.
- **Whether the gate reference is sound** (O31, density-dependent). That is out of scope here, and every gate verdict inherits it.

---

## Closing question: which A–G failures would the overfit-10 gate catch?

**I agree with "almost none", and would go further: none of the gate-blind ones.** The gate catches only failures that stop a
40M-parameter model from memorising 7,628 samples. In A–G the only failures of that kind are **crashes**: C2 (num_queries ≠ K)
and C5 (head width ≠ 22), both of which a shape check raises before any gate. A gross break of the observation path would also be
caught, but none is listed.

**Evidence that the rest slip through, from this audit:**

- **A1–A5 and B1–B5 (the whole CVAE)** slip through. Measured: the trained deployed function does not depend on the encoder (the
  oracle gap is 1.0e-7), and chunked BC, which has no CVAE, passes the same gate at 0.953. A decorative CVAE, the dead
  state-only branch, learned positional encodings, a mean-pooled readout, unmasked padding, or an encoder leaked into inference
  would all pass.
- **C1, C3, C4 and C6** slip through. Learned vs fixed queries, queries as `tgt`, reading `hs[-1]` of 7 layers, and anything
  consuming `is_pad_head` all train.
- **D1–D6** slip through. D6 actually **happened and passed**: KL collapsed from 0.486 to 5e-5 and the gate read PASS 0.920. β,
  `mean_kld` and the reduction scale are rescalings the optimizer absorbs. D3 unmasked constant dims are trivially predictable.
- **E1–E8** slip through. R1 (clipping) and R2 (init) **are present now, and the gate passed**. E4 pre-norm would help the gate.
  E7 and E8 need a validation set, and the gate has none.
- **F1–F4** slip through: the gate never rolls out.
- **G1–G10** would slip through, and G3 (capacity) and G5 (a hidden-state leak) would make the gate *better*. The gate would
  reward the confound.

The gate is a plumbing check. The correctness evidence for this ACT is items 1–4 above and the mutation-tested unit tests, not
the gate.
