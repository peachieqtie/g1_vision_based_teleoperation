# ACT correspondence table

What the reference implementation of ACT actually does, what we intend to do, and why
they differ. **Every row cites a file and line in the pinned clone or a page of the
paper.** Nothing in this document is stated from recollection; where a fact could not be
found in either source it says `NOT FOUND IN SOURCE`.

## The pinned reference

| | |
|---|---|
| repository | `https://github.com/tonyzhaozh/act` (resolved; not moved or renamed) |
| commit | **`742c753c0d4a5d87076c8f69e5628c79a8cc5488`** |
| commit date | 2024-01-28 12:18:07 -0800 |
| branch | `main` |
| clone location | `reference/act/` — **gitignored**, never vendored into the tracked tree |
| paper | Zhao, Kumar, Levine, Finn, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* |
| arXiv | **2304.13705v1**, submitted 23 Apr 2023 (v1 is the only version) |

That commit hash is what makes every citation below checkable. If the clone is
re-created, check out that hash or the line numbers here are meaningless.

`file:line` citations are relative to `reference/act/`.

## Status of this document

This session wrote **no model code**. The "our behaviour" column is therefore
**PROPOSED**, not implemented. Sessions 2 and 3 implement it and audit against it. Where
a row records a choice that has not been made, it says so and Part D of the session
report carries the options.

---

## 1. File inventory and what carries what

| file | lines | carries |
|---|---|---|
| `detr/models/detr_vae.py` | 278 | **the architecture**: `DETRVAE` (CVAE encoder + decoder wiring), `build_encoder`, `build`, `reparametrize`, `get_sinusoid_encoding_table` |
| `detr/models/transformer.py` | 314 | **the architecture**: `Transformer`, encoder/decoder layers, `build_transformer` |
| `detr/models/backbone.py` | 122 | **the architecture**: ResNet-18 vision backbone, frozen BatchNorm |
| `detr/models/position_encoding.py` | 93 | **the architecture**: 2-D sinusoidal / learned image position embeddings |
| `detr/main.py` | 114 | argument defaults and **optimizer construction** |
| `policy.py` | 84 | **the objective**: `ACTPolicy.__call__`, `kl_divergence` |
| `imitate_episodes.py` | 435 | **the training loop** (`train_bc`), **inference** (`eval_bc`), temporal ensembling |
| `utils.py` | 189 | dataset, normalization statistics, train/val split, `set_seed` |
| `constants.py` | 76 | `DT`, task configs, gripper conversions |
| `sim_env.py`, `ee_sim_env.py`, `scripted_policy.py`, `record_sim_episodes.py`, `visualize_episodes.py` | — | their MuJoCo environment and data collection; **not architecture** |
| `detr/util/misc.py`, `box_ops.py`, `plot_utils.py` | — | inherited DETR utilities; `box_ops` is unused by ACT |

---

## 2. The correspondence table

**"Gate-blind"** marks rows where getting it wrong would still pass an overfit-10 gate.
Those are the rows session 3 must check hardest: a 10-episode memorisation test does not
exercise generalisation, regularisation, inference-time behaviour, or anything that only
matters across episodes, so a mistake in any of them is invisible at the gate and
poisonous afterwards.

| # | component | reference behaviour (file:line) | our behaviour (PROPOSED) | verdict | reason | risk if wrong | gate-blind |
|---|---|---|---|---|---|---|---|
| 1 | Vision backbone | ResNet-18, ImageNet-pretrained, frozen BatchNorm, `backbone.py:92-96`; built unconditionally in `detr_vae.py:235-237` | **none** | **DROPPED** | See §3. State-only, simulation, ground-truth state; RQ is recurrence, not perception. | None for our RQ. Removes the largest parameter block and all image tokens. | no |
| 2 | Image position embedding | 2-D sinusoidal, `position_encoding.py:87`, `N_steps = hidden_dim // 2` `:84` | **none** | **DROPPED** | Follows row 1: no image tokens to position. | None. | no |
| 3 | Image → token sequence | 15×20×512 feature map flattened to 300 tokens/camera; cameras concatenated along width `detr_vae.py:129`; paper §C: 4 cameras → 1200 tokens | **none** | **DROPPED** | Follows row 1. | None. | no |
| 4 | Encoder token sequence | `[latent, proprio] ⧺ image tokens`, `transformer.py:62-63`. Non-image part is exactly **2 tokens**, evidenced by `additional_pos_embed = nn.Embedding(2, hidden_dim)` `detr_vae.py:76` | **`[latent, proprio]`, 2 tokens** (Option A, §4) | **ADAPTED** | Removing images removes image tokens and nothing else. The reference's own `nn.Embedding(2, ...)` is sized exactly for the two non-image tokens, so this is the reference's design for the non-image part rather than our invention. | A 2-token encoder is close to a no-op; if instead we invent extra tokens we are no longer running ACT. | **YES** |
| 5 | Latent injected into the policy | Vision path passes `latent_input` into the transformer `detr_vae.py:131`; **the state-only path at `:136` does NOT** — it calls `self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)` with 4 args, dropping `latent_input` entirely | latent **is** injected as a token | **ADAPTED (deliberate divergence)** | The reference's state-only branch is **unreachable dead code** — `build()` always constructs backbones `detr_vae.py:235-237`, so `backbones is None` never holds. Copying it would build an "ACT" with no CVAE conditioning at all. | Catastrophic and silent: the model would be chunked BC with an unused encoder, and RQ2/RQ3 would compare ACT against itself. | **YES** |
| 6 | CVAE encoder inputs | `cat([cls_embed, qpos_embed, action_embed], axis=1)` → `(bs, k+2, hidden)`, `detr_vae.py:95`. Paper §C: "k+2 length input". | same order: `[CLS, state, action chunk]` | **IDENTICAL** | — | Wrong order silently changes what the CLS token summarises. | **YES** |
| 7 | CVAE encoder positional encoding | **fixed sinusoidal**, `register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))` `detr_vae.py:72`, table built `:23-31`, applied `.clone().detach()` `:101` | fixed sinusoidal, same table | **IDENTICAL** | It is a buffer, not a parameter — not learned. | A learned table here trains fine on 10 episodes and changes the latent's meaning. | **YES** |
| 8 | CVAE encoder padding mask | `is_pad` prepended with 2 `False` so CLS and qpos are never masked, `detr_vae.py:98-99` | same | **IDENTICAL** | — | Masking CLS destroys the latent; masking nothing lets padded actions into the latent. | **YES** |
| 9 | Latent head | `encoder_output[0]` = **CLS token only** `detr_vae.py:105`; `latent_proj: Linear(hidden, latent_dim*2)` `:71`; split into `mu`, `logvar` `:107-108` | same | **IDENTICAL** | — | Taking the wrong token, or the mean of all tokens, still trains. | **YES** |
| 10 | Latent dimension | **32**, `detr_vae.py:67`, with the comment `# TODO tune` | 32 | **IDENTICAL** | Carried over unchanged; the reference itself flags it as untuned. | Low. Report it as the reference's untuned value, not as ours. | **YES** |
| 11 | Reparametrisation | `std = logvar.div(2).exp(); mu + std*eps`, `detr_vae.py:17-20` | same | **IDENTICAL** | — | A missing `div(2)` still trains. | **YES** |
| 12 | Inference-time latent | `latent_sample = torch.zeros([bs, latent_dim])` then through `latent_out_proj`, `detr_vae.py:113-114`; encoder not run (`mu = logvar = None` `:112`). Paper §B: "we set z to be the mean of the prior distribution i.e. zero" | same: zeros through `latent_out_proj` | **IDENTICAL** | — | **Gate-blind by construction**: the gate only measures training loss, where the encoder IS run. A broken inference path scores perfectly at the gate and fails every rollout. | **YES** |
| 13 | CVAE encoder at inference | discarded; paper §B: "discarded at test time" | discarded | **IDENTICAL** | — | Using it at test time leaks the ground-truth action chunk into the prediction — an oracle. | **YES** |
| 14 | Decoder queries | `query_embed = nn.Embedding(num_queries, hidden_dim)` `detr_vae.py:54` — **learned**. Paper §C says "the input sequence is a **fixed** position embedding". **CODE AND PAPER DISAGREE.** | **learned** (`nn.Embedding`), following the code | **IDENTICAL to code, DIVERGENT from paper prose** | The code is the artefact that produced the published results; the prose is a description of it. We follow the code and disclose the disagreement. | Low functional risk; high write-up risk if we cite the paper's wording while running the code's behaviour. | **YES** |
| 15 | Decoder input | `tgt = torch.zeros_like(query_embed)` `transformer.py:72`; queries enter as `query_pos` `:75` | same | **IDENTICAL** | — | Feeding queries as `tgt` instead of `query_pos` trains but is a different model. | **YES** |
| 16 | `num_queries` = chunk size | `'num_queries': args['chunk_size']` `imitate_episodes.py:58` | same: one query per predicted timestep | **IDENTICAL** | This is the mechanism by which the chunk is produced. | — | no |
| 17 | Action head | `action_head = nn.Linear(hidden_dim, state_dim)` `detr_vae.py:52`, `state_dim = 14` `:230` (hardcoded, with `# TODO hardcode`) | `nn.Linear(hidden_dim, 22)` | **ADAPTED** | Our action is 22-D (`g1_data/spec.py`), not 14-D. Pure dimensionality. | Low. | no |
| 18 | **Which decoder layer is read** | `hs = self.transformer(...)[0]` `detr_vae.py:131` and `:136`. `transformer.py:76` returns `(num_dec_layers, bs, num_queries, d)`, so `[0]` is the **FIRST** decoder layer. **MEASURED** on this commit: output shape `(7, 3, 11, 16)`; with the head reading `hs[0]`, decoder layer 1 gets grad L1 `7.41e-05` and **layers 2-7 get exactly `0.0`**. `dec_layers = 7` `imitate_episodes.py:55` | **RESOLVED: `DEC_LAYERS = 1`** (`act.py`) — the head reads `hs[0]`, and one layer is bit-identical to the reference's seven; see `NOTES.md` 2026-09-21 and CLAUDE.md §8 2026-09-23 (audit) | **ADAPTED** (was marked UNRESOLVED until decided 2026-09-21) | The reference builds 7 decoder layers and trains 1. Replicating it faithfully means shipping 6 dead layers; "fixing" it means not running ACT. | Either direction is defensible and must be stated. Silently picking one and reporting "7 decoder layers as in ACT" would be false. | **YES** |
| 19 | Transformer hidden dim | `--hidden_dim 512` `README.md:76`; default 256 `main.py:41`; paper Table III "hidden dimension 512" | 512 | **IDENTICAL** | The README command is the published configuration; the argparse default is overridden. | Low. | no |
| 20 | Heads | `nheads = 8` `imitate_episodes.py:56`; paper Table III "# heads 8" | 8 | **IDENTICAL** | — | Low. | no |
| 21 | Encoder layers | `enc_layers = 4` `imitate_episodes.py:54`; paper Table III "# encoder layers 4". **Shared** with the CVAE encoder — `build_encoder` reads the same `args.enc_layers` `detr_vae.py:217` with the comment `# TODO shared with VAE decoder` | 4 for both | **IDENTICAL** | Two separate `TransformerEncoder` instances, same depth. | Low, but note they are separate modules with separate weights. | no |
| 22 | Decoder layers | `dec_layers = 7` `imitate_episodes.py:55`; paper Table III "# decoder layers 7"; default 6 `main.py:37` | **RESOLVED: `DEC_LAYERS = 1`** — see row 18, `NOTES.md` 2026-09-21 and CLAUDE.md §8 2026-09-23 (audit) | **ADAPTED** (was marked UNRESOLVED until decided 2026-09-21) | — | — | **YES** |
| 23 | Feedforward dim | `--dim_feedforward 3200` `README.md:76`; default 2048 `main.py:39`; paper Table III "feedforward dimension 3200" | 3200 | **IDENTICAL** | — | Low. | no |
| 24 | Dropout | `0.1`, `main.py:43` (not overridden by the README command); paper Table III "dropout 0.1" | 0.1 | **IDENTICAL** | — | **Gate-blind**: the overfit-10 gate runs *without* regularisation by design, so a wrong dropout is invisible there. | **YES** |
| 25 | Activation | `"relu"`, hardcoded `detr_vae.py:219`; resolved `transformer.py:306-314` | ReLU | **IDENTICAL** | — | Low. | no |
| 26 | Normalisation placement | `normalize_before = args.pre_norm` `detr_vae.py:218`; `--pre_norm` is `store_true` `main.py:49` and **is not passed** in `README.md:76` → **post-norm**. Encoder norm is `None` when post-norm `transformer.py:30`; decoder always has a final `LayerNorm` `:35` | post-norm, decoder final LayerNorm | **IDENTICAL** | — | Pre-norm trains more stably and would *help* at the gate while being a different architecture. | **YES** |
| 27 | Weight init | `xavier_uniform_` on every parameter with `dim() > 1` of the **policy transformer only**, `transformer.py:44-47` (`_reset_parameters` is a method of `Transformer`, called at `:39`). The **CVAE encoder is NOT re-initialised**: `build_encoder` `detr_vae.py:212-226` never calls it, and `TransformerEncoder` deep-copies ONE layer `transformer.py:83, 289-290`, so its 4 layers start **identical**, with PyTorch's default init. Measured 2026-09-23: reference CVAE-encoder layers 0 and 3 identical, linear1 std 0.02553 (torch default) | xavier on the policy transformer **and on the CVAE encoder** (`act.py` `_reset_parameters`): 4 distinct layers, linear1 std 0.02321 | **ADAPTED** (was marked IDENTICAL until the 2026-09-23 audit, `docs/ACT_AUDIT_REPORT.md` R2) | Kept as built: it changes the starting point, not the function class, and the passing overfit-10 gate used this init. Changing it now would be a model change made for fidelity alone, with no measurement that it matters. Disclosed in `act.py` `_reset_parameters`. | Different init still overfits 10 episodes; it can change what the CVAE latent learns early, which the gate cannot see. | **YES** |
| 28 | Reconstruction loss | **L1**: `F.l1_loss(actions, a_hat, reduction='none')` `policy.py:30`. Paper Algorithm 1 line 9 says `Lreconst = MSE(...)`; paper §C says "We use L1 loss for reconstruction instead of the more common L2 loss". **PAPER'S OWN PSEUDOCODE CONTRADICTS ITS OWN PROSE AND THE CODE.** | L1 | **IDENTICAL to code and to paper prose; the Algorithm 1 box is wrong** | Code and prose agree on L1; two sources against one. Our Stage 2 loss is already L1 for this reason. | Using MSE would make the reconstruction term incomparable with BC and chunked BC and break the whole ladder. | no |
| 29 | Reconstruction reduction | `l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()` `policy.py:31` — zeroes padded entries but **divides by the full element count**, padding included | `train.masked_l1`: divide by the count of **contributing** elements | **ADAPTED (deliberate divergence)** | The reference's mean makes the loss depend on how much padding a batch happened to contain, so the same model scores differently at different K. Our ladder compares K=1 against K>1 directly, so the reduction must be K-invariant or the comparison measures padding. Also our dim mask excludes 6 constant dims the reference has no analogue for. | A scale factor on the loss looks like a hyperparameter change, not a bug, and it silently biases every cross-K comparison. | **YES** |
| 30 | Action-dim masking | none — all 14 dims scored | 16 of 22 scored via `spec.ACTION_MASK` | **ADAPTED** | 6 of our 22 action dims are constant by measurement (2026-09-10 audit); scoring them rewards reproducing a constant and dilutes every average. The reference has no constant dims. | Inflates or deflates all reported losses uniformly. | **YES** |
| 31 | KL term | `klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())`; `total_kld = klds.sum(1).mean(0, True)` `policy.py:79-80`. Summed over latent dims, averaged over batch. `loss_dict['kl'] = total_kld[0]` `:33` | same | **IDENTICAL** | — | `mean_kld` and `dim_wise_kld` are also returned `:84` and are **not** used; picking the wrong one rescales the KL by 32×. | **YES** |
| 32 | KL weight | `kl_weight`, `policy.py:15`; total loss `l1 + kl * kl_weight` `:34`; `--kl_weight 10` `README.md:76`; paper Table III "beta 10"; paper §B calls it β | β = 10, name `kl_weight` | **IDENTICAL** | — | **Gate-blind**: on 10 episodes the KL term is small and a wrong β still overfits. It governs generalisation, which the gate does not test. | **YES** |
| 33 | `is_pad_head` | `nn.Linear(hidden_dim, 1)` `detr_vae.py:53`, computed `:138`, returned `:139`, destructured `policy.py:27` — and **never used in any loss** | **DROPPED** | **DROPPED** | Dead output in the reference. Carrying it would add parameters that receive no gradient and imply a padding-prediction objective that does not exist. | None, provided we state that we checked and it is unused rather than that ACT has no such head. | no |
| 34 | Action chunk truncation | `actions = actions[:, :self.model.num_queries]` `policy.py:24`; same for `is_pad` `:25` | our loader emits exactly K | **ADAPTED** | Our `ChunkDataset` produces exactly K with its own mask, so truncation is unnecessary. Same effect, earlier. | Low. | no |
| 35 | Optimizer | `AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)` `main.py:87-88` | AdamW | **IDENTICAL** | — | Low. | no |
| 36 | Learning rate | `--lr 1e-5` `README.md:77`; paper Table III "learning rate 1e-5" | **RESOLVED: `LR = 1e-5`** (`act.py`), the reference's value, stated by ACT and never inherited — see `NOTES.md` 2026-09-21 and CLAUDE.md §9 TR28 | **IDENTICAL** (was marked ADAPTED until 2026-09-21) | ACT's 1e-5 is tuned for a 4-camera ResNet model at batch 8 for 2000 epochs. Ours is a much smaller state-only model, and 1e-5 was validated on it: at 1e-3 the latent collapsed (see `NOTES.md` 2026-09-21). Optimizer, weight decay and schedule also match (rows 35, 38, 39). Must be **identical between ACT and ACT-LSTM**. | A different lr between the two variants makes RQ3 meaningless. | **YES** |
| 37 | Backbone learning rate | separate param group at `lr_backbone = 1e-5` `imitate_episodes.py:51`, group built `main.py:80-86` | **DROPPED** | **DROPPED** | Follows row 1: no backbone, so no second param group. | None. | no |
| 38 | Weight decay | `1e-4`, `main.py:17`, not overridden | 1e-4 | **IDENTICAL** | — | **Gate-blind**: the gate runs unregularised. | **YES** |
| 39 | LR schedule | none. `--lr_drop 200` exists `main.py:19` but is marked `# not used` and no scheduler is constructed anywhere in `main.py` or `imitate_episodes.py` | none | **IDENTICAL** | — | Adding a schedule would help at the gate and change the comparison. | **YES** |
| 40 | Gradient clipping | **none applied**. `--clip_max_norm 0.1` exists `main.py:20` marked `# not used`; no `clip_grad_norm_` call anywhere in the repo | global-norm clip **1.0 for every model**: one constant `models.LADDER_GRAD_CLIP`, declared by each model config (`BCConfig.grad_clip`, `ACTConfig.grad_clip`), checked against the TrainConfig and refused if it differs from the ladder value (`train.assert_optimizer_source`); written into every run's metadata as `GRAD_CLIP_DISCLOSURE`. `TrainConfig.grad_clip` has **no default** since 2026-09-23 (it used to default to 1.0 - TR28's third occurrence) | **ADAPTED (deliberate divergence, disclosed)** | Measured (`docs/ACT_AUDIT_REPORT.md` R1): norm ~678 at init, median 1.35 at the converged ACT gate weights, above threshold on 19/20 batches - active throughout, not a safety net. Kept because ACT's passing gate used it, recurrent models are where clipping matters most, and a clip differing between ACT and ACT-LSTM would sit inside the headline comparison. | Clipping one variant and not the other confounds RQ3 directly - now structurally impossible. | **YES** |
| 41 | Batch size | `--batch_size 8` `README.md:76`; paper Table III "batch size 8" | **decision required** (our BC used 256) | **ADAPTED** | Theirs is sized by 4×480×640 images in VRAM; ours has no images and a 4 GB card. Must be identical across variants. | Low for correctness, high for comparability. | no |
| 42 | Epochs | `--num_epochs 2000` `README.md:77` | **decision required** | **ADAPTED** | Their dataset is 50 episodes of 400 steps. Must be identical across variants, or use identical early-stopping. | Training one variant longer is the easiest way to fake an RQ3 result. | **YES** |
| 43 | Seeding | `set_seed(1)` at `imitate_episodes.py:25`, and `set_seed(1000)` in `eval_bc` `:152`. `set_seed` sets `torch.manual_seed` and `np.random.seed` only `utils.py:187-189` — **no `random.seed`, no CUDA seeding, no `use_deterministic_algorithms`** | our `set_determinism` + `seeded_build` | **ADAPTED** | Stage 2 measured that seeding after model construction leaves weight init unseeded (6.0e-3 divergence). Our seeding is strictly stronger. | Non-reproducible runs. | no |
| 44 | Model selection | best checkpoint by **validation** loss `imitate_episodes.py:352-354`, where that loss is `forward_pass` with the action chunk supplied (`:346`) → `policy.py:23-34`: **L1 + β·KL with the CVAE encoder reading the target** (posterior z), in eval mode | best checkpoint by **deployment reconstruction**: `train.score_deployment` on the validation set (training set if none) — masked L1 only, **z = 0**, encoder never run, `train.py` `evaluate` / `SELECTION_CRITERION` | **ADAPTED** (was marked IDENTICAL until the 2026-09-23 audit, `docs/ACT_AUDIT_REPORT.md` R5) | Selection should score the function that is deployed. The reference's criterion lets the encoder see the target it is scored on and adds a KL term BC has no counterpart for, so it would not be comparable across BC, ACT and ACT-LSTM. Measured on the converged gate weights: the two paths differ by 1.0e-7, so on scripted data the choice is invisible - it will not be on piloted data if the latent is active (O32). | Selecting on train loss inflates the reported result; selecting on a target-reading loss rewards the oracle path. | **YES** |
| 45 | Normalisation stats | mean/std over **all** episodes before the split `utils.py:120` called after `load_data` computes indices `:115-117`; std clipped to `[1e-2, inf)` `:97,:102` | training split only, with seed provenance (`g1_data/dataset.py`) | **ADAPTED (deliberate divergence)** | The reference fits normalisation on train **and** val, leaking validation statistics into the normaliser. Our Stage 1 A3 work exists precisely to prevent that. We are stricter, and say so. | Leaks the validation set; invisible in every metric. | **YES** |
| 46 | Train/val split | random permutation, ratio 0.8 `utils.py:114-117`, **unseeded at that point** | binned-by-spawn stratified split, fixed, seed-provenanced | **ADAPTED** | Our Q6/Objective-4 design requires stratification by box position and a provable held-out patch. A random split cannot support the generalisation claim. | Validation measures a corner of the task. | no |
| 47 | Sampling unit | one random start timestep per episode per `__getitem__` `utils.py:35`; `__len__` = number of **episodes** `:21` | explicit `(episode, start_tick)` index over **every** tick | **ADAPTED** | The reference sees one random window per episode per epoch, so "epoch" means something different. Our deterministic index is what makes overfit-10 reproducible. | Changes what an epoch is; affects every epoch-count comparison with the paper. | no |
| 48 | Temporal ensembling | `imitate_episodes.py:250-259`. `k = 0.01` `:255`; `exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))` `:256`; normalised `:257`; weighted sum `:259`. Buffer `all_time_actions` `:219`. Enabled only by `--temporal_agg` (`store_true` `:433`) → **OFF by default**; when on, `query_frequency = 1` `:193` | **DROPPED in Stage 3; Stage 4 decision** | **DROPPED (this stage)** | Ensembling is an ACT mechanism, not a chunking one. Stage 3 isolates chunking; folding ensembling in would mean measuring chunking *and* ensembling with no way to separate them. | If ACT gets ensembling and chunked BC does not, RQ2 measures both. | no |
| 49 | Ensembling weight `m` | code: `k = 0.01` `imitate_episodes.py:255` (the variable is named `k`, colliding with the paper's `k` for chunk size). Paper: `w_i = exp(-m*i)`, "smaller m means faster incorporation" (§IV-A). **Paper gives no numeric value for m: NOT FOUND IN SOURCE.** | if used: 0.01, cited to code | — | The only numeric value in either source is the code's. | Quoting a value "from the paper" would be citing something that is not there. | no |
| 50 | Query frequency (no ensembling) | `query_frequency = num_queries` `imitate_episodes.py:191`; act open-loop for k steps, `raw_action = all_actions[:, t % query_frequency]` `:261` | Stage 3 uses `first_action` (re-plan every tick) | **ADAPTED** | Re-planning every tick without ensembling is the honest no-ensembling baseline. | Open-loop-for-K vs re-plan-every-tick is a large behavioural difference at rollout, invisible in training loss. | **YES** |
| 51 | Control rate | `DT = 0.02` `constants.py:36` → **50 Hz** | 25 Hz (D4) | **ADAPTED** | Fixed by our locomotion policy's native rate. | See §6: it changes what K=100 *means*. | no |
| 52 | Chunk size | `--chunk_size 100` `README.md:76`; paper Table III "chunk size 100" | see §6 | **UNRESOLVED** | — | — | no |
| 53 | Observation window | **does not exist.** Observation is a single timestep: `qpos = root['/observations/qpos'][start_ts]` `utils.py:37`; `forward(self, qpos, ...)` takes `qpos: batch, qpos_dim` `detr_vae.py:80` | **W_o = 12** (0.48 s at 25 Hz), the same for every model (BC, chunked BC, ACT-LSTM); a starting choice, not tuned — see CLAUDE.md §8 2026-09-25, 2026-09-27 | **ADAPTED** (reference uses 1; was the Part D2 recommendation of W_o = 1 until 2026-09-25) | Giving ACT and ACT-LSTM the same window leaves `use_lstm` the only difference between them, so the gap measures recurrence rather than extra history, with three models instead of a fourth windowed-ACT control. | ACT already sees 12 steps of history, so the ACT-LSTM gap measures recurrence over a window ACT also has, and will likely be smaller than against a W_o = 1 ACT. | **YES** |
| 54 | Recurrence / LSTM | **does not exist in ACT.** No `nn.LSTM`, `nn.GRU` or `nn.RNN` anywhere in the clone | ACT-LSTM adds it behind `use_lstm` | **ADDITION** | This is our contribution (D8) and the subject of RQ3. | See Part D3 for everything that could differ other than recurrence. | **YES** |
| 55 | `qvel` | read from the dataset `utils.py:38` and **never used** — not returned by `__getitem__` `:76` | not used | **IDENTICAL** | — | None. | no |
| 56 | Shared backbone across cameras | `self.backbones[0](image[:, cam_id])  # HARDCODED` `detr_vae.py:121` — one backbone for all cameras, and `build()` only ever appends one `:235-237` | n/a | **DROPPED** | Follows row 1. Recorded because the paper's Figure 11 shows a ResNet18 per camera, which the code does not do. | None for us. | no |

---

## 3. Why the vision pipeline is dropped — the Chapter 3 paragraph

ACT was designed for a real bimanual manipulator learning from four 480×640 RGB streams,
and roughly all of its parameter count and most of its engineering sits in converting
those streams into tokens a transformer can attend over. That machinery answers a
question this thesis does not ask. Our policy consumes privileged MuJoCo state, by
design and by the constraint recorded in the project's scope: the work is simulation-only
and the 47-D state vector is read directly from the physics engine, so the mapping from
pixels to state — which is what the ResNet-18 backbones, the 2-D sinusoidal position
embeddings and the 1200-token image sequence exist to learn — is already exact and free.
Retaining them would mean rendering camera images from a simulator whose ground-truth
state we already hold, in order to recover an approximation of that same state, and then
measuring how well a recurrent module helps a policy consume it. The perception error so
introduced would sit between the architecture and the research question as an
uncontrolled variable, and it would be the dominant one: differences in how well two
variants happened to train their visual encoders would appear in the results as
differences in the effect of recurrence.

The research questions are about temporal structure — whether predicting a chunk of
actions helps (RQ2), and whether recurrent state helps beyond that (RQ3). Neither is a
question about perception, and neither becomes better-posed by adding a perception stage
that the simulator makes unnecessary. Dropping vision therefore removes a confound rather
than a capability: every remaining component of ACT — the CVAE encoder, the style
variable, the transformer encoder and decoder, the learned queries, the chunked action
head and the L1 + βKL objective — is retained and is the part of the architecture the
research questions concern. What is lost is external validity with respect to
image-conditioned policies, and that limitation belongs in the limitations section
alongside the existing simulation-only and weld-grasp constraints, not in the results.

The cost of this decision is concentrated in one place, and it should be stated plainly:
with images removed, ACT's transformer encoder is left attending over two tokens (§4).
That is a real reduction in the encoder's role, and it is a consequence of the research
design rather than an oversight.

---

## 4. Forced structural choice: the encoder token sequence

Removing image tokens leaves the transformer encoder with whatever the non-image part of
`transformer.py:62-63` provides. The reference stacks exactly two: `latent_input` and
`proprio_input`. Four options, none of them free.

**Option A — two tokens: `[latent, proprio]`.**
Exactly the reference minus image tokens. The reference's own
`additional_pos_embed = nn.Embedding(2, hidden_dim)` (`detr_vae.py:76`) is sized precisely
for these two, which is direct evidence that two non-image tokens is ACT's design for the
non-image part rather than our invention.
*Against:* self-attention over two tokens is close to a no-op, and four encoder layers of
width 512 on a two-token sequence is a large amount of machinery doing very little. A
reviewer will notice.

**Option B — drop the transformer encoder entirely; the decoder cross-attends directly to
`[latent, proprio]`.**
Honest about Option A's weakness: if there is nothing to synthesise across, do not have a
synthesiser.
*Against:* it is a structural deletion, not an adaptation, and it makes "we implemented
ACT" materially harder to defend. It also changes the parameter count by more than any
other single choice here.

**Option C — one token per semantic group of the 47-D state** (box pose, base pose, left
palm, right palm, grippers, left arm, right arm, waist → 8 tokens, plus latent = 9).
Gives the encoder a real synthesis job, structurally analogous to synthesising across
camera viewpoints, which is what the paper says the encoder is for (§C: "the transformer
encoder synthesizes information from different camera viewpoints, the joint positions,
and the style variable").
*Against:* the grouping is our invention. It must be justified, held identical across
ACT and ACT-LSTM, and disclosed as a deviation. It also changes parameter count relative
to the reference and adds a design axis nobody asked for.

**Option D — one token per timestep of the observation window.**
The natural way to give a transformer a window.
*Against:* it entangles W_o with the token count, and it is precisely the mechanism that
could substitute for the LSTM. See D2.

**RECOMMENDATION: Option A.** It is the minimal defensible deviation — we removed image
tokens and changed nothing else — and it is the one supported by the reference's own
code rather than by our judgement. Its weakness (a two-token encoder) is a *disclosable
fact about what dropping vision costs*, not a bug, and it is better to report that
honestly than to invent structure to fill the gap and thereby stop running ACT. If a
reviewer later presses on the two-token encoder, Option C is the prepared fallback and
the argument for it is already written above; it should be adopted as a deliberate
documented change, never quietly.

Option A carries one obligation: **do not copy the reference's state-only branch**
(`detr_vae.py:132-136`). It is unreachable dead code and it drops the latent entirely
(row 5). Taking Option A means taking the *vision* branch's wiring
(`detr_vae.py:131` → `transformer.py:59-63`) and deleting only the image tokens from it.

---

## 5. Forced structural choice: how many decoder layers

Row 18, measured on this commit: `detr_vae.py:131` reads `hs[0]`, which is the **first**
decoder layer's output, while `dec_layers = 7`. Layers 2–7 run and receive exactly zero
gradient.

**Option 1 — replicate faithfully: build 7, read layer 1.**
Bit-for-bit what the published code does.
*Against:* ~6/7 of the decoder is dead weight, VRAM and time are spent on it, and the
thesis would report "7 decoder layers, as in ACT" for a model that trains one.

**Option 2 — build 1 decoder layer, read it.**
Functionally identical to the reference, honest about parameter count.
*Against:* the number in our methods table then differs from the number in the ACT paper's
Table III, and every reader will assume we got it wrong.

**Option 3 — read `hs[-1]`, keep 7 layers.**
What the architecture was presumably meant to do.
*Against:* this is **not ACT**. It is a deeper, different model, and any RQ2/RQ3 result
obtained with it cannot be attributed to ACT's design.

**RECOMMENDATION: Option 2, with Option 1 available as a check.** Build one decoder layer,
state in Chapter 3 that the reference constructs seven and trains one, cite the measured
gradient result, and note that our single layer is functionally equivalent to the
published model. This is honest in both directions — it neither claims seven layers we do
not use nor silently improves on the reference. If session 3 wants to verify equivalence,
Option 1 with `hs[0]` should produce the same losses to within initialisation noise, and
that is a cheap experiment worth running once.

**Whatever is chosen must be identical between ACT and ACT-LSTM.**

---

## 6. Chunk size and horizon

| | reference | ours |
|---|---|---|
| control rate | **50 Hz** (`DT = 0.02`, `constants.py:36`) | **25 Hz** (D4) |
| chunk size | **100** (`README.md:76`, paper Table III) | see below |
| horizon | **2.0 s** | K=100 → **4.0 s**; K=50 → 2.0 s |
| episode length | 400 steps = 8.0 s (`constants.py:9`) | 694–846 ticks = 27.8–33.8 s |
| chunk as fraction of episode | 100/400 = **25%** | K=100 → ~13%; K=50 → ~6.6% |

**Matching the number (K=100)** keeps every hyperparameter table row identical to the
paper and makes "chunk size 100, as in ACT" literally true, but it doubles the physical
horizon to 4.0 s and still covers a much smaller fraction of our far longer episodes.

**Matching the horizon (K=50)** predicts the same 2.0 s of future the reference does,
which is the quantity that plausibly transfers — a chunk is a unit of intent, and intent
has a duration in seconds, not in ticks — but it means reporting a number that differs
from the paper's and defending the conversion.

Stage 3 used K=100 as a provisional plumbing value on the grounds that it is ACT's
published number and nothing more. **Neither should be settled on scripted data.** The
right K depends on how long a piloted operator's intent stays coherent, which a scripted
demonstrator that never changes its mind cannot measure. What this table adds is that the
choice is not one number but two different claims, and the thesis must say which one it
is making.

---

## 7. Paper / code disagreements, collected

| topic | paper | code | follow |
|---|---|---|---|
| reconstruction loss | Algorithm 1 line 9: `MSE` | `F.l1_loss` `policy.py:30` | **code** — paper §C prose also says L1, so the Algorithm box is the outlier |
| decoder queries | §C: "a **fixed** position embedding" | `nn.Embedding` `detr_vae.py:54` — learned | **code** |
| decoder depth | Table III: 7 decoder layers | builds 7, trains 1 (`detr_vae.py:131`, measured) | see §5 |
| per-camera backbone | Fig. 11 shows ResNet18 per camera | one shared backbone `detr_vae.py:121,235-237` | **code** (moot for us) |
| ensembling weight `m` | `exp(-m*i)`, no numeric value given | `k = 0.01` `imitate_episodes.py:255` | **code** — the paper's value is NOT FOUND IN SOURCE |

---

## 8. Rows session 3 must check hardest

Every row marked gate-blind, in priority order — a wrong answer in any of these produces
a passing overfit-10 gate and a wrong thesis:

1. **Row 5** — latent actually reaches the policy. If not, "ACT" is chunked BC and RQ2/RQ3
   compare a model with itself.
2. **Row 12/13** — inference uses `z = 0` and does **not** run the CVAE encoder. The gate
   never exercises this path.
3. **Row 29/30** — loss reduction and dim masking. A uniform scale factor on the loss
   looks like a hyperparameter, not a bug.
4. **Row 31/32** — KL formulation and β. `mean_kld` instead of `total_kld` rescales by 32×.
5. **Row 18/22** — decoder depth, and which layer is read.
6. **Row 53/54** — W_o and the LSTM: the RQ3 confound.
7. **Rows 36, 40, 42** — lr, clipping, epochs identical across ACT and ACT-LSTM.
8. **Rows 6-9** — CVAE encoder input order, positional table, mask, and CLS extraction.
9. **Row 45** — normalisation fitted on the training split only.
10. **Rows 24, 26, 27, 38, 39** — dropout, pre/post-norm, init, weight decay, schedule.
