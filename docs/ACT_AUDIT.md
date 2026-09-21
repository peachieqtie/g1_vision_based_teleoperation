# ACT implementation audit checklist

**For session 3, or anyone auditing the ACT / ACT-LSTM implementation.**

You are trying to catch the implementer out. Assume the implementation is wrong until a
check says otherwise, and assume the implementer believed it was right — the failures
worth finding here do not crash and do not look wrong. Most of them produce a model that
passes an overfit-10 gate and a thesis result that is false.

**Do not read `docs/ACT_CORRESPONDENCE.md` first.** Answer from the reference and the
implementation, then compare. If you read the reasoning first you will find what it tells
you to find.

## Ground rules

- Every answer must be a **file and line** in `reference/act/` (commit
  `742c753c0d4a5d87076c8f69e5628c79a8cc5488`) or in our tree, or a **measured number**.
  "It looks right" is not an answer.
- If the clone is missing, re-create it at that exact hash. Line numbers below are
  meaningless against any other commit.
- Where a check says **MEASURE**, run it. Do not reason about it. Several items below
  were wrong when reasoned about and right when measured.
- The paper is arXiv **2304.13705v1**. Where paper and code disagree, the code is the
  artefact that produced the published results — but **record both**.

---

## A. The CVAE is actually present and actually conditional

These are the checks that separate ACT from chunked BC. If any fails, RQ2 and RQ3 are
comparing a model against itself.

**A1.** Does the latent reach the policy at training time? Trace it. In the reference the
vision path passes `latent_input` into the transformer (`detr_vae.py:131`); the
state-only path (`detr_vae.py:136`) **does not**. Which one does our code resemble?
*Verifiable answer:* name the line in our code where the latent becomes a token in the
encoder's input sequence.

**A2. MEASURE.** Zero the latent's contribution and confirm the training loss gets worse.
Run one short training with the latent path intact and one with `latent_input` forced to
zeros at training time. If the two losses are indistinguishable, the CVAE is decorative.
*Verifiable answer:* two numbers and their difference.

**A3. MEASURE.** Is `mu` non-constant across a batch? Feed two different action chunks
with the same state and confirm the encoder produces different `mu`. If `mu` does not
depend on the action chunk, the encoder is not encoding anything.
*Verifiable answer:* `max |mu(batch_1) - mu(batch_2)|` for two deliberately different
chunks.

**A4.** At inference, is the CVAE encoder run at all? In the reference it is not:
`mu = logvar = None` and `latent_sample = torch.zeros(...)` (`detr_vae.py:112-113`); the
paper says z is set to "the mean of the prior distribution i.e. zero" (§IV-B). Does our
inference path ever touch the ground-truth action chunk?
*Verifiable answer:* the line that constructs the inference latent, and a grep showing the
action chunk is not an argument on that path.

**A5.** *The oracle test.* If the encoder were accidentally run at inference, the model
would see the actions it is asked to predict and would score implausibly well. Is there a
test that would fail if that happened?
*Verifiable answer:* name the test, or report that none exists.

**A6.** What is the latent dimension, and where is it set? Reference: `32`
(`detr_vae.py:67`, annotated `# TODO tune`). Does our value match, and is it documented as
the reference's untuned value rather than as our choice?

**A7.** Is the latent sampled with reparametrisation at training and taken deterministically
at inference? Reference: `std = logvar.div(2).exp(); return mu + std*eps`
(`detr_vae.py:17-20`). Check for a missing `div(2)` — it trains fine and halves the
effective KL scale.

---

## B. The CVAE encoder's inputs

**B1.** What is the input sequence to the CVAE encoder, **in order**? Reference:
`cat([cls_embed, qpos_embed, action_embed], axis=1)` (`detr_vae.py:95`), giving length
`k+2` (paper §IV-C). Is our order the same?
*Verifiable answer:* the line, and the resulting sequence length as a function of K.

**B2.** Is the CVAE encoder's positional encoding **fixed sinusoidal** or learned?
Reference: a `register_buffer` (`detr_vae.py:72`), table constructed at `:23-31`, applied
with `.clone().detach()` at `:101` — a buffer, not a parameter.
*Verifiable answer:* is ours in `model.parameters()`? If yes, it is learned and wrong.
**This trains fine either way.**

**B3.** Is the positional table sized `1 + 1 + K`? If it is sized K, the CLS and state
tokens are getting the wrong positions and nothing will complain.

**B4.** Which token is read out of the encoder? Reference: `encoder_output[0]` — the CLS
token only (`detr_vae.py:105`). Check ours is not the mean over tokens, or the last token.
**Any of these trains.**

**B5.** Is the padding mask prepended with two `False` entries so CLS and state are never
masked? Reference: `detr_vae.py:98-99`. What happens in our code when K exceeds the
remaining episode length — are the padded action steps masked out of the latent?
*Verifiable answer:* construct a sample at the last tick of an episode and show which
encoder positions are masked.

---

## C. The decoder and the chunk

**C1.** Are the decoder queries learned or fixed? Reference code: `nn.Embedding`
(`detr_vae.py:54`) — **learned**. Reference paper §IV-C: "a **fixed** position embedding".
They disagree. Which did we implement, and is the disagreement disclosed?

**C2.** Is `num_queries == K`? Reference: `'num_queries': args['chunk_size']`
(`imitate_episodes.py:58`).

**C3.** Is the decoder's `tgt` zeros, with queries entering as `query_pos`? Reference:
`tgt = torch.zeros_like(query_embed)` (`transformer.py:72`), passed as `query_pos` at
`:75`. Feeding queries as `tgt` instead is a different model that trains.

**C4. MEASURE.** How many decoder layers receive gradient? Build the model, backward
through the action head, and count layers whose parameters have non-zero `.grad`.

On the reference this is **not** what its configuration suggests. `dec_layers = 7`
(`imitate_episodes.py:55`, paper Table III) but `detr_vae.py:131` reads `hs[0]`, and
`transformer.py:76` returns `(num_dec_layers, bs, num_queries, d)` — so `[0]` is the
**first** layer. Measured on this commit: layer 1 gets grad L1 `7.41e-05`, layers 2–7 get
exactly `0.0`.
*Verifiable answer:* a per-layer gradient table for our model, and a statement of which
option from `ACT_CORRESPONDENCE.md` §5 was taken and whether Chapter 3 says so.

**C5.** Does the action head map `hidden_dim → 22` (not 14)? Reference hardcodes
`state_dim = 14` (`detr_vae.py:230`).

**C6.** Is there an `is_pad_head`? The reference has one (`detr_vae.py:53`), computes it
(`:138`), returns it (`:139`) and **never uses it in any loss**. If ours has one, what
consumes it? If ours does not, is the omission recorded as "checked, unused in the
reference" rather than "ACT has no such head"?

---

## D. The objective

**D1.** Is the reconstruction loss L1? Reference code: `F.l1_loss(..., reduction='none')`
(`policy.py:30`). Reference paper §IV-C: "We use L1 loss ... instead of the more common L2
loss". Reference paper **Algorithm 1 line 9: `MSE`** — the paper contradicts itself. Two
sources against one; L1 is correct. Is ours L1?

**D2.** **How is the reconstruction term reduced?** Reference:
`(all_l1 * ~is_pad.unsqueeze(-1)).mean()` (`policy.py:31`) — zeroes padded entries and
then divides by the **full** element count, padding included. Ours divides by the count of
**contributing** elements (`g1_model/train.masked_l1`).
*Verifiable answer:* confirm ours is K-invariant. Construct the same model error at K=1
and K=100 and show the loss is equal. If it is not, every cross-K comparison in the thesis
is measuring padding. **This is gate-blind — a constant scale factor looks like a
hyperparameter.**

**D3.** Is the 22-dim action mask applied, and does the loss ignore the 6 constant dims?
*Verifiable answer:* a prediction wrong **only** on `spec.CONSTANT_ACTION_DIMS` must give
**exactly** `0.0`.

**D4.** Is the KL `total_kld`, not `mean_kld` or `dim_wise_kld`? Reference:
`klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())`, `total_kld = klds.sum(1).mean(0, True)`
(`policy.py:79-80`), consumed as `total_kld[0]` (`:33`). The other two are returned at
`:84` and unused. Picking `mean_kld` rescales the KL by the latent dimension — **32×** —
and still trains.
*Verifiable answer:* the line, plus a hand-computed KL for a known `mu`/`logvar` matching
our function's output.

**D5.** Is the total loss `L1 + β·KL`? Reference: `policy.py:34`. Is β = 10
(`README.md:76`, paper Table III "beta 10")? Is the variable name recorded as `kl_weight`
(`policy.py:15`)?
**Gate-blind:** on 10 episodes the KL term is small and a wrong β still overfits. β
governs generalisation, which the gate does not test.

**D6. MEASURE.** Does the KL term actually contribute? Log `l1` and `kl` separately for a
few epochs. If `kl` is ~0 from the first step the encoder has collapsed and ACT is
chunked BC with extra parameters; if it dominates, β is wrong.
*Verifiable answer:* both series for the first ~20 epochs.

---

## E. Training configuration

**E1.** Optimizer AdamW (`main.py:87`)? Weight decay `1e-4` (`main.py:17`, not overridden)?

**E2.** Is there a learning-rate schedule? Reference: **none**. `--lr_drop 200` exists
(`main.py:19`) marked `# not used`, and no scheduler is constructed anywhere.
*Verifiable answer:* grep our tree for a scheduler; if one exists, it is a deviation and
must be identical across variants.

**E3.** Is gradient clipping applied? Reference: **none applied** — `--clip_max_norm 0.1`
exists (`main.py:20`) marked `# not used` and there is no `clip_grad_norm_` call in the
repo. Ours defaults to `grad_clip = 1.0`.
*Verifiable answer:* the value used for **both** ACT and ACT-LSTM. Clipping the recurrent
variant and not the other confounds RQ3 directly.

**E4.** Is post-norm used (not pre-norm)? Reference: `--pre_norm` is `store_true`
(`main.py:49`) and is **not** passed in `README.md:76`. Pre-norm trains more stably, so
this is a change that *helps* at the gate while being a different architecture.

**E5.** Dropout `0.1` (`main.py:43`)? **Gate-blind:** the overfit-10 gate runs
unregularised by design, so a wrong dropout is invisible there.

**E6.** Xavier-uniform init on every parameter with `dim() > 1` (`transformer.py:44-47`)?

**E7.** Is model selection on **validation** loss? Reference:
`imitate_episodes.py:352-354`.

**E8.** Are normalisation statistics fitted on the **training split only**? The reference
fits on all episodes before the split (`utils.py:115-120`), leaking validation statistics
into the normaliser. We are stricter. Confirm `assert_norm_stats_match` is called and that
a val loader verifies against the **train** seeds.

---

## F. Inference and rollout

**F1.** With no temporal ensembling, what does the policy execute? Reference:
`query_frequency = num_queries` (`imitate_episodes.py:191`) — it acts **open-loop for K
steps**, indexing `all_actions[:, t % query_frequency]` (`:261`). Our Stage 3 rule is
`first_action` — re-plan every tick.
*Verifiable answer:* which one our rollout does, and where that is stated. **This is a
large behavioural difference that is completely invisible in training loss.**

**F2.** Is temporal ensembling present? It must **not** be in Stage 3. Reference:
`imitate_episodes.py:250-259`, enabled only by `--temporal_agg` (`store_true` `:433`) →
**off by default**; when on, `query_frequency = 1` (`:193`).
*Verifiable answer:* grep for the exponential weighting. If present, is it applied
identically to chunked BC, ACT and ACT-LSTM? If only some variants get it, RQ2 measures
chunking **and** ensembling.

**F3.** If ensembling is ever added, is the weight cited to the **code**? The only numeric
value in either source is `k = 0.01` (`imitate_episodes.py:255`, and note the variable is
named `k`, colliding with the paper's `k` for chunk size). The paper gives the form
`w_i = exp(-m*i)` but **no numeric m — NOT FOUND IN SOURCE**. Any value attributed to the
paper is fabricated.

**F4.** Is the exponential weighting normalised before the weighted sum? Reference:
`exp_weights = exp_weights / exp_weights.sum()` (`:257`) then `.sum(dim=0)` (`:259`).

---

## G. The RQ3 confound — ACT vs ACT-LSTM

The thesis claim is that a measured difference between ACT and ACT-LSTM is **caused by
recurrence**. Every item here is a way for that claim to be false.

**G1.** Are ACT and ACT-LSTM **one class behind a flag**? If there are two classes, the
difference includes every incidental implementation difference between them.
*Verifiable answer:* the class, the flag, and a test asserting that with the flag off the
module contains no recurrent parameters.

**G2.** What is `W_o` for each? ACT's observation is a **single timestep**
(`utils.py:37`, `detr_vae.py:80`) — there is no observation window in the reference. If
`W_o > 1`, the transformer already has history and the LSTM is no longer the only source
of it.
*Verifiable answer:* the W_o used by each variant. **They must be equal.** If `W_o > 1`,
Chapter 3 must state what is left for the LSTM to contribute.

**G3. MEASURE.** Parameter counts of both variants. The LSTM adds parameters, so any
improvement may be capacity rather than recurrence.
*Verifiable answer:* both counts, and whether a parameter-matched ACT control was run. If
not, that limitation must be stated.

**G4. MEASURE.** With `use_lstm=False`, are the weights **bit-identical** to an ACT built
without the flag existing? Constructing an LSTM consumes RNG and shifts the stream for
everything built after it, so the two "identical" models can start from different weights.
Stage 2 measured exactly this class of bug (seeding after construction; 6.0e-3 divergence).
*Verifiable answer:* a max-absolute-difference over all parameters.

**G5.** Is the LSTM hidden state **reset** at the same point for every episode, and never
carried across episode boundaries within a batch?
*Verifiable answer:* the reset call site, and a test that two orderings of the same
episodes give the same per-episode outputs. A leaking hidden state is an oracle across
episodes.

**G6.** Do both variants use the **same sampler and the same batch composition**? A
recurrent model often wants ordered/sequential batches while a feedforward one is shuffled.
If they differ, the data order differs and RQ3 measures that too.

**G7.** Do both use the same lr, batch size, epochs, clipping, dropout, weight decay and
early-stopping rule? Training one longer is the easiest way to manufacture an RQ3 result.
*Verifiable answer:* a side-by-side config diff that is empty except for `use_lstm`.

**G8.** Where does the LSTM sit, and does it change what the **CVAE encoder** sees? If
recurrent state feeds the encoder at training time, the latent changes too and the
difference is no longer attributable to recurrence alone.

**G9.** Does the LSTM add dropout of its own?

**G10.** Are both evaluated with the same rollout rule (F1) and the same ensembling
setting (F2)?

---

## H. Things that are true of the reference and easy to get wrong

**H1.** The CVAE encoder and the policy's transformer encoder are **separate modules with
separate weights**, both built with `args.enc_layers = 4` (`detr_vae.py:217`, annotated
`# TODO shared with VAE decoder`; `imitate_episodes.py:54`). Did we accidentally share one
module?

**H2.** `hidden_dim = 512` and `dim_feedforward = 3200` come from `README.md:76`, **not**
from the argparse defaults (256 and 2048, `main.py:41,39`). Quoting the defaults would be
quoting a configuration the paper never ran.

**H3.** `qvel` is read (`utils.py:38`) and never used (not returned at `:76`). If our
state vector includes velocities, that is a deviation — defensible, but it must be named.

**H4.** The reference samples **one random start timestep per episode per `__getitem__`**
(`utils.py:35`) and `__len__` is the number of **episodes** (`:21`). Our loader indexes
every tick. "Epoch" therefore means something different in the two codebases, and any
epoch-count comparison with the paper is invalid.

**H5.** Action std is clipped to `[1e-2, inf)` (`utils.py:97`). Our `NormStats` uses
`STD_FLOOR = 1e-6`. Different flooring changes the scale of low-variance dims.

**H6.** The reference's `set_seed` (`utils.py:187-189`) sets only `torch.manual_seed` and
`np.random.seed` — no `random.seed`, no CUDA seeding, no deterministic algorithms. Do not
copy it as a model of adequate seeding.

---

## I. Final questions

**I1.** Is there any number in our Chapter 3 methods table that cannot be traced to a line
in the clone, a page of the paper, or a measurement in our own repo? Name it.

**I2.** For every row in `ACT_CORRESPONDENCE.md` marked **ADAPTED** or **DROPPED**: is the
stated reason still true of the implementation as built?

**I3.** For every row marked **UNRESOLVED** (18, 22, 52, and the lr/batch/epoch rows): was
a decision made, is it recorded, and is it identical across ACT and ACT-LSTM?

**I4.** Does anything in the implementation cite the paper for a fact the paper does not
contain? The known trap is the temporal-ensembling `m` (F3).

**I5.** Run the overfit-10 gate for ACT and ACT-LSTM. Then ask: **which of the failures in
sections A–G would this gate have caught?** The expected answer is "almost none". If the
implementer's evidence of correctness is the gate, the audit is not finished.
