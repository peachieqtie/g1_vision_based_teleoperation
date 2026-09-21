"""ACT, state-only: the CVAE that chunked BC is missing.

WHAT THIS ADDS, AND ONLY THIS
-----------------------------
The ladder adds one thing per rung: BC -> chunked BC adds chunking, chunked BC ->
ACT adds the CVAE, ACT -> ACT-LSTM adds recurrence. So everything here that is
not the CVAE must be the same as chunked BC's setting: W_o = 1, K = 100, the same
loader, the same `train.masked_l1`, the same loop. What ACT contributes is the
style variable z and the transformer that consumes it.

THE REFERENCE, AND WHAT WAS DROPPED
-----------------------------------
Pinned at `tonyzhaozh/act` commit 742c753c0d4a5d87076c8f69e5628c79a8cc5488,
cloned (gitignored) to `reference/act/`. Paper arXiv 2304.13705v1. Every
architectural constant below is cited to a file and line in that clone;
`docs/ACT_CORRESPONDENCE.md` carries the row-by-row table and the reasons.

Vision is DROPPED: we are state-only, in simulation, with ground-truth state, and
the research question is recurrence, not perception (correspondence table §3).
That removes the ResNet backbones, the 2-D image position embeddings and the
~300-token-per-camera image sequence, and nothing else.

    !!! THE STATE-ONLY PATH IS BUILT FROM THE REFERENCE'S *VISION* PATH !!!

`DETRVAE.forward` has a `backbones is None` branch (`detr_vae.py:132-136`) that
looks like exactly what a state-only port wants. It is unreachable - `build()`
constructs backbones unconditionally (`detr_vae.py:235-237`) - and it calls the
transformer with FOUR arguments, never passing `latent_input`, where the vision
path passes seven (`detr_vae.py:131`). Copying it yields a model with no CVAE
conditioning at all: chunked BC with an unused encoder, which would pass every
gate we have and silently make RQ2/RQ3 a comparison of a model against itself.
This file therefore mirrors the VISION path (`detr_vae.py:116-131` ->
`transformer.py:59-63`) and deletes only the image tokens. `test_act.py`'s
latent-reaches-the-decoder test is the standing guard.

THE TRANSFORMER IS REIMPLEMENTED, NOT IMPORTED
----------------------------------------------
The clone is a read-only reference outside the tracked tree, so importing from it
would make this repo depend on an untracked directory. The layers below mirror
`transformer.py` with line citations, including the two details a stock
`nn.Transformer` does not reproduce: position embeddings are added to the
attention QUERY and KEY but never to the VALUE (`transformer.py:171-173`,
`:236-244`), and the decoder returns every layer's output so the head can select
one (`transformer.py:129-141`).
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from g1_data import spec

# ─── reference constants ──────────────────────────────────────────────────────
# Every value here is the reference's, cited. Values that are OURS are in
# ACTConfig and marked PROVISIONAL there.
REF_COMMIT = "742c753c0d4a5d87076c8f69e5628c79a8cc5488"
REF_PAPER = "arXiv:2304.13705v1"

#: reference/act/detr/models/detr_vae.py:67 — `self.latent_dim = 32  # TODO tune`.
#: Carried unchanged; the reference itself flags it as untuned.
LATENT_DIM: int = 32
#: reference/act/README.md:76 `--hidden_dim 512`; paper Table III "hidden dimension 512".
#: NOT main.py:41's default of 256, which the published command overrides.
HIDDEN_DIM: int = 512
#: reference/act/README.md:76 `--dim_feedforward 3200`; paper Table III.
DIM_FEEDFORWARD: int = 3200
#: reference/act/imitate_episodes.py:56 `nheads = 8`; paper Table III "# heads 8".
NHEADS: int = 8
#: reference/act/imitate_episodes.py:54 `enc_layers = 4`. Used for BOTH the CVAE
#: encoder and the policy's transformer encoder — `detr_vae.py:217` reads the same
#: `args.enc_layers` ("# TODO shared with VAE decoder"). Separate modules, same depth.
ENC_LAYERS: int = 4
#: reference/act/imitate_episodes.py:55 `dec_layers = 7`; paper Table III.
DEC_LAYERS: int = 7
#: reference/act/detr/main.py:43 `--dropout 0.1`, not overridden by README.md:76.
DROPOUT: float = 0.1
#: reference/act/detr/models/detr_vae.py:219 — hardcoded "relu".
ACTIVATION: str = "relu"
#: reference/act/detr/main.py:49 `--pre_norm` is store_true and is NOT passed in
#: README.md:76, so the published model is POST-norm.
PRE_NORM: bool = False
#: reference/act/README.md:76 `--kl_weight 10`; paper Table III "beta 10";
#: consumed as `policy.py:34` `l1 + kl * kl_weight`.
KL_WEIGHT: float = 10.0
#: reference/act/imitate_episodes.py:255 — `k = 0.01` inside the temporal-ensemble
#: block. The paper gives the FORM `w_i = exp(-m*i)` (§IV-A) but NO numeric value:
#: NOT FOUND IN SOURCE. Note the reference's variable is named `k`, colliding with
#: the paper's `k` for chunk size; renamed `m` here to match the paper's symbol.
TEMPORAL_ENSEMBLE_M: float = 0.01

#: reference/act/README.md:77 `--lr 1e-5`; paper Table III "learning rate 1e-5".
#: NOT `detr/main.py:14`'s default of 1e-4, which the published command overrides.
#: TR28: this value must be STATED by ACT, never inherited from another model's
#: config. The Stage-4 gate ran at 1e-3 because the shared runner drew `lr` from
#: `BCConfig`, and nothing raised.
LR: float = 1e-5
#: reference/act/detr/main.py:17 `--weight_decay 1e-4`, not overridden by README.md:76.
WEIGHT_DECAY: float = 1e-4
#: reference/act/detr/main.py:87 `torch.optim.AdamW`.
OPTIMIZER: str = "adamw"

#: MEASURED on the pinned commit, 2026-09-21. `detr_vae.py:131` reads `hs[0]`, and
#: `transformer.py:76` returns (num_dec_layers, bs, num_queries, d), so index 0 is
#: the FIRST decoder layer's output. Backward through `hs[0]` on a 7-layer decoder
#: gives layer 1 a gradient of L1 7.41e-05 and layers 2-7 exactly 0.0. We build the
#: reference's seven layers and read the same index it reads: fidelity by
#: construction. Recorded descriptively in NOTES.md 2026-09-21; not "fixed" here,
#: because reading hs[-1] would be a deeper, different model and any RQ2/RQ3 result
#: obtained with it could not be attributed to ACT's design.
DECODER_LAYER_READ: int = 0


def _get_activation_fn(activation: str):
    """reference/act/detr/models/transformer.py:306-314."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError("activation should be relu/gelu/glu, not %r" % activation)


def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    """reference/act/detr/models/transformer.py:289-290."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def sinusoid_table(n_position: int, d_hid: int) -> torch.Tensor:
    """reference/act/detr/models/detr_vae.py:23-31, reproduced exactly.

    FIXED, not learned. In the reference it is a `register_buffer`
    (`detr_vae.py:72`) applied with `.clone().detach()` (`:101`), so it never
    appears in `model.parameters()`. A learned table here would train perfectly
    well and would change what the CLS token summarises, which is why
    `test_act.py` asserts it is a buffer.
    """
    def angle_vec(position):
        return [position / np.power(10000, 2 * (j // 2) / d_hid) for j in range(d_hid)]

    table = np.array([angle_vec(p) for p in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])      # dim 2i
    table[:, 1::2] = np.cos(table[:, 1::2])      # dim 2i+1
    return torch.FloatTensor(table).unsqueeze(0)


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """reference/act/detr/models/detr_vae.py:17-20.

    `std = logvar.div(2).exp()`. A missing `div(2)` trains fine and silently
    halves the effective KL scale, so the factor is asserted in the tests.
    """
    std = logvar.div(2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


# ─── transformer, mirroring reference/act/detr/models/transformer.py ──────────
class _EncoderLayer(nn.Module):
    """reference/act/detr/models/transformer.py:144-201."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation,
                 normalize_before):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def _with_pos(tensor, pos):
        """transformer.py:163-164 — pos is added to Q and K, never to V."""
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_key_padding_mask=None, pos=None):
        if self.normalize_before:                        # transformer.py:181-193
            src2 = self.norm1(src)
            q = k = self._with_pos(src2, pos)
            src2 = self.self_attn(q, k, value=src2,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            return src + self.dropout2(src2)
        # transformer.py:166-179 (post-norm, the published configuration)
        q = k = self._with_pos(src, pos)
        src2 = self.self_attn(q, k, value=src,
                              key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class _Encoder(nn.Module):
    """reference/act/detr/models/transformer.py:79-100."""

    def __init__(self, layer, num_layers, norm=None):
        super().__init__()
        self.layers = _clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask=None, pos=None):
        out = src
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            out = self.norm(out)
        return out


class _DecoderLayer(nn.Module):
    """reference/act/detr/models/transformer.py:204-286."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation,
                 normalize_before):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def _with_pos(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None,
                memory_key_padding_mask=None):
        if self.normalize_before:                        # transformer.py:252-...
            tgt2 = self.norm1(tgt)
            q = k = self._with_pos(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)
            tgt2 = self.multihead_attn(query=self._with_pos(tgt2, query_pos),
                                       key=self._with_pos(memory, pos),
                                       value=memory,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            return tgt + self.dropout3(tgt2)
        # transformer.py:229-250 (post-norm, the published configuration)
        q = k = self._with_pos(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(query=self._with_pos(tgt, query_pos),
                                   key=self._with_pos(memory, pos),
                                   value=memory,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))


class _Decoder(nn.Module):
    """reference/act/detr/models/transformer.py:103-141.

    `return_intermediate=True` (set by `build_transformer`, `transformer.py:302`)
    stacks EVERY layer's output, with the last replaced by the final normed one
    (`:132-136`). The head then selects an index — see DECODER_LAYER_READ.
    """

    def __init__(self, layer, num_layers, norm, return_intermediate=True):
        super().__init__()
        self.layers = _clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, pos=None, query_pos=None,
                memory_key_padding_mask=None):
        out = tgt
        intermediate: List[torch.Tensor] = []
        for layer in self.layers:
            out = layer(out, memory, pos=pos, query_pos=query_pos,
                        memory_key_padding_mask=memory_key_padding_mask)
            if self.return_intermediate:
                intermediate.append(self.norm(out))
        if self.norm is not None:
            out = self.norm(out)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(out)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return out.unsqueeze(0)


# ─── configuration ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ACTConfig:
    """ACT's hyperparameters. Reference values are cited at their constants above.

    `obs_window` and `chunk_size` are REQUIRED with no default, as everywhere else
    in this codebase: K is what separates ACT from chunked BC on the ladder, and
    W_o is the parameter that could hand the transformer the history ACT-LSTM is
    meant to supply. Neither may be inherited silently.

    W_o = 1 is not a compromise, it is the reference: ACT's observation is a
    single timestep (`utils.py:37`, `detr_vae.py:80`).
    """

    obs_window: int                 # required; 1 to match the reference
    chunk_size: int                 # required; K = number of decoder queries
    #: REQUIRED, no default (TR28). ACT's published value is `LR` = 1e-5; the
    #: field is left unset so that a caller who does not think about it gets a
    #: TypeError instead of somebody else's learning rate. The Stage-4 gate at
    #: lr 1e-3 is what this field exists to prevent.
    lr: float

    hidden_dim: int = HIDDEN_DIM
    dim_feedforward: int = DIM_FEEDFORWARD
    nheads: int = NHEADS
    enc_layers: int = ENC_LAYERS
    dec_layers: int = DEC_LAYERS
    dropout: float = DROPOUT
    activation: str = ACTIVATION
    pre_norm: bool = PRE_NORM
    latent_dim: int = LATENT_DIM
    kl_weight: float = KL_WEIGHT
    weight_decay: float = WEIGHT_DECAY
    optimizer: str = OPTIMIZER

    #: OURS, PROVISIONAL. The reference's action head is `Linear(hidden, 14)`
    #: (`detr_vae.py:52`, `state_dim = 14` hardcoded at `:230`). Our action is
    #: 22-D and our state 47-D (`g1_data/spec.py`); pure dimensionality.
    state_dim: int = spec.STATE_DIM
    action_dim: int = spec.ACTION_DIM

    def __post_init__(self):
        if int(self.obs_window) < 1:
            raise ValueError("obs_window must be >= 1, got %r" % (self.obs_window,))
        if int(self.chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1, got %r" % (self.chunk_size,))
        if self.hidden_dim % self.nheads:
            raise ValueError("hidden_dim %d must be divisible by nheads %d"
                             % (self.hidden_dim, self.nheads))

    def optimizer_config(self) -> dict:
        """The optimizer settings ACT declares for itself (TR28).

        `train.make_optimizer` compares this against the `TrainConfig` it was
        handed and RAISES on any disagreement, so a runner that assembles the
        TrainConfig from another model's config cannot start. The Stage-4 leak
        was silent precisely because no such comparison existed.
        """
        return dict(lr=float(self.lr), weight_decay=float(self.weight_decay),
                    optimizer=str(self.optimizer))

    def as_metadata(self) -> dict:
        d = asdict(self)
        d.update(reference_commit=REF_COMMIT, reference_paper=REF_PAPER,
                 decoder_layer_read=DECODER_LAYER_READ,
                 reference_lr=LR, reference_weight_decay=WEIGHT_DECAY)
        return d


# ─── the model ────────────────────────────────────────────────────────────────
class ACTPolicy(nn.Module):
    """State-only ACT: a CVAE whose decoder is a transformer emitting K actions.

    Shape contract, identical to every other model in the ladder so that
    `train.train()` does not change between stages:

        obs : (B, W_o, 47)   ->   action : (B, K, 22)

    `forward(obs)` is inference. `forward(obs, actions=..., action_mask=...)` is
    training and additionally returns `mu`/`logvar`. Which mode is in force is
    decided by whether the action chunk was supplied (`detr_vae.py:85`
    `is_training = actions is not None`), so a caller cannot accidentally run the
    CVAE encoder at inference — it has nothing to run it on.

    WHERE THE LSTM WILL ATTACH (not built, D8): between `_encode_obs` and the
    transformer, as a third token or as a transform of `proprio_input`, so that
    the CVAE encoder, the decoder, the queries and the head are untouched and the
    only difference between ACT and ACT-LSTM is that one carries state across
    ticks. See the session report.
    """

    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.cfg = cfg
        #: Everything needed to rebuild this module from a checkpoint alone.
        self.hparams = dict(cfg=asdict(cfg))
        h, K = cfg.hidden_dim, cfg.chunk_size

        # ---- CVAE encoder (detr_vae.py:66-72) ----------------------------
        # [CLS] + proprio + action chunk -> mu, logvar. Runs ONLY at training.
        self.cls_embed = nn.Embedding(1, h)                        # :68
        self.encoder_action_proj = nn.Linear(cfg.action_dim, h)    # :69 (theirs 14)
        self.encoder_joint_proj = nn.Linear(
            cfg.state_dim * cfg.obs_window, h)                     # :70 (theirs 14)
        self.latent_proj = nn.Linear(h, cfg.latent_dim * 2)        # :71
        # FIXED sinusoidal, length 1 + 1 + K = [CLS], qpos, a_seq   # :72
        self.register_buffer("pos_table", sinusoid_table(1 + 1 + K, h))

        enc_layer = _EncoderLayer(h, cfg.nheads, cfg.dim_feedforward, cfg.dropout,
                                  cfg.activation, cfg.pre_norm)
        # build_encoder, detr_vae.py:212-226: norm only when pre_norm
        self.encoder = _Encoder(enc_layer, cfg.enc_layers,
                                nn.LayerNorm(h) if cfg.pre_norm else None)

        # ---- decoder side (detr_vae.py:74-76) -----------------------------
        self.latent_out_proj = nn.Linear(cfg.latent_dim, h)        # :75
        # LEARNED position embedding for exactly two tokens: [latent, proprio].
        # Sized 2 in the reference (:76) because those are the only non-image
        # tokens — which is the evidence that option A is the reference's own
        # design for the non-image part, not our invention.
        self.additional_pos_embed = nn.Embedding(2, h)             # :76
        self.input_proj_robot_state = nn.Linear(
            cfg.state_dim * cfg.obs_window, h)                     # :58 (theirs 14)

        # transformer encoder over the two tokens, then a K-query decoder
        t_enc_layer = _EncoderLayer(h, cfg.nheads, cfg.dim_feedforward,
                                    cfg.dropout, cfg.activation, cfg.pre_norm)
        self.t_encoder = _Encoder(t_enc_layer, cfg.enc_layers,
                                  nn.LayerNorm(h) if cfg.pre_norm else None)
        t_dec_layer = _DecoderLayer(h, cfg.nheads, cfg.dim_feedforward,
                                    cfg.dropout, cfg.activation, cfg.pre_norm)
        # decoder ALWAYS has a final LayerNorm (transformer.py:35)
        self.t_decoder = _Decoder(t_dec_layer, cfg.dec_layers, nn.LayerNorm(h),
                                  return_intermediate=True)

        # LEARNED queries (detr_vae.py:54). The paper §IV-C calls them "a fixed
        # position embedding"; code and paper disagree and we follow the code,
        # which is the artefact that produced the published results.
        self.query_embed = nn.Embedding(K, h)                      # :54
        self.action_head = nn.Linear(h, cfg.action_dim)            # :52 (theirs 14)
        # `is_pad_head` (:53) is DROPPED: the reference computes it (:138) and
        # returns it (:139) and never uses it in any loss. Carrying it would add
        # parameters that receive no gradient.

        self._reset_parameters()

    def optimizer_config(self) -> dict:
        """Hook read by `train.make_optimizer`. See `ACTConfig.optimizer_config`."""
        return self.cfg.optimizer_config()

    def _reset_parameters(self):
        """reference/act/detr/models/transformer.py:44-47 — xavier_uniform_ on
        every parameter with dim() > 1. Applied to the transformer stacks only,
        as in the reference, where `_reset_parameters` is a method of
        `Transformer` and does not touch the projections built in `DETRVAE`."""
        for m in (self.encoder, self.t_encoder, self.t_decoder):
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # ---- the CVAE encoder --------------------------------------------------
    def encode(self, obs: torch.Tensor, actions: torch.Tensor,
               action_mask: Optional[torch.Tensor] = None):
        """(mu, logvar) from [CLS] + proprio + action chunk. TRAINING ONLY.

        `action_mask` is our loader's True-on-real mask; the reference's `is_pad`
        is its complement (`utils.py:53-54`). D9: padded timesteps MUST be masked
        out of attention or z is computed partly from zeros that are not data —
        `detr_vae.py:98-99` prepends two `False` so [CLS] and qpos are never
        masked, then passes it as `src_key_padding_mask` (`:104`). This is
        gate-blind: without it the model trains perfectly and z is quietly wrong.
        """
        B = obs.shape[0]
        flat = obs.reshape(B, -1)
        action_embed = self.encoder_action_proj(actions)            # (B, K, h)
        qpos_embed = self.encoder_joint_proj(flat).unsqueeze(1)     # (B, 1, h)
        cls = self.cls_embed.weight.unsqueeze(0).repeat(B, 1, 1)    # (B, 1, h)
        src = torch.cat([cls, qpos_embed, action_embed], dim=1)     # (B, K+2, h)
        src = src.permute(1, 0, 2)                                  # (K+2, B, h)

        key_padding = None
        if action_mask is not None:
            is_pad = ~action_mask.bool()                            # (B, K)
            head = torch.zeros((B, 2), dtype=torch.bool, device=obs.device)
            key_padding = torch.cat([head, is_pad], dim=1)          # (B, K+2)

        pos = self.pos_table.clone().detach().permute(1, 0, 2)      # (K+2, 1, h)
        out = self.encoder(src, src_key_padding_mask=key_padding, pos=pos)
        latent_info = self.latent_proj(out[0])                      # CLS only, :105
        mu = latent_info[:, :self.cfg.latent_dim]
        logvar = latent_info[:, self.cfg.latent_dim:]
        return mu, logvar

    # ---- the policy --------------------------------------------------------
    def _decode(self, obs: torch.Tensor, latent_input: torch.Tensor) -> torch.Tensor:
        """The two-token transformer encoder and the K-query decoder.

        Mirrors the reference's VISION path with the image tokens removed:
        `transformer.py:62` stacks [latent, proprio] and `:63` concatenates the
        image tokens after them; with no images the stack IS the sequence, and
        `:59-60` reduces to `additional_pos_embed` alone.
        """
        B = obs.shape[0]
        proprio = self.input_proj_robot_state(obs.reshape(B, -1))   # (B, h)
        src = torch.stack([latent_input, proprio], dim=0)           # (2, B, h)
        pos = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, B, 1)
        query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (K, B, h)
        tgt = torch.zeros_like(query)                               # transformer.py:72

        memory = self.t_encoder(src, pos=pos)
        hs = self.t_decoder(tgt, memory, pos=pos, query_pos=query)  # (L, K, B, h)
        hs = hs.transpose(1, 2)                                     # (L, B, K, h)
        return hs[DECODER_LAYER_READ]                               # detr_vae.py:131

    def forward(self, obs: torch.Tensor, actions: Optional[torch.Tensor] = None,
                action_mask: Optional[torch.Tensor] = None):
        """(B, W_o, 47) -> (B, K, 22), plus (mu, logvar) when training.

        Mode is decided HERE, from whether an action chunk was supplied
        (`detr_vae.py:85`), not by a caller-set flag. At inference z is the prior
        mean, i.e. ZEROS (`detr_vae.py:113`; paper §IV-B "we set z to be the mean
        of the prior distribution i.e. zero"), and the CVAE encoder is not run at
        all — it is discarded at test time (paper §IV-B).
        """
        if obs.dim() != 3 or obs.shape[-1] != self.cfg.state_dim:
            raise ValueError("obs must be (B, W_o, %d), got %s"
                             % (self.cfg.state_dim, tuple(obs.shape)))
        if obs.shape[1] != self.cfg.obs_window:
            raise ValueError("this policy was built for obs_window=%d, got %d"
                             % (self.cfg.obs_window, obs.shape[1]))
        B = obs.shape[0]

        if actions is None:                                # inference
            mu = logvar = None
            z = torch.zeros(B, self.cfg.latent_dim, dtype=obs.dtype,
                            device=obs.device)
        else:                                              # training
            if actions.shape[1] != self.cfg.chunk_size:
                raise ValueError("action chunk must be K=%d, got %d"
                                 % (self.cfg.chunk_size, actions.shape[1]))
            mu, logvar = self.encode(obs, actions, action_mask)
            z = reparametrize(mu, logvar)

        latent_input = self.latent_out_proj(z)
        a_hat = self.action_head(self._decode(obs, latent_input))
        if actions is None:
            return a_hat
        return a_hat, mu, logvar


    # ---- the loop's optional hook (train.forward_loss) --------------------
    def loss_terms(self, obs, target, pad_mask, dim_mask=None):
        """(loss, reconstruction_l1, n_elements, extras) — ACT's L1 + beta*KL.

        This is what lets ONE training loop serve every rung of the ladder. The
        reconstruction term is `train.masked_l1`, the same function BC and
        chunked BC are scored with (D6); ACT only ADDS the KL, exactly as the
        reference does (`policy.py:34`, `l1 + kl * kl_weight`).

        The action chunk and its mask are passed to the CVAE encoder here, which
        is why the hook exists at all: a plain `model(obs)` cannot express a
        model whose training-time forward needs the target.
        """
        pred, mu, logvar = self(obs, actions=target, action_mask=pad_mask)
        loss, l1, kl, n = act_loss(pred, target, pad_mask, mu, logvar,
                                   self.cfg.kl_weight, dim_mask)
        return loss, l1, n, dict(kl=float(kl.detach()),
                                 kl_weighted=float((kl * self.cfg.kl_weight).detach()))


# ─── the objective ────────────────────────────────────────────────────────────
def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """reference/act/policy.py:71-84, the `total_kld` branch.

    `klds = -0.5 * (1 + logvar - mu^2 - exp(logvar))`, SUMMED over latent dims
    and averaged over the batch (`policy.py:79-80`), consumed as `total_kld[0]`
    (`:33`). The function also returns `dim_wise_kld` and `mean_kld` (`:84`) and
    neither is used; taking `mean_kld` instead would rescale the KL by the latent
    dimension — 32x — and would still train.
    """
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(1).mean(0)


def act_loss(pred, target, pad_mask, mu, logvar, kl_weight: float,
             dim_mask=None):
    """L1 + beta * KL (`reference/act/policy.py:34`).

    The reconstruction term is `train.masked_l1` — the SAME function BC and
    chunked BC use. One loss across every stage is the constraint the whole
    ladder rests on; a second implementation here would put an unknown offset
    inside every cross-model comparison.

    DIVERGENCE, DISCLOSED AND MEASURED: the reference reduces with
    `(all_l1 * ~is_pad).mean()` (`policy.py:31`), which zeroes padded entries and
    then divides by the FULL element count, padding and all 14 dims included.
    Ours divides by contributing elements (real timesteps x 16 trainable dims).
    Measured 2026-09-21 on 7,628 real samples at K=100: our reconstruction term
    is **1.4704x** the reference's scale, so preserving the reference's L1:KL
    balance would need beta' = 14.70 rather than 10. beta = 10 is used here
    because the overfit-10 gate cannot discriminate between them; the choice is
    deferred to a validation signal. BOTH numbers are written into run metadata.
    """
    from g1_model.train import masked_l1
    l1, n = masked_l1(pred, target, pad_mask, dim_mask)
    kl = kl_divergence(mu, logvar)
    return l1 + kl * float(kl_weight), l1, kl, n


# ─── A3: temporal ensembling (INFERENCE ONLY) ─────────────────────────────────
class TemporalEnsembler:
    """reference/act/imitate_episodes.py:218-259.

    An INFERENCE mechanism. The policy is queried every tick, so chunks overlap
    and several predictions exist for the same timestep; they are combined by an
    exponential weighting over how long ago each was made:

        exp_weights = np.exp(-m * np.arange(n))      imitate_episodes.py:256
        exp_weights = exp_weights / exp_weights.sum()                    :257
        raw_action  = (actions_for_curr_step * exp_weights).sum(dim=0)   :259

    `w_0` weights the OLDEST prediction still in the buffer (paper §IV-A), and a
    smaller `m` incorporates new observations more slowly. It is OFF by default
    in the reference — `--temporal_agg` is `store_true` (`:433`) — and when on it
    sets `query_frequency = 1` (`:193`).

    It touches no parameter and appears nowhere in training. Stage 3 deliberately
    excluded it so that chunked BC isolated chunking alone; it is built here
    because it is part of ACT, and whether the final comparison enables it for
    every rung or none is a decision that must be made once and applied uniformly.
    """

    def __init__(self, chunk_size: int, action_dim: int = spec.ACTION_DIM,
                 m: float = TEMPORAL_ENSEMBLE_M):
        self.K = int(chunk_size)
        self.action_dim = int(action_dim)
        self.m = float(m)
        self.reset()

    def reset(self) -> None:
        """Per episode. A buffer carried across episodes would average one
        episode's intent into the next."""
        self._buf: List[Optional[torch.Tensor]] = []
        self.t = 0

    @staticmethod
    def weights(n: int, m: float = TEMPORAL_ENSEMBLE_M) -> np.ndarray:
        """imitate_episodes.py:256-257, exactly. Index 0 is the OLDEST."""
        w = np.exp(-m * np.arange(n))
        return w / w.sum()

    def step(self, chunk: torch.Tensor) -> torch.Tensor:
        """Absorb a (K, 22) prediction made at the current tick; return the
        ensembled action for this tick, (22,)."""
        if chunk.shape != (self.K, self.action_dim):
            raise ValueError("expected a (%d, %d) chunk, got %s"
                             % (self.K, self.action_dim, tuple(chunk.shape)))
        self._buf.append(chunk)
        if len(self._buf) > self.K:
            self._buf.pop(0)
        # prediction made i ticks ago contributes its entry for the current tick
        n = len(self._buf)
        stack = torch.stack([self._buf[j][n - 1 - j] for j in range(n)], dim=0)
        w = torch.from_numpy(self.weights(n, self.m)).to(stack.device,
                                                         stack.dtype).unsqueeze(1)
        self.t += 1
        return (stack * w).sum(dim=0)


def build_act(cfg: ACTConfig = None, **kw) -> ACTPolicy:
    """Factory for `train.seeded_build` and `train.load_checkpoint` (D10)."""
    if cfg is not None and kw:
        raise ValueError("pass an ACTConfig or keyword arguments, not both")
    if cfg is None:
        cfg = ACTConfig(**kw["cfg"]) if "cfg" in kw else ACTConfig(**kw)
    elif isinstance(cfg, dict):
        cfg = ACTConfig(**cfg)
    return ACTPolicy(cfg)
