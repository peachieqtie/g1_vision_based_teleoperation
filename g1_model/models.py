"""The policies. Stage 3 has ONE class: BC at K=1, chunked BC at K>1.

=============================================================================
EVERY HYPERPARAMETER IN THIS FILE IS PROVISIONAL. NONE OF IT IS TUNED.
=============================================================================
They were chosen on SCRIPTED DEMONSTRATOR data - `data/synthetic`, produced by
`g1_data/scripted_demo.py`, not by a human at the ZED. The scripted action
distribution is smoother, more repeatable and lower-entropy than piloted
teleoperation will be: the same staged raise every episode, the same cosine
eases, no reaction time, no tracking dropout, no operator correction. A width, a
depth or a learning rate that suits that distribution has no claim on the one
that replaces it.

So: nothing here was swept, searched or tuned, deliberately. The values are
ordinary ones from the behavioural-cloning literature, written down so the next
session can see exactly what was assumed. Choosing them carefully on this data
would be worse than choosing them carelessly - it would produce numbers that
LOOK tuned, and the tuning would be to a distribution that is being thrown away.

WHEN PILOTED DATA EXISTS, EVERY VALUE BELOW IS RE-OPENED. A result obtained with
these numbers on scripted data is a plumbing check, never a finding.
=============================================================================

WHY AN MLP
----------
BC here is the FLOOR: the comparison it exists for is "does chunking help" (ACT)
and "does recurrence help" (ACT-LSTM), and a floor that is itself an interesting
architecture makes both of those harder to read. The proposal's D8 adds plain ACT
precisely so the LSTM is isolated rather than confounded with chunking; the same
logic applies downward. A plain MLP with no memory, no chunk and no attention is
the honest zero point.

THE SHAPE CONTRACT
------------------
Every model in this file, and every model added later, takes

    obs : (B, W_o, 47)      and returns      action : (B, K, 22)

so `train.train()` does not change between stages. BC is the W_o=1, K=1 corner
of that contract, not a different interface: it flattens the window and emits one
timestep. It predicts all 22 dims, including the 6 constant ones - the loss masks
them, and having the head emit 22 keeps the action vector one shape everywhere,
so nothing downstream has to know which 16 were trained.

Chunked BC is the SAME class at K>1. See `BCPolicy` for why that matters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from g1_data import spec

#: Read by `tools/train_bc.py` and copied into every run's metadata, so the
#: warning above travels with the numbers rather than staying in this file.
PROVISIONAL = (
    "PROVISIONAL HYPERPARAMETERS, NOT TUNED. Chosen on SCRIPTED demonstrator "
    "data, whose action distribution differs from piloted teleoperation. No "
    "sweep or search was run, deliberately. Every value is re-opened when "
    "piloted data exists; no result obtained with them may be cited as tuned."
)

#: The chunk length used for Stage 3 plumbing. PROVISIONAL, and NOT SWEPT.
#: 100 ticks is 4.0 s at the 25 Hz record rate (D4) and is the value ACT uses in
#: the original paper, which is the only reason it was picked: it is an ordinary
#: number from the literature rather than a measurement on this task. Episodes
#: here run 694-846 ticks, so it covers roughly an eighth of one. The right K
#: depends on how long a piloted operator's intent stays coherent, which cannot
#: be measured on a scripted demonstrator that never changes its mind. SETTLE IT
#: ON PILOTED DATA.
K_PROVISIONAL: int = 100

#: THE gradient-clipping threshold (global L2 norm) for EVERY model in the ladder.
#: Each model config declares it (`BCConfig.grad_clip`, `ACTConfig.grad_clip`), the
#: loop checks the declaration against the TrainConfig (`train.assert_optimizer_source`)
#: and REFUSES a model that declares any other value (`train.LADDER_INVARIANTS`).
#: It used to arrive as `TrainConfig`'s default - TR28's third occurrence
#: (docs/ACT_AUDIT_REPORT.md R1). To change it, change it HERE, for every model.
LADDER_GRAD_CLIP: float = 1.0

#: Written into every run's metadata. The deviation is stated where the numbers go.
GRAD_CLIP_DISCLOSURE = (
    "DISCLOSED DEVIATION FROM THE ACT REFERENCE: every model clips gradients to a "
    "global L2 norm of %.1f. The reference never clips: `--clip_max_norm 0.1` is "
    "marked '# not used' (reference/act/detr/main.py:20) and reference/act contains "
    "no clip_grad_norm_ call. MEASURED (docs/ACT_AUDIT_REPORT.md R1): the norm is ~678 "
    "at ACT's initialisation and median 1.35 at the converged ACT gate weights, above "
    "the threshold on 19 of 20 batches - the clip is active throughout training, not a "
    "safety net. KEPT, identically for every model, because (1) ACT's passing "
    "overfit-10 gate already used it, (2) recurrent models are where clipping matters "
    "most, and (3) a clip that differed between ACT and ACT-LSTM would sit inside the "
    "headline RQ3 comparison: identical-across-models matters more here than matching "
    "the reference." % LADDER_GRAD_CLIP)


@dataclass(frozen=True)
class BCConfig:
    """Every hyperparameter of the BC policy, in one place. See PROVISIONAL.

    `obs_window` AND `chunk_size` ARE REQUIRED, with no defaults, for the same
    reason `LoaderConfig` requires them: K is what separates BC from chunked BC,
    and a default would make one of the two the normal case and the other an
    opt-in. Stage 3 exists to isolate the effect of K, which it cannot do if K
    can be inherited silently.

    K_PROVISIONAL below is the value used for plumbing. It is NOT tuned and was
    NOT swept - see PROVISIONAL - and it must be settled on piloted data.
    """

    obs_window: int              # required (C3)
    chunk_size: int              # required (C3): 1 = BC, >1 = chunked BC
    hidden: Tuple[int, ...] = (512, 512)   # ordinary 2x512 MLP
    activation: str = "relu"
    dropout: float = 0.0         # the overfit-10 gate must NOT be regularized
    layer_norm: bool = False

    # training. DECLARED by the model (`optimizer_config`, attached by `build_bc`)
    # and checked against the TrainConfig, as ACT's are (TR28).
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    optimizer: str = "adamw"
    #: The ladder's one value; see LADDER_GRAD_CLIP and GRAD_CLIP_DISCLOSURE.
    grad_clip: float = LADDER_GRAD_CLIP

    provisional: str = PROVISIONAL

    def optimizer_config(self) -> dict:
        """The training hyperparameters BC declares for itself (TR28). Exactly
        `train.MODEL_DECLARED_FIELDS`; the loop refuses a partial declaration."""
        return dict(lr=float(self.lr), weight_decay=float(self.weight_decay),
                    optimizer=str(self.optimizer), grad_clip=float(self.grad_clip))

    def as_metadata(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


_ACT = dict(relu=nn.ReLU, gelu=nn.GELU, tanh=nn.Tanh, silu=nn.SiLU)


class BCPolicy(nn.Module):
    """Behavioural cloning: one observation in, K actions out.

    K=1 IS BC. K>1 IS CHUNKED BC. THEY ARE THE SAME CLASS, DELIBERATELY.

    Stage 2 refused `chunk_size > 1` here, so that chunked BC would have to be
    written as its own class. That was wrong, and lifting it is the whole point
    of Stage 3: if chunked BC were a separate class, the measured effect of
    chunking would include every incidental difference between two
    implementations - a different initialization, a different head, a different
    anything - and the isolation this stage exists to provide would be gone.
    The ONLY difference between the two conditions is now the number of actions
    predicted from one observation, which is the variable under study.

    The head emits K x 22 and reshapes. No temporal ensembling (C2): that is an
    ACT mechanism and belongs to Stage 4, and mixing it in here would mean Stage
    3 measured chunking-plus-ensembling. At deployment take `first_action`.
    """

    def __init__(self, obs_window: int, chunk_size: int,
                 hidden: Tuple[int, ...] = (512, 512), activation: str = "relu",
                 dropout: float = 0.0, layer_norm: bool = False):
        super().__init__()
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1, got %r" % (chunk_size,))
        if int(obs_window) < 1:
            raise ValueError("obs_window must be >= 1, got %r" % (obs_window,))
        if activation not in _ACT:
            raise ValueError("unknown activation %r (have %s)"
                             % (activation, ", ".join(sorted(_ACT))))
        self.obs_window = int(obs_window)
        self.chunk_size = int(chunk_size)
        #: Everything needed to rebuild this module from a checkpoint alone.
        self.hparams = dict(obs_window=int(obs_window),
                            chunk_size=int(chunk_size),
                            hidden=tuple(hidden), activation=activation,
                            dropout=float(dropout), layer_norm=bool(layer_norm))

        in_dim = self.obs_window * spec.STATE_DIM
        layers, d = [], in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(_ACT[activation]())
            if dropout:
                layers.append(nn.Dropout(float(dropout)))
            d = h
        layers.append(nn.Linear(d, self.chunk_size * spec.ACTION_DIM))
        self.net = nn.Sequential(*layers)
        #: Set by `build_bc` from a BCConfig. A BCPolicy built from bare keyword
        #: arguments (a checkpoint being loaded for scoring) declares nothing, and
        #: `train.make_optimizer` refuses to train it: it has no stated source for
        #: its learning rate, weight decay, optimizer or clip.
        self._declared_training: Optional[dict] = None

    def declare_training(self, optimizer_config: dict) -> "BCPolicy":
        """Attach the training hyperparameters this model is trained with. A
        declaration, not a change to the forward pass."""
        self._declared_training = dict(optimizer_config)
        return self

    def optimizer_config(self) -> Optional[dict]:
        """Hook read by `train.make_optimizer`; None when nothing was declared."""
        return (None if self._declared_training is None
                else dict(self._declared_training))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, W_o, 47) -> (B, K, 22)."""
        if obs.dim() != 3 or obs.shape[-1] != spec.STATE_DIM:
            raise ValueError("obs must be (B, W_o, %d), got %s"
                             % (spec.STATE_DIM, tuple(obs.shape)))
        if obs.shape[1] != self.obs_window:
            raise ValueError(
                "this policy was built for obs_window=%d but was given %d. W_o "
                "is not a free parameter at call time: it changes what the model "
                "can see, which is the thing the ACT/ACT-LSTM comparison is "
                "about." % (self.obs_window, obs.shape[1]))
        B = obs.shape[0]
        return self.net(obs.reshape(B, -1)).view(B, self.chunk_size,
                                                 spec.ACTION_DIM)


def first_action(pred: torch.Tensor) -> torch.Tensor:
    """The action a chunked policy ACTUALLY executes: the chunk's first step.

    (B, K, 22) -> (B, 22). No temporal ensembling (C2). ACT averages overlapping
    chunk predictions across timesteps, which is a real mechanism with a real
    effect, and folding it in here would mean Stage 3 measured chunking AND
    ensembling and could not say which contributed. Stage 4 adds it; until then
    a chunked policy commits to its first prediction and re-plans next tick.
    """
    if pred.dim() != 3 or pred.shape[-1] != spec.ACTION_DIM:
        raise ValueError("expected (B, K, %d), got %s"
                         % (spec.ACTION_DIM, tuple(pred.shape)))
    return pred[:, 0, :]


def build_bc(cfg: BCConfig = None, **kw) -> BCPolicy:
    """Factory used by `train.load_checkpoint` and by the training scripts."""
    if cfg is not None and kw:
        raise ValueError("pass a BCConfig or keyword arguments, not both")
    if cfg is not None:
        return BCPolicy(obs_window=cfg.obs_window, chunk_size=cfg.chunk_size,
                        hidden=cfg.hidden, activation=cfg.activation,
                        dropout=cfg.dropout, layer_norm=cfg.layer_norm
                        ).declare_training(cfg.optimizer_config())
    return BCPolicy(**kw)
