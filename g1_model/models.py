"""The policies. Stage 2 has exactly one: BC, the floor baseline.

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
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

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


@dataclass(frozen=True)
class BCConfig:
    """Every hyperparameter of the BC policy, in one place. See PROVISIONAL."""

    obs_window: int = 1          # BC is single-step by definition (C1)
    chunk_size: int = 1          # one action out
    hidden: Tuple[int, ...] = (512, 512)   # ordinary 2x512 MLP
    activation: str = "relu"
    dropout: float = 0.0         # the overfit-10 gate must NOT be regularized
    layer_norm: bool = False

    # training, for the record; consumed by tools/train_bc.py
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    optimizer: str = "adamw"

    provisional: str = PROVISIONAL

    def as_metadata(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


_ACT = dict(relu=nn.ReLU, gelu=nn.GELU, tanh=nn.Tanh, silu=nn.SiLU)


class BCPolicy(nn.Module):
    """Single-step behavioural cloning: one observation in, ONE action out.

    The flattened observation window goes through an MLP to 22 action dims. No
    memory, no chunk, no attention - see the module docstring.
    """

    def __init__(self, obs_window: int = 1, chunk_size: int = 1,
                 hidden: Tuple[int, ...] = (512, 512), activation: str = "relu",
                 dropout: float = 0.0, layer_norm: bool = False):
        super().__init__()
        if int(chunk_size) != 1:
            raise ValueError(
                "BCPolicy is single-step by definition (C1): chunk_size must be "
                "1, got %r. Chunked BC is a separate stage and must not be got "
                "by passing a bigger K to this class - the whole point of the "
                "BC/chunked-BC comparison is that they are different models."
                % (chunk_size,))
        if activation not in _ACT:
            raise ValueError("unknown activation %r (have %s)"
                             % (activation, ", ".join(sorted(_ACT))))
        self.obs_window = int(obs_window)
        self.chunk_size = 1
        #: Everything needed to rebuild this module from a checkpoint alone.
        self.hparams = dict(obs_window=int(obs_window), chunk_size=1,
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
        layers.append(nn.Linear(d, spec.ACTION_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, W_o, 47) -> (B, 1, 22)."""
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
        return self.net(obs.reshape(B, -1)).view(B, 1, spec.ACTION_DIM)


def build_bc(cfg: BCConfig = None, **kw) -> BCPolicy:
    """Factory used by `train.load_checkpoint` and by the training scripts."""
    if cfg is not None and kw:
        raise ValueError("pass a BCConfig or keyword arguments, not both")
    if cfg is not None:
        return BCPolicy(obs_window=cfg.obs_window, chunk_size=cfg.chunk_size,
                        hidden=cfg.hidden, activation=cfg.activation,
                        dropout=cfg.dropout, layer_norm=cfg.layer_norm)
    return BCPolicy(**kw)
