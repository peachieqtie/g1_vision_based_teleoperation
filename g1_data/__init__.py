"""Demonstration-collection harness (CLAUDE.md section 5).

Only `reset` exists so far; `recorder`, `episode`, `phases`, `success` and
`dataset` follow.
"""
from .reset import (EpisodeStart, LocomotionCarryover, reset_episode,
                    reset_policy_state, state_fingerprint)

__all__ = ["EpisodeStart", "LocomotionCarryover", "reset_episode",
           "reset_policy_state", "state_fingerprint"]
