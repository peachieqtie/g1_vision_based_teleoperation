"""Phase 4: the model codebase. Sibling to `g1_data/`, which it only reads.

`g1_data/` is the frozen data contract - the 47-D state, the 22-D action, the
masks, the clips, the recorder and the refusals. Nothing here may change it. The
one direction of dependency is `g1_model` -> `g1_data`, and a loader that needed
`g1_data` to change would be a loader that had stopped reading the dataset the
recorder actually wrote.

Stage 1 is the dataset loader and nothing else: no model, no training loop, no
deployment harness.
"""
from .loader import (ChunkDataset, LoaderConfig, TrackingPolicy,
                     collate_chunks)

__all__ = ["ChunkDataset", "LoaderConfig", "TrackingPolicy", "collate_chunks"]
