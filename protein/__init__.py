"""Stage 3 — frozen ESM-2 protein encoder + embedding store.

ESM-2 stays frozen for the entire project. We only ever *use* its outputs, so
this subpackage is a tight wrapper around the HuggingFace ESM-2 weights plus a
memory-mapped store optimized for the access pattern in Stage 4/5 (random pid
lookup, no full-array loads).
"""

from lattice_lab.protein.encoder import ESM2_DEFAULT_MODEL, ProteinEncoder, ProteinEncoderConfig
from lattice_lab.protein.store import EmbeddingStore, StoreManifest

__all__ = [
    "ESM2_DEFAULT_MODEL",
    "ProteinEncoder",
    "ProteinEncoderConfig",
    "EmbeddingStore",
    "StoreManifest",
]
