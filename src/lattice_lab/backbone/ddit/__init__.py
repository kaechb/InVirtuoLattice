"""Vendored DDiT architecture (from InVirtuoGen) for the discrete-flow backbone.

Kept here so the discrete-flow encoder has no runtime dependency on the external
``in_virtuo_gen`` package — only the checkpoint + tokenizer are external inputs.
``model_ddit`` does ``from . import rotary``, so ``rotary.py`` must sit beside it.
"""

from __future__ import annotations
