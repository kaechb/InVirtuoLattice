"""Wrapper around the FragMol backbone.

Two responsibilities:

1. Load the pretrained GPT model + config from ``software/FragMol/saved_models/``.
2. Provide a tokenizer + ``encode`` helper that mirrors FragMol's own
   ``encode_with_special_tokens`` but returns padded ``torch.LongTensor`` batches
   and attention masks suitable for the adapter.

We deliberately keep this module thin so the FragMol model code stays untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer

from lattice_lab.paths import FRAGMOL_SAVED_MODEL, FRAGMOL_TOKENIZER, ensure_fragmol_on_path

ensure_fragmol_on_path()

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from utils.transformer import GPT  # noqa: F401


@dataclass(frozen=True)
class FragMolBundle:
    """Loaded FragMol artifacts grouped together for ergonomic passing.

    ``sorted_vocab`` is precomputed for the longest-prefix tokenizer; recomputing
    it on every encode call dominates CPU when batch sizes are small.
    """

    model: "GPT"
    tokenizer: Tokenizer
    n_embd: int
    n_layer: int
    block_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    sorted_vocab: tuple[str, ...]


def load_fragmol(
    model_dir: Path | str = FRAGMOL_SAVED_MODEL,
    tokenizer_path: Path | str = FRAGMOL_TOKENIZER,
    device: str | torch.device = "cpu",
) -> FragMolBundle:
    """Load FragMol weights + tokenizer. Model is set to eval() and frozen."""
    from utils.transformer import GPT, GPTConfig

    model_dir = Path(model_dir)
    config_path = model_dir / "model_config.yml"
    weights_path = model_dir / "model_weights.pth"

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    mc = GPTConfig(
        vocab_size=cfg["vocab_size"],
        block_size=cfg.get("block_size", 200),
        num_props=cfg.get("num_props", 0),
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        embd_pdrop=cfg.get("embd_pdrop", 0.0),
        resid_pdrop=cfg.get("resid_pdrop", 0.0),
        attn_pdrop=cfg.get("attn_pdrop", 0.0),
    )
    model = GPT(mc)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tok = Tokenizer.from_file(str(tokenizer_path))
    pad_id = tok.token_to_id("[PAD]")
    bos_id = tok.token_to_id("[BOS]")
    eos_id = tok.token_to_id("[EOS]")
    assert pad_id is not None and bos_id is not None and eos_id is not None

    sorted_vocab = tuple(sorted(tok.get_vocab().keys(), key=len, reverse=True))

    return FragMolBundle(
        model=model,
        tokenizer=tok,
        n_embd=mc.n_embd,
        n_layer=mc.n_layer,
        block_size=mc.block_size,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        sorted_vocab=sorted_vocab,
    )


def custom_tokenize(sequence: str, sorted_vocab: Sequence[str]) -> list[str]:
    """Longest-prefix tokenization matching FragMol's ``custom_tokenize``."""
    tokens: list[str] = []
    i = 0
    n = len(sequence)
    while i < n:
        matched = False
        for t in sorted_vocab:
            if sequence.startswith(t, i):
                tokens.append(t)
                i += len(t)
                matched = True
                break
        if not matched:
            tokens.append(sequence[i])
            i += 1
    return tokens


def encode_view(bundle: FragMolBundle, view: str) -> list[int]:
    """Encode a FragMol view string to a list of token ids (BOS + body + EOS)."""
    tokens = custom_tokenize(view, bundle.sorted_vocab)
    tok = bundle.tokenizer
    vocab = tok.get_vocab()
    unk_id = tok.token_to_id("[UNK]")
    body: list[int] = []
    for t in tokens:
        if t in vocab:
            body.append(tok.token_to_id(t))
        else:
            body.append(unk_id)
    return [bundle.bos_id, *body, bundle.eos_id]


def pad_batch(
    sequences: Sequence[Sequence[int]],
    *,
    pad_id: int,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad to max length in the batch (or ``max_len`` if given). Returns (ids, mask).

    ``mask`` is 1 for real tokens (including BOS/EOS) and 0 for pad positions.
    """
    target_len = max_len if max_len is not None else max(len(s) for s in sequences)
    b = len(sequences)
    ids = np.full((b, target_len), pad_id, dtype=np.int64)
    mask = np.zeros((b, target_len), dtype=np.float32)
    for i, s in enumerate(sequences):
        ln = min(len(s), target_len)
        ids[i, :ln] = s[:ln]
        mask[i, :ln] = 1.0
    return torch.from_numpy(ids), torch.from_numpy(mask)
