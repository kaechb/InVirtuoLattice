"""Frozen ESM-2 wrapper that produces per-protein embeddings.

The README spec (Stage 3) is: ``embed_protein(seq) → z_p ∈ R^1280`` and a
precomputed store. We deliberately keep this module small — it only loads the
ESM-2 weights, runs them in eval/no-grad mode, and returns CPU tensors. The
*orchestration* (filtering, batching across thousands of proteins, writing to
disk) lives in ``lattice.protein.precompute`` so it stays testable without
needing to load 2.5 GB of weights.

Defaults track the README: ``facebook/esm2_t33_650M_UR50D``, ``d=1280``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


ESM2_DEFAULT_MODEL: str = "facebook/esm2_t33_650M_UR50D"
ESM2_DEFAULT_DIM: int = 1280
ESM2_MAX_LEN_DEFAULT: int = 1024


@dataclass(frozen=True)
class ProteinEncoderConfig:
    """Hyperparameters for the frozen ESM-2 wrapper.

    Attributes:
        model_name: HuggingFace model id. Defaults to the README's 650M variant.
        embedding_dim: Expected hidden size (sanity-check vs the loaded model).
        max_length: Truncate sequences longer than this. ESM-2 was trained at
            1024; longer windows usually degrade. Stage-1 filter already caps at
            1500 — this just enforces it on the encoder side too.
        dtype: Inference precision. ``float16`` is fine on GPU and halves memory.
        device: Where to place the model. ``"cuda"`` recommended; tests use
            ``"cpu"``.
        per_residue: If True, also return per-residue states (used by some
            downstream pocket-aware variants); default False to keep RAM small.
        prepend_bos: ESM-2 tokenizers add a BOS/EOS — handled by the tokenizer,
            this field is reserved for future custom tokenization.
    """

    model_name: str = ESM2_DEFAULT_MODEL
    embedding_dim: int = ESM2_DEFAULT_DIM
    max_length: int = ESM2_MAX_LEN_DEFAULT
    dtype: str = "float32"  # "float16" or "float32"
    device: str = "cpu"
    per_residue: bool = False
    extra: dict[str, str] = field(default_factory=dict)

    def torch_dtype(self) -> torch.dtype:
        return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[
            self.dtype
        ]


class ProteinEncoder(nn.Module):
    """Thin frozen-ESM-2 wrapper. Always eval mode, ``torch.no_grad`` on forward.

    Use ``embed_protein(seq)`` for one sequence (returns a 1-D tensor) or
    ``embed_batch(seqs)`` for many (returns ``[N, D]``). Both mean-pool over the
    valid (non-pad, non-special) residue positions, matching the README's
    "mean-pooled" requirement.
    """

    def __init__(
        self,
        cfg: ProteinEncoderConfig | None = None,
        *,
        model: nn.Module | None = None,
        tokenizer: object | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ProteinEncoderConfig()
        if model is None or tokenizer is None:
            model, tokenizer = self._load_pretrained(self.cfg)
        self.model = model
        self.tokenizer = tokenizer
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        # Sanity-check hidden size: protect against the user pointing at a
        # different ESM variant without updating ``embedding_dim``.
        cfg_hidden = getattr(getattr(self.model, "config", object()), "hidden_size", None)
        if cfg_hidden is not None and cfg_hidden != self.cfg.embedding_dim:
            raise ValueError(
                f"model.config.hidden_size={cfg_hidden} does not match "
                f"cfg.embedding_dim={self.cfg.embedding_dim}; update the config."
            )

    @staticmethod
    def _load_pretrained(cfg: ProteinEncoderConfig) -> tuple[nn.Module, object]:
        # Late import so `import lattice_lab.protein` does not require transformers
        # (the dep is real, but the import error is much clearer this way).
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - install-time error
            raise ImportError(
                "transformers is required to load ESM-2. Install with "
                "`pip install transformers`."
            ) from exc
        logger.info("loading ESM-2 weights: %s", cfg.model_name)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModel.from_pretrained(
            cfg.model_name, torch_dtype=cfg.torch_dtype()
        ).to(cfg.device)
        return model, tokenizer

    @torch.no_grad()
    def _forward_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(**batch, output_hidden_states=False, return_dict=True)
        # ``last_hidden_state`` shape: [B, L, D]
        return out.last_hidden_state

    @torch.no_grad()
    def embed_batch(self, seqs: Sequence[str]) -> torch.Tensor:
        """Encode a list of protein sequences and return ``[N, D]`` on CPU.

        Mean-pooled over residue positions excluding pad, BOS, and EOS tokens.
        Sequences longer than ``cfg.max_length`` are truncated.
        """
        if not seqs:
            return torch.empty(0, self.cfg.embedding_dim, dtype=torch.float32)
        seqs = [_validate_sequence(s) for s in seqs]
        enc = self.tokenizer(
            list(seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=True,
        )
        enc = {k: v.to(self.cfg.device) for k, v in enc.items()}
        h = self._forward_tokens(enc)  # [B, L, D]
        # Build a mask that excludes pads + special tokens; HuggingFace exposes
        # the attention mask; we additionally drop BOS (idx 0) and EOS (last
        # non-pad position) to match the "per-residue mean" semantics.
        mask = enc["attention_mask"].to(h.dtype)  # [B, L], 1 for real tokens
        mask = _strip_special_tokens(mask)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]
        return pooled.detach().to(torch.float32).cpu()

    @torch.no_grad()
    def embed_protein(self, seq: str) -> torch.Tensor:
        """Encode one sequence → ``[D]`` CPU tensor."""
        return self.embed_batch([seq])[0]

    @torch.no_grad()
    def embed_per_residue(self, seq: str) -> torch.Tensor:
        """Encode one sequence → ``[L_kept, D]`` CPU tensor (no special tokens)."""
        seq = _validate_sequence(seq)
        enc = self.tokenizer(
            seq,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=True,
        )
        enc = {k: v.to(self.cfg.device) for k, v in enc.items()}
        h = self._forward_tokens(enc)[0]  # [L, D]
        mask = _strip_special_tokens(enc["attention_mask"].to(h.dtype))[0]  # [L]
        return h[mask.bool()].detach().to(torch.float32).cpu()


def _validate_sequence(seq: str) -> str:
    """Normalize whitespace and uppercase; raise on empty input."""
    s = "".join(seq.split()).upper()
    if not s:
        raise ValueError("empty protein sequence")
    return s


def _strip_special_tokens(mask: torch.Tensor) -> torch.Tensor:
    """Zero out the leading BOS and the trailing EOS positions of an attention mask.

    Works on ``[B, L]`` or ``[1, L]``. The mask is 1 for valid tokens, 0 for pad.
    For each row we set ``mask[:, 0] = 0`` (BOS) and the last 1 → 0 (EOS).
    """
    out = mask.clone()
    if out.dim() != 2:
        raise ValueError(f"expected [B, L] mask, got shape {tuple(out.shape)}")
    out[:, 0] = 0.0
    # Index of last valid (non-pad) position per row; that's the EOS.
    lengths = mask.sum(dim=1).long()  # number of non-pad tokens
    last_idx = (lengths - 1).clamp_min(0)
    for b in range(out.shape[0]):
        if lengths[b] > 0:
            out[b, last_idx[b]] = 0.0
    return out
