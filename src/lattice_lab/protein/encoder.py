"""Frozen ESM-2 wrapper that produces per-protein embeddings.

The README spec (Stage 3) is: ``embed_protein(seq) → z_p ∈ R^1280`` and a
precomputed store. We deliberately keep this module small — it only loads the
ESM-2 weights, runs them in eval/no-grad mode, and returns CPU tensors. The
*orchestration* (filtering, batching across thousands of proteins, writing to
disk) lives in ``lattice_lab.protein.precompute`` so it stays testable without
needing to load 2.5 GB of weights.

Defaults track the README: ``facebook/esm2_t33_650M_UR50D``, ``d=1280``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)


ESM2_DEFAULT_MODEL: str = "facebook/esm2_t33_650M_UR50D"
ESM2_DEFAULT_DIM: int = 1280
ESM2_MAX_LEN_DEFAULT: int = 1024

# ESM C (Cambrian) — EvolutionaryScale's embedding-focused ESM-2 successor.
# Loaded via the `esm` SDK rather than transformers. 600M hidden size = 1152.
ESMC_DEFAULT_MODEL: str = "esmc_600m"
ESMC_DEFAULT_DIM: int = 1152
ESMC_MAX_LEN_DEFAULT: int = 2048


def _torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[
        name
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
        *,
        model_name: str = ESM2_DEFAULT_MODEL,
        embedding_dim: int = ESM2_DEFAULT_DIM,
        max_length: int = ESM2_MAX_LEN_DEFAULT,
        dtype: str = "float32",
        device: str = "cpu",
        per_residue: bool = False,
        model: nn.Module | None = None,
        tokenizer: object | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.dtype = dtype
        self.device = device
        self.per_residue = per_residue
        if model is None or tokenizer is None:
            model, tokenizer = self._load_pretrained(
                model_name=model_name,
                embedding_dim=embedding_dim,
                dtype=dtype,
                device=device,
            )
        self.model = model
        self.tokenizer = tokenizer
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        # Sanity-check hidden size: protect against the user pointing at a
        # different ESM variant without updating ``embedding_dim``.
        cfg_hidden = getattr(getattr(self.model, "config", object()), "hidden_size", None)
        if cfg_hidden is not None and cfg_hidden != embedding_dim:
            raise ValueError(
                f"model.config.hidden_size={cfg_hidden} does not match "
                f"embedding_dim={embedding_dim}; update the config."
            )

    @staticmethod
    def _load_pretrained(
        *,
        model_name: str,
        embedding_dim: int,
        dtype: str,
        device: str,
    ) -> tuple[nn.Module, object]:
        # Late import so `import lattice_lab.protein` does not require transformers
        # (the dep is real, but the import error is much clearer this way).
        from transformers import AutoModel, AutoTokenizer
        logger.info("loading ESM-2 weights: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=_torch_dtype(dtype)
        ).to(device)
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
        Sequences longer than ``max_length`` are truncated.
        """
        if not seqs:
            return torch.empty(0, self.embedding_dim, dtype=torch.float32)
        seqs = [_validate_sequence(s) for s in seqs]
        enc = self.tokenizer(
            list(seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
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
            max_length=self.max_length,
            add_special_tokens=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        h = self._forward_tokens(enc)[0]  # [L, D]
        mask = _strip_special_tokens(enc["attention_mask"].to(h.dtype))[0]  # [L]
        return h[mask.bool()].detach().to(torch.float32).cpu()


class ESMCEncoder:
    """Frozen ESM C (Cambrian) wrapper, mirroring :class:`ProteinEncoder`'s API.

    Exposes the same ``embed_batch`` / ``embed_protein`` / ``embed_per_residue``
    surface so :mod:`lattice_lab.protein.precompute` can drive either backend.
    Uses EvolutionaryScale's ``esm`` SDK; the per-residue embeddings it returns
    include the leading BOS and trailing EOS tokens, which we strip so the
    mean-pool matches the ESM-2 path (mean over true residues only).

    The SDK's public path is per-protein (``encode`` → ``logits``); since Stage 3
    is a one-shot precompute that path is fast enough and far more stable than
    poking the model's internal batched forward, so ``embed_batch`` just loops.
    """

    def __init__(
        self,
        *,
        model_name: str = ESMC_DEFAULT_MODEL,
        embedding_dim: int = ESMC_DEFAULT_DIM,
        max_length: int = ESMC_MAX_LEN_DEFAULT,
        dtype: str = "float32",
        device: str = "cpu",
        per_residue: bool = False,
        client: object | None = None,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.dtype = dtype
        self.device = device
        self.per_residue = per_residue
        if client is None:
            client = self._load_pretrained(model_name=model_name, device=device)
        self.client = client
        for p in self.client.parameters():  # type: ignore[attr-defined]
            p.requires_grad_(False)
        self.client.eval()  # type: ignore[attr-defined]
        # Best-effort hidden-size check: guards against pointing at a variant
        # whose dim doesn't match ``embedding_dim``. Skipped if unreadable.
        dim = getattr(getattr(self.client, "config", object()), "hidden_size", None)
        if dim is None:
            dim = getattr(self.client, "d_model", None)
        if dim is not None and dim != embedding_dim:
            raise ValueError(
                f"ESM C hidden size {dim} does not match embedding_dim="
                f"{embedding_dim}; pass the matching --embedding-dim."
            )

    @staticmethod
    def _load_pretrained(*, model_name: str, device: str) -> object:
        from esm.models.esmc import ESMC

        return ESMC.from_pretrained(model_name).to(device)

    @torch.no_grad()
    def _embed_one(self, seq: str) -> torch.Tensor:
        """One sequence → ``[L+2, D]`` CPU float32 (includes BOS/EOS rows)."""
        from esm.sdk.api import ESMProtein, LogitsConfig

        seq = _validate_sequence(seq)[: self.max_length]
        protein = ESMProtein(sequence=seq)
        tensor = self.client.encode(protein)  # type: ignore[attr-defined]
        out = self.client.logits(  # type: ignore[attr-defined]
            tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        return out.embeddings[0].detach().to(torch.float32).cpu()

    @torch.no_grad()
    def embed_per_residue(self, seq: str) -> torch.Tensor:
        """Encode one sequence → ``[L, D]`` CPU tensor (BOS/EOS stripped)."""
        return self._embed_one(seq)[1:-1].contiguous()

    @torch.no_grad()
    def embed_protein(self, seq: str) -> torch.Tensor:
        """Encode one sequence → ``[D]`` CPU tensor, mean-pooled over residues."""
        return self.embed_per_residue(seq).mean(dim=0)

    @torch.no_grad()
    def embed_batch(self, seqs: Sequence[str]) -> torch.Tensor:
        """Encode a list of sequences → ``[N, D]`` CPU tensor (mean-pooled)."""
        if not seqs:
             raise ValueError("no sequences to embed")
        return torch.stack([self.embed_protein(s) for s in seqs], dim=0)


def build_protein_encoder(
    backend: str = "esm2",
    **kwargs: object,
) -> ProteinEncoder | ESMCEncoder:
    """Construct the protein encoder for ``backend`` (``esm2`` or ``esmc``).

    Both encoders share the ``embed_batch`` / ``embed_protein`` /
    ``embed_per_residue`` surface and the same constructor keywords
    (``model_name``, ``embedding_dim``, ``max_length``, ``dtype``, ``device``,
    ``per_residue``), so callers can stay backend-agnostic.
    """
    b = backend.lower()
    if b == "esm2":
        return ProteinEncoder(**kwargs)  # type: ignore[arg-type]
    if b in ("esmc", "esm-c", "esm_c", "esm3"):
        return ESMCEncoder(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown protein backend {backend!r}; expected 'esm2' or 'esmc'")


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
