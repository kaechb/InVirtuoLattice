"""Discrete-flow (DDiT) SMILES backbone for LATTICE.

A minimal, self-contained alternative to the FragMol backbone
(``lattice/backbone/fragmol_loader.py`` + ``encoder.py``). It wraps the
pretrained ``DDiT`` discrete-flow model from InVirtuoFM/InVirtuoGEN and exposes
the *same* encoder contract as :class:`lattice.backbone.encoder.MoleculeEncoder`
so the rest of the pipeline (Stage-2 adapter SSL, decoy precompute, EBM head)
plugs in unchanged:

    DDiT (frozen) → last-L block hidden states → Adapter → z_m  [B, d_adapter]

It additionally supports the model's *own* discrete-flow pretraining objective
(:meth:`DiscreteFlowEncoder.discrete_flow_loss`), so the backbone can either be
loaded from a pretrained state dict or trained from scratch.

Design notes
------------
* This module deliberately depends only on ``in_virtuo_gen``'s ``DDiT`` class
  and a ``tokenizers``/``transformers`` SMILES tokenizer. The discrete-flow
  corruption math is inlined (a few lines) to avoid pulling the heavier
  InVirtuoFM Lightning stack.
* Hidden states are captured with forward hooks on ``DDiT.blocks`` — identical
  in spirit to :class:`lattice.backbone.encoder.HiddenStateCollector`, so the
  trainable :class:`lattice.backbone.adapter.Adapter` is reused verbatim
  (``d_fragmol`` simply becomes the DDiT ``hidden_size``).
* Token preprocessing matches the InVirtuoFM pretrain convention: drop the
  leading BOS and treat EOS as PAD for the backbone.

CLI::

    # load pretrained DDiT and encode a couple of SMILES
    python -m lattice.backbone.discrete_flow \
        --ckpt /path/to/invirtuo_gen.ckpt \
        --tokenizer /path/to/smiles_new.json \
        --smiles "CCO" "c1ccccc1"

    # build a fresh DDiT and run one discrete-flow training step
    python -m lattice.backbone.discrete_flow \
        --from-scratch --tokenizer /path/to/smiles_new.json --train-step
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerFast

from lattice_lab.backbone.adapter import Adapter, AdapterConfig

__all__ = [
    "DiscreteFlowConfig",
    "DiscreteFlowBundle",
    "DiscreteFlowEncoder",
    "load_ddit",
    "load_discrete_flow",
    "build_discrete_flow_encoder",
]


# --------------------------------------------------------------------------- #
# Config + checkpoint loading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiscreteFlowConfig:
    """Everything needed to build / load a DDiT backbone bundle.

    ``ckpt_path=None`` builds a fresh DDiT (train-from-scratch); otherwise the
    architecture + weights are read from the checkpoint and ``n_layer`` /
    ``n_embd`` below are ignored.
    """

    ckpt_path: Optional[str] = None
    tokenizer_path: str = "tokenizer/smiles_new.json"
    # Fresh-build architecture (ignored when ``ckpt_path`` is set).
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    n_conds: int = 0
    # Number of trailing DDiT blocks whose hidden states feed the adapter.
    n_backbone_layers: int = 4
    # Flow time fed to the backbone when *encoding* clean tokens (t=1 is clean).
    encode_time: float = 0.5
    # Lowest non-special token id used for the uniform-noise source.
    token_id_min: int = 4
    freeze_backbone: bool = True


def _strip_module_prefix(state: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for k, v in state.items():
        key = k[len("model.") :] if k.startswith("model.") else k
        out[key] = v
    return out


def load_ddit(
    cfg: DiscreteFlowConfig,
    *,
    vocab_size: int,
    map_location: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a ``DDiT`` and either load a checkpoint or return a fresh model.

    ``vocab_size`` is only used for the fresh build; when a checkpoint is given
    the vocab/hidden dims are derived from ``model.vocab_embed.weight`` (the
    saved ``hyper_parameters.vocab_size`` in InVirtuoFM checkpoints is
    unreliable).
    """
    from in_virtuo_gen.models.transformer.model_ddit import DDiT

    if not cfg.ckpt_path:
        model = DDiT(
            vocab_size=int(vocab_size),
            hidden_size=int(cfg.n_embd),
            n_heads=int(cfg.n_head),
            n_layer=int(cfg.n_layer),
            dropout=float(cfg.dropout),
            n_conds=int(cfg.n_conds),
        )
        meta = {
            "from_checkpoint": False,
            "vocab_size": int(vocab_size),
            "hidden_size": int(cfg.n_embd),
            "n_layer": int(cfg.n_layer),
        }
        return model, meta

    ckpt = torch.load(cfg.ckpt_path, map_location=map_location, weights_only=False)
    h = ckpt.get("hyper_parameters") or {}
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"invalid checkpoint state_dict in {cfg.ckpt_path!r}")

    embed_w = state.get("model.vocab_embed.weight", state.get("vocab_embed.weight"))
    if embed_w is not None:
        vocab = int(embed_w.shape[0])
        n_embd = int(embed_w.shape[1])
    else:
        vocab = int(h.get("vocab_size") or vocab_size)
        n_embd = int(h.get("n_embd") or h.get("hidden_size") or cfg.n_embd)

    n_layer = int(h.get("n_layer", h.get("num_layers", cfg.n_layer)))
    n_head = int(h.get("n_head", h.get("num_heads", cfg.n_head)))
    n_conds = int(h.get("n_conds", cfg.n_conds))
    dropout = float(h.get("dropout", cfg.dropout))

    model = DDiT(
        vocab_size=vocab,
        hidden_size=n_embd,
        n_heads=n_head,
        n_layer=n_layer,
        dropout=dropout,
        n_conds=n_conds,
    )
    missing, unexpected = model.load_state_dict(_strip_module_prefix(state), strict=False)
    meta = {
        "from_checkpoint": True,
        "vocab_size": vocab,
        "hidden_size": n_embd,
        "n_layer": n_layer,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return model, meta


# --------------------------------------------------------------------------- #
# Bundle (analogous to FragMolBundle)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiscreteFlowBundle:
    """Loaded DDiT artifacts grouped for ergonomic passing."""

    model: nn.Module
    tokenizer: PreTrainedTokenizerFast
    n_embd: int
    n_layer: int
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int


def _special_id(tok: PreTrainedTokenizerFast, token: str, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    ids = tok.encode(token, add_special_tokens=False)
    if not ids:
        raise ValueError(f"tokenizer has no id for {token!r}")
    return int(ids[0])


def load_discrete_flow(
    cfg: DiscreteFlowConfig | None = None,
    *,
    device: str | torch.device = "cpu",
    pad_id: Optional[int] = None,
    bos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> DiscreteFlowBundle:
    """Load tokenizer + DDiT (pretrained or fresh) into a frozen-by-default bundle."""
    cfg = cfg or DiscreteFlowConfig()
    tok_path = Path(cfg.tokenizer_path)
    if not tok_path.is_file():
        raise FileNotFoundError(f"tokenizer_path={cfg.tokenizer_path!r} is not a file")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tok_path))

    pad = _special_id(tokenizer, "[PAD]", pad_id)
    bos = _special_id(tokenizer, "[BOS]", bos_id)
    eos = _special_id(tokenizer, "[EOS]", eos_id)

    model, meta = load_ddit(cfg, vocab_size=len(tokenizer))
    model.to(device)
    if cfg.freeze_backbone:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    src = "checkpoint" if meta.get("from_checkpoint") else "fresh"
    n_params = sum(p.numel() for p in model.parameters())
    extra = ""
    if meta.get("from_checkpoint"):
        extra = (
            f", missing={len(meta.get('missing_keys', []))}"
            f", unexpected={len(meta.get('unexpected_keys', []))}"
        )
    print(
        f"[DiscreteFlow] DDiT ({src}): {n_params / 1e6:.1f}M params, "
        f"vocab={meta['vocab_size']}, hidden={meta['hidden_size']}, "
        f"n_layer={meta['n_layer']}{extra}",
        flush=True,
    )

    return DiscreteFlowBundle(
        model=model,
        tokenizer=tokenizer,
        n_embd=int(meta["hidden_size"]),
        n_layer=int(meta["n_layer"]),
        vocab_size=int(meta["vocab_size"]),
        pad_id=pad,
        bos_id=bos,
        eos_id=eos,
    )


# --------------------------------------------------------------------------- #
# Tokenization helpers
# --------------------------------------------------------------------------- #
def encode_smiles(bundle: DiscreteFlowBundle, smiles: str) -> list[int]:
    """SMILES → ``[BOS] body [EOS]`` token ids (no padding)."""
    body = bundle.tokenizer.encode(smiles, add_special_tokens=False)
    return [bundle.bos_id, *body, bundle.eos_id]


def pad_batch(
    sequences: Sequence[Sequence[int]],
    *,
    pad_id: int,
    max_len: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Pad to the batch max (or ``max_len``). Returns ``(ids, mask)``.

    ``mask`` is 1 for real tokens (incl. BOS/EOS) and 0 for pad positions.
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


def prepare_backbone_tokens(
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> tuple[Tensor, Tensor]:
    """Match InVirtuoFM pretrain: drop a leading BOS; treat EOS as PAD."""
    x = input_ids.long()
    mask = attention_mask.long().bool()
    if x.size(1) > 0 and bool((x[:, 0] == int(bos_id)).all()):
        x = x[:, 1:]
        mask = mask[:, 1:]
    eos = x == int(eos_id)
    x = x.masked_fill(eos, int(pad_id))
    mask = mask & ~eos
    return x, mask.long()


# --------------------------------------------------------------------------- #
# Discrete-flow corruption (inlined from InVirtuoFM)
# --------------------------------------------------------------------------- #
def _sample_timesteps(k: int, device: torch.device, *, t_cap: float = 1e-3) -> Tensor:
    u0 = torch.rand(1, device=device)
    idx = torch.arange(1, k + 1, dtype=torch.float32, device=device)
    t = (u0 + idx / k) % 1.0
    if t_cap > 0.0:
        t = t * (1.0 - float(t_cap))
    return t


def _sample_path(t: Tensor, x0: Tensor, x1: Tensor, *, n: float = 1.0) -> Tensor:
    sigma_t = 1.0 - t.pow(n)
    src = torch.rand(x1.shape, device=x1.device) < sigma_t.unsqueeze(-1)
    return torch.where(src, x0, x1)


# --------------------------------------------------------------------------- #
# Hidden-state collector (DDiT blocks)
# --------------------------------------------------------------------------- #
class _BlockHiddenCollector:
    """Forward hooks on the last ``n_layers`` DDiT blocks → ``[B, T, L*d]``."""

    def __init__(self, model: nn.Module, n_layers: int) -> None:
        self.model = model
        self.n_layers = int(n_layers)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._buffer: list[Tensor] = []

    def __enter__(self) -> "_BlockHiddenCollector":
        blocks = list(self.model.blocks)  # type: ignore[attr-defined]
        if self.n_layers > len(blocks):
            raise ValueError(
                f"requested {self.n_layers} layers but DDiT has only {len(blocks)}"
            )
        for b in blocks[-self.n_layers :]:
            self._handles.append(
                b.register_forward_hook(lambda _m, _i, out: self._buffer.append(out))
            )
        self._buffer = []
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def stack(self) -> Tensor:
        if len(self._buffer) != self.n_layers:
            raise RuntimeError(
                f"collector saw {len(self._buffer)} layers; expected {self.n_layers}"
            )
        return torch.cat(self._buffer, dim=-1)


# --------------------------------------------------------------------------- #
# Encoder (drop-in for MoleculeEncoder)
# --------------------------------------------------------------------------- #
class DiscreteFlowEncoder(nn.Module):
    """``DDiT (frozen) → last-L block hiddens → Adapter → z_m``.

    API-compatible with :class:`lattice.backbone.encoder.MoleculeEncoder`:
    ``encode_token_ids`` / ``encode_views`` / ``encode_molecule`` return the
    L2-normalized molecule latent ``z_m`` (optionally with the SimCLR
    projection). ``encode_views`` accepts **SMILES strings** (the discrete-flow
    model is a SMILES model — for contrastive SSL pass two augmented/randomized
    SMILES of the same molecule).
    """

    def __init__(
        self,
        bundle: DiscreteFlowBundle,
        adapter: Adapter | None = None,
        config: DiscreteFlowConfig | None = None,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.cfg = config or DiscreteFlowConfig()
        self.backbone = bundle.model
        self.adapter = adapter or Adapter(
            AdapterConfig(
                d_fragmol=bundle.n_embd,
                n_fragmol_layers=self.cfg.n_backbone_layers,
            )
        )
        if self.adapter.cfg.d_fragmol != bundle.n_embd:
            raise AssertionError(
                f"adapter d_fragmol={self.adapter.cfg.d_fragmol} != DDiT hidden={bundle.n_embd}"
            )
        if self.adapter.cfg.n_fragmol_layers != self.cfg.n_backbone_layers:
            raise AssertionError(
                f"adapter n_fragmol_layers={self.adapter.cfg.n_fragmol_layers} "
                f"!= n_backbone_layers={self.cfg.n_backbone_layers}"
            )
        # DDiT.forward gained ``return_post_hidden`` in the JEPA fork; detect it
        # so we can skip the output projection when only encoding.
        self._supports_post_hidden = "return_post_hidden" in inspect.signature(
            self.backbone.forward
        ).parameters

    # -- low-level ---------------------------------------------------------- #
    def _attn_mask(self, x: Tensor) -> Tensor:
        """Additive ``[B, 1, L, L]`` mask: -inf where the *key* is PAD."""
        b, length = x.shape
        valid = x != self.bundle.pad_id
        block = (~valid).unsqueeze(1).expand(b, length, length)
        return block.float().masked_fill(block, float("-inf")).unsqueeze(1)

    def _build_time(self, batch: int, device: torch.device, t: float) -> Tensor:
        return torch.full((batch,), float(t), device=device, dtype=torch.float32)

    def _backbone_hidden(self, x: Tensor) -> Tensor:
        """Run DDiT once on clean tokens; return last-L concat hiddens ``[B,T,L*d]``."""
        attn = self._attn_mask(x)
        t = self._build_time(x.size(0), x.device, self.cfg.encode_time)
        kwargs: dict[str, Any] = {"attn_mask": attn, "conds": None}
        if self._supports_post_hidden:
            kwargs["return_post_hidden"] = True
        with _BlockHiddenCollector(self.backbone, self.cfg.n_backbone_layers) as col:
            self.backbone(x, t, **kwargs)
            return col.stack()

    # -- public API (mirrors MoleculeEncoder) ------------------------------- #
    def encode_token_ids(
        self,
        ids: Tensor,
        mask: Tensor,
        *,
        return_projection: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        x, m = prepare_backbone_tokens(
            ids, mask, bos_id=self.bundle.bos_id, eos_id=self.bundle.eos_id, pad_id=self.bundle.pad_id
        )
        grad = any(p.requires_grad for p in self.backbone.parameters())
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            hs = self._backbone_hidden(x)
        return self.adapter(hs, m.to(hs.dtype), return_projection=return_projection)

    def encode_views(
        self,
        views: Sequence[str],
        device: torch.device | str = "cpu",
        *,
        return_projection: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        seqs = [encode_smiles(self.bundle, v) for v in views]
        ids, mask = pad_batch(seqs, pad_id=self.bundle.pad_id)
        return self.encode_token_ids(
            ids.to(device), mask.to(device), return_projection=return_projection
        )

    def encode_molecule(
        self, smiles: str, device: torch.device | str = "cpu"
    ) -> Tensor:
        return self.encode_views([smiles], device=device)  # type: ignore[return-value]

    # -- discrete-flow pretraining (train the backbone) --------------------- #
    def discrete_flow_loss(
        self,
        ids: Tensor,
        mask: Tensor,
        *,
        path_power: float = 1.0,
        t_cap: float = 1e-3,
        weight_eps: float = 1e-3,
    ) -> Tensor:
        """Uniform discrete-flow CE (the InVirtuoFM pretraining objective).

        ``x_0 ~ Uniform[token_id_min, vocab)``; ``x_t = path(t, x_0, x_clean)``;
        token CE to clean targets weighted by ``1/(1 - t^2)``. Backprops into the
        DDiT backbone, so use a non-frozen bundle to train from scratch.
        """
        x, m = prepare_backbone_tokens(
            ids, mask, bos_id=self.bundle.bos_id, eos_id=self.bundle.eos_id, pad_id=self.bundle.pad_id
        )
        b = x.size(0)
        valid = x != self.bundle.pad_id
        t = _sample_timesteps(b, x.device, t_cap=t_cap)
        x0 = torch.randint(
            int(self.cfg.token_id_min), int(self.bundle.vocab_size), x.shape, device=x.device
        ).masked_fill(~valid, self.bundle.pad_id)
        x_t = _sample_path(t, x0, x, n=float(path_power))
        attn = self._attn_mask(x_t)
        t_in = t.to(torch.float32)
        out = self.backbone(x_t, t_in, attn_mask=attn, conds=None)
        logits = out[0] if isinstance(out, tuple) else out
        targets = x.masked_fill(~valid, self.bundle.pad_id)
        ce = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), targets, reduction="none", ignore_index=self.bundle.pad_id
        )
        weights = 1.0 / ((1.0 - t.float().pow(2)) + float(weight_eps))
        denom = valid.sum(dim=1).clamp(min=1)
        per_seq = (ce.float() * valid.float()).sum(dim=1) / denom.float()
        return (per_seq * weights.float()).mean()


def build_discrete_flow_encoder(
    cfg: DiscreteFlowConfig | None = None,
    *,
    device: str | torch.device = "cpu",
    adapter: Adapter | None = None,
) -> DiscreteFlowEncoder:
    """One-call factory: load the bundle and wrap it in the encoder."""
    cfg = cfg or DiscreteFlowConfig()
    bundle = load_discrete_flow(cfg, device=device)
    enc = DiscreteFlowEncoder(bundle, adapter=adapter, config=cfg)
    enc.adapter.to(device)
    return enc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None, help="DDiT/InVirtuoFM checkpoint (omit for fresh build)")
    ap.add_argument("--from-scratch", action="store_true", help="build a fresh DDiT (ignore --ckpt)")
    ap.add_argument("--tokenizer", required=True, help="path to SMILES tokenizer json")
    ap.add_argument("--smiles", nargs="*", default=["CCO", "c1ccccc1"], help="SMILES to encode")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-backbone-layers", type=int, default=4)
    ap.add_argument("--train-step", action="store_true", help="run one discrete-flow training step")
    args = ap.parse_args()

    cfg = DiscreteFlowConfig(
        ckpt_path=None if args.from_scratch else args.ckpt,
        tokenizer_path=args.tokenizer,
        n_backbone_layers=args.n_backbone_layers,
        freeze_backbone=not args.train_step,
    )
    enc = build_discrete_flow_encoder(cfg, device=args.device)

    z = enc.encode_views(args.smiles, device=args.device)
    print(f"[DiscreteFlow] encoded {len(args.smiles)} SMILES → z_m {tuple(z.shape)} "
          f"(norm={z.norm(dim=-1).tolist()})")

    if args.train_step:
        seqs = [encode_smiles(enc.bundle, s) for s in args.smiles]
        ids, mask = pad_batch(seqs, pad_id=enc.bundle.pad_id)
        ids, mask = ids.to(args.device), mask.to(args.device)
        opt = torch.optim.AdamW(enc.backbone.parameters(), lr=1e-4)
        enc.backbone.train()
        loss = enc.discrete_flow_loss(ids, mask)
        loss.backward()
        opt.step()
        print(f"[DiscreteFlow] one discrete-flow train step: loss={loss.item():.4f}")


if __name__ == "__main__":
    _main()
