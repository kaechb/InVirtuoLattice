"""Discrete-flow (DDiT) SMILES backbone for LATTICE.

Wraps the pretrained ``DDiT`` discrete-flow model from InVirtuoFM/InVirtuoGEN and
exposes a molecule encoder for the rest of the pipeline (adapter SSL, decoy
precompute, EBM head):

    DDiT → block hidden states [start..end] → Adapter → z_m  [B, d_adapter]

(the DDiT backbone is frozen or trainable per ``freeze_backbone``)

It additionally supports the model's *own* discrete-flow pretraining objective
(:meth:`DiscreteFlowEncoder.discrete_flow_loss`), so the backbone can either be
loaded from a pretrained state dict or trained from scratch.

Design notes
------------
* This module deliberately depends only on ``in_virtuo_gen``'s ``DDiT`` class
  and a ``tokenizers``/``transformers`` SMILES tokenizer. The discrete-flow
  corruption math is inlined (a few lines) to avoid pulling the heavier
  InVirtuoFM Lightning stack.
* Hidden states are captured with forward hooks on ``DDiT.blocks`` (see
  :class:`_BlockHiddenCollector`), feeding the trainable
  :class:`lattice_lab.backbone.adapter.Adapter` (``d_backbone`` is the DDiT
  ``hidden_size``). This same encode path is used by Stage-2 SSL training,
  inference (:mod:`lattice_lab.eval.encode_utils`), and tests — there is no
  separate encoder implementation.
* Token preprocessing matches the InVirtuoFM pretrain convention: drop the
  leading BOS and treat EOS as PAD for the backbone.

CLI::

    # load pretrained DDiT and encode a couple of SMILES
    python -m lattice_lab.backbone.discrete_flow \
        --ckpt /path/to/invirtuo_gen.ckpt \
        --tokenizer /path/to/smiles_new.json \
        --smiles "CCO" "c1ccccc1"

    # build a fresh DDiT and run one discrete-flow training step
    python -m lattice_lab.backbone.discrete_flow \
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

from lattice_lab.backbone.adapter import Adapter
from lattice_lab.backbone.ddit.model_ddit import DDiT

__all__ = [
    "DiscreteFlowBundle",
    "DiscreteFlowEncoder",
    "load_ddit",
    "load_discrete_flow",
    "build_discrete_flow_encoder",
]


def _n_backbone_layers(layer_start: int, layer_end: int) -> int:
    return int(layer_end) - int(layer_start) + 1


def _validate_backbone_layer_range(n_blocks: int, layer_start: int, layer_end: int) -> None:
    if layer_start < 0 or layer_end >= n_blocks or layer_start > layer_end:
        raise ValueError(
            f"invalid backbone layer range [{layer_start}, {layer_end}] "
            f"for DDiT with {n_blocks} blocks (0-indexed, inclusive)"
        )


def _strip_module_prefix(state: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for k, v in state.items():
        key = k[len("model.") :] if k.startswith("model.") else k
        out[key] = v
    return out


def load_ddit(
    *,
    ckpt_path: Optional[str],
    vocab_size: int,
    n_layer: int = 12,
    n_head: int = 12,
    n_embd: int = 768,
    dropout: float = 0.1,
    n_conds: int = 0,
    force_n_conds: bool = False,
    map_location: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a ``DDiT`` and either load a checkpoint or return a fresh model.

    ``vocab_size`` is only used for the fresh build; when a checkpoint is given
    the vocab/hidden dims are derived from ``model.vocab_embed.weight`` (the
    saved ``hyper_parameters.vocab_size`` in InVirtuoFM checkpoints is
    unreliable). Fresh-build arch kwargs are ignored when ``ckpt_path`` is set,
    **except** ``n_conds`` when ``force_n_conds`` is set: the requested
    conditioning width is then used regardless of the checkpoint (the missing
    ``conds.*`` weights stay freshly initialized via ``strict=False``). This
    lets a conditional denoiser warm-start from an unconditional pretrained DDiT.
    """

    if not ckpt_path:
        model = DDiT(
            vocab_size=int(vocab_size),
            hidden_size=int(n_embd),
            n_heads=int(n_head),
            n_layer=int(n_layer),
            dropout=float(dropout),
            n_conds=int(n_conds),
        )
        meta = {
            "from_checkpoint": False,
            "vocab_size": int(vocab_size),
            "hidden_size": int(n_embd),
            "n_layer": int(n_layer),
        }
        return model, meta

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    h = ckpt.get("hyper_parameters") or {}
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"invalid checkpoint state_dict in {ckpt_path!r}")

    embed_w = state.get("model.vocab_embed.weight", state.get("vocab_embed.weight"))
    if embed_w is not None:
        vocab = int(embed_w.shape[0])
        n_embd_ckpt = int(embed_w.shape[1])
    else:
        vocab = int(h.get("vocab_size") or vocab_size)
        n_embd_ckpt = int(h.get("n_embd") or h.get("hidden_size") or n_embd)

    n_layer_ckpt = int(h.get("n_layer", h.get("num_layers", n_layer)))
    n_head_ckpt = int(h.get("n_head", h.get("num_heads", n_head)))
    n_conds_ckpt = int(n_conds) if force_n_conds else int(h.get("n_conds", n_conds))
    dropout_ckpt = float(h.get("dropout", dropout))

    model = DDiT(
        vocab_size=vocab,
        hidden_size=n_embd_ckpt,
        n_heads=n_head_ckpt,
        n_layer=n_layer_ckpt,
        dropout=dropout_ckpt,
        n_conds=n_conds_ckpt,
    )
    missing, unexpected = model.load_state_dict(_strip_module_prefix(state), strict=False)
    meta = {
        "from_checkpoint": True,
        "vocab_size": vocab,
        "hidden_size": n_embd_ckpt,
        "n_layer": n_layer_ckpt,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return model, meta


# --------------------------------------------------------------------------- #
# Bundle
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


def _special_id(tok: PreTrainedTokenizerFast, token: str) -> int:
    ids = tok.encode(token, add_special_tokens=False)
    if not ids:
        raise ValueError(f"tokenizer has no id for {token!r}")
    return int(ids[0])


def resolve_mask_token_id(
    tokenizer: PreTrainedTokenizerFast,
    *,
    override: int | None = None,
) -> int:
    """Token id for LeJEPA local masked views (never PAD)."""
    if override is not None:
        return int(override)
    for tok in ("<redacted_MASK>", "<MASK>", "<mask>"):
        try:
            return _special_id(tokenizer, tok)
        except ValueError:
            continue
    return _special_id(tokenizer, "[UNK]")


def load_discrete_flow(
    *,
    ckpt_path: Optional[str],
    tokenizer_path: str,
    freeze_backbone: bool,
    n_layer: int = 12,
    n_head: int = 12,
    n_embd: int = 768,
    dropout: float = 0.1,
    n_conds: int = 0,
    device: str | torch.device = "cpu",
) -> DiscreteFlowBundle:
    """Load tokenizer + DDiT (pretrained or fresh)."""
    tok_path = Path(tokenizer_path)
    if not tok_path.is_file():
        raise FileNotFoundError(f"tokenizer_path={tokenizer_path!r} is not a file")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tok_path))

    pad = _special_id(tokenizer, "[PAD]")
    bos = _special_id(tokenizer, "[BOS]")
    eos = _special_id(tokenizer, "[EOS]")

    model, meta = load_ddit(
        ckpt_path=ckpt_path,
        vocab_size=len(tokenizer),
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        n_conds=n_conds,
    )
    model.to(device)
    if freeze_backbone:
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
    """Forward hooks on DDiT blocks ``[layer_start, layer_end]`` → ``[B, T, L*d]``."""

    def __init__(self, model: nn.Module, layer_start: int, layer_end: int) -> None:
        self.model = model
        self.layer_start = int(layer_start)
        self.layer_end = int(layer_end)
        self.n_layers = _n_backbone_layers(self.layer_start, self.layer_end)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._buffer: list[Tensor] = []

    def __enter__(self) -> "_BlockHiddenCollector":
        blocks = list(self.model.blocks)  # type: ignore[attr-defined]
        _validate_backbone_layer_range(len(blocks), self.layer_start, self.layer_end)
        for b in blocks[self.layer_start : self.layer_end + 1]:
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
# Encoder
# --------------------------------------------------------------------------- #
class DiscreteFlowEncoder(nn.Module):
    """``DDiT → block hiddens [start..end] → Adapter → z_m`` (backbone frozen or
    trainable per ``freeze_backbone`` at build time).

    ``encode_token_ids`` / ``encode_views`` / ``encode_molecule`` return the
    L2-normalized molecule latent ``z_m`` (optionally with the SimCLR
    projection). ``encode_views`` accepts **SMILES strings** (the discrete-flow
    model is a SMILES model — for contrastive SSL pass two augmented/randomized
    SMILES of the same molecule).
    """

    def __init__(
        self,
        bundle: DiscreteFlowBundle,
        *,
        backbone_layer_start: int,
        backbone_layer_end: int,
        encode_time: float,
        learnable_time: bool,
        token_id_min: int,
        adapter: Adapter | None = None,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        # Filled in by ``build_discrete_flow_encoder`` so checkpoints can embed the
        # exact skeleton kwargs (``on_save_checkpoint``). ``None`` for encoders built
        # directly (e.g. unit tests), in which case loaders fall back to defaults.
        self.build_config: dict | None = None
        self.backbone_layer_start = int(backbone_layer_start)
        self.backbone_layer_end = int(backbone_layer_end)
        self.encode_time = float(encode_time)
        self.learnable_time = bool(learnable_time)
        self.token_id_min = int(token_id_min)
        n_backbone_layers = _n_backbone_layers(
            self.backbone_layer_start, self.backbone_layer_end
        )
        _validate_backbone_layer_range(
            bundle.n_layer, self.backbone_layer_start, self.backbone_layer_end
        )
        self.backbone = bundle.model
        self.adapter = adapter or Adapter(
            d_backbone=bundle.n_embd,
            n_backbone_layers=n_backbone_layers,
        )
        if self.adapter.d_backbone != bundle.n_embd:
            raise AssertionError(
                f"adapter d_backbone={self.adapter.d_backbone} != DDiT hidden={bundle.n_embd}"
            )
        if self.adapter.n_backbone_layers != n_backbone_layers:
            raise AssertionError(
                f"adapter n_backbone_layers={self.adapter.n_backbone_layers} "
                f"!= backbone layers {n_backbone_layers}"
            )
        # DDiT.forward gained ``return_post_hidden`` in the JEPA fork; detect it
        # so we can skip the output projection when only encoding.
        self._supports_post_hidden = "return_post_hidden" in inspect.signature(
            self.backbone.forward
        ).parameters

        # Optional learnable encode-time (sigmoid-bounded to (0, 1)).
        self._learnable_time = self.learnable_time
        if self._learnable_time:
            t0 = min(max(self.encode_time, 1e-4), 1.0 - 1e-4)
            self.time_logit = nn.Parameter(torch.logit(torch.tensor(float(t0))))

    @property
    def encode_time_value(self) -> float:
        """Current encode time as a plain float (for logging)."""
        if self._learnable_time:
            return float(torch.sigmoid(self.time_logit.detach()))
        return float(self.encode_time)

    # -- low-level ---------------------------------------------------------- #
    def _attn_mask(self, x: Tensor, *, hole_mask: Tensor | None = None) -> Tensor:
        """Additive ``[B, 1, L, L]`` mask: -inf where the *key* is PAD or a hole."""
        b, length = x.shape
        valid = (x != self.bundle.pad_id) & (x != self.bundle.bos_id) & (x != self.bundle.eos_id)

        block = (~valid).unsqueeze(1).expand(b, length, length)
        if hole_mask is not None:
            if hole_mask.shape != (b, length):
                raise ValueError(
                    f"hole_mask must be [B,T]=({b},{length}), got {tuple(hole_mask.shape)}"
                )
            block = block | hole_mask.unsqueeze(1).expand(b, length, length)
        return block.float().masked_fill(block, float("-inf")).unsqueeze(1)

    def _build_time(self, batch: int, device: torch.device) -> Tensor:
        if self._learnable_time:
            # Keep the graph so d(loss)/d(time_logit) flows through the backbone.
            return torch.sigmoid(self.time_logit).to(device).expand(batch)
        return torch.full(
            (batch,), float(self.encode_time), device=device, dtype=torch.float32
        )

    def _backbone_hidden(self, x: Tensor, *, hole_mask: Tensor | None = None) -> Tensor:
        """Run DDiT once on clean tokens; return concat hiddens ``[B,T,L*d]``."""
        attn = self._attn_mask(x, hole_mask=hole_mask)
        t = self._build_time(x.size(0), x.device)
        kwargs: dict[str, Any] = {"attn_mask": attn, "conds": None}
        if self._supports_post_hidden:
            kwargs["return_post_hidden"] = True
        with _BlockHiddenCollector(
            self.backbone, self.backbone_layer_start, self.backbone_layer_end
        ) as col:
            self.backbone(x, t, **kwargs)
            return col.stack()

    # -- public API -------------------------------------------------------- #
    def encode_token_ids(
        self,
        ids: Tensor,
        mask: Tensor,
        *,
        return_projection: bool = False,
        normalize: bool = True,
        return_tokens: bool = False,
        hole_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Encode token ids to ``z_m`` (see :meth:`Adapter.forward`).

        ``return_tokens`` returns ``(z_m, x)`` with per-token reps ``x`` aligned
        to the *post*-:func:`prepare_backbone_tokens` frame — i.e. the leading
        BOS is dropped, so column ``i`` of ``x`` corresponds to input column
        ``i + 1``. Callers pooling specific token spans must account for that.

        ``hole_mask`` ``[B,T]`` bool (post-BOS frame): when set, hole positions
        are blocked as attention *keys* in DDiT and the adapter so visible context
        reps cannot attend to ``<UNK>`` slots (I-JEPA context-only encoding).
        """
        x, m = prepare_backbone_tokens(
            ids, mask, bos_id=self.bundle.bos_id, eos_id=self.bundle.eos_id, pad_id=self.bundle.pad_id
        )
        if hole_mask is not None:
            hole_mask = hole_mask.to(device=x.device, dtype=torch.bool)
            if hole_mask.shape != x.shape:
                raise ValueError(
                    f"hole_mask must match post-BOS ids shape {tuple(x.shape)}, "
                    f"got {tuple(hole_mask.shape)}"
                )
        # Track the backbone graph when its params need grad OR when the encode
        # time is learnable (gradient must reach time_logit through the backbone).
        grad = self._learnable_time or any(p.requires_grad for p in self.backbone.parameters())
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            hs = self._backbone_hidden(x, hole_mask=hole_mask)
        return self.adapter(
            hs, m.to(hs.dtype), return_projection=return_projection,
            normalize=normalize, return_tokens=return_tokens, hole_mask=hole_mask,
        )

    def encode_views(
        self,
        views: Sequence[str],
        device: torch.device | str = "cpu",
        *,
        return_projection: bool = False,
        normalize: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor]:
        seqs = [encode_smiles(self.bundle, v) for v in views]
        ids, mask = pad_batch(seqs, pad_id=self.bundle.pad_id)
        return self.encode_token_ids(
            ids.to(device),
            mask.to(device),
            return_projection=return_projection,
            normalize=normalize,
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
            int(self.token_id_min), int(self.bundle.vocab_size), x.shape, device=x.device
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
    *,
    ckpt_path: Optional[str],
    tokenizer_path: str,
    backbone_layer_start: int,
    backbone_layer_end: int,
    d_adapter: int,
    adapter_n_layers: int,
    encode_time: float,
    learnable_time: bool,
    freeze_backbone: bool,
    token_id_min: int = 4,
    n_layer: int = 12,
    n_head: int = 12,
    n_embd: int = 768,
    dropout: float = 0.1,
    n_conds: int = 0,
    device: str | torch.device = "cpu",
) -> DiscreteFlowEncoder:
    """Hydra entrypoint: load DDiT + adapter and return a :class:`DiscreteFlowEncoder`.

    Every argument is explicit in ``configs/model/discrete_flow.yaml`` so there
    are no silent Python defaults on the training path. Fresh-build arch kwargs
    apply only when ``ckpt_path`` is null.
    """
    n_backbone_layers = _n_backbone_layers(backbone_layer_start, backbone_layer_end)
    bundle = load_discrete_flow(
        ckpt_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        freeze_backbone=freeze_backbone,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        n_conds=n_conds,
        device=device,
    )
    adapter = Adapter(
        d_backbone=bundle.n_embd,
        n_backbone_layers=n_backbone_layers,
        d_adapter=d_adapter,
        n_layers=adapter_n_layers,
    )
    enc = DiscreteFlowEncoder(
        bundle,
        backbone_layer_start=backbone_layer_start,
        backbone_layer_end=backbone_layer_end,
        encode_time=encode_time,
        learnable_time=learnable_time,
        token_id_min=token_id_min,
        adapter=adapter,
    )
    enc.adapter.to(device)
    # Stash the exact skeleton kwargs so a checkpoint can be made self-describing
    # (see EBM/SSL ``on_save_checkpoint``). The hook layer range is *not* derivable
    # from weights, so it must travel with the ckpt to avoid train/serve skew.
    enc.build_config = {
        "tokenizer_path": tokenizer_path,
        "backbone_layer_start": int(backbone_layer_start),
        "backbone_layer_end": int(backbone_layer_end),
        "d_adapter": int(d_adapter),
        "adapter_n_layers": int(adapter_n_layers),
        "token_id_min": int(token_id_min),
        "n_layer": int(bundle.n_layer),
        "n_head": int(n_head),
        "n_embd": int(bundle.n_embd),
        "dropout": float(dropout),
        "n_conds": int(n_conds),
    }
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
    ap.add_argument(
        "--backbone-layer-start",
        type=int,
        default=8,
        help="first DDiT block index to hook (0-indexed, inclusive)",
    )
    ap.add_argument(
        "--backbone-layer-end",
        type=int,
        default=11,
        help="last DDiT block index to hook (0-indexed, inclusive)",
    )
    ap.add_argument("--train-step", action="store_true", help="run one discrete-flow training step")
    args = ap.parse_args()

    enc = build_discrete_flow_encoder(
        ckpt_path=None if args.from_scratch else args.ckpt,
        tokenizer_path=args.tokenizer,
        backbone_layer_start=args.backbone_layer_start,
        backbone_layer_end=args.backbone_layer_end,
        d_adapter=512,
        adapter_n_layers=4,
        encode_time=0.5,
        learnable_time=False,
        freeze_backbone=not args.train_step,
        device=args.device,
    )

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


def sync_encoder_device(
    encoder: DiscreteFlowEncoder,
    device: str | torch.device,
    *,
    head: nn.Module | None = None,
) -> None:
    """Align backbone + adapter (+ optional head) after ``Trainer.fit`` teardown."""
    dev = torch.device(device)
    encoder.adapter.to(dev)
    encoder.backbone.to(dev)
    if head is not None:
        head.to(dev)


if __name__ == "__main__":
    _main()
