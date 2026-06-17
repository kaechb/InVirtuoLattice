"""Encoder / head builders shared by the LightningModules."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path

import torch

from lattice_lab.backbone.discrete_flow import (
    DiscreteFlowEncoder,
    build_discrete_flow_encoder,
)
from lattice_lab.ebm.head import EnergyHead

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "artifacts/tokenizer/smiles_new.json"
DEFAULT_BACKBONE_LAYER_START = 8
DEFAULT_BACKBONE_LAYER_END = 11

_ENCODER_PREFIX = "encoder."
_ADAPTER_PREFIX = "encoder.adapter."

# Skeleton kwargs used to rebuild an encoder from a *pre-``encoder_config``*
# checkpoint. New checkpoints embed their own ``encoder_config`` (see the
# LightningModules' ``on_save_checkpoint``); this only covers old ckpts and
# matches the architecture every trained model has used so far.
_FALLBACK_ENCODER_CONFIG: dict[str, object] = {
    "tokenizer_path": DEFAULT_TOKENIZER,
    "backbone_layer_start": DEFAULT_BACKBONE_LAYER_START,
    "backbone_layer_end": DEFAULT_BACKBONE_LAYER_END,
    "d_adapter": 512,
    "adapter_n_layers": 4,
    "token_id_min": 4,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "dropout": 0.1,
    "n_conds": 0,
}


def safe_torch_load(path: str | Path, *, weights_only: bool = True) -> dict:
    """``torch.load`` with ``pathlib`` classes allowlisted.

    Old checkpoints (pre-stringified cfg) embedded ``pathlib.PosixPath``; PyTorch
    >= 2.6 refuses those under the default ``weights_only=True``. We allowlist the
    path classes so weights still load strictly. Pass ``weights_only=False`` for
    full Lightning checkpoints that also carry non-tensor objects.
    """
    from pathlib import PosixPath, WindowsPath

    with torch.serialization.safe_globals([PosixPath, WindowsPath]):
        return torch.load(path, map_location="cpu", weights_only=weights_only)


_HEAD_PREFIX = "head."


def _checkpoint_state_dict(raw: object) -> dict[str, torch.Tensor]:
    """Return the ``state_dict`` of a full Lightning checkpoint.

    We only support whole-model Lightning ``.ckpt`` files (what ``ModelCheckpoint``
    writes): ``{"state_dict": {"encoder.*": ..., "head.*": ...}}``. A bare
    ``state_dict`` mapping is also accepted. There are intentionally no legacy
    partial-bundle formats — every stage saves the entire module, so loading is
    a single, unambiguous prefix split.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"checkpoint must be a dict, got {type(raw)}")
    state = raw.get("state_dict", raw)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no 'state_dict'")
    return state


def parse_head_checkpoint(raw: object) -> dict[str, torch.Tensor]:
    """Extract energy-head weights (``head.*``) from a full EBM Lightning ckpt."""
    state = _checkpoint_state_dict(raw)
    head_state = {
        k[len(_HEAD_PREFIX):]: v
        for k, v in state.items()
        if k.startswith(_HEAD_PREFIX)
    }
    if not head_state:
        raise ValueError(
            "no energy-head weights found (expected 'head.*' in a full "
            "Lightning EBM checkpoint)"
        )
    return head_state


def load_energy_head(
    head_ckpt: str | Path,
    *,
    d_adapter: int,
    d_protein: int,
    device: str | torch.device = "cpu",
) -> EnergyHead:
    """Load a trained Stage-5 :class:`EnergyHead` (frozen, ``eval()``).

    ``head_ckpt`` is a full EBM Lightning ``.ckpt``; the head is pulled out of its
    ``state_dict`` by the ``head.`` prefix.
    """
    raw = safe_torch_load(head_ckpt, weights_only=False)
    head = EnergyHead(d_m=d_adapter, d_p=d_protein)
    head.load_state_dict(parse_head_checkpoint(raw))
    head.to(device).eval()
    for p in head.parameters():
        p.requires_grad = False
    logger.info("loaded energy head from %s", head_ckpt)
    return head


def parse_adapter_state(raw: object) -> dict[str, torch.Tensor]:
    """Extract adapter weights (``encoder.adapter.*``) from a full Lightning ckpt.

    Used only for fingerprinting the latent space — both a Stage-2 SSL ckpt and a
    Stage-5 EBM ckpt carry the identical ``encoder.adapter.*`` entries.
    """
    state = _checkpoint_state_dict(raw)
    adapter_state = {
        k[len(_ADAPTER_PREFIX):]: v
        for k, v in state.items()
        if k.startswith(_ADAPTER_PREFIX)
    }
    if not adapter_state:
        raise ValueError(
            "no adapter weights found (expected 'encoder.adapter.*' in a full "
            "Lightning checkpoint)"
        )
    if not all(isinstance(v, torch.Tensor) for v in adapter_state.values()):
        raise ValueError("adapter weights must be tensors")
    return adapter_state


def resolve_adapter_ckpt(adapter_ckpt: str | Path) -> Path:
    """Resolve a checkpoint path (file, run dir, or checkpoints root)."""
    path = Path(adapter_ckpt)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"adapter checkpoint not found: {adapter_ckpt}")

    direct = path / "last.ckpt"
    if direct.is_file():
        return direct

    candidates = sorted(
        path.glob("*/last.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        chosen = candidates[0]
        logger.info("resolved adapter ckpt %s → %s", path, chosen)
        return chosen

    raise FileNotFoundError(
        f"no last.ckpt under {adapter_ckpt}; pass a file or "
        f"{adapter_ckpt}/<wandb_run_id>/last.ckpt"
    )


def adapter_state_fingerprint(adapter_state: Mapping[str, torch.Tensor]) -> str:
    """Stable SHA-1 over adapter weights — a fingerprint of the latent space.

    Two encoders produce comparable ``z_m`` iff their adapters share this hash.
    Used to guard z_m caches against being scored by a mismatched adapter
    (the path string alone is unreliable: the same weights live in both the
    Stage-2 adapter ckpt and every EBM ckpt that froze it).
    """
    h = hashlib.sha1()
    for k in sorted(adapter_state):
        v = adapter_state[k]
        h.update(k.encode())
        h.update(repr(tuple(v.shape)).encode())
        h.update(v.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def adapter_fingerprint(adapter_ckpt: str | Path) -> str:
    """Fingerprint the adapter baked into any Stage-2 / EBM checkpoint."""
    path = resolve_adapter_ckpt(adapter_ckpt)
    raw = safe_torch_load(path, weights_only=False)
    return adapter_state_fingerprint(parse_adapter_state(raw))


def load_encoder_from_ckpt(
    ckpt: str | Path,
    *,
    device: str | torch.device = "cpu",
    **overrides: object,
) -> DiscreteFlowEncoder:
    """Rebuild a frozen :class:`DiscreteFlowEncoder` from a single Lightning ckpt.

    The checkpoint is self-describing: it carries ``encoder_config`` (the exact
    skeleton kwargs used at build time — crucially the DDiT hook layer range,
    which cannot be recovered from the weights) alongside the full ``encoder.*``
    state (backbone + adapter + learnable time). We rebuild a fresh skeleton from
    that config and load every weight from this one file. No base DDiT, no
    per-caller layer range, so the adapter is always served the same layers it
    was trained on.

    Old checkpoints without ``encoder_config`` fall back to
    :data:`_FALLBACK_ENCODER_CONFIG` (override via ``**overrides``).
    """
    path = resolve_adapter_ckpt(ckpt)
    raw = safe_torch_load(path, weights_only=False)
    state = _checkpoint_state_dict(raw)
    enc_state = {
        k[len(_ENCODER_PREFIX):]: v
        for k, v in state.items()
        if k.startswith(_ENCODER_PREFIX)
    }
    if not enc_state:
        raise ValueError(f"no 'encoder.*' weights in checkpoint {path}")

    cfg = dict(_FALLBACK_ENCODER_CONFIG)
    embedded = raw.get("encoder_config") if isinstance(raw, dict) else None
    if embedded:
        cfg.update(embedded)
    else:
        logger.warning(
            "ckpt %s has no 'encoder_config'; rebuilding skeleton from defaults "
            "(layers %s-%s, d_adapter=%s) — retrain to embed it",
            path, cfg["backbone_layer_start"], cfg["backbone_layer_end"], cfg["d_adapter"],
        )
    cfg.update(overrides)

    # A learnable encode-time adds an ``encoder.time_logit`` parameter; the
    # skeleton must match the saved keys for a strict load.
    has_time = "time_logit" in enc_state
    encoder = build_discrete_flow_encoder(
        ckpt_path=None,
        tokenizer_path=str(cfg["tokenizer_path"]),
        backbone_layer_start=int(cfg["backbone_layer_start"]),
        backbone_layer_end=int(cfg["backbone_layer_end"]),
        d_adapter=int(cfg["d_adapter"]),
        adapter_n_layers=int(cfg["adapter_n_layers"]),
        encode_time=0.99,
        learnable_time=has_time,
        freeze_backbone=True,
        token_id_min=int(cfg["token_id_min"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg["dropout"]),
        n_conds=int(cfg["n_conds"]),
        device=device,
    )
    encoder.load_state_dict(enc_state, strict=True)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    logger.info(
        "loaded full encoder from %s (layers %d-%d, encode_time=%.4f)",
        path, encoder.backbone_layer_start, encoder.backbone_layer_end,
        encoder.encode_time_value,
    )
    return encoder


def build_eval_encoder(
    ckpt: str | Path,
    *,
    device: str | torch.device = "cpu",
    **overrides: object,
) -> DiscreteFlowEncoder:
    """Frozen encoder for eval / precompute CLIs — just loads the full ckpt."""
    return load_encoder_from_ckpt(ckpt, device=device, **overrides)


def build_energy_head(*, d_adapter: int, d_protein: int) -> EnergyHead:
    return EnergyHead(d_m=d_adapter, d_p=d_protein)
