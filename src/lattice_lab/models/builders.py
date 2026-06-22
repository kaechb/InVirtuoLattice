"""Encoder / head builders shared by the LightningModules."""

from __future__ import annotations

import hashlib
import inspect
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lattice_lab.backbone.discrete_flow import (
    DiscreteFlowEncoder,
    build_discrete_flow_encoder,
    pad_batch,
)
from lattice_lab.ebm.head import EnergyHead

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "artifacts/tokenizer/smiles_new.json"
DEFAULT_BACKBONE_LAYER_START = 8
DEFAULT_BACKBONE_LAYER_END = 11

_ENCODER_PREFIX = "encoder."
_ADAPTER_PREFIX = "encoder.adapter."
_STUDENT_PREFIX = "student."

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

    Some denoising-JEPA runs were saved with a leading ``student.`` prefix on
    every parameter; strip that when no bare ``encoder.*`` keys are present.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"checkpoint must be a dict, got {type(raw)}")
    state = raw.get("state_dict", raw)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no 'state_dict'")
    if not any(k.startswith(_ENCODER_PREFIX) for k in state):
        student = {
            k[len(_STUDENT_PREFIX):]: v
            for k, v in state.items()
            if k.startswith(_STUDENT_PREFIX)
        }
        if student:
            logger.info("remapping checkpoint state_dict: student.* → *")
            return student
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


def adapter_run_id(adapter_ckpt: str | Path) -> str:
    """W&B run id from ``.../checkpoints/<run_id>/last.ckpt``."""
    path = resolve_adapter_ckpt(adapter_ckpt)
    parent = path.parent
    if path.name != "last.ckpt" or parent.name == "checkpoints":
        raise ValueError(
            f"cannot infer adapter run id from {adapter_ckpt}; "
            "expected .../checkpoints/<wandb_run_id>/last.ckpt"
        )
    return parent.name


def zm_store_path(adapter_ckpt: str | Path, pool: str) -> Path:
    """Default Stage-4 z_m store for an adapter checkpoint.

    Layout: ``artifacts/decoys/<run_id>/{decoy_zm,bdb_zm}`` or
    ``artifacts/binders/<run_id>/binder_zm``.
    """
    rid = adapter_run_id(adapter_ckpt)
    if pool == "binder_zm":
        return Path(f"artifacts/binders/{rid}/binder_zm")
    if pool in ("decoy_zm", "bdb_zm"):
        return Path(f"artifacts/decoys/{rid}/{pool}")
    raise ValueError(f"unknown z_m pool {pool!r}")


def eval_zm_cache_path(adapter_run_id: str, cache_name: str) -> Path:
    """Default Stage-6 LIT-PCBA z_m cache for a Stage-2 adapter run."""
    return Path(f"artifacts/evaluation/{adapter_run_id}/{cache_name}")


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


class DenoisingJepaViewEncoder:
    """Frozen denoising-JEPA encoder for Stage-4 precompute / eval."""

    def __init__(self, module: Any) -> None:
        self._module = module
        self.latent_dim = int(module.encoder.pool.dim)

    @property
    def adapter(self) -> object:
        # ponytail: shim so Stage-4 scripts can read d_adapter without branching
        dim = self.latent_dim

        class _AdapterShim:
            d_adapter = dim

            def to(self, _device: object) -> _AdapterShim:
                return self

            def eval(self) -> _AdapterShim:
                return self

        return _AdapterShim()

    @property
    def backbone_layer_start(self) -> int:
        return 0

    @property
    def backbone_layer_end(self) -> int:
        return int(self._module.hparams.n_layer) - 1

    def encode_views(
        self,
        views: Sequence[str],
        device: torch.device | str = "cpu",
        **kwargs: object,
    ) -> Tensor:
        from lattice_lab.training.denoising_jepa import encode_pooled_latent

        bundle = self._module.bundle
        seqs = [
            [bundle.bos_id, *bundle.tokenizer.encode(v, add_special_tokens=False), bundle.eos_id]
            for v in views
        ]
        ids, mask = pad_batch(seqs, pad_id=bundle.pad_id)
        ids = ids.to(device)
        mask = mask.to(device)
        self._module.to(device)
        with torch.no_grad():
            return encode_pooled_latent(self._module, ids, mask, training=False)


def _flatten_mapping(obj: Mapping[str, object] | object) -> dict[str, object]:
    """Coerce Lightning / OmegaConf hyperparameter blobs to a plain dict."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)  # type: ignore[return-value]
    except ImportError:
        pass
    if isinstance(obj, Mapping):
        return dict(obj)
    return {}


def _denoising_jepa_init_kwargs(raw: Mapping[str, object]) -> dict[str, object]:
    from lattice_lab.models.denoising_jepa_ssl import DenoisingJEPAModule

    flat = _flatten_mapping(raw.get("hyper_parameters") or {})
    enc_cfg = _flatten_mapping(raw.get("encoder_config") or {})
    for src in (enc_cfg, flat):
        for k, v in src.items():
            flat.setdefault(k, v)

    params = inspect.signature(DenoisingJEPAModule.__init__).parameters
    kwargs: dict[str, object] = {}
    for name, param in params.items():
        if name == "self":
            continue
        if name in flat:
            kwargs[name] = flat[name]
        elif param.default is not inspect.Parameter.empty:
            kwargs[name] = param.default

    # Backbone weights come from state_dict; warm-start ckpt is unused at eval time.
    kwargs["ckpt_path"] = None
    kwargs["tokenizer_path"] = str(kwargs.get("tokenizer_path") or DEFAULT_TOKENIZER)
    return kwargs


def load_denoising_jepa_for_eval(
    ckpt: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> DenoisingJepaViewEncoder:
    """Rebuild a frozen denoising-JEPA module for ``encode_views`` precompute."""
    from lattice_lab.models.denoising_jepa_ssl import DenoisingJEPAModule

    path = resolve_adapter_ckpt(ckpt)
    raw = safe_torch_load(path, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"denoising-JEPA checkpoint {path} is not a dict")
    state = _checkpoint_state_dict(raw)
    if not raw.get("hyper_parameters") and not raw.get("encoder_config"):
        raise ValueError(f"denoising-JEPA checkpoint {path} has no hyper_parameters")

    module = DenoisingJEPAModule(**_denoising_jepa_init_kwargs(raw))
    module.load_state_dict(state, strict=True)
    module.to(device).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    logger.info("loaded denoising-JEPA encoder from %s (latent_dim=%d)", path, module.encoder.pool.dim)
    return DenoisingJepaViewEncoder(module)


def build_eval_encoder(
    ckpt: str | Path,
    *,
    device: str | torch.device = "cpu",
    **overrides: object,
) -> DiscreteFlowEncoder | DenoisingJepaViewEncoder:
    """Frozen encoder for eval / precompute CLIs."""
    path = resolve_adapter_ckpt(ckpt)
    raw = safe_torch_load(path, weights_only=False)
    state = _checkpoint_state_dict(raw)
    if any(k.startswith(_ADAPTER_PREFIX) for k in state):
        return load_encoder_from_ckpt(ckpt, device=device, **overrides)
    if any(k.startswith("encoder.pool.") for k in state):
        return load_denoising_jepa_for_eval(ckpt, device=device)
    raise ValueError(
        f"unrecognized checkpoint layout at {path}; "
        "expected discrete-flow adapter (encoder.adapter.*) or denoising-JEPA (encoder.pool.*)"
    )


def build_energy_head(*, d_adapter: int, d_protein: int) -> EnergyHead:
    return EnergyHead(d_m=d_adapter, d_p=d_protein)
