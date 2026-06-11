"""Encoder / head builders shared by the LightningModules.

Extracted (verbatim in behaviour) from the original ``train_ebm.build_encoder``
/ ``train_adapter.build_encoder`` / ``build_head`` so the new package carries no
dependency on the old monolithic trainers.
"""

from __future__ import annotations

from pathlib import Path

import torch

from lattice_lab.backbone.adapter import Adapter, AdapterConfig
from lattice_lab.backbone.encoder import EncoderConfig, MoleculeEncoder
from lattice_lab.backbone.fragmol_loader import load_fragmol
from lattice_lab.ebm.head import EnergyHead, EnergyHeadConfig


def build_ebm_encoder(
    *, adapter_ckpt: str | Path, n_fragmol_layers: int = 4, d_adapter: int = 512
) -> MoleculeEncoder:
    """FragMol (frozen) + adapter loaded from ``adapter_ckpt``, fully frozen.

    Note: the adapter is built with the default ``AdapterConfig.n_layers`` to
    match the checkpoint format produced by the adapter SSL stage.
    """
    bundle = load_fragmol(device="cpu")
    adapter = Adapter(
        AdapterConfig(
            d_fragmol=bundle.n_embd,
            n_fragmol_layers=n_fragmol_layers,
            d_adapter=d_adapter,
        )
    )
    # adapter_v1.pt from older revisions stored pathlib.PosixPath in its cfg
    # block; weights-only load (torch>=2.6) refuses that by default.
    from pathlib import PosixPath, WindowsPath

    with torch.serialization.safe_globals([PosixPath, WindowsPath]):
        state = torch.load(adapter_ckpt, map_location="cpu", weights_only=True)
    adapter.load_state_dict(state["adapter_state_dict"])
    encoder = MoleculeEncoder(
        fragmol=bundle,
        adapter=adapter,
        config=EncoderConfig(n_fragmol_layers=n_fragmol_layers),
    )
    encoder.adapter.eval()
    for p in encoder.adapter.parameters():
        p.requires_grad = False
    return encoder


def build_adapter_encoder(
    *, n_fragmol_layers: int = 4, d_adapter: int = 512, n_adapter_layers: int = 4
) -> MoleculeEncoder:
    """FragMol (frozen) + a fresh, trainable adapter for SSL pretraining."""
    bundle = load_fragmol(device="cpu")
    adapter = Adapter(
        AdapterConfig(
            d_fragmol=bundle.n_embd,
            n_fragmol_layers=n_fragmol_layers,
            d_adapter=d_adapter,
            n_layers=n_adapter_layers,
        )
    )
    return MoleculeEncoder(
        fragmol=bundle,
        adapter=adapter,
        config=EncoderConfig(n_fragmol_layers=n_fragmol_layers),
    )


def build_energy_head(*, d_adapter: int, d_protein: int, head_arch: str) -> EnergyHead:
    return EnergyHead(EnergyHeadConfig(d_m=d_adapter, d_p=d_protein, arch=head_arch))
