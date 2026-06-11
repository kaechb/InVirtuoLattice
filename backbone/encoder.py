"""End-to-end molecule encoder: ``FragMol (frozen) → Adapter → z_m``.

Provides:
- ``HiddenStateCollector``: forward-hook helper that captures hidden states
  from FragMol's last ``L`` blocks during a single forward pass.
- ``MoleculeEncoder``: ties everything together and exposes
  ``encode_molecule(smiles_or_view)`` and a batched ``encode_views``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import torch
from torch import nn

from lattice_lab.backbone.adapter import Adapter, AdapterConfig
from lattice_lab.backbone.fragmol_loader import (
    FragMolBundle,
    encode_view,
    pad_batch,
)
from lattice_lab.preprocessing.molecules import smiles_to_fragmol_views


class HiddenStateCollector:
    """Forward hooks on the last ``n_layers`` FragMol blocks.

    Use as a context manager::

        with HiddenStateCollector(model, n_layers=4) as collector:
            model(input_ids)
            hs = collector.stack()   # [B, T, L*d]
    """

    def __init__(self, model: nn.Module, n_layers: int) -> None:
        self.model = model
        self.n_layers = n_layers
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._buffer: list[torch.Tensor] = []

    def __enter__(self) -> "HiddenStateCollector":
        blocks = list(self.model.blocks)  # type: ignore[attr-defined]
        if self.n_layers > len(blocks):
            raise ValueError(
                f"requested {self.n_layers} layers but FragMol has only {len(blocks)}"
            )
        target = blocks[-self.n_layers :]
        # We want the post-residual output of each block (the input passed to the next).
        for i, b in enumerate(target):
            def make_hook(slot: int):
                def hook(_module, _inputs, output):
                    # Block.forward returns the residual-summed tensor [B,T,d].
                    self._buffer.append(output)
                return hook
            h = b.register_forward_hook(make_hook(i))
            self._handles.append(h)
        self._buffer = []
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def stack(self) -> torch.Tensor:
        """Concatenate captured hidden states along the feature axis: ``[B, T, L*d]``."""
        if len(self._buffer) != self.n_layers:
            raise RuntimeError(
                f"collector saw {len(self._buffer)} layers; expected {self.n_layers}. "
                "Did you call the model exactly once inside the context?"
            )
        return torch.cat(self._buffer, dim=-1)


@dataclass(frozen=True)
class EncoderConfig:
    n_fragmol_layers: int = 4
    max_len: int | None = None  # default: per-batch max
    fragmol_max_len: int = 200  # safety cap = FragMol block_size


class MoleculeEncoder(nn.Module):
    """``FragMol (frozen) → hook-captured hidden states → Adapter → z_m``."""

    def __init__(
        self,
        fragmol: FragMolBundle,
        adapter: Adapter | None = None,
        config: EncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.fragmol = fragmol
        self.cfg = config or EncoderConfig()
        self.adapter = adapter or Adapter(
            AdapterConfig(d_fragmol=fragmol.n_embd, n_fragmol_layers=self.cfg.n_fragmol_layers)
        )
        # Sanity: adapter must expect FragMol's hidden width and the configured layer count.
        if self.adapter.cfg.d_fragmol != fragmol.n_embd:
            raise AssertionError(
                f"adapter d_fragmol={self.adapter.cfg.d_fragmol} does not match "
                f"FragMol n_embd={fragmol.n_embd}"
            )
        if self.adapter.cfg.n_fragmol_layers != self.cfg.n_fragmol_layers:
            raise AssertionError(
                f"adapter n_fragmol_layers={self.adapter.cfg.n_fragmol_layers} "
                f"!= encoder.cfg.n_fragmol_layers={self.cfg.n_fragmol_layers}"
            )
        # FragMol params remain unbacked; verify caller didn't unfreeze it.
        for p in self.fragmol.model.parameters():
            assert not p.requires_grad, "FragMol must be frozen before wrapping"

    @torch.no_grad()
    def _fragmol_hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """Run FragMol once and return last-L concatenated hidden states ``[B,T,L*d]``."""
        with HiddenStateCollector(self.fragmol.model, self.cfg.n_fragmol_layers) as col:
            self.fragmol.model(ids)
            return col.stack()

    def encode_token_ids(self, ids: torch.Tensor, mask: torch.Tensor,
                         *, return_projection: bool = False
                         ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass given pre-tokenized ids ``[B,T]`` and attention mask ``[B,T]``."""
        hs = self._fragmol_hidden(ids)
        return self.adapter(hs, mask, return_projection=return_projection)

    def encode_views(
        self, views: Sequence[str], device: torch.device | str = "cpu",
        *, return_projection: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of FragMol-notation view strings."""
        seqs = [encode_view(self.fragmol, v) for v in views]
        cap = self.cfg.fragmol_max_len
        seqs = [s[:cap] for s in seqs]
        ids, mask = pad_batch(seqs, pad_id=self.fragmol.pad_id, max_len=self.cfg.max_len)
        ids = ids.to(device)
        mask = mask.to(device)
        return self.encode_token_ids(ids, mask, return_projection=return_projection)

    def encode_molecule(self, smiles: str, device: torch.device | str = "cpu",
                        *, seed: int | None = None) -> torch.Tensor:
        """Convenience: take a raw SMILES, fragmolize a single view, return ``z_m``."""
        views = smiles_to_fragmol_views(smiles, n_views=1, seed=seed)
        if not views:
            raise ValueError(f"could not fragmolize SMILES: {smiles!r}")
        return self.encode_views(views[:1], device=device)  # type: ignore[return-value]


def sync_encoder_device(
    encoder: MoleculeEncoder,
    device: str | torch.device,
    *,
    head: nn.Module | None = None,
) -> None:
    """Align FragMol + adapter (+ optional head) after ``Trainer.fit`` teardown."""
    dev = torch.device(device)
    encoder.adapter.to(dev)
    encoder.fragmol.model.to(dev)
    if head is not None:
        head.to(dev)
