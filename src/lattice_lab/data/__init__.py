"""LightningDataModules wrapping the proven lattice datasets/collators."""

from __future__ import annotations

from lattice_lab.data.adapter import AdapterDataModule
from lattice_lab.data.ebm import EBMDataModule

__all__ = ["AdapterDataModule", "EBMDataModule"]
