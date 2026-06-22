"""LightningModules for the LATTICE training stages."""

from __future__ import annotations

from lattice_lab.models.discrete_flow_ssl import DiscreteFlowSSLModule
from lattice_lab.models.ebm import EBMLitModule

__all__ = ["DiscreteFlowSSLModule", "EBMLitModule"]
