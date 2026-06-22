"""Stage 4: conditional energy-based head over (molecule latent, protein latent).

Public re-exports for convenience. The orchestration lives in
``lattice/training/train_ebm.py``; this package contains the model + losses
themselves.
"""

from lattice_lab.ebm.dataset import (
    BinderDataset,
    BinderRow,
    DecoyZmPool,
    EBMBatch,
    EBMCollator,
    HardNegativeCollator,
    stack_z_p,
)
from lattice_lab.ebm.head import EnergyHead
from lattice_lab.ebm.losses import (
    InfoNCEEnergyLoss,
    SinkhornEnergyLoss,
    cross_target_margin_loss,
    sample_target_prior,
    sinkhorn_divergence_1d,
)

__all__ = [
    "BinderDataset",
    "BinderRow",
    "DecoyZmPool",
    "EBMBatch",
    "EBMCollator",
    "EnergyHead",
    "HardNegativeCollator",
    "InfoNCEEnergyLoss",
    "SinkhornEnergyLoss",
    "cross_target_margin_loss",
    "sample_target_prior",
    "sinkhorn_divergence_1d",
    "stack_z_p",
]
