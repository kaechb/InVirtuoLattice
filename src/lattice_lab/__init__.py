"""lattice_lab — a self-contained Hydra + Lightning implementation of LATTICE.

The package carries its own copy of every pipeline stage (preprocessing, the
DDiT backbone + adapter, the ESM protein encoder/store, the energy head, losses,
datasets, ranking metrics and inference). There is no ``import lattice``; the
science kernels were re-homed from the original monorepo and the orchestration
was rewritten around:

- Hydra structured configs + ``hydra.utils.instantiate`` instead of the bespoke
  ``train_cli`` dataclass-flattening machinery.
- Real ``LightningDataModule`` / ``LightningModule`` classes with
  ``training_step`` / ``validation_step`` / ``on_validation_epoch_end``.
- Native ``ModelCheckpoint`` / ``WandbLogger`` / ``LearningRateMonitor`` instead
  of the hand-rolled callback that did ``torch.save`` + tqdm + ``wandb.log``.

See ``lattice_lab/README.md`` for the full 7-stage pipeline map.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
