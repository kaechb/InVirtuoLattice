"""lattice_lab — a clean Hydra + Lightning orchestration layer for LATTICE.

This package is a *thin, structured rewrite* of the training/eval entrypoints.
It does not re-implement any of the science: the energy head, adapter, encoder,
losses, datasets, protein store and ranking metrics are imported unchanged from
the proven :mod:`lattice` package. What changed is only the orchestration:

- Hydra structured configs + ``hydra.utils.instantiate`` instead of the bespoke
  ``train_cli`` dataclass-flattening machinery.
- Real ``LightningDataModule`` / ``LightningModule`` classes with
  ``training_step`` / ``validation_step`` / ``on_validation_epoch_end``.
- Native ``ModelCheckpoint`` / ``WandbLogger`` / ``LearningRateMonitor`` instead
  of the hand-rolled callback that did ``torch.save`` + tqdm + ``wandb.log``.

The original ``lattice.training.*`` entrypoints are left untouched so existing
runs stay reproducible. See ``lattice_lab/README.md`` for the migration map.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
