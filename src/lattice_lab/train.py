"""Unified Hydra + Lightning training entrypoint.

    python -m lattice_lab.train experiment=ebm_baseline
    python -m lattice_lab.train experiment=adapter_discrete_flow trainer=smoke

Everything downstream of the config is built with ``hydra.utils.instantiate``:
the datamodule, the LightningModule, the trainer, and the lists of callbacks /
loggers. There is no bespoke config-flattening or dataclass plumbing — the
config tree *is* the wiring.
"""

from __future__ import annotations

import logging
import os

# Silence the HF tokenizers fork warning and avoid the fork-after-parallelism
# deadlock risk: the SMILES tokenizer is only used (single-threaded) on the main
# process, while the DataLoader forks worker processes for decoy assembly.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hydra
import lightning as L
from omegaconf import DictConfig

from lattice_lab.training.run_logger import log_wandb_code
from lattice_lab.utils import (
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    seed_everything,
    wire_checkpoint_dirs_to_wandb,
)

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> dict[str, float]:
    if cfg.get("seed") is not None:
        seed_everything(int(cfg.seed))

    logger.info("instantiating datamodule <%s>", cfg.data._target_)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    logger.info("instantiating model <%s>", cfg.model._target_)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Validate a saved checkpoint against the (deterministic) val split and exit.
    # Stage 6 uses this to reproduce the stage-5 val metrics in the eval log. No
    # loggers/callbacks, so it never touches W&B or writes checkpoints.
    if cfg.get("validate_only"):
        val_trainer: L.Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=[], logger=False
        )
        val_trainer.validate(
            model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )
        metrics = {k: float(v) for k, v in val_trainer.callback_metrics.items()}
        logger.info("VAL-REPRO metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
        return metrics

    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    loggers = instantiate_loggers(cfg.get("logger"))
    wire_checkpoint_dirs_to_wandb(loggers, callbacks)

    logger.info("instantiating trainer <%s>", cfg.trainer._target_)
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    log_hyperparameters(
        cfg=cfg, model=model, datamodule=datamodule, trainer=trainer, loggers=loggers
    )
    log_wandb_code(loggers, root=cfg.paths.root_dir)

    # Optional cross-check: decoy latent dim must match the head's d_m. Test the
    # *class* for the property (reading the instance attr would assert before
    # setup()), then set up the data and compare.
    if hasattr(type(datamodule), "decoy_dim"):
        datamodule.setup()
        expected = int(cfg.model.get("d_adapter", datamodule.decoy_dim))
        if datamodule.decoy_dim != expected:
            raise ValueError(
                f"decoy pool dim {datamodule.decoy_dim} != model d_adapter {expected}; "
                "rebuild the decoy pool with the matching adapter."
            )

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    logger.info("final metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
