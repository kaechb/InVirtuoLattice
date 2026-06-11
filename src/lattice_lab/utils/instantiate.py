"""Instantiate lists of callbacks / loggers from a Hydra config node.

Both helpers accept a ``DictConfig`` whose values are themselves config nodes
carrying a ``_target_``. Missing / empty nodes are tolerated so a config can
disable all callbacks or loggers with ``callbacks: null`` / ``logger: null``.
"""

from __future__ import annotations

import logging

import hydra
from lightning.pytorch import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list[Callback]:
    callbacks: list[Callback] = []
    if not callbacks_cfg:
        return callbacks
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("callbacks config must be a DictConfig")
    for name, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info("instantiating callback %s <%s>", name, cb_conf._target_)
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Logger]:
    loggers: list[Logger] = []
    if not logger_cfg:
        return loggers
    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("logger config must be a DictConfig")
    for name, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.info("instantiating logger %s <%s>", name, lg_conf._target_)
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers
