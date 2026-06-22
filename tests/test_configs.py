"""Every Hydra config composes and fully resolves (catches interpolation typos,
bad defaults, and ``_target_`` paths that don't import)."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import lattice_lab

# Locate configs via the installed package (layout-independent).
CONFIG_DIR = str(Path(lattice_lab.__file__).resolve().parent / "configs")

TRAIN_CASES = [
    [],
    ["experiment=ebm_baseline"],
    ["experiment=ebm_hardneg"],
    ["experiment=adapter_discrete_flow"],
    ["experiment=adapter_discrete_flow", "model.ssl_loss=lejepa"],
    ["experiment=adapter_discrete_flow", "model.ssl_loss=ijepa"],
    ["experiment=adapter_discrete_flow", "model.ssl_loss=hybrid", "model.hybrid_anneal_steps=2000"],
    ["experiment=denoising_jepa"],
    ["experiment=ebm_hardneg_ntxent"],
    ["experiment=ebm_hardneg_lejepa"],
    ["trainer=smoke"],
    ["logger=csv"],
    ["logger=none"],
]


def _compose(config_name: str, overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        # Resolve every interpolation in the *user* config (this is what catches
        # typos / bad _target_ paths). The internal ``hydra`` node carries
        # sweep-only mandatory values, so mask it out first; the ${hydra:...}
        # resolver still works via the HydraConfig singleton set above.
        masked = OmegaConf.masked_copy(cfg, [k for k in cfg.keys() if k != "hydra"])
        container = OmegaConf.to_container(masked, resolve=True, throw_on_missing=True)
        return OmegaConf.create(container)


@pytest.mark.parametrize("overrides", TRAIN_CASES)
def test_train_config_resolves(overrides: list[str]) -> None:
    cfg = _compose("train", overrides)
    assert cfg.data._target_.startswith("lattice_lab.data")
    assert cfg.model._target_.startswith("lattice_lab.models")
    assert cfg.trainer._target_.endswith("Trainer")


def test_eval_config_resolves() -> None:
    cfg = _compose("eval", ["ckpt_path=/tmp/x.ckpt"])
    assert cfg.ckpt_path == "/tmp/x.ckpt"
    assert cfg.model._target_.startswith("lattice_lab.models")


def test_shared_dims_match_between_data_and_model() -> None:
    cfg = _compose("train", ["experiment=ebm_hardneg"])
    # n_decoys and hard_mining_mult must be identical across the two groups.
    assert cfg.data.n_decoys == cfg.model.n_decoys
    assert cfg.data.hard_mining_mult == cfg.model.hard_mining_mult == 3
