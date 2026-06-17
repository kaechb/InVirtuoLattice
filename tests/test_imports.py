"""Every public module imports cleanly (no instantiation / no I/O)."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    # entrypoints + orchestration
    "lattice_lab",
    "lattice_lab.train",
    "lattice_lab.evaluate",
    "lattice_lab.utils",
    "lattice_lab.utils.instantiate",
    "lattice_lab.utils.misc",
    "lattice_lab.data",
    "lattice_lab.data.ebm",
    "lattice_lab.data.cluster_sampler",
    "lattice_lab.data.fragment_views",
    "lattice_lab.models",
    "lattice_lab.models.ebm",
    "lattice_lab.models.discrete_flow_ssl",
    "lattice_lab.models.denoising_jepa_ssl",
    "lattice_lab.backbone.discrete_flow",
    "lattice_lab.models.builders",
    "lattice_lab.models.encode",
    "lattice_lab.models.schedules",
    "lattice_lab.callbacks",
    "lattice_lab.callbacks.sanity_gate",
    # re-homed pipeline CLIs (argparse entrypoints)
    "lattice_lab.preprocessing.run_bindingdb",
    "lattice_lab.preprocessing.run_preprocessing",
    "lattice_lab.protein.precompute",
    "lattice_lab.ebm.precompute_decoys",
    "lattice_lab.ebm.precompute_bdb_zm",
    "lattice_lab.eval.lit_pcba",
    "lattice_lab.eval.build_multiview_cache",
    "lattice_lab.eval.ensemble_eval",
    "lattice_lab.inference.predict",
    "lattice_lab.inference.predict_ensemble",
    # re-homed kernels (no `import lattice`)
    "lattice_lab.paths",
    "lattice_lab.backbone.adapter",
    "lattice_lab.backbone.ddit.rotary",
    "lattice_lab.backbone.ddit.model_ddit",
    "lattice_lab.protein.store",
    "lattice_lab.ebm.dataset",
    "lattice_lab.ebm.head",
    "lattice_lab.ebm.losses",
    "lattice_lab.eval.metrics",
    "lattice_lab.eval.sanity_check",
    "lattice_lab.preprocessing.molecules",
    "lattice_lab.preprocessing.homology",
    "lattice_lab.training.denoising_jepa",
    "lattice_lab.training.ssl_dataset",
    "lattice_lab.training.ssl_loss",
    "lattice_lab.training.ssl_val_probes",
    "lattice_lab.training.run_logger",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name: str) -> None:
    importlib.import_module(name)
