"""Sanity check #3 — linear probe of ``z_m`` on QM9 HOMO/LUMO.

Fits Ridge regression on adapter embeddings → quantum properties, on an 80/20
random split with a fixed seed. The README target is ``mean R² > 0.6`` across
HOMO and LUMO.

For full QM9 (~134K molecules) the encode pass is the dominant cost; pass
``n_subset`` to evaluate on a random subsample (e.g., 5K for a fast smoke run).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from lattice_lab.backbone.encoder import MoleculeEncoder
from lattice_lab.eval.encode_utils import encode_smiles_batched

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Qm9ProbeResult:
    r2_by_target: dict[str, float]
    mean_r2: float
    n_train: int
    n_test: int
    threshold: float
    passed: bool

    def as_metrics(self) -> dict[str, float | int | bool]:
        m: dict[str, float | int | bool] = {
            f"qm9/r2_{k}": v for k, v in self.r2_by_target.items()
        }
        m["qm9/mean_r2"] = self.mean_r2
        m["qm9/n_train"] = self.n_train
        m["qm9/n_test"] = self.n_test
        m["qm9/threshold"] = self.threshold
        m["qm9/pass"] = bool(self.passed)
        return m


def _load_qm9(path: Path | str, targets: Sequence[str]) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        raise ValueError(f"QM9 CSV must contain a 'smiles' column; got {list(df.columns)}")
    for t in targets:
        if t not in df.columns:
            raise ValueError(f"target column {t!r} not in CSV columns {list(df.columns)}")
    smiles = df["smiles"].astype(str).tolist()
    y = df[list(targets)].to_numpy(dtype=np.float32)
    return smiles, y


@torch.no_grad()
def evaluate_qm9_probe(
    encoder: MoleculeEncoder,
    qm9_csv: Path | str,
    *,
    targets: Sequence[str] = ("homo", "lumo"),
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    threshold: float = 0.6,
    test_size: float = 0.2,
    n_subset: int | None = None,
    seed: int = 0,
    ridge_alpha: float = 1.0,
    n_jobs: int | None = None,
) -> Qm9ProbeResult:
    """Encode QM9, fit Ridge per target, return per-target R² + mean R²."""
    smiles, y = _load_qm9(qm9_csv, targets)

    if n_subset is not None and n_subset < len(smiles):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(smiles), size=n_subset, replace=False)
        smiles = [smiles[i] for i in idx]
        y = y[idx]

    logger.info("encoding %d QM9 molecules", len(smiles))
    z, valid = encode_smiles_batched(
        encoder, smiles, batch_size=batch_size, device=device, seed=seed,
        desc="qm9 encode", n_jobs=n_jobs,
    )
    if not valid:
        return Qm9ProbeResult({t: 0.0 for t in targets}, 0.0, 0, 0, threshold, False)
    z_np = z.numpy()
    y_kept = y[valid]

    x_tr, x_te, y_tr, y_te = train_test_split(
        z_np, y_kept, test_size=test_size, random_state=seed
    )

    r2_by_target: dict[str, float] = {}
    for j, t in enumerate(targets):
        model = Ridge(alpha=ridge_alpha)
        model.fit(x_tr, y_tr[:, j])
        pred = model.predict(x_te)
        r2_by_target[t] = float(r2_score(y_te[:, j], pred))

    mean_r2 = float(np.mean(list(r2_by_target.values())))
    return Qm9ProbeResult(
        r2_by_target=r2_by_target,
        mean_r2=mean_r2,
        n_train=len(x_tr),
        n_test=len(x_te),
        threshold=threshold,
        passed=mean_r2 >= threshold,
    )
