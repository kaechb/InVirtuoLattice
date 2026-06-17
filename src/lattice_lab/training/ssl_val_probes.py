"""Validation probes for Stage-2 SSL (t-SNE + linear R² on chemistry props).

Each validation epoch (configurable) samples molecules from the MOSES val split,
encodes them to ``z_m``, fits Ridge probes for QED and molecular weight, and logs
2D t-SNE scatter plots (after PCA to 50 components) colored by those properties
to W&B. LeJEPA and hybrid (which anneals toward LeJEPA) use unnormalized ``z_m``
(matching SIGReg); NT-Xent uses L2-normalized ``z_m`` (matching downstream deploy).

Rank diagnostics are logged for BOTH the raw and L2-normalized ``z_m`` for every
method (``rank/{effective,numerical}_{raw,norm}``), so the InfoNCE-vs-LeJEPA rank
comparison is apples-to-apples and not confounded by the per-method probe
normalization. ``rank/{effective,numerical}`` remains the method-appropriate
primary (norm for NT-Xent, raw for LeJEPA/hybrid).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from lattice_lab.data.fragment_views import load_fragment_split_df
from lattice_lab.eval.encode_utils import encode_views_batched
from lattice_lab.preprocessing.molecules import molecule_qed_molwt

if TYPE_CHECKING:
    from lattice_lab.models.discrete_flow_ssl import DiscreteFlowSSLModule

logger = logging.getLogger(__name__)


def embedding_covariance_rank(
    z: np.ndarray,
    *,
    eps: float = 1e-12,
    rel_threshold: float = 1e-6,
) -> tuple[float, float]:
    """Rank diagnostics from the spectrum of centered embedding covariance.

    ``z`` is ``[N, D]``. Returns ``(effective, numerical)``: an entropy-based
    effective rank ``exp(-Σ p_i log p_i)`` and a numerical rank (count of
    eigenvalues above ``rel_threshold * max(eig)``).
    """
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        nan = float("nan")
        return nan, nan

    n = arr.shape[0]
    centered = arr - arr.mean(axis=0, keepdims=True)
    # Zc = U S Vt  =>  sample-cov eigenvalues are s^2 / (n - 1).
    singular = np.linalg.svd(centered, compute_uv=False)
    eig = (singular ** 2) / max(n - 1, 1)
    total = float(eig.sum())
    if total <= eps:
        return 0.0, 0.0

    probs = eig / total
    active = probs > eps
    effective = float(np.exp(-np.sum(probs[active] * np.log(probs[active]))))
    numerical = float(np.sum(eig > rel_threshold * eig.max()))
    return effective, numerical


def _l2_normalize_rows(z: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, eps, None)


@dataclass(frozen=True)
class SslValProbeResult:
    r2_qed: float
    r2_molwt: float
    mean_r2: float
    n_probe: int
    n_train: int
    n_test: int
    rank_effective: float
    rank_numerical: float
    rank_effective_raw: float
    rank_numerical_raw: float
    rank_effective_norm: float
    rank_numerical_norm: float

    def as_metrics(self) -> dict[str, float | int]:
        return {
            "val/probe_r2_qed": self.r2_qed,
            "val/probe_r2_molwt": self.r2_molwt,
            "val/probe_r2_mean": self.mean_r2,
            "val/probe_n": self.n_probe,
            "val/probe_n_train": self.n_train,
            "val/probe_n_test": self.n_test,
            # Primary (method-appropriate): normalized for NT-Xent, raw for LeJEPA.
            "rank/effective": self.rank_effective,
            "rank/numerical": self.rank_numerical,
            # Fair side-by-side: both spaces logged for every method.
            "rank/effective_raw": self.rank_effective_raw,
            "rank/numerical_raw": self.rank_numerical_raw,
            "rank/effective_norm": self.rank_effective_norm,
            "rank/numerical_norm": self.rank_numerical_norm,
        }


def _tsne_2d(x: np.ndarray, *, seed: int, perplexity: float | None) -> np.ndarray:
    n = x.shape[0]
    if n < 5:
        raise ValueError(f"t-SNE needs at least 5 points, got {n}")
    perp = float(perplexity) if perplexity is not None else max(5.0, min(30.0, n / 4))
    perp = min(perp, n - 1)
    return TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(x)


def _pca_tsne_2d(
    x: np.ndarray,
    *,
    seed: int,
    perplexity: float | None,
    pca_components: int = 50,
) -> np.ndarray:
    """Reduce to top PCA components, then run 2D t-SNE."""
    arr = np.asarray(x, dtype=np.float64)
    n_pca = min(int(pca_components), arr.shape[1], arr.shape[0] - 1)
    if n_pca < 2:
        raise ValueError(
            f"PCA→t-SNE needs at least 2 PCA components; got {n_pca} for shape {arr.shape}"
        )
    reduced = PCA(n_components=n_pca, random_state=seed).fit_transform(arr)
    return _tsne_2d(reduced, seed=seed, perplexity=perplexity)


def _ridge_r2(
    z: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    test_size: float,
    ridge_alpha: float,
) -> tuple[dict[str, float], int, int]:
    """Fit independent Ridge heads per target column; return per-target R²."""
    names = ("qed", "molwt")
    x_tr, x_te, y_tr, y_te = train_test_split(z, y, test_size=test_size, random_state=seed)
    r2: dict[str, float] = {}
    for j, name in enumerate(names):
        model = Ridge(alpha=ridge_alpha)
        model.fit(x_tr, y_tr[:, j])
        r2[name] = float(r2_score(y_te[:, j], model.predict(x_te)))
    return r2, len(x_tr), len(x_te)


def _tsne_figure(
    emb: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cbar_label: str,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    sc = ax.scatter(
        emb[:, 0], emb[:, 1], c=values, s=10, cmap="viridis", alpha=0.8, linewidths=0,
    )
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title=title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


class SSLValProbes:
    """Cached val-split probe set + epoch-end diagnostics."""

    def __init__(
        self,
        *,
        n_molecules: int = 2000,
        seed: int = 0,
        val_ratio: float = 0.005,
        test_ratio: float = 0.005,
        split_seed: int = 0,
        every_n_epochs: int = 1,
        encode_batch_size: int = 128,
        ridge_alpha: float = 1.0,
        probe_test_size: float = 0.2,
        tsne_perplexity: float | None = None,
    ) -> None:
        self.n_molecules = int(n_molecules)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.split_seed = int(split_seed)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.encode_batch_size = int(encode_batch_size)
        self.ridge_alpha = float(ridge_alpha)
        self.probe_test_size = float(probe_test_size)
        self.tsne_perplexity = tsne_perplexity
        self._views: list[str] | None = None
        self._props: np.ndarray | None = None  # [N, 2] = qed, molwt

    def prepare(self, shard_dir) -> None:
        """Load and cache a fixed val-split probe subset (idempotent)."""
        if self._views is not None:
            return
        if self.n_molecules <= 0:
            logger.info("ssl val probes disabled (n_molecules=%d)", self.n_molecules)
            return

        from pathlib import Path

        shards = sorted(Path(shard_dir).glob("shard_*.parquet"))
        if not shards:
            logger.warning("ssl val probes: no shards in %s", shard_dir)
            return

        df = load_fragment_split_df(
            shards,
            split="val",
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            split_seed=self.split_seed,
        )
        if "smiles" not in df.columns:
            logger.warning("ssl val probes: parquet missing 'smiles' column")
            return

        rng = np.random.default_rng(self.seed)
        if len(df) > self.n_molecules:
            df = df.iloc[rng.choice(len(df), size=self.n_molecules, replace=False)]

        views: list[str] = []
        props: list[tuple[float, float]] = []
        view_col = "fragment_view" if "fragment_view" in df.columns else "fragmol_view"
        for smi, view in zip(df["smiles"].astype(str), df[view_col].astype(str)):
            row = molecule_qed_molwt(smi)
            if row is None:
                continue
            views.append(view)
            props.append(row)

        if len(views) < 20:
            logger.warning("ssl val probes: only %d valid molecules after RDKit filter", len(views))
            return

        self._views = views
        self._props = np.asarray(props, dtype=np.float32)
        logger.info("ssl val probes: cached %d molecules from val split", len(views))

    @property
    def ready(self) -> bool:
        return self._views is not None and self._props is not None

    @torch.no_grad()
    def run(self, module: DiscreteFlowSSLModule) -> SslValProbeResult | None:
        if not self.ready:
            return None
        assert self._views is not None and self._props is not None

        module.encoder.eval()
        normalize_probe = module.ssl_loss == "ntxent"
        # Encode once unnormalized; L2-normalizing in numpy is exactly equivalent
        # to encoding with normalize=True (normalization is the final adapter step),
        # so we get both spaces from a single forward pass.
        z_raw = encode_views_batched(
            module.encoder,
            self._views,
            batch_size=self.encode_batch_size,
            device=module.device,
            desc="ssl val probe encode",
            normalize=False,
        ).numpy()
        z_norm = _l2_normalize_rows(z_raw)
        # Primary probe space matches downstream deploy semantics per method.
        z = z_norm if normalize_probe else z_raw

        r2_map, n_tr, n_te = _ridge_r2(
            z,
            self._props,
            seed=self.seed,
            test_size=self.probe_test_size,
            ridge_alpha=self.ridge_alpha,
        )
        eff_raw, num_raw = embedding_covariance_rank(z_raw)
        eff_norm, num_norm = embedding_covariance_rank(z_norm)
        eff_primary, num_primary = (eff_norm, num_norm) if normalize_probe else (eff_raw, num_raw)
        result = SslValProbeResult(
            r2_qed=r2_map["qed"],
            r2_molwt=r2_map["molwt"],
            mean_r2=float(np.mean(list(r2_map.values()))),
            n_probe=len(self._views),
            n_train=n_tr,
            n_test=n_te,
            rank_effective=eff_primary,
            rank_numerical=num_primary,
            rank_effective_raw=eff_raw,
            rank_numerical_raw=num_raw,
            rank_effective_norm=eff_norm,
            rank_numerical_norm=num_norm,
        )

        emb = _pca_tsne_2d(z, seed=self.seed, perplexity=self.tsne_perplexity, pca_components=50)
        qed = self._props[:, 0]
        molwt = self._props[:, 1]
        fig_qed = _tsne_figure(
            emb, qed, title="Val $z_m$ PCA(50)→t-SNE (QED)", cbar_label="QED",
        )
        fig_mw = _tsne_figure(
            emb, molwt, title="Val $z_m$ PCA(50)→t-SNE (molWt)", cbar_label="molWt",
        )
        _log_wandb_figures(
            module,
            {
                "val/tsne_qed": fig_qed,
                "val/tsne_molwt": fig_mw,
            },
        )
        return result

    def maybe_run(self, module: DiscreteFlowSSLModule) -> dict[str, float | int]:
        if not self.ready:
            return {}
        if (int(module.current_epoch) % self.every_n_epochs) != 0:
            return {}
        if module.trainer is not None and not module.trainer.is_global_zero:
            return {}
        out = self.run(module)
        return {} if out is None else out.as_metrics()


@torch.no_grad()
def _encode_jepa_zs(
    student: Any,
    views: list[str],
    *,
    batch_size: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Encode fragment-view strings to pooled latents ``z_s`` ``[N, D]`` (CPU).

    Tokenizes each view as ``[BOS] body [EOS]`` (no fragment shuffle — the probe
    set must be deterministic) and runs the JEPA encoder on the clean string.
    """
    from lattice_lab.backbone.discrete_flow import pad_batch

    student.encoder.eval()
    out: list[torch.Tensor] = []
    b = student.bundle
    for start in range(0, len(views), batch_size):
        batch = views[start : start + batch_size]
        seqs = [
            [b.bos_id, *b.tokenizer.encode(v, add_special_tokens=False), b.eos_id]
            for v in batch
        ]
        ids, mask = pad_batch(seqs, pad_id=b.pad_id)
        z = student.encoder(ids.to(device), mask.to(device))
        out.append(z.detach().cpu())
    return torch.cat(out, dim=0)


class JepaValProbes(SSLValProbes):
    """Val probes for the conditional denoising-JEPA module.

    Reuses the cached val-split subset, Ridge R² probes, rank diagnostics and
    PCA→t-SNE plots of :class:`SSLValProbes`, but the molecule embedding is the
    encoder's pooled latent ``z_s = student.encoder(ids, mask)`` (already
    LayerNorm'd by the attention pool, so the *raw* space is primary).
    """

    @torch.no_grad()
    def run(self, module: Any) -> SslValProbeResult | None:
        if not self.ready:
            return None
        assert self._views is not None and self._props is not None

        z_raw = _encode_jepa_zs(
            module.student,
            self._views,
            batch_size=self.encode_batch_size,
            device=module.device,
        ).numpy()
        z_norm = _l2_normalize_rows(z_raw)
        z = z_raw  # pooled z_s is LayerNorm'd; the raw space is the deploy space.

        r2_map, n_tr, n_te = _ridge_r2(
            z,
            self._props,
            seed=self.seed,
            test_size=self.probe_test_size,
            ridge_alpha=self.ridge_alpha,
        )
        eff_raw, num_raw = embedding_covariance_rank(z_raw)
        eff_norm, num_norm = embedding_covariance_rank(z_norm)
        result = SslValProbeResult(
            r2_qed=r2_map["qed"],
            r2_molwt=r2_map["molwt"],
            mean_r2=float(np.mean(list(r2_map.values()))),
            n_probe=len(self._views),
            n_train=n_tr,
            n_test=n_te,
            rank_effective=eff_raw,
            rank_numerical=num_raw,
            rank_effective_raw=eff_raw,
            rank_numerical_raw=num_raw,
            rank_effective_norm=eff_norm,
            rank_numerical_norm=num_norm,
        )

        emb = _pca_tsne_2d(z, seed=self.seed, perplexity=self.tsne_perplexity, pca_components=50)
        fig_qed = _tsne_figure(
            emb, self._props[:, 0], title="Val $z_s$ PCA(50)→t-SNE (QED)", cbar_label="QED",
        )
        fig_mw = _tsne_figure(
            emb, self._props[:, 1], title="Val $z_s$ PCA(50)→t-SNE (molWt)", cbar_label="molWt",
        )
        _log_wandb_figures(module, {"val/tsne_qed": fig_qed, "val/tsne_molwt": fig_mw})
        return result


def _log_wandb_figures(module: DiscreteFlowSSLModule, figures: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    pl_logger = module.logger
    if pl_logger is None:
        for fig in figures.values():
            plt.close(fig)
        return
    experiment = getattr(pl_logger, "experiment", None)
    if experiment is None:
        for fig in figures.values():
            plt.close(fig)
        return
    try:
        import wandb
    except ImportError:
        for fig in figures.values():
            plt.close(fig)
        return

    payload: dict[str, Any] = {}
    for key, fig in figures.items():
        payload[key] = wandb.Image(fig)
        plt.close(fig)
    experiment.log(payload, step=int(module.global_step))
