"""Latent-space diagnostics: qualitative plots of the EBM latent space.

Six publication-style figures that show whether the EBM latent space is
forming correctly, and in particular whether it has collapsed into the
frozen-decoy shortcut (the adapter partitioning ``z_m``) or the ``z_p``-bypass
(the head ignoring the protein).

Two evaluation sources (``--source``):

- ``lit-pcba`` (default): the independent, held-out **LIT-PCBA** test set;
  per target, experimentally confirmed actives (binders) and inactives.
  Figures reflect generalization.
- ``val``: a random subset of proteins from the **BindingDB validation
  split**, an in-distribution sanity check that does not need LIT-PCBA.
  ``--n-targets`` proteins are drawn at random (seeded), keeping only those
  with at least ``--min-actives-per-target`` actives.

Either way, every molecule is re-encoded with the current adapter, and the
energy convention is: lower ``E`` = stronger predicted binder.

Figures written to ``--output-dir``:

1. ``zm_binders_vs_decoys.png``: t-SNE of ``z_m`` for sampled actives vs
   inactives. **Healthy:** the two clouds *overlap*, ``z_m`` encodes
   chemistry, and actives/inactives are all drug-like. **Collapsed:** actives
   form a separate island, the adapter has partitioned the space.
2. ``energy_heatmap.png``: K×K matrix of mean ``E`` between each target's
   actives and each target's protein, true pairs on the diagonal.
   **Healthy:** dark diagonal. **Collapsed:** uniform rows, the head ignores
   ``z_p``.
3. ``cross_target_scatter.png``: ``E(active, correct protein)`` vs
   ``E(active, wrong protein)``. **Healthy:** cloud below the y=x diagonal.
4. ``zm_energy_two_proteins.png``: the active ``z_m`` t-SNE colored by energy
   under two different proteins. **Healthy:** the colour pattern *changes*
   between panels (the head conditions on ``z_p``).
5. ``zm_binders_by_target.png``: t-SNE of actives colored by target.
   **Healthy:** targets *overlap* with soft per-target enrichment, ``z_m`` is
   organized by chemistry. Disjoint per-target regions = ``z_m`` has wrongly
   encoded target identity (and contradicts the fact that one molecule can
   bind several targets).
6. ``energy_distribution_grid.png``: per-protein histogram of the energy over
   a random draw of the decoy ``z_m`` pool, overlaid with the target prior
   ``q*`` (the distribution the Stage-3 Sinkhorn loss matches to) and the
   protein's own actives. **Healthy:** a narrow decoy bulk at high energy with
   a thin low-energy tail, actives clearly left of it, and a shape close to
   ``q*``. A decoy bulk far from ``q*``'s Gaussian flags a prior/scale
   mismatch. Requires ``--decoy-store``; skipped if not given.

Unlike DrugCLIP (which aligns molecule and pocket in one shared space, so a
molecule embedding *should* cluster by target), LATTICE is an EBM: ``z_m`` is a
generic chemistry representation and the energy head does the binding. Hence
figures 1 and 5 expect actives and inactives (and different targets' actives)
to *mix*, not separate.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from lattice_lab.backbone.adapter import Adapter, AdapterConfig
from lattice_lab.backbone.encoder import EncoderConfig, MoleculeEncoder
from lattice_lab.backbone.fragmol_loader import load_fragmol
from lattice_lab.ebm.dataset import DecoyZmPool
from lattice_lab.ebm.head import EnergyHead, EnergyHeadConfig
from lattice_lab.ebm.losses import sample_target_prior
from lattice_lab.eval.lit_pcba import _fragmol_view, _safe_torch_load
from lattice_lab.protein.store import EmbeddingStore

logger = logging.getLogger(__name__)

# Consistent palette across the figures.
_C_ACTIVE = "#2C7FB8"   # actives / focal series
_C_INACTIVE = "#9E9E9E"  # inactives / reference
_C_GUIDE = "#555555"    # diagonal / guide lines
_C_PRIOR = "#E07B39"    # target prior q*


@dataclass
class DiagnosticsConfig:
    head_ckpt: Path
    adapter_ckpt: Path
    source: str = "lit-pcba"             # "lit-pcba" (held-out test) or "val"
    test_parquet: Path = Path(
        "01_preprocessing/processed_bindingdb/test_lit_pcba.parquet"
    )
    protein_store: Path = Path("03_protein_encoder/embeddings/esm2_650M/")
    decoy_store: Path | None = None      # decoy z_m pool, enables figure 6
    output_dir: Path = Path("06_evaluation/latent_diagnostics/")
    n_inactives: int = 1500              # inactives sampled for the t-SNE
    n_targets: int = 40                  # cap on targets used
    actives_per_target: int = 60         # actives sampled per target
    min_actives_per_target: int = 10     # val mode: drop proteins below this
    n_energy_panels: int = 9             # proteins shown in figure 6
    decoy_sample: int = 4000             # decoys scored per panel in figure 6
    batch_size: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    n_fragmol_layers: int = 4
    d_adapter: int = 512
    d_protein: int = 1280


# --------------------------------------------------------------------------
# Publication style
# --------------------------------------------------------------------------


def _apply_pub_style() -> None:
    """A clean, paper-figure matplotlib style (flat, sans-serif, no top/right
    spines, 300-dpi export)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "savefig.dpi": 300,
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 11,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9.5,
    })


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def _build_encoder(cfg: DiagnosticsConfig) -> MoleculeEncoder:
    bundle = load_fragmol(device=cfg.device)
    adapter = Adapter(AdapterConfig(
        d_fragmol=bundle.n_embd,
        n_fragmol_layers=cfg.n_fragmol_layers,
        d_adapter=cfg.d_adapter,
    ))
    adapter.load_state_dict(_safe_torch_load(cfg.adapter_ckpt)["adapter_state_dict"])
    encoder = MoleculeEncoder(
        fragmol=bundle, adapter=adapter,
        config=EncoderConfig(n_fragmol_layers=cfg.n_fragmol_layers),
    )
    encoder.adapter.to(cfg.device).eval()
    for p in encoder.adapter.parameters():
        p.requires_grad = False
    logger.info("loaded adapter from %s", cfg.adapter_ckpt)
    return encoder


def _build_head(cfg: DiagnosticsConfig) -> EnergyHead:
    state = _safe_torch_load(cfg.head_ckpt)
    arch = (state.get("cfg") or {}).get("head_arch", "cross_attn")
    head = EnergyHead(EnergyHeadConfig(d_m=cfg.d_adapter, d_p=cfg.d_protein, arch=arch))
    head.load_state_dict(state["head_state_dict"])
    head.to(cfg.device).eval()
    for p in head.parameters():
        p.requires_grad = False
    logger.info("loaded energy head arch=%s from %s", arch, cfg.head_ckpt)
    return head


def _load_eval_frame(
    cfg: DiagnosticsConfig, protein_store: EmbeddingStore, rng: np.random.Generator
) -> tuple[pd.DataFrame, str]:
    """Load + normalize the evaluation frame to ``(target_name, smiles, is_active)``.

    ``source="lit-pcba"`` reads the held-out LIT-PCBA test parquet as-is.
    ``source="val"`` reads the BindingDB validation split (``uniprot`` /
    ``is_binder_10uM``), renames it to the common schema, and draws a random
    subset of ``n_targets`` proteins, keeping only those with at least
    ``min_actives_per_target`` actives so every panel has signal.
    """
    if cfg.source == "lit-pcba":
        df = pd.read_parquet(cfg.test_parquet,
                             columns=["target_name", "smiles", "is_active"])
        label = "LIT-PCBA"
    elif cfg.source == "val":
        df = pd.read_parquet(cfg.test_parquet,
                             columns=["uniprot", "smiles", "is_binder_10uM"])
        df = df.rename(columns={"uniprot": "target_name",
                                "is_binder_10uM": "is_active"})
        label = "BindingDB val"
    else:
        raise ValueError(
            f"unknown source={cfg.source!r}; expected 'lit-pcba' or 'val'"
        )

    df["target_name"] = df["target_name"].astype(str)
    df["is_active"] = df["is_active"].astype(bool)

    present = df["target_name"].isin(protein_store.pid_to_row)
    missing = sorted(set(df.loc[~present, "target_name"]))
    if missing:
        shown = missing if len(missing) <= 20 else f"{missing[:20]} …(+{len(missing) - 20})"
        logger.warning("skipping %d targets missing from the protein store: %s",
                        len(missing), shown)
    df = df[present].reset_index(drop=True)

    if cfg.source == "val":
        act_counts = df.loc[df["is_active"]].groupby("target_name").size()
        eligible = sorted(act_counts[act_counts >= cfg.min_actives_per_target].index)
        if len(eligible) < 2:
            raise ValueError(
                f"only {len(eligible)} val proteins have >= "
                f"{cfg.min_actives_per_target} actives; lower "
                "--min-actives-per-target"
            )
        n_pick = min(cfg.n_targets, len(eligible))
        picked = set(rng.choice(np.array(eligible), size=n_pick, replace=False).tolist())
        df = df[df["target_name"].isin(picked)].reset_index(drop=True)
        logger.info("val mode: drew %d random proteins from %d eligible",
                    n_pick, len(eligible))

    return df, label


# --------------------------------------------------------------------------
# Encoding + energy
# --------------------------------------------------------------------------


def _encode_smiles(
    encoder: MoleculeEncoder, smiles: list[str], cfg: DiagnosticsConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Fragmolize + encode SMILES → ``(z_m [n_valid, d], valid_mask)``."""
    views = [_fragmol_view(s) for s in
             tqdm(smiles, desc="fragmolize", unit="mol", dynamic_ncols=True)]
    valid = np.array([v is not None for v in views])
    good = [v for v in views if v is not None]
    chunks: list[np.ndarray] = []
    for i in tqdm(range(0, len(good), cfg.batch_size), desc="encode z_m",
                  unit="batch", dynamic_ncols=True):
        with torch.no_grad():
            z = encoder.encode_views(good[i:i + cfg.batch_size], device=cfg.device)
        chunks.append(z.detach().cpu().to(torch.float32).numpy())
    z_m = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, cfg.d_adapter))
    return z_m, valid


def _paired_energy(
    head: EnergyHead, z_m: np.ndarray, z_p: np.ndarray, cfg: DiagnosticsConfig
) -> np.ndarray:
    """Energy for row-aligned ``z_m [N,d_m]`` and ``z_p [N,d_p]``."""
    out = np.empty(len(z_m), dtype=np.float32)
    for i in range(0, len(z_m), cfg.batch_size):
        zm = torch.from_numpy(z_m[i:i + cfg.batch_size]).float().to(cfg.device)
        zp = torch.from_numpy(z_p[i:i + cfg.batch_size]).float().to(cfg.device)
        with torch.no_grad():
            out[i:i + zm.shape[0]] = head(zm, zp).cpu().numpy()
    return out


def _tsne(x: np.ndarray, seed: int) -> np.ndarray:
    """2D t-SNE embedding of ``x [N, d]``."""
    from sklearn.manifold import TSNE

    perplexity = float(max(10, min(40, x.shape[0] // 30)))
    logger.info("t-SNE on %d×%d (perplexity=%.0f)…", *x.shape, perplexity)
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=seed).fit_transform(x)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_zm_binders_vs_decoys(
    emb: np.ndarray, n_actives: int, path: Path, source_label: str
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ina, act = emb[n_actives:], emb[:n_actives]
    ax.scatter(ina[:, 0], ina[:, 1], s=11, c=_C_INACTIVE, alpha=0.55, linewidths=0,
               label=f"inactives (n={len(ina)})")
    ax.scatter(act[:, 0], act[:, 1], s=13, c=_C_ACTIVE, alpha=0.75, linewidths=0,
               label=f"actives (n={len(act)})")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2",
           title=f"Molecule latent space $z_m$: {source_label} actives vs inactives")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", markerscale=2.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)


def plot_energy_heatmap(
    e: np.ndarray, target_names: list[str], path: Path, source_label: str
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    im = ax.imshow(e, cmap="viridis", aspect="equal", interpolation="nearest")
    k = len(target_names)
    ax.set_xticks(range(k)); ax.set_xticklabels(target_names, rotation=90, fontsize=7)
    ax.set_yticks(range(k)); ax.set_yticklabels(target_names, fontsize=7)
    ax.set(xlabel="protein target  j", ylabel="actives of target  i",
           title=f"Mean energy $E$: {source_label} targets × proteins")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean energy $E$  (lower = predicted binder)")
    hit = float(np.mean(e.argmin(axis=1) == np.arange(len(e))))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s (diagonal hit rate=%.3f)", path, hit)


def plot_cross_target(e_correct: np.ndarray, e_wrong: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    lo = float(min(e_correct.min(), e_wrong.min()))
    hi = float(max(e_correct.max(), e_wrong.max()))
    pad = 0.04 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1.1,
            color=_C_GUIDE, zorder=1, label="y = x")
    ax.scatter(e_correct, e_wrong, s=15, c=_C_ACTIVE, alpha=0.5, linewidths=0, zorder=2)
    ax.set(xlabel="$E$(active, correct protein)",
           ylabel="$E$(active, wrong protein)",
           title="Cross-target specificity")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    frac = float(np.mean(e_wrong > e_correct))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s (cross-target satisfied=%.3f)", path, frac)


def plot_zm_energy_two_proteins(
    emb: np.ndarray, e_a: np.ndarray, e_b: np.ndarray,
    name_a: str, name_b: str, path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8))
    vmin = float(min(e_a.min(), e_b.min()))
    vmax = float(max(e_a.max(), e_b.max()))
    sc = None
    for ax, e, name in [(axes[0], e_a, name_a), (axes[1], e_b, name_b)]:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=e, cmap="magma_r", s=15,
                        vmin=vmin, vmax=vmax, linewidths=0)
        ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title=f"colored by $E(\\cdot,\\ ${name}$)$")
        ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=axes, fraction=0.024, pad=0.02)
    cbar.set_label("energy $E$  (lower = predicted binder)")
    fig.suptitle("Active $z_m$ t-SNE: energy under two different proteins",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)


def plot_zm_binders_by_target(
    emb: np.ndarray, labels: np.ndarray, target_names: list[str], path: Path,
    source_label: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    cmap = plt.get_cmap("tab20")
    for k, name in enumerate(target_names):
        m = labels == k
        ax.scatter(emb[m, 0], emb[m, 1], s=16, color=cmap(k % 20), alpha=0.75,
                   linewidths=0, label=f"{name}  (n={int(m.sum())})")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2",
           title=f"Molecule latent space $z_m$: actives colored by {source_label} target")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=1.8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)


def _energy_distribution_panels(
    head: EnergyHead, decoy_store: Path, z_p_t: np.ndarray, tgt_used: list[str],
    a_label: np.ndarray, e_correct: np.ndarray, cfg: DiagnosticsConfig,
    rng: np.random.Generator,
) -> list[dict]:
    """Score one fixed random draw of decoy ``z_m`` under a random subset of
    proteins, returning per-protein energy clouds plus a ``q*`` reference.

    The decoy ``z_m`` are precomputed, so this only forwards the energy head,
    no FragMol. The pool must have been built with the *same* adapter as
    ``--adapter-ckpt``, hence the dimension check.
    """
    pool = DecoyZmPool.open(decoy_store)
    if pool.dim != cfg.d_adapter:
        raise ValueError(
            f"decoy pool dim {pool.dim} != d_adapter {cfg.d_adapter}; rebuild "
            "the pool with the matching adapter"
        )
    gen = torch.Generator().manual_seed(cfg.seed)
    decoy_z_m = pool.sample(cfg.decoy_sample, generator=gen).numpy().astype(np.float32)

    n_panels = min(cfg.n_energy_panels, len(tgt_used))
    panel_j = sorted(
        rng.choice(len(tgt_used), size=n_panels, replace=False).tolist()
    )
    logger.info("figure 6: scoring %d decoys under %d random proteins",
                cfg.decoy_sample, n_panels)

    panels: list[dict] = []
    for j in panel_j:
        zp = np.broadcast_to(z_p_t[j], (cfg.decoy_sample, cfg.d_protein)).copy()
        e_decoy = _paired_energy(head, decoy_z_m, zp, cfg)
        e_active = e_correct[a_label == j]
        # q*: anchor the binder delta on this protein's actual active energies
        # (fall back to the decoy cloud if the protein has no encoded actives).
        anchor = e_active if len(e_active) else e_decoy
        binder_e = anchor[rng.integers(0, len(anchor), size=cfg.decoy_sample)]
        q_gen = torch.Generator().manual_seed(cfg.seed + j + 1)
        q_star = sample_target_prior(
            1, torch.from_numpy(binder_e).float(), generator=q_gen,
        ).numpy().reshape(-1)
        panels.append({
            "name": tgt_used[j],
            "e_decoy": e_decoy,
            "e_active": e_active,
            "q_star": q_star,
        })
    return panels


def plot_energy_distributions(
    panels: list[dict], source_label: str, path: Path
) -> None:
    """Grid of per-protein energy histograms: decoys vs prior ``q*`` vs actives."""
    import matplotlib.pyplot as plt

    n = len(panels)
    ncol = 3 if n >= 3 else n
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.5 * nrow),
                             squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        p = panels[i]
        ref = [p["e_decoy"], p["q_star"]]
        if len(p["e_active"]):
            ref.append(p["e_active"])
        allv = np.concatenate(ref)
        lo, hi = np.percentile(allv, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 60)
        ax.hist(p["e_decoy"], bins=bins, density=True, color=_C_INACTIVE,
                alpha=0.75, linewidth=0, label="decoys")
        ax.hist(p["q_star"], bins=bins, density=True, histtype="step",
                color=_C_PRIOR, lw=1.7, label="prior $q^*$")
        if len(p["e_active"]):
            ax.hist(p["e_active"], bins=bins, density=True, color=_C_ACTIVE,
                    alpha=0.6, linewidth=0, label="actives")
            ax.axvline(float(np.mean(p["e_active"])), color=_C_ACTIVE,
                       lw=1.3, ls="--")
        ax.set_title(f"{p['name']}  (n_act={len(p['e_active'])})")
        ax.set_xlabel("energy $E$  (lower = predicted binder)")
        ax.set_ylabel("density")
        ax.set_yticks([])
        if i == 0:
            ax.legend(loc="best")
    fig.suptitle(
        f"Energy distribution per protein: decoys vs prior $q^*$  ({source_label})",
        fontsize=12.5, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s (%d panels)", path, n)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run(cfg: DiagnosticsConfig) -> None:
    _apply_pub_style()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    protein_store = EmbeddingStore.open(cfg.protein_store, mode="r")
    df, source_label = _load_eval_frame(cfg, protein_store, rng)
    targets = sorted(df["target_name"].unique())[:cfg.n_targets]
    logger.info("%s, targets used: %d", source_label, len(targets))
    if len(targets) < 2:
        raise ValueError(
            f"need at least 2 {source_label} targets in the protein store"
        )

    encoder = _build_encoder(cfg)
    head = _build_head(cfg)

    actives = df[df["is_active"] & df["target_name"].isin(targets)]
    inactives = df[~df["is_active"] & df["target_name"].isin(targets)]

    # --- per-target actives (figures 1-5) ------------------------------
    act_rows = []
    for t in targets:
        sub = actives[actives["target_name"] == t]
        if not sub.empty:
            act_rows.append(sub.sample(n=min(cfg.actives_per_target, len(sub)),
                                       random_state=cfg.seed))
    act_df = pd.concat(act_rows, ignore_index=True)
    z_m_a, valid_a = _encode_smiles(encoder, act_df["smiles"].tolist(), cfg)
    act_df = act_df[valid_a].reset_index(drop=True)
    n_act = len(act_df)

    tgt_used = [t for t in targets if (act_df["target_name"] == t).any()]
    t_index = {t: i for i, t in enumerate(tgt_used)}
    a_label = act_df["target_name"].map(t_index).to_numpy()
    z_p_t = np.stack([protein_store.get_mean(t) for t in tgt_used]).astype(np.float32)
    z_p_a = z_p_t[a_label]
    k = len(tgt_used)
    logger.info("actives encoded: %d over %d targets", n_act, k)

    # --- inactives (figure 1) ------------------------------------------
    ina_s = inactives.sample(n=min(cfg.n_inactives, len(inactives)), random_state=cfg.seed)
    z_m_i, _ = _encode_smiles(encoder, ina_s["smiles"].tolist(), cfg)

    # --- one t-SNE of [actives ; inactives] ----------------------------
    emb = _tsne(np.concatenate([z_m_a, z_m_i], axis=0), cfg.seed)
    emb_a = emb[:n_act]

    # Figure 1: actives vs inactives
    plot_zm_binders_vs_decoys(emb, n_act, cfg.output_dir / "zm_binders_vs_decoys.png",
                              source_label)

    # --- full active × target energy table (reused by figures 2 & 4) ---
    e_full = np.empty((n_act, k), dtype=np.float32)
    for j in range(k):
        zp_j = np.broadcast_to(z_p_t[j], (n_act, cfg.d_protein)).copy()
        e_full[:, j] = _paired_energy(head, z_m_a, zp_j, cfg)

    # Figure 2: K×K target×target mean-energy heatmap
    e_mat = np.stack([e_full[a_label == i].mean(axis=0) for i in range(k)])
    plot_energy_heatmap(e_mat, tgt_used, cfg.output_dir / "energy_heatmap.png",
                        source_label)

    # Figure 3: cross-target, active vs its protein / a wrong protein
    wrong = (a_label + 1 + rng.integers(0, k - 1, size=n_act)) % k
    e_correct = _paired_energy(head, z_m_a, z_p_a, cfg)
    e_wrong = _paired_energy(head, z_m_a, z_p_t[wrong], cfg)
    plot_cross_target(e_correct, e_wrong, cfg.output_dir / "cross_target_scatter.png")

    # Figure 4: active z_m t-SNE colored by energy under two proteins
    plot_zm_energy_two_proteins(emb_a, e_full[:, 0], e_full[:, 1],
                                tgt_used[0], tgt_used[1],
                                cfg.output_dir / "zm_energy_two_proteins.png")

    # Figure 5: active z_m t-SNE colored by target
    plot_zm_binders_by_target(emb_a, a_label, tgt_used,
                              cfg.output_dir / "zm_binders_by_target.png",
                              source_label)

    # Figure 6: per-protein energy distribution vs the prior q*
    n_figs = 5
    if cfg.decoy_store is not None:
        panels = _energy_distribution_panels(
            head, cfg.decoy_store, z_p_t, tgt_used, a_label, e_correct, cfg, rng,
        )
        plot_energy_distributions(panels, source_label,
                                  cfg.output_dir / "energy_distribution_grid.png")
        n_figs = 6
    else:
        logger.warning("figure 6 (energy_distribution_grid) skipped; pass "
                        "--decoy-store to enable it")

    logger.info("done: %d figures in %s", n_figs, cfg.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--head-ckpt", type=Path, required=True)
    parser.add_argument("--adapter-ckpt", type=Path, required=True)
    parser.add_argument("--source", choices=["lit-pcba", "val"], default="lit-pcba",
                        help="evaluation source: held-out LIT-PCBA test set, or "
                             "a random subset of the BindingDB validation split")
    parser.add_argument("--test-parquet", type=Path, default=None,
                        help="source parquet; defaults per --source to the "
                             "LIT-PCBA test parquet or the threshold_90 val split")
    parser.add_argument("--protein-store", type=Path,
                        default=Path("03_protein_encoder/embeddings/esm2_650M/"))
    parser.add_argument("--decoy-store", type=Path, default=None,
                        help="decoy z_m pool (e.g. 04_ebm_head/decoy_zm/); "
                             "enables figure 6; must match --adapter-ckpt")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("06_evaluation/latent_diagnostics/"))
    parser.add_argument("--n-inactives", type=int, default=1500,
                        help="inactives sampled for the t-SNE figures")
    parser.add_argument("--n-targets", type=int, default=40,
                        help="cap (lit-pcba) / random subset size (val) of targets")
    parser.add_argument("--actives-per-target", type=int, default=60,
                        help="actives sampled per target")
    parser.add_argument("--min-actives-per-target", type=int, default=10,
                        help="val mode: skip proteins with fewer actives")
    parser.add_argument("--n-energy-panels", type=int, default=9,
                        help="proteins shown in figure 6")
    parser.add_argument("--decoy-sample", type=int, default=4000,
                        help="decoys scored per protein in figure 6")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    test_parquet = args.test_parquet
    if test_parquet is None:
        test_parquet = (
            Path("01_preprocessing/processed_bindingdb/test_lit_pcba.parquet")
            if args.source == "lit-pcba"
            else Path("01_preprocessing/processed_bindingdb/threshold_90/val.parquet")
        )

    run(DiagnosticsConfig(
        head_ckpt=args.head_ckpt,
        adapter_ckpt=args.adapter_ckpt,
        source=args.source,
        test_parquet=test_parquet,
        protein_store=args.protein_store,
        decoy_store=args.decoy_store,
        output_dir=args.output_dir,
        n_inactives=args.n_inactives,
        n_targets=args.n_targets,
        actives_per_target=args.actives_per_target,
        min_actives_per_target=args.min_actives_per_target,
        n_energy_panels=args.n_energy_panels,
        decoy_sample=args.decoy_sample,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
