"""Stage 7: binding prediction for one target against a list of SMILES.

Given a single protein **sequence** and a list of ligand **SMILES**, this
script:

1. Encodes the protein on the fly with frozen ESM-2 → ``z_p ∈ R^1280``.
2. Encodes each ligand with frozen DDiT + the Stage-2 adapter →
   ``z_m ∈ R^512``.
3. Scores every pair with the trained energy head ``E_θ(z_m, z_p)``.
4. Writes a **CSV** ranked by predicted binding and a **PNG** violin plot of
   the predicted **energy ``E``** distribution (lower = stronger binder) at
   ``--output-png``.

Energy convention: lower ``E`` ⇒ stronger predicted binder, and
``p(bind) ∝ exp(−E)``. We report two columns:

- ``energy``: raw ``E`` (lower = better).
- ``score``: ``−E`` (higher = better); the value ranked.

**Reference sets (optional but recommended).** The raw score has no absolute
meaning on its own. Pass ``--binders`` (known actives for the target) and/or
``--nonbinders`` (known inactives / random molecules) and the script scores
them against the same target, so the violin plot brackets the screened library
between the two reference distributions: a screened molecule whose score sits
inside the known-binder violin is a strong hit; one sitting in the non-binder
violin is not. If neither is given, the plot shows only the screened set.

Unlike the Stage-6 LIT-PCBA evaluator, this takes a raw sequence (no
precomputed protein store) and needs no activity labels.

Example::

    python -m lattice_lab.inference.predict \\
        --target-fasta thrb.seq \\
        --smiles-file  my_library.csv \\
        --binders      known_binders.smi \\
        --nonbinders   known_decoys.smi \\
        --head-ckpt    artifacts/energy/checkpoints_gpu1/ebm_last.pt \\
        --adapter-ckpt artifacts/adapter/checkpoints/adapter_v1.pt \\
        --target-name  THRB \\
        --output-csv   artifacts/predictions/thrb_predictions.csv \\
        --output-png   artifacts/predictions/thrb_affinity_violin.png
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from tqdm.auto import tqdm

from lattice_lab.backbone.discrete_flow import DiscreteFlowEncoder
from lattice_lab.ebm.head import EnergyHead
from lattice_lab.eval.lit_pcba import _inchikey_or_none
from lattice_lab.models.builders import build_eval_encoder, load_energy_head
from lattice_lab.protein.encoder import (
    ESM2_DEFAULT_DIM,
    ESM2_DEFAULT_MODEL,
    ESMC_DEFAULT_DIM,
    ESMC_DEFAULT_MODEL,
    build_protein_encoder,
)

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


@contextmanager
def _phase(label: str):
    """Log a start/end banner with wall-clock duration for a pipeline phase."""
    logger.info("%s ...", label)
    t0 = time.perf_counter()
    yield
    logger.info("%s done in %.1fs", label, time.perf_counter() - t0)


# --------------------------------------------------------------------------
# Input parsing
# --------------------------------------------------------------------------


def read_smiles(path: Path) -> list[str]:
    """Read SMILES from a ``.csv``/``.tsv``/``.parquet`` (a ``smiles`` column,
    else the first column) or a plain text / ``.smi`` file (one SMILES per
    line; whitespace after the SMILES, e.g. an id, is ignored)."""
    if not path.exists():
        raise FileNotFoundError(f"SMILES file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".parquet"):
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        col = next(
            (c for c in df.columns if str(c).strip().lower() == "smiles"),
            df.columns[0],
        )
        smiles = df[col].astype(str).tolist()
    else:  # plain text / .smi: take the first whitespace-delimited token
        smiles = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smiles.append(line.split()[0])
    smiles = [s for s in (s.strip() for s in smiles) if s]
    if not smiles:
        raise ValueError(f"no SMILES parsed from {path}")
    logger.info("read %d SMILES from %s", len(smiles), path)
    return smiles


def read_target_sequence(seq: str | None, fasta: Path | None) -> str:
    """Resolve the target protein sequence from a literal string or a FASTA file."""
    if (seq is None) == (fasta is None):
        raise ValueError("provide exactly one of --target-seq / --target-fasta")
    if seq is not None:
        return "".join(seq.split()).upper()
    lines = [ln.strip() for ln in Path(fasta).read_text().splitlines()]
    body = [ln for ln in lines if ln and not ln.startswith(">")]
    if not body:
        raise ValueError(f"no sequence found in FASTA {fasta}")
    return "".join(body).upper()


# --------------------------------------------------------------------------
# Model loaders
# --------------------------------------------------------------------------


def build_encoder(args: argparse.Namespace) -> DiscreteFlowEncoder:
    """Frozen DDiT backbone + Stage-2 adapter (loaded from the full ckpt)."""
    logger.info("loading DDiT backbone + adapter (device=%s)…", args.device)
    return build_eval_encoder(args.adapter_ckpt, device=args.device)


def build_head(args: argparse.Namespace) -> EnergyHead:
    """Load the trained energy head."""
    return load_energy_head(
        args.head_ckpt, d_adapter=args.d_adapter, d_protein=args.d_protein, device=args.device,
    )


# --------------------------------------------------------------------------
# Encoding + scoring
# --------------------------------------------------------------------------


def encode_protein(args: argparse.Namespace, seq: str) -> np.ndarray:
    """Encode one protein sequence → ``[d_protein]`` float32 array."""
    dtype = "float16" if args.device.startswith("cuda") else "float32"
    encoder = build_protein_encoder(
        args.protein_backend,
        model_name=args.esm_model,
        embedding_dim=args.d_protein,
        device=args.device,
        dtype=dtype,
    )
    logger.info("encoding target protein (%d residues) with %s", len(seq), args.esm_model)
    z_p = encoder.embed_protein(seq)  # [d_protein] CPU float32
    return z_p.numpy().astype(np.float32)


def encode_ligands(
    args: argparse.Namespace, encoder: DiscreteFlowEncoder, smiles: list[str], *, desc: str,
) -> tuple[np.ndarray, list[bool]]:
    """Fragmentize + encode every SMILES, averaging ``cfg.n_views`` seeded rBRICS
    views per molecule (multi-view test-time augmentation, matching the LIT-PCBA
    eval encoding; ``n_views=1`` is plain single-view).

    Returns ``(z_m, valid)`` where ``z_m`` is ``[n_valid, d_adapter]`` (one
    view-averaged latent per valid molecule) and ``valid`` is a per-input-SMILES
    boolean mask (RDKit-unparseable SMILES are marked invalid and excluded from
    ``z_m``). An all-invalid input yields an empty ``z_m`` rather than raising.
    """
    from lattice_lab.eval.lit_pcba import fragment_views

    n_views = max(1, args.n_views)
    # Per molecule: a list of up to n_views deterministic views ([] if unparseable).
    # Spawning loky workers costs ~1 s each; only worth it for big libraries.
    use_parallel = args.n_jobs not in (0, 1) and len(smiles) >= 1000
    if not use_parallel:
        view_lists = [
            fragment_views(s, n_views)
            for s in tqdm(smiles, desc=f"fragmentize×{n_views} [{desc}]", unit="mol",
                          dynamic_ncols=True)
        ]
    else:
        from joblib import Parallel, delayed

        view_lists = list(
            tqdm(
                Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(
                    delayed(fragment_views)(s, n_views) for s in smiles
                ),
                total=len(smiles), desc=f"fragmentize×{n_views} [{desc}]", unit="mol",
                dynamic_ncols=True,
            )
        )

    valid = [len(vl) > 0 for vl in view_lists]
    n_bad = sum(1 for ok in valid if not ok)
    if n_bad:
        logger.warning("[%s] %d/%d SMILES could not be fragmentized; skipped",
                        desc, n_bad, len(smiles))

    # Flatten the valid molecules' views, tracking how many views each contributed.
    flat: list[str] = []
    offs: list[int] = []
    for vl in view_lists:
        if vl:
            offs.append(len(vl))
            flat.extend(vl)
    if not flat:
        logger.warning("[%s] no valid SMILES", desc)
        return np.zeros((0, args.d_adapter), dtype=np.float32), valid

    enc: list[np.ndarray] = []
    for i in tqdm(range(0, len(flat), args.batch_size),
                  desc=f"encode z_m [{desc}]", unit="batch", dynamic_ncols=True):
        with torch.no_grad():
            z_m = encoder.encode_views(flat[i : i + args.batch_size], device=args.device)
        enc.append(z_m.detach().cpu().to(torch.float32).numpy())
    all_z = np.concatenate(enc, axis=0)  # [sum(offs), d]

    # Average the views of each molecule → one latent per valid molecule.
    out = np.empty((len(offs), all_z.shape[1]), dtype=np.float32)
    p = 0
    for j, n in enumerate(offs):
        out[j] = all_z[p : p + n].mean(axis=0)
        p += n
    return out, valid


def score(
    head: EnergyHead, z_m: np.ndarray, z_p: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    """Return raw energies ``E`` for ``[n, d_m]`` ``z_m`` against one ``z_p``."""
    z_p_t = torch.from_numpy(z_p.astype(np.float32)).to(args.device)
    out = np.empty(z_m.shape[0], dtype=np.float32)
    for i in range(0, z_m.shape[0], args.batch_size):
        chunk = z_m[i : i + args.batch_size]
        z_m_t = torch.from_numpy(chunk.astype(np.float32)).to(args.device)
        z_p_b = z_p_t.unsqueeze(0).expand(z_m_t.shape[0], -1)
        with torch.no_grad():
            e = head(z_m_t, z_p_b)
        out[i : i + chunk.shape[0]] = e.cpu().numpy()
    return out


def score_smiles_set(
    args: argparse.Namespace,
    encoder: DiscreteFlowEncoder,
    head: EnergyHead,
    z_p: np.ndarray,
    smiles: list[str],
    *,
    desc: str,
) -> tuple[np.ndarray, list[bool]]:
    """Encode + score a SMILES list against one target.

    Returns ``(energy, valid)`` where ``energy`` has one entry per input SMILES
    (``NaN`` where the SMILES was unparseable) and ``valid`` is the parse mask.
    """
    z_m, valid = encode_ligands(args, encoder, smiles, desc=desc)
    energies = score(head, z_m, z_p, args)
    full = np.full(len(smiles), np.nan, dtype=np.float32)
    full[np.array(valid, dtype=bool)] = energies
    return full, valid


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def plot_violin(
    *,
    target_name: str,
    screened: np.ndarray | None,
    binders: np.ndarray | None,
    nonbinders: np.ndarray | None,
    path: Path,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Save a PNG violin plot of one per-molecule score, split by series.

    ``screened`` / ``binders`` / ``nonbinders`` carry the per-molecule value to
    plot on a single scale (e.g. energy ``E``); ``ylabel`` / ``ylim`` describe
    that scale and ``path`` is the PNG destination.

    Draws the screened library when given (``screened`` may be ``None`` for a
    pure reference comparison, e.g. LIT-PCBA actives vs inactives); adds a
    violin for each reference set (known binders / non-binders) that has at
    least 2 valid molecules. Series are laid out weak → strong predicted
    binding so the screened distribution can be read against the reference
    anchors.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    # (label, data, color), left-to-right: non-binders, screened, binders.
    series: list[tuple[str, np.ndarray, str]] = []
    if nonbinders is not None and nonbinders.size >= 2:
        series.append((f"Known\nnon-binders\n(n={nonbinders.size})", nonbinders, "#C44E52"))
    if screened is not None:
        series.append((f"Screened\nlibrary\n(n={screened.size})", screened, "#E08A2E"))
    if binders is not None and binders.size >= 2:
        series.append((f"Known\nbinders\n(n={binders.size})", binders, "#55A868"))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2.8 * len(series) + 1.5, 5.5))
    positions = list(range(1, len(series) + 1))
    parts = ax.violinplot(
        [s[1] for s in series], positions=positions,
        showmedians=True, showextrema=True, widths=0.8,
    )
    # Keep violin bodies pale so the overlaid points read clearly on top.
    for body, (_, _, color) in zip(parts["bodies"], series):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.30)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(1.2)

    def _darken(hex_color: str, factor: float = 0.62) -> tuple[float, float, float]:
        r, g, b = mcolors.to_rgb(hex_color)
        return (r * factor, g * factor, b * factor)

    # Jittered point overlay. Points use a darkened shade of the series color
    # with a white edge so they stay distinct from the pale violin behind them.
    # Large libraries are downsampled so the swarm never turns into a blob.
    rng = np.random.default_rng(0)
    max_points = 600
    for pos, (_, data, color) in zip(positions, series):
        shown = data
        if data.size > max_points:
            shown = rng.choice(data, size=max_points, replace=False)
        # Jitter width grows a little with count so dense sets spread out.
        jitter = 0.05 + 0.05 * min(1.0, shown.size / 300.0)
        x = rng.normal(pos, jitter, size=shown.size)
        ax.scatter(x, shown, s=26, facecolor=_darken(color), edgecolor="white",
                   linewidth=0.6, alpha=0.9, zorder=4)
        if data.size > max_points:
            logger.info("violin: showing %d/%d points for one series",
                        max_points, data.size)

    ax.set_xticks(positions)
    ax.set_xticklabels([s[0] for s in series])
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(f"Predicted affinity distribution, target: {target_name}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote affinity violin plot (%d series) to %s", len(series), path)


def predict(args: argparse.Namespace) -> pd.DataFrame:
    """Run the full prediction pipeline; return the ranked results frame."""
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    smiles = read_smiles(args.smiles_file)
    logger.info("pipeline: 1) encode target  2) load DDiT+adapter  "
                "3) load head  4) encode+score ligands")

    with _phase("[1/4] encode target protein with ESM-2 "
                "(loads a ~2.5 GB model; first run also downloads it)"):
        z_p = encode_protein(args, args.target_seq)
    with _phase("[2/4] load DDiT backbone + Stage-2 adapter"):
        encoder = build_encoder(args)
        # The ckpt is the source of truth for the latent dim.
        args.d_adapter = encoder.adapter.d_adapter
    with _phase("[3/4] load energy head"):
        head = build_head(args)

    # --- screened library: the molecules we actually report on ----------
    with _phase(f"[4/4] encode + score {len(smiles)} screened ligands"):
        energy, valid = score_smiles_set(
            args, encoder, head, z_p, smiles, desc="screened"
        )
    if not any(valid):
        raise ValueError(f"no valid SMILES to score in {args.smiles_file}")

    df = pd.DataFrame({
        "target": args.target_name,
        "smiles": smiles,
        "inchikey": [_inchikey_or_none(s) for s in smiles],
        "valid": valid,
        "energy": energy,                 # raw E ; lower = stronger binder
        "score": -energy,                 # −E    ; higher = stronger binder
    })
    df["rank"] = df["score"].rank(ascending=False, method="min").astype("Int64")
    df = df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)
    logger.info("wrote %d predictions to %s", len(df), args.output_csv)

    screened_scores = df.loc[df["valid"], "score"].to_numpy(dtype=np.float32)

    # --- optional reference sets: scored only to anchor the plot --------
    def _ref_scores(path: Path | None, desc: str) -> np.ndarray | None:
        if path is None:
            return None
        ref_smiles = read_smiles(path)
        ref_energy, _ = score_smiles_set(args, encoder, head, z_p, ref_smiles, desc=desc)
        s = -ref_energy[np.isfinite(ref_energy)]
        logger.info("[%s] scored %d valid molecules (median score=%.3f)",
                    desc, s.size, float(np.median(s)) if s.size else float("nan"))
        return s

    binder_scores = _ref_scores(args.binders_file, "binders")
    nonbinder_scores = _ref_scores(args.nonbinders_file, "nonbinders")

    # Violin PNG of the predicted ENERGY E (lower = stronger binder).
    def _neg(s: np.ndarray | None) -> np.ndarray | None:
        return None if s is None else -s

    plot_violin(
        target_name=args.target_name,
        screened=_neg(screened_scores),
        binders=_neg(binder_scores),
        nonbinders=_neg(nonbinder_scores),
        path=args.output_png,
        ylabel="energy E   (lower = stronger binder)",
    )

    # --- console summary -------------------------------------------------
    logger.info("screened library: median score=%.3f  [min=%.3f, max=%.3f]",
                float(np.median(screened_scores)),
                float(screened_scores.min()), float(screened_scores.max()))
    if nonbinder_scores is not None and nonbinder_scores.size:
        pct = float((nonbinder_scores < np.median(screened_scores)).mean() * 100.0)
        logger.info("screened median sits above %.1f%% of the non-binder reference", pct)
    top = df.loc[df["valid"]].head(5)
    logger.info("top predicted binders for %s:", args.target_name)
    for _, r in top.iterrows():
        logger.info("  rank=%-3d score=%+.3f  %s",
                     int(r["rank"]), r["score"], r["smiles"])
    return df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    tgt = parser.add_mutually_exclusive_group(required=True)
    tgt.add_argument("--target-seq", type=str, help="literal protein sequence")
    tgt.add_argument("--target-fasta", type=Path, help="FASTA file with one protein")
    parser.add_argument("--smiles-file", type=Path, required=True,
                        help=".csv/.tsv/.parquet (a 'smiles' column) or .smi/.txt "
                             "(one SMILES per line): the library to screen")
    parser.add_argument("--binders", type=Path, default=None,
                        help="optional SMILES file of KNOWN BINDERS for this target; "
                             "added as a reference violin")
    parser.add_argument("--nonbinders", type=Path, default=None,
                        help="optional SMILES file of KNOWN NON-BINDERS / random "
                             "molecules; added as a reference violin")
    parser.add_argument("--head-ckpt", type=Path,
                        default=Path("artifacts/energy/checkpoints_gpu1/ebm_last.pt"),
                        help="trained EBM checkpoint (Stage-5 Lightning .ckpt). The "
                             "adapter is read from this same file unless --adapter-ckpt "
                             "is given.")
    parser.add_argument("--adapter-ckpt", type=Path, default=None,
                        help="optional Stage-2 adapter ckpt; defaults to --head-ckpt so "
                             "the encoder always matches the trained head.")
    parser.add_argument("--output-csv", type=Path,
                        default=Path("artifacts/predictions/predictions.csv"))
    parser.add_argument("--output-png", type=Path,
                        default=Path("artifacts/predictions/affinity_distribution.png"))
    parser.add_argument("--target-name", type=str, default="target",
                        help="label used in the CSV and the plot title")
    parser.add_argument(
        "--protein-backend",
        default="esm2",
        choices=["esm2", "esmc"],
        help="protein encoder; must match the backend the head was trained on "
        "(esm2: d=1280, esmc: ESM C 600M d=1152)",
    )
    parser.add_argument(
        "--esm-model",
        type=str,
        default=None,
        help="override the backend's default checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="parallel workers for the CPU fragmentize step")
    parser.add_argument("--n-views", type=int, default=4,
                        help="seeded rBRICS views averaged per molecule (multi-view "
                             "test-time augmentation). 4 matches the reported LIT-PCBA "
                             "encoding; 1 = fast single-view.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args.adapter_ckpt = args.adapter_ckpt or args.head_ckpt
    args.target_seq = read_target_sequence(args.target_seq, args.target_fasta)
    args.binders_file = args.binders
    args.nonbinders_file = args.nonbinders
    # Provisional latent dim; overwritten from the ckpt once the encoder loads.
    args.d_adapter = 512
    if args.protein_backend == "esmc":
        args.d_protein = ESMC_DEFAULT_DIM
        if args.esm_model is None:
            args.esm_model = ESMC_DEFAULT_MODEL
    else:
        args.d_protein = ESM2_DEFAULT_DIM
        if args.esm_model is None:
            args.esm_model = ESM2_DEFAULT_MODEL
    predict(args)


if __name__ == "__main__":
    main()
