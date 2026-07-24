"""Energy-averaging ensemble evaluation on LIT-PCBA.

Scores LIT-PCBA with the mean per-ligand energy of N seed heads (lower energy =
stronger binder), then computes AUROC / BEDROC / EF the same way as
``lattice_lab.eval.lit_pcba``. This is the protocol behind the released 3-seed
number: average the per-ligand energies, then compute metrics once.

Reuses an already-built z_m cache (InChIKey-keyed EmbeddingStore), so no live encoding
encoding is needed and the whole thing runs comfortably on CPU — handy when both
GPUs are busy training. Optionally writes a per-target actives-vs-inactives
energy violin (``--violin-dir``) for the separation check.

    python -m lattice_lab.eval.ensemble_eval \
        --ckpts artifacts/energy/checkpoints/<run_id0>/last.ckpt \
                artifacts/energy/checkpoints/<run_id1>/last.ckpt \
                artifacts/energy/checkpoints/<run_id2>/last.ckpt \
        --zm-cache      artifacts/evaluation/lit_pcba_zm_mv4_lejepa \
        --protein-store artifacts/protein_store/embeddings/esm2_650M \
        --test-parquet  artifacts/preprocessing/processed/bindingdb/test_lit_pcba.parquet \
        --out           artifacts/evaluation/ensemble_hardneg_mv4_lejepa.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from lattice_lab.ebm.head import EnergyHead
from lattice_lab.eval.lit_pcba import (
    N_VIEWS_KEY,
    _inchikey_or_none,
    enforce_cache_adapter,
    enforce_cache_n_views,
)
from lattice_lab.eval.metrics import auroc, bedroc, ef_at_k
from lattice_lab.models.builders import adapter_fingerprint, load_energy_head
from lattice_lab.protein.store import EmbeddingStore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_METRIC_COLS = ("ap", "auroc", "bedroc", "ef@0.5%", "ef@1.0%", "ef@2.0%", "ef@5.0%")


def _summary_from_df(res: pd.DataFrame) -> dict[str, float | int]:
    summary = {f"mean/{k}": float(res[k].mean()) for k in _METRIC_COLS}
    summary.update({f"median/{k}": float(res[k].median()) for k in _METRIC_COLS})
    summary["n_targets"] = len(res)
    return summary


def _load_head(
    ckpt: Path,
    d_m: int | None,
    d_p: int | None,
    device: str,
) -> EnergyHead:
    return load_energy_head(ckpt, d_adapter=d_m, d_protein=d_p, device=device)


def _head_energies(
    heads,
    z_m: np.ndarray,
    z_p: np.ndarray,
    device: str,
    batch_size: int = 8192,
) -> np.ndarray:
    """Per-head energies [n_heads, L] for [L, d_m] z_m against one z_p. Lower = binder."""
    z_p_t = torch.from_numpy(z_p.astype(np.float32)).to(device)
    out = np.zeros((len(heads), z_m.shape[0]), dtype=np.float64)
    for i in range(0, z_m.shape[0], batch_size):
        chunk = torch.from_numpy(z_m[i:i + batch_size].astype(np.float32)).to(device)
        z_p_b = z_p_t.unsqueeze(0).expand(chunk.shape[0], -1)
        with torch.no_grad():
            for hi, h in enumerate(heads):
                out[hi, i:i + chunk.shape[0]] = (
                    h(chunk, z_p_b).cpu().numpy().astype(np.float64)
                )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--zm-cache", type=Path, required=True)
    ap.add_argument("--protein-store", type=Path, required=True)
    ap.add_argument("--test-parquet", type=Path,
                    default=Path("artifacts/preprocessing/processed/bindingdb/test_lit_pcba.parquet"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/evaluation/ensemble_eval.json"))
    ap.add_argument("--violin-dir", type=Path, default=None)
    ap.add_argument("--bedroc-alpha", type=float, default=80.5)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--d-adapter",
        type=int,
        default=None,
        help="override molecule latent dim (default: read from checkpoint)",
    )
    ap.add_argument(
        "--d-protein",
        type=int,
        default=None,
        help="override protein latent dim (default: read from checkpoint)",
    )
    a = ap.parse_args()

    df = pd.read_parquet(a.test_parquet, columns=["target_name", "smiles", "is_active"])
    df["target_name"] = df["target_name"].astype(str)
    df["smiles"] = df["smiles"].astype(str)
    df["is_active"] = df["is_active"].astype(int)

    pstore = EmbeddingStore.open(a.protein_store, mode="r")
    present = [t for t in sorted(df["target_name"].unique()) if t in pstore.pid_to_row]
    work = df[df["target_name"].isin(present)].copy()

    logger.info("computing InChIKeys for %d rows (n_jobs=%d)", len(work), a.n_jobs)
    if a.n_jobs in (0, 1):
        keys = [_inchikey_or_none(s) for s in work["smiles"]]
    else:
        from joblib import Parallel, delayed
        keys = list(Parallel(n_jobs=a.n_jobs, backend="loky")(
            delayed(_inchikey_or_none)(s) for s in work["smiles"]))
    work["inchikey"] = keys
    work = work.dropna(subset=["inchikey"]).reset_index(drop=True)

    zm = EmbeddingStore.open(a.zm_cache, mode="r")
    recorded_views = zm.manifest.extra.get(N_VIEWS_KEY)
    if recorded_views is None or int(recorded_views) < 2:
        raise SystemExit(
            f"{a.zm_cache} is not a multi-view cache ({N_VIEWS_KEY}={recorded_views!r}). "
            f"Run build_multiview_cache with --n-views >= 2 first."
        )
    enforce_cache_n_views(zm, int(recorded_views))
    work = work[work["inchikey"].isin(zm.pid_to_row)].reset_index(drop=True)

    # Guard: every ensemble member must share one adapter (else their z_m live in
    # different latent spaces), and that adapter must match the z_m cache.
    fps = [adapter_fingerprint(c) for c in a.ckpts]
    if len(set(fps)) > 1:
        raise ValueError(
            "ensemble checkpoints were trained with DIFFERENT adapters "
            f"(fingerprints {[f[:12] for f in fps]}); refusing to average their "
            "energies across mismatched latent spaces."
        )
    enforce_cache_adapter(zm, a.ckpts[0])

    heads = [_load_head(c, a.d_adapter, a.d_protein, a.device) for c in a.ckpts]

    if a.violin_dir is not None:
        a.violin_dir.mkdir(parents=True, exist_ok=True)
        from lattice_lab.inference.predict import plot_violin

    rows: list[dict] = []
    seed_rows: list[list[dict]] = [[] for _ in heads]
    for t in present:
        sub = work[work["target_name"] == t]
        if sub.empty:
            continue
        idx = np.fromiter((zm.pid_to_row[k] for k in sub["inchikey"]),
                          dtype=np.int64, count=len(sub))
        z_m = np.asarray(zm.mean_array[idx], dtype=np.float32)
        z_p = np.asarray(pstore.get_mean(t), dtype=np.float32)
        y_true = sub["is_active"].to_numpy(dtype=int)
        energies = _head_energies(heads, z_m, z_p, a.device)
        y_score = -energies.mean(axis=0)
        m = {
            "target": t, "n": int(y_true.size), "n_active": int(y_true.sum()),
            "ap": float(average_precision_score(y_true, y_score)),
            "auroc": auroc(y_true, y_score),
            "bedroc": bedroc(y_true, y_score, alpha=a.bedroc_alpha),
            "ef@0.5%": ef_at_k(y_true, y_score, 0.5),
            "ef@1.0%": ef_at_k(y_true, y_score, 1.0),
            "ef@2.0%": ef_at_k(y_true, y_score, 2.0),
            "ef@5.0%": ef_at_k(y_true, y_score, 5.0),
        }
        rows.append(m)
        logger.info(
            "%-8s [ensemble] n=%d n_a=%d auc=%.3f bedroc=%.3f ef1=%.2f",
            t, m["n"], m["n_active"], m["auroc"], m["bedroc"], m["ef@1.0%"],
        )
        for hi in range(len(heads)):
            y_seed = -energies[hi]
            seed_rows[hi].append({
                "target": t, "n": m["n"], "n_active": m["n_active"],
                "ap": float(average_precision_score(y_true, y_seed)),
                "auroc": auroc(y_true, y_seed),
                "bedroc": bedroc(y_true, y_seed, alpha=a.bedroc_alpha),
                "ef@0.5%": ef_at_k(y_true, y_seed, 0.5),
                "ef@1.0%": ef_at_k(y_true, y_seed, 1.0),
                "ef@2.0%": ef_at_k(y_true, y_seed, 2.0),
                "ef@5.0%": ef_at_k(y_true, y_seed, 5.0),
            })
        if a.violin_dir is not None:
            ea, ei = energies.mean(axis=0)[y_true == 1], energies.mean(axis=0)[y_true == 0]
            if ea.size >= 2 and ei.size >= 2:
                plot_violin(
                    target_name=t,
                    screened=None,
                    binders=ea,
                    nonbinders=ei,
                    path=a.violin_dir / f"{t}_energy_violin.png",
                    ylabel="energy E   (lower = stronger binder)",
                )

    res = pd.DataFrame(rows)
    summary = _summary_from_df(res)
    per_seed: list[dict] = []
    for hi, ckpt in enumerate(a.ckpts):
        seed_res = pd.DataFrame(seed_rows[hi])
        seed_summary = _summary_from_df(seed_res)
        entry = {
            "seed": hi,
            "run_id": ckpt.parent.name,
            "ckpt": str(ckpt),
            **seed_summary,
        }
        per_seed.append(entry)
        logger.info(
            "SEED %d (%s) -> auroc=%.4f bedroc=%.4f ef1=%.2f (n=%d targets)",
            hi, ckpt.parent.name, seed_summary["mean/auroc"],
            seed_summary["mean/bedroc"], seed_summary["mean/ef@1.0%"],
            seed_summary["n_targets"],
        )

    output = dict(summary)
    output["per_seed"] = per_seed
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out.with_suffix(".csv"), index=False)
    a.out.write_text(json.dumps(output, indent=2))
    logger.info(
        "ENSEMBLE %d heads -> auroc=%.4f bedroc=%.4f ef1=%.2f (n=%d targets)",
        len(heads), summary["mean/auroc"], summary["mean/bedroc"],
        summary["mean/ef@1.0%"], summary["n_targets"],
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    # ponytail: one-liner guard on summary aggregation
    _toy = pd.DataFrame({k: [1.0, 3.0] for k in _METRIC_COLS})
    assert _summary_from_df(_toy)["mean/auroc"] == 2.0
    main()
