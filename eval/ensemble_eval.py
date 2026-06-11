"""Energy-averaging ensemble evaluation on LIT-PCBA.

Scores LIT-PCBA with the mean per-ligand energy of N seed heads (lower energy =
stronger binder), then computes AUROC / BEDROC / EF the same way as
``lattice.eval.lit_pcba``. This is the protocol behind the released 3-seed
number: average the per-ligand energies, then compute metrics once.

Reuses an already-built z_m cache (InChIKey-keyed EmbeddingStore), so no FragMol
encoding is needed and the whole thing runs comfortably on CPU — handy when both
GPUs are busy training. Optionally writes a per-target actives-vs-inactives
energy violin (``--violin-dir``) for the separation check.

    python 05_training/ensemble_eval.py \
        --ckpts 05_training/exp_hardneg_seed0/ebm_best_ef1.pt \
                05_training/exp_hardneg_seed1/ebm_best_ef1.pt \
                05_training/exp_hardneg_seed2/ebm_best_ef1.pt \
        --zm-cache      06_evaluation/lit_pcba_zm_hardneg \
        --protein-store 03_protein_encoder/embeddings/esm2_650M \
        --test-parquet  01_preprocessing/processed_bindingdb/test_lit_pcba.parquet \
        --out           06_evaluation/ensemble_hardneg.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lattice_lab.ebm.head import EnergyHead, EnergyHeadConfig
from lattice_lab.eval.lit_pcba import _inchikey_or_none
from lattice_lab.eval.metrics import auroc, bedroc, ef_at_k
from lattice_lab.protein.store import EmbeddingStore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_head(ckpt: Path, d_m: int, d_p: int, device: str) -> EnergyHead:
    from pathlib import PosixPath, WindowsPath
    with torch.serialization.safe_globals([PosixPath, WindowsPath]):
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    arch = (state.get("cfg") or {}).get("head_arch", "cross_attn")
    head = EnergyHead(EnergyHeadConfig(d_m=d_m, d_p=d_p, arch=arch))
    head.load_state_dict(state["head_state_dict"])
    head.to(device).eval()
    for p in head.parameters():
        p.requires_grad = False
    logger.info("loaded head arch=%s from %s", arch, ckpt)
    return head


def _mean_energy(heads, z_m: np.ndarray, z_p: np.ndarray, device: str,
                 batch_size: int = 8192) -> np.ndarray:
    """Mean energy E over heads for [L, d_m] z_m against one z_p. Lower = binder."""
    z_p_t = torch.from_numpy(z_p.astype(np.float32)).to(device)
    out = np.zeros(z_m.shape[0], dtype=np.float64)
    for i in range(0, z_m.shape[0], batch_size):
        chunk = torch.from_numpy(z_m[i:i + batch_size].astype(np.float32)).to(device)
        z_p_b = z_p_t.unsqueeze(0).expand(chunk.shape[0], -1)
        acc = np.zeros(chunk.shape[0], dtype=np.float64)
        with torch.no_grad():
            for h in heads:
                acc += h(chunk, z_p_b).cpu().numpy().astype(np.float64)
        out[i:i + chunk.shape[0]] = acc / len(heads)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--zm-cache", type=Path, required=True)
    ap.add_argument("--protein-store", type=Path, required=True)
    ap.add_argument("--test-parquet", type=Path,
                    default=Path("01_preprocessing/processed_bindingdb/test_lit_pcba.parquet"))
    ap.add_argument("--out", type=Path, default=Path("06_evaluation/ensemble_eval.json"))
    ap.add_argument("--violin-dir", type=Path, default=None)
    ap.add_argument("--bedroc-alpha", type=float, default=80.5)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d-adapter", type=int, default=512)
    ap.add_argument("--d-protein", type=int, default=1280)
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
    work = work[work["inchikey"].isin(zm.pid_to_row)].reset_index(drop=True)

    heads = [_load_head(c, a.d_adapter, a.d_protein, a.device) for c in a.ckpts]

    if a.violin_dir is not None:
        a.violin_dir.mkdir(parents=True, exist_ok=True)
        from lattice_lab.inference.predict import PredictConfig, plot_violin

    rows = []
    for t in present:
        sub = work[work["target_name"] == t]
        if sub.empty:
            continue
        idx = np.fromiter((zm.pid_to_row[k] for k in sub["inchikey"]),
                          dtype=np.int64, count=len(sub))
        z_m = np.asarray(zm.mean_array[idx], dtype=np.float32)
        z_p = np.asarray(pstore.get_mean(t), dtype=np.float32)
        y_true = sub["is_active"].to_numpy(dtype=int)
        energy = _mean_energy(heads, z_m, z_p, a.device)
        y_score = -energy
        m = {
            "target": t, "n": int(y_true.size), "n_active": int(y_true.sum()),
            "auroc": auroc(y_true, y_score),
            "bedroc": bedroc(y_true, y_score, alpha=a.bedroc_alpha),
            "ef@0.5%": ef_at_k(y_true, y_score, 0.5),
            "ef@1.0%": ef_at_k(y_true, y_score, 1.0),
            "ef@5.0%": ef_at_k(y_true, y_score, 5.0),
        }
        rows.append(m)
        logger.info("%-8s n=%d n_a=%d auc=%.3f bedroc=%.3f ef1=%.2f",
                    t, m["n"], m["n_active"], m["auroc"], m["bedroc"], m["ef@1.0%"])
        if a.violin_dir is not None:
            ea, ei = energy[y_true == 1], energy[y_true == 0]
            if ea.size >= 2 and ei.size >= 2:
                pcfg = PredictConfig(target_seq="", smiles_file=Path("-"), target_name=t)
                plot_violin(pcfg, None, ea, ei,
                            path=a.violin_dir / f"{t}_energy_violin.png",
                            ylabel="energy E   (lower = stronger binder)")

    res = pd.DataFrame(rows)
    summary = {f"mean/{k}": float(res[k].mean()) for k in
               ["auroc", "bedroc", "ef@0.5%", "ef@1.0%", "ef@5.0%"]}
    summary.update({f"median/{k}": float(res[k].median()) for k in
                    ["auroc", "bedroc", "ef@0.5%", "ef@1.0%", "ef@5.0%"]})
    summary["n_targets"] = len(res)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out.with_suffix(".csv"), index=False)
    a.out.write_text(json.dumps(summary, indent=2))
    logger.info("ENSEMBLE %d heads -> auroc=%.4f bedroc=%.4f ef1=%.2f (n=%d targets)",
                len(heads), summary["mean/auroc"], summary["mean/bedroc"],
                summary["mean/ef@1.0%"], summary["n_targets"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
