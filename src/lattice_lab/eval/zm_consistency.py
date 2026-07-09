"""Stage-6 sanity check, logged beside the LIT-PCBA results.

The encoder baked into an EBM checkpoint (what Stage 6 rebuilds z_m from) must
reproduce the frozen Stage-4 ``binder_zm`` store (what Stage-5 *val* scored
against). This re-encodes a sample of the run's own binders through the eval
encoder using the *same* stored ``fragment_view`` and compares to the store by
cosine. A low cosine means the precompute path and the eval path have drifted —
a different encoder baked in, or a fragmentation / merge mismatch — which would
make the LIT-PCBA numbers untrustworthy while stage-5 val still looks fine.

Prints a single ``ZM-CONSISTENCY:`` line so it lands in stage6.out. Exits 1 on
FAIL; the stage-6 wrapper runs it non-fatally so results still get produced.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from lattice_lab.models.builders import adapter_fingerprint, build_eval_encoder
from lattice_lab.models.encode import encode_binders
from lattice_lab.protein.store import EmbeddingStore

logger = logging.getLogger(__name__)


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity between two ``[N, D]`` arrays."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (an * bn).sum(axis=1)


def _stored_views(extra: dict[str, str], smiles: list[str]) -> list[str | None]:
    """``fragment_view`` per SMILES from the store's source parquet(s), else None.

    Matches the exact view Stage-4 encoded so the only thing under test is the
    encoder (not runtime fragmentation)."""
    import pandas as pd
    import pyarrow.parquet as pq

    view: dict[str, str] = {}
    for key in ("source_train_parquet", "source_val_parquet"):
        p = extra.get(key)
        if not p or p == "None" or not Path(p).is_file():
            continue
        if "fragment_view" not in set(pq.read_schema(p).names):
            continue
        df = pd.read_parquet(p, columns=["smiles", "fragment_view"])
        for s, fv in zip(df["smiles"].astype(str), df["fragment_view"]):
            if s not in view and pd.notna(fv):
                view[s] = str(fv)
    return [view.get(s) for s in smiles]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ebm-ckpt", type=Path, required=True)
    ap.add_argument("--binder-store", type=Path, required=True)
    ap.add_argument("--n", type=int, default=256, help="binders to re-encode")
    ap.add_argument("--tol", type=float, default=0.999, help="min per-row cosine to PASS")
    ap.add_argument("--protein-store", type=Path, default=None)
    ap.add_argument("--d-protein", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not a.binder_store.is_dir():
        print(f"ZM-CONSISTENCY: SKIP (no binder store at {a.binder_store})", flush=True)
        return
    store = EmbeddingStore.open(a.binder_store, mode="r")
    if store.manifest.count == 0:
        print(f"ZM-CONSISTENCY: SKIP (empty binder store at {a.binder_store})", flush=True)
        return

    smiles = list(store.pid_to_row)[: a.n]
    views = _stored_views(store.manifest.extra, smiles)
    n_pre = sum(v is not None for v in views)

    enc = build_eval_encoder(a.ebm_ckpt, device=a.device)
    enc.adapter.to(a.device).eval()
    with torch.no_grad():
        fresh = encode_binders(enc, smiles, a.device, grad=False, views=views)
    fresh = fresh.detach().cpu().to(torch.float32).numpy()
    stored = np.stack([store.get_mean(s) for s in smiles]).astype(np.float32)

    cos = row_cosine(fresh, stored)
    mn, mean, med = float(cos.min()), float(cos.mean()), float(np.median(cos))
    ok = mn >= a.tol
    print(
        f"ZM-CONSISTENCY: {'PASS' if ok else 'FAIL'} "
        f"min_cos={mn:.5f} mean_cos={mean:.5f} median_cos={med:.5f} n={len(smiles)} "
        f"(stored fragment_view: {n_pre}/{len(smiles)}) tol={a.tol} "
        f"encoder_fp={adapter_fingerprint(a.ebm_ckpt)[:12]} "
        f"store_adapter={store.manifest.extra.get('adapter_run_id', '?')}",
        flush=True,
    )

    if a.protein_store is not None and a.protein_store.is_dir():
        ps = EmbeddingStore.open(a.protein_store, mode="r")
        dim = ps.manifest.embedding_dim
        dim_ok = a.d_protein is None or dim == a.d_protein
        print(
            f"ZM-CONSISTENCY: protein_store={a.protein_store} dim={dim} "
            f"targets={ps.manifest.count} d_protein={a.d_protein} "
            f"{'OK' if dim_ok else 'DIM-MISMATCH'}",
            flush=True,
        )

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
