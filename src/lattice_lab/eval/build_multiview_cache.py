"""Build a MULTI-VIEW z_m cache: encode each ligand with K seeded rBRICS
fragmentations and store the **average** z_m (test-time augmentation).

A single fragmentation is one arbitrary sample of a molecule's view distribution;
the adapter is SSL-trained to be view-invariant but isn't perfectly so, which is
why a single-view cache leaves EF@1% jittery. Averaging K seeded views denoises
z_m, is fully reproducible (fixed per-molecule seeds), and leaves only a small,
controlled uncertainty (≈ single-view spread / sqrt(K)).

    PYTHONPATH=. python artifacts/energy/build_multiview_cache.py \
        --n-views 8 --zm-cache artifacts/evaluation/lit_pcba_zm_mv8 \
        --adapter-ckpt artifacts/adapter/checkpoints_ssl2/adapter_v1.pt --n-jobs 32
"""
from __future__ import annotations
import argparse
import shutil
import numpy as np, pandas as pd, torch
from pathlib import Path
from rdkit import RDLogger
from lattice_lab.eval.lit_pcba import (
    ADAPTER_FP_KEY,
    _build_encoder,
    enforce_cache_adapter,
)
from lattice_lab.models.builders import adapter_fingerprint, adapter_run_id, eval_zm_cache_path
# NOTE: seeded_views lives in the torch-free `molecules` module on purpose — loky
# workers pickle it by reference and import only that module, so they don't
# cold-import torch (which dominated multi-view cache build time on Lustre).
from lattice_lab.preprocessing.molecules import seeded_views
from lattice_lab.protein.store import EmbeddingStore
RDLogger.DisableLog("rdApp.*")


def _clear_store(store_path: Path) -> None:
    """Delete an existing store so ``create`` rebuilds it from scratch."""
    removed = False
    for name in (EmbeddingStore.MANIFEST, EmbeddingStore.PIDS, EmbeddingStore.MEAN):
        f = store_path / name
        if f.exists():
            f.unlink()
            removed = True
    perres = store_path / EmbeddingStore.PERRES_DIR
    if perres.is_dir():
        shutil.rmtree(perres)
        removed = True
    if removed:
        print(f"cleared existing cache at {store_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-views", type=int, default=8)
    ap.add_argument("--zm-cache", type=Path, default=None,
                    help="default: artifacts/evaluation/<adapter_run_id>/lit_pcba_zm_mv<N>")
    ap.add_argument("--test-parquet", type=Path, default=Path("artifacts/preprocessing/processed/bindingdb/test_lit_pcba.parquet"))
    ap.add_argument("--adapter-ckpt", type=Path, required=True,
                    help="Stage-2 adapter .ckpt / run dir, OR a trained EBM .ckpt "
                         "(the adapter is extracted from it). Use the EBM ckpt so the "
                         "cache matches the heads that will score it.")
    ap.add_argument("--adapter-run-id", default=None,
                    help="Stage-2 W&B run id for the default zm-cache path when "
                         "--adapter-ckpt is an EBM checkpoint")
    ap.add_argument("--protein-store", type=Path, default=Path("artifacts/protein_store/embeddings/esm2_650M"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--limit-targets", type=int, default=-1,
                    help="encode only ligands for the first N protein-store targets (smoke)")
    a = ap.parse_args()
    a.limit_targets = None if a.limit_targets < 0 else a.limit_targets
    if a.zm_cache is None:
        rid = a.adapter_run_id
        if rid is None:
            try:
                rid = adapter_run_id(a.adapter_ckpt)
            except ValueError as e:
                raise SystemExit(
                    f"{e} — pass --adapter-run-id (Stage-2 W&B run id) or --zm-cache"
                ) from e
        a.zm_cache = eval_zm_cache_path(rid, f"lit_pcba_zm_mv{a.n_views}")

    _clear_store(a.zm_cache)

    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    # Worker lives in a torch-free module so loky workers spawn fast (no Lustre
    # torch cold-import); _inchikey_or_none would drag torch into every worker.
    from lattice_lab.preprocessing.molecules import inchikey_of

    print(f"reading {a.test_parquet} ...", flush=True)
    df = pd.read_parquet(a.test_parquet, columns=["target_name", "smiles", "is_active"])
    df["smiles"] = df["smiles"].astype(str)
    print(f"opening protein store {a.protein_store} ...", flush=True)
    ps = EmbeddingStore.open(a.protein_store, mode="r")
    present = df["target_name"].astype(str).isin(ps.pid_to_row)
    if a.limit_targets is not None:
        kept = sorted(df.loc[present, "target_name"].astype(str).unique())[: a.limit_targets]
        present = df["target_name"].astype(str).isin(kept)
        print(f"limit-targets={a.limit_targets} → {len(kept)} targets", flush=True)
    # Dedupe identical SMILES strings first: MolToInchiKey (InChI gen) is slow,
    # and LIT-PCBA repeats the same ligand across many rows. This collapses only
    # byte-identical strings, so distinct molecules are never merged here.
    smis = list(dict.fromkeys(df.loc[present, "smiles"].tolist()))
    print(f"computing InChIKeys for {len(smis)} unique SMILES (n_jobs={a.n_jobs}) ...", flush=True)
    if a.n_jobs == 1:
        keys = [inchikey_of(s) for s in tqdm(smis, desc="inchikey", unit="mol", dynamic_ncols=True)]
    else:
        keys = list(tqdm(
            Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator", batch_size=512)(
                delayed(inchikey_of)(s) for s in smis
            ),
            total=len(smis), desc="inchikey", unit="mol", dynamic_ncols=True,
        ))
    uniq = {}
    for k, s in zip(keys, smis):
        if k and k not in uniq:
            uniq[k] = s
    print(f"{len(smis)} unique SMILES -> {len(uniq)} unique ligands; {a.n_views} views each")

    iks = list(uniq.keys())
    # --- Phase 2: generate K seeded views per ligand IN PARALLEL (CPU-bound) ---
    print(f"generating views with n_jobs={a.n_jobs} (parallel rBRICS)...", flush=True)
    if a.n_jobs == 1:
        view_lists = [
            seeded_views(uniq[ik], a.n_views)
            for ik in tqdm(iks, desc="views", unit="mol", dynamic_ncols=True)
        ]
    else:
        view_lists = list(tqdm(
            Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator", batch_size=256)(
                delayed(seeded_views)(uniq[ik], a.n_views) for ik in iks
            ),
            total=len(iks), desc="views", unit="mol", dynamic_ncols=True,
        ))

    # --- Phase 3: batched GPU encode + per-ligand average ---
    enc = _build_encoder(adapter_ckpt=a.adapter_ckpt, device=a.device)
    d_adapter = int(enc.adapter.d_adapter)
    store = EmbeddingStore.create(a.zm_cache, embedding_dim=d_adapter,
        model_name=f"lattice-adapter-mv{a.n_views}", dtype="float16", per_residue=False,
        extra={"source": str(a.test_parquet), "n_views": str(a.n_views),
               "adapter_ckpt": str(a.adapter_ckpt),
               "adapter_run_id": str(a.zm_cache.parent.name),
               ADAPTER_FP_KEY: adapter_fingerprint(a.adapter_ckpt)})
    # Reject (or backfill) a reused cache that was built by a different adapter.
    enforce_cache_adapter(store, a.adapter_ckpt)
    bs = a.batch_size
    written = 0
    pbar = tqdm(range(0, len(iks), bs), desc="encode", unit="batch", dynamic_ncols=True)
    for i in pbar:
        chunk = iks[i:i+bs]; chunk_views = view_lists[i:i+bs]
        flat, offs, keep_ik = [], [], []
        for ik, vs in zip(chunk, chunk_views):
            if not vs:
                continue
            offs.append(len(vs)); flat.extend(vs); keep_ik.append(ik)
        if not flat:
            continue
        with torch.no_grad():
            zf = enc.encode_views(flat, device=a.device).detach().cpu().to(torch.float32).numpy()
        out, p = [], 0
        for n in offs:
            out.append(zf[p:p+n].mean(0)); p += n
        written += store.append_mean(keep_ik, np.asarray(out, dtype=np.float16))
        pbar.set_postfix(written=written)
    print(f"DONE: multi-view cache {a.zm_cache} has {store.manifest.count} entries")


if __name__ == "__main__":
    main()
