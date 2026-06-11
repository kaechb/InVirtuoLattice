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
import argparse, hashlib
import numpy as np, pandas as pd, torch
from pathlib import Path
from rdkit import Chem, RDLogger
from lattice_lab.eval.lit_pcba import _inchikey_or_none, _build_encoder, LitPcbaEvalConfig
from lattice_lab.preprocessing.molecules import smiles_to_fragmol_views
from lattice_lab.protein.store import EmbeddingStore
RDLogger.DisableLog("rdApp.*")


def seeded_views(smiles: str, k: int) -> list[str]:
    """Up to k distinct seeded views (deterministic); pad with canonical SMILES."""
    seed = int.from_bytes(hashlib.sha1(smiles.encode()).digest()[:4], "little")
    vs = smiles_to_fragmol_views(smiles, n_views=k, seed=seed)
    if not vs:
        m = Chem.MolFromSmiles(smiles)
        return [Chem.CanonSmiles(smiles)] if m else []
    return vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-views", type=int, default=8)
    ap.add_argument("--zm-cache", type=Path, required=True)
    ap.add_argument("--test-parquet", type=Path, default=Path("artifacts/processed/bindingdb/test_lit_pcba.parquet"))
    ap.add_argument("--adapter-ckpt", type=Path, default=Path("artifacts/adapter/checkpoints_ssl2/adapter_v1.pt"))
    ap.add_argument("--protein-store", type=Path, default=Path("artifacts/protein_store/embeddings/esm2_650M"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-jobs", type=int, default=32)
    a = ap.parse_args()

    df = pd.read_parquet(a.test_parquet, columns=["target_name", "smiles", "is_active"])
    df["smiles"] = df["smiles"].astype(str)
    ps = EmbeddingStore.open(a.protein_store, mode="r")
    present = df["target_name"].astype(str).isin(ps.pid_to_row)
    smis = df.loc[present, "smiles"].tolist()
    from joblib import Parallel, delayed
    keys = Parallel(n_jobs=a.n_jobs)(delayed(_inchikey_or_none)(s) for s in smis)
    uniq = {}
    for k, s in zip(keys, smis):
        if k and k not in uniq:
            uniq[k] = s
    print(f"{len(smis)} rows -> {len(uniq)} unique ligands; {a.n_views} views each")

    iks = list(uniq.keys())
    # --- Phase 2: generate K seeded views per ligand IN PARALLEL (CPU-bound) ---
    print(f"generating views with n_jobs={a.n_jobs} (parallel rBRICS)...", flush=True)
    view_lists = Parallel(n_jobs=a.n_jobs, backend="loky", batch_size=256)(
        delayed(seeded_views)(uniq[ik], a.n_views) for ik in iks
    )

    # --- Phase 3: batched GPU encode + per-ligand average ---
    enc = _build_encoder(LitPcbaEvalConfig(adapter_ckpt=a.adapter_ckpt, device=a.device))
    store = EmbeddingStore.create(a.zm_cache, embedding_dim=512,
        model_name=f"lattice-adapter-mv{a.n_views}", dtype="float16", per_residue=False,
        extra={"source": str(a.test_parquet), "n_views": str(a.n_views)})
    bs = a.batch_size
    written = 0
    for i in range(0, len(iks), bs):
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
        if (i // bs) % 50 == 0:
            print(f"  encoded {i+len(chunk)}/{len(iks)} ligands  (written={written})", flush=True)
    print(f"DONE: multi-view cache {a.zm_cache} has {store.manifest.count} entries")


if __name__ == "__main__":
    main()
