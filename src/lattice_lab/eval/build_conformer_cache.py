"""Stage-6 (Uni-Mol variant) — build the LIT-PCBA ``z_m`` cache with the 3D encoder.

Drop-in replacement for :mod:`lattice_lab.eval.build_multiview_cache` when the
EBM head was trained on 3D (Uni-Mol) ``z_m`` (Stage-4 ``ENCODER_3D=1``). It
writes a cache in the *exact same* :class:`EmbeddingStore` layout — keyed by
InChIKey, ``float16``, tagged with ``adapter_fp`` and ``n_views`` — so the
existing scorers (:mod:`lattice_lab.eval.lit_pcba` ``--skip-zm-precompute``,
:mod:`lattice_lab.eval.ensemble_eval`) consume it unchanged.

Two checkpoints are involved because they play different roles:

* ``--encoder3d-ckpt`` — the VIEW3D Stage-2 ckpt whose ``encoder_3d.*`` actually
  encodes the conformers.
* ``--fp-ckpt`` — the checkpoint whose 2D-encoder fingerprint the scorer will
  compare against (the EBM ckpt in the pipeline). Since Stage 5 bakes the same
  Stage-2 2D encoder into the EBM ckpt, ``adapter_fingerprint`` is identical for
  the Stage-2 and EBM ckpts, so tagging the cache with it keeps
  ``enforce_cache_adapter`` happy without weakening the "same run" guarantee.

"Multi-view" here means ``n_views`` seeded RDKit conformers per ligand, averaged
(the 3D analogue of seeded fragment views).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from tqdm.auto import tqdm

from lattice_lab.data.conformers import (
    DEFAULT_DICT_PATH,
    Dictionary,
    collate_conformers,
    featurize_conformer,
    generate_conformer,
    normalize_coordinates,
    remove_hydrogens,
)
from lattice_lab.eval.lit_pcba import ADAPTER_FP_KEY, N_VIEWS_KEY
from lattice_lab.models.builders import (
    adapter_fingerprint,
    adapter_run_id,
    eval_zm_cache_path,
    load_encoder_3d_from_ckpt,
)
from lattice_lab.protein.store import EmbeddingStore

RDLogger.DisableLog("rdApp.*")

MAX_ATOMS = 256


def _confs(smiles: str, n_views: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Up to ``n_views`` seeded heavy-atom conformers for one SMILES (torch-free)."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in range(n_views):
        ac = generate_conformer(smiles, seed=seed)
        if ac is None:
            continue
        atoms, coords = remove_hydrogens(*ac)
        if len(atoms) == 0:
            continue
        out.append((atoms, normalize_coordinates(coords)))
    return out


def _resolve_dictionary(enc3d) -> Dictionary:
    cfg = getattr(enc3d, "build_config", None)
    dict_path = cfg.get("dict_path") if isinstance(cfg, dict) else None
    if not dict_path or not Path(dict_path).is_file():
        dict_path = DEFAULT_DICT_PATH
    return Dictionary.load(str(dict_path))


def _clear_store(store_path: Path) -> None:
    import shutil

    for name in (EmbeddingStore.MANIFEST, EmbeddingStore.PIDS, EmbeddingStore.MEAN):
        f = store_path / name
        if f.exists():
            f.unlink()
    perres = store_path / EmbeddingStore.PERRES_DIR
    if perres.is_dir():
        shutil.rmtree(perres)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--zm-cache", type=Path, default=None,
                    help="default: artifacts/evaluation/<adapter_run_id>/lit_pcba_zm_mv<N>")
    ap.add_argument("--test-parquet", type=Path,
                    default=Path("artifacts/preprocessing/processed/bindingdb/test_lit_pcba.parquet"))
    ap.add_argument("--encoder3d-ckpt", type=Path, required=True,
                    help="VIEW3D Stage-2 ckpt (encoder_3d.* encodes the conformers)")
    ap.add_argument("--fp-ckpt", type=Path, default=None,
                    help="ckpt whose 2D-encoder fingerprint tags the cache (default: EBM ckpt "
                         "via --adapter-ckpt, else --encoder3d-ckpt)")
    ap.add_argument("--adapter-ckpt", type=Path, default=None,
                    help="alias for --fp-ckpt (the EBM ckpt in the pipeline)")
    ap.add_argument("--adapter-run-id", default=None,
                    help="Stage-2 W&B run id for the default zm-cache path")
    ap.add_argument("--protein-store", type=Path,
                    default=Path("artifacts/protein_store/embeddings/esm2_650M"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--limit-targets", type=int, default=-1)
    a = ap.parse_args()
    a.limit_targets = None if a.limit_targets < 0 else a.limit_targets
    fp_ckpt = a.fp_ckpt or a.adapter_ckpt or a.encoder3d_ckpt

    if a.zm_cache is None:
        rid = a.adapter_run_id or adapter_run_id(a.encoder3d_ckpt)
        a.zm_cache = eval_zm_cache_path(rid, f"lit_pcba_zm_mv{a.n_views}")
    _clear_store(a.zm_cache)

    from joblib import Parallel, delayed
    from lattice_lab.preprocessing.molecules import inchikey_of

    print(f"reading {a.test_parquet} ...", flush=True)
    df = pd.read_parquet(a.test_parquet, columns=["target_name", "smiles", "is_active"])
    df["smiles"] = df["smiles"].astype(str)
    ps = EmbeddingStore.open(a.protein_store, mode="r")
    present = df["target_name"].astype(str).isin(ps.pid_to_row)
    if a.limit_targets is not None:
        kept = sorted(df.loc[present, "target_name"].astype(str).unique())[: a.limit_targets]
        present = df["target_name"].astype(str).isin(kept)
    smis = list(dict.fromkeys(df.loc[present, "smiles"].tolist()))
    print(f"computing InChIKeys for {len(smis)} unique SMILES (n_jobs={a.n_jobs}) ...", flush=True)
    keys = list(tqdm(
        Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator", batch_size=512)(
            delayed(inchikey_of)(s) for s in smis
        ),
        total=len(smis), desc="inchikey", unit="mol", dynamic_ncols=True,
    )) if a.n_jobs != 1 else [inchikey_of(s) for s in smis]
    uniq: dict[str, str] = {}
    for k, s in zip(keys, smis):
        if k and k not in uniq:
            uniq[k] = s
    iks = list(uniq)
    print(f"{len(smis)} unique SMILES -> {len(iks)} unique ligands; {a.n_views} conformers each")

    print(f"generating conformers with n_jobs={a.n_jobs} ...", flush=True)
    conf_lists = list(tqdm(
        Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator", batch_size=64)(
            delayed(_confs)(uniq[ik], a.n_views) for ik in iks
        ),
        total=len(iks), desc="conformers", unit="mol", dynamic_ncols=True,
    )) if a.n_jobs != 1 else [_confs(uniq[ik], a.n_views) for ik in iks]

    enc3d = load_encoder_3d_from_ckpt(a.encoder3d_ckpt, device=a.device)
    dictionary = _resolve_dictionary(enc3d)
    d_out = int(getattr(enc3d, "output_dim"))
    store = EmbeddingStore.create(
        a.zm_cache, embedding_dim=d_out,
        model_name=f"lattice-unimol3d-mv{a.n_views}", dtype="float16", per_residue=False,
        extra={"source": str(a.test_parquet), N_VIEWS_KEY: str(a.n_views),
               "encoder": "unimol3d",
               "adapter_ckpt": str(fp_ckpt),
               "adapter_run_id": str(a.zm_cache.parent.name),
               ADAPTER_FP_KEY: adapter_fingerprint(fp_ckpt)})

    written = 0
    pbar = tqdm(range(0, len(iks), a.batch_size), desc="encode", unit="batch", dynamic_ncols=True)
    for i in pbar:
        chunk_ik = iks[i : i + a.batch_size]
        chunk_confs = conf_lists[i : i + a.batch_size]
        flat, offs, keep_ik = [], [], []
        for ik, confs in zip(chunk_ik, chunk_confs):
            if not confs:
                continue
            offs.append(len(confs))
            flat.extend(confs)
            keep_ik.append(ik)
        if not flat:
            continue
        feats = [featurize_conformer(at, co, dictionary, MAX_ATOMS) for at, co in flat]
        batch = collate_conformers(feats, key_prefix=enc3d.key_prefix)
        batch = {k: v.to(a.device) for k, v in batch.items()}
        with torch.no_grad():
            zf = enc3d(batch).detach().cpu().to(torch.float32).numpy()
        out, p = [], 0
        for n in offs:
            out.append(zf[p : p + n].mean(0))
            p += n
        written += store.append_mean(keep_ik, np.asarray(out, dtype=np.float16))
        pbar.set_postfix(written=written)
    print(f"DONE: 3D conformer cache {a.zm_cache} has {store.manifest.count} entries")


if __name__ == "__main__":
    main()
