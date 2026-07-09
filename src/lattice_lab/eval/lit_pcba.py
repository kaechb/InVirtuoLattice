"""Stage 6 — evaluate the trained EBM head on LIT-PCBA.

Pipeline:
1. Load the frozen DDiT + Stage-2 adapter and the Stage-5 energy head.
2. For each LIT-PCBA target with an ESM-2 embedding in the protein store,
   encode every ligand SMILES → ``z_m`` (cached on disk by InChIKey so
   re-runs are fast and idempotent).
3. Score with ``-E_θ(z_m, z_p)`` (higher = predicted binder).
4. Compute AUROC, BEDROC(α=80.5), EF@{0.5, 1, 5} % per target. Aggregate
   mean + median across targets.
5. Write a per-target CSV.

Targets whose pid is missing from the protein store are listed in the
output but skipped from metrics (Stage-3 likely rejected non-canonical
residues; pass ``--no-canonical-filter`` to ``lattice_lab.protein.precompute``
and re-embed to recover them).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from tqdm.auto import tqdm

from lattice_lab.backbone.discrete_flow import DiscreteFlowEncoder
from lattice_lab.ebm.head import EnergyHead
from lattice_lab.eval.metrics import auroc, bedroc, ef_at_k
from lattice_lab.models.builders import (
    adapter_fingerprint,
    build_eval_encoder,
    load_energy_head,
)
from lattice_lab.preprocessing.molecules import inchikey_of, seeded_views
from lattice_lab.protein.store import EmbeddingStore

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

ADAPTER_FP_KEY = "adapter_fp"
N_VIEWS_KEY = "n_views"


def enforce_cache_adapter(store: EmbeddingStore, adapter_ckpt: Path | str) -> None:
    """Guard that a z_m cache was built by the SAME adapter we're scoring with.

    Compares a fingerprint of the adapter *weights* (robust to path differences).
    - matching fingerprint: no-op.
    - mismatched fingerprint: raise (the cache lives in a different latent space).
    - cache predates fingerprinting: record it when the store is writable, else
      warn that it can't be verified.
    """
    fp = adapter_fingerprint(adapter_ckpt)
    recorded = store.manifest.extra.get(ADAPTER_FP_KEY)
    if recorded == fp:
        return
    if recorded is None:
        store.manifest.extra[ADAPTER_FP_KEY] = fp
        store.manifest.extra.setdefault("adapter_ckpt", str(adapter_ckpt))
        if store.mode != "r":
            store.save_manifest()
            logger.warning(
                "z_m cache %s had no adapter fingerprint; recorded the current "
                "adapter's (%s…). If this cache was built with a DIFFERENT adapter, "
                "delete it and rebuild.", store.path, fp[:12],
            )
        else:
            logger.warning(
                "z_m cache %s has no adapter fingerprint and is read-only; cannot "
                "verify it matches the scoring adapter (%s…).", store.path, fp[:12],
            )
        return
    raise ValueError(
        f"z_m cache {store.path} was built with a DIFFERENT adapter than the "
        f"checkpoint being scored (cache fp={recorded[:12]}…, scorer fp={fp[:12]}…). "
        f"Delete the cache and rebuild it with the matching adapter."
    )


def enforce_cache_n_views(store: EmbeddingStore, n_views: int) -> None:
    """Reject a z_m cache built with a different view count than we expect."""
    recorded = store.manifest.extra.get(N_VIEWS_KEY)
    if recorded is None:
        if n_views <= 1:
            return
        raise ValueError(
            f"z_m cache {store.path} has no {N_VIEWS_KEY} metadata but scoring "
            f"expects n_views={n_views}. Delete it and rebuild with "
            f"build_multiview_cache or lit_pcba --n-views {n_views}."
        )
    if int(recorded) != n_views:
        raise ValueError(
            f"z_m cache {store.path} was built with n_views={recorded} but scoring "
            f"expects n_views={n_views}. Delete the cache and rebuild."
        )


def require_zm_cache_complete(store: EmbeddingStore, inchikeys: list[str]) -> None:
    """Fail fast when scoring would silently miss ligands."""
    missing = sorted({ik for ik in inchikeys if ik not in store.pid_to_row})
    if missing:
        raise ValueError(
            f"z_m cache {store.path} is missing {len(missing)} / "
            f"{len(set(inchikeys))} unique ligands (e.g. {missing[:3]}). "
            f"Rebuild the cache before scoring."
        )


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def _build_encoder(
    *,
    adapter_ckpt: Path | str,
    device: str = "cpu",
) -> DiscreteFlowEncoder:
    return build_eval_encoder(adapter_ckpt, device=device)


def _build_head(
    head_ckpt: Path | str,
    *,
    d_adapter: int | None = None,
    d_protein: int | None = None,
    device: str = "cpu",
) -> EnergyHead:
    return load_energy_head(head_ckpt, d_adapter=d_adapter, d_protein=d_protein, device=device)


# --------------------------------------------------------------------------
# z_m precompute / cache
# --------------------------------------------------------------------------


def _inchikey_or_none(smiles: str) -> str | None:
    """Canonical InChIKey for cache lookup (same helper as preprocessing)."""
    return inchikey_of(smiles)


def fragment_views(smiles: str, n_views: int = 1) -> list[str]:
    """Up to ``n_views`` distinct **deterministic** rBRICS views of ``smiles``.

    The fragmentation is seeded with a per-molecule hash of the SMILES, so the
    views — and therefore every z_m cache / score derived from them — are
    bit-reproducible across runs and machines. Without seeding the fragment
    count is drawn from system entropy, which leaves AUROC/BEDROC stable but
    swings EF@1% by ±0.4–0.5 between encodings.

    Averaging the z_m of several views ("multi-view" test-time augmentation)
    denoises the molecule latent (single views of one molecule share only
    ~0.90 cosine). Falls back to the canonical SMILES if rBRICS can't produce a
    round-tripping cut, or returns ``[]`` if RDKit can't parse the SMILES.
    """
    # Faithful, full-coverage multi-granularity views (seeded per-molecule for
    # reproducibility) — averaging their z_m denoises the latent without ever
    # encoding a lossy fragment subset.
    return seeded_views(smiles, n_views)


def precompute_zm_for_smiles(
    encoder: DiscreteFlowEncoder,
    smiles_list: list[str],
    inchikeys: list[str],
    store: EmbeddingStore,
    *,
    batch_size: int,
    device: str,
    n_jobs: int = 1,
    n_views: int = 4,
) -> int:
    """Encode missing ``(inchikey → z_m)`` pairs, averaging ``n_views`` seeded views.

    Matches ``build_multiview_cache``: fragmentize K deterministic views per
    ligand, encode all views, store ``mean(z_m)`` per InChIKey. Idempotent.
    """
    if n_views < 1:
        raise ValueError(f"n_views must be >= 1, got {n_views}")

    # --- Phase 1: dedupe (fast, O(N) with a set) -------------------------
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for ik, smi in zip(
        tqdm(inchikeys, desc="dedupe", unit="row", dynamic_ncols=True),
        smiles_list,
    ):
        if ik in store.pid_to_row or ik in seen:
            continue
        seen.add(ik)
        unique.append((ik, smi))
    n_already_cached = len({ik for ik in inchikeys if ik in store.pid_to_row})
    logger.info(
        "dedupe: %d rows → %d unique ligands to encode (%d unique already cached)",
        len(inchikeys), len(unique), n_already_cached,
    )
    if not unique:
        return 0

    # --- Phase 2: K seeded views per ligand -------------------------------
    unique_smiles = [s for _, s in unique]
    logger.info(
        "fragmentizing %d unique ligands × %d views with n_jobs=%d",
        len(unique), n_views, n_jobs,
    )
    if n_jobs in (0, 1):
        view_lists = [
            fragment_views(s, n_views)
            for s in tqdm(unique_smiles, desc="fragmentize", unit="mol", dynamic_ncols=True)
        ]
    else:
        from joblib import Parallel, delayed

        view_lists = list(
            tqdm(
                Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
                    delayed(fragment_views)(s, n_views) for s in unique_smiles
                ),
                total=len(unique_smiles),
                desc="fragmentize",
                unit="mol",
                dynamic_ncols=True,
            )
        )

    pending_ids: list[str] = []
    pending_view_lists: list[list[str]] = []
    n_skipped_fragment = 0
    for (ik, _), vs in zip(unique, view_lists):
        if not vs:
            n_skipped_fragment += 1
            continue
        pending_ids.append(ik)
        pending_view_lists.append(vs)
    logger.info(
        "fragmentize: %d valid ligands, %d rdkit-rejected",
        len(pending_ids), n_skipped_fragment,
    )
    if not pending_ids:
        return 0

    # --- Phase 3: batched GPU encode + per-ligand view average ------------
    n_written = 0
    pbar = tqdm(
        range(0, len(pending_ids), batch_size),
        desc=f"encode z_m×{n_views}",
        unit="batch",
        dynamic_ncols=True,
    )
    for i in pbar:
        chunk_ids = pending_ids[i : i + batch_size]
        chunk_views = pending_view_lists[i : i + batch_size]
        flat: list[str] = []
        offs: list[int] = []
        for vs in chunk_views:
            offs.append(len(vs))
            flat.extend(vs)
        with torch.no_grad():
            zf = encoder.encode_views(flat, device=device).detach().cpu().to(torch.float32).numpy()
        out, p = [], 0
        for n in offs:
            out.append(zf[p : p + n].mean(0))
            p += n
        n_written += store.append_mean(chunk_ids, np.asarray(out, dtype=np.float16))
    return n_written


# --------------------------------------------------------------------------
# Scoring + metrics per target
# --------------------------------------------------------------------------


def _score_target(
    head: EnergyHead,
    z_m_rows: np.ndarray,
    z_p: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Return ``-E`` (higher = predicted binder) for ``[L, d_m]`` ``z_m_rows``."""
    z_p_t = torch.from_numpy(z_p.astype(np.float32)).to(device)
    out = np.empty(z_m_rows.shape[0], dtype=np.float32)
    for i in range(0, z_m_rows.shape[0], batch_size):
        chunk = z_m_rows[i : i + batch_size]
        z_m_t = torch.from_numpy(chunk.astype(np.float32)).to(device)
        z_p_b = z_p_t.unsqueeze(0).expand(z_m_t.shape[0], -1)
        with torch.no_grad():
            e = head(z_m_t, z_p_b)
        out[i : i + chunk.shape[0]] = (-e).cpu().numpy()
    return out


def _plot_target_energy_violin(
    target: str, y_true: np.ndarray, y_score: np.ndarray, out_dir: Path
) -> None:
    """Write the same energy-distribution violin as Stage-7 inference, with the
    target's actives as the *known-binder* reference and inactives as the
    *non-binder* reference (no screened library). ``y_score`` is ``-E``, so the
    plotted energy is ``E = -y_score`` (lower = stronger binder)."""
    # Local import avoids a module-level cycle (predict imports from this module).
    from lattice_lab.inference.predict import plot_violin

    energy = -y_score
    e_active = energy[y_true == 1]
    e_inactive = energy[y_true == 0]
    if e_active.size < 2 or e_inactive.size < 2:
        logger.warning("violin[%s]: need >=2 actives and inactives "
                       "(have %d/%d); skipping", target, e_active.size, e_inactive.size)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_violin(
        target_name=target,
        screened=None,
        binders=e_active,
        nonbinders=e_inactive,
        path=out_dir / f"{target}_energy_violin.png",
        ylabel="energy E   (lower = stronger binder)",
    )


def _per_target_metrics(
    y_true: np.ndarray, y_score: np.ndarray, args: argparse.Namespace
) -> dict[str, float]:
    out: dict[str, float] = {
        "n": float(y_true.size),
        "n_active": float(int(y_true.sum())),
        "auroc": auroc(y_true, y_score),
        "bedroc": bedroc(y_true, y_score, alpha=args.bedroc_alpha),
    }
    for p in args.ef_percents:
        out[f"ef@{p}%"] = ef_at_k(y_true, y_score, p)
    return out


# --------------------------------------------------------------------------
# Top-level evaluation
# --------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    """Run the whole Stage-6 pipeline. Returns the per-target results frame."""
    args.zm_cache.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading test parquet: %s", args.test_parquet)
    df = pd.read_parquet(args.test_parquet, columns=["target_name", "smiles", "is_active"])
    df["smiles"] = df["smiles"].astype(str)
    df["target_name"] = df["target_name"].astype(str)
    df["is_active"] = df["is_active"].astype(int)
    targets = sorted(df["target_name"].unique())
    logger.info("targets in test set: %d", len(targets))

    protein_store = EmbeddingStore.open(args.protein_store, mode="r")
    present_targets = [t for t in targets if t in protein_store.pid_to_row]
    missing_targets = [t for t in targets if t not in protein_store.pid_to_row]
    if missing_targets:
        logger.warning(
            "skipping %d/%d targets missing from protein store: %s. "
            "Re-run Stage 3 with --no-canonical-filter to embed them.",
            len(missing_targets), len(targets), missing_targets,
        )

    if args.limit_targets is not None:
        present_targets = present_targets[: args.limit_targets]
        logger.info("limit-targets=%d → scoring %d targets", args.limit_targets, len(present_targets))

    # Diagnostic: map each target to a DIFFERENT target's z_p (cyclic derangement)
    # so every ligand set is scored against the wrong protein. If the metrics
    # barely move vs the normal run, the head is ignoring z_p (target-independent
    # "binder-likeness"), which explains high val EF but random LIT-PCBA.
    zp_target = {t: t for t in present_targets}
    if args.shuffle_zp:
        n = len(present_targets)
        zp_target = {present_targets[i]: present_targets[(i + 1) % n] for i in range(n)}
        logger.warning("DIAGNOSTIC --shuffle-zp: scoring ligands against PERMUTED z_p")

    # The InChIKey is the z_m cache key, so we need one per row. But it is a
    # deterministic, expensive function of SMILES, and LIT-PCBA decoys are shared
    # across targets — so compute keys only for *unique* SMILES and persist them
    # to a sidecar parquet keyed by the test set (reused across zm-cache variants
    # and across reruns).
    work = df[df["target_name"].isin(present_targets)].copy()
    key_cache = args.test_parquet.with_suffix(".inchikeys.parquet")
    unique_smiles = work["smiles"].unique().tolist()
    mapping: dict[str, str | None] = {}
    if key_cache.exists():
        cached = pd.read_parquet(key_cache)
        mapping = dict(zip(cached["smiles"], cached["inchikey"]))
    todo = [s for s in unique_smiles if s not in mapping]
    logger.info("InChIKeys: %d rows, %d unique SMILES (%d cached, %d to compute, n_jobs=%d)…",
                len(work), len(unique_smiles), len(unique_smiles) - len(todo),
                len(todo), args.n_jobs)
    if todo:
        if args.n_jobs in (0, 1):
            new_keys = [
                _inchikey_or_none(s) for s in tqdm(
                    todo, desc="inchikey", unit="mol", dynamic_ncols=True
                )
            ]
        else:
            from joblib import Parallel, delayed

            new_keys = list(
                tqdm(
                    Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(
                        delayed(_inchikey_or_none)(s) for s in todo
                    ),
                    total=len(todo),
                    desc="inchikey", unit="mol", dynamic_ncols=True,
                )
            )
        mapping.update(zip(todo, new_keys))
        key_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"smiles": list(mapping), "inchikey": list(mapping.values())}).to_parquet(
            key_cache, index=False
        )
        logger.info("wrote InChIKey cache: %s (%d entries)", key_cache, len(mapping))
    work["inchikey"] = work["smiles"].map(mapping)
    n_dropped_parse = int(work["inchikey"].isna().sum())
    work = work.dropna(subset=["inchikey"]).reset_index(drop=True)
    if n_dropped_parse:
        logger.warning("dropped %d rows whose SMILES did not parse", n_dropped_parse)

    encoder = _build_encoder(adapter_ckpt=args.adapter_ckpt, device=args.device)
    d_adapter = encoder.adapter.d_adapter

    # z_m cache: same store layout as the protein store, keyed on InChIKey.
    zm_store = EmbeddingStore.create(
        args.zm_cache,
        embedding_dim=d_adapter,
        model_name=f"lattice-adapter-mv{args.n_views}" if args.n_views > 1 else "lattice-adapter-v1",
        dtype="float16",
        per_residue=False,
        extra={
            "source": str(args.test_parquet),
            "adapter_ckpt": str(args.adapter_ckpt),
            ADAPTER_FP_KEY: adapter_fingerprint(args.adapter_ckpt),
            N_VIEWS_KEY: str(args.n_views),
        },
    )
    enforce_cache_adapter(zm_store, args.adapter_ckpt)
    enforce_cache_n_views(zm_store, args.n_views)

    if args.skip_zm_precompute:
        require_zm_cache_complete(zm_store, work["inchikey"].tolist())
        n_new = 0
    else:
        n_new = precompute_zm_for_smiles(
            encoder,
            smiles_list=work["smiles"].tolist(),
            inchikeys=work["inchikey"].tolist(),
            store=zm_store,
            batch_size=args.batch_size,
            device=args.device,
            n_jobs=args.n_jobs,
            n_views=args.n_views,
        )
    logger.info("z_m cache: %d entries (%d newly written)", zm_store.manifest.count, n_new)

    head = _build_head(
        args.head_ckpt,
        d_adapter=d_adapter,
        d_protein=args.d_protein,
        device=args.device,
    )

    rows: list[dict[str, float | str]] = []
    for t in tqdm(present_targets, desc="targets", dynamic_ncols=True):
        sub = work[work["target_name"] == t]
        z_p = protein_store.get_mean(zp_target[t])
        # Pull z_m for the InChIKeys this target sees, in the row order.
        idx = np.fromiter((zm_store.pid_to_row[k] for k in sub["inchikey"]),
                          dtype=np.int64, count=len(sub))
        z_m_rows = np.asarray(zm_store.mean_array[idx], dtype=np.float32)
        y_true = sub["is_active"].to_numpy(dtype=int)
        y_score = _score_target(
            head, z_m_rows, z_p,
            batch_size=args.batch_size, device=args.device,
        )
        m = _per_target_metrics(y_true, y_score, args)
        m["target"] = t
        rows.append(m)
        if args.violin_dir is not None:
            _plot_target_energy_violin(t, y_true, y_score, args.violin_dir)
        logger.info("%-8s n=%d n_a=%d auc=%.3f bedroc=%.3f ef0.5=%.2f ef1=%.2f ef5=%.2f",
                    t, int(m["n"]), int(m["n_active"]), m["auroc"], m["bedroc"],
                    m["ef@0.5%"], m["ef@1.0%"], m["ef@5.0%"])

    # Rows for skipped targets so the CSV captures coverage.
    for t in missing_targets:
        rows.append({"target": t, "n": float("nan"), "n_active": float("nan"),
                     "auroc": float("nan"), "bedroc": float("nan"),
                     **{f"ef@{p}%": float("nan") for p in args.ef_percents}})

    results = pd.DataFrame(rows)
    results = results[
        ["target", "n", "n_active", "auroc", "bedroc"]
        + [f"ef@{p}%" for p in args.ef_percents]
    ]
    # Averages over scored targets only.
    scored = results[results["target"].isin(present_targets)]
    avg_row: dict[str, float | str] = {"target": "_mean"}
    median_row: dict[str, float | str] = {"target": "_median"}
    for col in ["auroc", "bedroc", *[f"ef@{p}%" for p in args.ef_percents]]:
        avg_row[col] = float(scored[col].mean(skipna=True))
        median_row[col] = float(scored[col].median(skipna=True))
    avg_row["n"] = int(scored["n"].sum())
    avg_row["n_active"] = int(scored["n_active"].sum())
    median_row["n"] = avg_row["n"]
    median_row["n_active"] = avg_row["n_active"]
    results = pd.concat(
        [results, pd.DataFrame([avg_row, median_row])], ignore_index=True
    )
    results.to_csv(args.output_csv, index=False)
    logger.info("wrote per-target results to %s", args.output_csv)

    summary: dict[str, float] = {}
    for col in ["auroc", "bedroc", *[f"ef@{p}%" for p in args.ef_percents]]:
        summary[f"avg/{col}"] = float(avg_row[col])
        summary[f"median/{col}"] = float(median_row[col])
    summary["n_targets_scored"] = float(len(present_targets))
    summary["n_targets_skipped"] = float(len(missing_targets))
    logger.info("summary: %s", json.dumps(summary, indent=2))

    return results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-parquet", type=Path,
                        default=Path("artifacts/preprocessing/processed/bindingdb/test_lit_pcba.parquet"))
    parser.add_argument("--head-ckpt", type=Path,
                        default=Path("artifacts/energy/checkpoints/ebm_last.pt"))
    parser.add_argument("--adapter-ckpt", type=Path,
                        default=Path("artifacts/adapter/checkpoints/adapter_v1.pt"))
    parser.add_argument("--protein-store", type=Path,
                        default=Path("artifacts/protein_store/embeddings/esm2_650M/"))
    parser.add_argument("--zm-cache", type=Path, default=None,
                        help="default: artifacts/evaluation/<adapter_run_id>/lit_pcba_zm_sv")
    parser.add_argument("--adapter-run-id", default=None,
                        help="Stage-2 W&B run id for the default zm-cache path")
    parser.add_argument("--output-csv", type=Path,
                        default=Path("artifacts/evaluation/lit_pcba_results.csv"))
    parser.add_argument("--violin-dir", type=Path, default=None,
                        help="if set, write a per-target energy-distribution violin "
                             "PNG (actives vs inactives) here, matching Stage-7 inference")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-views", type=int, default=4,
                        help="seeded rBRICS views per ligand; z_m is the mean embedding")
    parser.add_argument("--skip-zm-precompute", action="store_true",
                        help="score from an existing zm cache only (stage6 after build_multiview_cache)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel workers for the CPU fragmentize step "
                             "(set to e.g. cpu_count() - 1 to speed up the first run)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bedroc-alpha", type=float, default=80.5)
    parser.add_argument("--ef-percents", type=str, default="0.5,1,5",
                        help="comma-separated EF cutoffs in %% of the ranked list")
    parser.add_argument("--shuffle-zp", action="store_true",
                        help="DIAGNOSTIC: score each target's ligands against a "
                             "different target's z_p. If metrics barely drop, the "
                             "head is ignoring the protein.")
    parser.add_argument("--limit-targets", type=int, default=-1,
                        help="score only the first N protein-store targets (smoke runs)")
    parser.add_argument(
        "--d-protein",
        type=int,
        default=None,
        help="protein embedding dim (default: infer from head checkpoint)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.zm_cache is None:
        from lattice_lab.models.builders import adapter_run_id, eval_zm_cache_path

        rid = args.adapter_run_id
        if rid is None:
            try:
                rid = adapter_run_id(args.adapter_ckpt)
            except ValueError as e:
                raise SystemExit(
                    f"{e} — pass --adapter-run-id (Stage-2 W&B run id) or --zm-cache"
                ) from e
        args.zm_cache = eval_zm_cache_path(rid, f"lit_pcba_zm_mv{args.n_views}" if args.n_views > 1 else "lit_pcba_zm_sv")
    args.ef_percents = tuple(float(x) for x in args.ef_percents.split(","))
    args.limit_targets = None if args.limit_targets < 0 else args.limit_targets
    evaluate(args)


if __name__ == "__main__":
    main()
