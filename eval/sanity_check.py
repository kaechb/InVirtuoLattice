"""Stage 2 sanity-check orchestrator.

Runs all three checks (val alignment, bioisostere retrieval, QM9 linear probe),
logs results to W&B under namespaced keys, and returns a single dict telling
the caller whether the adapter passes the README freeze gate.

Two entry points:

- ``run_sanity_checks(encoder, ...)``  — call from inside ``train_adapter`` to
  log into the same W&B run.
- ``main()`` — standalone CLI that loads a saved checkpoint and runs the checks.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from lattice_lab.backbone.adapter import Adapter, AdapterConfig
from lattice_lab.backbone.encoder import EncoderConfig, MoleculeEncoder
from lattice_lab.backbone.fragmol_loader import load_fragmol
from lattice_lab.eval.baselines import (
    LITERATURE_REFERENCES,
    format_reference_table,
    morgan_bioisostere_baseline,
    morgan_qm9_baseline,
)
from lattice_lab.eval.bioisostere_retrieval import (
    DEFAULT_BIOISOSTERE_CSV,
    BioisostereResult,
    evaluate_bioisostere_retrieval,
)
from lattice_lab.eval.qm9_probe import Qm9ProbeResult, evaluate_qm9_probe
from lattice_lab.eval.val_alignment import ValAlignmentResult, evaluate_val_alignment
from lattice_lab.training.run_logger import RunLogger
from lattice_lab.training.ssl_dataset import PairedViewDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SanityReport:
    val: ValAlignmentResult | None
    bioiso: BioisostereResult | None
    qm9: Qm9ProbeResult | None
    bioiso_baseline: BioisostereResult | None
    qm9_baseline: Qm9ProbeResult | None
    all_passed: bool

    def as_metrics(self) -> dict[str, float | int | bool]:
        out: dict[str, float | int | bool] = {}
        if self.val is not None:
            out.update(self.val.as_metrics())
            # Random-chance reference for val top-1 = 1 / N.
            n = self.val.n_pairs
            if n > 0:
                out["reference/val_top1_random"] = 1.0 / n
        if self.bioiso is not None:
            out.update(self.bioiso.as_metrics())
        if self.qm9 is not None:
            out.update(self.qm9.as_metrics())
        if self.bioiso_baseline is not None:
            out["baseline/bioiso_morgan_recall@1"] = self.bioiso_baseline.recall_at_1
            out["baseline/bioiso_morgan_recall@5"] = self.bioiso_baseline.recall_at_5
            out["baseline/bioiso_morgan_recall@10"] = self.bioiso_baseline.recall_at_10
        if self.qm9_baseline is not None:
            for t, r2 in self.qm9_baseline.r2_by_target.items():
                out[f"baseline/qm9_morgan_r2_{t}"] = r2
            out["baseline/qm9_morgan_mean_r2"] = self.qm9_baseline.mean_r2
        # Literature reference values as flat scalars (W&B plots them as
        # horizontal lines when overlaid with the live metric).
        for metric, refs in LITERATURE_REFERENCES.items():
            for label, value, _source in refs:
                if "this run" in label.lower():
                    continue
                if value != value:  # NaN check without importing math
                    continue
                key = f"reference/{metric.replace('/', '_')}__{label}"
                # Slug-friendly key: strip parentheses, dots, etc.
                key = "".join(c if c.isalnum() or c in "_/-" else "_" for c in key)
                out[key] = float(value)
        out["sanity/all_pass"] = bool(self.all_passed)
        return out


def run_sanity_checks(
    encoder: MoleculeEncoder,
    *,
    val_shards: list[Path] | None,
    val_ratio: float = 0.005,
    test_ratio: float = 0.005,
    split_seed: int = 0,
    val_max_pairs: int | None = None,
    bioisostere_csv: Path | str | None = DEFAULT_BIOISOSTERE_CSV,
    qm9_csv: Path | str | None = None,
    qm9_n_subset: int | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    run_logger: RunLogger | None = None,
    step: int | None = None,
    include_baselines: bool = True,
    n_jobs: int | None = None,
) -> SanityReport:
    """Run all available checks; checks with missing inputs are skipped.

    Args:
        val_shards: Stage-1 parquet shards. ``None`` skips val alignment.
        bioisostere_csv: curated pairs CSV; pass ``None`` to skip the check.
        qm9_csv: QM9 CSV (``smiles, homo, lumo, ...``). ``None`` skips the probe.
        run_logger: if provided, results are streamed to W&B as well as returned.
        step: optional ``wandb.log`` step (defaults to ``None`` → wandb autoincrement).
    """
    val_result: ValAlignmentResult | None = None
    bioiso_result: BioisostereResult | None = None
    qm9_result: Qm9ProbeResult | None = None
    bioiso_baseline: BioisostereResult | None = None
    qm9_baseline: Qm9ProbeResult | None = None

    if val_shards:
        logger.info("running val-alignment check on %d shard(s)", len(val_shards))
        val_ds = PairedViewDataset(
            val_shards,
            seed=split_seed,
            split="val",
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
        )
        val_result = evaluate_val_alignment(
            encoder, val_ds, batch_size=batch_size, device=device,
            max_pairs=val_max_pairs,
        )
        logger.info("val alignment: %s", val_result)

    if bioisostere_csv is not None and Path(bioisostere_csv).exists():
        logger.info("running bioisostere-retrieval check on %s", bioisostere_csv)
        bioiso_result = evaluate_bioisostere_retrieval(
            encoder, bioisostere_csv, batch_size=batch_size, device=device,
            n_jobs=n_jobs,
        )
        logger.info("bioisostere: %s", bioiso_result)

    if qm9_csv is not None and Path(qm9_csv).exists():
        logger.info("running QM9 linear-probe (subset=%s)", qm9_n_subset)
        qm9_result = evaluate_qm9_probe(
            encoder, qm9_csv, batch_size=batch_size, device=device,
            n_subset=qm9_n_subset, n_jobs=n_jobs,
        )
        logger.info("qm9 probe: %s", qm9_result)

    if include_baselines:
        if bioisostere_csv is not None and Path(bioisostere_csv).exists():
            logger.info("computing Morgan FP / Tanimoto baseline on bioisostere set")
            bioiso_baseline = morgan_bioisostere_baseline(bioisostere_csv)
            logger.info("morgan bioiso baseline: %s", bioiso_baseline)
        if qm9_csv is not None and Path(qm9_csv).exists():
            logger.info("computing Morgan FP / Ridge baseline on QM9")
            qm9_baseline = morgan_qm9_baseline(qm9_csv, n_subset=qm9_n_subset)
            logger.info("morgan qm9 baseline: %s", qm9_baseline)

    # All-pass requires every *run* check to have passed (skipped → not penalized).
    flags = [r.passed for r in (val_result, bioiso_result, qm9_result) if r is not None]
    all_passed = bool(flags) and all(flags)
    report = SanityReport(
        val=val_result, bioiso=bioiso_result, qm9=qm9_result,
        bioiso_baseline=bioiso_baseline, qm9_baseline=qm9_baseline,
        all_passed=all_passed,
    )

    # Print reference comparison tables to stdout so users see context inline.
    if val_result is not None:
        print(format_reference_table("val/top1_acc", val_result.top1_acc))
    if bioiso_result is not None:
        b_val = bioiso_baseline.recall_at_10 if bioiso_baseline else None
        print(format_reference_table("bioiso/recall@10", bioiso_result.recall_at_10, b_val))
    if qm9_result is not None:
        for t, r2 in qm9_result.r2_by_target.items():
            b_val = qm9_baseline.r2_by_target.get(t) if qm9_baseline else None
            print(format_reference_table(f"qm9/r2_{t}", r2, b_val))
    if run_logger is not None:
        run_logger.log(report.as_metrics(), step=step)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="Path to adapter_v1.pt produced by Stage 2 training")
    parser.add_argument("--val-shards", type=Path, default=Path("01_preprocessing/processed"),
                        help="Shard dir containing Stage-1 parquet (val split derived from here)")
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument("--test-ratio", type=float, default=0.005)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-max-pairs", type=int, default=-1,
                        help="Cap val pairs for retrieval (default -1 = use full split). "
                             "Note: memory is O(N²); set a cap for very large val sets.")
    parser.add_argument("--bioisostere-csv", type=Path, default=DEFAULT_BIOISOSTERE_CSV)
    parser.add_argument("--qm9-csv", type=Path, default=Path("00_data/raw/qm9.csv"))
    parser.add_argument("--qm9-n-subset", type=int, default=-1,
                        help="QM9 random subsample (default -1 = use full 134K)")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip Morgan FP baselines (faster; less context)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="CPU workers for SMILES→view fragmentation "
                             "(default -1 = cpu_count - 1; pass 1 to disable)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="lattice")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    bundle = load_fragmol(device=args.device)
    # ``pathlib.PosixPath`` (and ``WindowsPath`` on win) appear in saved configs;
    # allowlist them so ``weights_only=True`` still works on legacy checkpoints
    # that were written before we started serializing paths as strings.
    import pathlib
    path_classes = [pathlib.PosixPath, pathlib.WindowsPath, pathlib.Path]
    with torch.serialization.safe_globals(path_classes):
        ck = torch.load(args.adapter, map_location="cpu", weights_only=True)
    saved_cfg = ck.get("cfg", {})
    adapter = Adapter(
        AdapterConfig(
            d_fragmol=bundle.n_embd,
            n_fragmol_layers=saved_cfg.get("n_fragmol_layers", 4),
            d_adapter=saved_cfg.get("d_adapter", 512),
            n_layers=saved_cfg.get("n_adapter_layers", 4),
        )
    )
    adapter.load_state_dict(ck["adapter_state_dict"])
    adapter.to(args.device).eval()
    encoder = MoleculeEncoder(bundle, adapter=adapter,
                              config=EncoderConfig(
                                  n_fragmol_layers=saved_cfg.get("n_fragmol_layers", 4)
                              ))

    val_shards = sorted(args.val_shards.glob("shard_*.parquet")) if args.val_shards else []
    qm9_subset = None if args.qm9_n_subset < 0 else args.qm9_n_subset
    val_cap = None if args.val_max_pairs < 0 else args.val_max_pairs
    n_jobs = None if args.n_jobs < 0 else args.n_jobs

    with RunLogger(
        project=args.wandb_project,
        run_name=args.wandb_run_name or "sanity-check",
        config={"adapter_path": str(args.adapter), **saved_cfg},
        tags=["stage2", "sanity"],
    ) as rl:
        report = run_sanity_checks(
            encoder,
            val_shards=val_shards or None,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
            val_max_pairs=val_cap,
            bioisostere_csv=args.bioisostere_csv,
            qm9_csv=args.qm9_csv if Path(args.qm9_csv).exists() else None,
            qm9_n_subset=qm9_subset,
            device=args.device,
            batch_size=args.batch_size,
            run_logger=rl,
            include_baselines=not args.no_baselines,
            n_jobs=n_jobs,
        )
    logger.info("== sanity report ==")
    logger.info("%s", asdict(report))
    if not report.all_passed:
        logger.warning("ADAPTER DID NOT PASS — see metrics above")
    raise SystemExit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
