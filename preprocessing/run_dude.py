"""Stage 1 — flatten the DUD-E benchmark into a test parquet + target FASTA.

DUD-E is a held-out test set only (no training rows, no homology split — that
filtering lives in ``run_bindingdb.py`` for LIT-PCBA). This orchestrator just
parses ``00_data/raw/dude/`` and writes, mirroring the LIT-PCBA test artifacts:

    <out>/test_dude.parquet      # one row per (target, ligand)
    <out>/dude_targets.fasta     # one entry per target (name -> receptor seq)
    <out>/manifest.json          # counts + target list

The parquet schema matches ``test_lit_pcba.parquet`` on the columns the
evaluators read (``target_name``, ``smiles``, ``is_active``), plus ``sequence``
and ``mol_id`` for provenance. ``uniprot`` is set to the target name (DUD-E ships
no UniProt mapping) so the column line up with the BindingDB / LIT-PCBA frames.

Idempotent: existing outputs are reused unless ``--overwrite`` is set.

Example::

    python -m lattice.preprocessing.run_dude \\
        --dude-dir   00_data/raw/dude \\
        --output-dir 01_preprocessing/processed_dude
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from lattice_lab.preprocessing import dude

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dude-dir", type=Path, default=Path("00_data/raw/dude"),
                   help="Directory with one subfolder per DUD-E target.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("01_preprocessing/processed_dude"))
    p.add_argument("--overwrite", action="store_true",
                   help="Re-write outputs even if they already exist.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    fasta_path = out_root / "dude_targets.fasta"
    parquet_path = out_root / "test_dude.parquet"
    manifest_path = out_root / "manifest.json"

    if (
        fasta_path.exists()
        and parquet_path.exists()
        and manifest_path.exists()
        and not args.overwrite
    ):
        logger.info("DUD-E outputs already present in %s — re-using "
                    "(pass --overwrite to regenerate).", out_root)
        return

    logger.info("loading DUD-E from %s", args.dude_dir)
    targets = dude.load_all(args.dude_dir)
    if not targets:
        raise SystemExit(f"no DUD-E targets found in {args.dude_dir}")

    dude.write_fasta(targets, fasta_path)
    logger.info("wrote %d target sequences to %s", len(targets), fasta_path)

    records = dude.to_records(targets)
    pd.DataFrame.from_records(records).to_parquet(parquet_path, index=False)
    n_actives = sum(len(t.actives) for t in targets)
    n_decoys = sum(len(t.decoys) for t in targets)
    logger.info("wrote %d ligand rows (%d actives, %d decoys) to %s",
                len(records), n_actives, n_decoys, parquet_path)

    manifest = {
        "dude_dir": str(args.dude_dir),
        "n_targets": len(targets),
        "n_rows": len(records),
        "n_actives": n_actives,
        "n_decoys": n_decoys,
        "targets": [t.name for t in targets],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
