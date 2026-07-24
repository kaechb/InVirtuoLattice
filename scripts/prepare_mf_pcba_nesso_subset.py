"""Prepare the eight publicly reconstructable Nesso/Boltz-2 MF-PCBA assays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "artifacts/benchmarks/mf_pcba/mf_pcba_bind_val_test_full.parquet"
OUT = REPO / "artifacts/benchmarks/mf_pcba/nesso_subset8_seed0.parquet"
FASTA = REPO / "artifacts/benchmarks/mf_pcba/nesso_subset8.fasta"

# The other two assays in Nesso/Boltz-2 (AID493155-485273 and
# AID588524-489030) are classified as phenotypic/PPI and are absent from the
# public PAINS-filtered MF-PCBA-Bind reconstruction.
AIDS = (
    "AID1053173-743445",
    "AID493248-485317",
    "AID434954-2097",
    "AID540297-493091",
    "AID463203-2650",
    "AID504329",
    "AID588689",
    "AID624273-588549",
)
EXPECTED_ACTIVES = {
    "AID1053173-743445": 144,
    "AID493248-485317": 976,
    "AID434954-2097": 522,
    "AID540297-493091": 782,
    "AID463203-2650": 612,
    "AID504329": 466,
    "AID588689": 486,
    "AID624273-588549": 159,
}


def prepare(raw: Path, output: Path, fasta: Path, seed: int, per_assay: int) -> pd.DataFrame:
    columns = ["CID", "smiles", "binds", "protein_name", "protein_accession",
               "amino_acid_sequence", "AID"]
    data = pd.read_parquet(raw, columns=columns, filters=[("AID", "in", AIDS)])
    rows = []
    fasta_records = []
    summary = []
    for aid in AIDS:
        assay = data.loc[data.AID.eq(aid)].copy()
        active = assay.loc[assay.binds.astype(bool)]
        inactive = assay.loc[~assay.binds.astype(bool)]
        if len(active) != EXPECTED_ACTIVES[aid]:
            raise ValueError(f"{aid}: expected {EXPECTED_ACTIVES[aid]} actives, got {len(active)}")
        n_inactive = per_assay - len(active)
        if n_inactive > len(inactive):
            raise ValueError(f"{aid}: only {len(inactive)} inactives available")
        sampled = pd.concat(
            [active, inactive.sample(n=n_inactive, random_state=seed)],
            ignore_index=True,
        ).sample(frac=1, random_state=seed).reset_index(drop=True)
        sampled["target_name"] = aid
        sampled["is_active"] = sampled.binds.astype(bool)
        rows.append(sampled)

        sequence = assay.amino_acid_sequence.iloc[0]
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"{aid}: missing protein sequence")
        fasta_records.append(f">{aid} {assay.protein_name.iloc[0]}\n{sequence}\n")
        summary.append({
            "target": aid,
            "protein": assay.protein_name.iloc[0],
            "accession": assay.protein_accession.iloc[0],
            "n": len(sampled),
            "n_active": len(active),
        })

    result = pd.concat(rows, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result[["target_name", "smiles", "is_active", "CID", "protein_name",
            "protein_accession"]].to_parquet(output, index=False)
    fasta.write_text("".join(fasta_records))
    output.with_suffix(".json").write_text(json.dumps({
        "source": str(raw),
        "seed": seed,
        "per_assay": per_assay,
        "assays": summary,
        "comparison_caveat": (
            "Eight of the ten Nesso/Boltz-2 assays; exact sampled compounds and "
            "random seed were not released by the authors."
        ),
    }, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"wrote {len(result)} rows -> {output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--fasta", type=Path, default=FASTA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-assay", type=int, default=50_000)
    args = parser.parse_args()
    prepared = prepare(args.raw, args.output, args.fasta, args.seed, args.per_assay)
    assert len(prepared) == len(AIDS) * args.per_assay
