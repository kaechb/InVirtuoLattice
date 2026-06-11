"""LIT-PCBA loader.

LIT-PCBA (Tran-Nguyen et al., JCIM 2020) is a held-out benchmark of 15 targets
with experimentally-determined actives and a much larger pool of decoys (sourced
from PubChem BioAssay). DrugCLIP uses it zero-shot — train on BindingDB / BioLiP,
evaluate retrieval (EF@1%) per target.

This module loads a LIT-PCBA dump prepared by ``00_data/copy_lit_pcba.sh``:

    00_data/raw/lit_pcba/<TARGET>/
        actives.smi               (SMILES <tab/space> PubChem CID)
        inactives.smi             (same format)
        <pdb>_protein.mol2        one representative protein structure / target

For each target we:

1.  Read ``actives.smi`` and ``inactives.smi`` into ``LitPcbaLigand`` rows.
2.  Pick the **first** protein mol2 (sorted lexicographically — deterministic),
    parse the @<TRIPOS>ATOM section, and recover the amino-acid sequence by
    walking residues in atom order.
3.  Return ``LitPcbaTarget`` records carrying ligands + sequence + the source
    PDB id so downstream code can write a FASTA for MMseqs2.

Why parse the mol2 directly instead of going via PDB:
- The LIT-PCBA distribution already ships the structure we will use as the
  reference. Re-fetching from PDB would risk a chain/segment mismatch.
- mol2 is parseable in <100 LOC without biopython; the alternative is a
  3-letter→1-letter table we already need anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Standard 20 residues + selenocysteine/pyrrolysine collapsed to X for parity
# with ESM-2's tokenizer behaviour on non-canonical residues.
_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    # Common protonation / tautomer variants.
    "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "CYM": "C",
    "ASH": "D", "GLH": "E", "LYN": "K",
    # Non-canonical.
    "MSE": "M", "SEC": "U", "PYL": "O",
}


@dataclass(frozen=True)
class LitPcbaLigand:
    """One row of an ``actives.smi`` / ``inactives.smi``."""

    smiles: str
    cid: str           # PubChem CID — second whitespace-separated column.
    is_active: bool


@dataclass
class LitPcbaTarget:
    """One LIT-PCBA target with reference structure + ligand pool."""

    name: str                              # e.g. "ADRB2", "ESR1_ago"
    sequence: str                          # amino-acid sequence from .mol2
    pdb_id: str                            # 4-letter PDB code of the reference
    actives: list[LitPcbaLigand] = field(default_factory=list)
    inactives: list[LitPcbaLigand] = field(default_factory=list)

    @property
    def ligands(self) -> list[LitPcbaLigand]:
        return self.actives + self.inactives


def parse_smi(path: str | Path, *, is_active: bool) -> list[LitPcbaLigand]:
    """Parse ``actives.smi`` / ``inactives.smi``.

    Format: ``<SMILES><whitespace><CID>`` per line. Blank lines / comment lines
    starting with ``#`` are ignored.
    """
    out: list[LitPcbaLigand] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles = parts[0]
            cid = parts[1] if len(parts) > 1 else ""
            out.append(LitPcbaLigand(smiles=smiles, cid=cid, is_active=is_active))
    return out


def _residue_key_from_mol2_line(parts: list[str]) -> tuple[str, str] | None:
    """Extract ``(residue_id, residue_3letter)`` from one ATOM record.

    mol2 ATOM line layout (tab-or-space separated):
        atom_id  atom_name  x  y  z  atom_type  subst_id  subst_name  charge

    ``subst_name`` is the residue label such as ``GLY124`` or ``HIS27``. We
    strip the trailing digits to obtain the 3-letter code; ``subst_id`` is the
    numeric residue id used to detect transitions.
    """
    if len(parts) < 8:
        return None
    subst_id = parts[6]
    subst_name = parts[7]
    # Strip trailing digits to expose the 3-letter prefix (e.g. GLY124 → GLY).
    prefix = "".join(ch for ch in subst_name if ch.isalpha())
    if len(prefix) < 3:
        return None
    return subst_id, prefix[:3].upper()


def parse_mol2_sequence(path: str | Path) -> str:
    """Return the one-letter amino-acid sequence parsed from a protein mol2.

    Walks the @<TRIPOS>ATOM section once; residues are emitted in the order
    they first appear (no sorting — preserves chain order produced by the
    upstream alignment tool). Non-canonical residues map to ``X``.
    """
    seq: list[str] = []
    seen_residues: set[str] = set()
    in_atom = False
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>"):
                in_atom = stripped == "@<TRIPOS>ATOM"
                continue
            if not in_atom:
                continue
            parts = stripped.split()
            kv = _residue_key_from_mol2_line(parts)
            if kv is None:
                continue
            subst_id, three = kv
            if subst_id in seen_residues:
                continue
            seen_residues.add(subst_id)
            seq.append(_THREE_TO_ONE.get(three, "X"))
    return "".join(seq)


def _pick_reference_mol2(target_dir: Path) -> Path | None:
    mol2s = sorted(target_dir.glob("*_protein.mol2"))
    return mol2s[0] if mol2s else None


def load_target(target_dir: str | Path) -> LitPcbaTarget:
    """Load one LIT-PCBA target folder into a ``LitPcbaTarget``."""
    target_dir = Path(target_dir)
    name = target_dir.name
    actives_path = target_dir / "actives.smi"
    inactives_path = target_dir / "inactives.smi"
    if not actives_path.exists() or not inactives_path.exists():
        raise FileNotFoundError(f"{name}: missing actives.smi / inactives.smi in {target_dir}")
    ref_mol2 = _pick_reference_mol2(target_dir)
    if ref_mol2 is None:
        raise FileNotFoundError(f"{name}: no *_protein.mol2 under {target_dir}")
    sequence = parse_mol2_sequence(ref_mol2)
    if not sequence:
        raise ValueError(f"{name}: empty sequence parsed from {ref_mol2}")
    pdb_id = ref_mol2.stem.split("_")[0]
    return LitPcbaTarget(
        name=name,
        sequence=sequence,
        pdb_id=pdb_id,
        actives=parse_smi(actives_path, is_active=True),
        inactives=parse_smi(inactives_path, is_active=False),
    )


def load_all(root: str | Path) -> list[LitPcbaTarget]:
    """Load every target subfolder under ``root``.

    Subfolders whose name starts with ``_`` are skipped (e.g. ``_feat_cache``).
    Targets are returned in lexicographic order by name.
    """
    root = Path(root)
    targets: list[LitPcbaTarget] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        try:
            targets.append(load_target(sub))
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("skipping LIT-PCBA target %s: %s", sub.name, exc)
    return targets


def write_fasta(targets: list[LitPcbaTarget], out_path: str | Path) -> None:
    """Write one FASTA entry per LIT-PCBA target (``>NAME`` header)."""
    with open(out_path, "w") as fh:
        for t in targets:
            fh.write(f">{t.name}\n{t.sequence}\n")


def to_records(targets: list[LitPcbaTarget]) -> list[dict[str, object]]:
    """Flatten LIT-PCBA actives + inactives into the shared ligand-row schema.

    One row per (target, ligand). ``uniprot`` mirrors ``target_name`` (the
    LIT-PCBA dump ships no UniProt mapping) so the column lines up with the
    BindingDB / DUD-E frames.
    """
    rows: list[dict[str, object]] = []
    for t in targets:
        for lig in t.ligands:
            rows.append(
                {
                    "target_name": t.name,
                    "uniprot": t.name,        # no UniProt mapping in LIT-PCBA dump; use target name as key
                    "pdb_id": t.pdb_id,
                    "sequence": t.sequence,
                    "smiles": lig.smiles,
                    "pubchem_cid": lig.cid,
                    "is_active": lig.is_active,
                }
            )
    return rows
