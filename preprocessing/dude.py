"""DUD-E loader.

DUD-E (Mysinger et al., J. Med. Chem. 2012) is a 102-target virtual-screening
benchmark: each target ships experimentally-confirmed actives plus ~50×
property-matched but topologically dissimilar decoys drawn from ZINC. Like
LIT-PCBA it is used zero-shot — train on BindingDB / BioLiP, evaluate
per-target early-recognition (EF@1%, BEDROC).

This module loads a DUD-E dump prepared by ``00_data/download_dude.sh``:

    00_data/raw/dude/<TARGET>/
        actives_final.ism        SMILES <space> internal_id [<space> ChEMBL_id]
        decoys_final.ism         SMILES <space> ZINC-style decoy_id
        receptor.pdb             one representative receptor structure / target

For each target we:

1.  Read ``actives_final.ism`` / ``decoys_final.ism`` into ``DudeLigand`` rows
    (SMILES is column 0; the molecule id is column 1).
2.  Parse ``receptor.pdb`` and recover the amino-acid sequence by walking the
    ATOM/HETATM records in file order, emitting one residue per
    ``(chain, residue-number, insertion-code)`` the first time it appears.
3.  Return ``DudeTarget`` records carrying ligands + sequence so downstream code
    can write a FASTA for ESM-2 and flatten to a test parquet.

This mirrors ``lattice/preprocessing/lit_pcba.py``; the only real differences
are the ligand file format (``.ism`` vs ``.smi``) and the reference structure
(a PDB receptor vs a mol2). The 3-letter→1-letter residue table is shared with
the LIT-PCBA loader so the two stay in lockstep.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from lattice_lab.preprocessing.lit_pcba import _THREE_TO_ONE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DudeLigand:
    """One row of an ``actives_final.ism`` / ``decoys_final.ism``."""

    smiles: str
    mol_id: str        # second whitespace-separated column (ChEMBL/ZINC-ish id).
    is_active: bool


@dataclass
class DudeTarget:
    """One DUD-E target with reference receptor sequence + ligand pool."""

    name: str                              # e.g. "aa2ar", "thrb"
    sequence: str                          # amino-acid sequence from receptor.pdb
    actives: list[DudeLigand] = field(default_factory=list)
    decoys: list[DudeLigand] = field(default_factory=list)

    @property
    def ligands(self) -> list[DudeLigand]:
        return self.actives + self.decoys


def parse_ism(path: str | Path, *, is_active: bool) -> list[DudeLigand]:
    """Parse a DUD-E ``.ism`` file.

    Format: ``<SMILES><whitespace><id>[<whitespace><id2>…]`` per line. SMILES is
    column 0 and the molecule id is column 1 (most ``actives_final.ism`` also
    carry a third ChEMBL column, which we ignore; a few — e.g. ``ampc`` — have
    only two columns). Blank lines and ``#`` comments are skipped.
    """
    out: list[DudeLigand] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles = parts[0]
            mol_id = parts[1] if len(parts) > 1 else ""
            out.append(DudeLigand(smiles=smiles, mol_id=mol_id, is_active=is_active))
    return out


def parse_pdb_sequence(path: str | Path) -> str:
    """Return the one-letter sequence of the **longest chain** in a receptor PDB.

    Many DUD-E receptors are oligomers (e.g. ``hmdh`` ships four copies) or
    hetero-complexes where the catalytic subunit is *not* the first chain (e.g.
    ``thrb`` lists thrombin's 27-residue light chain before its 250-residue
    heavy chain). Concatenating all chains would duplicate residues and skew the
    mean-pooled ESM-2 embedding, and taking the first chain would sometimes pick
    a small fragment — so we group residues per chain and return the longest
    one (ties broken by file order).

    Within a chain, residues are emitted once per
    ``(residue-number, insertion-code)`` in file order. Column positions follow
    the fixed PDB format. ``HETATM`` records are kept only when their residue
    name is a known (possibly modified) amino acid such as ``MSE`` — waters,
    ions and ligands are skipped rather than turned into ``X``. Unrecognised
    polymer (``ATOM``) residues map to ``X``.
    """
    chains: "OrderedDict[str, list[str]]" = OrderedDict()
    seen: dict[str, set[tuple[str, str]]] = {}
    with open(path) as fh:
        for line in fh:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            res_name = line[17:20].strip().upper()
            # HETATM that isn't a recognised amino acid is solvent/ligand/ion.
            if record == "HETATM" and res_name not in _THREE_TO_ONE:
                continue
            chain = line[21]
            key = (line[22:26].strip(), line[26])
            chains.setdefault(chain, [])
            seen.setdefault(chain, set())
            if key in seen[chain]:
                continue
            seen[chain].add(key)
            chains[chain].append(_THREE_TO_ONE.get(res_name, "X"))
    if not chains:
        return ""
    return "".join(max(chains.values(), key=len))


def load_target(target_dir: str | Path) -> DudeTarget:
    """Load one DUD-E target folder into a ``DudeTarget``."""
    target_dir = Path(target_dir)
    name = target_dir.name
    actives_path = target_dir / "actives_final.ism"
    decoys_path = target_dir / "decoys_final.ism"
    receptor_path = target_dir / "receptor.pdb"
    if not actives_path.exists() or not decoys_path.exists():
        raise FileNotFoundError(
            f"{name}: missing actives_final.ism / decoys_final.ism in {target_dir}"
        )
    if not receptor_path.exists():
        raise FileNotFoundError(f"{name}: no receptor.pdb under {target_dir}")
    sequence = parse_pdb_sequence(receptor_path)
    if not sequence:
        raise ValueError(f"{name}: empty sequence parsed from {receptor_path}")
    return DudeTarget(
        name=name,
        sequence=sequence,
        actives=parse_ism(actives_path, is_active=True),
        decoys=parse_ism(decoys_path, is_active=False),
    )


def load_all(root: str | Path) -> list[DudeTarget]:
    """Load every target subfolder under ``root``.

    Subfolders whose name starts with ``_`` are skipped. Targets are returned in
    lexicographic order by name.
    """
    root = Path(root)
    targets: list[DudeTarget] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        try:
            targets.append(load_target(sub))
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("skipping DUD-E target %s: %s", sub.name, exc)
    return targets


def write_fasta(targets: list[DudeTarget], out_path: str | Path) -> None:
    """Write one FASTA entry per DUD-E target (``>NAME`` header)."""
    with open(out_path, "w") as fh:
        for t in targets:
            fh.write(f">{t.name}\n{t.sequence}\n")


def to_records(targets: list[DudeTarget]) -> list[dict[str, object]]:
    """Flatten DUD-E actives + decoys into the shared ligand-row schema.

    One row per (target, ligand). ``uniprot`` mirrors ``target_name`` because
    DUD-E ships no UniProt mapping, so the column lines up with the BindingDB /
    LIT-PCBA frames. The evaluators only read ``target_name``, ``smiles`` and
    ``is_active``; ``sequence`` / ``mol_id`` are kept for provenance.
    """
    rows: list[dict[str, object]] = []
    for t in targets:
        for lig in t.ligands:
            rows.append(
                {
                    "target_name": t.name,
                    "uniprot": t.name,      # DUD-E ships no UniProt mapping; key on the target name
                    "sequence": t.sequence,
                    "smiles": lig.smiles,
                    "mol_id": lig.mol_id,
                    "is_active": lig.is_active,
                }
            )
    return rows
