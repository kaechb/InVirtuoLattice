"""SSL data for the discrete-flow backbone: fragment-shuffle contrastive pairs.

Each MOSES molecule has a space-separated fragmented-SMILES string (the
``fragment_view`` column from Stage-1). The contrastive augmentation is a
**fragment shuffle**: tokenize the view, split the token ids on the separator
id (the space token, id 4 in the discrete-flow tokenizer), shuffle the fragment
order, and rejoin. Two independent shuffles give the two views of a molecule.

The shuffle is a pure token-level op (:func:`shuffle_fragment_ids`) so it's unit
tested without the model. Stage 1 may store pretokenized ``body_ids`` in parquet
(``run_preprocessing --tokenizer-path``); otherwise workers or the LightningModule
tokenizes ``fragment_view`` strings at load time. Shuffle/mask always run online.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Sequence

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def fragment_split_mask(
    inchikeys: pd.Series,
    *,
    split: str,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> pd.Series:
    """Boolean mask selecting molecules for ``train`` / ``val`` / ``test``."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train/val/test, got {split!r}")
    buckets = inchikeys.astype(str).map(
        lambda k: (hash((split_seed, k)) % 10_000) / 10_000.0
    )
    if split == "train":
        return buckets >= (val_ratio + test_ratio)
    if split == "val":
        return (buckets >= test_ratio) & (buckets < val_ratio + test_ratio)
    return buckets < test_ratio


def load_fragment_split_df(
    shards: list[Path],
    *,
    split: str,
    val_ratio: float = 0.005,
    test_ratio: float = 0.005,
    split_seed: int = 0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load deduplicated ``view_idx==0`` rows for a MOSES split."""
    from lattice_lab.preprocessing.molecules import fragment_view_column_for_parquet

    requested_cols = columns or ["smiles", "inchikey", "view_idx", "fragment_view"]
    wants_view = "fragment_view" in requested_cols or "fragmol_view" in requested_cols
    frames = []
    for shard in shards:
        view_col = fragment_view_column_for_parquet(shard) if wants_view else None
        use_cols = [view_col if c in {"fragment_view", "fragmol_view"} else c for c in requested_cols]
        df = pd.read_parquet(shard, columns=use_cols)
        if view_col is not None and view_col != "fragment_view":
            df = df.rename(columns={view_col: "fragment_view"})
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["view_idx"] == 0].drop_duplicates("inchikey").reset_index(drop=True)
    mask = fragment_split_mask(
        df["inchikey"], split=split, val_ratio=val_ratio,
        test_ratio=test_ratio, split_seed=split_seed,
    )
    return df.loc[mask].reset_index(drop=True)


def split_fragment_ids(ids: list[int], sep_id: int) -> list[list[int]]:
    """Split token ids on ``sep_id`` into fragment lists (empty frags allowed)."""
    frags: list[list[int]] = [[]]
    for t in ids:
        if t == sep_id:
            frags.append([])
        else:
            frags[-1].append(t)
    return frags


def join_fragment_ids(frags: list[list[int]], sep_id: int) -> list[int]:
    """Rejoin fragment token lists with ``sep_id`` between fragments."""
    out: list[int] = []
    for i, frag in enumerate(frags):
        if i:
            out.append(sep_id)
        out.extend(frag)
    return out


def shuffle_frags(frags: list[list[int]], sep_id: int, rng: random.Random) -> list[int]:
    """Shuffle a pre-split fragment list and rejoin (no re-scan to split).

    A single fragment is returned (joined) unchanged. The input list is not
    mutated, so the same ``frags`` can be reused across multiple views.
    """
    if len(frags) <= 1:
        return join_fragment_ids(frags, sep_id)
    frags = list(frags)
    rng.shuffle(frags)
    return join_fragment_ids(frags, sep_id)


def shuffle_fragment_ids(ids: list[int], sep_id: int, rng: random.Random) -> list[int]:
    """Split ``ids`` on ``sep_id``, shuffle the fragments, rejoin with ``sep_id``.

    A single-fragment sequence (no ``sep_id``) is returned unchanged. Leading /
    trailing separators yield empty fragments, which are preserved (so the op is
    exactly invertible in fragment count). Thin wrapper over :func:`shuffle_frags`
    for callers that only have the flat token ids.
    """
    return shuffle_frags(split_fragment_ids(ids, sep_id), sep_id, rng)


def mask_frags(
    frags: list[list[int]],
    sep_id: int,
    mask_id: int,
    rng: random.Random,
    *,
    frac: float = 0.5,
    frag_idx: int | None = None,
) -> list[int]:
    """Fragment-list form of :func:`mask_fragment_ids` (no re-scan to split).

    Masks ``max(1, round(frac * n))`` of the ``n`` non-empty fragments, always
    leaving >= 1 fragment intact so the context keeps real signal (masking one
    fragment of a 3-fragment median molecule is too gentle — the masked and
    intact pooled embeddings stay nearly equal, so the invariance task is
    trivial). A single-fragment molecule has no fragment to spare, so it instead
    masks ~``frac`` of the *tokens* within that fragment (again leaving >= 1),
    keeping the ~19% single-fragment MOSES molecules from degenerating into an
    information-free all-``mask_id`` context. Uses ``mask_id`` (never PAD).

    ``frag_idx`` forces a single specific fragment to be fully masked (test hook;
    ignores ``frac``). The input list is not mutated.
    """
    frags = list(frags)
    non_empty = [i for i, frag in enumerate(frags) if frag]
    if not non_empty:
        return [mask_id] * (sum(len(f) for f in frags) + max(len(frags) - 1, 0))
    if frag_idx is not None:
        idx = frag_idx if frag_idx in non_empty else rng.choice(non_empty)
        frags[idx] = [mask_id] * len(frags[idx])
        return join_fragment_ids(frags, sep_id)
    if len(non_empty) >= 2:
        k = min(len(non_empty) - 1, max(1, round(frac * len(non_empty))))
        for idx in rng.sample(non_empty, k):
            frags[idx] = [mask_id] * len(frags[idx])
        return join_fragment_ids(frags, sep_id)
    # Single fragment: mask a fraction of its tokens, keeping >= 1 token.
    only = non_empty[0]
    tok = frags[only]
    k = min(len(tok) - 1, max(1, round(frac * len(tok)))) if len(tok) > 1 else 1
    masked_pos = set(rng.sample(range(len(tok)), k))
    frags[only] = [mask_id if i in masked_pos else t for i, t in enumerate(tok)]
    return join_fragment_ids(frags, sep_id)


def mask_fragment_ids(
    ids: list[int],
    sep_id: int,
    mask_id: int,
    rng: random.Random,
    *,
    frac: float = 0.5,
    frag_idx: int | None = None,
) -> list[int]:
    """Flat-ids wrapper over :func:`mask_frags`."""
    if not ids:
        return []
    return mask_frags(
        split_fragment_ids(ids, sep_id), sep_id, mask_id, rng,
        frac=frac, frag_idx=frag_idx,
    )


def mask_span_ids(
    ids: list[int],
    mask_id: int,
    rng: random.Random,
    *,
    frac: float = 0.5,
) -> list[int]:
    """Replace ~``frac`` of the sequence with one contiguous ``mask_id`` span.

    Always leaves >= 1 real token so the masked view keeps context. Spans may
    cross fragment boundaries (including ``sep_id`` tokens).
    """
    if not ids:
        return []
    n = len(ids)
    if n <= 1:
        return list(ids)
    k = min(n - 1, max(1, round(frac * n)))
    start = rng.randrange(0, n - k + 1)
    return [mask_id if start <= i < start + k else t for i, t in enumerate(ids)]


def mask_local_frags(
    frags: list[list[int]],
    ids: list[int],
    sep_id: int,
    mask_id: int,
    rng: random.Random,
    *,
    frac: float = 0.5,
    mode: str = "fragment",
) -> list[int]:
    """Local-view mask from a pre-split fragment list (no re-scan to split).

    ``ids`` is the joined sequence of ``frags``; span masks need it because they
    cross fragment boundaries (including ``sep_id`` tokens).
    """
    if mode == "fragment":
        return mask_frags(frags, sep_id, mask_id, rng, frac=frac)
    if mode == "span":
        return mask_span_ids(ids, mask_id, rng, frac=frac)
    if mode == "mixed":
        pick = "span" if rng.random() < 0.5 else "fragment"
        return mask_local_frags(frags, ids, sep_id, mask_id, rng, frac=frac, mode=pick)
    raise ValueError(f"mask mode must be fragment, span, or mixed, got {mode!r}")


def mask_local_ids(
    ids: list[int],
    sep_id: int,
    mask_id: int,
    rng: random.Random,
    *,
    frac: float = 0.5,
    mode: str = "fragment",
) -> list[int]:
    """Flat-ids wrapper over :func:`mask_local_frags`."""
    return mask_local_frags(
        split_fragment_ids(ids, sep_id), ids, sep_id, mask_id, rng,
        frac=frac, mode=mode,
    )


def _join_fragment_holes(
    frags: list[list[int]],
    hole_frags: list[list[bool]],
    sep_id: int,
) -> tuple[list[int], list[bool]]:
    ids: list[int] = []
    holes: list[bool] = []
    for i, (frag, hfrag) in enumerate(zip(frags, hole_frags)):
        if i:
            ids.append(sep_id)
            holes.append(False)
        ids.extend(frag)
        holes.extend(hfrag)
    return ids, holes


def _noise_fill(n: int, pool: Sequence[int], rng: random.Random) -> list[int]:
    return [int(rng.choice(pool)) for _ in range(n)]


def noise_frags(
    frags: list[list[int]],
    sep_id: int,
    noise_pool: Sequence[int],
    rng: random.Random,
    *,
    frac: float = 0.5,
    frag_idx: int | None = None,
) -> tuple[list[int], list[bool]]:
    """Like :func:`mask_frags`, but corrupts with i.i.d. uniform ``noise_pool`` tokens.

    Returns ``(body_ids, hole_flags)`` with one bool per body token (``True`` =
    corrupted). Fragment separators are never holes.
    """
    frags = [list(f) for f in frags]
    hole_frags = [[False] * len(f) for f in frags]
    non_empty = [i for i, frag in enumerate(frags) if frag]
    if not non_empty:
        n = sum(len(f) for f in frags) + max(len(frags) - 1, 0)
        return _noise_fill(n, noise_pool, rng), [True] * n
    if frag_idx is not None:
        idx = frag_idx if frag_idx in non_empty else rng.choice(non_empty)
        frags[idx] = _noise_fill(len(frags[idx]), noise_pool, rng)
        hole_frags[idx] = [True] * len(frags[idx])
        return _join_fragment_holes(frags, hole_frags, sep_id)
    if len(non_empty) >= 2:
        k = min(len(non_empty) - 1, max(1, round(frac * len(non_empty))))
        for idx in rng.sample(non_empty, k):
            frags[idx] = _noise_fill(len(frags[idx]), noise_pool, rng)
            hole_frags[idx] = [True] * len(frags[idx])
        return _join_fragment_holes(frags, hole_frags, sep_id)
    only = non_empty[0]
    tok = frags[only]
    k = min(len(tok) - 1, max(1, round(frac * len(tok)))) if len(tok) > 1 else 1
    masked_pos = set(rng.sample(range(len(tok)), k))
    frags[only] = [
        int(rng.choice(noise_pool)) if i in masked_pos else t
        for i, t in enumerate(tok)
    ]
    hole_frags[only] = [i in masked_pos for i in range(len(tok))]
    return _join_fragment_holes(frags, hole_frags, sep_id)


def noise_span_ids(
    ids: list[int],
    noise_pool: Sequence[int],
    rng: random.Random,
    *,
    frac: float = 0.5,
) -> tuple[list[int], list[bool]]:
    """Like :func:`mask_span_ids`, but fills the span with uniform noise tokens."""
    if not ids:
        return [], []
    n = len(ids)
    if n <= 1:
        return list(ids), [False] * n
    k = min(n - 1, max(1, round(frac * n)))
    start = rng.randrange(0, n - k + 1)
    out = list(ids)
    holes = [False] * n
    for i in range(start, start + k):
        out[i] = int(rng.choice(noise_pool))
        holes[i] = True
    return out, holes


def noise_local_frags(
    frags: list[list[int]],
    ids: list[int],
    sep_id: int,
    noise_pool: Sequence[int],
    rng: random.Random,
    *,
    frac: float = 0.5,
    mode: str = "fragment",
) -> tuple[list[int], list[bool]]:
    """I-JEPA local-view corruption with explicit hole flags (see :func:`noise_frags`)."""
    if mode == "fragment":
        return noise_frags(frags, sep_id, noise_pool, rng, frac=frac)
    if mode == "span":
        return noise_span_ids(ids, noise_pool, rng, frac=frac)
    if mode == "mixed":
        pick = "span" if rng.random() < 0.5 else "fragment"
        return noise_local_frags(
            frags, ids, sep_id, noise_pool, rng, frac=frac, mode=pick,
        )
    raise ValueError(f"mask mode must be fragment, span, or mixed, got {mode!r}")


class FragmentViewDataset(Dataset):
    """Yields fragmented-SMILES views or pretokenized ``body_ids`` from MOSES shards."""

    def __init__(
        self,
        shards: list[Path],
        *,
        split: str = "train",
        val_ratio: float = 0.005,
        test_ratio: float = 0.005,
        split_seed: int = 0,
        return_smiles: bool = False,
        tokenizer_path: str | Path | None = None,
        conformer_cache: dict | None = None,
        atom_dict=None,
        max_atoms: int = 256,
    ) -> None:
        from lattice_lab.preprocessing.molecules import (
            fragment_view_column,
            fragment_view_column_for_parquet,
            load_smiles_tokenizer,
            shards_have_body_ids,
        )

        self._return_smiles = return_smiles
        self._use_conformers = conformer_cache is not None
        self._conformer_cache = conformer_cache
        self._atom_dict = atom_dict
        self._max_atoms = int(max_atoms)
        if self._use_conformers and atom_dict is None:
            raise ValueError("atom_dict is required when conformer_cache is provided")
        self._use_body_ids = shards_have_body_ids(shards)
        self._tokenizer = None
        if not self._use_body_ids and tokenizer_path is not None:
            self._tokenizer = load_smiles_tokenizer(tokenizer_path)
        cols = ["inchikey", "view_idx"]
        if self._use_body_ids:
            cols.append("body_ids")
        else:
            cols.append(fragment_view_column_for_parquet(shards[0]))
        if return_smiles:
            cols.append("smiles")
        df = load_fragment_split_df(
            shards, split=split, val_ratio=val_ratio,
            test_ratio=test_ratio, split_seed=split_seed,
            columns=cols,
        )
        if self._use_conformers:
            # Keep only molecules with a cached conformer; all per-row lists below
            # (bodies/views/smiles/inchikeys) then derive from the same filtered df.
            before = len(df)
            df = df[df["inchikey"].astype(str).isin(self._conformer_cache)].reset_index(drop=True)
            logger.info(
                "fragment-view dataset split=%s: %d/%d molecules have a cached conformer",
                split, len(df), before,
            )
        if self._use_body_ids:
            self._bodies = [list(map(int, row)) for row in df["body_ids"].tolist()]
            self._views: list[str] = []
            logger.info(
                "fragment-view dataset split=%s: %d molecules (precomputed body_ids)",
                split, len(self._bodies),
            )
        else:
            view_col = fragment_view_column(df)
            self._views = df[view_col].astype(str).tolist()
            self._bodies = []
            logger.info("fragment-view dataset split=%s: %d molecules", split, len(self._views))
        self._smiles: list[str] = (
            df["smiles"].astype(str).tolist() if return_smiles else []
        )
        self._inchikeys: list[str] = (
            df["inchikey"].astype(str).tolist() if self._use_conformers else []
        )
        if self._use_conformers and not self._inchikeys:
            raise ValueError(
                f"no molecules with cached conformers in split={split!r} "
                f"from {len(shards)} shard(s)"
            )
        if not self._use_body_ids and not self._views:
            raise ValueError(f"no molecules in split={split!r} from {len(shards)} shard(s)")
        if self._use_body_ids and not self._bodies:
            raise ValueError(f"no molecules in split={split!r} from {len(shards)} shard(s)")

    def __len__(self) -> int:
        return len(self._bodies) if self._use_body_ids else len(self._views)

    def __getitem__(self, idx: int):
        if self._use_body_ids:
            item: str | list[int] = self._bodies[idx]
        else:
            view = self._views[idx]
            item = view
            if self._tokenizer is not None:
                item = self._tokenizer.encode(view, add_special_tokens=False)
        if self._use_conformers:
            from lattice_lab.data.conformers import featurize_conformer

            atoms, coords = self._conformer_cache[self._inchikeys[idx]]
            feat = featurize_conformer(atoms, coords, self._atom_dict, self._max_atoms)
            smi = self._smiles[idx] if self._return_smiles else None
            return item, smi, feat
        if self._return_smiles:
            return item, self._smiles[idx]
        return item


def collate_views(batch: list[str]) -> list[str]:
    return list(batch)


def collate_bodies(batch: list[list[int]]) -> list[list[int]]:
    return list(batch)


def collate_views_with_smiles(
    batch: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """Collator for ``return_smiles=True``: ``(views, smiles)``."""
    views, smiles = zip(*batch)
    return list(views), list(smiles)


def collate_bodies_with_smiles(
    batch: list[tuple[list[int], str]],
) -> tuple[list[list[int]], list[str]]:
    views, smiles = zip(*batch)
    return list(views), list(smiles)


def collate_with_conformers(batch: list[tuple]):
    """3D-enabled collate: ``(items, smiles|None, net_input_3d)``.

    Items may be fragment-view strings or ``body_ids`` (both just listed); the
    per-molecule featurized conformers are right-padded into the ``mol_src_*``
    batch dict read by :class:`PointCloudEncoder`. ``smiles`` is ``None`` when the
    dataset was built with ``return_smiles=False``.
    """
    from lattice_lab.data.conformers import collate_conformers

    items, smiles, feats = zip(*batch)
    smiles_list = None if smiles[0] is None else list(smiles)
    return list(items), smiles_list, collate_conformers(list(feats))


class FragmentViewDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        shard_dir: str | Path,
        batch_size: int = 256,
        val_ratio: float = 0.005,
        test_ratio: float = 0.005,
        split_seed: int = 0,
        run_validation: bool = True,
        num_workers: int = 0,
        return_smiles: bool = False,
        tokenizer_path: str | Path | None = None,
        conformer_cache: str | Path | None = None,
        atom_dict_path: str | Path | None = None,
        max_atoms: int = 256,
    ) -> None:
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.run_validation = run_validation
        self.num_workers = num_workers
        self.return_smiles = return_smiles
        self.tokenizer_path = tokenizer_path
        self.conformer_cache = conformer_cache
        self.atom_dict_path = atom_dict_path
        self.max_atoms = int(max_atoms)
        self._conformer_cache: dict | None = None
        self._atom_dict = None
        self._shards: list[Path] = []
        self._train: FragmentViewDataset | None = None
        self._val: FragmentViewDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return
        self._shards = sorted(self.shard_dir.glob("shard_*.parquet"))
        if not self._shards:
            raise FileNotFoundError(f"no parquet shards in {self.shard_dir}")
        if self.conformer_cache is not None:
            from lattice_lab.data.conformers import (
                DEFAULT_DICT_PATH,
                Dictionary,
                load_conformer_cache,
            )

            self._conformer_cache = load_conformer_cache(str(self.conformer_cache))
            self._atom_dict = Dictionary.load(
                str(self.atom_dict_path or DEFAULT_DICT_PATH)
            )
            logger.info("loaded %d cached conformers", len(self._conformer_cache))
        ds_kw = dict(
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            split_seed=self.split_seed,
            return_smiles=self.return_smiles,
            tokenizer_path=self.tokenizer_path,
            conformer_cache=self._conformer_cache,
            atom_dict=self._atom_dict,
            max_atoms=self.max_atoms,
        )
        self._train = FragmentViewDataset(self._shards, split="train", **ds_kw)
        if self.run_validation:
            self._val = FragmentViewDataset(self._shards, split="val", **ds_kw)

    def _pretokenized_batches(self) -> bool:
        from lattice_lab.preprocessing.molecules import shards_have_body_ids

        if self._train is not None:
            return bool(self._train._use_body_ids)
        return shards_have_body_ids(self._shards)

    @property
    def _collate(self):
        if self.conformer_cache is not None:
            # 3D-enabled batches always carry the (items, smiles|None, net_input_3d)
            # triple; item type (str vs body_ids) is handled inside the collate.
            return collate_with_conformers
        pretokenized = self._pretokenized_batches() or self.tokenizer_path is not None
        if pretokenized and self.return_smiles:
            return collate_bodies_with_smiles
        if pretokenized:
            return collate_bodies
        if self.return_smiles:
            return collate_views_with_smiles
        return collate_views

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True, collate_fn=self._collate,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._val is None:
            return None
        return DataLoader(
            self._val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=True, collate_fn=self._collate,
            persistent_workers=self.num_workers > 0,
        )
