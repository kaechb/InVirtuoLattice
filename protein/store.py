"""Memory-mapped store for per-protein ESM-2 embeddings.

The downstream loader (Stage 4/5) needs random ``pid → z_p`` lookup over tens
of thousands of proteins without holding the whole array in RAM. The format:

::

    store_dir/
        manifest.json          # model name, dim, dtype, count, per-residue flag
        pids.tsv               # row_idx \t pid           (1 line per row, idempotent index)
        mean.dat               # raw numpy memmap, shape (N, D), dtype from manifest
        per_residue/<pid>.npy  # optional, per protein (variable length)

Why ``.dat`` + ``pids.tsv`` instead of ``.npy``? ``.npy`` writes the full shape
in its header, which makes incremental append a pain (you'd rewrite the whole
file). We instead grow ``mean.dat`` with ``np.memmap(..., mode='r+')`` and
record the on-disk row layout in a separate text file. The ``EmbeddingStore``
abstracts both directions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StoreManifest:
    """Serializable description of a store on disk.

    Kept simple on purpose: this is the source of truth for shape/dtype, so
    readers don't have to introspect the ``.dat`` file. Stored as JSON in
    ``manifest.json``.
    """

    model_name: str
    embedding_dim: int
    dtype: str
    count: int
    per_residue: bool = False
    extra: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, object]) -> "StoreManifest":
        return StoreManifest(
            model_name=str(d["model_name"]),
            embedding_dim=int(d["embedding_dim"]),
            dtype=str(d["dtype"]),
            count=int(d["count"]),
            per_residue=bool(d.get("per_residue", False)),
            extra={k: str(v) for k, v in dict(d.get("extra", {})).items()},
        )


class EmbeddingStore:
    """Reader/writer for the ``mean.dat`` + ``pids.tsv`` + ``manifest.json`` layout.

    Typical use::

        store = EmbeddingStore.create(dir, dim=1280, dtype="float32", model_name="esm2-650M")
        store.append_mean(pids=["p1", "p2"], arr=np.random.randn(2, 1280).astype("float32"))
        store.save()

    or for reading::

        store = EmbeddingStore.open(dir, mode="r")
        z = store.get_mean("p1")            # → (1280,)
        all_z = store.mean_array            # full memmap, shape (N, D)

    The store is idempotent: ``append_mean`` skips pids that already exist.
    Use ``overwrite=True`` to replace an existing row.
    """

    MANIFEST = "manifest.json"
    PIDS = "pids.tsv"
    MEAN = "mean.dat"
    PERRES_DIR = "per_residue"

    def __init__(
        self,
        path: Path,
        manifest: StoreManifest,
        pid_to_row: dict[str, int],
        mode: str = "r+",
    ) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self.pid_to_row: dict[str, int] = dict(pid_to_row)
        self.mode = mode
        self._mean_memmap: np.memmap | None = None
        self._mean_ram: np.ndarray | None = None

    # ----- construction --------------------------------------------------

    @classmethod
    def create(
        cls,
        path: Path | str,
        *,
        embedding_dim: int,
        model_name: str,
        dtype: str = "float32",
        per_residue: bool = False,
        extra: dict[str, str] | None = None,
        exist_ok: bool = True,
    ) -> "EmbeddingStore":
        """Create a new store. If one already exists, returns it (when ``exist_ok``)."""
        path = Path(path)
        if (path / cls.MANIFEST).exists():
            if not exist_ok:
                raise FileExistsError(f"store already exists at {path}")
            return cls.open(path, mode="r+")
        path.mkdir(parents=True, exist_ok=True)
        if per_residue:
            (path / cls.PERRES_DIR).mkdir(exist_ok=True)
        manifest = StoreManifest(
            model_name=model_name,
            embedding_dim=embedding_dim,
            dtype=dtype,
            count=0,
            per_residue=per_residue,
            extra=dict(extra or {}),
        )
        # Touch empty files so mmap can later grow them.
        (path / cls.MEAN).touch()
        (path / cls.PIDS).touch()
        store = cls(path=path, manifest=manifest, pid_to_row={}, mode="r+")
        store.save()
        return store

    @classmethod
    def open(
        cls,
        path: Path | str,
        *,
        mode: str = "r",
        load_to_ram: bool = False,
    ) -> "EmbeddingStore":
        """Open an existing store. ``mode='r'`` is read-only memmap; ``'r+'`` allows append.

        ``load_to_ram=True`` copies ``mean.dat`` into RAM once (read-only). Use on
        Lustre when random row access into a memmap is slow.
        """
        path = Path(path)
        manifest_path = path / cls.MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}")
        with open(manifest_path) as fh:
            manifest = StoreManifest.from_dict(json.load(fh))
        pid_to_row: dict[str, int] = {}
        pids_path = path / cls.PIDS
        if pids_path.exists():
            with open(pids_path) as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 2:
                        continue
                    row_idx, pid = parts
                    pid_to_row[pid] = int(row_idx)
        store = cls(path=path, manifest=manifest, pid_to_row=pid_to_row, mode=mode)
        if load_to_ram and mode == "r" and manifest.count > 0:
            store._pin_mean_to_ram()
        return store

    def _pin_mean_to_ram(self) -> None:
        """Copy ``mean.dat`` into a float32 RAM array for fast random access."""
        if self.mode != "r":
            raise ValueError("load_to_ram is only supported for read-only stores")
        arr = np.asarray(self.mean_array, dtype=np.float32)
        self._mean_ram = arr
        nbytes = arr.nbytes
        logger.info(
            "pinned %d rows x %d to RAM (%.1f MB) from %s",
            self.manifest.count,
            self.manifest.embedding_dim,
            nbytes / (1024 ** 2),
            self.path,
        )

    # ----- mean embeddings -----------------------------------------------

    @property
    def mean_array(self) -> np.ndarray | np.memmap:
        """Lazy memmap covering the current ``mean.dat`` file.

        Reopened on every access if the count changed (i.e., after an append),
        so callers should reread this property after ``append_mean``.
        When ``load_to_ram`` was used at open, returns the in-RAM copy instead.
        """
        if self._mean_ram is not None:
            return self._mean_ram
        if self._mean_memmap is None or self._mean_memmap.shape[0] != self.manifest.count:
            shape = (self.manifest.count, self.manifest.embedding_dim)
            mode = "r" if self.mode == "r" else "r+"
            self._mean_memmap = np.memmap(
                self.path / self.MEAN,
                dtype=self.manifest.dtype,
                mode=mode if self.manifest.count > 0 else "r+",
                shape=shape if self.manifest.count > 0 else (0, self.manifest.embedding_dim),
            )
        return self._mean_memmap

    def contains(self, pid: str) -> bool:
        return pid in self.pid_to_row

    def get_mean(self, pid: str) -> np.ndarray:
        """Return a copy of the mean embedding for ``pid``."""
        if pid not in self.pid_to_row:
            raise KeyError(pid)
        return np.array(self.mean_array[self.pid_to_row[pid]], copy=True)

    def append_mean(
        self,
        pids: Sequence[str],
        arr: np.ndarray,
        *,
        overwrite: bool = False,
    ) -> int:
        """Append rows for pids not already in the store. Returns the number written.

        ``arr`` shape is ``(len(pids), D)``; rows for already-present pids are
        skipped unless ``overwrite=True`` (in which case their existing slot is
        overwritten in-place).
        """
        if self.mode == "r":
            raise PermissionError("store opened read-only")
        if arr.ndim != 2 or arr.shape[1] != self.manifest.embedding_dim:
            raise ValueError(
                f"arr shape {arr.shape} does not match dim {self.manifest.embedding_dim}"
            )
        if len(pids) != arr.shape[0]:
            raise ValueError(f"got {len(pids)} pids but {arr.shape[0]} rows")
        arr = np.ascontiguousarray(arr, dtype=self.manifest.dtype)

        new_pids: list[str] = []
        new_rows: list[np.ndarray] = []
        overwrites: list[tuple[int, np.ndarray]] = []
        for pid, row in zip(pids, arr):
            if pid in self.pid_to_row:
                if overwrite:
                    overwrites.append((self.pid_to_row[pid], row))
                continue
            new_pids.append(pid)
            new_rows.append(row)

        # Handle overwrites first (need an r+ memmap of current size).
        if overwrites:
            current = self.mean_array  # r+ memmap of current count
            for idx, row in overwrites:
                current[idx] = row
            current.flush()

        if not new_pids:
            return 0

        # Append: grow the .dat file by len(new_rows) * D * itemsize bytes.
        old_count = self.manifest.count
        new_count = old_count + len(new_pids)
        itemsize = np.dtype(self.manifest.dtype).itemsize
        nbytes = new_count * self.manifest.embedding_dim * itemsize
        # Resize the underlying file. Closing any open memmap first is required
        # on most platforms before truncate.
        self._mean_memmap = None
        with open(self.path / self.MEAN, "r+b") as fh:
            fh.truncate(nbytes)
        new_memmap = np.memmap(
            self.path / self.MEAN,
            dtype=self.manifest.dtype,
            mode="r+",
            shape=(new_count, self.manifest.embedding_dim),
        )
        new_memmap[old_count:new_count] = np.stack(new_rows, axis=0)
        new_memmap.flush()
        self._mean_memmap = new_memmap

        # Update index + manifest.
        with open(self.path / self.PIDS, "a") as fh:
            for offset, pid in enumerate(new_pids):
                fh.write(f"{old_count + offset}\t{pid}\n")
                self.pid_to_row[pid] = old_count + offset
        self.manifest.count = new_count
        self.save_manifest()
        return len(new_pids)

    # ----- per-residue (optional) ----------------------------------------

    def append_per_residue(self, pid: str, arr: np.ndarray, *, overwrite: bool = False) -> bool:
        """Persist a ``(L, D)`` per-residue array for one pid.

        Returns ``True`` on write, ``False`` if skipped because the file
        existed and ``overwrite=False``.
        """
        if self.mode == "r":
            raise PermissionError("store opened read-only")
        if not self.manifest.per_residue:
            raise RuntimeError("store was not created with per_residue=True")
        if arr.ndim != 2 or arr.shape[1] != self.manifest.embedding_dim:
            raise ValueError(
                f"per-residue array shape {arr.shape} does not match "
                f"dim {self.manifest.embedding_dim}"
            )
        target = self.path / self.PERRES_DIR / f"{pid}.npy"
        target.parent.mkdir(exist_ok=True)
        if target.exists() and not overwrite:
            return False
        np.save(target, arr.astype(self.manifest.dtype, copy=False))
        return True

    def get_per_residue(self, pid: str) -> np.ndarray:
        target = self.path / self.PERRES_DIR / f"{pid}.npy"
        if not target.exists():
            raise KeyError(pid)
        return np.load(target)

    # ----- persistence ---------------------------------------------------

    def save_manifest(self) -> None:
        with open(self.path / self.MANIFEST, "w") as fh:
            json.dump(self.manifest.as_dict(), fh, indent=2, sort_keys=True)

    def save(self) -> None:
        """Flush the memmap and rewrite the manifest. Safe to call repeatedly."""
        if self._mean_memmap is not None:
            self._mean_memmap.flush()
        self.save_manifest()

    # ----- diagnostics ---------------------------------------------------

    def __len__(self) -> int:
        return self.manifest.count

    def pids(self) -> list[str]:
        return sorted(self.pid_to_row, key=self.pid_to_row.get)  # type: ignore[arg-type]

    def checksum(self, n_rows: int = 32) -> str:
        """Cheap content checksum over the first ``n_rows`` rows.

        Not cryptographic; intended for "did the store change since last
        write?" sanity checks across reloads.
        """
        if self.manifest.count == 0:
            return "empty"
        h = hashlib.sha1()
        arr = self.mean_array[: min(n_rows, self.manifest.count)]
        h.update(arr.tobytes())
        return h.hexdigest()[:16]


def iter_missing_pids(store: EmbeddingStore, pids: Iterable[str]) -> list[str]:
    """Return the subset of ``pids`` that are not yet in ``store`` (preserves order)."""
    return [p for p in pids if p not in store.pid_to_row]
