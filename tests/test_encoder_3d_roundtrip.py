"""Self-check for the Uni-Mol 3D-encoder Stage-4/6 path.

Covers the load-bearing new logic:
1. ``load_encoder_3d_from_ckpt`` rebuilds the exact ``encoder_3d`` baked into a
   VIEW3D-style Lightning ckpt and it encodes a conformer batch to ``[B, dim]``.
2. ``load_conformer_cache(key_col=...)`` round-trips a SMILES-keyed cache (the
   Stage-5 binder store keying).

Run: ``python tests/test_encoder_3d_roundtrip.py`` (asserts, no framework).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lattice_lab.backbone.pointcloud import PointCloudEncoder
from lattice_lab.data.conformers import (
    Dictionary,
    collate_conformers,
    featurize_conformer,
    load_conformer_cache,
)
from lattice_lab.models.builders import load_encoder_3d_from_ckpt


def _tiny_dict(path: Path) -> Dictionary:
    path.write_text("C\nN\nO\n")
    return Dictionary.load(str(path))


def test_encoder_3d_roundtrip_and_encode() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dict_path = td / "dict_mol.txt"
        dictionary = _tiny_dict(dict_path)
        vocab = len(dictionary)

        cfg = dict(
            dict_path=str(dict_path), vocab_size=vocab, key_prefix="mol",
            encoder_layers=1, encoder_embed_dim=16, encoder_ffn_embed_dim=32,
            encoder_attention_heads=4, max_seq_len=64,
        )
        enc = PointCloudEncoder(
            vocab_size=vocab, key_prefix=cfg["key_prefix"],
            encoder_layers=cfg["encoder_layers"], encoder_embed_dim=cfg["encoder_embed_dim"],
            encoder_ffn_embed_dim=cfg["encoder_ffn_embed_dim"],
            encoder_attention_heads=cfg["encoder_attention_heads"], max_seq_len=cfg["max_seq_len"],
        ).eval()

        ckpt = td / "adapter3d_last.ckpt"
        state = {f"encoder_3d.{k}": v for k, v in enc.state_dict().items()}
        state["encoder.dummy"] = torch.zeros(1)  # 2D encoder also lives in real ckpts
        torch.save({"state_dict": state, "encoder_3d_config": cfg}, ckpt)

        loaded = load_encoder_3d_from_ckpt(ckpt, device="cpu")

        # Weights must match exactly (strict reload of the exact skeleton).
        for k, v in enc.state_dict().items():
            assert torch.equal(loaded.state_dict()[k], v), f"weight mismatch: {k}"

        atoms = np.array(["C", "O", "N"], dtype=object)
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.2, 0.0]], dtype=np.float32)
        feats = [featurize_conformer(atoms, coords, dictionary, 64) for _ in range(3)]
        batch = collate_conformers(feats, key_prefix=loaded.key_prefix)
        with torch.no_grad():
            z = loaded(batch)
        assert z.shape == (3, cfg["encoder_embed_dim"]), z.shape
        assert torch.isfinite(z).all()


def test_conformer_cache_key_col() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "binder_conformers.parquet"
        pd.DataFrame(
            {
                "smiles": ["CCO", "c1ccccc1"],
                "atoms": ["C C O", "C C C C C C"],
                "coords": [
                    np.arange(9, dtype=np.float32).tolist(),
                    np.arange(18, dtype=np.float32).tolist(),
                ],
            }
        ).to_parquet(p, index=False)

        cache = load_conformer_cache(str(p), key_col="smiles")
        assert set(cache) == {"CCO", "c1ccccc1"}
        atoms, coords = cache["CCO"]
        assert list(atoms) == ["C", "C", "O"]
        assert coords.shape == (3, 3)


if __name__ == "__main__":
    test_encoder_3d_roundtrip_and_encode()
    test_conformer_cache_key_col()
    print("ok")
