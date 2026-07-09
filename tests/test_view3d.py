"""Tests for the 3D point-cloud view: encoder, featurization, cache, and the
cross-modal predictor term wired into the SSL module.

Conformers are fabricated (atom symbols + random coords) so nothing here needs
RDKit; ``generate_conformer`` itself is exercised in an RDKit-gated test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lattice_lab.backbone.pointcloud import PointCloudEncoder, build_pointcloud_encoder
from lattice_lab.data.conformers import (
    DEFAULT_DICT_PATH,
    Dictionary,
    collate_conformers,
    featurize_conformer,
    load_conformer_cache,
)

TOKENIZER = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer" / "smiles_new.json"
requires_tokenizer = pytest.mark.skipif(
    not TOKENIZER.is_file(), reason=f"tokenizer not found at {TOKENIZER}"
)


def _fake_conformer(rng, n: int) -> tuple[np.ndarray, np.ndarray]:
    atoms = rng.choice(np.array(["C", "N", "O", "S"]), size=n)
    coords = rng.standard_normal((n, 3)).astype(np.float32)
    return atoms, coords


def _feats(dictionary, sizes, seed=0):
    rng = np.random.default_rng(seed)
    return [featurize_conformer(*_fake_conformer(rng, n), dictionary, 64) for n in sizes]


def test_dictionary_pad_is_zero_and_specials_present() -> None:
    d = Dictionary.load(DEFAULT_DICT_PATH)
    assert d.pad() == 0  # matches PointCloudEncoder.padding_idx
    assert d.index("C") > 3 and d.index("Xx") == d.index("[UNK]")
    assert len(d) == 31  # 4 specials + 26 atoms + [MASK]


def test_featurize_and_collate_shapes() -> None:
    d = Dictionary.load(DEFAULT_DICT_PATH)
    feats = _feats(d, [3, 6])
    # L = n_atoms + 2 (CLS/SEP); distance/edge_type are LxL.
    assert feats[0]["tokens"].shape == (5,)
    assert feats[0]["distance"].shape == (5, 5)
    assert feats[0]["edge_type"].shape == (5, 5)
    net = collate_conformers(feats)
    assert net["mol_src_tokens"].shape == (2, 8)  # padded to max L=8
    assert net["mol_src_distance"].shape == (2, 8, 8)
    assert net["mol_src_edge_type"].shape == (2, 8, 8)
    # padding rows are pad-id 0 in tokens
    assert (net["mol_src_tokens"][0, 5:] == 0).all()


def test_pointcloud_encoder_shape_and_pad_invariance() -> None:
    d = Dictionary.load(DEFAULT_DICT_PATH)
    enc = PointCloudEncoder(
        vocab_size=len(d), encoder_layers=2, encoder_embed_dim=32,
        encoder_ffn_embed_dim=64, encoder_attention_heads=4,
    ).eval()
    feats = _feats(d, [3, 6, 5])
    with torch.no_grad():
        z = enc(collate_conformers(feats))
        z_solo = enc(collate_conformers([feats[0]]))
    assert z.shape == (3, 32)
    assert torch.isfinite(z).all()
    # The CLS output for molecule 0 must not depend on the batch's padding.
    assert torch.allclose(z[0], z_solo[0], atol=1e-5)


def test_build_pointcloud_encoder_sizes_from_dict() -> None:
    enc = build_pointcloud_encoder(
        dict_path=DEFAULT_DICT_PATH, encoder_layers=2, encoder_embed_dim=32,
        encoder_ffn_embed_dim=64, encoder_attention_heads=4,
    )
    assert enc.output_dim == 32
    assert enc.build_config["vocab_size"] == 31
    assert enc.embed_tokens.num_embeddings == 31


def test_conformer_cache_roundtrip(tmp_path: Path) -> None:
    import pandas as pd

    rng = np.random.default_rng(1)
    rows = []
    for i in range(3):
        atoms, coords = _fake_conformer(rng, 4 + i)
        rows.append({
            "inchikey": f"IK{i}",
            "atoms": " ".join(atoms.tolist()),
            "coords": coords.reshape(-1).astype(np.float32).tolist(),
        })
    p = tmp_path / "conformers.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    cache = load_conformer_cache(str(p))
    assert set(cache) == {"IK0", "IK1", "IK2"}
    a, c = cache["IK2"]
    assert len(a) == 6 and c.shape == (6, 3) and c.dtype == np.float32


@requires_tokenizer
def test_view3d_grads_reach_encoder_3d_and_pred_3d() -> None:
    """A fast_dev_run with view3d enabled trains both the 3D co-encoder and the
    cross-modal predictor — the key check that configure_optimizers wires them in."""
    import lightning as L
    from torch.utils.data import DataLoader

    from lattice_lab.backbone.discrete_flow import build_discrete_flow_encoder
    from lattice_lab.data.fragment_views import collate_with_conformers
    from lattice_lab.models.discrete_flow_ssl import DiscreteFlowSSLModule

    enc = build_discrete_flow_encoder(
        ckpt_path=None, tokenizer_path=str(TOKENIZER),
        backbone_layer_start=0, backbone_layer_end=1, d_adapter=32, adapter_n_layers=1,
        encode_time=1.0, learnable_time=False, freeze_backbone=False,
        n_layer=2, n_head=4, n_embd=32, device="cpu",
    )
    enc3d = build_pointcloud_encoder(
        dict_path=DEFAULT_DICT_PATH, encoder_layers=2, encoder_embed_dim=32,
        encoder_ffn_embed_dim=64, encoder_attention_heads=4,
    )
    module = DiscreteFlowSSLModule(
        encoder=enc, ssl_loss="lejepa", encoder_3d=enc3d,
        view3d_weight=1.0, lejepa_lambda=0.1,
        ijepa_use_visreg=True, ijepa_visreg_num_projections=32,
        warmup_steps=1, val_probe_n_molecules=0, train_rank_every_n_steps=0,
    )
    d = Dictionary.load(DEFAULT_DICT_PATH)
    feats = _feats(d, [3, 4, 5, 6, 3, 4, 5, 6])
    smis = ["CCO", "c1ccccc1", "CCN", "CCCC", "OCC", "CCl", "CCOCC", "CC(=O)O"]
    samples = [(s, s, f) for s, f in zip(smis, feats)]
    loader = DataLoader(samples, batch_size=4, collate_fn=collate_with_conformers)

    tracked = {
        n: p.detach().clone()
        for n, p in module.named_parameters()
        if p.requires_grad and (n.startswith("encoder_3d.") or n.startswith("pred_3d."))
    }
    assert any(n.startswith("encoder_3d.") for n in tracked)
    assert any(n.startswith("pred_3d.") for n in tracked)

    trainer = L.Trainer(
        fast_dev_run=2, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=loader)

    current = dict(module.named_parameters())
    assert all(not torch.equal(tracked[n], current[n]) for n in tracked)


@requires_tokenizer
def test_ntxent_crossmodal_lejepa_with_visreg_1d() -> None:
    """ntxent (within-1D) + cross-modal LeJEPA (1D->3D, symmetric) with VISReg on
    *both* modalities: visreg_2d is built and training moves the 1D encoder, 3D
    co-encoder, and cross-modal predictor."""
    import lightning as L
    from torch.utils.data import DataLoader

    from lattice_lab.backbone.discrete_flow import build_discrete_flow_encoder
    from lattice_lab.data.fragment_views import collate_with_conformers
    from lattice_lab.models.discrete_flow_ssl import DiscreteFlowSSLModule

    enc = build_discrete_flow_encoder(
        ckpt_path=None, tokenizer_path=str(TOKENIZER),
        backbone_layer_start=0, backbone_layer_end=1, d_adapter=32, adapter_n_layers=1,
        encode_time=1.0, learnable_time=False, freeze_backbone=False,
        adapter_pool="attn", adapter_dual_pool=True,
        n_layer=2, n_head=4, n_embd=32, device="cpu",
    )
    enc3d = build_pointcloud_encoder(
        dict_path=DEFAULT_DICT_PATH, encoder_layers=2, encoder_embed_dim=32,
        encoder_ffn_embed_dim=64, encoder_attention_heads=4,
    )
    module = DiscreteFlowSSLModule(
        encoder=enc, ssl_loss="ntxent", encoder_3d=enc3d,
        view3d_weight=1.0, view3d_visreg_1d=True, lejepa_lambda=0.1,
        ijepa_visreg_num_projections=32,
        warmup_steps=1, val_probe_n_molecules=0, train_rank_every_n_steps=0,
    )
    assert module.visreg_2d is not None and module.visreg_3d is not None
    assert enc.adapter.dual_attn_pool

    d = Dictionary.load(DEFAULT_DICT_PATH)
    feats = _feats(d, [3, 4, 5, 6, 3, 4, 5, 6])
    smis = ["CCO", "c1ccccc1", "CCN", "CCCC", "OCC", "CCl", "CCOCC", "CC(=O)O"]
    samples = [(s, s, f) for s, f in zip(smis, feats)]
    loader = DataLoader(samples, batch_size=4, collate_fn=collate_with_conformers)

    tracked = {
        n: p.detach().clone()
        for n, p in module.named_parameters()
        if p.requires_grad
        and (n.startswith("encoder_3d.") or n.startswith("pred_3d.") or n.startswith("encoder.adapter."))
    }
    assert any(n.startswith("pred_3d.") for n in tracked)
    assert any(n.startswith("encoder.adapter.") for n in tracked)

    trainer = L.Trainer(
        fast_dev_run=2, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=loader)

    current = dict(module.named_parameters())
    assert any(not torch.equal(tracked[n], current[n]) for n in tracked)


@requires_tokenizer
def test_siglip_mode_trains_with_crossmodal_view3d() -> None:
    """ssl_loss=siglip reuses the ntxent two-view + cross-modal path; its learnable
    temperature/bias are optimized alongside the 3D co-encoder and predictor."""
    import lightning as L
    from torch.utils.data import DataLoader

    from lattice_lab.backbone.discrete_flow import build_discrete_flow_encoder
    from lattice_lab.data.fragment_views import collate_with_conformers
    from lattice_lab.models.discrete_flow_ssl import DiscreteFlowSSLModule

    enc = build_discrete_flow_encoder(
        ckpt_path=None, tokenizer_path=str(TOKENIZER),
        backbone_layer_start=0, backbone_layer_end=1, d_adapter=32, adapter_n_layers=1,
        encode_time=1.0, learnable_time=False, freeze_backbone=False,
        n_layer=2, n_head=4, n_embd=32, device="cpu",
    )
    enc3d = build_pointcloud_encoder(
        dict_path=DEFAULT_DICT_PATH, encoder_layers=2, encoder_embed_dim=32,
        encoder_ffn_embed_dim=64, encoder_attention_heads=4,
    )
    module = DiscreteFlowSSLModule(
        encoder=enc, ssl_loss="siglip", encoder_3d=enc3d,
        view3d_weight=1.0, view3d_visreg_1d=True, lejepa_lambda=0.1,
        ijepa_visreg_num_projections=32,
        warmup_steps=1, val_probe_n_molecules=0, train_rank_every_n_steps=0,
    )
    assert module.siglip_loss_fn is not None

    d = Dictionary.load(DEFAULT_DICT_PATH)
    feats = _feats(d, [3, 4, 5, 6, 3, 4, 5, 6])
    smis = ["CCO", "c1ccccc1", "CCN", "CCCC", "OCC", "CCl", "CCOCC", "CC(=O)O"]
    samples = [(s, s, f) for s, f in zip(smis, feats)]
    loader = DataLoader(samples, batch_size=4, collate_fn=collate_with_conformers)

    before = module.siglip_loss_fn.logit_scale.detach().clone()
    trainer = L.Trainer(
        fast_dev_run=2, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=loader)
    assert not torch.equal(before, module.siglip_loss_fn.logit_scale.detach())


@requires_tokenizer
def test_dual_pool_routes_lejepa_to_main_and_siglip_to_proj() -> None:
    """With dual_attn_pool + siglip, the cross-modal LeJEPA prediction reads only the
    main (regression) pool half and SigLIP only the projection pool half — so the
    predictor's gradient reaches pool_query but not proj_pool_query, and vice versa."""
    from torch.utils.data import DataLoader

    from lattice_lab.backbone.discrete_flow import build_discrete_flow_encoder
    from lattice_lab.data.fragment_views import collate_with_conformers
    from lattice_lab.models.discrete_flow_ssl import DiscreteFlowSSLModule

    enc = build_discrete_flow_encoder(
        ckpt_path=None, tokenizer_path=str(TOKENIZER),
        backbone_layer_start=0, backbone_layer_end=1, d_adapter=32, adapter_n_layers=1,
        encode_time=1.0, learnable_time=False, freeze_backbone=False,
        adapter_pool="attn", adapter_dual_pool=True,
        n_layer=2, n_head=4, n_embd=32, device="cpu",
    )
    enc3d = build_pointcloud_encoder(
        dict_path=DEFAULT_DICT_PATH, encoder_layers=2, encoder_embed_dim=32,
        encoder_ffn_embed_dim=64, encoder_attention_heads=4,
    )
    module = DiscreteFlowSSLModule(
        encoder=enc, ssl_loss="siglip", encoder_3d=enc3d,
        view3d_weight=1.0, view3d_visreg_1d=True, lejepa_lambda=0.1,
        ijepa_visreg_num_projections=32,
        warmup_steps=1, val_probe_n_molecules=0, train_rank_every_n_steps=0,
    )
    # Predictor reads the half-width main pool, not the full concat.
    assert module._view3d_main_half is True
    assert module.pred_3d.net[0].in_features == enc.adapter.d_pool == 16

    d = Dictionary.load(DEFAULT_DICT_PATH)
    feats = _feats(d, [3, 4, 5, 6])
    smis = ["CCO", "c1ccccc1", "CCN", "CCCC"]
    samples = [(s, s, f) for s, f in zip(smis, feats)]
    loader = DataLoader(samples, batch_size=4, collate_fn=collate_with_conformers)
    batch = next(iter(loader))
    views, smiles, net3d = module._split_batch(batch)
    ad = module.encoder.adapter

    def _no_signal(p) -> bool:  # None, or reachable-but-sliced-away (all-zero grad)
        return p.grad is None or float(p.grad.abs().sum()) == 0.0

    def _has_signal(p) -> bool:
        return p.grad is not None and float(p.grad.abs().sum()) > 0.0

    # Cross-modal LeJEPA term (+1D VISReg) alone -> gradient reaches the main pool only.
    module.zero_grad(set_to_none=True)
    _, _, z_a_pooled = module._encode_ntxent_views(views)
    l_3d, _, l_reg_1d, acc = module._view3d_loss(z_a_pooled, net3d)
    assert 0.0 <= acc <= 1.0  # cross-modal top-1 retrieval acc@1 is a fraction
    (l_3d + l_reg_1d).backward()
    assert _has_signal(ad.pool_query)
    assert _no_signal(ad.proj_pool_query)

    # SigLIP contrastive term alone -> gradient reaches the projection pool only.
    module.zero_grad(set_to_none=True)
    z_a, z_b, _ = module._encode_ntxent_views(views)
    loss_c, _ = module._compute_loss(z_a=z_a, z_b=z_b)
    loss_c.backward()
    assert _has_signal(ad.proj_pool_query)
    assert _no_signal(ad.pool_query)


@requires_tokenizer
def test_view3d_disabled_builds_nothing() -> None:
    """view3d_weight=0 (or no encoder_3d) leaves the 3D submodules unbuilt so
    plain 2D runs are unaffected and checkpoints keep their exact keys."""
    from lattice_lab.backbone.discrete_flow import build_discrete_flow_encoder
    from lattice_lab.models.discrete_flow_ssl import DiscreteFlowSSLModule

    enc = build_discrete_flow_encoder(
        ckpt_path=None, tokenizer_path=str(TOKENIZER),
        backbone_layer_start=0, backbone_layer_end=1, d_adapter=32, adapter_n_layers=1,
        encode_time=1.0, learnable_time=False, freeze_backbone=False,
        n_layer=2, n_head=4, n_embd=32, device="cpu",
    )
    m = DiscreteFlowSSLModule(encoder=enc, ssl_loss="lejepa", val_probe_n_molecules=0)
    assert m.encoder_3d is None and m.pred_3d is None and m.visreg_3d is None
    assert not m._view3d_enabled()


def test_generate_conformer_smoke() -> None:
    pytest.importorskip("rdkit")
    from lattice_lab.data.conformers import generate_conformer, remove_hydrogens

    ac = generate_conformer("CCO")
    assert ac is not None
    atoms, coords = remove_hydrogens(*ac)
    assert len(atoms) == 3 and coords.shape == (3, 3)
    assert generate_conformer("not a smiles") is None
