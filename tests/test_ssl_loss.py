"""Unit tests for SSL losses (NT-Xent + LeJEPA/SIGReg)."""

from __future__ import annotations

import numpy as np
import torch

from lattice_lab.training.ssl_loss import (
    IJEPALoss,
    LeJEPALoss,
    NTXentLoss,
    SIGReg,
    VICReg,
    lejepa_retrieval_acc1,
    top1_paired_accuracy,
)
from lattice_lab.models.discrete_flow_ssl import DiscreteFlowSSLModule
from lattice_lab.training.ssl_val_probes import embedding_covariance_rank


def _ijepa_batch(*, b: int = 2, t: int = 4, d: int = 32):
    tok = torch.randn(b, t, d)
    hole = torch.tensor(
        [
            [False, True, True, False],
            [True, False, True, False],
        ]
    )
    valid = torch.ones(b, t, dtype=torch.bool)
    target = torch.randn(int(hole.sum()), d)
    z_pooled = torch.randn(b, d)  # mean-pooled intact z_m batch (anti-collapse reg input)
    return tok, hole, valid, target, z_pooled


def test_sigreg_forward_shape() -> None:
    proj = torch.randn(2, 8, 16)
    loss = SIGReg(num_projections=32, knots=9)(proj)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_lejepa_forward_finite_terms() -> None:
    loss_fn = LeJEPALoss(lejepa_lambda=0.05, sigreg_num_projections=16, sigreg_knots=9)
    z_global = torch.randn(8, 2, 64)
    z_all = torch.cat([z_global, torch.randn(8, 1, 64)], dim=1)
    terms = loss_fn(z_global, z_all)
    for t in (terms.total, terms.inv, terms.sigreg, terms.inv_rel):
        assert t.ndim == 0 and torch.isfinite(t)
    assert terms.inv_rel.item() >= 0.0


def test_lejepa_inv_zero_when_all_views_equal_center() -> None:
    # Every view identical to the molecule's center => invariance MSE is 0.
    center = torch.randn(8, 64)
    z_global = center.unsqueeze(1).expand(8, 2, 64)
    z_all = center.unsqueeze(1).expand(8, 4, 64)
    terms = LeJEPALoss(lejepa_lambda=0.05, sigreg_num_projections=16, sigreg_knots=9)(
        z_global, z_all
    )
    assert terms.inv.item() == 0.0
    assert terms.inv_rel.item() == 0.0


def test_lejepa_inv_rel_low_when_views_cluster_high_when_random() -> None:
    # inv_rel = within-molecule view spread / between-molecule center spread.
    # Tight views around distinct centers => «1 (discriminative); fully random
    # views => >1 (no better than the batch, non-discriminative).
    torch.manual_seed(0)
    loss_fn = LeJEPALoss(lejepa_lambda=0.0, sigreg_num_projections=8, sigreg_knots=9)
    centers = torch.randn(256, 1, 64)
    tight_global = centers + torch.randn(256, 2, 64) * 0.01
    tight_all = torch.cat([tight_global, centers + torch.randn(256, 2, 64) * 0.01], dim=1)
    assert loss_fn(tight_global, tight_all).inv_rel.item() < 0.1

    rand_global = torch.randn(256, 2, 64)
    rand_all = torch.cat([rand_global, torch.randn(256, 2, 64)], dim=1)
    assert loss_fn(rand_global, rand_all).inv_rel.item() > 1.0


def test_lejepa_gradients_flow_to_inputs() -> None:
    loss_fn = LeJEPALoss(lejepa_lambda=0.05, sigreg_num_projections=32, sigreg_knots=9)
    z_global = torch.randn(8, 2, 128, requires_grad=True)
    z_all = z_global  # views == globals; still differentiable through inv+sigreg
    loss_fn(z_global, z_all).total.backward()
    assert z_global.grad is not None and z_global.grad.norm().item() > 1e-4


def test_lejepa_has_no_predictor() -> None:
    # Predictor-free by design: alignment is direct, no trainable loss params.
    loss_fn = LeJEPALoss(lejepa_lambda=0.5, sigreg_num_projections=8, sigreg_knots=9)
    assert not hasattr(loss_fn, "predictor")
    assert list(loss_fn.parameters()) == []


def test_ijepa_forward_finite_terms() -> None:
    loss_fn = IJEPALoss(dim=32, lejepa_lambda=0.05, sigreg_num_projections=16, sigreg_knots=9)
    tok, hole, valid, target, z_pooled = _ijepa_batch(d=32)
    terms = loss_fn(tok, hole, target, z_pooled, valid=valid)
    for t in (terms.total, terms.predict, terms.sigreg):
        assert t.ndim == 0 and torch.isfinite(t)


def test_ijepa_total_is_predict_plus_pooled_reg() -> None:
    """total = (1-lambda)*predict + lambda*reg(z_pooled), with reg on the pooled batch."""
    tok, hole, valid, target, z_pooled = _ijepa_batch(d=32)
    loss_fn = IJEPALoss(dim=32, lejepa_lambda=0.3, sigreg_num_projections=8, sigreg_knots=9)
    terms = loss_fn(tok, hole, target, z_pooled, valid=valid)
    expected = 0.7 * terms.predict + 0.3 * terms.sigreg
    assert abs(float(terms.total - expected)) < 1e-5


def test_vicreg_penalizes_collapse() -> None:
    """VICReg's variance hinge fires on collapsed rows, near-zero on spread rows."""
    torch.manual_seed(0)
    reg = VICReg(gamma=1.0, cov_coeff=1.0)
    collapsed = torch.zeros(64, 32) + 0.01 * torch.randn(1, 32)
    spread = torch.randn(64, 32)
    assert float(reg(collapsed)) > float(reg(spread))
    assert float(reg(spread)) < 0.5


def test_ijepa_use_vicreg_swaps_regularizer() -> None:
    """use_vicreg routes the pooled reg through VICReg, not SIGReg; grads still flow."""
    tok, hole, valid, target, _ = _ijepa_batch(d=16)
    loss_fn = IJEPALoss(dim=16, lejepa_lambda=0.5, use_vicreg=True)
    assert loss_fn.sigreg is None and isinstance(loss_fn.vicreg, VICReg)
    z_pooled = torch.randn(2, 16, requires_grad=True)
    terms = loss_fn(tok, hole, target, z_pooled, valid=valid)
    assert torch.isfinite(terms.sigreg)
    terms.total.backward()
    assert z_pooled.grad is not None and z_pooled.grad.norm().item() > 1e-6


def test_gather_hole_tokens_position_aligned() -> None:
    tok = torch.tensor(
        [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]]]
    )
    hole = torch.tensor([[False, True, True], [True, False, False]])
    out = DiscreteFlowSSLModule._gather_hole_tokens(tok, hole)
    assert out.shape == (3, 2)
    assert out[0, 0].item() == 2.0
    assert out[1, 0].item() == 3.0
    assert out[2, 0].item() == 4.0


def test_gather_visible_tokens() -> None:
    tok = torch.tensor(
        [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]]]
    )
    hole = torch.tensor([[False, True, True], [True, False, False]])
    valid = torch.ones(2, 3, dtype=torch.bool)
    out = DiscreteFlowSSLModule._gather_visible_tokens(tok, hole, valid)
    assert out.shape == (3, 2)
    assert out[0, 0].item() == 1.0
    assert out[1, 0].item() == 4.0
    assert out[2, 0].item() == 5.0


def test_ijepa_has_predictor_and_trains() -> None:
    loss_fn = IJEPALoss(dim=32, sigreg_num_projections=8, sigreg_knots=9)
    assert isinstance(loss_fn.predictor, torch.nn.Module)
    assert loss_fn.predictor.transformer.num_layers == 1
    assert len(list(loss_fn.predictor.parameters())) > 4
    assert not hasattr(loss_fn, "sigreg_norm")


def test_ijepa_predict_invariant_to_target_scale() -> None:
    """Cosine regression is invariant to the target's magnitude."""
    loss_fn = IJEPALoss(dim=16, lejepa_lambda=0.0, sigreg_num_projections=8, sigreg_knots=9)
    tok, hole, valid, target, z_pooled = _ijepa_batch(d=16)
    base = loss_fn(tok, hole, target, z_pooled, valid=valid).predict.item()
    scaled = loss_fn(tok, hole, target * 100.0, z_pooled, valid=valid).predict.item()
    assert abs(base - scaled) < 1e-5


def test_ijepa_predict_stopgrad_blocks_target_grad() -> None:
    loss_fn = IJEPALoss(dim=16, lejepa_lambda=0.0, sigreg_num_projections=8, sigreg_knots=9)
    tok = torch.randn(1, 5, 16, requires_grad=True)
    hole = torch.tensor([[False, True, True, False, False]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    target = torch.randn(2, 16, requires_grad=True)
    z_pooled = torch.randn(1, 16)
    loss_fn(tok, hole, target, z_pooled, valid=valid).predict.backward()
    assert tok.grad is not None and tok.grad.norm().item() > 1e-6
    assert target.grad is None


def test_ijepa_reg_grad_flows_to_pooled() -> None:
    """The anti-collapse regularizer back-props into the pooled z_m batch."""
    loss_fn = IJEPALoss(dim=16, lejepa_lambda=1.0, sigreg_num_projections=8, sigreg_knots=9)
    tok, hole, valid, target, _ = _ijepa_batch(d=16)
    z_pooled = torch.randn(2, 16, requires_grad=True)
    loss_fn(tok, hole, target, z_pooled, valid=valid).total.backward()
    assert z_pooled.grad is not None and z_pooled.grad.norm().item() > 1e-6


def test_ijepa_predictor_reduces_loss_when_visible_encodes_target() -> None:
    torch.manual_seed(0)
    d = 16
    loss_fn = IJEPALoss(dim=d, lejepa_lambda=0.0, sigreg_num_projections=8, sigreg_knots=9)
    target = torch.randn(2, d)
    tok = torch.zeros(1, 4, d)
    hole = torch.tensor([[False, True, True, False]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    z_pooled = torch.randn(1, d)
    tok[0, 0] = target[0]
    tok[0, 3] = target[1]
    opt = torch.optim.Adam(loss_fn.parameters(), lr=1e-3)
    initial = loss_fn(tok, hole, target, z_pooled, valid=valid).predict.item()
    for _ in range(500):
        opt.zero_grad()
        loss_fn(tok, hole, target, z_pooled, valid=valid).predict.backward()
        opt.step()
    assert loss_fn(tok, hole, target, z_pooled, valid=valid).predict.item() < initial * 0.5


def test_ijepa_condition_bypass_gap_keys() -> None:
    loss_fn = IJEPALoss(dim=16, lejepa_lambda=0.0, sigreg_num_projections=8, sigreg_knots=9)
    tok, hole, valid, target, _ = _ijepa_batch(d=16)
    stats = loss_fn.condition_bypass_gap(tok, hole, target, valid=valid)
    for k in ("predict_true", "predict_shuf", "predict_zero", "gap_zero", "gap_shuf"):
        assert k in stats and np.isfinite(stats[k])


def test_ijepa_condition_gap_after_training() -> None:
    """Visible context should beat zero/shuffled after the predictor trains."""
    torch.manual_seed(0)
    d = 16
    loss_fn = IJEPALoss(dim=d, lejepa_lambda=0.0, sigreg_num_projections=8, sigreg_knots=9)
    target = torch.randn(2, d)
    tok = torch.zeros(1, 4, d)
    hole = torch.tensor([[False, True, True, False]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    z_pooled = torch.randn(1, d)
    tok[0, 0] = target[0]
    tok[0, 3] = target[1]
    opt = torch.optim.Adam(loss_fn.parameters(), lr=1e-3)
    for _ in range(300):
        opt.zero_grad()
        loss_fn(tok, hole, target, z_pooled, valid=valid).predict.backward()
        opt.step()
    stats = loss_fn.condition_bypass_gap(tok, hole, target, valid=valid)
    assert stats["predict_true"] < stats["predict_zero"]
    assert stats["gap_zero"] > 0.0


def test_ijepa_gradients_flow_to_tok() -> None:
    loss_fn = IJEPALoss(dim=32, lejepa_lambda=0.05, sigreg_num_projections=16, sigreg_knots=9)
    tok = torch.randn(2, 5, 32, requires_grad=True)
    hole = torch.tensor([[False, True, True, False, False], [True, False, True, False, False]])
    valid = torch.ones(2, 5, dtype=torch.bool)
    target = torch.randn(int(hole.sum()), 32)
    z_pooled = torch.randn(2, 32, requires_grad=True)
    loss_fn(tok, hole, target, z_pooled, valid=valid).total.backward()
    assert tok.grad is not None and tok.grad.norm().item() > 1e-6
    assert z_pooled.grad is not None and z_pooled.grad.norm().item() > 1e-6


def test_ijepa_predictor_batched_matches_variable_lengths() -> None:
    """Batched predictor stacks variable-length ctx+query seqs correctly."""
    from lattice_lab.training.ssl_loss import _IJEPAPredictor

    torch.manual_seed(0)
    pred = _IJEPAPredictor(dim=16, n_layers=1, n_heads=2, max_positions=32)
    tok = torch.randn(3, 6, 16)
    hole = torch.tensor(
        [
            [False, True, True, False, False, False],
            [False, False, True, True, True, False],
            [True, False, False, False, True, False],
        ]
    )
    valid = torch.ones(3, 6, dtype=torch.bool)
    out = pred(tok, hole, valid=valid)
    assert out.shape == (int(hole.sum()), 16)
    assert torch.isfinite(out).all()


def test_ijepa_predictor_uses_visible_position() -> None:
    """Visible reps carry position embeddings, so permuting visible-token order
    changes the hole prediction (a positionless bag would be order-invariant)."""
    from lattice_lab.training.ssl_loss import _IJEPAPredictor

    torch.manual_seed(0)
    pred = _IJEPAPredictor(dim=16, n_layers=1, n_heads=2, max_positions=32).eval()
    tok = torch.randn(1, 5, 16)
    hole = torch.tensor([[False, False, True, False, False]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        base = pred(tok, hole, valid=valid)
        swapped = tok.clone()
        swapped[0, [0, 1, 3, 4]] = tok[0, [4, 3, 1, 0]]  # reorder visible rows
        perm = pred(swapped, hole, valid=valid)
    assert not torch.allclose(base, perm, atol=1e-5)


def test_body_ids_passthrough() -> None:
    ids = [1, 2, 3, 4]
    assert DiscreteFlowSSLModule._body_ids(ids, tokenizer=object()) is ids


def test_subsample_rows() -> None:
    z = torch.arange(20).float().unsqueeze(1).expand(20, 4)
    out = DiscreteFlowSSLModule._subsample_rows(z, 8)
    assert out.shape == (8, 4)
    assert DiscreteFlowSSLModule._subsample_rows(z, 0).shape == (20, 4)


def test_adapter_hole_mask_changes_visible_reps() -> None:
    from lattice_lab.backbone.adapter import Adapter

    adapter = Adapter(d_backbone=8, n_backbone_layers=1, d_adapter=16, n_layers=1, n_heads=2)
    hs = torch.randn(1, 4, 8)
    attn = torch.ones(1, 4)
    hole = torch.tensor([[False, True, False, False]])
    _, free = adapter(hs, attn, return_tokens=True, normalize=False)
    _, blocked = adapter(hs, attn, return_tokens=True, normalize=False, hole_mask=hole)
    assert not torch.allclose(free[0, 0], blocked[0, 0])
    assert not torch.allclose(free[0, 2], blocked[0, 2])


def test_ntxent_perfect_pairs_lower_than_shuffled() -> None:
    z = torch.randn(4, 32)
    z = torch.nn.functional.normalize(z, dim=-1)
    loss_fn = NTXentLoss(temperature=0.1)
    perfect = loss_fn(z, z)
    shuffled = loss_fn(z, z.roll(1, dims=0))
    assert perfect.item() < shuffled.item()


def test_top1_paired_accuracy() -> None:
    z = torch.eye(4)
    assert top1_paired_accuracy(z, z) == 1.0


def test_embedding_covariance_rank_collapsed_vs_full() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    full = rng.standard_normal((128, 32))
    collapsed = rng.standard_normal((128, 1)) @ rng.standard_normal((1, 32))
    full_eff, full_num = embedding_covariance_rank(full)
    collapsed_eff, collapsed_num = embedding_covariance_rank(collapsed)
    assert full_eff > collapsed_eff
    assert full_num > collapsed_num


def test_embedding_covariance_rank_bounded_by_batch() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    z = rng.standard_normal((64, 256))
    eff, num = embedding_covariance_rank(z)
    assert eff <= 64.0
    assert num <= 64.0


def test_lejepa_retrieval_acc1_global_views() -> None:
    c = torch.randn(8, 16)
    z_g = torch.stack([c, c], dim=1)
    z_all = torch.cat([z_g, torch.randn(8, 2, 16)], dim=1)
    assert lejepa_retrieval_acc1(z_g, z_g) == 1.0
    assert lejepa_retrieval_acc1(z_g, z_all) is not None
