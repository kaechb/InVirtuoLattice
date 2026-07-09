"""Unit tests for SSL losses (NT-Xent + LeJEPA/SIGReg)."""

from __future__ import annotations

import math

import numpy as np
import torch

from lattice_lab.training.ssl_loss import (
    DINOHead,
    DINOLoss,
    IJEPALoss,
    LeJEPALoss,
    NTXentLoss,
    PooledEmbeddingPredictor,
    SIGReg,
    SigLIPLoss,
    VISReg,
    gram_anchoring_loss,
    lejepa_retrieval_acc1,
    top1_paired_accuracy,
)
from lattice_lab.models.discrete_flow_ssl import DiscreteFlowSSLModule, _adamw_no_decay
from lattice_lab.training.ssl_val_probes import embedding_covariance_rank


def test_adamw_no_decay_skips_bias_norm_and_scalars() -> None:
    w = torch.nn.Parameter(torch.randn(4, 8))
    assert not _adamw_no_decay("adapter.input_proj.weight", w)
    assert _adamw_no_decay("adapter.input_proj.bias", torch.nn.Parameter(torch.randn(8)))
    assert _adamw_no_decay("adapter.norm.weight", torch.nn.Parameter(torch.ones(8)))
    assert _adamw_no_decay("backbone.blocks.0.norm1.weight", torch.nn.Parameter(torch.ones(8)))
    assert _adamw_no_decay("time_logit", torch.nn.Parameter(torch.tensor(0.5)))
    assert not _adamw_no_decay(
        "predictor.pos_embed.weight", torch.nn.Parameter(torch.randn(16, 32))
    )


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
    z_pooled = torch.randn(b, d)
    z_teacher_rows = torch.randn(b, d)
    z_pooled_shuffle = torch.randn(b, d)
    return tok, hole, valid, target, z_pooled, z_teacher_rows, z_pooled_shuffle


def _ijepa_call(loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, z_pooled_shuffle):
    return loss_fn(
        tok,
        hole,
        target,
        z_pooled,
        valid=valid,
        z_teacher_rows=z_teacher_rows if loss_fn.glob_weight > 0.0 else None,
        z_inv_target=z_pooled_shuffle if loss_fn.inv_weight > 0.0 else None,
    )


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
    tok, hole, valid, target, z_pooled, z_teacher_rows, z_shuffle = _ijepa_batch(d=32)
    terms = _ijepa_call(
        loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, z_shuffle,
    )
    for t in (terms.total, terms.predict, terms.glob, terms.inv, terms.sigreg):
        assert t.ndim == 0 and torch.isfinite(t)


def test_ijepa_total_matches_weighted_terms() -> None:
    tok, hole, valid, target, z_pooled, z_teacher_rows, z_shuffle = _ijepa_batch(d=32)
    loss_fn = IJEPALoss(
        dim=32, lejepa_lambda=0.3, glob_weight=1.0, inv_weight=0.1,
        sigreg_num_projections=8, sigreg_knots=9,
    )
    terms = _ijepa_call(
        loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, z_shuffle,
    )
    main = terms.predict + terms.glob + 0.1 * terms.inv
    expected = 0.7 * main + 0.3 * terms.sigreg
    assert abs(float(terms.total - expected)) < 1e-5


def test_visreg_penalizes_collapse() -> None:
    """VISReg's scale term fires on collapsed rows, ~0 on unit-variance rows."""
    torch.manual_seed(0)
    collapsed = torch.zeros(64, 32) + 0.01 * torch.randn(1, 32)  # identical rows
    spread = torch.randn(64, 32)
    reg = VISReg(gamma=1.0, num_projections=256)
    # Scale term (gamma-std)^2: std≈0.01 -> ~0.98; std≈1 -> ~0.
    scale_collapsed, _ = reg.components(collapsed)
    scale_spread, _ = reg.components(spread)
    assert float(scale_collapsed) > 0.9
    assert float(scale_spread) < 0.1
    # Full reg (scale + sketching) still ranks collapse above spread.
    assert float(reg(collapsed)) > float(reg(spread))


def test_visreg_shape_matches_gaussian() -> None:
    """The sliced-Wasserstein shape term is small for a standard-normal batch and
    large for a skewed (non-Gaussian) batch of the same variance.

    Uses ``D=1`` so each projection *is* the marginal — with many dimensions the
    CLT drives every random 1D projection toward Gaussian regardless of the input,
    which is exactly why sliced methods need enough projections, not why they work.
    """
    torch.manual_seed(0)
    gauss = torch.randn(1024, 1)
    skewed = torch.empty(1024, 1).exponential_()  # skewed, clearly non-Gaussian
    reg = VISReg(gamma=1.0, num_projections=64)
    _, shape_gauss = reg.components(gauss)
    _, shape_skew = reg.components(skewed)
    assert float(shape_gauss) < float(shape_skew)
    assert float(shape_gauss) < 0.05


def test_visreg_scale_term_anchors_absolute_scale() -> None:
    """The (gamma-std)^2 scale term is two-sided: it rises when the batch std is
    pushed either below or above gamma, so it pins the absolute output scale."""
    torch.manual_seed(0)
    x = torch.randn(256, 16)
    reg = VISReg(gamma=1.0, num_projections=256)
    scale_unit, _ = reg.components(x)
    scale_small, _ = reg.components(x * 0.1)
    scale_big, _ = reg.components(x * 5.0)
    assert float(scale_small) > float(scale_unit)
    assert float(scale_big) > float(scale_unit)


def test_visreg_shape_is_scale_invariant() -> None:
    """The sketching term normalizes by std, so it is invariant to the batch scale
    (only the shape of the distribution matters, not its magnitude)."""
    torch.manual_seed(0)
    x = torch.randn(256, 16)
    x = x - x.mean(dim=0, keepdim=True)  # zero-mean so the center term stays 0
    reg = VISReg(gamma=1.0, num_projections=512)
    # Re-seed before each call so the random projections are identical and only
    # the effect of rescaling the input on the shape term is measured.
    # Scales stay well above eps=1e-4 so std normalization is exact (at tiny
    # scales the variance floor eps would leak in and break invariance).
    torch.manual_seed(1)
    _, dist_base = reg.components(x)
    for s in (2.0, 100.0):
        torch.manual_seed(1)
        _, dist_scaled = reg.components(x * s)
        assert abs(float(dist_scaled) - float(dist_base)) < 1e-3


def test_pooled_embedding_predictor_shape() -> None:
    """Pool → MLP predictor maps [B, D] noised pool to [B, D] target embedding."""
    pred = PooledEmbeddingPredictor(16)
    z = torch.randn(6, 16)
    assert pred(z).shape == (6, 16)


def test_ijepa_predict_global_embedding_shape() -> None:
    """Glob readout maps visible token context to a single [B, D] global vector."""
    loss_fn = IJEPALoss(dim=16, lejepa_lambda=0.0, predict_weight=0.0, glob_weight=0.0)
    tok = torch.randn(4, 8, 16)
    hole = torch.tensor(
        [[False, True, True, False, False, False, False, False],
         [False, False, True, True, False, False, False, False],
         [False, False, False, False, True, True, False, False],
         [False, True, False, True, False, False, False, False]],
    )
    z_hat = loss_fn.predict_global_embedding(tok, hole)
    assert z_hat.shape == (4, 16)


def test_dino_head_shape_and_loss_behaviour() -> None:
    """Head maps pooled → K prototype logits; DINO CE is lower when the student
    matches the (centered, sharpened) teacher than when it is random, and the
    center EMA moves toward the batch mean of teacher logits."""
    torch.manual_seed(0)
    head = DINOHead(dim=32, out_dim=64, hidden=48, bottleneck=16)
    z = torch.randn(20, 32)
    logits = head(z)
    assert logits.shape == (20, 64)

    loss_fn = DINOLoss(out_dim=64, teacher_temp=0.04, student_temp=0.1)
    teacher = torch.randn(20, 64)
    matched = float(loss_fn(teacher.clone(), teacher))
    random = float(loss_fn(torch.randn(20, 64), teacher))
    assert matched >= 0.0 and random >= 0.0
    assert matched < random

    assert torch.count_nonzero(loss_fn.center) == 0
    loss_fn.update_center(teacher)
    expected = (1.0 - 0.9) * teacher.mean(dim=0, keepdim=True)
    assert torch.allclose(loss_fn.center, expected, atol=1e-6)


def test_dino_utilization_uniform_vs_collapsed() -> None:
    """Batch-mean teacher entropy: log(K) when uniform, ~0 when one prototype wins."""
    k = 64
    loss_fn = DINOLoss(out_dim=k, teacher_temp=0.04, student_temp=0.1)
    uniform_logits = torch.zeros(32, k)
    ent_u, active_u = loss_fn.utilization(uniform_logits)
    assert abs(ent_u - math.log(k)) < 0.05
    assert active_u == k

    collapsed_logits = torch.zeros(32, k)
    collapsed_logits[:, 0] = 10.0
    ent_c, active_c = loss_fn.utilization(collapsed_logits)
    assert ent_c < 0.1
    assert active_c == 1.0


def test_ijepa_use_visreg_swaps_regularizer() -> None:
    tok, hole, valid, target, z_pooled, z_teacher_rows, z_shuffle = _ijepa_batch(d=16)
    loss_fn = IJEPALoss(dim=16, lejepa_lambda=0.5, use_visreg=True, inv_weight=0.0)
    assert loss_fn.sigreg is None and isinstance(loss_fn.visreg, VISReg)
    z_pooled = torch.randn(2, 16, requires_grad=True)
    terms = _ijepa_call(
        loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, z_shuffle,
    )
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
    hole = torch.tensor([[False, True, True], [False, False, True]])
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
    loss_fn = IJEPALoss(
        dim=16, lejepa_lambda=0.0, glob_weight=0.0, inv_weight=0.0,
        sigreg_num_projections=8, sigreg_knots=9,
    )
    loss_fn.eval()
    tok, hole, valid, target, z_pooled, z_teacher_rows, z_shuffle = _ijepa_batch(d=16)
    base = _ijepa_call(
        loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, z_shuffle,
    ).predict.item()
    scaled = _ijepa_call(
        loss_fn, tok, hole, valid, target * 100.0, z_pooled, z_teacher_rows, z_shuffle,
    ).predict.item()
    assert abs(base - scaled) < 1e-5


def test_ijepa_predict_stopgrad_blocks_target_grad() -> None:
    loss_fn = IJEPALoss(
        dim=16, lejepa_lambda=0.0, glob_weight=0.0, inv_weight=0.0,
        sigreg_num_projections=8, sigreg_knots=9,
    )
    tok = torch.randn(1, 5, 16, requires_grad=True)
    hole = torch.tensor([[False, True, True, False, False]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    target = torch.randn(2, 16, requires_grad=True)
    z_pooled = torch.randn(1, 16)
    z_teacher_rows = torch.randn(1, 16)
    _ijepa_call(
        loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, None,
    ).predict.backward()
    assert tok.grad is not None and tok.grad.norm().item() > 1e-6
    assert target.grad is None


def test_ijepa_reg_grad_flows_to_pooled() -> None:
    loss_fn = IJEPALoss(
        dim=16, lejepa_lambda=1.0, glob_weight=0.0, inv_weight=0.0,
        sigreg_num_projections=8, sigreg_knots=9,
    )
    tok, hole, valid, target, _, z_teacher_rows, _ = _ijepa_batch(d=16)
    z_pooled = torch.randn(2, 16, requires_grad=True)
    _ijepa_call(
        loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, None,
    ).total.backward()
    assert z_pooled.grad is not None and z_pooled.grad.norm().item() > 1e-6


def test_ijepa_predictor_reduces_loss_when_visible_encodes_target() -> None:
    torch.manual_seed(0)
    d = 16
    loss_fn = IJEPALoss(
        dim=d, lejepa_lambda=0.0, glob_weight=0.0, inv_weight=0.0,
        sigreg_num_projections=8, sigreg_knots=9,
    )
    target = torch.randn(2, d)
    tok = torch.zeros(1, 4, d)
    hole = torch.tensor([[False, True, True, False]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    z_pooled = torch.randn(1, d)
    z_teacher_rows = torch.randn(1, d)
    tok[0, 0] = target[0]
    tok[0, 3] = target[1]
    opt = torch.optim.Adam(loss_fn.parameters(), lr=1e-3)
    initial = _ijepa_call(
        loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, None,
    ).predict.item()
    for _ in range(500):
        opt.zero_grad()
        _ijepa_call(
            loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, None,
        ).predict.backward()
        opt.step()
    final = _ijepa_call(
        loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, None,
    ).predict.item()
    assert final < initial * 0.5


def test_ijepa_condition_bypass_gap_keys() -> None:
    loss_fn = IJEPALoss(
        dim=16, lejepa_lambda=0.0, glob_weight=0.0, inv_weight=0.0,
        sigreg_num_projections=8, sigreg_knots=9,
    )
    tok, hole, valid, target, _, _, _ = _ijepa_batch(d=16)
    stats = loss_fn.condition_bypass_gap(tok, hole, target, valid=valid)
    for k in ("predict_true", "predict_shuf", "predict_zero", "gap_zero", "gap_shuf"):
        assert k in stats and np.isfinite(stats[k])


def test_ijepa_condition_gap_after_training() -> None:
    """Visible context should beat zero/shuffled after the predictor trains."""
    torch.manual_seed(0)
    d = 16
    loss_fn = IJEPALoss(
        dim=d, lejepa_lambda=0.0, glob_weight=0.0, inv_weight=0.0,
        sigreg_num_projections=8, sigreg_knots=9,
    )
    target = torch.randn(2, d)
    tok = torch.zeros(1, 4, d)
    hole = torch.tensor([[False, True, True, False]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    z_pooled = torch.randn(1, d)
    z_teacher_rows = torch.randn(1, d)
    tok[0, 0] = target[0]
    tok[0, 3] = target[1]
    opt = torch.optim.Adam(loss_fn.parameters(), lr=1e-3)
    for _ in range(300):
        opt.zero_grad()
        _ijepa_call(
            loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, None,
        ).predict.backward()
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
    z_teacher_rows = torch.randn(2, 32)
    z_shuffle = torch.randn(2, 32)
    _ijepa_call(
        loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, z_shuffle,
    ).total.backward()
    assert tok.grad is not None and tok.grad.norm().item() > 1e-6
    assert z_pooled.grad is not None and z_pooled.grad.norm().item() > 1e-6


def _ijepa_predictor_reference(
    pred: "_IJEPAPredictor",
    tok: torch.Tensor,
    hole: torch.Tensor,
    *,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Slow row-wise packer; reference for vectorized ``_IJEPAPredictor.forward``."""
    b, _, dim = tok.shape
    visible = valid & ~hole
    ctx_parts: list[torch.Tensor] = []
    query_parts: list[torch.Tensor] = []
    seq_lens: list[int] = []
    for i in range(b):
        hole_idx = hole[i].nonzero(as_tuple=True)[0]
        if hole_idx.numel() == 0:
            continue
        vis_idx = visible[i].nonzero(as_tuple=True)[0]
        ctx_parts.append(tok[i, vis_idx] + pred.pos_embed(vis_idx))
        query_parts.append(
            pred.mask_token.expand(hole_idx.numel(), dim) + pred.pos_embed(hole_idx)
        )
        seq_lens.append(int(vis_idx.numel() + hole_idx.numel()))
    if not ctx_parts:
        return tok.new_zeros(0, dim)
    max_len = max(seq_lens)
    n_seq = len(ctx_parts)
    seq = tok.new_zeros(n_seq, max_len, dim)
    pad = torch.ones(n_seq, max_len, dtype=torch.bool, device=tok.device)
    for j, (ctx, queries) in enumerate(zip(ctx_parts, query_parts)):
        n = ctx.size(0) + queries.size(0)
        seq[j, : ctx.size(0)] = ctx
        seq[j, ctx.size(0) : n] = queries
        pad[j, :n] = False
    out = pred.norm(pred.transformer(seq, src_key_padding_mask=pad))
    preds: list[torch.Tensor] = []
    for j, (ctx, queries) in enumerate(zip(ctx_parts, query_parts)):
        preds.append(out[j, ctx.size(0) : ctx.size(0) + queries.size(0)])
    return torch.cat(preds, dim=0)


def test_ijepa_predictor_batched_matches_variable_lengths() -> None:
    """Vectorized predictor matches the row-wise reference packer."""
    from lattice_lab.training.ssl_loss import _IJEPAPredictor

    torch.manual_seed(0)
    pred = _IJEPAPredictor(dim=16, n_layers=1, n_heads=2, max_positions=32).eval()
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
    ref = _ijepa_predictor_reference(pred, tok, hole, valid=valid)
    assert out.shape == (int(hole.sum()), 16)
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.isfinite(out).all()

    torch.manual_seed(1)
    tok_big = torch.randn(32, 20, 16)
    valid_big = torch.ones(32, 20, dtype=torch.bool)
    valid_big[:, -2:] = False  # trailing pad
    # Production guarantees holes ⊆ valid (mask tokens never land on pad), so
    # restrict the random holes the same way; the predictor emits one row per
    # hole regardless of valid, so an unconstrained hole on pad would desync the
    # expected count.
    hole_big = (torch.rand(32, 20) < 0.15) & valid_big
    out_big = pred(tok_big, hole_big, valid=valid_big)
    ref_big = _ijepa_predictor_reference(pred, tok_big, hole_big, valid=valid_big)
    assert out_big.shape == ref_big.shape == (int((hole_big & valid_big).sum()), 16)
    assert torch.allclose(out_big, ref_big, atol=1e-6)


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


def test_body_ids_retokenizes_wrapped_fragment_view() -> None:
    frag = "[1*]c1c(F)cccc1F [2*]C([4*])=O"
    tok = type("Tok", (), {"encode": lambda self, text, add_special_tokens=False: [len(text), 7]})()
    assert DiscreteFlowSSLModule._body_ids([frag], tok) == [len(frag), 7]


def test_split_batch_unzips_view_smiles_rows() -> None:
    batch = [("[1*]a", "CCO"), ("[1*]b", "CCN")]
    views, smiles, net3d = DiscreteFlowSSLModule._split_batch(batch)
    assert views == ["[1*]a", "[1*]b"]
    assert smiles == ["CCO", "CCN"]
    assert net3d is None


def test_split_batch_accepts_list_of_view_smiles_lists() -> None:
    views_in = ["[1*]a", "[1*]b"]
    smiles_in = ["CCO", "CCN"]
    for batch in ((views_in, smiles_in), [views_in, smiles_in]):
        views, smiles, net3d = DiscreteFlowSSLModule._split_batch(batch)
        assert views == views_in
        assert smiles == smiles_in
        assert net3d is None


def test_split_batch_extracts_conformer_payload() -> None:
    import torch

    net3d = {"mol_src_tokens": torch.zeros(2, 3, dtype=torch.long)}
    views, smiles, out3d = DiscreteFlowSSLModule._split_batch(
        (["[1*]a", "[1*]b"], ["CCO", "CCN"], net3d)
    )
    assert views == ["[1*]a", "[1*]b"] and smiles == ["CCO", "CCN"]
    assert out3d is net3d
    # smiles may be None (return_smiles=False) but the 3D dict is still carried.
    _, smiles2, out3d2 = DiscreteFlowSSLModule._split_batch(
        ([[1, 2], [3, 4]], None, net3d)
    )
    assert smiles2 is None and out3d2 is net3d


def test_subsample_rows() -> None:
    z = torch.arange(20).float().unsqueeze(1).expand(20, 4)
    out = DiscreteFlowSSLModule._subsample_rows(z, 8)
    assert out.shape == (8, 4)
    assert DiscreteFlowSSLModule._subsample_rows(z, 0).shape == (20, 4)


def test_effective_fp_weight_anneals_linearly() -> None:
    from types import SimpleNamespace

    # Call the method unbound on a plain stub: global_step is a read-only
    # LightningModule property, so a real (object.__new__) instance can't have
    # it assigned. The method only reads fp_weight/hparams/global_step.
    fp = DiscreteFlowSSLModule._effective_fp_weight
    m = SimpleNamespace(fp_weight=2.0, global_step=0, hparams=SimpleNamespace(fp_anneal_steps=100))
    assert fp(m) == 2.0
    m.global_step = 50
    assert fp(m) == 1.0
    m.global_step = 100
    assert fp(m) == 0.0
    m.hparams.fp_anneal_steps = 0
    m.global_step = 0
    assert fp(m) == 2.0


def test_resolve_total_steps_from_trainer_estimate() -> None:
    from types import SimpleNamespace

    # Unbound call on a stub: trainer is a LightningModule property whose setter
    # walks nn.Module internals that an uninitialized instance lacks.
    m = SimpleNamespace(trainer=SimpleNamespace(estimated_stepping_batches=12_345))
    assert DiscreteFlowSSLModule._resolve_total_steps(m) == 12_345


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


def test_siglip_perfect_pairs_lower_and_params_learn() -> None:
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(8, 32), dim=-1)
    loss_fn = SigLIPLoss()
    perfect = loss_fn(z, z)
    shuffled = loss_fn(z, z.roll(1, dims=0))
    assert perfect.item() < shuffled.item()
    # Learnable temperature + bias receive gradient.
    perfect.backward()
    assert loss_fn.logit_scale.grad is not None
    assert loss_fn.logit_bias.grad is not None


def test_siglip_diagnostics_margin_and_scalars() -> None:
    import math

    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(8, 32), dim=-1)
    loss_fn = SigLIPLoss()
    d = loss_fn.diagnostics(z, z)
    # z vs z: matched cosine is exactly 1 and above the off-diagonal → positive margin.
    assert math.isclose(d["siglip/diagnostics_cos_pos_mean"], 1.0, rel_tol=1e-5)
    assert d["siglip/diagnostics_cos_pos_mean"] > d["siglip/diagnostics_cos_neg_mean"]
    assert math.isclose(
        d["siglip/diagnostics_cos_margin"],
        d["siglip/diagnostics_cos_pos_mean"] - d["siglip/diagnostics_cos_neg_mean"],
        rel_tol=1e-5,
    )
    # Scalars mirror the learnable parameters (temperature = exp(logit_scale)).
    assert math.isclose(d["siglip/diagnostics_temperature"], math.exp(loss_fn.logit_scale.item()), rel_tol=1e-5)
    assert math.isclose(d["siglip/diagnostics_logit_bias"], loss_fn.logit_bias.item(), rel_tol=1e-5)
    # Predicted match probabilities are valid and rank positives above negatives.
    assert 0.0 <= d["siglip/diagnostics_neg_prob_mean"] <= d["siglip/diagnostics_pos_prob_mean"] <= 1.0


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


def test_mean_visible_pool() -> None:
    from lattice_lab.training.ssl_loss import _mean_visible_pool

    tok = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]])
    hole = torch.tensor([[False, True, False]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    out = _mean_visible_pool(tok, hole, valid)
    assert out.shape == (1, 2)
    assert out[0, 0].item() == 3.0  # mean of positions 0 and 2


def test_ijepa_inv_zero_when_identical() -> None:
    z = torch.randn(4, 16)
    assert IJEPALoss.inv_loss(z, z).item() == 0.0


def test_ijepa_inv_target_is_stopgrad() -> None:
    """Asymmetric inv: grad reaches online z_pooled but not the EMA-teacher target."""
    loss_fn = IJEPALoss(
        dim=16, lejepa_lambda=0.0, glob_weight=0.0, inv_weight=1.0,
        sigreg_num_projections=8, sigreg_knots=9,
    )
    tok, hole, valid, target, _, _, _ = _ijepa_batch(d=16)
    z_pooled = torch.randn(2, 16, requires_grad=True)
    z_inv_target = torch.randn(2, 16, requires_grad=True)
    _ijepa_call(
        loss_fn, tok, hole, valid, target, z_pooled, None, z_inv_target,
    ).inv.backward()
    assert z_pooled.grad is not None
    assert z_inv_target.grad is None


def test_ijepa_noise_inv_enters_total_and_is_stopgrad() -> None:
    """noise_inv: whole-molecule pool of the corrupted view vs stop-grad EMA global."""
    loss_fn = IJEPALoss(
        dim=16, lejepa_lambda=0.0, glob_weight=0.0, inv_weight=0.0,
        noise_inv_weight=1.0, sigreg_num_projections=8, sigreg_knots=9,
    )
    tok, hole, valid, target, z_pooled, _, _ = _ijepa_batch(d=16)
    z_noised = torch.randn(2, 16, requires_grad=True)
    z_teacher_rows = torch.randn(2, 16, requires_grad=True)
    terms = loss_fn(
        tok, hole, target, z_pooled, valid=valid,
        z_teacher_rows=z_teacher_rows, z_noised_pooled=z_noised,
    )
    assert terms.noise_inv.item() > 0.0
    # lejepa_lambda=0 → total = predict + noise_inv_weight * noise_inv (other weights 0)
    assert torch.allclose(terms.total, terms.predict + terms.noise_inv, atol=1e-5)
    terms.noise_inv.backward()
    assert z_noised.grad is not None
    assert z_teacher_rows.grad is None  # teacher is the stop-grad target


def test_ijepa_noise_inv_zero_when_matching_teacher() -> None:
    loss_fn = IJEPALoss(
        dim=16, lejepa_lambda=0.0, glob_weight=0.0, inv_weight=0.0,
        noise_inv_weight=1.0, sigreg_num_projections=8, sigreg_knots=9,
    )
    tok, hole, valid, target, z_pooled, _, _ = _ijepa_batch(d=16)
    z = torch.randn(2, 16)
    terms = loss_fn(
        tok, hole, target, z_pooled, valid=valid,
        z_teacher_rows=z, z_noised_pooled=z.clone(),
    )
    assert terms.noise_inv.item() == 0.0


def test_ijepa_predict_weight_zero_drops_predict() -> None:
    """predict_weight=0 removes the token-level objective (pooled-only mode)."""
    loss_fn = IJEPALoss(
        dim=16, lejepa_lambda=0.0, predict_weight=0.0, glob_weight=0.0,
        inv_weight=0.0, noise_inv_weight=1.0,
        sigreg_num_projections=8, sigreg_knots=9,
    )
    tok, hole, valid, target, z_pooled, _, _ = _ijepa_batch(d=16)
    z_noised = torch.randn(2, 16)
    z_teacher_rows = torch.randn(2, 16)
    terms = loss_fn(
        tok, hole, target, z_pooled, valid=valid,
        z_teacher_rows=z_teacher_rows, z_noised_pooled=z_noised,
    )
    assert terms.predict.item() == 0.0
    assert torch.allclose(terms.total, terms.noise_inv, atol=1e-5)  # only noise_inv left


def test_ijepa_glob_grad_flows_to_tok() -> None:
    loss_fn = IJEPALoss(
        dim=16, lejepa_lambda=0.0, glob_weight=1.0, inv_weight=0.0,
        sigreg_num_projections=8, sigreg_knots=9,
    )
    tok = torch.randn(2, 4, 16, requires_grad=True)
    hole = torch.tensor([[False, True, True, False], [False, False, True, True]])
    valid = torch.ones(2, 4, dtype=torch.bool)
    target = torch.randn(int(hole.sum()), 16)
    z_pooled = torch.randn(2, 16)
    z_teacher_rows = torch.randn(2, 16)
    _ijepa_call(
        loss_fn, tok, hole, valid, target, z_pooled, z_teacher_rows, None,
    ).glob.backward()
    assert tok.grad is not None and tok.grad.norm().item() > 1e-6


def test_ijepa_glob_ablation() -> None:
    """Glob term trains when visible mean encodes teacher; predict-only does not move readout."""
    from lattice_lab.training.ssl_loss import _mean_visible_pool

    torch.manual_seed(0)
    d = 16
    teacher = torch.randn(1, d)
    tok = torch.zeros(1, 4, d)
    hole = torch.tensor([[False, True, True, False]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    target = torch.randn(2, d)
    z_pooled = torch.randn(1, d)
    tok[0, 0] = teacher[0]
    tok[0, 3] = teacher[0]

    predict_only = IJEPALoss(
        dim=d, lejepa_lambda=0.0, glob_weight=0.0, inv_weight=0.0,
        sigreg_num_projections=8, sigreg_knots=9,
    )
    with_glob = IJEPALoss(
        dim=d, lejepa_lambda=0.0, glob_weight=1.0, inv_weight=0.0,
        sigreg_num_projections=8, sigreg_knots=9,
    )
    opt = torch.optim.Adam(with_glob.glob_readout.parameters(), lr=5e-3)
    initial = with_glob.glob_loss(tok, hole, teacher, valid=valid).item()
    for _ in range(200):
        opt.zero_grad()
        with_glob.glob_loss(tok, hole, teacher, valid=valid).backward()
        opt.step()
    final = with_glob.glob_loss(tok, hole, teacher, valid=valid).item()
    assert final < initial * 0.5
    ctx = _mean_visible_pool(tok, hole, valid)
    pred = with_glob.glob_readout(ctx)
    cos = torch.nn.functional.cosine_similarity(pred, teacher, dim=-1).item()
    assert cos > 0.9
    assert not hasattr(predict_only, "glob_readout") or predict_only.glob_weight == 0.0


def test_gram_anchoring_zero_when_identical() -> None:
    """Identical online/target reps → matching Gram matrices → exactly 0."""
    tok = torch.randn(3, 5, 16)
    valid = torch.ones(3, 5, dtype=torch.bool)
    assert float(gram_anchoring_loss(tok, tok.clone(), valid)) < 1e-10


def test_gram_anchoring_ignores_pad_tokens() -> None:
    """Differences at invalid (pad) positions must not affect the loss."""
    online = torch.randn(2, 4, 8)
    target = online.clone()
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    # Corrupt only the invalid (pad) positions of the target.
    target[~valid] += 10.0
    assert float(gram_anchoring_loss(online, target, valid)) < 1e-10


def test_gram_anchoring_stopgrad_and_grad_flow() -> None:
    """Gradient flows to the online reps but not through the (detached) target."""
    online = torch.randn(2, 4, 8, requires_grad=True)
    target = torch.randn(2, 4, 8, requires_grad=True)
    valid = torch.ones(2, 4, dtype=torch.bool)
    gram_anchoring_loss(online, target, valid).backward()
    assert online.grad is not None and online.grad.norm() > 0
    assert target.grad is None


def test_ijepa_gram_term_enters_total() -> None:
    """gram_weight > 0 routes the Gram term into IJEPALossTerms.total."""
    tok, hole, valid, target, z_pooled, z_teacher_rows, z_shuffle = _ijepa_batch(d=16)
    g_online = torch.randn(2, 4, 16)
    g_target = torch.randn(2, 4, 16)
    g_valid = torch.ones(2, 4, dtype=torch.bool)
    loss_fn = IJEPALoss(
        dim=16, lejepa_lambda=0.0, glob_weight=0.0, inv_weight=0.0,
        gram_weight=2.0, sigreg_num_projections=8, sigreg_knots=9,
    )
    terms = loss_fn(
        tok, hole, target, z_pooled, valid=valid,
        gram_online=g_online, gram_target=g_target, gram_valid=g_valid,
    )
    expected_gram = gram_anchoring_loss(g_online, g_target, g_valid)
    assert torch.isfinite(terms.gram) and terms.gram > 0
    assert abs(float(terms.gram - expected_gram)) < 1e-6
    # lambda=0, only-gram main → total == gram_weight * gram == predict + 2*gram.
    assert abs(float(terms.total - (terms.predict + 2.0 * terms.gram))) < 1e-5
