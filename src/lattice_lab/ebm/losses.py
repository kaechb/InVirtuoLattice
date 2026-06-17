"""Losses for the Stage 4 EBM head.

Three components combined per the README::

    L_total = L_InfoNCE + λ_sink · L_Sinkhorn + λ_neg · L_cross_target

- ``L_InfoNCE``: temperature-scaled contrastive between the energy of the
  positive (binder) and ``N`` decoys per target. Implemented as
  ``cross_entropy`` over logits = ``-E / T`` with the binder at index 0.
- ``L_Sinkhorn``: Sinkhorn divergence between the observed decoy-energy
  distribution and a target prior ``q*`` = delta on the binder + narrow
  Gaussian for negatives + heavy Student-t tail. The tail absorbs false
  negatives without paying penalty. We use an entropy-regularised log-domain
  Sinkhorn on 1D scalars — no external geomloss dependency, matching the
  README §Engineering "minimise non-essential deps" preference (the dep is
  documented as optional).
- ``L_cross_target``: hinge that pushes ``E(z_m+, z_p_correct)`` below
  ``E(z_m+, z_p_other)`` by a configurable margin, forcing the head to learn
  target specificity rather than drug-likeness.

All three accept (binder_energies, decoy_energies) or the components needed
to compute them, and return scalar losses ready to ``.backward()``.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn


def _prior_default(name: str, fallback: str) -> float:
    return float(os.environ.get(name, fallback))


def _flatten_pair(
    binder_e: torch.Tensor, decoy_e: torch.Tensor
) -> tuple[torch.Tensor, int, int]:
    """Validate and reshape (binder, decoys) tensors to a canonical [B, 1+N] form.

    ``binder_e``: shape ``[B]`` — one energy per target (the binder).
    ``decoy_e``:  shape ``[B, N]`` — N decoy energies per target.
    Returns the stacked ``[B, 1+N]`` matrix with the binder at index 0, plus
    ``(B, N)``.
    """
    if binder_e.ndim != 1:
        raise ValueError(f"binder_e must be 1D, got shape {tuple(binder_e.shape)}")
    if decoy_e.ndim != 2 or decoy_e.shape[0] != binder_e.shape[0]:
        raise ValueError(
            "decoy_e must be [B, N] aligned with binder_e [B]; got "
            f"{tuple(decoy_e.shape)} vs {tuple(binder_e.shape)}"
        )
    return torch.cat([binder_e.unsqueeze(1), decoy_e], dim=1), binder_e.shape[0], decoy_e.shape[1]


class InfoNCEEnergyLoss(nn.Module):
    """Per-target InfoNCE on (negative) energy logits.

    For each row in the batch we have one binder energy ``e+`` and ``N`` decoy
    energies ``e-_1…e-_N``. The logit for class ``i`` is ``-e_i / T``; the
    target class is the binder (index 0). Cross-entropy then minimises
    ``-log softmax(-e+/T)``, which is exactly the contrastive objective
    pushing ``e+`` below the decoy energies.

    Also reports a top-1 accuracy (binder ranked first) for monitoring.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(
        self, binder_e: torch.Tensor, decoy_e: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        all_e, b, _ = _flatten_pair(binder_e, decoy_e)
        logits = -all_e / self.temperature           # [B, 1+N]
        targets = torch.zeros(b, dtype=torch.long, device=all_e.device)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        with torch.no_grad():
            ranked = logits.argmax(dim=1)
            top1 = (ranked == 0).float().mean().item()
        return loss, {"infonce/top1": top1, "infonce/loss": float(loss.detach().item())}


# --------------------------------------------------------------------------
# Sinkhorn divergence on 1D scalars
# --------------------------------------------------------------------------


def sinkhorn_divergence_1d(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    blur: float = 0.05,
    n_iter: int = 50,
) -> torch.Tensor:
    """Symmetrised Sinkhorn divergence ``S_ε(x, y) = OT_ε(x, y) − ½[OT_ε(x,x) + OT_ε(y,y)]``.

    Inputs are 1D point clouds in ``R^1``, each shape ``[..., M]``. We use
    squared-Euclidean ground cost and log-domain Sinkhorn iterations with
    entropy regularisation ``ε = blur²``. The leading dims of ``x`` and ``y``
    must match — the divergence is computed per-row, then averaged.

    No external dependency: ``geomloss`` would be faster but its KeOps backend
    is heavy and platform-fragile. For the cardinalities we use (≤ 1024
    points) the pure-torch implementation is fast enough on GPU.
    """
    if x.shape[:-1] != y.shape[:-1]:
        raise ValueError(
            f"leading shapes must match: {tuple(x.shape[:-1])} vs {tuple(y.shape[:-1])}"
        )
    eps = blur * blur
    ot_xy = _entropic_ot(x, y, eps=eps, n_iter=n_iter)
    ot_xx = _entropic_ot(x, x, eps=eps, n_iter=n_iter)
    ot_yy = _entropic_ot(y, y, eps=eps, n_iter=n_iter)
    return (ot_xy - 0.5 * (ot_xx + ot_yy)).mean()


def _entropic_ot(
    x: torch.Tensor, y: torch.Tensor, *, eps: float, n_iter: int
) -> torch.Tensor:
    """Entropy-regularised OT between uniform measures on 1D point sets.

    Returns a tensor of shape ``x.shape[:-1]`` — one OT value per leading batch.
    """
    # Cost: squared distance, [..., M, N]
    x_ = x.unsqueeze(-1)        # [..., M, 1]
    y_ = y.unsqueeze(-2)        # [..., 1, N]
    cost = (x_ - y_).pow(2)
    m = x.shape[-1]
    n = y.shape[-1]
    # Uniform log-weights.
    log_a = torch.full_like(x, -math.log(m))
    log_b = torch.full_like(y, -math.log(n))
    # Log-Sinkhorn iterations.
    K = -cost / eps  # [..., M, N]
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(n_iter):
        log_v = log_b - torch.logsumexp(K + log_u.unsqueeze(-1), dim=-2)
        log_u = log_a - torch.logsumexp(K + log_v.unsqueeze(-2), dim=-1)
    # Plan-weighted cost.
    log_pi = K + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    plan = log_pi.exp()
    return (plan * cost).sum(dim=(-1, -2))


# --------------------------------------------------------------------------
# Target prior q*
# --------------------------------------------------------------------------


def sample_target_prior(
    n_samples: int,
    binder_e: torch.Tensor,
    *,
    alpha: float | None = None,
    pi: float | None = None,
    mu_neg: float | None = None,
    sigma_neg: float = 0.3,
    student_t_df: float | None = None,
    student_t_scale: float = 1.0,
    student_t_loc: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``n_samples`` per row from the target prior ``q*``.

    Mixture (README §Stage 4):
    - ``alpha`` mass on the binder delta (placed at the empirical binder energy).
    - ``(1 - alpha) * pi`` mass on a narrow Gaussian for true non-binders.
    - ``(1 - alpha) * (1 - pi)`` mass on a Student-t heavy tail to absorb
      false negatives (decoys that are actually binders).

    Args:
        n_samples: how many points to draw per target.
        binder_e: ``[B]`` — the binder energies, used to place the delta and to
            anchor the device/dtype.

    Returns:
        ``[B, n_samples]`` samples from ``q*``.
    """
    alpha = _prior_default("LATTICE_PRIOR_ALPHA", "0.1") if alpha is None else alpha
    pi = _prior_default("LATTICE_PRIOR_PI", "0.97") if pi is None else pi
    mu_neg = _prior_default("LATTICE_PRIOR_MU_NEG", "1.0") if mu_neg is None else mu_neg
    student_t_df = _prior_default("LATTICE_PRIOR_DF", "3.0") if student_t_df is None else student_t_df
    device = binder_e.device
    dtype = binder_e.dtype
    b = binder_e.shape[0]

    # Per-(target, slot) Bernoulli draws for the mixture component.
    u = torch.rand(b, n_samples, generator=generator, device=device, dtype=dtype)
    is_binder = u < alpha
    is_neg = (u >= alpha) & (u < alpha + (1 - alpha) * pi)
    # Else: heavy-tail.

    # Gaussian negatives.
    gaussian = (
        torch.randn(b, n_samples, generator=generator, device=device, dtype=dtype)
        * sigma_neg
        + mu_neg
    )
    # Student-t heavy tail = Normal / sqrt(ChiSquared / df).
    z = torch.randn(b, n_samples, generator=generator, device=device, dtype=dtype)
    df = student_t_df
    chi2 = torch.distributions.Chi2(df).sample((b, n_samples)).to(device=device, dtype=dtype)
    student = student_t_loc + student_t_scale * z / (chi2 / df).sqrt()

    binder_delta = binder_e.detach().unsqueeze(1).expand(b, n_samples)
    out = torch.where(is_binder, binder_delta, torch.where(is_neg, gaussian, student))
    return out


class SinkhornEnergyLoss(nn.Module):
    """Sinkhorn divergence between the observed energies and the prior ``q*``.

    The "observed" point cloud per target is ``{e+, e-_1, …, e-_N}``; ``q*`` is
    sampled by :func:`sample_target_prior` at the same cardinality.
    """

    def __init__(
        self,
        *,
        prior_alpha: float | None = None,
        prior_pi: float | None = None,
        prior_mu_neg: float | None = None,
        prior_sigma_neg: float = 0.3,
        prior_student_t_df: float | None = None,
        prior_student_t_scale: float = 1.0,
        prior_student_t_loc: float = 0.0,
        blur: float = 0.05,
        n_iter: int = 50,
    ) -> None:
        super().__init__()
        self.prior_alpha = prior_alpha
        self.prior_pi = prior_pi
        self.prior_mu_neg = prior_mu_neg
        self.prior_sigma_neg = prior_sigma_neg
        self.prior_student_t_df = prior_student_t_df
        self.prior_student_t_scale = prior_student_t_scale
        self.prior_student_t_loc = prior_student_t_loc
        self.blur = blur
        self.n_iter = n_iter

    def forward(
        self, binder_e: torch.Tensor, decoy_e: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        all_e, _, _ = _flatten_pair(binder_e, decoy_e)
        # Detach the binder when seeding the prior so gradients to the prior
        # only flow through the structure of q*, not the live binder value.
        target_samples = sample_target_prior(
            n_samples=all_e.shape[1],
            binder_e=binder_e.detach(),
            alpha=self.prior_alpha,
            pi=self.prior_pi,
            mu_neg=self.prior_mu_neg,
            sigma_neg=self.prior_sigma_neg,
            student_t_df=self.prior_student_t_df,
            student_t_scale=self.prior_student_t_scale,
            student_t_loc=self.prior_student_t_loc,
        )
        div = sinkhorn_divergence_1d(
            all_e, target_samples, blur=self.blur, n_iter=self.n_iter
        )
        return div, {"sinkhorn/loss": float(div.detach().item())}


# --------------------------------------------------------------------------
# Cross-target margin
# --------------------------------------------------------------------------


def cross_target_margin_loss(
    binder_e: torch.Tensor,
    wrong_target_e: torch.Tensor,
    *,
    margin: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hinge: ``E(z_m+, z_p_wrong) > E(z_m+, z_p_correct) + margin``.

    Args:
        binder_e: ``[B]`` energies for the binder on its true target.
        wrong_target_e: ``[B]`` energies for the same binder on a random
            other target (sampled by the dataset / training loop).
        margin: separation we want between correct- and wrong-target energy.

    Returns:
        Scalar loss + a small dict of monitoring stats.
    """
    if binder_e.shape != wrong_target_e.shape or binder_e.ndim != 1:
        raise ValueError(
            f"shape mismatch: binder_e {tuple(binder_e.shape)} "
            f"vs wrong_target_e {tuple(wrong_target_e.shape)}"
        )
    delta = margin + binder_e - wrong_target_e          # want this <= 0
    loss = torch.clamp(delta, min=0).mean()
    with torch.no_grad():
        viol = (delta > 0).float().mean().item()
    return loss, {
        "cross_target/loss": float(loss.detach().item()),
        "cross_target/violation_rate": viol,
    }
