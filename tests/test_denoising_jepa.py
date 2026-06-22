"""Tests for the conditional denoising-JEPA objective."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from lattice_lab.training.denoising_jepa import (
    AttentionPool,
    DenoisingJEPAModel,
    VAEHead,
    build_jepa,
    condition_bypass_gap,
    corrupt_tokens,
    denoising_loss,
    effective_rank,
    encode_pooled_latent,
    reconstruction_loss,
    train_step,
)

torch.manual_seed(0)

# Tiny model + batch dims (fast on CPU).
VOCAB = 16
HIDDEN = 32
HEADS = 4
LAYERS = 2
POOL_HEADS = 4
PAD_ID = 3
TOKEN_ID_MIN = 4
B, L, D = 8, 7, HIDDEN


def _tiny_jepa() -> DenoisingJEPAModel:
    return build_jepa(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        n_heads=HEADS,
        n_layer=LAYERS,
        pool_heads=POOL_HEADS,
        dropout=0.0,
        encode_time=1.0,
        pad_id=PAD_ID,
        token_id_min=TOKEN_ID_MIN,
    )


def _tiny_jepa_with_vae() -> DenoisingJEPAModel:
    model = _tiny_jepa()
    model.vae_head = VAEHead(D, free_bits=0.0)
    return model


def _tiny_batch(*, seed: int = 0, with_pad: bool = True):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(TOKEN_ID_MIN, VOCAB, (B, L), generator=g)
    mask = torch.ones(B, L)
    if with_pad:
        ids[:, -2:] = PAD_ID
        mask[:, -2:] = 0.0
    return ids, mask


# --------------------------------------------------------------------------- #
# AttentionPool (the z_s encoder head)
# --------------------------------------------------------------------------- #
def test_attention_pool_output_shape() -> None:
    pool = AttentionPool(D, num_heads=POOL_HEADS).eval()
    assert pool(torch.randn(B, L, D)).shape == (B, D)


def test_attention_pool_permutation_invariance() -> None:
    pool = AttentionPool(D, num_heads=POOL_HEADS).eval()
    tokens = torch.randn(B, L, D)
    perm = torch.randperm(L)
    assert torch.allclose(pool(tokens), pool(tokens[:, perm, :]), atol=1e-5)


def test_attention_pool_respects_padding() -> None:
    pool = AttentionPool(D, num_heads=POOL_HEADS).eval()
    tokens = torch.randn(B, L, D)
    kpm = torch.zeros(B, L, dtype=torch.bool)
    kpm[:, -2:] = True
    out = pool(tokens, key_padding_mask=kpm)
    tampered = tokens.clone()
    tampered[:, -2:, :] = torch.randn(B, 2, D)
    assert torch.allclose(out, pool(tampered, key_padding_mask=kpm), atol=1e-5)


# --------------------------------------------------------------------------- #
# Corruption + noised-position mask
# --------------------------------------------------------------------------- #
def test_corrupt_returns_noised_mask_within_valid() -> None:
    ids, mask = _tiny_batch()
    x_t, t, noised = corrupt_tokens(
        ids, mask, vocab_size=VOCAB, token_id_min=TOKEN_ID_MIN, pad_id=PAD_ID, t=0.2
    )
    assert x_t.shape == ids.shape and noised.shape == ids.shape
    # No padding position is ever marked noised.
    assert not noised[mask == 0].any()
    # Pad positions remain PAD.
    assert (x_t[mask == 0] == PAD_ID).all()


def test_corrupt_time_controls_noise_level() -> None:
    ids, mask = _tiny_batch()
    _, _, hi_noise = corrupt_tokens(
        ids, mask, vocab_size=VOCAB, token_id_min=TOKEN_ID_MIN, pad_id=PAD_ID, t=0.05
    )
    _, _, clean = corrupt_tokens(
        ids, mask, vocab_size=VOCAB, token_id_min=TOKEN_ID_MIN, pad_id=PAD_ID, t=1.0
    )
    valid = mask.bool().sum()
    assert hi_noise.float().sum() / valid > 0.7   # t≈0 → almost all noised
    assert clean.float().sum() == 0               # t=1 → nothing noised


def test_uniform_time_range_sampled_per_sample() -> None:
    ids, mask = _tiny_batch()
    _, t, _ = corrupt_tokens(
        ids, mask, vocab_size=VOCAB, token_id_min=TOKEN_ID_MIN, pad_id=PAD_ID,
        t=(0.1, 0.6),
    )
    assert t.shape == (B,)
    assert float(t.min()) >= 0.1 and float(t.max()) < 0.6


# --------------------------------------------------------------------------- #
# reconstruction_loss (pure generative CE at the noised positions)
# --------------------------------------------------------------------------- #
def test_reconstruction_loss_only_counts_noised_positions() -> None:
    logits = torch.zeros(B, L, VOCAB)
    clean = torch.zeros(B, L, dtype=torch.long)
    noised = torch.zeros(B, L, dtype=torch.bool)
    # No noised positions → loss is exactly 0 (no terms).
    loss, _ = reconstruction_loss(logits, clean, noised)
    assert float(loss) == 0.0
    # A single noised position contributes uniform-logit CE = log(VOCAB).
    noised[0, 0] = True
    loss, _ = reconstruction_loss(logits, clean, noised)
    assert math.isclose(float(loss), math.log(VOCAB), rel_tol=1e-5)


# --------------------------------------------------------------------------- #
# denoising_loss / gradient flow (no teacher, no stop-grad)
# --------------------------------------------------------------------------- #
def test_optimizer_over_model_params_only() -> None:
    model = _tiny_jepa()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model_ids = {id(p) for p in model.parameters()}
    opt_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert opt_ids == model_ids


def test_gradient_reaches_encoder_pool_and_denoiser() -> None:
    model = _tiny_jepa()

    def _has_grad(module) -> bool:
        return any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )

    # The conditional denoiser receives gradient immediately.
    loss, _ = denoising_loss(model, _tiny_batch(), corrupt_t=0.1)
    loss.backward()
    assert _has_grad(model.denoiser)

    # DDiT's adaLN conditioning is zero-initialized, so the gradient w.r.t. the
    # conditioning vector (hence z_s and the encoder pool) is exactly 0 on a
    # fresh model — it only flows once adaLN moves off zero. A few steps warm it
    # up; then z_s is trained through reconstruction.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(10):
        train_step(model, opt, _tiny_batch(), corrupt_t=0.1)
    model.zero_grad(set_to_none=True)
    loss, _ = denoising_loss(model, _tiny_batch(), corrupt_t=0.1)
    loss.backward()
    assert _has_grad(model.encoder.pool)


def test_alignment_zero_for_identical_views() -> None:
    """Two identical views → cos(z_s, z_s_b) = 1 → align term is 0."""
    model = _tiny_jepa()
    b = _tiny_batch()
    loss, m = denoising_loss(model, b, batch_b=b, align_lambda=0.5, corrupt_t=0.1)
    assert "align" in m and "align_cos" in m
    assert m["align"] == pytest.approx(0.0, abs=1e-5)
    assert m["align_cos"] == pytest.approx(1.0, abs=1e-5)


def test_alignment_vae_both_views_post_vae() -> None:
    """With VAE, z_b must go through the same head as z_s (μ at eval)."""
    model = _tiny_jepa_with_vae()
    model.eval()
    b = _tiny_batch()
    _, m = denoising_loss(model, b, batch_b=b, align_lambda=0.5, corrupt_t=0.1)
    assert m["align"] == pytest.approx(0.0, abs=1e-5)


def test_encode_pooled_latent_applies_vae_mu_at_eval() -> None:
    model = _tiny_jepa_with_vae()
    model.eval()
    ids, mask = _tiny_batch()
    z_raw = model.encoder(ids, mask)
    z = encode_pooled_latent(model, ids, mask, training=False)
    assert not torch.allclose(z_raw, z)
    assert torch.allclose(z, model.vae_head.mu(z_raw))


def test_denoising_loss_reports_logvar_when_vae() -> None:
    model = _tiny_jepa_with_vae()
    _, m = denoising_loss(model, _tiny_batch(), corrupt_t=0.2)
    assert "logvar_mean" in m and "logvar_std" in m and "logvar_exp_mean" in m


def test_latent_consistency_when_enabled() -> None:
    model = _tiny_jepa()
    _, metrics, _, terms = denoising_loss(
        model,
        _tiny_batch(),
        corrupt_t=0.2,
        latent_consistency_lambda=0.5,
        return_outputs=True,
    )
    assert "latent" in metrics and "latent_cos" in metrics
    assert metrics["latent"] >= 0.0
    assert "latent" in terms


def test_train_step_returns_finite_metrics() -> None:
    model = _tiny_jepa()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_step(model, opt, _tiny_batch(), corrupt_t=(0.1, 0.6))
    assert set(metrics) >= {"loss", "recon", "recon_acc", "rank_s", "noised_frac"}
    assert all(math.isfinite(v) for v in metrics.values())


def test_loss_is_zero_when_clean() -> None:
    """t=1 → nothing noised → the reconstruction loss has no terms (returns 0)."""
    model = _tiny_jepa()
    loss, metrics = denoising_loss(model, _tiny_batch(), corrupt_t=1.0)
    assert metrics["noised_frac"] == 0.0
    assert float(loss) == 0.0


def test_denoising_loss_returns_weighted_terms() -> None:
    model = _tiny_jepa()
    _, _, _, terms = denoising_loss(
        model,
        _tiny_batch(),
        batch_b=_tiny_batch(seed=1),
        align_lambda=0.5,
        corrupt_t=0.3,
        return_outputs=True,
    )
    assert set(terms) == {"recon", "align"}
    assert all(t.requires_grad for t in terms.values())


def test_log_grad_norms_logs_active_terms() -> None:
    import lightning as L

    from lattice_lab.models.denoising_jepa_ssl import DenoisingJEPAModule

    model = _tiny_jepa()
    harness = DenoisingJEPAModule.__new__(DenoisingJEPAModule)
    L.LightningModule.__init__(harness)
    harness.encoder = model.encoder
    harness.denoiser = model.denoiser
    harness.pad_id = model.pad_id
    harness.vocab_size = model.vocab_size
    harness.token_id_min = model.token_id_min
    harness.vae_head = None
    _, _, _, terms = denoising_loss(
        model, _tiny_batch(), corrupt_t=0.3, return_outputs=True, align_lambda=0.0
    )
    logged: dict[str, float] = {}
    harness.log_dict = lambda d, **kw: logged.update(d)  # type: ignore[method-assign]
    DenoisingJEPAModule._log_grad_norms(harness, **terms)
    assert set(logged) == {"train/grad_norm_recon"}
    assert math.isfinite(logged["train/grad_norm_recon"]) and logged["train/grad_norm_recon"] > 0.0


def test_loss_is_zero_when_clean() -> None:
    """t=1 → nothing noised → the reconstruction loss has no terms (returns 0)."""
    model = _tiny_jepa()
    loss, metrics = denoising_loss(model, _tiny_batch(), corrupt_t=1.0)
    assert metrics["noised_frac"] == 0.0
    assert float(loss) == 0.0


# --------------------------------------------------------------------------- #
# condition_bypass_gap
# --------------------------------------------------------------------------- #
def test_condition_bypass_gap_is_nonzero_after_training() -> None:
    """After fitting a tiny batch the denoiser should rely on z_s (positive gap)."""
    model = _tiny_jepa()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    batch = _tiny_batch()
    for _ in range(60):
        train_step(model, opt, batch, corrupt_t=0.1)
    stats = condition_bypass_gap(model, batch, corrupt_t=0.1)
    assert set(stats) == {"recon_real", "recon_zeroed", "gap"}
    assert all(math.isfinite(v) for v in stats.values())
    # Zeroing z_s should not *help* reconstruction (gap >= ~0); after overfitting
    # a fixed batch it should be strictly positive.
    assert stats["gap"] > 0.0


# --------------------------------------------------------------------------- #
# effective_rank
# --------------------------------------------------------------------------- #
def test_effective_rank_collapsed_is_low() -> None:
    z = torch.ones(B, D)
    z[:, 0] = torch.linspace(0, 1, B)
    assert effective_rank(z) < 2.0


def test_effective_rank_full_is_high() -> None:
    assert effective_rank(torch.randn(64, D)) > 5.0


# --------------------------------------------------------------------------- #
# Smoke + informativeness guard
# --------------------------------------------------------------------------- #
def test_smoke_single_train_step() -> None:
    model = _tiny_jepa()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_step(model, opt, _tiny_batch(), corrupt_t=(0.1, 0.6))
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["rank_s"])


def test_rank_does_not_collapse_over_50_steps() -> None:
    torch.manual_seed(0)  # order-independent: don't inherit prior tests' RNG state
    model = _tiny_jepa()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ranks = []
    for step in range(50):
        m = train_step(model, opt, _tiny_batch(seed=step), corrupt_t=(0.1, 0.6))
        ranks.append(m["rank_s"])
    assert all(math.isfinite(r) for r in ranks)
    assert ranks[-1] > 1.5  # crude guard: z_s stays informative


# --------------------------------------------------------------------------- #
# LightningModule wiring (drops into the repo's Hydra/Lightning flow)
# --------------------------------------------------------------------------- #
TOKENIZER = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer" / "smiles_new.json"
requires_tokenizer = pytest.mark.skipif(
    not TOKENIZER.is_file(), reason=f"tokenizer not found at {TOKENIZER}"
)


@requires_tokenizer
def test_lightning_module_fast_dev_run() -> None:
    import lightning as L
    from torch.utils.data import DataLoader

    from lattice_lab.models.denoising_jepa_ssl import DenoisingJEPAModule

    module = DenoisingJEPAModule(
        ckpt_path=None,
        tokenizer_path=str(TOKENIZER),
        freeze_backbone=False,
        encode_time=1.0,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
        corrupt_t=(0.1, 0.6),
        warmup_steps=1,
        total_steps=10,
        train_rank_every_n_steps=1,
    )
    p0 = next(p for p in module.denoiser.parameters() if p.requires_grad).detach().clone()

    views = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC", "OCC", "C1CCCCC1", "CCl"]
    loader = DataLoader(views, batch_size=4, collate_fn=list)
    trainer = L.Trainer(
        fast_dev_run=2, accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=loader)

    p1 = next(p for p in module.denoiser.parameters() if p.requires_grad).detach().clone()
    assert not torch.equal(p0, p1)  # denoiser trained


def test_split_batch_keeps_smiles() -> None:
    """`(views, smiles)` batches keep SMILES (for the Tanimoto FP target); a bare
    list of views yields `smiles=None`."""
    from lattice_lab.models.denoising_jepa_ssl import DenoisingJEPAModule

    views, smiles = DenoisingJEPAModule._split_batch((["CCO", "CCN"], ["CCO", "CCN"]))
    assert views == ["CCO", "CCN"] and smiles == ["CCO", "CCN"]
    views, smiles = DenoisingJEPAModule._split_batch(["CCO", "CCN"])
    assert views == ["CCO", "CCN"] and smiles is None


@requires_tokenizer
def test_fp_distillation_is_finite_and_feeds_pool() -> None:
    """The Tanimoto distillation term is finite, off when disabled, and (with a
    monkeypatched FP cache, so no rdkit needed) routes gradient into the pool."""
    import numpy as np

    from lattice_lab.models.denoising_jepa_ssl import DenoisingJEPAModule

    module = DenoisingJEPAModule(
        ckpt_path=None,
        tokenizer_path=str(TOKENIZER),
        freeze_backbone=False,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
        fp_weight=1.0,
        fp_bits=64,
    )

    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
    ids, mask = module._tokenize(smiles)

    off = DenoisingJEPAModule(
        ckpt_path=None,
        tokenizer_path=str(TOKENIZER),
        freeze_backbone=False,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
        fp_weight=0.0,
    )
    assert off._fp_distillation(module.encoder(ids, mask), smiles) is None

    # Deterministic fake fingerprints (distinct per row) so Tanimoto is non-trivial.
    rng = np.random.default_rng(0)
    bits = (rng.random((len(smiles), 64)) < 0.3).astype(np.float32)

    class _FakeCache:
        def bits(self, s):
            return bits

    module._fp_cache = _FakeCache()
    z_s = module.encoder(ids, mask)
    fp = module._fp_distillation(z_s, smiles)
    assert fp is not None and math.isfinite(float(fp)) and float(fp) >= 0.0

    fp.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in module.encoder.pool.parameters()
    )


@requires_tokenizer
def test_lightning_checkpoint_embeds_config() -> None:
    from lattice_lab.models.denoising_jepa_ssl import DenoisingJEPAModule

    module = DenoisingJEPAModule(
        ckpt_path=None,
        tokenizer_path=str(TOKENIZER),
        freeze_backbone=False,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
    )
    ckpt: dict = {}
    module.on_save_checkpoint(ckpt)
    assert ckpt["encoder_config"]["n_embd"] == 32
