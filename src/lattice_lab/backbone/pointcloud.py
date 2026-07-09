"""Uni-Mol-style 3D point-cloud encoder.

Vendored from InVirtuoLabs/InVirtuoCLIP
(``src/invirtuo/models/encoders/pointcloud.py``) — a self-contained
re-implementation of the Uni-Mol molecule tower that needs no ``unicore``
runtime dependency. SE(3)-invariant: it consumes atom-type tokens plus a
pairwise-distance matrix (encoded through a Gaussian radial basis into an
attention bias), never raw coordinates, and returns the CLS (position 0)
representation ``[B, embed_dim]``.

``build_pointcloud_encoder`` is the Hydra ``_target_`` factory (mirrors
:func:`lattice_lab.backbone.discrete_flow.build_discrete_flow_encoder`): it sizes
the encoder from the atom dictionary, optionally loads pretrained Uni-Mol weights,
and stashes a ``build_config`` so checkpoints self-describe the skeleton.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Vendored unicore layer helpers
# ---------------------------------------------------------------------------

class LayerNorm(nn.LayerNorm):
    """Subclass for compatibility with unicore weight names."""
    pass


def _gelu(x):
    return F.gelu(x)


_ACTIVATION_REGISTRY = {
    "relu": F.relu,
    "gelu": _gelu,
    "tanh": torch.tanh,
}


def get_activation_fn(name: str):
    return _ACTIVATION_REGISTRY[name]


# ---------------------------------------------------------------------------
# TransformerEncoderLayer (simplified from unicore)
# ---------------------------------------------------------------------------

class _SelfAttn(nn.Module):
    """Packed QKV self-attention matching unicore checkpoint naming."""

    def __init__(self, embed_dim: int, attention_dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attention_dropout)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.head_dim = embed_dim // attention_heads

        self.self_attn = _SelfAttn(embed_dim, attention_dropout)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.dropout = nn.Dropout(dropout)

        self.self_attn_layer_norm = LayerNorm(embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim)
        self.post_ln = post_ln

    def forward(self, x, padding_mask=None, attn_bias=None, return_attn=False):
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)

        bsz, seq_len, _ = x.size()
        qkv = self.self_attn.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.attention_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            attn_logits = attn_logits + attn_bias.view(bsz, self.attention_heads, seq_len, seq_len)

        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.self_attn.attn_dropout(attn_probs)

        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        attn_out = self.self_attn.out_proj(attn_out)
        attn_out = self.dropout(attn_out)
        x = residual + attn_out

        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x

        if self.post_ln:
            x = self.final_layer_norm(x)

        # Return pre-softmax logits as updated pair representation
        out_attn = attn_logits.view(-1, seq_len, seq_len)
        return x, out_attn, None


# ---------------------------------------------------------------------------
# TransformerEncoderWithPair
# ---------------------------------------------------------------------------

def _encoder_layer_forward(layer, x, padding_mask, attn_mask):
    x, attn_mask, _ = layer(
        x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
    )
    return x, attn_mask


class TransformerEncoderWithPair(nn.Module):
    """Transformer encoder with pair representation (attention bias).

    Faithful re-implementation of the Uni-Mol architecture; layer names match the
    original so pretrained weights load by name.
    """

    def __init__(
        self,
        encoder_layers: int = 15,
        embed_dim: int = 512,
        ffn_embed_dim: int = 2048,
        attention_heads: int = 64,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 512,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(embed_dim)

        self.final_layer_norm = None if post_ln else LayerNorm(embed_dim)
        self.final_head_layer_norm = None if no_final_head_layer_norm else LayerNorm(attention_heads)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                attention_heads=attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                post_ln=post_ln,
            )
            for _ in range(encoder_layers)
        ])

    def forward(self, emb, attn_mask=None, padding_mask=None):
        bsz = emb.size(0)
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        if attn_mask is not None and padding_mask is not None:
            attn_mask = attn_mask.view(bsz, -1, seq_len, seq_len)
            attn_mask.masked_fill_(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_mask = attn_mask.view(-1, seq_len, seq_len)
            padding_mask = None

        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x, attn_mask = checkpoint(
                    _encoder_layer_forward,
                    layer,
                    x,
                    padding_mask,
                    attn_mask,
                    use_reentrant=False,
                )
            else:
                x, attn_mask, _ = layer(
                    x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
                )

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        if input_padding_mask is not None:
            delta_pair_repr_view = delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            delta_pair_repr_view.masked_fill_(
                input_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), 0
            )
            delta_pair_repr = delta_pair_repr_view.view(-1, seq_len, seq_len)

        attn_mask_out = attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        delta_pair_repr = delta_pair_repr.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, attn_mask_out, delta_pair_repr, None, None


# ---------------------------------------------------------------------------
# GaussianLayer
# ---------------------------------------------------------------------------

def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K: int = 128, edge_types: int = 1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


# ---------------------------------------------------------------------------
# PointCloudEncoder
# ---------------------------------------------------------------------------

class NonLinearHead(nn.Module):
    def __init__(self, input_dim, out_dim, activation_fn):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class PointCloudEncoder(nn.Module):
    """Self-contained Uni-Mol point-cloud encoder.

    Reads ``{key_prefix}_src_tokens`` ``[B, L]``, ``{key_prefix}_src_distance``
    ``[B, L, L]`` and ``{key_prefix}_src_edge_type`` ``[B, L, L]`` from a batch
    dict and returns the CLS representation ``[B, embed_dim]`` (position 0).
    """

    def __init__(
        self,
        vocab_size: int,
        key_prefix: str = "mol",
        encoder_layers: int = 15,
        encoder_embed_dim: int = 512,
        encoder_ffn_embed_dim: int = 2048,
        encoder_attention_heads: int = 64,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 512,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        gradient_checkpointing: bool = False,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.key_prefix = key_prefix
        self.output_dim = output_dim if output_dim is not None else encoder_embed_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.padding_idx = 0  # [PAD] is always index 0
        self.embed_tokens = nn.Embedding(vocab_size, encoder_embed_dim, self.padding_idx)

        K = 128
        n_edge_type = vocab_size * vocab_size
        self.gbf = GaussianLayer(K, n_edge_type)
        self.gbf_proj = NonLinearHead(K, encoder_attention_heads, activation_fn)

        self.encoder = TransformerEncoderWithPair(
            encoder_layers=encoder_layers,
            embed_dim=encoder_embed_dim,
            ffn_embed_dim=encoder_ffn_embed_dim,
            attention_heads=encoder_attention_heads,
            emb_dropout=emb_dropout,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            max_seq_len=max_seq_len,
            activation_fn=activation_fn,
            post_ln=post_ln,
            no_final_head_layer_norm=True,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """Encode and return the CLS representation ``[B, embed_dim]`` (position 0)."""
        p = self.key_prefix
        src_tokens = batch[f"{p}_src_tokens"]
        src_distance = batch[f"{p}_src_distance"]
        src_edge_type = batch[f"{p}_src_edge_type"]

        padding_mask = src_tokens.eq(self.padding_idx)
        x = self.embed_tokens(src_tokens)

        n_node = src_distance.size(-1)
        gbf_feature = self.gbf(src_distance, src_edge_type)
        gbf_result = self.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)

        encoder_rep, _, _, _, _ = self.encoder(
            x, padding_mask=padding_mask, attn_mask=graph_attn_bias
        )
        return encoder_rep[:, 0, :]

    @classmethod
    def from_pretrained_checkpoint(
        cls,
        checkpoint_path: str,
        vocab_size: int,
        sd_prefix: str = "",
        **kwargs,
    ) -> "PointCloudEncoder":
        """Load weights from a Uni-Mol/CLIP-style checkpoint.

        ``sd_prefix`` selects a tower from a joint checkpoint (e.g. ``"mol_model."``).
        """
        model = cls(vocab_size=vocab_size, **kwargs)
        state = torch.load(checkpoint_path, map_location="cpu")
        sd = state.get("model", state)
        if sd_prefix:
            sd = {k[len(sd_prefix):]: v for k, v in sd.items() if k.startswith(sd_prefix)}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[PointCloudEncoder] Missing keys: {missing}")
        if unexpected:
            print(f"[PointCloudEncoder] Unexpected keys: {unexpected}")
        return model


# Backward compatibility for checkpoints saved before the rename.
UniMolEncoder = PointCloudEncoder


# ---------------------------------------------------------------------------
# Hydra factory
# ---------------------------------------------------------------------------

def build_pointcloud_encoder(
    *,
    dict_path: str,
    ckpt_path: Optional[str] = None,
    sd_prefix: str = "mol_model.",
    key_prefix: str = "mol",
    encoder_layers: int = 15,
    encoder_embed_dim: int = 512,
    encoder_ffn_embed_dim: int = 2048,
    encoder_attention_heads: int = 64,
    max_seq_len: int = 512,
    gradient_checkpointing: bool = False,
    device: str | torch.device = "cpu",
) -> PointCloudEncoder:
    """Hydra entrypoint: size a :class:`PointCloudEncoder` from the atom dictionary
    and optionally load pretrained Uni-Mol weights.

    ``dict_path`` is the atom-type vocab (``dict_mol.txt``); its length (incl. the
    ``[PAD]/[CLS]/[SEP]/[UNK]/[MASK]`` specials) sets ``vocab_size``. When
    ``ckpt_path`` is given, the ``sd_prefix`` tower is loaded (non-strict).
    """
    from lattice_lab.data.conformers import Dictionary  # lazy: avoid import cycle

    d = Path(dict_path)
    if not d.is_file():
        raise FileNotFoundError(f"dict_path={dict_path!r} is not a file")
    vocab_size = len(Dictionary.load(str(d)))

    kwargs = dict(
        key_prefix=key_prefix,
        encoder_layers=encoder_layers,
        encoder_embed_dim=encoder_embed_dim,
        encoder_ffn_embed_dim=encoder_ffn_embed_dim,
        encoder_attention_heads=encoder_attention_heads,
        max_seq_len=max_seq_len,
        gradient_checkpointing=gradient_checkpointing,
    )
    if ckpt_path:
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"ckpt_path={ckpt_path!r} is not a file")
        enc = PointCloudEncoder.from_pretrained_checkpoint(
            ckpt_path, vocab_size=vocab_size, sd_prefix=sd_prefix, **kwargs
        )
    else:
        enc = PointCloudEncoder(vocab_size=vocab_size, **kwargs)
    enc.to(device)
    # Self-describing skeleton for checkpoint reconstruction (mirrors the DDiT encoder).
    enc.build_config = {
        "dict_path": str(dict_path),
        "vocab_size": int(vocab_size),
        "key_prefix": key_prefix,
        "encoder_layers": int(encoder_layers),
        "encoder_embed_dim": int(encoder_embed_dim),
        "encoder_ffn_embed_dim": int(encoder_ffn_embed_dim),
        "encoder_attention_heads": int(encoder_attention_heads),
        "max_seq_len": int(max_seq_len),
    }
    return enc
