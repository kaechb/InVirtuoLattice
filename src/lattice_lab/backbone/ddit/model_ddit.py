import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from . import rotary


class AttentionPoolingHead(nn.Module):
    def __init__(self, hidden_size, num_classes, linear_hidden_size=None, num_heads=8, dropout=0.1):
        super().__init__()

        # --- FIX 1: Prevent Softmax Saturation ---
        # OLD: self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size)) (Norm ~27 -> Dead Gradients)
        # NEW: Small variance initialization (Norm ~1 -> Healthy Gradients)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(hidden_size)
        self.norm_input = LayerNorm(hidden_size)
        # --- FIX 2: Break Symmetry + Safe Start ---
        linear_hidden_size = linear_hidden_size if linear_hidden_size is not None else hidden_size
        self.mlp = nn.Sequential(nn.Linear(hidden_size, linear_hidden_size), nn.GELU(), nn.Linear(linear_hidden_size, num_classes))
        self.length = nn.Linear(1, hidden_size)
        # Init Hidden Layer (Random to break symmetry)
        nn.init.kaiming_normal_(self.mlp[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.mlp[0].bias)

        # Init Output Layer (Zero to prevent Loss Shock)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x, token_mask, return_embedding=False):
        batch_size = x.shape[0]
        query = self.cls_token.expand(batch_size, -1, -1)
        length = self.length(token_mask.sum(dim=1).unsqueeze(1)/50-1)
        # Correct Masking Logic
        padding_mask = ~token_mask if token_mask is not None else None
        attn_out, _ = self.attention(query=query, key=self.norm_input(x), value=self.norm_input(x), key_padding_mask=padding_mask)

        pooled = (query + attn_out).squeeze(1)
        logits = self.mlp(self.norm(pooled + length))
        if return_embedding:
            return logits, attn_out.squeeze(1)
        return logits



def bias_dropout_add_scale(x: Tensor, scale: Tensor, residual: Optional[Tensor], prob: float, training: bool) -> Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])

        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(time: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(time)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_conds: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        gated: bool = False,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be devisable by n_heads"

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout
        self.gated = gated

        self.head_dim = self.dim // self.n_heads

        self.norm1 = LayerNorm(dim=dim)

        self.qw = nn.Linear(dim, dim, bias=False)
        self.kw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)

        if self.gated:
            self.gw = nn.Linear(dim, dim, bias=False)

        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        self.adaLN_modulation = nn.Linear(n_conds, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, rotary_cos_sin: Tensor, c: Tensor, attn_mask: Tensor) -> Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c)[
            :, None
        ].chunk(6, dim=2)

        x_skip = x
        x = modulate(x=self.norm1(x), shift=shift_msa, scale=scale_msa)

        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        # <--- Conditional Gate Calculation
        g = None
        if self.gated:
            g = self.gw(x)
            g = g.view(batch_size, seq_len, self.n_heads, self.head_dim)
            g = g.transpose(1, 2)
        # ---------------------------------

        q, k, v = (item.view(batch_size, seq_len, self.n_heads, self.head_dim) for item in (q, k, v))

        # with torch.amp.autocast("cuda", enabled=False):
        cos, sin = rotary_cos_sin
        original_dtype = q.dtype

        q = rotary.apply_rotary_emb_torch(x=q, cos=cos, sin=sin).to(original_dtype)
        k = rotary.apply_rotary_emb_torch(x=k, cos=cos, sin=sin).to(original_dtype)

        q, k, v = (item.transpose(1, 2) for item in (q, k, v))

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_mask)

        # <--- Conditional Gating Application
        if self.gated and g is not None:
            x = x * F.silu(g)
        # -----------------------------------

        x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)
        x = bias_dropout_add_scale(
            x=self.attn_out(x),
            scale=gate_msa,
            residual=x_skip,
            prob=self.dropout,
            training=self.training,
        )
        x = bias_dropout_add_scale(
            x=self.mlp(modulate(x=self.norm2(x), shift=shift_mlp, scale=scale_mlp)),
            scale=gate_mlp,
            residual=x,
            prob=self.dropout,
            training=self.training,
        )

        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, n_conds: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(n_conds, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(x=self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)

        return x


class DDiT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        t_emb_dim: int = 128,
        n_heads: int = 12,
        n_layer: int = 12,
        dropout: float = 0.1,
        num_classes: int = 0,
        n_conds: int = 0,
        n_descriptors: int = 0,
        gated: bool = False,
        linear_hidden_size: int = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_descriptors = n_descriptors
        self.vocab_embed = nn.Embedding(self.vocab_size, hidden_size)
        if n_descriptors > 0:

            self.aux_head = AttentionPoolingHead(hidden_size=hidden_size, num_classes=n_descriptors, linear_hidden_size=linear_hidden_size)
        else:
            self.aux_head = None
        self.time_embedding = TimestepEmbedder(hidden_size=t_emb_dim)
        self.rotary_emb = rotary.Rotary(dim=hidden_size // n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=hidden_size,
                    n_heads=n_heads,
                    n_conds=t_emb_dim,
                    dropout=dropout,
                    gated=gated,
                )
                for _ in range(n_layer)
            ]
        )

        self.output_layer = DDitFinalLayer(
            hidden_size=hidden_size,
            out_channels=vocab_size,
            n_conds=t_emb_dim,
        )
        if n_conds > 0:
            self.conds = nn.Sequential(
                nn.Linear(n_conds, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 2 * hidden_size),
            )
            # FiLM γ/β start at 0 → h = h * 1 + 0; denoiser can ignore z_s at init.
            last = self.conds[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            self.conds = None

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        attn_mask: Tensor,
        conds: Optional[Tensor] = None,
        return_hidden=False,
        token_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        classification: bool = False,
        return_aux_embedding: bool = False,
    ) -> Tensor:
        # attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        # with torch.amp.autocast("cuda", enabled=False):
        # Embed the input token IDs
        if token_mask is not None and x is not None:
            assert (x[token_mask] != 3).all(), "Descriptor token should not be masked"
        if inputs_embeds is not None:
            x = inputs_embeds
        elif x is not None:
            x = self.vocab_embed(x)  # Shape: (batch_size, seq_len, hidden_size)
        c = F.silu(self.time_embedding(time=t))
        if conds is not None:
            if self.conds is None:
                raise ValueError("conds passed but model has n_conds=0")
            if conds.ndim == 1:
                conds = conds.unsqueeze(0)
            gamma, beta = self.conds(conds[: x.size(0)]).chunk(2, dim=-1)
            x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        else:
            assert self.conds is None

        # Get rotary embeddings
        rotary_cos_sin = self.rotary_emb(x=x)

        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                hidden_states = x.clone()
            x = block(x=x, rotary_cos_sin=rotary_cos_sin, c=c, attn_mask=attn_mask)

        logits = self.output_layer(x=x, c=c)
        if self.aux_head is not None and classification:
            if return_aux_embedding:
                aux_logits, aux_emb = self.aux_head(x, token_mask=token_mask, return_embedding=True)
                return logits, aux_logits, aux_emb
            return logits, self.aux_head(x, token_mask=token_mask)
        if not return_hidden:
            return logits
        else:
            return logits, hidden_states
