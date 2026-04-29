"""
Models for brain tumour classification.

PoPEViT  — proposed model: ViT with Polar Positional Embeddings (PoPE)
RoPEViT  — ViT baseline:   ViT with Rotary Positional Embeddings (RoPE)

All models share the same input/output contract:
    Input:  float32 tensor (batch, 3, image_size, image_size), values in [0, 1]
            after ImageNet normalisation.
    Output: raw logits (batch, num_classes)

Usage:
    from model import PoPEViT, RoPEViT
    model = PoPEViT(image_size=224, patch_size=16, num_classes=4,
                    dim=512, depth=6, heads=8, mlp_dim=1024, dropout=0.1)
    model = RoPEViT(image_size=224, patch_size=16, num_classes=4,
                    dim=512, depth=6, heads=8, mlp_dim=1024, dropout=0.1)
"""

import torch
from torch import nn
import torch.nn.functional as F
from PoPE_pytorch import PoPE
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ── Shared feed-forward block ─────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── PoPE attention & transformer ──────────────────────────────────────────────

class PoPEAttention(nn.Module):
    """
    Multi-head self-attention where position is encoded via PoPE.

    PoPE converts Q and K into polar form: each value becomes a magnitude (via
    softplus) rotated by a frequency-scaled position angle (cos/sin). Keys also
    receive a learned per-head phase bias, separating *what* a token is from
    *where* it is.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.pope = PoPE(dim=dim_head, heads=heads)

    def forward(self, x):
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        q, k = PoPE.apply_pope_to_qk(self.pope(x.shape[1]), q, k)
        attn = self.dropout(self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        return self.to_out(rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)'))


class PoPETransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = PoPEAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class PoPEViT(nn.Module):
    """
    Vision Transformer with PoPE positional embeddings (proposed model).

    Uses coarse absolute position embeddings before the transformer plus fine-
    grained relative position via PoPE inside each attention layer.
    """
    def __init__(self, image_size=224, patch_size=16, num_classes=4,
                 dim=512, depth=6, heads=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(3 * patch_size ** 2, dim),
        )
        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout       = nn.Dropout(dropout)
        dim_head = dim // heads
        self.transformer = nn.ModuleList([
            PoPETransformerBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm     = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.patch_embed(img)
        x = torch.cat((repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0]), x), dim=1)
        x = self.dropout(x + self.pos_embedding)
        for block in self.transformer:
            x = block(x)
        return self.mlp_head(self.norm(x).mean(dim=1))


# ── RoPE attention & transformer ──────────────────────────────────────────────

class RoPEAttention(nn.Module):
    """
    Multi-head self-attention with 1-D Rotary Position Embeddings (RoPE).

    Q and K are rotated by position-dependent angles before the dot product,
    making attention scores depend only on relative positions. No absolute
    position embedding is needed alongside RoPE.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        half = dim_head // 2
        theta = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.rope_theta: torch.Tensor
        self.register_buffer('rope_theta', theta)

    def _apply_rope(self, x):
        # x: (b, h, n, d)  — rotate each (x[...,2i], x[...,2i+1]) pair by pos * theta_i
        n, d = x.shape[-2], x.shape[-1]
        half = d // 2
        pos = torch.arange(n, device=x.device, dtype=x.dtype)
        angles = pos[:, None] * self.rope_theta[None, :]          # (n, half)
        cos = angles.cos()[None, None]                             # (1, 1, n, half)
        sin = angles.sin()[None, None]
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x):
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        q, k = self._apply_rope(q), self._apply_rope(k)
        attn = self.dropout(self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        return self.to_out(rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)'))


class RoPETransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = RoPEAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class RoPEViT(nn.Module):
    """
    Vision Transformer with RoPE positional embeddings (ViT baseline).

    No absolute position embedding is added — RoPE inside each attention layer
    encodes relative positions directly, making the two approaches directly
    comparable as ablations of the positional encoding strategy.
    """
    def __init__(self, image_size=224, patch_size=16, num_classes=4,
                 dim=512, depth=6, heads=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(3 * patch_size ** 2, dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout   = nn.Dropout(dropout)
        dim_head = dim // heads
        self.transformer = nn.ModuleList([
            RoPETransformerBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm     = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.patch_embed(img)
        x = torch.cat((repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0]), x), dim=1)
        x = self.dropout(x)
        for block in self.transformer:
            x = block(x)
        return self.mlp_head(self.norm(x).mean(dim=1))
