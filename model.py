"""
PoPEViT — Vision Transformer with PoPE positional embeddings (proposed model).

INTEGRATION GUIDE
-----------------
Instantiate the model:

    from model import PoPEViT
    model = PoPEViT(image_size=256, patch_size=16, num_classes=4,
                    dim=512, depth=6, heads=8, mlp_dim=1024, dropout=0.1)

Input:  float32 tensor of shape (batch, 3, 256, 256), pixel values in [0, 1].
        This matches BrainTumorDataset output after v2.ToDtype(torch.float32, scale=True).
        Images must be 3-channel RGB — add v2.RGB() to transforms if grayscale inputs
        are possible.

Output: raw logits of shape (batch, 4). Class order matches BrainTumorDataset:
            0 = no_tumor
            1 = meningioma_tumor
            2 = glioma_tumor
            3 = pituitary_tumor

Training loss:
    loss = F.cross_entropy(model(images), labels)   # takes logits directly

Inference / AUROC:
    probs = F.softmax(model(images), dim=-1)        # convert to per-class probabilities
    # probs shape: (batch, 4) — pass to sklearn.metrics.roc_auc_score

Baseline (standard ViT with learned positional embeddings, no PoPE):
    from vit_pytorch import ViT
    baseline = ViT(image_size=256, patch_size=16, num_classes=4,
                   dim=512, depth=6, heads=8, mlp_dim=1024)
    # same input/output contract as PoPEViT
"""

import torch
from torch import nn
import torch.nn.functional as F
from PoPE_pytorch import PoPE
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PoPEAttention(nn.Module):
    """
    Multi-head self-attention where position is encoded via PoPE instead of RoPE.

    PoPE converts Q and K into polar form: each value becomes a magnitude (via softplus)
    rotated by a frequency-scaled position angle (cos/sin). Keys additionally receive a
    learned per-head phase bias, so queries encode pure content magnitude while keys encode
    position-shifted magnitude. This explicitly separates *what* a token is from *where* it
    is, which is useful when both location and appearance carry independent diagnostic signal.
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
        """Projects x to Q/K/V, applies PoPE to Q and K, then computes scaled dot-product attention."""
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        q, k = PoPE.apply_pope_to_qk(self.pope(x.shape[1]), q, k)
        attn = self.dropout(self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        return self.to_out(rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)'))


class PoPETransformerBlock(nn.Module):
    """
    Single Pre-Norm transformer block: PoPE attention followed by a feed-forward MLP,
    each wrapped in a residual connection with layer normalisation applied before the
    sub-layer (Pre-Norm). Pre-Norm is used over Post-Norm for more stable training at
    larger depths.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = PoPEAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PoPEViT(nn.Module):
    """
    Full Vision Transformer using PoPE attention throughout.

    Images are split into fixed-size patches and linearly projected to token embeddings.
    A learnable CLS token is prepended, and coarse absolute position embeddings are added
    before the transformer. Fine-grained relative position is then handled by PoPE inside
    each attention layer — the two forms are complementary, not redundant.
    After all transformer blocks, tokens are averaged and mapped to class logits by a
    single linear layer.
    """
    def __init__(self, image_size=256, patch_size=16, num_classes=4,
                 dim=512, depth=6, heads=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(3 * patch_size ** 2, dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        dim_head = dim // heads
        self.transformer = nn.ModuleList([
            PoPETransformerBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.patch_embed(img)
        x = torch.cat((repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0]), x), dim=1)
        x = self.dropout(x + self.pos_embedding)
        for block in self.transformer:
            x = block(x)
        return self.mlp_head(self.norm(x).mean(dim=1))
