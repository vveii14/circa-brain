"""
3DSC-TF image encoder: 3D Depthwise Separable Convolution + Transformer.

Adapted from: https://github.com/NWPU-903PR/3DSC-TF
Paper: Classification of Alzheimer's Disease by Jointing 3D Depthwise
       Separable Convolutional Neural Network and Transformer.

Input: (B, 1, D, H, W) full 3D volume.
Output: (B, feat_dim) feature vector.

The volume is split into non-overlapping cubic patches. Each patch is
independently encoded by a lightweight 3D depthwise-separable CNN (DSC).
The resulting patch tokens are fed into a standard Transformer encoder
(with CLS token + positional embedding) to produce the final feature.
"""

from functools import partial
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Tuple, Union, Optional


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# DSC: 3D Depthwise Separable Conv block — processes a single 3D patch
# ---------------------------------------------------------------------------
class DSC(nn.Module):
    """
    Encode one 3D patch into a vector of length `embed_dim`.
    Conv3d -> DW-Sep Conv3d (with residual) -> AdaptiveAvgPool -> flatten.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, embed_dim, 3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim, embed_dim, 1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.dw_sep = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, 3, groups=embed_dim, padding=1),
            nn.Conv3d(embed_dim, embed_dim, 1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim, embed_dim, 3, groups=embed_dim, padding=1),
            nn.Conv3d(embed_dim, embed_dim, 1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, ps, ps, ps)
        h = self.stem(x)
        h = h + self.dw_sep(h)  # residual
        h = self.pool(h)
        return h.flatten(1)  # (B, embed_dim)


# ---------------------------------------------------------------------------
# Shared DSC: single DSC applied to all patches (parameter efficient)
# ---------------------------------------------------------------------------
class SharedPatchEmbed(nn.Module):
    """
    Extract non-overlapping patches from the volume, encode each with a
    *shared* DSC module.  Patches are processed in chunks to limit peak
    GPU memory (important when many patches share a single GPU).
    """

    def __init__(self, img_size: int, patch_size: int, embed_dim: int,
                 patch_chunk: int = 32, use_grad_ckpt: bool = True):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 3
        self.patch_chunk = patch_chunk
        self.use_grad_ckpt = use_grad_ckpt
        self.dsc = DSC(embed_dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def _run_dsc(self, x: torch.Tensor) -> torch.Tensor:
        return self.dsc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W)
        B = x.size(0)
        ps = self.patch_size
        N = self.num_patches
        x = x.unfold(2, ps, ps).unfold(3, ps, ps).unfold(4, ps, ps)
        x = x.contiguous().view(B * N, 1, ps, ps, ps)
        chunks = []
        for i in range(0, B * N, self.patch_chunk):
            chunk = x[i:i + self.patch_chunk]
            if self.use_grad_ckpt and self.training:
                chunks.append(grad_checkpoint(self._run_dsc, chunk,
                                              use_reentrant=False))
            else:
                chunks.append(self._run_dsc(chunk))
        tokens = torch.cat(chunks, dim=0)  # (B*N, embed_dim)
        tokens = tokens.view(B, N, -1)
        return self.norm(tokens)


# ---------------------------------------------------------------------------
# Transformer blocks (same as original 3DSC-TF)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Filled during forward for interpretability (e.g., attention rollout).
        self.last_attn = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.last_attn = attn
        if self.last_attn.requires_grad:
            self.last_attn.retain_grad()
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Full 3DSC-TF image encoder
# ---------------------------------------------------------------------------
class ImageEncoder3DSCTF(nn.Module):
    """
    3DSC-TF: 3D Depthwise Separable Conv + Transformer image encoder.

    Constructor args follow the same convention as other image encoders in
    this codebase (feat_dim, in_channels, dropout, **kwargs).

    For 128^3 input with patch_size=32: 4x4x4 = 64 non-overlapping patches.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        in_channels: int = 1,
        dropout: float = 0.2,
        img_size: int = 128,
        patch_size: int = 32,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

        self.patch_embed = SharedPatchEmbed(img_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=dropout, attn_drop=dropout,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W) or (B, D, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B = x.size(0)
        tokens = self.patch_embed(x)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        return self.proj(cls_out)
