"""
3D ViT for image branch, aligned with ViT_recipe_for_AD (Vision_Transformer3D).
Input (B, 1, D, H, W), output (B, feat_dim). Load ViT-B checkpoint then finetune.
"""

import math
import os
import torch
import torch.nn as nn
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


# ---------- Structure aligned with ViT_recipe_for_AD for loading their checkpoint ----------
class PatchEmbed3D(nn.Module):
    """Same as ViT_recipe: (B,1,D,H,W) -> (B, n_patches, embed_dim)."""

    def __init__(self, img_size: Union[int, Tuple], patch_size: int, embed_dim: int, in_chans: int = 1):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size,) * 3
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        out = self.proj(torch.rand(1, 1, *self.img_size))
        self.n_patches = out.flatten(2).shape[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """ViT_recipe style: qkv + proj (same state_dict keys)."""

    def __init__(self, dim: int, n_heads: int = 12, qkv_bias: bool = True, attn_p: float = 0.0, proj_p: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    """ViT_recipe style: fc1, act, fc2, drop."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, p: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """ViT_recipe Block: norm1, attn, norm2, mlp, drop_path."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        p: float = 0.0,
        attn_p: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features, dim, p=p)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ImageEncoderViT3D(nn.Module):
    """
    3D ViT encoder aligned with ViT_recipe Vision_Transformer3D (ViT-B).
    - Load ViT-B checkpoint via pretrained_checkpoint (state_dict key 'net' or full dict).
    - No classification head; we use a projection 768 -> feat_dim (trained from scratch).
    """

    def __init__(
        self,
        img_size: Union[int, Tuple] = 96,
        patch_size: int = 16,
        in_channels: int = 1,
        feat_dim: int = 256,
        embed_dim: int = 768,
        depth: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        pretrained_checkpoint: Optional[str] = None,
        checkpoint_key: str = "net",
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size,) * 3
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(img_size, patch_size, embed_dim, in_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=dpr[i], p=dropout, attn_p=dropout)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # Our projection (not in ViT_recipe; trained from scratch when loading pretrained)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

        path = str(pretrained_checkpoint).strip() if pretrained_checkpoint else ""
        if path and os.path.isfile(path):
            self._load_pretrained(path, checkpoint_key)
        elif path:
            print(f"[ViT3D] Pretrained path not found, training from scratch: {path}")

    def _load_pretrained(self, path: str, checkpoint_key: str = "net") -> None:
        """Load ViT-B checkpoint (ViT_recipe format: state['net']). Skip head and shape-mismatch (e.g. pos_embed)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and checkpoint_key in ckpt:
            state = ckpt[checkpoint_key]
        else:
            state = ckpt
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        self_state = self.state_dict()
        loaded = {}
        for k, v in state.items():
            if k not in self_state:
                continue
            if self_state[k].shape != v.shape:
                continue
            loaded[k] = v
        self_state.update(loaded)
        self.load_state_dict(self_state, strict=False)
        n_loaded = len(loaded)
        n_skipped = len(state) - n_loaded
        print(f"[ViT3D] Pretrained: loaded {n_loaded} params, skipped {n_skipped} (shape mismatch or our proj).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_final = x[:, 0]
        return self.proj(cls_final)
