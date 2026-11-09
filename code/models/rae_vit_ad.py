"""
RAE-ViT-AD: Regional Attention-Enhanced ViT for AD (https://github.com/jomeiri/RAE-ViT-AD).
Copied and adapted for use as image encoder: single-scale patch embed to match pos_embed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionalAttentionModule(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.region_weight = nn.Parameter(torch.ones(1))

    def forward(self, x, region_mask):
        B, N, C = x.shape
        q = self.query(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.key(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if region_mask.dim() == 2:
            attn = attn + region_mask.unsqueeze(1).unsqueeze(-1) * self.region_weight
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x


class HierarchicalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        # Inputs are (B, N, C); keep attention in batch-first mode.
        self.local_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.global_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.w_local = nn.Parameter(torch.ones(1))
        self.w_global = nn.Parameter(torch.ones(1))

    def forward(self, x):
        local_out, _ = self.local_attn(x, x, x)
        global_out, _ = self.global_attn(x, x, x)
        return self.w_local * local_out + self.w_global * global_out


class MultiScalePatchEmbedding(nn.Module):
    """Single-scale (patch_size=8) to match pos_embed length (img_size//8)**3."""

    def __init__(self, img_size=128, patch_size=8, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.n_patches = (img_size // patch_size) ** 3

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class RAEViT(nn.Module):
    """RAE-ViT backbone; forward returns logits. Use ImageEncoderRAEViT for (B, feat_dim)."""

    def __init__(self, img_size=128, embed_dim=768, num_heads=8, num_classes=3):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_embed = MultiScalePatchEmbedding(img_size, patch_size=8, embed_dim=embed_dim)
        self.ram = RegionalAttentionModule(embed_dim, num_heads)
        self.hsa = HierarchicalSelfAttention(embed_dim, num_heads)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, smri, pet=None, region_mask=None):
        x = self.patch_embed(smri)
        cls_token = self.cls_token.expand(smri.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1) + self.pos_embed
        if region_mask is None:
            region_mask = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        x = self.ram(x, region_mask)
        x = self.hsa(x)
        return self.mlp_head(x[:, 0])

    def forward_features(self, smri, region_mask=None):
        """Return CLS token before classification head (B, embed_dim)."""
        x = self.patch_embed(smri)
        cls_token = self.cls_token.expand(smri.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1) + self.pos_embed
        if region_mask is None:
            region_mask = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        x = self.ram(x, region_mask)
        x = self.hsa(x)
        return x[:, 0]
