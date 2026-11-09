"""
Image encoder wrapper for RAE-ViT-AD (https://github.com/jomeiri/RAE-ViT-AD).
Input (B, 1, D, H, W). Supports both 96^3 and 128^3: set img_size=96 for native 96^3
(no resize), or img_size=128 to run at 128^3 (input is resized if different). Outputs (B, feat_dim).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from .rae_vit_ad import RAEViT


def _img_size_to_int(img_size: Union[int, tuple, list]) -> int:
    """Config may give img_size as 128, or [96,96,96]; normalize to int."""
    if isinstance(img_size, int):
        return img_size
    if isinstance(img_size, (tuple, list)) and len(img_size) >= 1:
        v = int(img_size[0])
        if len(img_size) == 3 and (img_size[1], img_size[2]) != (v, v):
            raise ValueError(f"RAE-ViT expects cubic volume; got img_size={img_size}")
        return v
    raise ValueError(f"img_size must be int or (D,H,W) with D=H=W; got {img_size}")


class ImageEncoderRAEViT(nn.Module):
    """
    RAE-ViT-AD as image encoder: 3D volume -> CLS features -> feat_dim.
    Supports 96^3 or 128^3: pass img_size=96 or img_size=128; input is resized only if it does not match.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feat_dim: int = 256,
        img_size: Union[int, tuple, list] = 128,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.2,
        pretrained_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.img_size = _img_size_to_int(img_size)
        self.backbone = RAEViT(
            img_size=self.img_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_classes=2,
        )
        self.backbone.mlp_head = nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if pretrained_checkpoint and str(pretrained_checkpoint).strip():
            path = str(pretrained_checkpoint).strip()
            if __import__("os").path.isfile(path):
                self._load_pretrained(path)
            else:
                print(f"[RAE-ViT] Pretrained not found, from scratch: {path}")

    def _load_pretrained(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[RAE-ViT] Loaded: missing {len(missing)}, unexpected {len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size or x.shape[4] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size,) * 3,
                mode="trilinear",
                align_corners=False,
            )
        z = self.backbone(x)
        return self.proj(z)
