"""
Full model: image encoder + ROI encoder + fusion head.
All components are resolved from the registry by name (config-driven, pluggable).
"""

import torch
import torch.nn as nn
from typing import Optional

from . import registry


class ImageROIFusionModel(nn.Module):
    """
    Combines image-based and ROI-based branches with configurable fusion.
    Image encoder, ROI encoder, and fusion module are all replaceable via config (registry).
    """

    def __init__(
        self,
        n_rois: int,
        num_classes: int = 2,
        use_image_branch: bool = True,
        use_roi_branch: bool = True,
        image_encoder: str = "conv3d",
        image_feat_dim: int = 256,
        roi_encoder: str = "connectivity",
        roi_feat_dim: int = 256,
        fusion: str = "concat",
        contrastive_temperature: float = 0.07,
        contrastive_weight: float = 0.1,
        dropout: float = 0.3,
        in_channels: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.use_image_branch = use_image_branch
        self.use_roi_branch = use_roi_branch
        self.contrastive_weight = contrastive_weight

        # Optional kwargs for image encoder (e.g. vit3d: img_size, patch_size, embed_dim, depth, n_heads, mlp_ratio)
        image_enc_kwargs = kwargs.get("image_encoder_kwargs") or {}
        if use_image_branch:
            self.image_enc = registry.build_image_encoder(
                image_encoder,
                feat_dim=image_feat_dim,
                in_channels=in_channels,
                dropout=dropout,
                **image_enc_kwargs,
            )
        else:
            self.image_enc = None
            image_feat_dim = 0

        if use_roi_branch:
            roi_kw = {k: v for k, v in kwargs.items() if k in ("max_T", "num_classes")}
            if "num_classes" not in roi_kw:
                roi_kw["num_classes"] = num_classes
            roi_enc_kwargs = kwargs.get("roi_encoder_kwargs") or {}
            roi_kw.update(roi_enc_kwargs)
            self.roi_enc = registry.build_roi_encoder(
                roi_encoder,
                n_rois=n_rois,
                feat_dim=roi_feat_dim,
                dropout=dropout,
                **roi_kw,
            )
        else:
            self.roi_enc = None
            roi_feat_dim = 0

        if use_image_branch and use_roi_branch:
            fusion_kwargs = dict(kwargs.get("fusion_kwargs") or {})
            fusion_kwargs.setdefault("temperature", contrastive_temperature)
            fusion_kwargs.setdefault("dropout", dropout)
            self.fusion_head = registry.build_fusion(
                fusion,
                image_dim=image_feat_dim,
                roi_dim=roi_feat_dim,
                num_classes=num_classes,
                **fusion_kwargs,
            )
        elif use_image_branch:
            self.fusion_head = nn.Linear(image_feat_dim, num_classes)
        else:
            self.fusion_head = nn.Linear(roi_feat_dim, num_classes)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        roi_data: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        z_img = self.image_enc(image) if self.use_image_branch and image is not None else None
        z_roi = self.roi_enc(roi_data) if self.use_roi_branch and roi_data is not None else None

        if self.use_image_branch and self.use_roi_branch:
            logits = self.fusion_head(z_img, z_roi, return_embeddings=return_embeddings)
            if return_embeddings:
                logits, (z_img, z_roi) = logits
        elif self.use_image_branch:
            logits = self.fusion_head(z_img)
        else:
            logits = self.fusion_head(z_roi)

        if return_embeddings and self.use_image_branch and self.use_roi_branch:
            return logits, (z_img, z_roi)
        return logits

    def get_contrastive_loss(
        self,
        image: Optional[torch.Tensor] = None,
        roi_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not (self.use_image_branch and self.use_roi_branch) or image is None or roi_data is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        z_img = self.image_enc(image)
        z_roi = self.roi_enc(roi_data)
        return self.fusion_head.get_contrastive_loss(z_img, z_roi)
