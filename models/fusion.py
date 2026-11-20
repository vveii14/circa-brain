"""
Fusion heads: cross-view contrastive alignment (proposed) and plain concatenation (ablation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """Simple concatenation of image and ROI features + classifier."""

    def __init__(
        self,
        image_dim: int,
        roi_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.fused_dim = image_dim + roi_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        z_img: torch.Tensor,
        z_roi: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        z = torch.cat([z_img, z_roi], dim=1)
        logits = self.classifier(z)
        if return_embeddings:
            return logits, (z_img, z_roi)
        return logits

    def get_contrastive_loss(self, z_img: torch.Tensor, z_roi: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=z_img.device)


class ContrastiveFusion(nn.Module):
    """
    Align image and ROI embeddings with InfoNCE, then concat and classify.
    L_cls + lambda * L_contrastive.
    """

    def __init__(
        self,
        image_dim: int,
        roi_dim: int,
        num_classes: int,
        proj_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.temperature = temperature
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.ReLU(inplace=True),
        )
        self.roi_proj = nn.Sequential(
            nn.Linear(roi_dim, proj_dim),
            nn.ReLU(inplace=True),
        )
        self.fused_dim = image_dim + roi_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        z_img: torch.Tensor,
        z_roi: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        z = torch.cat([z_img, z_roi], dim=1)
        logits = self.classifier(z)
        if return_embeddings:
            return logits, (z_img, z_roi)
        return logits

    def get_contrastive_loss(self, z_img: torch.Tensor, z_roi: torch.Tensor) -> torch.Tensor:
        p_img = F.normalize(self.image_proj(z_img), dim=1)
        p_roi = F.normalize(self.roi_proj(z_roi), dim=1)
        logits_ij = torch.mm(p_img, p_roi.t()) / self.temperature  # (B, B): image i vs ROI j
        labels = torch.arange(p_img.size(0), device=z_img.device)
        # Symmetric InfoNCE: image→ROI and ROI→image
        loss_img2roi = F.cross_entropy(logits_ij, labels)
        loss_roi2img = F.cross_entropy(logits_ij.t(), labels)
        return (loss_img2roi + loss_roi2img) / 2
