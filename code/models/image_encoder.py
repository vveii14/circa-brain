"""
Image-based branch: 3D CNN on brain volume (optionally from 4D temporal mean).
"""

import torch
import torch.nn as nn
from typing import Optional


class ImageEncoder3D(nn.Module):
    """Simple 3D CNN backbone. Input (B, 1, D, H, W), output (B, feat_dim)."""

    def __init__(
        self,
        in_channels: int = 1,
        feat_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W)
        if x.dim() == 5 and x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        elif x.dim() == 4:
            x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)


class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.BatchNorm3d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class ImageEncoderResNet3D(nn.Module):
    """Lightweight 3D ResNet-style encoder."""

    def __init__(
        self,
        in_channels: int = 1,
        feat_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.layer1 = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResBlock3D(64),
            nn.MaxPool3d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            ResBlock3D(128),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5 and x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        elif x.dim() == 4:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.fc(x)
