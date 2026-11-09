"""
ROI-based branch: connectivity matrix or timeseries -> embedding.
"""

import torch
import torch.nn as nn
from typing import Optional


class ROIEncoder(nn.Module):
    """
    Encodes ROI data: (B, n_rois, n_rois) connectivity or (B, T, n_rois) timeseries.
    """

    def __init__(
        self,
        n_rois: int,
        feat_dim: int = 256,
        mode: str = "connectivity",  # connectivity | timeseries_1d | transformer | roi_vector
        max_T: Optional[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.feat_dim = feat_dim
        self.mode = mode
        self.max_T = max_T or 200

        if mode == "connectivity":
            # Symmetric matrix: use upper triangle or full -> MLP
            self.proj = nn.Sequential(
                nn.Linear(n_rois * n_rois, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, feat_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        elif mode == "timeseries_1d":
            self.conv = nn.Sequential(
                nn.Conv1d(n_rois, 64, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, feat_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        elif mode == "roi_vector":
            # Single vector per subject (e.g. atlas mean intensity per ROI for sMRI)
            self.proj = nn.Sequential(
                nn.Linear(n_rois, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, feat_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        elif mode == "transformer":
            self.embed = nn.Linear(n_rois, 64)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                dropout=dropout,
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.proj = nn.Sequential(
                nn.Linear(64 * self.max_T, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, feat_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "connectivity":
            # (B, n_rois, n_rois) -> (B, n_rois*n_rois)
            x = x.flatten(1)
            return self.proj(x)
        elif self.mode == "roi_vector":
            # (B, n_rois)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.proj(x)
        elif self.mode == "timeseries_1d":
            # (B, T, n_rois) -> (B, n_rois, T)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.dim() == 3:
                x = x.transpose(1, 2)
            x = self.conv(x)
            return self.proj(x)
        else:
            # transformer: (B, T, n_rois)
            B, T, _ = x.shape
            x = self.embed(x)
            x = self.transformer(x)
            x = x.reshape(B, -1)
            if x.shape[1] > 64 * self.max_T:
                x = x[:, : 64 * self.max_T]
            elif x.shape[1] < 64 * self.max_T:
                x = torch.nn.functional.pad(x, (0, 64 * self.max_T - x.shape[1]))
            return self.proj(x)
