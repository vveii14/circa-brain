"""
ROI encoder: Chen et al. 2019-style DNN on single-atlas connectivity.
DOI: 10.1148/ryai.2019190012 — multichannel DNN; here we use single atlas:
  input (B, n_rois, n_rois) -> flatten upper triangle -> FC+BN+ReLU+Dropout -> feat_dim.
"""

import torch
import torch.nn as nn
from typing import Optional


class Chen2019ROIEncoder(nn.Module):
    """
    Single-atlas version of Chen et al. 2019: connectivity matrix -> flatten upper triangle
    -> FC + BN + ReLU + Dropout (×2) -> feat_dim. No multi-atlas; same DNN style.
    """

    def __init__(
        self,
        n_rois: int,
        feat_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5,
        hidden: int = 512,
        mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.feat_dim = feat_dim
        # Upper triangle (no diagonal): n_rois*(n_rois-1)//2
        self.input_dim = n_rois * (n_rois - 1) // 2

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_rois, n_rois) connectivity matrix, or (B, n_rois) ROI vector (e.g. brain_structure)
        device = x.device
        if x.dim() == 2 and x.size(-1) == self.n_rois:
            # ROI vector (B, n_rois) -> similarity matrix via pairwise z-distance (more informative than rank-1 outer product)
            eps = 1e-6
            z = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True, unbiased=False) + eps)
            dist = torch.abs(z.unsqueeze(2) - z.unsqueeze(1))
            sim01 = torch.exp(-dist)
            x = sim01.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
            eye = torch.eye(self.n_rois, device=device, dtype=x.dtype).unsqueeze(0)
            x = x * (1.0 - eye) + eye
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        B = x.size(0)
        # Upper triangle indices (offset=1, no diagonal)
        i, j = torch.triu_indices(self.n_rois, self.n_rois, offset=1, device=device)
        flat = x[:, i, j]  # (B, input_dim)
        return self.net(flat)
