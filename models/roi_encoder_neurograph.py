"""
NeuroGraph ROI encoder.

Wraps the ResidualGNNs backbone of NeuroGraph (https://github.com/Anwar-Said/NeuroGraph)
and maps a batch of ROI connectivity matrices (B, n_rois, n_rois) to graph-level
embeddings (B, feat_dim).

Graph construction per subject:
  * node features: the rows of the connectivity matrix (n_rois-dim per node);
  * edges: the strongest `edge_top_p` fraction of upper-triangular entries
    (made undirected).
"""

from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import ChebConv, GATConv, GCNConv, GINConv, GraphConv, SAGEConv

from .neurograph_residual_gnn import ResidualGNNs

_GNN_MAP = {
    "GCNConv": GCNConv,
    "GATConv": GATConv,
    "GINConv": GINConv,
    "SAGEConv": SAGEConv,
    "GraphConv": GraphConv,
    "ChebConv": ChebConv,
}


def _matrix_to_graph(C: torch.Tensor, n_rois: int, edge_top_p: float) -> Data:
    """(n_rois, n_rois) connectivity matrix -> PyG Data."""
    C = C.float()
    triu_idx = torch.triu_indices(n_rois, n_rois, offset=1, device=C.device)
    vals = C[triu_idx[0], triu_idx[1]]
    k = max(1, int(vals.numel() * edge_top_p))
    _, top_idx = torch.topk(vals, k)
    edge_index = triu_idx[:, top_idx]
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(x=C, edge_index=edge_index)


class NeuroGraphROIEncoder(nn.Module):
    """(B, n_rois, n_rois) connectivity matrices -> (B, feat_dim) embeddings."""

    def __init__(
        self,
        n_rois: int,
        feat_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5,
        hidden_channels: int = 32,
        hidden_mlp: int = 64,
        num_layers: int = 2,
        gnn_name: str = "GCNConv",
        edge_top_p: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.feat_dim = feat_dim
        self.edge_top_p = edge_top_p

        gnn_cls = _GNN_MAP.get(gnn_name)
        if gnn_cls is None:
            raise ValueError(f"Unknown gnn_name: {gnn_name}. Available: {list(_GNN_MAP)}")

        # ResidualGNNs reads `args.model`, `args.num_classes` and `train_dataset[0].num_features`.
        args = types.SimpleNamespace(model=gnn_name, num_classes=num_classes)
        dataset_stub = [types.SimpleNamespace(num_features=n_rois)]
        self.gnn = ResidualGNNs(args, dataset_stub, hidden_channels, hidden_mlp, num_layers, gnn_cls)

        # Input of the MLP: upper triangle (incl. diagonal) of the matrix + pooled hidden states.
        triu_len = (n_rois * (n_rois + 1)) // 2
        self.gnn.bn = nn.BatchNorm1d(triu_len)
        self.gnn.mlp[0] = nn.Linear(triu_len + hidden_channels * num_layers, hidden_mlp)

        # The embedding is the input of the last MLP layer (the pre-logit representation).
        self._pre_logit_dim = hidden_mlp // 2
        self.proj = nn.Sequential(
            nn.Linear(self._pre_logit_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self._captured: Optional[torch.Tensor] = None

    def _capture_pre_logit(self, module, inputs):
        # Keep the graph attached so gradients reach the GNN backbone.
        self._captured = inputs[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B, {self.n_rois}, {self.n_rois}) connectivity matrices, got {tuple(x.shape)}")
        # BatchNorm1d needs more than one sample in training mode: duplicate a single graph.
        single_graph = x.size(0) == 1
        if single_graph:
            x = x.repeat(2, 1, 1)
        batch = Batch.from_data_list([_matrix_to_graph(c, self.n_rois, self.edge_top_p) for c in x]).to(x.device)
        self._captured = None
        handle = self.gnn.mlp[-1].register_forward_pre_hook(self._capture_pre_logit)
        try:
            self.gnn(batch)
        finally:
            handle.remove()
        if self._captured is None:
            raise RuntimeError("NeuroGraph ROI encoder: pre-logit tensor not captured")
        out = self.proj(self._captured)
        return out[:1] if single_graph else out
