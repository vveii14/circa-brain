"""
ResidualGNNs from NeuroGraph (inlined for standalone release). https://github.com/Anwar-Said/NeuroGraph
Original: NeuroGraph model.py
"""
import torch
from torch import nn
from torch.nn import ModuleList
from torch_geometric.nn import aggr

softmax = torch.nn.LogSoftmax(dim=1)


class ResidualGNNs(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset[0].num_features
        if args.model == "ChebConv":
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels, K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels, K=5))
        elif args.model == "GINConv":
            mlp = nn.Sequential(
                nn.Linear(num_features, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GNN(mlp))
            for _ in range(num_layers - 1):
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.convs.append(GNN(mlp))
        else:
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        input_dim1 = int(((num_features * num_features) / 2) + (num_features / 2) + (hidden_channels * num_layers))
        input_dim = int(((num_features * num_features) / 2) + (num_features / 2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels * num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden // 2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        upper_tri_indices = torch.triu_indices(x.shape[1], x.shape[1])
        for i, xx in enumerate(xs):
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t[upper_tri_indices[0], upper_tri_indices[1]] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)
        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return softmax(x)
