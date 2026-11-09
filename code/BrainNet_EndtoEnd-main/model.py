import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import aggr
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import ModuleList

softmax = torch.nn.LogSoftmax(dim=1)

class TimeSeriesEncoder(nn.Module):
    def __init__(self, num_rois, time_steps, embedding_size):
        super(TimeSeriesEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=time_steps, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=embedding_size, kernel_size=3, stride=1, padding=1)
        # Removing the linear layer that caused the mismatch
        # If you want to add a linear layer, ensure its input matches the output of conv2
        # self.fc = nn.Linear(num_rois, num_rois)  # Commented out for now

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, time_steps, num_rois) -> (batch_size, num_rois, time_steps)
        x = F.relu(self.conv1(x))  # (batch_size, embedding_size, num_rois)
        x = F.relu(self.conv2(x))  # (batch_size, embedding_size, num_rois)
        x = x.permute(0, 2, 1)  # (batch_size, num_rois, embedding_size)
        return x

# Graph Generator
class GraphGenerator(nn.Module):
    def __init__(self, embedding_size, num_rois):
        super(GraphGenerator, self).__init__()
        self.fc = nn.Linear(embedding_size, num_rois)

    def forward(self, x):
        # print(x.size())
        hA = F.softmax(self.fc(x), dim=-1)  # hA should now be (batch_size, num_rois, num_rois)
        # print(hA.size())
        A = torch.bmm(hA, hA.transpose(1, 2))  # This ensures A has the shape (batch_size, num_rois, num_rois)
        return A

# Graph Predictor using GCN
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# End-to-end model
class FBNetGen(nn.Module):
    def __init__(self, num_rois, time_steps, embedding_size, hidden_channels, num_classes):
        super(FBNetGen, self).__init__()
        self.encoder = TimeSeriesEncoder(num_rois, time_steps, embedding_size)
        self.graph_generator = GraphGenerator(embedding_size, num_rois)
        self.gcn = GCN(embedding_size, hidden_channels, num_classes)

    def forward(self, x, save_graph=False, batch_idx=None):
        x = self.encoder(x)
        # print('x', x.size())
        A = self.graph_generator(x)
        if save_graph and batch_idx is not None:
            torch.save(A, f"saved_graphs/graph_batch_{batch_idx}.pt")
        
        node_features = x.reshape(-1, x.size(2))
        num_nodes = node_features.size(0)
        edge_index = torch.randint(0, num_nodes, (2, 500)).to(x.device)
        batch = torch.repeat_interleave(torch.arange(x.size(0)), x.size(1)).to(x.device)
        output = self.gcn(node_features, edge_index, batch)
        return output


class ResidualGNNs(torch.nn.Module):
    def __init__(self,args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        # 
        num_features = train_dataset[0].num_features
        if args.model=="ChebConv":
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels,K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels,K=5))
        elif args.model=="GINConv":
            mlp = nn.Sequential(
                nn.Linear(num_features, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GNN(mlp))
            for _ in range(num_layers - 1):
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(GNN(mlp))
        else:
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))
        
        input_dim1 = int(((num_features * num_features)/2)+ (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)+ (num_features/2))
            
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

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
