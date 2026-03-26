import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
        self.conv2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        self.conv3 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
