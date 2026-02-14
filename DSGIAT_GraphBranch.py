import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, LabelPropagation
from GIATConv import GIATConvLayer

class DSGIAT_GraphBranch(nn.Module):
    """
    Multi-omics graph branch
    Input:
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        batch: [num_nodes] mapping nodes to patient
    Output:
        Graph embedding per patient: [batch_size, 128]
    """
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GIATConvLayer(in_channels, hidden_channels, heads=4)
        self.conv2 = GIATConvLayer(hidden_channels*4, hidden_channels, heads=4)
        self.lp = LabelPropagation(num_layers=2, alpha=0.5)

        # Jump Knowledge concat dimension
        jk_dim = in_channels + hidden_channels*4*2
        self.post_pool_mlp = nn.Sequential(
            nn.Linear(jk_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, x, edge_index, batch):
        h0 = x
        # First GIATConv + LP
        h1 = torch.relu(self.conv1(h0, edge_index))
        h1 = self.lp(h1, edge_index)

        # Second GIATConv + LP
        h2 = torch.relu(self.conv2(h1, edge_index))
        h2 = self.lp(h2, edge_index)

        # Jump Knowledge concatenation
        combined = torch.cat([h0, h1, h2], dim=-1)
        graph_embedding = global_mean_pool(combined, batch)  # [batch_size, jk_dim]
        return self.post_pool_mlp(graph_embedding)  # [batch_size, 128]
