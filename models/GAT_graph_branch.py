import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm, global_max_pool

# our GAT
class GAT_graph_branch(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, heads=2):
        super(GAT_graph_branch, self).__init__()
        # if hidden_channels dims change we need to project to map num_node_features to the output dimension before adding skip connections:
        self.skip_conn_projection1 = torch.nn.Linear(num_node_features, hidden_channels * heads)
        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_channels, edge_dim=num_edge_features,
                               heads=heads)
        self.bn1 = BatchNorm(hidden_channels * heads)

        # (no need if hidden_channels is the same):
        self.skip_conn_projection2 = torch.nn.Linear(hidden_channels * heads, hidden_channels)
        self.conv2 = GATv2Conv(in_channels=hidden_channels * heads, out_channels=hidden_channels,
                               edge_dim=num_edge_features, heads=1)
        self.bn2 = BatchNorm(hidden_channels)


    def forward(self, x, edge_index, edge_attr, batch, return_attention_weights=False):
        x_projected = self.skip_conn_projection1(x)
        if return_attention_weights:
            x, att_weights = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = x + x_projected
        x = F.elu(x)

        x_projected = self.skip_conn_projection2(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = x + x_projected
        x = F.elu(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)  # TODO try without
        x = torch.cat([x_mean, x_max], dim=1)  # now x has dimension hidden_channels * 2

        if return_attention_weights:
            return x, att_weights
        return x

    def get_attention(self, x, edge_index, edge_attr, batch):
        return self.forward(x, edge_index, edge_attr, batch, return_attention_weights=True)
