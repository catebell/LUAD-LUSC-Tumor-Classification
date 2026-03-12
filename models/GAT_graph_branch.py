import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, LabelPropagation

class GAT_graph_branch(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, heads=2):
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels=num_node_features,
            out_channels=hidden_channels,
            edge_dim = num_edge_features,
            heads=heads,
            concat=True
        ) # output dimension = hidden_channels * heads (because concat=True)


        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            edge_dim = num_edge_features,
            heads=heads,
            concat=True
        ) # output dimension = hidden_channels * heads

        # Label Propagation: https://medium.com/@ivavrtaric/graph-neural-networks-the-label-propagation-algorithm-46f75a0c468c
        self.lp = LabelPropagation(num_layers=2, alpha=0.5)

        # Skip Knowledge concatenation dimension:
        # h0 (input) + h1 (hidden_channels*heads) + h2 (hidden_channels*heads)
        skip_knowledge_dim = num_node_features + hidden_channels * heads * 2

        # MLP after graph pooling
        self.post_pool_mlp = nn.Sequential(
            nn.Linear(skip_knowledge_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, x, edge_index, batch):
        h0 = x
        # Each node attends over its neighbors, multi-head attention increases expressive power
        h1 = F.elu(self.conv1(h0, edge_index))
        h1 = self.lp(h1, edge_index)  # label propagation smooths node embeddings

        # second GAT layer learns higher-level interactions
        h2 = F.elu(self.conv2(h1, edge_index))
        h2 = self.lp(h2, edge_index)

        # skip connections: concatenate representations from different depths
        combined = torch.cat([h0, h1, h2], dim=-1)

        # aggregate node embeddings into graph embedding
        graph_embedding = global_mean_pool(combined, batch)

        return self.post_pool_mlp(graph_embedding)  # out features = 128