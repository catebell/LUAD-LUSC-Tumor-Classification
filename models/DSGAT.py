import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, LabelPropagation

class DSGAT_GraphBranch(nn.Module):
    """
    Multi-omics graph branch using Graph Attention Networks (GAT)

    Input:
        x:            Node features              [num_nodes, in_channels]
        edge_index:   Graph connectivity (COO)   [2, num_edges]
        batch:        Node-to-graph mapping      [num_nodes]

    Output:
        Graph-level embedding per patient        [batch_size, 128]
    """

    def __init__(self, in_channels, hidden_channels, heads=4):
        super().__init__()

        # First GAT layer:
        # - Multi-head attention
        # - Output dimension = hidden_channels * heads (because concat=True)
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True
        )

        # Second GAT layer:
        # - Takes concatenated output of first layer
        # - Again multi-head attention
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True
        )

        # Label Propagation layer to smooth node representations
        self.lp = LabelPropagation(num_layers=2, alpha=0.5)

        # Jump Knowledge concatenation dimension:
        # h0 (input) +
        # h1 (hidden_channels*heads) +
        # h2 (hidden_channels*heads)
        jk_dim = in_channels + hidden_channels * heads * 2

        # MLP after graph pooling
        self.post_pool_mlp = nn.Sequential(
            nn.Linear(jk_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, x, edge_index, batch):

        # Save initial node features (for Jump Knowledge)
        h0 = x

        # ----- First GAT layer -----
        # Each node attends over its neighbors
        # Multi-head attention increases expressive power
        h1 = F.elu(self.conv1(h0, edge_index))

        # Label propagation smooths node embeddings
        h1 = self.lp(h1, edge_index)

        # ----- Second GAT layer -----
        # Learns higher-level interactions
        h2 = F.elu(self.conv2(h1, edge_index))

        # Again smooth with label propagation
        h2 = self.lp(h2, edge_index)

        # ----- Jump Knowledge -----
        # Concatenate representations from different depths
        combined = torch.cat([h0, h1, h2], dim=-1)

        # ----- Graph-level pooling -----
        # Aggregate node embeddings into graph embedding
        graph_embedding = global_mean_pool(combined, batch)

        # Final graph representation
        return self.post_pool_mlp(graph_embedding)