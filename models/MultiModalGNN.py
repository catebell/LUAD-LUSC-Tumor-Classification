import torch.nn as nn
import torch

from models.GAT_graph_branch import GAT_graph_branch
from models.MLP_clinical_branch import MLP_clinical_branch


class MultiModalGNN(nn.Module):
    """
    Full multimodal classifier:
    - Graph branch (GAT-based)
    - Clinical MLP branch
    - Final fusion classifier
    """

    def __init__(self, num_node_features, num_edge_features, clinical_input_dim, hidden_channels):
        super().__init__()

        self.graph_branch = GAT_graph_branch(
            num_node_features,
            num_edge_features,
            hidden_channels
        )

        # Clinical feature encoder
        self.clinical_branch = MLP_clinical_branch(clinical_input_dim)

        # Final classifier after feature fusion
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # LUAD vs LUSC
        )

    def forward(self, graph_data: torch.Tensor, edge_index, edge_attr, clinical_data: torch.Tensor, batch):

        emb_graph = self.graph_branch(graph_data, edge_index, edge_attr, batch)  # graph-level representation embedding [batch_size, 128]

        emb_clinical = self.clinical_branch(clinical_data)  # clinical representation embedding [batch_size, 32]

        emb_combined = torch.cat([emb_graph, emb_clinical], dim=-1)  # feature fusion [batch_size, 160]

        return self.classifier(emb_combined)  # [batch_size, 2]
