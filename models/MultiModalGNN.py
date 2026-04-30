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

    def __init__(self, num_node_features, num_edge_features, clinical_input_dim, hidden_channels, num_classes):
        super().__init__()

        # genes graph features encoder
        self.graph_branch = GAT_graph_branch(
            num_node_features,
            num_edge_features,
            hidden_channels
        )  # output has dim hidden_channels * 2

        # clinical feature encoder
        self.clinical_branch = MLP_clinical_branch(clinical_input_dim)  # output has dim 8

        # gate will look at both data type and decide how much importance give to the clinical features for this patient
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 8, num_classes)  # Tumor Classes
        )


    def forward(self, graph_data: torch.Tensor, edge_index, edge_attr, clinical_data: torch.Tensor, batch):

        emb_graph = self.graph_branch(graph_data, edge_index, edge_attr, batch)  # graph-level representation embedding

        emb_clinical = self.clinical_branch(clinical_data)  # clinical representation embedding

        emb_graph = torch.nn.functional.normalize(emb_graph, p=2, dim=1)
        emb_clinical = torch.nn.functional.normalize(emb_clinical, p=2, dim=1)

        # gate_score gives relative importance to clinical features
        gate_score = self.gate(torch.cat([emb_graph, emb_clinical], dim=1))  # gate assigns a [0-1] coefficient
        emb_combined = torch.cat([emb_graph, emb_clinical * gate_score], dim=-1)  # feature fusion

        return self.classifier(emb_combined)
