from DSGAT import DSGAT_GraphBranch
import torch.nn as nn
from MLP_clinical import ClinicalMLP
import torch

class DSGAT_Classifier(nn.Module):
    """
    Full multimodal classifier:
    - Graph branch (GAT-based)
    - Clinical MLP branch
    - Final fusion classifier
    """

    def __init__(self, omic_in_channels, clinical_input_dim, hidden_channels):
        super().__init__()

        self.graph_branch = DSGAT_GraphBranch(
            omic_in_channels,
            hidden_channels
        )

        # Clinical feature encoder
        self.clinical_branch = ClinicalMLP(clinical_input_dim)

        # Final classifier after feature fusion
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x, edge_index, batch, clinical_data):

        # Graph-level representation
        v_graph = self.graph_branch(x, edge_index, batch)      # [batch_size, 128]

        # Clinical representation
        v_clinical = self.clinical_branch(clinical_data)       # [batch_size, 32]

        # Feature fusion
        combined = torch.cat([v_graph, v_clinical], dim=-1)    # [batch_size, 160]

        # Classification logits
        return self.classifier(combined)                     # [batch_size, 2]
