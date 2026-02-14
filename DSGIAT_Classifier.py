import torch
import torch.nn as nn
from DSGIAT_GraphBranch import DSGIAT_GraphBranch
from MLP_clinical import ClinicalMLP

class DSGIAT_Classifier(nn.Module):
    """
    Full multimodal classifier
    """
    def __init__(self, omic_in_channels, clinical_input_dim, hidden_channels):
        super().__init__()
        self.graph_branch = DSGIAT_GraphBranch(omic_in_channels, hidden_channels)
        self.clinical_branch = ClinicalMLP(clinical_input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(128+32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x, edge_index, batch, clinical_data):
        v_graph = self.graph_branch(x, edge_index, batch)      # [batch_size, 128]
        v_clinical = self.clinical_branch(clinical_data)       # [batch_size, 32]
        combined = torch.cat([v_graph, v_clinical], dim=-1)    # [batch_size, 160]
        logits = self.classifier(combined)                     # [batch_size, 2]
        return logits
