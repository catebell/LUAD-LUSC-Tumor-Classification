import torch.nn as nn

class MLP_clinical_branch(nn.Module):
    def __init__(self, num_patient_features):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_patient_features, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(0.7),

            nn.Linear(16, 4),
            nn.LayerNorm(4),
            nn.ReLU(),
            nn.Dropout(0.6)
        )

    def forward(self, x):
        return self.mlp(x)
