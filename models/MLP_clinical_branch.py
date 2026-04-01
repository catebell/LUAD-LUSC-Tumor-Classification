import torch.nn as nn

class MLP_clinical_branch(nn.Module):
    def __init__(self, num_patient_features):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_patient_features, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 8),
            nn.LayerNorm(8),
            nn.SiLU(),
            nn.Dropout(0.2)
        )


    def forward(self, x):
        return self.mlp(x)
