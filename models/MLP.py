import torch
import torch.nn as nn

class MLP(nn.Module):
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

        self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        x = self.mlp(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)

        return self.classifier(x)