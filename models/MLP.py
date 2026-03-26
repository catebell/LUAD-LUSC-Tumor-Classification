import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_patient_features):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_patient_features, 64),
            #nn.BatchNorm1d(64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 8),
            #nn.BatchNorm1d(8),
            nn.LayerNorm(8),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(8, 2)


    def forward(self, x):
        x = self.mlp(x)
        #x = torch.nn.functional.normalize(x, p=2, dim=1)

        return self.classifier(x)