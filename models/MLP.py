import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_patient_features, num_classes):
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

        self.classifier = nn.Linear(8, num_classes)


    def forward(self, x):
        x = self.mlp(x)

        return self.classifier(x)