import torch.nn as nn

class MLP_clinical_branch(nn.Module):
    def __init__(self, num_patient_features):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_patient_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.net(x)