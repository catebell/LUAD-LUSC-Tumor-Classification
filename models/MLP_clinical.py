import torch.nn as nn

class ClinicalMLP(nn.Module):
    """
        Input: [batch_size, num_clinical_features]
        Output: embedding [batch_size, 32]
    """
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
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