import torch
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
import torch.nn.functional as F

class CancerGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        super(CancerGNN, self).__init__()
        # GINEConv needs an MLP
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_node_features, hidden_channels), torch.nn.ReLU())
        self.conv1 = GINEConv(nn1, edge_dim=num_edge_features)

        self.classifier = torch.nn.Linear(hidden_channels, 2)  # LUAD vs LUSC

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        x = global_mean_pool(x, batch)  # graph to single vector per patient

        x = F.dropout(x, p=0.5, training=self.training)
        return self.classifier(x)

# TODO F.dropout(x, p=0.5, training=self.training) tra i layer del modello.

'''  # nice ma non usare senza GPU
class CancerGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        super(CancerGNN, self).__init__()

        # Layer 1
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINEConv(nn1, edge_dim=num_edge_features)
        self.bn1 = BatchNorm(hidden_channels)

        # Layer 2
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINEConv(nn2, edge_dim=num_edge_features)
        self.bn2 = BatchNorm(hidden_channels)

        self.classifier = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, edge_attr, batch):
        # Primo passaggio
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # Secondo passaggio (con Skip Connection)
        h = self.conv2(x, edge_index, edge_attr=edge_attr)
        h = self.bn2(h)
        x = F.relu(x + h)  # Residual connection

        x = global_mean_pool(x, batch)
        return self.classifier(x)
'''
