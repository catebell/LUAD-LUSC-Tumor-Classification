import torch
from torch_geometric.nn import GINEConv, global_mean_pool
import torch.nn.functional as F


class CancerGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        super(CancerGNN, self).__init__()
        # GINEConv richiede un MLP per aggiornare le feature
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_node_features, hidden_channels), torch.nn.ReLU())
        self.conv1 = GINEConv(nn1, edge_dim=num_edge_features)

        self.classifier = torch.nn.Linear(hidden_channels, 2)  # LUAD vs LUSC

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Message passing
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        # 2. Global Pooling (trasforma il grafo in un vettore per paziente)
        x = global_mean_pool(x, batch)

        # 3. Classificazione
        return self.classifier(x)