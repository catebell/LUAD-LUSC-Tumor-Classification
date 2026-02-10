import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from GIATConv import GIATConvLayer

class DSGIAT_GraphBranch(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(DSGIAT_GraphBranch, self).__init__()

        # Primo Layer GIATConv (assumendo 4 teste di default nel GIATConvLayer)
        self.conv1 = GIATConvLayer(in_channels, hidden_channels, heads=4)
        # Secondo Layer GIATConv
        self.conv2 = GIATConvLayer(hidden_channels * 4, hidden_channels, heads=4)

        # Modulo G: Riduce la dimensione delle feature concatenate
        # in_channels (h0) + hidden*4 (h1) + hidden*4 (h2)
        self.post_pool_mlp = nn.Sequential(
            nn.Linear(in_channels + (hidden_channels * 4 * 2), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, x, edge_index, batch):
        h0 = x

        h1 = torch.relu(self.conv1(h0, edge_index))
        # Qui si potrebbe inserire la Label Propagation reale

        h2 = torch.relu(self.conv2(h1, edge_index))

        # Concatenazione Jump Knowledge
        combined_node_features = torch.cat([h0, h1, h2], dim=-1)

        # Pooling Globale: da molti nodi a un solo vettore per paziente
        graph_representation = global_mean_pool(combined_node_features, batch)

        return self.post_pool_mlp(graph_representation)


class DSGIAT_Classifier(nn.Module):
    def __init__(self, omic_in_channels, clinical_in_channels, hidden_channels):
        super(DSGIAT_Classifier, self).__init__()

        # 1. RAMO GRAFO (Omiche)
        self.graph_branch = DSGIAT_GraphBranch(omic_in_channels, hidden_channels)

        # 2. RAMO CLINICO (MLP semplice)
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # 3. MLP FINALE DI CLASSIFICAZIONE (Modulo C)
        # Unisce 128 (output grafo) + 64 (output clinico)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            # Output per classificazione binaria: 2 classi (LUAD=0, LUSC=1)
            nn.Linear(32, 2)
        )

    def forward(self, x, edge_index, batch, clinical_data):
        # Stream 1: Omics via GIATConv
        v_graph = self.graph_branch(x, edge_index, batch)

        # Stream 2: Clinica via MLP
        v_clinical = self.clinical_branch(clinical_data)

        # Fusione (Concatenazione C)
        combined = torch.cat([v_graph, v_clinical], dim=-1)

        # Classificazione finale
        logits = self.classifier(combined)
        return logits

# ESEMPIO DI INIZIALIZZAZIONE:
# model = DSGIAT_Classifier(omic_in_channels=3, clinical_in_channels=10, hidden_channels=32)