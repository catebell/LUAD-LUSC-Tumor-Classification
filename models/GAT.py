import torch
from torch_geometric.nn import GATv2Conv, global_mean_pool

# GATv2Conv: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GATv2Conv.html


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_edge_features, hidden_channels, heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_channels, edge_dim=num_edge_features,
                               heads=heads)
        self.conv2 = GATv2Conv(in_channels=hidden_channels * heads, out_channels=hidden_channels, edge_dim=num_edge_features,
                               heads=1)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)  # LUAD vs LUSC

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = x.relu()

        x = global_mean_pool(x, batch)  # graph to single vector per patient

        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        return self.classifier(x)


'''
1. Multi-Head Attention
Il vero potere dei GAT risiede nelle "teste" multiple. Usare heads=1 limita il modello a un solo "punto di vista" sulle relazioni tra geni.
Usando più teste (es. 4 o 8), permetti al modello di imparare diverse sottoreti funzionali contemporaneamente.

2. Batch Normalization e Skip Connections
Con grafi di 20.000 nodi, il segnale tende a "diluirsi" o a svanire (vanishing gradient).
La BatchNorm stabilizza l'apprendimento, mentre le skip connections (sommare l'input all'output del layer) aiutano a preservare l'informazione originale del gene.

3. Pooling Evoluto
Il global_mean_pool fa la media di tutti i geni. Ma in oncologia, spesso sono pochi geni chiave a determinare il fenotipo.
Usare una combinazione di mean e max pooling (o un GlobalAttention) aiuta a catturare sia il segnale globale che i picchi di espressione/mutazione.

4. Struttura del Classifier
Passare da 64 canali direttamente alle 2 classi è un salto brusco. Un piccolo MLP alla fine aiuta a distillare meglio il vettore del paziente.
'''