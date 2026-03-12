import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, BatchNorm


# GATv2Conv: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GATv2Conv.html


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_edge_features, hidden_channels, heads=2):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_channels, edge_dim=num_edge_features,
                               heads=heads)
        self.bn1 = BatchNorm(hidden_channels * heads)
        # if hidden_channels dims change we need to project to map num_node_features to the output dimension before adding skip connections:
        self.skip_conn_projection1 = torch.nn.Linear(num_node_features, hidden_channels * heads)

        self.conv2 = GATv2Conv(in_channels=hidden_channels * heads, out_channels=hidden_channels, edge_dim=num_edge_features,
                               heads=1)
        self.bn2 = BatchNorm(hidden_channels)
        # (no need if hidden_channels is the same):
        self.skip_conn_projection2 = torch.nn.Linear(hidden_channels * heads, hidden_channels)

        self.classifier = torch.nn.Linear(hidden_channels * 2, num_classes)  # LUAD vs LUSC

        # LUAD vs LUSC
        '''
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(hidden_channels, num_classes)
        )
        
        oppure
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3), 
            torch.nn.Linear(hidden_channels, num_classes)
        )
        '''

    def forward(self, x, edge_index, edge_attr, batch):
        x_projected = self.skip_conn_projection1(x)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = x + x_projected
        x = F.elu(x)

        x_projected = self.skip_conn_projection2(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = x + x_projected
        x = F.elu(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)  #now x has dimension hidden_channels * 2

        # TODO provare senza e magari mettere dropout nelle conv
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        return self.classifier(x)


'''
1. Gestione del Dropout
Attualmente hai il dropout solo sul classificatore. Nei GAT è molto efficace applicare il dropout anche ai coefficienti di attenzione e all'input dei layer convolutivi.

self.conv1 = GATv2Conv(..., dropout=0.2) # Dropout sulle attention heads, forse meglio 0.1
self.conv2 = GATv2Conv(..., dropout=0.2)

2. Batch Normalization e Skip Connections
Con grafi di 20.000 nodi, il segnale tende a "diluirsi" o a svanire (vanishing gradient).
La BatchNorm stabilizza l'apprendimento, mentre le skip connections (sommare l'input all'output del layer) aiutano a preservare l'informazione originale del gene.

- Skip Connections (ResNet-style):  x = x + identity permette al modello di "decidere" quanto usare dell'output del layer corrente
    e quanto mantenere dell'input precedente. Se un layer non è utile, il modello può semplicemente azzerarne i pesi e far passare l'identità.

- Proiezioni Lineari: Poiché l'input ha 5 feature e l'uscita del primo layer ne ha 64 * heads, non puoi sommarli direttamente.
    skip_conn_projection serve a portare l'input alla stessa "forma" dell'output per permettere la somma.


3. Pooling Evoluto
Il global_mean_pool fa la media di tutti i geni. Ma in oncologia, spesso sono pochi geni chiave a determinare il fenotipo.
Usare una combinazione di mean e max pooling (o un GlobalAttention) aiuta a catturare sia il segnale globale che i picchi di espressione/mutazione.

Tuttavia, la scelta tra Global Attention e Mean+Max Pooling non è solo tecnica, ma dipende da cosa vuoi che il modello "impari":
- Concatenazione Mean + Max Pooling (Soluzione Robusta)
    Questa è la soluzione più comune in bioinformatica.
    Mean Pooling: Cattura lo stato metabolico/trascrizionale "medio" del tumore.
    Max Pooling: Identifica i "punti caldi" (es. un singolo gene con un'espressione altissima o una mutazione critica).
    Pro: Molto stabile, non aggiunge parametri da addestrare, cattura segnali contrastanti.

- Global Attention Pooling (Soluzione Sofisticata)
    Questa tecnica usa un piccolo network neurale separato per assegnare un punteggio di importanza a ogni gene prima di sommarli. In pratica, il modello decide quali geni sono "biomarcatori" più rilevanti per la classificazione LUAD/LUSC.
    Pro: Potenzialmente più preciso; permette di fare interpretabilità (puoi estrarre i pesi per vedere quali geni il modello considera importanti).
    Contro: Rende il training più instabile e richiede più dati per non andare in overfitting.


4. Struttura del Classifier
Passare da 64 canali direttamente alle 2 classi è un salto brusco. Un piccolo MLP alla fine aiuta a distillare meglio il vettore del paziente.

"Dense" Classifier

self.classifier = torch.nn.Linear(hidden_channels * 2, num_classes) è un singolo layer lineare: Linear(128, 2).
Dopo aver condensato 20.000 geni in un vettore da 128 (64 mean + 64 max), un passaggio intermedio aiuta a modellare
interazioni non lineari tra queste feature aggregate:

self.classifier = torch.nn.Sequential(
    torch.nn.Linear(hidden_channels * 2, hidden_channels),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(hidden_channels, num_classes)
)


5. Learning Rate Scheduler
https://medium.com/@theom/a-very-short-visual-introduction-to-learning-rate-schedulers-with-code-189eddffdb00
'''