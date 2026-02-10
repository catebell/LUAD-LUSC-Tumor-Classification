import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class GIATConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4):
        # Aggregazione "add" per preservare le proprietà di isomorfismo (GIN-style)
        super(GIATConvLayer, self).__init__(aggr='add')

        self.heads = heads
        self.out_channels = out_channels

        # Trasformazioni per i dati multi-omici (RNA, Met, CNV)
        self.lin_q = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.lin_k = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.lin_v = nn.Linear(in_channels, out_channels * heads, bias=False)

        # Vettore di attenzione per il calcolo dello score alpha
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        # x: [N, features_omiche] (RNA, Met, CNV concatenati)
        H, C = self.heads, self.out_channels

        # Generazione Q, K, V
        q = self.lin_q(x).view(-1, H, C)
        k = self.lin_k(x).view(-1, H, C)
        v = self.lin_v(x).view(-1, H, C)

        # Inizia il Message Passing (Propagazione nel grafo delle proteine/geni)
        return self.propagate(edge_index, q=q, k=k, v=v)

    def message(self, q_i, k_j, v_j, index, ptr, size_i):
        # Calcolo dell'attenzione tra Gene i e Gene j
        # Uniamo la Query del ricevente e la Key del mittente
        alpha = torch.cat([q_i, k_j], dim=-1)
        alpha = (alpha * self.att).sum(dim=-1)

        # Normalizzazione: quanto è importante il gene j per il gene i?
        alpha = softmax(alpha, index, ptr, size_i)

        # Restituiamo il segnale omico pesato
        return v_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        # Dopo l'aggregazione, concateniamo le teste per il pooling finale (G)
        return aggr_out.view(-1, self.heads * self.out_channels)