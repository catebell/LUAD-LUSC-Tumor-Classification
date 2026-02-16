import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class GIATConvLayer(MessagePassing):
    """
    Graph Isomorphism + Attention Layer (multi-head)
    Input: node features x [num_nodes, in_channels], edge_index [2, num_edges]
    Output: updated node features [num_nodes, heads*out_channels]
    """
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__(aggr='add')  # GIN-style aggregation
        self.heads = heads
        self.out_channels = out_channels

        # Linear projections for multi-head attention
        self.lin_q = nn.Linear(in_channels, out_channels*heads, bias=False)
        self.lin_k = nn.Linear(in_channels, out_channels*heads, bias=False)
        self.lin_v = nn.Linear(in_channels, out_channels*heads, bias=False)

        # Learnable attention vector per head
        self.att = nn.Parameter(torch.Tensor(1, heads, 2*out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        H, C = self.heads, self.out_channels
        q = self.lin_q(x).view(-1, H, C)
        k = self.lin_k(x).view(-1, H, C)
        v = self.lin_v(x).view(-1, H, C)
        return self.propagate(edge_index, q=q, k=k, v=v)

    def message(self, q_i, k_j, v_j, index, ptr, size_i):
        # Compute attention between target node i and source node j
        alpha = torch.cat([q_i, k_j], dim=-1)
        alpha = (alpha*self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        return v_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        # Concat heads
        return aggr_out.view(-1, self.heads*self.out_channels)