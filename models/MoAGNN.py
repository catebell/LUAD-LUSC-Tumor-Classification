#!/usr/bin/env python3
"""
networks.py
original file took from https://github.com/leebiomicslab/MoAGNN.git
-----------
MoAGNN: GCN + 3×SAGPooling with hierarchical readouts.
Readout uses global max + mean at each pooling stage, then CONCAT T1||T2||T3.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import SAGPooling


class MoAGNN(nn.Module):
    def __init__(self, args):
        """
        Args (expected in `args`):
            num_features: int
            nhid: int
            num_classes: int
            pooling_ratio: float  (SAGPooling ratio P)
            dropout_ratio: float
        """
        super().__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        # ----- Encoder + Pooling blocks -----
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

        # Each readout T_i is [gmp||gap] -> 2*nhid
        # We CONCAT T1||T2||T3 -> 6*nhid
        readout_dim = self.nhid * 2 * 3

        # ----- MLP head -----
        self.lin1 = nn.Linear(readout_dim, self.nhid)
        self.lin2 = nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = nn.Linear(self.nhid // 2, self.num_classes)

    def _stage(self, x, edge_index, batch, conv, pool):
        x = F.relu(conv(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = pool(x, edge_index, None, batch)
        # readout: global max + global mean
        t = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        return x, edge_index, batch, t

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x, edge_index, batch, T1 = self._stage(x, edge_index, batch, self.conv1, self.pool1)
        x, edge_index, batch, T2 = self._stage(x, edge_index, batch, self.conv2, self.pool2)
        x, edge_index, batch, T3 = self._stage(x, edge_index, batch, self.conv3, self.pool3)

        # CONCAT T1||T2||T3  (instead of sum)
        x = torch.cat([T1, T2, T3], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x