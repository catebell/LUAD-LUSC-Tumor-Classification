import torch
from torch_geometric.nn import GATv2Conv, global_mean_pool

# GATv2Conv: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GATv2Conv.html


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_edge_features, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_channels, edge_dim=num_edge_features,
                               heads=1)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)  # LUAD vs LUSC

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()

        x = global_mean_pool(x, batch)  # graph to single vector per patient

        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        return self.classifier(x)