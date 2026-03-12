import logging

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.loader import DataLoader
from collections import Counter

from PatientGraphDataset import PatientGraphDataset
from models.CancerGNN import CancerGNN
from models.GAT import GAT
from models.MultiModalGNN import MultiModalGNN


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('execution.log', mode='w'),
        logging.StreamHandler()
    ]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info("Device: " + str(device))
torch.cuda.empty_cache()

#model = CancerGNN(num_node_features=5, num_edge_features=3, hidden_channels=64).to(device)
model = GAT(num_node_features=5, num_edge_features=3, num_classes=2, hidden_channels=64).to(device)
#model = MultiModalGNN(num_node_features=5, num_edge_features=3, clinical_input_dim=16, hidden_channels=64).to(device)

file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t').dropna()
patient_split_df = pd.read_csv('files/clinical/patient_split_cleaned.csv')

file_mapping_df_train = file_mapping_df[file_mapping_df['case_id'].isin(
    patient_split_df[patient_split_df['split'] == 'train']['cases.case_id'])]

file_mapping_df_test = file_mapping_df[file_mapping_df['case_id'].isin(
    patient_split_df[patient_split_df['split'] == 'test']['cases.case_id'])]

file_mapping_df_val = file_mapping_df[file_mapping_df['case_id'].isin(
    patient_split_df[patient_split_df['split'] == 'val']['cases.case_id'])]

node_map_df = pd.read_csv('downloaded_files/gene_ids_mapped.tsv', sep='\t')
node_map = dict(zip(node_map_df.gene_id, node_map_df.gene_id_mapped))

# graph dataset initialization; if not exists, it gets created
logging.info("Train Dataset init...")
train_dataset = PatientGraphDataset(root='data_graphs_processed_train', file_mapping_df=file_mapping_df_train, node_map=node_map)
logging.info("Test Dataset init...")
test_dataset = PatientGraphDataset(root='data_graphs_processed_test', file_mapping_df=file_mapping_df_test, node_map=node_map)
logging.info("Val Dataset init...")
val_dataset = PatientGraphDataset(root='data_graphs_processed_validation', file_mapping_df=file_mapping_df_val, node_map=node_map)

dataset = train_dataset

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')

print(dataset[0])
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {dataset[0].num_nodes}')
print(f'Number of edges: {dataset[0].num_edges}')
print(f'Average node degree: {dataset[0].num_edges / dataset[0].num_nodes:.2f}')
print(f'Has isolated nodes: {dataset[0].has_isolated_nodes()}')
print(f'Has self-loops: {dataset[0].has_self_loops()}')
print(f'Is undirected: {dataset[0].is_undirected()}')
print()

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

node_feat_scaler = StandardScaler()
edge_attr_scaler = MinMaxScaler()
clinical_feat_scaler = StandardScaler()

# scalers fitted only on test data
for data in train_loader:
    node_feat_scaler.partial_fit(data.x[:, :4].numpy())  # only first 4 cols (5 is binary mask)
    edge_attr_scaler.partial_fit(data.edge_attr[:,2].numpy().reshape(-1,1))  # only cols of number of links
    # TODO clinical_feat_scaler.partial_fit(data.clinical.numpy()) solo pack_years_smoked, tobacco_years, age_at_index


# to make things faster by applying scaler transformations manually:
x_mean = torch.tensor(node_feat_scaler.mean_, dtype=torch.float, device=device)
x_std = torch.tensor(node_feat_scaler.scale_, dtype=torch.float, device=device)
e_min = torch.tensor(edge_attr_scaler.data_min_, dtype=torch.float, device=device)
e_max = torch.tensor(edge_attr_scaler.data_max_, dtype=torch.float, device=device)
clinical_mean = torch.tensor(clinical_feat_scaler.mean_, dtype=torch.float, device=device)
clinical_std = torch.tensor(clinical_feat_scaler.scale_, dtype=torch.float, device=device)

# train loop

lr_min = 0.00001
lr_max = 0.001
max_epochs = 50

optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)  # lr = Learning Rate
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr_min)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
'''
ReduceLROnPlateau: Invece di calare "a prescindere" dal tempo (come fanno Exponential o Cosine), cala solo quando la
Val Loss smette di migliorare. È molto più intelligente per dati biologici dove la velocità di convergenza può variare
tra un run e l'altro.
'''

# different weights to classes based on number of samples
train_labels = [data.y.item() for data in train_dataset]
counts = Counter(train_labels)
# N_tot / (N_classes * N_elem_of_class_n)
w0 = len(train_labels) / (2 * counts[0])
w1 = len(train_labels) / (2 * counts[1])
weights = torch.tensor([w0, w1], dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weights)

logging.info(model)


def train():
    model.train()
    total_loss = 0
    model.zero_grad()

    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        # .transform() works with numpy, library on GPU (not CUDA-efficient)
        #data.x[:, :4] = torch.tensor(node_feat_scaler.transform(data.x[:, :4].numpy())).to(device)
        #data.edge_attr[:,2] = torch.tensor(edge_attr_scaler.transform(data.edge_attr[:,2].numpy().reshape(-1,1))).to(device)

        # StandardScale: (x - mean) / std
        data.x[:, :4] = (data.x[:, :4] - x_mean) / (x_std + 1e-6)
        # TODO data.clinical
        # MinMaxScaler: (x - min) / (max - min)
        data.edge_attr[:,2] = (data.edge_attr[:,2] - e_min) / (e_max - e_min + 1e-6)

        # perform a single forward pass
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        loss = criterion(out, data.y)  # Compute the loss.
        scaled_loss = loss
        scaled_loss.backward()  # Derive gradients.

        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_dataset)


def validate():
    model.eval()
    total_loss = 0
    for data in val_loader:
        data = data.to(device)
        data.x[:, :4] = (data.x[:, :4] - x_mean) / (x_std + 1e-6)
        # TODO data.clinical
        data.edge_attr[:, 2] = (data.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(val_dataset)


def test(loader):
     model.eval()
     correct = 0

     for data in loader:  # Iterate in batches over the training/test dataset.
         data = data.to(device)

         #inefficient
         #data.x[:, :4] = torch.tensor(node_feat_scaler.transform(data.x[:, :4].cpu().numpy())).to(device)
         #data.edge_attr[:, 2:] = torch.tensor(edge_attr_scaler.transform(data.edge_attr[:, 2].cpu().numpy().reshape(-1,1))).to(device)

         data.x[:, :4] = (data.x[:, :4] - x_mean) / (x_std + 1e-6)
         # TODO data.clinical
         data.edge_attr[:, 2] = (data.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

         out = model(data.x, data.edge_index, data.edge_attr, data.batch)

         pred = out.argmax(dim=1)  # Use the class with the highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


best_val_loss = float('inf')

for epoch in range(1, max_epochs + 1):
    train_loss = train()
    val_loss = validate()

    train_acc = test(train_loader)
    val_acc = test(val_loader)
    test_acc = test(test_loader)

    scheduler.step(val_loss)  # update learning rate

    logging.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f},'
                 f' Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    #save best model based on Val Loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_classification_gnn.pth')
        logging.info("--- Found and saved a better model! ---")
