import logging

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.loader import DataLoader
from collections import Counter

from PatientGraphDataset import PatientGraphDataset
from models.GAT import GAT
from models.GNN import GNN
from models.CancerGNN import CancerGNN

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

file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t').dropna()
patient_split_df = pd.read_csv('files/clinical/patient_split_cleaned.csv')

file_mapping_df_train = file_mapping_df[file_mapping_df['case_id'].isin(
    patient_split_df[patient_split_df['split'] == 'train']['cases.case_id'])]

file_mapping_df_test = file_mapping_df[file_mapping_df['case_id'].isin(
    patient_split_df[patient_split_df['split'] == 'test']['cases.case_id'])]

file_mapping_df_val = file_mapping_df[file_mapping_df['case_id'].isin(
    patient_split_df[patient_split_df['split'] == 'val']['cases.case_id'])]

# dataset initialization; if not exists, it gets created
logging.info("Train Dataset init...")
train_dataset = PatientGraphDataset(root='data_graphs_processed_train', file_mapping_df=file_mapping_df_train)
logging.info("Test Dataset init...")
test_dataset = PatientGraphDataset(root='data_graphs_processed_test', file_mapping_df=file_mapping_df_test)
logging.info("Val Dataset init...")
val_dataset = PatientGraphDataset(root='data_graphs_processed_validation', file_mapping_df=file_mapping_df_val)

node_feat_scaler = StandardScaler()
edge_attr_scaler = MinMaxScaler()

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

'''
# manual split over whole dataset (80% train, 20% test)
torch.manual_seed(12345)
dataset = dataset.shuffle()
train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]
'''

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# scalers fitted only o n test data
for graph in train_loader:
    node_feat_scaler.partial_fit(graph.x[:, :4].numpy())  # only first 4 cols (5 is binary mask)
    edge_attr_scaler.partial_fit(graph.edge_attr[:,2].numpy().reshape(-1,1))  # only cols of number of links

# to make things faster by applying scaler transformations manually:
x_mean = torch.tensor(node_feat_scaler.mean_, dtype=torch.float, device=device)
x_std = torch.tensor(node_feat_scaler.scale_, dtype=torch.float, device=device)
e_min = torch.tensor(edge_attr_scaler.data_min_, dtype=torch.float).to(device)
e_max = torch.tensor(edge_attr_scaler.data_max_, dtype=torch.float).to(device)

'''
for step, data in enumerate(train_loader):
    print(f'\nStep {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
'''

# train loop

#model = GNN(num_node_features=5, num_classes=2, hidden_channels=64).to(device)
model = CancerGNN(num_node_features=5, num_edge_features=3, hidden_channels=64).to(device)
#model = GAT(num_node_features=5, num_edge_features=3, num_classes=2, hidden_channels=64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # lr = Learning Rate

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

    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)

        # .transform() works with numpy, library on GPU (not CUDA-efficient)
        #data.x[:, :4] = torch.tensor(node_feat_scaler.transform(data.x[:, :4].numpy())).to(device)
        #data.edge_attr[:,2:] = torch.tensor(edge_attr_scaler.transform(data.edge_attr[:,2].numpy().reshape(-1,1))).to(device)

        # StandardScale: (x - mean) / std
        data.x[:, :4] = (data.x[:, :4] - x_mean) / (x_std + 1e-6)
        # MinMaxScaler: (x - min) / (max - min)
        data.edge_attr[:,2:] = (data.edge_attr[:,2:] - e_min) / (e_max - e_min + 1e-6)

        # perform a single forward pass
        #out = model(data.x, data.edge_index, data.batch)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        #out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        partial_loss = criterion(out, data.y)  # Compute the loss.
        partial_loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        total_loss += partial_loss.item() * data.num_graphs
    return total_loss / len(train_dataset)


def test(loader):
     model.eval()
     correct = 0

     for data in loader:  # Iterate in batches over the training/test dataset.
         data = data.to(device)

         #inefficient
         #data.x[:, :4] = torch.tensor(node_feat_scaler.transform(data.x[:, :4].cpu().numpy())).to(device)
         #data.edge_attr[:, 2:] = torch.tensor(edge_attr_scaler.transform(data.edge_attr[:, 2].cpu().numpy().reshape(-1,1))).to(device)

         # StandardScale: (x - mean) / std
         data.x[:, :4] = (data.x[:, :4] - x_mean) / (x_std + 1e-6)
         # MinMaxScaler: (x - min) / (max - min)
         data.edge_attr[:, 2:] = (data.edge_attr[:, 2:] - e_min) / (e_max - e_min + 1e-6)

         #out = model(data.x, data.edge_index, data.batch)
         out = model(data.x, data.edge_index, data.edge_attr, data.batch)
         #out = model(data.x, data.edge_index, data.edge_attr, data.batch)

         pred = out.argmax(dim=1)  # Use the class with the highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

