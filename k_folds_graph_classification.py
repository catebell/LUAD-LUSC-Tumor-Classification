import logging

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from collections import Counter

from PatientGraphDataset import PatientGraphDataset
from models.CancerGNN import CancerGNN
from models.GAT import GAT
from models.MLP import MLP
from models.MultiModalGNN import MultiModalGNN

import warnings
warnings.filterwarnings("ignore")

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

file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t').dropna()
patient_split_df = pd.read_csv('files/clinical/patient_split_cleaned.csv')

file_mapping_df_train = file_mapping_df[file_mapping_df['case_id'].isin(
    patient_split_df[patient_split_df['split'] == 'train']['cases.case_id'])]

file_mapping_df_test = file_mapping_df[file_mapping_df['case_id'].isin(
    patient_split_df[patient_split_df['split'] == 'test']['cases.case_id'])]

file_mapping_df_val = file_mapping_df[file_mapping_df['case_id'].isin(
    patient_split_df[patient_split_df['split'] == 'val']['cases.case_id'])]

node_map_df = pd.read_csv('files/clinical/gene_ids_mapped.tsv', sep='\t')
node_map = dict(zip(node_map_df.gene_id, node_map_df.gene_id_mapped))

# graph dataset initialization; if not exists, it gets created
logging.info("Train Dataset init...")
train_dataset = PatientGraphDataset(root='data_graphs_processed_train', file_mapping_df=file_mapping_df_train, node_map=node_map)
logging.info("Test Dataset init...")
test_dataset = PatientGraphDataset(root='data_graphs_processed_test', file_mapping_df=file_mapping_df_test, node_map=node_map)
logging.info("Val Dataset init...")
val_dataset = PatientGraphDataset(root='data_graphs_processed_validation', file_mapping_df=file_mapping_df_val, node_map=node_map)

full_train_dataset = train_dataset + val_dataset

y_labels = []
for i in range(len(full_train_dataset)):
    y_labels.append(full_train_dataset[i].y.item())
y_labels = np.array(y_labels)

k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

def get_optimizer(model, lr, default):
    """To give different parameters to the two different sub-models."""
    mlp_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'mlp' in name or 'classifier' in name:
            mlp_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.Adam([
        {'params': other_params, 'weight_decay': default},
        {'params': mlp_params, 'weight_decay': 1e-2}
    ], lr=lr)

fold_results = []

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data_copy = data.clone() # to prevent inplace modifications through folds/loops
        data_copy = data_copy.to(device)

        # StandardScale: (x - mean) / std
        data_copy.x[:, :4] = (data_copy.x[:, :4] - x_mean) / (x_std + 1e-6)
        data_copy.clinical[:, :3] = (data_copy.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
        # MinMaxScaler: (x - min) / (max - min)
        data_copy.edge_attr[:, 2] = (data_copy.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

        optimizer.zero_grad()

        if model.__class__ == CancerGNN or model.__class__ == GAT:
            out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.batch)  # just graph
        elif model.__class__ == MLP:
            out = model(data_copy.clinical) # just clinical features
        else:  # MultiModalGNN
            out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.clinical, data_copy.batch) # both

        loss = criterion(out, data_copy.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_copy.num_graphs
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for data in loader:
        data_copy = data.clone()
        data_copy = data_copy.to(device)

        data_copy.x[:, :4] = (data_copy.x[:, :4] - x_mean) / (x_std + 1e-6)
        data_copy.clinical[:, :3] = (data_copy.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
        data_copy.edge_attr[:, 2] = (data_copy.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

        if model.__class__ == CancerGNN or model.__class__ == GAT:
            out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.batch)  # just graph
        elif model.__class__ == MLP:
            out = model(data_copy.clinical)  # just clinical features
        else:  # MultiModalGNN
            out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.clinical,
                        data_copy.batch)  # both

        loss = criterion(out, data_copy.y)
        total_loss += loss.item() * data_copy.num_graphs
    return total_loss / len(loader.dataset)


def test(model, loader):
    model.eval()
    correct = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        data_copy = data.clone()
        data_copy = data_copy.to(device)

        data_copy.x[:, :4] = (data_copy.x[:, :4] - x_mean) / (x_std + 1e-6)
        data_copy.clinical[:, :3] = (data_copy.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
        data_copy.edge_attr[:, 2] = (data_copy.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

        if model.__class__ == CancerGNN or model.__class__ == GAT:
            out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.batch)  # just graph
        elif model.__class__ == MLP:
            out = model(data_copy.clinical)  # just clinical features
        else:  # MultiModalGNN
            out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.clinical,
                        data_copy.batch)  # both

        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        correct += int((pred == data_copy.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
max_epochs = 100

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_labels)), y_labels)):
    logging.info(f"--- FOLD {fold + 1}/{k_folds} ---\n")

    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=True)

    # re-initialization for current fold
    #model = CancerGNN(num_node_features=5, num_edge_features=3, hidden_channels=64).to(device)
    #model = GAT(num_node_features=5, num_edge_features=3, num_classes=2, hidden_channels=64).to(device)
    #model = MLP(num_patient_features=53).to(device)
    model = MultiModalGNN(num_node_features=5, num_edge_features=3, clinical_input_dim=53, hidden_channels=64).to(device)
    logging.info(model)

    optimizer = get_optimizer(model, 0.001, 1e-4)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # different weights to classes based on number of samples
    fold_train_labels = y_labels[train_idx]
    counts = Counter(fold_train_labels)
    # class n weight = N_tot / (N_classes * N_elem_of_class_n)
    w0 = len(fold_train_labels) / (2 * counts[0])
    w1 = len(fold_train_labels) / (2 * counts[1])
    weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    node_feat_scaler = StandardScaler()
    edge_attr_scaler = MinMaxScaler()
    clinical_feat_scaler = StandardScaler()

    # scalers fitted only on test data
    for data in train_loader:
        node_feat_scaler.partial_fit(data.x[:, :4].numpy())  # only first 4 cols (5 is binary mask)
        edge_attr_scaler.partial_fit(data.edge_attr[:, 2].numpy().reshape(-1, 1))  # only cols of number of links
        clinical_feat_scaler.partial_fit(
            data.clinical[:, :3].numpy())  # only pack_years_smoked, tobacco_years, age_at_index

    # to make things faster by applying scaler transformations manually:
    x_mean = torch.tensor(node_feat_scaler.mean_, dtype=torch.float, device=device)
    x_std = torch.tensor(node_feat_scaler.scale_, dtype=torch.float, device=device)
    e_min = torch.tensor(edge_attr_scaler.data_min_, dtype=torch.float, device=device)
    e_max = torch.tensor(edge_attr_scaler.data_max_, dtype=torch.float, device=device)
    clinical_mean = torch.tensor(clinical_feat_scaler.mean_, dtype=torch.float, device=device)
    clinical_std = torch.tensor(clinical_feat_scaler.scale_, dtype=torch.float, device=device)

    best_fold_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        train_acc = test(model, train_loader)
        val_acc = test(model, val_loader)

        scheduler.step(val_loss)

        if val_loss < best_fold_val_loss:
            early_stopping_counter = 0
            best_fold_val_loss = val_loss

            logging.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

            torch.save(model.state_dict(), f'model_fold_{fold + 1}.pth')
            logging.info("--- Found and saved a better model! ---\n")

        if early_stopping_counter > 20:
            logging.info("--- Stopping training due to early stopping ---\n")
            break
        else:
            early_stopping_counter += 1


    # best model found in this fold tested on the fixed test set
    model.load_state_dict(torch.load(f'model_fold_{fold + 1}.pth'))
    fold_test_acc = test(model, test_loader)
    fold_results.append(fold_test_acc)
    logging.info(f"Fold {fold + 1} Test Acc: {fold_test_acc:.4f}\n")

logging.info(f"Mean Test Accuracy: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}\n")
