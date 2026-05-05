import logging
import argparse
import os
import config
import pandas as pd
import torch
import importlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.loader import DataLoader
from collections import Counter
from PatientGraphDataset import PatientGraphDataset
from models.GINEConvGNN import GINEConvGNN
from models.GAT import GAT
from models.MLP import MLP
from models.MultiModalGNN import MultiModalGNN

import warnings
warnings.filterwarnings("ignore")  # to temporarily not see warnings

def get_available_datasets():
    """Return all subfolders inside original_dataset/ so the dataset available"""
    if not os.path.isdir(config.DATASET):
        return []

    return sorted([
        folder
        for folder in os.listdir(config.DATASET)
        if os.path.isdir(os.path.join(config.DATASET, folder))
    ])

def get_available_models():
    """Return all the models available in the models/ folder"""
    models_dir = "models"

    if not os.path.isdir(models_dir):
        return []

    models = []

    for file in os.listdir(models_dir):
        if file.endswith(".py") and file != "__init__.py":
            models.append(os.path.splitext(file)[0])

    return sorted(models)

def parse_args():
    """ Parse command line arguments or set defaul values for model_name and tumor dataset """
    datasets = get_available_datasets()
    models = get_available_models()

    parser = argparse.ArgumentParser(description="Training GNN")

    parser.add_argument(
        "--dataset",
        type=str,
        default=config.tumor,
        choices=datasets,
        help=f"Available datasets: {', '.join(datasets)}"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=config.model,
        choices=models,
        help=f"Available models: {', '.join(models)}"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    global model
    model = load_model(args.model_name, device)
    train_and_save_model(args.dataset, model, args.model_name)

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

def load_model(model_name, device):
    module = importlib.import_module(f"models.{model_name}")
    model_class = getattr(module, model_name)

    clinical_df = pd.read_csv(f'files/{config.tumor}/clinical/features_considered.tsv', sep='\t')
    num_patient_features = len(clinical_df.columns.tolist()[2:])

    if model_name in ["GINEConvGNN", "GAT"]:
        model = model_class(num_node_features=5, num_edge_features=3, hidden_channels=64, num_classes=2)
    elif model_name == "MLP":
        model = model_class(num_patient_features=num_patient_features, num_classes=2)
    elif model_name == "MultiModalGNN":
        model = model_class(num_node_features=5, num_edge_features=3, clinical_input_dim=num_patient_features, hidden_channels=64, num_classes=2)
    else:
        model = model_class()

    return model.to(device)


def prepare_datasets(dataset_name):
    file_mapping_df = pd.read_csv(f'{config.FILES}/{dataset_name}/clinical/file_case_mapping.tsv', sep='\t').dropna()
    patient_split_df = pd.read_csv(f'{config.FILES}/{dataset_name}/clinical/patient_split_cleaned.csv')

    file_mapping_df_train = file_mapping_df[file_mapping_df['case_id'].isin(
        patient_split_df[patient_split_df['split'] == 'train']['cases.case_id'])]

    file_mapping_df_test = file_mapping_df[file_mapping_df['case_id'].isin(
        patient_split_df[patient_split_df['split'] == 'test']['cases.case_id'])]

    file_mapping_df_val = file_mapping_df[file_mapping_df['case_id'].isin(
        patient_split_df[patient_split_df['split'] == 'val']['cases.case_id'])]

    # graph dataset initialization; if not exists, it gets created
    logging.info("Train Dataset init...")
    train_dataset = PatientGraphDataset(
        root=os.path.join('data_graphs_processed', dataset_name, 'data_graphs_processed_train'),
        file_mapping_df=file_mapping_df_train,
        dataset=dataset_name
    )
    logging.info("Test Dataset init...")
    test_dataset = PatientGraphDataset(
        root=os.path.join('data_graphs_processed', dataset_name, 'data_graphs_processed_test'),
        file_mapping_df=file_mapping_df_test,
        dataset=dataset_name
    )
    logging.info("Val Dataset init...")
    val_dataset = PatientGraphDataset(
        root=os.path.join('data_graphs_processed', dataset_name, 'data_graphs_processed_validation'),
        file_mapping_df=file_mapping_df_val,
        dataset=dataset_name
    )
    return train_dataset, test_dataset, val_dataset


def train_and_save_model(dataset_name, model, model_name):
    global train_loader, test_loader, val_loader
    global x_mean, x_std, e_min, e_max, clinical_mean, clinical_std
    global optimizer, scheduler, criterion

    train_dataset, test_dataset, val_dataset = prepare_datasets(dataset_name)

    dataset = train_dataset

    print()
    print(f'Train Dataset: {dataset}:')
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
        node_feat_scaler.partial_fit(data.x[:,:4].numpy())  # only first 4 cols (5 is binary mask)
        edge_attr_scaler.partial_fit(data.edge_attr[:,2].numpy().reshape(-1,1))  # only cols of number of links
        clinical_feat_scaler.partial_fit(data.clinical[:,:3].numpy()) # only pack_years_smoked, tobacco_years, age_at_index

    # to make things faster by applying scaler transformations manually:
    x_mean = torch.tensor(node_feat_scaler.mean_, dtype=torch.float, device=device)
    x_std = torch.tensor(node_feat_scaler.scale_, dtype=torch.float, device=device)
    e_min = torch.tensor(edge_attr_scaler.data_min_, dtype=torch.float, device=device)
    e_max = torch.tensor(edge_attr_scaler.data_max_, dtype=torch.float, device=device)
    clinical_mean = torch.tensor(clinical_feat_scaler.mean_, dtype=torch.float, device=device)
    clinical_std = torch.tensor(clinical_feat_scaler.scale_, dtype=torch.float, device=device)

    # train loop
    max_epochs = 100

    optimizer = get_optimizer(model, 0.001, 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # different weights to classes based on number of samples
    train_labels = [data.y.item() for data in train_dataset]
    counts = Counter(train_labels)

    criterion = torch.nn.CrossEntropyLoss()

    logging.info(str(model) + '\n')

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train(model)
        val_loss = validate(model)

        train_acc = test(model, train_loader)
        val_acc = test(model, val_loader)
        test_acc = test(model, test_loader)

        scheduler.step(val_loss)  # update learning rate

        logging.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f},'
                     f' Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

        #save best model based on Val Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0

            torch.save(model.state_dict(), f'best_{model_name}_{dataset_name}.pth')
            logging.info("--- Found and saved a better model! ---\n")

        if early_stopping_counter > 20:
            logging.info("--- Stopping training due to early stopping ---")
            break
        else:
            early_stopping_counter += 1


max_epochs = 100

def get_optimizer(model, lr, default):
    """To assign different optimizer params to different sections of the model."""
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


def train(model):
    model.train()
    total_loss = 0
    model.zero_grad()

    for og in train_loader:  # Iterate in batches over the training dataset.
        data = og.clone()
        data = data.to(device)

        # StandardScale: (x - mean) / std
        data.x[:, :4] = (data.x[:, :4] - x_mean) / (x_std + 1e-6)
        data.clinical[:, :3] = (data.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
        # MinMaxScaler: (x - min) / (max - min)
        data.edge_attr[:,2] = (data.edge_attr[:,2] - e_min) / (e_max - e_min + 1e-6)

        # perform a single forward pass
        if model.__class__ == GINEConvGNN or model.__class__ == GAT:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # just graph
        elif model.__class__ == MLP:
            out = model(data.clinical) # just clinical features
        else:  # MultiModalGNN
            out = model(data.x, data.edge_index, data.edge_attr, data.clinical, data.batch) # both

        loss = criterion(out, data.y)
        loss.backward()

        optimizer.step()  # parameters update based on gradients
        optimizer.zero_grad()

        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


def validate(model):
    model.eval()
    total_loss = 0
    for og in val_loader:
        data = og.clone()
        data = data.to(device)
        data.x[:, :4] = (data.x[:, :4] - x_mean) / (x_std + 1e-6)
        data.clinical[:, :3] = (data.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
        data.edge_attr[:, 2] = (data.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

        with torch.no_grad():
            if model.__class__ == GINEConvGNN or model.__class__ == GAT:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # just graph
            elif model.__class__ == MLP:
                out = model(data.clinical)  # just clinical features
            else:  # MultiModalGNN
                out = model(data.x, data.edge_index, data.edge_attr, data.clinical, data.batch)  # both

            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(val_loader.dataset)


def test(model, loader):
     model.eval()
     correct = 0

     for og in loader:  # iterate in batches over the training/test dataset.
         data = og.clone()
         data = data.to(device)
         data.x[:, :4] = (data.x[:, :4] - x_mean) / (x_std + 1e-6)
         data.clinical[:, :3] = (data.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
         data.edge_attr[:, 2] = (data.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

         if model.__class__ == GINEConvGNN or model.__class__ == GAT:
             out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # just graph
         elif model.__class__ == MLP:
             out = model(data.clinical)  # just clinical features
         else:  # MultiModalGNN
             out = model(data.x, data.edge_index, data.edge_attr, data.clinical, data.batch)  # both

         pred = out.argmax(dim=1)  # class with the highest probability
         correct += int((pred == data.y).sum())  # check against ground-truth labels
     return correct / len(loader.dataset)


if __name__ == "__main__":
    main()