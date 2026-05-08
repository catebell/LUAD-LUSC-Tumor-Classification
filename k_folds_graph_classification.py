import logging
import os
import argparse
import importlib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, average_precision_score, confusion_matrix
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import global_mean_pool

from PatientGraphDataset import PatientGraphDataset
from models.BasicGraphConvGNN import BasicGraphConvGNN
from models.GINEConvGNN import GINEConvGNN
from models.GAT import GAT
from models.MLP import MLP
from models.MultiModalGNN import MultiModalGNN
import config

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


def get_available_datasets():
    """Return all subfolders inside {config.DATASET}/ so the dataset available"""
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


def load_model(model_name, num_classes, device):
    module = importlib.import_module(f"models.{model_name}")
    model_class = getattr(module, model_name)

    clinical_df = pd.read_csv(f'{config.FILES}/{dataset_name}/clinical/features_considered.tsv', sep='\t')
    num_patient_features = len(clinical_df.columns.tolist()[2:])

    if model_name in ["GINEConvGNN", "GAT"]:
        model = model_class(num_node_features=5, num_edge_features=3, hidden_channels=64, num_classes=num_classes)
    elif model_name == "BasicGraphConvGNN":
        model = model_class(num_node_features=5, hidden_channels=64, num_classes=num_classes)
    elif model_name == "MLP":
        #model = model_class(num_patient_features=num_patient_features, hidden_channels = 64, num_classes=num_classes)
        model = model_class(num_patient_features=5, hidden_channels=64, num_classes=num_classes)
    elif model_name == "MultiModalGNN":
        model = model_class(num_node_features=5, num_edge_features=3, clinical_input_dim=num_patient_features,
                            hidden_channels=64, num_classes=num_classes)
    else:
        model = model_class()

    return model.to(device)


args = parse_args()
model_name = args.model_name
dataset_name = args.dataset

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
train_dataset = PatientGraphDataset(root=f'data_graphs_processed/{dataset_name}/data_graphs_processed_train', file_mapping_df=file_mapping_df_train)
logging.info("Test Dataset init...")
test_dataset = PatientGraphDataset(root=f'data_graphs_processed/{dataset_name}/data_graphs_processed_test', file_mapping_df=file_mapping_df_test)
logging.info("Val Dataset init...")
val_dataset = PatientGraphDataset(root=f'data_graphs_processed/{dataset_name}/data_graphs_processed_validation', file_mapping_df=file_mapping_df_val)

full_train_dataset = train_dataset + val_dataset

y_labels = []
for i in range(len(full_train_dataset)):
    y_labels.append(full_train_dataset[i].y.item())
y_labels = np.array(y_labels)

unique_labels = np.unique(y_labels)
logging.info(f"Unique labels found in dataset: {unique_labels}")
num_classes = len(unique_labels)

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
        data_copy = data.clone()  # to prevent inplace modifications through folds/loops
        data_copy = data_copy.to(device)

        # StandardScale: (x - mean) / std
        data_copy.x[:, :4] = (data_copy.x[:, :4] - x_mean) / (x_std + 1e-6)
        data_copy.clinical[:, :3] = (data_copy.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
        # MinMaxScaler: (x - min) / (max - min)
        data_copy.edge_attr[:, 2] = (data_copy.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

        optimizer.zero_grad()

        if model.__class__ == GINEConvGNN or model.__class__ == GAT:
            out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.batch)  # just graph
        elif model.__class__ == BasicGraphConvGNN:
            out = model(data_copy.x, data_copy.edge_index, data_copy.batch)
        elif model.__class__ == MLP:
            # out = model(data_copy.clinical)  # just clinical features
            # need to aggregate graph nodes
            x_pooled = global_mean_pool(data_copy.x, data_copy.batch)
            out = model(x_pooled)
        else:  # MultiModalGNN
            out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.clinical,
                        data_copy.batch)  # both

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

        with torch.no_grad():
            if model.__class__ == GINEConvGNN or model.__class__ == GAT:
                out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.batch)  # just graph
            elif model.__class__ == BasicGraphConvGNN:
                out = model(data_copy.x, data_copy.edge_index, data_copy.batch)
            elif model.__class__ == MLP:
                # out = model(data_copy.clinical)  # just clinical features
                # need to aggregate graph nodes
                x_pooled = global_mean_pool(data_copy.x, data_copy.batch)
                out = model(x_pooled)
            else:  # MultiModalGNN
                out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.clinical,
                            data_copy.batch)  # both

            loss = criterion(out, data_copy.y)
            total_loss += loss.item() * data_copy.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data in loader:  # iterate in batches.
            data_copy = data.clone().to(device)

            data_copy.x[:, :4] = (data_copy.x[:, :4] - x_mean) / (x_std + 1e-6)
            data_copy.clinical[:, :3] = (data_copy.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
            data_copy.edge_attr[:, 2] = (data_copy.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

            if model.__class__ == GINEConvGNN or model.__class__ == GAT:
                out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.batch)  # just graph
            elif model.__class__ == BasicGraphConvGNN:
                out = model(data_copy.x, data_copy.edge_index, data_copy.batch)
            elif model.__class__ == MLP:
                #out = model(data_copy.clinical)  # just clinical features
                # need to aggregate graph nodes
                x_pooled = global_mean_pool(data_copy.x, data_copy.batch)
                out = model(x_pooled)
            else:  # MultiModalGNN
                out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.clinical,
                            data_copy.batch)  # both

            probs = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1)  # use the class with the highest probability

            all_targets.extend(data_copy.y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    if len(unique_labels) == 2:
        auc_val = roc_auc_score(all_targets, all_probs[:, 1])
    elif len(set(all_targets)) > 2:
        auc_val = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    else:
        auc_val = 0.0

    metrics = {
        'acc': np.mean(all_preds == all_targets),
        'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        # Muliti-Class AUC --> OvR strategy (One-vs-Rest)
        'auc': auc_val,
    }

    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)

    return metrics, report, all_targets, all_preds


def plot_final_confusion_matrix(model, all_targets, all_preds):
    features_df = pd.read_csv(f'{config.FILES}/{dataset_name}/clinical/features_considered.tsv', sep='\t')
    classes = sorted(features_df["project.project_id"].dropna().unique())
    # for project.project_id, remap tumor class to numbers
    class_mapping = {label: i for i, label in enumerate(classes)}
    class_mapping_inv = {v: k for k, v in class_mapping.items()}

    filename = f'confusion_matrix_{model._get_name()}.png'

    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, cmap='Blues',
                xticklabels=[f'Pred {class_mapping_inv.get(l)}' for l in unique_labels],
                yticklabels=[f'True {class_mapping_inv.get(l)}' for l in unique_labels])

    plt.title('Global Confusion Matrix (Sum and avg for all folds)')
    plt.ylabel('Real Label')
    plt.xlabel('Model Prediction')

    plt.savefig(f'models/final_metrics_k_fold/{dataset_name}/{filename}', bbox_inches='tight', dpi=300)
    plt.show()
    logging.info(f"Confusion Matrix saved.")

max_epochs = 100
cumulative_targets = []
cumulative_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_labels)), y_labels)):
    logging.info(f"--- FOLD {fold + 1}/{k_folds} ---\n")

    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # re-initialization for current fold
    model = load_model(model_name, len(unique_labels), device)
    logging.info(model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer = get_optimizer(model, 0.001, 1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # different weights to classes based on number of samples
    '''
    train_labels = [data.y.item() for data in train_dataset]
    counts = Counter(train_labels)
    weights = [0] * len(counts)
    for i in range(len(counts)):
        # N_tot / (N_classes * N_elem_of_class_n)
        weights[i] = len(train_labels) / (len(counts) * counts[i])
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    '''
    criterion = torch.nn.CrossEntropyLoss()

    node_feat_scaler = StandardScaler()
    edge_attr_scaler = MinMaxScaler()
    clinical_feat_scaler = StandardScaler()

    # scalers fitted only on evaluate data
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

        train_metrics, train_report, _, _ = evaluate(model, train_loader)
        val_metrics, val_report, _, _ = evaluate(model, val_loader)

        scheduler.step(val_loss)

        if val_loss < best_fold_val_loss:
            early_stopping_counter = 0
            best_fold_val_loss = val_loss

            logging.info(
                f'Epoch: {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n'
                f'Train -> Acc: {train_metrics["acc"]:.4f}, F1: {train_metrics["f1"]:.4f}, AUC: {train_metrics["auc"]:.4f}\n'
                f'Val -> Acc: {val_metrics["acc"]:.4f}, F1: {val_metrics["f1"]:.4f}, AUC: {val_metrics["auc"]:.4f}\n'
                #f"Val Class 0 (LUAD) Precision: {val_report['0']['precision']:.4f} | "
                #f"Val Class 1 (LUSC) Precision: {val_report['1']['precision']:.4f}"
            )

            torch.save(model.state_dict(), f'{dataset_name}_{model_name}_fold_{fold + 1}.pth')
            logging.info("--- Found and saved a better model! ---\n")

        if early_stopping_counter > 20:
            logging.info("--- Stopping training due to early stopping ---\n")
            break
        else:
            early_stopping_counter += 1

    # best model found in this fold tested on the fixed evaluate set
    model.load_state_dict(torch.load(f'{dataset_name}_{model_name}_fold_{fold + 1}.pth'))
    fold_test_metrics, test_report, all_targets, all_preds = evaluate(model, test_loader)
    fold_results.append(fold_test_metrics)
    cumulative_targets.extend(all_targets)
    cumulative_preds.extend(all_preds)

    logging.info(
        f"Fold {fold + 1} Test Acc: {fold_test_metrics['acc']:.4f} | AUC: {fold_test_metrics['auc']:.4f}\n"
        f"Precision: {fold_test_metrics['precision']:.4f} | Recall: {fold_test_metrics['recall']:.4f}\n"
        #f"Class 0 (LUAD) Precision: {test_report['0']['precision']:.4f} | "
        #f"Class 1 (LUSC) Precision: {test_report['1']['precision']:.4f}\n"
    )

avg_acc = np.mean([metrics['acc'] for metrics in fold_results])
dev_std = np.std([metrics['acc'] for metrics in fold_results])
avg_f1 = np.mean([metrics['f1'] for metrics in fold_results])
avg_auc = np.mean([metrics['auc'] for metrics in fold_results])
avg_precision = np.mean([metrics['precision'] for metrics in fold_results])
avg_recall = np.mean([metrics['recall'] for metrics in fold_results])

logging.info(f"Mean Test Accuracy: {avg_acc:.4f} +/- {dev_std:.4f}")
logging.info(f"F1: {avg_f1:.4f}, AUC: {avg_auc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

results_for_csv = []

for i, res in enumerate(fold_results):
    row = {
        'fold': i + 1,
        'accuracy': res['acc'],
        'precision': res['precision'],
        'recall': res['recall'],
        'f1_score': res['f1'],
        'auc_roc': res['auc'],
    }
    results_for_csv.append(row)

df_results = pd.DataFrame(results_for_csv)

mean_row = df_results.mean(numeric_only=True).to_dict()
mean_row['fold'] = 'MEAN'
df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)

std_row = df_results.iloc[:-1].std(numeric_only=True).to_dict()
std_row['fold'] = 'STD_DEV'
df_results = pd.concat([df_results, pd.DataFrame([std_row])], ignore_index=True)

path = Path(f"models/final_metrics_k_fold/{dataset_name}")
path.mkdir(parents=True, exist_ok=True)
csv_filename = f'{path}/{model._get_name()}_final_metrics_comparison.csv'

df_results.to_csv(csv_filename, index=False)

logging.info(f"Results saved in {csv_filename}\n")

plot_final_confusion_matrix(model, cumulative_targets, cumulative_preds)
