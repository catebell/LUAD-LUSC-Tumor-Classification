import logging
import warnings
import os
import argparse
import importlib
import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report
)
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from PatientGraphDataset import PatientGraphDataset
from models.GAT import GAT
from models.MLP import MLP
from models.MultiModalGNN import MultiModalGNN
from models.BasicGraphConvGNN import BasicGraphConvGNN
from models.GINEConvGNN import GINEConvGNN
from models.MoAGNN import MoAGNN
import config

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

MONTE_CARLO_ITER = 30
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 4
LR = 0.001
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


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
    """Parse command line arguments or set default values for model_name and tumor dataset"""
    datasets = get_available_datasets()
    models = get_available_models()

    parser = argparse.ArgumentParser(description="Monte Carlo Cross-Validation GNN Training")

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


def load_model(model_name, dataset_name, num_classes, device):
    module = importlib.import_module(f"models.{model_name}")
    model_class = getattr(module, model_name)

    clinical_df = pd.read_csv(f'{config.FILES}/{dataset_name}/clinical/features_considered.tsv', sep='\t')
    num_patient_features = len(clinical_df.columns.tolist()[2:])

    if model_name in ["GINEConvGNN", "GAT"]:
        model = model_class(num_node_features=5, num_edge_features=3, hidden_channels=64, num_classes=num_classes)
    elif model_name == "BasicGraphConvGNN":
        model = model_class(num_node_features=5, hidden_channels=64, num_classes=num_classes)
    elif model_name == "MLP":
        model = model_class(num_patient_features=5, hidden_channels=64, num_classes=num_classes)
    elif model_name == "MultiModalGNN":
        model = model_class(num_node_features=5, num_edge_features=3, clinical_input_dim=num_patient_features,
                            hidden_channels=64, num_classes=num_classes)
    elif model_name == "MoAGNN":
        class Args:
            pass
        args = Args()
        args.num_features = 5
        args.nhid = 128
        args.num_classes = num_classes
        args.pooling_ratio = 0.5
        args.dropout_ratio = 0.0
        model = model_class(args)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def get_optimizer(model, lr, default_wd=1e-4):
    """Give different weight decay to MLP/classifier params vs graph params."""
    mlp_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'mlp' in name or 'classifier' in name:
            mlp_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.Adam([
        {'params': other_params, 'weight_decay': default_wd},
        {'params': mlp_params, 'weight_decay': 1e-2}
    ], lr=lr)


def load_dataset(dataset_name):
    logging.info("Loading dataset...")
    file_mapping_df = pd.read_csv(f'{config.FILES}/{dataset_name}/clinical/file_case_mapping.tsv', sep='\t').dropna()
    patient_split_df = pd.read_csv(f'{config.FILES}/{dataset_name}/clinical/patient_split_cleaned.csv')

    train_df = file_mapping_df[file_mapping_df['case_id'].isin(
        patient_split_df[patient_split_df['split'] == 'train']['cases.case_id'])]
    val_df = file_mapping_df[file_mapping_df['case_id'].isin(
        patient_split_df[patient_split_df['split'] == 'val']['cases.case_id'])]
    test_df = file_mapping_df[file_mapping_df['case_id'].isin(
        patient_split_df[patient_split_df['split'] == 'test']['cases.case_id'])]

    train_dataset = PatientGraphDataset(f'data_graphs_processed/{dataset_name}/data_graphs_processed_train', train_df)
    val_dataset = PatientGraphDataset(f'data_graphs_processed/{dataset_name}/data_graphs_processed_validation', val_df)
    test_dataset = PatientGraphDataset(f'data_graphs_processed/{dataset_name}/data_graphs_processed_test', test_df)

    full_dataset = train_dataset + val_dataset + test_dataset

    y_labels = np.array([full_dataset[i].y.item() for i in range(len(full_dataset))])
    unique_labels = np.unique(y_labels)
    num_classes = len(unique_labels)

    logging.info(f"Dataset size: {len(full_dataset)}")
    logging.info(f"Unique labels found in dataset: {unique_labels}")

    return full_dataset, y_labels, num_classes, unique_labels


def fit_scalers(train_loader):
    """Fit StandardScaler on node features and clinical, MinMaxScaler on edge attributes."""
    node_feat_scaler = StandardScaler()
    edge_attr_scaler = MinMaxScaler()
    clinical_feat_scaler = StandardScaler()

    for data in train_loader:
        node_feat_scaler.partial_fit(data.x[:, :4].numpy())
        edge_attr_scaler.partial_fit(data.edge_attr[:, 2].numpy().reshape(-1, 1))
        clinical_feat_scaler.partial_fit(data.clinical[:, :3].numpy())

    x_mean = torch.tensor(node_feat_scaler.mean_, dtype=torch.float, device=device)
    x_std = torch.tensor(node_feat_scaler.scale_, dtype=torch.float, device=device)
    e_min = torch.tensor(edge_attr_scaler.data_min_, dtype=torch.float, device=device)
    e_max = torch.tensor(edge_attr_scaler.data_max_, dtype=torch.float, device=device)
    clinical_mean = torch.tensor(clinical_feat_scaler.mean_, dtype=torch.float, device=device)
    clinical_std = torch.tensor(clinical_feat_scaler.scale_, dtype=torch.float, device=device)

    return x_mean, x_std, e_min, e_max, clinical_mean, clinical_std


def apply_scaling(model, data_copy, x_mean, x_std, e_min, e_max, clinical_mean, clinical_std):
    data_copy.x[:, :4] = (data_copy.x[:, :4] - x_mean) / (x_std + 1e-6)
    if model.__class__ != MoAGNN:
        data_copy.clinical[:, :3] = (data_copy.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
        data_copy.edge_attr[:, 2] = (data_copy.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)
    return data_copy


def forward_pass(model, data_copy):
    if model.__class__ in [GINEConvGNN, GAT]:
        return model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.batch)
    elif model.__class__ == BasicGraphConvGNN:
        return model(data_copy.x, data_copy.edge_index, data_copy.batch)
    elif model.__class__ == MLP:
        x_pooled = global_mean_pool(data_copy.x, data_copy.batch)
        return model(x_pooled)
    elif model.__class__ == MoAGNN:
        return model(data_copy)
    else:  # MultiModalGNN
        return model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.clinical, data_copy.batch)


def train_epoch(model, loader, optimizer, criterion, scalers):
    model.train()
    total_loss = 0

    for data in loader:
        data_copy = data.clone().to(device)
        data_copy = apply_scaling(model, data_copy, *scalers)

        optimizer.zero_grad()
        out = forward_pass(model, data_copy)
        loss = criterion(out, data_copy.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data_copy.num_graphs

    return total_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, scalers):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            data_copy = data.clone().to(device)
            data_copy = apply_scaling(model, data_copy, *scalers)

            out = forward_pass(model, data_copy)
            loss = criterion(out, data_copy.y)
            total_loss += loss.item() * data_copy.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, unique_labels, scalers):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data in loader:
            data_copy = data.clone().to(device)
            data_copy = apply_scaling(model, data_copy, *scalers)

            out = forward_pass(model, data_copy)
            probs = torch.exp(out) if model.__class__ == MoAGNN else torch.softmax(out, dim=1)
            pred = out.argmax(dim=1)

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

    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)

    # tn, tp, fn, fp from report support and recall (binary only)
    if len(unique_labels) == 2:
        tn = int(round(report['0']['recall'] * report['0']['support']))
        fp = int(report['0']['support']) - tn
        tp = int(round(report['1']['recall'] * report['1']['support']))
        fn = int(report['1']['support']) - tp
    else:
        tn = fp = fn = tp = None

    metrics = {
        "accuracy": np.mean(all_targets == all_preds),
        "f1_score": f1_score(all_targets, all_preds, average='macro', zero_division=0),
        "auc": auc_val,
        "precision": precision_score(all_targets, all_preds, average='macro', zero_division=0),
        "recall": recall_score(all_targets, all_preds, average='macro', zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return metrics


def run_montecarlo(dataset_name, model_name):
    full_dataset, y_labels, num_classes, unique_labels = load_dataset(dataset_name)

    splitter = StratifiedShuffleSplit(
        n_splits=MONTE_CARLO_ITER,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    results = []

    for i, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y_labels)), y_labels)):
        logging.info(f"\n===== MONTE CARLO ITER {i + 1}/{MONTE_CARLO_ITER} =====")

        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

        model = load_model(model_name, dataset_name, num_classes, device)
        logging.info(model)

        if model.__class__ == MoAGNN:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
            criterion = torch.nn.NLLLoss()
        else:
            optimizer = get_optimizer(model, LR)
            criterion = torch.nn.CrossEntropyLoss()

        scalers = fit_scalers(train_loader)

        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_model_path = f'{dataset_name}_{model_name}_mc_iter_{i + 1}.pth'

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scalers)
            val_loss = validate_epoch(model, test_loader, criterion, scalers)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                early_stopping_counter = 0
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

                if epoch % 10 == 0 or epoch == 1:
                    train_metrics = evaluate(model, train_loader, unique_labels, scalers)
                    logging.info(
                        f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Train F1: {train_metrics['f1_score']:.4f} | Train AUC: {train_metrics['auc']:.4f}"
                    )
                logging.info("--- Found and saved a better model! ---")
            else:
                early_stopping_counter += 1

            if early_stopping_counter > 20:
                logging.info("--- Early stopping triggered ---")
                break

        model.load_state_dict(torch.load(best_model_path))
        metrics = evaluate(model, test_loader, unique_labels, scalers)
        results.append(metrics)

        logging.info(
            f"Iter {i + 1} | ACC {metrics['accuracy']:.4f} | "
            f"F1 {metrics['f1_score']:.4f} | AUC {metrics['auc']:.4f}"
        )

        del model
        torch.cuda.empty_cache()

    return results


def save_iteration_results(results, model_name, dataset_name):
    df = pd.DataFrame(results)
    df["model"] = model_name
    df["evaluation"] = "montecarlo"

    path = Path(f"metrics/{dataset_name}")
    path.mkdir(parents=True, exist_ok=True)

    out_path = f"{path}/montecarlo_{model_name}_iteration_results.csv"
    df.to_csv(out_path, index=False)
    logging.info(f"Iteration results saved to {out_path}")


def summarize_results(results, model_name, dataset_name):
    df = pd.DataFrame(results)
    rows = []

    for metric in ["accuracy", "f1_score", "auc", "precision", "recall"]:
        mean = df[metric].mean()
        std = df[metric].std()
        n = len(df)
        ci = 1.96 * std / np.sqrt(n)

        rows.append({
            "metric": metric,
            "mean": mean,
            "std": std,
            "ci_lower": mean - ci,
            "ci_upper": mean + ci,
            "model": model_name,
            "evaluation": "montecarlo"
        })

    summary_df = pd.DataFrame(rows)

    path = Path(f"metrics/{dataset_name}")
    path.mkdir(parents=True, exist_ok=True)

    out_path = f"{path}/montecarlo_{model_name}_summary_results.csv"
    summary_df.to_csv(out_path, index=False)

    logging.info("\n===== FINAL RESULTS =====")
    logging.info(summary_df.to_string(index=False))


def main():
    args = parse_args()

    dataset_name = args.dataset
    model_name = args.model_name

    logging.info(f"Dataset: {dataset_name} | Model: {model_name} | Iterations: {MONTE_CARLO_ITER}")

    results = run_montecarlo(dataset_name, model_name)
    save_iteration_results(results, model_name, dataset_name)
    summarize_results(results, model_name, dataset_name)


if __name__ == "__main__":
    main()