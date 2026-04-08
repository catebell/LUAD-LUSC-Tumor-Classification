import logging
import warnings
import torch
import numpy as np
import pandas as pd

from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix
)

from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from PatientGraphDataset import PatientGraphDataset
from models.CancerGNN import CancerGNN
from models.GAT import GAT
from models.MLP import MLP
from models.MultiModalGNN import MultiModalGNN

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================

K_FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 4
LR = 0.001

MODEL_TYPE = "MultiModalGNN"

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("execution.log", mode="w"),
        logging.StreamHandler()
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")

torch.cuda.empty_cache()

# =====================================================
# MODEL BUILDER
# =====================================================

def build_model():

    if MODEL_TYPE == "CancerGNN":
        model = CancerGNN(num_node_features=5, num_edge_features=3, hidden_channels=64)

    elif MODEL_TYPE == "GAT":
        model = GAT(num_node_features=5, num_edge_features=3, num_classes=2, hidden_channels=64)

    elif MODEL_TYPE == "MLP":
        model = MLP(num_patient_features=53, num_classes=2)

    elif MODEL_TYPE == "MultiModalGNN":
        model = MultiModalGNN(
            num_node_features=5,
            num_edge_features=3,
            clinical_input_dim=53,
            hidden_channels=64,
            num_classes=2
        )

    return model.to(device)

# =====================================================
# DATASET
# =====================================================

def load_datasets():

    file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t').dropna()
    patient_split_df = pd.read_csv('files/clinical/patient_split_cleaned.csv')

    node_map_df = pd.read_csv('files/clinical/gene_ids_mapped.tsv', sep='\t')
    node_map = dict(zip(node_map_df.gene_id, node_map_df.gene_id_mapped))

    train_df = file_mapping_df[file_mapping_df['case_id'].isin(
        patient_split_df[patient_split_df['split'] == 'train']['cases.case_id'])]

    val_df = file_mapping_df[file_mapping_df['case_id'].isin(
        patient_split_df[patient_split_df['split'] == 'val']['cases.case_id'])]

    test_df = file_mapping_df[file_mapping_df['case_id'].isin(
        patient_split_df[patient_split_df['split'] == 'test']['cases.case_id'])]

    train_dataset = PatientGraphDataset("data_graphs_processed_train", train_df, node_map)
    val_dataset = PatientGraphDataset("data_graphs_processed_validation", val_df, node_map)
    test_dataset = PatientGraphDataset("data_graphs_processed_test", test_df, node_map)

    full_train_dataset = train_dataset + val_dataset

    return full_train_dataset, test_dataset

# =====================================================
# TRAIN
# =====================================================

def train_epoch(model, loader, optimizer, criterion):

    model.train()
    total_loss = 0

    for data in loader:

        data = data.to(device)

        optimizer.zero_grad()

        if model.__class__ in [CancerGNN, GAT]:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        elif model.__class__ == MLP:
            out = model(data.clinical)

        else:
            out = model(data.x, data.edge_index, data.edge_attr, data.clinical, data.batch)

        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

# =====================================================
# EVALUATION
# =====================================================

def evaluate(model, loader):

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for data in loader:

            data = data.to(device)

            if model.__class__ in [CancerGNN, GAT]:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)

            elif model.__class__ == MLP:
                out = model(data.clinical)

            else:
                out = model(data.x, data.edge_index, data.edge_attr, data.clinical, data.batch)

            prob = torch.softmax(out, dim=1)[:,1]
            pred = out.argmax(dim=1)

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {}

    metrics["accuracy"] = np.mean(y_true == y_pred)
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    metrics["precision"] = precision_score(y_true, y_pred)
    metrics["recall"] = recall_score(y_true, y_pred)
    metrics["auprc"] = average_precision_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)

    metrics["tn"] = cm[0,0]
    metrics["fp"] = cm[0,1]
    metrics["fn"] = cm[1,0]
    metrics["tp"] = cm[1,1]

    return metrics

# =====================================================
# KFOLD TRAINING
# =====================================================

def run_kfold():

    dataset, test_dataset = load_datasets()

    y_labels = np.array([dataset[i].y.item() for i in range(len(dataset))])

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_labels)), y_labels)):

        logging.info(f"\n===== FOLD {fold+1}/{K_FOLDS} =====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        model = build_model()

        train_labels = y_labels[train_idx]
        counts = Counter(train_labels)

        w0 = len(train_labels) / (2 * counts[0])
        w1 = len(train_labels) / (2 * counts[1])

        weights = torch.tensor([w0, w1], dtype=torch.float).to(device)

        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10
        )

        best_val_loss = float("inf")
        early_counter = 0

        for epoch in range(EPOCHS):

            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = evaluate(model, val_loader)

            scheduler.step(train_loss)

            logging.info(
                f"Epoch {epoch+1} | "
                f"TrainLoss {train_loss:.4f} | "
                f"ValACC {val_metrics['accuracy']:.4f} "
                f"F1 {val_metrics['f1']:.4f} "
                f"ROC {val_metrics['roc_auc']:.4f}"
            )

            if train_loss < best_val_loss:

                best_val_loss = train_loss
                early_counter = 0

                torch.save(model.state_dict(), f"model_fold_{fold}.pth")

            else:

                early_counter += 1

                if early_counter > 20:
                    logging.info("Early stopping")
                    break

        model.load_state_dict(torch.load(f"model_fold_{fold}.pth"))

        test_metrics = evaluate(model, test_loader)

        fold_results.append(test_metrics)

        logging.info(
            f"Fold Test ACC {test_metrics['accuracy']:.4f} "
            f"F1 {test_metrics['f1']:.4f} "
            f"ROC {test_metrics['roc_auc']:.4f}"
        )

    return fold_results

# =====================================================
# FINAL STATS
# =====================================================

def summarize_results(results):

    df = pd.DataFrame(results)

    logging.info("\n===== FINAL RESULTS =====")

    for metric in ["accuracy","f1","roc_auc","precision","recall","auprc"]:

        mean = df[metric].mean()
        std = df[metric].std()

        logging.info(f"{metric}: {mean:.4f} ± {std:.4f}")

# =====================================================
# MAIN
# =====================================================

def main():

    results = run_kfold()

    summarize_results(results)


if __name__ == "__main__":
    main()