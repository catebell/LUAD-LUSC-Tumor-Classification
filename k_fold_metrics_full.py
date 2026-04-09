import logging
import sys
import warnings
import numpy as np
import pandas as pd
import torch

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
BATCH_SIZE = 4
EPOCHS = 100
LR = 0.001
MODEL_TYPE = "MultiModalGNN"

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"execution_{MODEL_TYPE}.log", mode="w"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")

# =====================================================
# MODEL
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

def load_dataset():

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

    logging.info(f"Train dataset size: {len(full_train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    return full_train_dataset, test_dataset

# =====================================================
# TRAIN
# =====================================================

def train_epoch(model, loader, optimizer, criterion):

    model.train()
    total_loss = 0

    for data in loader:

        data = data.clone().to(device)

        normalize_data(data)

        optimizer.zero_grad()

        out = forward_model(model, data)

        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

# =====================================================
# VALIDATION
# =====================================================

def validate(model, loader, criterion):

    model.eval()
    total_loss = 0

    with torch.no_grad():

        for data in loader:

            data = data.clone().to(device)

            normalize_data(data)

            out = forward_model(model, data)

            loss = criterion(out, data.y)

            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

# =====================================================
# FORWARD
# =====================================================

def forward_model(model, data):

    if model.__class__ in [CancerGNN, GAT]:
        return model(data.x, data.edge_index, data.edge_attr, data.batch)

    elif model.__class__ == MLP:
        return model(data.clinical)

    else:
        return model(data.x, data.edge_index, data.edge_attr, data.clinical, data.batch)

# =====================================================
# NORMALIZATION
# =====================================================

def normalize_data(data):

    data.x[:, :4] = (data.x[:, :4] - x_mean) / (x_std + 1e-6)
    data.clinical[:, :3] = (data.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
    data.edge_attr[:, 2] = (data.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

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

            data = data.clone().to(device)

            normalize_data(data)

            out = forward_model(model, data)

            prob = torch.softmax(out, dim=1)[:,1]
            pred = out.argmax(dim=1)

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": np.mean(y_true == y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auprc": average_precision_score(y_true, y_prob),
        "tn": cm[0,0],
        "fp": cm[0,1],
        "fn": cm[1,0],
        "tp": cm[1,1]
    }

# =====================================================
# CROSS VALIDATION
# =====================================================

def run_cross_validation(train_dataset, test_dataset):

    y_labels = np.array([train_dataset[i].y.item() for i in range(len(train_dataset))])

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    results = []

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_labels)), y_labels)):

        logging.info(f"\n===== FOLD {fold+1}/{K_FOLDS} =====")

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        model = build_model()

        train_labels = y_labels[train_idx]
        counts = Counter(train_labels)

        weights = torch.tensor([
            len(train_labels)/(2*counts[0]),
            len(train_labels)/(2*counts[1])
        ], dtype=torch.float).to(device)

        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        fit_scalers(train_loader)

        best_val = float("inf")
        early_stop = 0

        for epoch in range(EPOCHS):

            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss = validate(model, val_loader, criterion)

            scheduler.step(val_loss)

            if val_loss < best_val:

                best_val = val_loss
                early_stop = 0

                torch.save(model.state_dict(), f"model_fold_{fold}.pth")

                logging.info(
                    f"Epoch {epoch+1} | "
                    f"Train {train_loss:.4f} | "
                    f"Val {val_loss:.4f}"
                )

            else:

                early_stop += 1

            if early_stop > 20:
                logging.info("Early stopping")
                break

        model.load_state_dict(torch.load(f"model_fold_{fold}.pth"))

        metrics = evaluate(model, test_loader)

        results.append(metrics)

        logging.info(
            f"ACC {metrics['accuracy']:.4f} | "
            f"F1 {metrics['f1_score']:.4f} | "
            f"ROC {metrics['roc_auc']:.4f}"
        )

    return results

# =====================================================
# SCALERS
# =====================================================

def fit_scalers(loader):

    global x_mean,x_std,e_min,e_max,clinical_mean,clinical_std

    node_scaler = StandardScaler()
    edge_scaler = MinMaxScaler()
    clinical_scaler = StandardScaler()

    for data in loader:

        node_scaler.partial_fit(data.x[:, :4].numpy())
        edge_scaler.partial_fit(data.edge_attr[:, 2].numpy().reshape(-1,1))
        clinical_scaler.partial_fit(data.clinical[:, :3].numpy())

    x_mean = torch.tensor(node_scaler.mean_,device=device)
    x_std = torch.tensor(node_scaler.scale_,device=device)

    e_min = torch.tensor(edge_scaler.data_min_,device=device)
    e_max = torch.tensor(edge_scaler.data_max_,device=device)

    clinical_mean = torch.tensor(clinical_scaler.mean_,device=device)
    clinical_std = torch.tensor(clinical_scaler.scale_,device=device)

# =====================================================
# SAVE RESULTS
# =====================================================

def save_results(results):

    df = pd.DataFrame(results)

    df["model"] = MODEL_TYPE
    df["evaluation"] = "cross_validation"

    df.to_csv(f"metrics/cv_{MODEL_TYPE}_iteration_results.csv", index=False)

    rows = []

    for metric in ["accuracy","f1_score","roc_auc","precision","recall","auprc"]:

        mean = df[metric].mean()
        std = df[metric].std()
        n = len(df)

        ci = 1.96 * std / np.sqrt(n)

        rows.append({
            "metric":metric,
            "mean":mean,
            "ci_lower":mean-ci,
            "ci_upper":mean+ci,
            "model":MODEL_TYPE,
            "evaluation":"cross_validation"
        })

    pd.DataFrame(rows).to_csv(f"metrics/cv_{MODEL_TYPE}_summary_results.csv", index=False)

# =====================================================
# MAIN
# =====================================================

def main():

    train_dataset, test_dataset = load_dataset()

    results = run_cross_validation(train_dataset, test_dataset)

    save_results(results)

if __name__ == "__main__":
    main()