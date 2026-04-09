import logging
import warnings
import torch
import numpy as np
import pandas as pd

from collections import Counter

from sklearn.model_selection import StratifiedShuffleSplit
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

MONTE_CARLO_ITER = 30
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 4
LR = 0.001

MODEL_TYPE = "MultiModalGNN"

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

    full_train_dataset = train_dataset + val_dataset + test_dataset

    return full_train_dataset


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

    metrics = {"accuracy": np.mean(y_true == y_pred), "f1": f1_score(y_true, y_pred),
               "roc_auc": roc_auc_score(y_true, y_prob), "precision": precision_score(y_true, y_pred),
               "recall": recall_score(y_true, y_pred), "auprc": average_precision_score(y_true, y_prob)}

    cm = confusion_matrix(y_true, y_pred)

    metrics["tn"] = cm[0,0]
    metrics["fp"] = cm[0,1]
    metrics["fn"] = cm[1,0]
    metrics["tp"] = cm[1,1]

    return metrics


def run_montecarlo():
    dataset = load_dataset()
    y_labels = np.array([dataset[i].y.item() for i in range(len(dataset))])

    splitter = StratifiedShuffleSplit(
        n_splits=MONTE_CARLO_ITER,
        test_size=TEST_SIZE,
        random_state=42
    )

    results = []

    for i, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y_labels)), y_labels)):

        logging.info(f"\n===== MONTE CARLO ITER {i+1}/{MONTE_CARLO_ITER} =====")

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE)

        model = build_model()

        train_labels = y_labels[train_idx]
        counts = Counter(train_labels)

        w0 = len(train_labels) / (2 * counts[0])
        w1 = len(train_labels) / (2 * counts[1])

        weights = torch.tensor([w0, w1], dtype=torch.float).to(device)

        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion)

        metrics = evaluate(model, test_loader)

        results.append(metrics)

        logging.info(
            f"ACC {metrics['accuracy']:.4f} "
            f"F1 {metrics['f1']:.4f} "
            f"ROC {metrics['roc_auc']:.4f} "
            f"AUPRC {metrics['auprc']:.4f}"
        )

    return results


def summarize_results(results):
    df = pd.DataFrame(results)
    logging.info("\n===== FINAL RESULTS =====")

    for metric in ["accuracy","f1","roc_auc","precision","recall","auprc"]:

        mean = df[metric].mean()
        std = df[metric].std()

        logging.info(f"{metric}: {mean:.4f} ± {std:.4f}")


def main():

    results = run_montecarlo()

    summarize_results(results)

if __name__ == "__main__":
    main()