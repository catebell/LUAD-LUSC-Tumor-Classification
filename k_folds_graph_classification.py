import logging
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, average_precision_score, confusion_matrix
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
train_dataset = PatientGraphDataset(root='data_graphs_processed_train', file_mapping_df=file_mapping_df_train,
                                    node_map=node_map)
logging.info("Test Dataset init...")
test_dataset = PatientGraphDataset(root='data_graphs_processed_test', file_mapping_df=file_mapping_df_test,
                                   node_map=node_map)
logging.info("Val Dataset init...")
val_dataset = PatientGraphDataset(root='data_graphs_processed_validation', file_mapping_df=file_mapping_df_val,
                                  node_map=node_map)

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
        data_copy = data.clone()  # to prevent inplace modifications through folds/loops
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
            out = model(data_copy.clinical)  # just clinical features
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


def evaluate(model, loader):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    correct = 0

    with torch.no_grad():
        for data in loader:  # iterate in batches.
            data_copy = data.clone().to(device)

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

            probs = torch.softmax(out, dim=1)[:, 1]  # class 1 probability
            pred = out.argmax(dim=1)  # use the class with the highest probability

            all_targets.extend(data_copy.y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0, 1]).ravel()

    metrics = {
        'acc': np.mean(np.array(all_preds) == np.array(all_targets)),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
        'auc': roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0,
        'auprc': average_precision_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)

    return metrics, report


test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
max_epochs = 100

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_labels)), y_labels)):
    logging.info(f"--- FOLD {fold + 1}/{k_folds} ---\n")

    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=True)

    # re-initialization for current fold
    # model = CancerGNN(num_node_features=5, num_edge_features=3, hidden_channels=64).to(device)
    # model = GAT(num_node_features=5, num_edge_features=3, num_classes=2, hidden_channels=64).to(device)
    # model = MLP(num_patient_features=53, num_classes=2).to(device)
    model = MultiModalGNN(num_node_features=5, num_edge_features=3, clinical_input_dim=53, hidden_channels=64,
                          num_classes=2).to(device)
    logging.info(model)

    optimizer = get_optimizer(model, 0.001, 1e-4)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
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

        train_metrics, train_report = evaluate(model, train_loader)
        val_metrics, val_report = evaluate(model, val_loader)

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

            torch.save(model.state_dict(), f'model_fold_{fold + 1}.pth')
            logging.info("--- Found and saved a better model! ---\n")

        if early_stopping_counter > 20:
            logging.info("--- Stopping training due to early stopping ---\n")
            break
        else:
            early_stopping_counter += 1

    # best model found in this fold tested on the fixed evaluate set
    model.load_state_dict(torch.load(f'model_fold_{fold + 1}.pth'))
    fold_test_metrics, test_report = evaluate(model, test_loader)
    fold_results.append(fold_test_metrics)

    logging.info(
        f"Fold {fold + 1} Test Acc: {fold_test_metrics['acc']:.4f} | AUC: {fold_test_metrics['auc']:.4f} | AUPRC: {fold_test_metrics['auprc']:.4f}\n"
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
avg_auprc = np.mean([metrics['auprc'] for metrics in fold_results])
total_tn = np.sum([metrics['tn'] for metrics in fold_results])
total_fp = np.sum([metrics['fp'] for metrics in fold_results])
total_fn = np.sum([metrics['fn'] for metrics in fold_results])
total_tp = np.sum([metrics['tp'] for metrics in fold_results])

logging.info(f"Mean Test Accuracy: {avg_acc:.4f} +/- {dev_std:.4f}")
logging.info(f"F1: {avg_f1:.4f}, AUC: {avg_auc:.4f}, , AUPRC: {avg_auprc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")
logging.info(f"Global Confusion Matrix -> TP: {total_tp}, TN: {total_tn}, FP: {total_fp}, FN: {total_fn}")

results_for_csv = []

for i, res in enumerate(fold_results):
    row = {
        'fold': i + 1,
        'accuracy': res['acc'],
        'precision': res['precision'],
        'recall': res['recall'],
        'f1_score': res['f1'],
        'auc_roc': res['auc'],
        'auprc': res['auprc'],
        'true_positive': res['tp'],
        'true_negative': res['tn'],
        'false_positive': res['fp'],
        'false_negative': res['fn']
    }
    results_for_csv.append(row)

df_results = pd.DataFrame(results_for_csv)

mean_row = df_results.mean(numeric_only=True).to_dict()
mean_row['fold'] = 'MEAN'
df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)

std_row = df_results.iloc[:-1].std(numeric_only=True).to_dict()
std_row['fold'] = 'STD_DEV'
df_results = pd.concat([df_results, pd.DataFrame([std_row])], ignore_index=True)

if not os.path.exists("models/final_metrics_k_fold"):
    os.mkdir('models/final_metrics_k_fold')

if model.__class__ == CancerGNN or model.__class__ == GAT:
    csv_filename = 'models/final_metrics_k_fold/GNN_final_metrics_comparison.csv'
elif model.__class__ == MLP:
    csv_filename = 'models/final_metrics_k_fold/MLP_final_metrics_comparison.csv'
else:  # MultiModalGNN
    csv_filename = 'models/final_metrics_k_fold/MultiModalGNN_final_metrics_comparison.csv'

df_results.to_csv(csv_filename, index=False)

logging.info(f"Results saved in {csv_filename}\n")


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_final_confusion_matrix(fold_results, filename='confusion_matrix_total.png'):
    total_tp = sum([res['tp'] for res in fold_results])
    total_tn = sum([res['tn'] for res in fold_results])
    total_fp = sum([res['fp'] for res in fold_results])
    total_fn = sum([res['fn'] for res in fold_results])

    cm = [[total_tn, total_fp],
          [total_fn, total_tp]]

    cm_norm = np.array(cm) / np.array(cm).sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, cmap='Blues',
                xticklabels=['Pred LUAD (0)', 'Pred LUSC (1)'],
                yticklabels=['True LUAD (0)', 'True LUSC (1)'])

    plt.title('Global Confusion Matrix (Sum for all folds)')
    plt.ylabel('Real Label')
    plt.xlabel('Model Prediction')

    plt.savefig(f'models/final_metrics_k_fold/{filename}', bbox_inches='tight', dpi=300)
    plt.show()
    logging.info(f"Confusion Matrix saved.")

plot_final_confusion_matrix(fold_results)