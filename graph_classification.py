import logging

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.loader import DataLoader
from collections import Counter

from PatientGraphDataset import PatientGraphDataset
from models.CancerGNN import CancerGNN
from models.GAT import GAT
from models.MLP import MLP
from models.MultiModalGNN import MultiModalGNN

import warnings
warnings.filterwarnings("ignore")
# TODO rimuovere sta cosa

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
#model = GAT(num_node_features=5, num_edge_features=3, num_classes=2, hidden_channels=64).to(device)
#model = MLP(num_patient_features=53).to(device)
model = MultiModalGNN(num_node_features=5, num_edge_features=3, clinical_input_dim=53, hidden_channels=64).to(device)

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

def get_optimizer(model, lr, default):
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


optimizer = get_optimizer(model, 0.001, 1e-4)
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

criterion = torch.nn.CrossEntropyLoss(weights, label_smoothing=0.1)

logging.info(model)


def train():
    model.train()
    total_loss = 0
    model.zero_grad()

    for og in train_loader:  # Iterate in batches over the training dataset.
        data = og.clone()
        data = data.to(device)
        # .transform() works with numpy, library on GPU (not CUDA-efficient)
        #data.x[:, :4] = torch.tensor(node_feat_scaler.transform(data.x[:, :4].numpy())).to(device)
        #data.edge_attr[:,2] = torch.tensor(edge_attr_scaler.transform(data.edge_attr[:,2].numpy().reshape(-1,1))).to(device)

        # StandardScale: (x - mean) / std
        data.x[:, :4] = (data.x[:, :4] - x_mean) / (x_std + 1e-6)
        data.clinical[:, :3] = (data.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
        # MinMaxScaler: (x - min) / (max - min)
        data.edge_attr[:,2] = (data.edge_attr[:,2] - e_min) / (e_max - e_min + 1e-6)

        # perform a single forward pass
        if model.__class__ == CancerGNN or model.__class__ == GAT:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # just graph
        elif model.__class__ == MLP:
            out = model(data.clinical) # just clinical features
        else:  # MultiModalGNN
            out = model(data.x, data.edge_index, data.edge_attr, data.clinical, data.batch) # both

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
    for og in val_loader:
        data = og.clone()
        data = data.to(device)
        data.x[:, :4] = (data.x[:, :4] - x_mean) / (x_std + 1e-6)
        data.clinical[:, :3] = (data.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
        data.edge_attr[:, 2] = (data.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

        with torch.no_grad():
            if model.__class__ == CancerGNN or model.__class__ == GAT:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # just graph
            elif model.__class__ == MLP:
                out = model(data.clinical)  # just clinical features
            else:  # MultiModalGNN
                out = model(data.x, data.edge_index, data.edge_attr, data.clinical, data.batch)  # both

            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(val_dataset)


def test(model, loader):
     model.eval()
     correct = 0

     for og in loader:  # Iterate in batches over the training/test dataset.
         data = og.clone()
         data = data.to(device)
         data.x[:, :4] = (data.x[:, :4] - x_mean) / (x_std + 1e-6)
         data.clinical[:, :3] = (data.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
         data.edge_attr[:, 2] = (data.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

         if model.__class__ == CancerGNN or model.__class__ == GAT:
             out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # just graph
         elif model.__class__ == MLP:
             out = model(data.clinical)  # just clinical features
         else:  # MultiModalGNN
             out = model(data.x, data.edge_index, data.edge_attr, data.clinical, data.batch)  # both

         pred = out.argmax(dim=1)  # Use the class with the highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def explain_clinical_importance(model, loader, clinical_feature_names):
    """
    Clinical features importance computed by Permutation Importance.
    0.05 --> accuracy drops by 5% without that feature, 0.00 --> useless feature, accuracy doesn't drop.
    """
    model.eval()
    baseline_acc = test(model, loader)
    feature_importances = {}

    for i, name in enumerate(clinical_feature_names):
        correct = 0
        total = 0

        for data in loader:
            data_copy = data.clone().to(device)
            data_copy.clinical[:, :3] = (data_copy.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
            data_copy.x[:, :4] = (data_copy.x[:, :4] - x_mean) / (x_std + 1e-6)
            data_copy.edge_attr[:, 2] = (data_copy.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

            # Permutazione della feature i-esima all'interno del batch
            perm = torch.randperm(data_copy.clinical.size(0))
            data_copy.clinical[:, i] = data_copy.clinical[perm, i]

            with torch.no_grad():
                if model.__class__ == CancerGNN or model.__class__ == GAT:
                    out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.batch)  # just graph
                elif model.__class__ == MLP:
                    out = model(data_copy.clinical)  # just clinical features
                else:  # MultiModalGNN
                    out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.clinical, data_copy.batch)  # both

                pred = out.argmax(dim=1)
                correct += int((pred == data_copy.y).sum())
                total += data_copy.num_graphs

        permuted_acc = correct / total
        feature_importances[name] = baseline_acc - permuted_acc  # importance given by how much accuracy drops

    return sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)


def get_gene_attention_weights(model, loader, node_map_inv):
    """
    Extract genes with the highest attention weight from GAT.
    """
    model.eval()

    # we process data by batches, so total numbers of nodes in a data from loader is len(node_map_inv) * num_batches
    num_unique_genes = len(node_map_inv)
    gene_scores = torch.zeros(num_unique_genes).to(device)
    counts = torch.zeros(num_unique_genes).to(device)

    for data in loader:
        data_copy = data.clone().to(device)
        data_copy.clinical[:, :3] = (data_copy.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
        data_copy.x[:, :4] = (data_copy.x[:, :4] - x_mean) / (x_std + 1e-6)
        data_copy.edge_attr[:, 2] = (data_copy.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

        with torch.no_grad():
            # retrieve attention from graph branch
            _, (edge_index, att_weights) = model.graph_branch.get_attention(
                data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.batch
            )

            # avg for attention heads
            mean_att = att_weights.mean(dim=1)

            # remapping to unique genes in a batch
            # target_nodes = idx in batch (es. 0...77963 if batch_size=4 and genes=19491)
            target_nodes_batch = edge_index[1]

            # remap to single patient genes idx (0...19490)
            target_nodes_original = target_nodes_batch % num_unique_genes

            gene_scores.scatter_add_(0, target_nodes_original, mean_att)

            ones = torch.ones_like(mean_att)
            counts.scatter_add_(0, target_nodes_original, ones)

    avg_scores = (gene_scores / (counts + 1e-6)).cpu().numpy()

    importance_list = []
    for idx, score in enumerate(avg_scores):
        gene_id = node_map_inv.get(idx, f"Unknown_{idx}")
        # just genes present in the dataset
        if counts[idx] > 0:
            importance_list.append((gene_id, score))

    return sorted(importance_list, key=lambda x: x[1], reverse=True)


def get_gene_saliency(model, loader, node_map_inv):
    model.eval()
    gene_accumulation = {}
    gene_counts = {}

    for data in loader:
        data_copy = data.clone().to(device)
        data_copy.x[:, :4] = (data_copy.x[:, :4] - x_mean) / (x_std + 1e-6)
        data_copy.edge_attr[:, 2] = (data_copy.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)
        data_copy.clinical[:, :3] = (data_copy.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)

        data_copy.x.requires_grad = True

        # Forward & Backward
        out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.clinical, data_copy.batch)
        probs = torch.softmax(out, dim=1)
        max_probs, _ = torch.max(probs, dim=1)

        model.zero_grad()
        max_probs.backward(torch.ones_like(max_probs))

        # Saliency: avg of absolute gradient on node features
        saliency = data_copy.x.grad.abs().mean(dim=1).cpu().numpy()

        # retrieve original node indexes because data contains a batch of graphs (num nodes = num genes * num batches)
        num_nodes_per_graph = data_copy.x.size(0) // data_copy.num_graphs

        for i in range(data_copy.x.size(0)):
            # Calcoliamo l'indice del gene relativo al singolo grafo
            # (Assumendo che ogni grafo nel batch abbia gli stessi geni nello stesso ordine)
            gene_idx_in_map = i % (data_copy.x.size(0) // data_copy.num_graphs)
            score = saliency[i]
            gene_id = node_map_inv.get(gene_idx_in_map, f"Unknown_{gene_idx_in_map}")

            gene_accumulation[gene_id] = gene_accumulation.get(gene_id, 0) + score
            gene_counts[gene_id] = gene_counts.get(gene_id, 0) + 1

    # Calcolo media e normalizzazione
    final_importance = []
    for gene_id in gene_accumulation:
        avg_score = gene_accumulation[gene_id] / gene_counts[gene_id]
        final_importance.append((gene_id, avg_score))

    # Ordinamento
    final_importance.sort(key=lambda x: x[1], reverse=True)

    # Normalizzazione 0-1 per il primo della lista
    if final_importance:
        max_val = final_importance[0][1]
        final_importance = [(g, s / max_val) for g, s in final_importance]

    return final_importance

'''
best_val_loss = float('inf')
early_stopping_counter = 0

for epoch in range(1, max_epochs + 1):
    train_loss = train()
    val_loss = validate()

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

        torch.save(model.state_dict(), 'best_classification_gnn.pth')
        logging.info("--- Found and saved a better model! ---\n")

    if early_stopping_counter > 20:
        logging.info("--- Stopping training due to early stopping ---")
        break
    else:
        early_stopping_counter += 1
'''

clinical_names = ['age_at_index', 'tobacco_years', 'pack_years_smoked', 'country_of_residence_at_enrollment_Australia', 'country_of_residence_at_enrollment_Germany', 'country_of_residence_at_enrollment_United States', 'country_of_residence_at_enrollment_Switzerland', 'country_of_residence_at_enrollment_Russia', 'country_of_residence_at_enrollment_Canada', 'country_of_residence_at_enrollment_Ukraine', 'country_of_residence_at_enrollment_Romania', 'country_of_residence_at_enrollment_Vietnam', 'ethnicity_not hispanic or latino', 'ethnicity_hispanic or latino', 'gender_male', 'gender_female', 'race_white', 'race_black or african american', 'race_asian', 'ajcc_pathologic_m_M0', 'ajcc_pathologic_m_M1a', 'ajcc_pathologic_m_M1', 'ajcc_pathologic_m_M1b', 'ajcc_pathologic_n_N1', 'ajcc_pathologic_n_N0', 'ajcc_pathologic_n_N2', 'ajcc_pathologic_n_N3', 'ajcc_pathologic_t_T2a', 'ajcc_pathologic_t_T2b', 'ajcc_pathologic_t_T2', 'ajcc_pathologic_t_T3', 'ajcc_pathologic_t_T4', 'ajcc_pathologic_t_T1b', 'ajcc_pathologic_t_T1', 'ajcc_pathologic_t_T1a', '3', '1', '2', '9', '8', '0', 'laterality_Left', 'laterality_Right', 'sites_of_involvement_Peripheral Lung', 'sites_of_involvement_Central Lung', 'tissue_or_organ_of_origin_Lower lobe, lung', 'tissue_or_organ_of_origin_Upper lobe, lung', 'tissue_or_organ_of_origin_Middle lobe, lung', 'tissue_or_organ_of_origin_Lung, NOS', 'tissue_or_organ_of_origin_Overlapping lesion of lung', 'tissue_or_organ_of_origin_Main bronchus', 'tobacco_smoker', 'ajcc_pathologic_stage']

node_map_inv = {v: k for k, v in node_map.items()}

logging.info("--- Feature Importance analysis (Best Model Saved) ---\n")

model.load_state_dict(torch.load('best_classification_gnn.pth', map_location=device))

'''
clinical_imp = explain_clinical_importance(model, test_loader, clinical_names)
logging.info("Clinical Features importance:")
for name, imp in clinical_imp:
    logging.info(f"{name}: {imp:.4f}\n")
'''

gene_alias = pd.read_csv('STRING_downloaded_files/9606.protein.aliases.gene.tsv', sep='\t', usecols=['gene_id', 'alias'])
gene_alias = gene_alias.groupby('gene_id')['alias'].apply(list).reset_index(name='names')

#gene_imp = get_gene_attention_weights(model, test_loader, node_map_inv)  # (GAT Attention)
gene_sal = get_gene_saliency(model, test_loader, node_map_inv)
logging.info("Top 100 Genes saliency:")

for gene_id, score in gene_sal[:100]:
    names = gene_alias[gene_alias['gene_id'] == gene_id]['names']
    logging.info(f"{gene_id}: {score:.4f}   {names}")
