import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from graph_classification import test, clinical_mean, clinical_std, x_mean, x_std, e_min, e_max, model, device, \
    test_loader, node_map
from models.CancerGNN import CancerGNN
from models.GAT import GAT
from models.MLP import MLP


def explain_clinical_importance(model, device, loader, clinical_features_names):
    """
    Clinical features importance computed by Permutation Importance.
    0.05 --> accuracy drops by 5% without that feature, 0.00 --> useless feature, accuracy doesn't drop.
    """
    model.eval()
    baseline_acc = test(model, loader)
    feature_importances = {}

    features = ['age_at_index','tobacco_years','pack_years_smoked','country_of_residence_at_enrollment','ethnicity',
                'gender','race','ajcc_pathologic_m','ajcc_pathologic_n','ajcc_pathologic_t','icd_10_code','laterality',
                'sites_of_involvement','tissue_or_organ_of_origin','tobacco_smoker','ajcc_pathologic_stage']
    '''
    for example:
    country_of_residence_at_enrollment_Germany,
    country_of_residence_at_enrollment_Ukraine,
    country_of_residence_at_enrollment_Switzerland
    will all be found by contains.('country_of_residence_at_enrollment')
    '''

    for target_feat in features:
        # find all cols idx corresponding to a single feature to regroup the one-hot-encoded ones:
        col_indices = []
        for i, col_name in enumerate(clinical_features_names):
            if col_name == target_feat or col_name.startswith(target_feat + "_"):
                col_indices.append(i)

        correct = 0
        total = 0

        for data in loader:
            data_copy = data.clone().to(device)

            data_copy.clinical[:, :3] = (data_copy.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)
            data_copy.x[:, :4] = (data_copy.x[:, :4] - x_mean) / (x_std + 1e-6)
            data_copy.edge_attr[:, 2] = (data_copy.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)

            perm = torch.randperm(data_copy.clinical.size(0))

            # permutation applied to every col of the current feature
            for idx in col_indices:
                data_copy.clinical[:, idx] = data_copy.clinical[perm, idx]

            with torch.no_grad():
                if model.__class__ == CancerGNN or model.__class__ == GAT:
                    out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.batch)
                elif model.__class__ == MLP:
                    out = model(data_copy.clinical)
                else:  # MultiModalGNN
                    out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.clinical,
                                data_copy.batch)

                pred = out.argmax(dim=1)
                correct += int((pred == data_copy.y).sum())
                total += data_copy.num_graphs

        permuted_acc = correct / total
        feature_importances[target_feat] = baseline_acc - permuted_acc

    clinical_imp = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    logging.info("Clinical Features importance:")
    for name, imp in clinical_imp:
        logging.info(f"{name}: {imp:.4f}\n")

    return clinical_imp


def get_gene_attention_weights(model, device, loader, node_map_inv):
    """Extract genes with the highest attention weights from GAT.
    Says which ones are important "hubs" in the graph structure."""
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

    gene_importance = sorted(importance_list, key=lambda x: x[1], reverse=True)

    logging.info("Genes with attention importance = 1.000:")

    #TODO chekc data format e guarda solo quelli con score=1
    for gene_id, score in gene_imp:
        names = gene_alias[gene_alias['gene_id'] == gene_id]['names'].iloc[0]
        logging.info(f"{gene_id}: {score:.4f}   {names}")

    return gene_importance


def get_gene_saliency(model, device, loader, node_map_inv):
    """Compute genes saliency, which indicates what genes cause the prediction to move towards LUAD or LUSC."""
    model.eval()
    gene_accumulation = {}
    gene_counts = {}

    for data in loader:
        data_copy = data.clone().to(device)
        data_copy.x[:, :4] = (data_copy.x[:, :4] - x_mean) / (x_std + 1e-6)
        data_copy.edge_attr[:, 2] = (data_copy.edge_attr[:, 2] - e_min) / (e_max - e_min + 1e-6)
        data_copy.clinical[:, :3] = (data_copy.clinical[:, :3] - clinical_mean) / (clinical_std + 1e-6)

        data_copy.x.requires_grad = True  # to trace operations on the tensor and accumulate grads in x.grad attribute

        out = model(data_copy.x, data_copy.edge_index, data_copy.edge_attr, data_copy.clinical, data_copy.batch)
        probs = torch.softmax(out, dim=1)
        max_probs, _ = torch.max(probs, dim=1)

        model.zero_grad()
        max_probs.backward(torch.ones_like(max_probs))

        # avg of absolute gradient on node features
        saliency = data_copy.x.grad.abs().mean(dim=1).cpu().numpy()

        num_nodes_per_graph = data_copy.x.size(0) // data_copy.num_graphs

        for i in range(data_copy.x.size(0)):
            # retrieve original node indexes because data contains a batch of graphs (num nodes = num genes * num batches)
            # --> each graph has same gene idx order
            gene_idx_in_map = i % (data_copy.x.size(0) // data_copy.num_graphs)
            score = saliency[i]
            gene_id = node_map_inv.get(gene_idx_in_map, f"Unknown_{gene_idx_in_map}")

            gene_accumulation[gene_id] = gene_accumulation.get(gene_id, 0) + score
            gene_counts[gene_id] = gene_counts.get(gene_id, 0) + 1

    gene_saliency = []
    for gene_id in gene_accumulation:
        avg_score = gene_accumulation[gene_id] / gene_counts[gene_id]  # avg per gene_id
        gene_saliency.append((gene_id, avg_score))

    gene_saliency.sort(key=lambda x: x[1], reverse=True)

    # normalization 0-1 for first in the list
    if gene_saliency:
        max_val = gene_saliency[0][1]
        gene_saliency = [(g, s / max_val) for g, s in gene_saliency]

    logging.info("Top 100 Genes saliency:")

    for gene_id, score in gene_saliency[:100]:
        names = gene_alias[gene_alias['gene_id'] == gene_id]['names'].iloc[0]
        logging.info(f"{gene_id}: {score:.4f}   {names}")

    return gene_saliency


#TODO this
# I geni top dalla tua Saliency
top_genes = {
    'ENSG00000185201': 'IFITM2',
    'ENSG00000205420': 'KRT6C',
    'ENSG00000186081': 'KRT5',
    'ENSG00000147381': 'MAGEA4',
    'ENSG00000019582': 'CD74'
}


def plot_gene_distributions(genes_expression_df, gene_dict):
    plt.figure(figsize=(15, 10))
    for i, (ensg, name) in enumerate(gene_dict.items(), 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='label', y=ensg, data=genes_expression_df, palette='Set2')
        plt.title(f"{name}\n({ensg})")
        plt.xlabel("Sottotipo (0=LUAD, 1=LUSC)")
        plt.ylabel("Log2 Expression")

    plt.tight_layout()
    plt.show()

logging.info("--- Feature Importance analysis (Best Model Saved) ---\n")

model.load_state_dict(torch.load('best_k_fold_gnn.pth', map_location=device))  # currently model_fold_3.pth

clinical_names = ['age_at_index', 'tobacco_years', 'pack_years_smoked', 'country_of_residence_at_enrollment_Australia', 'country_of_residence_at_enrollment_Germany', 'country_of_residence_at_enrollment_United States', 'country_of_residence_at_enrollment_Switzerland', 'country_of_residence_at_enrollment_Russia', 'country_of_residence_at_enrollment_Canada', 'country_of_residence_at_enrollment_Ukraine', 'country_of_residence_at_enrollment_Romania', 'country_of_residence_at_enrollment_Vietnam', 'ethnicity_not hispanic or latino', 'ethnicity_hispanic or latino', 'gender_male', 'gender_female', 'race_white', 'race_black or african american', 'race_asian', 'ajcc_pathologic_m_M0', 'ajcc_pathologic_m_M1a', 'ajcc_pathologic_m_M1', 'ajcc_pathologic_m_M1b', 'ajcc_pathologic_n_N1', 'ajcc_pathologic_n_N0', 'ajcc_pathologic_n_N2', 'ajcc_pathologic_n_N3', 'ajcc_pathologic_t_T2a', 'ajcc_pathologic_t_T2b', 'ajcc_pathologic_t_T2', 'ajcc_pathologic_t_T3', 'ajcc_pathologic_t_T4', 'ajcc_pathologic_t_T1b', 'ajcc_pathologic_t_T1', 'ajcc_pathologic_t_T1a', '3', '1', '2', '9', '8', '0', 'laterality_Left', 'laterality_Right', 'sites_of_involvement_Peripheral Lung', 'sites_of_involvement_Central Lung', 'tissue_or_organ_of_origin_Lower lobe, lung', 'tissue_or_organ_of_origin_Upper lobe, lung', 'tissue_or_organ_of_origin_Middle lobe, lung', 'tissue_or_organ_of_origin_Lung, NOS', 'tissue_or_organ_of_origin_Overlapping lesion of lung', 'tissue_or_organ_of_origin_Main bronchus', 'tobacco_smoker', 'ajcc_pathologic_stage']
node_map_inv = {v: k for k, v in node_map.items()}

gene_alias = pd.read_csv('STRING_downloaded_files/9606.protein.aliases.gene.tsv', sep='\t', usecols=['gene_id', 'alias'])
gene_alias = gene_alias.groupby('gene_id')['alias'].apply(list).reset_index(name='names')
gene_alias['gene_id_mapped'] = gene_alias['gene_id'].map(node_map)
gene_alias.set_index('gene_id_mapped', inplace=True)

clinical_imp = explain_clinical_importance(model, device, test_loader, clinical_names)

#gene_imp = get_gene_attention_weights(model, device, test_loader, node_map_inv)  # (GAT Attention)

gene_sal = get_gene_saliency(model, device, test_loader, node_map_inv)

#plot_gene_distributions(gene_sal[:10], node_map)
