import logging
import time

import numpy as np
import pandas as pd
import torch
import torch_geometric
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.transforms import ToUndirected

from preprocessing_CNV_to_df import create_cnv_df
from preprocessing_RNA_to_df import create_rna_df
from preprocessing_methylation_to_df import create_meth_df
from models.GAT import GAT

"""
    Computes and returns the torch_geometric graph (torch_geometric.data.Data) of genes (gene_id is node identifier)
    for the specified patient (case_id) using coded protein-protein interactions and features from RNA, CNV and
    methylation data.
    :param case_id:
    :return torch_geometric.data.Data:
"""

pd.set_option('display.max_colwidth', None)

start_time = time.time()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../execution.log', mode='w'),
        logging.StreamHandler()
    ]
)

ppi_score_threshold = 0.7  # minimum interaction probability score to create edges

file_mapping_df = pd.read_csv('../files/clinical/file_case_mapping.tsv', sep='\t').dropna()

clinical_features_df = pd.read_csv('../files/clinical/features_encoded.tsv', sep='\t')
# map {case_id --> tumor class (LUAD-LUSC)}
labels_dict = dict(zip(clinical_features_df['case_id'], clinical_features_df['project_id'].astype(int)))

# GENES ALIASES WITH PROTEINS AND GENE IDS MAPPING
# file extracted using create_tsv_from_STRING_files.create_gene_aliases_proteins_ids_mapping_file()
logging.info("Reading protein-aliases-gene file...")
genes_mapping_df = pd.read_csv('../STRING_downloaded_files/9606.protein.aliases.gene.tsv', sep='\t')

# unique genes_ids mapping to numerical index
unique_nodes = genes_mapping_df['gene_id'].unique()
node_map = {node: i for i, node in enumerate(unique_nodes)}  # TODO maybe with a LabelEncoder (https://stackoverflow.com/questions/44617871/how-to-convert-a-list-of-strings-into-a-tensor-in-pytorch)


# PROTEINS LINKS
# file downloaded from https://string-db.org/cgi/download.pl selecting organism = Homo sapiens
# --> 9606.protein.links file under INTERACTION DATA, place the .txt extracted into original_dataset/
logging.info("Reading protein-links file...")
protein_links_df = pd.read_csv('../STRING_downloaded_files/9606.protein.links.v12.0.txt', sep=' ')

# refactor the score in a [0-1] interval, like returned by stringdb.get_network()
protein_links_df['combined_score'] = protein_links_df['combined_score'] / 1000

logging.info("Dropping interactions with combined probability score lower than " + str(ppi_score_threshold) + "...")
# filter based on score (probability of interacting)
protein_links_df.drop(protein_links_df[protein_links_df['combined_score'] < ppi_score_threshold].index, inplace=True)
protein_links_df.reset_index(inplace=True, drop=True)


# METHYLATION ILLUMINA MANIFEST FOR CpG-GENE MAPPING
logging.info("Reading Illumina manifest...")
# file downloaded from https://support.illumina.com/downloads/infinium_humanmethylation450_product_files.html
# place .csv file into methylation_manifests/originals_downloaded, then run methylation_manifest_to_tsv.py
meth_manifest_df = pd.read_csv("../methylation_manifests/methylation_manifest450.tsv", sep='\t', dtype=str)

###

'''
#patient_list = file_mapping_df['case_id'].dropna().unique()  # case_id list
print(file_mapping_df['case_id'].value_counts())
# there are some patients with more or less than 3 omics files: we leave them out
patient_list = file_mapping_df.groupby('case_id').agg(case_id = ('case_id','first'), num_omics_present=('filename', 'count'))
patient_list.drop(patient_list[patient_list['num_omics_present'] != 3].index, inplace=True)
patient_list.reset_index(inplace=True, drop=True)
'''

case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb'  # for example

df_rna, network_df = create_rna_df(case_id, file_mapping_df, genes_mapping_df, protein_links_df)
df_rna['gene_id_mapped'] = df_rna['gene_id'].map(node_map)

df_cnv = create_cnv_df(case_id, file_mapping_df, genes_mapping_df)

node_features_df = pd.merge(df_rna, df_cnv, how='left', on=['gene_id'])

df_meth = create_meth_df(case_id, file_mapping_df, genes_mapping_df, meth_manifest_df)
node_features_df = pd.merge(node_features_df, df_meth, how='left', on=['gene_id'])

node_features_df[['copy_number', 'cnv_min_max_diff', 'weighted_beta_value']] = node_features_df[['copy_number', 'cnv_min_max_diff', 'weighted_beta_value']].fillna(0)

node_features_df['meth_data_present'] = np.where(node_features_df['weighted_beta_value'] > 0, 1, 0)
# so the net should learn that when meth_data_present = 0 weighted_beta_value doesn't matter

# only numerical features assigned to nodes, to be used for classification
features_cols = ['tpm_unstranded', 'copy_number', 'cnv_min_max_diff', 'weighted_beta_value', 'meth_data_present']
node_features_df[features_cols] = node_features_df[features_cols].astype(float)


# unique nodes df
node_features_df = node_features_df.groupby('gene_id_mapped').agg({
    'gene_id': 'first',
    'gene_name': 'first',
    'tpm_unstranded': 'mean',
    'copy_number': 'mean',
    'cnv_min_max_diff': 'max',
    'weighted_beta_value': 'mean',
    'meth_data_present': 'max'
})

node_features_df['tpm_unstranded'] = np.log1p(node_features_df['tpm_unstranded'])

# interactions aggregation by genes
edge_features_df = network_df.groupby(['gene1', 'gene2']).agg(avg_combined_score=('combined_score', 'mean'),
                                                              max_combined_score=('combined_score', 'max'),
                                                              num_protein_links=('combined_score', 'count')  # how many proteins of these 2 genes interacts
).reset_index()


# no bidirectional edges, added later: here we remove duplicate pairs after sort (gene1 < gene2)
edge_features_df[['gene1', 'gene2']] = np.sort(edge_features_df[['gene1', 'gene2']].values, axis=1)
edge_features_df = edge_features_df.groupby(['gene1', 'gene2']).agg({'avg_combined_score': 'mean',
                                                                    'max_combined_score': 'max',
                                                                    'num_protein_links': 'mean'}).reset_index()

# add correlations computed on omics in file merged_gene_matrix.tsv
#omic_correlations_df = pd.read_csv('edge_weights/merged_gene_matrix.tsv', sep='\t')
#edge_features_df = pd.merge(edge_features_df, omic_correlations_df, how='left', on=['gene1','gene2'])

# not necessary for computation/classification, torch_geometric is enough
'''
print("Creating NetworkX graph...")

# values are duplicated (both ways interactions), keep only one score for pair since nx.Graph() is undirected by default
cols = ['gene1', 'gene2']
network_df.loc[:, cols] = np.sort(network_df[cols].values, axis=1)

G = nx.from_pandas_edgelist(edge_features_df.drop_duplicates(), source='gene1', target='gene2', edge_attr='combined_score')

# adding features of a node (gene) and isolated nodes
# TODO maybe only numerical features like for pytorch_geometric, or maybe here we don't mind
df_nodes_features = df_rna[['gene_id', 'gene_name', 'protein_id', 'tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded']]
df_nodes_features.set_index('gene_id', inplace=True)
G.add_nodes_from((n, dict(d)) for n, d in df_nodes_features.iterrows())
'''

logging.info("Creating torch_geometric.data.Data graph...")

# data = torch_geometric.utils.from_networkx(G)  # TOO SLOW, INTERNAL CONVERSION TO TENSORS IS BOTTLENECK

# we need to consider in x every possible gene, so the edge index will have the correct numbers (pos. in x = gene id)
num_nodes = len(node_map)

# only the rows corresponding to this specific patient genes are filled (non-zero), with gene_id_mapped as index
x_df = node_features_df.reindex(np.arange(0, num_nodes), fill_value=0)
x = torch.tensor(x_df.loc[x_df.index, features_cols].values.astype(float), dtype=torch.float)

edge_index = torch.as_tensor(np.stack([
    edge_features_df['gene1'].map(node_map).values, edge_features_df['gene2'].map(node_map).values
]), dtype=torch.long)
edge_attr = torch.tensor(edge_features_df[['avg_combined_score', 'max_combined_score', 'num_protein_links']].values, dtype=torch.float)

data = torch_geometric.data.Data(x=x, edge_index=edge_index,
                                 edge_attr=edge_attr)  # graph of genes from patient 'case_id'


transform = T.Compose([T.AddSelfLoops(attr='edge_attr'), ToUndirected()])
data = transform(data)


# ADD CLINICAL FEATURES TENSOR
clinical_values = clinical_features_df[clinical_features_df['case_id'] == case_id].iloc[:, 2:]

data.clinical = torch.tensor(clinical_values.values.astype(float), dtype=torch.float)

logging.info(data)

print("\n--- %s seconds ---" % (time.time() - start_time))
logging.info("\nGRAPH FOR PATIENT %s CREATED\n" % case_id)

scaler = StandardScaler()
#print(scaler.fit(node_features_df[features_cols]))
#print(scaler.mean_)
#print(scaler.transform(node_features_df[features_cols]))
print(scaler)

# scaling
cols_to_scale = ['tpm_unstranded', 'copy_number', 'cnv_min_max_diff', 'weighted_beta_value']
node_features_df[cols_to_scale] = StandardScaler().fit_transform(node_features_df[cols_to_scale])

cols_to_scale = ['num_protein_links']
edge_features_df[cols_to_scale] = MinMaxScaler().fit_transform(edge_features_df[cols_to_scale])

data.y = torch.tensor([labels_dict[case_id]], dtype=torch.long)  # instead of torch.int
criterion = torch.nn.CrossEntropyLoss()
model = GAT(num_node_features=5, num_edge_features=3, num_classes=2, hidden_channels=64)
out = model(data.x, data.edge_index, data.edge_attr, data.batch)
loss = criterion(out, data.y)

print(loss)
