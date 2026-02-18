import os
import time

import numpy as np
import torch
import pandas as pd
import torch_geometric
from torch_geometric.data import Dataset
import torch_geometric.transforms as T

from extract_CNV_data import create_cnv_df
from extract_RNA_data import create_rna_df
from extract_methylation_data import create_meth_df


start_time = time.time()

file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')
patients_features_df = pd.read_csv('files/clinical/features.tsv', sep='\t')

ppi_score_threshold = 0.7  # minimum interaction probability score to create edges


# GENES ALIASES WITH PROTEINS AND GENE IDS MAPPING
# file extracted using genes_proteins_aliases_ensg_mapping.py
print("Reading protein-aliases-gene file...")
genes_mapping_df = pd.read_csv('downloaded_files/9606.protein.aliases.gene.tsv', sep='\t')

# unique genes_ids mapping to numerical index
unique_nodes = genes_mapping_df['gene_id'].unique()
node_map = {node: i for i, node in enumerate(unique_nodes)}  # TODO maybe with a LabelEncoder (https://stackoverflow.com/questions/44617871/how-to-convert-a-list-of-strings-into-a-tensor-in-pytorch)


# PROTEINS LINKS
# file downloaded from https://string-db.org/cgi/download.pl selecting organism = Homo sapiens
# --> 9606.protein.links file under INTERACTION DATA, place the .txt extracted into original_dataset/
print("Reading protein-links file...")
protein_links_df = pd.read_csv('downloaded_files/9606.protein.links.v12.0.txt', sep=' ')

# refactor the score in a [0-1] interval, like returned by stringdb.get_network()
protein_links_df['combined_score'] = protein_links_df['combined_score'] / 1000

print("Dropping interactions with combined probability score lower than " + str(ppi_score_threshold) + "...")
# filter based on score (probability of interacting)
protein_links_df.drop(protein_links_df[protein_links_df['combined_score'] < ppi_score_threshold].index, inplace=True)
protein_links_df.reset_index(inplace=True, drop=True)


# METHYLATION ILLUMINA MANIFEST FOR CpG-GENE MAPPING
print("Reading Illumina manifest...")
# file downloaded from https://support.illumina.com/downloads/infinium_humanmethylation450_product_files.html
# place .csv file into methylation_manifests/originals, then run methylation_manifest_to_tsv.py
meth_manifest_df = pd.read_csv("methylation_manifests/methylation_manifest450.tsv", sep='\t', dtype=str)

###

pd.set_option('display.max_colwidth', None)


class PatientGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        Class for the LUAD/LUSC cancer patient graphs Dataset.
        :param root: directory where to save processed data
        """

        # case_id list
        self.patient_list = file_mapping_df.groupby('case_id').agg(num_omics_present=('filename', 'count'))
        # drop patient with incorrect number if omic files (should be 3)
        self.patient_list.drop(self.patient_list[self.patient_list['num_omics_present'] != 3].index, inplace=True)

        self.node_map = node_map  # global dictionary for Gene_ID -> Index mapping
        self.genes_mapping_df = genes_mapping_df  # proteins-genes mapping df

        mapping_project_id = {
            'TCGA-LUAD': 0,
            'TCGA-LUSC': 1
        }
        patients_features_df['project.project_id'] = patients_features_df['project.project_id'].map(mapping_project_id)
        # map {case_id --> tumor class (LUAD-LUSC)}
        self.labels_dict = dict(zip(patients_features_df['cases.case_id'], patients_features_df['project.project_id']))

        self.standard_transform = T.Compose([T.AddSelfLoops(attr='edge_attr')])

        super(PatientGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        :return: Original files list (for each patient: RNA.tsv, CNV.tsv, methylation.txt)
        """
        '''
        return [(
            f"files/RNA/{file_mapping_df[(file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'RNA')]['filename'].to_string(index=False)},"
            f"files/CNV/{file_mapping_df[(file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'CNV')]['filename'].to_string(index=False)},"
            f"files/methylation/{file_mapping_df[(file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'methylation')]['filename'].to_string(index=False)}"
        ) for case_id in self.patient_list]
        '''
        return []

    @property
    def processed_file_names(self):
        # File names saved after graph conversion.
        return [f"data_{case_id}.pt" for case_id in self.patient_list]

    def download(self):
        # Download to `self.raw_dir`. Not required if data already present on disk.
        #path = download_url(url, self.raw_dir)
        pass

    def process(self):
        # Single execution to convert everything in .pt
        for case_id in self.patient_list.index:
            df_rna, network_df = create_rna_df(case_id, file_mapping_df, genes_mapping_df, protein_links_df)
            df_rna['gene_id_mapped'] = df_rna['gene_id'].map(node_map)

            df_cnv = create_cnv_df(case_id, file_mapping_df, genes_mapping_df)

            node_features_df = pd.merge(df_rna, df_cnv, how='left', on=['gene_id'])

            df_meth = create_meth_df(case_id, file_mapping_df, genes_mapping_df, meth_manifest_df)
            node_features_df = pd.merge(node_features_df, df_meth, how='left', on=['gene_id'])

            node_features_df[['copy_number', 'cnv_min_max_diff', 'weighted_beta_value']] = node_features_df[
                ['copy_number', 'cnv_min_max_diff', 'weighted_beta_value']].fillna(0)

            node_features_df['meth_data_present'] = np.where(node_features_df['weighted_beta_value'] > 0, 1, 0)
            # so the net should learn that when meth_data_present = 0 weighted_beta_value doesn't matter

            data = self._create_graph(node_features_df, network_df)

            data.y = torch.tensor([self.labels_dict[case_id]], dtype=torch.long)

            data = self.standard_transform(data)
            torch.save(data, os.path.join(self.processed_dir, f'data_{case_id}.pt'))

    def _create_graph(self, node_features_df, network_df):
        """
        Creates graph of genes for patient 'case_id'.
        :param node_features_df:
        :param network_df:
        :return:
        """

        # only numerical features assigned to nodes, to be used for classification
        features_cols = ['tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded', 'copy_number',
                         'cnv_min_max_diff', 'weighted_beta_value', 'meth_data_present']
        node_features_df[features_cols] = node_features_df[features_cols].astype(float)

        # unique nodes df
        node_features_df = node_features_df.groupby('gene_id_mapped').agg({
            'gene_id': 'first',
            'gene_name': 'first',
            'tpm_unstranded': 'mean',
            'fpkm_unstranded': 'mean',
            'fpkm_uq_unstranded': 'mean',
            'copy_number': 'mean',
            'cnv_min_max_diff': 'max',
            'weighted_beta_value': 'mean',
            'meth_data_present': 'max'
        })

        # interactions aggregation by genes
        edge_features_df = network_df.groupby(['gene1', 'gene2']).agg(avg_combined_score=('combined_score', 'mean'),
                                                                      max_combined_score=('combined_score', 'max'),
                                                                      # how many proteins of these 2 genes interacts
                                                                      num_protein_links=('combined_score', 'count')
                                                                      ).reset_index()

        # we need to consider in x every possible gene, so the edge index will have the correct numbers (pos. in x = gene id)
        num_nodes = len(self.node_map)

        # only the rows corresponding to this specific patient genes are filled (non-zero), with gene_id_mapped as index
        x_df = node_features_df.reindex(np.arange(0, num_nodes), fill_value=0)
        x = torch.tensor(x_df.loc[x_df.index, features_cols].values.astype(float), dtype=torch.float)

        edge_index = torch.as_tensor(np.stack([
            edge_features_df['gene1'].map(node_map).values, edge_features_df['gene2'].map(node_map).values
        ]), dtype=torch.long)
        edge_attr = torch.tensor(
            edge_features_df[['avg_combined_score', 'max_combined_score', 'num_protein_links']].values,
            dtype=torch.float)

        data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Upload processed graph from disk.
        case_id = self.patient_list[idx]
        data = torch.load(os.path.join(self.processed_dir, f'data_{case_id}.pt'))


from torch_geometric.loader import DataLoader

# init
dataset = PatientGraphDataset(root='data_graphs_processed')

print("\n--- %s seconds ---" % (time.time() - start_time))
print("\nALL DATA PROCESSED\n")

# Train/Test subdivision
'''
train_dataset = dataset[:80]
test_dataset = dataset[80:]
'''

# DataLoader
'''
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for batch in train_loader:
    # batch.x -> Feature di tutti i nodi del batch
    # batch.edge_index -> Archi aggiornati per il batch
    # batch.batch -> Vettore che indica a quale paziente appartiene ogni nodo
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    # loss = ...
'''