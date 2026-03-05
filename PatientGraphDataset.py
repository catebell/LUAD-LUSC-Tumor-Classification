import os
import time

import numpy as np
import torch
import pandas as pd
import torch_geometric
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.transforms import ToUndirected


pd.set_option('display.max_colwidth', None)


class PatientGraphDataset(Dataset):
    def __init__(self, root, file_mapping_df: pd.DataFrame, transform=None, pre_transform=None):
        """
        Class for the LUAD/LUSC cancer patient graphs Dataset.
        :param root: directory where to save and retrieve processed data.
        :param file_mapping_df: df of data portions to consider.
        """
        self.file_mapping_df = file_mapping_df
        self.patients_features_df = pd.read_csv('files/clinical/features.tsv', sep='\t')

        # case_id list
        self.patient_list = self.file_mapping_df.groupby('case_id').agg(case_id=('case_id', 'first'),
                                                                   num_omics_present=('filename', 'count'))
        # drop patient with incorrect number if omic files (should be 3)
        self.patient_list.drop(self.patient_list[self.patient_list['num_omics_present'] != 3].index, inplace=True)
        self.patient_list = self.patient_list['case_id'].to_list()
        self.node_map = pd.read_csv('downloaded_files/gene_ids_mapped.tsv', sep='\t')

        mapping_project_id = {
            'TCGA-LUAD': 0,
            'TCGA-LUSC': 1
        }
        self.patients_features_df['project.project_id'] = self.patients_features_df['project.project_id'].map(mapping_project_id)
        # map {case_id --> tumor class (LUAD-LUSC)}
        self.labels_dict = dict(zip(self.patients_features_df['cases.case_id'], self.patients_features_df['project.project_id']))

        self.standard_transform = T.Compose([T.AddSelfLoops(attr='edge_attr'), ToUndirected()])

        self.ppi_score_threshold = 0.7  # minimum interaction probability score to create edges

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
        from extract_CNV_data import create_cnv_df
        from extract_RNA_data import create_rna_df
        from extract_methylation_data import create_meth_df

        start_time = time.time()

        # GENES ALIASES WITH PROTEINS AND GENE IDS MAPPING
        # file extracted using string_files_to_tsv.py --> create_protein_links()
        genes_mapping_df = pd.read_csv('downloaded_files/9606.protein.aliases.gene.tsv',
                                            sep='\t')  # proteins-genes mapping df

        # PROTEINS LINKS
        # file downloaded from https://string-db.org/cgi/download.pl selecting organism = Homo sapiens
        # --> 9606.protein.links file under INTERACTION DATA, place the .txt extracted into original_dataset/
        print("Reading protein-links file...")
        protein_links_df = pd.read_csv('downloaded_files/9606.protein.links.v12.0.txt', sep=' ')

        # refactor the score in a [0-1] interval, like returned by stringdb.get_network()
        protein_links_df['combined_score'] = protein_links_df['combined_score'] / 1000

        print("Dropping interactions with combined probability score lower than " + str(self.ppi_score_threshold) + "...")
        # filter based on score (probability of interacting)
        protein_links_df.drop(protein_links_df[protein_links_df['combined_score'] < self.ppi_score_threshold].index,
                              inplace=True)
        protein_links_df.reset_index(inplace=True, drop=True)

        # METHYLATION ILLUMINA MANIFEST FOR CpG-GENE MAPPING
        print("Reading Illumina manifest...")
        # file downloaded from https://support.illumina.com/downloads/infinium_humanmethylation450_product_files.html
        # place .csv file into methylation_manifests/originals, then run methylation_manifest_to_tsv.py
        meth_manifest_df = pd.read_csv("methylation_manifests/methylation_manifest450.tsv", sep='\t', dtype=str)


        for case_id in self.patient_list:
            df_rna, network_df = create_rna_df(case_id, self.file_mapping_df, genes_mapping_df, protein_links_df)
            df_rna['gene_id_mapped'] = df_rna['gene_id'].map(self.node_map)

            df_cnv = create_cnv_df(case_id, self.file_mapping_df, genes_mapping_df)

            node_features_df = pd.merge(df_rna, df_cnv, how='left', on=['gene_id'])

            df_meth = create_meth_df(case_id, self.file_mapping_df, genes_mapping_df, meth_manifest_df)
            node_features_df = pd.merge(node_features_df, df_meth, how='left', on=['gene_id'])

            node_features_df[['copy_number', 'cnv_min_max_diff', 'weighted_beta_value']] = node_features_df[
                ['copy_number', 'cnv_min_max_diff', 'weighted_beta_value']].fillna(0)

            node_features_df['meth_data_present'] = np.where(node_features_df['weighted_beta_value'] > 0, 1, 0)
            # so the net should learn that when meth_data_present = 0 weighted_beta_value doesn't matter

            data = self._create_graph(node_features_df, network_df)

            data.y = torch.tensor([self.labels_dict[case_id]], dtype=torch.int)

            data = self.standard_transform(data)
            torch.save(data, os.path.join(self.processed_dir, f'data_{case_id}.pt'))

        print("\n--- %s seconds ---" % (time.time() - start_time))
        print("\nALL DATA PROCESSED\n")

    def _create_graph(self, node_features_df, network_df):
        """
        Creates graph of genes for patient 'case_id'.
        :param node_features_df:
        :param network_df:
        :return:
        """

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

        node_features_df['tpm_unstranded'] = np.log1p(node_features_df['tpm_unstranded'])  # so not to have too different scales

        # interactions aggregation by genes
        edge_features_df = network_df.groupby(['gene1', 'gene2']).agg(avg_combined_score=('combined_score', 'mean'),
                                                                      max_combined_score=('combined_score', 'max'),
                                                                      # how many proteins of these 2 genes interacts
                                                                      num_protein_links=('combined_score', 'count')
                                                                      ).reset_index()

        # no bidirectional edges, added later: here we remove duplicate pairs after sort (gene1 < gene2)
        edge_features_df[['gene1', 'gene2']] = np.sort(edge_features_df[['gene1', 'gene2']].values, axis=1)
        edge_features_df = edge_features_df.groupby(['gene1', 'gene2']).agg({'avg_combined_score': 'mean',
                                                                             'max_combined_score': 'max',
                                                                             'num_protein_links': 'mean'}).reset_index()

        # we need to consider in x every possible gene, so the edge index will have the correct numbers (pos. in x = gene id)
        num_nodes = len(self.node_map)

        # only the rows corresponding to this specific patient genes are filled (non-zero), with gene_id_mapped as index
        x_df = node_features_df.reindex(np.arange(0, num_nodes), fill_value=0)
        x = torch.tensor(x_df.loc[x_df.index, features_cols].values.astype(float), dtype=torch.float)

        edge_index = torch.as_tensor(np.stack([
            edge_features_df['gene1'].map(self.node_map).values, edge_features_df['gene2'].map(self.node_map).values
        ]), dtype=torch.int64)
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
        data = torch.load(os.path.join(self.processed_dir, f'data_{case_id}.pt'), weights_only=False)
        return data
