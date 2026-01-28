import pandas as pd
import torch
from torch_geometric.data import Data

pd.set_option('display.max_colwidth', None)

case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example

file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')

df_rna = pd.read_csv(f"files/RNA/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'RNA')
    ]['filename'].to_string(index=False)}", sep='\t' ,comment='#') # 'comment=' to ignore the first line in files
print(df_rna.head(1))

df_cnv = pd.read_csv(f"files/CNV/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'CNV')
    ]['filename'].to_string(index=False)}", sep='\t')
print(df_cnv.head(1))

# TODO Methylation (richiede una mappatura sonda -> gene)
'''
df_meth_manifest = pd.read_csv("files/methylation_manifests/methylation_manifest27.tsv", sep='\t')

df_meth = pd.read_csv(f"files/methylation/{dataframe_tsv[
    (dataframe_tsv['case_id'] == case_id) & (dataframe_tsv['omic'] == 'methylation')
]['filename'].to_string(index=False)}", sep='\t', names=['cg_id', 'beta_value'])

df_meth = pd.merge(df_meth,df_meth_manifest[['cg_id','chr', 'gene_id']],on='cg_id', how='left')
print(df_meth.head(1))
'''

df_omics = pd.merge(df_rna[['gene_id', 'gene_name', 'tpm_unstranded']],
                       df_cnv[['gene_id', 'gene_name', 'copy_number']],
                       on=['gene_id','gene_name'], how='inner')

df_omics = df_omics.fillna(0)

print(df_omics)

# adjacency matrix
num_nodes = len(df_omics)
edge_index = torch.tensor([[i, (i+1)%num_nodes] for i in range(num_nodes)], dtype=torch.long).t()
# TODO penso che ora crei edge a caso tra due nodi con stesso gene_id, bisogna trovare una soglia o qualcosa del genere

print(edge_index)

# features of a node (gene)
x = torch.tensor(df_omics[['tpm_unstranded', 'copy_number']].values, dtype=torch.float)

data = Data(x=x, edge_index=edge_index) # graph of case_id omics

print(data)