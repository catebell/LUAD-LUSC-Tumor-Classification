import pandas as pd
import torch
import torch_geometric.data

pd.set_option('display.max_colwidth', None)

case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example

file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')

df_rna = pd.read_csv(f"files/RNA/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'RNA')
    ]['filename'].to_string(index=False)}", sep='\t' ,comment='#')  # 'comment=' to ignore the first line in RNA files
df_rna['gene_id'] = df_rna.gene_id.str.split('.', expand=True)[0]  # remove gene_id version (ENSG00000000003.15 --> ENSG00000000003)
df_rna.drop(df_rna[df_rna['gene_id'].str.startswith('ENSG') == False].index, inplace=True)  # to drop metadata
print(df_rna.head(1))

df_cnv = pd.read_csv(f"files/CNV/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'CNV')
    ]['filename'].to_string(index=False)}", sep='\t')
df_cnv['gene_id'] = df_cnv.gene_id.str.split('.', expand=True)[0]  # remove gene_id version (ENSG00000000003.15 --> ENSG00000000003)
print(df_cnv.head(1))

# TODO Methylation (richiede una mappatura sonda -> gene)
df_meth_manifest = pd.read_csv("dataset/matched_cpg_genes_converted.csv", dtype=str)
df_meth_manifest.rename(columns={"cpg_IlmnID": "cg_id"}, inplace=True)

df_meth = pd.read_csv(f"files/methylation/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'methylation')
]['filename'].to_string(index=False)}", sep='\t', names=['cg_id', 'beta_value'])

df_meth = pd.merge(df_meth,df_meth_manifest[['cg_id', 'gene_id', 'gene_symbol', 'gene_chr', 'gene_strand', 'gene_start',
                                             'gene_end', 'cpg_island', 'cpg_chr']],on='cg_id', how='left')
print(df_meth.head(1))

df_omics = pd.merge(df_rna[['gene_id', 'gene_name', 'tpm_unstranded']],
                       df_cnv[['gene_id', 'gene_name', 'copy_number']],
                       on=['gene_id','gene_name'], how='inner')

df_omics = df_omics.fillna(0)

print(df_omics)

# adjacency matrix
num_nodes = len(df_omics)
edge_index = torch.tensor([[i, (i+1)%num_nodes] for i in range(num_nodes)], dtype=torch.long).t()
# TODO penso che ora crei edge a caso tra due nodi consecutivi, bisogna trovare una soglia o qualcosa del genere

print(edge_index)

# features of a node (gene)
x = torch.tensor(df_omics[['tpm_unstranded', 'copy_number']].values, dtype=torch.float)

data = torch_geometric.data.Data(x=x, edge_index=edge_index) # graph of case_id omics

print(data)