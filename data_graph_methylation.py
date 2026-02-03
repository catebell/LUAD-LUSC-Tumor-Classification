import pandas as pd
import stringdb

pd.set_option('display.max_colwidth', None)

case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example
file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')

df_meth_manifest = pd.read_csv("dataset/matched_cpg_genes_converted.csv", dtype=str)
df_meth_manifest.rename(columns={"cpg_IlmnID": "cg_id"}, inplace=True)

df_meth = pd.read_csv(f"files/methylation/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'methylation')
    ]['filename'].to_string(index=False)}", sep='\t', names=['cg_id', 'beta_value'])

# merge data of methylated gene
df_meth = pd.merge(df_meth,df_meth_manifest[['cg_id', 'gene_id', 'gene_symbol', 'gene_chr', 'gene_strand', 'gene_start',
                                             'gene_end', 'cpg_island', 'cpg_chr']],on='cg_id', how='left')
print("Methylation df created:")
print(df_meth.head(1))
print("...")


print(df_meth['gene_symbol'])
print(df_meth['gene_symbol'].isna().sum())

# mean of b-values of cpg islands associated to same gene

'''
genes = df_meth
string_ids = stringdb.get_string_ids(genes)
enrichment_df = stringdb.get_enrichment(string_ids.queryItem)
network = stringdb.get_network(string_ids.queryItem) # ppi
print(string_ids)
'''