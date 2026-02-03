import pandas as pd
import stringdb

pd.set_option('display.max_colwidth', None)

case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example
file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')

df_meth_manifest450 = pd.read_csv("methylation_manifests/methylation_manifest450.tsv", sep='\t', dtype=str)

df_meth = pd.read_csv(f"files/methylation/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'methylation')
    ]['filename'].to_string(index=False)}", sep='\t', names=['cpg_IlmnID', 'beta_value'])

print("Methylation df created:")
print(df_meth.head(1))
print("...")

# drop rows not about 'cg...'
df_meth.drop(df_meth[df_meth['cpg_IlmnID'].str.startswith('cg') == False].index, inplace=True)

# drop rows with na, not useful
df_meth.dropna(inplace=True)

# merge data of methylated gene
df_meth = pd.merge(df_meth, df_meth_manifest450, on='cpg_IlmnID', how='outer')

print("Merged with methylation_manifest450.tsv")
print("Correspondences cpgIDs-symbols still missing: " + str(df_meth['gene_symbol'].isna().sum()))

#df_meth = pd.merge(df_meth,df_meth_manifest[['cpg_IlmnID', 'gene_id', 'gene_symbol', 'gene_chr', 'gene_strand', 'gene_start', 'gene_end', 'cpg_island', 'cpg_chr']],on='cpg_IlmnID', how='left')
# TODO unire i dati da altri manifest? quali? strand +/- corrispondono?

print("Removing values not tracking to a gene:")
print(df_meth['gene_symbol'].isna().sum())
df_meth.dropna(subset=['gene_symbol'], axis=0, inplace=True)

symbols_list = df_meth['gene_symbol'].str.split(';')
df_meth['gene_symbol'] = symbols_list.apply(lambda x: list(set(x)))  # so to have single symbols variants

# TODO mean of b-values of cpg islands associated to same gene
print(' ')
