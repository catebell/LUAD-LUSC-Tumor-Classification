import pandas as pd
import stringdb

pd.set_option('display.max_colwidth', None)

case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example
file_manifestIllumina = 'files/clinical/file_case_mapping.tsv'

file_mapping_df = pd.read_csv(file_manifestIllumina, sep='\t')

df_meth_manifest = pd.read_csv("methylation_manifests/methylation_manifest450.tsv", sep='\t', dtype=str)

print('Reading methylation data...')

df_meth = pd.read_csv(f"files/methylation/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'methylation')
    ]['filename'].to_string(index=False)}", sep='\t', names=['cpg_IlmnID', 'beta_value'])
print("Methylation df created:")
print(str(df_meth.head(3)) + '\n')

print("Dropping data not useful...")
print(str(len(df_meth)) + " rows")
# drop rows not about 'cg...'
df_meth.drop(df_meth[df_meth['cpg_IlmnID'].str.startswith('cg') == False].index, inplace=True)
# drop rows with na
df_meth.dropna(inplace=True)
df_meth.reset_index(inplace=True, drop=True)
print("--> " + str(len(df_meth)) + " rows\n")

print("Merging with data of methylated gene from " + str(file_manifestIllumina) + "...")
df_meth = pd.merge(df_meth, df_meth_manifest, on='cpg_IlmnID', how='outer')

print("Removing " + str(df_meth['gene_symbol'].isna().sum()) + " cpgIDs-symbols values still missing (gene not tracked)...")
print(str(len(df_meth)) + " rows")
df_meth.dropna(subset=['gene_symbol'], axis=0, inplace=True)
df_meth.reset_index(inplace=True, drop=True)
print("--> " + str(len(df_meth)) + " rows\n")

#df_meth = pd.merge(df_meth,df_meth_manifest[['cpg_IlmnID', 'gene_id', 'gene_symbol', 'gene_chr', 'gene_strand', 'gene_start', 'gene_end', 'cpg_island', 'cpg_chr']],on='cpg_IlmnID', how='left')
# TODO perhaps unire i dati da altri manifest? quali? strand +/- corrispondono?

print("Getting gene symbols variants...")
symbols_list = df_meth['gene_symbol'].str.split(';')
df_meth['gene_symbol'] = symbols_list.apply(lambda x: list(set(x)))  # so to have single symbols variants

# TODO mean of b-values of cpg islands associated to same gene
print(' ')
