import pandas as pd
import stringdb

pd.set_option('display.max_colwidth', None)

case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example
file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')

df_rna = pd.read_csv(f"files/RNA/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'RNA')
    ]['filename'].to_string(index=False)}", sep='\t', dtype=str, comment='#')  # 'comment=' to ignore the first line in RNA files

# remove gene_id version (ENSG00000000003.15 --> ENSG00000000003)
df_rna['gene_id'] = df_rna.gene_id.str.split('.', expand=True)[0]
df_rna.drop(df_rna[df_rna['gene_id'].str.startswith('ENSG') == False].index, inplace=True)  # to drop metadata
print("RNA df created:")
print(df_rna.head(1))
print("...")

#

print((df_rna[(df_rna['gene_name'] == 'TNMD') | (df_rna['gene_name'] == 'TSPAN6') | (df_rna['gene_name'] == 'DPM1')]))

genes = ['TNMD', 'TSPAN6', 'DPM1']
# genes = ['TP53', 'BRCA1', 'FANCD1', 'FANCL']
print(df_rna['gene_name'].unique())
string_ids = stringdb.get_string_ids(genes)
enrichment_df = stringdb.get_enrichment(string_ids.queryItem)
network = stringdb.get_network(string_ids.queryItem) # ppi
print(string_ids)

# 'stringId' stripped before the dot (9606 = Homo sapiens)
