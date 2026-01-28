# mRNA_seq <-> mRNS_seq

import pandas as pd

pd.set_option('display.max_colwidth', None)

case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example

file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')

df_rna = pd.read_csv(f"files/RNA/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'RNA')
    ]['filename'].to_string(index=False)}", sep='\t' ,comment='#')  # 'comment=' to ignore the first line in RNA files
df_rna['gene_id'] = df_rna.gene_id.str.split('.', expand=True)[0]  # remove gene_id version (ENSG00000000003.15 --> ENSG00000000003)
df_rna.drop(df_rna[df_rna['gene_id'].str.startswith('ENSG') == False].index, inplace=True)  # to drop metadata
print(df_rna.head(1))