import pandas as pd
from duckdb.experimental.spark import DataFrame

pd.set_option('display.max_colwidth', None)

# case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example
case_id = '23421041-fc75-4610-8a59-3246fb6df7e8'
file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')

df_cnv = pd.read_csv(f"files/CNV/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'CNV')
    ]['filename'].to_string(index=False)}", sep='\t')

# remove gene_id version (ENSG00000000003.15 --> ENSG00000000003)
df_cnv['gene_id'] = df_cnv.gene_id.str.split('.', expand=True)[0]
print("CNV df created:")
print(df_cnv.head)
print("...")
# print(df_cnv.dtypes)
# print(df_cnv.isna().sum())

columns = ["gene_id", "copy_number", "min_copy_number", "max_copy_number"]
cnv = pd.DataFrame(data=df_cnv, columns = columns)
cnv.dropna(inplace=True)
print("CNV:\n", cnv.head)

for index, row in cnv.iterrows():
    cnv["copy_number_diff"] = cnv["max_copy_number"] - cnv["min_copy_number"]
print(cnv.head)

cnv.to_csv(r"files/clinical/cnv.tsv", sep="\t", index=False)