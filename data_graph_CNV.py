import time

import pandas as pd
import stringdb
from duckdb.experimental.spark import DataFrame

pd.set_option('display.max_colwidth', None)

# case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example
case_id = '23421041-fc75-4610-8a59-3246fb6df7e8'
file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')


# PROTEINS INFO
# file downloaded from https://string-db.org/cgi/download.pl selecting organism = Homo sapiens
# --> first file under ACCESSORY DATA, place the .txt extracted into dataset/
print("Uploading protein-info file...")
protein_info_df = pd.read_csv('dataset/9606.protein.info.v12.0.txt', sep='\t')

# unique stringIds mapping to numerical index
unique_nodes = protein_info_df['#string_protein_id'].unique()
node_map = {node: i for i, node in enumerate(unique_nodes)}  # TODO maybe with a LabelEncoder (https://stackoverflow.com/questions/44617871/how-to-convert-a-list-of-strings-into-a-tensor-in-pytorch)


print('Reading CNV data...')

start_time = time.time()

df_cnv = pd.read_csv(f"files/CNV/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'CNV')
    ]['filename'].to_string(index=False)}", sep='\t')

# remove ids (Ensembl) version (ENSG00000000003.15 --> ENSG00000000003)
df_cnv['gene_id'] = df_cnv.gene_id.str.split('.', expand=True)[0]

df_cnv.dropna(inplace=True)
df_cnv.reset_index(inplace=True, drop=True)

print("CNV df created:")
print(str(df_cnv.head(3)) + '\n')

'''
columns = ["gene_id", "copy_number", "min_copy_number", "max_copy_number"]
cnv = pd.DataFrame(data=df_cnv, columns = columns)
cnv.dropna(inplace=True)
cnv.reset_index(inplace=True, drop=True)

print("CNV:\n", cnv.head)

for index, row in cnv.iterrows():
    cnv["copy_number_diff"] = cnv["max_copy_number"] - cnv["min_copy_number"]
print(cnv.head)

cnv.to_csv(r"files/clinical/cnv.tsv", sep="\t", index=False)
'''

# TODO rimuovere i geni con normal state? (2 2 2) fare prove

genes = list(df_cnv['gene_name'].astype(str))

# TODO split nomi dopo il .? Come identificare quelli protein coding, usare file RNA associato con i gene id?
df_cnv['gene_name'] = df_cnv.gene_name.str.split('.', expand=True)[0]

print("\nRetrieving genes Ensembl ids from STRING...")
string_ids = stringdb.get_string_ids(genes)

# drop genes not found in string db
# TODO ma può essere utile tenerne se hanno valori sballati anche se non protein coding, magari aggungendo nodi isolati
df_cnv.drop(df_cnv[df_cnv['gene_name'].isin(string_ids['queryItem']) == False].index, inplace=True)
df_cnv.reset_index(inplace=True, drop=True)

print("--- %s seconds ---" % (time.time() - start_time))
print("\nPROCESSED RNA DATA\n")