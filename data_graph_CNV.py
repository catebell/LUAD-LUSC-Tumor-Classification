import time
import pandas as pd
import stringdb

pd.set_option('display.max_colwidth', None)

# case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example
case_id = '23421041-fc75-4610-8a59-3246fb6df7e8'
file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')


# PROTEINS ALIASES
# file downloaded from https://string-db.org/cgi/download.pl selecting organism = Homo sapiens
# --> 9606.protein.aliases file under ACCESSORY DATA, place the .txt extracted into dataset/
print("Reading protein-aliases file...")
protein_aliases_df = pd.read_csv('dataset/9606.protein.aliases.v12.0.txt', sep='\t', usecols=['#string_protein_id', 'alias'])
#protein_aliases_df['alias'] = protein_aliases_df.alias.str.split('.', expand=True)[0] # BOTTLENECK
protein_aliases_df.drop_duplicates(inplace=True)
protein_aliases_df.reset_index(drop=True, inplace=True)
protein_aliases_df.rename(columns={'#string_protein_id': 'protein_id'}, inplace=True)

# unique protein_ids mapping to numerical index
unique_nodes = protein_aliases_df['protein_id'].unique()
node_map = {node: i for i, node in enumerate(unique_nodes)}  # TODO maybe with a LabelEncoder (https://stackoverflow.com/questions/44617871/how-to-convert-a-list-of-strings-into-a-tensor-in-pytorch)


print('Reading CNV data...')

start_time = time.time()

df_cnv = pd.read_csv(f"files/CNV/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'CNV')
    ]['filename'].to_string(index=False)}", sep='\t')

df_cnv.dropna(inplace=True)
df_cnv.reset_index(inplace=True, drop=True)
print(str(len(df_cnv)) + " rows\n")

# remove ids (Ensembl) version (ENSG00000000003.15 --> ENSG00000000003)
df_cnv['gene_id'] = df_cnv.gene_id.str.split('.', expand=True)[0]
# remove genes (names) version (AL627309.1 --> AL627309)
df_cnv['gene_name'] = df_cnv.gene_name.str.split('.', expand=True)[0]

print("\nCNV df created, like:")
print(str(df_cnv.head(1)) + '\n')

# TODO rimuovere i geni con normal state? (2 2 2) fare prove

print("\nAdding matches from file-aliases...")
protein_aliases_df.rename(columns={"alias": "gene_name"}, inplace=True)
df_cnv = pd.merge(df_cnv, protein_aliases_df,  how='left', on=['gene_name'])
df_cnv.dropna(subset=['protein_id'], inplace=True)
df_cnv.reset_index(inplace=True, drop=True)
print("--> " + str(len(df_cnv)) + " actual rows\n")

# TODO forse tenere da parte i geni con CNV sballati ma non protein-coing e aggiungerli come nodi isolati?



print("\n--- %s seconds ---" % (time.time() - start_time))
print("\nPROCESSED RNA DATA\n")