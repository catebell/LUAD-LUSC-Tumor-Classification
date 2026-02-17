import mygene
import pandas as pd

'''
Run to re-create the 9606.protein.aliases.gene txt file with protein_id, gene_name and gene_id mapping and excluding
data not useful. Output will be save in 'output_filepath'.
Takes a lot of time.
'''

output_filepath = "downloaded_files/9606.protein.aliases.gene.tsv"

def map_symbols_to_ensg(gene_symbols):
    mg = mygene.MyGeneInfo()

    results = mg.querymany(gene_symbols,
                           scopes='symbol,alias',  # search in symbols column and expanding to aliases
                           fields='ensembl.gene',  # request Ensembl IDs
                           species='human',
                           returnall=False)

    mapping = {}
    for res in results:
        symbol = res['query']
        if 'ensembl' in res:
            # Ensembl might return a list if there are more IDs (rare for genes)
            ensg_data = res['ensembl']
            if isinstance(ensg_data, list):
                mapping[symbol] = ensg_data[0]['gene']
            else:
                mapping[symbol] = ensg_data['gene']

    return mapping

'''
# Usage example:
my_genes = ["ARF5", "TSPAN6", "EAW88598"]
mapping_dict = map_symbols_to_ensg(my_genes)

print(mapping_dict)
# Output: {'ARF5': 'ENSG00000000233', 'TSPAN6': 'ENSG00000000003', ...}
'''

# PROTEINS ALIASES
# file downloaded from https://string-db.org/cgi/download.pl selecting organism = Homo sapiens
# --> 9606.protein.aliases file under ACCESSORY DATA, place the .txt extracted into original_dataset/
print("Reading protein-aliases file...")
df_iter = pd.read_csv('downloaded_files/9606.protein.aliases.v12.0.txt', sep='\t',
                      usecols=['#string_protein_id', 'alias'], chunksize=10000)  # Process 50000 rows at a time

protein_aliases_df = pd.DataFrame(columns=['protein_id', 'alias', 'gene_id'])

for chunk in df_iter:  # Process each chunk separately
    chunk.drop_duplicates(inplace=True)
    chunk.rename(columns={'#string_protein_id': 'protein_id'}, inplace=True)

    mapping_dict = map_symbols_to_ensg(chunk['alias'])

    chunk['gene_id'] = chunk['alias'].map(mapping_dict).astype(object)
    chunk.dropna(subset=['gene_id'], inplace=True)  # remove genes without ID Ensembl found

    protein_aliases_df = pd.merge(protein_aliases_df, chunk, on=['protein_id', 'alias','gene_id'], how='outer')

print("Kept " + str(len(protein_aliases_df)) + " rows.")
print("Dropping eventual remaining duplicates...")
protein_aliases_df.drop_duplicates(inplace=True)

# save to file
protein_aliases_df.to_csv(output_filepath, sep="\t", index=False)