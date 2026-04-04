import pandas as pd

def main():
    create_genes_id_mapping_file()
    create_gene_aliases_proteins_ids_mapping_file()  # ⚠ COMMENT AFTER DOING ONCE


def create_genes_id_mapping_file():
    """Creates the tsv file that nodes/genes use to get their associated unique ordered id.
    Output saved in 'files/clinical/gene_ids_mapped.tsv'"""

    # GENES ALIASES WITH PROTEINS AND GENE IDS MAPPING
    # file extracted using string_files_to_tsv.py --> create_protein_links()
    genes_mapping_df = pd.read_csv('STRING_downloaded_files/9606.protein.aliases.gene.tsv',
                                   sep='\t')  # proteins-genes mapping df

    # unique genes_ids mapping to numerical index
    unique_nodes = genes_mapping_df['gene_id'].unique()
    node_map = {node: i for i, node in enumerate(
        unique_nodes)}
    node_map = node_map  # global dictionary for Gene_ID -> Index mapping

    genes_id_mapping_df = pd.DataFrame(node_map.items(), columns=['gene_id', 'gene_id_mapped'])

    # save to file
    genes_id_mapping_df.to_csv('files/clinical/gene_ids_mapped.tsv', sep="\t", index=False)


def create_gene_aliases_proteins_ids_mapping_file():
    """Run to re-create the 9606.protein.aliases.gene txt file with protein_id, gene_name and gene_id (retrieved using
    mygene lib) mapping and excluding data not useful.
    Output will be saved in 'STRING_downloaded_files/9606.protein.aliases.gene.tsv'.
    Takes a lot of time."""

    import mygene

    output_filepath = "STRING_downloaded_files/9606.protein.aliases.gene.tsv"

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
    df_iter = pd.read_csv('STRING_downloaded_files/9606.protein.aliases.v12.0.txt', sep='\t',
                          usecols=['#string_protein_id', 'alias'], chunksize=10000)  # Process 50000 rows at a time

    protein_aliases_df = pd.DataFrame(columns=['protein_id', 'alias', 'gene_id'])

    for chunk in df_iter:  # Process each chunk separately
        chunk.drop_duplicates(inplace=True)
        chunk.rename(columns={'#string_protein_id': 'protein_id'}, inplace=True)

        mapping_dict = map_symbols_to_ensg(chunk['alias'])

        chunk['gene_id'] = chunk['alias'].map(mapping_dict).astype(object)
        chunk.dropna(subset=['gene_id'], inplace=True)  # remove genes without ID Ensembl found

        protein_aliases_df = pd.merge(protein_aliases_df, chunk, on=['protein_id', 'alias', 'gene_id'], how='outer')

    print("Kept " + str(len(protein_aliases_df)) + " rows.")
    print("Dropping eventual remaining duplicates...")
    protein_aliases_df.drop_duplicates(inplace=True)

    # save to file
    protein_aliases_df.to_csv(output_filepath, sep="\t", index=False)

if __name__ == "__main__":
    main()
