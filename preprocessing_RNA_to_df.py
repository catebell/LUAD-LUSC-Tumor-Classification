import pandas as pd
import time

pd.set_option('display.max_colwidth', None)

# ISOLATED EXECUTION
def main():
    file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')

    example_case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb'  # for example
    ppi_score_threshold = 0.7  # minimum interaction probability score to create edges

    # GENES ALIASES WITH PROTEINS AND GENE IDS MAPPING
    # file extracted using create_tsv_from_STRING_files.create_gene_aliases_proteins_ids_mapping_file()
    print("Reading protein-aliases-gene file...")
    genes_mapping_df = pd.read_csv('STRING_downloaded_files/9606.protein.aliases.gene.tsv', sep='\t')

    # PROTEINS LINKS
    # file downloaded from https://string-db.org/cgi/download.pl selecting organism = Homo sapiens
    # --> 9606.protein.links file under INTERACTION DATA, place the .txt extracted into original_dataset/
    print("Reading protein-links file...")
    protein_links_df = pd.read_csv('STRING_downloaded_files/9606.protein.links.v12.0.txt', sep=' ')

    # refactor the score in a [0-1] interval, like returned by stringdb.get_network()
    protein_links_df['combined_score'] = protein_links_df['combined_score'] / 1000

    print("Dropping interactions with combined probability score lower than " + str(ppi_score_threshold) + "...")
    # filter based on score (probability of interacting)
    protein_links_df.drop(protein_links_df[protein_links_df['combined_score'] < ppi_score_threshold].index,
                          inplace=True)
    protein_links_df.reset_index(inplace=True, drop=True)


    rna, network = create_rna_df(example_case_id, file_mapping_df, genes_mapping_df, protein_links_df)
    print(rna.head(3))
    print(network.head(3))


def create_rna_df(case_id: str, file_mapping_df: pd.DataFrame, genes_mapping_df: pd.DataFrame, protein_links_df: pd.DataFrame):
    """
    Computes and returns a pd.Dataframe with RNA filtered data and a pd.Dataframe of protein-protein interactions for
    the specified patient (case_id) using only protein-coding genes.
    :param case_id:
    :param file_mapping_df:
    :param genes_mapping_df:
    :param protein_links_df:
    :return torch_geometric.data.Data:
    :return pd.DataFrame:
    """

    print('Reading RNA genes-reads data...')

    start_time = time.time()

    df_rna = pd.read_csv(f"files/RNA/{file_mapping_df[
        (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'RNA')
        ]['filename'].to_string(index=False)}", sep='\t', dtype=str, usecols=['gene_id','gene_name','gene_type',
                                                                              'tpm_unstranded'], comment='#')  # 'comment=' to ignore the first line in RNA files

    df_rna.drop(df_rna[df_rna['gene_id'].str.startswith('ENSG') == False].index, inplace=True)  # drop metadata
    df_rna.dropna(inplace=True)
    df_rna.reset_index(inplace=True, drop=True)
    print("--> " + str(len(df_rna)) + " rows")

    n_rows = len(df_rna)
    df_rna.drop(df_rna[df_rna['gene_type'] != "protein_coding"].index, inplace=True)
    df_rna.drop('gene_type', axis=1, inplace=True)  # not useful anymore
    print("Removed " + str(n_rows - len(df_rna)) + " non protein-coding genes.")

    n_rows = len(df_rna)

    # remove gene_ids (Ensembl) version (ENSG00000000003.15 --> ENSG00000000003)
    df_rna['gene_id'] = df_rna.gene_id.str.split('.', expand=True)[0]
    # remove genes (names) version (AL627309.1 --> AL627309)
    df_rna['gene_name'] = df_rna.gene_name.str.split('.', expand=True)[0]

    # Retrieves only preferred protein_ids mapped to preferred STRING gene_name (we lose eventual aliases info)
    # Might be fine for RNA data, but for other omics a lot of genes get grouped as one, and eventual multiple proteins
    # are not retrieved by the string function alone. Moreover, it's a bottleneck operation.
    '''
    import stringdb

    #genes = ['TP53', 'BRCA1', 'FANCD1', 'FANCL']  # example
    genes = list(df_rna['gene_name'].astype(str))

    # STRING REQ. SUPPORTS 2000 NODES AT MOST, CANNOT USE THIS METHOD
    # network = stringdb.get_network(genes) # ppi
    # columns = ['stringId1', 'stringId2', 'preferredName_A', 'preferredName_B', 'score']
    # --> ErrorMessage input too large. STRING website does not support networks larger than 2000 nodes.

    print("\nRetrieving gene proteins ids from STRING...")
    string_ids = stringdb.get_string_ids(genes)

    # drop genes not found in string db
    df_rna.drop(df_rna[df_rna['gene_name'].isin(string_ids['queryItem']) == False].index, inplace=True)
    df_rna.reset_index(inplace=True, drop=True)

    # nodes data integration
    df_rna['protein_id'] = string_ids[string_ids['queryItem'] == df_rna['gene_name']]['stringId']
    df_rna['preferredName'] = string_ids[string_ids['queryItem'] == df_rna['gene_name']]['preferredName']
    '''

    # nodes data integration
    print("Adding matches from protein.aliases.gene file to find all protein isoforms coded per gene...")
    genes_mapping_df.rename(columns={"alias": "gene_name"}, inplace=True)
    # add all protein_ids associated to a gene as multiple rows
    df_rna = pd.merge(df_rna, genes_mapping_df, how='left', on=['gene_name'])
    df_rna.dropna(inplace=True)
    df_rna = df_rna.rename(columns={'gene_id_y': 'gene_id'}).drop(columns='gene_id_x')
    df_rna.reset_index(drop=True, inplace=True)

    print("Retrieving protein-protein interactions from file protein.links...")
    # retrieve only interactions between genes both present in df_rna. Both ways interactions (p1-->p2 and p2-->p1)
    network_df = protein_links_df[(protein_links_df['protein1'].isin(df_rna['protein_id'])) &
                                  (protein_links_df['protein2'].isin(df_rna['protein_id']))].copy()
    network_df.reset_index(inplace=True, drop=True)

    # map {protein_id --> gene_id}
    prot_to_gene = dict(zip(df_rna['protein_id'], df_rna['gene_id']))

    network_df['gene1'] = network_df['protein1'].map(prot_to_gene)
    network_df['gene2'] = network_df['protein2'].map(prot_to_gene)
    network_df = network_df.dropna(subset=['gene1', 'gene2'])

    print(str(len(df_rna)) + " final rows and found " + str(len(network_df)/2) + " protein isoforms interactions.")

    print("--- %s seconds ---\n" % (time.time() - start_time))

    return df_rna, network_df


if __name__ == "__main__":
    main()
