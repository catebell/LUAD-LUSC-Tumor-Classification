import numpy as np
import pandas as pd
import stringdb
import time

pd.set_option('display.max_colwidth', None)

tpm_unstranded_threshold = 1  # gene expression threshold to ignore lower-expression genes

# ISOLATED EXECUTION
def main():
    file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')

    example_case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb'  # for example
    ppi_score_threshold = 0.7  # minimum interaction probability score to create edges

    # GENES ALIASES WITH PROTEINS AND GENE IDS MAPPING
    # file extracted using genes_proteins_aliases_ensg_mapping.py
    print("Reading protein-aliases-gene file...")
    genes_mapping_df = pd.read_csv('downloaded_files/9606.protein.aliases.gene.tsv', sep='\t')

    # PROTEINS LINKS
    # file downloaded from https://string-db.org/cgi/download.pl selecting organism = Homo sapiens
    # --> 9606.protein.links file under INTERACTION DATA, place the .txt extracted into original_dataset/
    print("Reading protein-links file...")
    protein_links_df = pd.read_csv('downloaded_files/9606.protein.links.v12.0.txt', sep=' ')

    # refactor the score in a [0-1] interval, like returned by stringdb.get_network()
    protein_links_df['combined_score'] = protein_links_df['combined_score'] / 1000

    print("Dropping interactions with combined probability score lower than " + str(ppi_score_threshold) + "...")
    # filter based on score (probability of interacting)
    protein_links_df.drop(protein_links_df[protein_links_df['combined_score'] < ppi_score_threshold].index,
                          inplace=True)
    protein_links_df.reset_index(inplace=True, drop=True)


    create_rna_df(example_case_id, file_mapping_df, genes_mapping_df, protein_links_df)


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
    print(str(len(df_rna)) + " rows\n")

    print("Removing non protein-coding genes...")
    df_rna.drop(df_rna[df_rna['gene_type'] != "protein_coding"].index, inplace=True)
    df_rna.drop('gene_type', axis=1, inplace=True)  # not useful anymore
    print("--> " + str(len(df_rna)) + " actual rows\n")

    print("Removing genes with expression (tpm_unstranded) under " + str(tpm_unstranded_threshold) + "...")
    df_rna.drop(df_rna[df_rna['tpm_unstranded'].astype(float) < tpm_unstranded_threshold].index, inplace=True)
    df_rna.reset_index(inplace=True, drop=True)
    print("--> " + str(len(df_rna)) + " actual rows\n")

    # remove gene_ids (Ensembl) version (ENSG00000000003.15 --> ENSG00000000003)
    df_rna['gene_id'] = df_rna.gene_id.str.split('.', expand=True)[0]
    # remove genes (names) version (AL627309.1 --> AL627309)
    df_rna['gene_name'] = df_rna.gene_name.str.split('.', expand=True)[0]
    print("\nRNA df created, like:")
    print(str(df_rna.head(1)))

    # Retrieves only preferred protein_ids mapped to preferred STRING gene_name (we lose eventual aliases info)
    # Might be fine for RNA data, but for other omics a lot of genes get grouped as one, and eventual multiple proteins
    # are not retrieved by the string function alone. Moreover, it's a bottleneck operation.
    '''
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
    print("\nAdding matches from protein.aliases.gene file to find all coded proteins per gene...")
    genes_mapping_df.rename(columns={"alias": "gene_name"}, inplace=True)
    # add all protein_ids associated to a gene as multiple rows
    df_rna = pd.merge(df_rna, genes_mapping_df, how='left', on=['gene_name'])
    df_rna.dropna(inplace=True)
    df_rna['gene_id'] = np.where(df_rna['gene_id_x'] == df_rna['gene_id_y'], df_rna['gene_id_y'], df_rna['gene_id_y'])
    df_rna.drop(columns=['gene_id_x', 'gene_id_y'], inplace=True)
    df_rna.reset_index(drop=True, inplace=True)
    print("--> " + str(len(df_rna)) + " actual rows\n")

    print("\nRetrieving protein-protein interactions from file protein.links...")
    # retrieve only interactions between genes both present in df_rna. Both ways interactions (p1-->p2 and p2-->p1)
    network_df = protein_links_df[(protein_links_df['protein1'].isin(df_rna['protein_id'])) &
                                  (protein_links_df['protein2'].isin(df_rna['protein_id']))].copy()
    network_df.reset_index(inplace=True, drop=True)

    # map {protein_id --> gene_id}
    prot_to_gene = dict(zip(df_rna['protein_id'], df_rna['gene_id']))

    network_df['gene1'] = network_df['protein1'].map(prot_to_gene)
    network_df['gene2'] = network_df['protein2'].map(prot_to_gene)
    network_df = network_df.dropna(subset=['gene1', 'gene2'])

    print("Found " + str(len(network_df) / 2) + " bidirectional interactions.\n")

    print("\n--- %s seconds ---\n" % (time.time() - start_time))

    return df_rna, network_df


if __name__ == "__main__":
    main()


# INFO UTILI

# Motivzioni per tpm_unstranded_threshold:
'''
Rilevanza Biologica: Un'interazione proteina-proteina può avvenire solo se entrambe le proteine sono presenti.
Se il trascritto è a 0, la proteina non verrà prodotta, quindi l'interazione è fisicamente impossibile in quel contesto.

Riduzione dei Falsi Positivi: STRING database contiene milioni di interazioni potenziali.
Filtrando i geni non espressi, la rete risultante sarà specifica per il campione.

Efficienza delle API: STRING ha dei limiti sul numero di geni che puoi inviare in una singola richiesta.
Rimuovendo i geni "spenti", si risparmia spazio per quelli più interessanti.

Spesso in bioinformatica non ci si limita a togliere lo zero assoluto, ma si usa una soglia minima di TPM
(es. tpm_unstranded > 0.5 o > 1), perché valori molto bassi potrebbero essere solo rumore di fondo del sequenziamento.
'''

# Conversione combined_score/1000:
'''
stringdb.get_network() ritorna score[0.34, 0.68, etc] mentre sul file si trova combined_score[145, 456, etc].
La differenza tra i due è puramente formale (una questione di scala), ma il significato statistico è identico.
Il `combined_score` nei file scaricati (es. 173, 471) è espresso in una scala che va da 0 a 1000.
Lo `score` restituito dalle API o dalle librerie Python è espresso in una scala da 0 a 1.

--> combined_score=471 nel file scaricato corrisponde esattamente a score=0.471 nella risposta della funzione Python.

In entrambi i casi, il valore rappresenta la confidenza probabilistica che l'interazione sia reale,
basata sull'integrazione di diverse prove (esperimenti, co-espressione, database, text-mining, ecc.).
STRING non usa una semplice somma delle prove, ma un modello probabilistico. Con più canali di prova
(es. "escore" per esperimenti, "nscore" per vicinanza genica), il punteggio viene calcolato come una loro combinazione
normalizzando gli score di ogni singolo canale.
'''
