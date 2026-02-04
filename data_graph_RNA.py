import pandas as pd
import stringdb

pd.set_option('display.max_colwidth', None)

case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example
tpm_unstranded_threshold = 0.1  # gene expression threshold to ignore low-expression genes
ppi_score_threshold = 0.5  # minimum interaction probability score to create edges

file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')

print('Reading RNA reads data...')

df_rna = pd.read_csv(f"files/RNA/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'RNA')
    ]['filename'].to_string(index=False)}", sep='\t', dtype=str, comment='#')  # 'comment=' to ignore the first line in RNA files

print("Dropping values not useful...")
print(str(len(df_rna)) + " rows")
df_rna.drop(df_rna[df_rna['gene_id'].str.startswith('ENSG') == False].index, inplace=True) # drop metadata

print("Removing non protein-coding genes...")
df_rna.drop(df_rna[df_rna['gene_type'] != "protein_coding"].index, inplace=True)
print("--> " + str(len(df_rna)) + " rows\n")

print("Removing genes with expression (tpm_unstranded) under " + str(tpm_unstranded_threshold) + "...")
df_rna.drop(df_rna[df_rna['tpm_unstranded'].astype(float) < tpm_unstranded_threshold].index, inplace=True)
df_rna.reset_index(inplace=True, drop=True)
print("--> " + str(len(df_rna)) + " rows\n")

# remove ids (Ensembl) version (ENSG00000000003.15 --> ENSG00000000003)
df_rna['gene_id'] = df_rna.gene_id.str.split('.', expand=True)[0]
print("RNA df created:")
print(df_rna.head(3))

#genes = ['TP53', 'BRCA1', 'FANCD1', 'FANCL']
genes = list(df_rna['gene_name'].astype(str))

# STRING REQ. SUPPORTS 2000 NODES AT MOST, CANNOT USE THIS METHOD
# network = stringdb.get_network(genes) # ppi
# columns = ['stringId1', 'stringId2', 'preferredName_A', 'preferredName_B', 'score']
''' ErrorMessage input too large. STRING website does not support networks larger than 2000 nodes. '''

print("Uploading protein-links file...")
# file downloaded from https://string-db.org/cgi/download.pl into dataset/
protein_links_df = pd.read_csv('dataset/9606.protein.links.v12.0.txt', sep=' ')

# to refactor the score in a [0-1] interval, like returned by stringdb.get_network()
protein_links_df['combined_score'] = protein_links_df['combined_score']/1000

print("Dropping interactions with combined probability score lower than " + str(ppi_score_threshold) + "...")
# filter based on score (probability of interacting)
protein_links_df.drop(protein_links_df[protein_links_df['combined_score'] < ppi_score_threshold].index, inplace=True)
protein_links_df.reset_index(inplace=True, drop=True)

print("Retrieving genes Ensembl ids from STRING...")
string_ids = stringdb.get_string_ids(genes)

print("Retrieving protein-protein interactions from file uploaded...")
# interactions are retrieved in both ways
network_df = protein_links_df[(protein_links_df['protein1'].isin(string_ids['stringId'])) &
                            (protein_links_df['protein2'].isin(string_ids['stringId']))]
network_df.reset_index(inplace=True, drop=True)
print("Found " + str(len(network_df)/2) + " interactions.")

print("DONE")

# http manual requests (instead of stringdb.get_network()), still, input too large to use it
'''
import requests ## python -m pip install requests

string_api_url = "https://version-12-0.string-db.org/api"
output_format = "tsv"
method = "network"
request_url = "/".join([string_api_url, output_format, method])

my_genes = ["CDC42","CDK1","KIF23","PLK1", "RAC2","RACGAP1","RHOA","RHOB"]

params = {
    "identifiers" : "%0d".join(my_genes), # your protein
    "species" : 9606, # NCBI/STRING taxon identifier
    "caller_identity" : "BioInfoProject" # your app name
}

response = requests.post(request_url, data=params)
df_response = pd.read_csv(io.StringIO(response.text), sep="\t")
df_network = df_response[df_response['escore'] > 0.4]
print(df_network[['preferredName_A', 'preferredName_B', 'escore']].head())
'''

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

# Motivazioni per ppi_score_threshold
'''
Soglie di confidenza per gli score; STRING suggerisce solitamente tre "cut-off" standard per filtrare i risultati:

| Low Confidence | > 0.150 | > 150 |
| Medium Confidence | > 0.400 | > 400 |
| High Confidence | > 0.700 | > 700 |
| Highest Confidence | > 0.900 | > 900 |

`0` in colonne come `nscore` (neighborhood score) o `fscore` (fusion score), succede perché:
1. Per quelle specifiche coppie di proteine, non ci sono prove derivanti da quel particolare canale.
2. Spesso le API restituiscono lo score totale (`score`) e i singoli sub-score solo se esplicitamente richiesti o se
superano una certa soglia interna.
'''
