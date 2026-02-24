import time

import numpy as np
import pandas as pd

pd.set_option('display.max_colwidth', None)

# ISOLATED EXECUTION
def main():
    file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')
    example_case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb'

    # GENES ALIASES WITH PROTEINS AND GENE IDS MAPPING
    # file extracted using genes_proteins_aliases_ensg_mapping.py
    print("Reading protein-aliases-gene file...")
    genes_mapping_df = pd.read_csv('downloaded_files/9606.protein.aliases.gene.tsv', sep='\t')

    create_cnv_df(example_case_id, file_mapping_df, genes_mapping_df)


def create_cnv_df(case_id: str, file_mapping_df: pd.DataFrame, genes_mapping_df: pd.DataFrame):
    """
    Computes and returns a pd.Dataframe with CNV filtered data for the specified patient (case_id).
    :param case_id:
    :param file_mapping_df:
    :param genes_mapping_df:
    :return pd.DataFrame:
    """

    print('Reading CNV data...')

    start_time = time.time()

    df_cnv = pd.read_csv(f"files/CNV/{file_mapping_df[
        (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'CNV')
        ]['filename'].to_string(index=False)}", sep='\t', usecols=['gene_id','gene_name','copy_number',
                                                                   'min_copy_number','max_copy_number'])

    df_cnv.dropna(inplace=True)
    df_cnv.reset_index(inplace=True, drop=True)
    print('--> ' + str(len(df_cnv)) + " rows")

    # remove ids (Ensembl) version (ENSG00000000003.15 --> ENSG00000000003)
    df_cnv['gene_id'] = df_cnv.gene_id.str.split('.', expand=True)[0]
    # remove genes (names) version (AL627309.1 --> AL627309)
    df_cnv['gene_name'] = df_cnv.gene_name.str.split('.', expand=True)[0]

    # the bigger the diff, the higher the region instability
    df_cnv['cnv_min_max_diff'] = df_cnv['max_copy_number'] - df_cnv['min_copy_number']
    df_cnv.drop(columns=['min_copy_number', 'max_copy_number'], inplace=True)

    print("CNV df created, like:")
    print(str(df_cnv.head(1)))

    # nodes data integration
    print("Adding matches from protein.aliases.gene file to find gene Ensembl ids...")

    genes_mapping_df.rename(columns={"alias": "gene_name"}, inplace=True)
    df_cnv = pd.merge(df_cnv, genes_mapping_df.drop(columns='protein_id'), how='left', on=['gene_name'])
    df_cnv.dropna(inplace=True)  # only genes protein coding kept (not present in mapping file from STRING)
    df_cnv.drop_duplicates(inplace=True)
    # if there are discrepancies, keep gene_id from file for correct mapping
    df_cnv['gene_id'] = np.where(df_cnv['gene_id_x'] == df_cnv['gene_id_y'], df_cnv['gene_id_y'], df_cnv['gene_id_y'])
    df_cnv.drop(columns=['gene_id_x', 'gene_id_y'], inplace=True)
    df_cnv.reset_index(drop=True, inplace=True)

    print("Grouping by gene_id (mean) if more present...")
    df_cnv_grouped = df_cnv.groupby('gene_id').agg({
        'copy_number': 'mean',
        'cnv_min_max_diff': 'mean'
    }).reset_index()
    print("--> " + str(len(df_cnv)) + " actual rows")

    print("\n--- %s seconds ---\n" % (time.time() - start_time))

    return df_cnv_grouped


if __name__ == "__main__":
    main()