import numpy as np
import pandas as pd
import time

pd.set_option('display.max_colwidth', None)

# ISOLATED EXECUTION
def main():
    file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t')
    example_case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb'  #TODO prendi il primo paziente invece di uno specifico

    # GENES ALIASES WITH PROTEINS AND GENE IDS MAPPING
    # file extracted using create_tsv_from_STRING_files.create_gene_aliases_proteins_ids_mapping_file()
    print("Reading protein-aliases-gene file...")
    genes_mapping_df = pd.read_csv('STRING_downloaded_files/9606.protein.aliases.gene.tsv', sep='\t')

    # METHYLATION ILLUMINA MANIFEST FOR CpG-GENE MAPPING
    print("Reading Illumina manifest...")
    # file downloaded from https://support.illumina.com/downloads/infinium_humanmethylation450_product_files.html
    # place .csv file into methylation_manifests/originals_downloaded, then run methylation_manifest_to_tsv.py
    meth_manifest_df = pd.read_csv("methylation_manifests/methylation_manifest450.tsv", sep='\t', dtype=str)

    meth = create_meth_df(example_case_id, file_mapping_df, genes_mapping_df, meth_manifest_df)
    print(meth.head(3))


def create_meth_df(case_id: str, file_mapping_df: pd.DataFrame, genes_mapping_df: pd.DataFrame, meth_manifest_df: pd.DataFrame):
    """
    Computes and returns a pd.Dataframe with CNV filtered data for the specified patient (case_id).
    :param case_id:
    :param file_mapping_df:
    :param genes_mapping_df:
    :param meth_manifest_df:
    :return pd.DataFrame:
    """

    print('Reading methylation data...')

    start_time = time.time()

    df_meth = pd.read_csv(f"files/methylation/{file_mapping_df[
        (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'methylation')
        ]['filename'].to_string(index=False)}", sep='\t', names=['cpg_IlmnID', 'beta_value'])

    df_meth.dropna(inplace=True)
    df_meth.reset_index(inplace=True, drop=True)
    print('--> ' + str(len(df_meth)) + " rows")

    n_rows = len(df_meth)
    df_meth.drop(df_meth[df_meth['cpg_IlmnID'].str.startswith('cg') == False].index, inplace=True)
    print("Removed " + str(n_rows - len(df_meth)) + " rows not strictly about CpG islands.")

    print("Merging with data of methylated gene from Illumina Manifest file...")
    df_meth = pd.merge(df_meth, meth_manifest_df, on='cpg_IlmnID', how='left')

    df_meth.rename(columns={'gene_symbol': 'gene_name'}, inplace=True)

    n_rows = len(df_meth)
    df_meth.dropna(subset=['gene_name'], axis=0, inplace=True)
    df_meth.reset_index(inplace=True, drop=True)
    print("Removed " + str(n_rows - len(df_meth)) + " cpgIDs-names values still missing (gene not tracked).")

    print("Splitting gene name variants and keeping only promoter regions...")
    # UCSC terms identifying Promoters (TSS200 = Proximal promoter, TSS1500 = Distal promoter)
    promoter_terms = ['TSS200', 'TSS1500', '1stExon', "5'UTR"]
    # many 450k probes map on more transcripts: ".str.contains('|'.join(promoter_terms), na=False)" catches also the ones classified as Body for a gene and promoter for another
    is_promoter = df_meth['cpg_region'].str.contains('|'.join(promoter_terms), na=False)
    df_meth = df_meth[is_promoter]

    # have different rows for the possible gene names variants
    values_list = df_meth['gene_name'].str.split(';')
    df_meth['gene_name'] = values_list.apply(lambda x: list(set(x)))  # so to have single name variants as list
    df_meth = df_meth.explode(column=['gene_name']).reset_index(drop=True)

    # nodes data integration
    print("Adding matches from protein.aliases.gene file to find gene Ensembl ids...")
    genes_mapping_df.rename(columns={"alias": "gene_name"}, inplace=True)
    df_meth = pd.merge(df_meth, genes_mapping_df.drop(columns='protein_id'), how='left', on=['gene_name'])
    df_meth.dropna(inplace=True)  # only genes protein coding kept (not present in mapping file from STRING)
    df_meth.drop_duplicates(inplace=True)

    pos_priority_map = {
        'Island': 1.0,
        'N_Shore': 0.6,
        'S_Shore': 0.6,
        'N_Shelf': 0.2,
        'S_Shelf': 0.2
    }

    df_meth['weight'] = df_meth['cpg_position'].map(pos_priority_map)

    df_meth_grouped = df_meth.groupby('gene_id').apply(
        lambda x: np.average(x['beta_value'], weights=x['weight']), include_groups=False
    ).reset_index()
    df_meth_grouped.rename(columns={0: 'weighted_beta_value'}, inplace=True)

    print("Grouped by gene_id (averaging b-values) where more present: " + str(len(df_meth_grouped)) + " final rows")

    print("--- %s seconds ---\n" % (time.time() - start_time))

    return df_meth_grouped


if __name__ == "__main__":
    main()