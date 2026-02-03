import pandas as pd

'''
df_meth27_manifest = pd.read_excel('methylation_manifests/originals/illumina_humanmethylation27_content.xlsx',
                                   usecols=['Name', 'Chr', 'Gene_Strand', 'Symbol'], dtype=str)
df_meth27_manifest.rename(columns={"Name": "cpg_IlmnID", "Chr": "gene_chr", "Gene_Strand":"gene_strand", "Symbol":"gene_symbol"}, inplace=True)
# drop rows not about 'cg...'
df_meth27_manifest.drop(df_meth27_manifest[df_meth27_manifest['cpg_IlmnID'].str.startswith('cg') == False].index, inplace=True)
# drop rows with no cpg_IlmnID or no gene_symbol (not useful)
df_meth27_manifest.dropna(subset=["cpg_IlmnID", "gene_symbol"], inplace=True)
df_meth27_manifest.to_csv(r"methylation_manifests/methylation_manifest27.tsv", sep="\t", index=False)
''' # not useful, too little values

# DE-COMMENT TO EXTRACT TSV

df_meth450_manifest = pd.read_csv('methylation_manifests/originals/humanmethylation450_15017482_v1-2.csv',
                                  comment='#', usecols=['IlmnID', 'CHR', 'Strand', 'UCSC_RefGene_Name'], dtype=str)
df_meth450_manifest.rename(columns={"IlmnID": "cpg_IlmnID", "CHR": "gene_chr", "Strand":"gene_strand", "UCSC_RefGene_Name":"gene_symbol"}, inplace=True)
# drop rows not about 'cg...'
df_meth450_manifest.drop(df_meth450_manifest[df_meth450_manifest['cpg_IlmnID'].str.startswith('cg') == False].index, inplace=True)
# drop rows with no cpg_IlmnID or no gene_symbol (not useful)
df_meth450_manifest.dropna(subset=["cpg_IlmnID", "gene_symbol"], inplace=True)
# remap gene_strand F/R to +/-
mapping = {
    'F': '+',
    'R': '-'
}
df_meth450_manifest['gene_strand'] = df_meth450_manifest['gene_strand'].map(mapping)
df_meth450_manifest.to_csv(r"methylation_manifests/methylation_manifest450.tsv", sep="\t", index=False)


'''
df_methEPICb4_manifest = pd.read_csv('methylation_manifests/originals/MethylationEPIC_v-1-0_B4.csv',
                                     comment='#', usecols=['IlmnID', 'CHR', 'UCSC_RefGene_Name', 'GencodeBasicV12_Accession'], dtype=str)
df_methEPICb4_manifest.rename(columns={"IlmnID": "cpg_IlmnID", "CHR": "gene_chr", "UCSC_RefGene_Name":"gene_symbol", "GencodeBasicV12_Accession":"gene_id"}, inplace=True)
# drop rows not about 'cg...'
df_methEPICb4_manifest.drop(df_methEPICb4_manifest[df_methEPICb4_manifest['cpg_IlmnID'].str.startswith('cg') == False].index,
                            inplace=True)
# drop rows with no cg_id or no gene_symbol (not useful)
df_methEPICb4_manifest.dropna(subset=["cpg_IlmnID", "gene_symbol"], inplace=True)
df_methEPICb4_manifest.to_csv(r"methylation_manifests/methylation_manifestEPICb4.tsv", sep="\t", index=False)


df_methEPICb5_manifest = pd.read_csv('methylation_manifests/originals/infinium-methylationepic-v-1-0-b5-manifest-file.csv',
                                     comment='#', usecols=['IlmnID', 'CHR', 'UCSC_RefGene_Name', 'GencodeBasicV12_Accession'], dtype=str)
df_methEPICb5_manifest.rename(columns={"IlmnID": "cpg_IlmnID", "CHR": "gene_chr", "UCSC_RefGene_Name":"gene_symbol", "GencodeBasicV12_Accession":"gene_id"},
                              inplace=True)
# drop rows not about 'cg...'
df_methEPICb5_manifest.drop(df_methEPICb5_manifest[df_methEPICb5_manifest['cpg_IlmnID'].str.startswith('cg') == False].index, inplace=True)
# drop rows with no cg_id or no gene_symbol (not useful)
df_methEPICb5_manifest.dropna(subset=["cpg_IlmnID", "gene_symbol"], inplace=True)
df_methEPICb5_manifest.to_csv(r"methylation_manifests/methylation_manifestEPICb5.tsv", sep="\t", index=False)
''' # probably not useful, 450k is enough

# EXAMPLE TO SEE VALUES

pd.set_option('display.max_colwidth', None) # or else it can't take whole filename from file_case_mapping.tsv

file_mapping_df = pd.read_csv('files/clinical/file_case_mapping.tsv', sep='\t', dtype=str)

case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example

df_meth = pd.read_csv(f"files/methylation/{file_mapping_df[
    (file_mapping_df['case_id'] == case_id) & (file_mapping_df['omic'] == 'methylation')
    ]['filename'].to_string(index=False)}", sep='\t', names=['cpg_IlmnID', 'beta_value'], dtype=str)

df_match_cpg_gene = pd.read_csv("dataset/matched_cpg_genes_converted.csv", dtype=str)
df_meth = pd.merge(df_meth, df_match_cpg_gene[['cpg_IlmnID', 'gene_chr', 'gene_id', 'gene_symbol']], on='cpg_IlmnID', how='left')
print("Added symbols from matched_cpg_genes_converted.csv")
print("Tot cpgIDs: " + str(len(df_meth)))
print("Correspondences cpgIDs-symbols still missing: " + str(df_meth['gene_symbol'].isna().sum()))

df_meth_manifest450 = pd.read_csv("methylation_manifests/methylation_manifest450.tsv", sep='\t', dtype=str)
df_meth = pd.merge(df_meth, df_meth_manifest450[['cpg_IlmnID', 'gene_symbol']], on='cpg_IlmnID', how='left')
print("Added symbols from methylation_manifest450.tsv")
#print("Correspondences cpgIDs-symbols missing with only 450k: " + str(df_meth['gene_symbol'].isna().sum()))
print("Correspondences cpgIDs-symbols missing with only 450k: " + str(df_meth.iloc[:,-1].isna().sum()))


#inner_joined = pd.merge(df_meth, df_meth_manifest450[['cpg_IlmnID', 'gene_symbol']], on='cpg_IlmnID', how='inner')
# concatenare  quelli presenti in entrambi e trovare un modo per tenere solo i not na tra le due colonne dai due file


# not working:

#df_meth['gene_symbol'] = df_meth['gene_symbol'].fillna(' ')
#symbols = df_meth['gene_symbol'].copy()


#print("Correspondences cpgIDs-symbols still missing: " + str(df_meth.iloc[:,-1].isna().sum()))
#df_meth.iloc[:,-1] = df_meth.iloc[:,-1].fillna(' ')
# now the last 2 cols are symbols from the different files, we need to unify them
#symbols = symbols + ";" + df_meth.iloc[:,-1]
#df_meth.drop(axis=1, columns=df_meth.columns[-2:], inplace=True)
#df_meth['gene_symbol'] = symbols.copy()

#print("Correspondences cpgIDs-symbols still missing: " + str(df_meth['gene_symbol'].isna().sum()))
