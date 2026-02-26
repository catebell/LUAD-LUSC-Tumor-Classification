import os
import glob
import numpy as np
import pandas as pd

from multiomics_graph_creation import ppi_score_threshold

# ===============================
# CONFIGURAZIONE (meno restrittiva)
# ===============================

METHYLATION_DIR = "files/methylation"
OUTPUT_DIR = "weight_edges/methylation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MANIFEST_FILE = "methylation_manifests/methylation_manifest450.tsv"
STRING_EDGES_FILE = "downloaded_files/9606.protein.links.v12.0.txt"

MIN_PRESENCE = 0.6      # prima 0.8 → più inclusivo
MIN_VARIANCE = 0        # rimosso filtro varianza

# ===============================
# 1️⃣ LOAD METHYLATION FILES
# ===============================

print("\nLoading methylation files...")

all_files = glob.glob(os.path.join(METHYLATION_DIR, "*.txt"))
if len(all_files) == 0:
    raise ValueError("No methylation .txt files found")

dfs = []
for file in all_files:
    sample = os.path.basename(file).replace(".txt", "")
    df = pd.read_csv(file, sep="\t", header=None, names=["CpG", sample])
    dfs.append(df)

beta_matrix = dfs[0]
for df in dfs[1:]:
    beta_matrix = beta_matrix.merge(df, on="CpG", how="outer")

beta_matrix = beta_matrix.set_index("CpG")
print("Initial CpG × sample shape:", beta_matrix.shape)

# ===============================
# 2️⃣ CpG → gene_symbol
# ===============================

manifest = pd.read_csv(MANIFEST_FILE, sep="\t", dtype=str, encoding="utf-8-sig")
manifest.columns = manifest.columns.str.strip()

manifest = manifest[["cpg_IlmnID", "gene_symbol"]].dropna()

manifest["gene_symbol"] = manifest["gene_symbol"].str.split(";")
manifest = manifest.explode("gene_symbol")
manifest["gene_symbol"] = manifest["gene_symbol"].str.strip()

beta_gene = beta_matrix.reset_index().merge(
    manifest,
    left_on="CpG",
    right_on="cpg_IlmnID",
    how="inner"
)

# media CpG → gene
gene_matrix = (
    beta_gene
    .drop(columns=["CpG", "cpg_IlmnID"])
    .groupby("gene_symbol")
    .mean()
)

print("Gene_symbol × sample shape:", gene_matrix.shape)

# ===============================
# 3️⃣ gene_symbol → ENSG
# ===============================

# GENES ALIASES WITH PROTEINS AND GENE IDS MAPPING
# file extracted using genes_proteins_aliases_ensg_mapping.py
print("Reading protein-aliases-gene file...")
genes_mapping_df = pd.read_csv('downloaded_files/9606.protein.aliases.gene.tsv', sep='\t', dtype=str)
genes_mapping_df.rename(columns={"alias": "gene_name"}, inplace=True)

gene_matrix.index = gene_matrix.index.str.strip()

gene_matrix = gene_matrix.reset_index().merge(
    genes_mapping_df,
    left_on="gene_symbol",
    right_on="alias",
    how="inner"
)

gene_matrix = gene_matrix.drop(columns=["gene_symbol", "alias"])
gene_matrix = gene_matrix.groupby("gene_id").mean()

print("Final ENSG × sample shape:", gene_matrix.shape)

# ===============================
# 4️⃣ FILTRO PRESENZA (più permissivo)
# ===============================

presence = gene_matrix.notna().mean(axis=1)
gene_matrix = gene_matrix.loc[presence >= MIN_PRESENCE]

print("After relaxed filtering:", gene_matrix.shape)

# Riempimento NA con media del gene (aumenta copertura)
gene_matrix = gene_matrix.apply(
    lambda row: row.fillna(row.mean()),
    axis=1
)

# ===============================
# 5️⃣ LOAD STRING BACKBONE
# ===============================

print("\nLoading STRING backbone...")

string_edges_df = pd.read_csv(STRING_EDGES_FILE, sep="\s+", dtype=str)
string_edges_df.drop(string_edges_df[string_edges_df['combined_score'] < ppi_score_threshold].index, inplace=True)

protein2gene = pd.read_csv('downloaded_files/9606.protein.aliases.gene.tsv', sep="\t", dtype=str)
protein2gene = protein2gene[["protein_id", "gene_id"]].dropna()
protein2gene = protein2gene[protein2gene["gene_id"].str.startswith("ENSG")]

# protein1 → gene1
string_edges_df = string_edges_df.merge(
    protein2gene,
    left_on="protein1",
    right_on="protein_id",
    how="left"
).rename(columns={"gene_id": "gene1"}).drop(columns="protein_id")

# protein2 → gene2
string_edges_df = string_edges_df.merge(
    protein2gene,
    left_on="protein2",
    right_on="protein_id",
    how="left"
).rename(columns={"gene_id": "gene2"}).drop(columns="protein_id")

string_edges_df = string_edges_df.dropna(subset=["gene1", "gene2"])

# rendi non direzionale
string_edges_df[["gene1","gene2"]] = pd.DataFrame(
    string_edges_df[["gene1","gene2"]].apply(lambda x: sorted(x), axis=1).tolist(),
    index=string_edges_df.index
)

string_edges_df = string_edges_df.drop_duplicates(subset=["gene1","gene2"])

selected_genes = pd.unique(string_edges_df[["gene1","gene2"]].values.ravel())
print(f"STRING backbone genes: {len(selected_genes)}")

# ===============================
# 6️⃣ Subset methylation genes
# ===============================

gene_matrix_sub = gene_matrix.loc[
    gene_matrix.index.isin(selected_genes)
]

print(f"Methylation genes overlapping STRING: {gene_matrix_sub.shape[0]}")

# ===============================
# 7️⃣ Spearman correlation
# ===============================

print("\nComputing Spearman correlation matrix...")
corr_matrix = gene_matrix_sub.T.corr(method="spearman")

# ===============================
# 8️⃣ Estrai coppie STRING
# ===============================

edges_df = string_edges_df[
    string_edges_df["gene1"].isin(corr_matrix.index) &
    string_edges_df["gene2"].isin(corr_matrix.index)
].copy()

edges_df["weight"] = [
    corr_matrix.loc[g1, g2]
    for g1, g2 in zip(edges_df["gene1"], edges_df["gene2"])
]

edges_df = edges_df[["gene1","gene2","weight"]]

edges_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "methylation_edges_spearman2_for_STRING.tsv"
    ),
    sep="\t",
    index=False
)

print(f"Edges extracted for STRING backbone: {len(edges_df)}")

print("\nMethylation backbone-weighting complete (max gene coverage mode).")
