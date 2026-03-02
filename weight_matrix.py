import pandas as pd

# --------------------------------------------------
# 1. Carica i file
# --------------------------------------------------
protein_links = pd.read_csv(
    "downloaded_files/9606.protein.links.v12.0.txt",
    sep=" ", header=0
)

protein_aliases = pd.read_csv(
    "downloaded_files/9606.protein.aliases.gene.tsv",
    sep="\t", header=0
)

cnv_edges = pd.read_csv(
    "edge_weights/CNV_edges_spearman_for_STRING.tsv",
    sep="\t", header=0
)

methylation_edges = pd.read_csv(
    "edge_weights/methylation_edges_spearman2_for_STRING.tsv",
    sep="\t", header=0
)

print("Protein links:", protein_links.shape)
print("CNV Spearman:", cnv_edges.shape)
print("Methylation:", methylation_edges.shape)

# --------------------------------------------------
# 2. Mappa protein_id → gene_id
# --------------------------------------------------

# Per protein1
protein_links = protein_links.merge(
    protein_aliases[['protein_id', 'gene_id']],
    left_on='protein1',
    right_on='protein_id',
    how='left'
).rename(columns={'gene_id': 'gene1'}).drop(columns=['protein_id'])

# Per protein2
protein_links = protein_links.merge(
    protein_aliases[['protein_id', 'gene_id']],
    left_on='protein2',
    right_on='protein_id',
    how='left'
).rename(columns={'gene_id': 'gene2'}).drop(columns=['protein_id'])

# Rimuovi eventuali righe con mapping mancante
protein_links = protein_links.dropna(subset=['gene1', 'gene2'])

# --------------------------------------------------
# 3. Normalizza STRING combined_score
# --------------------------------------------------
protein_links['protein_links_weight'] = (
    protein_links['combined_score'] / 1000
)

protein_links = protein_links[
    ['gene1', 'gene2', 'protein_links_weight']
]

# --------------------------------------------------
# 4. Aggrega per coppia di geni
# --------------------------------------------------
protein_links_agg = (
    protein_links
    .groupby(['gene1', 'gene2'], as_index=False)
    .mean()
)

# --------------------------------------------------
# 5. Rendi tutte le reti NON direzionali
# --------------------------------------------------
def make_undirected(df):
    df[['gene1', 'gene2']] = pd.DataFrame(
        df[['gene1', 'gene2']].apply(lambda x: sorted(x), axis=1).tolist(),
        index=df.index
    )
    return df

protein_links_agg = make_undirected(protein_links_agg)
cnv_edges = make_undirected(cnv_edges)
methylation_edges = make_undirected(methylation_edges)

# --------------------------------------------------
# 6. Rinominare colonne peso
# --------------------------------------------------
cnv_edges = cnv_edges.rename(columns={'weight': 'cnv_weight'})
methylation_edges = methylation_edges.rename(columns={'weight': 'methylation_weight'})

# --------------------------------------------------
# 7. Merge (STRING come base)
# --------------------------------------------------
merged = pd.merge(
    protein_links_agg,
    cnv_edges,
    on=['gene1', 'gene2'],
    how='left'
)

merged = pd.merge(
    merged,
    methylation_edges,
    on=['gene1', 'gene2'],
    how='left'
)

# --------------------------------------------------
# 8. Statistiche
# --------------------------------------------------
num_protein = merged['protein_links_weight'].notna().sum()
num_cnv = merged['cnv_weight'].notna().sum()
num_methyl = merged['methylation_weight'].notna().sum()

num_protein_cnv = (
    merged['protein_links_weight'].notna() &
    merged['cnv_weight'].notna()
).sum()

num_all_three = (
    merged['protein_links_weight'].notna() &
    merged['cnv_weight'].notna() &
    merged['methylation_weight'].notna()
).sum()

print("\n--- STATISTICHE ---")
print(f"Righe con STRING: {num_protein}")
print(f"Righe con CNV: {num_cnv}")
print(f"Righe con Methylation: {num_methyl}")
print(f"STRING + CNV: {num_protein_cnv}")
print(f"Tutti e tre i layer: {num_all_three}")

# --------------------------------------------------
# 9. Esempi con tutti e tre presenti
# --------------------------------------------------
all_present = merged[
    merged['protein_links_weight'].notna() &
    merged['cnv_weight'].notna() &
    merged['methylation_weight'].notna()
]

print("\nEsempi con tutti e tre i pesi:")
print(all_present.head(10))

# --------------------------------------------------
# 10. Salva file finale
# --------------------------------------------------
merged.to_csv(
    "merged_gene_matrix.tsv",
    sep="\t",
    index=False,
    float_format="%.3f"
)

print("\nFile merged_gene_matrix.tsv creato correttamente!")
