import os
import numpy as np
import pandas as pd
import PyWGCNA
import matplotlib.pyplot as plt
import seaborn as sns

RNA_DIR = "files/RNA"
MAPPING_FILE = "files/clinical/file_case_mapping.tsv"
SPLIT_FILE = "files/clinical/patient_split_cleaned.csv"
OUTPUT_DIR = "weight_edges/RNA"

TPM_THRESHOLD = 0.1
MIN_SAMPLE_FRACTION = 0.2
TOP_N_GENES = 2000
EDGE_THRESHOLD = 0.1

sns.set_theme(style="white")

file_mapping_df = pd.read_csv(MAPPING_FILE, sep="\t")
split_df = pd.read_csv(SPLIT_FILE)

train_case_ids = split_df.loc[
    split_df["split"] == "train",
    "cases.case_id"
].astype(str).unique()

rna_mapping = file_mapping_df[
    (file_mapping_df["omic"] == "RNA") &
    (file_mapping_df["case_id"].isin(train_case_ids))
][["case_id", "filename"]].reset_index(drop=True)

print(f"\nTrain RNA samples found: {len(rna_mapping)}")

expression_dfs = []

for _, row in rna_mapping.iterrows():
    case_id = row["case_id"]
    filename = row["filename"].strip()
    path = os.path.join(RNA_DIR, filename)

    print(f"Processing {case_id}")

    df_rna = pd.read_csv(path, sep="\t", dtype=str, comment="#")

    df_rna = df_rna[df_rna["gene_id"].str.startswith("ENSG")]
    df_rna = df_rna[df_rna["gene_type"] == "protein_coding"]

    df_rna["gene_id"] = df_rna["gene_id"].str.split(".", expand=True)[0]
    df_rna["tpm_unstranded"] = df_rna["tpm_unstranded"].astype(float)

    df_rna = (
        df_rna.sort_values("tpm_unstranded", ascending=False)
              .drop_duplicates(subset="gene_name")
    )

    df_rna[case_id] = np.log2(df_rna["tpm_unstranded"] + 1)

    expression_dfs.append(df_rna[["gene_id", case_id]])

expr_matrix = expression_dfs[0]
for df in expression_dfs[1:]:
    expr_matrix = expr_matrix.merge(df, on="gene_id", how="outer")

expr_matrix.set_index("gene_id", inplace=True)
expr_matrix = expr_matrix.astype(float).fillna(0)

log_thresh = np.log2(TPM_THRESHOLD + 1)
min_samples = int(np.ceil(MIN_SAMPLE_FRACTION * expr_matrix.shape[1]))

expr_matrix = expr_matrix[
    (expr_matrix > log_thresh).sum(axis=1) >= min_samples
]

expr_matrix = expr_matrix.loc[expr_matrix.var(axis=1) > 0]

print("\nFinal expression matrix shape:", expr_matrix.shape)

expr_for_wgcna = expr_matrix.T  # samples x genes

wgcna = PyWGCNA.WGCNA(
    name="RNA_WGCNA_train",
    geneExp=expr_for_wgcna
)

wgcna.runWGCNA()

gene2module = pd.Series(
    wgcna.datExpr.var["moduleColors"].values,
    index=wgcna.datExpr.var.index,
    name="module"
)

print("\nNumber of genes per module:")
print(gene2module.value_counts())

gene2module.to_csv(os.path.join(OUTPUT_DIR,"WGCNA_gene_to_module.tsv"), sep='\t')

MEs = wgcna.MEs

sns.clustermap(
    MEs,
    cmap="vlag",
    center=0,
    standard_scale=1,
    figsize=(10, 8)
)

plt.title("Module Eigengenes Heatmap")
plt.savefig(os.path.join(OUTPUT_DIR, "WGCNA_ModuleEigengenes_heatmap.png"), dpi=300)
plt.show()

MEs.to_csv(os.path.join(OUTPUT_DIR, "WGCNA_module_eigengenes.tsv"), sep='\t')

TOM = pd.DataFrame(
    wgcna.TOM,
    index=wgcna.datExpr.var.index,
    columns=wgcna.datExpr.var.index
)

connectivity = TOM.sum(axis=1)

connectivity_df = pd.DataFrame({
    "gene": connectivity.index,
    "connectivity": connectivity.values,
    "module": gene2module.loc[connectivity.index].values
}).sort_values(by="connectivity", ascending=False)

print("\nTop 10 hub genes:")
print(connectivity_df.head(10))

connectivity_df.to_csv(os.path.join(OUTPUT_DIR, "WGCNA_gene_connectivity.tsv"), sep='\t', index=False)

top_genes = connectivity_df.head(200)["gene"].values
expr_top = expr_matrix.loc[top_genes]

sns.clustermap(
    expr_top,
    cmap="vlag",
    z_score=0,
    figsize=(12, 12)
)

plt.title("Top 200 Hub Genes")
plt.savefig(os.path.join(OUTPUT_DIR, "Top_hub_genes_heatmap.png"), dpi=300)
plt.show()

selected_genes = connectivity_df.head(TOP_N_GENES)["gene"].values

pd.Series(selected_genes).to_csv(
    os.path.join(OUTPUT_DIR, "WGCNA_selected_genes_for_STRING.tsv"),
    sep='\t',
    index=False
)

print(f"\nSelected backbone genes: {len(selected_genes)}")

edges = []

for i, g1 in enumerate(selected_genes):
    for g2 in selected_genes[i+1:]:
        weight = TOM.loc[g1, g2]
        if weight > EDGE_THRESHOLD:
            edges.append((g1, g2, weight))

edges_df = pd.DataFrame(edges, columns=["gene1", "gene2", "weight"])

print("Edges over threshold:", len(edges_df))

edges_df.to_csv(os.path.join(OUTPUT_DIR, "WGCNA_edges_for_STRING_intersection.tsv"), sep='\t', index=False)

plt.figure(figsize=(8,5))
plt.hist(edges_df["weight"], bins=50)
plt.title("Distribution TOM weights")
plt.xlabel("TOM weight")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "WGCNA_edge_weight_distribution.png"), dpi=300)
plt.show()

print("\nPipeline weight_edges complete.")
