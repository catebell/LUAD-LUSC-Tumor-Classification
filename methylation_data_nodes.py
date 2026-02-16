import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurazioni ---
METH_DIR = "files/methylation"
OUTPUT_DIR = "WGCNA/methylation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAPPING_FILE = "files/clinical/file_case_mapping.tsv"
SPLIT_FILE = "files/clinical/patient_split_cleaned.csv"
CPG_GENE_MAP_FILE = "dataset/matched_cpg_genes_converted.csv"
STRING_GENES_FILE = "WGCNA/WGCNA_selected_genes_for_STRING.tsv"

EDGE_THRESHOLD = 0.1
MIN_SAMPLE_FRACTION = 0.2
PROM_UP = 1500
PROM_DOWN = 500

sns.set_theme(style="white")

# --- Leggi mapping e split ---
file_mapping_df = pd.read_csv(MAPPING_FILE, sep="\t")
split_df = pd.read_csv(SPLIT_FILE)
train_case_ids = split_df.loc[split_df["split"]=="train", "cases.case_id"].astype(str).unique()

meth_mapping = file_mapping_df[(file_mapping_df["omic"].str.lower()=="methylation") &
                               (file_mapping_df["case_id"].isin(train_case_ids))][["case_id","filename"]].reset_index(drop=True)
print(f"\nTrain methylation samples found: {len(meth_mapping)}")

# --- Leggi CpG-gene mapping ---
cpg_map = pd.read_csv(CPG_GENE_MAP_FILE)
# colonne usate: gene_id,gene_strand,gene_start,gene_end,cpg_island,cpg_IlmnID

# --- Costruisci beta per gene per sample ---
gene_beta = {}

for _, row in meth_mapping.iterrows():
    case_id = row["case_id"]
    filename = row["filename"].strip()
    path = os.path.join(METH_DIR, filename)

    print(f"Processing methylation for {case_id}")
    df_cpg = pd.read_csv(path, sep="\t", header=None, names=["cpg_IlmnID","beta_value"])
    df_cpg["beta_value"] = df_cpg["beta_value"].astype(float)

    # join con mapping CpG-gene
    df_annot = df_cpg.merge(cpg_map, on="cpg_IlmnID", how="inner")

    # calcola promoter start/end
    df_annot["prom_start"] = np.where(df_annot["gene_strand"]=="+",
                                      df_annot["gene_start"]-PROM_UP,
                                      df_annot["gene_end"]-PROM_UP)
    df_annot["prom_end"] = np.where(df_annot["gene_strand"]=="+",
                                    df_annot["gene_start"]+PROM_DOWN,
                                    df_annot["gene_end"]+PROM_DOWN)

    # seleziona CpG nel promotore
    df_prom = df_annot[(df_annot["cpg_island"] >= df_annot["prom_start"]) &
                       (df_annot["cpg_island"] <= df_annot["prom_end"])]

    # media beta per gene
    beta_gene = df_prom.groupby("gene_id")["beta_value"].mean()
    for gene, beta in beta_gene.items():
        if gene not in gene_beta:
            gene_beta[gene] = {}
        gene_beta[gene][case_id] = beta

# --- Trasforma in matrice gene x sample ---
beta_matrix = pd.DataFrame(gene_beta).T.fillna(0)

# --- Mantieni solo geni presenti in STRING ---
selected_genes = pd.read_csv(STRING_GENES_FILE, header=None, names=["gene"])["gene"].values
beta_matrix = beta_matrix.loc[beta_matrix.index.isin(selected_genes)]

# Filtri minima presenza campioni e varianza
min_samples = int(np.ceil(MIN_SAMPLE_FRACTION * beta_matrix.shape[1]))
beta_matrix = beta_matrix[(beta_matrix!=0).sum(axis=1)>=min_samples]
beta_matrix = beta_matrix.loc[beta_matrix.var(axis=1)>0]

print("\nFinal methylation matrix shape after filtering:", beta_matrix.shape)
beta_matrix.to_csv(os.path.join(OUTPUT_DIR,"methylation_node_matrix_filtered.tsv"), sep="\t")

# --- Features nodi methylation ---
nodes_df = pd.DataFrame({"gene": beta_matrix.index})
nodes_df["METH_mean"] = beta_matrix.mean(axis=1).values
nodes_df["METH_var"] = beta_matrix.var(axis=1).values
nodes_df["METH_status"] = np.where(nodes_df["METH_mean"]>=0.6,"methylated",
                                   np.where(nodes_df["METH_mean"]<=0.3,"unmethylated","intermediate"))
nodes_df.to_csv(os.path.join(OUTPUT_DIR,"methylation_node_features.tsv"), sep="\t", index=False)

# --- Funzione per calcolare matrice correlazione e edge list ---
def compute_edges(matrix, method_name):
    print(f"\nComputing {method_name} correlation...")
    corr_matrix = matrix.T.corr(method=method_name)

    # Edge list filtrata
    adj_sub = corr_matrix.copy()
    adj_sub[adj_sub < EDGE_THRESHOLD] = 0
    g_idx = np.where(adj_sub.values > EDGE_THRESHOLD)
    edges = [(adj_sub.index[i], adj_sub.columns[j], adj_sub.iloc[i,j])
             for i,j in zip(*g_idx) if i<j]
    edges_df = pd.DataFrame(edges, columns=["gene1","gene2","weight"])
    edges_df.to_csv(os.path.join(OUTPUT_DIR, f"methylation_edges_{method_name}_for_STRING.tsv"), sep="\t", index=False)

    print(f"{method_name} edges over threshold: {len(edges_df)}")
    return edges_df

# --- Pearson, Spearman, Kendall ---
edges_pearson = compute_edges(beta_matrix, "pearson")
edges_spearman = compute_edges(beta_matrix, "spearman")
edges_kendall = compute_edges(beta_matrix, "kendall")

# --- Confronto qualità metrica ---
summary = pd.DataFrame({
    "method":["pearson","spearman","kendall"],
    "n_edges":[len(edges_pearson), len(edges_spearman), len(edges_kendall)],
    "mean_weight":[edges_pearson["weight"].mean(),
                   edges_spearman["weight"].mean(),
                   edges_kendall["weight"].mean()]
})
print("\nComparison of correlation methods:")
print(summary)

# --- Istogrammi dei pesi ---
plt.figure(figsize=(12,4))
for i,(edges_df,name) in enumerate(zip([edges_pearson, edges_spearman, edges_kendall],
                                       ["Pearson","Spearman","Kendall"])):
    plt.subplot(1,3,i+1)
    plt.hist(edges_df["weight"], bins=50)
    plt.title(f"{name} weight distribution")
    plt.xlabel("Correlation")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"methylation_edge_weight_distributions_all_methods.png"), dpi=300)
plt.show()

print("\nMethylation pipeline complete with Pearson, Spearman and Kendall correlations.")
