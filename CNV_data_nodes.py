import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurazioni ---
CNV_DIR = "files/CNV"
OUTPUT_DIR = "WGCNA/CNV"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAPPING_FILE = "files/clinical/file_case_mapping.tsv"
SPLIT_FILE = "files/clinical/patient_split_cleaned.csv"
STRING_GENES_FILE = "WGCNA/WGCNA_selected_genes_for_STRING.tsv"

k_diff = 0.3
EDGE_THRESHOLD = 0.1
MIN_SAMPLE_FRACTION = 0.2  # fraction di pazienti con CNV non nullo

sns.set_theme(style="white")

# --- Leggi mapping e split ---
file_mapping_df = pd.read_csv(MAPPING_FILE, sep="\t")
split_df = pd.read_csv(SPLIT_FILE)
train_case_ids = split_df.loc[split_df["split"]=="train", "cases.case_id"].astype(str).unique()

cnv_mapping = file_mapping_df[(file_mapping_df["omic"]=="CNV") &
                              (file_mapping_df["case_id"].isin(train_case_ids))][["case_id","filename"]].reset_index(drop=True)

print(f"\nTrain CNV samples found: {len(cnv_mapping)}")

# --- Funzione per calcolare CNV_node ---
def compute_CNV_node(row, k=k_diff):
    CN = row["copy_number"]
    CN_adj = CN if CN > 0 else 0.01
    CN_log2 = np.log2(CN_adj / 2)
    diff_term = k if row["diff"] != 0 else 0
    return CN_log2 + diff_term

# --- Leggi i file CNV e costruisci matrice ---
cnv_dfs = []

for _, row in cnv_mapping.iterrows():
    case_id = row["case_id"]
    filename = row["filename"].strip()
    path = os.path.join(CNV_DIR, filename)

    print(f"Processing CNV for {case_id}")
    df_cnv = pd.read_csv(path, sep="\t", dtype=str, comment="#")

    df_cnv["copy_number"] = df_cnv["copy_number"].astype(float)
    df_cnv["min_copy_number"] = df_cnv["min_copy_number"].astype(float)
    df_cnv["max_copy_number"] = df_cnv["max_copy_number"].astype(float)
    df_cnv["gene_id"] = df_cnv["gene_id"].str.split(".", expand=True)[0]

    df_gene = df_cnv.groupby("gene_id").agg({
        "copy_number":"mean",
        "min_copy_number":"min",
        "max_copy_number":"max"
    }).reset_index()
    df_gene["diff"] = df_gene["max_copy_number"] - df_gene["min_copy_number"]
    df_gene[case_id] = df_gene.apply(compute_CNV_node, axis=1)
    cnv_dfs.append(df_gene[["gene_id", case_id]])

# --- Merge di tutti i pazienti ---
cnv_matrix = cnv_dfs[0]
for df in cnv_dfs[1:]:
    cnv_matrix = cnv_matrix.merge(df, on="gene_id", how="outer")

cnv_matrix.set_index("gene_id", inplace=True)
cnv_matrix = cnv_matrix.astype(float).fillna(0)

# --- Filtri ---
min_samples = int(np.ceil(MIN_SAMPLE_FRACTION * cnv_matrix.shape[1]))
cnv_matrix = cnv_matrix[(cnv_matrix != 0).sum(axis=1) >= min_samples]
cnv_matrix = cnv_matrix.loc[cnv_matrix.var(axis=1) > 0]

print("\nFinal CNV matrix shape after filtering:", cnv_matrix.shape)
cnv_matrix.to_csv(os.path.join(OUTPUT_DIR, "CNV_node_matrix_filtered.tsv"), sep='\t')

# --- Mantieni solo i geni selezionati per STRING ---
selected_genes = pd.read_csv(STRING_GENES_FILE, header=None, names=["gene"])["gene"].values
cnv_matrix_sub = cnv_matrix.loc[cnv_matrix.index.isin(selected_genes)]
print(f"Using {cnv_matrix_sub.shape[0]} CNV genes present in STRING backbone")

# --- Features dei nodi CNV ---
nodes_df = pd.DataFrame({"gene": cnv_matrix_sub.index})
nodes_df["CNV_mean"] = cnv_matrix_sub.mean(axis=1).values
nodes_df["CNV_var"] = cnv_matrix_sub.var(axis=1).values
nodes_df.to_csv(os.path.join(OUTPUT_DIR, "CNV_node_features.tsv"), sep='\t', index=False)

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
    edges_df.to_csv(os.path.join(OUTPUT_DIR, f"CNV_edges_{method_name}_for_STRING.tsv"), sep="\t", index=False)

    print(f"{method_name} edges over threshold: {len(edges_df)}")
    return edges_df

# --- Pearson, Spearman, Kendall ---
edges_pearson = compute_edges(cnv_matrix_sub, "pearson")
edges_spearman = compute_edges(cnv_matrix_sub, "spearman")
edges_kendall = compute_edges(cnv_matrix_sub, "kendall")

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
plt.savefig(os.path.join(OUTPUT_DIR,"CNV_edge_weight_distributions_all_methods.png"), dpi=300)
plt.show()

print("\nCNV pipeline complete with Pearson, Spearman and Kendall correlations.")