import pandas as pd

df = pd.read_csv(f"dataset/matched_cpg_genes_converted.csv", sep=",", dtype=str)

print(df.shape)

print(df["cpg_IlmnID"].nunique())

print(df["gene_id"].nunique())