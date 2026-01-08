import os
import pandas as pd

BASE_DIR = "files"
OMICS = ["CNV", "RNA", "methylation"]

rows = []

for omic in OMICS:
    omic_dir = os.path.join(BASE_DIR, omic)

    for root, _, files in os.walk(omic_dir):
        for fname in files:

            if "." not in fname:
                continue

            # CNV: TCGA-LUAD/LUSC.<file_id>.*
            if omic == "CNV":
                first_part = fname.split(".", 2)[1]
                file_id = first_part

            # RNA / methylation: <file_id>.*
            else:
                file_id = fname.split(".", 1)[0]

            rows.append({
                "omic": omic,
                "filename": fname,
                "file_id": file_id,
                "path": os.path.join(root, fname)
            })

df = pd.DataFrame(rows)

print(df.head())

for omic in OMICS:
    subset = df[df["omic"] == omic]
    print(f"\n=== {omic} ===")
    print(subset.iloc[0])

output_file = "files\clinical\omics_files.tsv"
df.to_csv(output_file, sep="\t", index=False)

print(f"\nFile TSV created: {output_file}")
print(df.head())
print(df.shape)
print(df.isnull().sum().sum())