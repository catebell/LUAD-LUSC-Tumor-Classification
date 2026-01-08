import pandas as pd

df = pd.read_csv("dataset/clinical/clinical.tsv", sep="\t")
df.dropna(inplace=True)
print("Shape: ", df.shape)

print("Null values: ")


print("Data values: ")
print("cases.case_id: ", df["cases.case_id"].nunique())
print("project.project_id: ", df["project.project_id"].unique())
