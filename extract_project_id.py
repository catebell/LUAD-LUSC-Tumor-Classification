import pandas as pd

file_mapping = pd.read_csv(r"files/clinical/file_case_mapping.tsv", sep="\t")
clinical = pd.read_csv(r"files/clinical/features.tsv", sep="\t")

merged = pd.merge(
    clinical,
    file_mapping,
    left_on='cases.case_id',
    right_on='case_id',
    how='left'
)

merged = merged.drop(columns=['cases.case_id'])
merged = merged.rename(columns={'project.project_id': 'project_id'})

final_df = merged.to_csv(
    r"files/clinical/file_case_with_project.tsv",
    sep="\t",
    index=False
)

print("File created: file_case_with_project.tsv")
print(merged.head())
print(merged.shape)
print(merged.isnull().sum())
print("\nRows with null values:")
print(merged[merged.isnull().any(axis=1)])