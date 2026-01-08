import pandas as pd

file_mapping = pd.read_csv(r"files/clinical/file_case_mapping.tsv", sep="\t")
clinical = pd.read_csv(r"dataset/clinical/clinical.tsv", sep="\t")

merged = pd.merge(
    file_mapping,
    clinical[['cases.case_id', 'project.project_id']],
    left_on='case_id',
    right_on='cases.case_id',
    how='left'
)

merged = merged.rename(columns={'project.project_id': 'project_id'})

final_df = merged[['project_id', 'case_id', 'file_id', 'omic', 'filename']]

final_df.to_csv(r"files/clinical/file_case_with_project.tsv", sep="\t", index=False)

print("File created: file_case_with_project.tsv")
print(final_df.head())
print(final_df.shape)
print(final_df.isnull().sum())
print("\nRows with null values:")
print(final_df[final_df.isnull().any(axis=1)])