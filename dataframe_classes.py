import pandas as pd

df = pd.read_csv(r"files/clinical/file_case_with_project.tsv", sep="\t")

df = df.drop_duplicates()
df.dropna(inplace=True)

df = df.drop(columns=['filename'])

df = df.sort_values(by="project_id").reset_index(drop=True)

df.to_csv(r"files/clinical/dataframe.tsv", sep="\t", index=False)

print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df['project_id'].unique())
print(df['project_id'].value_counts())
