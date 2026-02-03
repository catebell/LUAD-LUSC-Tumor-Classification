import pandas as pd

df = pd.read_csv(r'files/clinical/file_case_mapping.tsv', sep='\t')

all_omics = set(['CNV', 'RNA', 'methylation'])

grouped = df.groupby('case_id')['omic'].unique().apply(set).reset_index()

incomplete = grouped[grouped['omic'].apply(lambda x: x != all_omics)]

print(f"{'CASE_ID': <40} | {'OMICS'}")
print("-" * 70)

for _, row in incomplete.iterrows():
    presents = ", ".join(sorted(list(row['omic'])))
    print(f"{row['case_id']:<40} | {presents}")

print("Length: ", len(incomplete))