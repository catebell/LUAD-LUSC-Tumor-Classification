import pandas as pd
from sklearn.model_selection import train_test_split

mapping_df = pd.read_csv(r'files/clinical/file_case_mapping.tsv', sep='\t')
features_df = pd.read_csv(r'files/clinical/features.tsv', sep='\t')

omic_counts = mapping_df.groupby('case_id')['omic'].nunique()
complete_case_ids = omic_counts[omic_counts == 3].index.tolist()

print("Original unique patients in mapping: ", len(omic_counts))
print("Patients with 3/3 omics: ", len(complete_case_ids))
print("Patients to be removed: ",len(omic_counts) - len(complete_case_ids))

filtered_patients = features_df[features_df['cases.case_id'].isin(complete_case_ids)]
filtered_patients = filtered_patients[['cases.case_id', 'project.project_id']].drop_duplicates()

# Perform Stratified Split: 80% (Train+Val) and 20% Test
train_val, test = train_test_split(
    filtered_patients,
    test_size=0.20,
    stratify=filtered_patients['project.project_id'],
    random_state=42
)

# Split Train+Val into Train (70% total) and Val (10% total): 0.125 * 0.8 = 0.1
train, val = train_test_split(
    train_val,
    test_size=0.125,
    stratify=train_val['project.project_id'],
    random_state=42
)

filtered_patients['split'] = 'none'
filtered_patients.loc[filtered_patients['cases.case_id'].isin(train['cases.case_id']), 'split'] = 'train'
filtered_patients.loc[filtered_patients['cases.case_id'].isin(val['cases.case_id']), 'split'] = 'val'
filtered_patients.loc[filtered_patients['cases.case_id'].isin(test['cases.case_id']), 'split'] = 'test'

filtered_patients = filtered_patients.sort_values(by="split").reset_index(drop=True)
filtered_patients.to_csv('files/clinical/patient_split_cleaned.csv', index=False)

stats = filtered_patients.groupby(['split', 'project.project_id']).size().unstack()
print("Stats: ", stats)
print("Final number of patients: ", len(filtered_patients))