import pandas as pd

'''
    controlli sui dati del file clinical.tsv
    rimozione duplicati case_id e tenere dati solo relativi a tumori primary
'''

df = pd.read_csv("../original_dataset/clinical/clinical.tsv", sep="\t")
df.dropna(inplace=True)
print("Shape: ", df.shape)

primary = df[df["diagnoses.classification_of_tumor"] == "primary"]
print("Primary shape: ", primary.shape)
no_duplicates = primary.drop_duplicates(subset=["cases.case_id"])
no_duplicates.reset_index(inplace=True, drop=True)

print("No duplicates shape: ", no_duplicates.shape)
print("Classification of tumor: ", no_duplicates['diagnoses.classification_of_tumor'].unique())

'''
print("Null values: ")
for i in no_duplicates.columns:
    print("\t", i , ": ", len(no_duplicates[no_duplicates[i] == "\'--"]))
'''

# print('Columns to consider: ')
columns = []
for i in no_duplicates.columns:
    if len(no_duplicates[no_duplicates[i] == "\'--"]) <= no_duplicates.shape[0]/2:
        if i.startswith("treatments."): break
        columns.append(i)
        # print("\t", i)
print("Possibly ", len(columns), " columns with useful data.")


# don't consider columns with same value for everyone
cols_no_single_value = []
print("Number of unique values per column: ")
for i in columns:
    print("\t", i, ": ", no_duplicates[i].nunique())
    if no_duplicates[i].nunique() > 1:
        cols_no_single_value.append(i)
print("Consider in total ", len(cols_no_single_value), " columns:")
print(cols_no_single_value)



print("Print data unique values: ")
for i in cols_no_single_value:
    if no_duplicates[i].nunique() <= 50:
        print("\t", i, ": ", no_duplicates[i].unique())
print(no_duplicates["project.project_id"].value_counts())
print(no_duplicates['diagnoses.primary_diagnosis'].value_counts())  # I would say not to consider because matches classification



# cols printed are:
'''
[
'project.project_id',
'cases.case_id', 
'cases.consent_type', 
'cases.days_to_consent', 
'cases.disease_type', 
'cases.lost_to_followup', 
'cases.submitter_id', 
'demographic.age_at_index',
'demographic.age_is_obfuscated', 
'demographic.country_of_residence_at_enrollment',
'demographic.days_to_birth',
'demographic.demographic_id',
'demographic.ethnicity',
'demographic.gender',
'demographic.race',
'demographic.submitter_id',
'demographic.vital_status',
'diagnoses.age_at_diagnosis',
'diagnoses.ajcc_pathologic_m',
'diagnoses.ajcc_pathologic_n',
'diagnoses.ajcc_pathologic_stage',
'diagnoses.ajcc_pathologic_t',
'diagnoses.ajcc_staging_system_edition',
'diagnoses.days_to_diagnosis',
'diagnoses.days_to_last_follow_up',
'diagnoses.diagnosis_id',
'diagnoses.icd_10_code',
'diagnoses.laterality',
'diagnoses.morphology',
'diagnoses.primary_diagnosis', 
'diagnoses.prior_malignancy', 
'diagnoses.prior_treatment', 
'diagnoses.residual_disease', 
'diagnoses.submitter_id', 
'diagnoses.synchronous_malignancy', 
'diagnoses.tissue_or_organ_of_origin', 
'diagnoses.year_of_diagnosis'
]
'''