import pandas as pd

# EXPOSURE.TSV

df_exposure = pd.read_csv("dataset/clinical/exposure.tsv", sep="\t")
df_exposure.dropna(inplace=True) # remove rows with wrong formatting ()

# consider only columns with >50% of data not null
columns = []
for i in df_exposure.columns:
    if len(df_exposure[df_exposure[i] == "\'--"]) <= df_exposure.shape[0]/2:
        columns.append(i)
print("Columns kept from exposure.tsv: " + str(columns))

features_df = pd.DataFrame(data=df_exposure, columns = columns)
features_df.drop(['exposures.exposure_type'], axis=1, inplace=True)

# add cols
new_col_smoker = features_df.loc[:,'project.project_id'].astype(str).copy()
new_col_smoker.loc[:] = '\'--' # Null, True or False
new_col_years = features_df.loc[:,'project.project_id'].astype(str).copy()
new_col_years.loc[:] = '\'--' # Null, count_years (int)


# change obj type to int
columns_smoking_years = ["exposures.tobacco_smoking_onset_year", "exposures.tobacco_smoking_quit_year"]

for i in columns_smoking_years:
    features_df[i] = features_df[i].astype(str)
    features_df.loc[features_df[i] == '\'--', i] = '0'
    features_df[i] = features_df[i].astype(int)
#print("Dtype Features:\n",features_df.dtypes)

# if both years not null --> True, data_stop - data_start
new_col_smoker.loc[(features_df['exposures.tobacco_smoking_quit_year'] != 0) & (features_df['exposures.tobacco_smoking_onset_year'] != 0)] = True
new_col_years.loc[(features_df['exposures.tobacco_smoking_quit_year'] != 0) & (features_df['exposures.tobacco_smoking_onset_year'] != 0)] = features_df['exposures.tobacco_smoking_quit_year'] - features_df['exposures.tobacco_smoking_onset_year']

# if only one year missing --> True, Null
new_col_smoker.loc[
    ((features_df['exposures.tobacco_smoking_quit_year'] != 0) & (features_df['exposures.tobacco_smoking_onset_year'] == 0))
    | ((features_df['exposures.tobacco_smoking_quit_year'] == 0) & (features_df['exposures.tobacco_smoking_onset_year'] != 0)
        )] = True

# if both years missing --> check status: if 'Lifelong Non-Smoker' --> False, else Null
new_col_smoker.loc[
    (features_df['exposures.tobacco_smoking_quit_year'] == 0) &
    (features_df['exposures.tobacco_smoking_onset_year'] == 0) &
    (features_df['exposures.tobacco_smoking_status'] == 'Lifelong Non-Smoker')] = False

# add new cols
features_df['exposures.tobacco_smoker'] = new_col_smoker # Null, True or False
features_df['exposures.tobacco_years'] = new_col_years # Null, count_years (int)
print("Columns added: ['exposures.tobacco_smoker', 'exposures.tobacco_years']")

# drop cols used and not more useful
features_df.drop(['exposures.tobacco_smoking_onset_year'], axis=1, inplace=True)
features_df.drop(['exposures.tobacco_smoking_quit_year'], axis=1, inplace=True)
features_df.drop(['exposures.tobacco_smoking_status'], axis=1, inplace=True)

# DEBUG
'''
print("Features Shape: ", features_df.shape)
print("Null values: ")
for i in features_df.columns:
    print("\t", i , ": ", len(features_df[features_df[i] == "\'--"]))
print("Data unique values: ")
for i in features_df.columns:
    if features_df[i].nunique() != features_df.shape[0]:
        print("\t", i, ": ", features_df[i].unique())
'''


# CLINICAL.TSV

df_clinical = pd.read_csv("dataset/clinical/clinical.tsv", sep="\t")
df_exposure.dropna(inplace=True) # remove rows with wrong formatting

# for each patient (case_id) keep only first row with classification_of_tumor == 'primary'
only_primary_df = df_clinical[df_clinical["diagnoses.classification_of_tumor"] == "primary"]
only_primary_df.drop_duplicates(subset=["cases.case_id"], inplace=True)
only_primary_df.reset_index(inplace=True, drop=True)

cols = [
    'project.project_id',
    'cases.case_id',
    'cases.submitter_id',
    'demographic.age_at_index',
    'demographic.age_is_obfuscated',
    'demographic.country_of_residence_at_enrollment',
    'demographic.ethnicity',
    'demographic.gender',
    'demographic.race',
    'diagnoses.ajcc_pathologic_m',
    'diagnoses.ajcc_pathologic_n',
    'diagnoses.ajcc_pathologic_stage',
    'diagnoses.ajcc_pathologic_t',
    'diagnoses.ajcc_staging_system_edition',
    'diagnoses.icd_10_code',
    'diagnoses.laterality',
    'diagnoses.morphology',
    'diagnoses.sites_of_involvement',
    'diagnoses.tissue_or_organ_of_origin']

index_cols = ['project.project_id', 'cases.case_id', 'cases.submitter_id']

features_df = features_df.join(no_duplicates_primary_df[cols].set_index(index_cols), on=index_cols)
print("Columns joined from clinical.tsv: " + str(cols))

# for project.project_id
mapping_project_id = {
    'TCGA-LUAD': 0,
    'TCGA-LUSC': 1
}
# remap tumor class to 0-1
features_df['project.project_id'] = features_df['project.project_id'].map(mapping_project_id)


# CREATE FILE
features_df.to_csv(r"files/clinical/features.tsv", sep="\t", index=False)

print("DONE")