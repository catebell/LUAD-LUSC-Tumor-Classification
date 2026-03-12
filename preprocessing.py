import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# EXPOSURE.TSV

df_exposure = pd.read_csv("original_dataset/clinical/exposure.tsv", sep="\t",
                          usecols=['project.project_id','cases.case_id','exposures.pack_years_smoked',
                                   'exposures.tobacco_smoking_onset_year','exposures.tobacco_smoking_quit_year',
                                   'exposures.tobacco_smoking_status'])
df_exposure.dropna(inplace=True) # remove rows with wrong formatting

# consider only columns with >50% of data not null
cols = []
for i in df_exposure.columns:
    if len(df_exposure[df_exposure[i] == "\'--"]) <= df_exposure.shape[0]/2:
        cols.append(i)
print("Columns kept from exposure.tsv: " + str(cols))

features_df = pd.DataFrame(data=df_exposure, columns = cols)

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


# CLINICAL.TSV

df_clinical = pd.read_csv("original_dataset/clinical/clinical.tsv", sep="\t")
df_exposure.dropna(inplace=True) # remove rows with wrong formatting

# for each patient (case_id) keep only first row with classification_of_tumor == 'primary'
only_primary_df = df_clinical[df_clinical["diagnoses.classification_of_tumor"] == "primary"]
only_primary_df = only_primary_df.drop_duplicates(subset=["cases.case_id"])
only_primary_df.reset_index(inplace=True, drop=True)

cols = [
    'project.project_id',
    'cases.case_id',
    'demographic.age_at_index',
    'demographic.country_of_residence_at_enrollment',
    'demographic.ethnicity',
    'demographic.gender',
    'demographic.race',
    'diagnoses.ajcc_pathologic_m',
    'diagnoses.ajcc_pathologic_n',
    'diagnoses.ajcc_pathologic_stage',
    'diagnoses.ajcc_pathologic_t',
    'diagnoses.icd_10_code',
    'diagnoses.laterality',
    'diagnoses.sites_of_involvement',
    'diagnoses.tissue_or_organ_of_origin']


features_df = features_df.join(only_primary_df[cols].set_index(['project.project_id', 'cases.case_id']),
                               on=['project.project_id', 'cases.case_id'])
print("Columns joined from clinical.tsv: " + str(cols))

features_df.replace(['\'--', 'not reported', 'Unknown', 'MX', 'NX', 'TX'], pd.NA, inplace=True)

features_df.to_csv(r"files/clinical/features_considered.tsv", sep="\t", index=False)

print("\nEncoding features to numerical...")

#print(features_df.nunique())

# for project.project_id, remap tumor class to 0-1
mapping = {
    'TCGA-LUAD': 0,
    'TCGA-LUSC': 1
}
features_df['project.project_id'] = features_df['project.project_id'].map(mapping)

# for exposures.tobacco_smoker 0 will be missing value
mapping = {
    True: 1,
    False: -1
}
features_df['exposures.tobacco_smoker'] = features_df['exposures.tobacco_smoker'].map(mapping).fillna(0)

# features already numerical, set NA to 0:
cols = ['exposures.pack_years_smoked', 'exposures.tobacco_years', 'demographic.age_at_index']
for col in cols:
    median_val = pd.to_numeric(features_df[col], errors='coerce').median()  # if set to 0 it may be considered outlier
    features_df[col] = features_df[col].fillna(median_val)

# for diagnoses.ajcc_pathologic_stage, ordinal:
stage_map = {'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1, 'Stage II': 2, 'Stage IIA': 2,
             'Stage IIB': 2, 'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IV': 4}
features_df['diagnoses.ajcc_pathologic_stage'] = features_df['diagnoses.ajcc_pathologic_stage'].map(stage_map).fillna(0)

# for categorical features:
cols = [
    'demographic.country_of_residence_at_enrollment',
    'demographic.ethnicity',
    'demographic.gender',
    'demographic.race',
    'diagnoses.ajcc_pathologic_m',
    'diagnoses.ajcc_pathologic_n',
    'diagnoses.ajcc_pathologic_t',
    'diagnoses.icd_10_code',
    'diagnoses.laterality',
    'diagnoses.sites_of_involvement',
    'diagnoses.tissue_or_organ_of_origin'
]

categories = [features_df[col].dropna().unique() for col in cols]  # all possible not NA vals of each category

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(categories=categories, handle_unknown='ignore'), cols)],
    remainder='passthrough'  # Keep other columns as-is
)
encoded_array = ct.fit_transform(features_df)
encoded_df = pd.DataFrame(encoded_array, columns=ct.get_feature_names_out())

# to change encoded names from 'encoder__demographic.country_of_residence_at_enrollment_Australia' to 'country_of_residence_at_enrollment_Australia'
encoded_df = encoded_df.rename(columns=dict(zip(encoded_df.columns, [s.split('.')[-1] for s in encoded_df.columns])))

# CREATE FILE
encoded_df.to_csv(r"files/clinical/features_encoded.tsv", sep="\t", index=False)

print("DONE")