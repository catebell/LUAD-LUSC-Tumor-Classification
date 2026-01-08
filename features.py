import pandas as pd

df = pd.read_csv("dataset/clinical/exposure.tsv", sep="\t")
df.dropna(inplace=True) # remove rows with wrong formatting

# if we want to be sure the first two cols have a specified value:
# https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value

features = pd.DataFrame(data=df, columns=['project.project_id', 'cases.case_id', 'cases.submitter_id', 'exposures.exposure_type', 'exposures.pack_years_smoked', 'exposures.tobacco_smoking_onset_year', 'exposures.tobacco_smoking_quit_year', 'exposures.tobacco_smoking_status'])
print(features)

# for exposures.tobacco_smoking_status
mapping1 = {
    '\'--': 0,
    'Not Reported': 0,
    'Unknown': 0,
    'Lifelong Non-Smoker': 1,
    'Current Reformed Smoker for < or = 15 yrs': 2,
    'Current Reformed Smoker for > 15 yrs': 3,
    'Current Reformed Smoker, Duration Not Specified': 4,
    'Current Smoker': 5
}

# for project.project_id
mapping2 = {
    'TCGA-LUSC': 0,
    'TCGA-LUAD': 1
}

features['exposures.tobacco_smoking_status'] = features['exposures.tobacco_smoking_status'].map(mapping1)
print("exposures.tobacco_smoking_status: ", features["exposures.tobacco_smoking_status"].unique())

features['project.project_id'] = features['project.project_id'].map(mapping2)
print("project.project_id: ", features["project.project_id"].unique())


df2 = pd.read_csv("dataset/clinical/clinical.tsv", sep="\t")