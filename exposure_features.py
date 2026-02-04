import pandas as pd
import numpy as np

'''
    print e controlli sui dati del file exposure_features.tsv
'''

df = pd.read_csv("dataset/clinical/exposure.tsv", sep="\t")
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)
print("Shape: ", df.shape)

print("Null values: ")
for i in df.columns:
    print("\t", i , ": ", len(df[df[i] == "\'--"]))

print("Columns to consider: ")
columns = []
for i in df.columns:
    if len(df[df[i] == "\'--"]) <= df.shape[0]/2:
        columns.append(i)
        print("\t", i)
print("Consider in total ", len(columns), " columns")

print("Number of unique values per column: ")
for i in columns:
    print("\t", i, ": ", df[i].nunique())

print("Data unique values: ")
for i in columns:
    if df[i].nunique() != df.shape[0]:
        print("\t", i, ": ", df[i].unique())

print("Data types smoking years:")
print("\t", df["exposures.tobacco_smoking_onset_year"].dtypes)
print("\t", df["exposures.tobacco_smoking_quit_year"].dtypes)

columns_smoking_years = [
    "exposures.tobacco_smoking_onset_year",
    "exposures.tobacco_smoking_quit_year"
]

smoking_status_years = df.loc[:, [
    "exposures.tobacco_smoking_status",
    *columns_smoking_years
]]

# TODO: does not work
smoking_status_years.loc[:, columns_smoking_years] = (
    smoking_status_years.loc[:, columns_smoking_years]
    .replace("'--", np.nan)
    .apply(pd.to_numeric, errors="coerce")
    .astype("Int64")
)
print("Data types: ", smoking_status_years.dtypes)

print("Current Reformed Smoker for < or = 15 yrs")
ref_smokers_min = smoking_status_years[
    smoking_status_years["exposures.tobacco_smoking_status"] == "Current Reformed Smoker for < or = 15 yrs"
]
print("Number of Current Reformed Smoker for < or = 15 yrs: ", len(ref_smokers_min))
for i in columns_smoking_years:
    print("\t", i, ": ", ref_smokers_min[i].isnull().sum())

print("Current Reformed Smoker for > 15 yrs")
ref_smokers_max = smoking_status_years[
    smoking_status_years["exposures.tobacco_smoking_status"] == "Current Reformed Smoker for > 15 yrs"
]
print("Number of Current Reformed Smoker for > 15 yrs: ", len(ref_smokers_max))
for i in columns_smoking_years:
    print("\t", i, ": ", ref_smokers_max[i].isnull().sum())

print("Current Smoker")
smokers = smoking_status_years[
    smoking_status_years["exposures.tobacco_smoking_status"] == "Current Smoker"
]
print("Number of Current Smoker: ", len(smokers))
for i in columns_smoking_years:
    print("\t", i, ": ", smokers[i].isnull().sum())
# print(smokers[smokers["exposures.tobacco_smoking_quit_year"].notnull()])

print("Lifelong Non-Smoker")
non_smokers = smoking_status_years[
    smoking_status_years["exposures.tobacco_smoking_status"] == "Lifelong Non-Smoker"
]
print("Number of Lifelong Non-Smoker: ", len(non_smokers))
for i in columns_smoking_years:
    print("\t", i, ": ", non_smokers[i].isnull().sum())

print("Not Reported")
not_reported = smoking_status_years[
    smoking_status_years["exposures.tobacco_smoking_status"] == "Not Reported"
]
print("Number of Not Reported: ", len(not_reported))
for i in columns_smoking_years:
    print("\t", i, ": ", not_reported[i].isnull().sum())

print("Unknown")
unknown = smoking_status_years[
    smoking_status_years["exposures.tobacco_smoking_status"] == "Unknown"
]
print("Number of Unknown: ", len(unknown))
for i in columns_smoking_years:
    print("\t", i, ": ", unknown[i].isnull().sum())

print("Current Reformed Smoker, Duration Not Specified")
ref_smokers = smoking_status_years[
    smoking_status_years["exposures.tobacco_smoking_status"] == "Current Reformed Smoker, Duration Not Specified"
]
print("Number of Current Reformed Smoker, Duration Not Specified: ", len(ref_smokers))
for i in columns_smoking_years:
    print("\t", i, ": ", ref_smokers[i].isnull().sum())