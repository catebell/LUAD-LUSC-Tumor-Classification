import pandas as pd

'''
    print e controlli sui dati del file clinical.tsv
    rimozione duplicati case_id e tenere dati solo relativi a tumori primary
'''

df = pd.read_csv("dataset/clinical/clinical.tsv", sep="\t")
df.dropna(inplace=True)
print("Shape: ", df.shape)

primary = df[df["diagnoses.classification_of_tumor"] == "primary"]
print("Primary shape: ", primary.shape)
no_duplicates = primary.drop_duplicates(subset=["cases.case_id"])
print("No duplicates shape: ", no_duplicates.shape)
print("Classification of tumor: ", no_duplicates['diagnoses.classification_of_tumor'].unique())

'''
print("Null values: ")
for i in no_duplicates.columns:
    print("\t", i , ": ", len(no_duplicates[no_duplicates[i] == "\'--"]))
'''

# print("Columns to consider: ")
columns = []
for i in no_duplicates.columns:
    if len(no_duplicates[no_duplicates[i] == "\'--"]) <= no_duplicates.shape[0]/2:
        if i.startswith("treatments."): break
        columns.append(i)
        # print("\t", i)
print("Consider in total ", len(columns), " columns")

print("Number of unique values per column: ")
for i in columns:
    print("\t", i, ": ", no_duplicates[i].nunique())

print("Data unique values: ")
for i in columns:
    if no_duplicates[i].nunique() <= 50:
        print("\t", i, ": ", no_duplicates[i].unique())
print(no_duplicates["project.project_id"].value_counts())