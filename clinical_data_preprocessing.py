import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from MLP_clinical import ClinicalMLP
import torch

df = pd.read_csv(r"files/clinical/features.tsv", sep="\t")
# print(df.dtypes)

y = df["project.project_id"].astype(int) # target

cols_to_drop = [
    "project.project_id",
    "cases.case_id",
    "cases.submitter_id",
    "demographic.age_is_obfuscated",
    "diagnoses.ajcc_staging_system_edition"
]

df.drop(columns=cols_to_drop, inplace=True)

numerical_cols = [
    "exposures.pack_years_smoked",
    "exposures.tobacco_years",
    "demographic.age_at_index"
]

for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# print(df.dtypes)

categorical_cols = [
    "demographic.country_of_residence_at_enrollment",
    "demographic.ethnicity",
    "demographic.race",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.icd_10_code",
    "diagnoses.laterality",
    "diagnoses.morphology",
    "diagnoses.sites_of_involvement",
    "diagnoses.tissue_or_organ_of_origin",
    "exposures.tobacco_smoker"
]

binary_cols = ["demographic.gender"]

df["demographic.gender"] = df["demographic.gender"].map({
    "male": 0,
    "female": 1
})

df[categorical_cols] = df[categorical_cols].replace("'--", "Unknown")
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
        ("bin", "passthrough", binary_cols)
    ]
)

X = preprocessor.fit_transform(df) # X.shape = (n_patients, input_dim)
print("X shape:", X.shape)

input_dim = X.shape[1]
print("Input dimension:", input_dim)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

model = ClinicalMLP(input_dim=X.shape[1])
z_clinical = model(X_tensor)
print("Clinical shape:", z_clinical.shape)