import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def process_clinical_data():

    # =========================
    # 1️⃣ Load split
    # =========================

    split_df = pd.read_csv("files/clinical/patient_split_cleaned.csv")

    train_ids = split_df.loc[split_df["split"] == "train", "cases.case_id"].tolist()
    val_ids   = split_df.loc[split_df["split"] == "val", "cases.case_id"].tolist()
    test_ids  = split_df.loc[split_df["split"] == "test", "cases.case_id"].tolist()

    # =========================
    # 2️⃣ Load clinical features
    # =========================

    df = pd.read_csv("files/clinical/features.tsv", sep="\t")

    df["project.project_id"] = df["project.project_id"].astype(int)

    # =========================
    # 3️⃣ Column definitions
    # =========================

    numerical_cols = [
        "exposures.pack_years_smoked",
        "exposures.tobacco_years",
        "demographic.age_at_index"
    ]

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
        "diagnoses.sites_of_involvement",
        "diagnoses.tissue_or_organ_of_origin",
        "exposures.tobacco_smoker"
    ]

    binary_cols = ["demographic.gender"]

    # =========================
    # 4️⃣ Basic cleaning
    # =========================

    df["demographic.gender"] = df["demographic.gender"].map({
        "male": 0,
        "female": 1
    })

    df[categorical_cols] = df[categorical_cols].replace("'--", "Unknown")
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================
    # 5️⃣ Split dataframe
    # =========================

    df_train = df[df["cases.case_id"].isin(train_ids)].reset_index(drop=True)
    df_val   = df[df["cases.case_id"].isin(val_ids)].reset_index(drop=True)
    df_test  = df[df["cases.case_id"].isin(test_ids)].reset_index(drop=True)

    # =========================
    # 6️⃣ Preprocessor (FIT SOLO SU TRAIN)
    # =========================

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("bin", "passthrough", binary_cols)
    ])

    X_train = preprocessor.fit_transform(df_train)
    X_val   = preprocessor.transform(df_val)
    X_test  = preprocessor.transform(df_test)

    y_train = df_train["project.project_id"].values
    y_val   = df_val["project.project_id"].values
    y_test  = df_test["project.project_id"].values

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        train_ids, val_ids, test_ids
    )