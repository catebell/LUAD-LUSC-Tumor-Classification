import os
import argparse
import pandas as pd
import config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

DEBUG = False

EXCLUDED_CLINICAL = [
    "cases.submitter_id",
    "cases.consent_type",
    "cases.days_to_consent",
    "cases.disease_type",
    "cases.lost_to_followup",
    "demographic.age_is_obfuscated",
    "demographic.days_to_birth",
    "demographic.demographic_id",
    "demographic.submitter_id",
    "demographic.vital_status",
    "diagnoses.age_at_diagnosis",
    "diagnoses.ajcc_staging_system_edition",
    "diagnoses.days_to_diagnosis",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.diagnosis_id",
    "diagnoses.morphology",
    "diagnoses.primary_diagnosis",
    "diagnoses.prior_malignancy",
    "diagnoses.prior_treatment",
    "diagnoses.residual_disease",
    "diagnoses.submitter_id",
    "diagnoses.synchronous_malignancy",
    "diagnoses.year_of_diagnosis",
    "treatments.submitter_id",
    "treatments.treatment_id",
    "treatments.treatment_intent_type",
    "treatments.treatment_or_therapy",
    "treatments.treatment_type"
]

EXCLUDED_EXPOSURE = [
    "cases.submitter_id"
]

def get_available_datasets():
    """Return subfolders inside original_dataset/"""
    if not os.path.isdir(config.DATASET):
        return []

    return sorted([
        d for d in os.listdir(config.DATASET)
        if os.path.isdir(os.path.join(config.DATASET, d))
    ])


def parse_args():
    """Parse command line arguments"""
    datasets = get_available_datasets()

    parser = argparse.ArgumentParser(
        description="Clinical preprocessing pipeline"
    )

    parser.add_argument(
        "--dataset",
        default=config.tumor,
        choices=datasets,
        help="Available datasets: " + ", ".join(datasets) + ""
    )

    return parser.parse_args()


def columns_to_keep(df, excluded=None):
    """
    Keep columns not in the excluded list if:
    1) missing values are <= 50%
    2) values with unique count > 1
    """

    if excluded is None:
        excluded = []

    cols = []

    for i in df.columns:

        if i in excluded:
            continue

        missing_count = len(df[df[i] == "'--"])
        unique_count = df[i].nunique()

        if (
            missing_count <= df.shape[0] / 2
            and unique_count > 1
        ):
            cols.append(i)

    print(f"Columns to keep: {len(cols)}")
    print(cols)

    return cols

def debug_df(df, name):
    """Debug dataframe"""
    if not DEBUG:
        return

    print(f"\n{name} DEBUG")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    for col in df.columns:
        print(f"\n{col}")
        print("unique:", df[col].nunique())
        print(df[col].value_counts().head(5))

def add_smoker_features(df_exposure):
    """
    Create:
    - exposures.tobacco_smoker
    - exposures.tobacco_years
    """

    onset = "exposures.tobacco_smoking_onset_year"
    quit = "exposures.tobacco_smoking_quit_year"
    status = "exposures.tobacco_smoking_status"

    if onset not in df_exposure.columns or quit not in df_exposure.columns:
        return df_exposure

    # add cols
    new_col_smoker = pd.Series(pd.NA, index=range(len(df_exposure.index)))  # Null, True or False
    new_col_years = pd.Series(pd.NA, index=range(len(df_exposure.index)))  # Null, count_years (int)

    # change obj type to int
    columns_smoking_years = ["exposures.tobacco_smoking_onset_year", "exposures.tobacco_smoking_quit_year"]
    for i in columns_smoking_years:
        df_exposure[i] = df_exposure[i].astype(str)
        df_exposure.loc[df_exposure[i] == '\'--', i] = '0'
        df_exposure[i] = df_exposure[i].astype(int)

    # if both years not null --> True, data_stop - data_start
    new_col_smoker.loc[(df_exposure['exposures.tobacco_smoking_quit_year'] != 0) & (
                df_exposure['exposures.tobacco_smoking_onset_year'] != 0)] = True
    new_col_years.loc[(df_exposure['exposures.tobacco_smoking_quit_year'] != 0) & (
                df_exposure['exposures.tobacco_smoking_onset_year'] != 0)] = df_exposure[
                                                                                 'exposures.tobacco_smoking_quit_year'] - \
                                                                             df_exposure[
                                                                                 'exposures.tobacco_smoking_onset_year']

    # if only one year missing --> True, Null
    new_col_smoker.loc[
        ((df_exposure['exposures.tobacco_smoking_quit_year'] != 0) & (
                    df_exposure['exposures.tobacco_smoking_onset_year'] == 0))
        | ((df_exposure['exposures.tobacco_smoking_quit_year'] == 0) & (
                    df_exposure['exposures.tobacco_smoking_onset_year'] != 0)
           )] = True

    # if both years missing --> check status: if 'Lifelong Non-Smoker' --> False, else Null
    new_col_smoker.loc[
        (df_exposure['exposures.tobacco_smoking_quit_year'] == 0) &
        (df_exposure['exposures.tobacco_smoking_onset_year'] == 0) &
        (df_exposure['exposures.tobacco_smoking_status'] == 'Lifelong Non-Smoker')] = False

    # add new cols
    df_exposure['exposures.tobacco_smoker'] = new_col_smoker  # Null, True or False
    df_exposure['exposures.tobacco_years'] = new_col_years  # Null, count_years (int)
    print("Columns added: ['exposures.tobacco_smoker', 'e"
          "xposures.tobacco_years']")

    # drop cols used and not more useful
    df_exposure.drop(['exposures.tobacco_smoking_onset_year'], axis=1, inplace=True)
    df_exposure.drop(['exposures.tobacco_smoking_quit_year'], axis=1, inplace=True)
    df_exposure.drop(['exposures.tobacco_smoking_status'], axis=1, inplace=True)

    return df_exposure


def process_exposure(path):
    """Preprocessing of the exposure file"""
    df_exposure = pd.read_csv(path, sep="\t")
    debug_df(df_exposure, "EXPOSURE")

    df_exposure.dropna(inplace=True) # remove rows with wrong formatting

    cols = columns_to_keep(df_exposure, excluded=EXCLUDED_EXPOSURE)

    features_df = pd.DataFrame(data=df_exposure, columns=cols)

    features_df = add_smoker_features(features_df)

    return features_df

def process_clinical(path):
    """Preprocessing of the clinical file"""
    df_clinical = pd.read_csv(path, sep="\t")
    debug_df(df_clinical, "CLINICAL")

    df_clinical.dropna(inplace=True)  # remove rows with wrong formatting

    # for each patient (case_id) keep only the first row with classification_of_tumor == 'primary'
    if "diagnoses.classification_of_tumor" in df_clinical.columns:
        only_primary_df = df_clinical[df_clinical["diagnoses.classification_of_tumor"] == "primary"]
        only_primary_df = only_primary_df.drop_duplicates(subset=["cases.case_id"])
        only_primary_df.reset_index(inplace=True, drop=True)

    cols = columns_to_keep(only_primary_df, excluded=EXCLUDED_CLINICAL)

    features_df = pd.DataFrame(data=only_primary_df, columns=cols)

    return features_df

def build_features_considered(dataset):
    """Build the features_considered file by joining the exposure and clinical files"""
    dataset_dir = os.path.join(config.DATASET, dataset)
    output_dir = os.path.join(config.FILES, dataset, "clinical")

    os.makedirs(output_dir, exist_ok=True)

    exposure_path = os.path.join(dataset_dir, "clinical", "exposure.tsv")
    clinical_path = os.path.join(dataset_dir, "clinical", "clinical.tsv")

    df_exposure = process_exposure(exposure_path)
    df_clinical = process_clinical(clinical_path)

    features_df = df_exposure.merge(
        df_clinical,
        on=['project.project_id', 'cases.case_id'],
        how='left'
    )

    features_df.replace(
        ["'--", "not reported", "Unknown", "MX", "NX", "TX"],
        pd.NA,
        inplace=True
    )

    out_file = os.path.join(output_dir, "features_considered.tsv")
    features_df.to_csv(out_file, sep="\t", index=False)

    print(f"\nSaved: {out_file}")
    print("Final shape:", features_df.shape)

def encode_stage(value):
    """Generic AJCC stage encoder"""

    if pd.isna(value):
        return 0

    value = str(value).upper()

    if "IV" in value:
        return 4
    elif "III" in value:
        return 3
    elif "II" in value:
        return 2
    elif "I" in value:
        return 1

    return 0

def build_features_encoded(dataset):
    """Build the features_encoded file starting from features_considered"""

    dataset_dir = os.path.join(config.DATASET, dataset)
    output_dir = os.path.join(config.FILES, dataset, "clinical")

    os.makedirs(output_dir, exist_ok=True)

    input_file = os.path.join(output_dir, "features_considered.tsv")
    out_file = os.path.join(output_dir, "features_encoded.tsv")

    features_df = pd.read_csv(input_file, sep="\t")

    print("\nEncoding features to numerical...")

    if "project.project_id" in features_df.columns:

        classes = sorted(features_df["project.project_id"].dropna().unique())

        # for project.project_id, remap tumor class to numbers
        mapping = {label: i for i, label in enumerate(classes)}

        features_df["project.project_id"] = (
            features_df["project.project_id"].map(mapping)
        )

        print("Target mapping:", mapping)

    if "exposures.tobacco_smoker" in features_df.columns:

        smoker_map = {
            True: 1,
            False: -1
        }

        # for exposures.tobacco_smoker 0 will be missing value
        features_df["exposures.tobacco_smoker"] = (
            features_df["exposures.tobacco_smoker"]
            .map(smoker_map)
            .fillna(0)
        )

    # for diagnoses.ajcc_pathologic_stage, ordinal:
    if "diagnoses.ajcc_pathologic_stage" in features_df.columns:

        features_df["diagnoses.ajcc_pathologic_stage"] = (
            features_df["diagnoses.ajcc_pathologic_stage"]
            .apply(encode_stage)
        )

    if "exposures.tobacco_smoking_status" in features_df.columns:
        smoking_map = {
            "Not_Reported": "unknown",
            "Lifelong_Non-Smoker": "non_smoker",
            "Current_Smoker": "smoker",
            "Current_Reformed_Smoker_for_<_or_=_15_yrs": "former_smoker",
            "Current_Reformed_Smoker_for_>_15_yrs": "former_smoker",
            "Current_Reformed_Smoker,_Duration_Not_Specified": "former_smoker"
        }

        features_df["exposures.tobacco_smoking_status"] = (
            features_df["exposures.tobacco_smoking_status"]
            .astype("string")
            .str.replace(" ", "_", regex=False)
            .map(smoking_map)
            .fillna("unknown")
        )

    # IDs must not go into encoding
    id_cols = ["project.project_id", "cases.case_id"]

    # features already numerical, set NA to median
    numeric_cols = []

    for col in features_df.columns:

        if col in id_cols:
            continue

        converted = pd.to_numeric(features_df[col], errors="coerce")

        if converted.notna().sum() > 0:
            numeric_cols.append(col)

    for col in numeric_cols:

        features_df[col] = pd.to_numeric(
            features_df[col],
            errors="coerce"
        )

        median_val = features_df[col].median()

        features_df[col] = features_df[col].fillna(median_val)

    # for categorical features:
    categorical_cols = []

    for col in features_df.columns:

        if col in id_cols:
            continue

        if col not in numeric_cols:
            categorical_cols.append(col)

    print("Categorical columns: ", categorical_cols)

    for col in categorical_cols:

        features_df[col] = (
            features_df[col]
            .astype("string")
            .str.replace(".", "_", regex=False)
            .str.replace(" ", "_", regex=False)
        )

    categories = [
        features_df[col].dropna().unique()
        for col in categorical_cols
    ]

    # remove IDs before sklearn
    X = features_df.drop(columns=id_cols)

    ct = ColumnTransformer(
        transformers=[
            (
                'encoder',
                OneHotEncoder(
                    categories=categories,
                    handle_unknown='ignore',
                    sparse_output=False
                ),
                categorical_cols
            )
        ],
        remainder='passthrough'
    )

    encoded_array = ct.fit_transform(X)

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=ct.get_feature_names_out(),
        index=X.index
    )

    # clean column names
    encoded_df.columns = [
        c.split("__")[-1].split(".")[-1]
        for c in encoded_df.columns
    ]

    # re-add IDs (like original pipeline behavior)
    encoded_df["project.project_id"] = features_df["project.project_id"].values
    encoded_df["cases.case_id"] = features_df["cases.case_id"].values

    # reorder names
    priority_cols = [
        "project.project_id",
        "cases.case_id",
        "age_at_index",
        "tobacco_years",
        "pack_years_smoked"
    ]

    existing_priority = [
        c for c in priority_cols
        if c in encoded_df.columns
    ]

    other_cols = [
        c for c in encoded_df.columns
        if c not in existing_priority
    ]

    encoded_df = encoded_df[
        existing_priority + other_cols
    ]

    # CREATE FILE
    encoded_df.to_csv(
        out_file,
        sep="\t",
        index=False
    )

    print(f"Saved: {out_file}")
    print("Final shape:", encoded_df.shape)

def main():
    args = parse_args()
    build_features_considered(args.dataset)
    build_features_encoded(args.dataset)

if __name__ == "__main__":
    main()