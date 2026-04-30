import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import config

def get_available_datasets():
    """Return all subfolders inside original_dataset/ so the dataset available"""
    if not os.path.isdir(config.DATASET):
        return []

    return sorted([
        folder
        for folder in os.listdir(config.DATASET)
        if os.path.isdir(os.path.join(config.DATASET, folder))
    ])

def parse_args():
    """Parse the command line arguments"""
    datasets = get_available_datasets()

    parser = argparse.ArgumentParser(
        description="Patient split creation"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=config.tumor,
        choices=datasets,
        help=f"Available datasets: {', '.join(datasets)}"
    )

    return parser.parse_args()


def build_patient_split_cleaned(dataset):
    """Create patient_split_cleaned using only complete patients so with 3 omics"""

    input_mapping = os.path.join(
        config.FILES,
        dataset,
        "clinical",
        "file_case_mapping.tsv"
    )

    input_features = os.path.join(
        config.FILES,
        dataset,
        "clinical",
        "features_considered.tsv"
    )

    output_file = os.path.join(
        config.FILES,
        dataset,
        "clinical",
        "patient_split_cleaned.csv"
    )

    mapping_df = pd.read_csv(input_mapping, sep="\t")
    features_df = pd.read_csv(input_features, sep="\t")

    dup_counts = (
        mapping_df
        .groupby(["case_id", "omic"])
        .size()
        .reset_index(name="count")
    )

    patients_with_duplicates = dup_counts[
        dup_counts["count"] > 1
    ]["case_id"].unique()

    mapping_no_dup = mapping_df[
        ~mapping_df["case_id"].isin(patients_with_duplicates)
    ]

    omic_counts = mapping_no_dup.groupby("case_id")["omic"].nunique()

    complete_case_ids = omic_counts[
        omic_counts == 3
    ].index.tolist()

    print("Original unique patients in mapping:",
          mapping_df["case_id"].nunique())

    print("Patients removed due to duplicated omics:",
          len(patients_with_duplicates))

    print("Patients with exactly 3 omics (no duplicates):",
          len(complete_case_ids))

    print("Total removed patients:",
          mapping_df["case_id"].nunique() - len(complete_case_ids))

    filtered_patients = features_df[
        features_df["cases.case_id"].isin(complete_case_ids)
    ]

    filtered_patients = (
        filtered_patients[
            ["cases.case_id", "project.project_id"]
        ]
        .drop_duplicates()
    )

    print(
        "Count classes:\n",
        filtered_patients["project.project_id"].value_counts()
    )

    # Train+Val (80%) vs Test (20%)
    train_val, test = train_test_split(
        filtered_patients,
        test_size=0.20,
        stratify=filtered_patients["project.project_id"],
        random_state=7
    )

    # Train (70%) vs Val (10%)
    train, val = train_test_split(
        train_val,
        test_size=0.125,
        stratify=train_val["project.project_id"],
        random_state=7
    )

    filtered_patients["split"] = "none"

    filtered_patients.loc[
        filtered_patients["cases.case_id"].isin(train["cases.case_id"]),
        "split"
    ] = "train"

    filtered_patients.loc[
        filtered_patients["cases.case_id"].isin(val["cases.case_id"]),
        "split"
    ] = "val"

    filtered_patients.loc[
        filtered_patients["cases.case_id"].isin(test["cases.case_id"]),
        "split"
    ] = "test"

    filtered_patients = (
        filtered_patients
        .sort_values(by="split")
        .reset_index(drop=True)
    )

    filtered_patients.to_csv(output_file, index=False)

    stats = (
        filtered_patients
        .groupby(["split", "project.project_id"])
        .size()
        .unstack(fill_value=0)
    )

    print("Stats for split and project:\n", stats)
    print("Final number of patients:", len(filtered_patients))
    print("Saved:", output_file)


if __name__ == "__main__":
    build_patient_split_cleaned(parse_args().dataset)