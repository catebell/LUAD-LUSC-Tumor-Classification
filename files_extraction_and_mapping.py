import json
import os
import shutil

import pandas as pd

""" Extract files from dataset/ into files/ """

# EXTRACT_FILES

def safe_copy(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    base = os.path.basename(src)
    dst = os.path.join(dst_dir, base)
    if os.path.exists(dst):
        return

    shutil.copy2(src, dst)


def extract_files(base_dir, extension, out_dir):
    for root, dirs, files in os.walk(base_dir):

        dirs[:] = [d for d in dirs if d.lower() != "logs"]

        for file in files:
            if "annotation" in file.lower():
                continue

            if file.lower().endswith(extension):
                src_path = os.path.join(root, file)
                safe_copy(src_path, out_dir)


def files_extraction():
    """Extract all files from the directories CNV, RNA and methylation -> put in files"""

    ROOT_DIR = "dataset"
    OUTPUT_DIR = "files"
    TSV_EXT = ".tsv"
    TXT_EXT = ".txt"

    paths = {
        "CNV": TSV_EXT,
        "RNA": TSV_EXT,
        "methylation": TXT_EXT
    }

    for folder, ext in paths.items():
        base_path = os.path.join(ROOT_DIR, folder)
        out_path = os.path.join(OUTPUT_DIR, folder)

        if os.path.isdir(base_path):
            extract_files(base_path, ext, out_path)
        else:
            print(f"Directory not found: {base_path}")

    print("Completed")



# EXTRACT_FILE_ID

def extract_file_id():
    """Extract file_id from filename in files -> put in omics_files.tsv"""

    BASE_DIR = "files"
    OMICS = ["CNV", "RNA", "methylation"]

    rows = []

    for omic in OMICS:
        omic_dir = os.path.join(BASE_DIR, omic)

        for root, _, files in os.walk(omic_dir):
            for fname in files:

                if "." not in fname:
                    continue

                # CNV: TCGA-LUAD/LUSC.<file_id>.*
                if omic == "CNV":
                    first_part = fname.split(".", 2)[1]
                    file_id = first_part

                # RNA / methylation: <file_id>.*
                else:
                    file_id = fname.split(".", 1)[0]

                rows.append({
                    "omic": omic,
                    "filename": fname,
                    "file_id": file_id,
                    "path": os.path.join(root, fname)
                })

    df = pd.DataFrame(rows)

    print(df.head())

    for omic in OMICS:
        subset = df[df["omic"] == omic]
        print(f"\n=== {omic} ===")
        print(subset.iloc[0])

    output_file = "files\\clinical\\omics_files.tsv"
    df.to_csv(output_file, sep="\t", index=False)

    print(f"\nFile TSV created: {output_file}")
    print(df.head())
    print(df.shape)
    print(df.isnull().sum().sum())



# EXTRACT_FILE_ID_CASE_ID

def extract_file_id_case_id():
    """Search the file_id in the LUAD_LUSC_metadata.json file and take the corresponding case_id -> put in file_case_mapping.tsv"""

    tsv_file = r"files/clinical/omics_files.tsv"
    json_file = r"dataset/clinical/LUAD_LUSC_metadata.json"

    df = pd.read_csv(tsv_file, sep="\t")

    with open(json_file, "r") as f:
        json_data = json.load(f)

    # CNV: entity_id -> case_id
    entity_to_case = {}
    # RNA / Methylation: file_name -> case_id
    file_name_to_case = {}

    for entry in json_data:
        data_category = entry.get("data_category", "").lower()

        # CNV
        if data_category == "copy number variation":
            for entity in entry.get("associated_entities", []):
                entity_id = entity.get("entity_id")
                case_id = entity.get("case_id")
                if entity_id and case_id:
                    entity_to_case[entity_id] = case_id

        # RNA / Methylation
        elif data_category in ["transcriptome profiling", "dna methylation"]:
            file_name = entry.get("file_name")
            case_id = None
            if entry.get("associated_entities"):
                case_id = entry["associated_entities"][0].get("case_id")
            if file_name and case_id:
                file_name_to_case[file_name] = case_id

    # Associate with case_id
    def get_case_id(row):
        omic_type = row['omic'].lower()
        if omic_type == "cnv":
            return entity_to_case.get(row['file_id'])
        else:
            # RNA / Methylation → mapping with filename
            return file_name_to_case.get(row['filename'])

    df["case_id"] = df.apply(get_case_id, axis=1)

    df_out = df[["file_id", "case_id", "omic", "filename"]]
    df_out.to_csv(r"files/clinical/file_case_mapping.tsv", sep="\t", index=False)

    print(df_out.head())
    print(df_out.isnull().sum())
    print("\nRows with null values:")
    print(df_out[df_out.isnull().any(axis=1)])



# EXTRACT_PROJECT_ID

def extract_project_id():
    """Search the case_id (cases.case_id) in features.tsv and take the corresponding project.project_id (LUAD/LUSC label) -> put in file_case_with_project.tsv"""

    file_mapping = pd.read_csv(r"files/clinical/file_case_mapping.tsv", sep="\t")
    clinical = pd.read_csv(r"files/clinical/features.tsv", sep="\t")

    merged = pd.merge(
        clinical,
        file_mapping,
        left_on='cases.case_id',
        right_on='case_id',
        how='left'
    )

    merged = merged.drop(columns=['cases.case_id'])
    merged = merged.rename(columns={'project.project_id': 'project_id'})
    merged = merged.sort_values(by="project_id").reset_index(drop=True)

    final_df = merged.to_csv(
        r"files/clinical/file_case_with_project.tsv",
        sep="\t",
        index=False
    )

    print("File created: file_case_with_project.tsv")
    print(merged.head())
    print(merged.shape)
    print(merged.isnull().sum())
    print("\nRows with null values:")
    print(merged[merged.isnull().any(axis=1)])



# MAIN

def main():
    files_extraction()
    extract_file_id()
    extract_file_id_case_id()
    extract_project_id()

if __name__ == "__main__":
    main()
