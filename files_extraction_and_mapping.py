import json
import glob
import os
import shutil
import pandas as pd
import config

""" Extract files from original_dataset/ into files/ """

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

def count_partial_files(base_dir):
    """Count all files with extension .partial in the subfolders of the dataset"""

    print("\nCheck the presence of incomplete files (.partial)")
    partial_files = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".partial"):
                partial_files.append(os.path.join(root, file))

    if partial_files:
        print(f"Attention: Found {len(partial_files)} incomplete files!")
    else:
        print("No file .partial found")

    return len(partial_files)


def files_extraction(INPUT_DIR, OUTPUT_DIR):
    """Extract all files from the directories CNV, RNA and methylation -> put in files"""

    TSV_EXT = ".tsv"
    TXT_EXT = ".txt"

    paths = {
        "CNV": TSV_EXT,
        "RNA": TSV_EXT,
        "methylation": TXT_EXT
    }

    for folder, ext in paths.items():
        base_path = os.path.join(INPUT_DIR, folder)
        out_path = os.path.join(OUTPUT_DIR, folder)

        if os.path.isdir(base_path):
            extract_files(base_path, ext, out_path)
        else:
            print(f"Directory not found: {base_path}")

    print("Completed")


# EXTRACT_FILE_ID

def extract_file_id(FILES_DIR):
    """Extract file_id from filename in files -> put in omics_files.tsv"""

    # BASE_DIR = "files"
    OMICS = ["CNV", "RNA", "methylation"]

    rows = []

    for omic in OMICS:
        omic_dir = os.path.join(FILES_DIR, omic)

        for root, _, files in os.walk(omic_dir):
            for fname in files:

                if "." not in fname:
                    continue

                if omic == "CNV":
                    if "ascat" not in fname.lower():
                        continue
                    if fname.startswith("TCGA"):
                        # CNV: TCGA-<class>.<file_id>.*
                        file_id = fname.split(".", 2)[1]
                    else:
                        # CNV: <file_id>.wgs.ASCAT*
                        file_id = fname.split(".", 1)[0]

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

    output_file = os.path.join(FILES_DIR, "clinical", "omics_files.tsv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, sep="\t", index=False)

    print(f"\nFile TSV created: {output_file}")
    print(df.head())
    print(df.shape)
    print(df.isnull().sum().sum())

# EXTRACT_FILE_ID_CASE_ID

def extract_file_id_case_id(DATASET_DIR, FILES_DIR):
    """Search the file_id in the metadata.json file and take the corresponding case_id -> put in file_case_mapping.tsv"""

    tsv_file = os.path.join(FILES_DIR, "clinical", "omics_files.tsv")

    json_files = glob.glob(os.path.join(DATASET_DIR, "clinical", "*.json"))

    if len(json_files) != 1:
        raise ValueError(f"Expected 1 JSON file, found {len(json_files)}")

    json_file = json_files[0]

    print(f"Using json file: {json_file}")

    df = pd.read_csv(tsv_file, sep="\t")

    with open(json_file, "r") as f:
        json_data = json.load(f)

    entity_to_case    = {} # CNV: entity_id -> case_id
    file_id_to_case   = {} # CNV: file_id   -> case_id
    file_name_to_case = {} # RNA / Methylation: file_name -> case_id

    for entry in json_data:
        data_category = entry.get("data_category", "").lower()
        associated    = entry.get("associated_entities", [])

        # CNV
        if data_category == "copy number variation":
            file_name = entry.get("file_name")
            file_id   = entry.get("file_id")

            if associated:
                case_id = associated[0].get("case_id")

                if file_name and case_id:
                    file_name_to_case[file_name] = case_id

                if file_id and case_id:
                    file_id_to_case[file_id] = case_id

            for entity in associated:
                entity_id = entity.get("entity_id")
                case_id   = entity.get("case_id")

                if entity_id and case_id:
                    entity_to_case[entity_id] = case_id

        # RNA / Methylation
        elif data_category in ["transcriptome profiling", "dna methylation"]:
            file_name = entry.get("file_name")

            case_id = None
            if associated:
                case_id = associated[0].get("case_id")

            if file_name and case_id:
                file_name_to_case[file_name] = case_id

    # Associate with case_id
    def get_case_id(row):
        omic_type = row["omic"].lower()

        if omic_type == "cnv":
            return (
                file_name_to_case.get(row["filename"]) or
                file_id_to_case.get(row["file_id"]) or
                entity_to_case.get(row["file_id"])
            )
        else:
            return file_name_to_case.get(row["filename"])

    df["case_id"] = df.apply(get_case_id, axis=1)

    df_out = df[["file_id", "case_id", "omic", "filename"]]

    print("\n=== Null check per columns ===")
    print(df_out.isnull().sum())

    null_rows = df_out[df_out.isnull().any(axis=1)]
    if not null_rows.empty:
        print("\nRows with null values:")
        print(null_rows.head())

    out_file = os.path.join(FILES_DIR, "clinical", "file_case_mapping.tsv")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    df_out.to_csv(out_file, sep="\t", index=False)

    print(f"\nFile TSV created: {out_file}")
    print(df_out.head())
    print(df_out.shape)

def process_dataset(dataset_dir, files_dir):
    num_partial = count_partial_files(dataset_dir)

    if num_partial > 0:
        print(f"Skipped {dataset_dir}")
        return

    files_extraction(INPUT_DIR=dataset_dir, OUTPUT_DIR=files_dir)
    extract_file_id(FILES_DIR=files_dir)
    extract_file_id_case_id(FILES_DIR=files_dir, DATASET_DIR=dataset_dir)

def main():
    print("\n=== LUNG ===")
    process_dataset(config.DATASET_LUNG, config.FILES_LUNG)

    print("\n=== KIDNEY ===")
    process_dataset(config.DATASET_KIDNEY, config.FILES_KIDNEY)

if __name__ == "__main__":
    main()
