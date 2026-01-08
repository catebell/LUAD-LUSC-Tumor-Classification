import pandas as pd
import json

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