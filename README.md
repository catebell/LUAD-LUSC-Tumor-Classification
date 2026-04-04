Doc condiviso → https://docs.google.com/document/d/1N98eTXyuQqsBdpS_juzpIXrvbeDfqw0gvggBHOVsTjU/edit?usp=sharing
(Drive con Dataset)

## Execution and Generalizability

### Get the data

The desired dataset can be downloaded and saved in a folder named **original_datset/**. Our data came in this form:

```
original_dataset/
    clinical/
        clinical.tsv
        exposure.tsv
        LUAD_LUSC_metadata.json : file mapping, necessary to map exposure and clinical to CNV,RNA and methylation data (different file_id)
    CNV/
        722 patients folders
    methylation/
        758 patients folders
    RNA/
        757 patients folders
```

To extract the correctly formatted files in a new **files/** folder, start the script [files_extraction_and_mapping.py](files_extraction_and_mapping.py). This will be our new reference folder:

```
files/
    clinical/
        file_case_mapping.tsv
        file_case_with_project.tsv
        omics_files.tsv
    CNV/
        extracted patients .tsv files
    methylation/
        extracted patients .txt files
    RNA/
        extracted patients .tsv files
```

We downloaded from [STRING](https://string-db.org/cgi/download?sessionId=bUEKUGQV7g5H&species_text=Homo+sapiens&settings_expanded=0&min_download_score=0&filter_redundant_pairs=0&delimiter_type=txt) the following files (click to start the download):
- [9606.protein.aliases.v12.0.txt](https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz) 
- [9606.protein.links.v12.0.txt](https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz)

→ Put them in a new **STRING_downloaded_files/** folder and run [STRING_files_to_tsv.py](STRING_files_to_tsv.py): the first function creates **files/clinical/file_case_mapping.tsv**, the second one creates **STRING_downloaded_files/9606.protein.aliases.gene.tsv**.
> ⚠️ The execution of the second function takes a few hours, do it only once and then comment the call.

We need also a methylation manifest for the preprocessing of methylation data; we downloaded from [here](https://support.illumina.com/downloads/infinium_humanmethylation450_product_files.html) the one relative to the Illumina “450 K array” technology ([click to start the download](https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv)).

→ Put it in **methylation_manifests/originals_downloaded/** and run [methylation_manifest_to_tsv.py](methylation_manifest_to_tsv.py).

### Preprocessing

Run [preprocessing_clinical_features_to_file.py](preprocessing_clinical_features_to_file.py) to obtain the following files:

```
files/
    clinical/
        features_considered.tsv  ←
        features_encoded.tsv  ←
        file_case_mapping.tsv
        file_case_with_project.tsv
        file_case_mapping.tsv
        omics_files.tsv
```

Run [train_test_patients_split.py](train_test_patients_split.py) to assign to each patient (case_id) a label [*train, val, test*]:

```
files/
    clinical/
        features_considered.tsv
        features_encoded.tsv
        file_case_mapping.tsv
        file_case_with_project.tsv
        file_case_mapping.tsv
        omics_files.tsv
        patient_split_cleaned.csv  ←
```
Run [graph_classification.py](graph_classification.py) to create the patients graphs Dataset (if the folders do not already exist). You will then see three new folders:
- **data_graphs_processed_test/**
- **data_graphs_processed_train/**
- **data_graphs_processed_validation/**
> ⚠️ The first execution takes a few hours. It will not start a second time unless the folders get deleted or renamed.




## Architecture

<p align="center">
  <img src="images/architecture-light.png#gh-light-mode-only" width="100%" alt="Project Architecture">
  <img src="images/architecture-dark.png#gh-dark-mode-only" width="100%" alt="Project Architecture">
</p>

Overview dell'architettura del progetto.