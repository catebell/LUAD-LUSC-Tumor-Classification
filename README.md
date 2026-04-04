Doc condiviso → https://docs.google.com/document/d/1N98eTXyuQqsBdpS_juzpIXrvbeDfqw0gvggBHOVsTjU/edit?usp=sharing
(Drive con Dataset)

[Execution and Generalizability]

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
    CNV/
        extracted patients .tsv files
    methylation/
        extracted patients .txt files
    RNA/
        extracted patients .tsv files
```
