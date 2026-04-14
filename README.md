→ [Paper](https://drive.google.com/file/d/1Q4lXf_c4RhPrcQ3jVWPxu1M_9AghVad1/view?usp=drive_link)
- [click here](https://drive.google.com/file/d/1T_uj8j4KXscNW_96eOw5EVnfU75Fwzr9/view?usp=drive_link) to download our original dataset retrieved from GCD Data Portal
- [click here](https://drive.google.com/file/d/1mIzf35TOcNY9JKtysnPBZnZqJvtpP_q_/view?usp=drive_link) to download the three folders [train/val/test] of our processed graphs Dataset

## Architecture

<p align="center">
  <img src="images/architecture-light.png#gh-light-mode-only" width="100%" alt="Project Architecture">
  <img src="images/architecture-dark.png#gh-dark-mode-only" width="100%" alt="Project Architecture">
</p>
<p align="center">
  <strong>Figure 1.</strong> High-level overview of the project architecture.
</p>

## Explainability

[model_analysis_functions.py](model_analysis_functions.py) provides functions for the interpretation of the learned models decisional processes and what features they focus on:

- Graph branch attention scores (get_gene_attention_weights()) retrieves the attention scores given to each gene by the GAT-GNN layers, with higher scores to the genes that the model learned to monitor more closely.

```
2026-04-09 11:56:00,592 - INFO - Genes with attention importance = 1.000:
2026-04-09 11:56:00,737 - INFO - ENSG00000128422: 1.0000   ['PC2', 'PC2', 'PC2', 'PC2', 'Pc2', '39.1', 'CK-17', 'K17', 'KRT17', 'PC2', 'PCHC1']
2026-04-09 11:56:00,741 - INFO - ENSG00000119147: 1.0000   ['C2orf40', 'ECRG4']
2026-04-09 11:56:00,742 - INFO - ENSG00000119632: 1.0000   ['FAM14A', 'IFI27L2', 'ISG12B', 'TLH29']
2026-04-09 11:56:00,857 - INFO - ENSG00000124107: 1.0000   ['ALP', 'ALK1', 'BLPI', 'HUSI', 'HUSI-I', 'SLPI', 'WAP4', 'WFDC4', 'ALP']
2026-04-09 11:56:02,203 - INFO - ENSG00000162733: 1.0000   ['DDR2', 'MIG20a', 'NTRKR3', 'TYRO10', 'WRCN']
...
```

- Saliency (get_gene_saliency()) computes which genes influence the model's decision by having more relevance in the gradient computation.

```
2026-04-09 17:10:59,946 - INFO - Top 100 Genes saliency:
2026-04-09 17:10:59,950 - INFO - ENSG00000185201: 1.0000   ['1-8D', 'DSPA2c', 'IFITM2']
2026-04-09 17:10:59,951 - INFO - ENSG00000205420: 0.6538   ['CK-6C', 'CK-6E', 'K6C', 'KRT6C', 'PC3', 'CK-6C', 'CK-6E', 'CK6A', 'CK6C', 'CK6D', 'K6A', 'K6C', 'K6D', 'KRT6A', 'KRT6C', 'KRT6D', 'PC3']
2026-04-09 17:10:59,952 - INFO - ENSG00000011600: 0.5574   ['DAP12', 'KARAP', 'PLOSL', 'PLOSL1', 'TYROBP']
2026-04-09 17:10:59,953 - INFO - ENSG00000173599: 0.4210   ['PC', 'PC', 'PC', 'PC', 'PCB']
2026-04-09 17:10:59,954 - INFO - ENSG00000019582: 0.3975   ['CD74', 'CLIP', 'DHLAG', 'HLADG', 'Ia-GAMMA', 'CLIP', 'II', 'II', 'P33', 'p33']
2026-04-09 17:10:59,955 - INFO - ENSG00000186395: 0.3811   ['EHK', 'BCIE', 'BIE', 'CK10', 'K10', 'KPP', 'KRT10', 'EHK2']
2026-04-09 17:10:59,957 - INFO - ENSG00000171401: 0.3503   ['CK13', 'K13', 'KRT13', 'WSN2', 'K13']
2026-04-09 17:10:59,958 - INFO - ENSG00000186832: 0.3351   ['CK16', 'FNEPPK', 'K16', 'K1CP', 'KRT16', 'KRT16A']
2026-04-09 17:10:59,959 - INFO - ENSG00000186081: 0.3284   ['CK5', 'DDD', 'DDD1', 'EBS2', 'K5', 'KRT5', 'KRT5A']
...
```

→ Boxplots are employed in the visualization of the selected genes features values through the test patients dataset:

<p align="center">
  <img src="https://github.com/catebell/LUAD-LUSC-Tumor-Classification/blob/master/example1_model_with_analysis/analysis_plots/genes_expression_boxplot.png">
  <img src="https://github.com/catebell/LUAD-LUSC-Tumor-Classification/blob/master/example1_model_with_analysis/analysis_plots/genes_cnv_boxplot.png">
  <img src="https://github.com/catebell/LUAD-LUSC-Tumor-Classification/blob/master/example1_model_with_analysis/analysis_plots/genes_beta_value_boxplot.png">
</p>

- Clinical importance (explain_clinical_importance()) explains which clinical features, if any, most influence the prediction accuracy.

```
2026-04-01 15:12:34,071 - INFO - Clinical Features importance:
2026-04-01 15:12:34,071 - INFO - age_at_index: 0.0070
2026-04-01 15:12:34,071 - INFO - country_of_residence_at_enrollment: 0.0070
2026-04-01 15:12:34,071 - INFO - gender: 0.0070
2026-04-01 15:12:34,071 - INFO - ajcc_pathologic_n: 0.0070
2026-04-01 15:12:34,072 - INFO - tissue_or_organ_of_origin: 0.0070
2026-04-01 15:12:34,072 - INFO - ethnicity: 0.0000
2026-04-01 15:12:34,072 - INFO - race: 0.0000
...
```

## How to execute

### Get the data

1) The dataset must be downloaded from the desired source and saved in a folder named **original_dataset/**. For example, our data (click [here](https://drive.google.com/file/d/1T_uj8j4KXscNW_96eOw5EVnfU75Fwzr9/view?usp=drive_link) to download) was originally in this form:

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

2) Extract the correctly formatted files in a new **files/** folder by executing [files_extraction_and_mapping.py](files_extraction_and_mapping.py). This will be our new reference folder:

```
files/
    clinical/
        file_case_mapping.tsv
        omics_files.tsv
    CNV/
        extracted patients .tsv files
    methylation/
        extracted patients .txt files
    RNA/
        extracted patients .tsv files
```

3) We downloaded from the [STRING](https://string-db.org/cgi/download?sessionId=bUEKUGQV7g5H&species_text=Homo+sapiens&settings_expanded=0&min_download_score=0&filter_redundant_pairs=0&delimiter_type=txt) database the following files, used later on to retrieve genes properties and build the graphs based on their codified proteins (click to start the download):
- [9606.protein.aliases.v12.0.txt](https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz) 
- [9606.protein.links.v12.0.txt](https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz)

→ Put them in a new **STRING_downloaded_files/** folder and run [STRING_files_to_tsv.py](STRING_files_to_tsv.py): the first function creates **STRING_downloaded_files/9606.protein.aliases.gene.tsv**, the second one creates **files/clinical/gene_ids_mapped.tsv**.
> ⚠️ The execution of the first function can take a few hours.

4) We need also a methylation manifest for the preprocessing of methylation data; we downloaded from the relative [Illumina support page](https://support.illumina.com/downloads/infinium_humanmethylation450_product_files.html) the one relative to the Illumina “450 K array” technology (click [here](https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv) to start the download).

→ Put it in **methylation_manifests/originals_downloaded/** and run [methylation_manifest_to_tsv.py](methylation_manifest_to_tsv.py) to extract only the needed information correctly formatted.


### Preprocessing

1) Run [preprocessing_clinical_features_to_file.py](preprocessing_clinical_features_to_file.py) to obtain the following files:

```
files/
    clinical/
        features_considered.tsv  ←
        features_encoded.tsv  ←
        file_case_mapping.tsv
        gene_ids_mapped.tsv
        omics_files.tsv
```

2) Run [train_test_patients_split.py](train_test_patients_split.py) to assign to each patient (case_id) a label [*train, val, test*]:

```
files/
    clinical/
        features_considered.tsv
        features_encoded.tsv
        file_case_mapping.tsv
        gene_ids_mapped.tsv
        omics_files.tsv
        patient_split_cleaned.csv  ←
```

3) Run [graph_classification.py](graph_classification.py) to create the patients graphs Dataset (if the folders do not already exist). You will then see three new folders:
- **data_graphs_processed_test/**
- **data_graphs_processed_train/**
- **data_graphs_processed_validation/**
> ⚠️ The first execution can take a few hours. It will not start again unless the folders get deleted or renamed.
###### Click [here](https://drive.google.com/file/d/1mIzf35TOcNY9JKtysnPBZnZqJvtpP_q_/view?usp=drive_link) to download the three folders [train/val/test] of our processed graphs Dataset

## Try with different tumor classes

If needed, this model can be adapted to classify different tumor types (given the same kind of biological data).

In [preprocessing_clinical_features_to_file.py](preprocessing_clinical_features_to_file.py) change this mapping:
<p></p>

```
# for project.project_id, remap tumor class to 0-1
mapping = {
    'TCGA-LUAD': 0,
    'TCGA-LUSC': 1
}
features_df['project.project_id'] = features_df['project.project_id'].map(mapping)
```

#### If you want to use the default [models/MultiModalGNN](models/MultiModalGNN.py) and consider both graphs and clinical features: 
1) Change the content of [preprocessing_clinical_features_to_file.py](preprocessing_clinical_features_to_file.py) with respect to the clinical data you have.
2) Change the number of clinical features and/or the number of classes in every model initialization:
   <p></p>
   
   ```
   model = MultiModalGNN(num_node_features=5, num_edge_features=3, clinical_input_dim=53, hidden_channels=64, num_classes=2).to(device)
   ```

#### If you don't want to consider the clinical features, use only [models/GAT](models/GAT.py):
1) Change the model initialization in [graph_classification.py](graph_classification.py) and/or other files where it is needed:
   <p></p>
   
   ```
   → #model = GAT(num_node_features=5, num_edge_features=3, num_classes=2, hidden_channels=64).to(device)
   #model = MLP(num_patient_features=53, num_classes=2).to(device)
   → model = MultiModalGNN(num_node_features=5, num_edge_features=3, clinical_input_dim=53, hidden_channels=64, num_classes=2).to(device)
   ```
   ```
   → model = GAT(num_node_features=5, num_edge_features=3, num_classes=2, hidden_channels=64).to(device)
   #model = MLP(num_patient_features=53, num_classes=2).to(device)
   → #model = MultiModalGNN(num_node_features=5, num_edge_features=3, clinical_input_dim=53, hidden_channels=64, num_classes=2).to(device)
   ```

When the graph Dataset is created, class labels are automatically stored, so no need to modify [PatientGraphDataset.py](PatientGraphDataset.py):
```
data.y = torch.tensor([self.labels_dict[case_id]], dtype=torch.long)
```
