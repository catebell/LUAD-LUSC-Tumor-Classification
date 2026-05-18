import os
import argparse
import config
import torch
import subprocess
import sys
import logging
from files_extraction_and_mapping import process_dataset
from STRING_files_to_tsv import create_gene_aliases_proteins_ids_mapping_file, create_genes_id_mapping_file
from preprocessing_clinical_features_to_file import build_features_considered, build_features_encoded
from train_test_val_patients_split import build_patient_split_cleaned
from methylation_manifest_to_tsv import create_meth_manifest

"""
Main script to run the project.
"""

def get_available_datasets():
    """Read folders inside {config.DATASET}/ to know the dataset available"""

    if not os.path.isdir(config.DATASET):
        return []

    datasets = []

    for folder in os.listdir(config.DATASET):

        full_path = os.path.join(config.DATASET, folder)

        if os.path.isdir(full_path):
            datasets.append(folder)

    return sorted(datasets)


def get_available_models():
    """Read python files inside models/ to select the model to use"""

    models_dir = "models"

    if not os.path.isdir(models_dir):
        return []

    models = []

    for file in os.listdir(models_dir):

        full_path = os.path.join(models_dir, file)

        if os.path.isfile(full_path) and file.endswith(".py"):

            if file == "__init__.py":
                continue

            model_name = os.path.splitext(file)[0]
            models.append(model_name)

    return sorted(models)


def build_paths(dataset_name):
    """Build the paths for the dataset and the files"""
    dataset_dir = os.path.join(config.DATASET, dataset_name)
    files_dir = os.path.join(config.FILES, dataset_name)

    return dataset_dir, files_dir


def parse_args():
    """Parse command line arguments"""

    datasets = get_available_datasets()
    models = get_available_models()

    parser = argparse.ArgumentParser(
        description="Main script for the project",
        usage="main.py [<args>] [-h | --help]"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=config.tumor,
        choices=datasets,
        help=f"Choose dataset: {', '.join(datasets)}"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=config.model,
        choices=models,
        help=f"Choose model: {', '.join(models)}" if models else "No models found"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate files even if already existing"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="kfold",
        choices=["kfold", "gridsearch", "montecarlo"],
        help="Choose execution mode"
    )

    return parser.parse_args()

def run_script(script_name, args_list):
    """Run a script with the given arguments"""
    cmd = [sys.executable, script_name] + args_list
    logging.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    args = parse_args()

    dataset_dir, files_dir = build_paths(args.dataset)

    print(f"\nSelected dataset: {args.dataset}")
    print(f"Input path : {dataset_dir}")
    print(f"Output path: {files_dir}")
    print(f"Force mode : {args.force}")

    ## Get the data

    print("\nFiles extraction and mapping:")
    process_dataset(dataset_dir, files_dir)

    print("\nCreate tsv files using the STRING downloaded files:")
    create_gene_aliases_proteins_ids_mapping_file(args.force)
    create_genes_id_mapping_file()

    print("\nCreate methylation manifest tsv file:")
    create_meth_manifest()

    ## Preprocessing

    print("\nPreprocessing features:")
    build_features_considered(args.dataset)
    build_features_encoded(args.dataset)

    print("\nSplit patients into train, val and test:")
    build_patient_split_cleaned(args.dataset)

    ## Run the model

    print("\nRunning the model:")
    if args.model_name:
        print(f"\nSelected model: {args.model_name}")

    if args.mode:
        print(f"\nSelected mode: {args.mode}")

    common_args = [
        "--dataset", args.dataset,
        "--model-name", args.model_name
    ]

    if args.mode == "kfold":
        run_script("k_folds_graph_classification.py", common_args)

    elif args.mode == "gridsearch":
        run_script("graph_classification_grid_search.py", common_args)

    elif args.mode == "montecarlo":
        run_script("montecarlo_graph_classification.py", common_args)

    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()