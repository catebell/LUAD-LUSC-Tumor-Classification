import os
import glob
import itertools
import logging
from pathlib import Path
import config

import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = "TestsModels"

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc_roc"
]


def load_model_csv(csv_path):
    """
    Load CSV and remove MEAN / STD_DEV rows
    """

    df = pd.read_csv(csv_path)

    df = df[df["fold"].astype(str).str.isnumeric()].copy()

    df["fold"] = df["fold"].astype(int)

    return df


def compare_models(model1_name, df1, model2_name, df2):
    """
    Compare two models on all metrics
    """

    results = []

    logging.info(f"\nComparing {model1_name} vs {model2_name}")

    merged = pd.merge(
        df1,
        df2,
        on="fold",
        suffixes=("_m1", "_m2")
    )

    for metric in METRICS:

        values1 = merged[f"{metric}_m1"].values
        values2 = merged[f"{metric}_m2"].values

        logging.info(f"\nMetric: {metric}")
        logging.info(f"{model1_name} mean = {values1.mean():.4f}")
        logging.info(f"{model2_name} mean = {values2.mean():.4f}")

        metric_result = {
            "model_1": model1_name,
            "model_2": model2_name,
            "metric": metric,
            "mean_model_1": values1.mean(),
            "mean_model_2": values2.mean(),
        }

        # Paired t-test
        try:
            t_stat, t_p = stats.ttest_rel(values1, values2)

            metric_result["paired_t_stat"] = t_stat
            metric_result["paired_t_p"] = t_p

            logging.info(
                f"Paired t-test: t={t_stat:.4f}, p={t_p:.4f}"
            )

        except Exception:
            metric_result["paired_t_stat"] = None
            metric_result["paired_t_p"] = None

        # Welch t-test
        try:
            w_stat, w_p = stats.ttest_ind(
                values1,
                values2,
                equal_var=False
            )

            metric_result["welch_t_stat"] = w_stat
            metric_result["welch_t_p"] = w_p

            logging.info(
                f"Welch t-test: t={w_stat:.4f}, p={w_p:.4f}"
            )

        except Exception:
            metric_result["welch_t_stat"] = None
            metric_result["welch_t_p"] = None

        # Wilcoxon
        try:
            wil_stat, wil_p = stats.wilcoxon(values1, values2)

            metric_result["wilcoxon_stat"] = wil_stat
            metric_result["wilcoxon_p"] = wil_p

            logging.info(
                f"Wilcoxon: stat={wil_stat:.4f}, p={wil_p:.4f}"
            )

        except Exception:
            metric_result["wilcoxon_stat"] = None
            metric_result["wilcoxon_p"] = None

        # Mann-Whitney
        try:
            mw_stat, mw_p = stats.mannwhitneyu(
                values1,
                values2,
                alternative="two-sided"
            )

            metric_result["mannwhitney_stat"] = mw_stat
            metric_result["mannwhitney_p"] = mw_p

            logging.info(
                f"Mann-Whitney: U={mw_stat:.4f}, p={mw_p:.4f}"
            )

        except Exception:
            metric_result["mannwhitney_stat"] = None
            metric_result["mannwhitney_p"] = None

        results.append(metric_result)

    return results


def main():

    csv_files = glob.glob(
        os.path.join(BASE_DIR, "*", "*.csv")
    )

    if len(csv_files) < 2:
        logging.error("Need at least 2 CSV")
        return

    models = {}

    for csv_path in csv_files:

        model_name = os.path.basename(
            os.path.dirname(csv_path)
        )

        try:
            df = load_model_csv(csv_path)
            models[model_name] = df

            logging.info(f"Loaded {model_name}")

        except Exception as e:
            logging.warning(
                f"Error in loading {csv_path}: {e}"
            )

    all_results = []

    for model1, model2 in itertools.combinations(models.keys(), 2):

        results = compare_models(
            model1,
            models[model1],
            model2,
            models[model2]
        )

        all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    output_csv = "statistical_tests_results.csv"

    results_df.to_csv(output_csv, index=False)

    logging.info(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    main()