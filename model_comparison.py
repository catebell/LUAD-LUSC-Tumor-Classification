import pandas as pd
from scipy import stats
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def compare_models(file1, file2, metrics):
    """
    Confronta due modelli usando test statistici su risultati di cross-validation.

    Args:
        file1 (str): CSV modello 1
        file2 (str): CSV modello 2
        metrics (list): metriche da confrontare
    """

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    model1 = df1["model"].iloc[0]
    model2 = df2["model"].iloc[0]

    logging.info(f"\nComparing models: {model1} vs {model2}")

    if len(df1) != len(df2):
        raise ValueError("I due file devono avere lo stesso numero di fold")

    for metric in metrics:

        values1 = df1[metric].values
        values2 = df2[metric].values

        logging.info(f"\nMetric: {metric}")
        logging.info(f"{model1} mean = {values1.mean():.4f}")
        logging.info(f"{model2} mean = {values2.mean():.4f}")

        # Paired t-test
        t_stat, t_p = stats.ttest_rel(values1, values2)
        logging.info(f"Paired t-test: t={t_stat:.4f}, p={t_p:.4f}")

        # Wilcoxon signed-rank test
        try:
            w_stat, w_p = stats.wilcoxon(values1, values2)
            logging.info(f"Wilcoxon test: stat={w_stat:.4f}, p={w_p:.4f}")
        except ValueError:
            logging.warning("Wilcoxon test not applicable (identical values)")


if __name__ == "__main__":

    file_model1 = "metrics/montecarlo_GAT_iteration_results.csv"
    file_model2 = "metrics/montecarlo_MultiModalGNN_iteration_results.csv"

    metrics = [
        "accuracy",
        "f1_score",
        "roc_auc",
        "precision",
        "recall",
        "auprc"
    ]

    compare_models(file_model1, file_model2, metrics)