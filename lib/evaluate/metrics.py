import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)


class MetricsEvaluator:
    def __init__(self, config: dict):
        self.metrics = config.get("metrics", [])
        n_out = config.get("n_outputs", None)
        if n_out is None:
            raise ValueError("MetricsEvaluator requires config['n_outputs']")
        
        self.labels = list(range(n_out))

    def evaluate(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray = None
    ) -> dict:

        results = {}
        gt = ground_truth
        pr = predictions

        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy_score(gt, pr)

        # Only if >=2 classes present
        if "kappa" in self.metrics:
            unique_labels = set(gt) | set(pr)
            if len(unique_labels) < 2:
                results["kappa"] = 0.0
            else:
                results["kappa"] = cohen_kappa_score(
                    gt, pr, labels=self.labels
                )

        if "precision" in self.metrics:
            results["precision"] = precision_score(
                gt, pr, average="macro", zero_division=0
            )

        if "recall" in self.metrics:
            results["recall"] = recall_score(
                gt, pr, average="macro", zero_division=0
            )

        if "f1_score" in self.metrics:
            results["f1_score"] = f1_score(
                gt, pr, average="macro", zero_division=0
            )

        if "confusion_matrix" in self.metrics:
            results["confusion_matrix"] = confusion_matrix(
                gt, pr, labels=self.labels
            )

        if "roc_curve" in self.metrics and probabilities is not None:
            roc_results = {}
            for cls in self.labels:
                binary_truth = (gt == cls).astype(int)
                fpr, tpr, thresh = roc_curve(binary_truth, probabilities[:, cls])
                roc_results[cls] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresh,
                    "auc": auc(fpr, tpr),
                }
            results["roc_curve"] = roc_results

        return results
