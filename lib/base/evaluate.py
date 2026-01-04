import os
import numpy as np
import pandas as pd
from lib.evaluate.visuals import VisualEvaluator
from lib.evaluate.metrics import MetricsEvaluator
from lib.logging import logger

logger = logger.get()

class BaselineEvaluator:
    def __init__(self, config: dict):
        quant = dict(config.evaluators.get("quantitative", {}))
        quant["n_outputs"] = config.experiment.model.n_outputs
        self.metrics = MetricsEvaluator(quant)

        qual = config.get("qualitative", {})
        self.visualizer = VisualEvaluator(qual)
        self.output_dir = qual.get("output_dir", "./evaluation_plots")
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_all(self, wrapper):
        rows = []

        # Single-subject
        for subj, runs in wrapper.results_by_experiment["single"].items():
            per_run_metrics = []
            for run_res in runs:
                gt = np.array(run_res["ground_truth"])
                pr = np.array(run_res["predictions"])
                prb = run_res.get("probabilities", None)
                per_run_metrics.append(self.metrics.evaluate(gt, pr, prb))

            for i, m in enumerate(per_run_metrics, start=1):
                rows.append({
                    "mode": "single",
                    "subject": subj,
                    "run": i,
                    **m
                })

            agg = {}
            for k, v0 in per_run_metrics[0].items():
                vals = [m[k] for m in per_run_metrics]
                if isinstance(v0, (int, float, np.floating)):
                    agg[k] = float(np.mean(vals))
                else:
                    agg[k] = v0
            rows.append({
                "mode":   "single",
                "subject": subj,
                "run":    "mean",
                **agg
            })

            last = runs[-1]
            gt   = np.array(last["ground_truth"])
            pr   = np.array(last["predictions"])
            self.visualizer.plot_confusion_matrix(
                gt,
                pr,
                filename=f"cm_subj_{subj}.png"
            )

        # Pooled
        pooled_runs = wrapper.results_by_experiment["pooled"]
        per_run_metrics = []
        for run_res in pooled_runs:
            gt  = np.array(run_res["ground_truth"])
            pr  = np.array(run_res["predictions"])
            prb = run_res.get("probabilities", None)
            per_run_metrics.append(self.metrics.evaluate(gt, pr, prb))

        for i, m in enumerate(per_run_metrics, start=1):
            rows.append({
                "mode":   "pooled",
                "subject":"all",
                "run":    i,
                **m
            })

        agg = {}
        for k, v0 in per_run_metrics[0].items():
            vals = [m[k] for m in per_run_metrics]
            if isinstance(v0, (int, float, np.floating)):
                agg[k] = float(np.mean(vals))
            else:
                agg[k] = v0
        rows.append({
            "mode":   "pooled",
            "subject":"all",
            "run":    "mean",
            **agg
        })

        last = pooled_runs[-1]
        gt   = np.array(last["ground_truth"])
        pr   = np.array(last["predictions"])
        self.visualizer.plot_confusion_matrix(
            gt,
            pr,
            filename="cm_pooled.png"
        )

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, "baseline_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info("\n%s", df.to_markdown(index=False))

