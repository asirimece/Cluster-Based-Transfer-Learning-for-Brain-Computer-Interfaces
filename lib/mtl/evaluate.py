
import os
import pickle
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from lib.base.train import BaseWrapper
from lib.evaluate.metrics import MetricsEvaluator
from lib.evaluate.visuals import VisualEvaluator
from lib.logging import logger
import matplotlib.pyplot as plt
import seaborn as sns

logger = logger.get()


class MTLEvaluator:
    def __init__(self, mtl_wrapper, experiment_cfg):
        if not isinstance(experiment_cfg, DictConfig):
            experiment_cfg = OmegaConf.create(experiment_cfg)
        # unwrap nested 'experiment' keys until we find 'evaluators'
        while isinstance(experiment_cfg, DictConfig) \
          and 'evaluators' not in experiment_cfg \
          and 'experiment' in experiment_cfg:
            experiment_cfg = experiment_cfg.experiment
        if not isinstance(experiment_cfg, DictConfig) or 'evaluators' not in experiment_cfg:
            raise ValueError(
                "MTLEvaluator expected config containing 'evaluators'; got:\n"
                f"{OmegaConf.to_yaml(experiment_cfg)}"
            )

        self.experiment_cfg = experiment_cfg
        self.mtl_wrapper    = mtl_wrapper

        # quantitative metrics
        n_out = self.experiment_cfg.mtl.model.n_outputs
        qc = self.experiment_cfg.evaluators.quantitative
        self.metrics         = MetricsEvaluator({'metrics': qc.metrics, 'n_outputs': n_out})
        self.cluster_metrics = MetricsEvaluator({'metrics': qc.cluster_metrics, 'n_outputs': n_out})

        # qualitative visuals, include full label set so confusion_matrix is always N×N
        qv = self.experiment_cfg.evaluators.qualitative
        out_dir = qv.get('output_dir', self.experiment_cfg.evaluators.mtl_output_dir)
        full_labels = list(range(n_out))
        self.visuals = VisualEvaluator({
            'visualizations':   qv.visualizations,
            'pca_n_components': qv.pca_n_components,
            'tsne':             qv.tsne,
            'output_dir':       out_dir,
            'labels':           full_labels,
        })
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # baseline wrappers
        base_cfg = OmegaConf.load('config/experiment/base.yaml')
        single_wrapped = pickle.load(open(base_cfg.logging.single_results_path, 'rb'))
        pooled_wrapped = pickle.load(open(base_cfg.logging.pooled_results_path, 'rb'))
        self.baseline_wrapper = BaseWrapper({
            'single': single_wrapped,
            'pooled': pooled_wrapped
        })

        logger.info("MTLEvaluator initialized.")

        # load subject representations
        feat_file = self.experiment_cfg.features_file
        with open(feat_file, "rb") as f:
            all_feats = pickle.load(f)

        self.subject_reprs = {}
        for subj, data in all_feats.items():
            if isinstance(data, dict):
                reps = []
                for split_data in data.values():
                    if isinstance(split_data, dict) and "combined" in split_data:
                        reps.append(split_data["combined"])
                if reps:
                    all_data = np.concatenate(reps, axis=0)
                    self.subject_reprs[subj] = np.mean(all_data, axis=0)
            elif isinstance(data, np.ndarray):
                self.subject_reprs[subj] = np.mean(data, axis=0)
            else:
                raise TypeError(f"Unexpected type for subject {subj}: {type(data)}")

    def evaluate(self):
        # Skip expensive global t-SNE if too many total samples
        if "tsne" in self.visuals.visualizations:
            total_samples = sum(len(runs) for runs in self.mtl_wrapper.results_by_subject.values())
            if total_samples > 500:
                logger.info(f"MTLEvaluator: skipping t-SNE on {total_samples} samples (>500).")
                self.visuals.visualizations.remove("tsne")

        n_out = self.experiment_cfg.mtl.model.n_outputs
        full_labels = list(range(n_out))
        viz_list = self.experiment_cfg.evaluators.qualitative.visualizations

        from sklearn.metrics import confusion_matrix, cohen_kappa_score

        def safe_cm_kappa(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred, labels=full_labels)
            if len(set(y_true) | set(y_pred)) < 2:
                kappa = 0.0
            else:
                kappa = cohen_kappa_score(y_true, y_pred, labels=full_labels)
            return cm, kappa

        # Subject level
        subj_run = []
        for subj, runs in self.mtl_wrapper.results_by_subject.items():
            for ridx, run in enumerate(runs):
                gt = np.array(run['ground_truth'])
                pr = np.array(run['predictions'])

                m = self.metrics.evaluate(gt, pr)
                _, kappa = safe_cm_kappa(gt, pr)
                m.update({'kappa': kappa})

                row = {'run': ridx, 'subject': subj}
                row.update(m)
                subj_run.append(row)

                if ridx == 0 and 'confusion_matrix' in viz_list:
                    self.visuals.plot_confusion_matrix(
                        gt, pr,
                        labels=full_labels,
                        filename=f"cm_mtl_subj_{subj}.png" 
                    )

        df_subj_run = pd.DataFrame(subj_run)
        df_subj_run.to_csv(
            os.path.join(self.out_dir, "mtl_subject_run_metrics.csv"),
            index=False
        )
        subj_stats = (
            df_subj_run.groupby('subject')
            .agg(['mean','std'])
        )
        subj_stats.columns = [f"{met}_{st}" for met, st in subj_stats.columns]
        subj_stats = subj_stats.reset_index()
        subj_stats['cluster'] = subj_stats['subject'].map(self.mtl_wrapper.cluster_assignments)
        cols = ['subject','cluster'] + [c for c in subj_stats.columns if c not in ('subject','cluster')]
        subj_stats = subj_stats[cols]
        subj_stats.to_csv(
            os.path.join(self.out_dir, "mtl_subject_stats_metrics.csv"),
            index=False
        )

        cl_run = []
        max_runs = max(len(runs) for runs in self.mtl_wrapper.results_by_subject.values())
        for ridx in range(max_runs):
            for subj, runs in self.mtl_wrapper.results_by_subject.items():
                if ridx < len(runs):
                    gt = np.array(runs[ridx]['ground_truth'])
                    pr = np.array(runs[ridx]['predictions'])

                    m = self.cluster_metrics.evaluate(gt, pr)
                    _, kappa = safe_cm_kappa(gt, pr)
                    m.update({'kappa': kappa})

                    cl = self.mtl_wrapper.cluster_assignments.get(subj, "None")
                    row = {'run': ridx, 'cluster': cl}
                    row.update(m)
                    cl_run.append(row)

                    if ridx == 0 and 'confusion_matrix' in viz_list:
                        self.visuals.plot_confusion_matrix(
                            gt, pr,
                            labels=full_labels,
                            filename=f"cm_mtl_cluster_{cl}.png" 
                        )

        df_cl_run = pd.DataFrame(cl_run)
        df_cl_run.to_csv(
            os.path.join(self.out_dir, "mtl_cluster_run_metrics.csv"),
            index=False
        )
        cl_stats = (
            df_cl_run.groupby('cluster')
            .agg(['mean','std'])
        )
        cl_stats.columns = [f"{met}_{st}" for met, st in cl_stats.columns]
        cl_stats = cl_stats.reset_index()
        cl_stats.to_csv(
            os.path.join(self.out_dir, "mtl_cluster_stats_metrics.csv"),
            index=False
        )

        # Pooled level
        pool_run = []
        for run_idx in range(max_runs):
            all_gt, all_pr = [], []
            for runs in self.mtl_wrapper.results_by_subject.values():
                if run_idx < len(runs):
                    all_gt.extend(runs[run_idx]['ground_truth'])
                    all_pr.extend(runs[run_idx]['predictions'])
            gt = np.array(all_gt)
            pr = np.array(all_pr)

            m = self.metrics.evaluate(gt, pr)
            _, kappa = safe_cm_kappa(gt, pr)
            m.update({'kappa': kappa})

            row = {'run': run_idx}
            row.update(m)
            pool_run.append(row)

            if 'confusion_matrix' in viz_list:
                self.visuals.plot_confusion_matrix(
                    gt, pr,
                    labels=full_labels,
                    filename=f"cm_mtl_pooled_run{run_idx}.png"
                )

        df_pool_run = pd.DataFrame(pool_run)
        df_pool_run.to_csv(
            os.path.join(self.out_dir, 'mtl_pooled_run_metrics.csv'),
            index=False
        )
        pool_stats = df_pool_run.drop(columns=['run']).agg(['mean','std']).T
        pool_stats.columns = ['mean','std']
        pool_stats = pool_stats.reset_index().rename(columns={'index':'metric'})
        pool_stats.to_csv(
            os.path.join(self.out_dir, 'mtl_pooled_stats_metrics.csv'),
            index=False
        )

        # Baseline cluster level
        base_cl_run = []
        base_single = self.baseline_wrapper.get_experiment_results('single')
        max_base = max(len(r) for r in base_single.values())
        for run_idx in range(max_base):
            for subj, runs in base_single.items():
                if run_idx < len(runs):
                    run = runs[run_idx]
                    gt, pr = map(np.array, (run['ground_truth'], run['predictions']))
                    cl = self.mtl_wrapper.cluster_assignments.get(subj, 'None')
                    m  = self.cluster_metrics.evaluate(gt, pr)
                    row = {'run': run_idx, 'cluster': cl}
                    row.update(m)
                    base_cl_run.append(row)
                    if 'confusion_matrix' in viz_list:
                        self.visuals.plot_confusion_matrix(
                            gt, pr,
                            filename=f"cm_base_cluster_{cl}_run{run_idx}.png"
                        )
        df_base_cl_run = pd.DataFrame(base_cl_run)
        df_base_cl_run.to_csv(os.path.join(self.out_dir, 'baseline_cluster_run_metrics.csv'), index=False)

        base_cl_stats = df_base_cl_run.groupby('cluster').agg(['mean','std'])
        base_cl_stats.columns = [f"{met}_{st}" for met, st in base_cl_stats.columns]
        base_cl_stats = base_cl_stats.reset_index()
        base_cl_stats.to_csv(os.path.join(self.out_dir, 'baseline_cluster_stats_metrics.csv'), index=False)

        # Baseline single subject
        base_subj_run = []
        for subj, runs in self.baseline_wrapper.get_experiment_results('single').items():
            for run_idx, run in enumerate(runs):
                gt, pr = map(np.array, (run['ground_truth'], run['predictions']))
                m  = self.metrics.evaluate(gt, pr)
                row = {'run': run_idx, 'subject': subj}
                row.update(m)
                base_subj_run.append(row)
        df_base_subj_run = pd.DataFrame(base_subj_run)
        df_base_subj_run.to_csv(os.path.join(self.out_dir, 'baseline_subject_run_metrics.csv'), index=False)

        # Baseline pooled
        base_pool_run = []
        for run_idx, run in enumerate(self.baseline_wrapper.get_experiment_results('pooled')):
            gt, pr = map(np.array, (run['ground_truth'], run['predictions']))
            m  = self.metrics.evaluate(gt, pr)
            row = {'run': run_idx}
            row.update(m)
            base_pool_run.append(row)
        df_base_pool_run = pd.DataFrame(base_pool_run)
        df_base_pool_run.to_csv(os.path.join(self.out_dir, 'baseline_pooled_run_metrics.csv'), index=False)

        # Comparison
        df_cmp_subj = df_subj_run.merge(df_base_subj_run,
                                        on=['subject','run'],
                                        suffixes=('_mtl','_base'))
        for metric in self.experiment_cfg.evaluators.quantitative.metrics:
            df_cmp_subj[f'{metric}_delta'] = (
                df_cmp_subj[f'{metric}_mtl'] - df_cmp_subj[f'{metric}_base']
            )
        df_cmp_subj.to_csv(os.path.join(self.out_dir, 'mtl_vs_baseline_subject_run.csv'), index=False)

        df_cmp_cl = df_cl_run.merge(df_base_cl_run,
                                    on=['cluster','run'],
                                    suffixes=('_mtl','_base'))
        for metric in self.experiment_cfg.evaluators.quantitative.cluster_metrics:
            df_cmp_cl[f'{metric}_delta'] = (
                df_cmp_cl[f'{metric}_mtl'] - df_cmp_cl[f'{metric}_base']
            )
        df_cmp_cl.to_csv(os.path.join(self.out_dir, 'mtl_vs_baseline_cluster_run.csv'), index=False)

        df_cmp_pool = df_pool_run.merge(df_base_pool_run,
                                        on='run',
                                        suffixes=('_mtl','_base'))
        for metric in self.experiment_cfg.evaluators.quantitative.metrics:
            df_cmp_pool[f'{metric}_delta'] = (
                df_cmp_pool[f'{metric}_mtl'] - df_cmp_pool[f'{metric}_base']
            )
        df_cmp_pool.to_csv(os.path.join(self.out_dir, 'mtl_vs_baseline_pooled_run.csv'), index=False)

        plt.figure(figsize=(6,4))
        plt.plot(df_cmp_pool['run'], df_cmp_pool['accuracy_mtl'],
                 marker='o', label='MTL', color='blue')
        plt.plot(df_cmp_pool['run'], df_cmp_pool['accuracy_base'],
                 marker='s', label='Baseline', color='green')
        plt.xlabel('Run'); plt.ylabel('Accuracy')
        plt.title('Pooled Accuracy: MTL vs Baseline'); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'pooled_accuracy_comparison.png'))
        plt.close()


        cl_mt = df_cl_run.groupby('cluster')['accuracy'].agg(['mean','std']).rename(
            columns={'mean':'mtl_mean','std':'mtl_std'})
        cl_bs = df_base_cl_run.groupby('cluster')['accuracy'].agg(['mean','std']).rename(
            columns={'mean':'base_mean','std':'base_std'})
        cmpdf = cl_mt.join(cl_bs, how='outer').fillna(0)
        clusters = list(cmpdf.index)
        x = np.arange(len(clusters))
        w = 0.35
        plt.figure(figsize=(8,4))

        plt.bar(x - w/2,
                cmpdf['mtl_mean'],
                w,
                yerr=cmpdf['mtl_std'],
                label='MTL',
                color='blue',
                capsize=4)

        plt.bar(x + w/2,
                cmpdf['base_mean'],
                w,
                yerr=cmpdf['base_std'],
                label='Baseline',
                color='green',
                capsize=4)
        plt.xticks(x, clusters)
        plt.xlabel('Cluster'); plt.ylabel('Accuracy')
        plt.title('Cluster Accuracy: MTL vs Baseline'); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'cluster_accuracy_comparison.png'))
        plt.close()

        plt.figure(figsize=(6,4))
        sns.boxplot(y=df_cmp_subj['accuracy_delta'], color='lightgray')
        plt.title('Subject-level Δ Accuracy (MTL − Baseline)')
        plt.ylabel('Δ Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'subject_accuracy_delta_boxplot.png'))
        plt.close()

        if 'confusion_matrix' in viz_list:
            # pooled MTL
            all_gt, all_pr = [], []
            for runs in self.mtl_wrapper.results_by_subject.values():
                for run in runs:
                    all_gt.extend(run['ground_truth'])
                    all_pr.extend(run['predictions'])
            self.visuals.plot_confusion_matrix(
                np.array(all_gt), np.array(all_pr),
                filename="cm_mtl_pooled_agg.png"
            )
            all_gt_bs, all_pr_bs = [], []
            for run in self.baseline_wrapper.get_experiment_results('pooled'):
                all_gt_bs.extend(run['ground_truth'])
                all_pr_bs.extend(run['predictions'])
            self.visuals.plot_confusion_matrix(
                np.array(all_gt_bs), np.array(all_pr_bs),
                filename="cm_baseline_pooled_agg.png"
            )
            for cl in cl_stats['cluster'].tolist():
                # MTL cluster
                gt_cl, pr_cl = [], []
                for subj, runs in self.mtl_wrapper.results_by_subject.items():
                    if self.mtl_wrapper.cluster_assignments.get(subj) == cl:
                        for run in runs:
                            gt_cl.extend(run['ground_truth'])
                            pr_cl.extend(run['predictions'])
                self.visuals.plot_confusion_matrix(
                    np.array(gt_cl), np.array(pr_cl),
                    filename=f"cm_mtl_cluster_{cl}_agg.png"
                )
                gt_bs_cl, pr_bs_cl = [], []
                for subj, runs in self.baseline_wrapper.get_experiment_results('single').items():
                    if self.mtl_wrapper.cluster_assignments.get(subj) == cl:
                        for run in runs:
                            gt_bs_cl.extend(run['ground_truth'])
                            pr_bs_cl.extend(run['predictions'])
                self.visuals.plot_confusion_matrix(
                    np.array(gt_bs_cl), np.array(pr_bs_cl),
                    filename=f"cm_baseline_cluster_{cl}_agg.png"
                )

        # Cluster scatter
        if 'cluster_scatter' in viz_list:
            self.visuals.plot_cluster_scatter(
                self.subject_reprs,
                self.mtl_wrapper.cluster_assignments,
                method="pca",
                filename="cluster_scatter_pca.png"
            )
            self.visuals.plot_cluster_scatter(
                self.subject_reprs,
                self.mtl_wrapper.cluster_assignments,
                method="tsne",
                filename="cluster_scatter_tsne.png"
            )

        return {
            'mtl_subject_run': df_subj_run,
            'mtl_subject_stats': subj_stats,
            'mtl_cluster_run': df_cl_run,
            'mtl_cluster_stats': cl_stats,
            'baseline_cluster_run': df_base_cl_run,
            'baseline_cluster_stats': base_cl_stats,
            'mtl_pooled_run': df_pool_run,
            'mtl_pooled_stats': pool_stats,
            'baseline_subject_run': df_base_subj_run,
            'baseline_pooled_run': df_base_pool_run,
            'delta_subject_run': df_cmp_subj,
            'delta_cluster_run': df_cmp_cl,
            'delta_pooled_run': df_cmp_pool,
        }
