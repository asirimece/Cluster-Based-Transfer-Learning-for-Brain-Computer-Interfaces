import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import defaultdict
from lib.tl.train import TLTrainer
from omegaconf import DictConfig, OmegaConf
from lib.base.train import BaseWrapper
from lib.evaluate.metrics import MetricsEvaluator
from lib.evaluate.visuals import VisualEvaluator
from lib.logging import logger

logger = logger.get()


class TLEvaluator:
    def __init__(self, tl_results: dict[int, dict[int, dict]], config: DictConfig):
        if hasattr(tl_results, "results_by_subject"):
            raise ValueError(
                "TLEvaluator expects a dict of runs-subjects."
            )
        self.tl_results = tl_results

        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)
        while "evaluators" not in config and "experiment" in config:
            config = config.experiment
        self.cfg = config

        n_out = self.cfg.transfer.model.n_outputs
        qc = self.cfg.evaluators.quantitative
        self.metrics = MetricsEvaluator({"metrics": qc.metrics, "n_outputs": n_out})

        qv = self.cfg.evaluators.qualitative
        out_dir = self.cfg.evaluators.tl_output_dir
        labels = list(range(n_out))
        self.visuals = VisualEvaluator({
            "visualizations":   qv.visualizations,
            "pca_n_components": qv.pca_n_components,
            "tsne":             qv.tsne,
            "output_dir":       out_dir,
            "labels":           labels,
        })
        os.makedirs(out_dir, exist_ok=True)

        # optionally load baseline & mtl for comparisons
        try:
            base_cfg = OmegaConf.load("config/experiment/base.yaml")
            single = pickle.load(open(base_cfg.logging.single_results_path, "rb"))
            pooled = pickle.load(open(base_cfg.logging.pooled_results_path, "rb"))
            bw = BaseWrapper({"single": single, "pooled": pooled})
            self.baseline = bw.get_experiment_results("single")
        except Exception:
            self.baseline = None

        try:
            mtl_wrapped = pickle.load(open(
                os.path.join(self.cfg.mtl.mtl_model_output, "mtl_wrapper.pkl"), "rb"
            ))
            self.mtl = mtl_wrapped.results_by_subject
        except Exception:
            self.mtl = None


    def evaluate(self):
        logger.info("TLEvaluator initialized.")
        rows = []
        for run_idx, subj_dict in self.tl_results.items():
            for subj, res in subj_dict.items():
                gt, pr = res["ground_truth"], res["predictions"]
                m = self.metrics.evaluate(gt, pr)
                row = {"run": run_idx, "subject": subj}
                row.update(m)
                rows.append(row)

                if run_idx == 0 and "confusion_matrix" in self.visuals.visualizations:
                    self.visuals.plot_confusion_matrix(
                        gt, pr,
                        filename=f"cm_tl_subject_{subj}.png"
                    )

        df_runs = pd.DataFrame(rows)
        out = self.cfg.evaluators.tl_output_dir
        df_runs.to_csv(os.path.join(out, "tl_subject_run_metrics.csv"), index=False)

        # subject‐level stats
        subj_stats = df_runs.groupby("subject").agg(["mean","std"])
        subj_stats.columns = [f"{met}_{st}" for met, st in subj_stats.columns]
        subj_stats = subj_stats.reset_index()
        subj_stats.to_csv(os.path.join(out, "tl_subject_stats.csv"), index=False)

        # pooled: all runs & subjects
        pooled = df_runs.drop(columns=["subject","run"]).agg(["mean","std"]).T
        pooled.columns = ["mean","std"]
        pooled = pooled.reset_index().rename(columns={"index":"metric"})
        pooled.to_csv(os.path.join(out, "tl_pooled_stats.csv"), index=False)

        outputs = {
            "tl_subject_run": df_runs,
            "tl_subject_stats": subj_stats,
            "tl_pooled_stats": pooled
        }

        # TL vs baseline
        if self.baseline is not None:
            cmp_df = self._compare(self.tl_results, self.baseline, "tl", "baseline")
            cmp_df.to_csv(os.path.join(out, "tl_vs_baseline.csv"), index=False)
            outputs["tl_vs_baseline"] = cmp_df

        # TL vs MTL vs baseline
        if self.baseline is not None and self.mtl is not None:
            all_rows = []
            for label, src in (("baseline", self.baseline),
                               ("tl", self.tl_results),
                               ("mtl", self.mtl)):
                if label == "tl":
                    for run_idx, subj_dict in src.items():
                        for subj, res in subj_dict.items():
                            m = self.metrics.evaluate(res["ground_truth"], res["predictions"])
                            row = {"model":label,"run":run_idx,"subject":subj}
                            row.update(m)
                            all_rows.append(row)
                else:
                    for subj, runs in src.items():
                        for i, res in enumerate(runs):
                            m = self.metrics.evaluate(res["ground_truth"], res["predictions"])
                            row = {"model":label,"run":i,"subject":subj}
                            row.update(m)
                            all_rows.append(row)

            df_all = pd.DataFrame(all_rows)
            df_all.to_csv(os.path.join(out, "all_model_comparison.csv"), index=False)
            try:
                plt.figure(figsize=(8,4))
                sns.boxplot(data=df_all, x="model", y="accuracy", palette="pastel")
                plt.title("Accuracy: Baseline vs TL vs MTL")
                plt.tight_layout()
                plt.savefig(os.path.join(out, "model_accuracy_boxplot.png"))
                plt.close()
            except Exception:
                pass
            outputs["all_model_comparison"] = df_all

        logger.info("TL evaluation complete.")
        return outputs

    def _compare(self, a_runs, b_runs, la, lb):
        rows = []
        def flatten(src, label):
            tmp = {}
            if label == "tl":
                for run_idx, subj_dict in src.items():
                    for subj, res in subj_dict.items():
                        tmp.setdefault(subj, []).append(res)
            else:
                tmp = src
            return tmp

        A = flatten(a_runs, la)
        B = flatten(b_runs, lb)

        for subj, runs_a in A.items():
            runs_b = B.get(subj, [])
            for i in range(min(len(runs_a), len(runs_b))):
                ra, rb = runs_a[i], runs_b[i]
                ma = self.metrics.evaluate(ra["ground_truth"], ra["predictions"])
                mb = self.metrics.evaluate(rb["ground_truth"], rb["predictions"])
                row = {"subject": subj, "run": i}
                for k in ma:
                    row[f"{la}_{k}"]   = ma[k]
                    row[f"{lb}_{k}"]   = mb[k]
                    row[f"delta_{k}"] = ma[k] - mb[k]
                rows.append(row)

        return pd.DataFrame(rows)



class TargetTLEvaluator:
    """
    Evaluates in-session TL performance.
    """
    def __init__(self, trainer, config):
        self.trainer = trainer
        self.cfg     = config
        self.mode    = trainer.mode 

        with open(trainer.target_cfg.data_path, 'rb') as f:
            self.target_cfg = pickle.load(f)

        n_out = self.cfg.transfer.model.n_outputs
        self.metrics = MetricsEvaluator({
            'metrics':   self.cfg.evaluators.quantitative.metrics,
            'n_outputs': n_out
        })
        self.visuals = VisualEvaluator({
            'visualizations':   self.cfg.evaluators.qualitative.visualizations,
            'pca_n_components': self.cfg.evaluators.qualitative.pca_n_components,
            'tsne':             self.cfg.evaluators.qualitative.tsne,
            'output_dir':       self.cfg.evaluators.tl_output_dir,
            'labels':           list(range(n_out))
        })

        self.out_dir = self.cfg.evaluators.tl_output_dir
        os.makedirs(self.out_dir, exist_ok=True)
    
    def _get_pooled_state(self):
        """Return a state dict for the pooled model, if available.
        """
        if hasattr(self.trainer, "pooled_source") and self.trainer.pooled_source:
            path = self.trainer.pooled_source
            if isinstance(path, str) and os.path.exists(path):
                return torch.load(path, map_location="cpu")

        # in_session fallback: use the already-loaded pooled weights
        if getattr(self.trainer, "_pretrained", None):
            return self.trainer._pretrained

        # nothing available
        return None

    def evaluate(self):
        """Run all in_session evaluations and save results."""
        # baseline performance
        df_base = self._baseline_target()
        
        # cluster distance diagnostics
        df_dist = self._cluster_distance()
        
        # incremental calibration vs trials
        df_inc  = self._incremental_by_trials()
        
        # delta vs distance scatter and summary file
        self._plot_delta_vs_distance(df_base, df_dist, df_inc)
        
        # confusion matrices after final calibration
        self._plot_confusion_after_calib()
        
        # PCA overlay of clusters
        self._plot_cluster_overlay()
        
        # cluster‐mate comparison 
        self._plot_cluster_comparison(df_base, df_inc, df_dist)
        
        # similarity vs. accuracy scatter
        self._plot_similarity_vs_accuracy(df_inc)
        
        logger.info('[In-Session] Evaluation complete.')


    def _baseline_target(self):
        """
        Write both no-calibration baselines:
        - pooled head-0 
        - zero-shot via in_session cluster assignment 
        """
        col_uni = 'pooled_target_accuracy'
        col_zs  = 'zero_shot_cluster_accuracy'
        rows = []

        for subj, splits in self.target_cfg.items():
            X, y = splits['train'].get_data(), splits['train'].events[:, -1]

            preds_uni = self.trainer._predict_with_state(self.trainer._pretrained or {}, X, head_id=0)
            m_uni = self.metrics.evaluate(y, np.array(preds_uni))

            cid = dist = None
            preds_zs = None
            if (self.trainer.cluster_wrapper is not None and
                self.cfg.transfer.model.n_clusters_pretrained > 1):
                try:
                    cid, dist = self.trainer._assign_cluster_target(subj, unlabeled_epochs=X, prefer_eval=True)
                    preds_zs = self.trainer._predict_with_state(self.trainer._pretrained or {}, X, head_id=cid)
                    m_zs = self.metrics.evaluate(y, np.array(preds_zs))
                    acc_zs = m_zs['accuracy']
                except Exception as e:
                    logger.warning(f"[In-Session] zero-shot cluster baseline failed for {subj}: {e}")
                    acc_zs = np.nan
            else:
                acc_zs = np.nan

            rows.append({
                'subject': subj,
                col_uni: m_uni['accuracy'],
                col_zs:  acc_zs,
                'assigned_cluster': (None if cid is None else int(cid)),
                'cluster_distance': (None if dist is None else float(dist)),
            })

            if 'confusion_matrix' in self.metrics.metrics:
                self.visuals.plot_confusion_matrix(
                    y, np.array(preds_uni),
                    filename=f'cm_target_pooled_subject_{subj}.png'
                )
                if preds_zs is not None:
                    self.visuals.plot_confusion_matrix(
                        y, np.array(preds_zs),
                        filename=f'cm_target_zero_shot_subject_{subj}.png'
                    )

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.out_dir, f'target_baselines.csv'), index=False)
        return df

    def _cluster_distance(self):
        """
        Compute cluster id and distance for every subject.
        """
        import numpy as np
        import pandas as pd

        rows = []
        for subj, splits in self.target_cfg.items():
            subj_str = str(subj)

            Xu = None
            try:
                if isinstance(splits, dict):
                    if "day2" in splits and "unlabeled" in splits["day2"]:
                        Xu = splits["day2"]["unlabeled"].get_data()
                    elif "train" in splits:
                        Xu = splits["train"].get_data()
            except Exception:
                Xu = None

            cid, dist = None, None
            try:
                cid, dist = self.trainer._assign_cluster_target(subj_str, unlabeled_epochs=Xu, prefer_eval=True)
            except Exception:
                try:
                    rep = self.trainer._get_subject_repr(subj_str)
                    km  = self.trainer.cluster_wrapper.model
                    cid = int(km.predict(rep[None, :])[0])
                    if hasattr(km, "transform"):
                        dist = float(np.min(km.transform(rep[None, :])[0]))
                    else:
                        d = np.linalg.norm(km.cluster_centers_ - rep[None, :], axis=1)
                        dist = float(d[cid])
                except Exception as e:
                    logger.warning(f"[In-Session] distance/cluster failed for {subj_str}: {e}")

            rows.append({
                "subject":          subj_str,
                "assigned_cluster": (None if cid  is None else int(cid)),
                "cluster_distance": (None if dist is None else float(dist)),
            })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.out_dir, "target_cluster_distances.csv"), index=False)
        self._dist_rows = rows
        self._dist_map = {r["subject"]: r.get("cluster_distance") for r in rows}
        return df

    def _incremental_by_trials(self):
        """
        Few-shot curves for both inits per subject:
        """
        k_max = int(self.cfg.transfer.in_session.k_shot)
        rows  = []

        for subj, splits in self.target_cfg.items():
            Xall, yall = splits['train'].get_data(), splits['train'].events[:, -1]

            cid = None
            if (self.trainer.cluster_wrapper is not None and
                self.cfg.transfer.model.n_clusters_pretrained > 1):
                try:
                    cid, _ = self.trainer._assign_cluster_target(subj, unlabeled_epochs=Xall, prefer_eval=True)
                    cid = int(cid)
                except Exception as e:
                    logger.warning(f"[In-Session] could not assign cluster for {subj}: {e}")
                    cid = None

            for k in range(1, k_max + 1):
                Xc, yc = Xall[:k], yall[:k]
                Xt, yt = Xall[k:], yall[k:]

                # cluster-init few-shot 
                if cid is not None:
                    state_c = self.trainer.calibrate_subject_from_cluster(
                        pooled_state=self.trainer._pretrained or {},
                        cluster_id=cid,
                        subject_id=subj,
                        X_calib=Xc, y_calib=yc
                    )
                    preds_c = self.trainer._predict_with_state(state_c, Xt, subject_id=subj)
                    mc = self.metrics.evaluate(yt, np.array(preds_c))
                    mc.update({'subject': subj, 'k_shot': k, 'approach': 'cluster'})
                    rows.append(mc)

                # pooled-init few-shot 
                state_p = self.trainer.calibrate_subject(
                    pooled_state=self.trainer._pretrained or {},
                    subject_id=subj,
                    X_calib=Xc, y_calib=yc,
                    calib_epochs=self.cfg.transfer.in_session.calib_epochs,
                    calib_lr=self.cfg.transfer.in_session.calib_lr
                )
                preds_p = self.trainer._predict_with_state(state_p, Xt, subject_id=subj)
                mp = self.metrics.evaluate(yt, np.array(preds_p))
                mp.update({'subject': subj, 'k_shot': k, 'approach': 'pooled'})
                rows.append(mp)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.out_dir, 'target_incremental_both.csv'), index=False)

        for subj, gsub in df.groupby('subject'):
            plt.figure()
            for app, g in gsub.groupby('approach'):
                plt.plot(g['k_shot'], g['accuracy'], '-o', label=app)
            plt.xlabel('Calibration Trials'); plt.ylabel('Accuracy')
            plt.title(f'in_session TL ({self.mode}): Accuracy vs # Calibration Trials — {subj}')
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, f'target_acc_vs_k_{self.mode}_{subj}.png'))
            plt.close()

        return df

    def _plot_delta_vs_distance(self, df_base, df_dist, df_inc):
        import pandas as pd
        import matplotlib.pyplot as plt

        for df in (df_base, df_inc, df_dist):
            if df is not None and "subject" in getattr(df, "columns", []):
                df["subject"] = df["subject"].astype(str)

        need_dist = (
            df_dist is None
            or "cluster_distance" not in getattr(df_dist, "columns", [])
            or df_dist.empty
            or not set(df_base["subject"]).issubset(set(df_dist["subject"]))
        )
        if need_dist:
            df_dist = self._cluster_distance()

        k = int(self.cfg.transfer.in_session.k_shot)
        finals = (
            df_inc.loc[df_inc.k_shot == k, ["subject", "accuracy", "approach"]]
                .rename(columns={"accuracy": "acc_after"})
        )

        base = df_base.merge(df_dist[["subject", "cluster_distance"]], on="subject", how="left")
        pooled = finals[finals.approach == "pooled"].merge(base, on="subject", how="left")
        if "pooled_target_accuracy" in pooled.columns:
            pooled["accuracy_delta"] = pooled["acc_after"] - pooled["pooled_target_accuracy"]
        else:
            pooled["accuracy_delta"] = pd.NA

        cluster = finals[finals.approach == "cluster"].merge(base, on="subject", how="left")
        if "zero_shot_cluster_accuracy" in cluster.columns:
            cluster["accuracy_delta"] = cluster["acc_after"] - cluster["zero_shot_cluster_accuracy"]
        else:
            cluster["accuracy_delta"] = pd.NA

        # save combined summary
        out = pd.concat([pooled.assign(approach="pooled"),
                        cluster.assign(approach="cluster")], ignore_index=True)
        out.to_csv(os.path.join(self.out_dir, "target_subject_summary.csv"), index=False)

        if "cluster_distance" not in out.columns or out["cluster_distance"].isna().all():
            logger.warning("[In-Session] cluster_distance unavailable after merge; skipping Δ vs Distance plot.")
            return

        plt.figure()
        g_p = out[out.approach == "pooled"].dropna(subset=["accuracy_delta", "cluster_distance"])
        g_c = out[out.approach == "cluster"].dropna(subset=["accuracy_delta", "cluster_distance"])
        if not g_p.empty:
            plt.scatter(g_p["cluster_distance"], g_p["accuracy_delta"], alpha=0.8, label="pooled → Δ vs pooled")
        if not g_c.empty:
            plt.scatter(g_c["cluster_distance"], g_c["accuracy_delta"], alpha=0.8, marker="^", label="cluster → Δ vs zero-shot")

        plt.xlabel("Cluster Distance")
        plt.ylabel("Δ Accuracy")
        plt.title(f"ΔAccuracy vs Distance ({self.mode})")
        if (not g_p.empty) or (not g_c.empty):
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f"target_delta_vs_distance_{self.mode}.png"))
        plt.close()

    def _plot_confusion_after_calib(self):
        k = self.cfg.transfer.in_session.k_shot
        for subj, splits in self.target_cfg.items():
            Xall, yall = splits['train'].get_data(), splits['train'].events[:,-1]
            Xc,yc = Xall[:k], yall[:k]; Xt,yt = Xall[k:], yall[k:]
            if self.trainer.cluster_wrapper and self.cfg.transfer.model.n_clusters_pretrained>1:
                rep = self.trainer._get_subject_repr(subj)
                cid = int(self.trainer.cluster_wrapper.model.predict(rep[None,:])[0])
                state = self.trainer.calibrate_subject_from_cluster(
                    pooled_state=self.trainer._pretrained or {},
                    cluster_id=cid, subject_id=subj, X_calib=Xc, y_calib=yc
                )
            else:
                state = self.trainer.calibrate_subject(
                    pooled_state=self.trainer._pretrained or {},
                    subject_id=subj, X_calib=Xc, y_calib=yc,
                    calib_epochs=self.cfg.transfer.in_session.calib_epochs,
                    calib_lr=self.cfg.transfer.in_session.calib_lr
                )
            preds = self.trainer._predict_with_state(state, Xt, subject_id=subj)
            if 'confusion_matrix' in self.metrics.metrics:
                self.visuals.plot_confusion_matrix(
                    yt, np.array(preds),
                    filename=f'cm_target_{self.mode}_after_{k}_subject_{subj}.png'
                )

    def _plot_cluster_overlay(self):
        from sklearn.decomposition import PCA
        import seaborn as sns, matplotlib.pyplot as plt

        orig_ids  = list(self.trainer.cluster_wrapper.subject_representations)
        orig_repr = np.stack([self.trainer._get_subject_repr(s) for s in orig_ids], axis=0)
        orig_lbls = np.array([self.trainer.cluster_wrapper.get_cluster_for_subject(s) for s in orig_ids])

        pca2 = PCA(n_components=2)
        proj = pca2.fit_transform(orig_repr)

        plt.figure(figsize=(6,6))
        for cid in np.unique(orig_lbls):
            mask = orig_lbls == cid
            plt.scatter(proj[mask,0], proj[mask,1], label=f'cluster {cid}', alpha=0.4, s=40, edgecolor='none',
                        color=sns.color_palette('tab10')[cid])

        for rec in self._dist_rows:
            sid = rec['subject']
            Xunl = self.target_cfg[sid]['train'].get_data()

            if sid in getattr(self.trainer, "target_features", {}):
                rep = self.trainer._repr_from_precomputed_features(sid, prefer_eval=True)
            else:
                rep = self.trainer._compute_subject_repr_from_epochs(Xunl)
            nr = pca2.transform(rep[None, :])
            cid = int(rec['assigned_cluster'])
            plt.plot(nr[:,0], nr[:,1], '*', markersize=15, markeredgecolor='k',
                    markerfacecolor=sns.color_palette('tab10')[cid], label=f'S{sid} (cluster {cid})')

        plt.title('Original Clusters + New Subjects')
        plt.xlabel('PCA 1'); plt.ylabel('PCA 2')
        plt.legend(bbox_to_anchor=(1,1), fontsize='small', frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f'target_overlay_{self.mode}.png'))
        plt.close()


    def _plot_cluster_comparison(self, df_base, df_inc, df_dist):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from collections import defaultdict

        univ_state = self._get_pooled_state()
        if univ_state is None:
            logger.warning("[In-Session] No pooled state available; skipping cluster-mate comparison.")
            return

        off = self.trainer._evaluate_with_state(univ_state)
        accs = {
            subj: self.metrics.evaluate(r['ground_truth'], r['predictions'])['accuracy']
            for subj, r in off.items()
        }

        cm = self.trainer.cluster_wrapper
        cluster_accs = defaultdict(list)
        for subj, acc in accs.items():
            cluster_accs[cm.get_cluster_for_subject(subj)].append(acc)

        for new_subj in df_base['subject'].unique():
            new_cid = int(df_dist.loc[df_dist.subject == new_subj, 'assigned_cluster'].iloc[0])

            splits = self.target_data[new_subj]
            Xb, yb = splits['train'].get_data(), splits['train'].events[:, -1]
            base_preds = self.trainer._predict_with_state(univ_state, Xb, head_id=new_cid)
            uni_acc = self.metrics.evaluate(yb, np.array(base_preds))['accuracy']

            # Calibrated accuracy at k_max
            k_max = self.cfg.transfer.in_session.k_shot
            cal_acc = float(
                df_inc[(df_inc.subject == new_subj) & (df_inc.k_shot == k_max)]['accuracy'].iloc[0]
            )

            mates = cluster_accs.get(new_cid, [])
            if not mates:
                logger.warning(f"[In-Session] No cluster-mate accuracies for cluster {new_cid}; skipping.")
                continue

            plt.figure(figsize=(5,4))
            sns.violinplot(data=[mates], inner='quart')
            plt.scatter([-0.1], [uni_acc], s=100, color='blue', marker='o', label='new pooled')
            plt.scatter([+0.1], [cal_acc], s=100, color='red',  marker='*', label='new calibrated')
            plt.xticks([], [])
            plt.ylabel('Accuracy')
            plt.title(f'Cluster {new_cid} mates (n={len(mates)}) vs S{new_subj}')
            plt.legend()
            plt.tight_layout()
            out = os.path.join(self.out_dir, f'target_cluster_{new_cid}_S{new_subj}_comparison.png')
            plt.savefig(out); plt.close()

    def _plot_similarity_vs_accuracy(self, df_inc):
        univ_state = self._get_pooled_state()
        if univ_state is None:
            logger.warning("[In-Session] No pooled state available; skipping similarity vs. accuracy plot.")
            return

        off = self.trainer._evaluate_with_state(univ_state)

        # Build new subject representation in the same space as KMeans
        new_subj = str(df_inc["subject"].iloc[0])
        try:
            if new_subj in getattr(self.trainer, "target_features", {}):
                rep_new = self.trainer._repr_from_precomputed_features(new_subj, prefer_eval=True)
            else:
                # prefer eval-day unlabeled if present; else train
                splits = self.target_data[new_subj]
                Xu = None
                if isinstance(splits, dict):
                    if "day2" in splits and "unlabeled" in splits["day2"]:
                        Xu = splits["day2"]["unlabeled"].get_data()
                    elif "train" in splits:
                        Xu = splits["train"].get_data()
                rep_new = self.trainer._compute_subject_repr_from_epochs(Xu)
        except Exception as e:
            logger.warning(f"[In-Session] could not form in_session representation for {new_subj}: {e}")
            return

        rows = []
        for s, r in off.items():
            acc  = self.metrics.evaluate(r['ground_truth'], r['predictions'])['accuracy']
            rep_s = self.trainer.cluster_wrapper.subject_representations[s]
            dist = float(np.linalg.norm(rep_s - rep_new))
            rows.append({'subject': s, 'distance_to_new': dist, 'accuracy': float(acc)})
        df = pd.DataFrame(rows)

        new_cal = df_inc[df_inc.k_shot == self.cfg.transfer.in_session.k_shot].iloc[0].accuracy

        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4))
        plt.scatter(df.distance_to_new, df.accuracy, alpha=0.5, label='cluster mates')
        plt.scatter(0, new_cal, s=120, marker='*', label='new calibrated')
        plt.xlabel('Distance to New Subject'); plt.ylabel('cross-subject Accuracy')
        plt.title('Similarity vs. Accuracy')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir,'target_similarity_vs_accuracy.png'))
        plt.close()
