import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from omegaconf import OmegaConf
import mne
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from lib.dataset.dataset import EEGDataset
from lib.model.deep4net import Deep4NetModel
from lib.base.evaluate import BaselineEvaluator
from lib.logging import logger

matplotlib.use("Agg")

logger = logger.get()


class BaseWrapper:
    """ 
    Wraps baseline training results. 
    """
    def __init__(self, results_by_experiment):
        self.results_by_experiment = results_by_experiment

    def get_experiment_results(self, key):
        return self.results_by_experiment.get(key)


class BaselineTrainer:
    def __init__(self,
                 base_config_path="config/experiment/base.yaml",
                 model_config_path="config/model/deep4net.yaml"):
        
        self.base_config = OmegaConf.load(base_config_path)
        self.model_config = OmegaConf.load(model_config_path)

        exp_cfg = self.base_config.experiment
        self.device = exp_cfg.device
        self.single_cfg = exp_cfg.single
        self.pooled_cfg = exp_cfg.pooled

        self.single_results_path = self.base_config.logging.single_results_path
        self.pooled_results_path = self.base_config.logging.pooled_results_path

        with open(self.base_config.data.preprocessed_data, "rb") as f:
            self.preprocessed_data = pickle.load(f)

    def _train_deep4net_model(self,
                             X_train, y_train, train_ids,
                             X_test,  y_test,  test_ids,
                             model_cfg, train_cfg, device="cpu"):

        model_inst = Deep4NetModel(model_cfg)
        model = model_inst.get_model().to(device)

        train_ds = EEGDataset(X_train, y_train, train_ids)
        test_ds = EEGDataset(X_test,  y_test,  test_ids)
        train_loader = DataLoader(train_ds,
                                  batch_size=train_cfg.batch_size,
                                  shuffle=True)
        test_loader = DataLoader(test_ds,
                                  batch_size=train_cfg.batch_size,
                                  shuffle=False)

        Optim = getattr(torch.optim, train_cfg.optimizer)
        optimizer = Optim(model.parameters(),
                          lr=train_cfg.learning_rate,
                          weight_decay=train_cfg.weight_decay)

        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(train_cfg.epochs):
            total_loss, correct, total = 0.0, 0, 0
            batch_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{train_cfg.epochs}",
                unit="batch",
                leave=False
            )
            for Xb, yb, _ in batch_iter:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(Xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                preds   = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += Xb.size(0)
                total_loss += loss.item() * Xb.size(0)

                batch_iter.set_postfix({
                    "loss": f"{(total_loss/total):.4f}",
                    "acc":  f"{(correct/total):.4f}",
                    "wd":   f"{train_cfg.weight_decay}"
                })

            avg_loss = total_loss / total
            acc      = correct / total
            logger.info(f"[BaselineTrainer] Epoch {epoch+1}/{train_cfg.epochs} "
                  f"Loss={avg_loss:.4f} Acc={acc:.4f} "
                  f"WD={train_cfg.weight_decay}")

        model.eval()
        all_logits, all_tids, all_y = [], [], []
        with torch.no_grad():
            for Xb, yb, tid in test_loader:
                Xb = Xb.to(device)
                out = model(Xb).cpu().numpy()
                all_logits.append(out)
                all_tids.append(tid.numpy())
                all_y.extend(yb.numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_tids   = np.concatenate(all_tids,   axis=0)
        unique_t   = np.unique(all_tids)

        preds, labels = [], []
        for t in unique_t:
            idx = np.where(all_tids == t)[0]
            avg_logit = all_logits[idx].mean(axis=0)
            preds.append(int(avg_logit.argmax()))
            labels.append(int(all_y[idx[0]]))

        acc   = accuracy_score(labels, preds)
        kappa = cohen_kappa_score(labels, preds)
        cm    = confusion_matrix(labels, preds)
        logger.info(f"Trial‑level Test - Acc: {acc:.4f}, Kappa: {kappa:.4f}")

        return model, {"ground_truth": labels, "predictions": preds}


    def _train_subject(self, subj, subject_data):
        tr = subject_data["train"]
        te = subject_data["test"]
        Xtr, ytr = tr.get_data(), tr.events[:, -1]
        Xte, yte = te.get_data(),  te.events[:, -1]
        tid_tr   = tr.events[:, 1]
        tid_te   = te.events[:, 1]

        common = {k: self.model_config[k]
                  for k in ["name","in_chans","n_classes","n_times","final_conv_length"]
                  if k in self.model_config}
        merged_cfg = {**common, **self.model_config.get("single", {})}

        results_runs = []
        for run_i in range(self.single_cfg.n_runs):
            seed = self.single_cfg.seed_start + run_i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            logger.info(f"Single subject run {run_i+1}/{self.single_cfg.n_runs}"
                  f" for Subject {subj} (seed={seed})")

            _, trial_res = self._train_deep4net_model(
                Xtr, ytr, tid_tr,
                Xte, yte, tid_te,
                merged_cfg, self.single_cfg,
                device=self.device
            )
            results_runs.append(trial_res)

        return results_runs


    def _train_pooled(self):
        Xtr_list, ytr_list, tid_tr_list = [], [], []
        Xte_list, yte_list, tid_te_list = [], [], []

        for idx, (_, data) in enumerate(self.preprocessed_data.items()):
            #tr = data["0train"]; te = data["1test"]
            tr = data["train"]; te = data["test"]
            offset = idx * 1_000_000
            Xtr_list.append(tr.get_data());     
            ytr_list.append(tr.events[:,-1])
            tid_tr_list.append(tr.events[:,1] + offset)
            Xte_list.append(te.get_data());     
            yte_list.append(te.events[:,-1])
            tid_te_list.append(te.events[:,1] + offset)

        Xtr = np.concatenate(Xtr_list, axis=0)
        ytr = np.concatenate(ytr_list, axis=0)
        tid_tr= np.concatenate(tid_tr_list, axis=0)
        Xte = np.concatenate(Xte_list, axis=0)
        yte = np.concatenate(yte_list, axis=0)
        tid_te= np.concatenate(tid_te_list, axis=0)

        common = {k: self.model_config[k]
                  for k in ["name","in_chans","n_classes","n_times","final_conv_length"]
                  if k in self.model_config}
        merged_cfg = {**common, **self.model_config.get("pooled", {})}

        pooled_runs = []
        for run_i in range(self.pooled_cfg.n_runs):
            seed = self.pooled_cfg.seed_start + run_i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            logger.info(f"Pooled run {run_i+1}/{self.pooled_cfg.n_runs} (seed={seed})")

            _, trial_res = self._train_deep4net_model(
                Xtr, ytr, tid_tr,
                Xte, yte, tid_te,
                merged_cfg, self.pooled_cfg,
                device=self.device
            )
            pooled_runs.append(trial_res)

        return pooled_runs


    def run(self):
        single_res = {}
        for subj, data in self.preprocessed_data.items():
            single_res[subj] = self._train_subject(subj, data)
        os.makedirs(os.path.dirname(self.single_results_path), exist_ok=True)
        with open(self.single_results_path, "wb") as f:
            pickle.dump(single_res, f)
        logger.info(f"Single-subject training results are saved.")

        pooled_res = self._train_pooled()
        os.makedirs(os.path.dirname(self.pooled_results_path), exist_ok=True)
        with open(self.pooled_results_path, "wb") as f:
            pickle.dump(pooled_res, f)
        logger.info(f"Pooled training results are saved.")

        return BaseWrapper({"single": single_res,
                            "pooled": pooled_res})

