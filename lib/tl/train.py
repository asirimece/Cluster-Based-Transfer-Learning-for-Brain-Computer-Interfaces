import os
import random
import pickle
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from tqdm.auto import tqdm
from lib.augment.augment import mixup_batch_torch
from omegaconf import DictConfig

from lib.dataset.dataset import TLSubjectDataset
from lib.tl.model import TLModel
from lib.utils.utils import _prefix_mtl_keys
from lib.logging import logger

logger = logger.get()


def freeze_backbone_layers(backbone: nn.Module, freeze_until_layer: str | None = None):
    """
    Freeze all backbone parameters up to (and including) the named layer.
    """
    found = False
    for name, module in backbone.named_children():
        for p in module.parameters():
            p.requires_grad = False
        if freeze_until_layer and name == freeze_until_layer:
            found = True
            break
    if freeze_until_layer and not found:
        raise ValueError(f"freeze_until_layer '{freeze_until_layer}' not found")


class TLTrainer:
    """
    TL Trainer supporting modes:
      - Pooled: pooled fine-tune on all subjects; evaluate per subject.
      - Zero_shot: for each held subject s*: pooled–pooled on others vs clustered–zero-shot on s*.
      - Few_shot: for each s*: pooled–pooled on others, then k-per-class calibration on s*.
      - In_session: pooled fine-tune, then calibration on new subject recording.
    """

    def __init__(self, config: DictConfig):
        tl_cfg = config.experiment.experiment.transfer
        self.cfg = tl_cfg
        self.mode = getattr(tl_cfg, "mode", "pooled")
        self.device = torch.device(tl_cfg.device)
        self.aug_cfg  = config.augment.augmentations
        
        self.do_mixup = bool(
            getattr(self.cfg, "augment", False) and
            self.aug_cfg.get("mixup", {}).get("enabled", False)
        )
        self.mixup_alpha = float(self.aug_cfg.get("mixup", {}).get("alpha", 0.0))
        
        self.n_runs         = tl_cfg.n_runs
        self.seed_start     = tl_cfg.seed_start
        self.batch_size     = tl_cfg.batch_size
        self.weight_decay   = tl_cfg.weight_decay

        self.univ_epochs = tl_cfg.pooled_epochs
        self.patience    = tl_cfg.early_stop_patience
        self.val_frac    = getattr(tl_cfg, "val_fraction", 0.1)

        self.head_lr           = tl_cfg.head_lr
        self.backbone_lr       = tl_cfg.backbone_lr
        self.freeze_backbone   = tl_cfg.freeze_backbone
        self.freeze_until_layer= tl_cfg.freeze_until_layer

        self.target_cfg = getattr(tl_cfg, "in_session", None) 
        self.use_cluster = getattr(tl_cfg, "use_cluster", False)
        
            
        with open(config.experiment.experiment.preprocessed_file, "rb") as f:
            self.data = pickle.load(f)

        self.subject_ids = sorted(self.data.keys())  

        self._pretrained = None
        if not tl_cfg.init_from_scratch and tl_cfg.pretrained_mtl_model:
            state = torch.load(tl_cfg.pretrained_mtl_model, map_location=self.device)
            self._pretrained = _prefix_mtl_keys(state)
            logger.info("[TLTrainer] loaded pretrained MTL weights")

        # Load cluster wrapper
        mtl_out = config.experiment.experiment.mtl.mtl_model_output
        wrapper_path = os.path.join(mtl_out, "cluster_wrapper.pkl")
        self.cluster_wrapper = None
        if os.path.exists(wrapper_path):
            with open(wrapper_path, "rb") as f:
                self.cluster_wrapper = pickle.load(f)
            try:
                ncl = self.cluster_wrapper.get_num_clusters()
            except Exception:
                ncl = getattr(self.cluster_wrapper, "n_clusters_", None) or getattr(
                    getattr(self.cluster_wrapper, "model", None), "n_clusters", "?"
                )
            logger.info(f"[TLTrainer] loaded cluster wrapper (n_clusters={ncl})")

        os.makedirs(tl_cfg.tl_model_output, exist_ok=True)
        self.pooled_out = os.path.join(tl_cfg.tl_model_output, "tl_pooled.pth")

        feat_fp = config.experiment.experiment.features_file
        with open(feat_fp, "rb") as f:
            self.features = pickle.load(f)
            
        self.features_mask = None
        if self.mode == "in_session":
            mask_fp_cfg = getattr(config.experiment.experiment, "features_mask_file", None)
            mask_fp_inf = None
            if isinstance(feat_fp, str) and feat_fp.endswith(".pkl"):
                base = feat_fp[:-4]
                for cand in (base + "_mask.pkl", os.path.join(os.path.dirname(feat_fp), "features_mask.pkl")):
                    if os.path.exists(cand):
                        mask_fp_inf = cand
                        break
            for cand in (mask_fp_cfg, mask_fp_inf):
                if cand and os.path.exists(cand):
                    try:
                        with open(cand, "rb") as f:
                            m = pickle.load(f)
                        m = np.array(m)
                        if m.dtype == bool or np.issubdtype(m.dtype, np.integer):
                            self.features_mask = m
                            logger.info(f"[In-Session] loaded features_mask with shape {self.features_mask.shape} (dtype={self.features_mask.dtype})")
                        else:
                            logger.warning(f"[In-Session] ignored features_mask (unsupported dtype {m.dtype})")
                    except Exception as e:
                        logger.warning(f"[In-Session] failed to load features_mask from {cand}: {e}")
                    break

        self.target_features = {}
        try:
            ofp = getattr(config.experiment.experiment, "target_features_file", None)
            if ofp is None and hasattr(self.cfg, "in_session") and getattr(self.cfg.in_session, "features_file", None):
                ofp = self.cfg.in_session.features_file
            if ofp:
                with open(ofp, "rb") as f:
                    self.target_features = pickle.load(f)  # dict: subject_id -> nested dict / arrays
                logger.info(f"[TLTrainer] loaded in_session features from {ofp} (subjects={len(self.target_features)})")
        except Exception as e:
            logger.warning(f"[TLTrainer] could not load in_session features: {e}")

        self.features_mask = None
        try:
            mfp = getattr(config.experiment.experiment, "features_mask_file", None)
            if mfp is None and hasattr(self.cfg, "in_session") and getattr(self.cfg.in_session, "features_mask_file", None):
                mfp = self.cfg.in_session.features_mask_file
            if mfp:
                with open(mfp, "rb") as f:
                    self.features_mask = pickle.load(f)  # np.ndarray[bool] or indices
                logger.info(f"[TLTrainer] loaded features mask from {mfp} (len={len(self.features_mask)})")
        except Exception as e:
            logger.warning(f"[TLTrainer] could not load features mask: {e}")

        self._feat_scaler = getattr(self.cluster_wrapper, "scaler", None)
        self._feat_pca    = getattr(self.cluster_wrapper, "pca",    None)
        if self.mode == "in_session":
            n_expected = getattr(getattr(self.cluster_wrapper, "model", None), "n_features_in_", None)
            if (self._feat_pca is None or self._feat_scaler is None) and n_expected is not None:
                logger.warning(
                    "[In-Session] cluster_wrapper is missing scaler and/or pca. "
                    "If KMeans was trained in the PCA space, persist both in cluster_wrapper.pkl."
                )

    def _extract_head_keys_from_state(self, keys):
        pat = re.compile(r"(?:^|\.)(?:mtl_net\.)?heads\.([^\.]+)\.")
        out = set()
        for k in keys:
            m = pat.search(k)
            if m:
                out.add(m.group(1))
        return sorted(out)

    def _state_has_head(self, state, head_key):
        prefixes = (f"heads.{head_key}.", f"mtl_net.heads.{head_key}.")
        return any(any(k.startswith(p) for p in prefixes) for k in state.keys())
                   
    def _get_subject_repr(self, subject_id):
        """
        Return the D-dim vector used for clustering/assignment.
        """
        if self.cluster_wrapper and hasattr(self.cluster_wrapper, "subject_representations"):
            if subject_id in self.cluster_wrapper.subject_representations:
                return self.cluster_wrapper.subject_representations[subject_id]

        raw = self.features[subject_id]
        combined = []
        for key in ("train", "test"):
            sess = raw.get(key, raw)
            if isinstance(sess, dict) and "combined" in sess:
                combined.append(sess["combined"])
        if not combined:
            raise RuntimeError(f"No combined features for subject {subject_id}")
        Xall = np.concatenate(combined, axis=0)
        return Xall.mean(axis=0)

    def _assign_cluster(self, subject_id: int):
        """
        Use the saved wrapper's model to assign a subject to a cluster.
        """
        if self.cluster_wrapper is None:
            raise RuntimeError("Cluster assignment requested but no cluster wrapper loaded.")
        model = getattr(self.cluster_wrapper, "model", None)
        if model is None:
            raise RuntimeError("cluster_wrapper.model missing.")
        rep = self._get_subject_repr(subject_id)[None, :]
        cid = int(model.predict(rep)[0])

        dist = None
        if hasattr(model, "transform"):
            drow = model.transform(rep)[0]  # distances to all centroids
            dist = float(np.min(drow))
        return cid, dist

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self, n_chans: int, window: int, n_clusters: int) -> TLModel:
        head_kw = {"hidden_dim": self.cfg.head_hidden_dim, "dropout": self.cfg.head_dropout}
        model = TLModel(
            n_chans=n_chans,
            n_outputs=self.cfg.model.n_outputs,
            n_clusters_pretrained=n_clusters,
            window_samples=window,
            head_kwargs=head_kw
        ).to(self.device)
        if self._pretrained is not None:
            model.load_state_dict(self._pretrained, strict=False)
            if not hasattr(self, "_pretrained_debug"):
                numeric = [k for k in model.heads.keys() if k.isdigit()]
                other   = [k for k in model.heads.keys() if not k.isdigit()]
                try:
                    expect = set(map(str, range(n_clusters)))
                    assert set(numeric) >= expect, \
                        f"Missing cluster heads. expect={sorted(expect)} have={sorted(numeric)}"
                except Exception as e:
                    logger.warning("[TLTrainer]", e)
                self._pretrained_debug = True
        if self.freeze_backbone or self.freeze_until_layer:
            freeze_backbone_layers(model.shared_backbone, self.freeze_until_layer)
        return model

    def _build_pooled_loaders(self, exclude_subj=None):
        """
        Build subject-disjoint train/val loaders.
        """
        pool = [sid for sid in self.subject_ids if sid != exclude_subj]
        n_subj = len(pool)
        if n_subj < 2:
            raise RuntimeError(f"Need at least 2 subjects to form a subject-disjoint val split; got {n_subj}.")

        n_val_subj = max(1, int(round(self.val_frac * n_subj)))
        perm = np.random.permutation(n_subj)
        val_subj   = [pool[i] for i in perm[:n_val_subj]]
        train_subj = [s for s in pool if s not in val_subj]

        train_dsets, val_dsets = [], []
        for sid in train_subj:
            splits = self.data[sid]
            X, y = splits["train"].get_data(), splits["train"].events[:, -1]
            train_dsets.append(TLSubjectDataset(X, y))
        for sid in val_subj:
            splits = self.data[sid]
            X, y = splits["train"].get_data(), splits["train"].events[:, -1]
            val_dsets.append(TLSubjectDataset(X, y))

        train_full = ConcatDataset(train_dsets)
        val_full   = ConcatDataset(val_dsets)
        
        train_loader = DataLoader(train_full, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_full,   batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader


    def _sample_k_shots_per_class(self, X: np.ndarray, y: np.ndarray, k_per_class: int, seed: int):
        rng = np.random.default_rng(seed)
        Xc_list, yc_list = [], []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            if len(idx) < k_per_class:
                raise ValueError(f"Class {cls}: need {k_per_class}, have {len(idx)}")
            take = rng.choice(idx, size=k_per_class, replace=False)
            Xc_list.append(X[take]); yc_list.append(y[take])
        return np.concatenate(Xc_list, axis=0), np.concatenate(yc_list, axis=0)

    # In-session helpers
    def _compute_subject_repr_from_epochs(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_wrapper is None or not hasattr(self.cluster_wrapper, "model"):
            raise RuntimeError("[In-Session] cluster_wrapper with a fitted .model (KMeans) is required.")

        kmeans = self.cluster_wrapper.model
        needed = getattr(kmeans, "n_features_in_", None)
        if needed is None:
            raise RuntimeError("[In-Session] KMeans model lacks n_features_in_. Was it fitted?")

        # Use exact cross-subject featurizer
        if not hasattr(self.cluster_wrapper, "featurizer"):
            raise RuntimeError("[In-Session] cluster_wrapper.featurizer missing. Persist the cross-subject extractor.")
        feat = self.cluster_wrapper.featurizer

        def _call_feat(f, x):
            try:
                return f(x)                          
            except Exception:
                try:
                    return f(x.reshape(x.shape[0], -1))  
                except Exception as e:
                    raise RuntimeError(
                        "[In-Session] featurizer could not process unlabeled epochs. "
                        f"Error: {e}"
                    )

        if hasattr(feat, "compute_features"):
            F = _call_feat(feat.compute_features, X)  # (N, D0)
        elif hasattr(feat, "transform"):
            F = _call_feat(feat.transform, X)         # (N, D0)
        elif callable(feat):
            F = _call_feat(feat, X)                   # (N, D0)
        else:
            raise RuntimeError("[In-Session] featurizer lacks compute_features/transform/callable.")

        if not isinstance(F, np.ndarray) or F.ndim != 2:
            raise RuntimeError(f"[In-Session] featurizer returned invalid shape {getattr(F,'shape',None)}; expected (N, D0).")

        # 2) mask → 3) scaler → 4) projector (PCA) in the same order as cross-subject
        Fm = self._apply_feature_mask(F)

        scaler = getattr(self.cluster_wrapper, "scaler", None) or getattr(self.cluster_wrapper, "standardizer", None)
        Fs = scaler.transform(Fm) if (scaler is not None and hasattr(scaler, "transform")) else Fm

        proj = getattr(self.cluster_wrapper, "projector", None) or getattr(self.cluster_wrapper, "pca", None)
        Z = proj.transform(Fs) if (proj is not None and hasattr(proj, "transform")) else Fs

        rep = Z.mean(axis=0)  

        if rep.shape[0] != needed:
            raise ValueError(
                f"[In-Session] representation has {rep.shape[0]} dims, but KMeans expects {needed}. "
                "Ensure featurizer/mask/scaler/projector match the cross-subject pipeline."
            )
        return rep


    def _assign_cluster_from_unlabeled(self, X_unlabeled: np.ndarray):
        """
        Assign cluster using a representation computed from this subject's own unlabeled epochs.
        """
        if self.cluster_wrapper is None or not hasattr(self.cluster_wrapper, "model"):
            raise RuntimeError("Cluster assignment requires a loaded cluster_wrapper with a .model")
        rep = self._compute_subject_repr_from_epochs(X_unlabeled)[None, :]
        model = self.cluster_wrapper.model
        cid = int(model.predict(rep)[0])
        dist = None
        if hasattr(model, "transform"):
            dist = float(np.min(model.transform(rep)[0]))
        return cid, dist

    def _predict_cluster_head_on_array(self, X: np.ndarray, cluster_id: int):
        """
        Zero-shot: run the pretrained cluster head on raw array X (N, C, T).
        """
        X0 = self.data[self.subject_ids[0]]["train"].get_data()
        n_ch, win = X0.shape[1], X0.shape[2]
        n_clusters = self.cfg.model.n_clusters_pretrained
        m = TLModel(
            n_chans=n_ch, n_outputs=self.cfg.model.n_outputs,
            n_clusters_pretrained=n_clusters, window_samples=win,
            head_kwargs={"hidden_dim": self.cfg.head_hidden_dim, "dropout": self.cfg.head_dropout}
        ).to(self.device)
        if self._pretrained is not None:
            m.load_state_dict(self._pretrained, strict=False)
        m.eval()
        loader = DataLoader(TLSubjectDataset(X, np.zeros(len(X))), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for Xb, _ in loader:
                Xb = Xb.to(self.device)
                preds.extend(m(Xb, [cluster_id]*len(Xb)).argmax(dim=1).cpu().numpy())
        return np.array(preds, dtype=int)

    def _split_k_shot_rest(self, X: np.ndarray, y: np.ndarray, k: int, seed: int):
        """
        Stratified k-per-class selection for calibration; remaining trials used for evaluation.
        """
        rng = np.random.default_rng(seed)
        take_idx = []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            if len(idx) < k:
                raise ValueError(f"Class {cls}: need {k}, have {len(idx)}")
            take_idx.extend(rng.choice(idx, size=k, replace=False))
        take_idx = np.array(sorted(take_idx))
        mask = np.ones(len(y), dtype=bool)
        mask[take_idx] = False
        Xc, yc = X[take_idx], y[take_idx]
        Xt, yt = X[mask], y[mask]
        return (Xc, yc), (Xt, yt)

    def _extract_target_splits(self, splits: dict):
        """
        Flexible extractor for target subject recording dicts.
        """
        def _xy(ep):
            return ep.get_data(), ep.events[:, -1]

        out = {
            "calib_labeled_X": None, "calib_labeled_y": None,
            "calib_unlabeled_Xu": None,
            "eval_unlabeled_Xu": None,
            "eval_X": None, "eval_y": None
        }

        if "day1" in splits and "day2" in splits:
            d1, d2 = splits["day1"], splits["day2"]
            if "train" in d1:
                X, y = _xy(d1["train"])
                out["calib_labeled_X"], out["calib_labeled_y"] = X, y
            if "unlabeled" in d1:
                out["calib_unlabeled_Xu"] = d1["unlabeled"].get_data()
            if "test" in d2:
                Xte, yte = _xy(d2["test"])
                out["eval_X"], out["eval_y"] = Xte, yte
            if "unlabeled" in d2:
                out["eval_unlabeled_Xu"] = d2["unlabeled"].get_data()

            if out["eval_unlabeled_Xu"] is None and "train" in d2:
                out["eval_unlabeled_Xu"] = d2["train"].get_data()
            if out["calib_unlabeled_Xu"] is None and "train" in d1:
                out["calib_unlabeled_Xu"] = d1["train"].get_data()

        else:
            if "train" in splits and "test" in splits:
                Xtr, ytr = _xy(splits["train"])
                Xte, yte = _xy(splits["test"])
                out["calib_labeled_X"], out["calib_labeled_y"] = Xtr, ytr
                out["calib_unlabeled_Xu"] = Xtr
                out["eval_unlabeled_Xu"] = Xte
                out["eval_X"], out["eval_y"] = Xte, yte
            else:
                raise RuntimeError("in_session data structure not recognized. Expected day1/day2 or train/test keys.")
        return out
    
    def _apply_feature_mask(self, X: np.ndarray) -> np.ndarray:
        """Apply a boolean/index feature mask if available. X: (N, D)."""
        if self.features_mask is None:
            return X
        if self.features_mask.dtype == bool:
            if self.features_mask.shape[-1] != X.shape[-1]:
                raise ValueError(f"[In-Session] Boolean features_mask length {self.features_mask.shape[-1]} != feature dim {X.shape[-1]}")
            return X[:, self.features_mask]
        return X[:, self.features_mask.astype(int)]

    def _gather_feature_arrays(self, node):
        out = []
        if node is None:
            return out
        if isinstance(node, dict):
            for v in node.values():
                out.extend(self._gather_feature_arrays(v))
        elif isinstance(node, (list, tuple)):
            for v in node:
                out.extend(self._gather_feature_arrays(v))
        elif isinstance(node, np.ndarray) and node.ndim == 2:
            out.append(node)
        return out

    def _repr_from_precomputed_features(self, subject_id, prefer_eval=True):
        """
        Build the subject representation from precomputed features.
        """
        if subject_id not in self.target_features:
            raise KeyError(f"[In-Session] features for subject '{subject_id}' not found in target_features_file.")

        cand = self.target_features[subject_id]
        arrays = []
        if isinstance(cand, dict) and ("day2" in cand or "day1" in cand):
            if prefer_eval and "day2" in cand:
                arrays = self._gather_feature_arrays(cand["day2"])
            elif not prefer_eval and "day1" in cand:
                arrays = self._gather_feature_arrays(cand["day1"])
        if not arrays:
            arrays = self._gather_feature_arrays(cand)

        if not arrays:
            raise RuntimeError(f"[In-Session] No 2D feature arrays found for subject '{subject_id}'.")

        F = np.concatenate(arrays, axis=0)  # (N, D0)

        if self.features_mask is not None:
            F = F[:, self.features_mask]

        if self._feat_scaler is not None:
            F = self._feat_scaler.transform(F)
        if self._feat_pca is not None:
            F = self._feat_pca.transform(F)

        km = getattr(self.cluster_wrapper, "model", None)
        if km is not None and hasattr(km, "n_features_in_"):
            if F.shape[1] != km.n_features_in_:
                raise RuntimeError(
                    f"[In-Session] Feature dim {F.shape[1]} ≠ KMeans expects {km.n_features_in_}. "
                    "Persist scaler/pca (and mask) in cluster_wrapper, or ensure in_session features are in the same space."
                )

        rep = F.mean(axis=0)  # (D,)
        return rep

    def _assign_cluster_target(self, subject_id: int | str, unlabeled_epochs: np.ndarray | None, prefer_eval=True):
        """
        Target data cluster assignment with support for precomputed features.
        """
        if self.cluster_wrapper is None or not hasattr(self.cluster_wrapper, "model"):
            raise RuntimeError("[In-Session] Cluster assignment requires cluster_wrapper with a .model (KMeans).")

        if subject_id in getattr(self, "target_features", {}):
            rep = self._repr_from_precomputed_features(subject_id, prefer_eval=prefer_eval)
        else:
            if unlabeled_epochs is None:
                raise RuntimeError("[In-Session] No in_session features for subject and no unlabeled epochs provided.")
            if hasattr(self.cluster_wrapper, "featurizer") and hasattr(self.cluster_wrapper.featurizer, "compute_features"):
                F = self.cluster_wrapper.featurizer.compute_features(unlabeled_epochs)  # (N, D0)
                if self.features_mask is not None:
                    F = F[:, self.features_mask]
                if self._feat_scaler is not None:
                    F = self._feat_scaler.transform(F)
                if self._feat_pca is not None:
                    F = self._feat_pca.transform(F)
                km = self.cluster_wrapper.model
                if hasattr(km, "n_features_in_") and F.shape[1] != km.n_features_in_:
                    raise RuntimeError(
                        f"[In-Session] Featurizer output dim {F.shape[1]} ≠ KMeans expects {km.n_features_in_}. "
                        "Ensure featurizer+mask+scaler+pca match the cross-subject pipeline."
                    )
                rep = F.mean(axis=0)
            else:
                raise RuntimeError(
                    "[In-Session] No precomputed features for subject and no featurizer saved in cluster_wrapper. "
                )

        km = self.cluster_wrapper.model
        cid = int(km.predict(rep[None, :])[0])
        dist = None
        if hasattr(km, "transform"):
            dist = float(np.min(km.transform(rep[None, :])[0]))
        return cid, dist

    # Ealuation helpers

    def _eval_subject_head(self, state: dict, subject_id: int, head_id: int):
        """
        Evaluate 'state' using head_id on subject_id's TEST set.
        """
        X0 = self.data[self.subject_ids[0]]["train"].get_data()
        n_ch, win = X0.shape[1], X0.shape[2]
        m = self._build_model(n_ch, win, self.cfg.model.n_clusters_pretrained)
        head_key = m.add_new_head(head_id)  # "subj_*"
        m.load_state_dict(state, strict=False)  
        m.eval()

        splits = self.data[subject_id]["test"]
        Xte, yte = splits.get_data(), splits.events[:, -1]
        loader = DataLoader(TLSubjectDataset(Xte, yte), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for Xb, _ in loader:
                Xb = Xb.to(self.device)
                preds.extend(m(Xb, [head_key] * len(Xb)).argmax(dim=1).cpu().numpy())
        return {"ground_truth": yte.astype(int), "predictions": np.array(preds, dtype=int)}

    def _eval_subject_clustered_zero_shot(self, subject_id: int):
        """
        Build a model with the pretrained trunk+cluster heads and evaluate the assigned cluster head.
        """
        if self.cluster_wrapper is None:
            raise RuntimeError("Clustered zero-shot requested but no cluster wrapper loaded.")
        X0 = self.data[self.subject_ids[0]]["train"].get_data()
        n_ch, win = X0.shape[1], X0.shape[2]
        n_clusters = self.cfg.model.n_clusters_pretrained

        m = TLModel(
            n_chans=n_ch, n_outputs=self.cfg.model.n_outputs,
            n_clusters_pretrained=n_clusters, window_samples=win,
            head_kwargs={"hidden_dim": self.cfg.head_hidden_dim, "dropout": self.cfg.head_dropout}
        ).to(self.device)
        if self._pretrained is not None:
            m.load_state_dict(self._pretrained, strict=False)
        m.eval()

        cid, dist = self._assign_cluster(subject_id)
        splits = self.data[subject_id]["test"]
        Xte, yte = splits.get_data(), splits.events[:, -1]
        loader = DataLoader(TLSubjectDataset(Xte, yte), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for Xb, _ in loader:
                Xb = Xb.to(self.device)
                preds.extend(m(Xb, [cid] * len(Xb)).argmax(dim=1).cpu().numpy())
        return {"ground_truth": yte.astype(int),
                "predictions": np.array(preds, dtype=int),
                "assigned_cluster": int(cid),
                "assignment_distance": dist}

    def _evaluate_zero_shot_all(self):
        """
        Evaluate clustered zero-shot for all subjects.
        """
        results = {}
        for sid in self.subject_ids:
            results[sid] = self._eval_subject_clustered_zero_shot(sid)
        return results


    def run(self) -> dict:
        any_s = self.subject_ids[0]
        X0 = self.data[any_s]["train"].get_data()
        n_chans, window = X0.shape[1], X0.shape[2]
        n_clusters = self.cfg.model.n_clusters_pretrained

        # Zero-shot
        if self.mode == "loso":
            loso_res_pooled = {}
            loso_res_clustered = {}

            for held in tqdm(self.subject_ids, desc="LOSO subjects"):
                tr, vl = self._build_pooled_loaders(exclude_subj=held)
                state_univ = self._train_pooled(n_chans, window, 1, tr, vl)
                loso_res_pooled[held] = self._eval_subject_head(state_univ, held, head_id=0)

                if self.cluster_wrapper is not None and n_clusters > 1:
                    loso_res_clustered[held] = self._eval_subject_clustered_zero_shot(held)

            runs = {0: loso_res_pooled}
            if len(loso_res_clustered) > 0:
                runs[1] = loso_res_clustered  
            return runs

        # Few-shot
        if self.mode == "loso_few_shot":
            k = None
            if self.target_cfg is not None and hasattr(self.target_cfg, "k_shot"):
                k = int(self.target_cfg.k_shot)
            elif hasattr(self.cfg, "k_shots"):
                k = int(self.cfg.k_shots)
            if k is None or k <= 0:
                raise RuntimeError("loso_few_shot requires in_session.k_shot or cfg.k_shots > 0 (shots per class).")

            loso_fs = {}
            for held in tqdm(self.subject_ids, desc="LOSO few-shot"):
                tr, vl = self._build_pooled_loaders(exclude_subj=held)
                state_univ = self._train_pooled(n_chans, window, 1, tr, vl)

                Xtr, ytr = self.data[held]["train"].get_data(), self.data[held]["train"].events[:, -1]
                Xc, yc = self._sample_k_shots_per_class(Xtr, ytr, k_per_class=k, seed=self.seed_start)

                if self.use_cluster and self.cluster_wrapper is not None and n_clusters > 1:
                    cid, _ = self._assign_cluster(held)
                    cstate = self.calibrate_subject_from_cluster(state_univ, cid, held, Xc, yc)
                else:
                    cstate = self.calibrate_subject(
                        state_univ, held, Xc, yc,
                        calib_epochs=self.target_cfg.calib_epochs if self.target_cfg else 10,
                        calib_lr=self.target_cfg.calib_lr if self.target_cfg else self.head_lr
                    )

                loso_fs[held] = self._test_with_state(cstate, held)

            return {0: loso_fs}

        all_runs = {}
        for run_i in range(self.n_runs):
            seed = self.seed_start + run_i
            self._set_seed(seed)

            tr, vl = self._build_pooled_loaders(exclude_subj=None)
            state = self._train_pooled(n_chans, window, n_clusters, tr, vl)
            if run_i == self.n_runs - 1:
                torch.save(state, self.pooled_out)
                logger.info(f"Saved pooled pooled model → {self.pooled_out}")

            results = self._evaluate_with_state(state)

            if self.mode == "in_session" and self.target_cfg and self.target_cfg.enabled:
                R = int(getattr(self.target_cfg, "n_repeats", 50))
                k_grid = list(getattr(self.target_cfg, "k_grid",
                                      ([int(self.target_cfg.k_shot)] if hasattr(self.target_cfg, "k_shot") else [1, 2, 4, 8])))
                assign_both = bool(getattr(self.target_cfg, "assignment_both", True))
                with open(self.target_cfg.data_path, "rb") as f:
                    target_cfg = pickle.load(f)

                _old_mixup = self.do_mixup
                self.do_mixup = False
                try:
                    for sid, splits in target_cfg.items():
                        ex = self._extract_target_splits(splits)
                        X_cal_lab, y_cal_lab = ex["calib_labeled_X"], ex["calib_labeled_y"]
                        Xu_cal  = ex["calib_unlabeled_Xu"]
                        Xu_eval = ex["eval_unlabeled_Xu"]
                        X_eval, y_eval = ex["eval_X"], ex["eval_y"]

                        zero_shot = {}
                        try:
                            cid_eval, dist_eval = self._assign_cluster_target(sid, unlabeled_epochs=Xu_eval, prefer_eval=True)
                            zs_preds_eval = self._predict_cluster_head_on_array(X_eval, cid_eval)
                            zero_shot["target_assignment"] = {
                                "assigned_cluster": int(cid_eval),
                                "assignment_distance": dist_eval,
                                "assignment_source": ("target_features" if sid in self.target_features else "eval_unlabeled"),
                                "predictions": zs_preds_eval,
                                "ground_truth": y_eval.astype(int),
                            }
                        except Exception as e:
                            logger.warning(f"[In-Session] eval-day assignment for subj {sid} failed: {e}")

                        if assign_both:
                            try:
                                cid_cal, dist_cal = self._assign_cluster_target(sid, unlabeled_epochs=Xu_cal, prefer_eval=False)
                                zs_preds_cal = self._predict_cluster_head_on_array(X_eval, cid_cal)
                                zero_shot["conservative_assignment"] = {
                                    "assigned_cluster": int(cid_cal),
                                    "assignment_distance": dist_cal,
                                    "assignment_source": ("target_features" if sid in self.target_features else "calib_unlabeled"),
                                    "predictions": zs_preds_cal,
                                    "ground_truth": y_eval.astype(int),
                                }
                            except Exception as e:
                                logger.warning(f"[In-Session] calib-day assignment for subj {sid} failed: {e}")

                        few_shot = {}
                        for k in k_grid:
                            runs_k = []
                            for r in range(R):
                                (Xc, yc), _ = self._split_k_shot_rest(X_cal_lab, y_cal_lab, k, seed=self.seed_start + r)

                                clustered_preds = None
                                assigned_key_for_cluster = None
                                if "target_assignment" in zero_shot:
                                    cid_use = zero_shot["target_assignment"]["assigned_cluster"]
                                    assigned_key_for_cluster = "target_assignment"
                                elif "conservative_assignment" in zero_shot:
                                    cid_use = zero_shot["conservative_assignment"]["assigned_cluster"]
                                    assigned_key_for_cluster = "conservative_assignment"
                                else:
                                    cid_use = None

                                if cid_use is not None:
                                    cstate = self.calibrate_subject_from_cluster(state, cid_use, sid, Xc, yc)
                                    clustered_preds = np.array(self._predict_with_state(cstate, X_eval, sid), dtype=int)

                                pstate = self.calibrate_subject(
                                    state, sid, Xc, yc,
                                    calib_epochs=self.target_cfg.calib_epochs,
                                    calib_lr=self.target_cfg.calib_lr
                                )
                                pooled_preds = np.array(self._predict_with_state(pstate, X_eval, sid), dtype=int)

                                runs_k.append({
                                    "k": int(k),
                                    "repeat": int(r),
                                    "pooled_preds": pooled_preds,
                                    "clustered_preds": clustered_preds,
                                    "ground_truth": y_eval.astype(int),
                                    "cluster_init_from": assigned_key_for_cluster
                                })
                            few_shot[int(k)] = runs_k

                        results[sid] = {"zero_shot": zero_shot, "few_shot": few_shot}
                finally:
                    self.do_mixup = _old_mixup

            all_runs[run_i] = results
        return all_runs


    def _train_pooled(
        self,
        n_chans: int,
        window: int,
        n_clusters_pre: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self._set_seed(self.seed_start)

        model    = self._build_model(n_chans, window, n_clusters_pre)
        old_keys = set(model.heads.keys())

        head_key = model.add_new_head(0)  # returns "subj_0"
        new_keys = set(model.heads.keys()) - old_keys
        if len(new_keys) != 1:
            raise RuntimeError(f"Expected exactly one new head, got {new_keys!r}")
        head_str    = new_keys.pop()
        head_id_int = int(head_str.split("_")[-1])

        optim_params = []
        if not self.freeze_backbone:
            optim_params.append({"params": model.shared_backbone.parameters(),
                                 "lr":      self.backbone_lr})
        optim_params.append({"params": model.heads[head_str].parameters(),
                             "lr":      self.head_lr})
        optimizer = torch.optim.Adam(optim_params, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val, no_imp, best_state = float("inf"), 0, None
        for ep in range(1, self.univ_epochs + 1):
            model.train()
            t_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                if self.do_mixup:
                    # torch-native mixup
                    Xmix, ya, yb2, lam = mixup_batch_torch(Xb, yb, self.mixup_alpha)
                    out  = model(Xmix, [head_key] * len(Xmix))
                    loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb2)
                else:
                    out = model(Xb, [head_key] * len(Xb))
                    loss  = criterion(out, yb)
                    
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * yb.size(0)
            t_loss /= len(train_loader.dataset)

            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(self.device), yv.to(self.device)
                    out = model(Xv, [head_key] * len(Xv))
                    v_loss += criterion(out, yv).item() * yv.size(0)
            v_loss /= len(val_loader.dataset)

            logger.info(f"[TLTrainer] Epoch {ep}: train={t_loss:.4f}, val={v_loss:.4f}")
            if v_loss < best_val - 1e-4:
                best_val   = v_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                no_imp     = 0
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    logger.info(f"[TLTrainer] Early stop at epoch {ep}")
                    break

        if best_state is None:
            raise RuntimeError("No checkpoint saved during pooled fine-tuning")
        return best_state

    def _evaluate_with_state(self, state: dict):
        """
        Evaluate a given state for all subjects' TEST sets using their cluster head.
        """
        results = {}

        for subj, splits in self.data.items():
            Xtr = self.data[self.subject_ids[0]]["train"].get_data()
            n_ch, win = Xtr.shape[1], Xtr.shape[2]
            m = self._build_model(n_ch, win, self.cfg.model.n_clusters_pretrained)
            m.load_state_dict(state, strict=False)
            m.to(self.device)
            m.eval()

            if self.cluster_wrapper is not None:
                if hasattr(self.cluster_wrapper, "get_cluster_for_subject"):
                    cid = int(self.cluster_wrapper.get_cluster_for_subject(subj))
                else:
                    rep = self._get_subject_repr(subj)
                    cid = int(self.cluster_wrapper.model.predict(rep[None, :])[0])
            else:
                cid = 0

            head_key = str(cid)
            
            if not self._state_has_head(state, head_key):
                avail = self._extract_head_keys_from_state(state.keys())
                raise RuntimeError(
                    f"[TLTrainer] Loaded state missing cluster head '{head_key}'. "
                    f"Available heads: {avail[:12]}"
                )

            Xte, yte = splits["test"].get_data(), splits["test"].events[:, -1]
            loader = DataLoader(TLSubjectDataset(Xte, yte),
                                batch_size=self.batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for Xb, _ in loader:
                    Xb = Xb.to(self.device)
                    out = m(Xb, [head_key] * len(Xb))
                    preds.extend(out.argmax(dim=1).cpu().numpy())

            results[subj] = {
                "ground_truth": np.array(yte, dtype=int),
                "predictions":  np.array(preds, dtype=int),
            }
        return results



    def _test_with_state(self, state: dict, subject_id: int):
        """
        Evaluate a calibrated state using subject-specific head on that subject's TEST set.
        """
        X0 = self.data[self.subject_ids[0]]["train"].get_data()
        n_ch, win = X0.shape[1], X0.shape[2]
        m = self._build_model(n_ch, win, self.cfg.model.n_clusters_pretrained)

        _ = m.add_new_head(0)                 
        subj_key = m.add_new_head(subject_id)
        m.load_state_dict(state, strict=False)
        if not self._state_has_head(state, subj_key):
            avail = self._extract_head_keys_from_state(state.keys())
            raise RuntimeError(
                f"[TLTrainer] Loaded state is missing '{subj_key}'. "
                f"Available heads in state: {avail[:12]}"
            )
        m.eval()

        splits = self.data[subject_id]["test"]
        Xte, yte = splits.get_data(), splits.events[:, -1]
        loader = DataLoader(TLSubjectDataset(Xte, yte), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for Xb, _ in loader:
                Xb  = Xb.to(self.device)
                out = m(Xb, [subj_key] * len(Xb))
                preds.extend(out.argmax(dim=1).cpu().numpy())
        return {"ground_truth": yte.astype(int), "predictions": np.array(preds, dtype=int)}

    def calibrate_subject(
        self,
        pooled_state: dict,
        subject_id: int | str,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        calib_epochs: int = 10,
        calib_lr: float    = None
    ) -> dict:
        """
        Pooled few-shot: freeze pooled trunk, add new subject head, fine-tune head.
        """
        #self.calib_batch_size     = getattr(self.cfg, "calib_batch_size", self.batch_size)
        
        if calib_lr is None:
            calib_lr = self.head_lr

        X0 = self.data[self.subject_ids[0]]["train"].get_data()
        n_ch, win = X0.shape[1], X0.shape[2]
        m = self._build_model(n_ch, win, self.cfg.model.n_clusters_pretrained)
        m.add_new_head(0)
        m.load_state_dict(pooled_state, strict=False)

        # freeze trunk and add subject head
        freeze_backbone_layers(m.shared_backbone)

        subj_key = m.add_new_head(subject_id)
        optimizer = torch.optim.Adam(m.heads[subj_key].parameters(),
                                    lr=(calib_lr if calib_lr is not None else self.head_lr))
        criterion = nn.CrossEntropyLoss()
        ds     = TLSubjectDataset(X_calib, y_calib)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        m.train()
        for ep in range(1, calib_epochs + 1):
            total = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                if self.do_mixup:
                    Xmix, ya, yb2, lam = mixup_batch_torch(Xb, yb, self.mixup_alpha)
                    out  = m(Xmix, [subj_key] * len(Xmix))
                    loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb2)
                else:
                    out   = m(Xb, [subj_key] * len(Xb))
                    loss  = criterion(out, yb)
                
                loss.backward()
                optimizer.step()
                total += loss.item() * len(yb)
            logger.info(f"[Calib subj={subject_id}] Ep{ep}/{calib_epochs} loss={(total/len(loader.dataset)):.4f}")
        return m.state_dict()


    def calibrate_subject_from_cluster(
        self,
        pooled_state: dict,
        cluster_id: int,
        subject_id: int,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> dict:
        """
        Clustered few-shot: copy the assigned cluster head into a new subject head,
        freeze pooled trunk, fine-tune head.
        """
        #self.calib_batch_size     = getattr(self.cfg, "calib_batch_size", self.batch_size)
        X0 = self.data[self.subject_ids[0]]["train"].get_data()
        n_chans, window = X0.shape[1], X0.shape[2]
        n_clusters = self.cfg.model.n_clusters_pretrained

        # Build model; add all cluster heads; load PRETRAINED (to get cluster head weights)
        m = self._build_model(n_chans, window, n_clusters)
        # 1) load pretrained (gives you cluster heads "0..k-1"), then overlay pooled
        if self._pretrained is not None:
            m.load_state_dict(self._pretrained, strict=False)
        m.load_state_dict(pooled_state, strict=False)
        
        m.add_new_head(0)

        # create subject head and init from the cluster head
        subj_key = m.add_new_head(subject_id)
        cluster_key = str(int(cluster_id))
        assert cluster_key in m.heads, f"Cluster head '{cluster_key}' missing; have {list(m.heads.keys())[:10]}"
        m.heads[subj_key].load_state_dict(m.heads[cluster_key].state_dict())
        
        # freeze backbone, fine-tune subject head
        freeze_backbone_layers(m.shared_backbone, self.freeze_until_layer)
        optimizer = torch.optim.Adam(m.heads[subj_key].parameters(),
                                    lr=(self.target_cfg.calib_lr if self.target_cfg else self.head_lr))
        criterion = nn.CrossEntropyLoss()
        ds = TLSubjectDataset(X_calib, y_calib)

        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        m.train()
        for epoch in range(1, (self.target_cfg.calib_epochs if self.target_cfg else 10) + 1):
            total = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                if self.do_mixup:
                    Xmix, ya, yb2, lam = mixup_batch_torch(Xb, yb, self.mixup_alpha)
                    out  = m(Xmix, [subj_key] * len(Xmix))
                    loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb2)
                else:
                    out = m(Xb, [subj_key] * len(Xb))
                    loss = criterion(out, yb)
                    
                loss.backward()
                optimizer.step()
                total += loss.item() * len(yb)
            logger.info(f"[Cluster-Calib subj={subject_id}] Ep{epoch} loss={total/len(loader.dataset):.4f}")

        return m.state_dict()


    def _predict_with_state(self, state: dict, X, head_id=None, subject_id=None):
        """
        Predict on X with a given state.
        """
        Xnp = X if isinstance(X, np.ndarray) else X.get_data()
        n_ch, win = Xnp.shape[1], Xnp.shape[2]

        avail_heads = set(self._extract_head_keys_from_state(state.keys()))  # e.g., ['0','1','2','subj_subject_001']
        desired_key = None              
        add_subject_arg = None            

        if subject_id is not None:
            cand = f"subj_{subject_id}"
            if cand in avail_heads:
                desired_key = cand
                add_subject_arg = subject_id

        if desired_key is None and isinstance(head_id, str) and not head_id.isdigit():
            cand = f"subj_{head_id}"
            if cand in avail_heads:
                desired_key = cand
                add_subject_arg = head_id

        if desired_key is None and head_id is not None and \
        (isinstance(head_id, (int, np.integer)) or (isinstance(head_id, str) and head_id.isdigit())):
            desired_key = str(head_id)

        # Route by cluster if we have a subject_id
        if desired_key is None and subject_id is not None and \
        self.cluster_wrapper and self.cfg.model.n_clusters_pretrained > 1:
            rep = self._get_subject_repr(subject_id)
            cid = int(self.cluster_wrapper.model.predict(rep[None, :])[0])
            desired_key = str(cid)

        if desired_key is None:
            desired_key = "0"

        m = self._build_model(n_ch, win, self.cfg.model.n_clusters_pretrained)

        if add_subject_arg is not None:
            _ = m.add_new_head(add_subject_arg)
            
        m.load_state_dict(state, strict=False)
        m.to(self.device)
        m.eval()

        if desired_key.startswith("subj_") and desired_key not in m.heads:
            _ = m.add_new_head(add_subject_arg if add_subject_arg is not None else desired_key.replace("subj_", ""))

        loader = DataLoader(TLSubjectDataset(Xnp, np.zeros(len(Xnp), dtype=int)),
                            batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for Xb, _ in loader:
                Xb = Xb.to(self.device)
                out = m(Xb, [desired_key] * len(Xb))
                preds.extend(out.argmax(dim=1).cpu().numpy())
        return preds
