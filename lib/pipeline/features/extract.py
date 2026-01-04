import os
import numpy as np
import mne
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.preprocessing import StandardScaler
import pickle
from lib.logging import logger
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV

logger = logger.get()


class FeatureExtractor:
    def __init__(self, feat_cfg: DictConfig):
        self.config = feat_cfg

    def extract_erd_ers(self, epochs):
        kwargs = self.config.methods[0].get('kwargs', {}) if self.config.methods else {}
        baseline_window = kwargs.get('baseline_window', [0.0, 0.5])
        analysis_window = kwargs.get('analysis_window', [0.5, 4.0])
        frequency_bands = kwargs.get('frequency_bands', {'mu': [8, 12], 'beta': [13, 30]})

        sfreq = epochs.info['sfreq']
        times = epochs.times
        data = epochs.get_data().astype(np.float64)

        def time_to_index(sec):
            sec_clamped = np.clip(sec, times[0], times[-1])
            return np.searchsorted(times, sec_clamped)

        b_start = time_to_index(baseline_window[0])
        b_end = time_to_index(baseline_window[1])
        a_start = time_to_index(analysis_window[0])
        a_end = time_to_index(analysis_window[1])

        n_epochs = data.shape[0]
        n_bands = len(frequency_bands)
        features = np.zeros((n_epochs, n_bands))
        band_names = list(frequency_bands.keys())

        for i, band in enumerate(band_names):
            low, high = frequency_bands[band]
            band_data = data.copy()
            for ep in range(n_epochs):
                band_data[ep] = mne.filter.filter_data(
                    band_data[ep], sfreq=sfreq,
                    l_freq=low, h_freq=high, verbose=False
                )
            baseline_power = np.mean(band_data[:, :, b_start:b_end] ** 2, axis=(1,2))
            analysis_power = np.mean(band_data[:, :, a_start:a_end] ** 2, axis=(1,2))
            baseline_power = np.maximum(baseline_power, 1e-12)
            features[:, i] = (analysis_power - baseline_power) / baseline_power * 100.0

        return features

    def extract_csp(self, epochs):
        kwargs = self.config.methods[0].get('kwargs', {}) if self.config.methods else {}
        n_comp = kwargs.get('n_components', 4)
        labels = epochs.events[:, -1]
        data = epochs.get_data().astype(np.float64)

        if len(np.unique(labels)) < 2:
            raise ValueError("CSP requires at least 2 classes.")

        csp = CSP(n_components=n_comp, reg=None, norm_trace=False)
        return csp.fit_transform(data, labels)

    def extract_fbcsp(self, epochs):
        kwargs = self.config.methods[0].get('kwargs', {}) if self.config.methods else {}
        freq_bands = kwargs.get('frequency_bands', [[4, 8], [8, 12], [12, 16]])
        n_per_band = kwargs.get('n_components_per_band', 2)
        sfreq = epochs.info['sfreq']
        labels = epochs.events[:, -1]
        raw_data = epochs.get_data().astype(np.float64)

        all_feats = []
        for low, high in freq_bands:
            band_data = raw_data.copy()
            for ep in range(band_data.shape[0]):
                band_data[ep] = mne.filter.filter_data(
                    band_data[ep], sfreq=sfreq,
                    l_freq=low, h_freq=high, verbose=False
                )
            csp = CSP(n_components=n_per_band, norm_trace=False)
            all_feats.append(csp.fit_transform(band_data, labels))

        return np.concatenate(all_feats, axis=1)

    def extract_riemannian(self, epochs):
        kwargs = self.config.methods[1].get('kwargs', {}) if len(self.config.methods) > 1 else {}
        estimator = kwargs.get('estimator', 'oas')
        mapping = kwargs.get('mapping', 'tangent')

        data = epochs.get_data().astype(np.float64)
        cov = Covariances(estimator=estimator)
        mats = cov.fit_transform(data)

        if mapping == 'tangent':
            ts = TangentSpace()
            return ts.fit_transform(mats)
        else:
            n_ep, n_ch, _ = mats.shape
            tri_idx = np.triu_indices(n_ch)
            feats = np.zeros((n_ep, len(tri_idx[0])))
            for i in range(n_ep):
                feats[i] = mats[i][tri_idx]
            return feats

    def feature_extraction(self, epochs):
        feats = []
        for m in self.config.methods:
            name = m['name']
            if name == 'erd_ers':
                feats.append(self.extract_erd_ers(epochs))
            elif name == 'csp':
                feats.append(self.extract_csp(epochs))
            elif name == 'fbcsp':
                feats.append(self.extract_fbcsp(epochs))
            elif name == 'riemannian':
                feats.append(self.extract_riemannian(epochs))
            else:
                logger.warning(f"Unsupported method {name}")

        combined = np.concatenate(feats, axis=1)

        labels = epochs.events[:, -1]
        return combined, labels

    @staticmethod
    def run(config: DictConfig, preprocessed_data, mode: str = "in-session", save=True):
        """
        Change mode to "in-session" for target dataset.
        """
        feat_cfg = config.transform.feature_extraction
        extractor = FeatureExtractor(feat_cfg)
        
        raw_feats = {}
        for subj, sessions in preprocessed_data.items():
            raw_feats[subj] = {}
            for sess_label, epochs in sessions.items():
                X_raw, y = extractor.feature_extraction(epochs)
                raw_feats[subj][sess_label] = {"combined": X_raw, "labels": y}

        X_train_list, y_train_list = [], []
        for subj, sessions in raw_feats.items():
            for sess_label, d in sessions.items():
                if sess_label.endswith("train"):
                    X_train_list.append(d["combined"])
                    y_train_list.append(d["labels"])
        X_pool = np.vstack(X_train_list)
        y_pool = np.concatenate(y_train_list)

        scaler = StandardScaler().fit(X_pool)
        
        X_pool = scaler.transform(X_pool)
        
        for subj, sessions in raw_feats.items():
            for label, d in sessions.items():
                d['combined'] = scaler.transform(d['combined'])

        D_pool = X_pool.shape[1]
        pca_cfg = getattr(config.transform, 'dimensionality_reduction', None)
        threshold = getattr(pca_cfg, 'threshold', 300) if pca_cfg else 300
        if D_pool > threshold or pca_cfg is not None:
            if pca_cfg and 'n_components' in pca_cfg.kwargs:
                n_comp = pca_cfg.kwargs.n_components
                pca = PCA(n_components=n_comp)
            else:
                var = pca_cfg.kwargs.get('explained_variance', 0.95) if pca_cfg else 0.95
                pca = PCA(n_components=var)
            pca.fit(X_pool)
            X_pool = pca.transform(X_pool)
            for subj, sessions in raw_feats.items():
                for label, d in sessions.items():
                    d['combined'] = pca.transform(d['combined'])
        else:
            logger.info(f"Skipped PCA: {D_pool} dims <= {threshold}")

        fs_cfg = config.transform.feature_selection.kwargs
        mask_fp = config.transform.feature_selection.mask_file
        if mode == "cross-subject":
            svc = SVC(kernel='linear', C=fs_cfg.get('svc_C', 1.0), random_state=42)
            rfecv = RFECV(
                estimator=svc,
                step=fs_cfg.get('step', 5),
                cv=fs_cfg.get('cv', 3),
                scoring=fs_cfg.get('scoring', 'accuracy'),
                min_features_to_select=fs_cfg.get('min_features_to_select', 1),
                n_jobs=-1
            )
            logger.info(f"Fitting RFECV on pooled train: {X_pool.shape} \n This will take a while!")
            rfecv.fit(X_pool, y_pool)
            logger.info(f"RFECV complete: {rfecv.n_features_} features selected")
     
            with open(mask_fp, "wb") as mf:
                pickle.dump(rfecv.support_, mf)
            logger.info(f"Saved feature‐selection mask → {mask_fp}")
            mask = rfecv.support_
        else:
            with open(mask_fp, "rb") as mf:
                mask = pickle.load(mf)

        selected = {}
        for subj, sessions in raw_feats.items():
            selected[subj] = {}
            for label, d in sessions.items():
                X_sel = d['combined'][:, mask]
                selected[subj][label] = {"combined": X_sel, "labels": d['labels']}

        if save:
            out_dir = getattr(config.transform.feature_selection, "out_dir", ".")
            fname = f"features_{mode.replace('-', '_')}.pkl"
            out_path = os.path.join(out_dir, fname)
            with open(out_path, "wb") as f:
                pickle.dump(selected, f)
            logger.info(f"Saved features → {out_path}")

        return selected

def save_features(features, filename):
    with open(filename, 'wb') as f:
        pickle.dump(features, f)
