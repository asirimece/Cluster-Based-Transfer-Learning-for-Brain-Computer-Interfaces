import numpy as np
import pickle
from lib.pipeline.cluster.methods import (
    evaluate_k_means,
    kmeans_clustering,
    hierarchical_clustering,
    dbscan_clustering,
)


class ClusterWrapper:
    def __init__(self, subject_ids, labels, model, subject_representations):
        """
        Wraps clustering results.
        """
        self.subject_ids = list(subject_ids)   
        self.labels = dict(labels)
        self.model = model
        self.subject_representations = dict(subject_representations)

    def get_cluster_for_subject(self, subject_id):
        return self.labels.get(subject_id, None)

    def get_num_clusters(self):
        unique = np.unique(list(self.labels.values()))
        return len(unique)
    
    def predict(self, feature_vector: np.ndarray) -> int:
        """
        Assign a new feature‐vector to the nearest cluster learned during MTL.
        """
        cid = self.model.predict(feature_vector[None, :])[0]
        return int(cid)

    def summary(self):
        unique, counts = np.unique(list(self.labels.values()), return_counts=True)
        lines = [f"Cluster {u}: {c} subjects" for u, c in zip(unique, counts)]
        return "Clustering Summary:\n" + "\n".join(lines)


class SubjectClusterer:
    def __init__(self, features_file, config):
        self.features_file = features_file
        self.config = config
        self.features = self._load_features()
        self.subject_representations = self._compute_subject_representation()

    def _load_features(self):
        with open(self.features_file, "rb") as f:
            return pickle.load(f)

    def _compute_subject_representation(self):
        subject_repr = {}
        for subj, sessions in self.features.items():
            if isinstance(sessions, dict):
                combined_all = []
                for sess in sessions.values():
                    if isinstance(sess, dict) and 'combined' in sess:
                        combined_all.append(sess['combined'])
                if combined_all:
                    reps_concat = np.concatenate(combined_all, axis=0)
                    subject_repr[subj] = np.mean(reps_concat, axis=0)
                else:
                    raise ValueError(f"No valid 'combined' features for subject {subj}")
            # If sessions is already an array (legacy format)
            elif isinstance(sessions, np.ndarray):
                subject_repr[subj] = np.mean(sessions, axis=0)
            else:
                raise TypeError(f"Unsupported session type for subject {subj}: {type(sessions)}")
        return subject_repr


    def cluster_subjects(self, method=None):
        if method is None:
            method = self.config.get('method', 'kmeans')
        subject_ids = list(self.subject_representations.keys())
        X = np.stack([self.subject_representations[sid] for sid in subject_ids], axis=0)
        method = method.lower()
        if method == 'kmeans':
            params = self.config.get('kmeans', {})
            labels, model = kmeans_clustering(X, **params)
        elif method == 'hierarchical':
            params = self.config.get('hierarchical', {})
            labels, model = hierarchical_clustering(X, **params)
        elif method == 'dbscan':
            params = self.config.get('dbscan', {})
            labels, model = dbscan_clustering(X, **params)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Build a subject_id-label dict
        cluster_labels = {sid: lab for sid, lab in zip(subject_ids, labels)}
        return ClusterWrapper(subject_ids, cluster_labels, model, self.subject_representations)
