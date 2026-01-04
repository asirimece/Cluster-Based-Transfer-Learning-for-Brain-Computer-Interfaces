import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """
    Wraps EEG data, labels, and trial IDs for sub-epochs.
    """
    def __init__(self, epochs_or_data, labels=None, trial_ids=None, transform=None, subject_id=None, session_key=None):
        self.transform = transform
        
        if hasattr(epochs_or_data, "get_data"):
            data = epochs_or_data.get_data()  
            self.labels = epochs_or_data.events[:, -1]
            self.trial_ids = epochs_or_data.events[:, 1]
        else:
            data = epochs_or_data
            self.labels = labels
            self.trial_ids = trial_ids
        
        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(self.labels, dtype=torch.long)
        self.trial_ids = torch.tensor(self.trial_ids, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = (self.X[idx], self.y[idx], self.trial_ids[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class EEGMultiTaskDataset(Dataset):
    """
    MTL Dataset.
    """
    def __init__(self, data, labels, subject_ids, cluster_wrapper):
        self.data = data
        self.labels = labels
        self.subject_ids = subject_ids
        self.cluster_wrapper = cluster_wrapper

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        subject_id = self.subject_ids[index]
        cluster_id = self.cluster_wrapper.get_cluster_for_subject(subject_id)
        return sample, label, subject_id, cluster_id


class TLSubjectDataset(Dataset):
    """
    TL Dataset.
    """
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
