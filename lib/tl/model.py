import numpy as np
import torch
import torch.nn as nn
from lib.mtl.model import MultiTaskDeep4Net


class TLModel(nn.Module):
    """
    TL Model with MTL backbone
    """
    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_clusters_pretrained: int,
        window_samples: int,
        head_kwargs: dict | None = None,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_clusters_pretrained = n_clusters_pretrained
        self.window_samples = window_samples

        self.head_kwargs = head_kwargs or {}
        self.hidden_dim = self.head_kwargs.get("hidden_dim", 128)
        self.dropout = self.head_kwargs.get("dropout", 0.5)

        self.mtl_net = MultiTaskDeep4Net(
            n_chans=self.n_chans,
            n_outputs=self.n_outputs,
            n_clusters=self.n_clusters_pretrained,
            backbone_kwargs={"n_times": window_samples},
            head_kwargs={
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
            },
        )

        self._id_map: dict[object, int] = {}
        self._next_id: int = 0

    def _encode_id(self, raw_id: object) -> int:
        """Map any hashable raw_id to a unique small integer index."""
        if raw_id not in self._id_map:
            self._id_map[raw_id] = self._next_id
            self._next_id += 1
        return self._id_map[raw_id]

    def forward(self, x: torch.Tensor, ids) -> torch.Tensor:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        elif not isinstance(ids, (list, tuple)):
            ids = [ids] * x.shape[0]

        feats = self.mtl_net.shared_backbone(x)
        outs = []
        route_counts = {"cluster": 0, "subject_direct": 0, "subject_map": 0}

        for i, raw in enumerate(ids):
            if isinstance(raw, str) and raw.startswith("subj_"):
                key = raw
                route_counts["subject_direct"] += 1

            elif isinstance(raw, (int, np.integer)) or (isinstance(raw, str) and raw.isdigit()):
                key = str(int(raw))
                route_counts["cluster"] += 1

            else:
                sid = self._encode_id(raw)
                key = f"subj_{sid}"
                route_counts["subject_map"] += 1

            if key not in self.mtl_net.heads:
                raise KeyError(f"Head '{key}' not found")

            outs.append(self.mtl_net.heads[key](feats[i:i+1]))

        if not hasattr(self, "_route_debug_emitted"):
            self._route_debug_emitted = True

        return torch.cat(outs, dim=0)


    def add_new_head(
        self,
        subject_id: object,
        feature_dim: int | None = None,
        dummy_input: torch.Tensor | None = None,
    ) -> str:
        idx = self._encode_id(subject_id)
        if feature_dim is None:
            if dummy_input is None:
                device = next(self.parameters()).device
                dummy_input = torch.zeros(1, self.n_chans, self.window_samples, device=device)
            with torch.no_grad():
                feature_dim = self.mtl_net.shared_backbone(dummy_input).shape[1]

        mlp_head = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_outputs),
        )
        key = f"subj_{idx}"
        self.mtl_net.heads[key] = mlp_head
        return key

    @property
    def shared_backbone(self) -> nn.Module:
        return self.mtl_net.shared_backbone

    @property
    def heads(self) -> nn.ModuleDict:
        return self.mtl_net.heads
