import os
import glob
import json
import argparse
import torch
import joblib
from sklearn.pipeline import Pipeline
from torch import nn
from lib.tl.model import TLModel


class Deep4NetTLWrapper:
    """
    TL Pipeline wrapper to export pipeline for the EEG-Recorder developed by CAI
    """
    def __init__(self, model_path, model_ctor_kwargs, device="cpu"):
        self.model_path        = model_path
        self.model_ctor_kwargs = model_ctor_kwargs
        self.device            = torch.device(device)
        self._load_model()

    def _load_model(self):
        self.model = TLModel(**self.model_ctor_kwargs).to(self.device)

        self.model.add_new_head(0)

        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)

        self.model.eval()


    def fit(self, X, y=None):
        return self

    def predict(self, X):
        import numpy as np
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32, device=self.device)
            batch = tx.shape[0]
            sids = [0]*batch
            out = self.model(tx, sids).cpu().numpy()
            return out.argmax(axis=1)

    def predict_proba(self, X):
        import numpy as np
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32, device=self.device)
            batch = tx.shape[0]
            sids = [0]*batch
            out = self.model(tx, sids).cpu().numpy()
            exp = np.exp(out - out.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)

def export_models(
    pth_dir: str,
    glob_pattern: str,
    device: str,
    n_chans: int,
    n_outputs: int,
    n_clusters_pretrained: int,
    window_samples: int,
    head_hidden_dim: int,
    head_dropout: float,
    cluster_mode: bool = False,
    pca_path: str = None,
    recordings: list = None
):

    model_ctor_kwargs = {
        "n_chans":               n_chans,
        "n_outputs":             n_outputs,
        "n_clusters_pretrained": n_clusters_pretrained,
        "window_samples":        window_samples,
        "head_kwargs": {
            "hidden_dim": head_hidden_dim,
            "dropout":    head_dropout
        }
    }

    pth_paths = glob.glob(os.path.join(pth_dir, glob_pattern))
    if not pth_paths:
        print(f"[!] No .pth files found in {pth_dir} matching {glob_pattern}")
        return

    for pth in pth_paths:
        print(f"[+] Processing {pth}")
        wrapper = Deep4NetTLWrapper(
            model_path=pth,
            model_ctor_kwargs=model_ctor_kwargs,
            device=device
        )
        pipe = Pipeline([("deep4net_tl", wrapper)])
        joblib_path = pth.replace(".pth", ".joblib")

        if not cluster_mode:
            joblib.dump(pipe, joblib_path)
            print(f"  → Saved pipeline to {joblib_path}")
        else:
            if pca_path is None or recordings is None:
                raise ValueError("cluster_mode=True requires --pca_path and --recordings")
            to_save = {
                "model": pipe,
                "metadata": {
                    "pca_model_path": pca_path,
                    "recordings":     recordings
                }
            }
            joblib.dump(to_save, joblib_path)
            print(f"  → Saved pipeline+metadata to {joblib_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export TL .pth models to sklearn .joblib pipelines for the Recorder"
    )
    parser.add_argument("dir", help="Directory containing .pth model files")
    parser.add_argument(
        "--pattern", default="*.pth",
        help="Glob pattern to match .pth (e.g. pooled_model.pth or base_model_cluster*.pth)"
    )
    parser.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    parser.add_argument("--n_chans",             type=int, required=True)
    parser.add_argument("--n_outputs",           type=int, required=True)
    parser.add_argument("--n_clusters_pretrained",type=int, required=True)
    parser.add_argument("--window_samples",      type=int, required=True)
    parser.add_argument("--head_hidden_dim",     type=int,   default=128)
    parser.add_argument("--head_dropout",        type=float, default=0.5)

    parser.add_argument(
        "--cluster_mode", action="store_true",
        help="Wrap each model with metadata (requires --pca_path and --recordings)"
    )
    parser.add_argument(
        "--pca_path", help="Path to the PCA .joblib used for mahalanobis"
    )
    parser.add_argument(
        "--recordings",
        help="JSON list of recordings (e.g. '[\"subj1_run1.fif\",\"subj2_run1.fif\"]')"
    )

    args = parser.parse_args()

    recs = None
    if args.cluster_mode:
        if not args.recordings:
            parser.error("--cluster_mode requires --recordings")
        recs = json.loads(args.recordings)

    export_models(
        pth_dir=args.dir,
        glob_pattern=args.pattern,
        device=args.device,
        n_chans=args.n_chans,
        n_outputs=args.n_outputs,
        n_clusters_pretrained=args.n_clusters_pretrained,
        window_samples=args.window_samples,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        cluster_mode=args.cluster_mode,
        pca_path=args.pca_path,
        recordings=recs
    )
