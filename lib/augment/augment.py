import numpy as np
import torch
from scipy import signal

def add_gaussian_noise(epoch: np.ndarray, sigma: float) -> np.ndarray:
    """
    per-channel std (over time)
    """
    ch_std = epoch.std(axis=1, keepdims=True) + 1e-8
    noise = np.random.randn(*epoch.shape) * (ch_std * sigma)
    return epoch + noise

def time_warp(epoch: np.ndarray, warp_ratio: float, max_seg: float=0.5):
    ch, T = epoch.shape
    seg_len = max(1, int(T * max_seg))
    if seg_len >= T:  # guard
        return epoch
    start = np.random.randint(0, T - seg_len)
    factor = 1.0 + np.random.uniform(-warp_ratio, warp_ratio)
    warped = signal.resample(epoch[:, start:start+seg_len], max(1, int(seg_len * factor)), axis=1)
    # pad/crop back to seg_len
    if warped.shape[1] < seg_len:
        pad = np.zeros((epoch.shape[0], seg_len - warped.shape[1]), dtype=epoch.dtype)
        warped = np.concatenate([warped, pad], axis=1)
    else:
        warped = warped[:, :seg_len]
    return np.concatenate([epoch[:, :start], warped, epoch[:, start+seg_len:]], axis=1)

def frequency_shift(epoch: np.ndarray, fs: float, shift_hz: float):
    """
    Hilbert-based, avoids circular wrap-around.
    """
    ch, T = epoch.shape
    t = np.arange(T, dtype=np.float64) / float(fs)
    analytic = signal.hilbert(epoch, axis=1)
    carrier  = np.exp(1j * 2.0 * np.pi * shift_hz * t)[None, :]
    shifted  = np.real(analytic * carrier)
    return shifted

def mixup_batch_torch(X, y, alpha: float):
    """
    Torch-native mixup for CE loss.
    X: (B, C, T) or (B, D) torch.Tensor
    y: (B,) torch.LongTensor
    Returns Xmix, y_a, y_b, lam(float)
    """
    if alpha <= 0:
        return X, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    Xmix = lam * X + (1 - lam) * X[idx]
    return Xmix, y, y[idx], lam

def apply_raw_augmentations(X: np.ndarray, cfg: dict) -> np.ndarray:
    out = []
    do_noise = cfg["gaussian_noise"]["enabled"]
    do_warp  = cfg["time_warp"]["enabled"]
    do_shift = cfg["frequency_shift"]["enabled"]
    for epoch in X:
        e = epoch
        if do_noise:
            e = add_gaussian_noise(e, cfg["gaussian_noise"]["sigma"])
        if do_warp:
            tp = cfg["time_warp"]
            e = time_warp(e, tp["warp_ratio"], tp["max_seg"])
        if do_shift:
            fsf = cfg["frequency_shift"]
            e = frequency_shift(e, fsf["fs"], fsf["shift_hz"])
        out.append(e)
    return np.stack(out, axis=0)
