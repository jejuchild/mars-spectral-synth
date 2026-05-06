"""Per-band R² / SAM / MAE on Mastcam-Z 11-band reflectance prediction."""
from __future__ import annotations

import numpy as np


def per_band_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch {yt.shape} vs {yp.shape}")
    if yt.ndim != 2:
        raise ValueError(f"expected (N, B), got {yt.shape}")
    sse = ((yt - yp) ** 2).sum(axis=0)
    sst = ((yt - yt.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = np.where(sst > 1e-12, 1.0 - sse / np.where(sst > 1e-12, sst, 1.0), 0.0)
    return r2.astype(np.float32)


def per_band_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y_true) - np.asarray(y_pred)).mean(axis=0).astype(np.float32)


def spectral_angle_mapper(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape or yt.ndim != 2:
        raise ValueError("SAM expects (N, B) arrays of equal shape")
    dot = (yt * yp).sum(axis=1)
    nt = np.linalg.norm(yt, axis=1) + 1e-9
    np_ = np.linalg.norm(yp, axis=1) + 1e-9
    cos = np.clip(dot / (nt * np_), -1.0, 1.0)
    return np.arccos(cos).astype(np.float32)


def summarize(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    core_band_indices: tuple[int, ...] | list[int],
) -> dict:
    r2 = per_band_r2(y_true, y_pred)
    mae = per_band_mae(y_true, y_pred)
    sam = spectral_angle_mapper(y_true, y_pred)
    core = list(core_band_indices)
    core_r2 = float(r2[core].mean())
    return {
        "per_band_r2": r2.tolist(),
        "per_band_mae": mae.tolist(),
        "sam_mean_rad": float(sam.mean()),
        "sam_std_rad": float(sam.std()),
        "core_band_indices": core,
        "mean_r2_core": core_r2,
        "mean_r2_all": float(r2.mean()),
        "n_samples": int(y_true.shape[0]),
        "n_bands": int(y_true.shape[1]),
    }


def train_test_split_indices(n: int, test_size: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1.0 - test_size))
    return idx[:cut], idx[cut:]
