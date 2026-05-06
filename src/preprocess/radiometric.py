"""Radiometric normalization placeholders.

Phase 0 keeps this minimal — clipping + simple per-band rescaling. The interface
matches what a Carson-pipeline-output reader would provide later (Rice 2023
Mastcam-Z calibration target, HiRISE Lambertian assumption).
"""
import numpy as np


def clip_reflectance(arr: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=hi, neginf=lo), lo, hi).astype(np.float32)


def normalize_per_band(arr: np.ndarray, axis: tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    mean = a.mean(axis=axis, keepdims=True)
    std = a.std(axis=axis, keepdims=True) + 1e-6
    return ((a - mean) / std).astype(np.float32)


def apply_atmospheric_proxy(reflectance: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Toy atmospheric correction: reflectance / exp(-tau).

    Beer-Lambert proxy. Real correction (DISORT / volcano-scan) is out of scope
    for Phase 0 — see ATM_INVERSION_DESIGN.md for the full treatment in idea 05.
    """
    if reflectance.ndim < 1:
        raise ValueError("reflectance array must be at least 1D")
    tau = np.asarray(tau, dtype=np.float32).reshape((-1,) + (1,) * (reflectance.ndim - 1))
    return (reflectance / np.exp(-tau)).astype(np.float32)
