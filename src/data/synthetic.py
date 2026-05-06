"""Deterministic synthetic paired HiRISE + DTM + Mastcam-Z dataset.

Design intent (mirrors `context/SPECTRAL_SYNTH_DESIGN.md` §6 Track A):
  - Each patch has a latent "mineral mix" vector that controls the spectrum.
  - HiRISE (3-band) sees broad averages of the spectrum.
  - Mastcam-Z (11-band) is the supervision: sub-band reflectance.
  - DTM and (zero-fill) geometry/tau provide auxiliary structure.
The relationship between HiRISE and Mastcam-Z is non-trivial but learnable —
that is the Phase 0 prototype contract, not a claim about the real Mars problem.
"""
from dataclasses import dataclass, field

import numpy as np

from ..constants import (
    HIRISE_BAND_CENTERS_NM,
    MASTCAMZ_BAND_CENTERS_NM,
    N_HIRISE_BANDS,
    N_MASTCAMZ_BANDS,
)


@dataclass
class SyntheticDataset:
    X_hirise: np.ndarray   # (N, P, P, 3)
    X_dtm: np.ndarray      # (N, P, P)
    X_geometry: np.ndarray # (N, 3) phase, incidence, emission (deg)
    X_tau: np.ndarray      # (N,) atmospheric opacity
    y_mastcamz: np.ndarray # (N, 11) per-patch median reflectance label
    band_centers_nm: np.ndarray = field(default_factory=lambda: MASTCAMZ_BAND_CENTERS_NM.copy())


def _gaussian_basis(centers_nm: np.ndarray, peak_nm: float, width_nm: float) -> np.ndarray:
    return np.exp(-0.5 * ((centers_nm - peak_nm) / width_nm) ** 2)


def make_paired_dataset(
    n_patches: int = 200,
    patch_size: int = 32,
    noise: float = 0.02,
    seed: int = 42,
) -> SyntheticDataset:
    if n_patches <= 0:
        raise ValueError("n_patches must be > 0")
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")
    rng = np.random.default_rng(seed)

    components = np.array([
        _gaussian_basis(MASTCAMZ_BAND_CENTERS_NM, peak_nm=500.0, width_nm=120.0),
        _gaussian_basis(MASTCAMZ_BAND_CENTERS_NM, peak_nm=700.0, width_nm=80.0),
        _gaussian_basis(MASTCAMZ_BAND_CENTERS_NM, peak_nm=900.0, width_nm=70.0),
        _gaussian_basis(MASTCAMZ_BAND_CENTERS_NM, peak_nm=620.0, width_nm=40.0),
        _gaussian_basis(MASTCAMZ_BAND_CENTERS_NM, peak_nm=860.0, width_nm=50.0),
    ], dtype=np.float32)
    components = components / components.max(axis=1, keepdims=True)

    weights = rng.dirichlet(np.ones(components.shape[0]) * 0.6, size=n_patches).astype(np.float32)
    base_albedo = rng.uniform(0.10, 0.35, size=(n_patches, 1)).astype(np.float32)
    spectra = base_albedo * (weights @ components)
    spectra = np.clip(spectra, 0.0, 1.0)

    hirise_widths = np.array([200.0, 100.0, 150.0], dtype=np.float32)
    hirise_filters = np.stack([
        _gaussian_basis(MASTCAMZ_BAND_CENTERS_NM, peak_nm=HIRISE_BAND_CENTERS_NM[i], width_nm=hirise_widths[i])
        for i in range(N_HIRISE_BANDS)
    ], axis=0)
    hirise_filters = hirise_filters / hirise_filters.sum(axis=1, keepdims=True)
    hirise_mean = spectra @ hirise_filters.T  # (N, 3)

    P = patch_size
    spatial_var = rng.normal(0.0, 0.015, size=(n_patches, P, P, 1)).astype(np.float32)
    X_hirise = hirise_mean[:, None, None, :] + spatial_var
    X_hirise += rng.normal(0.0, noise * 0.5, size=X_hirise.shape).astype(np.float32)
    X_hirise = np.clip(X_hirise, 0.0, 1.0).astype(np.float32)

    elev_base = rng.uniform(-2500.0, -1500.0, size=(n_patches, 1, 1)).astype(np.float32)
    yy, xx = np.mgrid[0:P, 0:P].astype(np.float32)
    slope_x = rng.normal(0.0, 0.5, size=(n_patches, 1, 1)).astype(np.float32)
    slope_y = rng.normal(0.0, 0.5, size=(n_patches, 1, 1)).astype(np.float32)
    X_dtm = elev_base + slope_x * xx + slope_y * yy
    X_dtm += rng.normal(0.0, 0.3, size=X_dtm.shape).astype(np.float32)

    X_geometry = np.zeros((n_patches, 3), dtype=np.float32)
    X_geometry[:, 0] = rng.uniform(10.0, 80.0, size=n_patches)
    X_geometry[:, 1] = rng.uniform(20.0, 60.0, size=n_patches)
    X_geometry[:, 2] = rng.uniform(0.0, 30.0, size=n_patches)

    X_tau = rng.uniform(0.3, 1.2, size=n_patches).astype(np.float32)

    y_mastcamz = spectra + rng.normal(0.0, noise, size=spectra.shape).astype(np.float32)
    y_mastcamz = np.clip(y_mastcamz, 0.0, 1.0).astype(np.float32)

    return SyntheticDataset(
        X_hirise=X_hirise,
        X_dtm=X_dtm.astype(np.float32),
        X_geometry=X_geometry,
        X_tau=X_tau,
        y_mastcamz=y_mastcamz,
    )
