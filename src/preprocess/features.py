"""Feature matrix assembly: HiRISE patch stats + DTM derivatives + (optional) geometry/τ."""
import numpy as np

from ..constants import N_HIRISE_BANDS


def hirise_patch_stats(X_hirise: np.ndarray) -> np.ndarray:
    """Mean / std / 25-50-75 percentile per band per patch."""
    if X_hirise.ndim != 4 or X_hirise.shape[-1] != N_HIRISE_BANDS:
        raise ValueError(f"expected (N,H,W,3), got {X_hirise.shape}")
    flat = X_hirise.reshape(X_hirise.shape[0], -1, X_hirise.shape[-1])
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    p25 = np.percentile(flat, 25, axis=1)
    p50 = np.percentile(flat, 50, axis=1)
    p75 = np.percentile(flat, 75, axis=1)
    return np.concatenate([mean, std, p25, p50, p75], axis=1).astype(np.float32)


def dtm_derivatives(X_dtm: np.ndarray) -> np.ndarray:
    """Slope magnitude + roughness (std of laplacian) + mean elevation."""
    if X_dtm.ndim != 3:
        raise ValueError(f"expected (N,H,W), got {X_dtm.shape}")
    n = X_dtm.shape[0]
    if X_dtm.shape[1] < 2 or X_dtm.shape[2] < 2:
        return np.zeros((n, 3), dtype=np.float32)
    gy, gx = np.gradient(X_dtm, axis=(1, 2))
    slope = np.sqrt(gx ** 2 + gy ** 2).reshape(n, -1).mean(axis=1)
    lap = np.abs(np.diff(X_dtm, axis=1, n=2)).reshape(n, -1).mean(axis=1) if X_dtm.shape[1] >= 3 else np.zeros(n)
    elev = X_dtm.reshape(n, -1).mean(axis=1)
    return np.stack([slope, lap, elev], axis=1).astype(np.float32)


def build_feature_matrix(
    X_hirise: np.ndarray,
    X_dtm: np.ndarray | None = None,
    X_geometry: np.ndarray | None = None,
    X_tau: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    n = X_hirise.shape[0]
    parts: list[np.ndarray] = []
    names: list[str] = []

    hstats = hirise_patch_stats(X_hirise)
    parts.append(hstats)
    band_labels = ["bg", "red", "nir"]
    for stat in ("mean", "std", "p25", "p50", "p75"):
        for b in band_labels:
            names.append(f"hirise_{b}_{stat}")

    if X_dtm is not None:
        d = dtm_derivatives(X_dtm)
        parts.append(d)
        names += ["dtm_slope", "dtm_roughness", "dtm_elev"]
        names += ["dtm_present_flag"]
        parts.append(np.ones((n, 1), dtype=np.float32))
    else:
        parts.append(np.zeros((n, 4), dtype=np.float32))
        names += ["dtm_slope", "dtm_roughness", "dtm_elev", "dtm_present_flag"]

    if X_geometry is not None:
        if X_geometry.shape != (n, 3):
            raise ValueError(f"geometry shape {X_geometry.shape} != ({n}, 3)")
        parts.append(X_geometry.astype(np.float32))
        parts.append(np.ones((n, 1), dtype=np.float32))
    else:
        parts.append(np.zeros((n, 3), dtype=np.float32))
        parts.append(np.zeros((n, 1), dtype=np.float32))
    names += ["geom_phase", "geom_incidence", "geom_emission", "geom_present_flag"]

    if X_tau is not None:
        if X_tau.shape != (n,):
            raise ValueError(f"tau shape {X_tau.shape} != ({n},)")
        parts.append(X_tau.astype(np.float32).reshape(n, 1))
        parts.append(np.ones((n, 1), dtype=np.float32))
    else:
        parts.append(np.zeros((n, 1), dtype=np.float32))
        parts.append(np.zeros((n, 1), dtype=np.float32))
    names += ["atm_tau", "atm_present_flag"]

    X_feat = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(X_feat).all():
        raise ValueError("feature matrix contains NaN or inf")
    return X_feat, names
