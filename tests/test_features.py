import numpy as np
import pytest

from src.data.synthetic import make_paired_dataset
from src.preprocess.features import build_feature_matrix, hirise_patch_stats, dtm_derivatives
from src.preprocess.radiometric import clip_reflectance, normalize_per_band, apply_atmospheric_proxy


def test_feature_matrix_dim_and_no_nan():
    ds = make_paired_dataset(n_patches=20, patch_size=8, seed=0)
    X_feat, names = build_feature_matrix(ds.X_hirise, ds.X_dtm, ds.X_geometry, ds.X_tau)
    assert X_feat.shape[0] == 20
    assert X_feat.shape[1] == len(names)
    assert X_feat.shape[1] == 5 * 3 + 4 + 4 + 2  # hstats + dtm + geom + tau
    assert np.isfinite(X_feat).all()


def test_feature_matrix_deterministic():
    a = make_paired_dataset(n_patches=5, patch_size=8, seed=99)
    b = make_paired_dataset(n_patches=5, patch_size=8, seed=99)
    fa, _ = build_feature_matrix(a.X_hirise, a.X_dtm, a.X_geometry, a.X_tau)
    fb, _ = build_feature_matrix(b.X_hirise, b.X_dtm, b.X_geometry, b.X_tau)
    np.testing.assert_array_equal(fa, fb)


def test_feature_matrix_optional_inputs_zero_filled():
    ds = make_paired_dataset(n_patches=8, patch_size=8, seed=1)
    X_feat, names = build_feature_matrix(ds.X_hirise, X_dtm=None, X_geometry=None, X_tau=None)
    assert "dtm_present_flag" in names
    flag_idx = names.index("dtm_present_flag")
    assert (X_feat[:, flag_idx] == 0.0).all()


def test_clip_reflectance_handles_nan_and_out_of_range():
    arr = np.array([[-0.1, 0.5, 1.2, np.nan, np.inf, -np.inf]], dtype=np.float32)
    out = clip_reflectance(arr)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert np.isfinite(out).all()


def test_normalize_per_band_zero_mean_unit_std():
    arr = np.random.default_rng(5).normal(size=(4, 8, 8, 3)).astype(np.float32)
    out = normalize_per_band(arr)
    np.testing.assert_allclose(out.mean(axis=(0, 1, 2)), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=(0, 1, 2)), 1.0, atol=1e-3)


def test_apply_atmospheric_proxy_increases_signal():
    refl = np.full((4, 8, 8, 3), 0.3, dtype=np.float32)
    tau = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    out = apply_atmospheric_proxy(refl, tau)
    assert out.shape == refl.shape
    assert (out > refl).all()


def test_hirise_patch_stats_shape_check():
    bad = np.zeros((4, 8, 8, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        hirise_patch_stats(bad)


def test_dtm_derivatives_handles_tiny_patch():
    tiny = np.zeros((3, 1, 1), dtype=np.float32)
    out = dtm_derivatives(tiny)
    assert out.shape == (3, 3)
    assert (out == 0).all()
