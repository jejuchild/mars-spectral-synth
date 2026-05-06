import numpy as np
import pytest

from src.eval.metrics import per_band_r2, per_band_mae, spectral_angle_mapper, summarize, train_test_split_indices


def test_perfect_prediction_r2_one():
    y = np.random.default_rng(0).uniform(0, 1, size=(50, 11)).astype(np.float32)
    r2 = per_band_r2(y, y)
    assert np.allclose(r2, 1.0, atol=1e-5)


def test_zero_info_prediction_r2_near_zero_or_negative():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 1, size=(50, 11)).astype(np.float32)
    y_pred = np.full_like(y, y.mean())
    r2 = per_band_r2(y, y_pred)
    assert (r2 <= 1e-3).all()


def test_all_zero_truth_safe():
    y = np.zeros((20, 11), dtype=np.float32)
    y_pred = np.zeros_like(y)
    r2 = per_band_r2(y, y_pred)
    assert np.isfinite(r2).all()


def test_mae_zero_when_perfect():
    y = np.random.default_rng(2).uniform(0, 1, size=(30, 11)).astype(np.float32)
    assert per_band_mae(y, y).max() < 1e-6


def test_sam_zero_when_perfect():
    y = np.random.default_rng(3).uniform(0, 1, size=(30, 11)).astype(np.float32)
    sam = spectral_angle_mapper(y, y)
    assert sam.max() < 1e-3


def test_summarize_keys_and_core_mean():
    y = np.random.default_rng(4).uniform(0, 1, size=(40, 11)).astype(np.float32)
    res = summarize(y, y, core_band_indices=(3, 5, 6, 7, 8, 9, 10))
    assert set(res.keys()) >= {"per_band_r2", "per_band_mae", "sam_mean_rad", "mean_r2_core", "mean_r2_all"}
    assert abs(res["mean_r2_core"] - 1.0) < 1e-4


def test_split_indices_disjoint_and_full_coverage():
    tr, te = train_test_split_indices(100, test_size=0.2, seed=1)
    assert len(tr) + len(te) == 100
    assert len(set(tr).intersection(set(te))) == 0
    assert len(tr) == 80


def test_shape_mismatch_raises():
    y = np.zeros((10, 11), dtype=np.float32)
    bad = np.zeros((10, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        per_band_r2(y, bad)
