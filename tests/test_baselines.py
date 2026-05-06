import numpy as np
import pytest

from src.data.synthetic import make_paired_dataset
from src.preprocess.features import build_feature_matrix
from src.model.baselines import REGISTRY, fit_predict, make_model


def _tiny_split(seed=42, n=80, patch=8):
    ds = make_paired_dataset(n_patches=n, patch_size=patch, seed=seed, noise=0.01)
    X, _ = build_feature_matrix(ds.X_hirise, ds.X_dtm, ds.X_geometry, ds.X_tau)
    y = ds.y_mastcamz
    cut = int(n * 0.8)
    return X[:cut], y[:cut], X[cut:], y[cut:]


@pytest.mark.parametrize("name", ["ridge", "random_forest", "xgboost", "mlp"])
def test_baseline_fit_predict_shape(name):
    Xtr, ytr, Xte, yte = _tiny_split()
    res = fit_predict(name, Xtr, ytr, Xte, yte)
    assert res.y_pred.shape == yte.shape
    assert np.isfinite(res.y_pred).all()


def test_registry_has_four_models():
    assert set(REGISTRY) == {"ridge", "random_forest", "xgboost", "mlp"}


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        make_model("svm")


def test_ridge_learns_signal():
    Xtr, ytr, Xte, yte = _tiny_split(seed=7, n=160, patch=8)
    res = fit_predict("ridge", Xtr, ytr, Xte, yte)
    nir_band = 8
    sse = ((res.y_pred[:, nir_band] - yte[:, nir_band]) ** 2).sum()
    sst = ((yte[:, nir_band] - yte[:, nir_band].mean()) ** 2).sum() + 1e-9
    r2 = 1.0 - sse / sst
    assert r2 > 0.0, f"ridge failed to learn even non-negative R² on synthetic NIR band: {r2:.3f}"
