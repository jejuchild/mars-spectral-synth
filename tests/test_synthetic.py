import numpy as np
import pytest

from src.data.synthetic import make_paired_dataset
from src.data.real_reader import discover_carson_pipeline
from src.constants import N_HIRISE_BANDS, N_MASTCAMZ_BANDS


def test_synthetic_shapes():
    ds = make_paired_dataset(n_patches=10, patch_size=16, seed=42)
    assert ds.X_hirise.shape == (10, 16, 16, N_HIRISE_BANDS)
    assert ds.X_dtm.shape == (10, 16, 16)
    assert ds.X_geometry.shape == (10, 3)
    assert ds.X_tau.shape == (10,)
    assert ds.y_mastcamz.shape == (10, N_MASTCAMZ_BANDS)
    assert ds.X_hirise.dtype == np.float32
    assert ds.y_mastcamz.dtype == np.float32


def test_synthetic_deterministic():
    a = make_paired_dataset(n_patches=5, patch_size=8, seed=123)
    b = make_paired_dataset(n_patches=5, patch_size=8, seed=123)
    np.testing.assert_array_equal(a.X_hirise, b.X_hirise)
    np.testing.assert_array_equal(a.y_mastcamz, b.y_mastcamz)


def test_synthetic_value_range():
    ds = make_paired_dataset(n_patches=20, patch_size=8, seed=7)
    assert ds.X_hirise.min() >= 0.0
    assert ds.X_hirise.max() <= 1.0
    assert ds.y_mastcamz.min() >= 0.0
    assert ds.y_mastcamz.max() <= 1.0
    assert (ds.X_tau > 0).all() and (ds.X_tau < 2).all()


def test_synthetic_correlation_hirise_to_label():
    ds = make_paired_dataset(n_patches=200, patch_size=8, seed=42, noise=0.01)
    hirise_mean = ds.X_hirise.mean(axis=(1, 2))
    nir_band = ds.y_mastcamz[:, 8]
    corr = np.corrcoef(hirise_mean[:, 2], nir_band)[0, 1]
    assert corr > 0.3, f"HiRISE NIR ↔ Mastcam-Z 910nm correlation too weak: {corr:.3f}"


def test_synthetic_invalid_args():
    with pytest.raises(ValueError):
        make_paired_dataset(n_patches=0)
    with pytest.raises(ValueError):
        make_paired_dataset(patch_size=0)


def test_real_reader_dry_run_missing_dirs(tmp_path):
    manifest = discover_carson_pipeline(
        hirise_dir=tmp_path / "no_such_hirise",
        mastcamz_dir=tmp_path / "no_such_mastcamz",
    )
    assert len(manifest.missing_dirs) == 2
    assert manifest.hirise_files == 0
    assert manifest.mastcamz_files == 0
    assert manifest.paired_estimate == 0


def test_real_reader_dry_run_existing_empty(tmp_path):
    h = tmp_path / "hirise"
    m = tmp_path / "mastcamz"
    h.mkdir()
    m.mkdir()
    (h / "scene.lbl").write_text("dummy")
    manifest = discover_carson_pipeline(hirise_dir=h, mastcamz_dir=m)
    assert manifest.missing_dirs == []
    assert manifest.hirise_files == 1
    assert manifest.mastcamz_files == 0
    assert manifest.paired_estimate == 0
