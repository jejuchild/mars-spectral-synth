"""End-to-end Phase 0 pipeline: synthetic or real-data dry-run."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.constants import CORE_BAND_INDICES, MASTCAMZ_BAND_CENTERS_NM
from src.data.real_reader import discover_carson_pipeline
from src.data.synthetic import make_paired_dataset
from src.eval.metrics import summarize, train_test_split_indices
from src.model.baselines import REGISTRY, fit_predict
from src.preprocess.features import build_feature_matrix
from src.preprocess.radiometric import clip_reflectance
from src.utils.checks import RuntimeBudget, assert_cpu_only
from src.utils.io import dump_npz, ensure_dir, read_yaml, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mars-spectral-synth Phase 0 pipeline")
    p.add_argument("--config", default="configs/phase0.yaml")
    p.add_argument("--synthetic", action="store_true", help="Run on synthetic data (default)")
    p.add_argument("--real", action="store_true", help="Run on real Carson-pipeline output")
    p.add_argument("--dry-run", action="store_true", help="Real mode: only print manifest")
    p.add_argument("--hirise", default=None)
    p.add_argument("--mastcamz", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def run_synthetic(cfg: dict, log) -> int:
    budget = RuntimeBudget(cfg["runtime"]["budget_seconds"])
    syn = cfg["data"]["synthetic"]
    log.info("Generating synthetic dataset n=%d patch=%d noise=%.3f", syn["n_patches"], syn["patch_size"], syn["noise"])
    ds = make_paired_dataset(
        n_patches=syn["n_patches"],
        patch_size=syn["patch_size"],
        noise=syn["noise"],
        seed=cfg["seed"],
    )

    X_hirise = clip_reflectance(ds.X_hirise)
    X_feat, names = build_feature_matrix(
        X_hirise,
        ds.X_dtm if cfg["preprocess"]["use_dtm"] else None,
        ds.X_geometry if cfg["preprocess"]["use_geometry"] else None,
        ds.X_tau if cfg["preprocess"]["use_atmospheric_tau"] else None,
    )
    log.info("Feature matrix shape=%s features=%d", X_feat.shape, len(names))

    tr_idx, te_idx = train_test_split_indices(
        n=X_feat.shape[0],
        test_size=cfg["eval"]["test_size"],
        seed=cfg["seed"],
    )
    Xtr, ytr = X_feat[tr_idx], ds.y_mastcamz[tr_idx]
    Xte, yte = X_feat[te_idx], ds.y_mastcamz[te_idx]

    artifacts = ensure_dir(ROOT / cfg["output"]["artifacts_dir"])
    metrics_all = {}
    preds_all = {}
    for name in cfg["model"]["baselines"]:
        kwargs = cfg["model"].get(name, {})
        kwargs = {k: v for k, v in kwargs.items() if k != "early_stopping"} if name == "mlp" else kwargs
        kwargs = cfg["model"].get(name, {}) if name != "mlp" else cfg["model"][name]
        log.info("Training baseline=%s ...", name)
        t0 = time.perf_counter()
        res = fit_predict(name, Xtr, ytr, Xte, yte, **kwargs)
        elapsed = time.perf_counter() - t0
        m = summarize(res.y_true, res.y_pred, core_band_indices=cfg["eval"]["core_band_indices"])
        m["train_seconds"] = round(elapsed, 3)
        metrics_all[name] = m
        preds_all[name] = res.y_pred
        log.info(
            "  %s: mean_r2_core=%.3f mean_r2_all=%.3f sam=%.3f rad train=%.2fs",
            name, m["mean_r2_core"], m["mean_r2_all"], m["sam_mean_rad"], elapsed,
        )
        budget.check()

    write_json(metrics_all, artifacts / "metrics.json")
    dump_npz(
        artifacts / "predictions.npz",
        y_true=yte,
        band_centers=MASTCAMZ_BAND_CENTERS_NM,
        core_band_indices=np.array(cfg["eval"]["core_band_indices"], dtype=np.int32),
        **{f"y_pred_{k}": v for k, v in preds_all.items()},
    )
    log.info("Wrote metrics.json + predictions.npz to %s", artifacts)

    threshold = cfg["eval"]["success_threshold"]["mean_r2_core_synthetic"]
    best = max(metrics_all.values(), key=lambda d: d["mean_r2_core"])
    best_name = [k for k, v in metrics_all.items() if v is best][0]
    if best["mean_r2_core"] >= threshold:
        log.info("PASS: best=%s mean_r2_core=%.3f >= %.2f", best_name, best["mean_r2_core"], threshold)
        return 0
    log.error("FAIL: best=%s mean_r2_core=%.3f < %.2f", best_name, best["mean_r2_core"], threshold)
    return 1


def run_real(cfg: dict, args: argparse.Namespace, log) -> int:
    real = cfg["data"]["real"]
    h = args.hirise or real["hirise_dir"]
    m = args.mastcamz or real["mastcamz_dir"]
    log.info("Real-data manifest discovery: hirise=%s mastcamz=%s", h, m)
    manifest = discover_carson_pipeline(
        hirise_dir=h,
        mastcamz_dir=m,
        spice_kernel=real.get("spice_kernel"),
        crism_dir=real.get("crism_dir"),
        themis_dir=real.get("themis_dir"),
        meda_dir=real.get("meda_dir"),
    )
    log.info("Manifest: hirise_files=%d mastcamz_files=%d paired_estimate=%d",
             manifest.hirise_files, manifest.mastcamz_files, manifest.paired_estimate)
    log.info("  spice=%s crism=%s themis=%s meda=%s",
             manifest.spice_kernel_present, manifest.crism_present,
             manifest.themis_present, manifest.meda_present)
    if manifest.missing_dirs:
        log.warning("Missing input dirs: %s", manifest.missing_dirs)
    if args.dry_run:
        log.info("Dry-run only — full real ingest is Carson follow-up (Phase 1+).")
        return 0
    log.error("Real-data full ingest not implemented in Phase 0 prototype.")
    raise NotImplementedError("Phase 0 prototype: real ingest is Carson follow-up; pass --dry-run.")


def main() -> int:
    assert_cpu_only()
    args = parse_args()
    cfg_path = ROOT / args.config
    cfg = read_yaml(cfg_path)
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_global_seed(cfg["seed"])
    log_dir = ROOT / cfg["output"]["log_dir"]
    log = setup_logging(log_dir=log_dir, name="spectral_synth")
    log.info("config=%s seed=%d", cfg_path.name, cfg["seed"])

    if args.real:
        cfg["data"]["mode"] = "real"
        return run_real(cfg, args, log)
    cfg["data"]["mode"] = "synthetic"
    return run_synthetic(cfg, log)


if __name__ == "__main__":
    raise SystemExit(main())
