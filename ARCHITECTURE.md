# ARCHITECTURE — `mars-spectral-synth` Phase 0 prototype

작성: 2026-05-04 (Phase 2)
입력: PRD.md §7 preview + codex critique (`/tmp/codex-out-01-arch.txt`)
방침: codex 제안 tree 채택, **30분 phase 3 budget**에 맞춰 일부 모듈 합치고 plot/texture는 제외 (numeric metric only).

## 1. CRITICAL issue 처리

Codex가 conditional critical 5개 제시 — 모두 PRD 위반 아님:
- ❌ GPU/diffusion/foundation model 사용 → PRD §8 명시 금지
- ❌ real-data primary → PRD §5 G2/G3은 synthetic primary, real은 G5 dry-run only
- ❌ synthetic generator 부재 → PRD §6 step 1 명시
- ❌ monolithic script → 본 ARCHITECTURE 모듈 분리 명시
- ❌ 30분 cap risk → sub-phase 3a~3e 분할 + 각 ≤6분 budget

→ Phase 3 진행 시 critical block 없음.

## 2. Module tree (확정)

```
ideas/01-spectral-synth/
├── PRD.md                  # phase 1
├── ARCHITECTURE.md         # phase 2 (this)
├── README.md               # phase 5
├── DECISIONS.md            # phase 5
├── TODO.md                 # phase 5
├── requirements.txt        # phase 5
├── configs/
│   └── phase0.yaml         # runtime knobs (seed, sample count, model choice, paths)
├── src/
│   ├── __init__.py
│   ├── constants.py        # Mastcam-Z band centers (442~978 nm), 7 core indices
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic.py    # PRIMARY: deterministic synthetic generator (HiRISE+DTM+Mastcam-Z paired)
│   │   └── real_reader.py  # FOLLOW-UP: discover real PDS/Carson-pipeline output paths (dry-run only Phase 0)
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── radiometric.py  # clip/normalize, calibration target placeholder
│   │   └── features.py     # build feature matrix: 3-band stats + DTM slope/roughness + (optional) geometry/τ context
│   ├── model/
│   │   ├── __init__.py
│   │   └── baselines.py    # 4 baselines: ridge / random_forest / xgboost / mlp (sklearn-style fit/predict)
│   ├── eval/
│   │   ├── __init__.py
│   │   └── metrics.py      # per-band R² / SAM / MAE; 7-core mean R²; hold-out splitter; report writer
│   └── utils/
│       ├── __init__.py
│       ├── io.py           # ensure_dir / read_yaml / write_json / dump_npz
│       ├── logging.py      # setup_logging (console + file)
│       ├── seed.py         # set_global_seed (numpy + sklearn + xgboost + torch)
│       └── checks.py       # assert_cpu_only (no torch.cuda); assert_runtime_budget
├── tests/
│   ├── __init__.py
│   ├── test_synthetic.py   # synthetic generator: shape, determinism, ground-truth correlation
│   ├── test_features.py    # feature matrix: no NaN, expected dim, deterministic
│   ├── test_baselines.py   # each model: fit + predict on tiny synthetic
│   ├── test_metrics.py     # R²/SAM/MAE on known cases (perfect pred, zero-info, all-zero)
│   └── test_smoke_run.py   # integration: run.py --synthetic completes < 60s, mean R² > 0.4 on 7 core bands
├── artifacts/              # gitignored: model_*.pkl, metrics.json, predictions.npz, synth_dataset.npz
├── logs/                   # gitignored
└── run.py                  # CLI entrypoint: --synthetic | --real --hirise PATH --mastcamz PATH [--dry-run]
```

## 3. Data flow

### Synthetic (Phase 0 primary)

```
configs/phase0.yaml
  → utils.seed.set_global_seed(42)
  → data.synthetic.make_paired_dataset(n=200, patch=32, seed=42)
       returns:
         X_hirise   (N, 32, 32, 3)   float32 reflectance
         X_dtm      (N, 32, 32)      float32 elevation (m)
         y_mastcamz (N,  8,  8, 11)  float32 reflectance (downsampled label)
         meta       (N,)             dict: synthetic spectra params + noise level
  → preprocess.radiometric.normalize_reflectance (clip [0,1])
  → preprocess.features.build_feature_matrix
       returns:
         X_feat     (N, F)           F ≈ 24 (band stats + DTM slope/roughness + zero-fill geometry/τ + flag cols)
         y_flat     (N, 11)          per-patch median Mastcam-Z 11-band reflectance (or center-pixel)
  → model.baselines.train_all (ridge, rf, xgb, mlp)
       per-model checkpoint + per-band R²/SAM/MAE on hold-out
  → eval.metrics.summarize → metrics.json
  → utils.io.dump_npz → predictions.npz
```

### Real (Phase 0 dry-run only, full ingest = Carson follow-up)

```
data.real_reader.discover_carson_pipeline(data/HiRISE, data/Mastcam-Z)
  → returns manifest dict {hirise_count, mastcamz_count, paired_estimate, missing_dirs}
  → if --dry-run: print manifest + exit (G5 met)
  → else:        raise NotImplementedError("Phase 0 prototype: real ingest is Carson follow-up")
```

## 4. CPU-only guards (utils/checks.py)

1. `assert_cpu_only()` — `if torch is not None and torch.cuda.is_available(): raise RuntimeError`. Default "torch not imported in baselines."
2. `os.environ['CUDA_VISIBLE_DEVICES'] = ''` 강제 (utils/seed.py at import)
3. xgboost: `tree_method='hist'`, `device='cpu'`
4. lightgbm: `device='cpu'`
5. mlp baseline: sklearn `MLPRegressor` (CPU), torch는 base_model에서 사용 안 함 (req에서 fallback only)

## 5. Sub-phase split (30-min cap × 5 = total ≤150min for phase 3)

| sub-phase | 모듈 | acceptance | tag |
|---|---|---|---|
| **3a** | `data/synthetic.py` + `data/real_reader.py` + `constants.py` + `utils/{seed,io,logging,checks}.py` + `configs/phase0.yaml` | `pytest tests/test_synthetic.py -v` pass (≥3 tests) | 01-spectral-synth-phase-3a-done |
| **3b** | `preprocess/radiometric.py` + `preprocess/features.py` | `pytest tests/test_features.py -v` pass (≥2 tests), feature matrix deterministic + no NaN | -3b-done |
| **3c** | `model/baselines.py` (4 baselines) | `pytest tests/test_baselines.py -v` pass (≥4 tests, one per model) | -3c-done |
| **3d** | `eval/metrics.py` | `pytest tests/test_metrics.py -v` pass (≥3 tests including perfect-pred / zero-info / all-zero edge cases) | -3d-done |
| **3e** | `run.py` integration + smoke test | `python run.py --synthetic` < 60s, mean R² > 0.4 on 7 core bands | -3-done (alias) |

## 6. configs/phase0.yaml schema

```yaml
seed: 42
data:
  mode: synthetic         # synthetic | real
  synthetic:
    n_patches: 200
    patch_size: 32
    label_size: 8
    noise: 0.02
  real:
    hirise_dir: ../../data/HiRISE
    mastcamz_dir: ../../data/Mastcam-Z
    spice_kernel: ../../data/SPICE/m2020_coregister.tm
preprocess:
  clip_reflectance: [0.0, 1.0]
  use_dtm: true
  use_geometry: true       # zero-fill if absent
  use_atmospheric_tau: true
model:
  baselines: [ridge, random_forest, xgboost, mlp]
  ridge:    {alpha: 1.0}
  random_forest: {n_estimators: 80, max_depth: 12, n_jobs: 1}
  xgboost:  {n_estimators: 100, max_depth: 6, n_jobs: 1, tree_method: hist}
  mlp:      {hidden_layer_sizes: [64, 64], max_iter: 200, early_stopping: true}
eval:
  test_size: 0.2
  band_centers_nm: [442, 528, 567, 605, 686, 754, 800, 866, 910, 939, 978]
  core_band_indices: [3, 5, 6, 7, 8, 9, 10]   # HiRISE NIR-IR overlap (605~978 nm)
  success_threshold:
    mean_r2_core_synthetic: 0.4
output:
  artifacts_dir: artifacts
  log_dir: logs
runtime:
  cpu_only: true
  budget_seconds: 60
```

## 7. Test coverage map (G4 target ≥60%)

| 모듈 | unit tests | boundary cases | integration |
|---|---|---|---|
| `data/synthetic` | shape, determinism, target correlation | n=0, patch=1, noise=0 | (in smoke) |
| `data/real_reader` | dry-run manifest | all dirs missing → graceful | (in smoke) |
| `preprocess/features` | dim, no NaN/inf, determinism | all-zero HiRISE, all-NaN DTM → handled | (in smoke) |
| `model/baselines` | each model fit+predict on tiny synth | 1 sample → handled or graceful raise | (in smoke) |
| `eval/metrics` | R² perfect=1, zero-info≈0, all-zero pred | NaN guard, single-band | (in smoke) |
| `run.py` | — | — | smoke 60s + R²>0.4 |

## 8. Out-of-scope reaffirm

- No `pyproject.toml` (codex suggested, but `requirements.txt` per MD §⑦.5 sufficient for prototype)
- No `eval/plots.py` (numeric metrics only — Carson follow-up adds matplotlib visualizations)
- No `features/texture.py` (DTM slope/roughness + band stats sufficient)
- No `preprocess/geometry.py` (synthetic doesn't require alignment; real path is dry-run)
- No CLI multi-script (`scripts/run_phase0.py` + `scripts/smoke_synthetic.py`) — single `run.py` per MD §⑦.3.3e

---

**End of ARCHITECTURE. Phase 3 implementation starts at sub-phase 3a.**
