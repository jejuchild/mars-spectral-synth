# DECISIONS — autonomous-mode defaults (Carson 검토 후 변경 가능)

본 prototype은 overnight 자율 모드에서 작성됨. 아래 default들은 §⑨ 자율 룰의 "best-of-3 default 자동 선택" 적용 결과. Carson이 검토 후 변경 가능 (각 항목 옆 *Change-tag*).

## 1. Spec 해석

- **D1 (★)**: PRD §5 G3의 "synthetic mean R² > 0.4 on 7 core bands" 기준은 **best-of-4 baseline** 으로 측정 (모든 baseline이 통과할 필요 없음). *Change-tag*: spec-strict
- **D2**: 7 core band 인덱스 = `[3, 5, 6, 7, 8, 9, 10]` (HiRISE NIR-IR overlap 영역 추정, design doc §11.1 + codex deepdive §4.4 종합). *Change-tag*: core-band-mapping

## 2. Synthetic generator

- **D3**: 5-component Gaussian basis (peaks 500/620/700/860/900 nm, widths 40-120 nm) + Dirichlet(0.6) mixing → spectra. 진짜 광물 spectra (basalt, hematite, phyllosilicate, sulfate) 모방 아님 — Phase 0 model 검증용 toy. *Change-tag*: synth-realism
- **D4**: HiRISE 3-band은 Mastcam-Z spectra를 Gaussian filter (200/100/150 nm width @ 500/700/900 nm)로 적분. Beer-Lambert 대기 + BRDF 효과 모델 안 함. *Change-tag*: synth-physics
- **D5**: noise default 0.02 (reflectance unit). *Change-tag*: synth-noise
- **D6**: n_patches default 200, patch_size 32. CPU smoke 5s 안에 끝나는 양. *Change-tag*: synth-size

## 3. Feature engineering

- **D7**: HiRISE patch stats = mean/std/p25/p50/p75 per band (15 dims). texture/Gabor/CNN feature 미사용. *Change-tag*: hirise-features
- **D8**: DTM 3-feature = slope-magnitude / |Laplacian| / mean elevation. roughness 별도 측정 없음. *Change-tag*: dtm-features
- **D9**: geometry/τ 부재 시 zero-fill + `{name}_present_flag` 컬럼 추가. 모델이 학습으로 무시 가능하도록. *Change-tag*: missing-data-policy
- **D10**: 라벨 = patch-level 11-band 평균 reflectance (per-pixel 학습 X). Phase 0 prototype은 patch-to-spectrum mapping 만. *Change-tag*: label-granularity

## 4. Model baselines

- **D11**: 4 baseline = ridge / random_forest / xgboost / mlp. SVR / GradientBoosting / lightgbm 미포함 (overlap). *Change-tag*: baseline-set
- **D12**: hyperparameter = sklearn/xgboost default, no tuning. RF n_estimators=80, depth=12; XGB n_estimators=100, depth=6, hist; MLP (64,64) max_iter=200 early_stopping. *Change-tag*: hp-tuning
- **D13**: multi-output strategy = 단일 estimator (sklearn native multi-output) 또는 `MultiOutputRegressor` (xgboost). Per-band 별도 학습 X. *Change-tag*: multi-output-strategy
- **D14**: torch 미사용 (CPU torch는 wheel 설치되어 있음 단지 fallback 용). MLP는 sklearn `MLPRegressor`. *Change-tag*: torch-usage

## 5. Eval

- **D15**: train/test split = random 80/20 with seed 42. spatial split / temporal split / k-fold X. *Change-tag*: split-strategy
  - **주의**: `SPECTRAL_SYNTH_DESIGN.md` §7.1 명시 "*공간적으로* 분리되어야 함, 단순 sol split 아님". 진짜 데이터에선 spatial hold-out 강제. Phase 0 synthetic은 random OK.
- **D16**: SAM은 Mastcam-Z 11-band 전체 vector로 계산 (radian, 작을수록 좋음). 7-core 한정 SAM 미계산. *Change-tag*: sam-scope
- **D17**: Bootstrap CI / k-fold variance 미계산. design doc §7.2 명시되지만 Phase 0 OOS. *Change-tag*: ci-bootstrap

## 6. Real-data path

- **D18**: Phase 0의 `--real` 모드는 manifest discovery + dry-run only. Full PDS reader / Mastcam-Z calibration target apply / SPICE pose 변환 등 미구현. *Change-tag*: real-ingest
- **D19**: 데이터 위치는 `/disk1/cspark/mastcam/coregister_data/output/{hirise,mastcamz}` (Carson 기존 pipeline 출력). symlink로 `data/HiRISE`, `data/Mastcam-Z`. *Change-tag*: real-data-path
- **D20**: CRISM / THEMIS / MEDA 디렉토리 학교 서버 부재 → manifest의 `*_present` 플래그가 False. lazy-download 미실행 (overnight cap). *Change-tag*: aux-data

## 7. Codex 협업

- **D21**: Phase 2 codex critique은 sandbox 제약으로 PRD를 직접 못 읽음 → conditional CRITICAL 5건만 받음. 본 prototype은 그 5건 모두 위반 안 함을 확인 후 진행. *Change-tag*: codex-sandbox
- **D22**: Phase 4 tests는 claude solo 작성 (codex BLOCK 회피). boundary case 명시 (perfect / zero-info / all-zero / shape-mismatch / empty-dir). *Change-tag*: test-author

## 8. Runtime / 환경

- **D23**: Python 3.13.11, sklearn 1.8.0, xgboost 3.2.0, lightgbm 4.6.0, torch 2.11.0+cpu (unused). `requirements.txt`에 freeze. *Change-tag*: env-freeze
- **D24**: workspace `~/mars-auto/ideas/01-spectral-synth`, venv `~/mars-auto/.venv`. GitHub push는 `gh` CLI (jejuchild auth, repo+workflow scope). *Change-tag*: workspace
- **D25**: log → `logs/spectral_synth.log`, artifact → `artifacts/{metrics.json, predictions.npz}`. gitignored. *Change-tag*: artifact-policy

## 9. 확신 낮은 추측 (Carson confirm 필요)

- **D26**: 7 core band index 매핑 — design doc에 정확한 index 표 없음. 605~978 nm 범위로 지정. 만약 Carson이 다른 범위 의도면 `core_band_indices` 변경. ★ Change-tag: **core-band-mapping** (재실행 영향 큼)
- **D27**: synthetic 3-band → 11-band mapping이 진짜 HiRISE↔Mastcam-Z scenario를 얼마나 닮았는지 — Phase 0 prototype은 학습 가능성을 보일 뿐, paper claim 아님. ★ Change-tag: **synth-realism**

## 10. cycle-008 lesson 반영 ("Carson 강점 ★4+ hard requirement")

- **D28**: Carson HiRISE↔Mastcam-Z 정합 pipeline은 Phase 0 reader가 그 출력 format을 그대로 받도록 설계됨 (`/disk1/cspark/mastcam/coregister_data/output/...`). Real-mode dry-run으로 path 자동 인식 검증. *Change-tag*: carson-pipeline-binding
- **D29**: SPICE pose 입력 슬롯 마련 (`X_geometry` (N, 3) 컬럼). Phase 0 synthetic은 random fill, real-mode에선 `m2020_coregister.tm` 로딩 후 spiceypy로 변환 — Carson 후속. *Change-tag*: spice-binding
- **D30**: multi-instrument (CRISM/THEMIS/MEDA) 슬롯 미리 마련 (zero-fill + present_flag). Phase 1+ 활성화. *Change-tag*: multi-instrument-binding

---

**End of DECISIONS. 30개 default. Carson change-tag 단위로 재실행 필요 항목 식별 가능.**
