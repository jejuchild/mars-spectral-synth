# PRD — `mars-spectral-synth` Phase 0 prototype

작성: 2026-05-04 (overnight smoke test, idea 01)
참조 spec: `~/mars-auto/context/SPECTRAL_SYNTH_DESIGN.md` §6 Track A + §9 PRD draft, `codex-spectral-synth-deepdive.md` §3.3.2 + §4.4
범위: **Phase 0 prototype skeleton (CPU-only, synthetic+real branch, ~30분 phase budget cap)**

## 1. Problem statement

위성 HiRISE 25 cm/px 3-band orthoimage → 같은 ground patch에서 Mastcam-Z가 봤을 11-band reflectance cube (442–978 nm)를 학습으로 합성. 미래 sample-return mission planning + Starship multi-rover 사전 광물 분석 input. *Pre-encounter prediction*이라는 framing이 1961 OPC-GS의 post-observation 비판을 우회.

## 2. Goals / Non-goals

### Goals (Phase 0 prototype, CPU-only synthetic 위에서 검증)
- end-to-end pipeline 동작 (synthetic data 위에서 ingest → preprocess → train → eval → run)
- Track A regression (ridge / RF / XGBoost / small MLP) 4종 baseline 비교 가능
- HiRISE NIR-IR overlap 7 core bands에 대해 **mean R² > 0.4** (synthetic, easy case)
- per-band SAM (Spectral Angle Mapper) + per-band σ uncertainty 산출
- 실데이터 path 분기: `python run.py --synthetic` / `python run.py --real --hirise data/HiRISE/... --mastcamz data/Mastcam-Z/...`
- 모든 unit test pass (sub-phase 별 ≥1, 전체 ≥10)

### Non-goals (Phase 0)
- Diffusion / LoRA / Foundation model — MD §⑥.1 명시 금지
- GPU 학습 — `.cpu()` 강제, `torch.cuda.is_available()` 호출 금지
- 실제 Mars hold-out region R² > 0.5 — Carson 후속 (paper-ready 작업, 별도 70-100h)
- Cheyava/Bright Angel anchor visualization — 본 prototype은 numeric metric 위주
- Onboard real-time deployment / 화성 전 영역 / dust storm prediction (design doc §9.8 OOS 그대로)
- `gh repo deploy` 외 외부 deploy (vercel/render/HF Spaces 금지, MARS_AUTO_OVERNIGHT.md §❷)

## 3. Inputs

| 입력 | 경로 | shape | optional |
|---|---|---|---|
| HiRISE 3-band patch | `data/HiRISE/` (symlink → mastcam coregister output) + `data/HiRISE_full_pds/` | `(H, W, 3)` float32, BG/RED/NIR-IR | required |
| HiRISE DTM | `data/HiRISE_full_pds/` 안의 lbl 매칭 | `(H, W)` float32 m | optional (slope/roughness derived) |
| Mastcam-Z 11-band patch (label) | `data/Mastcam-Z/` (symlink) | `(h, w, 11)` float32, 442-978 nm | required for train |
| Sun-target-camera geometry | `data/SPICE/m2020_coregister.tm` (symlink) | `(3,)` phase/incidence/emission deg | optional |
| Atmospheric τ | `data/MEDA/` (현재 부재) | scalar float per sol | optional (없으면 zero-fill) |
| CRISM coarse summary | `data/CRISM/` (현재 부재) | `(h, w, K)` float32 | optional |

## 4. Outputs

| 산출물 | 경로 | 형식 |
|---|---|---|
| 학습된 model 체크포인트 | `artifacts/model_{ridge,rf,xgb,mlp}.pkl` (sklearn pickled) | `.pkl` |
| per-band metric 표 | `artifacts/metrics.json` | `{model, band_idx, R2, SAM, mae}` |
| hold-out predictions | `artifacts/predictions.npz` | `y_pred (N, 11)`, `y_true (N, 11)`, `band_centers (11,)` |
| run log | `logs/01-spectral-synth-run.log` | text |
| (option) synthetic data dump | `artifacts/synth_dataset.npz` | reproducibility |

## 5. Success criteria (binding for Phase 0 prototype)

- [G1] `pytest -v` 모든 test pass (≥10 tests)
- [G2] `python run.py --synthetic` smoke run 60초 이내 완료 (CPU)
- [G3] synthetic data 위 7 core bands (HiRISE NIR-IR overlap: Mastcam-Z indices ≈ {3,5,6,7,8,9,10} per design doc §11.1 mapping) **mean R² > 0.4**
- [G4] coverage > 60% (`pytest --cov=src`)
- [G5] `python run.py --real --dry-run` 시 symlink된 실데이터 path를 정상 인식 + 데이터 부재 시 graceful fallback 메시지 (`CRISM not present → skip`)

## 6. Pipeline (5-step)

1. **Ingest**: synthetic generator 또는 real PDS reader → patch dataframe `(N, H, W, 3)` HiRISE + `(N, h, w, 11)` Mastcam-Z label
2. **Preprocess**: HiRISE→pixel feature flatten + DTM slope/roughness 추가 + (optional) geometry/τ context vector concat → `(N, F)` feature matrix
3. **Train**: 4 model 학습 (ridge / RF / XGBoost / small MLP), 80/20 split
4. **Eval**: per-band R² + SAM + MAE, 7 core bands mean R² 표시, hold-out predictions 저장
5. **Report**: `metrics.json` + `predictions.npz` + console summary

## 7. Architecture preview (Phase 2에서 정식)

```
ideas/01-spectral-synth/
├── src/
│   ├── data/        # ingest, synthetic generator, real PDS reader (lazy)
│   ├── preprocess/  # patch alignment, feature engineering, atmospheric correction stub
│   ├── model/       # 4 baselines: ridge / rf / xgb / mlp (sklearn-style fit/predict)
│   └── eval/        # metrics: R², SAM, MAE; hold-out splitter
├── tests/           # unit + integration (synthetic path)
├── artifacts/       # model checkpoints + metrics + predictions
├── run.py           # CLI: --synthetic | --real --hirise PATH --mastcamz PATH
├── PRD.md
├── ARCHITECTURE.md  # phase 2
├── DECISIONS.md     # phase 5
├── TODO.md          # phase 5
├── README.md        # phase 5
└── requirements.txt # phase 5
```

## 8. Out of scope (overnight prototype)

- Diffusion / LoRA / pre-trained Earth foundation model
- GPU 학습 / CUDA 호출
- 5GB 초과 PDS download (cap)
- 실제 Mars Jezero hold-out region R² > 0.5 (paper-ready)
- Cheyava / Bright Angel anchor visualization
- Mineral classifier downstream
- Atmospheric correction physical model (Hapke BRDF, full DISORT)
- arXiv draft / paper writing

## 9. Carson 강점 sell (cycle-008 ★4+ hard requirement)

- HiRISE↔Mastcam-Z 정합 pipeline 실전 보유 (`/disk1/cspark/mastcam/coregister_data/output/{hirise,mastcamz}/`) → real-data path가 Carson pipeline 출력을 그대로 입력으로 받아들이도록 reader 설계
- SPICE 기반 sun-target-camera geometry 가용 (m2020_coregister.tm) → optional feature로 명시
- multi-instrument (CRISM/THEMIS/MEDA) 슬롯 미리 마련, Phase 0에선 zero-fill, 후속 phase에서 활성화
- Mastcam-Z calibration 도메인 지식 (Rice 2023 reference) → calibration target normalization placeholder

## 10. Open decisions (autonomous default → DECISIONS.md 기록)

1. 7 core bands index 매핑: `[3,5,6,7,8,9,10]` (HiRISE NIR-IR overlap, design doc §11.1 추론) — Carson 검토 후 변경 가능
2. patch size: 32×32 (synthetic), 실데이터는 사용자 지정
3. train/test split: 80/20 random, seed=42
4. baseline 4종 hyperparameter: sklearn default (no tuning)
5. uncertainty σ 추정: per-band MAD on train residual (post-hoc), 모델 내부 prediction interval 아님
6. atmospheric τ 부재 시: zero-fill + flag column (모델이 학습으로 무시 가능)
7. CRISM/THEMIS 부재 시: zero-fill column

## 11. Risks (Phase 0 prototype 수준)

| Risk | 확률 | mitigation |
|---|---|---|
| 실데이터 reader가 Carson pipeline 출력 format과 mismatch | 중 | 본 prototype은 dry-run + path inspection만, full read는 Carson 후속 |
| synthetic data가 너무 easy → R² inflated | 중 | synthetic generator에 noise + spectral correlation 의도 추가 |
| sklearn API drift (1.8.x) | 낮 | `requirements.txt`에 정확 버전 freeze |
| codex BLOCK 누적 → solo 격하 | 낮 | §⑧.3 fallback, DECISIONS.md 명시 |

---

**End of PRD. Phase 2 Architecture가 다음 단계.**
