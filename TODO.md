# TODO — Carson 다음 작업 (Phase 1~3, paper-ready 방향)

본 Phase 0 prototype 위에 70-100h 추가 작업이 paper draft까지의 path. `SPECTRAL_SYNTH_DESIGN.md` §10 4지선다 게이트와 정렬.

## 즉시 (week 1, ~10h)

- [ ] **DECISIONS.md 검토** — D1~D30 전체 검토, change-tag 단위로 변경 결정
- [ ] **D26 7 core band index 확정** — Mastcam-Z 11-band 중 HiRISE NIR-IR overlap 영역 정확히 어디? 광학 design + Rice 2023 calibration table로 binding
- [ ] **`src/data/real_reader.py` 확장** — Carson pipeline 출력 디렉토리 구조 매칭 (`*_aligned_meta.json`, `*_ortho.png` 등 실제 파일명) → sample N=10 patch 로드 + `clip_reflectance` 적용 후 동일 pipeline 통과 확인
- [ ] **CRISM / MEDA / SPICE pose 슬롯 활성화** — `data/SPICE/m2020_coregister.tm` 로딩 (spiceypy), MEDA τ 다운로드 (5GB cap 안에서), `X_geometry` 와 `X_tau` 진짜 값으로 채움

## Phase 1 (week 2-4, ~25h) — Track A POC on real data (50-100 patch)

- [ ] Jezero sol 100-700 paired patch 50-100건 (Carson pipeline 재사용)
- [ ] `python run.py --real --hirise ... --mastcamz ...` 풀 ingest 동작
- [ ] 5 baseline 비교 표 추가 (bilinear interpolation + 본 4 baseline)
- [ ] **Phase 1 gate** (`SPECTRAL_SYNTH_DESIGN.md` §9.5): per-band R² 분포 + SAM < bilinear -20% 통과 여부

## Phase 2 (month 2-3, ~25h) — anchor visualization + ablation

- [ ] Cheyava Falls (sol 1175) hold-out patch에서 11-band 합성 vs 실측 visualization (figure-1 candidate)
- [ ] Bright Angel formation anchor 동일
- [ ] Ablation: DTM 빼면 / geometry 빼면 / τ 빼면 R² drop 분해
- [ ] Per-band uncertainty σ 추정 (bootstrap residual MAD on train) → trust-gate

## Phase 3 (month 4-9) — paper / submission

- [ ] **Phase 0/1 4지선다 게이트** (`SPECTRAL_SYNTH_DESIGN.md` §10):
  - 6+ band R² > 0.5 + Cheyava OK → GO Track A+B (full ambition)
  - 3-5 band R² > 0.4 → GO Track A only (IEEE TGRS short, 350-400h)
  - 1-2 band only → PIVOT to dataset paper (#4 CROSSVIEW-DB)
  - all band R² < 0.3 → KILL
- [ ] Track A only paper draft (IEEE TGRS short / RSE)
  - Method: 본 4 baseline + ablation
  - Data: Jezero hold-out region split (spatial, sol 700-1300)
  - Mineral class agreement metric (CRISM ground truth)
- [ ] (option, advisor 동의 시) Track B = pre-trained Earth diffusion + LoRA — **별도 lab GPU 협상 + 5-10일 점유** 후 시도. Carson 강점 ★3-4 borderline.

## Engineering follow-up (low priority but useful)

- [ ] `src/eval/plots.py` — pred-vs-true scatter, residual heatmap, SAM histogram (matplotlib)
- [ ] `src/preprocess/calibration.py` — Mastcam-Z calibration target apply (Rice 2023), HiRISE Lambertian
- [ ] k-fold cross-validation + bootstrap CI
- [ ] LightGBM baseline 추가 (`requirements.txt`에 이미 있음)
- [ ] OOD detection (uncertainty 임계 초과 patch 자동 reject)
- [ ] `src/data/real_reader.py`: `discover_carson_pipeline` 결과를 `pytest tests/test_real_reader.py` 로 unit test (현재는 smoke만)
- [ ] CI: `pytest -v --cov=src` GitHub Actions에서 자동 실행

## 알려진 한계 (수정 안 함, 명시만)

- Synthetic data가 진짜 Mars 광물 spectra (basalt / hematite / phyllosilicate / sulfate) 와 무관 — Phase 0 prototype 검증용 toy
- Patch-to-spectrum (single 11-band vector per patch) 매핑. Per-pixel 11-band cube 합성 X — design doc §6 Track A 수준
- Spectral extrapolation 영역 (Mastcam-Z 442/978 nm) 자신감 없음 — design doc §11.1 한계 그대로
- Reviewer "이거 진짜인지 어떻게 아냐" critique 대응 (anchor figure + uncertainty σ + 5 baseline) 은 Phase 2까지 안 옴

## 호들갑 금지

본 README/DECISIONS/TODO는 prototype 수준 honest framing. paper-ready 주장 X. cycle-008 / cycle-013 lesson:

- ★4+ Carson 강점 (HiRISE↔Mastcam-Z 정합 pipeline + SPICE) 가 main contribution이 되도록 paper 구성
- novelty ★3 Track A regression 단독으로 IEEE TGRS short publishable (`SPECTRAL_SYNTH_DESIGN.md` §8.5 reduced)
- Track B 진입은 advisor 협의 후 (D26-D27 + GPU 5-10일 점유 협상)
