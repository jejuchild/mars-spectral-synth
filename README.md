# mars-spectral-synth — Phase 0 prototype

CPU-only Track A regression skeleton for HiRISE 3-band → Mastcam-Z 11-band reflectance synthesis.

> *"Idea 01 — SPECTRAL-SYNTH reduced. ★4.0 / 1순위 동률"* (`CARSON_FINAL_5IDEA_VERDICT.md`).
> Phase 0 prototype = pipeline skeleton on synthetic data. Real-data ingest + paper-ready experiments are Carson follow-up.

## Status

| Gate | Target | Result |
|---|---|---|
| G1 unit + integration tests | ≥10 pass | **32 pass** |
| G2 synthetic smoke wall-clock | <60s | **~5s** |
| G3 mean R² on 7 core bands (synthetic) | >0.4 | **0.704 (ridge)** |
| G4 src coverage | >60% | **72%** |
| G5 real dry-run handles missing dirs | graceful | **OK** |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU-only by design — no CUDA, no diffusion, no foundation model.

## Run

```bash
# synthetic (default, fast)
python run.py --synthetic

# real-data dry run (manifest only — full ingest is Carson follow-up)
python run.py --real --dry-run \
    --hirise   /disk1/cspark/mastcam/coregister_data/output/hirise \
    --mastcamz /disk1/cspark/mastcam/coregister_data/output/mastcamz
```

Output:
- `artifacts/metrics.json` — per-baseline `mean_r2_core` / `mean_r2_all` / `sam_mean_rad` / `per_band_*`
- `artifacts/predictions.npz` — `y_true`, `y_pred_{ridge,random_forest,xgboost,mlp}`, `band_centers`, `core_band_indices`
- `logs/spectral_synth.log`

## Layout

```
src/
├── constants.py        Mastcam-Z 11 band centers + 7 core indices
├── data/
│   ├── synthetic.py    deterministic paired generator
│   └── real_reader.py  Carson-pipeline manifest discovery (Phase 0: dry-run)
├── preprocess/
│   ├── radiometric.py  clip / normalize / Beer-Lambert atmospheric proxy
│   └── features.py     HiRISE patch stats + DTM derivatives + geometry/τ slots
├── model/
│   └── baselines.py    ridge / random_forest / xgboost / mlp (sklearn-compatible)
├── eval/
│   └── metrics.py      per-band R²/MAE, spectral angle mapper, train/test split
└── utils/              seed, io, logging, cpu-only guard, runtime budget
```

See `ARCHITECTURE.md` for module-by-module responsibility, `PRD.md` for the success-criteria contract, `DECISIONS.md` for autonomous-mode default choices, `TODO.md` for next-step plan.

## What this is *not*

Phase 0 prototype — *not* paper-ready. No real Mars Jezero hold-out R², no Cheyava/Bright Angel anchor visualization, no diffusion / LoRA, no GPU. Those are Carson follow-up work (~70-100h per `SPECTRAL_SYNTH_DESIGN.md` §8.1 reduced version timeline).

## Spec sources (read order)

1. `~/mars-auto/context/CARSON_FINAL_5IDEA_VERDICT.md` — 5 idea ranking, ★4.0 binding
2. `~/mars-auto/context/SPECTRAL_SYNTH_DESIGN.md` — Track A regression baseline contract (§6, §9)
3. `~/mars-auto/context/codex-spectral-synth-deepdive.md` — Earth prior art + spectral coverage gap analysis (§3.3.2, §4.4)
4. `/disk1/cspark/mastcam/research/{00,02,03,04,05,06,07,08,09}_*.md` + `SUMMARY.md` — Carson domain corpus

License: research prototype, no license header (Carson decides).
