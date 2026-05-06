"""Real-data single-band radiance proxy: HiRISE RED vs Mastcam-Z LEFT radiance.

Honest framing per codex plan: no Mastcam-Z 11-band cubes locally and no HiRISE
3-band stack — only single bands of each. So we cannot test the original PRD's
HiRISE 3-band → Mastcam-Z 11-band synthesis. Reduced scope:
- HiRISE RED orthoimage (single band, 16-bit) — orbital prior
- Mastcam-Z LEFT radiance ZLF_*RASLN*.IMG (single band) — ground observation
- Without strict spatial co-registration, report **distributional similarity**
  via 1D Wasserstein distance between normalized intensity histograms

Inputs:
  HiRISE: /disk1/cspark/mastcam/coregister_data/pds_cache/hirise/ESP_065431_1985_RED.JP2
  Mastcam-Z: /disk1/cspark/mastcam/coregister_data/pds_cache/mastcamz/sol{N}/ZLF_*RASLN*.IMG
Outputs: artifacts_real/real_radiance_proxy.json + figure
"""
from __future__ import annotations

import argparse
import glob
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, write_json
from src.utils.logging import setup_logging

Image.MAX_IMAGE_PIXELS = None

DEFAULT_HIRISE = "/disk1/cspark/mastcam/coregister_data/pds_cache/hirise/ESP_065431_1985_RED.JP2"
DEFAULT_MASTCAMZ = "/disk1/cspark/mastcam/coregister_data/pds_cache/mastcamz"


def parse_pds4_xml(xml_path: Path) -> dict:
    """Extract array shape + dtype + element offset from a PDS4 label."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"p": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    info = {"path": str(xml_path)}

    def find_text(name: str):
        for el in root.iter():
            tag = el.tag.split("}")[-1]
            if tag == name:
                return (el.text or "").strip()
        return None

    info["lines"] = int(find_text("axes")[0]) if False else None
    axes = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "Axis_Array":
            axis = {}
            for c in el:
                ctag = c.tag.split("}")[-1]
                axis[ctag] = (c.text or "").strip()
            axes.append(axis)
    info["axes"] = axes
    info["data_type"] = find_text("data_type")
    info["offset"] = int(find_text("offset") or 0)
    return info


def parse_mastcamz_array(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    info = {"bands": None, "lines": None, "samples": None,
            "dtype": ">i2", "offset": 0, "scaling_factor": 1.0, "value_offset": 0.0}
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "Array_3D_Image":
            for c in el.iter():
                ctag = c.tag.split("}")[-1]
                if ctag == "offset" and c.text:
                    info["offset"] = int(c.text.strip())
                elif ctag == "Element_Array":
                    for cc in c:
                        cct = cc.tag.split("}")[-1]
                        if cct == "data_type":
                            info["dtype_str"] = (cc.text or "").strip()
                        elif cct == "scaling_factor" and cc.text:
                            info["scaling_factor"] = float(cc.text.strip())
                        elif cct == "value_offset" and cc.text:
                            info["value_offset"] = float(cc.text.strip())
                elif ctag == "Axis_Array":
                    name = ""
                    elems = 0
                    for cc in c:
                        cct = cc.tag.split("}")[-1]
                        if cct == "axis_name":
                            name = (cc.text or "").strip().lower()
                        elif cct == "elements":
                            elems = int((cc.text or "0").strip())
                    if name == "band":
                        info["bands"] = elems
                    elif name == "line":
                        info["lines"] = elems
                    elif name == "sample":
                        info["samples"] = elems
            break
    dt_str = info.get("dtype_str", "")
    if dt_str == "SignedMSB2":
        info["dtype"] = ">i2"
    elif dt_str == "SignedLSB2":
        info["dtype"] = "<i2"
    elif dt_str == "UnsignedMSB2":
        info["dtype"] = ">u2"
    elif dt_str == "IEEE754MSBSingle":
        info["dtype"] = ">f4"
    return info


def read_zlf_img_radiance(img_path: Path, xml_path: Path | None = None) -> np.ndarray:
    """Read Mastcam-Z RASLN 3-band radiance, scaled to physical units."""
    if xml_path is None:
        xml_path = img_path.with_suffix(".xml")
    if not xml_path.exists():
        raise FileNotFoundError(f"missing PDS4 label: {xml_path}")
    meta = parse_mastcamz_array(xml_path)
    bands, lines, samples = meta["bands"], meta["lines"], meta["samples"]
    if not (bands and lines and samples):
        raise ValueError(f"could not parse axes from {xml_path}")
    n = bands * lines * samples
    raw = np.fromfile(img_path, dtype=meta["dtype"], count=n, offset=meta["offset"])
    arr = raw.reshape(bands, lines, samples).astype(np.float32)
    return arr * float(meta["scaling_factor"]) + float(meta["value_offset"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hirise", default=DEFAULT_HIRISE)
    p.add_argument("--mastcamz", default=DEFAULT_MASTCAMZ)
    p.add_argument("--n-hirise-samples", type=int, default=500_000)
    p.add_argument("--n-mastcam-files", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reduce", type=int, default=4, help="reduce JP2 by factor")
    p.add_argument("--artifacts", default=str(ROOT / "artifacts_real"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging(log_dir=ROOT / "logs", name="spectral_synth_real")

    log.info("Loading HiRISE RED JP2 (reduce=%d)", args.reduce)
    img = Image.open(args.hirise)
    if img.mode == "I;16":
        log.info("Converting I;16 → I for reduce compatibility")
        img = img.convert("I")
    if args.reduce > 1:
        img = img.reduce(args.reduce)
    arr = np.asarray(img, dtype=np.float32)
    log.info("HiRISE shape: %s, range [%.0f, %.0f]", arr.shape, arr.min(), arr.max())

    rng = np.random.default_rng(args.seed)
    H, W = arr.shape
    nz = arr > 0
    if nz.sum() < args.n_hirise_samples:
        log.warning("only %d non-zero HiRISE pixels", nz.sum())
    flat_nz = arr[nz]
    take = min(args.n_hirise_samples, len(flat_nz))
    idx = rng.choice(len(flat_nz), size=take, replace=False)
    hirise_samples = flat_nz[idx]
    log.info("HiRISE samples: %d, mean=%.1f std=%.1f", len(hirise_samples),
             hirise_samples.mean(), hirise_samples.std())

    sol_dirs = sorted(Path(args.mastcamz).glob("sol*"))
    log.info("Mastcam-Z sols available: %d", len(sol_dirs))
    raslnpattern = "ZLF_*RASLN*.IMG"
    chosen = []
    for sd in sol_dirs:
        files = list(sd.glob(raslnpattern))
        if files:
            chosen.append(files[0])
        if len(chosen) >= args.n_mastcam_files:
            break
    log.info("Mastcam-Z RASLN files chosen: %d (sols: %s)",
             len(chosen), [f.parent.name for f in chosen])

    mastcam_samples_list = []
    per_file_records = []
    for img_path in chosen:
        try:
            r = read_zlf_img_radiance(img_path)  # (3, L, S) Bayer R/G/B radiance
        except Exception as e:
            log.warning("failed %s: %s", img_path.name, e)
            continue
        avg = r.mean(axis=0)  # average over R/G/B → broadband radiance proxy
        valid = avg[avg > 0]
        if len(valid) == 0:
            continue
        per_file_records.append({
            "file": img_path.name,
            "sol": img_path.parent.name,
            "shape": list(r.shape),
            "n_valid": int(len(valid)),
            "mean": float(valid.mean()),
            "std": float(valid.std()),
            "median": float(np.median(valid)),
            "per_band_mean": [float(r[b][r[b] > 0].mean()) if (r[b] > 0).any() else 0.0 for b in range(r.shape[0])],
        })
        mastcam_samples_list.append(valid)
        log.info("  %s: shape=%s n_valid=%d mean=%.4f std=%.4f bands=%s",
                 img_path.name, r.shape, len(valid), valid.mean(), valid.std(),
                 per_file_records[-1]["per_band_mean"])

    if not mastcam_samples_list:
        log.error("no Mastcam-Z RASLN files loaded")
        return 1

    mastcam_samples = np.concatenate([
        s[rng.choice(len(s), size=min(50_000, len(s)), replace=False)]
        for s in mastcam_samples_list
    ])
    log.info("Mastcam-Z combined samples: %d, mean=%.4f std=%.4f",
             len(mastcam_samples), mastcam_samples.mean(), mastcam_samples.std())

    def normalize(a):
        return (a - a.mean()) / (a.std() + 1e-9)

    h_norm = normalize(hirise_samples)
    m_norm = normalize(mastcam_samples)
    n_for_wd = min(50_000, len(h_norm), len(m_norm))
    rng2 = np.random.default_rng(args.seed + 1)
    h_sub = rng2.choice(h_norm, size=n_for_wd, replace=False)
    m_sub = rng2.choice(m_norm, size=n_for_wd, replace=False)
    wd_normalized = float(wasserstein_distance(h_sub, m_sub))
    wd_raw = float(wasserstein_distance(
        rng2.choice(hirise_samples, size=n_for_wd, replace=False),
        rng2.choice(mastcam_samples, size=n_for_wd, replace=False),
    ))

    log.info("Wasserstein-1 (z-normalized): %.4f", wd_normalized)
    log.info("Wasserstein-1 (raw radiance): %.4f", wd_raw)

    summary = {
        "hirise": {
            "path": args.hirise,
            "reduce_factor": args.reduce,
            "image_shape": list(arr.shape),
            "n_samples": int(len(hirise_samples)),
            "mean": float(hirise_samples.mean()),
            "std": float(hirise_samples.std()),
            "median": float(np.median(hirise_samples)),
        },
        "mastcam_z_left_radiance": {
            "n_files": len(per_file_records),
            "n_combined_samples": int(len(mastcam_samples)),
            "files": per_file_records,
            "mean": float(mastcam_samples.mean()),
            "std": float(mastcam_samples.std()),
            "median": float(np.median(mastcam_samples)),
        },
        "wasserstein_1d": {
            "z_normalized": wd_normalized,
            "raw_radiance": wd_raw,
            "n_samples_compared": n_for_wd,
        },
        "honesty_note": (
            "HiRISE RED is a 16-bit DN/I-over-F orbital image; Mastcam-Z RASLN is "
            "calibrated radiance in W/m²/sr/nm (different physical unit and band). "
            "We compare z-normalized distributions because the absolute scales are "
            "incomparable. This is NOT spectral synthesis — it is a distributional "
            "sanity check on real Mars data."
        ),
    }
    artifacts = ensure_dir(args.artifacts)
    write_json(summary, artifacts / "real_radiance_proxy.json")
    log.info("Wrote %s", artifacts / "real_radiance_proxy.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(hirise_samples, bins=80, color="steelblue", alpha=0.7, label="HiRISE RED (DN)")
        axes[0].set_xlabel("HiRISE pixel value")
        axes[0].set_ylabel("count")
        axes[0].set_title("HiRISE raw distribution")
        axes[0].legend()

        axes[1].hist(h_sub, bins=60, color="steelblue", alpha=0.5, label="HiRISE z-norm", density=True)
        axes[1].hist(m_sub, bins=60, color="darkorange", alpha=0.5, label="Mastcam-Z RASLN z-norm", density=True)
        axes[1].set_xlabel("z-normalized intensity")
        axes[1].set_ylabel("density")
        axes[1].set_title(f"z-normalized overlay (W₁={wd_normalized:.3f})")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(artifacts / "real_radiance_proxy.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote figure: %s", artifacts / "real_radiance_proxy.png")
    except Exception as e:
        log.warning("figure failed: %s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
