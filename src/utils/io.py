import json
from pathlib import Path

import numpy as np
import yaml


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(obj, path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return p


def dump_npz(path, **arrays) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    np.savez_compressed(p, **arrays)
    return p


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")
