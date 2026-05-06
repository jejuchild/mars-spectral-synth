import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str | Path | None = None, name: str = "spectral_synth") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(d / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
