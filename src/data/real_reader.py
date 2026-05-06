"""Discover real Carson-pipeline output. Phase 0 supports dry-run only."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RealDataManifest:
    hirise_dir: Path
    mastcamz_dir: Path
    hirise_files: int
    mastcamz_files: int
    paired_estimate: int
    missing_dirs: list
    spice_kernel_present: bool
    crism_present: bool
    themis_present: bool
    meda_present: bool


def discover_carson_pipeline(
    hirise_dir: str | Path,
    mastcamz_dir: str | Path,
    spice_kernel: str | Path | None = None,
    crism_dir: str | Path | None = None,
    themis_dir: str | Path | None = None,
    meda_dir: str | Path | None = None,
) -> RealDataManifest:
    hirise = Path(hirise_dir)
    mastcamz = Path(mastcamz_dir)

    missing = []
    if not hirise.exists():
        missing.append(str(hirise))
    if not mastcamz.exists():
        missing.append(str(mastcamz))

    hirise_files = 0
    mastcamz_files = 0
    if hirise.exists():
        hirise_files = sum(1 for _ in hirise.rglob("*") if _.is_file())
    if mastcamz.exists():
        mastcamz_files = sum(1 for _ in mastcamz.rglob("*") if _.is_file())

    paired = min(hirise_files, mastcamz_files)

    def _present(p):
        if p is None:
            return False
        try:
            return Path(p).exists()
        except OSError:
            return False

    return RealDataManifest(
        hirise_dir=hirise,
        mastcamz_dir=mastcamz,
        hirise_files=hirise_files,
        mastcamz_files=mastcamz_files,
        paired_estimate=paired,
        missing_dirs=missing,
        spice_kernel_present=_present(spice_kernel),
        crism_present=_present(crism_dir),
        themis_present=_present(themis_dir),
        meda_present=_present(meda_dir),
    )
