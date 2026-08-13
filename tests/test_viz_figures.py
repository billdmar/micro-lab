"""Figure-generation smoke + determinism tests.

The plotting code (src/viz/*) is coverage-omitted; these tests assert the figure
pipeline RUNS from the committed fixtures with no raw data, produces all five
PNGs, and is byte-stable on a second run (the project's script-generated +
deterministic figure requirement). Kept fast (writes to a tmp dir)."""

from __future__ import annotations

import importlib.util
import os

import matplotlib

matplotlib.use("Agg")

# Load scripts/make_figures.py by path (scripts/ is not an importable package).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "make_figures", os.path.join(_ROOT, "scripts", "make_figures.py")
)
make_figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_figures)

EXPECTED = [
    "01_ofi_horizon_profile.png",
    "02_ofi_linearity.png",
    "03_recovery_power.png",
    "04_robustness_heatmap.png",
    "05_sign_autocorr_decay.png",
]


def _generate(outdir: str) -> None:
    fam = make_figures._load("study_family_results.csv")
    rob = make_figures._load("robustness_by_symbol.csv")
    binned = make_figures._load("ofi_linearity_binned.csv")
    fit = make_figures._load("ofi_linearity_fit.csv")
    power = make_figures._load("recovery_power_curve.csv")
    make_figures.fig_horizon_profile(fam, outdir)
    make_figures.fig_linearity(binned, fit, outdir)
    make_figures.fig_recovery_power(power, outdir)
    make_figures.fig_robustness(rob, outdir)
    make_figures.fig_sign_autocorr(fam, outdir)


def test_all_figures_generate_from_fixtures(tmp_path):
    out = tmp_path / "figs"
    out.mkdir()
    _generate(str(out))
    for name in EXPECTED:
        p = out / name
        assert p.exists() and p.stat().st_size > 1000, f"{name} missing or too small"


def test_figures_are_byte_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _generate(str(a))
    _generate(str(b))
    for name in EXPECTED:
        assert (a / name).read_bytes() == (b / name).read_bytes(), (
            f"{name} not byte-identical across runs (non-deterministic figure)"
        )
