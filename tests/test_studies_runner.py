"""Tests for the study runner. The pure bucketing / p-value helpers run
in CI; the full real-data family run is marked slow (needs raw LOBSTER data)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from config import registry as R
from src.studies import runner

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def test_bucket_sums_nonoverlapping_and_nan_safe():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], dtype="float64")
    # bucket=3 -> [1+2+0, 4+5+6] = [3, 15] (NaN treated as 0 within a bucket)
    out = runner._bucket_sums(x, 3)
    assert np.allclose(out, [3.0, 15.0])
    # incomplete trailing bucket is dropped
    assert runner._bucket_sums(np.arange(5.0), 3).tolist() == [0 + 1 + 2]


def test_bucket_mid_change_is_log_return_across_bucket_closes():
    mid = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], dtype="float64")
    # bucket=2 -> closes at idx 1,3,5 = [101,103,105]; diffs of logs
    out = runner._bucket_mid_change(mid, 2)
    expected = np.diff(np.log([101.0, 103.0, 105.0]))
    assert np.allclose(out, expected)


def test_bucket_mid_change_too_short_returns_empty():
    assert runner._bucket_mid_change(np.arange(3.0), 2).size == 0


def _has_raw() -> bool:
    return os.path.exists(runner._raw_paths(R.DATA.tickers[0])[0])


def _synth_symbol(ticker: str, n: int = 6000, beta: float = 3e-4, seed: int = 0):
    """A small synthetic SymbolData with a planted OFI->forward-price link, so the
    rewritten studies can be exercised in CI without raw LOBSTER data."""
    rng = np.random.default_rng(R.MASTER_SEED + seed)
    ofi = rng.standard_normal(n) * 100.0
    # mid driven by lagged OFI (a real forward relationship) + noise
    logmid = np.cumsum(beta * 1e-2 * np.r_[0.0, ofi[:-1]] + rng.standard_normal(n) * 1e-4)
    mid = 30_0000 * np.exp(logmid)  # ~ $30 in 1/10000-dollar units
    signed = rng.standard_normal(n) * 50.0
    tick = rng.choice([-1.0, 1.0], size=n)
    qi = np.tanh(rng.standard_normal(n))
    return runner.SymbolData(
        ticker=ticker,
        n_events=n,
        mid=mid,
        ofi_event=ofi,
        signed_size=signed,
        tick_sign=tick,
        queue_imbalance=qi,
    )


def test_studies_run_on_synthetic_symbols_ci_covered():
    """Exercise the rewritten study functions (contemporaneous slope, direct OOS
    forward coef, block-AUC queue, per-symbol-averaged sign ACF, binned impact) on
    synthetic symbols so the riskiest logic is covered in CI (no raw data)."""
    syms = [_synth_symbol(t, seed=i) for i, t in enumerate(R.DATA.tickers)]

    slope, r2 = runner.study_ofi_contemporaneous(syms)
    assert slope.metric == "coef" and np.isfinite(slope.point)
    assert "r2" in slope.extra and 0.0 <= r2.point <= 1.0

    fwd = runner.study_ofi_forward(syms)
    assert len(fwd) == len(R.HORIZON_GRID_EVENTS)
    for res in fwd:
        assert np.isfinite(res.point) and res.ci_low <= res.point <= res.ci_high
        assert "oos_ic" in res.extra  # the OOS information coefficient is reported

    auc = runner.study_queue_imbalance(syms)
    assert auc.ci_method.startswith("bootstrap_block")
    assert 0.0 <= auc.point <= 1.0 and 0.0 < auc.p_value_raw <= 1.0

    ac = runner.study_sign_autocorrelation(syms)
    assert len(ac) == len(R.HORIZON_GRID_EVENTS)
    for res in ac:
        assert res.ci_method == "bartlett_per_symbol_weighted"
        assert res.ci_low <= res.point <= res.ci_high  # CI contains the estimate

    imp = runner.study_impact_curve(syms)
    assert imp.metric == "coef" and np.isfinite(imp.point)
    assert imp.extra["n_bins"] >= 5  # binning actually happened


@pytest.mark.slow
def test_full_family_replicates_canonical_effects():
    """The registered family, on real data, reproduces the canonical order-flow
    effects with the expected signs/magnitudes and FDR control. Skips without data."""
    if not _has_raw():
        pytest.skip("raw LOBSTER data not present (run scripts/get_data.sh)")
    df = runner.run_family()
    assert len(df) == R.family_size() == 17
    by = df.set_index("name")

    # OFI -> contemporaneous price: the FDR-entering test is the HAC slope, which is
    # positive with its CI above 0 (Cont-Kukanov-Stoikov). (R^2 is reported
    # descriptively; the per-symbol R^2 is checked in the robustness test.)
    assert by.loc["ofi_contemporaneous", "metric"] == "coef"
    assert by.loc["ofi_contemporaneous", "point"] > 0
    assert by.loc["ofi_contemporaneous", "ci_low"] > 0

    # OFI -> FORWARD price: the DIRECT out-of-sample coefficient is positive at
    # every horizon (no spurious sign flip) and its CI stays above 0.
    fwd = by[by.index.str.startswith("ofi_forward_h")]
    assert (fwd["point"] > 0).all()
    assert (fwd["ci_low"] > 0).all()

    # Queue imbalance predicts the next move: AUC clearly above 0.5.
    assert by.loc["queue_imbalance_next_move", "point"] > 0.55
    assert by.loc["queue_imbalance_next_move", "ci_low"] > 0.5

    # Trade-sign autocorrelation decays and stays positive at short lags (long memory).
    assert by.loc["sign_autocorr_lag1", "point"] > 0.5
    assert (
        by.loc["sign_autocorr_lag1", "point"]
        > by.loc["sign_autocorr_lag10", "point"]
        > by.loc["sign_autocorr_lag50", "point"]
    )
    # Every reported autocorrelation CI must contain its own point estimate
    # (the Bartlett interval; the block bootstrap did NOT — see DESIGN D-stats.5).
    for lag in R.HORIZON_GRID_EVENTS:
        row = by.loc[f"sign_autocorr_lag{lag}"]
        assert row["ci_low"] <= row["point"] <= row["ci_high"]

    # Impact is concave (log-log exponent strictly between 0 and 1).
    assert 0 < by.loc["impact_curve", "point"] < 1

    # FDR control leaves the bulk significant but is capable of a non-rejection
    # (the long-lag sign autocorrelations decay into noise).
    assert int(df["rejected"].fillna(False).sum()) >= 12


@pytest.mark.slow
def test_robustness_shows_large_tick_queue_effect():
    """Per-symbol split reproduces the Gould-Bonart large-tick signature: queue
    imbalance predicts far better in large-tick names (INTC, MSFT) than in the
    high-price small-tick names (AAPL, GOOG). Skips without data."""
    if not _has_raw():
        pytest.skip("raw LOBSTER data not present")
    rob = runner.run_robustness_by_symbol().set_index("ticker")
    assert rob.loc["INTC", "qi_auc"] > rob.loc["AAPL", "qi_auc"]
    assert rob.loc["MSFT", "qi_auc"] > rob.loc["GOOG", "qi_auc"]
    # OFI R^2 positive for every symbol (robust, not driven by one name).
    assert (rob["ofi_R2_lo"] > 0).all()
