"""verification — the verification asserted as tests.

These tests assert the machinery thresholds from ``config/registry.py`` that must
pass before ANY registered study runs on real data. The synthetic-recovery,
placebo, leakage-detection, and estimator-golden checks run on generated/known
data and execute in the default (CI) suite. The reconstruction-invariant check
needs the raw LOBSTER files and is marked ``slow`` (deselected in CI, run locally
before the studies). No threshold is widened here — they are imported from the
registry.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from config import registry as R
from src.cv.leakage import audit_fold, detect_future_features
from src.cv.splitter import PurgedWalkForwardSplitter
from src.schema import FeatureSpec
from src.sim.simulator import OrderFlowSimulator, SimConfig
from src.stats import estimators as E

# numpy 2.2 emits spurious FP-flag RuntimeWarnings from vectorized matmul inside
# ols_hac even though every estimate/CI is finite; silence only that class here.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# --------------------------------------------------------------------------- #
# (b) Synthetic recovery — injected beta recovered within CI at measured power
# --------------------------------------------------------------------------- #


def _recover(beta: float, n_events: int, seed: int, lags: int = 5):
    sim = OrderFlowSimulator(SimConfig(beta=beta))
    _, truth = sim.generate_with_truth(n_events, seed)
    g, y = truth.driver, truth.fwd_ret_1
    mask = np.isfinite(g) & np.isfinite(y)
    res = E.ols_hac(g[mask].reshape(-1, 1), y[mask], name="recovery", lags=lags)
    covers = res.ci_low <= beta <= res.ci_high
    rejects_zero = not (res.ci_low <= 0.0 <= res.ci_high)
    return res.point, covers, rejects_zero


def test_synthetic_recovery_is_unbiased_and_covers():
    """Injected beta recovered with negligible bias and CI coverage >= the
    registry minimum, across a grid of signal strengths and seeds."""
    n_seeds = 30
    for beta in (3e-4, 6e-4):
        points, covers, power = [], 0, 0
        for s in range(1, n_seeds + 1):
            pt, cov, rej = _recover(beta, 20_000, s)
            points.append(pt)
            covers += int(cov)
            power += int(rej)
        mean_est = float(np.mean(points))
        rel_err = abs(mean_est - beta) / beta
        assert rel_err < 0.05, (
            f"beta={beta}: mean estimate {mean_est:.3e} is biased ({rel_err:.1%})"
        )
        assert covers / n_seeds >= R.RECOVERY_CI_COVERAGE_MIN, (
            f"beta={beta}: CI coverage {covers / n_seeds:.2f} < {R.RECOVERY_CI_COVERAGE_MIN}"
        )
        # A real signal at this strength should be detected essentially always.
        assert power / n_seeds >= 0.90, f"beta={beta}: power {power / n_seeds:.2f} too low"


# --------------------------------------------------------------------------- #
# (c) Placebo — beta = 0 rejects at ~ the nominal alpha (no false discoveries)
# --------------------------------------------------------------------------- #


def test_placebo_null_holds_false_positive_rate():
    """With beta = 0 (a true null), the fraction of runs whose 95% CI excludes 0
    sits near the nominal alpha — the pipeline stays silent on noise."""
    n_seeds = 60
    false_pos = sum(_recover(0.0, 20_000, s)[2] for s in range(1, n_seeds + 1))
    rate = false_pos / n_seeds
    # Allow sampling slack around the nominal 0.05 (n=60): must not be inflated.
    assert rate <= 0.15, f"placebo false-positive rate {rate:.3f} is inflated above nominal"


# --------------------------------------------------------------------------- #
# (d) Leakage detection — constructed leaks are caught
# --------------------------------------------------------------------------- #


def test_leakage_detectors_catch_constructed_leaks():
    # A future-timestamped feature (info_offset > 0 but not a label) must be flagged.
    peeking = FeatureSpec("peek", "uses future info", info_offset=5.0)
    # FeatureSpec forbids a non-label horizon>0, but info_offset>0 with horizon 0
    # is exactly the "feature that peeks" leak the detector must catch.
    findings = detect_future_features([peeking])
    assert any(f.kind == "future_feature" for f in findings), "peeking feature not flagged"

    # A train block whose label window overlaps the test block (missing purge)
    # must be flagged by the fold audit.
    train = np.arange(0, 100)
    test = np.arange(100, 150)  # adjacent -> label windows of train reach into test
    leaky = audit_fold(train, test, label_horizon=20, embargo=0, specs=())
    assert any(f.kind in ("label_overlap", "embargo_breach") for f in leaky), (
        "adjacent train/test with horizon>embargo not flagged"
    )


def test_clean_purged_split_has_no_leakage_and_is_disjoint():
    """A correct purged + embargoed walk-forward split raises no findings and
    produces disjoint train/test with the embargo respected."""
    n = 2_000
    horizon = 10
    embargo = R.CV_EMBARGO_EVENTS
    splitter = PurgedWalkForwardSplitter(mode="walkforward")
    folds = list(splitter.split(n, label_horizon=horizon, embargo=embargo))
    assert folds, "splitter produced no folds"
    for train, test in folds:
        assert set(train).isdisjoint(set(test)), "train/test overlap"
        # strictly causal: all train indices precede the test block
        assert train.max() < test.min(), "walk-forward train is not strictly in the past"
        # embargo respected: no train index within `embargo` before the test block
        gap = test.min() - train.max()
        assert gap > embargo, f"embargo {embargo} not respected (gap {gap})"
        findings = audit_fold(train, test, label_horizon=horizon, embargo=embargo, specs=())
        assert not findings, f"clean split flagged: {findings}"


# --------------------------------------------------------------------------- #
# (e) Estimator goldens — sanity that the golden machinery is wired (full goldens
#     live in tests/test_stats_estimators.py). Here: HAC recovers a known slope.
# --------------------------------------------------------------------------- #


def test_hac_estimator_recovers_known_ols_slope():
    rng = np.random.default_rng(R.MASTER_SEED)
    n = 5_000
    x = rng.standard_normal(n)
    y = 2.5 * x + rng.standard_normal(n)
    res = E.ols_hac(x.reshape(-1, 1), y, name="golden", lags=10)
    assert res.ci_low <= 2.5 <= res.ci_high
    assert abs(res.point - 2.5) < 0.1


def test_g1_gate_recovery_curve_smoke():
    """Exercise the gate's recovery_curve harness on a tiny grid (fast): the null
    stays near nominal and a strong signal is recovered close to its beta."""
    from src.studies.g1_gate import recovery_curve

    rows = recovery_curve(betas=(0.0, 6e-4), n_events=6_000, n_seeds=6, lags=5)
    by_beta = {r.beta: r for r in rows}
    assert by_beta[0.0].reject_rate <= 0.5  # null: few discoveries on this tiny grid
    strong = by_beta[6e-4]
    # This is a fast harness smoke test (only 6 seeds), so assert the point
    # estimate is on target and coverage is plausible; the real coverage gate is
    # test_synthetic_recovery_is_unbiased_and_covers (30 seeds).
    assert abs(strong.mean_estimate - 6e-4) / 6e-4 < 0.1
    assert strong.ci_coverage >= 0.5
    assert 0.0 <= strong.ci_coverage <= 1.0 and strong.n_seeds == 6


# --------------------------------------------------------------------------- #
# (a) Reconstruction transition invariant — SLOW, needs raw LOBSTER data
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_reconstruction_transition_invariant_all_tickers():
    """On every ticker, the exact per-event price-level transition invariant holds
    at 100% (tolerance 0) — the independent correctness oracle. Skips cleanly if
    the raw data has not been downloaded."""
    from src.lobster import parser
    from src.studies.g1_gate import _raw_paths, transition_invariant

    if not os.path.exists(_raw_paths(R.DATA.tickers[0])[0]):
        pytest.skip("raw LOBSTER data not present (run scripts/get_data.sh)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total_checked = 0
        for ticker in R.DATA.tickers:
            msg_path, ob_path = _raw_paths(ticker)
            msg = parser.read_message_frame(msg_path)
            ob = parser.read_orderbook_frame(ob_path, levels=10)
            checked, viol = transition_invariant(msg, ob)
            total_checked += checked
            assert checked > 0, f"{ticker}: no transitions checked"
            rate = 1.0 - viol / checked
            assert rate >= R.RECON_SUBMIT_INVARIANT_TARGET, (
                f"{ticker}: transition invariant {rate:.6f} < target "
                f"({viol} violations / {checked} checked)"
            )
        assert total_checked > 1_000_000  # sanity: we really swept the day


@pytest.mark.slow
def test_g1_gate_reconstruct_ticker_and_report(capsys):
    """Exercise the empty-start reconstruction path (reconstruct_ticker) on one
    ticker and the human-readable print_report end to end. Skips without raw data."""
    from src.studies.g1_gate import _raw_paths, print_report, reconstruct_ticker

    if not os.path.exists(_raw_paths(R.DATA.tickers[0])[0]):
        pytest.skip("raw LOBSTER data not present (run scripts/get_data.sh)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rr = reconstruct_ticker("GOOG")
        assert rr.transition_invariant_rate == 1.0
        assert rr.transitions_checked > 0
        assert 0.0 <= rr.seeded_touch_match <= 1.0
        assert 0.0 <= rr.fullbook_message_only <= 1.0
        print_report()
    out = capsys.readouterr().out
    assert "verification" in out
    assert "RECONSTRUCTION DIFFERENTIAL" in out
    assert "SYNTHETIC RECOVERY" in out
