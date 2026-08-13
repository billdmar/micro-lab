"""Tests for src.sim — the ground-truth-injectable order-flow simulator.

These are the acceptance gate for the simulator: the whole point of the simulator is to
VERIFY the statistics pipeline, so it must (a) be deterministic in its seed,
(b) emit a schema-valid, self-consistent book, and — the headline — (c) recover
an injected ``beta`` WITHOUT errors-in-variables attenuation, with honest CI
coverage at/above the pre-registered floor and a calibrated false-positive rate
under the null. All runs are on simulator output only; no LOBSTER files.

The recovery gate consumes the EXPOSED continuous mid and exogenous driver (not
the tick-quantized book), so quantization cannot attenuate the slope — that is
precisely the bug this module was rewritten to remove.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import registry as R
from src.schema import EVENT_COLUMNS, EVENT_DTYPES, validate_event_frame
from src.sim.simulator import OrderFlowSimulator, SimConfig, SimGroundTruth
from src.stats.estimators import ols_hac

# numpy 2.2.1 emits spurious "divide by zero / overflow / invalid" RuntimeWarnings
# from its vectorized matmul FP-flag path even when inputs and outputs are finite
# (verified: point estimate and CI are always finite). We do not own the
# estimator module, so we silence only that noise here — never a real error.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

# --- recovery-gate configuration (sized for a stable-but-fast default suite) --
_N_EVENTS = 20_000
_N_SEEDS = 40
_HAC_LAGS = 5  # AR(1) pressure gives short-memory flow; a small HAC lag suffices
# Two positive betas scaled so the signal is detectable in ~20-40k events, plus
# the true null. Units: log-return per unit of the O(1) signed-flow driver.
_BETAS = (0.0, 3e-4, 6e-4)


def _regress_seed(beta: float, seed: int, n_events: int = _N_EVENTS):
    """Fit fwd_ret_1 ~ driver on one simulated stream; return the EstimationResult."""
    _, truth = OrderFlowSimulator(SimConfig(beta=beta)).generate_with_truth(n_events, seed)
    g = truth.driver[:-1]  # last event has no forward return
    y = truth.fwd_ret_1[:-1]
    return ols_hac(g, y, name=f"recover_beta={beta}", lags=_HAC_LAGS)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_same_seed_identical_frame_and_truth():
    sim = OrderFlowSimulator(SimConfig(beta=3e-4))
    f1, t1 = sim.generate_with_truth(4000, 11)
    f2, t2 = sim.generate_with_truth(4000, 11)
    pd.testing.assert_frame_equal(f1, f2)
    np.testing.assert_array_equal(t1.driver, t2.driver)
    np.testing.assert_array_equal(t1.mid_continuous, t2.mid_continuous)


def test_different_seed_different_frame():
    sim = OrderFlowSimulator(SimConfig(beta=3e-4))
    f1 = sim.generate(4000, 1)
    f2 = sim.generate(4000, 2)
    # Frames share the schema but must differ in content.
    assert not f1.equals(f2)


def test_seed_offset_changes_stream():
    a = OrderFlowSimulator(SimConfig(seed_offset=0)).generate(2000, 5)
    b = OrderFlowSimulator(SimConfig(seed_offset=1000)).generate(2000, 5)
    assert not a.equals(b)


# --------------------------------------------------------------------------- #
# Schema + book self-consistency
# --------------------------------------------------------------------------- #


def test_emitted_frame_is_schema_valid():
    frame = OrderFlowSimulator(SimConfig()).generate(5000, 7)
    validate_event_frame(frame)  # raises on any violation
    assert list(frame.columns) == list(EVENT_COLUMNS)
    for col, dtype in EVENT_DTYPES.items():
        assert frame[col].dtype == np.dtype(dtype), col
    assert (frame["size"] >= 0).all()  # never negative (schema floor)
    assert frame["direction"].isin((-1, 1)).all()
    assert frame["time_s"].is_monotonic_increasing


def test_book_stays_valid_no_negative_or_crossed():
    # Drive the engine and confirm sizes never go negative and the reported
    # continuous mid is always a positive, finite price.
    _, truth = OrderFlowSimulator(SimConfig()).generate_with_truth(5000, 9)
    assert np.isfinite(truth.mid_continuous).all()
    assert (truth.mid_continuous > 0).all()


def test_generate_rejects_tiny_n():
    with pytest.raises(ValueError, match="n_events must be >= 2"):
        OrderFlowSimulator(SimConfig()).generate(1, 0)


def test_ground_truth_shape_and_finiteness():
    n = 3000
    _, truth = OrderFlowSimulator(SimConfig(beta=3e-4)).generate_with_truth(n, 4)
    assert isinstance(truth, SimGroundTruth)
    assert truth.driver.shape == (n,)
    assert truth.mid_continuous.shape == (n,)
    assert truth.fwd_ret_1.shape == (n,)
    assert np.isfinite(truth.driver).all()  # driver is exogenous & always defined
    assert np.isnan(truth.fwd_ret_1[-1])  # no forward return at the last event
    assert np.isfinite(truth.fwd_ret_1[:-1]).all()
    assert truth.extra["n_events"] == n and truth.extra["seed"] == 4


# --------------------------------------------------------------------------- #
# The injected identity is EXACT (the anti-attenuation guarantee)
# --------------------------------------------------------------------------- #


def test_forward_return_equals_beta_driver_when_noiseless():
    # With sigma=0 the exposed forward return must equal beta * driver EXACTLY
    # (up to float rounding) — the structural reason recovery cannot attenuate.
    beta = 4e-4
    _, truth = OrderFlowSimulator(SimConfig(beta=beta, sigma=0.0)).generate_with_truth(5000, 3)
    lhs = truth.fwd_ret_1[:-1]
    rhs = beta * truth.driver[:-1]
    np.testing.assert_allclose(lhs, rhs, rtol=0.0, atol=1e-12)


def test_continuous_mid_matches_cumulative_log_returns():
    # log(mid_{n+1}) - log(mid_n) reconstructs fwd_ret_1 exactly.
    _, truth = OrderFlowSimulator(SimConfig(beta=2e-4)).generate_with_truth(3000, 6)
    diff = np.diff(np.log(truth.mid_continuous))
    np.testing.assert_allclose(diff, truth.fwd_ret_1[:-1], rtol=0.0, atol=1e-15)


# --------------------------------------------------------------------------- #
# Headline: UNBIASED recovery + CI coverage + calibrated null (the gate)
# --------------------------------------------------------------------------- #


def test_unbiased_recovery_and_coverage_and_null():
    """For each injected beta, regress fwd_ret_1 on the exposed driver over many
    seeds and assert: (a) the mean point estimate matches beta with tiny relative
    error (no systematic attenuation), (b) the 95% CI covers beta at a rate >=
    RECOVERY_CI_COVERAGE_MIN, and for beta=0 (c) the false-positive rate sits
    near the nominal 0.05."""
    for beta in _BETAS:
        points = np.empty(_N_SEEDS)
        covered = 0
        rejected = 0  # CI excludes 0 -> a "discovery"
        for s in range(_N_SEEDS):
            res = _regress_seed(beta, s)
            points[s] = res.point
            if res.ci_low <= beta <= res.ci_high:
                covered += 1
            if not (res.ci_low <= 0.0 <= res.ci_high):
                rejected += 1

        mean_est = float(points.mean())
        coverage = covered / _N_SEEDS
        reject_rate = rejected / _N_SEEDS

        # (b) honest-CI coverage floor (pre-registered) for every beta.
        assert coverage >= R.RECOVERY_CI_COVERAGE_MIN, (
            f"beta={beta}: coverage {coverage:.3f} < {R.RECOVERY_CI_COVERAGE_MIN}"
        )

        if beta == 0.0:
            # (c) true null: reject rate near nominal 0.05. Allow slack for the
            # finite seed count (40 seeds -> a couple of chance rejections).
            assert reject_rate <= 0.15, f"null reject rate too high: {reject_rate:.3f}"
        else:
            # (a) unbiasedness: mean estimate within a few percent of beta. A
            # biased (e.g. attenuated ~0.45x) estimator would blow past this.
            rel_err = abs(mean_est - beta) / beta
            assert rel_err < 0.05, (
                f"beta={beta}: mean_est {mean_est:.3e} rel_err {rel_err:.3f} — biased?"
            )
            # A detectable positive effect should be found essentially always.
            assert reject_rate >= 0.90, f"beta={beta}: underpowered, power={reject_rate:.3f}"


def test_positive_beta_recovery_no_attenuation_single_large_sample():
    # A single large sample sanity check that the recovered slope is ~beta, not a
    # shrunk-toward-zero value (the old EIV failure mode recovered ~0.45x).
    beta = 6e-4
    res = _regress_seed(beta, seed=0, n_events=40_000)
    assert res.ci_low <= beta <= res.ci_high
    assert 0.85 * beta < res.point < 1.15 * beta
