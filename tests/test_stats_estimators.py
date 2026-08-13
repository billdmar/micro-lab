"""Tests for src.stats — golden checks against reference libraries and honest-CI
behavior. Everything runs on small synthetic data with a known reference answer;
no LOBSTER files are touched.

The headline verifications (per MISSION):
  * HAC/Newey-West SEs vs statsmodels OLS(cov_type="HAC") to GOLDEN_RTOL.
  * AUC vs sklearn.metrics.roc_auc_score.
  * Rank IC vs scipy.stats.spearmanr; Pearson IC vs scipy.stats.pearsonr.
  * Block-bootstrap CI coverage on a synthetic AR(1) series near nominal 95%.
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

from config import registry as R
from src.schema import EstimationResult
from src.stats import estimators as E

# numpy 2.2 emits spurious FP-flag RuntimeWarnings from its vectorized matmul path
# on the deliberately-degenerate (constant / tiny) series these goldens feed in;
# the estimates and CIs are finite and golden-matched. Silence only that class
# (same convention as the sibling test modules); the strict error::DeprecationWarning
# rule for our own code is untouched.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

RTOL = R.GOLDEN_RTOL


def _ar1(n: int, phi: float, rng: np.random.Generator, sigma: float = 1.0) -> np.ndarray:
    """Zero-mean AR(1) series x_t = phi x_{t-1} + eps_t."""
    x = np.empty(n, dtype=np.float64)
    x[0] = rng.normal(scale=sigma / np.sqrt(1 - phi**2))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(scale=sigma)
    return x


# --------------------------------------------------------------------------- #
# OLS + HAC golden
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("lags", [0, 1, 4, 10])
def test_ols_hac_matches_statsmodels(lags):
    rng = np.random.default_rng(R.MASTER_SEED + 7)
    n = 250
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    e = rng.normal(size=n)
    for t in range(1, n):  # autocorrelated errors so HAC != OLS SEs
        e[t] += 0.6 * e[t - 1]
    y = 1.5 + 2.0 * x1 - 0.7 * x2 + e
    X = np.column_stack([x1, x2])

    res = E.ols_hac(X, y, name="g", lags=lags, coef_index=1)

    Xsm = sm.add_constant(X)
    ref = sm.OLS(y, Xsm).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    ref_ci = ref.conf_int(alpha=1 - R.CI_LEVEL)

    np.testing.assert_allclose(res.extra["coef"], ref.params, rtol=RTOL)
    np.testing.assert_allclose(res.extra["bse"], ref.bse, rtol=RTOL)
    np.testing.assert_allclose(res.extra["p_value_raw"], ref.pvalues, rtol=RTOL, atol=1e-300)
    np.testing.assert_allclose(res.extra["r2"], ref.rsquared, rtol=RTOL)
    # Reported coefficient (index 1) point/SE/CI line up with statsmodels.
    assert res.point == pytest.approx(ref.params[1], rel=RTOL)
    assert res.std_error == pytest.approx(ref.bse[1], rel=RTOL)
    assert res.ci_low == pytest.approx(ref_ci[1, 0], rel=RTOL)
    assert res.ci_high == pytest.approx(ref_ci[1, 1], rel=RTOL)
    assert res.ci_method == "hac" and res.metric == "coef" and res.n_obs == n


def test_ols_hac_default_coef_index_and_no_const():
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    y = 3.0 * x + rng.normal(size=80)
    # No intercept: the single regressor is index 0 and default target.
    res = E.ols_hac(x, y, name="noconst", lags=2, add_const=False)
    ref = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    assert res.point == pytest.approx(ref.params[0], rel=RTOL)
    assert res.std_error == pytest.approx(ref.bse[0], rel=RTOL)


def test_ols_hac_estimator_protocol():
    from src import interfaces

    est = E.OLSHACEstimator(lags=3)
    assert isinstance(est, interfaces.Estimator)
    rng = np.random.default_rng(2)
    x = rng.normal(size=100)
    y = 1.0 + x + rng.normal(size=100)
    out = est.estimate(x, y, name="proto")
    assert isinstance(out, EstimationResult) and out.ci_method == "hac"


# --------------------------------------------------------------------------- #
# Logistic regression
# --------------------------------------------------------------------------- #


def test_logit_matches_statsmodels_and_ci():
    rng = np.random.default_rng(11)
    n = 400
    x = rng.normal(size=n)
    prob = 1.0 / (1.0 + np.exp(-(0.5 + 1.2 * x)))
    y = (rng.random(n) < prob).astype(int)

    res = E.logit_coef(x, y, name="qimb")
    Xsm = sm.add_constant(x)
    ref = sm.Logit(y, Xsm).fit(disp=0)

    assert res.point == pytest.approx(ref.params[1], rel=RTOL)
    assert res.std_error == pytest.approx(ref.bse[1], rel=RTOL)
    assert res.p_value_raw == pytest.approx(ref.pvalues[1], rel=RTOL)
    z = stats.norm.ppf(0.5 + R.CI_LEVEL / 2)
    assert res.ci_low == pytest.approx(ref.params[1] - z * ref.bse[1], rel=RTOL)
    assert res.ci_high == pytest.approx(ref.params[1] + z * ref.bse[1], rel=RTOL)


def test_logit_rejects_non_binary():
    rng = np.random.default_rng(3)
    x = rng.normal(size=20)
    with pytest.raises(ValueError):
        E.logit_coef(x, np.full(20, 2), name="bad")


def test_logit_estimator_protocol_and_explicit_coef_index():
    from src import interfaces

    est = E.LogitEstimator(coef_index=0)  # report the intercept
    assert isinstance(est, interfaces.Estimator)
    rng = np.random.default_rng(4)
    x = rng.normal(size=200)
    y = (rng.random(200) < 1 / (1 + np.exp(-(0.3 + x)))).astype(int)
    out = est.estimate(x, y, name="intercept")
    ref = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    assert out.point == pytest.approx(ref.params[0], rel=RTOL)


# --------------------------------------------------------------------------- #
# Information coefficient (Pearson / Spearman) goldens
# --------------------------------------------------------------------------- #


def test_pearson_ic_matches_scipy():
    rng = np.random.default_rng(21)
    n = 200
    pred = rng.normal(size=n)
    outcome = 0.6 * pred + rng.normal(size=n)
    res = E.information_coefficient(pred, outcome, name="ic", method="pearson")
    ref = stats.pearsonr(pred, outcome)
    ref_ci = ref.confidence_interval(R.CI_LEVEL)
    assert res.point == pytest.approx(ref.statistic, rel=RTOL)
    assert res.p_value_raw == pytest.approx(ref.pvalue, rel=RTOL)
    assert res.ci_low == pytest.approx(ref_ci.low, rel=RTOL)
    assert res.ci_high == pytest.approx(ref_ci.high, rel=RTOL)
    assert res.ci_method == "fisher_z"


def test_spearman_ic_matches_scipy():
    rng = np.random.default_rng(22)
    n = 200
    pred = rng.normal(size=n)
    outcome = pred**3 + rng.normal(size=n)  # monotone -> strong rank corr
    res = E.information_coefficient(pred, outcome, name="ric", method="spearman")
    ref = stats.spearmanr(pred, outcome)
    assert res.point == pytest.approx(ref.statistic, rel=RTOL)
    assert res.p_value_raw == pytest.approx(ref.pvalue, rel=RTOL)
    # Fisher-z interval on the rank correlation (n > 3, |r| < 1).
    se = 1.0 / np.sqrt(n - 3)
    z = stats.norm.ppf(0.5 + R.CI_LEVEL / 2)
    assert res.ci_low == pytest.approx(np.tanh(np.arctanh(ref.statistic) - z * se), rel=RTOL)


def test_ic_bad_method_and_small_n_nan_ci():
    with pytest.raises(ValueError):
        E.information_coefficient(np.arange(5.0), np.arange(5.0), name="x", method="kendall")
    # n <= 3: interval undefined, honest NaN (point still defined).
    res = E.information_coefficient(np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0]), name="s")
    assert np.isnan(res.ci_low) and np.isnan(res.ci_high)
    assert not np.isnan(res.point)


# --------------------------------------------------------------------------- #
# AUC golden + bootstrap CI
# --------------------------------------------------------------------------- #


def test_auc_point_matches_sklearn():
    rng = np.random.default_rng(31)
    n = 300
    y = rng.integers(0, 2, size=n)
    scores = y * 0.8 + rng.normal(size=n)  # informative scores
    res = E.auc_ci(scores, y, name="auc", n_resamples=200)
    assert res.point == pytest.approx(roc_auc_score(y, scores), rel=RTOL)
    assert res.metric == "AUC" and res.ci_method == "bootstrap_paired"
    # CI brackets the point estimate and stays inside [0, 1].
    assert 0.0 <= res.ci_low <= res.point <= res.ci_high <= 1.0


def test_auc_is_deterministic_under_seed():
    rng = np.random.default_rng(32)
    y = rng.integers(0, 2, size=150)
    scores = rng.normal(size=150) + y
    a = E.auc_ci(scores, y, name="a", n_resamples=300, seed=123)
    b = E.auc_ci(scores, y, name="a", n_resamples=300, seed=123)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_auc_rejects_single_class():
    with pytest.raises(ValueError):
        E.auc_ci(np.arange(10.0), np.ones(10, dtype=int), name="bad")


def test_auc_skips_degenerate_resamples():
    # A single positive among many negatives makes some bootstrap resamples
    # all-negative; those must be skipped rather than crash roc_auc_score.
    y = np.zeros(30, dtype=int)
    y[0] = 1
    scores = np.arange(30.0)
    res = E.auc_ci(scores, y, name="rare", n_resamples=200, seed=1)
    assert res.extra["n_resamples_effective"] < 200  # some resamples were skipped
    assert res.ci_low <= res.ci_high


# --------------------------------------------------------------------------- #
# Autocorrelation + block bootstrap
# --------------------------------------------------------------------------- #


def test_autocorrelation_matches_reference():
    rng = np.random.default_rng(41)
    x = _ar1(500, 0.5, rng)
    # Reference: numpy correlate-based biased acf at lag 1.
    a = x - x.mean()
    ref = float(a[1:] @ a[:-1] / (a @ a))
    assert E.autocorrelation(x, 1) == pytest.approx(ref, rel=1e-12)
    assert np.isnan(E.autocorrelation(np.zeros(10), 1))  # constant -> undefined


@pytest.mark.parametrize("method", ["stationary", "circular"])
def test_block_bootstrap_is_deterministic(method):
    rng = np.random.default_rng(42)
    x = _ar1(400, 0.4, rng)
    stat = lambda s: E.autocorrelation(s, 1)  # noqa: E731
    a = E.block_bootstrap_ci(x, stat, name="ac", block=50, n_resamples=300, seed=7, method=method)
    b = E.block_bootstrap_ci(x, stat, name="ac", block=50, n_resamples=300, seed=7, method=method)
    assert (a.point, a.ci_low, a.ci_high) == (b.point, b.ci_low, b.ci_high)
    assert a.ci_low <= a.point <= a.ci_high
    assert a.ci_method == f"block_bootstrap_{method}"


def test_block_bootstrap_bad_method():
    with pytest.raises(ValueError):
        E.block_bootstrap_ci(np.arange(20.0), lambda s: float(s.mean()), name="x", method="iid")


@pytest.mark.parametrize("method", ["stationary", "circular"])
def test_block_bootstrap_ci_coverage_on_ar1(method):
    """Coverage check on a serially correlated series: over repeated AR(1) draws,
    does the CI for the sample mean cover the true mean (0) at ~95%?

    The mean of an AR(1) is the canonical case where the block bootstrap is
    needed: it preserves the dependence and reaches nominal coverage, whereas an
    i.i.d. bootstrap ignores the autocorrelation and under-covers badly. We
    assert both — the block CI clears the registry floor AND beats the naive
    i.i.d. interval on the same draws. (The percentile CI for the autocorrelation
    *coefficient* is more biased and under-covers; we do not overstate it here.)"""
    phi = 0.5
    true_mean = 0.0  # AR(1) is zero-mean by construction
    n_trials = 300
    n = 600
    master = np.random.default_rng(R.MASTER_SEED)
    stat = lambda s: float(s.mean())  # noqa: E731
    covered_block = 0
    covered_iid = 0
    for _ in range(n_trials):
        x = _ar1(n, phi, master)
        boot_seed = int(master.integers(0, 2**31 - 1))
        res = E.block_bootstrap_ci(
            x, stat, name="mean", block=40, n_resamples=400, seed=boot_seed, method=method
        )
        if res.ci_low <= true_mean <= res.ci_high:
            covered_block += 1
        # Naive i.i.d. percentile bootstrap of the same statistic, same seed.
        rng = np.random.default_rng(boot_seed)
        iid = np.array([x[rng.integers(0, n, size=n)].mean() for _ in range(400)])
        if np.quantile(iid, 0.025) <= true_mean <= np.quantile(iid, 0.975):
            covered_iid += 1
    cov_block = covered_block / n_trials
    cov_iid = covered_iid / n_trials
    # Nominal 95%; block CIs may under-cover slightly -> registry floor. The
    # dependence-aware interval must beat the i.i.d. one that ignores it.
    assert cov_block >= R.RECOVERY_CI_COVERAGE_MIN, (cov_block, cov_iid)
    assert cov_block > cov_iid, (cov_block, cov_iid)


def test_autocorr_bartlett_ci_matches_statsmodels_and_contains_point():
    """The Bartlett ACF interval matches statsmodels' acf(..., alpha=) confidence
    band (which uses Bartlett's formula) and always contains its own point."""
    master = np.random.default_rng(R.MASTER_SEED)
    # AR(1) so there is real autocorrelation to bound.
    n, phi = 8000, 0.6
    x = np.zeros(n)
    eps = master.standard_normal(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]

    from statsmodels.tsa.stattools import acf as sm_acf

    max_lag = 20
    acf_vals, confint = sm_acf(x, nlags=max_lag, alpha=0.05, fft=False, bartlett_confint=True)
    for lag in (1, 5, 10, 20):
        res = E.autocorr_bartlett_ci(x, lag, name=f"acf{lag}")
        # point matches the biased/divide-by-n acf statsmodels reports
        assert abs(res.point - acf_vals[lag]) < 1e-9
        # CI half-width matches statsmodels' Bartlett band (centered form)
        sm_half = (confint[lag][1] - confint[lag][0]) / 2.0
        our_half = (res.ci_high - res.ci_low) / 2.0
        assert abs(our_half - sm_half) < 1e-6
        # the interval must contain its own estimate (the property the block
        # bootstrap violated; see DESIGN D-stats.5)
        assert res.ci_low <= res.point <= res.ci_high
        assert np.isfinite(res.p_value_raw)


def test_autocorr_bartlett_ci_constant_series_is_nan():
    res = E.autocorr_bartlett_ci(np.ones(100), 1, name="const")
    assert np.isnan(res.point) or np.isnan(res.ci_low)


def test_auc_ci_iid_path_unchanged_and_block_path_valid():
    """The default (block=None) path stays the i.i.d. paired bootstrap (goldens
    rely on ci_method), and the block path is a valid stationary-block resample
    with a genuine two-sided bootstrap p-value against AUC=0.5."""
    rng = np.random.default_rng(R.MASTER_SEED)
    n = 2000
    score = rng.standard_normal(n)
    label = (rng.random(n) < 1.0 / (1.0 + np.exp(-score))).astype(int)

    iid = E.auc_ci(score, label, name="iid", n_resamples=400)
    assert iid.ci_method == "bootstrap_paired"
    assert iid.ci_low < iid.point < iid.ci_high
    assert 0.0 < iid.p_value_raw <= 1.0

    blk = E.auc_ci(score, label, name="blk", n_resamples=400, block=50)
    assert blk.ci_method == "bootstrap_block_stationary"
    assert blk.point == iid.point  # same sklearn point estimate
    assert blk.ci_low < blk.point < blk.ci_high
    assert 0.0 < blk.p_value_raw <= 1.0


def test_auc_ci_block_p_value_is_large_for_no_skill_scores():
    """Random scores unrelated to labels -> AUC ~ 0.5 -> the bootstrap p-value
    against 0.5 should be large (not a fabricated 0)."""
    rng = np.random.default_rng(R.MASTER_SEED + 7)
    n = 1500
    label = (rng.random(n) < 0.5).astype(int)
    score = rng.standard_normal(n)  # independent of label
    res = E.auc_ci(score, label, name="noskill", n_resamples=400, block=40)
    assert res.p_value_raw > 0.10  # cannot reject no-skill
    assert res.p_value_raw >= 1.0 / (res.extra["n_resamples_effective"] + 1)
