"""Estimators with honest uncertainty — every result carries a confidence interval.

This module is the statistics engine behind the registered study family. Each
public entry point returns a :class:`schema.EstimationResult` with ``ci_low`` /
``ci_high`` always populated (that is the project's non-negotiable claims rule).
Where a closed-form interval exists we use it and verify it against a reference
library to :data:`config.registry.GOLDEN_RTOL`; where the sampling distribution
is intractable (AUC, autocorrelated series) we use a seeded block/paired
bootstrap.

Conventions, chosen to reproduce the reference libraries exactly:

* **HAC / Newey-West** uses the Bartlett kernel with weights ``1 - l/(L+1)`` and
  the meat divisor ``n`` (no small-sample scaling) — this is statsmodels'
  ``OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L})`` default, so our SEs
  match it to numerical precision.
* Coefficient p-values and CIs use the **standard normal** (statsmodels uses
  ``use_t=False`` for HAC), i.e. ``z = Phi^{-1}(1 - alpha/2)``.
* Correlation CIs use the **Fisher z-transform** with ``se = 1/sqrt(n - 3)``,
  which reproduces ``scipy.stats.pearsonr(...).confidence_interval`` exactly and
  is the standard approximate interval for the Spearman rank correlation.

All randomness derives from :data:`config.registry.MASTER_SEED`; nothing here
reads the wall clock.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

from config import registry as R
from src.schema import EstimationResult

# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #


def _design(X: np.ndarray, add_const: bool) -> np.ndarray:
    """Return a 2-d float design matrix, optionally with a leading intercept
    column. A 1-d ``X`` is treated as a single regressor."""
    A = np.asarray(X, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if add_const:
        A = np.column_stack([np.ones(A.shape[0]), A])
    return A


def _z(ci_level: float) -> float:
    """Two-sided normal critical value for the given confidence level."""
    return float(stats.norm.ppf(0.5 + ci_level / 2.0))


def _percentile_ci(samples: np.ndarray, ci_level: float) -> tuple[float, float]:
    """Percentile-bootstrap interval at ``ci_level`` (equal-tailed)."""
    alpha = 1.0 - ci_level
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


# --------------------------------------------------------------------------- #
# OLS with Newey-West / HAC standard errors
# --------------------------------------------------------------------------- #


def _hac_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance of the OLS coefficients (Bartlett kernel,
    meat divisor ``n`` — matches statsmodels' HAC default exactly)."""
    XtX_inv = np.linalg.inv(X.T @ X)
    u = resid
    # Lag-0 term: sum_t u_t^2 x_t x_t'.
    S = (X * (u**2)[:, None]).T @ X
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        xt, xtl = X[lag:], X[:-lag]
        ut, utl = u[lag:], u[:-lag]
        # sum_t u_t u_{t-l} x_t x_{t-l}'  (t running over the overlap).
        g = (xt * (ut * utl)[:, None]).T @ xtl
        S += w * (g + g.T)
    return XtX_inv @ S @ XtX_inv


def ols_hac(
    X: np.ndarray,
    y: np.ndarray,
    *,
    name: str,
    lags: int,
    coef_index: int | None = None,
    add_const: bool = True,
    ci_level: float = R.CI_LEVEL,
) -> EstimationResult:
    """OLS fit with Newey-West/HAC standard errors.

    Reports the coefficient at ``coef_index`` (into the design matrix, which
    includes the intercept when ``add_const``); by default the first slope
    (index 1 with a constant, else 0). The full coefficient vector, HAC SEs,
    two-sided p-values, per-coefficient CIs, and the centered R^2 are returned
    in ``extra`` so a caller can report R^2 or any other coefficient without
    refitting.
    """
    Xd = _design(X, add_const)
    yv = np.asarray(y, dtype=np.float64)
    n, k = Xd.shape
    if coef_index is None:
        coef_index = 1 if (add_const and k > 1) else 0

    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    beta = XtX_inv @ (Xd.T @ yv)
    resid = yv - Xd @ beta
    cov = _hac_cov(Xd, resid, lags)
    bse = np.sqrt(np.diag(cov))

    # Centered R^2 (matches statsmodels when an intercept is present).
    ss_res = float(resid @ resid)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    zval = beta / bse
    p_all = 2.0 * stats.norm.sf(np.abs(zval))
    z = _z(ci_level)
    ci_lo_all = beta - z * bse
    ci_hi_all = beta + z * bse

    i = coef_index
    return EstimationResult(
        name=name,
        metric="coef",
        point=float(beta[i]),
        ci_low=float(ci_lo_all[i]),
        ci_high=float(ci_hi_all[i]),
        n_obs=n,
        std_error=float(bse[i]),
        p_value_raw=float(p_all[i]),
        ci_level=ci_level,
        ci_method="hac",
        extra={
            "coef": beta.copy(),
            "bse": bse.copy(),
            "p_value_raw": p_all.copy(),
            "ci_low": ci_lo_all.copy(),
            "ci_high": ci_hi_all.copy(),
            "r2": r2,
            "lags": lags,
            "coef_index": i,
        },
    )


class OLSHACEstimator:
    """:class:`interfaces.Estimator` implementation for OLS with HAC SEs."""

    def __init__(
        self,
        *,
        lags: int,
        coef_index: int | None = None,
        add_const: bool = True,
        ci_level: float = R.CI_LEVEL,
    ) -> None:
        self.lags = lags
        self.coef_index = coef_index
        self.add_const = add_const
        self.ci_level = ci_level

    def estimate(self, X: np.ndarray, y: np.ndarray, *, name: str) -> EstimationResult:
        return ols_hac(
            X,
            y,
            name=name,
            lags=self.lags,
            coef_index=self.coef_index,
            add_const=self.add_const,
            ci_level=self.ci_level,
        )


# --------------------------------------------------------------------------- #
# Logistic regression (queue-imbalance next-move classification)
# --------------------------------------------------------------------------- #


def logit_coef(
    X: np.ndarray,
    y: np.ndarray,
    *,
    name: str,
    coef_index: int | None = None,
    add_const: bool = True,
    ci_level: float = R.CI_LEVEL,
) -> EstimationResult:
    """Logistic regression via statsmodels; reports one coefficient with a Wald
    (normal-approximation) CI and p-value.

    ``y`` must be binary in {0, 1}. By default the first slope is reported. The
    full parameter vector, SEs, p-values, and CIs are returned in ``extra``.
    """
    Xd = _design(X, add_const)
    yv = np.asarray(y, dtype=np.float64)
    if not np.isin(np.unique(yv), (0.0, 1.0)).all():
        raise ValueError("logit target must be binary in {0, 1}")
    k = Xd.shape[1]
    if coef_index is None:
        coef_index = 1 if (add_const and k > 1) else 0

    res = sm.Logit(yv, Xd).fit(disp=0)
    beta = np.asarray(res.params, dtype=np.float64)
    bse = np.asarray(res.bse, dtype=np.float64)
    p_all = np.asarray(res.pvalues, dtype=np.float64)
    z = _z(ci_level)
    ci_lo_all = beta - z * bse
    ci_hi_all = beta + z * bse

    i = coef_index
    return EstimationResult(
        name=name,
        metric="coef",
        point=float(beta[i]),
        ci_low=float(ci_lo_all[i]),
        ci_high=float(ci_hi_all[i]),
        n_obs=Xd.shape[0],
        std_error=float(bse[i]),
        p_value_raw=float(p_all[i]),
        ci_level=ci_level,
        ci_method="wald",
        extra={
            "coef": beta.copy(),
            "bse": bse.copy(),
            "p_value_raw": p_all.copy(),
            "ci_low": ci_lo_all.copy(),
            "ci_high": ci_hi_all.copy(),
            "coef_index": i,
        },
    )


class LogitEstimator:
    """:class:`interfaces.Estimator` implementation for logistic regression."""

    def __init__(
        self,
        *,
        coef_index: int | None = None,
        add_const: bool = True,
        ci_level: float = R.CI_LEVEL,
    ) -> None:
        self.coef_index = coef_index
        self.add_const = add_const
        self.ci_level = ci_level

    def estimate(self, X: np.ndarray, y: np.ndarray, *, name: str) -> EstimationResult:
        return logit_coef(
            X,
            y,
            name=name,
            coef_index=self.coef_index,
            add_const=self.add_const,
            ci_level=self.ci_level,
        )


# --------------------------------------------------------------------------- #
# Information coefficient (Pearson / Spearman) with a Fisher-z CI
# --------------------------------------------------------------------------- #


def information_coefficient(
    pred: np.ndarray,
    outcome: np.ndarray,
    *,
    name: str,
    method: str = "pearson",
    ci_level: float = R.CI_LEVEL,
) -> EstimationResult:
    """Correlation of a prediction with its outcome — the "IC" of the study.

    ``method="pearson"`` gives the linear IC (point and p-value from
    ``scipy.stats.pearsonr``); ``method="spearman"`` gives the rank IC (point
    and p-value from ``scipy.stats.spearmanr``). Both intervals use the Fisher
    z-transform with ``se = 1/sqrt(n - 3)`` — exact against scipy's Pearson CI,
    and the standard approximate interval for the rank correlation. With fewer
    than four observations the interval is undefined and returned as NaN.
    """
    p = np.asarray(pred, dtype=np.float64)
    o = np.asarray(outcome, dtype=np.float64)
    n = p.shape[0]
    if method == "pearson":
        res = stats.pearsonr(p, o)
    elif method == "spearman":
        res = stats.spearmanr(p, o)
    else:
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    r = float(res.statistic)
    p_raw = float(res.pvalue)

    if n > 3 and abs(r) < 1.0:
        se = 1.0 / np.sqrt(n - 3)
        z = _z(ci_level)
        zr = np.arctanh(r)
        ci_low = float(np.tanh(zr - z * se))
        ci_high = float(np.tanh(zr + z * se))
    else:
        # Fisher transform is undefined at |r| == 1 or n <= 3: report an honest
        # unknown rather than a fabricated interval.
        ci_low = ci_high = float("nan")

    return EstimationResult(
        name=name,
        metric="IC",
        point=r,
        ci_low=ci_low,
        ci_high=ci_high,
        n_obs=n,
        p_value_raw=p_raw,
        ci_level=ci_level,
        ci_method="fisher_z",
        extra={"method": method},
    )


# --------------------------------------------------------------------------- #
# AUC (ROC) with a paired bootstrap CI
# --------------------------------------------------------------------------- #


def auc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
    n_resamples: int = R.BOOTSTRAP_N_RESAMPLES,
    ci_level: float = R.CI_LEVEL,
    seed: int = R.MASTER_SEED,
    block: int | None = None,
    method: str = "stationary",
) -> EstimationResult:
    """ROC AUC (point from ``sklearn.metrics.roc_auc_score``) with a seeded
    paired-bootstrap percentile CI and a genuine two-sided bootstrap p-value
    against the no-skill null AUC = 0.5.

    ``labels`` must be binary in {0, 1}. The bootstrap resamples (score, label)
    pairs with replacement; resamples in which only one class survives are
    skipped (AUC is undefined there).

    When ``block`` is None the resample is i.i.d. (each pair drawn independently);
    when ``block`` is a positive int the resample draws contiguous blocks
    (``method`` = "stationary" Politis-Romano or "circular"), which preserves the
    serial dependence of consecutive observations — the honest choice when the
    (score, label) rows are autocorrelated (e.g. successive order-book buckets),
    for which the i.i.d. interval under-covers. The bootstrap SE and the two-sided
    p-value are reported in ``extra``; ``ci_method`` records which resample was used.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    classes = np.unique(y)
    if not np.isin(classes, (0, 1)).all() or classes.size < 2:
        raise ValueError("AUC requires binary labels with both classes present")
    point = float(roc_auc_score(y, s))

    n = s.shape[0]
    rng = np.random.default_rng(seed)
    if block is None:
        gen = None
        ci_method = "bootstrap_paired"
    elif method == "stationary":
        gen = _stationary_block_index
        ci_method = "bootstrap_block_stationary"
    elif method == "circular":
        gen = _circular_block_index
        ci_method = "bootstrap_block_circular"
    else:
        raise ValueError(f"method must be 'stationary' or 'circular', got {method!r}")

    boot = np.empty(n_resamples, dtype=np.float64)
    valid = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n) if gen is None else gen(n, block, rng)
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue  # degenerate resample — AUC undefined, skip it
        boot[valid] = roc_auc_score(yb, s[idx])
        valid += 1
    boot = boot[:valid]
    ci_low, ci_high = _percentile_ci(boot, ci_level)

    # Two-sided bootstrap p-value against AUC = 0.5, clipped so an all-above-0.5
    # bootstrap yields a finite ~1/(valid+1) rather than a fabricated exact 0.
    if valid > 0:
        tail = min(float(np.mean(boot <= 0.5)), float(np.mean(boot >= 0.5)))
        p_value = min(1.0, max(2.0 * tail, 1.0 / (valid + 1)))
    else:
        p_value = float("nan")

    return EstimationResult(
        name=name,
        metric="AUC",
        point=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n_obs=n,
        std_error=float(np.std(boot, ddof=1)) if valid > 1 else float("nan"),
        p_value_raw=p_value,
        ci_level=ci_level,
        ci_method=ci_method,
        extra={"n_resamples_effective": valid, "block": block},
    )


# --------------------------------------------------------------------------- #
# Block bootstrap for autocorrelated series
# --------------------------------------------------------------------------- #


def autocorrelation(x: np.ndarray, lag: int) -> float:
    """Sample autocorrelation of ``x`` at ``lag`` (biased/divide-by-n estimator,
    the standard one for order-flow long-memory work)."""
    a = np.asarray(x, dtype=np.float64)
    a = a - a.mean()
    denom = float(a @ a)
    if denom == 0.0:
        return float("nan")
    return float(a[lag:] @ a[:-lag] / denom)


def autocorr_bartlett_ci(
    x: np.ndarray, lag: int, *, name: str, ci_level: float = R.CI_LEVEL
) -> EstimationResult:
    """Sample autocorrelation at ``lag`` with a Bartlett large-sample confidence
    interval — the textbook (Box-Jenkins) interval for the ACF of a serially
    correlated series.

    Bartlett's formula for the variance of the lag-k sample autocorrelation of an
    MA-like process is ``Var(r_k) ~= (1 + 2 * sum_{j=1}^{k-1} r_j^2) / n``. Unlike
    a block-bootstrap percentile interval — which resamples blocks and therefore
    ATTENUATES the very autocorrelation being estimated, shifting the interval off
    the point estimate (see docs/DESIGN.md §D-stats.5) — the Bartlett interval is
    centred on the estimate and correctly widens at longer lags as earlier-lag
    correlation accumulates. Returns an ``EstimationResult`` with the two-sided
    p-value against the null r_k = 0. NaN CI/p if the series is constant."""
    a = np.asarray(x, dtype=np.float64)
    n = a.shape[0]
    r_k = autocorrelation(a, lag)
    if not np.isfinite(r_k) or n <= lag + 1:
        return EstimationResult(
            name=name,
            metric="autocorr",
            point=r_k,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_obs=n,
            ci_level=ci_level,
            ci_method="bartlett",
        )
    r_prev = np.array([autocorrelation(a, j) for j in range(1, lag)])
    se = float(np.sqrt((1.0 + 2.0 * np.sum(r_prev**2)) / n))
    z = _z(ci_level)
    p = float(2.0 * stats.norm.sf(abs(r_k) / se)) if se > 0 else float("nan")
    return EstimationResult(
        name=name,
        metric="autocorr",
        point=r_k,
        ci_low=r_k - z * se,
        ci_high=r_k + z * se,
        n_obs=n,
        std_error=se,
        p_value_raw=p,
        ci_level=ci_level,
        ci_method="bartlett",
        extra={"lag": lag},
    )


def _circular_block_index(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Index vector of length ``n`` from wrapped blocks of fixed length ``block``
    starting at uniformly random positions (circular block bootstrap)."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def _stationary_block_index(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Index vector of length ``n`` for the stationary bootstrap (Politis &
    Romano, 1994): geometric block lengths with mean ``block``, wrapped."""
    p = 1.0 / block
    idx = np.empty(n, dtype=np.int64)
    cur = int(rng.integers(0, n))
    for i in range(n):
        idx[i] = cur
        cur = int(rng.integers(0, n)) if rng.random() < p else (cur + 1) % n
    return idx


def block_bootstrap_ci(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    name: str,
    metric: str = "autocorr",
    block: int = R.BOOTSTRAP_BLOCK_EVENTS,
    n_resamples: int = R.BOOTSTRAP_N_RESAMPLES,
    ci_level: float = R.CI_LEVEL,
    seed: int = R.MASTER_SEED,
    method: str = "stationary",
) -> EstimationResult:
    """Percentile CI for ``statistic`` on an autocorrelated series via a seeded
    block bootstrap.

    ``method="stationary"`` (default, Politis-Romano geometric blocks) or
    ``"circular"`` (fixed-length wrapped blocks). Both preserve short-range
    dependence so the interval is honest for serially correlated data (e.g.
    trade-sign autocorrelation). The point estimate is ``statistic(x)`` on the
    original series; the bootstrap SE is reported in ``extra``.
    """
    a = np.asarray(x, dtype=np.float64)
    n = a.shape[0]
    if method == "circular":
        gen = _circular_block_index
    elif method == "stationary":
        gen = _stationary_block_index
    else:
        raise ValueError(f"method must be 'stationary' or 'circular', got {method!r}")

    point = float(statistic(a))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        boot[b] = statistic(a[gen(n, block, rng)])
    ci_low, ci_high = _percentile_ci(boot, ci_level)

    return EstimationResult(
        name=name,
        metric=metric,
        point=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n_obs=n,
        std_error=float(np.std(boot, ddof=1)),
        ci_level=ci_level,
        ci_method=f"block_bootstrap_{method}",
        extra={"block": block, "n_resamples": n_resamples},
    )
