"""Real-data study runners.

Runs the pre-registered study family (``config.registry.STUDY_FAMILY``) on the
LOBSTER sample through the verified machinery — the feature library, the
estimators (HAC errors, block-bootstrap CIs, AUC), and the purged
walk-forward splitter — producing an ``EstimationResult`` per test, each with a
confidence interval and a raw p-value ready for FDR control.

Methodological choices (recorded in docs/DESIGN.md §):

* **Features are computed from LOBSTER's REFERENCE orderbook**, not from our
  reconstruction. The reconstruction differential () already established that
  we can rebuild the book correctly from messages; for the studies we use the
  reference book so feature values are exact at every displayed level and the
  free sample's deep-level limitation (DESIGN §D1.1) is irrelevant to them.
* **Interval (event-bucket) OFI regressions**, following Cont–Kukanov–Stoikov:
  the day is cut into non-overlapping buckets of ``bucket`` events; per bucket we
  sum the per-event OFI and take the mid-price change. The contemporaneous study
  regresses ΔP on OFI in the SAME bucket (and reports the HAC slope as the test,
  R² as a descriptive statistic); the forward study regresses the mid change over
  the NEXT bucket on this bucket's OFI, out-of-sample, under a per-symbol purged
  walk-forward split — strictly point-in-time.
* **Point-in-time**: every predictor is known at bucket close; every forward
  label is realized strictly later; the forward study fits only on held-out folds
  of each symbol's own timeline (per-symbol purged + embargoed walk-forward),
  pooling the out-of-sample pairs afterward.
* **Valid, per-test p-values for the FDR family**: coefficient/slope tests carry
  the HAC analytic p-value; the AUC carries a genuine two-sided block-bootstrap
  p-value against 0.5; the autocorrelations carry the Bartlett analytic p-value.
  Each test enters Benjamini–Hochberg with a p-value valid for its own estimator
  (no p-value is manufactured from a CI half-width). The R² statistic — bounded
  and one-sided under the null — is reported descriptively with a bootstrap CI,
  never as an FDR test.

No PnL, tradability, or forecast claim is made anywhere — only effect sizes with
CIs under FDR control. Determinism: all randomness seeds from ``MASTER_SEED``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config import registry as R
from src.features.engine import tick_rule_sign
from src.lobster import parser
from src.schema import (
    NO_ASK_PRICE,
    NO_BID_PRICE,
    EstimationResult,
    EventType,
    mid_price_series,
)
from src.stats import estimators as E

RAW = "data/raw"
LEVELS = 10
EXEC_TYPES = (int(EventType.EXECUTE_VISIBLE), int(EventType.EXECUTE_HIDDEN))


# --------------------------------------------------------------------------- #
# Data loading + per-event primitives (from the reference book)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SymbolData:
    """Per-event primitives for one ticker, all aligned and point-in-time."""

    ticker: str
    n_events: int
    mid: np.ndarray  # mid price (1/10000 dollar), NaN on empty touch
    ofi_event: np.ndarray  # per-event OFI increment (NaN where undefined)
    signed_size: np.ndarray  # signed executed size (+buy/-sell), 0 off-trades
    tick_sign: np.ndarray  # tick-rule trade sign at executions, NaN elsewhere
    queue_imbalance: np.ndarray  # (bid_sz1 - ask_sz1)/(sum) at the touch


def _raw_paths(ticker: str) -> tuple[str, str]:
    stem = f"{ticker}_{R.DATA.session}_34200000_57600000"
    return f"{RAW}/{stem}_message_{LEVELS}.csv", f"{RAW}/{stem}_orderbook_{LEVELS}.csv"


def load_symbol(ticker: str) -> SymbolData:
    """Load one ticker and compute the per-event primitives from the reference
    orderbook (exact book) and the message stream."""
    msg_path, ob_path = _raw_paths(ticker)
    events = parser.read_message_frame(msg_path)
    book = parser.read_orderbook_frame(ob_path, levels=LEVELS)

    mid = mid_price_series(book).to_numpy()

    # Per-event OFI increment from best-quote changes (Cont-Kukanov-Stoikov).
    pb = book["bid_px_1"].to_numpy(dtype="float64")
    qb = book["bid_sz_1"].to_numpy(dtype="float64")
    pa = book["ask_px_1"].to_numpy(dtype="float64")
    qa = book["ask_sz_1"].to_numpy(dtype="float64")
    ofi = np.full(len(book), np.nan, dtype="float64")
    if len(book) > 1:
        bid_ge = (pb[1:] >= pb[:-1]).astype("float64")
        bid_le = (pb[1:] <= pb[:-1]).astype("float64")
        ask_le = (pa[1:] <= pa[:-1]).astype("float64")
        ask_ge = (pa[1:] >= pa[:-1]).astype("float64")
        e = bid_ge * qb[1:] - bid_le * qb[:-1] - ask_le * qa[1:] + ask_ge * qa[:-1]
        present = (book["bid_px_1"] != NO_BID_PRICE) & (book["ask_px_1"] != NO_ASK_PRICE)
        present = present.to_numpy()
        valid = present[1:] & present[:-1]
        e[~valid] = np.nan
        ofi[1:] = e

    # Signed executed size (+1 buyer-initiated / -1 seller-initiated) * shares.
    # LOBSTER direction is the RESTING side, so trade-initiator sign = -direction.
    etype = events["event_type"].to_numpy()
    is_exec = np.isin(etype, EXEC_TYPES)
    signed_size = np.zeros(len(events), dtype="float64")
    signed_size[is_exec] = (
        -events["direction"].to_numpy()[is_exec] * events["size"].to_numpy()[is_exec]
    )

    # Tick-rule sign at executions (inference; NaN elsewhere).
    tick = np.full(len(events), np.nan, dtype="float64")
    exec_prices = events.loc[is_exec, "price"]
    tick[is_exec] = tick_rule_sign(exec_prices).to_numpy()

    denom = qb + qa
    qi = np.where(denom != 0, (qb - qa) / denom, np.nan)

    return SymbolData(
        ticker=ticker,
        n_events=len(events),
        mid=mid,
        ofi_event=ofi,
        signed_size=signed_size,
        tick_sign=tick,
        queue_imbalance=qi,
    )


# --------------------------------------------------------------------------- #
# Bucketing helper (non-overlapping event buckets)
# --------------------------------------------------------------------------- #


def _bucket_sums(x: np.ndarray, bucket: int) -> np.ndarray:
    """Sum ``x`` over non-overlapping buckets of ``bucket`` events (NaNs treated
    as 0 within a bucket — an undefined per-event OFI contributes nothing)."""
    n = (len(x) // bucket) * bucket
    if n == 0:
        return np.array([], dtype="float64")
    xf = np.nan_to_num(x[:n], nan=0.0)
    return xf.reshape(-1, bucket).sum(axis=1)


def _bucket_mid_change(mid: np.ndarray, bucket: int) -> np.ndarray:
    """Log mid change across each non-overlapping bucket: log(mid at bucket end)
    - log(mid at previous bucket end). NaN where a bucket boundary mid is invalid."""
    n = (len(mid) // bucket) * bucket
    if n < 2 * bucket:
        return np.array([], dtype="float64")
    ends = mid[bucket - 1 : n : bucket]  # mid at the close of each bucket
    log_ends = np.log(np.where(ends > 0, ends, np.nan))
    return np.diff(log_ends)  # change over each bucket vs the previous close


# --------------------------------------------------------------------------- #
# The five registered studies (each pools the five symbols)
# --------------------------------------------------------------------------- #


def study_ofi_contemporaneous(symbols: list[SymbolData], bucket: int = 50):
    """OFI vs contemporaneous mid change over the same bucket. Reports the slope
    (HAC) and the R^2 with a block-bootstrap CI. Pools symbols (stacked buckets)."""
    xs, ys = [], []
    for s in symbols:
        x = _bucket_sums(s.ofi_event, bucket)
        dp = _bucket_mid_change(s.mid, bucket)
        k = min(len(x), len(dp))
        # contemporaneous: align bucket OFI with the SAME bucket's mid change
        # (_bucket_mid_change[j] is the change across bucket j+1; drop the first OFI)
        xs.append(x[1 : k + 1] if k > 0 else np.array([]))
        ys.append(dp[:k])
    X = np.concatenate([a for a in xs if a.size]) if any(a.size for a in xs) else np.array([])
    Y = np.concatenate([a for a in ys if a.size]) if any(a.size for a in ys) else np.array([])
    m = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[m], Y[m]
    # The FDR-entering test for this study is the HAC-robust OFI->price slope: it
    # has a valid analytic p-value and CI. R^2 is a bounded, one-sided-under-null
    # statistic, so we report it descriptively (with a block-bootstrap CI) rather
    # than fabricate a two-sided normal p-value for it.
    slope = E.ols_hac(X.reshape(-1, 1), Y, name="ofi_contemporaneous", lags=10)
    r = float(np.corrcoef(X, Y)[0, 1]) if X.size > 2 else float("nan")
    r2_point = r * r

    def _r2(idx_series: np.ndarray) -> float:
        # statistic over an index reshuffle of paired rows (block bootstrap on rows)
        xi = X[idx_series.astype(int)]
        yi = Y[idx_series.astype(int)]
        rr = np.corrcoef(xi, yi)[0, 1] if xi.size > 2 else np.nan
        return rr * rr

    r2 = E.block_bootstrap_ci(
        np.arange(X.size, dtype="float64"),
        _r2,
        name="ofi_contemporaneous_r2",
        metric="R2",
        block=R.BOOTSTRAP_BLOCK_EVENTS,
        n_resamples=R.BOOTSTRAP_N_RESAMPLES,
    )
    r2 = dataclasses.replace(r2, point=r2_point)
    slope = dataclasses.replace(
        slope,
        extra={
            **slope.extra,
            "n_obs": int(X.size),
            "r2": r2_point,
            "r2_ci": (r2.ci_low, r2.ci_high),
        },
    )
    return slope, r2


def study_ofi_forward(symbols: list[SymbolData]):
    """OFI vs FORWARD mid change by horizon, out-of-sample under purged walk-forward
    CV. One EstimationResult per horizon (bucket size = horizon).

    The reported effect is the DIRECT out-of-sample OFI -> forward-return
    coefficient: for each symbol independently we run a strictly-causal purged,
    embargoed walk-forward split (so the split respects that symbol's own timeline
    and never crosses symbol boundaries), keep only the held-out (out-of-sample)
    (OFI, forward-return) pairs, pool those across symbols, and fit a single
    HAC-robust OLS of forward return on OFI. Because we regress the realized return
    on the observed OFI (not on a train-fitted prediction), the coefficient is the
    interpretable effect size the study hypothesis claims, and its HAC standard
    error is well-defined (no generated-regressor problem). Every fold is checked by
    the leakage detector before its data is used.
    """
    from src.cv.leakage import audit_fold
    from src.cv.splitter import PurgedWalkForwardSplitter

    splitter = PurgedWalkForwardSplitter(mode="walkforward")
    results = []
    for h in R.HORIZON_GRID_EVENTS:
        # embargo/purge are applied in BUCKET-row units; convert the event-based
        # registry embargo to buckets (>=1) so it means the same span at every h.
        embargo_buckets = max(1, -(-R.CV_EMBARGO_EVENTS // h))  # ceil division
        oos_ofi, oos_ret = [], []
        for s in symbols:
            ofi_b = _bucket_sums(s.ofi_event, h)
            dp = _bucket_mid_change(s.mid, h)  # dp[j] = change across bucket j+1
            k = min(len(ofi_b) - 1, len(dp))
            if k <= 0:
                continue
            # forward: bucket-j OFI predicts the NEXT bucket's mid change (dp[j])
            x = ofi_b[:k]
            y = dp[:k]
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            for tr, te in splitter.split(len(x), label_horizon=1, embargo=embargo_buckets):
                # adversarial check: the fold geometry must be leakage-free
                assert not audit_fold(tr, te, label_horizon=1, embargo=embargo_buckets), (
                    f"leakage detected in {s.ticker} h={h}"
                )
                if len(tr) < 10 or len(te) < 2:
                    continue
                oos_ofi.append(x[te])  # held-out FUTURE OFI of this symbol
                oos_ret.append(y[te])  # its realized forward return
        if not oos_ofi:
            continue
        X = np.concatenate(oos_ofi)
        Y = np.concatenate(oos_ret)
        res = E.ols_hac(X.reshape(-1, 1), Y, name=f"ofi_forward_h{h}", lags=min(2 * h, 50))
        # An out-of-sample rank information coefficient, reported alongside (not a
        # separate family test — the family stays 17).
        ic = E.information_coefficient(X, Y, name=f"ofi_forward_ic_h{h}", method="spearman")
        res = dataclasses.replace(
            res,
            extra={
                **res.extra,
                "horizon": h,
                "n_obs": int(Y.size),
                "oos": True,
                "oos_ic": ic.point,
                "oos_ic_ci": (ic.ci_low, ic.ci_high),
            },
        )
        results.append(res)  # ols_hac already carries a valid HAC p-value
    return results


def study_queue_imbalance(symbols: list[SymbolData], bucket: int = 20):
    """Queue imbalance predicts the sign of the NEXT mid move: block-bootstrap AUC
    with CI and a genuine bootstrap p-value against 0.5. Point-in-time (imbalance
    at bucket close -> next move)."""
    scores, labels = [], []
    for s in symbols:
        # sample queue imbalance at bucket closes; label = sign of next bucket move
        dp = _bucket_mid_change(s.mid, bucket)  # dp[j] across bucket j+1
        n = (len(s.mid) // bucket) * bucket
        qi_close = s.queue_imbalance[bucket - 1 : n : bucket]  # imbalance at each close
        k = min(len(qi_close) - 1, len(dp))
        if k <= 0:
            continue
        qi_j = qi_close[:k]
        move = dp[:k]
        valid = np.isfinite(qi_j) & np.isfinite(move) & (move != 0.0)
        scores.append(qi_j[valid])
        labels.append((move[valid] > 0).astype(int))
    S = np.concatenate(scores)
    L = np.concatenate(labels)
    # Block bootstrap: consecutive bucket observations are serially correlated, so
    # an i.i.d. paired bootstrap would under-cover. The block AUC also carries a
    # genuine two-sided bootstrap p-value against the no-skill null (AUC = 0.5).
    auc = E.auc_ci(
        S,
        L,
        name="queue_imbalance_next_move",
        n_resamples=R.BOOTSTRAP_N_RESAMPLES,
        block=R.BOOTSTRAP_BLOCK_EVENTS,
        method="stationary",
    )
    return dataclasses.replace(auc, extra={**auc.extra, "n_obs": int(S.size)})


def study_sign_autocorrelation(symbols: list[SymbolData]):
    """Trade-sign autocorrelation (long memory) at each horizon-grid lag, with a
    Bartlett large-sample CI. One EstimationResult per lag.

    We use the Bartlett (Box-Jenkins) interval, NOT a block bootstrap: a block
    bootstrap resamples blocks and thereby ATTENUATES the very autocorrelation
    being estimated, producing an interval that does not even contain the point
    estimate at intermediate lags (see docs/DESIGN.md §D-stats.5). The Bartlett
    interval is the standard, correctly-centred CI for the sample ACF.

    The autocorrelation is computed PER SYMBOL and combined by a trade-count-
    weighted average (weights n_i / sum n), so no cross-symbol boundary pairs ever
    enter the estimate — concatenating the series would contaminate the lag-k
    products at the symbol seams. The combined SE treats symbols as independent:
    se = sqrt(sum w_i^2 se_i^2) with each se_i the Bartlett SE. Note the sign is
    the TICK-RULE inference, not LOBSTER's true resting side; any misclassification
    attenuates the measured memory, so this is a conservative (downward-biased)
    estimate — stated deliberately."""
    z = 1.959963984540054  # two-sided 95% normal critical value
    sign_series = [
        s.tick_sign[np.isfinite(s.tick_sign)]
        for s in symbols
        if np.isfinite(s.tick_sign).sum() > max(R.HORIZON_GRID_EVENTS) + 10
    ]
    counts = np.array([sg.size for sg in sign_series], dtype="float64")
    weights = counts / counts.sum()
    total_n = int(counts.sum())

    results = []
    for lag in R.HORIZON_GRID_EVENTS:
        per_symbol = [E.autocorr_bartlett_ci(sg, lag, name="acf") for sg in sign_series]
        points = np.array([r.point for r in per_symbol])
        ses = np.array([r.std_error for r in per_symbol])
        point = float(np.sum(weights * points))
        se = float(np.sqrt(np.sum((weights**2) * (ses**2))))
        p = float(2.0 * sp_stats.norm.sf(abs(point) / se)) if se > 0 else float("nan")
        res = EstimationResult(
            name=f"sign_autocorr_lag{lag}",
            metric="autocorr",
            point=point,
            ci_low=point - z * se,
            ci_high=point + z * se,
            n_obs=total_n,
            std_error=se,
            p_value_raw=p,
            ci_level=R.CI_LEVEL,
            ci_method="bartlett_per_symbol_weighted",
            extra={"lag": lag, "n_obs": total_n, "n_symbols": len(sign_series)},
        )
        results.append(res)
    return results


def study_impact_curve(symbols: list[SymbolData], bucket: int = 20, n_bins: int = 20):
    """Concave price impact: contemporaneous |mid change| as a function of |signed
    traded volume| in a bucket, fit as a log-log exponent gamma (< 1 => concave,
    sub-linear impact — the square-root-impact stylized fact). No tradability claim.

    To avoid selection on the dependent variable (dropping zero-move buckets would
    truncate |dp| from below and bias the exponent upward — a problem precisely in
    the large-tick names with many zero-move buckets), we do NOT filter on |dp|>0.
    Instead we bin buckets by |signed volume| into quantile bins and regress the
    log of the MEAN |dp| in each bin on the log of the mean |V|. Averaging over a
    bin makes mean|dp| strictly positive even where individual buckets did not move,
    so every non-zero-volume bucket contributes. The exponent is the HAC slope over
    the bin points (few lags — the regression is on ~n_bins points)."""
    av_all, adp_all = [], []
    for s in symbols:
        v = _bucket_sums(s.signed_size, bucket)  # signed volume per bucket
        dp = _bucket_mid_change(s.mid, bucket)
        k = min(len(v), len(dp))
        if k <= 0:
            continue
        vv = v[1 : k + 1]  # volume in the bucket whose mid change is dp[:k]
        pp = dp[:k]
        m = np.isfinite(vv) & np.isfinite(pp) & (np.abs(vv) > 0)  # only V != 0
        av_all.append(np.abs(vv[m]))
        adp_all.append(np.abs(pp[m]))
    AV = np.concatenate(av_all)
    ADP = np.concatenate(adp_all)

    edges = np.quantile(AV, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    b = np.clip(np.digitize(AV, edges[1:-1]), 0, len(edges) - 2)
    xb, yb = [], []
    for i in range(b.max() + 1):
        mask = b == i
        if mask.sum() < 5:
            continue
        mean_v = AV[mask].mean()
        mean_dp = ADP[mask].mean()  # zeros retained -> no selection on the DV
        if mean_v > 0 and mean_dp > 0:
            xb.append(mean_v)
            yb.append(mean_dp)
    logv = np.log(np.asarray(xb))
    logdp = np.log(np.asarray(yb))
    res = E.ols_hac(logv.reshape(-1, 1), logdp, name="impact_curve", lags=2)
    return dataclasses.replace(
        res,
        metric="coef",
        extra={
            **res.extra,
            "interpretation": "log-log impact exponent (gamma), binned to avoid DV selection",
            "n_obs": int(AV.size),
            "n_bins": int(len(xb)),
        },
    )


# --------------------------------------------------------------------------- #
# Orchestration: run the whole registered family and close it under FDR
# --------------------------------------------------------------------------- #


def run_family(tickers: tuple[str, ...] = R.DATA.tickers) -> pd.DataFrame:
    """Run every registered study on the given tickers, apply Benjamini-Hochberg
    across the whole family, and return a tidy results table (one row per test)."""
    from src.multi.fdr import BenjaminiHochbergController

    symbols = [load_symbol(t) for t in tickers]

    results = []
    slope, _r2 = study_ofi_contemporaneous(symbols)
    results.append(slope)  # the HAC slope is the FDR-entering test for this study
    results.extend(study_ofi_forward(symbols))
    results.append(study_queue_imbalance(symbols))
    results.extend(study_sign_autocorrelation(symbols))
    results.append(study_impact_curve(symbols))

    controller = BenjaminiHochbergController()
    adjusted = controller.control(results, family_id="microlab_study_family", alpha=R.FDR_ALPHA)

    rows = []
    for res in adjusted:
        rows.append(
            {
                "name": res.name,
                "metric": res.metric,
                "point": res.point,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "p_raw": res.p_value_raw,
                "p_adj": res.p_value_adj,
                "rejected": res.rejected,
                "n_obs": res.extra.get("n_obs"),
                "horizon": res.extra.get("horizon", res.extra.get("lag")),
            }
        )
    return pd.DataFrame(rows)


def run_robustness_by_symbol(tickers: tuple[str, ...] = R.DATA.tickers) -> pd.DataFrame:
    """Per-symbol robustness split: re-run the two headline single-shot effects
    (OFI contemporaneous R^2, queue-imbalance AUC) on each ticker separately, so
    the note can show the effect is not driven by one name. No FDR here — this is
    a robustness view of already-registered effects, not new hypotheses."""
    rows = []
    for t in tickers:
        syms = [load_symbol(t)]
        slope, r2 = study_ofi_contemporaneous(syms)
        auc = study_queue_imbalance(syms)
        rows.append(
            {
                "ticker": t,
                "ofi_R2": r2.point,
                "ofi_R2_lo": r2.ci_low,
                "ofi_R2_hi": r2.ci_high,
                "ofi_slope": slope.point,
                "qi_auc": auc.point,
                "qi_auc_lo": auc.ci_low,
                "qi_auc_hi": auc.ci_high,
            }
        )
    return pd.DataFrame(rows)


# Committed derived artifacts (CI runs on these; raw data is never committed).
RESULTS_DIR = "data/fixtures"
FAMILY_CSV = f"{RESULTS_DIR}/study_family_results.csv"
ROBUSTNESS_CSV = f"{RESULTS_DIR}/robustness_by_symbol.csv"


def main() -> None:
    import os

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = run_family()
    df.to_csv(FAMILY_CSV, index=False)
    print(df.to_string(index=False))
    n_rej = int(df["rejected"].fillna(False).sum())
    print(
        f"\nFamily size m = {len(df)} (registry says {R.family_size()}); "
        f"FDR-significant at alpha={R.FDR_ALPHA}: {n_rej}"
    )

    rob = run_robustness_by_symbol()
    rob.to_csv(ROBUSTNESS_CSV, index=False)
    print("\nPer-symbol robustness (OFI R^2, queue-imbalance AUC):")
    print(rob.to_string(index=False))
    print(f"\nSaved: {FAMILY_CSV} and {ROBUSTNESS_CSV}")


if __name__ == "__main__":
    main()
