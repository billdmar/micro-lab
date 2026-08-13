"""Point-in-time feature/label library.

Implements :class:`PointInTimeFeatureEngine` per ``interfaces.FeatureEngine``.
Every produced column carries a ``schema.FeatureSpec`` whose point-in-time
metadata is the contract's teeth: backward-looking *features* are known at the
event (``is_label=False``, ``horizon=0``, ``info_offset=0``); forward *labels*
of horizon h become knowable only h events later (``is_label=True``,
``horizon=h``, ``info_offset=h``). A feature at row i depends only on rows <= i;
a label at row i looks forward exactly its horizon and is NaN in the final h
rows. Genuine unknowns (row 0 OFI, empty-touch spread, the leading realized-vol
window, past-the-end labels) are NaN — never a fabricated number.

Units follow ``schema``: prices are integer 1/10000-dollar, sizes are integer
shares, ``direction`` is the resting side (+1 buy limit / -1 sell limit). See
docs/DESIGN.md for the rationale behind each methodological choice.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from config import registry as R

from ..schema import (
    NO_ASK_PRICE,
    NO_BID_PRICE,
    EventType,
    FeatureSpec,
    mid_price_series,
)

# Columns of the reconstructed book frame this engine reads at the touch.
_BEST = ("ask_px_1", "ask_sz_1", "bid_px_1", "bid_sz_1")
_SZ_COL = re.compile(r"^(ask|bid)_sz_\d+$")  # any-depth size column (for total depth)

#: Execution event codes carry a trade; the tick rule classifies these only.
_EXEC_TYPES = (int(EventType.EXECUTE_VISIBLE), int(EventType.EXECUTE_HIDDEN))


# --------------------------------------------------------------------------- #
# Trade-sign inference (tick rule) — shared by the feature and its validator
# --------------------------------------------------------------------------- #


def tick_rule_sign(prices: pd.Series) -> pd.Series:
    """Classic tick-rule trade-sign inference over an ordered trade-price series.

    A trade priced above the previous trade is buyer-initiated (+1), below is
    seller-initiated (-1); a zero tick carries forward the last non-zero sign
    (Lee & Ready 1991). The first trade — and any leading run of zero ticks
    before the first price move — has no reference and is NaN (honest unknown),
    NOT guessed. Returns a float series index-aligned to ``prices``.
    """
    step = np.sign(prices.astype("float64").diff())  # +1 / 0 / -1, NaN at the first
    step = step.replace(0.0, np.nan)  # zero ticks defer to the carried sign
    return step.ffill()


def tick_rule_accuracy(events: pd.DataFrame) -> float:
    """Measure tick-rule sign accuracy against LOBSTER ground truth (validation).

    LOBSTER ``direction`` is the RESTING order's side, so at an execution the
    aggressor (trade initiator) is the opposite side: true sign = -direction.
    Compares the inferred tick-rule sign to that truth over execution rows where
    the rule produced a sign. Returns NaN if there is no classified trade.
    """
    is_exec = events["event_type"].isin(_EXEC_TYPES)
    inferred = tick_rule_sign(events.loc[is_exec, "price"])
    truth = -events.loc[is_exec, "direction"].astype("float64")
    graded = inferred.notna()
    n = int(graded.sum())
    if n == 0:
        return float("nan")
    return float((inferred[graded] == truth[graded]).sum()) / n


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


class PointInTimeFeatureEngine:
    """Produces the backward-looking feature columns and the forward mid-return
    labels in one index-aligned frame. ``rv_window`` is the backward window (in
    events) for short-horizon realized volatility; ``label_horizons`` defaults to
    the pre-registered event horizon grid.
    """

    def __init__(
        self,
        rv_window: int = 20,
        label_horizons: Sequence[int] = R.HORIZON_GRID_EVENTS,
    ) -> None:
        if rv_window < 1:
            raise ValueError(f"rv_window must be >= 1, got {rv_window}")
        self.rv_window = rv_window
        self.label_horizons = tuple(label_horizons)
        self._rv_name = f"rv_{rv_window}"

    # -- contract ----------------------------------------------------------- #

    def specs(self) -> Sequence[FeatureSpec]:
        """Every column this engine emits, with its point-in-time metadata."""
        features = [
            FeatureSpec("ofi", "Cont-Kukanov-Stoikov order-flow imbalance from best-quote changes"),
            FeatureSpec(
                "queue_imbalance", "(bid_sz_1 - ask_sz_1) / (bid_sz_1 + ask_sz_1) at the touch"
            ),
            FeatureSpec(
                "tick_sign", "tick-rule inferred trade sign at execution events (NaN otherwise)"
            ),
            FeatureSpec("depth", "total displayed size across all book levels (both sides)"),
            FeatureSpec("spread", "best_ask - best_bid in 1/10000 dollar"),
            FeatureSpec(
                self._rv_name,
                f"realized vol of mid log-returns over the trailing {self.rv_window} events",
            ),
        ]
        labels = [
            FeatureSpec(
                f"fwd_ret_{h}",
                f"forward mid log-return over {h} events",
                is_label=True,
                horizon=float(h),
                horizon_unit="events",
                info_offset=float(h),
                causal=False,
            )
            for h in self.label_horizons
        ]
        return features + labels

    def compute(self, events: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with one column per spec, aligned to ``events.index``."""
        missing = [c for c in _BEST if c not in book.columns]
        if missing:
            raise ValueError(f"book frame missing touch columns: {missing}")
        idx = events.index

        mid = mid_price_series(book).to_numpy()  # 1/10000 dollar, NaN on empty touch

        out = pd.DataFrame(index=idx)
        out["ofi"] = self._ofi(book)
        out["queue_imbalance"] = self._queue_imbalance(book)
        out["tick_sign"] = self._tick_sign(events)
        out["depth"] = self._depth(book)
        out["spread"] = self._spread(book)
        out[self._rv_name] = self._realized_vol(mid, idx)
        for h in self.label_horizons:
            out[f"fwd_ret_{h}"] = self._forward_return(mid, h, idx)
        return out

    # -- individual columns ------------------------------------------------- #

    def _ofi(self, book: pd.DataFrame) -> pd.Series:
        """Per-event OFI contribution e_n from best-quote price/size changes.

        e_n = 1[Pb_n>=Pb_{n-1}]qb_n - 1[Pb_n<=Pb_{n-1}]qb_{n-1}
            - 1[Pa_n<=Pa_{n-1}]qa_n + 1[Pa_n>=Pa_{n-1}]qa_{n-1}
        (bid side adds when the bid improves/holds, subtracts when it worsens;
        ask symmetric). Row 0 has no predecessor -> NaN; rows with an empty touch
        (either side) at n or n-1 are NaN because the change is ill-defined.
        """
        pb = book["bid_px_1"].astype("float64")
        qb = book["bid_sz_1"].astype("float64")
        pa = book["ask_px_1"].astype("float64")
        qa = book["ask_sz_1"].astype("float64")
        pb_p, qb_p = pb.shift(1), qb.shift(1)
        pa_p, qa_p = pa.shift(1), qa.shift(1)

        # Indicators as floats: a boolean Series must not be negated with unary
        # minus (that is logical NOT, not arithmetic sign).
        bid_ge = (pb >= pb_p).astype("float64")
        bid_le = (pb <= pb_p).astype("float64")
        ask_le = (pa <= pa_p).astype("float64")
        ask_ge = (pa >= pa_p).astype("float64")
        bid_term = bid_ge * qb - bid_le * qb_p
        ask_term = -ask_le * qa + ask_ge * qa_p
        e = bid_term + ask_term

        present = (book["bid_px_1"] != NO_BID_PRICE) & (book["ask_px_1"] != NO_ASK_PRICE)
        valid = present & present.shift(1, fill_value=False)  # both touches at n and n-1
        return e.where(valid)

    @staticmethod
    def _queue_imbalance(book: pd.DataFrame) -> pd.Series:
        bid = book["bid_sz_1"].astype("float64")
        ask = book["ask_sz_1"].astype("float64")
        denom = bid + ask
        # 0/0 (empty touch) yields NaN, which is the honest "undefined" here.
        return (bid - ask) / denom.where(denom != 0)

    @staticmethod
    def _tick_sign(events: pd.DataFrame) -> pd.Series:
        """Tick-rule trade sign, defined at executions and NaN at other events."""
        is_exec = events["event_type"].isin(_EXEC_TYPES)
        sign = pd.Series(np.nan, index=events.index, dtype="float64")
        sign.loc[is_exec] = tick_rule_sign(events.loc[is_exec, "price"])
        return sign

    @staticmethod
    def _depth(book: pd.DataFrame) -> pd.Series:
        sz_cols = [c for c in book.columns if _SZ_COL.match(c)]
        # Empty LOBSTER levels carry size 0, so a plain sum is total displayed size.
        return book[sz_cols].astype("float64").sum(axis=1)

    @staticmethod
    def _spread(book: pd.DataFrame) -> pd.Series:
        ask = book["ask_px_1"].astype("float64")
        bid = book["bid_px_1"].astype("float64")
        present = (book["bid_px_1"] != NO_BID_PRICE) & (book["ask_px_1"] != NO_ASK_PRICE)
        return (ask - bid).where(present)

    def _realized_vol(self, mid: np.ndarray, idx: pd.Index) -> pd.Series:
        """sqrt of the trailing sum of squared mid log-returns over rv_window
        events. NaN until a full window of returns is available (backward only).
        """
        log_mid = pd.Series(np.log(mid), index=idx)  # NaN where mid is NaN/<=0
        r2 = log_mid.diff() ** 2
        return np.sqrt(r2.rolling(self.rv_window, min_periods=self.rv_window).sum())

    @staticmethod
    def _forward_return(mid: np.ndarray, h: int, idx: pd.Index) -> pd.Series:
        """Forward mid log-return over h events: log(mid_{i+h}) - log(mid_i).
        The final h rows have no forward price and are NaN (label look-forward).
        """
        log_mid = np.log(mid)
        fwd = np.full(mid.shape, np.nan, dtype="float64")
        if h < mid.shape[0]:
            fwd[: mid.shape[0] - h] = log_mid[h:] - log_mid[: mid.shape[0] - h]
        return pd.Series(fwd, index=idx)
