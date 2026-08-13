"""Definitional + point-in-time tests for the feature engine.

All inputs are tiny hand-built books whose right answers are computed by hand in
the assertions (or a comment). No raw LOBSTER data is touched. The point-in-time
tests are the teeth: a backward feature at row i is unchanged when the frame is
truncated after row i (no look-ahead), and a forward label equals the realized
forward value and is NaN in the final h rows.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from config import registry as R
from src.features import (
    PointInTimeFeatureEngine,
    tick_rule_accuracy,
    tick_rule_sign,
)
from src.schema import NO_ASK_PRICE, NO_BID_PRICE, EventType, book_columns

# --------------------------------------------------------------------------- #
# Fixture builders (shared, tiny, hand-checkable)
# --------------------------------------------------------------------------- #


def _book(rows: list[list[int]], levels: int = 1) -> pd.DataFrame:
    """Book frame from interleaved LOBSTER rows (ask_px,ask_sz,bid_px,bid_sz,...)."""
    return pd.DataFrame(rows, columns=book_columns(levels)).astype("int64")


def _events(event_types: list[int], prices: list[int], directions: list[int]) -> pd.DataFrame:
    """Minimal event frame carrying only the columns the engine reads."""
    n = len(prices)
    return pd.DataFrame(
        {
            "time_s": np.arange(n, dtype="float64"),
            "event_type": np.asarray(event_types, dtype="int8"),
            "order_id": np.arange(1, n + 1, dtype="int64"),
            "size": np.ones(n, dtype="int64"),
            "price": np.asarray(prices, dtype="int64"),
            "direction": np.asarray(directions, dtype="int8"),
        }
    )


#: A three-row level-1 book used across the definitional tests.
#   row0: ask 110x20, bid 100x10   mid 105
#   row1: ask 110x20, bid 101x15   mid 105.5  (bid improves)
#   row2: ask 109x8,  bid 100x5    mid 104.5  (bid worsens, ask improves)
_BOOK3 = _book(
    [
        [110, 20, 100, 10],
        [110, 20, 101, 15],
        [109, 8, 100, 5],
    ]
)
_EVENTS3 = _events(
    event_types=[int(EventType.EXECUTE_VISIBLE)] * 3,
    prices=[105, 106, 104],
    directions=[1, 1, 1],
)


# --------------------------------------------------------------------------- #
# OFI
# --------------------------------------------------------------------------- #


def test_ofi_matches_hand_computation():
    out = PointInTimeFeatureEngine(rv_window=2, label_horizons=(1,)).compute(_EVENTS3, _BOOK3)
    ofi = out["ofi"]
    assert math.isnan(ofi.iloc[0])  # no predecessor
    # row1: bid improves 100->101 (+15); ask holds (- q_a + q_a_prev = -20+20 = 0)
    assert ofi.iloc[1] == 15.0
    # row2: bid worsens 101->100 (-15); ask improves 110->109 (-8, +0)
    assert ofi.iloc[2] == -23.0


def test_ofi_is_nan_when_a_touch_is_empty():
    book = _book(
        [
            [110, 20, 100, 10],
            [NO_ASK_PRICE, 0, 100, 10],  # ask side empties -> change ill-defined
            [110, 20, 100, 10],
        ]
    )
    ev = _events([int(EventType.SUBMIT)] * 3, [100, 100, 100], [1, 1, 1])
    ofi = PointInTimeFeatureEngine(label_horizons=(1,)).compute(ev, book)["ofi"]
    assert math.isnan(ofi.iloc[1])  # n has empty touch
    assert math.isnan(ofi.iloc[2])  # n-1 had empty touch


# --------------------------------------------------------------------------- #
# Queue imbalance / spread / depth
# --------------------------------------------------------------------------- #


def test_queue_imbalance_matches_hand_computation():
    qi = PointInTimeFeatureEngine(label_horizons=(1,)).compute(_EVENTS3, _BOOK3)["queue_imbalance"]
    assert qi.iloc[0] == pytest.approx((10 - 20) / 30)
    assert qi.iloc[1] == pytest.approx((15 - 20) / 35)
    assert qi.iloc[2] == pytest.approx((5 - 8) / 13)


def test_queue_imbalance_nan_on_empty_touch():
    book = _book([[NO_ASK_PRICE, 0, NO_BID_PRICE, 0]])
    ev = _events([int(EventType.SUBMIT)], [100], [1])
    qi = PointInTimeFeatureEngine(label_horizons=(1,)).compute(ev, book)["queue_imbalance"]
    assert math.isnan(qi.iloc[0])


def test_spread_matches_hand_computation():
    sp = PointInTimeFeatureEngine(label_horizons=(1,)).compute(_EVENTS3, _BOOK3)["spread"]
    assert list(sp) == [10.0, 9.0, 9.0]


def test_depth_sums_all_levels():
    book = _book(
        [
            [110, 20, 100, 10, 111, 5, 99, 7],  # L2 sizes add: 20+10+5+7 = 42
        ],
        levels=2,
    )
    ev = _events([int(EventType.SUBMIT)], [100], [1])
    depth = PointInTimeFeatureEngine(label_horizons=(1,)).compute(ev, book)["depth"]
    assert depth.iloc[0] == 42.0


# --------------------------------------------------------------------------- #
# Realized volatility (backward window)
# --------------------------------------------------------------------------- #


def test_realized_vol_backward_window():
    rv = PointInTimeFeatureEngine(rv_window=2, label_horizons=(1,)).compute(_EVENTS3, _BOOK3)[
        "rv_2"
    ]
    r1 = math.log(105.5) - math.log(105.0)
    r2 = math.log(104.5) - math.log(105.5)
    assert math.isnan(rv.iloc[0])  # no return yet
    assert math.isnan(rv.iloc[1])  # only one return, window needs two
    assert rv.iloc[2] == pytest.approx(math.sqrt(r1**2 + r2**2))


# --------------------------------------------------------------------------- #
# Tick-rule sign inference + validation helper
# --------------------------------------------------------------------------- #


def test_tick_rule_sign_carries_forward_and_leaves_leading_nan():
    prices = pd.Series([100, 101, 101, 100, 99])
    sign = tick_rule_sign(prices)
    assert math.isnan(sign.iloc[0])  # first trade: no reference
    assert list(sign.iloc[1:]) == [1.0, 1.0, -1.0, -1.0]  # zero tick carries the +1


def test_tick_sign_feature_defined_only_at_executions():
    ev = _events(
        event_types=[
            int(EventType.EXECUTE_VISIBLE),
            int(EventType.SUBMIT),  # not a trade
            int(EventType.EXECUTE_VISIBLE),
        ],
        prices=[100, 100, 101],
        directions=[1, 1, 1],
    )
    book = _book([[110, 5, 100, 5]] * 3)
    ts = PointInTimeFeatureEngine(label_horizons=(1,)).compute(ev, book)["tick_sign"]
    assert math.isnan(ts.iloc[0])  # first execution: no reference
    assert math.isnan(ts.iloc[1])  # non-execution row is never signed
    # tick rule runs over executions only: 100 -> 101 is an up-tick (+1)
    assert ts.iloc[2] == 1.0


def test_tick_rule_accuracy_against_ground_truth():
    # true aggressor sign = -direction. Rows 1..4 are graded (row0 unsigned).
    # inferred: [nan, +1, +1, -1, -1]; truths via dirs [1,-1,-1,1,-1] -> [-1,+1,+1,-1,+1]
    # matches on rows 1,2,3, mismatch on row4 -> 3/4.
    ev = _events(
        event_types=[int(EventType.EXECUTE_VISIBLE)] * 5,
        prices=[100, 101, 101, 100, 99],
        directions=[1, -1, -1, 1, -1],
    )
    assert tick_rule_accuracy(ev) == pytest.approx(0.75)


def test_tick_rule_accuracy_nan_without_trades():
    ev = _events([int(EventType.SUBMIT)] * 2, [100, 101], [1, 1])
    assert math.isnan(tick_rule_accuracy(ev))


# --------------------------------------------------------------------------- #
# Forward-return labels
# --------------------------------------------------------------------------- #


def test_forward_return_label_known_by_hand_and_nan_tail():
    out = PointInTimeFeatureEngine(rv_window=2, label_horizons=(1, 2)).compute(_EVENTS3, _BOOK3)
    mids = [105.0, 105.5, 104.5]
    f1 = out["fwd_ret_1"]
    assert f1.iloc[0] == pytest.approx(math.log(mids[1]) - math.log(mids[0]))
    assert f1.iloc[1] == pytest.approx(math.log(mids[2]) - math.log(mids[1]))
    assert math.isnan(f1.iloc[2])  # final h=1 row has no forward price
    f2 = out["fwd_ret_2"]
    assert f2.iloc[0] == pytest.approx(math.log(mids[2]) - math.log(mids[0]))
    assert math.isnan(f2.iloc[1])  # final h=2 rows NaN
    assert math.isnan(f2.iloc[2])


def test_forward_return_all_nan_when_horizon_exceeds_frame():
    out = PointInTimeFeatureEngine(rv_window=2, label_horizons=(5,)).compute(_EVENTS3, _BOOK3)
    assert out["fwd_ret_5"].isna().all()  # only 3 rows, no 5-ahead price exists


# --------------------------------------------------------------------------- #
# Specs / contract
# --------------------------------------------------------------------------- #


def test_specs_metadata_is_point_in_time_correct():
    eng = PointInTimeFeatureEngine(rv_window=20, label_horizons=(1, 5, 10))
    specs = {s.name: s for s in eng.specs()}
    for name in ("ofi", "queue_imbalance", "tick_sign", "depth", "spread", "rv_20"):
        s = specs[name]
        assert not s.is_label and s.horizon == 0.0 and s.info_offset == 0.0 and s.causal
    for h in (1, 5, 10):
        s = specs[f"fwd_ret_{h}"]
        assert s.is_label and s.horizon == float(h) and s.info_offset == float(h)
        assert not s.causal and s.horizon_unit == "events"


def test_compute_columns_match_specs_exactly():
    eng = PointInTimeFeatureEngine(rv_window=3, label_horizons=(1, 2))
    out = eng.compute(_EVENTS3, _BOOK3)
    assert list(out.columns) == [s.name for s in eng.specs()]
    assert out.index.equals(_EVENTS3.index)


def test_default_label_horizons_are_the_registry_grid():
    specs = PointInTimeFeatureEngine().specs()
    label_horizons = tuple(int(s.horizon) for s in specs if s.is_label)
    assert label_horizons == R.HORIZON_GRID_EVENTS


def test_rv_window_must_be_positive():
    with pytest.raises(ValueError):
        PointInTimeFeatureEngine(rv_window=0)


def test_compute_raises_on_missing_touch_columns():
    eng = PointInTimeFeatureEngine(label_horizons=(1,))
    bad_book = pd.DataFrame({"ask_px_1": [110], "bid_px_1": [100]})  # missing sizes
    ev = _events([int(EventType.SUBMIT)], [100], [1])
    with pytest.raises(ValueError, match="missing touch columns"):
        eng.compute(ev, bad_book)


# --------------------------------------------------------------------------- #
# Point-in-time: truncation invariance (the anti-look-ahead teeth)
# --------------------------------------------------------------------------- #

_FEATURE_COLS = ("ofi", "queue_imbalance", "tick_sign", "depth", "spread", "rv_2")


@pytest.mark.parametrize("col", _FEATURE_COLS)
def test_feature_at_row_i_unchanged_by_truncation(col):
    eng = PointInTimeFeatureEngine(rv_window=2, label_horizons=(1,))
    full = eng.compute(_EVENTS3, _BOOK3)[col]
    for i in range(len(_EVENTS3)):
        truncated = eng.compute(_EVENTS3.iloc[: i + 1], _BOOK3.iloc[: i + 1])[col]
        a, b = full.iloc[i], truncated.iloc[i]
        assert (math.isnan(a) and math.isnan(b)) or a == b


def test_label_becomes_nan_when_future_is_truncated_away():
    eng = PointInTimeFeatureEngine(rv_window=2, label_horizons=(1,))
    full = eng.compute(_EVENTS3, _BOOK3)["fwd_ret_1"]
    # Truncating after row0 removes row1 -> the h=1 label at row0 is now unknown.
    truncated = eng.compute(_EVENTS3.iloc[:1], _BOOK3.iloc[:1])["fwd_ret_1"]
    assert not math.isnan(full.iloc[0])  # known on the full frame
    assert math.isnan(truncated.iloc[0])  # forward value truncated away
