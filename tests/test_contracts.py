"""Contract-level tests for the frozen schema, interfaces, and registry.

These guard the seams the parallel workstreams build against. They are fast and
run on no external data.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from config import registry as R
from src import interfaces, schema


def test_event_construction_and_validation():
    e = schema.Event(34200.004, schema.EventType.SUBMIT, 16113575, 18, 5853300, 1)
    assert e.event_type is schema.EventType.SUBMIT
    assert e.price == 5853300
    with pytest.raises(ValueError):
        schema.Event(1.0, schema.EventType.SUBMIT, 1, 10, 100, 0)  # bad direction
    with pytest.raises(ValueError):
        schema.Event(1.0, schema.EventType.SUBMIT, 1, -5, 100, 1)  # negative size


def test_book_state_mid_and_flatten():
    bs = schema.BookState((5859400, 5859800), (200, 200), (5853300, 5853000), (18, 150))
    assert bs.levels == 2
    assert bs.best_ask == 5859400
    assert bs.best_bid == 5853300
    assert bs.mid_price == (5859400 + 5853300) / 2.0
    assert bs.to_row() == [5859400, 200, 5853300, 18, 5859800, 200, 5853000, 150]


def test_book_state_empty_touch_is_nan_mid():
    bs = schema.BookState((schema.NO_ASK_PRICE,), (0,), (5853300,), (18,))
    assert math.isnan(bs.mid_price)


def test_book_columns_layout():
    assert schema.book_columns(1) == ["ask_px_1", "ask_sz_1", "bid_px_1", "bid_sz_1"]
    assert len(schema.book_columns(10)) == 40


def test_feature_spec_rules():
    feat = schema.FeatureSpec("ofi", "order flow imbalance")
    assert feat.info_offset == 0.0 and not feat.is_label
    label = schema.FeatureSpec("ret10", "fwd ret", is_label=True, horizon=10, info_offset=10)
    assert label.is_label and label.horizon == 10
    with pytest.raises(ValueError):
        schema.FeatureSpec("bad", "label w/o horizon", is_label=True, horizon=0)
    with pytest.raises(ValueError):
        schema.FeatureSpec("bad", "feature w/ horizon", horizon=5)


def test_estimation_result_requires_ordered_ci():
    er = schema.EstimationResult("ofi_R2", "R2", 0.42, 0.39, 0.45, 10_000)
    assert er.ci_low <= er.point <= er.ci_high
    with pytest.raises(ValueError):
        schema.EstimationResult("bad", "coef", 0.0, 1.0, -1.0, 10)  # ci_low > ci_high


def test_events_frame_roundtrip_and_validation():
    events = [
        schema.Event(1.0, schema.EventType.SUBMIT, 1, 10, 100, 1),
        schema.Event(2.0, schema.EventType.EXECUTE_VISIBLE, 1, 5, 100, 1),
    ]
    df = schema.events_to_frame(events)
    assert list(df.columns) == list(schema.EVENT_COLUMNS)
    assert str(df["direction"].dtype) == "int8"
    schema.validate_event_frame(df)  # should not raise


def test_validate_event_frame_catches_unsorted_time():
    df = schema.events_to_frame(
        [
            schema.Event(2.0, schema.EventType.SUBMIT, 1, 10, 100, 1),
            schema.Event(1.0, schema.EventType.SUBMIT, 2, 10, 100, 1),
        ]
    )
    df = df.iloc[::-1]  # this ordering is still fine
    df2 = df.copy()
    df2.loc[:, "time_s"] = [2.0, 1.0]  # force a descending time
    with pytest.raises(ValueError):
        schema.validate_event_frame(df2)


def test_mid_price_series_vectorized():
    book = pd.DataFrame(
        {
            "ask_px_1": [5859400, schema.NO_ASK_PRICE],
            "bid_px_1": [5853300, 5853000],
        }
    )
    mid = schema.mid_price_series(book)
    assert mid.iloc[0] == (5859400 + 5853300) / 2.0
    assert math.isnan(mid.iloc[1])


def test_registry_family_is_frozen_and_counted():
    assert R.family_size() == sum(s.n_tests for s in R.STUDY_FAMILY)
    assert R.family_size() == 17  # 1 + 7 + 1 + 7 + 1 across the five studies
    assert R.FDR_ALPHA == 0.05
    assert R.CI_LEVEL == 0.95
    assert max(R.HORIZON_GRID_EVENTS) == R.CV_EMBARGO_EVENTS
    assert R.DATA.tickers == ("AAPL", "AMZN", "GOOG", "INTC", "MSFT")
    ids = {s.study_id for s in R.STUDY_FAMILY}
    assert ids == {
        "ofi_contemporaneous",
        "ofi_forward",
        "queue_imbalance_next_move",
        "sign_autocorrelation",
        "impact_curve",
    }


def test_protocols_are_runtime_checkable():
    # Contracts expose the seams the parallel workstreams implement.
    for proto in (
        interfaces.BookReconstructor,
        interfaces.OrderFlowSimulator,
        interfaces.FeatureEngine,
        interfaces.Estimator,
        interfaces.Splitter,
        interfaces.MultipleTestingController,
    ):
        assert hasattr(proto, "_is_runtime_protocol")
