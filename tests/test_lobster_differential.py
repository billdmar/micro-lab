"""Differential tests: reconstructed vs reference book, cell-for-cell.

The fast tests build both frames by hand so the match rates are known exactly.
One @pytest.mark.slow test runs the full pipeline (parse -> reconstruct ->
differential) on ONE real LOBSTER ticker to sanity-check end to end; it is
deselected on CI and skipped if the raw file is absent (raw data is never
committed).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import registry as R
from src.lobster import (
    BookReconstructor,
    differential,
    read_message_frame,
    read_orderbook_frame,
)
from src.schema import NO_BID_PRICE, book_columns, book_frame

_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
_TICKER = "INTC_2012-06-21_34200000_57600000"
_LEVELS = 10


def _book(rows: list[list[int]], levels: int = 1) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=book_columns(levels)).astype("int64")


# --------------------------------------------------------------------------- #
# Fast, fully hand-built differentials
# --------------------------------------------------------------------------- #


def test_identical_frames_match_perfectly():
    recon = _book([[110, 20, 100, 10], [110, 20, 101, 15]])
    ref = recon.copy()
    rep = differential(recon, ref, levels=1)
    assert rep.n_rows == 2
    assert rep.row_match_rate == 1.0
    assert rep.cell_match_rate == 1.0
    assert rep.per_level_row_match_rate == {1: 1.0}
    assert rep.first_mismatch_rows == []
    assert rep.first_mismatches == []


def test_single_cell_mismatch_lowers_rates_and_is_reported():
    recon = _book([[110, 20, 100, 10], [110, 20, 101, 15]])
    ref = recon.copy()
    ref.loc[1, "bid_sz_1"] = 99  # one cell differs on row 1
    rep = differential(recon, ref, levels=1)
    assert rep.row_match_rate == 0.5  # one of two rows fails
    # 7 of 8 cells match.
    assert rep.cell_match_rate == pytest.approx(7 / 8)
    assert rep.per_level_row_match_rate == {1: 0.5}
    assert rep.first_mismatch_rows == [1]
    mm = rep.first_mismatches[0]
    assert mm["row"] == 1
    assert mm["reconstructed"]["bid_sz_1"] == 15
    assert mm["reference"]["bid_sz_1"] == 99


def test_per_level_isolates_deep_level_mismatch():
    # Level 1 matches on every row; level 2 mismatches on the second row. This
    # is the shape of the documented deep-level edge case.
    recon = _book(
        [
            [110, 20, 100, 10, 111, 5, 99, 7],
            [110, 20, 100, 10, 112, 5, 99, 7],
        ],
        levels=2,
    )
    ref = recon.copy()
    ref.loc[1, "ask_px_2"] = 113  # deep level differs on the second row only
    rep = differential(recon, ref, levels=2)
    assert rep.per_level_row_match_rate[1] == 1.0  # interior intact
    assert rep.per_level_row_match_rate[2] == 0.5  # deepest level churns
    assert rep.row_match_rate == 0.5


def test_reconstructor_output_differences_against_a_hand_book():
    # Drive the reconstructor with events, then difference its emitted frame
    # against a hand-written reference book that we know it should reproduce.
    from src.schema import Event, EventType

    events = [
        Event(0.0, EventType.SUBMIT, 1, 20, 110, -1),  # ask 110x20
        Event(1.0, EventType.SUBMIT, 2, 10, 100, 1),  # bid 100x10
        Event(2.0, EventType.EXECUTE_VISIBLE, 1, 8, 110, -1),  # ask -> 12
    ]
    states = list(BookReconstructor(levels=1).run(events))
    recon = book_frame(states, levels=1)
    ref = _book(
        [
            [110, 20, NO_BID_PRICE, 0],
            [110, 20, 100, 10],
            [110, 12, 100, 10],
        ]
    )
    rep = differential(recon, ref, levels=1)
    assert rep.row_match_rate == 1.0
    assert rep.per_level_row_match_rate == {1: 1.0}


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_differential_rejects_row_count_mismatch():
    recon = _book([[110, 20, 100, 10]])
    ref = _book([[110, 20, 100, 10], [110, 20, 100, 10]])
    with pytest.raises(ValueError, match="row-count mismatch"):
        differential(recon, ref, levels=1)


def test_differential_rejects_missing_columns():
    recon = pd.DataFrame({"ask_px_1": [110], "ask_sz_1": [20]})  # missing bid cols
    ref = _book([[110, 20, 100, 10]])
    with pytest.raises(ValueError, match="missing columns"):
        differential(recon, ref, levels=1)


def test_empty_frames_report_nan_rates():
    empty = _book([]).astype("int64")
    rep = differential(empty, empty, levels=1)
    assert rep.n_rows == 0
    assert math.isnan(rep.row_match_rate)
    assert math.isnan(rep.cell_match_rate)
    assert math.isnan(rep.per_level_row_match_rate[1])


def test_tolerance_is_zero_by_registry():
    # A single off-by-one cell must count as a mismatch (exact integer compare).
    recon = _book([[110, 20, 100, 10]])
    ref = _book([[110, 21, 100, 10]])
    assert R.RECON_CELL_TOLERANCE == 0
    rep = differential(recon, ref, levels=1)
    assert rep.row_match_rate == 0.0


# --------------------------------------------------------------------------- #
# Slow end-to-end sanity check on ONE real ticker (deselected on CI)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_real_ticker_reconstruction_differential(capsys):
    """End-to-end sanity check on ONE real ticker (deselected on CI).

    IMPORTANT HONEST FINDING (documented, not hidden by loosening tolerance):
    the *free* LOBSTER sample "_10" message file is LEVEL-RESTRICTED — it only
    records events touching the visible top-10 levels. Message-only
    reconstruction therefore CANNOT reproduce LOBSTER's orderbook file over a
    full session, for two compounding reasons proven in docs/DESIGN.md:

      1. pre-open resting liquidity is never declared in the messages (it only
         appears in orderbook row 0), so the book starts empty; and
      2. when a top-of-book order is removed, the level-11 order promoted into
         view was never seen (its submit is off-window), and off-window removals
         of orders that later drift out of the top-10 are likewise absent — so
         phantom levels accumulate as the intraday price drifts.

    The consequence is a LOW full-day differential (ask touch ~ tens of %, bid
    far lower once price drifts down), NOT the ~100% a complete message feed
    would give. We assert the things that ARE true and meaningful rather than a
    match target this data cannot support:

      * row-for-row alignment (parser + reconstructor emit one state per event);
      * the per-event SUBMIT arithmetic is EXACT against LOBSTER ground truth —
        for every submit at a price inside LOBSTER's visible window, the
        reference size at that price changes by exactly the event size (zero
        violations). This validates the event semantics the reconstructor
        encodes, independent of the un-reconstructable pre-open / level-11
        liquidity. The observed differential rates are recorded for the note.
    """
    msg_path = _RAW / f"{_TICKER}_message_{_LEVELS}.csv"
    ob_path = _RAW / f"{_TICKER}_orderbook_{_LEVELS}.csv"
    if not msg_path.exists() or not ob_path.exists():
        pytest.skip("raw LOBSTER data not present (never committed)")

    events = read_message_frame(msg_path)
    reference = read_orderbook_frame(ob_path, levels=_LEVELS)

    from src.schema import Event, EventType

    rows = (
        Event(t, EventType(int(et)), int(oid), int(sz), int(px), int(d))
        for t, et, oid, sz, px, d in events.itertuples(index=False, name=None)
    )
    states = list(BookReconstructor(levels=_LEVELS).run(rows))
    recon = book_frame(states, levels=_LEVELS)

    rep = differential(recon, reference, levels=_LEVELS)
    assert rep.n_rows == len(reference)  # one emitted state per message

    # Per-event SUBMIT arithmetic is exact against LOBSTER ground truth: within
    # the visible price window, ref size at the submitted price rises by exactly
    # event.size. This is the meaningful end-to-end validation of the semantics
    # the reconstructor implements (its hand-built unit tests prove it applies
    # them; this proves the semantics match real LOBSTER data).
    et = events["event_type"].to_numpy()
    px = events["price"].to_numpy()
    sz = events["size"].to_numpy()
    dr = events["direction"].to_numpy()
    ask_px = reference[[f"ask_px_{lvl}" for lvl in range(1, _LEVELS + 1)]].to_numpy()
    ask_sz = reference[[f"ask_sz_{lvl}" for lvl in range(1, _LEVELS + 1)]].to_numpy()
    bid_px = reference[[f"bid_px_{lvl}" for lvl in range(1, _LEVELS + 1)]].to_numpy()
    bid_sz = reference[[f"bid_sz_{lvl}" for lvl in range(1, _LEVELS + 1)]].to_numpy()

    def _size_at(i: int, price: int, side: int) -> int | None:
        prices, sizes = (bid_px, bid_sz) if side == 1 else (ask_px, ask_sz)
        hit = np.flatnonzero(prices[i] == price)
        return int(sizes[i, hit[0]]) if hit.size else None

    violations = 0
    for i in range(1, len(events)):
        if et[i] != int(EventType.SUBMIT):
            continue
        price, side = int(px[i]), int(dr[i])
        prices_row = bid_px[i] if side == 1 else ask_px[i]
        if not (int(prices_row.min()) <= price <= int(prices_row.max())):
            continue  # submit landed outside the visible window; not comparable
        before = _size_at(i - 1, price, side)
        after = _size_at(i, price, side)
        if before is None or after is None:
            continue
        if after - before != int(sz[i]):
            violations += 1
    assert violations == 0, f"{violations} SUBMIT events disagree with LOBSTER size deltas"

    # Record the observed (data-limited) differential rates for the research note.
    with capsys.disabled():
        print(
            f"\n[{_TICKER}] real differential (level-restricted free sample): "
            f"row={rep.row_match_rate:.4f} "
            f"per_level={ {k: round(v, 4) for k, v in rep.per_level_row_match_rate.items()} }"
        )
