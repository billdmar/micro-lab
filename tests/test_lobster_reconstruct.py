"""Reconstructor tests: message stream -> book states by hand.

Every input is a tiny hand-built Event sequence whose resulting BookState is
worked out by hand in the assertions. The reconstructor consumes ONLY events
(never the orderbook file), so these are self-contained. Event codes: 1 submit,
2 cancel-partial, 3 delete, 4 execute-visible, 5 execute-hidden, 7 halt.
"""

from __future__ import annotations

from src.lobster import BookReconstructor
from src.schema import NO_ASK_PRICE, NO_BID_PRICE, BookState, Event, EventType


def _submit(oid: int, price: int, size: int, side: int, t: float = 0.0) -> Event:
    return Event(t, EventType.SUBMIT, oid, size, price, side)


# --------------------------------------------------------------------------- #
# Submit
# --------------------------------------------------------------------------- #


def test_submit_builds_touch_with_sentinels_for_empty_side():
    r = BookReconstructor(levels=1)
    # First a lone bid: the ask side is empty -> sentinel price, size 0.
    state = r.apply(_submit(1, 100, 10, side=1))
    assert state.ask_px == (NO_ASK_PRICE,)
    assert state.ask_sz == (0,)
    assert state.bid_px == (100,)
    assert state.bid_sz == (10,)
    # Now an ask: both touches populated.
    state = r.apply(_submit(2, 110, 20, side=-1))
    assert state.ask_px == (110,) and state.ask_sz == (20,)
    assert state.bid_px == (100,) and state.bid_sz == (10,)


def test_two_orders_same_price_aggregate_size():
    r = BookReconstructor(levels=1)
    r.apply(_submit(1, 100, 10, side=1))
    state = r.apply(_submit(2, 100, 15, side=1))
    assert state.bid_px == (100,)
    assert state.bid_sz == (25,)  # 10 + 15 aggregated at one price level


def test_levels_sorted_ask_ascending_bid_descending():
    r = BookReconstructor(levels=2)
    for oid, (px, side) in enumerate([(110, -1), (112, -1), (100, 1), (98, 1)], start=1):
        state = r.apply(_submit(oid, px, 5, side))
    # Asks ascending from the touch; bids descending.
    assert state.ask_px == (110, 112)
    assert state.bid_px == (100, 98)


# --------------------------------------------------------------------------- #
# Cancel-partial (type 2)
# --------------------------------------------------------------------------- #


def test_cancel_partial_reduces_level_size():
    r = BookReconstructor(levels=1)
    r.apply(_submit(1, 100, 10, side=1))
    state = r.apply(Event(1.0, EventType.CANCEL_PARTIAL, 1, 4, 100, 1))
    assert state.bid_px == (100,)
    assert state.bid_sz == (6,)  # 10 - 4


def test_cancel_partial_only_reduces_the_named_order_at_a_shared_level():
    r = BookReconstructor(levels=1)
    r.apply(_submit(1, 100, 10, side=1))
    r.apply(_submit(2, 100, 15, side=1))  # level total 25
    state = r.apply(Event(1.0, EventType.CANCEL_PARTIAL, 1, 4, 100, 1))
    assert state.bid_sz == (21,)  # 25 - 4; order 2 untouched


# --------------------------------------------------------------------------- #
# Delete (type 3)
# --------------------------------------------------------------------------- #


def test_delete_removes_order_and_empties_level():
    r = BookReconstructor(levels=1)
    r.apply(_submit(1, 100, 10, side=1))
    state = r.apply(Event(1.0, EventType.DELETE, 1, 10, 100, 1))
    # The only bid is gone -> sentinel touch.
    assert state.bid_px == (NO_BID_PRICE,)
    assert state.bid_sz == (0,)


def test_delete_leaves_a_sibling_order_at_the_same_price():
    r = BookReconstructor(levels=1)
    r.apply(_submit(1, 100, 10, side=1))
    r.apply(_submit(2, 100, 15, side=1))
    state = r.apply(Event(1.0, EventType.DELETE, 1, 10, 100, 1))
    assert state.bid_px == (100,)
    assert state.bid_sz == (15,)  # order 2's size remains


# --------------------------------------------------------------------------- #
# Execute-visible (type 4) — reduces/removes the resting order hit
# --------------------------------------------------------------------------- #


def test_execute_visible_reduces_resting_order():
    r = BookReconstructor(levels=1)
    r.apply(_submit(1, 110, 20, side=-1))  # resting ask
    state = r.apply(Event(1.0, EventType.EXECUTE_VISIBLE, 1, 8, 110, -1))
    assert state.ask_px == (110,)
    assert state.ask_sz == (12,)  # 20 - 8


def test_execute_visible_fully_consuming_removes_the_level():
    r = BookReconstructor(levels=1)
    r.apply(_submit(1, 110, 20, side=-1))
    state = r.apply(Event(1.0, EventType.EXECUTE_VISIBLE, 1, 20, 110, -1))
    assert state.ask_px == (NO_ASK_PRICE,)
    assert state.ask_sz == (0,)


# --------------------------------------------------------------------------- #
# Execute-hidden (5) and halt (7) — book UNCHANGED
# --------------------------------------------------------------------------- #


def test_execute_hidden_leaves_book_unchanged():
    r = BookReconstructor(levels=1)
    before = r.apply(_submit(1, 100, 10, side=1))
    # A hidden execution references order_id 0 (not on the visible book).
    after = r.apply(Event(1.0, EventType.EXECUTE_HIDDEN, 0, 5, 100, -1))
    assert after == before


def test_halt_leaves_book_unchanged():
    r = BookReconstructor(levels=1)
    before = r.apply(_submit(1, 100, 10, side=1))
    after = r.apply(Event(1.0, EventType.HALT, 0, 0, NO_BID_PRICE, -1))
    assert after == before


# --------------------------------------------------------------------------- #
# Pre-window orders: an unknown order_id is skipped (honest unknown), not guessed
# --------------------------------------------------------------------------- #


def test_unknown_order_id_on_delete_is_skipped():
    r = BookReconstructor(levels=1)
    before = r.apply(_submit(1, 100, 10, side=1))
    # order_id 99 was never submitted (pre-window); its price is unknowable.
    after = r.apply(Event(1.0, EventType.DELETE, 99, 10, 100, 1))
    assert after == before  # book unchanged, no fabricated level touched


def test_unknown_order_id_on_execute_is_skipped():
    r = BookReconstructor(levels=1)
    before = r.apply(_submit(1, 110, 20, side=-1))
    after = r.apply(Event(1.0, EventType.EXECUTE_VISIBLE, 42, 5, 110, -1))
    assert after == before


def test_cancel_qty_clamped_to_remaining_size():
    # A cancel larger than the tracked remaining size cannot drive a level
    # negative; it removes the order and empties the level.
    r = BookReconstructor(levels=1)
    r.apply(_submit(1, 100, 10, side=1))
    state = r.apply(Event(1.0, EventType.CANCEL_PARTIAL, 1, 999, 100, 1))
    assert state.bid_px == (NO_BID_PRICE,)
    assert state.bid_sz == (0,)


# --------------------------------------------------------------------------- #
# run() streaming + contract
# --------------------------------------------------------------------------- #


def test_run_yields_one_state_per_event_matching_apply():
    events = [
        _submit(1, 110, 20, side=-1),
        _submit(2, 100, 10, side=1),
        Event(1.0, EventType.EXECUTE_VISIBLE, 1, 8, 110, -1),
        Event(2.0, EventType.DELETE, 2, 10, 100, 1),
    ]
    streamed = list(BookReconstructor(levels=1).run(events))
    assert len(streamed) == len(events)
    # The final state: ask reduced to 12, bid emptied.
    assert streamed[-1] == BookState(
        ask_px=(110,), ask_sz=(12,), bid_px=(NO_BID_PRICE,), bid_sz=(0,)
    )


def test_levels_must_be_positive():
    import pytest

    with pytest.raises(ValueError, match="levels must be"):
        BookReconstructor(levels=0)


def test_multi_level_snapshot_pads_missing_depth_with_sentinels():
    r = BookReconstructor(levels=3)
    r.apply(_submit(1, 110, 20, side=-1))
    state = r.apply(_submit(2, 100, 10, side=1))
    # Only one level per side exists; deeper levels are sentinel-padded.
    assert state.ask_px == (110, NO_ASK_PRICE, NO_ASK_PRICE)
    assert state.ask_sz == (20, 0, 0)
    assert state.bid_px == (100, NO_BID_PRICE, NO_BID_PRICE)
    assert state.bid_sz == (10, 0, 0)
