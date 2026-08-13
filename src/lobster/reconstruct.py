"""Independent limit-order-book reconstructor.

:class:`BookReconstructor` consumes ONLY the message/event stream and emits the
top-``levels`` book state after each event, so its output can be differenced
row-for-row against LOBSTER's reference orderbook file. It NEVER reads that
file — that independence is the whole point of the differential.

State is a per-order map ``order_id -> (price, size, side)`` plus per-price-level
size aggregates for each side. Events are applied per the LOBSTER codes:

* ``1 SUBMIT``           add a new order at its price/side;
* ``2 CANCEL_PARTIAL``   reduce the referenced resting order by the event size;
* ``3 DELETE``           remove the referenced resting order entirely;
* ``4 EXECUTE_VISIBLE``  reduce/remove the visible resting order that was hit;
* ``5 EXECUTE_HIDDEN``   hidden order, not on the visible book -> no change;
* ``7 HALT``             trading-halt indicator -> no change.

Honest-unknown edge case: the sample only carries events for orders that touch
the tracked visible depth, so a cancel/delete/execution can reference an order
submitted BEFORE the recording window (never seen here). Its price cannot be
known from the messages alone, so such an event is skipped rather than guessed.
These pre-existing orders linger mostly at the deepest displayed level, which is
where the differential shows its documented, enumerated mismatch. See
docs/DESIGN.md and the differential's per-level report.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..schema import (
    NO_ASK_PRICE,
    NO_BID_PRICE,
    BookState,
    Event,
    EventType,
)


class BookReconstructor:
    """A LOBSTER book reconstructor implementing ``interfaces.BookReconstructor``.

    ``levels`` is the displayed depth of each emitted :class:`schema.BookState`
    (LOBSTER sample files are level-10). Book sides are ``price -> total size``
    dicts; the per-order map lets cancels/deletes/executions find the exact
    resting price and size to remove.
    """

    def __init__(self, levels: int) -> None:
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        self.levels = levels
        # order_id -> [price, remaining_size, side(+1 bid / -1 ask)]
        self._orders: dict[int, list[int]] = {}
        # price -> aggregated resting size, one dict per side
        self._bids: dict[int, int] = {}
        self._asks: dict[int, int] = {}

    # -- contract ----------------------------------------------------------- #

    def apply(self, event: Event) -> BookState:
        """Apply one event to the book and return the resulting snapshot."""
        etype = event.event_type
        if etype == EventType.SUBMIT:
            self._add(event.order_id, event.price, event.size, event.direction)
        elif etype in (EventType.CANCEL_PARTIAL, EventType.EXECUTE_VISIBLE):
            self._reduce(event.order_id, event.size)
        elif etype == EventType.DELETE:
            self._remove(event.order_id)
        # EXECUTE_HIDDEN (5) and HALT (7) leave the visible book unchanged.
        return self._snapshot()

    def run(self, events: Iterable[Event]) -> Iterator[BookState]:
        """Stream one book state per event, in order."""
        for event in events:
            yield self.apply(event)

    # -- state mutation ----------------------------------------------------- #

    def _side(self, side: int) -> dict[int, int]:
        return self._bids if side == 1 else self._asks

    def _add(self, order_id: int, price: int, size: int, side: int) -> None:
        """Record a new resting order and add its size to the price level."""
        self._orders[order_id] = [price, size, side]
        level = self._side(side)
        level[price] = level.get(price, 0) + size

    def _reduce(self, order_id: int, qty: int) -> None:
        """Reduce a known resting order by ``qty`` (cancel-partial / execution).

        An unknown ``order_id`` is a pre-existing order (submitted before the
        window); its price is unknowable from messages, so the event is skipped.
        ``qty`` is clamped to the tracked remaining size for robustness.
        """
        order = self._orders.get(order_id)
        if order is None:
            return
        price, size, side = order
        delta = min(qty, size)
        self._decrement_level(self._side(side), price, delta)
        remaining = size - delta
        if remaining <= 0:
            del self._orders[order_id]
        else:
            order[1] = remaining

    def _remove(self, order_id: int) -> None:
        """Delete a known resting order entirely (unknown ids are skipped)."""
        order = self._orders.pop(order_id, None)
        if order is None:
            return
        price, size, side = order
        self._decrement_level(self._side(side), price, size)

    @staticmethod
    def _decrement_level(level: dict[int, int], price: int, delta: int) -> None:
        """Subtract ``delta`` shares at ``price``, dropping emptied levels."""
        remaining = level.get(price, 0) - delta
        if remaining <= 0:
            level.pop(price, None)
        else:
            level[price] = remaining

    # -- snapshot ----------------------------------------------------------- #

    def _snapshot(self) -> BookState:
        """Emit the top-``levels`` state: asks ascending, bids descending, with
        LOBSTER sentinels padding any level that does not exist."""
        ask_prices = sorted(self._asks)[: self.levels]
        bid_prices = sorted(self._bids, reverse=True)[: self.levels]

        ask_px = [NO_ASK_PRICE] * self.levels
        ask_sz = [0] * self.levels
        bid_px = [NO_BID_PRICE] * self.levels
        bid_sz = [0] * self.levels
        for i, p in enumerate(ask_prices):
            ask_px[i], ask_sz[i] = p, self._asks[p]
        for i, p in enumerate(bid_prices):
            bid_px[i], bid_sz[i] = p, self._bids[p]

        return BookState(
            ask_px=tuple(ask_px),
            ask_sz=tuple(ask_sz),
            bid_px=tuple(bid_px),
            bid_sz=tuple(bid_sz),
        )
