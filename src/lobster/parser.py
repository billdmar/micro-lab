"""LOBSTER CSV parsers -> canonical schema frames.

Two LOBSTER sample files describe one session per ticker:

* the **message** file — one order-flow event per row, columns
  ``time, type, order_id, size, price, direction`` — which maps 1:1 onto the
  canonical :data:`schema.EVENT_COLUMNS` (same order), and
* the **orderbook** file — the reference book state AFTER each message, as
  ``4*levels`` interleaved columns ``ask_px_1, ask_sz_1, bid_px_1, bid_sz_1,
  ask_px_2, ...`` — matching :func:`schema.book_columns`.

Row ``i`` of the orderbook is the book after message ``i``; the two files are
row-for-row aligned. Units follow LOBSTER exactly (prices integer 1/10000
dollar, sizes integer shares, time float seconds), so the reconstruction
differential is a literal integer comparison. See docs/DESIGN.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schema import EVENT_COLUMNS, EVENT_DTYPES, book_columns, validate_event_frame

#: Raw LOBSTER message columns, in file order. They align positionally with
#: EVENT_COLUMNS (time_s, event_type, order_id, size, price, direction).
_RAW_MESSAGE_COLUMNS: tuple[str, ...] = (
    "time",
    "type",
    "order_id",
    "size",
    "price",
    "direction",
)


def read_message_frame(path: str | Path) -> pd.DataFrame:
    """Read a LOBSTER message CSV into a canonical, validated event frame.

    The six raw columns map positionally onto ``schema.EVENT_COLUMNS``; the
    frame is cast to ``EVENT_DTYPES`` and checked by
    :func:`schema.validate_event_frame` before it is returned.
    """
    raw = pd.read_csv(
        path,
        header=None,
        names=list(_RAW_MESSAGE_COLUMNS),
        usecols=range(len(_RAW_MESSAGE_COLUMNS)),
    )
    df = raw.set_axis(list(EVENT_COLUMNS), axis=1).astype(EVENT_DTYPES)[list(EVENT_COLUMNS)]
    validate_event_frame(df)
    return df


def read_orderbook_frame(path: str | Path, levels: int) -> pd.DataFrame:
    """Read a LOBSTER orderbook CSV into a canonical ``levels``-deep book frame.

    The file carries ``4*levels`` interleaved integer columns in LOBSTER's
    ``ask_px, ask_sz, bid_px, bid_sz`` order (see :func:`schema.book_columns`);
    empty levels already use LOBSTER's price sentinels. All cells are integers.
    """
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    cols = book_columns(levels)
    df = pd.read_csv(
        path,
        header=None,
        names=cols,
        usecols=range(len(cols)),
    )
    return df.astype("int64")
