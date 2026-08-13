"""Frozen data contracts for micro-lab (core contract module; changes are made deliberately here).

Everything downstream — parsers, the reconstructor, the simulator, features,
estimators, CV — speaks these types. Two representations coexist by design:

* **Row objects** (``Event``, ``BookState``) — immutable dataclasses for
  single-item APIs, hand-built test fixtures, and readable assertions.
* **Columnar frames** — for the ~2.1M-event working sets we process as pandas
  DataFrames with the canonical column names/dtypes defined here. The column
  constants are the contract; helpers convert between the two views.

Units follow LOBSTER exactly so the reconstruction differential is a literal
comparison: prices are integers in 1/10000 dollar ("price times 10000"),
sizes are integer shares, time is float seconds after midnight (millisecond+
precision). See docs/DESIGN.md for the rationale behind each choice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Order-flow events (the "message" stream — LOBSTER-native and simulator-native)
# --------------------------------------------------------------------------- #


class EventType(IntEnum):
    """LOBSTER message event codes (see the sample-files ReadMe).

    Code 6 (auction/cross) does not occur in the equity sample and is omitted.
    """

    SUBMIT = 1  # submission of a new limit order
    CANCEL_PARTIAL = 2  # partial cancellation (size reduced)
    DELETE = 3  # total deletion of a limit order
    EXECUTE_VISIBLE = 4  # execution against a visible limit order
    EXECUTE_HIDDEN = 5  # execution against a hidden order (not on the book)
    HALT = 7  # trading-halt indicator


#: Canonical column order + names for a message/event frame. This IS the schema:
#: parsers, the simulator, and every consumer must produce/accept these exactly.
EVENT_COLUMNS: tuple[str, ...] = (
    "time_s",  # float64  seconds after midnight
    "event_type",  # int8     one of EventType
    "order_id",  # int64    exchange order reference (0 for hidden executions)
    "size",  # int64    shares
    "price",  # int64    dollar price * 10000
    "direction",  # int8     +1 buy limit, -1 sell limit
)

EVENT_DTYPES: dict[str, str] = {
    "time_s": "float64",
    "event_type": "int8",
    "order_id": "int64",
    "size": "int64",
    "price": "int64",
    "direction": "int8",
}


@dataclass(frozen=True, slots=True)
class Event:
    """A single order-flow event. Prices are integer 1/10000-dollar; time is
    float seconds after midnight; direction is the side of the resting limit
    order (+1 buy, -1 sell) — for an execution (type 4/5) the aggressor is the
    opposite side (executing a sell limit = buyer-initiated trade)."""

    time_s: float
    event_type: EventType
    order_id: int
    size: int
    price: int
    direction: int

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError(f"direction must be +/-1, got {self.direction}")
        if self.size < 0:
            raise ValueError(f"size must be non-negative, got {self.size}")


# --------------------------------------------------------------------------- #
# Book states (the reconstructed / reference "orderbook" stream)
# --------------------------------------------------------------------------- #

#: LOBSTER sentinels for a price level that does not exist at a given depth.
NO_ASK_PRICE: int = 9999999999  # empty ask level price
NO_BID_PRICE: int = -9999999999  # empty bid level price


def book_columns(levels: int) -> list[str]:
    """Canonical interleaved column names for an L-level book snapshot frame,
    matching LOBSTER's orderbook layout: ask_px_1, ask_sz_1, bid_px_1,
    bid_sz_1, ask_px_2, ... — 4*levels columns."""
    cols: list[str] = []
    for lvl in range(1, levels + 1):
        cols += [f"ask_px_{lvl}", f"ask_sz_{lvl}", f"bid_px_{lvl}", f"bid_sz_{lvl}"]
    return cols


@dataclass(frozen=True, slots=True)
class BookState:
    """Top-of-book-through-level-L snapshot AFTER applying one event.

    ``ask_px``/``bid_px`` hold integer 1/10000-dollar prices (ascending in
    depth from the touch); sizes are shares. Missing levels use the LOBSTER
    sentinels. This is the unit the reconstruction differential compares."""

    ask_px: tuple[int, ...]
    ask_sz: tuple[int, ...]
    bid_px: tuple[int, ...]
    bid_sz: tuple[int, ...]

    @property
    def levels(self) -> int:
        return len(self.ask_px)

    @property
    def best_ask(self) -> int:
        return self.ask_px[0]

    @property
    def best_bid(self) -> int:
        return self.bid_px[0]

    @property
    def mid_price(self) -> float:
        """Mid in 1/10000 dollar. Undefined (NaN) if either touch is empty."""
        if self.ask_px[0] == NO_ASK_PRICE or self.bid_px[0] == NO_BID_PRICE:
            return float("nan")
        return (self.ask_px[0] + self.bid_px[0]) / 2.0

    def to_row(self) -> list[int]:
        """Flatten to the interleaved LOBSTER column order (len 4*levels)."""
        row: list[int] = []
        for i in range(self.levels):
            row += [self.ask_px[i], self.ask_sz[i], self.bid_px[i], self.bid_sz[i]]
        return row


# --------------------------------------------------------------------------- #
# Feature specifications — point-in-time metadata is the contract's teeth
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Declares one column produced by a FeatureEngine and, crucially, *when its
    information is known*. Point-in-time discipline is enforced structurally by
    reading these specs: a feature (``is_label=False``) computed for the event
    at time t may depend only on information with timestamp <= t; a label
    (``is_label=True``) looks forward exactly ``horizon`` and therefore becomes
    known only at t + horizon. The CV/leakage machinery uses ``horizon`` to
    purge and embargo, and the point-in-time audit uses ``info_offset`` to prove
    no feature peeks past its own information time.

    ``info_offset`` is the delay, in the units of ``horizon_unit``, between the
    anchor event time and the moment the value is actually knowable:
      * a backward-looking feature has ``info_offset == 0`` (known at the event);
      * a forward label of horizon h has ``info_offset == h``.
    """

    name: str
    description: str
    is_label: bool = False
    horizon: float = 0.0  # forward look for labels; 0 for features
    horizon_unit: str = "events"  # "events" or "seconds"
    info_offset: float = 0.0  # when the value becomes known (see above)
    causal: bool = True  # backward-looking only (features)

    def __post_init__(self) -> None:
        if self.horizon_unit not in ("events", "seconds"):
            raise ValueError(f"bad horizon_unit: {self.horizon_unit}")
        if self.is_label and self.horizon <= 0:
            raise ValueError("a label must have a positive horizon")
        if (not self.is_label) and self.horizon != 0:
            raise ValueError("a non-label feature must have horizon 0")


# --------------------------------------------------------------------------- #
# Estimation results — effect sizes with CIs and multiple-testing bookkeeping
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EstimationResult:
    """The output of any estimator: an effect size with honest uncertainty and
    the metadata needed for FDR control across the registered study family.

    No result is ever reported without ``ci_low``/``ci_high``. The multiple-
    testing fields (``family_id``, ``p_value_raw``, ``p_value_adj``,
    ``alpha``, ``rejected``) are populated by the FDR machinery once the whole
    family is known; a lone estimate carries the raw p-value and leaves the
    adjusted fields as NaN/None until the family is closed."""

    name: str  # human label for the estimate
    metric: str  # "coef" | "R2" | "IC" | "AUC" | "autocorr" | ...
    point: float  # the effect-size point estimate
    ci_low: float
    ci_high: float
    n_obs: int
    std_error: float = float("nan")  # HAC/Newey-West SE where applicable
    p_value_raw: float = float("nan")
    ci_level: float = 0.95
    ci_method: str = ""  # "hac" | "block_bootstrap" | "analytic" | ...
    # --- multiple-testing bookkeeping (filled by src/multi) --------------------
    family_id: str = ""
    p_value_adj: float = float("nan")
    alpha: float = float("nan")
    rejected: bool | None = None
    extra: dict = field(default_factory=dict)  # estimator-specific diagnostics

    def __post_init__(self) -> None:
        if self.ci_low > self.ci_high:
            raise ValueError(f"ci_low ({self.ci_low}) > ci_high ({self.ci_high})")
        if not (0.0 < self.ci_level < 1.0):
            raise ValueError(f"ci_level must be in (0,1), got {self.ci_level}")


# --------------------------------------------------------------------------- #
# Frame <-> row helpers (single source of truth for column handling)
# --------------------------------------------------------------------------- #


def events_to_frame(events: Sequence[Event]) -> pd.DataFrame:
    """Build a canonical, dtype-correct event frame from Event rows."""
    df = pd.DataFrame(
        {
            "time_s": [e.time_s for e in events],
            "event_type": [int(e.event_type) for e in events],
            "order_id": [e.order_id for e in events],
            "size": [e.size for e in events],
            "price": [e.price for e in events],
            "direction": [e.direction for e in events],
        }
    )
    return df.astype(EVENT_DTYPES)[list(EVENT_COLUMNS)]


def validate_event_frame(df: pd.DataFrame) -> None:
    """Raise if ``df`` is not a well-formed event frame (schema gatekeeper)."""
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"event frame missing columns: {missing}")
    bad_dir = ~df["direction"].isin((-1, 1))
    if bad_dir.any():
        raise ValueError(f"{int(bad_dir.sum())} events have direction not in (-1,1)")
    if (df["size"] < 0).any():
        raise ValueError("event frame has negative sizes")
    if not df["time_s"].is_monotonic_increasing:
        raise ValueError("event frame is not sorted by time_s")


def book_frame(states: Sequence[BookState], levels: int) -> pd.DataFrame:
    """Build a canonical book-snapshot frame (one row per state)."""
    cols = book_columns(levels)
    data = np.array([s.to_row() for s in states], dtype=np.int64)
    return pd.DataFrame(data, columns=cols)


def mid_price_series(book: pd.DataFrame) -> pd.Series:
    """Vectorized mid price (1/10000 dollar) from a book frame; NaN where a
    touch level is empty (LOBSTER sentinel)."""
    ask = book["ask_px_1"].astype("float64")
    bid = book["bid_px_1"].astype("float64")
    mid = (ask + bid) / 2.0
    mid[(book["ask_px_1"] == NO_ASK_PRICE) | (book["bid_px_1"] == NO_BID_PRICE)] = np.nan
    return mid
