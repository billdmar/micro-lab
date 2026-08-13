"""Frozen protocol interfaces for micro-lab (core contract module).

These Protocols are the seams between the modules. Each module implements the
protocol it owns; the pipeline wires them together. They are intentionally
minimal — just enough to let the pipeline run unchanged on either real LOBSTER
data or simulator output (both emit the Event schema).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .schema import BookState, EstimationResult, Event, FeatureSpec


@runtime_checkable
class BookReconstructor(Protocol):
    """Consumes the message/event stream ONLY and emits the book state after
    each event. Its output is differenced row-for-row against LOBSTER's
    reference orderbook file — so it must never look at that file."""

    levels: int

    def apply(self, event: Event) -> BookState:
        """Apply one event and return the resulting top-of-book-through-L state."""
        ...

    def run(self, events: Iterable[Event]) -> Iterator[BookState]:
        """Stream states for a sequence of events (one per event)."""
        ...


@runtime_checkable
class OrderFlowSimulator(Protocol):
    """Generates synthetic order flow in the Event schema, optionally with an
    injected ground-truth relationship linking a feature to future mid drift
    (for the recovery gate) or pure noise (for the placebo gate)."""

    def generate(self, n_events: int, seed: int) -> pd.DataFrame:
        """Return a canonical event frame of ``n_events`` rows, deterministic
        in ``seed``."""
        ...


@runtime_checkable
class FeatureEngine(Protocol):
    """Computes one or more point-in-time feature/label columns from an event
    frame and the aligned reconstructed book frame. Every produced column is
    described by a FeatureSpec so the leakage machinery can reason about it."""

    def specs(self) -> Sequence[FeatureSpec]:
        """The FeatureSpecs this engine produces (feature and/or label cols)."""
        ...

    def compute(self, events: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with one column per spec, index-aligned to ``events``.
        Backward-looking columns at row i use only rows <= i; label columns look
        forward exactly their declared horizon."""
        ...


@runtime_checkable
class Estimator(Protocol):
    """Fits a relationship between feature(s) X and target y and returns an
    EstimationResult carrying the effect size WITH a confidence interval."""

    def estimate(self, X: np.ndarray, y: np.ndarray, *, name: str) -> EstimationResult: ...


@runtime_checkable
class Splitter(Protocol):
    """Yields (train_idx, test_idx) pairs for purged & embargoed walk-forward
    cross-validation. Label horizons overlapping the test window are purged
    from train and an embargo removes the post-test-boundary contamination."""

    def split(
        self, n: int, *, label_horizon: int, embargo: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...


@runtime_checkable
class MultipleTestingController(Protocol):
    """Applies FDR control (Benjamini-Hochberg) across a registered family of
    EstimationResults, filling their adjusted-p / rejected / alpha fields."""

    def control(
        self, results: Sequence[EstimationResult], *, family_id: str, alpha: float
    ) -> list[EstimationResult]: ...
