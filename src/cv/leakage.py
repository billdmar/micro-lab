"""Leakage detectors — the adversarial check on the CV machinery.

The splitter is *supposed* to make look-ahead impossible; these detectors prove
it, and catch hand-constructed leaks in tests. Given the FeatureSpecs of the
columns in play and one fold's ``(train_idx, test_idx)``, they flag three
distinct contaminations (Lopez de Prado, ch. 7):

  (a) **label_overlap** — a TRAIN index whose forward label window
      ``[i, i+label_horizon]`` reaches into the (contiguous) test block: its
      label is co-determined with the test period, so purging failed.
  (b) **future_feature** — a FEATURE (``is_label=False``) declared with
      ``info_offset > 0``: it is future-timestamped yet would be used as if
      known at the anchor event. Labels legitimately carry ``info_offset > 0``.
  (c) **embargo_breach** — a TRAIN index within ``embargo`` positions of either
      side of the test block: serial-correlation contamination that survives
      purging alone.

Each detector returns a list of ``LeakageFinding``; an empty list means clean.
The functions read only integer index arrays and FeatureSpec metadata — no data
values — so they are cheap enough to run on every fold as an audit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.schema import FeatureSpec


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    """One flagged contamination. ``kind`` is the machine-readable category
    ("label_overlap" | "future_feature" | "embargo_breach"); ``count`` is the
    number of offending indices (or specs); ``detail`` is a human message."""

    kind: str
    count: int
    detail: str


def detect_future_features(specs: Sequence[FeatureSpec]) -> list[LeakageFinding]:
    """Flag any non-label feature whose information is not known at its anchor
    (``info_offset > 0``). Such a column peeks into the future while being used
    as a contemporaneous predictor. Labels are exempt — they are known only at
    ``t + horizon`` by definition."""
    offenders = [s.name for s in specs if not s.is_label and s.info_offset > 0]
    if not offenders:
        return []
    return [
        LeakageFinding(
            "future_feature",
            len(offenders),
            f"features with info_offset>0 used as contemporaneous: {offenders}",
        )
    ]


def detect_label_overlap(
    train_idx: np.ndarray, test_idx: np.ndarray, label_horizon: int
) -> list[LeakageFinding]:
    """Flag TRAIN indices whose label window ``[i, i+label_horizon]`` overlaps
    the (contiguous) test block. Empty result means purge succeeded."""
    if train_idx.size == 0 or test_idx.size == 0:
        return []
    test_lo, test_hi = int(test_idx.min()), int(test_idx.max())
    # Window [i, i+h] intersects [test_lo, test_hi] iff i+h >= test_lo and i <= test_hi.
    overlap = (train_idx + label_horizon >= test_lo) & (train_idx <= test_hi)
    n_bad = int(overlap.sum())
    if n_bad == 0:
        return []
    return [
        LeakageFinding(
            "label_overlap",
            n_bad,
            f"{n_bad} train indices have a label window reaching into the test block "
            f"[{test_lo},{test_hi}] at horizon {label_horizon}",
        )
    ]


def detect_embargo_breach(
    train_idx: np.ndarray, test_idx: np.ndarray, embargo: int
) -> list[LeakageFinding]:
    """Flag TRAIN indices sitting within ``embargo`` positions on either side of
    the test block. The embargo zone is ``[test_lo-embargo, test_lo)`` before and
    ``(test_hi, test_hi+embargo]`` after; any train index there is boundary
    contamination."""
    if train_idx.size == 0 or test_idx.size == 0 or embargo == 0:
        return []
    test_lo, test_hi = int(test_idx.min()), int(test_idx.max())
    before = (train_idx >= test_lo - embargo) & (train_idx < test_lo)
    after = (train_idx > test_hi) & (train_idx <= test_hi + embargo)
    n_bad = int((before | after).sum())
    if n_bad == 0:
        return []
    return [
        LeakageFinding(
            "embargo_breach",
            n_bad,
            f"{n_bad} train indices lie within embargo={embargo} of the test block "
            f"[{test_lo},{test_hi}]",
        )
    ]


def audit_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    label_horizon: int,
    embargo: int,
    specs: Sequence[FeatureSpec] = (),
) -> list[LeakageFinding]:
    """Run all three detectors on one fold and return the combined findings
    (empty == clean). ``specs`` defaults to none so callers auditing only the
    index geometry need not pass them."""
    findings: list[LeakageFinding] = []
    findings += detect_label_overlap(train_idx, test_idx, label_horizon)
    findings += detect_embargo_breach(train_idx, test_idx, embargo)
    findings += detect_future_features(specs)
    return findings
