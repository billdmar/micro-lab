"""Purged/embargoed walk-forward CV and leakage detection."""

from __future__ import annotations

from src.cv.leakage import (
    LeakageFinding,
    audit_fold,
    detect_embargo_breach,
    detect_future_features,
    detect_label_overlap,
)
from src.cv.splitter import PurgedWalkForwardSplitter

__all__ = [
    "PurgedWalkForwardSplitter",
    "LeakageFinding",
    "audit_fold",
    "detect_label_overlap",
    "detect_future_features",
    "detect_embargo_breach",
]
