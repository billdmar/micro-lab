"""Tests for the leakage detectors — the adversarial core of the CV module.

The point of these tests is DELIBERATE leaks that MUST be caught: an overlapping
label window across the fold boundary, a future-timestamped feature, and a
train index inside the embargo zone. A correct purged+embargoed split (built by
the splitter) must produce zero findings — that positive case lives in
test_cv_splitter.py's leakage-audit test.
"""

from __future__ import annotations

import numpy as np

from src.cv.leakage import (
    LeakageFinding,
    audit_fold,
    detect_embargo_breach,
    detect_future_features,
    detect_label_overlap,
)
from src.schema import FeatureSpec

# --- (a) label-window overlap across the boundary (missing purge) ----------- #


def test_label_overlap_across_boundary_is_flagged():
    # Train index 8 with a horizon-5 label window [8,13] reaches into test block
    # [10,14]; index 4's window [4,9] does not. Exactly one offender.
    train_idx = np.array([0, 4, 8])
    test_idx = np.array([10, 11, 12, 13, 14])
    findings = detect_label_overlap(train_idx, test_idx, label_horizon=5)
    assert len(findings) == 1
    assert findings[0].kind == "label_overlap"
    assert findings[0].count == 1


def test_label_overlap_boundary_touch_counts():
    # Window end exactly equal to test_lo is an overlap (closed window [i,i+h]).
    train_idx = np.array([5])  # window [5,10]
    test_idx = np.array([10, 11])
    assert detect_label_overlap(train_idx, test_idx, label_horizon=5)[0].count == 1


def test_label_overlap_clean_when_purged():
    # All train indices' label windows end strictly before the test block.
    train_idx = np.array([0, 1, 2, 3])  # h=5 -> windows end at <= 8 < 10
    test_idx = np.array([10, 11, 12])
    assert detect_label_overlap(train_idx, test_idx, label_horizon=5) == []


def test_label_overlap_zero_horizon_needs_actual_membership():
    # With h=0 the label window is a point; only a train index literally inside
    # the block overlaps (i.e. a train/test intersection).
    assert detect_label_overlap(np.array([0, 9]), np.array([10, 11]), label_horizon=0) == []
    assert detect_label_overlap(np.array([10]), np.array([10, 11]), label_horizon=0)[0].count == 1


# --- (b) future-timestamped feature used as contemporaneous ----------------- #


def test_future_feature_is_flagged():
    specs = [
        FeatureSpec("ofi", "order flow imbalance"),  # info_offset 0 -> ok
        FeatureSpec("peek", "uses a future book snapshot", info_offset=3.0),  # leak
    ]
    findings = detect_future_features(specs)
    assert len(findings) == 1
    assert findings[0].kind == "future_feature"
    assert findings[0].count == 1
    assert "peek" in findings[0].detail


def test_label_with_info_offset_is_not_a_feature_leak():
    # A label is legitimately known only at t+horizon; info_offset>0 is expected.
    specs = [FeatureSpec("ret10", "fwd ret", is_label=True, horizon=10, info_offset=10.0)]
    assert detect_future_features(specs) == []


def test_all_backward_features_are_clean():
    specs = [FeatureSpec("a", "x"), FeatureSpec("b", "y")]
    assert detect_future_features(specs) == []


# --- (c) train/test adjacency without embargo (boundary contamination) ------ #


def test_embargo_breach_before_block_is_flagged():
    # Train index 9 sits one position before test block [10,14] with embargo 3.
    train_idx = np.array([0, 5, 9])
    test_idx = np.array([10, 11, 12, 13, 14])
    findings = detect_embargo_breach(train_idx, test_idx, embargo=3)
    assert len(findings) == 1
    assert findings[0].kind == "embargo_breach"
    assert findings[0].count == 1


def test_embargo_breach_after_block_is_flagged():
    # Train index 16 sits within embargo=3 after test block ending at 14.
    train_idx = np.array([16, 30])
    test_idx = np.array([10, 11, 12, 13, 14])
    assert detect_embargo_breach(train_idx, test_idx, embargo=3)[0].count == 1


def test_embargo_breach_both_sides_counted():
    train_idx = np.array([8, 9, 15, 16])  # all within embargo=3 of [10,14]
    test_idx = np.array([10, 11, 12, 13, 14])
    assert detect_embargo_breach(train_idx, test_idx, embargo=3)[0].count == 4


def test_embargo_zero_never_breaches():
    train_idx = np.array([9, 15])
    test_idx = np.array([10, 11, 12, 13, 14])
    assert detect_embargo_breach(train_idx, test_idx, embargo=0) == []


def test_embargo_clean_when_gap_is_wide_enough():
    # Nearest train indices are exactly `embargo`+1 away on each side.
    train_idx = np.array([6, 18])  # block [10,14], embargo=3 -> zone [7,9] & [15,17]
    test_idx = np.array([10, 11, 12, 13, 14])
    assert detect_embargo_breach(train_idx, test_idx, embargo=3) == []


# --- audit_fold aggregation ------------------------------------------------- #


def test_audit_fold_aggregates_all_three_kinds():
    # A single fold rigged to trip all three detectors at once.
    train_idx = np.array([8, 9])  # 8: label overlap (h=5 -> [8,13]); 9: embargo breach
    test_idx = np.array([10, 11, 12])
    specs = [FeatureSpec("peek", "future", info_offset=2.0)]
    findings = audit_fold(train_idx, test_idx, label_horizon=5, embargo=3, specs=specs)
    kinds = {f.kind for f in findings}
    assert kinds == {"label_overlap", "embargo_breach", "future_feature"}


def test_audit_fold_clean_returns_empty():
    train_idx = np.array([0, 1, 2])
    test_idx = np.array([10, 11, 12])
    specs = [FeatureSpec("ofi", "backward")]
    assert audit_fold(train_idx, test_idx, label_horizon=5, embargo=3, specs=specs) == []


def test_detectors_handle_empty_index_arrays():
    empty = np.empty(0, dtype=int)
    test_idx = np.array([10, 11])
    assert detect_label_overlap(empty, test_idx, label_horizon=5) == []
    assert detect_embargo_breach(empty, test_idx, embargo=3) == []
    assert detect_label_overlap(np.array([1]), empty, label_horizon=5) == []


def test_leakage_finding_is_frozen():
    f = LeakageFinding("label_overlap", 1, "x")
    assert f.kind == "label_overlap"
    assert f.count == 1
