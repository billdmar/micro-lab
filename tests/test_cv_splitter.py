"""Tests for the purged/embargoed walk-forward splitter.

Split geometry is verified by hand on small n where the answer is enumerable,
and the invariants a correct purged+embargoed split must satisfy (train/test
disjoint, no label overlap, embargo respected) are asserted directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import registry
from src import interfaces
from src.cv.leakage import audit_fold
from src.cv.splitter import PurgedWalkForwardSplitter


def test_implements_splitter_protocol():
    assert isinstance(PurgedWalkForwardSplitter(), interfaces.Splitter)


def test_walkforward_indices_by_hand():
    # n=10, 5 folds -> blocks [0,2)[2,4)[4,6)[6,8)[8,10). h=emb=1.
    # Fold 0 (test [0,1]) has no usable past -> skipped. Purge/embargo zone is
    # max(h,emb)=1 index before each block.
    s = PurgedWalkForwardSplitter(n_folds=5, mode="walkforward")
    folds = list(s.split(10, label_horizon=1, embargo=1))
    got = [(tr.tolist(), te.tolist()) for tr, te in folds]
    assert got == [
        ([0], [2, 3]),
        ([0, 1, 2], [4, 5]),
        ([0, 1, 2, 3, 4], [6, 7]),
        ([0, 1, 2, 3, 4, 5, 6], [8, 9]),
    ]


def test_purged_kfold_indices_by_hand():
    # Same tiling; train = everything minus test block, its pre-block purge zone
    # (max(h,emb) before) and both embargo zones (emb after).
    s = PurgedWalkForwardSplitter(n_folds=5, mode="purged_kfold")
    folds = list(s.split(10, label_horizon=1, embargo=1))
    got = [(tr.tolist(), te.tolist()) for tr, te in folds]
    assert got == [
        ([3, 4, 5, 6, 7, 8, 9], [0, 1]),
        ([0, 5, 6, 7, 8, 9], [2, 3]),
        ([0, 1, 2, 7, 8, 9], [4, 5]),
        ([0, 1, 2, 3, 4, 9], [6, 7]),
        ([0, 1, 2, 3, 4, 5, 6], [8, 9]),
    ]


def test_walkforward_is_strictly_causal():
    # Every train index must precede the whole test block (past-only).
    s = PurgedWalkForwardSplitter(n_folds=5, mode="walkforward")
    for train_idx, test_idx in s.split(300, label_horizon=20, embargo=50):
        assert train_idx.max() < test_idx.min()


def test_correct_split_passes_all_leakage_audits():
    # A properly purged+embargoed split must be disjoint, have no label overlap,
    # and keep every train index outside the embargo zone.
    h, emb = 10, registry.CV_EMBARGO_EVENTS
    for mode in ("walkforward", "purged_kfold"):
        s = PurgedWalkForwardSplitter(n_folds=5, mode=mode)
        n_folds_seen = 0
        for train_idx, test_idx in s.split(2000, label_horizon=h, embargo=emb):
            n_folds_seen += 1
            assert np.intersect1d(train_idx, test_idx).size == 0  # disjoint
            assert audit_fold(train_idx, test_idx, label_horizon=h, embargo=emb) == []
            # No train index within `embargo` of the test block, either side.
            lo, hi = test_idx.min(), test_idx.max()
            assert not ((train_idx >= lo - emb) & (train_idx < lo)).any()
            assert not ((train_idx > hi) & (train_idx <= hi + emb)).any()
        assert n_folds_seen >= 1


def test_test_blocks_tile_the_range_contiguously():
    # The union of all test blocks is exactly range(n) with no gaps/overlaps,
    # and each block is a contiguous forward slice.
    s = PurgedWalkForwardSplitter(n_folds=5, mode="purged_kfold")
    seen: list[int] = []
    for _, test_idx in s.split(53, label_horizon=3, embargo=2):
        assert test_idx.tolist() == list(range(test_idx.min(), test_idx.max() + 1))
        seen += test_idx.tolist()
    assert seen == list(range(53))


def test_embargo_wider_than_horizon_controls_purge_zone():
    # When embargo > horizon the removed pre-block zone is embargo-wide, not h-wide.
    s = PurgedWalkForwardSplitter(n_folds=5, mode="walkforward")
    folds = list(s.split(50, label_horizon=1, embargo=5))
    # Block 1 is [10,20); train_end = 10 - max(1,5) = 5 -> train {0..4}.
    train_idx, test_idx = folds[0]
    assert test_idx.min() == 10
    assert train_idx.tolist() == list(range(5))


def test_zero_horizon_zero_embargo_only_removes_test_block():
    # With no forward label and no embargo, purged_kfold train is the complement
    # of the test block exactly.
    s = PurgedWalkForwardSplitter(n_folds=5, mode="purged_kfold")
    train_idx, test_idx = next(s.split(10, label_horizon=0, embargo=0))
    assert test_idx.tolist() == [0, 1]
    assert train_idx.tolist() == list(range(2, 10))


def test_default_n_folds_from_registry():
    s = PurgedWalkForwardSplitter()
    assert s.n_folds == registry.CV_N_FOLDS


def test_constructor_validation():
    with pytest.raises(ValueError):
        PurgedWalkForwardSplitter(n_folds=1)
    with pytest.raises(ValueError):
        PurgedWalkForwardSplitter(mode="shuffle")


def test_split_argument_validation():
    s = PurgedWalkForwardSplitter()
    with pytest.raises(ValueError):
        list(s.split(0, label_horizon=1, embargo=1))
    with pytest.raises(ValueError):
        list(s.split(10, label_horizon=-1, embargo=1))
    with pytest.raises(ValueError):
        list(s.split(10, label_horizon=1, embargo=-1))


def test_degenerate_small_n_skips_empty_blocks():
    # n < n_folds: some blocks are empty and are silently skipped, and any block
    # whose purge+embargo zones (symmetric, max(h, embargo) each side) leave no
    # train index is also skipped. With n=3 singleton blocks and label_horizon=1,
    # the middle block (index 1) purges indices 0 and 2 on both sides -> empty
    # train -> skipped; the first and last blocks keep one train index each.
    s = PurgedWalkForwardSplitter(n_folds=5, mode="purged_kfold")
    folds = list(s.split(3, label_horizon=1, embargo=0))
    assert len(folds) == 2  # middle singleton has no usable train after symmetric purge
    for tr, te in folds:
        assert te.size == 1
        assert tr.size >= 1
        assert np.intersect1d(tr, te).size == 0
