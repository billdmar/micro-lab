"""Purged & embargoed walk-forward cross-validation splitters.

Point-in-time discipline continues into the CV layer: a naive KFold split leaks
future information because a training sample whose forward LABEL window reaches
into the test block, or whose FEATURES sit immediately after the test block,
shares information with the test set. Following Lopez de Prado (*Advances in
Financial Machine Learning*, ch. 7), we defend with two mechanisms:

* **Purge** — drop from train any index whose label window ``[i, i+horizon]``
  overlaps the test block (its label is co-determined with the test period).
* **Embargo** — additionally drop ``embargo`` indices on each side of the test
  block, killing the serial-correlation contamination that survives purging
  (the test block's own forward labels bleed into the indices just after it).

Two split policies share this machinery (pick via the constructor ``mode``):

* ``"walkforward"`` (default) — strictly causal: train is PAST only, so no
  fold is ever trained on data from after its test window. This mirrors how a
  real point-in-time study is run and is the honest default for forward-horizon
  claims. With ``n_folds`` blocks it yields ``n_folds - 1`` usable folds (the
  first block has no past to train on).
* ``"purged_kfold"`` — train is the whole series MINUS the test block, its
  purge zone, and both embargo zones. Uses more data per fold at the cost of
  training partly on the future; offered for robustness comparison only.

Blocks are contiguous and deterministic (equal-size via ``linspace`` bounds);
there is no shuffling, so the splitter carries no RNG state at all.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from config import registry

_MODES = ("walkforward", "purged_kfold")


class PurgedWalkForwardSplitter:
    """Walk-forward CV with purged label windows and symmetric embargo.

    Implements ``interfaces.Splitter``. ``n_folds`` contiguous test blocks tile
    ``range(n)``; each block is a test set once. See the module docstring for
    the ``mode`` policies.
    """

    def __init__(self, n_folds: int = registry.CV_N_FOLDS, mode: str = "walkforward") -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.n_folds = n_folds
        self.mode = mode

    def split(
        self, n: int, *, label_horizon: int, embargo: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` index arrays for each usable fold.

        ``label_horizon`` is the forward look (in events) of the label being
        predicted; ``embargo`` is the number of indices removed on each side of
        the test block. Both must be non-negative. Folds whose train set comes
        out empty (e.g. the first walk-forward block) are skipped.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if label_horizon < 0:
            raise ValueError(f"label_horizon must be >= 0, got {label_horizon}")
        if embargo < 0:
            raise ValueError(f"embargo must be >= 0, got {embargo}")

        bounds = np.linspace(0, n, self.n_folds + 1).astype(int)
        for f in range(self.n_folds):
            t_start, t_end = int(bounds[f]), int(bounds[f + 1])
            if t_end <= t_start:
                continue  # degenerate empty block (n < n_folds); nothing to test
            test_idx = np.arange(t_start, t_end)

            if self.mode == "walkforward":
                train_idx = self._walkforward_train(t_start, label_horizon, embargo)
            else:
                train_idx = self._purged_kfold_train(n, t_start, t_end, label_horizon, embargo)

            if train_idx.size == 0:
                continue  # no usable past (first walk-forward block); skip
            yield train_idx, test_idx

    @staticmethod
    def _walkforward_train(t_start: int, label_horizon: int, embargo: int) -> np.ndarray:
        """Past-only train indices, with the purge+embargo zone before the test
        block removed. The removed zone is ``[t_start - max(h, embargo), t_start)``:
        purging drops ``[t_start - h, t_start)`` (label windows reaching the block)
        and the embargo drops ``[t_start - embargo, t_start)``; their union is the
        wider of the two."""
        train_end = t_start - max(label_horizon, embargo)
        if train_end <= 0:
            return np.empty(0, dtype=int)
        return np.arange(0, train_end)

    @staticmethod
    def _purged_kfold_train(
        n: int, t_start: int, t_end: int, label_horizon: int, embargo: int
    ) -> np.ndarray:
        """All indices outside the test block, its purge zone, and both embargo
        zones. A train index within ``label_horizon`` of the block leaks (its label
        window overlaps the block) and one within ``embargo`` leaks via serial
        correlation, so BOTH sides remove ``max(label_horizon, embargo)`` indices —
        symmetric, so the post-block side is safe even if ``embargo < label_horizon``
        (the pre-block side already used the max; the post-block side must too, or a
        train label reaches forward into the test block)."""
        gap = max(label_horizon, embargo)
        forbidden = np.zeros(n, dtype=bool)
        forbidden[t_start:t_end] = True  # the test block itself
        before = max(0, t_start - gap)
        forbidden[before:t_start] = True  # purge + embargo, pre-block
        after = min(n, t_end + gap)
        forbidden[t_end:after] = True  # purge + embargo, post-block
        return np.flatnonzero(~forbidden)
