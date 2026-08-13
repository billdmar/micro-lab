"""Book-reconstruction differential.

Compares a reconstructed book frame against LOBSTER's reference orderbook frame
cell-for-cell. Both are canonical ``schema.book_columns(levels)`` integer frames
(prices 1/10000 dollar, sizes shares), so the comparison is exact — the
tolerance is :data:`config.registry.RECON_CELL_TOLERANCE` (zero mismatched
cells). A correct reconstructor reproduces LOBSTER exactly on the well-defined
interior; the deepest displayed level can mismatch when an order from OUTSIDE
the tracked window surfaces there (see reconstruct.py's honest-unknown note),
so we report the match rate PER LEVEL rather than loosen the tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import registry as R

from ..schema import book_columns


@dataclass(frozen=True, slots=True)
class DifferentialReport:
    """Outcome of a reconstruction differential.

    ``row_match_rate`` is the fraction of rows matching on EVERY cell;
    ``cell_match_rate`` the fraction of individual cells matching;
    ``per_level_row_match_rate`` maps level -> fraction of rows whose four cells
    at that level all match (the diagnostic that isolates the deep-level edge
    case). ``first_mismatches`` holds a few offending rows for inspection.
    """

    n_rows: int
    levels: int
    row_match_rate: float
    cell_match_rate: float
    per_level_row_match_rate: dict[int, float]
    first_mismatch_rows: list[int] = field(default_factory=list)
    first_mismatches: list[dict] = field(default_factory=list)


def differential(
    reconstructed: pd.DataFrame,
    reference: pd.DataFrame,
    levels: int,
    *,
    max_examples: int = 5,
) -> DifferentialReport:
    """Difference ``reconstructed`` against LOBSTER ``reference`` cell-for-cell.

    Both frames must be ``book_columns(levels)`` integer frames with identical
    row counts (they are row-for-row aligned event snapshots). Returns overall
    and per-level match rates plus the first few mismatching rows. Comparison is
    exact (integer equality within ``RECON_CELL_TOLERANCE``).
    """
    cols = book_columns(levels)
    missing_recon = [c for c in cols if c not in reconstructed.columns]
    missing_ref = [c for c in cols if c not in reference.columns]
    if missing_recon or missing_ref:
        raise ValueError(f"book frames missing columns (recon={missing_recon}, ref={missing_ref})")
    if len(reconstructed) != len(reference):
        raise ValueError(
            f"row-count mismatch: reconstructed {len(reconstructed)} vs reference {len(reference)}"
        )

    recon = reconstructed[cols].to_numpy(dtype=np.int64)
    ref = reference[cols].to_numpy(dtype=np.int64)
    n_rows = recon.shape[0]

    # Integer, exact comparison; tolerance is documented to be zero.
    cell_ok = np.abs(recon - ref) <= R.RECON_CELL_TOLERANCE  # (n_rows, 4*levels)
    row_ok = cell_ok.all(axis=1)

    cell_match_rate = float(cell_ok.mean()) if n_rows else float("nan")
    row_match_rate = float(row_ok.mean()) if n_rows else float("nan")

    per_level: dict[int, float] = {}
    for lvl in range(1, levels + 1):
        base = 4 * (lvl - 1)  # ask_px, ask_sz, bid_px, bid_sz block for this level
        level_ok = cell_ok[:, base : base + 4].all(axis=1)
        per_level[lvl] = float(level_ok.mean()) if n_rows else float("nan")

    mismatch_idx = np.flatnonzero(~row_ok)[:max_examples]
    first_mismatches = [
        {
            "row": int(i),
            "reconstructed": {c: int(v) for c, v in zip(cols, recon[i], strict=True)},
            "reference": {c: int(v) for c, v in zip(cols, ref[i], strict=True)},
        }
        for i in mismatch_idx
    ]

    return DifferentialReport(
        n_rows=n_rows,
        levels=levels,
        row_match_rate=row_match_rate,
        cell_match_rate=cell_match_rate,
        per_level_row_match_rate=per_level,
        first_mismatch_rows=[int(i) for i in mismatch_idx],
        first_mismatches=first_mismatches,
    )
