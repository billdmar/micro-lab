"""Benjamini-Hochberg false-discovery-rate control across the registered family.

The registered study family (:data:`config.registry.STUDY_FAMILY`) contributes
many individual tests — one per horizon for the forward-looking studies — so the
p-values must be adjusted together as a single family or the FDR is not
controlled. This module owns that adjustment: the standard BH step-up procedure
(Benjamini & Hochberg, 1995) and a :class:`interfaces.MultipleTestingController`
that stamps each :class:`schema.EstimationResult` with its adjusted p-value,
``alpha``, ``family_id``, and reject decision.

Conventions, chosen to reproduce the reference library exactly:

* **Adjusted p-values ("q-values")** are the monotone BH values reported by
  ``statsmodels.stats.multitest.multipletests(..., method="fdr_bh")``: sort the
  ``m`` raw p-values ascending, form ``m * p_(k) / k`` at each rank ``k``, take
  the running minimum from the largest rank down (enforcing monotonicity), clip
  to ``<= 1``, and return in the original order. A hypothesis is rejected iff its
  adjusted p-value ``<= alpha`` — equivalent to the step-up rejection threshold.
* **Ties** in the raw p-values are handled by ``argsort``'s stable order; the
  running-minimum step makes the adjusted values tie-invariant (equal raw
  p-values receive equal adjusted values), matching statsmodels.
* **Honest unknowns**: a result whose ``p_value_raw`` is NaN carries no test
  (the estimate was undefined — e.g. a Fisher-z CI at ``|r| == 1``). Such
  results are excluded from the ``m`` used by BH and passed through with
  ``rejected=None`` and ``p_value_adj=NaN`` rather than being counted as a
  non-rejection, which would silently inflate ``m`` and bias the adjustment.

Nothing here reads the wall clock; the procedure is a pure function of its
inputs and is fully deterministic.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np

from config import registry as R
from src.schema import EstimationResult

# --------------------------------------------------------------------------- #
# The Benjamini-Hochberg step-up procedure
# --------------------------------------------------------------------------- #


def benjamini_hochberg(
    p_values: Sequence[float] | np.ndarray, alpha: float = R.FDR_ALPHA
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control at level ``alpha``.

    Returns ``(rejected, adjusted)`` where ``adjusted`` are the BH-adjusted
    p-values (q-values) in the ORIGINAL input order and ``rejected`` is the
    boolean mask ``adjusted <= alpha``. Both arrays have the same length as the
    input. Empty input returns two empty arrays.

    The adjusted value at rank ``i`` (1-indexed, ascending) is
    ``min_{k >= i} ( m * p_(k) / k )`` clipped to ``[0, 1]``; this monotone,
    tie-invariant form is exactly ``statsmodels`` ``method="fdr_bh"``.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.shape[0]
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)

    order = np.argsort(p, kind="stable")  # ascending; stable so ties keep order
    p_sorted = p[order]

    ranks = np.arange(1, m + 1, dtype=np.float64)
    # Adjusted p-values (q-values): raw BH factor m*p_(k)/k at each ascending
    # rank, made monotone by the running minimum from the largest rank downward,
    # then clipped to [0, 1].
    factor = m * p_sorted / ranks
    q_sorted = np.minimum.accumulate(factor[::-1])[::-1]
    np.clip(q_sorted, 0.0, 1.0, out=q_sorted)

    # Reject via the BH step-up threshold rather than ``q <= alpha`` directly:
    # find the largest rank k with p_(k) <= (k/m)*alpha and reject every
    # hypothesis up to it. This is the canonical BH decision and — unlike the
    # algebraically-equivalent ``q_sorted <= alpha`` — is immune to the
    # floating-point drift in m*p/k at exact boundaries (e.g. 6*0.2/6 rounds to
    # just above 0.2). It reproduces statsmodels' fdr_bh reject mask exactly.
    below = p_sorted <= (ranks / m) * alpha
    rejected_sorted = np.zeros(m, dtype=bool)
    if below.any():
        kmax = int(np.max(np.nonzero(below)))
        rejected_sorted[: kmax + 1] = True

    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = q_sorted
    rejected = np.empty(m, dtype=bool)
    rejected[order] = rejected_sorted
    return rejected, adjusted


# --------------------------------------------------------------------------- #
# The MultipleTestingController implementation
# --------------------------------------------------------------------------- #


class BenjaminiHochbergController:
    """:class:`interfaces.MultipleTestingController` using Benjamini-Hochberg.

    ``control`` reads each result's ``p_value_raw``, runs BH across the whole
    list as one family, and returns NEW :class:`schema.EstimationResult` objects
    (the dataclass is frozen, so we construct copies via
    :func:`dataclasses.replace`) with ``p_value_adj``, ``alpha``, ``rejected``,
    and ``family_id`` filled in.

    Results whose ``p_value_raw`` is NaN carry no test and are excluded from the
    BH computation; they pass through with ``rejected=None`` and
    ``p_value_adj=NaN`` (only ``alpha`` and ``family_id`` are stamped), so they
    never inflate the family size ``m``.
    """

    def control(
        self,
        results: Sequence[EstimationResult],
        *,
        family_id: str,
        alpha: float = R.FDR_ALPHA,
    ) -> list[EstimationResult]:
        raw = np.array([r.p_value_raw for r in results], dtype=np.float64)
        tested = ~np.isnan(raw)

        # Run BH only over the results that actually carry a p-value.
        rej_tested, adj_tested = benjamini_hochberg(raw[tested], alpha)

        adjusted = np.full(raw.shape[0], np.nan, dtype=np.float64)
        adjusted[tested] = adj_tested
        rejected = np.zeros(raw.shape[0], dtype=bool)
        rejected[tested] = rej_tested

        out: list[EstimationResult] = []
        for i, res in enumerate(results):
            if tested[i]:
                out.append(
                    dataclasses.replace(
                        res,
                        family_id=family_id,
                        p_value_adj=float(adjusted[i]),
                        alpha=alpha,
                        rejected=bool(rejected[i]),
                    )
                )
            else:
                # No test performed — honest unknown, not a non-rejection.
                out.append(
                    dataclasses.replace(
                        res,
                        family_id=family_id,
                        p_value_adj=float("nan"),
                        alpha=alpha,
                        rejected=None,
                    )
                )
        return out


# --------------------------------------------------------------------------- #
# Convenience: close the whole registered family at once
# --------------------------------------------------------------------------- #


def close_family(
    results_by_study: Mapping[str, Sequence[EstimationResult]],
    family_id: str,
    alpha: float = R.FDR_ALPHA,
) -> dict[str, list[EstimationResult]]:
    """Close FDR control over an entire family passed study-by-study.

    ``results_by_study`` maps each ``study_id`` to that study's list of
    :class:`schema.EstimationResult` (e.g. one per horizon). All of them are
    pooled into ONE Benjamini-Hochberg family — that pooling is the whole point,
    since the FDR is only controlled across the family as a unit — adjusted
    together, then regrouped by study for the caller. Study insertion order is
    preserved. Useful for the study runner, which holds the registered family
    keyed by study id.
    """
    study_ids = list(results_by_study.keys())
    flat: list[EstimationResult] = []
    counts: list[int] = []
    for sid in study_ids:
        group = list(results_by_study[sid])
        flat.extend(group)
        counts.append(len(group))

    controlled = BenjaminiHochbergController().control(flat, family_id=family_id, alpha=alpha)

    regrouped: dict[str, list[EstimationResult]] = {}
    pos = 0
    for sid, k in zip(study_ids, counts, strict=True):
        regrouped[sid] = controlled[pos : pos + k]
        pos += k
    return regrouped
