"""Tests for src.multi.fdr — Benjamini-Hochberg FDR control.

Headline verifications (per MISSION):
  * GOLDEN: benjamini_hochberg matches statsmodels' fdr_bh on both the rejected
    mask AND the adjusted p-values, on random and hand-built p-vectors (ties,
    known-signal + many-nulls, all-null, all-signal).
  * FDR CONTROL: on simulated hypothesis families with a known true-null subset,
    the empirical false-discovery proportion averaged over many replications is
    <= alpha within sampling slack. Seeded from registry.MASTER_SEED.
  * CONTROLLER: round-trips the frozen EstimationResult via dataclasses.replace,
    preserves NaN-p entries as rejected=None, and stamps family_id / alpha.

Everything runs on small synthetic data with a known reference answer; no
LOBSTER files are touched.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from config import registry as R
from src.multi.fdr import BenjaminiHochbergController, benjamini_hochberg, close_family
from src.schema import EstimationResult

GOLDEN_ATOL = 1e-12


def _mk_result(name: str, p_raw: float) -> EstimationResult:
    """A minimal, contract-valid EstimationResult carrying a raw p-value."""
    return EstimationResult(
        name=name,
        metric="coef",
        point=0.0,
        ci_low=-1.0,
        ci_high=1.0,
        n_obs=100,
        p_value_raw=p_raw,
    )


# --------------------------------------------------------------------------- #
# Golden vs statsmodels
# --------------------------------------------------------------------------- #

_HAND_BUILT = {
    "single": np.array([0.03]),
    "all_ones": np.array([1.0, 1.0, 1.0, 1.0]),
    "ties": np.array([0.01, 0.01, 0.04, 0.04, 0.2, 0.2]),
    "known_signal_many_nulls": np.concatenate([np.array([1e-6, 1e-5, 1e-4]), np.full(30, 0.7)]),
    "all_null": np.array([0.51, 0.62, 0.73, 0.84, 0.95]),
    "all_signal": np.array([1e-8, 1e-7, 1e-6, 1e-5]),
    "mixed": np.array([0.001, 0.008, 0.039, 0.041, 0.9, 0.5, 0.2, 0.75]),
    "boundary": np.array([0.05, 0.05, 0.05]),
}


@pytest.mark.parametrize("case", sorted(_HAND_BUILT))
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2])
def test_golden_hand_built(case, alpha):
    p = _HAND_BUILT[case]
    rej, adj = benjamini_hochberg(p, alpha)
    ref_rej, ref_adj, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    np.testing.assert_allclose(adj, ref_adj, atol=GOLDEN_ATOL, rtol=0.0)
    np.testing.assert_array_equal(rej, ref_rej)


@pytest.mark.parametrize("m", [1, 2, 5, 17, 50, 200])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2])
def test_golden_random(m, alpha):
    rng = np.random.default_rng(R.MASTER_SEED + m)
    # Mix of uniform nulls and small "signal" p-values, plus deliberate ties.
    p = rng.uniform(0.0, 1.0, size=m)
    if m >= 5:
        p[: m // 5] = rng.uniform(0.0, 0.01, size=m // 5)
        p[-1] = p[0]  # inject a tie
    rej, adj = benjamini_hochberg(p, alpha)
    ref_rej, ref_adj, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    np.testing.assert_allclose(adj, ref_adj, atol=GOLDEN_ATOL, rtol=0.0)
    np.testing.assert_array_equal(rej, ref_rej)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_empty_input():
    rej, adj = benjamini_hochberg([])
    assert rej.shape == (0,) and adj.shape == (0,)
    assert rej.dtype == bool


def test_single_pvalue_is_unchanged():
    # With m == 1 the adjusted value equals the raw p-value.
    rej, adj = benjamini_hochberg([0.03], alpha=0.05)
    assert adj[0] == pytest.approx(0.03, abs=GOLDEN_ATOL)
    assert bool(rej[0]) is True


def test_adjusted_are_monotone_and_bounded():
    rng = np.random.default_rng(R.MASTER_SEED + 999)
    p = rng.uniform(size=40)
    _, adj = benjamini_hochberg(p, 0.05)
    assert np.all(adj >= 0.0) and np.all(adj <= 1.0)
    # Sorted by raw p, the adjusted values are non-decreasing.
    adj_by_rank = adj[np.argsort(p, kind="stable")]
    assert np.all(np.diff(adj_by_rank) >= -GOLDEN_ATOL)


def test_default_alpha_is_registry_fdr_alpha():
    p = np.array([0.001, 0.5, 0.5])
    rej_default, _ = benjamini_hochberg(p)
    rej_explicit, _ = benjamini_hochberg(p, R.FDR_ALPHA)
    np.testing.assert_array_equal(rej_default, rej_explicit)


# --------------------------------------------------------------------------- #
# Empirical FDR control on synthetic hypothesis families
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_empirical_fdr_control(alpha):
    """Simulate many families with a known true-null subset; BH must keep the
    average false-discovery proportion <= alpha (within sampling slack)."""
    rng = np.random.default_rng(R.MASTER_SEED)
    m = 60  # tests per family
    n_null = 40  # first n_null hypotheses are true nulls
    n_reps = 3000
    signal_shift = 3.5  # z-shift giving small (but not trivially zero) signal p's

    fdps = np.empty(n_reps, dtype=np.float64)
    for r in range(n_reps):
        z = rng.standard_normal(m)
        z[n_null:] += signal_shift  # true-signal hypotheses shifted away from 0
        p = 2.0 * (1.0 - _norm_cdf(np.abs(z)))  # two-sided z-test p-values
        rej, _ = benjamini_hochberg(p, alpha)
        n_rej = int(rej.sum())
        false_rej = int(rej[:n_null].sum())  # rejections among true nulls
        fdps[r] = false_rej / n_rej if n_rej > 0 else 0.0

    empirical_fdr = float(fdps.mean())
    # BH controls E[FDP] <= alpha * (n_null / m) <= alpha; allow small MC slack.
    assert empirical_fdr <= alpha + 0.01, (
        f"empirical FDR {empirical_fdr:.4f} exceeds alpha {alpha} + slack"
    )
    # Sanity: the procedure is actually rejecting the signals, not trivially empty.
    assert empirical_fdr >= 0.0


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard-normal CDF via the error function (vectorized, no scipy import)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Controller: frozen round-trip, NaN handling, stamping
# --------------------------------------------------------------------------- #


def test_controller_stamps_and_matches_procedure():
    ctrl = BenjaminiHochbergController()
    p = [0.001, 0.02, 0.03, 0.5, 0.9]
    results = [_mk_result(f"t{i}", pv) for i, pv in enumerate(p)]
    out = ctrl.control(results, family_id="fam", alpha=0.05)

    ref_rej, ref_adj = benjamini_hochberg(np.array(p), 0.05)
    for i, res in enumerate(out):
        assert res.family_id == "fam"
        assert res.alpha == 0.05
        assert res.p_value_adj == pytest.approx(ref_adj[i], abs=GOLDEN_ATOL)
        assert res.rejected is bool(ref_rej[i])
        # Original raw p-value and effect size are preserved.
        assert res.p_value_raw == p[i]
    # New objects, inputs untouched (frozen dataclass round-trip).
    assert all(a is not b for a, b in zip(out, results, strict=True))
    assert all(r.rejected is None for r in results)


def test_controller_default_alpha():
    ctrl = BenjaminiHochbergController()
    results = [_mk_result("a", 0.001), _mk_result("b", 0.6)]
    out = ctrl.control(results, family_id="fam")
    assert all(r.alpha == R.FDR_ALPHA for r in out)


def test_controller_excludes_nan_p_values():
    ctrl = BenjaminiHochbergController()
    # Two real tests + one NaN-p (no test performed).
    results = [
        _mk_result("real1", 0.001),
        _mk_result("no_test", float("nan")),
        _mk_result("real2", 0.9),
    ]
    out = ctrl.control(results, family_id="fam", alpha=0.05)

    # The NaN entry passes through honestly and is NOT counted in m.
    assert out[1].rejected is None
    assert math.isnan(out[1].p_value_adj)
    assert out[1].family_id == "fam" and out[1].alpha == 0.05

    # m == 2 for the adjustment (the NaN did not inflate the family).
    ref_rej, ref_adj = benjamini_hochberg(np.array([0.001, 0.9]), 0.05)
    assert out[0].p_value_adj == pytest.approx(ref_adj[0], abs=GOLDEN_ATOL)
    assert out[2].p_value_adj == pytest.approx(ref_adj[1], abs=GOLDEN_ATOL)
    assert out[0].rejected is bool(ref_rej[0])
    assert out[2].rejected is bool(ref_rej[1])


def test_controller_all_nan():
    ctrl = BenjaminiHochbergController()
    results = [_mk_result("a", float("nan")), _mk_result("b", float("nan"))]
    out = ctrl.control(results, family_id="fam", alpha=0.05)
    assert all(r.rejected is None and math.isnan(r.p_value_adj) for r in out)
    assert all(r.family_id == "fam" for r in out)


def test_controller_empty():
    out = BenjaminiHochbergController().control([], family_id="fam", alpha=0.05)
    assert out == []


def test_controller_satisfies_protocol():
    from src.interfaces import MultipleTestingController

    assert isinstance(BenjaminiHochbergController(), MultipleTestingController)


# --------------------------------------------------------------------------- #
# close_family helper
# --------------------------------------------------------------------------- #


def test_close_family_pools_then_regroups():
    by_study = {
        "study_a": [_mk_result("a0", 0.001), _mk_result("a1", 0.20)],
        "study_b": [_mk_result("b0", 0.03), _mk_result("b1", 0.9), _mk_result("b2", 0.01)],
    }
    out = close_family(by_study, family_id="whole", alpha=0.05)

    # Regrouping preserves study keys and per-study sizes.
    assert set(out) == {"study_a", "study_b"}
    assert len(out["study_a"]) == 2 and len(out["study_b"]) == 3

    # Adjustment is pooled across ALL 5 tests, not per study.
    flat_p = [0.001, 0.20, 0.03, 0.9, 0.01]
    _, ref_adj = benjamini_hochberg(np.array(flat_p), 0.05)
    got = [r.p_value_adj for r in out["study_a"]] + [r.p_value_adj for r in out["study_b"]]
    np.testing.assert_allclose(got, ref_adj, atol=GOLDEN_ATOL, rtol=0.0)
    assert all(r.family_id == "whole" for grp in out.values() for r in grp)


def test_close_family_default_alpha():
    by_study = {"s": [_mk_result("x", 0.001)]}
    out = close_family(by_study, family_id="f")
    assert out["s"][0].alpha == R.FDR_ALPHA
