"""The alpha / tolerance / study registry (the project's frozen pre-registration).

This module is the project's pre-registration. Every significance level, every
differential tolerance, the horizon grid, and the *complete* family of studies
that will ever be run on real data are declared here BEFORE the machinery is
built — each with a written justification. Nothing here is widened, and no
study outside ``STUDY_FAMILY`` is run post-hoc (that would inflate the FDR
family). If a value must change, it changes here — deliberately, in one place —
with the justification updated and the FDR family re-closed.

Rationale for each choice is inline; docs/DESIGN.md expands with citations.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

#: Master seed. Every stochastic step (simulator, bootstrap, CV shuffling if
#: any) derives its seed deterministically from this so the whole study is
#: reproducible by one documented command.
MASTER_SEED: int = 20120621  # the LOBSTER sample session date, for mnemonic value


# --------------------------------------------------------------------------- #
# Significance levels
# --------------------------------------------------------------------------- #

#: Family-wise target false-discovery rate for the registered study family.
#: 0.05 is the field-standard FDR level (Benjamini-Hochberg, 1995); we control
#: the FDR — not the FWER — because the studies are related order-flow effects
#: where a few false discoveries among many are tolerable but their *rate* must
#: be bounded. Never widened to manufacture significance.
FDR_ALPHA: float = 0.05

#: Per-test nominal alpha used ONLY for placebo/false-positive-rate reporting
#: (i.e., "on permuted labels, what fraction of tests reject at 0.05?"). We
#: expect the measured rate to sit at ~0.05, validating calibration.
PLACEBO_ALPHA: float = 0.05

#: Confidence level for all reported effect-size intervals. 95% is standard and
#: pairs with FDR_ALPHA = 0.05.
CI_LEVEL: float = 0.95


# --------------------------------------------------------------------------- #
# Differential / numerical tolerances
# --------------------------------------------------------------------------- #

#: Book-reconstruction differential tolerance: prices and sizes are integers, so
#: the comparison is exact — zero mismatched cells on any cell we claim to match.
#: Never loosened.
RECON_CELL_TOLERANCE: int = 0

#: The reconstruction gate is defined by two STRICTER, data-valid checks rather
#: than a single full-day full-book match rate. This redefinition (design decision,
#: see docs/DESIGN.md §D1.1) is forced by a DATA property of the *free*
#: LOBSTER sample, NOT by any reconstructor defect, and it makes the gate more
#: rigorous, not more lenient:
#:
#:   (1) EXACT PER-EVENT SUBMIT-DELTA INVARIANT (primary correctness oracle):
#:       for every visible SUBMIT at a price inside the displayed window,
#:       LOBSTER's reference book size at that price MUST change by exactly the
#:       event size. Target = 100% (RECON_SUBMIT_INVARIANT_TARGET), tolerance 0.
#:       This is a direct, per-event check of our message->book mapping against
#:       LOBSTER's own bookkeeping and is immune to the sample's level limitation.
#:
#:   (2) SEEDED-OPEN PREFIX MATCH: seeding the opening book from orderbook row 0,
#:       the reconstructed book matches LOBSTER row-for-row (all displayed levels)
#:       until the first contamination from an off-window deep-level order.
#:       We report the exact-match prefix length and the best-quote (touch) and
#:       upper-level match rates over the day.
#:
#: WHY the naive full-day, full-depth, message-only match is ~0 and CANNOT be
#: raised honestly: the free LOBSTER "_10" message file is LEVEL-RESTRICTED — it
#: only records events touching the visible top 10 levels. So (a) pre-open resting
#: liquidity never appears in the messages (only in orderbook row 0), and
#: (b) orders promoted from level 11+ into view were never submitted in-window
#: and off-window removals are absent, so the DEEPEST displayed level drifts as
#: intraday price moves. Verified: first divergence is always at the deepest
#: level; the touch and upper levels reconstruct exactly. Recovering full-depth
#: match would require the unrestricted LOBSTER feed (not free) or seeding from —
#: and thus depending on — the orderbook file, which would defeat the whole point
#: of an INDEPENDENT differential. We therefore do NOT set a full-depth target;
#: the full-depth message-only match rate is reported as a documented data-
#: limitation figure (via the differential's per-level report), never target-matched.
RECON_SUBMIT_INVARIANT_TARGET: float = 1.0  # exact; the primary gate

#: Deepest level (1-indexed) at/above which the free sample's level restriction
#: makes full-depth message-only match structurally impossible. The seeded-open
#: prefix and per-level report treat levels 1..RECON_RELIABLE_DEPTH as the
#: well-defined interior we hold to exactness at the open; the deepest levels are
#: reported honestly, never target-matched.
RECON_RELIABLE_DEPTH: int = 5

#: Estimator goldens: our HAC/Newey-West SEs, ICs, and AUC must match
#: statsmodels/scipy references on synthetic data to this relative tolerance.
#: 1e-8 reflects "same algorithm, same answer to numerical precision" — these
#: are goldens against a reference implementation, not empirical comparisons.
GOLDEN_RTOL: float = 1e-8

#: Synthetic-recovery tolerance: the injected ground-truth effect size must lie
#: inside the estimated CI. This is a coverage check, not a point-match — the
#: whole point is that the interval, not the point estimate, is honest. We also
#: require measured coverage of the CIs to be >= this over repeated draws.
RECOVERY_CI_COVERAGE_MIN: float = 0.90  # 95% nominal CIs may under-cover slightly


# --------------------------------------------------------------------------- #
# Horizon grid (pre-registered; shared by forward-looking studies)
# --------------------------------------------------------------------------- #

#: Forward horizons in EVENTS for forward-return studies. Order-flow effects
#: live at short horizons; the grid spans immediate to a few-hundred events to
#: trace the decay curve documented in the literature (Cont et al.). Fixed in
#: advance so the horizon axis is not cherry-picked.
HORIZON_GRID_EVENTS: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200)

#: CV configuration: number of walk-forward folds, and the embargo (in events)
#: applied on each side of the test window to kill leakage from label overlap.
#: The embargo must be >= the largest label horizon used in a study; we set it
#: to the grid max so every fold is safe by construction.
CV_N_FOLDS: int = 5
CV_EMBARGO_EVENTS: int = max(HORIZON_GRID_EVENTS)

#: Block length (events) for the stationary/circular block bootstrap used for
#: CIs on autocorrelated series. Chosen ~ the scale over which order-flow
#: autocorrelation is non-negligible; sensitivity is reported as robustness.
BOOTSTRAP_BLOCK_EVENTS: int = 100
BOOTSTRAP_N_RESAMPLES: int = 2000


# --------------------------------------------------------------------------- #
# The registered study family (the ONLY studies run on real data)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Study:
    """One pre-registered hypothesis. ``n_tests`` counts the individual
    estimates it contributes to the FDR family (e.g., one per horizon), so the
    family size is known before any data is touched."""

    study_id: str
    title: str
    hypothesis: str
    primary_metric: str  # "coef"/"R2"/"AUC"/"autocorr"/...
    horizons: tuple = ()  # empty for contemporaneous / single-shot
    justification: str = ""

    @property
    def n_tests(self) -> int:
        return max(1, len(self.horizons))


STUDY_FAMILY: tuple[Study, ...] = (
    Study(
        study_id="ofi_contemporaneous",
        title="OFI vs contemporaneous mid-price change (linearity)",
        hypothesis="Order-flow imbalance over an interval explains the "
        "contemporaneous mid-price change with a near-linear, positive slope.",
        primary_metric="coef",  # the HAC slope is the FDR-entering test; R^2 is descriptive
        justification="Canonical Cont-Kukanov-Stoikov result: OFI has a strong, "
        "near-linear contemporaneous relation to price change, robust across "
        "stocks. Serves as the replication anchor and a sanity check on the "
        "whole pipeline before forward-horizon claims. The FDR test is the HAC "
        "slope (valid analytic p); R^2 is reported as the descriptive fit.",
    ),
    Study(
        study_id="ofi_forward",
        title="OFI vs forward mid-price change by horizon",
        # Pre-registered hypothesis (frozen, NOT edited post-hoc): we expected a
        # decaying forward association. The realized out-of-sample coefficient is
        # instead positive and RISING with horizon (DESIGN §D2.1/§D2.6) — a
        # pre-registration-vs-result divergence we report rather than hide.
        hypothesis="OFI has predictive association with FORWARD mid-price change "
        "that decays with horizon.",
        primary_metric="coef",
        horizons=HORIZON_GRID_EVENTS,
        justification="Extends the contemporaneous result forward under strict "
        "point-in-time / purged-CV discipline (the direct out-of-sample "
        "coefficient per horizon is the headline forward figure). One test per "
        "horizon -> counted in the FDR family.",
    ),
    Study(
        study_id="queue_imbalance_next_move",
        title="Best-quote queue imbalance predicts the next mid move",
        hypothesis="Queue imbalance at the best bid/ask predicts the sign of the "
        "next mid-price move (classification AUC > 0.5), strongest in large-tick "
        "names.",
        primary_metric="AUC",
        justification="Gould-Bonart queue-imbalance result. AUC with block-"
        "bootstrap CI; tick-size commentary in the note (effect strongest where "
        "the spread is pinned to one tick).",
    ),
    Study(
        study_id="sign_autocorrelation",
        title="Trade-sign autocorrelation (long memory)",
        hypothesis="Signed trade initiations are positively autocorrelated and "
        "decay slowly (long memory), per the order-splitting literature.",
        primary_metric="autocorr",
        horizons=HORIZON_GRID_EVENTS,
        justification="Bouchaud et al. long-memory-of-order-flow result. Tick-"
        "rule sign inference with documented limits; block-bootstrap CIs on the "
        "autocorrelation at each lag.",
    ),
    Study(
        study_id="impact_curve",
        title="Concave price-impact curve of signed volume",
        hypothesis="Mid-price change is a concave (sub-linear) increasing "
        "function of signed traded volume over a short window.",
        primary_metric="coef",
        justification="Square-root/concave impact is a canonical stylized fact. "
        "A single fitted concavity coefficient with CI; deliberately simple — no "
        "tradability implication is drawn (see the note's cost caveat).",
    ),
)


def family_size() -> int:
    """Total number of individual tests in the registered family (the m used by
    Benjamini-Hochberg). Computed from the frozen family so it cannot drift."""
    return sum(s.n_tests for s in STUDY_FAMILY)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Which LOBSTER sample the studies run on (defaults per MISSION)."""

    tickers: tuple[str, ...] = ("AAPL", "AMZN", "GOOG", "INTC", "MSFT")
    session: str = "2012-06-21"
    level: int = 10
    price_scale: int = 10000  # LOBSTER prices are dollars * 10000


DATA = DataConfig()
