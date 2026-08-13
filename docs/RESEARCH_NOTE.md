# Order Flow and Short-Horizon Price Formation on NASDAQ: A Verified Replication

**micro-lab research note.** A replication of canonical limit-order-book microstructure
results — order-flow imbalance, queue-imbalance prediction, trade-sign long memory, and
price impact — on LOBSTER NASDAQ sample data, computed through a statistics pipeline that
was **itself verified** before it touched real data.

> **Claims discipline.** Everything below is an effect size with a confidence interval,
> under false-discovery-rate control. There are **no** PnL, strategy, tradability, or
> forecasting claims anywhere in this project. The measured effects are real but small and,
> at these horizons, cost-dominated — §6 explains why plainly. This note is descriptive
> microstructure research, not investment advice.

All numbers are reproducible from one command (`python -m src.studies.runner`) and the
figures from another (`python scripts/make_figures.py`); both are seeded and deterministic.

---

## 1. Verification of the machinery (the part most projects skip)

The differentiating claim of this project is not any single effect size — those are
textbook. It is that **the pipeline was proven to work before it was trusted**: it recovers
signals that are really there, stays silent on noise, catches its own leaks, and matches
reference implementations to numerical precision. This section comes first deliberately.

**Reconstruction correctness oracle.** We rebuild the limit order book from the LOBSTER
message stream and check it against LOBSTER's own reference orderbook file. The correctness
measure is the **exact per-event price-level transition invariant**: for every visible
event (submit, cancel, delete, execution) at a price inside the displayed window, the
reference book's size at that price must change by exactly the event's signed size. This
holds at **100.0000% — 0 violations across 1,673,089 transitions** on all five tickers
(tolerance 0, integer arithmetic). This is a direct, per-event check of our message→book
semantics against the exchange's own bookkeeping.

We report this invariant rather than a full-day, full-depth row-match rate for an honest
reason (see §7 and `docs/DESIGN.md §D1.1`): the *free* LOBSTER level-10 sample is
level-restricted — it records only events touching the visible top 10 levels — so pre-open
resting liquidity and off-window deep-level orders are absent from the messages, and a naive
message-only full-book reconstruction diverges at the deepest level as price drifts. That is
a property of the free sample, **not** a defect of the reconstructor (whose per-event
semantics are exactly correct, as the invariant shows). We do not loosen a tolerance to
paper over it; we report the invariant that is valid and exact.

**Synthetic ground-truth recovery.** A configurable order-flow simulator
(`src/sim/simulator.py`) injects a known relationship — a coefficient β linking an exogenous
signed-flow driver to future mid-price drift — and the pipeline must recover it. Across a
grid of signal strengths (40 seeds each, through the real HAC estimator), the recovered
coefficient is **within 0.05% of the injected β** (no attenuation), the 95% confidence
interval covers β at **97.5%**, and statistical power rises from **2.5% at β = 0 to 100%**
at the tested effect sizes (Figure 3). The simulator's design — an *exogenous* driver and a
*continuous* efficient mid — was chosen precisely so that recovery is unbiased; an earlier
design that drove price with the book-derived OFI suffered errors-in-variables attenuation
(~45%) and was discarded (`DESIGN §D1.2`). A biased simulator would make this gate lie.

**Placebo nulls.** With β = 0 (a true null), the fraction of runs whose 95% CI excludes zero
is **2.5%** — at or below the nominal 5%. The pipeline does not manufacture significance from
noise.

**Leakage detection.** Point-in-time discipline is structural: every feature carries a
`FeatureSpec` with an information-timestamp offset, and the CV/leakage machinery
(`src/cv/`) refuses look-ahead. Deliberately constructed leaks — a label window overlapping
the train set, a future-timestamped feature (`info_offset > 0`), a train/test boundary
without embargo — are each caught by the detectors, verified in
`tests/test_cv_leakage.py` and `tests/test_integration_g1.py`.

**Estimator goldens.** The HAC/Newey–West standard errors match `statsmodels`, the AUC
matches `scikit-learn`, the information coefficients match `scipy`, and the Bartlett
autocorrelation interval matches `statsmodels`' ACF band — all to a relative tolerance of
**1e-8** on synthetic data where the answer is known.

**Coverage & determinism.** ~97% line coverage on the CI suite (gate: ≥ 90%); every
published statistic reproduces bit-for-bit on a reseeded rerun.

*The one-line version:* **I verified my
pipeline could find planted signals and stay silent on noise before I let it near real data.**

---

## 2. Data and methods

**Data.** LOBSTER reconstructs NASDAQ limit order books from TotalView-ITCH. We use its free
academic sample — session **2012-06-21**, message + orderbook file pairs at depth level 10 —
for **AAPL, AMZN, GOOG, INTC, MSFT**, a working set of **2,110,860 events** (AAPL 400,391 ·
AMZN 269,748 · GOOG 147,916 · INTC 624,040 · MSFT 668,765). Raw files are downloaded by
`scripts/get_data.sh` and never redistributed (academic-use terms); only small derived
fixtures are committed. LOBSTER units are kept exactly: integer 1/10000-dollar prices,
integer shares, float seconds after midnight (`DESIGN §D0.1–D0.2`).

**Features** (`src/features/engine.py`) are computed from LOBSTER's reference orderbook (exact
at every displayed level), which sidesteps the level-restriction limitation for feature
values. Order-flow imbalance follows Cont–Kukanov–Stoikov: a per-event contribution from
best-quote price/size changes. Queue imbalance is `(bid_sz₁ − ask_sz₁)/(bid_sz₁ + ask_sz₁)`.
Trade sign is inferred by the tick rule (Lee–Ready), and its accuracy is *validated* against
LOBSTER's true resting-order side. Empty-touch and warm-up values are `NaN`, never fabricated.

**Estimation** (`src/stats/estimators.py`). OLS with Newey–West/HAC standard errors;
logistic regression; information coefficients (Fisher-z CI); ROC AUC (paired bootstrap CI);
the Bartlett large-sample autocorrelation interval; a stationary/circular block bootstrap.

**Design.** OFI regressions use non-overlapping event buckets (interval OFI vs interval
mid-change), following the canonical construction. The forward study runs through a **purged,
embargoed walk-forward** splitter (5 folds, embargo 200 events = the largest horizon), and
reports the **out-of-sample** relationship. All five studies and every threshold — FDR level
0.05 (Benjamini–Hochberg), 95% CIs, the horizon grid {1, 5, 10, 20, 50, 100, 200} events —
are **pre-registered** in `config/registry.py` before any data was touched, a family of
**17 individual tests**. Multiple testing is controlled by Benjamini–Hochberg across the
whole family (`src/multi/fdr.py`, golden-matched to `statsmodels`).

---

## 3. Results

All effects are pooled across the five symbols, reported with 95% CIs, and FDR-adjusted over
the 17-test family. **15 of 17 tests are FDR-significant at α = 0.05** (the two long-lag
trade-sign autocorrelations are the honest non-rejections — the memory has decayed to noise).

Each study's FDR-entering test carries an analytic or genuine-bootstrap p-value; the
OFI-forward coefficients are the **direct out-of-sample** OFI→forward-return slopes (return per
unit OFI, hence small in magnitude) fit only on held-out folds under a per-symbol purged
walk-forward split.

| Study | Metric | Estimate | 95% CI | FDR-adj p | Significant |
|---|---|---:|---|---:|:---:|
| OFI → contemporaneous mid | slope | **5.8e-9** | [5.4e-9, 6.2e-9] | 1.2e-192 | ✓ |
| OFI → forward mid, h=1 (OOS) | coef | **1.3e-10** | [1.0e-10, 1.5e-10] | 2.0e-22 | ✓ |
| OFI → forward mid, h=5 (OOS) | coef | 5.1e-10 | [4.4e-10, 5.8e-10] | 5.8e-46 | ✓ |
| OFI → forward mid, h=10 (OOS) | coef | 7.2e-10 | [6.3e-10, 8.2e-10] | 2.0e-49 | ✓ |
| OFI → forward mid, h=20 (OOS) | coef | 9.2e-10 | [8.0e-10, 1.0e-9] | 2.8e-48 | ✓ |
| OFI → forward mid, h=50 (OOS) | coef | 1.4e-9 | [1.3e-9, 1.6e-9] | 1.6e-51 | ✓ |
| OFI → forward mid, h=100 (OOS) | coef | **2.1e-9** | [1.9e-9, 2.3e-9] | 3.2e-77 | ✓ |
| OFI → forward mid, h=200 (OOS) | coef | 1.7e-9 | [1.5e-9, 2.0e-9] | 1.5e-52 | ✓ |
| Queue imbalance → next move | AUC | **0.637** | [0.617, 0.658] | 6.1e-4 | ✓ |
| Trade-sign autocorr, lag 1 | ρ | **0.764** | [0.758, 0.769] | ~0 | ✓ |
| Trade-sign autocorr, lag 5 | ρ | 0.515 | [0.503, 0.527] | ~0 | ✓ |
| Trade-sign autocorr, lag 10 | ρ | 0.385 | [0.370, 0.400] | ~0 | ✓ |
| Trade-sign autocorr, lag 20 | ρ | 0.249 | [0.231, 0.266] | 2.2e-165 | ✓ |
| Trade-sign autocorr, lag 50 | ρ | 0.092 | [0.073, 0.112] | 4.5e-20 | ✓ |
| Trade-sign autocorr, lag 100 | ρ | 0.017 | [−0.003, 0.037] | 9.8e-2 | **✗** |
| Trade-sign autocorr, lag 200 | ρ | −0.013 | [−0.033, 0.007] | 2.0e-1 | **✗** |
| Price impact (log-log) | exponent | **0.026** | [0.005, 0.046] | 1.6e-2 | ✓ |

**OFI and contemporaneous price** (Figure 2). Order-flow imbalance explains contemporaneous
mid-price change with a positive HAC slope (5.8e-9, essentially p ≈ 0) and **R² = 0.150**
[0.120, 0.186] — the Cont–Kukanov–Stoikov near-linear result, replicated. (We register the
slope as the significance test because R² is bounded and one-sided under the null; R² is
reported as the descriptive strength-of-fit with its own bootstrap CI.)

**OFI and forward price** (Figure 1). Out-of-sample, under a strictly per-symbol purged
walk-forward split, the **direct** OFI→forward-return coefficient is positive and
FDR-significant at **every** horizon, rising from 1.3e-10 at h=1 to ~2.1e-9 by h=100 as the
coefficient accumulates the price response over a longer forward window. (This is the corrected
estimator: an earlier version reported the slope of realized-return on the fitted prediction — a
calibration statistic ≈1 by construction, not the coefficient — which spuriously showed a
negative h=1; the direct coefficient is positive throughout.)

**Queue imbalance and the next move.** Queue imbalance predicts the sign of the next mid move
with AUC = 0.637 pooled (block-bootstrap CI [0.617, 0.658], genuine bootstrap p = 6.1e-4) —
well above 0.5, and (§5) far stronger in large-tick names.

**Trade-sign long memory** (Figure 5). Signed trade initiations are strongly, slowly-decaying
autocorrelated: ρ falls from 0.764 at lag 1 to ~0.017 at lag 100, becoming statistically
indistinguishable from zero by lag 100–200 — the two non-rejections in the family, and honest
ones (the memory has genuinely decayed). Autocorrelations are computed per symbol and combined
by a trade-count-weighted average, so no cross-symbol boundary pairs contaminate the estimate.

**Price impact.** Mid-price change is a concave (sub-linear) function of signed volume: the
log-log exponent, fit on volume-binned means to avoid selecting on the dependent variable, is
0.026 [0.005, 0.046] — strongly concave (well below the linear-impact value of 1), consistent
with the square-root-impact stylized fact. **No tradability implication is drawn** (§6).

---

## 4. Comparison to the published literature

- **Cont, Kukanov & Stoikov (2014)** established that OFI has a strong, near-linear relation
  to contemporaneous price change, robust across stocks and timescales. Our contemporaneous
  R² = 0.150 with a positive linear HAC slope reproduces this; the positive out-of-sample
  forward coefficient at every horizon extends it forward under point-in-time discipline.
- **Gould & Bonart (2016)** showed best-quote queue imbalance predicts the next mid-price
  move, strongest in large-tick stocks. Our pooled AUC = 0.637 and the per-symbol split (§5)
  reproduce both the effect and its tick-size dependence.
- **Bouchaud et al.** documented long memory in signed order flow, driven by order splitting.
  Our slowly-decaying trade-sign autocorrelation (0.764 → ~0 over ~100–200 trades) matches the
  qualitative shape.

These are mechanism-backed, repeatedly-replicated results; reproducing them honestly — with
CIs, FDR control, and point-in-time discipline — is the exercise, and it inoculates the
project against the "your backtest is overfit" critique because **there is no backtest**.

---

## 5. The tick-size signature (per-symbol robustness)

Splitting the queue-imbalance study by symbol reproduces the canonical **large-tick effect**
(Figure 4):

| Symbol | Approx. price | OFI R² [95% CI] | Queue-imbalance AUC [95% CI] |
|---|---:|---|---|
| AAPL | ~$586 | 0.271 [0.205, 0.359] | 0.556 [0.547, 0.566] |
| GOOG | ~$580 | 0.243 [0.178, 0.328] | 0.578 [0.560, 0.596] |
| AMZN | ~$224 | 0.110 [0.076, 0.307] | 0.614 [0.600, 0.628] |
| INTC | ~$28 | 0.287 [0.221, 0.362] | **0.907** [0.873, 0.937] |
| MSFT | ~$31 | 0.356 [0.289, 0.426] | **0.923** [0.896, 0.945] |

Queue imbalance is far more predictive in large-tick names (INTC, MSFT — low price, so the
spread is pinned to a single tick and the queue is informative) than in high-price,
small-relative-tick names (AAPL, GOOG). This is exactly Gould–Bonart's finding. OFI R² is
positive with its CI above zero for every symbol, so the effect is not driven by one name.

---

## 6. Why these effects are not money (the transaction-cost caveat)

This section is mandatory and load-bearing. The effects above are **real and statistically
strong**, yet they carry **no tradability claim**, for reasons that are structural, not
coy:

1. **The horizons are tiny and the per-event edge is minuscule.** The forward OFI coefficient
   is significant across the grid but its magnitude is a few times 1e-10 to 1e-9 in return per
   unit of OFI — a sub-basis-point mid move per typical bucket, at event horizons that are
   sub-second at these tickers' rates. Capturing it requires acting inside the very order flow
   being measured, at latencies this project does not model.
2. **Costs dominate the edge.** Any attempt to trade these signals pays the **bid-ask spread**
   (at least one tick — and it is precisely the large-tick names, where the queue signal is
   strongest, that have the *widest* relative spread), **exchange/clearing fees**, and
   **market impact** — the very concave impact we measured in §3. A predictive R² of 0.15 on
   mid-price change, or an AUC of 0.64, is an *association*, not an expected return net of
   these frictions.
3. **Adverse selection.** Queue imbalance predicts the next move partly *because* informed
   flow is on one side; a naive taker of that signal is selected against by exactly the flow
   that makes it predictive.
4. **Mid-price ≠ executable price.** Our labels are mid-price changes. You cannot trade the
   mid; you cross the spread. The gap between a mid-price association and a net-of-cost return
   is where most "profitable backtests" quietly die.

So the honest statement is: *these are well-established descriptive regularities of price
formation, measured cleanly and reported with uncertainty — and turning any of them into money
is a separate, much harder problem this project does not claim to have solved.*

---

## 7. Limitations

- **Breadth.** Five tickers, one trading session (2012-06-21). The results are a faithful
  replication on a small sample, not a claim about all markets or regimes. The pre-registered
  design and committed fixtures make extension straightforward, but we do not over-generalize.
- **Data level restriction.** The free LOBSTER level-10 sample omits pre-open and deep
  liquidity (§1), which is why the reconstruction oracle is the per-event transition invariant
  rather than a full-book match. Features use the reference book, so this does not bias the
  studies, but it bounds what the reconstruction differential can certify from messages alone.
- **Tick-rule sign inference.** Trade signs use the tick rule, which is a proxy; we validate
  its accuracy against LOBSTER's true resting side but the sign-memory study inherits any
  residual misclassification.
- **Effect-size units.** The OFI-forward coefficients are return-per-unit-OFI and therefore
  small in absolute magnitude; they are reported as measured, not rescaled to look larger. The
  interpretable takeaway is the sign, significance, and horizon shape, not the raw number.
- **Cross-symbol combination.** Family-level OFI estimates pool held-out folds across symbols
  (each split runs on its own symbol's timeline); the sign-memory ACF is a per-symbol
  trade-count-weighted average. The per-symbol split (§5) confirms no single name drives the
  pooled result.

---

## Figures

1. `docs/figures/01_ofi_horizon_profile.png` — OFI → forward-return coefficient vs horizon (95% CI).
2. `docs/figures/02_ofi_linearity.png` — binned contemporaneous OFI vs mid change, with the OLS fit (R² = 0.150).
3. `docs/figures/03_recovery_power.png` — synthetic-recovery power and CI-coverage vs injected β (the verification differentiator).
4. `docs/figures/04_robustness_heatmap.png` — per-symbol OFI R² and queue-imbalance AUC (the large-tick signature).
5. `docs/figures/05_sign_autocorr_decay.png` — trade-sign autocorrelation vs lag (Bartlett 95% CI).

## Reproduce

```bash
bash scripts/get_data.sh                 # download the LOBSTER sample (not committed)
python -m src.studies.runner             # rerun the 17-test family -> data/fixtures/*.csv
python scripts/make_figures.py           # regenerate docs/figures/*.png (deterministic)
python -m pytest                         # full verification suite (add -m "not slow" for CI subset)
python -m src.studies.g1_gate            # print the machinery-gate evidence report
```

*Data: LOBSTER (lobsterdata.com), NASDAQ sample 2012-06-21, academic-use. Methodology and all
choices are logged in `docs/DESIGN.md`.*
