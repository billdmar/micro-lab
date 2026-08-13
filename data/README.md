# data/

Two subdirectories, with very different rules:

## `raw/` — NOT committed (gitignored)
The raw LOBSTER limit-order-book sample files (message + orderbook CSV pairs for
AAPL, AMZN, GOOG, INTC, MSFT, session 2012-06-21, depth level 10). Downloaded by
`bash scripts/get_data.sh` (or `make data`) and subject to LOBSTER's academic-use
terms — **never committed or redistributed** here. See `docs/DESIGN.md §D0.1` for
provenance and `NOTICE` for attribution.

## `fixtures/` — committed (small, derived, reproducible)
Derived artifacts CI and the figure scripts run on, so nothing here depends on the
raw data. Each is regenerated deterministically (seeded from
`config.registry.MASTER_SEED`):

| File | Produced by | Contents |
|---|---|---|
| `study_family_results.csv` | `python -m src.studies.runner` | one row per registered test: name, metric, point, ci_low, ci_high, p_raw, p_adj, rejected, n_obs, horizon |
| `robustness_by_symbol.csv` | `python -m src.studies.runner` | per-symbol OFI R² (+CI, slope) and queue-imbalance AUC (+CI) |
| `ofi_linearity_binned.csv` | figure-prep (contemporaneous OFI binning) | binned OFI mean, mean forward mid-return, SE, n per bin |
| `ofi_linearity_fit.csv` | figure-prep | the OLS slope/intercept/R² and n for the linearity fit |
| `recovery_power_curve.csv` | `src.studies.g1_gate.recovery_curve` | injected β vs mean estimate, CI coverage, power, n_seeds (40) |

Regenerate the study fixtures with `make studies` (needs `make data` first) and the
figures with `make figures` (needs only the committed fixtures).
