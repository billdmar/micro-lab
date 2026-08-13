#!/usr/bin/env python3
"""Deterministic showcase-figure generator for micro-lab.

Regenerates every showcase PNG under docs/figures/ from the COMMITTED derived
fixtures in data/fixtures/ — no raw LOBSTER data is read, so this runs anywhere
(including CI) and reproduces identically (headless Agg backend, fixed dpi).

Figures:
    01_ofi_horizon_profile      OFI->forward-return coef vs horizon, with 95% CI band
    02_ofi_linearity          binned contemporaneous OFI vs mid change + OLS fit (R^2)
    03_recovery_power         synthetic-recovery power & CI-coverage vs injected beta
    04_robustness_heatmap     per-symbol OFI R^2 and queue-imbalance AUC (large-tick)
    05_sign_autocorr_decay    trade-sign autocorrelation vs lag, with Bartlett CI band

Usage:
    python scripts/make_figures.py                 # data/fixtures -> docs/figures/
    python scripts/make_figures.py --outdir /tmp/f # write elsewhere
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Run-as-script: put the repo root on sys.path so `src` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viz.style import (  # noqa: E402
    CATEGORICAL,
    PALETTE,
    SERIES,
    apply_house_style,
    footer,
    savefig,
)

FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fixtures"
)


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(FIXTURES, name))


# --------------------------------------------------------------------------- #
# 01 — OFI -> forward mid-change coefficient by horizon (out-of-sample).
# --------------------------------------------------------------------------- #
def fig_horizon_profile(fam: pd.DataFrame, outdir: str) -> str:
    import matplotlib.pyplot as plt

    apply_house_style()
    fwd = fam[fam["name"].str.startswith("ofi_forward_h")].copy()
    fwd["h"] = fwd["horizon"].astype(int)
    fwd = fwd.sort_values("h")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    color = CATEGORICAL[0]
    ax.fill_between(fwd["h"], fwd["ci_low"], fwd["ci_high"], color=color, alpha=0.15, linewidth=0)
    ax.plot(fwd["h"], fwd["point"], color=color, marker="o", label="direct OOS coef (HAC 95% CI)")
    ax.axhline(0.0, color=PALETTE["axis"], linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(fwd["h"])
    ax.set_xticklabels([str(h) for h in fwd["h"]])
    ax.set_xlabel("forward horizon (events, log scale)")
    ax.set_ylabel("OFI → forward return coefficient")
    ax.set_title("Order-flow imbalance predicts forward price at every horizon (out-of-sample)")
    ax.legend(loc="upper left")
    footer(fig)
    path = os.path.join(outdir, "01_ofi_horizon_profile.png")
    savefig(fig, path)
    return path


# --------------------------------------------------------------------------- #
# 02 — Contemporaneous OFI vs mid-price change: the near-linear relationship.
# --------------------------------------------------------------------------- #
def fig_linearity(binned: pd.DataFrame, fit: pd.DataFrame, outdir: str) -> str:
    import matplotlib.pyplot as plt

    apply_house_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    color = CATEGORICAL[0]
    ax.errorbar(
        binned["ofi_mean"],
        binned["ret_mean"],
        yerr=binned["ret_se"],
        fmt="o",
        color=color,
        ecolor=PALETTE["muted"],
        elinewidth=0.8,
        capsize=2,
        label="binned mean ± SE",
    )
    xs = np.linspace(binned["ofi_mean"].min(), binned["ofi_mean"].max(), 100)
    slope = float(fit["slope"].iloc[0])
    intercept = float(fit["intercept"].iloc[0])
    r2 = float(fit["r2"].iloc[0])
    ax.plot(xs, intercept + slope * xs, color=CATEGORICAL[1], label=f"OLS fit (R² = {r2:.3f})")
    ax.axhline(0.0, color=PALETTE["axis"], linewidth=0.8)
    ax.axvline(0.0, color=PALETTE["axis"], linewidth=0.8)
    ax.set_xlabel("order-flow imbalance (bucket sum)")
    ax.set_ylabel("contemporaneous mid log-return")
    ax.set_title("OFI vs contemporaneous price change is near-linear")
    ax.legend(loc="upper left")
    footer(fig)
    path = os.path.join(outdir, "02_ofi_linearity.png")
    savefig(fig, path)
    return path


# --------------------------------------------------------------------------- #
# 03 — Synthetic-recovery power curve (the "we verified the pipeline" figure).
# --------------------------------------------------------------------------- #
def fig_recovery_power(power: pd.DataFrame, outdir: str) -> str:
    import matplotlib.pyplot as plt

    apply_house_style()
    p = power.sort_values("beta")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(p["beta"], p["power"], color=CATEGORICAL[0], marker="o", label="power (CI excludes 0)")
    ax.plot(
        p["beta"],
        p["ci_coverage"],
        color=CATEGORICAL[2],
        marker="s",
        linestyle="--",
        label="95% CI coverage of β",
    )
    ax.axhline(0.95, color=PALETTE["muted"], linewidth=0.8, linestyle=":")
    ax.axhline(0.05, color=PALETTE["muted"], linewidth=0.8, linestyle=":")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("injected effect size β (log-return per unit signed flow)")
    ax.set_ylabel("rate over 40 seeds")
    ax.set_title("The pipeline recovers planted signals and stays silent on noise")
    ax.legend(loc="center right")
    footer(fig, "Ground-truth simulator; verification of the statistics pipeline itself.")
    path = os.path.join(outdir, "03_recovery_power.png")
    savefig(fig, path)
    return path


# --------------------------------------------------------------------------- #
# 04 — Per-symbol robustness: the large-tick queue-imbalance signature.
# --------------------------------------------------------------------------- #
def fig_robustness(rob: pd.DataFrame, outdir: str) -> str:
    import matplotlib.pyplot as plt

    apply_house_style()
    rob = rob.set_index("ticker").loc[list(SERIES.keys())]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.2))
    tickers = list(rob.index)
    colors = [SERIES[t] for t in tickers]

    ax1.bar(tickers, rob["ofi_R2"], color=colors, width=0.62)
    ax1.errorbar(
        tickers,
        rob["ofi_R2"],
        yerr=[rob["ofi_R2"] - rob["ofi_R2_lo"], rob["ofi_R2_hi"] - rob["ofi_R2"]],
        fmt="none",
        ecolor=PALETTE["ink_secondary"],
        elinewidth=0.9,
        capsize=3,
    )
    ax1.set_title("OFI R² by symbol")
    ax1.set_ylabel("contemporaneous R²")
    ax1.set_ylim(0, None)

    ax2.bar(tickers, rob["qi_auc"], color=colors, width=0.62)
    ax2.errorbar(
        tickers,
        rob["qi_auc"],
        yerr=[rob["qi_auc"] - rob["qi_auc_lo"], rob["qi_auc_hi"] - rob["qi_auc"]],
        fmt="none",
        ecolor=PALETTE["ink_secondary"],
        elinewidth=0.9,
        capsize=3,
    )
    ax2.axhline(0.5, color=PALETTE["axis"], linewidth=0.8, label="no skill (0.5)")
    ax2.set_title("Queue-imbalance AUC by symbol (large-tick effect)")
    ax2.set_ylabel("next-move AUC")
    ax2.set_ylim(0.5, 1.0)
    ax2.legend(loc="upper left")
    fig.suptitle(
        "Effects hold across symbols; queue imbalance predicts far better in large-tick names",
        fontsize=12,
        fontweight="bold",
    )
    footer(fig)
    path = os.path.join(outdir, "04_robustness_heatmap.png")
    savefig(fig, path)
    return path


# --------------------------------------------------------------------------- #
# 05 — Trade-sign autocorrelation decay (long memory), Bartlett CI band.
# --------------------------------------------------------------------------- #
def fig_sign_autocorr(fam: pd.DataFrame, outdir: str) -> str:
    import matplotlib.pyplot as plt

    apply_house_style()
    ac = fam[fam["name"].str.startswith("sign_autocorr_lag")].copy()
    ac["lag"] = ac["horizon"].astype(int)
    ac = ac.sort_values("lag")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    color = CATEGORICAL[3]
    ax.fill_between(ac["lag"], ac["ci_low"], ac["ci_high"], color=color, alpha=0.15, linewidth=0)
    ax.plot(
        ac["lag"],
        ac["point"],
        color=color,
        marker="o",
        label="sign autocorrelation (Bartlett 95% CI)",
    )
    ax.axhline(0.0, color=PALETTE["axis"], linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(ac["lag"])
    ax.set_xticklabels([str(x) for x in ac["lag"]])
    ax.set_xlabel("lag (trades, log scale)")
    ax.set_ylabel("autocorrelation of trade sign")
    ax.set_title("Trade-sign autocorrelation decays slowly — the long-memory signature")
    ax.legend(loc="upper right")
    footer(fig)
    path = os.path.join(outdir, "05_sign_autocorr_decay.png")
    savefig(fig, path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    default_out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures"
    )
    ap.add_argument("--outdir", default=default_out)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fam = _load("study_family_results.csv")
    rob = _load("robustness_by_symbol.csv")
    binned = _load("ofi_linearity_binned.csv")
    fit = _load("ofi_linearity_fit.csv")
    power = _load("recovery_power_curve.csv")

    paths = [
        fig_horizon_profile(fam, args.outdir),
        fig_linearity(binned, fit, args.outdir),
        fig_recovery_power(power, args.outdir),
        fig_robustness(rob, args.outdir),
        fig_sign_autocorr(fam, args.outdir),
    ]
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
