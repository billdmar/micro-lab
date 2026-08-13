"""House style for micro-lab showcase figures — one consistent visual system.

Design follows the dataviz skill's validated default palette (light surface),
matched to the sibling repos (vol-lab) so the portfolio reads as one system:
categorical hues assigned in fixed order (never cycled), thin marks, recessive
grid/axes, text in ink tokens rather than series colors, a legend for >= 2
series, and a subtle provenance footer on every figure. Everything here is
deterministic and headless (Agg backend), so figures regenerate byte-stably.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless, before any pyplot import

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# Chart chrome & ink (dataviz reference palette, light surface).
PALETTE = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

# Categorical slots, fixed order (blue, orange, aqua, yellow, magenta). The five
# LOBSTER sample tickers are pinned to fixed slots so color follows the entity
# consistently across every figure in the note and README.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SERIES = {
    "AAPL": CATEGORICAL[0],
    "AMZN": CATEGORICAL[1],
    "GOOG": CATEGORICAL[2],
    "INTC": CATEGORICAL[3],
    "MSFT": CATEGORICAL[4],
}

# Sequential ramps (single hue, light->dark) for encoding a magnitude such as
# forward horizon across the grid, or a match rate on a heatmap.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_ORANGE = ["#f7cbb4", "#f3a883", "#ef8656", "#eb6834", "#c9531f", "#a34115", "#7d310e"]

FONT_FAMILY = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]


def apply_house_style() -> None:
    """Install the micro-lab rcParams. Call once before creating any figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["surface"],
            "savefig.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": 10,
            "text.color": PALETTE["ink"],
            "axes.edgecolor": PALETTE["axis"],
            "axes.linewidth": 0.8,
            "axes.labelcolor": PALETTE["ink_secondary"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelcolor": PALETTE["ink_secondary"],
            "ytick.labelcolor": PALETTE["ink_secondary"],
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "figure.dpi": 140,
            "savefig.dpi": 140,
        }
    )


def footer(
    fig: Figure, note: str = "LOBSTER sample 2012-06-21; descriptive only, no tradability claim."
) -> None:
    """Stamp the shared provenance footer below the plotting area (descriptive-only).

    Reserves a strip of figure height for the footer so it never collides with the
    x-axis label, then anchors the note in that strip. Call AFTER all axes/labels
    are set."""
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.06)
    fig.text(
        0.99,
        0.015,
        note,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=PALETTE["muted"],
    )


def sample_ramp(ramp: list[str], n: int) -> list[str]:
    """Pick `n` evenly spaced colors from a light->dark sequential ramp."""
    if n <= 1:
        return [ramp[len(ramp) // 2]]
    idx = [round(i * (len(ramp) - 1) / (n - 1)) for i in range(n)]
    return [ramp[j] for j in idx]


def savefig(fig: Figure, path: str) -> None:
    """Save a figure at house dpi with a tight-ish, deterministic layout."""
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
