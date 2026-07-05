"""Shared setup for every figure cell in the book.

Import this first in every ``{python}`` code cell:

    #| echo: false
    import _common

It gives the whole book one visual identity (a single palette + Matplotlib
house style) and one place to put shared constants and the canonical demo case,
so multiple chapters can replot the *same* result from different angles.

Design follows Tufte: white background, faint gridlines, no chartjunk, gray for
context, and saturated color spent only on the mark the eye should land on.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa: F401  (re-exported for convenience in cells)
import numpy as np  # noqa: F401  (re-exported for convenience in cells)

# If the project uses JAX and you want deterministic float64 CPU renders, enable:
# import os
# os.environ.setdefault("JAX_PLATFORMS", "cpu")
# import jax
# jax.config.update("jax_enable_x64", True)

# --- Palette -----------------------------------------------------------------
# A categorical palette with enough contrast to be colorblind-legible. Slot
# order is a CONVENTION you reuse across chapters (e.g. SERIES[0] = "our method",
# SERIES[1] = "baseline") so the same color means the same thing book-wide.
SERIES = [
    "#2a78d6",  # blue
    "#1baf7a",  # green
    "#eda100",  # amber
    "#d65a2a",  # orange-red
    "#8250c4",  # purple
    "#c0392b",  # brick
]

# Context colors — gray does the heavy lifting so color can mean "look here".
TEXT_PRIMARY = "#222222"
TEXT_SECONDARY = "#6b6b6b"
GRID = "#d9d9d9"

# --- House style -------------------------------------------------------------
mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,          # gridlines behind the data
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,        # despine: kill the box
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",    # left-aligned titles read as topic sentences
        "axes.labelsize": 10,
        "axes.labelcolor": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,         # prefer direct labels over a legend box
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def label_line(ax, x, y, text, color, dx=0.0, dy=0.0, **kwargs):
    """Direct-label a line at its end (Tufte) instead of using a legend.

    Place near the last plotted point so the eye never trips to a key.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        color=color,
        va="center",
        fontweight="bold",
        fontsize=9,
        **kwargs,
    )


def annotate_event(ax, x, text, color=TEXT_SECONDARY):
    """Draw a labelled vertical reference line so a figure carries its own story."""
    ax.axvline(x, color=color, linewidth=0.8, linestyle="--", zorder=0)
    ax.text(
        x, ax.get_ylim()[1], f" {text}",
        color=color, va="top", ha="left", fontsize=8, rotation=0,
    )


# --- Shared demo case --------------------------------------------------------
# Solve ONE canonical case here and reuse it across chapters. Replace this stub
# with your project's real setup; assert the property that must hold so a broken
# pipeline fails the render loudly rather than plotting garbage.
def demo_case():
    """Return the canonical result object every chapter plots from."""
    # result = my_package.solve(canonical_input)
    # assert result.converged, "demo_case did not converge — figures would be wrong"
    # return result
    raise NotImplementedError("Wire demo_case() to your project's canonical run.")
