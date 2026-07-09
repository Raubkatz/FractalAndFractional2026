#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fractal_function_intuition.py

Creates the fractal-function intuition plots used in the article.

Output structure
----------------
plots_intuition/
    weierstrass_partial_sum_1.{png,eps}
    weierstrass_partial_sum_2.{png,eps}
    weierstrass_partial_sum_3.{png,eps}
    weierstrass_partial_sum_4.{png,eps}
    weierstrass_partial_sum_5.{png,eps}

    weierstrass_tanh_activation_depth_1.{png,eps}
    weierstrass_tanh_activation_depth_2.{png,eps}
    weierstrass_tanh_activation_depth_3.{png,eps}
    weierstrass_tanh_activation_depth_4.{png,eps}
    weierstrass_tanh_activation_depth_5.{png,eps}

Content
-------
The script plots the two depth-wise constructions used in the intuition
section of the article:

    1. Finite Weierstrass-type partial sums

       W_M(x) = sum_{m=0}^{M-1} a^m cos(b^m pi x)

    2. Modified Weierstrass--Tanh activation depths

       phi_{W,M}(x)
       = tanh(x)
         + sum_{m=0}^{M-1} (-1)^m a^m cos(b^m pi x) exp(-mu_e |x|)

The plots are saved as PNG and EPS.

Colour scheme
-------------
Project hexadecimal palette:
    #589641  #367DCD  #874BD2  #DC6923  #E1AA14  #2896A0  #BE372D  #2F7F74  #C05A8A
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0) SETTINGS
# =============================================================================

OUT_DIR = Path("plots_intuition")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE: List[str] = [
    "#589641",   # 0  green
    "#367DCD",   # 1  blue
    "#874BD2",   # 2  purple
    "#DC6923",   # 3  orange
    "#E1AA14",   # 4  gold
    "#2896A0",   # 5  teal
    "#BE372D",   # 6  red
    "#2F7F74",   # 7  dark teal
    "#C05A8A",   # 8  pink
]

BG_COLOR = "#FFFFFF"

# Depths used in the article figures.
MAX_DEPTH = 5

# Shared x-axis ranges.
X_MIN_WEIERSTRASS, X_MAX_WEIERSTRASS = -2.0, 2.0
X_MIN_ACTIVATION, X_MAX_ACTIVATION = -2.0, 2.0
N_POINTS = 4000

# Weierstrass-type function parameters.
A_WEIERSTRASS = 0.55
B_WEIERSTRASS = 3.0

# Modified Weierstrass--Tanh activation parameters.
A_ACTIVATION = 0.55
B_ACTIVATION = 3.0
MU_E = 0.85

# Save settings.
DPI = 220

# Square plot settings.
FIGSIZE = (4.0, 4.0)
LINE_WIDTH = 3.0

# Fixed subplot geometry.
# Keeps every PNG/EPS canvas and axes placement identical.
SUBPLOT_ADJUST = {
    "left": 0.13,
    "right": 0.96,
    "bottom": 0.13,
    "top": 0.96,
}

# Clean panel style.
SHOW_TICKS = False
SHOW_GRID = False
SHOW_ZERO_LINES = True
TRANSPARENT_BG = True

# Matplotlib output settings.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
})


# =============================================================================
# 1) FUNCTIONS
# =============================================================================

x_weierstrass = np.linspace(
    X_MIN_WEIERSTRASS,
    X_MAX_WEIERSTRASS,
    N_POINTS,
    dtype=np.float64,
)

x_activation = np.linspace(
    X_MIN_ACTIVATION,
    X_MAX_ACTIVATION,
    N_POINTS,
    dtype=np.float64,
)


def weierstrass_partial_sum(
    x: np.ndarray,
    depth: int,
    a: float = A_WEIERSTRASS,
    b: float = B_WEIERSTRASS,
) -> np.ndarray:
    """
    Finite Weierstrass-type partial sum

        W_M(x) = sum_{m=0}^{M-1} a^m cos(b^m pi x).
    """
    y = np.zeros_like(x, dtype=np.float64)
    for m in range(depth):
        y += (a ** m) * np.cos((b ** m) * np.pi * x)
    return y


def weierstrass_tanh_activation_depth(
    x: np.ndarray,
    depth: int,
    a: float = A_ACTIVATION,
    b: float = B_ACTIVATION,
    mu_e: float = MU_E,
) -> np.ndarray:
    """
    Modified Weierstrass--Tanh activation at depth M

        phi_{W,M}(x)
        = tanh(x)
          + sum_{m=0}^{M-1} (-1)^m a^m cos(b^m pi x) exp(-mu_e |x|).
    """
    y = np.tanh(x).astype(np.float64)
    envelope = np.exp(-mu_e * np.abs(x))

    for m in range(depth):
        y += ((-1.0) ** m) * (a ** m) * np.cos((b ** m) * np.pi * x) * envelope

    return y


# Compute all curves first so that all depths in each family share y-limits.
weierstrass_curves = [
    weierstrass_partial_sum(x_weierstrass, depth)
    for depth in range(1, MAX_DEPTH + 1)
]

activation_curves = [
    weierstrass_tanh_activation_depth(x_activation, depth)
    for depth in range(1, MAX_DEPTH + 1)
]


def symmetric_ylim(curves: List[np.ndarray], margin: float = 0.08) -> Tuple[float, float]:
    """Create symmetric y-limits around zero for a list of curves."""
    max_abs = max(float(np.max(np.abs(y))) for y in curves)
    max_abs *= 1.0 + margin
    return -max_abs, max_abs


Y_LIM_WEIERSTRASS = symmetric_ylim(weierstrass_curves)
Y_LIM_ACTIVATION = symmetric_ylim(activation_curves)


# =============================================================================
# 2) PLOTTING UTILITIES
# =============================================================================

def save_fig(fig: plt.Figure, stem: str) -> None:
    """Save figure as PNG and EPS."""
    png_path = OUT_DIR / f"{stem}.png"
    eps_path = OUT_DIR / f"{stem}.eps"

    fig.savefig(
        png_path,
        format="png",
        dpi=DPI,
        facecolor=fig.get_facecolor(),
        transparent=TRANSPARENT_BG,
    )

    fig.savefig(
        eps_path,
        format="eps",
        facecolor=BG_COLOR,
    )

    plt.close(fig)

    print(f"    saved  {png_path.name}  +  {eps_path.name}")


def decorate_axes(
    ax: plt.Axes,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    """Apply shared clean style to one panel."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if SHOW_ZERO_LINES:
        ax.axhline(0.0, color="#444444", linewidth=0.6, alpha=0.50, zorder=0)
        ax.axvline(0.0, color="#444444", linewidth=0.6, alpha=0.50, zorder=0)

    if SHOW_GRID:
        ax.grid(True, color="#BBBBBB", linestyle="--", linewidth=0.6, alpha=0.40)
    else:
        ax.grid(False)

    if not SHOW_TICKS:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_curve(
    x: np.ndarray,
    y: np.ndarray,
    stem: str,
    color: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    """Create one square curve plot."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

    ax.plot(
        x,
        y,
        color=color,
        linewidth=LINE_WIDTH,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=3,
    )

    decorate_axes(ax=ax, xlim=xlim, ylim=ylim)

    fig.subplots_adjust(**SUBPLOT_ADJUST)
    save_fig(fig, stem)


# =============================================================================
# 3) MAIN
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("Fractal-function intuition plots")
    print(f"  Output dir             : {OUT_DIR.resolve()}")
    print(f"  Weierstrass x range    : [{X_MIN_WEIERSTRASS}, {X_MAX_WEIERSTRASS}]")
    print(f"  Activation x range     : [{X_MIN_ACTIVATION}, {X_MAX_ACTIVATION}]")
    print(f"  Points                 : {N_POINTS}")
    print(f"  Max depth              : {MAX_DEPTH}")
    print(f"  W parameters           : a={A_WEIERSTRASS}, b={B_WEIERSTRASS}")
    print(f"  Activation parameters  : a={A_ACTIVATION}, b={B_ACTIVATION}, mu_e={MU_E}")
    print("=" * 72)

    print("\n[01] Weierstrass-type partial sums")
    for depth, y in enumerate(weierstrass_curves, start=1):
        plot_curve(
            x=x_weierstrass,
            y=y,
            stem=f"weierstrass_partial_sum_{depth}",
            color=PALETTE[6],
            xlim=(X_MIN_WEIERSTRASS, X_MAX_WEIERSTRASS),
            ylim=Y_LIM_WEIERSTRASS,
        )

    print("\n[02] Modified Weierstrass--Tanh activation depths")
    for depth, y in enumerate(activation_curves, start=1):
        plot_curve(
            x=x_activation,
            y=y,
            stem=f"weierstrass_tanh_activation_depth_{depth}",
            color=PALETTE[6],
            xlim=(X_MIN_ACTIVATION, X_MAX_ACTIVATION),
            ylim=Y_LIM_ACTIVATION,
        )

    print("\n" + "=" * 72)
    print("Done.  10 figures × 2 formats = 20 files")
    print(f"       {OUT_DIR.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()