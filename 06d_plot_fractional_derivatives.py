#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fractional_derivative_intuition.py

Creates the fractional-derivative intuition plots used in the article.

Output structure
----------------
plots_fractional_derivative_intuition/
    polynomial_function.{png,eps}
    polynomial_first_derivative.{png,eps}
    polynomial_second_derivative.{png,eps}
    polynomial_fractional_derivative_15.{png,eps}
    polynomial_derivative_comparison.{png,eps}

Content
-------
The script uses the polynomial

    f(x) = x^3,  x >= 0,

and plots:

    1. f(x)
    2. f'(x)
    3. f''
    4. Caputo derivative of order 1.5

The Caputo derivative is evaluated numerically from

    C D_{0+}^alpha f(x)
      = 1/Gamma(n-alpha) * int_0^x f^{(n)}(tau) (x-tau)^{n-alpha-1} dtau,

with n = ceil(alpha).  For alpha = 1.5, n = 2.

Colour scheme
-------------
Project hexadecimal palette:
    #589641  #367DCD  #874BD2  #DC6923  #E1AA14  #2896A0  #BE372D  #2F7F74  #C05A8A
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple
from math import ceil, gamma

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0) SETTINGS
# =============================================================================

OUT_DIR = Path("plots_fractional_derivative_intuition")
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

# Global font scaling.
# Increase this value to scale all plot fonts up while preserving their ratios.
FONT_SCALE = 1.7

# Shared x-axis range for all plots.
# x starts at 0 because the Caputo derivative is taken with lower terminal 0.
X_MIN, X_MAX = 0.0, 2.0
N_POINTS = 1200

# Caputo derivative order.
ALPHA_CAPUTO = 1.5

# Numerical quadrature resolution for the Caputo integral.
N_QUAD = 2500

# Save settings.
DPI = 220

# Square plots.
FIGSIZE_SINGLE = (5.0, 5.0)
FIGSIZE_COMBINED = FIGSIZE_SINGLE

# Fixed subplot geometry.
# This keeps the single plots and the combined plot identical in canvas size
# and axes placement.
SUBPLOT_ADJUST = {
    "left": 0.18,
    "right": 0.96,
    "bottom": 0.16,
    "top": 0.88,
}

# Plot line width.
LINEWIDTH = 3.0

# Matplotlib output settings.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11 * FONT_SCALE,
    "axes.titlesize": 13 * FONT_SCALE,
    "axes.labelsize": 12 * FONT_SCALE,
    "legend.fontsize": 9 * FONT_SCALE,
    "xtick.labelsize": 10 * FONT_SCALE,
    "ytick.labelsize": 10 * FONT_SCALE,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
})


# =============================================================================
# 1) FUNCTIONS AND DERIVATIVES
# =============================================================================

x_np = np.linspace(X_MIN, X_MAX, N_POINTS, dtype=np.float64)


def f_poly(x: np.ndarray) -> np.ndarray:
    """Base function f(x) = x^3."""
    return x ** 3


def df_poly(x: np.ndarray) -> np.ndarray:
    """First derivative f'(x) = 3x^2."""
    return 3.0 * x ** 2


def d2f_poly(x: np.ndarray) -> np.ndarray:
    """Second derivative f''(x) = 6x."""
    return 6.0 * x


def caputo_derivative_numeric(
    f_nth_derivative: Callable[[np.ndarray], np.ndarray],
    x_values: np.ndarray,
    alpha: float,
    n_quad: int = 2000,
) -> np.ndarray:
    """
    Numerically compute the Caputo derivative of order alpha.

    For n - 1 < alpha < n,

        C D_{0+}^alpha f(x)
          = 1/Gamma(n-alpha)
            int_0^x f^{(n)}(tau) (x-tau)^{n-alpha-1} dtau.

    A variable substitution removes the weak endpoint singularity:

        tau = x * (1 - s^r),   r = 1 / (n - alpha),   s in [0, 1].

    This gives

        C D_{0+}^alpha f(x)
          = r * x^{n-alpha} / Gamma(n-alpha)
            int_0^1 f^{(n)}(x(1-s^r)) ds.

    This implementation is written for clarity and robustness, not speed.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if abs(alpha - round(alpha)) < 1e-12:
        raise ValueError("Use the ordinary derivative for integer alpha.")

    n = int(ceil(alpha))
    beta = n - alpha

    if not (0.0 < beta < 1.0):
        raise ValueError("This implementation expects n-1 < alpha < n.")

    r = 1.0 / beta
    s_grid = np.linspace(0.0, 1.0, n_quad, dtype=np.float64)

    out = np.zeros_like(x_values, dtype=np.float64)
    prefactor_const = r / gamma(beta)

    for i, x in enumerate(x_values):
        if x <= 0.0:
            out[i] = 0.0
            continue

        tau = x * (1.0 - s_grid ** r)
        integrand = f_nth_derivative(tau)
        integral = np.trapz(integrand, s_grid)

        out[i] = prefactor_const * (x ** beta) * integral

    return out


# Compute all curves.
y_f = f_poly(x_np)
y_df = df_poly(x_np)
y_d2f = d2f_poly(x_np)
y_caputo_15 = caputo_derivative_numeric(
    f_nth_derivative=d2f_poly,
    x_values=x_np,
    alpha=ALPHA_CAPUTO,
    n_quad=N_QUAD,
)

# Analytic reference for a small diagnostic print only:
# C D_{0+}^{3/2} x^3 = Gamma(4)/Gamma(5/2) * x^{3/2}.
y_caputo_15_ref = gamma(4.0) / gamma(4.0 - ALPHA_CAPUTO) * x_np ** (3.0 - ALPHA_CAPUTO)
max_abs_err = float(np.max(np.abs(y_caputo_15 - y_caputo_15_ref)))

# Shared y-axis limits for all individual plots.
Y_MIN = -0.25
Y_MAX = 1.08 * float(
    max(
        np.max(y_f),
        np.max(y_df),
        np.max(y_d2f),
        np.max(y_caputo_15),
    )
)


# =============================================================================
# 2) PLOTTING UTILITIES
# =============================================================================

def save_fig(fig: plt.Figure, stem: str) -> None:
    """Save figure as PNG and EPS."""
    png_path = OUT_DIR / f"{stem}.png"
    eps_path = OUT_DIR / f"{stem}.eps"

    fig.savefig(png_path, dpi=DPI)
    fig.savefig(eps_path, format="eps")
    plt.close(fig)

    print(f"    saved  {png_path.name}  +  {eps_path.name}")


def decorate_axes(
    ax: plt.Axes,
    title: str,
    ylabel: str = "value",
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Apply shared axis formatting."""
    ax.axhline(0.0, color="#444444", linewidth=0.7, zorder=0)
    ax.axvline(0.0, color="#444444", linewidth=0.7, zorder=0)

    ax.set_xlim(X_MIN, X_MAX)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=10)

    ax.grid(True, color="#BBBBBB", linestyle="--", linewidth=0.6)
    ax.set_box_aspect(1)


def plot_single_curve(
    stem: str,
    title: str,
    label: str,
    y: np.ndarray,
    color: str,
) -> None:
    """Create one square plot with shared x/y limits."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.plot(x_np, y, color=color, linewidth=LINEWIDTH, label=label, zorder=3)

    decorate_axes(
        ax=ax,
        title=title,
        ylabel="value",
        ylim=(Y_MIN, Y_MAX),
    )

    ax.legend(loc="upper left", frameon=True, framealpha=1.0)
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    save_fig(fig, stem)


def plot_combined() -> None:
    """Create one square comparison plot with all curves."""
    fig, ax = plt.subplots(figsize=FIGSIZE_COMBINED)

    ax.plot(
        x_np,
        y_f,
        color=PALETTE[0],
        linewidth=LINEWIDTH,
        label=r"$f(x)=x^3$",
        zorder=3,
    )
    ax.plot(
        x_np,
        y_df,
        color=PALETTE[1],
        linewidth=LINEWIDTH,
        label=r"$f'(x)=3x^2$",
        zorder=3,
    )
    ax.plot(
        x_np,
        y_d2f,
        color=PALETTE[2],
        linewidth=LINEWIDTH,
        label=r"$f''(x)=6x$",
        zorder=3,
    )
    ax.plot(
        x_np,
        y_caputo_15,
        color=PALETTE[6],
        linewidth=LINEWIDTH,
        label=r"${}^{C}D_{0+}^{1.5} f(x)$",
        zorder=3,
    )

    decorate_axes(
        ax=ax,
        title="Deriv. of $f(x)=x^3$",
        ylabel="value",
        ylim=(Y_MIN, Y_MAX),
    )

    ax.legend(loc="upper left", frameon=True, framealpha=1.0)
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    save_fig(fig, "polynomial_derivative_comparison")


# =============================================================================
# 3) MAIN
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("Fractional-derivative intuition plots")
    print(f"  Output dir        : {OUT_DIR.resolve()}")
    print(f"  x range           : [{X_MIN}, {X_MAX}]")
    print(f"  shared y range    : [{Y_MIN:.3f}, {Y_MAX:.3f}]")
    print(f"  points            : {N_POINTS}")
    print(f"  Caputo alpha      : {ALPHA_CAPUTO}")
    print(f"  Caputo quadrature : {N_QUAD} points")
    print(f"  max Caputo error  : {max_abs_err:.6e} against analytic reference")
    print("=" * 72)

    plot_single_curve(
        stem="polynomial_function",
        title=r"Base function",
        label=r"$f(x)=x^3$",
        y=y_f,
        color=PALETTE[0],
    )

    plot_single_curve(
        stem="polynomial_first_derivative",
        title=r"First derivative",
        label=r"$f'(x)=3x^2$",
        y=y_df,
        color=PALETTE[1],
    )

    plot_single_curve(
        stem="polynomial_second_derivative",
        title=r"Second derivative",
        label=r"$f''(x)=6x$",
        y=y_d2f,
        color=PALETTE[2],
    )

    plot_single_curve(
        stem="polynomial_fractional_derivative_15",
        title=r"$1.5$. derivative",
        label=r"${}^{C}D_{0+}^{1.5} f(x)$",
        y=y_caputo_15,
        color=PALETTE[6],
    )

    plot_combined()

    print("\n" + "=" * 72)
    print("Done.  5 figures × 2 formats = 10 files")
    print(f"       {OUT_DIR.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()