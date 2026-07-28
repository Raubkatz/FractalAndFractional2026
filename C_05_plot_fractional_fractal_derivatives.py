#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_weierstrass_fractional_coupling_intuition.py

Creates intuition plots for the interaction between fractal functions and
fractional-memory derivatives.

Output structure
----------------
plots_weierstrass_fractional_coupling/
    weierstrass_function.{png,eps}
    weierstrass_pointwise_derivative.{png,eps}
    weierstrass_fractional_derivative_gl.{png,eps}
    weierstrass_derivative_comparison.{png,eps}

Purpose
-------
The script uses a finite Weierstrass-type function

    W_N(x) = sum_{m=0}^{N-1} a^m cos(b^m pi x),

with a b > 1.  In this regime, higher-frequency terms have decreasing
amplitude but their classical derivative has increasing contribution,
because

    d/dx [a^m cos(b^m pi x)]
      = -pi a^m b^m sin(b^m pi x).

Thus, the finite classical derivative exists, but it becomes increasingly
large and oscillatory as N grows.

The script compares:

    1. The original finite Weierstrass-type function W_N(x).
    2. A pointwise numerical first derivative of W_N(x).
    3. A finite-memory Grunwald--Letnikov derivative of order nu < 1.
    4. A combined plot of all three objects.

Finite-memory Grunwald--Letnikov derivative
-------------------------------------------
For sampled values u_i = W_N(x_i), spacing h, and order nu, the finite-memory
backward GL derivative is computed as

    D_h^nu u_i
      = h^{-nu} sum_{k=0}^{min(i,K-1)} c_k^(nu) u_{i-k},

where

    c_0^(nu) = 1,
    c_k^(nu) = (1 - (nu + 1)/k) c_{k-1}^(nu).

This is the same coefficient recursion used in the memory-based fractional
optimizer sections.  Here it is applied to function values rather than to
gradient sequences, only for visualization.

Colour scheme
-------------
Project hexadecimal palette:
    #589641  #367DCD  #874BD2  #DC6923  #E1AA14  #2896A0  #BE372D  #2F7F74  #C05A8A
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
from math import log

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0) SETTINGS
# =============================================================================

OUT_DIR = Path("plots_weierstrass_fractional_coupling")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE: List[str] = [
    "#2896A0",  # 5  teal
    "#DC6923",  # 3  orange
    "#367DCD",   # 1  blue
    "#874BD2",   # 2  purple
    "#E1AA14",   # 4  gold
    "#589641",  # 0  green
    "#BE372D",   # 6  red
    "#2F7F74",   # 7  dark teal
    "#C05A8A",   # 8  pink
]

BG_COLOR = "#FFFFFF"

# Global font scaling.
# Increase this value to scale all plot fonts up while preserving their ratios.
FONT_SCALE = 1.35

# Display window.  All plots use this same x-window.
X_DISPLAY_MIN, X_DISPLAY_MAX = -0.05, 0.05

# Display window.  Used only when Y_AXIS_MODE = "global".
Y_DISPLAY_MIN, Y_DISPLAY_MAX = 0.0, 30.0

# Y-axis limit mode.
# Options:
#   "automatic"  : let Matplotlib choose y-limits for each plot
#   "individual" : use separately computed y-limits for each plot
#   "global"     : use Y_DISPLAY_MIN, Y_DISPLAY_MAX for all plots
Y_AXIS_MODE = "automatic"

# Extended computation window.
# The GL derivative is causal/backward-looking, so the left padding avoids
# boundary artifacts inside the displayed window.
X_LEFT_PAD = 0.35

# Sampling resolution.
# The high-frequency terms require a dense grid.
N_POINTS_EXTENDED = 90000

# Weierstrass-type function parameters.
# The product a*b is larger than 1, so the classical derivative becomes
# increasingly oscillatory as more terms are included.
A_WEIERSTRASS = 0.55
B_WEIERSTRASS = 3.0
N_TERMS = 8

# Fractional derivative parameters.
# A value below 1 gives an intermediate, memory-based derivative order.
NU_FRACTIONAL = 0.5 #0.5og
GL_MEMORY_SIZE = 512

# Save settings.
DPI = 220

# Square plot settings.
FIGSIZE_SINGLE = (5.0, 5.0)
FIGSIZE_COMBINED = FIGSIZE_SINGLE

# Fixed subplot geometry.
# Keeps all PNG/EPS canvases and axes placements identical.
SUBPLOT_ADJUST = {
    "left": 0.18,
    "right": 0.96,
    "bottom": 0.16,
    "top": 0.88,
}

# Plot line width.
LINEWIDTH = 0.6

# Clean plot style.
SHOW_GRID = True
SHOW_ZERO_LINES = True
TRANSPARENT_BG = False

# Plot colours.
COLOR_FUNCTION = PALETTE[6]
COLOR_CLASSICAL_DERIVATIVE = PALETTE[1]
COLOR_FRACTIONAL_DERIVATIVE = PALETTE[0]

# Numerical derivative setting.
# If False, the plot uses the pointwise finite-difference derivative.
# If True, it uses the closed-form derivative of the finite sum.
USE_ANALYTIC_CLASSICAL_DERIVATIVE_FOR_PLOT = False

# Matplotlib output settings.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11 * FONT_SCALE,
    "axes.titlesize": 13 * FONT_SCALE,
    "axes.labelsize": 12 * FONT_SCALE,
    "legend.fontsize": 8.5 * FONT_SCALE,
    "xtick.labelsize": 10 * FONT_SCALE,
    "ytick.labelsize": 10 * FONT_SCALE,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
})


# =============================================================================
# 1) WEIERSTRASS FUNCTION AND DERIVATIVES
# =============================================================================

x_ext = np.linspace(
    X_DISPLAY_MIN - X_LEFT_PAD,
    X_DISPLAY_MAX,
    N_POINTS_EXTENDED,
    dtype=np.float64,
)

DX = float(x_ext[1] - x_ext[0])

display_mask = (x_ext >= X_DISPLAY_MIN) & (x_ext <= X_DISPLAY_MAX)
x_plot = x_ext[display_mask]


def weierstrass_partial_sum(
    x: np.ndarray,
    depth: int = N_TERMS,
    a: float = A_WEIERSTRASS,
    b: float = B_WEIERSTRASS,
) -> np.ndarray:
    """
    Finite Weierstrass-type partial sum

        W_N(x) = sum_{m=0}^{N-1} a^m cos(b^m pi x).
    """
    y = np.zeros_like(x, dtype=np.float64)

    for m in range(depth):
        y += (a ** m) * np.cos((b ** m) * np.pi * x)

    return y


def weierstrass_derivative_exact(
    x: np.ndarray,
    depth: int = N_TERMS,
    a: float = A_WEIERSTRASS,
    b: float = B_WEIERSTRASS,
) -> np.ndarray:
    """
    Closed-form derivative of the finite Weierstrass-type partial sum

        dW_N/dx
          = -pi sum_{m=0}^{N-1} a^m b^m sin(b^m pi x).

    This exists because N is finite.  Its scale grows with (a b)^m.
    """
    y = np.zeros_like(x, dtype=np.float64)

    for m in range(depth):
        y += -np.pi * (a ** m) * (b ** m) * np.sin((b ** m) * np.pi * x)

    return y


def pointwise_numerical_derivative(
    y: np.ndarray,
    dx: float,
) -> np.ndarray:
    """
    Pointwise numerical first derivative using central differences in the
    interior and one-sided differences at the boundaries.
    """
    return np.gradient(y, dx, edge_order=2)


def gl_coefficients(
    nu: float,
    memory_size: int,
) -> np.ndarray:
    """
    Grunwald--Letnikov coefficients

        c_0^(nu) = 1,
        c_k^(nu) = (1 - (nu + 1)/k) c_{k-1}^(nu).

    For nu = 1 and memory_size >= 2, the sequence starts [1, -1, 0, ...],
    which recovers the ordinary backward first difference after division
    by h.
    """
    if not (0.0 < nu < 2.0):
        raise ValueError("nu must satisfy 0 < nu < 2.")
    if memory_size < 1:
        raise ValueError("memory_size must be at least 1.")

    coeffs = np.zeros(memory_size, dtype=np.float64)
    coeffs[0] = 1.0

    for k in range(1, memory_size):
        coeffs[k] = (1.0 - (nu + 1.0) / k) * coeffs[k - 1]

    return coeffs


def finite_memory_gl_derivative(
    values: np.ndarray,
    dx: float,
    nu: float,
    memory_size: int,
) -> np.ndarray:
    """
    Finite-memory backward Grunwald--Letnikov derivative

        D_h^nu u_i
          = h^{-nu} sum_{k=0}^{min(i,K-1)} c_k^(nu) u_{i-k}.

    Implemented by causal convolution with the GL coefficient sequence.
    """
    coeffs = gl_coefficients(nu=nu, memory_size=memory_size)

    conv = np.convolve(values, coeffs, mode="full")[: values.shape[0]]
    deriv = conv / (dx ** nu)

    return deriv


# Compute function and derivatives on the extended window.
y_function_ext = weierstrass_partial_sum(x_ext)

y_classical_numeric_ext = pointwise_numerical_derivative(
    y=y_function_ext,
    dx=DX,
)

y_classical_exact_ext = weierstrass_derivative_exact(x_ext)

y_fractional_gl_ext = finite_memory_gl_derivative(
    values=y_function_ext,
    dx=DX,
    nu=NU_FRACTIONAL,
    memory_size=GL_MEMORY_SIZE,
)

# Select the original display window.
y_function = y_function_ext[display_mask]
y_classical_numeric = y_classical_numeric_ext[display_mask]
y_classical_exact = y_classical_exact_ext[display_mask]
y_fractional_gl = y_fractional_gl_ext[display_mask]

if USE_ANALYTIC_CLASSICAL_DERIVATIVE_FOR_PLOT:
    y_classical = y_classical_exact
    classical_label = r"closed-form $dW_N/dx$"
    classical_title = "Ord. first der."
else:
    y_classical = y_classical_numeric
    classical_label = r"p. num. $dW_N/dx$"
    classical_title = "P. num. first der."

# Diagnostic difference between numerical and exact derivative of the finite sum.
max_abs_deriv_error = float(np.max(np.abs(y_classical_numeric - y_classical_exact)))

# Roughness exponent of the infinite Weierstrass-type idealization.
ROUGHNESS_EXPONENT = -log(A_WEIERSTRASS) / log(B_WEIERSTRASS)


# =============================================================================
# 2) PLOTTING UTILITIES
# =============================================================================

def full_ylim(
    curves: List[np.ndarray],
    margin: float = 0.08,
    symmetric: bool = True,
) -> Tuple[float, float]:
    """
    Return y-limits that include all finite values of all curves.
    """
    finite_values = []

    for curve in curves:
        curve = np.asarray(curve, dtype=np.float64)
        finite_values.append(curve[np.isfinite(curve)])

    values = np.concatenate(finite_values)

    if symmetric:
        max_abs = float(np.max(np.abs(values)))
        max_abs = max_abs * (1.0 + margin)
        if max_abs <= 0:
            max_abs = 1.0
        return -max_abs, max_abs

    ymin = float(np.min(values))
    ymax = float(np.max(values))
    span = ymax - ymin
    if span <= 0:
        span = 1.0

    return ymin - margin * span, ymax + margin * span


Y_LIM_FUNCTION = full_ylim([y_function], symmetric=True)
Y_LIM_CLASSICAL = full_ylim([y_classical], symmetric=True)
Y_LIM_FRACTIONAL = full_ylim([y_fractional_gl], symmetric=True)
Y_LIM_COMBINED = full_ylim(
    [y_function, y_classical, y_fractional_gl],
    symmetric=True,
)

Y_LIM_GLOBAL = (Y_DISPLAY_MIN, Y_DISPLAY_MAX)


def get_ylim(kind: str) -> Optional[Tuple[float, float]]:
    """
    Select y-axis limits according to Y_AXIS_MODE.

    kind must be one of:
        "function", "classical", "fractional", "combined".
    """
    if Y_AXIS_MODE == "automatic":
        return None

    if Y_AXIS_MODE == "global":
        return Y_LIM_GLOBAL

    if Y_AXIS_MODE == "individual":
        if kind == "function":
            return Y_LIM_FUNCTION
        if kind == "classical":
            return Y_LIM_CLASSICAL
        if kind == "fractional":
            return Y_LIM_FRACTIONAL
        if kind == "combined":
            return Y_LIM_COMBINED
        raise ValueError(f"Unknown y-limit kind: {kind}")

    raise ValueError(
        "Y_AXIS_MODE must be one of 'automatic', 'individual', or 'global'."
    )


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
    title: str,
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Apply shared axis formatting."""
    if SHOW_ZERO_LINES:
        ax.axhline(0.0, color="#444444", linewidth=0.7, alpha=0.55, zorder=0)
        ax.axvline(0.0, color="#444444", linewidth=0.7, alpha=0.55, zorder=0)

    ax.set_xlim(X_DISPLAY_MIN, X_DISPLAY_MAX)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=10)

    if SHOW_GRID:
        ax.grid(True, color="#BBBBBB", linestyle="--", linewidth=0.6, alpha=0.55)
    else:
        ax.grid(False)

    ax.set_box_aspect(1)


def plot_single_curve(
    stem: str,
    title: str,
    label: str,
    y: np.ndarray,
    color: str,
    ylim: Optional[Tuple[float, float]],
    ylabel: str = "value",
) -> None:
    """Create one square plot with the shared x-window."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    ax.plot(
        x_plot,
        y,
        color=color,
        linewidth=LINEWIDTH,
        label=label,
        zorder=3,
    )

    decorate_axes(
        ax=ax,
        title=title,
        ylabel=ylabel,
        ylim=ylim,
    )

    ax.legend(loc="upper left", frameon=True, framealpha=1.0)
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    save_fig(fig, stem)


def plot_combined() -> None:
    """Create one square comparison plot with all curves."""
    fig, ax = plt.subplots(figsize=FIGSIZE_COMBINED)

    ax.plot(
        x_plot,
        y_classical,
        color=COLOR_CLASSICAL_DERIVATIVE,
        linewidth=LINEWIDTH,
        label=classical_label,
        zorder=3,
    )


    ax.plot(
        x_plot,
        y_fractional_gl,
        color=COLOR_FRACTIONAL_DERIVATIVE,
        linewidth=LINEWIDTH*1.2,
        label=rf"$D_h^{{{NU_FRACTIONAL:.2f}}}W_{{{N_TERMS}}}(x)$",
        zorder=3,
    )

    ax.plot(
        x_plot,
        y_function,
        color=COLOR_FUNCTION,
        linewidth=LINEWIDTH*1.25,
        label=rf"$W_{{{N_TERMS}}}(x)$",
        zorder=3,
    )





    decorate_axes(
        ax=ax,
        title="All deriv.",
        ylabel="value",
        ylim=get_ylim("combined"),
    )

    ax.legend(loc="upper left", frameon=True, framealpha=1.0)
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    save_fig(fig, "weierstrass_derivative_comparison")


# =============================================================================
# 3) MAIN
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("Weierstrass / fractional-memory derivative intuition plots")
    print(f"  Output dir             : {OUT_DIR.resolve()}")
    print(f"  Display x range        : [{X_DISPLAY_MIN}, {X_DISPLAY_MAX}]")
    print(f"  Y-axis mode            : {Y_AXIS_MODE}")
    if Y_AXIS_MODE == "global":
        print(f"  Display y range        : [{Y_DISPLAY_MIN}, {Y_DISPLAY_MAX}]")
    print(f"  Left computation pad   : {X_LEFT_PAD}")
    print(f"  Extended grid points   : {N_POINTS_EXTENDED}")
    print(f"  dx                     : {DX:.6e}")
    print(f"  Weierstrass terms      : {N_TERMS}")
    print(f"  Weierstrass parameters : a={A_WEIERSTRASS}, b={B_WEIERSTRASS}, a*b={A_WEIERSTRASS * B_WEIERSTRASS:.3f}")
    print(f"  Roughness exponent     : H=-log(a)/log(b)={ROUGHNESS_EXPONENT:.3f}")
    print(f"  GL order               : nu={NU_FRACTIONAL}")
    print(f"  GL memory size         : K={GL_MEMORY_SIZE}")
    print(f"  Classical derivative   : {classical_label}")
    print(f"  max |numeric-exact|    : {max_abs_deriv_error:.6e}")
    print("=" * 72)

    plot_single_curve(
        stem="weierstrass_function",
        title=rf"Frac. func. $W_{{{N_TERMS}}}$",
        label=rf"$W_{{{N_TERMS}}}(x)$",
        y=y_function,
        color=COLOR_FUNCTION,
        ylim=get_ylim("function"),
        ylabel="function value",
    )

    plot_single_curve(
        stem="weierstrass_pointwise_derivative",
        title=classical_title,
        label=classical_label,
        y=y_classical,
        color=COLOR_CLASSICAL_DERIVATIVE,
        ylim=get_ylim("classical"),
        ylabel="derivative value",
    )

    plot_single_curve(
        stem="weierstrass_fractional_derivative_gl",
        title=rf"GL deriv. $\nu={NU_FRACTIONAL:.2f}$",
        label=rf"$D_h^{{{NU_FRACTIONAL:.2f}}}W_{{{N_TERMS}}}(x)$",
        y=y_fractional_gl,
        color=COLOR_FRACTIONAL_DERIVATIVE,
        ylim=get_ylim("fractional"),
        ylabel="fractional derivative value",
    )

    plot_combined()

    print("\n" + "=" * 72)
    print("Done.  4 figures × 2 formats = 8 files")
    print(f"       {OUT_DIR.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()