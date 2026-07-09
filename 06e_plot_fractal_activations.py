#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_selected_fractal_activations.py

Plots the selected activation functions used in the final comparison.

Output structure
----------------
plots_selected_fractal_activations/
    01_selected_regular_and_fractal_activations.{png,eps}

Selected activations
--------------------
The script plots exactly six activation functions in one figure:

    1. relu
    2. tanh
    3. weierstrass_mandelbrot_function_xpsin
    4. modulated_blancmange_curve
    5. decaying_cosine_function_tf
    6. modified_weierstrass_function_tanh

Colour scheme
-------------
Project hexadecimal palette (same palette used in the optimizer and analysis plots):
    #589641  #367DCD  #874BD2  #DC6923  #E1AA14  #2896A0  #BE372D  #2F7F74  #C05A8A
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fractal_activation_functions as faf


# =============================================================================
# 0) SETTINGS
# =============================================================================

OUT_DIR = Path("plots_selected_fractal_activations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Project hexadecimal palette  (9 colours — the selected plot uses 6 colours)
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

# x-axis range and resolution
X_MIN, X_MAX = -2.0, 2.0
N_POINTS     = 2000

# Save settings
DPI = 200

# Figure dimensions
FIGSIZE = (11, 6)


# =============================================================================
# 1) SHARED UTILITIES
# =============================================================================

x_np: np.ndarray = np.linspace(X_MIN, X_MAX, N_POINTS, dtype=np.float32)
x_tf: tf.Tensor  = tf.constant(x_np)


def eval_fn(fn: Callable) -> np.ndarray:
    """
    Evaluate `fn(x_tf)` and return a float64 numpy array of shape (N_POINTS,).
    Returns an all-NaN array on any exception so the rest of the plot is unaffected.
    """
    try:
        y = fn(x_tf)
        arr = np.asarray(y, dtype=np.float64).ravel()
        if arr.shape[0] != N_POINTS:
            arr = np.full(N_POINTS, np.nan)
        return arr
    except Exception as exc:
        print(f"    [WARN] evaluation failed: {exc}")
        return np.full(N_POINTS, np.nan)


def save_fig(fig: plt.Figure, stem: str) -> None:
    """Save `fig` to PNG (DPI=200) and EPS inside OUT_DIR."""
    png_path = OUT_DIR / f"{stem}.png"
    eps_path = OUT_DIR / f"{stem}.eps"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"    saved  {png_path.name}  +  {eps_path.name}")


def _decorate_axes(ax: plt.Axes, title: str, ylim: Optional[Tuple] = None) -> None:
    """Apply consistent style to an Axes object."""
    ax.axhline(0, color="#444444", linewidth=0.6, alpha=0.45, zorder=0)
    ax.axvline(0, color="#444444", linewidth=0.6, alpha=0.45, zorder=0)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.20, linestyle="--", linewidth=0.7)
    if ylim is not None:
        ax.set_ylim(*ylim)


# =============================================================================
# 2) SELECTED ACTIVATION PLOT
# =============================================================================

def plot_selected_activations(
    functions: List[Tuple[str, Callable]],
    stem: str,
    title: str,
    ylim: Optional[Tuple] = None,
) -> None:
    """
    Plot the selected regular and fractal activation functions in one figure.

    Parameters
    ----------
    functions : list of (label, callable)
        Exactly the selected activation functions to be shown.
    stem : str
        Output filename stem (no extension).
    title : str
        Figure title.
    ylim : (lo, hi) or None
        Optional manual y-axis limits.
    """
    if len(functions) > len(PALETTE):
        raise ValueError(
            f"Group '{stem}' has {len(functions)} functions but palette only has "
            f"{len(PALETTE)} colours. Split into smaller groups."
        )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for idx, (label, fn) in enumerate(functions):
        y = eval_fn(fn)
        ax.plot(x_np, y, color=PALETTE[idx], linewidth=2.0, label=label, zorder=3)

    _decorate_axes(ax, title, ylim)
    ax.legend(fontsize=9, loc="best", framealpha=0.88)
    fig.tight_layout()
    save_fig(fig, stem)


# =============================================================================
# 3) MAIN  —  orchestrate selected plot
# =============================================================================

def main() -> None:

    print("=" * 72)
    print("Selected regular and fractal activation function plot")
    print(f"  Output dir : {OUT_DIR.resolve()}")
    print(f"  DPI        : {DPI}")
    print(f"  x range    : [{X_MIN}, {X_MAX}]  ({N_POINTS} points)")
    print("  Activations: relu, tanh, wm_xpsin, blancmange, decaying cosine, mod. Weierstrass tanh")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # SELECTED REGULAR + FRACTAL ACTIVATIONS  (6 functions, one plot)
    # -------------------------------------------------------------------------

    print("\n[01] Selected regular and fractal activations")
    plot_selected_activations(
        functions=[
            ("relu",
             tf.nn.relu),
            ("tanh",
             tf.nn.tanh),
            ("weierstrass_mandelbrot_function_xpsin",
             faf.weierstrass_mandelbrot_function_xpsin),
            ("modulated_blancmange_curve",
             faf.modulated_blancmange_curve),
            ("decaying_cosine_function_tf",
             faf.decaying_cosine_function_tf),
            ("modified_weierstrass_function_tanh",
             faf.modified_weierstrass_function_tanh),
        ],
        stem="01_selected_regular_and_fractal_activations",
        title="Selected regular and fractal activation functions",
    )

    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    n_files = 1 * 2
    print(f"Done.  1 figure × 2 formats = {n_files} files")
    print(f"       {OUT_DIR.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()