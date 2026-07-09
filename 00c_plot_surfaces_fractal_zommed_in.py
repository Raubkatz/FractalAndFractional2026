#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


OUTPUT_DIR = Path("surface_visualizations_contours_only_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_SCALE = 1.25
DPI_PNG = 400
GRID_N = 350
ZOOM_GRID_N = 500

ZOOM_WINDOW_SIZE = 0.30
ZOOM_HALF_WIDTH = ZOOM_WINDOW_SIZE / 2.0

CLIP_Z_FOR_PLOTS = True
Z_CLIP_PERCENTILE_LOW = 1.0
Z_CLIP_PERCENTILE_HIGH = 99.0

ZOOM_NEAR_MINIMUM_OFFSET = {
    "ackley": (0.18, 0.12),
    "rastrigin": (0.18, 0.12),
    "himmelblau": (0.18, 0.12),
    "levy": (0.18, 0.12),
}

COLOR_PALETTE_A = [
    "#589641", "#367DCD", "#874BD2", "#DC6923", "#E1AA14",
    "#2896A0", "#BE372D", "#2F7F74", "#C05A8A",
]

COLOR_PALETTE_B = [
    "#E7F5E0", "#E5F1FF", "#F2E8FF", "#FFEFE4", "#FFF8DC",
    "#E1F7F8", "#FFE8E6", "#E4F4F1", "#FCEAF2",
]

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "paper_heatmap_cmap",
    [COLOR_PALETTE_B[0], COLOR_PALETTE_B[1], COLOR_PALETTE_A[0], COLOR_PALETTE_A[1], COLOR_PALETTE_A[3]],
    N=256,
)

plt.rcParams.update({
    "font.size": 10 * FONT_SCALE,
    "axes.titlesize": 12 * FONT_SCALE,
    "axes.labelsize": 11 * FONT_SCALE,
    "xtick.labelsize": 9 * FONT_SCALE,
    "ytick.labelsize": 9 * FONT_SCALE,
    "legend.fontsize": 9 * FONT_SCALE,
    "figure.titlesize": 13 * FONT_SCALE,
    "axes.linewidth": 0.9,
    "savefig.dpi": DPI_PNG,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

SURFACE_SETTINGS: Dict[str, Dict[str, float]] = {
    "ackley": {"xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0, "frac_a": 0.55, "frac_b": 2.4, "frac_k_max": 8, "lambda_additive": 0.2, "lambda_multiplicative": 0.1, "lambda_warped": 0.2},
    "rastrigin": {"xmin": -5.12, "xmax": 5.12, "ymin": -5.12, "ymax": 5.12, "frac_a": 0.55, "frac_b": 2.7, "frac_k_max": 8, "lambda_additive": 0.20, "lambda_multiplicative": 0.1, "lambda_warped": 0.2},
    "himmelblau": {"xmin": -6.0, "xmax": 6.0, "ymin": -6.0, "ymax": 6.0, "frac_a": 0.55, "frac_b": 2.5, "frac_k_max": 8, "lambda_additive": 0.2, "lambda_multiplicative": 0.1, "lambda_warped": 0.2},
    "levy": {"xmin": -6.0, "xmax": 6.0, "ymin": -6.0, "ymax": 6.0, "frac_a": 0.55, "frac_b": 2.5, "frac_k_max": 8, "lambda_additive": 0.2, "lambda_multiplicative": 0.1, "lambda_warped": 0.2},
}


def ackley(X, Y):
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (X**2 + Y**2))) - np.exp(0.5 * (np.cos(2.0 * np.pi * X) + np.cos(2.0 * np.pi * Y))) + 20.0 + np.e


def rastrigin(X, Y):
    return 20.0 + X**2 + Y**2 - 10.0 * np.cos(2.0 * np.pi * X) - 10.0 * np.cos(2.0 * np.pi * Y)


def himmelblau(X, Y):
    return (X**2 + Y - 11.0)**2 + (X + Y**2 - 7.0)**2


def levy(X, Y):
    wx = 1.0 + (X - 1.0) / 4.0
    wy = 1.0 + (Y - 1.0) / 4.0
    return np.sin(np.pi * wx)**2 + (wx - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * wx + 1.0)**2) + (wy - 1.0)**2 * (1.0 + np.sin(2.0 * np.pi * wy)**2)


BASE_FUNCTIONS: Dict[str, Callable] = {
    "ackley": ackley,
    "rastrigin": rastrigin,
    "himmelblau": himmelblau,
    "levy": levy,
}


def fractal_term(X, Y, a, b, k_max):
    out = np.zeros_like(X, dtype=np.float64)
    for k in range(k_max + 1):
        out += (a**k) * (np.cos((b**k) * np.pi * X) + np.cos((b**k) * np.pi * Y))
    return out


def warped_coordinates(X, Y, a, b, k_max, lam):
    wx = np.zeros_like(X, dtype=np.float64)
    wy = np.zeros_like(Y, dtype=np.float64)
    for k in range(k_max + 1):
        wx += (a**k) * np.cos((b**k) * np.pi * X)
        wy += (a**k) * np.cos((b**k) * np.pi * Y)
    return X + lam * wx, Y + lam * wy


def evaluate_variant(surface_name, variant_name, X, Y):
    s = SURFACE_SETTINGS[surface_name]
    base_fn = BASE_FUNCTIONS[surface_name]

    a = float(s["frac_a"])
    b = float(s["frac_b"])
    k_max = int(s["frac_k_max"])

    if variant_name == "standard":
        return base_fn(X, Y)

    P = fractal_term(X, Y, a, b, k_max)

    if variant_name == "additive":
        return base_fn(X, Y) + float(s["lambda_additive"]) * P

    if variant_name == "multiplicative":
        return base_fn(X, Y) * (1.0 + float(s["lambda_multiplicative"]) * P)

    if variant_name == "warped":
        U, V = warped_coordinates(X, Y, a, b, k_max, float(s["lambda_warped"]))
        return base_fn(U, V)

    raise ValueError(f"Unknown variant: {variant_name}")


def make_grid(surface_name):
    s = SURFACE_SETTINGS[surface_name]
    x = np.linspace(float(s["xmin"]), float(s["xmax"]), GRID_N)
    y = np.linspace(float(s["ymin"]), float(s["ymax"]), GRID_N)
    return np.meshgrid(x, y)


def make_custom_grid(xmin, xmax, ymin, ymax, n):
    x = np.linspace(float(xmin), float(xmax), n)
    y = np.linspace(float(ymin), float(ymax), n)
    return np.meshgrid(x, y)


def clipped_for_plot(Z):
    if not CLIP_Z_FOR_PLOTS:
        return Z

    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        return Z

    lo = np.percentile(finite, Z_CLIP_PERCENTILE_LOW)
    hi = np.percentile(finite, Z_CLIP_PERCENTILE_HIGH)

    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return Z

    return np.clip(Z, lo, hi)


def known_minima(surface_name):
    return {
        "ackley": np.array([[0.0, 0.0]], dtype=np.float64),
        "rastrigin": np.array([[0.0, 0.0]], dtype=np.float64),
        "himmelblau": np.array([[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]], dtype=np.float64),
        "levy": np.array([[1.0, 1.0]], dtype=np.float64),
    }[surface_name]


def add_minima_stars(ax, surface_name, xmin, xmax, ymin, ymax):
    minima = known_minima(surface_name)
    eps = 1e-12

    mask = (
        (minima[:, 0] >= xmin - eps)
        & (minima[:, 0] <= xmax + eps)
        & (minima[:, 1] >= ymin - eps)
        & (minima[:, 1] <= ymax + eps)
    )

    minima_visible = minima[mask]
    if len(minima_visible) == 0:
        return

    ax.scatter(
        minima_visible[:, 0],
        minima_visible[:, 1],
        marker="*",
        s=120,
        c="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
        clip_on=True,
        label="Known minimum",
    )
    ax.legend(loc="upper right", framealpha=0.85)


def surface_bounds(surface_name):
    s = SURFACE_SETTINGS[surface_name]
    return float(s["xmin"]), float(s["xmax"]), float(s["ymin"]), float(s["ymax"])


def fixed_size_zoom_window(surface_name, center_x, center_y, window_size):
    xmin, xmax, ymin, ymax = surface_bounds(surface_name)
    half = window_size / 2.0

    zxmin = center_x - half
    zxmax = center_x + half
    zymin = center_y - half
    zymax = center_y + half

    if zxmin < xmin:
        shift = xmin - zxmin
        zxmin += shift
        zxmax += shift
    if zxmax > xmax:
        shift = zxmax - xmax
        zxmin -= shift
        zxmax -= shift
    if zymin < ymin:
        shift = ymin - zymin
        zymin += shift
        zymax += shift
    if zymax > ymax:
        shift = zymax - ymax
        zymin -= shift
        zymax -= shift

    return float(zxmin), float(zxmax), float(zymin), float(zymax)


def zoom_windows(surface_name):
    base_minimum = known_minima(surface_name)[0]
    dx, dy = ZOOM_NEAR_MINIMUM_OFFSET[surface_name]

    return {
        "zoom_minimum": fixed_size_zoom_window(surface_name, float(base_minimum[0]), float(base_minimum[1]), ZOOM_WINDOW_SIZE),
        "zoom_near_minimum": fixed_size_zoom_window(surface_name, float(base_minimum[0] + dx), float(base_minimum[1] + dy), ZOOM_WINDOW_SIZE),
    }


def valid_contour_levels(Z, n_levels):
    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, n_levels)

    zmin = float(np.min(finite))
    zmax = float(np.max(finite))

    if not np.isfinite(zmin) or not np.isfinite(zmax) or np.isclose(zmin, zmax):
        eps = max(1e-9, abs(zmin) * 1e-6)
        return np.linspace(zmin - eps, zmax + eps, n_levels)

    return np.linspace(zmin, zmax, n_levels)


def nice_title(surface_name, variant_name, suffix):
    surface = surface_name.capitalize()
    variant_map = {
        "standard": "Standard",
        "additive": "Additive Fractal Perturbation",
        "multiplicative": "Multiplicative Fractal Perturbation",
        "warped": "Warped-Coordinate Fractal Perturbation",
    }
    suffix_map = {
        "full": "Full Domain",
        "zoom_minimum": "Zoom Around Minimum",
        "zoom_near_minimum": "Zoom Near Minimum",
    }
    return f"{surface}: {variant_map[variant_name]} ({suffix_map[suffix]})"


def save_figure(fig, out_base):
    fig.savefig(out_base.with_suffix(".png"), dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_base.with_suffix(".eps"), format="eps", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_single_contour(X, Y, Z, surface_name, variant_name, out_dir, suffix, with_labels):
    Zp = clipped_for_plot(Z)

    xmin = float(X[0, 0])
    xmax = float(X[0, -1])
    ymin = float(Y[0, 0])
    ymax = float(Y[-1, 0])

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    cf = ax.contourf(X, Y, Zp, levels=valid_contour_levels(Zp, 60), cmap=HEATMAP_CMAP)
    cs = ax.contour(X, Y, Zp, levels=valid_contour_levels(Zp, 18), colors="black", linewidths=0.35, alpha=0.45)

    if with_labels:
        ax.clabel(cs, inline=True, fontsize=max(6, int(6 * FONT_SCALE)), fmt="%.3g")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    add_minima_stars(ax, surface_name, xmin, xmax, ymin, ymax)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_title(nice_title(surface_name, variant_name, suffix))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("$f(x,y)$")

    fig.tight_layout()

    label_suffix = "with_labels" if with_labels else "no_labels"
    save_figure(fig, out_dir / f"{surface_name}_{variant_name}_{suffix}_2d_contour_{label_suffix}")


def plot_contour(X, Y, Z, surface_name, variant_name, out_dir, suffix="full"):
    plot_single_contour(X, Y, Z, surface_name, variant_name, out_dir, suffix, True)
    plot_single_contour(X, Y, Z, surface_name, variant_name, out_dir, suffix, False)


def main():
    variants = ["standard", "additive", "multiplicative", "warped"]

    for surface_name in ["ackley", "rastrigin", "himmelblau", "levy"]:
        surface_dir = OUTPUT_DIR / surface_name
        surface_dir.mkdir(parents=True, exist_ok=True)

        X, Y = make_grid(surface_name)

        for variant_name in variants:
            Z = np.asarray(evaluate_variant(surface_name, variant_name, X, Y), dtype=np.float64)
            plot_contour(X, Y, Z, surface_name, variant_name, surface_dir, suffix="full")

            for zoom_name, (xmin, xmax, ymin, ymax) in zoom_windows(surface_name).items():
                Xz, Yz = make_custom_grid(xmin, xmax, ymin, ymax, ZOOM_GRID_N)
                Zz = np.asarray(evaluate_variant(surface_name, variant_name, Xz, Yz), dtype=np.float64)
                plot_contour(Xz, Yz, Zz, surface_name, variant_name, surface_dir, suffix=zoom_name)

    print(f"\nDone. Saved plots to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()