#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A_00b_ackley_perturbation_and_minima.py

Prior stage of the Ackley surface study. This script is the single place in
which the fractal perturbation of the Ackley surface is defined. It

    1) defines the perturbation and the base surface,
    2) plots the standard and the perturbed surface side by side, over the
       full domain and zoomed around every minimum,
    3) locates the minima of the perturbed surface exactly, and
    4) writes those minima together with a fingerprint of the perturbation
       to ackley_perturbed_reference_minima.json.

The optimization script A_02b_ackley_loss_surface_optimization.py loads that
file for the perturbed variant and refuses to run if its own perturbation
constants, base minima or objective values disagree with the fingerprint. The
standard variant needs no file, because its minima are the analytic ones.

Run this first:

    python3 A_00b_ackley_perturbation_and_minima.py
    python3 A_02b_ackley_loss_surface_optimization.py

Outputs
-------
    <OUTPUT_DIR>/ackley_perturbed_reference_minima.json
    <OUTPUT_DIR>/perturbation_and_minima_report.txt
    <OUTPUT_DIR>/minima_table.csv
    <OUTPUT_DIR>/plots/*.png, *.eps
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# 0) SETTINGS
# =============================================================================

SURFACE_NAME = "ackley"
STANDARD_VARIANT_NAME = "standard_ackley"
PERTURBED_VARIANT_NAME = "fractional_ackley_additive"
PERTURBATION_MODE = "additive"

OUTPUT_DIR = Path("ackley_perturbation_and_minima")
PLOT_DIR = OUTPUT_DIR / "plots"
REFERENCE_FILE_NAME = "ackley_perturbed_reference_minima.json"

# Visualization domain. Must match the optimization script.
XMIN, XMAX = -5.0, 5.0
YMIN, YMAX = -5.0, 5.0

# -----------------------------------------------------------------------------
# The perturbation. These four constants and the two formulas below define the
# perturbed objective completely. They are written into the fingerprint and
# checked by the optimization script.
FRAC_A = 0.55
FRAC_B = 2.4
FRAC_K_MAX = 8
FRAC_LAMBDA_ADDITIVE = 0.2

BASE_FORMULA = "-20 exp(-0.2 sqrt(0.5 (x^2 + y^2))) - exp(0.5 (cos(2 pi x) + cos(2 pi y))) + 20 + e"
FRACTAL_FORMULA = "sum_{k=0}^{k_max} a^k (cos(b^k pi x) + cos(b^k pi y))"
PERTURBED_FORMULA = "base + lambda * fractal"

# Analytic minima of the standard surface.
BASE_MINIMA = np.array(
    [
        [0.0, 0.0],
    ],
    dtype=np.float64,
)

# -----------------------------------------------------------------------------
# Minimum search. Identical settings and identical code to the optimization
# script, so that both would find the same points.
REFERENCE_SEARCH_RADIUS = 0.25
REFERENCE_SEARCH_LATTICE_SIDE = 101
REFERENCE_DESCENT_ITERATIONS = 600
REFERENCE_DESCENT_INITIAL_STEP = 1e-4
REFERENCE_DESCENT_MIN_STEP = 1e-10
REFERENCE_DESCENT_MAX_STEP = 1e-2
REFERENCE_DESCENT_GROW = 1.2
REFERENCE_DESCENT_SHRINK = 0.5
REFERENCE_EPSILON_RADIUS = 1e-12

# Probe points at which the objective values are recorded in the fingerprint.
# The optimization script re-evaluates its own objective at these points, so a
# changed formula is detected even when every constant still agrees.
N_PROBE_POINTS = 32
PROBE_SEED = 20260102
FINGERPRINT_FLOAT_TOLERANCE = 1e-9

# -----------------------------------------------------------------------------
# Plotting. Style follows the surface plotting routine of the study.
FONT_SCALE = 1.25
DPI_PNG = 400
GRID_N = 350
ZOOM_GRID_N = 500
ZOOM_WINDOW_SIZES = [0.30, 0.05]

SUCCESS_DISTANCE_THRESHOLD = 0.2
MARK_SUCCESS_RADIUS_AROUND_MINIMA = True
SUCCESS_RADIUS_LINEWIDTH = 1.8

CLIP_Z_FOR_PLOTS = True
Z_CLIP_PERCENTILE_LOW = 1.0
Z_CLIP_PERCENTILE_HIGH = 99.0

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
    [COLOR_PALETTE_B[0], COLOR_PALETTE_B[1], COLOR_PALETTE_A[0],
     COLOR_PALETTE_A[1], COLOR_PALETTE_A[3]],
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


# =============================================================================
# 1) OBJECTIVE FUNCTIONS
# =============================================================================

def ackley_base_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    square_mean = 0.5 * (x ** 2 + y ** 2)
    cosine_mean = 0.5 * (np.cos(2.0 * np.pi * x) + np.cos(2.0 * np.pi * y))
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(square_mean))
        - np.exp(cosine_mean)
        + 20.0
        + np.e
    )


def ackley_base_grad_numpy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    square_mean = 0.5 * (x ** 2 + y ** 2)
    radius = np.sqrt(np.maximum(square_mean, REFERENCE_EPSILON_RADIUS))
    exponential_factor = 2.0 * np.exp(-0.2 * radius) / radius
    cosine_mean = 0.5 * (np.cos(2.0 * np.pi * x) + np.cos(2.0 * np.pi * y))
    cosine_factor = np.exp(cosine_mean) * np.pi
    gx = exponential_factor * x + cosine_factor * np.sin(2.0 * np.pi * x)
    gy = exponential_factor * y + cosine_factor * np.sin(2.0 * np.pi * y)
    return gx, gy


def fractal_component_numpy(
    x: np.ndarray,
    y: np.ndarray,
    a: float = FRAC_A,
    b: float = FRAC_B,
    k_max: int = FRAC_K_MAX,
) -> np.ndarray:
    value = np.zeros_like(np.asarray(x, dtype=np.float64))

    for k in range(k_max + 1):
        ak = a ** k
        bk = b ** k
        value = value + ak * (np.cos(bk * np.pi * x) + np.cos(bk * np.pi * y))

    return value


def fractal_component_grad_numpy(
    x: np.ndarray,
    y: np.ndarray,
    a: float = FRAC_A,
    b: float = FRAC_B,
    k_max: int = FRAC_K_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(np.asarray(x, dtype=np.float64))
    gy = np.zeros_like(np.asarray(y, dtype=np.float64))

    for k in range(k_max + 1):
        coeff = (a ** k) * (b ** k) * np.pi
        gx = gx - coeff * np.sin(b ** k * np.pi * x)
        gy = gy - coeff * np.sin(b ** k * np.pi * y)

    return gx, gy


def fractal_component_curvature_numpy(
    x: np.ndarray,
    y: np.ndarray,
    a: float = FRAC_A,
    b: float = FRAC_B,
    k_max: int = FRAC_K_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonal second derivatives of the perturbation ladder."""
    hx = np.zeros_like(np.asarray(x, dtype=np.float64))
    hy = np.zeros_like(np.asarray(y, dtype=np.float64))

    for k in range(k_max + 1):
        coeff = (a ** k) * (b ** k * np.pi) ** 2
        hx = hx - coeff * np.cos(b ** k * np.pi * x)
        hy = hy - coeff * np.cos(b ** k * np.pi * y)

    return hx, hy


def fractional_ackley_additive_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ackley_base_numpy(x, y) + FRAC_LAMBDA_ADDITIVE * fractal_component_numpy(x, y)


def fractional_ackley_additive_grad_numpy(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    bx, by = ackley_base_grad_numpy(x, y)
    fx, fy = fractal_component_grad_numpy(x, y)
    return bx + FRAC_LAMBDA_ADDITIVE * fx, by + FRAC_LAMBDA_ADDITIVE * fy


VARIANTS: Dict[str, Dict[str, Any]] = {
    STANDARD_VARIANT_NAME: {
        "np_fn": ackley_base_numpy,
        "grad_fn": ackley_base_grad_numpy,
        "is_perturbed": False,
        "label": "Standard",
        "description": "Classical Ackley function.",
    },
    PERTURBED_VARIANT_NAME: {
        "np_fn": fractional_ackley_additive_numpy,
        "grad_fn": fractional_ackley_additive_grad_numpy,
        "is_perturbed": True,
        "label": "Additive Fractal",
        "description": "Ackley plus additive multi-scale cosine perturbation.",
    },
}


# =============================================================================
# 2) MINIMUM SEARCH
# =============================================================================
# The two functions below are character-identical to the ones in the
# optimization script, so that the minima written here are exactly the minima
# that script would compute itself.

def monotone_descent(
    np_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    grad_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    x0: np.ndarray,
    y0: np.ndarray,
    n_iterations: int = REFERENCE_DESCENT_ITERATIONS,
    initial_step: float = REFERENCE_DESCENT_INITIAL_STEP,
    min_step: float = REFERENCE_DESCENT_MIN_STEP,
    max_step: float = REFERENCE_DESCENT_MAX_STEP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic local minimization of a rough objective.

    Plain gradient descent with a per-point step size that grows while the
    objective decreases and halves whenever a step would not decrease it. A
    proposed step is accepted only if it lowers the objective, so the iteration
    is monotone and cannot jump out of a narrow local minimum of the
    perturbation ladder. Every point of the input lattice is advanced in
    parallel, which keeps the cost independent of the lattice size up to the
    array operations themselves. No random numbers are used.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    y = np.array(y0, dtype=np.float64, copy=True)
    f = np.asarray(np_fn(x, y), dtype=np.float64)
    step = np.full(x.shape, float(initial_step), dtype=np.float64)

    for _ in range(int(n_iterations)):
        gx, gy = grad_fn(x, y)
        x_new = x - step * gx
        y_new = y - step * gy
        f_new = np.asarray(np_fn(x_new, y_new), dtype=np.float64)

        improved = f_new < f
        x = np.where(improved, x_new, x)
        y = np.where(improved, y_new, y)
        f = np.where(improved, f_new, f)
        step = np.where(
            improved,
            np.minimum(step * REFERENCE_DESCENT_GROW, max_step),
            np.maximum(step * REFERENCE_DESCENT_SHRINK, min_step),
        )

    return x, y, f


def deepest_minimum_near(
    np_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    grad_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    base_point: np.ndarray,
    radius: float = REFERENCE_SEARCH_RADIUS,
    lattice_side: int = REFERENCE_SEARCH_LATTICE_SIDE,
) -> Dict[str, Any]:
    """
    Locate the lowest local minimum of `np_fn` within `radius` of `base_point`,
    and also the local minimum reached by descent from `base_point` itself.
    """
    offsets = np.linspace(-radius, radius, int(lattice_side))
    dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
    inside_start = (dx ** 2 + dy ** 2) <= radius ** 2

    x0 = float(base_point[0]) + dx[inside_start]
    y0 = float(base_point[1]) + dy[inside_start]

    x, y, f = monotone_descent(np_fn, grad_fn, x0, y0)

    inside_end = ((x - float(base_point[0])) ** 2
                  + (y - float(base_point[1])) ** 2) <= radius ** 2
    if not np.any(inside_end):
        inside_end = np.ones_like(f, dtype=bool)
    x, y, f = x[inside_end], y[inside_end], f[inside_end]

    best = int(np.argmin(f))
    deepest = np.array([x[best], y[best]], dtype=np.float64)

    xn, yn, fn = monotone_descent(
        np_fn, grad_fn,
        np.array([float(base_point[0])], dtype=np.float64),
        np.array([float(base_point[1])], dtype=np.float64),
    )
    relaxed = np.array([float(xn[0]), float(yn[0])], dtype=np.float64)

    return {
        "deepest_point": deepest,
        "deepest_value": float(f[best]),
        "deepest_distance": float(np.linalg.norm(deepest - np.asarray(base_point, dtype=np.float64))),
        "relaxed_point": relaxed,
        "relaxed_value": float(fn[0]),
        "relaxed_distance": float(np.linalg.norm(relaxed - np.asarray(base_point, dtype=np.float64))),
        "n_starts": int(x0.size),
        "n_distinct_endpoints": int(
            np.unique(np.round(np.stack([x, y], axis=1), 5), axis=0).shape[0]
        ),
        "endpoint_value_min": float(np.min(f)),
        "endpoint_value_max": float(np.max(f)),
    }



def resolve_perturbed_minima() -> Dict[str, Any]:
    """
    Locate the minima of the perturbed surface and audit the geometry of the
    base minima under it.
    """
    np_fn = VARIANTS[PERTURBED_VARIANT_NAME]["np_fn"]
    grad_fn = VARIANTS[PERTURBED_VARIANT_NAME]["grad_fn"]

    audit: List[Dict[str, Any]] = []
    reference_minima: List[np.ndarray] = []

    for index, base_point in enumerate(BASE_MINIMA):
        bx = np.array(base_point[0])
        by = np.array(base_point[1])

        gx, gy = fractal_component_grad_numpy(bx, by)
        pert_grad_norm = float(FRAC_LAMBDA_ADDITIVE * np.hypot(float(gx), float(gy)))
        cx, cy = fractal_component_curvature_numpy(bx, by)
        vgx, vgy = grad_fn(bx, by)

        found = deepest_minimum_near(np_fn, grad_fn, base_point)
        reference_minima.append(found["deepest_point"])

        audit.append({
            "index": int(index),
            "base_minimum": [float(base_point[0]), float(base_point[1])],
            "base_objective_value": float(ackley_base_numpy(bx, by)),
            "variant_objective_value_at_base_minimum": float(np_fn(bx, by)),
            "perturbation_gradient_norm_at_base_minimum": pert_grad_norm,
            "variant_gradient_norm_at_base_minimum": float(
                np.hypot(float(vgx), float(vgy))
            ),
            "perturbation_curvature_at_base_minimum": [
                float(FRAC_LAMBDA_ADDITIVE * float(cx)),
                float(FRAC_LAMBDA_ADDITIVE * float(cy)),
            ],
            "base_minimum_is_critical_point": bool(pert_grad_norm < 1e-8),
            "reference_minimum": found["deepest_point"].tolist(),
            "reference_objective_value": found["deepest_value"],
            "displacement_from_base_minimum": found["deepest_distance"],
            "relaxed_minimum": found["relaxed_point"].tolist(),
            "relaxed_objective_value": found["relaxed_value"],
            "relaxed_displacement": found["relaxed_distance"],
            "n_search_starts": found["n_starts"],
            "n_distinct_descent_endpoints": found["n_distinct_endpoints"],
            "endpoint_value_min": found["endpoint_value_min"],
            "endpoint_value_max": found["endpoint_value_max"],
        })

    values = [row["reference_objective_value"] for row in audit]

    return {
        "reference_minima": np.stack(reference_minima, axis=0),
        "reference_values": values,
        "floor_value": float(min(values)),
        "floor_index": int(np.argmin(values)),
        "audit": audit,
    }


# =============================================================================
# 3) FINGERPRINT
# =============================================================================

def probe_points() -> np.ndarray:
    """
    Fixed points at which the objective values are recorded. The lattice is
    deterministic and spans the domain, with the base minima appended so that
    the most relevant locations are covered.
    """
    side = int(np.ceil(np.sqrt(max(N_PROBE_POINTS - len(BASE_MINIMA), 1))))
    xs = np.linspace(XMIN + 0.137, XMAX - 0.231, side)
    ys = np.linspace(YMIN + 0.311, YMAX - 0.179, side)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return np.concatenate([BASE_MINIMA, grid], axis=0)


def build_fingerprint() -> Dict[str, Any]:
    points = probe_points()
    base_values = ackley_base_numpy(points[:, 0], points[:, 1])
    pert_values = fractional_ackley_additive_numpy(points[:, 0], points[:, 1])

    return {
        "surface_name": SURFACE_NAME,
        "standard_variant_name": STANDARD_VARIANT_NAME,
        "perturbed_variant_name": PERTURBED_VARIANT_NAME,
        "perturbation_mode": PERTURBATION_MODE,
        "frac_a": float(FRAC_A),
        "frac_b": float(FRAC_B),
        "frac_k_max": int(FRAC_K_MAX),
        "frac_lambda_additive": float(FRAC_LAMBDA_ADDITIVE),
        "base_formula": BASE_FORMULA,
        "fractal_formula": FRACTAL_FORMULA,
        "perturbed_formula": PERTURBED_FORMULA,
        "base_minima": BASE_MINIMA.tolist(),
        "domain": [float(XMIN), float(XMAX), float(YMIN), float(YMAX)],
        "finest_perturbation_wavelength": float(2.0 / FRAC_B ** FRAC_K_MAX),
        "probe_points": points.tolist(),
        "probe_base_values": np.asarray(base_values, dtype=np.float64).tolist(),
        "probe_perturbed_values": np.asarray(pert_values, dtype=np.float64).tolist(),
        "float_tolerance": float(FINGERPRINT_FLOAT_TOLERANCE),
    }


# =============================================================================
# 4) PLOTTING
# =============================================================================

def clipped_for_plot(Z: np.ndarray) -> np.ndarray:
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


def valid_contour_levels(Z: np.ndarray, n_levels: int) -> np.ndarray:
    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, n_levels)

    zmin = float(np.min(finite))
    zmax = float(np.max(finite))

    if not np.isfinite(zmin) or not np.isfinite(zmax) or np.isclose(zmin, zmax):
        eps = max(1e-9, abs(zmin) * 1e-6)
        return np.linspace(zmin - eps, zmax + eps, n_levels)

    return np.linspace(zmin, zmax, n_levels)


def make_grid(xmin: float, xmax: float, ymin: float, ymax: float, n: int):
    x = np.linspace(float(xmin), float(xmax), n)
    y = np.linspace(float(ymin), float(ymax), n)
    return np.meshgrid(x, y)


def fixed_size_zoom_window(center_x: float, center_y: float, window_size: float):
    half = window_size / 2.0

    zxmin = center_x - half
    zxmax = center_x + half
    zymin = center_y - half
    zymax = center_y + half

    if zxmin < XMIN:
        shift = XMIN - zxmin
        zxmin += shift
        zxmax += shift
    if zxmax > XMAX:
        shift = zxmax - XMAX
        zxmin -= shift
        zxmax -= shift
    if zymin < YMIN:
        shift = YMIN - zymin
        zymin += shift
        zymax += shift
    if zymax > YMAX:
        shift = zymax - YMAX
        zymin -= shift
        zymax -= shift

    return float(zxmin), float(zxmax), float(zymin), float(zymax)


def mark_minima(
    ax,
    base_minima: np.ndarray,
    perturbed_minima: np.ndarray | None,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    """
    Mark the analytic base minima and, on a perturbed panel, the minima of the
    perturbed surface. The success radius is drawn around the minima that the
    corresponding success criterion refers to.
    """
    eps = 1e-12

    def visible(points: np.ndarray) -> np.ndarray:
        mask = (
            (points[:, 0] >= xmin - eps)
            & (points[:, 0] <= xmax + eps)
            & (points[:, 1] >= ymin - eps)
            & (points[:, 1] <= ymax + eps)
        )
        return points[mask]

    criterion_minima = base_minima if perturbed_minima is None else perturbed_minima

    if MARK_SUCCESS_RADIUS_AROUND_MINIMA:
        for minimum in visible(np.asarray(criterion_minima, dtype=np.float64)):
            ax.add_patch(
                plt.Circle(
                    (float(minimum[0]), float(minimum[1])),
                    float(SUCCESS_DISTANCE_THRESHOLD),
                    fill=False,
                    edgecolor=COLOR_PALETTE_A[6],
                    linewidth=SUCCESS_RADIUS_LINEWIDTH,
                    zorder=9,
                    clip_on=True,
                )
            )

    base_visible = visible(np.asarray(base_minima, dtype=np.float64))
    if len(base_visible) > 0:
        ax.scatter(
            base_visible[:, 0], base_visible[:, 1],
            marker="*", s=160, c="white", edgecolors="black", linewidths=0.8,
            zorder=10, clip_on=True, label="Standard-surface minimum",
        )

    if perturbed_minima is not None:
        pert_visible = visible(np.asarray(perturbed_minima, dtype=np.float64))
        if len(pert_visible) > 0:
            ax.scatter(
                pert_visible[:, 0], pert_visible[:, 1],
                marker="X", s=110, c=COLOR_PALETTE_A[6], edgecolors="black",
                linewidths=0.8, zorder=11, clip_on=True,
                label="Perturbed-surface minimum",
            )

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", framealpha=0.85)


def plot_side_by_side(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    grid_n: int,
    perturbed_minima: np.ndarray,
    suffix: str,
    title_suffix: str,
    with_labels: bool,
) -> None:
    """Standard surface on the left, perturbed surface on the right."""
    X, Y = make_grid(xmin, xmax, ymin, ymax, grid_n)

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.2))

    for ax, variant_name in zip(axes, (STANDARD_VARIANT_NAME, PERTURBED_VARIANT_NAME)):
        variant = VARIANTS[variant_name]
        Z = np.asarray(variant["np_fn"](X, Y), dtype=np.float64)
        Zp = clipped_for_plot(Z)

        cf = ax.contourf(X, Y, Zp, levels=valid_contour_levels(Zp, 60), cmap=HEATMAP_CMAP)
        cs = ax.contour(
            X, Y, Zp, levels=valid_contour_levels(Zp, 18),
            colors="black", linewidths=0.35, alpha=0.45,
        )

        if with_labels:
            ax.clabel(cs, inline=True, fontsize=max(6, int(6 * FONT_SCALE)), fmt="%.3g")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")

        mark_minima(
            ax,
            BASE_MINIMA,
            perturbed_minima if variant["is_perturbed"] else None,
            xmin, xmax, ymin, ymax,
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{SURFACE_NAME.capitalize()}: {variant['label']} ({title_suffix})")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("$f(x,y)$")

    fig.tight_layout()

    label_suffix = "with_labels" if with_labels else "no_labels"
    out_base = PLOT_DIR / f"{SURFACE_NAME}_standard_vs_additive_{suffix}_{label_suffix}"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_base.with_suffix(".eps"), format="eps", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_all_plots(perturbed_minima: np.ndarray) -> None:
    for with_labels in (True, False):
        plot_side_by_side(
            XMIN, XMAX, YMIN, YMAX, GRID_N, perturbed_minima,
            "full", "Full Domain", with_labels,
        )

    for index, base_point in enumerate(BASE_MINIMA):
        for window in ZOOM_WINDOW_SIZES:
            zxmin, zxmax, zymin, zymax = fixed_size_zoom_window(
                float(base_point[0]), float(base_point[1]), float(window)
            )
            tag = f"zoom_min{index}_w{str(window).replace('.', 'p')}"
            title = f"Zoom {window} Around Minimum {index}"
            for with_labels in (True, False):
                plot_side_by_side(
                    zxmin, zxmax, zymin, zymax, ZOOM_GRID_N, perturbed_minima,
                    tag, title, with_labels,
                )


# =============================================================================
# 5) OUTPUT
# =============================================================================

def write_reference_file(result: Dict[str, Any], fingerprint: Dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / REFERENCE_FILE_NAME

    payload = {
        "format_version": 1,
        "produced_by": "A_00b_ackley_perturbation_and_minima.py",
        "fingerprint": fingerprint,
        "search": {
            "reference_search_radius": float(REFERENCE_SEARCH_RADIUS),
            "reference_search_lattice_side": int(REFERENCE_SEARCH_LATTICE_SIDE),
            "reference_descent_iterations": int(REFERENCE_DESCENT_ITERATIONS),
            "reference_descent_initial_step": float(REFERENCE_DESCENT_INITIAL_STEP),
            "reference_descent_min_step": float(REFERENCE_DESCENT_MIN_STEP),
            "reference_descent_max_step": float(REFERENCE_DESCENT_MAX_STEP),
            "reference_descent_grow": float(REFERENCE_DESCENT_GROW),
            "reference_descent_shrink": float(REFERENCE_DESCENT_SHRINK),
            "reference_epsilon_radius": float(REFERENCE_EPSILON_RADIUS),
        },
        "reference_minima": result["reference_minima"].tolist(),
        "reference_values": result["reference_values"],
        "floor_value": result["floor_value"],
        "floor_index": result["floor_index"],
        "audit": result["audit"],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


def write_minima_table(result: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in result["audit"]:
        rows.append({
            "index": row["index"],
            "base_x": row["base_minimum"][0],
            "base_y": row["base_minimum"][1],
            "perturbed_x": row["reference_minimum"][0],
            "perturbed_y": row["reference_minimum"][1],
            "displacement": row["displacement_from_base_minimum"],
            "lambda_times_grad_P_norm": row["perturbation_gradient_norm_at_base_minimum"],
            "base_minimum_is_critical_point": row["base_minimum_is_critical_point"],
            "value_at_base_minimum": row["variant_objective_value_at_base_minimum"],
            "value_at_perturbed_minimum": row["reference_objective_value"],
            "n_distinct_descent_endpoints": row["n_distinct_descent_endpoints"],
            "endpoint_value_min": row["endpoint_value_min"],
            "endpoint_value_max": row["endpoint_value_max"],
        })

    with open(OUTPUT_DIR / "minima_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(result: Dict[str, Any], fingerprint: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 110)
    lines.append(f"Perturbation and minima of the {SURFACE_NAME} surface")
    lines.append("=" * 110)
    lines.append("")
    lines.append(f"base surface          : {BASE_FORMULA}")
    lines.append(f"perturbation          : {FRACTAL_FORMULA}")
    lines.append(f"combination           : {PERTURBED_FORMULA}")
    lines.append(
        f"constants             : a = {FRAC_A}, b = {FRAC_B}, "
        f"k_max = {FRAC_K_MAX}, lambda = {FRAC_LAMBDA_ADDITIVE}"
    )
    lines.append(
        f"finest wavelength     : {2.0 / FRAC_B ** FRAC_K_MAX:.6f}"
    )
    lines.append(
        f"search                : radius {REFERENCE_SEARCH_RADIUS}, "
        f"lattice {REFERENCE_SEARCH_LATTICE_SIDE}x{REFERENCE_SEARCH_LATTICE_SIDE}, "
        f"{REFERENCE_DESCENT_ITERATIONS} descent iterations"
    )
    lines.append("")
    lines.append(
        "The perturbation gradient at a base minimum is non-zero wherever the"
    )
    lines.append(
        "base minimum is not a critical point of the perturbed surface. The"
    )
    lines.append(
        "perturbed minima below are the points the optimization metrics refer to."
    )
    lines.append("")

    for row in result["audit"]:
        bm = row["base_minimum"]
        rm = row["reference_minimum"]
        rl = row["relaxed_minimum"]
        cur = row["perturbation_curvature_at_base_minimum"]
        lines.append(
            f"[{row['index']}] standard minimum        = ({bm[0]:+.6f}, {bm[1]:+.6f})\n"
            f"     value there (perturbed)  = {row['variant_objective_value_at_base_minimum']:+.8f}\n"
            f"     lambda * |grad P|        = {row['perturbation_gradient_norm_at_base_minimum']:.6f}\n"
            f"     |grad f_perturbed|       = {row['variant_gradient_norm_at_base_minimum']:.6f}\n"
            f"     is a critical point      = {row['base_minimum_is_critical_point']}\n"
            f"     lambda * d2P/dx2, d2P/dy2= ({cur[0]:+.2f}, {cur[1]:+.2f})\n"
            f"     perturbed minimum        = ({rm[0]:+.6f}, {rm[1]:+.6f})\n"
            f"     value there              = {row['reference_objective_value']:+.8f}\n"
            f"     displacement             = {row['displacement_from_base_minimum']:.6f}\n"
            f"     relaxed from standard min= ({rl[0]:+.6f}, {rl[1]:+.6f})"
            f"  value {row['relaxed_objective_value']:+.8f}"
            f"  distance {row['relaxed_displacement']:.6f}\n"
            f"     descent endpoints        = {row['n_distinct_descent_endpoints']} distinct,"
            f" values in [{row['endpoint_value_min']:+.6f},"
            f" {row['endpoint_value_max']:+.6f}]\n"
        )

    values = result["reference_values"]
    lines.append("")
    lines.append(
        f"perturbed minimum values: min {min(values):+.8f}  max {max(values):+.8f}"
        f"  range {max(values) - min(values):.8f}"
    )
    lines.append(
        f"floor_value written to the reference file = {result['floor_value']:+.8f}"
        f"  (perturbed minimum index {result['floor_index']})"
    )
    if max(values) - min(values) > 1e-9:
        lines.append("")
        lines.append(
            "The values differ between the perturbed minima, so these minima are"
        )
        lines.append(
            "not equivalent and a distance criterion alone does not separate them."
        )
    lines.append("")
    lines.append(f"probe points in the fingerprint: {len(fingerprint['probe_points'])}")
    lines.append("=" * 110)

    text = "\n".join(lines)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "perturbation_and_minima_report.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return text


# =============================================================================
# 6) MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(f"Perturbation definition and minima of the {SURFACE_NAME} surface")
    print("=" * 96)
    print(
        f"a = {FRAC_A}, b = {FRAC_B}, k_max = {FRAC_K_MAX}, "
        f"lambda = {FRAC_LAMBDA_ADDITIVE}"
    )

    print("\nLocating the minima of the perturbed surface")
    result = resolve_perturbed_minima()

    print("Building the perturbation fingerprint")
    fingerprint = build_fingerprint()

    print("Plotting the standard and perturbed surfaces side by side")
    make_all_plots(result["reference_minima"])

    write_minima_table(result)
    path = write_reference_file(result, fingerprint)
    text = write_report(result, fingerprint)

    print("")
    print(text)
    print("")
    print(f"Reference file: {path}")
    print(f"Plots:          {PLOT_DIR}")
    print(
        "\nRun the optimization script next; it loads this file for the"
        "\nperturbed variant and verifies the perturbation against the"
        "\nfingerprint before starting."
    )


if __name__ == "__main__":
    main()
