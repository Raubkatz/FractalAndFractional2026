#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fractional_ackley_optimizer_comparison.py

This script compares TensorFlow optimizers on several Ackley-type objective
surfaces, including fractional/fractal perturbation variants.

Multi-run experimental design
-----------------------------
This version uses the same infrastructure as the earlier multi-run Ackley
and Rastrigin scripts, but with a more controlled randomized starting-point
design. Each run uses:

    1. a reproducible base run seed,
    2. a reproducible optimizer-specific seed,
    3. one common starting point shared by every optimizer in the run,
    4. a fixed-radius starting point around the standard Ackley minimum.

The starting-point design is intentionally controlled. The complete list of
starting points is generated once, before any optimization, by
generate_common_start_points(). It contains exactly one point per run, and
within a run that same point is used by every optimizer. No starting point is
drawn at random.

The runs are radiated around the standard Ackley minimum: their angles are
spread evenly over the full circle, so that n runs sharing a reference minimum
use the angles

    theta_i = START_POINT_ANGLE_OFFSET + 2 pi i / n,   i = 0, ..., n - 1,

and the starting point of run i is

    p_0 = m + r [cos(theta_i), sin(theta_i)],

where m is the standard Ackley global minimum and r is
START_RADIUS_AROUND_MINIMUM. All optimizers in a run therefore begin at
identical coordinates, which removes the optimizer-specific initial direction
from the comparison, while the set of runs still covers all directions around
the minimum.

The point is accepted only if it lies inside the plotting and optimization
domain [XMIN, XMAX] x [YMIN, YMAX]. If an angle gives a point outside the
domain, the angle is advanced deterministically in steps of
START_POINT_ANGLE_SEARCH_INCREMENT, up to START_POINT_MAX_RESAMPLE_ATTEMPTS
times. If no valid angle is found, the script falls back to a clipped point in
the direction of the original angle.

This allows later analysis of:

    - mean final loss,
    - standard deviation of final loss,
    - best and worst final loss,
    - mean best loss reached during the trajectory,
    - mean distance to the standard Ackley global minimum,
    - success rate according to a distance threshold,
    - average number of steps,
    - wall-clock runtime per optimizer run,
    - aggregate runtime statistics across runs,
    - stability across optimizer-specific starting points.

The starting points are generated once from N_RUNS and the selected minimum
indices alone, without any random draw. They are saved in run_design.json and
then reused for all objective variants and for every optimizer, so that a given
run index denotes the same starting point throughout the experiment.

Output structure
----------------
The script creates:

    ackley_comparison_fin_std_add_5_lr_vderiv_grid/
        run_design.json
        all_results.json
        all_aggregate_statistics.json
        all_best_configuration_per_optimizer.json

        <objective_name>/
            aggregate_statistics.json
            aggregate_statistics_by_configuration.json
            best_configuration_per_optimizer.json
            aggregate_summary.txt
            aggregate_summary_by_configuration.txt

            run_000_seed_<seed>/
                run_results.json
                summary.txt
                trajectories.png
                convergence.png
                gradient_norms.png
                fractional_orders.png
                <objective_name>_optimization.mp4

            run_001_seed_<seed>/
                ...

The per-run files keep the detailed trajectories, losses, gradient norms,
fractional orders, optimizer-specific starting points, and runtime measurements.
The aggregate files contain the statistics needed for later plotting and
reporting, including runtime statistics.

The base two-dimensional Ackley function is

    A(x, y)
      = -20 exp(-0.2 sqrt(0.5 (x^2 + y^2)))
        - exp(0.5 [cos(2 pi x) + cos(2 pi y)])
        + 20 + e.

It has its standard global minimum at

    (0.000000, 0.000000), with A(0, 0) = 0.

This script evaluates two objective variants:

1. Standard Ackley
------------------
    f_0(x, y) = A(x, y)

2. Additive fractional/fractal Ackley
-------------------------------------
    f_add(x, y)
      = A(x, y)
        + lambda * sum_{k=0}^{K} a^k [
              cos(b^k pi x) + cos(b^k pi y)
          ]


Why this family fits the previous experiments
---------------------------------------------
The Ackley function is a standard multimodal benchmark with one known global
minimum and a broad basin structure around the center. Compared with Rastrigin,
it is usually less harsh because the oscillations are embedded in a smoother
global envelope. Compared with Ackley, it has one standard global minimum
instead of four equal minima. This makes it a useful third benchmark between
basin-selection behavior and dense periodic local traps.

Plot ordering
-------------
All trajectory, convergence, gradient-norm, fractional-order, and animation
plots order optimizers by convergence quality. The default ranking is ascending
final objective value:

    best optimizer = lowest final_loss.

Line styling
------------
The parameters TRAJECTORY_LINEWIDTH, CONVERGENCE_LINEWIDTH, GRADIENT_LINEWIDTH,
ORDER_LINEWIDTH, and ANIMATION_LINEWIDTH control line thickness. Line styles are
cycled from LINE_STYLES, and colors are drawn from COLOR_PALETTE_A.

Required local optimizer files:

    class_b_tf_fractional_optimizers_OHA.py
    class_c_tf_fractional_optimizers_SR.py
    class_minimal_fractional_adaptive_optimizers.py
    class_d_related_work_optimizers.py

Active optimizer groups
-----------------------
    1. Standard: SGD, RMSprop, Adam, Adadelta
    2. Herrera-style fractional: FSGD, FRMSprop, FAdam, FAdadelta
    3. Memory fractional: MemoryFSGD, MemoryFRMSprop, MemoryFAdam, MemoryFAdadelta
    4. Minimal adaptive memory fractional: AdaptiveMemoryFSGD,
       AdaptiveMemoryFRMSprop, AdaptiveMemoryFAdam, AdaptiveMemoryFAdadelta
    5. Related-work fractional (class_d): AdaGL, FCSGD_GL, FCAdam_GL,
       AOFGD_SGD, AOFGD_Adam
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# ---------------------------------------------------------------------
# LOCAL OPTIMIZER IMPORTS
# ---------------------------------------------------------------------

from class_b_tf_fractional_optimizers_OHA import (
    FSGD,
    FAdam,
    FAdagrad,
    FAdadelta,
    FRMSprop,
    FAdamW,
)

from class_c_tf_fractional_optimizers_SR import (
    MemoryFSGD,
    MemoryFRMSprop,
    MemoryFAdam,
    MemoryFAdadelta,
)

from class_minimal_fractional_adaptive_optimizers import (
    AdaptiveMemoryFSGD,
    AdaptiveMemoryFRMSprop,
    AdaptiveMemoryFAdam,
    AdaptiveMemoryFAdadelta,
)

from class_d_related_work_optimizers import (
    AdaGL,
    FCSGD_GL,
    FCAdam_GL,
    AOFGD_SGD,
    AOFGD_Adam,
)


# =============================================================================
# 0) GLOBAL SETTINGS
# =============================================================================

OUTPUT_ROOT = Path("ackley_comparison_fin_std_add_5_lr_vderiv_grid")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 42
DTYPE = tf.float32
tf.keras.backend.set_floatx("float32")

# Multi-run settings
#N_RUNS = 100
N_RUNS = 100

RUN_SEEDS = [1000 + i for i in range(N_RUNS)]

START_POINT_MODE = "common_radius_around_minimum"

# Legacy options kept for compatibility with older start-point modes.
START_POINT_X_RANGE = (-5.0, 5.0)
START_POINT_Y_RANGE = (-5.0, 5.0)

# Controlled radius-based start-point design.
# Each optimizer receives a different point at this exact radius around the
# selected reference minimum.
START_RADIUS_AROUND_MINIMUM = 2.5

# Deterministic angular layout of the common starting points.
# The runs that share a reference minimum are radiated around it with angles
# spread evenly over the full circle, starting from START_POINT_ANGLE_OFFSET.
# If an angle places a point outside the domain, the angle is advanced in steps
# of START_POINT_ANGLE_SEARCH_INCREMENT until a valid point is found.
START_POINT_ANGLE_OFFSET = 0.0
START_POINT_ANGLE_SEARCH_INCREMENT = 0.01

# Minimum selection:
#   "standard" -> always use the standard Ackley minimum (0, 0)
#   "random"   -> selected reproducibly from ACKLEY_MINIMA
#   integer    -> always use this minimum index
START_MINIMUM_SELECTION_MODE = "standard"

START_POINT_MAX_RESAMPLE_ATTEMPTS = 1000

SAVE_PER_RUN_PLOTS = True
SAVE_PER_RUN_ANIMATIONS = False
SAVE_AGGREGATE_JSON = True

SUCCESS_DISTANCE_THRESHOLD = 0.3

# -----------------------------------------------------------------------------
# Reference minima of each objective variant.
#
# The additive fractal perturbation moves the minima of the base surface: the
# base minima are no longer critical points of the perturbed objective. The
# constants below control how the minima of the *perturbed* objective are
# located, so that the reported distance, success and loss metrics refer to the
# objective that was actually optimized. The standard variant keeps the
# analytic base minima unchanged.
#
# The search is a deterministic, vectorized monotone gradient descent started
# from a fixed lattice of offsets inside a disc around each base minimum. It
# uses no random numbers and no external optimizer, so it reproduces exactly.
RESOLVE_PERTURBED_REFERENCE_MINIMA = True
REFERENCE_SEARCH_RADIUS = 0.25
REFERENCE_SEARCH_LATTICE_SIDE = 101
REFERENCE_DESCENT_ITERATIONS = 600
REFERENCE_DESCENT_INITIAL_STEP = 1e-4
REFERENCE_DESCENT_MIN_STEP = 1e-10
REFERENCE_DESCENT_MAX_STEP = 1e-2
REFERENCE_DESCENT_GROW = 1.2
REFERENCE_DESCENT_SHRINK = 0.5
REFERENCE_EPSILON_RADIUS = 1e-12

# "legacy"  -> ranking unchanged (mean_final_loss, ...)
# "variant" -> rank by the normalized regret against the variant's own floor
AGGREGATE_RANKING_METRIC = "legacy"

# -----------------------------------------------------------------------------
# Source of the reference minima of the perturbed variant.
#
#   "file"    -> load them from the file written by A_00b_ackley_perturbation_and_minima.py, and
#                refuse to run unless the perturbation recorded there agrees
#                with the perturbation defined in this script
#   "compute" -> resolve them in this script, without the prior stage
#
# The standard variant never loads anything: its minima are the analytic ones.
REFERENCE_MINIMA_SOURCE = "file"
REFERENCE_MINIMA_FILE = Path("ackley_perturbation_and_minima") / "ackley_perturbed_reference_minima.json"
REFERENCE_FINGERPRINT_FLOAT_TOLERANCE = 1e-9

# Written into the fingerprint comparison. These strings must match the ones in
# A_00b_ackley_perturbation_and_minima.py exactly; a difference means the two scripts
# describe different objectives.
PERTURBATION_MODE = "additive"
BASE_FORMULA = "-20 exp(-0.2 sqrt(0.5 (x^2 + y^2))) - exp(0.5 (cos(2 pi x) + cos(2 pi y))) + 20 + e"
FRACTAL_FORMULA = "sum_{k=0}^{k_max} a^k (cos(b^k pi x) + cos(b^k pi y))"
PERTURBED_FORMULA = "base + lambda * fractal"


# Visualization domain
XMIN, XMAX = -5.0, 5.0
YMIN, YMAX = -5.0, 5.0
GRID_N = 300

# Optimization settings
INITIAL_POINT = np.array([3.5, -2.5], dtype=np.float32)
MAX_STEPS = 250
GRAD_CLIP_NORM = None
STOP_IF_GRAD_NORM_BELOW = None

# Animation settings
FPS = 20
TRAIL_LENGTH = None

# Plot settings
CONTOUR_LEVELS = 80
FIGSIZE = (10, 8)

# Line thickness settings
TRAJECTORY_LINEWIDTH = 1.8
CONVERGENCE_LINEWIDTH = 1.8
GRADIENT_LINEWIDTH = 1.8
ORDER_LINEWIDTH = 1.8
ANIMATION_LINEWIDTH = 2.0

# Line style cycle
LINE_STYLES = [
    "-",
    "--",
    "-.",
    ":",
]

# Colour palette
COLOR_PALETTE_A = [
    "#589641",
    "#367DCD",
    "#874BD2",
    "#DC6923",
    "#E1AA14",
    "#2896A0",
    "#BE372D",
    "#2F7F74",
    "#C05A8A",
]

COLOR_PALETTE_B = [
    "#E7F5E0",
    "#E5F1FF",
    "#F2E8FF",
    "#FFEFE4",
    "#FFF8DC",
    "#E1F7F8",
    "#FFE8E6",
    "#E4F4F1",
    "#FCEAF2",
]

# Fractional/fractal surface parameters
FRAC_A = 0.55
FRAC_B = 2.4
FRAC_K_MAX = 8
#FRAC_LAMBDA_ADDITIVE = 0.08

FRAC_LAMBDA_ADDITIVE = 0.2

# Known standard Ackley global minimum
ACKLEY_MINIMA = np.array(
    [
        [0.0, 0.0],
    ],
    dtype=np.float32,
)

# Optimizer defaults
HERRERA_VDERIV = 0.8
MEMORY_VDERIV = 0.8
MEMORY_HISTORY_SIZE = 6
MEMORY_NORMALIZE_COEFFS = False

MINIMAL_ADAPTIVE_VDERIV = 0.9
MINIMAL_ADAPTIVE_HISTORY_SIZE = 4
MINIMAL_ADAPTIVE_NORMALIZE_COEFFS = True
MINIMAL_ADAPTIVE_INITIAL_MIX = 0.10
MINIMAL_ADAPTIVE_MIN_MIX = 0.0
MINIMAL_ADAPTIVE_MAX_MIX = 0.30
MINIMAL_ADAPTIVE_ADAPT_STEP = 0.01
MINIMAL_ADAPTIVE_MIX_EMA_GAMMA = 0.90
MINIMAL_ADAPTIVE_STABILITY_EMA_GAMMA = 0.95
MINIMAL_ADAPTIVE_STABILITY_THRESHOLD = 0.25
MINIMAL_ADAPTIVE_STABILITY_CLIP = 10.0
MINIMAL_ADAPTIVE_NORM_MATCH = True
MINIMAL_ADAPTIVE_USE_RAW_SECOND_MOMENT = True
MINIMAL_ADAPTIVE_LOSS_EMA_BETA = 0.95
MINIMAL_ADAPTIVE_LOSS_WORSEN_TOLERANCE = 1e-4
MINIMAL_ADAPTIVE_LOSS_MIX_DECREASE = 0.02
MINIMAL_ADAPTIVE_WARMUP_STEPS = 0


LR_SGD = 1e-2
LR_ADAM = 1e-2
LR_RMSPROP = 5e-3
LR_ADAGRAD = 2e-2
LR_ADADELTA = 1.0
LR_ADAMW = 5e-3

# Hyperparameter grids
# The standard optimizers are evaluated at four learning rates centered around
# the original defaults. The original learning rate remains one grid point.
STANDARD_LEARNING_RATE_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0]
STANDARD_LEARNING_RATE_GRID = {
    "SGD": [LR_SGD * m for m in STANDARD_LEARNING_RATE_MULTIPLIERS],
    "RMSprop": [LR_RMSPROP * m for m in STANDARD_LEARNING_RATE_MULTIPLIERS],
    "Adam": [LR_ADAM * m for m in STANDARD_LEARNING_RATE_MULTIPLIERS],
    "Adadelta": [LR_ADADELTA * m for m in STANDARD_LEARNING_RATE_MULTIPLIERS],
}

# Four fractional-order candidates are used for Herrera, GL-Memory,
# Adaptive-Memory, AdaGL, FCSGD_GL, and FCAdam_GL.
FRACTIONAL_ORDER_GRID = [0.75, 1.25, 1.50, 1.75]

# AOFGD adapts its order inside [AOFGD_ALPHA_MIN, AOFGD_ALPHA_MAX]. Its swept
# value is therefore the initial order alpha_init and must remain inside the
# original adaptive interval. The candidate count stays four.
AOFGD_INITIAL_ORDER_GRID = [0.75, 1.00, 1.25, 1.50]

SGD_MOMENTUM = 0.9
SGD_NESTEROV = False

RMSPROP_RHO = 0.9
RMSPROP_MOMENTUM = 0.0
RMSPROP_CENTERED = False

ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-7
ADAM_AMSGRAD = False

ADAGRAD_INITIAL_ACCUMULATOR = 0.1
ADAGRAD_EPSILON = 1e-7

ADADELTA_RHO = 0.95
ADADELTA_EPSILON = 1e-7

ADAMW_WEIGHT_DECAY = 4e-3
HERRERA_FRAC_EPSILON = 1e-6

# Related-work optimizer defaults (class_d)
# AdaGL (Chen, Zhang & Mu, 2024)
ADAGL_ALPHA = 0.9
ADAGL_HISTORY_SIZE = 10

# FCSGD_GL / FCAdam_GL (Zhou, Zhao & Huang, 2023)
FCGL_ALPHA = 0.9
FCGL_HISTORY_SIZE = 11          # 11 slots = 1 current + 10 historical, matching paper Eq. 10
FCGL_PERTURBATION_PROB = 0.5

# AOFGD (Xiang et al., 2025)
AOFGD_ALPHA_INIT = 0.9
AOFGD_ALPHA_MIN = 0.5
AOFGD_ALPHA_MAX = 1.5
AOFGD_DELTA_ALPHA = 0.02
AOFGD_CEF_THRESHOLD = 0.05
AOFGD_EMA_GAMMA = 0.95
AOFGD_FRAC_EPSILON = 1e-6
AOFGD_EPS_CEF = 1e-8

OPTIMIZER_NAMES = [
    "SGD",
    "RMSprop",
    "Adam",
    "Adadelta",

    "FSGD",
    "FRMSprop",
    "FAdam",
    "FAdadelta",

    "MemoryFSGD",
    "MemoryFRMSprop",
    "MemoryFAdam",
    "MemoryFAdadelta",

    "AdaptiveMemoryFSGD",
    "AdaptiveMemoryFRMSprop",
    "AdaptiveMemoryFAdam",
    "AdaptiveMemoryFAdadelta",
    # Related-work optimizers (class_d)
    "AdaGL",
    "FCSGD_GL",
    "FCAdam_GL",
    "AOFGD_SGD",
    "AOFGD_Adam",
]

STANDARD_OPTIMIZER_NAMES = ["SGD", "RMSprop", "Adam", "Adadelta"]


def optimizer_base_learning_rate(optimizer_name: str) -> float:
    if optimizer_name in ("SGD", "FSGD", "MemoryFSGD", "AdaptiveMemoryFSGD", "FCSGD_GL", "AOFGD_SGD"):
        return float(LR_SGD)
    if optimizer_name in ("RMSprop", "FRMSprop", "MemoryFRMSprop", "AdaptiveMemoryFRMSprop"):
        return float(LR_RMSPROP)
    if optimizer_name in ("Adam", "FAdam", "MemoryFAdam", "AdaptiveMemoryFAdam", "AdaGL", "FCAdam_GL", "AOFGD_Adam"):
        return float(LR_ADAM)
    if optimizer_name in ("Adadelta", "FAdadelta", "MemoryFAdadelta", "AdaptiveMemoryFAdadelta"):
        return float(LR_ADADELTA)
    if optimizer_name == "FAdagrad":
        return float(LR_ADAGRAD)
    if optimizer_name == "FAdamW":
        return float(LR_ADAMW)
    raise ValueError(f"No base learning rate is defined for optimizer: {optimizer_name}")


def optimizer_fractional_parameter_name(optimizer_name: str) -> str | None:
    if optimizer_name in STANDARD_OPTIMIZER_NAMES:
        return None
    if optimizer_name in ("AdaGL", "FCSGD_GL", "FCAdam_GL"):
        return "alpha"
    if optimizer_name in ("AOFGD_SGD", "AOFGD_Adam"):
        return "alpha_init"
    return "vderiv"


def format_configuration_value(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p").replace("+", "")


def build_experiment_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for optimizer_name in OPTIMIZER_NAMES:
        if optimizer_name in STANDARD_OPTIMIZER_NAMES:
            for learning_rate in STANDARD_LEARNING_RATE_GRID[optimizer_name]:
                learning_rate = float(learning_rate)
                configs.append(
                    {
                        "optimizer_name": optimizer_name,
                        "learning_rate": learning_rate,
                        "fractional_order": None,
                        "swept_parameter_name": "learning_rate",
                        "swept_parameter_value": learning_rate,
                        "configuration_label": (
                            f"{optimizer_name}__lr_{format_configuration_value(learning_rate)}"
                        ),
                    }
                )
            continue

        parameter_name = optimizer_fractional_parameter_name(optimizer_name)
        order_grid = (
            AOFGD_INITIAL_ORDER_GRID
            if optimizer_name in ("AOFGD_SGD", "AOFGD_Adam")
            else FRACTIONAL_ORDER_GRID
        )
        for fractional_order in order_grid:
            fractional_order = float(fractional_order)
            configs.append(
                {
                    "optimizer_name": optimizer_name,
                    "learning_rate": optimizer_base_learning_rate(optimizer_name),
                    "fractional_order": fractional_order,
                    "swept_parameter_name": parameter_name,
                    "swept_parameter_value": fractional_order,
                    "configuration_label": (
                        f"{optimizer_name}__{parameter_name}_{format_configuration_value(fractional_order)}"
                    ),
                }
            )

    return configs


EXPERIMENT_CONFIGS = build_experiment_configs()


# =============================================================================
# 1) REPRODUCIBILITY
# =============================================================================

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_global_seed(GLOBAL_SEED)


# =============================================================================
# 2) RUN DESIGN
# =============================================================================

def derive_optimizer_seed(base_run_seed: int, optimizer_index: int) -> int:
    """
    Derive a deterministic optimizer-specific seed.

    The seed changes with:
        - the base run seed,
        - the optimizer index.

    This gives every optimizer its own random stream while remaining
    reproducible.
    """
    return int(base_run_seed + optimizer_index * 100_000)


def derive_configuration_seed(base_run_seed: int, configuration_index: int) -> int:
    """
    Derive a deterministic configuration-specific seed.

    Every optimizer--hyperparameter configuration receives its own reproducible
    random stream, while all configurations of a run still share the same
    controlled starting point.
    """
    return int(base_run_seed + configuration_index * 100_000)


def select_reference_minimum(run_idx: int, run_seed: int) -> Tuple[int, np.ndarray]:
    """
    Select the reference minimum used for radius-based starting points.
    """
    if START_MINIMUM_SELECTION_MODE == "standard":
        minimum_idx = 0
    elif START_MINIMUM_SELECTION_MODE == "random":
        rng = np.random.default_rng(int(run_seed))
        minimum_idx = int(rng.integers(0, len(ACKLEY_MINIMA)))
    elif isinstance(START_MINIMUM_SELECTION_MODE, int):
        minimum_idx = int(START_MINIMUM_SELECTION_MODE)
        if minimum_idx < 0 or minimum_idx >= len(ACKLEY_MINIMA):
            raise ValueError(
                f"START_MINIMUM_SELECTION_MODE index out of range: {minimum_idx}"
            )
    else:
        raise ValueError(
            f"Unknown START_MINIMUM_SELECTION_MODE: {START_MINIMUM_SELECTION_MODE}"
        )

    return minimum_idx, ACKLEY_MINIMA[minimum_idx].astype(np.float32)


def common_start_angles(n_points: int, angle_offset: float = 0.0) -> List[float]:
    """
    Return n_points angles spread evenly over the full circle.

    The angles are deterministic and depend only on n_points and the offset.
    """
    if n_points <= 0:
        return []
    return [
        float(angle_offset + 2.0 * np.pi * float(i) / float(n_points))
        for i in range(n_points)
    ]


def point_inside_domain(point: np.ndarray) -> bool:
    """
    Check whether a point lies inside the plotting and optimization domain.
    """
    x, y = float(point[0]), float(point[1])
    return (XMIN <= x <= XMAX) and (YMIN <= y <= YMAX)


def point_at_angle(
    reference_minimum: np.ndarray,
    radius: float,
    theta: float,
) -> np.ndarray:
    """
    Return the point at the given radius and angle around a reference minimum.
    """
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    point = reference_minimum.astype(np.float32) + np.float32(radius) * direction
    return point.astype(np.float32)


def resolve_point_in_domain(
    reference_minimum: np.ndarray,
    radius: float,
    theta: float,
) -> Tuple[np.ndarray, float, bool]:
    """
    Return a point at the given radius whose coordinates lie inside the domain.

    The requested angle is tried first. If the resulting point lies outside the
    configured domain, the angle is advanced deterministically in steps of
    START_POINT_ANGLE_SEARCH_INCREMENT. No random draw is involved. The function
    returns the point, the angle actually used, and whether a valid angle was
    found without clipping.
    """
    for attempt in range(START_POINT_MAX_RESAMPLE_ATTEMPTS):
        candidate_theta = float(theta + float(attempt) * START_POINT_ANGLE_SEARCH_INCREMENT)
        point = point_at_angle(reference_minimum, radius, candidate_theta)

        if point_inside_domain(point):
            return point, candidate_theta, True

    fallback_point = point_at_angle(reference_minimum, radius, theta)
    clipped_point = np.array(
        [
            np.clip(fallback_point[0], XMIN, XMAX),
            np.clip(fallback_point[1], YMIN, YMAX),
        ],
        dtype=np.float32,
    )
    return clipped_point, float(theta), False


def generate_common_start_points() -> List[Dict[str, Any]]:
    """
    Build the common starting-point list once, before any optimization runs.

    The list holds exactly one starting point per run. Runs that share a
    reference minimum are radiated around it, with their angles spread evenly
    over the full circle, so the number of generated points follows directly
    from N_RUNS and the minimum assignment. Within a run the point is identical
    for every optimizer, which removes optimizer-specific initial directions
    from the comparison.
    """
    minimum_assignments = [
        select_reference_minimum(run_idx, seed)
        for run_idx, seed in enumerate(RUN_SEEDS)
    ]

    runs_per_minimum: Dict[int, List[int]] = {}
    for run_idx, (minimum_idx, _reference_minimum) in enumerate(minimum_assignments):
        runs_per_minimum.setdefault(int(minimum_idx), []).append(int(run_idx))

    angle_by_run: Dict[int, float] = {}
    for minimum_idx in sorted(runs_per_minimum.keys()):
        run_indices = runs_per_minimum[minimum_idx]
        angles = common_start_angles(len(run_indices), START_POINT_ANGLE_OFFSET)
        for position, run_idx in enumerate(run_indices):
            angle_by_run[int(run_idx)] = float(angles[position])

    common_start_points: List[Dict[str, Any]] = []

    for run_idx, (minimum_idx, reference_minimum) in enumerate(minimum_assignments):
        requested_theta = angle_by_run[int(run_idx)]

        point, theta, accepted = resolve_point_in_domain(
            reference_minimum=reference_minimum,
            radius=START_RADIUS_AROUND_MINIMUM,
            theta=requested_theta,
        )

        common_start_points.append(
            {
                "run_idx": int(run_idx),
                "seed": int(RUN_SEEDS[run_idx]),
                "selected_minimum_index": int(minimum_idx),
                "selected_minimum": reference_minimum.tolist(),
                "requested_start_angle": float(requested_theta),
                "start_angle": float(theta),
                "start_point": point.tolist(),
                "start_point_accepted": bool(accepted),
            }
        )

    return common_start_points


def sample_point_at_fixed_radius(
    reference_minimum: np.ndarray,
    radius: float,
    seed: int,
) -> Tuple[np.ndarray, float, bool]:
    """
    Sample a point at a fixed radius around a reference minimum.

    The point is

        p = reference_minimum + radius * [cos(theta), sin(theta)].

    The function resamples theta until the point lies inside the configured
    domain. It returns:
        - point,
        - theta,
        - whether the point was accepted without clipping.
    """
    rng = np.random.default_rng(int(seed))

    last_theta = 0.0
    last_point = reference_minimum.copy().astype(np.float32)

    for _ in range(START_POINT_MAX_RESAMPLE_ATTEMPTS):
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        point = reference_minimum.astype(np.float32) + np.float32(radius) * direction

        last_theta = theta
        last_point = point.astype(np.float32)

        if point_inside_domain(point):
            return point.astype(np.float32), theta, True

    clipped_point = np.array(
        [
            np.clip(last_point[0], XMIN, XMAX),
            np.clip(last_point[1], YMIN, YMAX),
        ],
        dtype=np.float32,
    )
    return clipped_point, last_theta, False


def generate_starting_points() -> List[Dict[str, Any]]:
    """
    Generate the complete run design.

    In the default mode the starting points come from the common list built by
    generate_common_start_points(): one point per run, placed at a fixed radius
    around the standard Ackley minimum, and shared by every optimizer in that
    run. The legacy per-optimizer and uniform modes are kept unchanged below.
    """
    rng = np.random.default_rng(GLOBAL_SEED)
    run_design = []

    # The common starting points are built once, before the run loop, so the
    # complete list exists before any optimization starts.
    common_start_points = (
        generate_common_start_points()
        if START_POINT_MODE == "common_radius_around_minimum"
        else None
    )

    for run_idx, seed in enumerate(RUN_SEEDS):
        if START_POINT_MODE == "common_radius_around_minimum":
            entry = common_start_points[run_idx]

            point = np.array(entry["start_point"], dtype=np.float32)
            minimum_idx = int(entry["selected_minimum_index"])
            reference_minimum = np.array(entry["selected_minimum"], dtype=np.float32)

            # Every optimizer in this run starts at exactly the same point.
            optimizer_initial_points = {
                optimizer_name: point.tolist()
                for optimizer_name in OPTIMIZER_NAMES
            }
            optimizer_start_angles = {
                optimizer_name: float(entry["start_angle"])
                for optimizer_name in OPTIMIZER_NAMES
            }
            optimizer_start_seeds = {
                optimizer_name: int(derive_optimizer_seed(seed, opt_idx))
                for opt_idx, optimizer_name in enumerate(OPTIMIZER_NAMES)
            }
            optimizer_start_accepted = {
                optimizer_name: bool(entry["start_point_accepted"])
                for optimizer_name in OPTIMIZER_NAMES
            }

            run_design.append(
                {
                    "run_idx": int(run_idx),
                    "seed": int(seed),
                    "selected_minimum_index": int(minimum_idx),
                    "selected_minimum": reference_minimum.tolist(),
                    "start_radius_around_minimum": float(START_RADIUS_AROUND_MINIMUM),
                    "common_start_point": point.tolist(),
                    "requested_start_angle": float(entry["requested_start_angle"]),
                    "start_angle": float(entry["start_angle"]),
                    "start_point_accepted": bool(entry["start_point_accepted"]),
                    "optimizer_initial_points": optimizer_initial_points,
                    "optimizer_start_angles": optimizer_start_angles,
                    "optimizer_start_seeds": optimizer_start_seeds,
                    "optimizer_start_accepted": optimizer_start_accepted,
                }
            )

        elif START_POINT_MODE == "radius_around_minimum_per_optimizer":
            minimum_idx, reference_minimum = select_reference_minimum(run_idx, seed)

            optimizer_initial_points: Dict[str, Any] = {}
            optimizer_start_angles: Dict[str, float] = {}
            optimizer_start_seeds: Dict[str, int] = {}
            optimizer_start_accepted: Dict[str, bool] = {}

            for opt_idx, optimizer_name in enumerate(OPTIMIZER_NAMES):
                optimizer_seed = derive_optimizer_seed(seed, opt_idx)
                start_seed = int(optimizer_seed + 17_000_003)

                point, theta, accepted = sample_point_at_fixed_radius(
                    reference_minimum=reference_minimum,
                    radius=START_RADIUS_AROUND_MINIMUM,
                    seed=start_seed,
                )

                optimizer_initial_points[optimizer_name] = point.tolist()
                optimizer_start_angles[optimizer_name] = float(theta)
                optimizer_start_seeds[optimizer_name] = int(start_seed)
                optimizer_start_accepted[optimizer_name] = bool(accepted)

            run_design.append(
                {
                    "run_idx": int(run_idx),
                    "seed": int(seed),
                    "selected_minimum_index": int(minimum_idx),
                    "selected_minimum": reference_minimum.tolist(),
                    "start_radius_around_minimum": float(START_RADIUS_AROUND_MINIMUM),
                    "optimizer_initial_points": optimizer_initial_points,
                    "optimizer_start_angles": optimizer_start_angles,
                    "optimizer_start_seeds": optimizer_start_seeds,
                    "optimizer_start_accepted": optimizer_start_accepted,
                }
            )

        elif START_POINT_MODE == "uniform":
            x0 = rng.uniform(START_POINT_X_RANGE[0], START_POINT_X_RANGE[1])
            y0 = rng.uniform(START_POINT_Y_RANGE[0], START_POINT_Y_RANGE[1])
            point = np.array([x0, y0], dtype=np.float32)

            optimizer_initial_points = {
                optimizer_name: point.tolist()
                for optimizer_name in OPTIMIZER_NAMES
            }

            run_design.append(
                {
                    "run_idx": int(run_idx),
                    "seed": int(seed),
                    "initial_point": point.tolist(),
                    "optimizer_initial_points": optimizer_initial_points,
                }
            )

        elif START_POINT_MODE == "fixed":
            point = INITIAL_POINT.copy().astype(np.float32)

            optimizer_initial_points = {
                optimizer_name: point.tolist()
                for optimizer_name in OPTIMIZER_NAMES
            }

            run_design.append(
                {
                    "run_idx": int(run_idx),
                    "seed": int(seed),
                    "initial_point": point.tolist(),
                    "optimizer_initial_points": optimizer_initial_points,
                }
            )

        else:
            raise ValueError(f"Unknown START_POINT_MODE: {START_POINT_MODE}")

    return run_design


def save_run_design(run_design: List[Dict[str, Any]]) -> None:
    payload = {
        "global_seed": GLOBAL_SEED,
        "n_runs": N_RUNS,
        "run_seeds": RUN_SEEDS,
        "start_point_mode": START_POINT_MODE,
        "start_point_x_range": START_POINT_X_RANGE,
        "start_point_y_range": START_POINT_Y_RANGE,
        "start_radius_around_minimum": START_RADIUS_AROUND_MINIMUM,
        "start_minimum_selection_mode": START_MINIMUM_SELECTION_MODE,
        "start_point_max_resample_attempts": START_POINT_MAX_RESAMPLE_ATTEMPTS,
        "start_point_angle_offset": START_POINT_ANGLE_OFFSET,
        "start_point_angle_search_increment": START_POINT_ANGLE_SEARCH_INCREMENT,
        "domain": {
            "xmin": XMIN,
            "xmax": XMAX,
            "ymin": YMIN,
            "ymax": YMAX,
        },
        "optimizer_names": OPTIMIZER_NAMES,
        "standard_learning_rate_grid": STANDARD_LEARNING_RATE_GRID,
        "fractional_order_grid": FRACTIONAL_ORDER_GRID,
        "aofgd_initial_order_grid": AOFGD_INITIAL_ORDER_GRID,
        "experiment_configs": EXPERIMENT_CONFIGS,
        "run_design": run_design,
    }

    with open(OUTPUT_ROOT / "run_design.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# 3) OBJECTIVE FUNCTIONS
# =============================================================================

def ackley_base_tf(xy: tf.Tensor) -> tf.Tensor:
    x = tf.cast(xy[0], DTYPE)
    y = tf.cast(xy[1], DTYPE)

    pi_t = tf.constant(np.pi, dtype=DTYPE)
    e_t = tf.constant(np.e, dtype=DTYPE)

    square_mean = tf.cast(0.5, DTYPE) * (tf.square(x) + tf.square(y))
    cosine_mean = tf.cast(0.5, DTYPE) * (
        tf.cos(tf.cast(2.0, DTYPE) * pi_t * x)
        + tf.cos(tf.cast(2.0, DTYPE) * pi_t * y)
    )

    return (
        -tf.cast(20.0, DTYPE) * tf.exp(-tf.cast(0.2, DTYPE) * tf.sqrt(square_mean))
        -tf.exp(cosine_mean)
        + tf.cast(20.0, DTYPE)
        + e_t
    )


def fractal_component_tf(
    x: tf.Tensor,
    y: tf.Tensor,
    a: float = FRAC_A,
    b: float = FRAC_B,
    k_max: int = FRAC_K_MAX,
) -> tf.Tensor:
    value = tf.constant(0.0, dtype=DTYPE)
    pi_t = tf.constant(np.pi, dtype=DTYPE)

    for k in range(k_max + 1):
        ak = tf.cast(a ** k, DTYPE)
        bk = tf.cast(b ** k, DTYPE)
        value += ak * (tf.cos(bk * pi_t * x) + tf.cos(bk * pi_t * y))

    return value


def fractional_ackley_additive_tf(xy: tf.Tensor) -> tf.Tensor:
    x = tf.cast(xy[0], DTYPE)
    y = tf.cast(xy[1], DTYPE)
    base = ackley_base_tf(xy)
    frac = fractal_component_tf(x, y)
    return base + tf.cast(FRAC_LAMBDA_ADDITIVE, DTYPE) * frac



def ackley_base_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    square_mean = 0.5 * (x ** 2 + y ** 2)
    cosine_mean = 0.5 * (np.cos(2.0 * np.pi * x) + np.cos(2.0 * np.pi * y))
    return -20.0 * np.exp(-0.2 * np.sqrt(square_mean)) - np.exp(cosine_mean) + 20.0 + np.e


def fractal_component_numpy(
    x: np.ndarray,
    y: np.ndarray,
    a: float = FRAC_A,
    b: float = FRAC_B,
    k_max: int = FRAC_K_MAX,
) -> np.ndarray:
    value = np.zeros_like(x, dtype=np.float64)

    for k in range(k_max + 1):
        ak = a ** k
        bk = b ** k
        value += ak * (np.cos(bk * np.pi * x) + np.cos(bk * np.pi * y))

    return value


def fractional_ackley_additive_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    base = ackley_base_numpy(x, y)
    frac = fractal_component_numpy(x, y)
    return base + FRAC_LAMBDA_ADDITIVE * frac




OBJECTIVE_VARIANTS: Dict[str, Dict[str, Any]] = {
    "standard_ackley": {
        "tf_fn": ackley_base_tf,
        "np_fn": ackley_base_numpy,
        "description": "Classical two-dimensional Ackley function.",
    },
    "fractional_ackley_additive": {
        "tf_fn": fractional_ackley_additive_tf,
        "np_fn": fractional_ackley_additive_numpy,
        "description": "Ackley plus additive multi-scale cosine perturbation.",
    },
}


# =============================================================================
# 3b) REFERENCE MINIMA OF EACH OBJECTIVE VARIANT
# =============================================================================
#
# The additive perturbation lambda * P_J shifts every minimum of the base
# surface and removes the exact degeneracy between them: the base minima have
# a non-zero perturbed gradient, and the perturbed local minima sit at a small
# but non-zero distance with different depths. Everything in this section
# locates the minima of the objective that is actually being optimized and
# reports the geometry explicitly, so that no metric refers to the minima of a
# different function.


def ackley_base_numpy_grad_numpy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def fractional_ackley_additive_numpy_grad_numpy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bx, by = ackley_base_numpy_grad_numpy(x, y)
    fx, fy = fractal_component_grad_numpy(x, y)
    return bx + FRAC_LAMBDA_ADDITIVE * fx, by + FRAC_LAMBDA_ADDITIVE * fy


VARIANT_NUMPY_GRADIENTS: Dict[str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = {
    "standard_ackley": ackley_base_numpy_grad_numpy,
    "fractional_ackley_additive": fractional_ackley_additive_numpy_grad_numpy,
}

# Variants whose minima differ from the analytic base minima.
PERTURBED_VARIANT_NAMES = ("fractional_ackley_additive",)

SURFACE_NAME = "ackley"
STANDARD_VARIANT_NAME = "standard_ackley"
PERTURBED_VARIANT_NAME = "fractional_ackley_additive"
PRIOR_SCRIPT_NAME = "A_00b_ackley_perturbation_and_minima.py"
REFERENCE_BASE_MINIMA = ACKLEY_MINIMA.astype(np.float64)


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


def resolve_variant_reference_minima(
    variant_name: str,
    objective_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Reference minima for one objective variant, together with the geometry
    audit of the base minima under that variant.

    For the standard variant the analytic base minima are returned unchanged.
    For a perturbed variant each base minimum is replaced by the lowest local
    minimum of the perturbed objective within REFERENCE_SEARCH_RADIUS.
    """
    np_fn = objective_info["np_fn"]
    grad_fn = VARIANT_NUMPY_GRADIENTS[variant_name]
    base_minima = ACKLEY_MINIMA.astype(np.float64)

    perturbed = (
        variant_name in PERTURBED_VARIANT_NAMES
        and RESOLVE_PERTURBED_REFERENCE_MINIMA
    )

    audit: List[Dict[str, Any]] = []
    reference_minima = []

    for index, base_point in enumerate(base_minima):
        bx = np.array(base_point[0]); by = np.array(base_point[1])
        gx, gy = fractal_component_grad_numpy(bx, by)
        pert_grad_norm = float(
            FRAC_LAMBDA_ADDITIVE * np.hypot(float(gx), float(gy))
        )
        cx, cy = fractal_component_curvature_numpy(bx, by)
        vgx, vgy = grad_fn(bx, by)

        row: Dict[str, Any] = {
            "index": int(index),
            "base_minimum": base_point.tolist(),
            "base_objective_value": float(np_fn(bx, by)),
            "variant_objective_value_at_base_minimum": float(np_fn(bx, by)),
            "perturbation_gradient_norm_at_base_minimum": pert_grad_norm,
            "variant_gradient_norm_at_base_minimum": float(
                np.hypot(float(vgx), float(vgy))
            ),
            "perturbation_curvature_at_base_minimum": [
                float(FRAC_LAMBDA_ADDITIVE * float(cx)),
                float(FRAC_LAMBDA_ADDITIVE * float(cy)),
            ],
        }

        if perturbed:
            found = deepest_minimum_near(np_fn, grad_fn, base_point)
            reference_minima.append(found["deepest_point"])
            row.update({
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
                "base_minimum_is_critical_point": bool(pert_grad_norm < 1e-8),
            })
        else:
            reference_minima.append(base_point.copy())
            row.update({
                "reference_minimum": base_point.tolist(),
                "reference_objective_value": float(np_fn(bx, by)),
                "displacement_from_base_minimum": 0.0,
                "relaxed_minimum": base_point.tolist(),
                "relaxed_objective_value": float(np_fn(bx, by)),
                "relaxed_displacement": 0.0,
                "n_search_starts": 0,
                "n_distinct_descent_endpoints": 0,
                "endpoint_value_min": float(np_fn(bx, by)),
                "endpoint_value_max": float(np_fn(bx, by)),
                "base_minimum_is_critical_point": True,
            })

        reference_minima[-1] = np.asarray(reference_minima[-1], dtype=np.float64)
        audit.append(row)

    reference_array = np.stack(reference_minima, axis=0).astype(np.float32)
    floor_value = float(min(r["reference_objective_value"] for r in audit))

    return {
        "variant_name": variant_name,
        "is_perturbed": bool(perturbed),
        "reference_minima": reference_array,
        "reference_values": [r["reference_objective_value"] for r in audit],
        "floor_value": floor_value,
        "floor_index": int(
            np.argmin([r["reference_objective_value"] for r in audit])
        ),
        "search_radius": float(REFERENCE_SEARCH_RADIUS) if perturbed else 0.0,
        "audit": audit,
    }


def _fingerprint_mismatches(
    fingerprint: Dict[str, Any],
    objective_info: Dict[str, Any],
) -> List[str]:
    """
    Compare the perturbation recorded in the reference file with the
    perturbation defined in this script. Returns a list of differences, empty
    when the two agree.
    """
    problems: List[str] = []
    tol = float(REFERENCE_FINGERPRINT_FLOAT_TOLERANCE)

    def same_string(key: str, mine: str) -> None:
        theirs = fingerprint.get(key)
        if theirs != mine:
            problems.append(f"{key}: file has {theirs!r}, this script has {mine!r}")

    def same_float(key: str, mine: float) -> None:
        theirs = fingerprint.get(key)
        if theirs is None or abs(float(theirs) - float(mine)) > tol:
            problems.append(f"{key}: file has {theirs!r}, this script has {mine!r}")

    def same_int(key: str, mine: int) -> None:
        theirs = fingerprint.get(key)
        if theirs is None or int(theirs) != int(mine):
            problems.append(f"{key}: file has {theirs!r}, this script has {mine!r}")

    same_string("surface_name", SURFACE_NAME)
    same_string("standard_variant_name", STANDARD_VARIANT_NAME)
    same_string("perturbed_variant_name", PERTURBED_VARIANT_NAME)
    same_string("perturbation_mode", PERTURBATION_MODE)
    same_string("base_formula", BASE_FORMULA)
    same_string("fractal_formula", FRACTAL_FORMULA)
    same_string("perturbed_formula", PERTURBED_FORMULA)
    same_float("frac_a", FRAC_A)
    same_float("frac_b", FRAC_B)
    same_int("frac_k_max", FRAC_K_MAX)
    same_float("frac_lambda_additive", FRAC_LAMBDA_ADDITIVE)

    # This script stores its base minima in float32, because they are fed to
    # float32 variables, while the prior stage writes float64 literals. The
    # comparison is therefore made at float32 precision; any real change of a
    # coordinate is far larger than that.
    mine_minima = np.asarray(REFERENCE_BASE_MINIMA, dtype=np.float32)
    file_minima = np.asarray(fingerprint.get("base_minima", []), dtype=np.float32)
    if file_minima.shape != mine_minima.shape:
        problems.append(
            f"base_minima: file has shape {file_minima.shape}, "
            f"this script has {mine_minima.shape}"
        )
    elif np.max(np.abs(file_minima - mine_minima)) > 1e-6:
        problems.append(
            "base_minima: largest coordinate difference "
            f"{float(np.max(np.abs(file_minima - mine_minima))):.3e}"
        )

    mine_domain = np.array([XMIN, XMAX, YMIN, YMAX], dtype=np.float64)
    file_domain = np.asarray(fingerprint.get("domain", []), dtype=np.float64)
    if file_domain.shape != mine_domain.shape:
        problems.append("domain: shape differs")
    elif np.max(np.abs(file_domain - mine_domain)) > 1e-9:
        problems.append(
            f"domain: file has {file_domain.tolist()}, "
            f"this script has {mine_domain.tolist()}"
        )

    # The decisive check: evaluate the objectives of this script at the probe
    # points of the file and compare the values. A changed formula is caught
    # here even when every constant still agrees.
    points = np.asarray(fingerprint.get("probe_points", []), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        problems.append("probe_points: missing or malformed in the reference file")
        return problems

    for key, variant_name in (
        ("probe_base_values", STANDARD_VARIANT_NAME),
        ("probe_perturbed_values", PERTURBED_VARIANT_NAME),
    ):
        stored = np.asarray(fingerprint.get(key, []), dtype=np.float64)
        if stored.shape[0] != points.shape[0]:
            problems.append(f"{key}: {stored.shape[0]} values for {points.shape[0]} points")
            continue
        mine = np.asarray(
            OBJECTIVE_VARIANTS[variant_name]["np_fn"](points[:, 0], points[:, 1]),
            dtype=np.float64,
        )
        scale = np.maximum(np.abs(stored), 1.0)
        deviation = float(np.max(np.abs(mine - stored) / scale))
        if deviation > tol:
            worst = int(np.argmax(np.abs(mine - stored) / scale))
            problems.append(
                f"{key}: largest relative deviation {deviation:.3e} at "
                f"({points[worst, 0]:+.6f}, {points[worst, 1]:+.6f}); "
                f"file {stored[worst]:+.10f}, this script {mine[worst]:+.10f}"
            )

    return problems


def load_variant_reference_minima(
    variant_name: str,
    objective_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Reference minima of one objective variant.

    The standard variant uses the analytic base minima and loads nothing. The
    perturbed variant loads the minima written by the prior stage and verifies
    that the perturbation recorded there is the perturbation of this script.
    """
    if variant_name not in PERTURBED_VARIANT_NAMES:
        np_fn = objective_info["np_fn"]
        reference = np.asarray(REFERENCE_BASE_MINIMA, dtype=np.float64)
        values = [
            float(np_fn(np.array(p[0]), np.array(p[1]))) for p in reference
        ]
        return {
            "variant_name": variant_name,
            "is_perturbed": False,
            "reference_minima": reference.astype(np.float32),
            "reference_values": values,
            "floor_value": float(min(values)),
            "floor_index": int(np.argmin(values)),
            "search_radius": 0.0,
            "source": "analytic base minima of this script",
            "audit": [],
        }

    path = Path(REFERENCE_MINIMA_FILE)
    if not path.is_file():
        raise RuntimeError(
            f"The reference minima of the perturbed variant were not found at "
            f"{path}. Run the prior stage first:\n"
            f"    python3 {PRIOR_SCRIPT_NAME}\n"
            f"It defines the perturbation, plots the standard and the perturbed "
            f"surface side by side, locates the minima of the perturbed surface "
            f"and writes them together with a fingerprint of the perturbation."
        )

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    fingerprint = payload.get("fingerprint", {})
    problems = _fingerprint_mismatches(fingerprint, objective_info)
    if problems:
        raise RuntimeError(
            "The perturbation recorded in "
            f"{path} does not match the perturbation defined in this script, "
            "so the stored minima do not belong to the objective that would be "
            "optimized here. Differences:\n  - "
            + "\n  - ".join(problems)
            + f"\n\nEither restore the settings or rerun {PRIOR_SCRIPT_NAME} "
              "with the current ones."
        )

    reference = np.asarray(payload["reference_minima"], dtype=np.float64)
    stored_values = [float(v) for v in payload["reference_values"]]

    np_fn = objective_info["np_fn"]
    recomputed = [
        float(np_fn(np.array(p[0]), np.array(p[1]))) for p in reference
    ]
    worst = max(
        abs(a - b) / max(abs(b), 1.0) for a, b in zip(recomputed, stored_values)
    ) if stored_values else 0.0
    if worst > float(REFERENCE_FINGERPRINT_FLOAT_TOLERANCE):
        raise RuntimeError(
            f"The objective of this script does not reproduce the values stored "
            f"for the reference minima in {path} (largest relative deviation "
            f"{worst:.3e}). Rerun {PRIOR_SCRIPT_NAME}."
        )

    print(
        f"  loaded {len(reference)} reference minima for {variant_name} from {path}"
    )
    print(
        f"  perturbation verified: a = {FRAC_A}, b = {FRAC_B}, "
        f"k_max = {FRAC_K_MAX}, lambda = {FRAC_LAMBDA_ADDITIVE}, "
        f"{len(fingerprint.get('probe_points', []))} probe points matched"
    )

    return {
        "variant_name": variant_name,
        "is_perturbed": True,
        "reference_minima": reference.astype(np.float32),
        "reference_values": stored_values,
        "floor_value": float(payload["floor_value"]),
        "floor_index": int(payload["floor_index"]),
        "search_radius": float(
            payload.get("search", {}).get("reference_search_radius", 0.0)
        ),
        "source": str(path),
        "audit": payload.get("audit", []),
    }


def get_variant_reference_minima(
    variant_name: str,
    objective_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Dispatch on REFERENCE_MINIMA_SOURCE."""
    if REFERENCE_MINIMA_SOURCE == "compute":
        return resolve_variant_reference_minima(variant_name, objective_info)
    if REFERENCE_MINIMA_SOURCE != "file":
        raise ValueError(
            f"Unknown REFERENCE_MINIMA_SOURCE: {REFERENCE_MINIMA_SOURCE!r}"
        )
    return load_variant_reference_minima(variant_name, objective_info)


def write_reference_minima_audit(
    reference_info: Dict[str, Any],
    objective_description: str,
    out_dir: Path,
) -> None:
    """Write the geometry audit of one variant as text and JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "reference_minima_audit.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant_name": reference_info["variant_name"],
                "is_perturbed": reference_info["is_perturbed"],
                "search_radius": reference_info["search_radius"],
                "lattice_side": REFERENCE_SEARCH_LATTICE_SIDE,
                "descent_iterations": REFERENCE_DESCENT_ITERATIONS,
                "frac_a": FRAC_A,
                "frac_b": FRAC_B,
                "frac_k_max": FRAC_K_MAX,
                "frac_lambda_additive": FRAC_LAMBDA_ADDITIVE,
                "finest_perturbation_wavelength": float(2.0 / FRAC_B ** FRAC_K_MAX),
                "reference_minima": reference_info["reference_minima"].tolist(),
                "reference_values": reference_info["reference_values"],
                "floor_value": reference_info["floor_value"],
                "floor_index": reference_info["floor_index"],
                "audit": reference_info["audit"],
            },
            f,
            indent=2,
        )

    lines: List[str] = []
    lines.append("=" * 110)
    lines.append(f"Reference minima and geometry audit: {reference_info['variant_name']}")
    lines.append("=" * 110)
    lines.append(objective_description)
    lines.append("")
    lines.append(f"is_perturbed = {reference_info['is_perturbed']}")
    lines.append(f"source = {reference_info.get('source', 'computed in this script')}")
    lines.append(
        f"perturbation: lambda = {FRAC_LAMBDA_ADDITIVE}, a = {FRAC_A}, "
        f"b = {FRAC_B}, k_max = {FRAC_K_MAX}"
    )
    lines.append(
        f"finest perturbation wavelength = {2.0 / FRAC_B ** FRAC_K_MAX:.6f}"
    )
    if reference_info["is_perturbed"]:
        lines.append(
            f"search radius = {reference_info['search_radius']}, "
            f"lattice side = {REFERENCE_SEARCH_LATTICE_SIDE}, "
            f"descent iterations = {REFERENCE_DESCENT_ITERATIONS}"
        )
    lines.append("")
    lines.append(
        "A non-zero perturbation gradient at a base minimum means that the base"
    )
    lines.append(
        "minimum is not a critical point of this variant, so the minima of this"
    )
    lines.append(
        "variant are the reference minima below and not the analytic base minima."
    )
    lines.append("")

    for row in reference_info["audit"]:
        bm = row["base_minimum"]
        rm = row["reference_minimum"]
        rl = row["relaxed_minimum"]
        cur = row["perturbation_curvature_at_base_minimum"]
        lines.append(
            f"[{row['index']}] base minimum           = ({bm[0]:+.6f}, {bm[1]:+.6f})\n"
            f"     objective at base min    = {row['variant_objective_value_at_base_minimum']:+.8f}\n"
            f"     lambda * |grad P_J|      = {row['perturbation_gradient_norm_at_base_minimum']:.6f}\n"
            f"     |grad f_variant|         = {row['variant_gradient_norm_at_base_minimum']:.6f}\n"
            f"     base min is critical     = {row['base_minimum_is_critical_point']}\n"
            f"     lambda * d2P/dx2, d2P/dy2= ({cur[0]:+.2f}, {cur[1]:+.2f})\n"
            f"     reference minimum        = ({rm[0]:+.6f}, {rm[1]:+.6f})\n"
            f"     objective at reference   = {row['reference_objective_value']:+.8f}\n"
            f"     displacement             = {row['displacement_from_base_minimum']:.6f}\n"
            f"     relaxed from base min    = ({rl[0]:+.6f}, {rl[1]:+.6f})"
            f"  value {row['relaxed_objective_value']:+.8f}"
            f"  distance {row['relaxed_displacement']:.6f}\n"
            f"     descent endpoints        = {row['n_distinct_descent_endpoints']}"
            f" distinct, values in"
            f" [{row['endpoint_value_min']:+.6f}, {row['endpoint_value_max']:+.6f}]\n"
        )

    values = reference_info["reference_values"]
    lines.append("")
    lines.append(
        f"reference values: min {min(values):+.8f}  max {max(values):+.8f}"
        f"  range {max(values) - min(values):.8f}"
    )
    lines.append(
        f"floor_value used for the gap and regret metrics = "
        f"{reference_info['floor_value']:+.8f}"
        f"  (reference minimum index {reference_info['floor_index']})"
    )
    if max(values) - min(values) > 1e-9:
        lines.append(
            "The reference values differ, so the minima of this variant are not"
        )
        lines.append(
            "equivalent and a distance-based success criterion alone does not"
        )
        lines.append("distinguish them.")

    with open(out_dir / "reference_minima_audit.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("")
    print("\n".join(lines))


def print_reference_minima_audit(reference_info: Dict[str, Any]) -> None:
    print(
        f"  reference minima for {reference_info['variant_name']}: "
        f"{reference_info['reference_minima'].tolist()}"
    )


# =============================================================================
# 4) OPTIMIZER FACTORY
# =============================================================================

def build_optimizer(
    optimizer_name: str,
    learning_rate: float | None = None,
    fractional_order: float | None = None,
) -> tf.keras.optimizers.Optimizer:
    lr = optimizer_base_learning_rate(optimizer_name) if learning_rate is None else float(learning_rate)

    if optimizer_name == "SGD":
        return tf.keras.optimizers.SGD(lr=lr, momentum=SGD_MOMENTUM, nesterov=SGD_NESTEROV)

    if optimizer_name == "Adam":
        return tf.keras.optimizers.Adam(lr=lr, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON, amsgrad=ADAM_AMSGRAD)

    if optimizer_name == "RMSprop":
        return tf.keras.optimizers.RMSprop(lr=lr, rho=RMSPROP_RHO, momentum=RMSPROP_MOMENTUM, epsilon=ADAM_EPSILON, centered=RMSPROP_CENTERED)

    if optimizer_name == "Adadelta":
        return tf.keras.optimizers.Adadelta(
            lr=lr,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
        )

    if optimizer_name == "FSGD":
        vderiv = HERRERA_VDERIV if fractional_order is None else float(fractional_order)
        return FSGD(lr=lr, momentum=SGD_MOMENTUM, nesterov=SGD_NESTEROV, vderiv=vderiv, frac_epsilon=HERRERA_FRAC_EPSILON)

    if optimizer_name == "FAdam":
        vderiv = HERRERA_VDERIV if fractional_order is None else float(fractional_order)
        return FAdam(lr=lr, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON, amsgrad=ADAM_AMSGRAD, vderiv=vderiv, frac_epsilon=HERRERA_FRAC_EPSILON)

    if optimizer_name == "FRMSprop":
        vderiv = HERRERA_VDERIV if fractional_order is None else float(fractional_order)
        return FRMSprop(lr=lr, rho=RMSPROP_RHO, momentum=RMSPROP_MOMENTUM, epsilon=ADAM_EPSILON, centered=RMSPROP_CENTERED, vderiv=vderiv, frac_epsilon=HERRERA_FRAC_EPSILON)

    if optimizer_name == "FAdagrad":
        vderiv = HERRERA_VDERIV if fractional_order is None else float(fractional_order)
        return FAdagrad(lr=lr, initial_accumulator_value=ADAGRAD_INITIAL_ACCUMULATOR, epsilon=ADAGRAD_EPSILON, vderiv=vderiv, frac_epsilon=HERRERA_FRAC_EPSILON)

    if optimizer_name == "FAdadelta":
        vderiv = HERRERA_VDERIV if fractional_order is None else float(fractional_order)
        return FAdadelta(lr=lr, rho=ADADELTA_RHO, epsilon=ADADELTA_EPSILON, vderiv=vderiv, frac_epsilon=HERRERA_FRAC_EPSILON)

    if optimizer_name == "FAdamW":
        vderiv = HERRERA_VDERIV if fractional_order is None else float(fractional_order)
        return FAdamW(lr=lr, weight_decay=ADAMW_WEIGHT_DECAY, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON, amsgrad=ADAM_AMSGRAD, vderiv=vderiv, frac_epsilon=HERRERA_FRAC_EPSILON)

    if optimizer_name == "MemoryFSGD":
        vderiv = MEMORY_VDERIV if fractional_order is None else float(fractional_order)
        return MemoryFSGD(lr=lr, momentum=SGD_MOMENTUM, nesterov=SGD_NESTEROV, vderiv=vderiv, history_size=MEMORY_HISTORY_SIZE, normalize_coefficients=MEMORY_NORMALIZE_COEFFS)

    if optimizer_name == "MemoryFRMSprop":
        vderiv = MEMORY_VDERIV if fractional_order is None else float(fractional_order)
        return MemoryFRMSprop(lr=lr, rho=RMSPROP_RHO, momentum=RMSPROP_MOMENTUM, epsilon=ADAM_EPSILON, centered=RMSPROP_CENTERED, vderiv=vderiv, history_size=MEMORY_HISTORY_SIZE, normalize_coefficients=MEMORY_NORMALIZE_COEFFS)

    if optimizer_name == "MemoryFAdam":
        vderiv = MEMORY_VDERIV if fractional_order is None else float(fractional_order)
        return MemoryFAdam(lr=lr, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON, amsgrad=ADAM_AMSGRAD, vderiv=vderiv, history_size=MEMORY_HISTORY_SIZE, normalize_coefficients=MEMORY_NORMALIZE_COEFFS)

    if optimizer_name == "MemoryFAdadelta":
        vderiv = MEMORY_VDERIV if fractional_order is None else float(fractional_order)
        return MemoryFAdadelta(lr=lr, rho=ADADELTA_RHO, epsilon=ADADELTA_EPSILON, vderiv=vderiv, history_size=MEMORY_HISTORY_SIZE, normalize_coefficients=MEMORY_NORMALIZE_COEFFS)

    if optimizer_name == "AdaptiveMemoryFSGD":
        vderiv = MINIMAL_ADAPTIVE_VDERIV if fractional_order is None else float(fractional_order)
        return AdaptiveMemoryFSGD(
            lr=lr,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            vderiv=vderiv,
            history_size=MINIMAL_ADAPTIVE_HISTORY_SIZE,
            normalize_coefficients=MINIMAL_ADAPTIVE_NORMALIZE_COEFFS,
            initial_mix=MINIMAL_ADAPTIVE_INITIAL_MIX,
            min_mix=MINIMAL_ADAPTIVE_MIN_MIX,
            max_mix=MINIMAL_ADAPTIVE_MAX_MIX,
            adapt_step=MINIMAL_ADAPTIVE_ADAPT_STEP,
            mix_ema_gamma=MINIMAL_ADAPTIVE_MIX_EMA_GAMMA,
            stability_ema_gamma=MINIMAL_ADAPTIVE_STABILITY_EMA_GAMMA,
            stability_threshold=MINIMAL_ADAPTIVE_STABILITY_THRESHOLD,
            stability_clip=MINIMAL_ADAPTIVE_STABILITY_CLIP,
            norm_match_fractional_gradient=MINIMAL_ADAPTIVE_NORM_MATCH,
            use_raw_gradient_for_second_moment=MINIMAL_ADAPTIVE_USE_RAW_SECOND_MOMENT,
            loss_ema_beta=MINIMAL_ADAPTIVE_LOSS_EMA_BETA,
            loss_worsen_tolerance=MINIMAL_ADAPTIVE_LOSS_WORSEN_TOLERANCE,
            loss_mix_decrease=MINIMAL_ADAPTIVE_LOSS_MIX_DECREASE,
            warmup_steps=MINIMAL_ADAPTIVE_WARMUP_STEPS,
        )

    if optimizer_name == "AdaptiveMemoryFRMSprop":
        vderiv = MINIMAL_ADAPTIVE_VDERIV if fractional_order is None else float(fractional_order)
        return AdaptiveMemoryFRMSprop(
            lr=lr,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
            vderiv=vderiv,
            history_size=MINIMAL_ADAPTIVE_HISTORY_SIZE,
            normalize_coefficients=MINIMAL_ADAPTIVE_NORMALIZE_COEFFS,
            initial_mix=MINIMAL_ADAPTIVE_INITIAL_MIX,
            min_mix=MINIMAL_ADAPTIVE_MIN_MIX,
            max_mix=MINIMAL_ADAPTIVE_MAX_MIX,
            adapt_step=MINIMAL_ADAPTIVE_ADAPT_STEP,
            mix_ema_gamma=MINIMAL_ADAPTIVE_MIX_EMA_GAMMA,
            stability_ema_gamma=MINIMAL_ADAPTIVE_STABILITY_EMA_GAMMA,
            stability_threshold=MINIMAL_ADAPTIVE_STABILITY_THRESHOLD,
            stability_clip=MINIMAL_ADAPTIVE_STABILITY_CLIP,
            norm_match_fractional_gradient=MINIMAL_ADAPTIVE_NORM_MATCH,
            use_raw_gradient_for_second_moment=MINIMAL_ADAPTIVE_USE_RAW_SECOND_MOMENT,
            loss_ema_beta=MINIMAL_ADAPTIVE_LOSS_EMA_BETA,
            loss_worsen_tolerance=MINIMAL_ADAPTIVE_LOSS_WORSEN_TOLERANCE,
            loss_mix_decrease=MINIMAL_ADAPTIVE_LOSS_MIX_DECREASE,
            warmup_steps=MINIMAL_ADAPTIVE_WARMUP_STEPS,
        )

    if optimizer_name == "AdaptiveMemoryFAdam":
        vderiv = MINIMAL_ADAPTIVE_VDERIV if fractional_order is None else float(fractional_order)
        return AdaptiveMemoryFAdam(
            lr=lr,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv=vderiv,
            history_size=MINIMAL_ADAPTIVE_HISTORY_SIZE,
            normalize_coefficients=MINIMAL_ADAPTIVE_NORMALIZE_COEFFS,
            initial_mix=MINIMAL_ADAPTIVE_INITIAL_MIX,
            min_mix=MINIMAL_ADAPTIVE_MIN_MIX,
            max_mix=MINIMAL_ADAPTIVE_MAX_MIX,
            adapt_step=MINIMAL_ADAPTIVE_ADAPT_STEP,
            mix_ema_gamma=MINIMAL_ADAPTIVE_MIX_EMA_GAMMA,
            stability_ema_gamma=MINIMAL_ADAPTIVE_STABILITY_EMA_GAMMA,
            stability_threshold=MINIMAL_ADAPTIVE_STABILITY_THRESHOLD,
            stability_clip=MINIMAL_ADAPTIVE_STABILITY_CLIP,
            norm_match_fractional_gradient=MINIMAL_ADAPTIVE_NORM_MATCH,
            use_raw_gradient_for_second_moment=MINIMAL_ADAPTIVE_USE_RAW_SECOND_MOMENT,
            loss_ema_beta=MINIMAL_ADAPTIVE_LOSS_EMA_BETA,
            loss_worsen_tolerance=MINIMAL_ADAPTIVE_LOSS_WORSEN_TOLERANCE,
            loss_mix_decrease=MINIMAL_ADAPTIVE_LOSS_MIX_DECREASE,
            warmup_steps=MINIMAL_ADAPTIVE_WARMUP_STEPS,
        )

    if optimizer_name == "AdaptiveMemoryFAdadelta":
        vderiv = MINIMAL_ADAPTIVE_VDERIV if fractional_order is None else float(fractional_order)
        return AdaptiveMemoryFAdadelta(
            lr=lr,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
            vderiv=vderiv,
            history_size=MINIMAL_ADAPTIVE_HISTORY_SIZE,
            normalize_coefficients=MINIMAL_ADAPTIVE_NORMALIZE_COEFFS,
            initial_mix=MINIMAL_ADAPTIVE_INITIAL_MIX,
            min_mix=MINIMAL_ADAPTIVE_MIN_MIX,
            max_mix=MINIMAL_ADAPTIVE_MAX_MIX,
            adapt_step=MINIMAL_ADAPTIVE_ADAPT_STEP,
            mix_ema_gamma=MINIMAL_ADAPTIVE_MIX_EMA_GAMMA,
            stability_ema_gamma=MINIMAL_ADAPTIVE_STABILITY_EMA_GAMMA,
            stability_threshold=MINIMAL_ADAPTIVE_STABILITY_THRESHOLD,
            stability_clip=MINIMAL_ADAPTIVE_STABILITY_CLIP,
            norm_match_fractional_gradient=MINIMAL_ADAPTIVE_NORM_MATCH,
            use_raw_gradient_for_second_moment=MINIMAL_ADAPTIVE_USE_RAW_SECOND_MOMENT,
            loss_ema_beta=MINIMAL_ADAPTIVE_LOSS_EMA_BETA,
            loss_worsen_tolerance=MINIMAL_ADAPTIVE_LOSS_WORSEN_TOLERANCE,
            loss_mix_decrease=MINIMAL_ADAPTIVE_LOSS_MIX_DECREASE,
            warmup_steps=MINIMAL_ADAPTIVE_WARMUP_STEPS,
        )

    # -------------------------------------------------------------------------
    # Related-work optimizers (class_d_related_work_optimizers)
    # -------------------------------------------------------------------------

    if optimizer_name == "AdaGL":
        alpha = ADAGL_ALPHA if fractional_order is None else float(fractional_order)
        return AdaGL(
            lr=lr,
            alpha=alpha,
            history_size=ADAGL_HISTORY_SIZE,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
        )

    if optimizer_name == "FCSGD_GL":
        alpha = FCGL_ALPHA if fractional_order is None else float(fractional_order)
        return FCSGD_GL(
            lr=lr,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            alpha=alpha,
            history_size=FCGL_HISTORY_SIZE,
            perturbation_prob=FCGL_PERTURBATION_PROB,
        )

    if optimizer_name == "FCAdam_GL":
        alpha = FCGL_ALPHA if fractional_order is None else float(fractional_order)
        return FCAdam_GL(
            lr=lr,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            alpha=alpha,
            history_size=FCGL_HISTORY_SIZE,
            perturbation_prob=FCGL_PERTURBATION_PROB,
        )

    if optimizer_name == "AOFGD_SGD":
        alpha_init = AOFGD_ALPHA_INIT if fractional_order is None else float(fractional_order)
        return AOFGD_SGD(
            lr=lr,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            alpha_init=alpha_init,
            alpha_min=AOFGD_ALPHA_MIN,
            alpha_max=AOFGD_ALPHA_MAX,
            delta_alpha=AOFGD_DELTA_ALPHA,
            cef_threshold=AOFGD_CEF_THRESHOLD,
            ema_gamma=AOFGD_EMA_GAMMA,
            frac_epsilon=AOFGD_FRAC_EPSILON,
            eps_cef=AOFGD_EPS_CEF,
        )

    if optimizer_name == "AOFGD_Adam":
        alpha_init = AOFGD_ALPHA_INIT if fractional_order is None else float(fractional_order)
        return AOFGD_Adam(
            lr=lr,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            alpha_init=alpha_init,
            alpha_min=AOFGD_ALPHA_MIN,
            alpha_max=AOFGD_ALPHA_MAX,
            delta_alpha=AOFGD_DELTA_ALPHA,
            cef_threshold=AOFGD_CEF_THRESHOLD,
            ema_gamma=AOFGD_EMA_GAMMA,
            frac_epsilon=AOFGD_FRAC_EPSILON,
            eps_cef=AOFGD_EPS_CEF,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


# =============================================================================
# 5) OPTIMIZATION
# =============================================================================

def nearest_reference_minimum(
    point: np.ndarray,
    reference_minima: np.ndarray,
) -> Tuple[int, np.ndarray, float]:
    """Nearest minimum of the variant that was actually optimized."""
    reference = np.asarray(reference_minima, dtype=np.float32)
    diffs = reference - point[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return idx, reference[idx], float(dists[idx])


def nearest_ackley_minimum(point: np.ndarray) -> Tuple[np.ndarray, float]:
    diffs = ACKLEY_MINIMA - point[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return ACKLEY_MINIMA[idx], float(dists[idx])


def optimize_surface(
    optimizer_name: str,
    objective_name: str,
    objective_tf_fn: Callable[[tf.Tensor], tf.Tensor],
    initial_point: np.ndarray,
    run_seed: int,
    max_steps: int = MAX_STEPS,
    selected_minimum_index: int | None = None,
    selected_minimum: List[float] | None = None,
    start_angle: float | None = None,
    start_radius: float | None = None,
    start_seed: int | None = None,
    start_point_accepted: bool | None = None,
    reference_info: Dict[str, Any] | None = None,
    learning_rate: float | None = None,
    fractional_order: float | None = None,
    swept_parameter_name: str | None = None,
    swept_parameter_value: float | None = None,
    configuration_label: str | None = None,
) -> Dict[str, Any]:
    set_global_seed(run_seed)

    xy = tf.Variable(initial_point.astype(np.float32), dtype=DTYPE, name=f"xy_{optimizer_name}")
    optimizer = build_optimizer(
        optimizer_name,
        learning_rate=learning_rate,
        fractional_order=fractional_order,
    )

    trajectory = []
    losses = []
    grad_norms = []
    orders = []
    lr_factors = []

    optimization_start_time = time.perf_counter()

    for step in range(max_steps):
        with tf.GradientTape() as tape:
            loss = objective_tf_fn(xy)

        grads = tape.gradient(loss, [xy])

        if GRAD_CLIP_NORM is not None:
            grads = [tf.clip_by_norm(g, GRAD_CLIP_NORM) if g is not None else g for g in grads]

        if hasattr(optimizer, "set_current_loss"):
            try:
                optimizer.set_current_loss(loss)
            except Exception:
                pass

        optimizer.apply_gradients(zip(grads, [xy]))

        xy_np = xy.numpy().copy()
        grad_np = grads[0].numpy().copy()
        grad_norm = float(np.linalg.norm(grad_np))

        trajectory.append(xy_np.tolist())
        losses.append(float(loss.numpy()))
        grad_norms.append(grad_norm)

        if hasattr(optimizer, "current_orders"):
            try:
                curr_orders = optimizer.current_orders()
                mean_order = float(np.mean([float(o.numpy()) for o in curr_orders]))
            except Exception:
                mean_order = None
        elif hasattr(optimizer, "current_alpha_values"):
            # AOFGD family exposes adaptive fractional order via current_alpha_values()
            try:
                curr_alphas = optimizer.current_alpha_values()
                mean_order = float(np.mean([float(a.numpy()) for a in curr_alphas]))
            except Exception:
                mean_order = None
        else:
            mean_order = None

        if hasattr(optimizer, "current_lr_factor"):
            try:
                lr_factor = float(optimizer.current_lr_factor())
            except Exception:
                lr_factor = None
        else:
            lr_factor = None

        orders.append(mean_order)
        lr_factors.append(lr_factor)

        if STOP_IF_GRAD_NORM_BELOW is not None and grad_norm < STOP_IF_GRAD_NORM_BELOW:
            break

    optimization_runtime_seconds = time.perf_counter() - optimization_start_time

    final_xy = np.array(trajectory[-1], dtype=np.float32)
    final_loss = float(objective_tf_fn(tf.constant(final_xy, dtype=DTYPE)).numpy())
    nearest_min, nearest_dist = nearest_ackley_minimum(final_xy)

    # Metrics that refer to the minima of the variant actually optimized.
    if reference_info is not None:
        reference_minima = reference_info["reference_minima"]
        floor_value = float(reference_info["floor_value"])
    else:
        reference_minima = ACKLEY_MINIMA
        floor_value = float("nan")

    variant_idx, variant_min, variant_dist = nearest_reference_minimum(
        final_xy, reference_minima
    )
    initial_loss = float(losses[0])
    best_loss = float(np.min(losses))
    gap_final = float(max(final_loss - floor_value, 0.0))
    gap_best = float(max(best_loss - floor_value, 0.0))
    initial_gap = float(initial_loss - floor_value)
    if np.isfinite(initial_gap) and initial_gap > 1e-12:
        regret_final = float(min(max(gap_final / initial_gap, 0.0), 1.0e12))
        regret_best = float(min(max(gap_best / initial_gap, 0.0), 1.0e12))
    else:
        regret_final = float("nan")
        regret_best = float("nan")

    return {
        "objective_name": objective_name,
        "optimizer_name": optimizer_name,
        "configuration_label": configuration_label if configuration_label is not None else optimizer_name,
        "learning_rate": None if learning_rate is None else float(learning_rate),
        "fractional_order": None if fractional_order is None else float(fractional_order),
        "swept_parameter_name": swept_parameter_name,
        "swept_parameter_value": None if swept_parameter_value is None else float(swept_parameter_value),
        "run_seed": int(run_seed),
        "trajectory": trajectory,
        "losses": losses,
        "grad_norms": grad_norms,
        "orders": orders,
        "lr_factors": lr_factors,
        "n_steps": len(trajectory),
        "optimization_runtime_seconds": float(optimization_runtime_seconds),
        "runtime_seconds": float(optimization_runtime_seconds),
        "mean_time_per_step_seconds": float(optimization_runtime_seconds / max(1, len(trajectory))),
        "initial_point": initial_point.tolist(),
        "selected_minimum_index": selected_minimum_index,
        "selected_minimum": selected_minimum,
        "start_angle": start_angle,
        "start_radius": start_radius,
        "start_seed": start_seed,
        "start_point_accepted": start_point_accepted,
        "initial_distance_to_selected_minimum": float(
            np.linalg.norm(
                initial_point.astype(np.float32)
                - np.array(selected_minimum, dtype=np.float32)
            )
        ) if selected_minimum is not None else None,
        "final_point": final_xy.tolist(),
        "final_loss": final_loss,
        "best_loss_reached": float(np.min(losses)),
        "nearest_standard_ackley_minimum": nearest_min.tolist(),
        "distance_to_nearest_standard_ackley_minimum": nearest_dist,
        "success_nearest_minimum": bool(nearest_dist <= SUCCESS_DISTANCE_THRESHOLD),
        "nearest_variant_minimum_index": variant_idx,
        "nearest_variant_minimum": np.asarray(variant_min, dtype=np.float64).tolist(),
        "distance_to_nearest_variant_minimum": variant_dist,
        "success_variant_minimum": bool(variant_dist <= SUCCESS_DISTANCE_THRESHOLD),
        "initial_loss": initial_loss,
        "variant_floor_value": floor_value,
        "gap_to_variant_floor": gap_final,
        "best_gap_to_variant_floor": gap_best,
        "normalized_regret": regret_final,
        "best_normalized_regret": regret_best,
    }


def sort_results_by_best_convergence(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        results,
        key=lambda r: (
            float(r["final_loss"]),
            float(np.min(r["losses"])) if len(r["losses"]) > 0 else np.inf,
            int(r["n_steps"]),
        ),
    )


def resolve_plot_minima(
    reference_info: Dict[str, Any] | None,
) -> Tuple[np.ndarray, str]:
    """
    Minima to mark on a surface plot. A perturbed variant is marked with its
    own minima, not with the analytic minima of the base surface.
    """
    if reference_info is None:
        return ACKLEY_MINIMA, "Standard Ackley minimum"
    if reference_info.get("is_perturbed", False):
        return (
            np.asarray(reference_info["reference_minima"], dtype=np.float32),
            "Perturbed-surface minima",
        )
    return (
        np.asarray(reference_info["reference_minima"], dtype=np.float32),
        "Standard Ackley minimum",
    )


def get_plot_style(index: int) -> Tuple[str, str]:
    color = COLOR_PALETTE_A[index % len(COLOR_PALETTE_A)]
    linestyle = LINE_STYLES[(index // len(COLOR_PALETTE_A)) % len(LINE_STYLES)]
    return color, linestyle


# =============================================================================
# 6) GRID AND PLOTS
# =============================================================================

def compute_surface_grid(objective_np_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(XMIN, XMAX, GRID_N)
    ys = np.linspace(YMIN, YMAX, GRID_N)
    X, Y = np.meshgrid(xs, ys)
    Z = objective_np_fn(X, Y)
    return X, Y, Z


def save_static_trajectory_plot(objective_name: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, results: List[Dict[str, Any]], out_path: Path, reference_info: Dict[str, Any] | None = None) -> None:
    results = sort_results_by_best_convergence(results)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    contour = ax.contourf(X, Y, Z, levels=CONTOUR_LEVELS)
    plt.colorbar(contour, ax=ax, label="Objective value")

    marker_minima, marker_label = resolve_plot_minima(reference_info)
    ax.scatter(marker_minima[:, 0], marker_minima[:, 1], marker="*", s=140, c="white", edgecolors="black", linewidths=0.8, label=marker_label, zorder=5)

    for idx, res in enumerate(results):
        color, linestyle = get_plot_style(idx)
        traj = np.array(res["trajectory"])
        ax.plot(traj[:, 0], traj[:, 1], linewidth=TRAJECTORY_LINEWIDTH, linestyle=linestyle, color=color, label=res.get("configuration_label", res["optimizer_name"]))
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=35, color=color)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=50, color=color)

    ax.set_title(f"Optimizer trajectories: {objective_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.legend(fontsize=8, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_convergence_plot(results: List[Dict[str, Any]], out_path: Path) -> None:
    results = sort_results_by_best_convergence(results)
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, res in enumerate(results):
        color, linestyle = get_plot_style(idx)
        ax.plot(res["losses"], label=res.get("configuration_label", res["optimizer_name"]), linewidth=CONVERGENCE_LINEWIDTH, linestyle=linestyle, color=color)

    ax.set_title("Objective value during optimization")
    ax.set_xlabel("Step")
    ax.set_ylabel("f(x, y)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_gradient_norm_plot(results: List[Dict[str, Any]], out_path: Path) -> None:
    results = sort_results_by_best_convergence(results)
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, res in enumerate(results):
        color, linestyle = get_plot_style(idx)
        ax.plot(res["grad_norms"], label=res.get("configuration_label", res["optimizer_name"]), linewidth=GRADIENT_LINEWIDTH, linestyle=linestyle, color=color)

    ax.set_title("Gradient norm during optimization")
    ax.set_xlabel("Step")
    ax.set_ylabel("||grad||")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_order_plot(results: List[Dict[str, Any]], out_path: Path) -> None:
    results = sort_results_by_best_convergence(results)
    fig, ax = plt.subplots(figsize=(10, 6))

    any_orders = False
    for idx, res in enumerate(results):
        color, linestyle = get_plot_style(idx)
        orders = res["orders"]
        clean_orders = [np.nan if v is None else float(v) for v in orders]
        if not np.all(np.isnan(clean_orders)):
            any_orders = True
            ax.plot(clean_orders, label=res.get("configuration_label", res["optimizer_name"]), linewidth=ORDER_LINEWIDTH, linestyle=linestyle, color=color)

    if not any_orders:
        plt.close(fig)
        return

    ax.set_title("Mean fractional order during optimization")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean order")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_animation(objective_name: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, results: List[Dict[str, Any]], out_path: Path, fps: int = FPS, reference_info: Dict[str, Any] | None = None) -> None:
    results = sort_results_by_best_convergence(results)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    contour = ax.contourf(X, Y, Z, levels=CONTOUR_LEVELS)
    plt.colorbar(contour, ax=ax, label="Objective value")

    marker_minima, _ = resolve_plot_minima(reference_info)
    ax.scatter(marker_minima[:, 0], marker_minima[:, 1], marker="*", s=140, c="white", edgecolors="black", linewidths=0.8, zorder=5)

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Optimization trajectories: {objective_name}")

    max_len = max(res["n_steps"] for res in results)

    line_handles = {}
    point_handles = {}

    for idx, res in enumerate(results):
        color, linestyle = get_plot_style(idx)
        name = res.get("configuration_label", res["optimizer_name"])
        line, = ax.plot([], [], linewidth=ANIMATION_LINEWIDTH, linestyle=linestyle, color=color, label=name)
        point, = ax.plot([], [], marker="o", markersize=5, color=color)
        line_handles[name] = line
        point_handles[name] = point

    ax.legend(fontsize=8, loc="upper right")

    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def init():
        for res in results:
            name = res.get("configuration_label", res["optimizer_name"])
            line_handles[name].set_data([], [])
            point_handles[name].set_data([], [])
        info_text.set_text("")
        return list(line_handles.values()) + list(point_handles.values()) + [info_text]

    def update(frame: int):
        text_lines = [f"step = {frame + 1}"]

        for res in results:
            name = res.get("configuration_label", res["optimizer_name"])
            traj = np.array(res["trajectory"])
            last_idx = min(frame, len(traj) - 1)

            if TRAIL_LENGTH is None:
                start_idx = 0
            else:
                start_idx = max(0, last_idx - TRAIL_LENGTH + 1)

            segment = traj[start_idx:last_idx + 1]
            line_handles[name].set_data(segment[:, 0], segment[:, 1])
            point_handles[name].set_data(traj[last_idx, 0], traj[last_idx, 1])
            text_lines.append(f"{name}: loss={res['losses'][last_idx]:.5f}")

        info_text.set_text("\n".join(text_lines))
        return list(line_handles.values()) + list(point_handles.values()) + [info_text]

    anim = FuncAnimation(fig, update, frames=max_len, init_func=init, blit=False, interval=1000 / fps)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(str(out_path), writer=writer, dpi=160)
    plt.close(fig)


# =============================================================================
# 7) EXPORT AND STATISTICS
# =============================================================================

def save_results_json(results: List[Dict[str, Any]], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def write_summary_txt(objective_name: str, objective_description: str, results: List[Dict[str, Any]], out_path: Path) -> None:
    sorted_results = sort_results_by_best_convergence(results)

    lines = []
    lines.append("=" * 100)
    lines.append(f"Objective: {objective_name}")
    lines.append("=" * 100)
    lines.append(objective_description)
    lines.append("")
    lines.append("Ranked results:")
    lines.append("")

    for rank, res in enumerate(sorted_results, start=1):
        fp = res["final_point"]
        nm = res["nearest_standard_ackley_minimum"]
        lines.append(
            f"[{rank:02d}] {res.get('configuration_label', res['optimizer_name'])}\n"
            f"     run_seed = {res['run_seed']}\n"
            f"     final_loss = {res['final_loss']:.10f}\n"
            f"     best_loss_reached = {res['best_loss_reached']:.10f}\n"
            f"     steps = {res['n_steps']}\n"
            f"     runtime_seconds = {res['runtime_seconds']:.10f}\n"
            f"     mean_time_per_step_seconds = {res['mean_time_per_step_seconds']:.10f}\n"
            f"     initial_point = ({res['initial_point'][0]:.8f}, {res['initial_point'][1]:.8f})\n"
            f"     selected_minimum_index = {res.get('selected_minimum_index')}\n"
            f"     selected_minimum = {res.get('selected_minimum')}\n"
            f"     start_radius = {res.get('start_radius')}\n"
            f"     start_angle = {res.get('start_angle')}\n"
            f"     start_seed = {res.get('start_seed')}\n"
            f"     start_point_accepted = {res.get('start_point_accepted')}\n"
            f"     initial_distance_to_selected_minimum = {res.get('initial_distance_to_selected_minimum')}\n"
            f"     final_point = ({fp[0]:.8f}, {fp[1]:.8f})\n"
            f"     nearest_standard_ackley_minimum = ({nm[0]:.8f}, {nm[1]:.8f})\n"
            f"     distance_to_nearest_standard_ackley_minimum = {res['distance_to_nearest_standard_ackley_minimum']:.10f}\n"
            f"     success_nearest_minimum = {res['success_nearest_minimum']}\n"
            f"     nearest_variant_minimum_index = {res.get('nearest_variant_minimum_index')}\n"
            f"     distance_to_nearest_variant_minimum = "
            f"{res.get('distance_to_nearest_variant_minimum', float('nan')):.10f}\n"
            f"     success_variant_minimum = {res.get('success_variant_minimum')}\n"
            f"     initial_loss = {res.get('initial_loss', float('nan')):.10f}\n"
            f"     variant_floor_value = {res.get('variant_floor_value', float('nan')):.10f}\n"
            f"     gap_to_variant_floor = {res.get('gap_to_variant_floor', float('nan')):.10f}\n"
            f"     best_gap_to_variant_floor = "
            f"{res.get('best_gap_to_variant_floor', float('nan')):.10f}\n"
            f"     normalized_regret = {res.get('normalized_regret', float('nan')):.10f}\n"
            f"     best_normalized_regret = "
            f"{res.get('best_normalized_regret', float('nan')):.10f}\n"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def aggregate_statistics(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for res in results:
        configuration_label = res.get("configuration_label", res["optimizer_name"])
        grouped.setdefault(configuration_label, []).append(res)

    # Floor of the variant actually optimized. The value found by the
    # reference search is used unless some run of the study reached a lower
    # value, in which case that value is the floor, so that the gap and the
    # normalized regret are non-negative by construction.
    declared_floors = [
        float(r["variant_floor_value"]) for r in results
        if r.get("variant_floor_value") is not None
        and np.isfinite(float(r["variant_floor_value"]))
    ]
    observed = [
        float(r["best_loss_reached"]) for r in results
        if np.isfinite(float(r["best_loss_reached"]))
    ]
    if declared_floors and observed:
        effective_floor = float(min(min(declared_floors), min(observed)))
    elif observed:
        effective_floor = float(min(observed))
    else:
        effective_floor = float("nan")

    def _regret(final_value: float, initial_value: float) -> float:
        gap = float(final_value) - effective_floor
        span = float(initial_value) - effective_floor
        if not np.isfinite(gap) or not np.isfinite(span) or span <= 1e-12:
            return float("nan")
        return float(min(max(gap / span, 0.0), 1.0e12))

    aggregate = []

    for configuration_label, rows in grouped.items():
        optimizer_name = rows[0]["optimizer_name"]
        final_losses = np.array([r["final_loss"] for r in rows], dtype=np.float64)
        best_losses = np.array([r["best_loss_reached"] for r in rows], dtype=np.float64)
        distances = np.array([r["distance_to_nearest_standard_ackley_minimum"] for r in rows], dtype=np.float64)
        steps = np.array([r["n_steps"] for r in rows], dtype=np.float64)
        runtimes = np.array([r["runtime_seconds"] for r in rows], dtype=np.float64)
        time_per_step = np.array([r["mean_time_per_step_seconds"] for r in rows], dtype=np.float64)
        successes = np.array([bool(r["success_nearest_minimum"]) for r in rows], dtype=bool)
        variant_successes = np.array(
            [bool(r.get("success_variant_minimum", False)) for r in rows], dtype=bool
        )
        variant_distances = np.array(
            [
                np.nan if r.get("distance_to_nearest_variant_minimum") is None
                else float(r["distance_to_nearest_variant_minimum"])
                for r in rows
            ],
            dtype=np.float64,
        )
        initial_losses = np.array(
            [
                np.nan if r.get("initial_loss") is None else float(r["initial_loss"])
                for r in rows
            ],
            dtype=np.float64,
        )
        gaps_final = np.array(
            [max(float(r["final_loss"]) - effective_floor, 0.0) for r in rows],
            dtype=np.float64,
        )
        gaps_best = np.array(
            [max(float(r["best_loss_reached"]) - effective_floor, 0.0) for r in rows],
            dtype=np.float64,
        )
        regrets_final = np.array(
            [_regret(r["final_loss"], r.get("initial_loss", np.nan)) for r in rows],
            dtype=np.float64,
        )
        regrets_best = np.array(
            [_regret(r["best_loss_reached"], r.get("initial_loss", np.nan)) for r in rows],
            dtype=np.float64,
        )
        start_distances = np.array(
            [
                np.nan if r.get("initial_distance_to_selected_minimum") is None
                else float(r["initial_distance_to_selected_minimum"])
                for r in rows
            ],
            dtype=np.float64,
        )

        aggregate.append(
            {
                "optimizer_name": optimizer_name,
                "configuration_label": configuration_label,
                "learning_rate": rows[0].get("learning_rate"),
                "fractional_order": rows[0].get("fractional_order"),
                "swept_parameter_name": rows[0].get("swept_parameter_name"),
                "swept_parameter_value": rows[0].get("swept_parameter_value"),
                "n_runs": int(len(rows)),
                "mean_final_loss": float(np.mean(final_losses)),
                "std_final_loss": float(np.std(final_losses, ddof=0)),
                "median_final_loss": float(np.median(final_losses)),
                "min_final_loss": float(np.min(final_losses)),
                "max_final_loss": float(np.max(final_losses)),
                "mean_best_loss_reached": float(np.mean(best_losses)),
                "std_best_loss_reached": float(np.std(best_losses, ddof=0)),
                "mean_distance_to_nearest_standard_ackley_minimum": float(np.mean(distances)),
                "std_distance_to_nearest_standard_ackley_minimum": float(np.std(distances, ddof=0)),
                "success_count": int(np.sum(successes)),
                "success_rate": float(np.mean(successes)),
                "mean_steps": float(np.mean(steps)),
                "std_steps": float(np.std(steps, ddof=0)),
                "mean_runtime_seconds": float(np.mean(runtimes)),
                "std_runtime_seconds": float(np.std(runtimes, ddof=0)),
                "median_runtime_seconds": float(np.median(runtimes)),
                "min_runtime_seconds": float(np.min(runtimes)),
                "max_runtime_seconds": float(np.max(runtimes)),
                "mean_time_per_step_seconds": float(np.mean(time_per_step)),
                "std_time_per_step_seconds": float(np.std(time_per_step, ddof=0)),
                "mean_initial_distance_to_selected_minimum": float(np.nanmean(start_distances)),
                "std_initial_distance_to_selected_minimum": float(np.nanstd(start_distances, ddof=0)),
                "variant_floor_value_used": effective_floor,
                "mean_distance_to_nearest_variant_minimum": float(np.nanmean(variant_distances)),
                "std_distance_to_nearest_variant_minimum": float(np.nanstd(variant_distances, ddof=0)),
                "variant_success_count": int(np.sum(variant_successes)),
                "variant_success_rate": float(np.mean(variant_successes)),
                "mean_initial_loss": float(np.nanmean(initial_losses)),
                "mean_gap_to_variant_floor": float(np.nanmean(gaps_final)),
                "std_gap_to_variant_floor": float(np.nanstd(gaps_final, ddof=0)),
                "median_gap_to_variant_floor": float(np.nanmedian(gaps_final)),
                "mean_best_gap_to_variant_floor": float(np.nanmean(gaps_best)),
                "mean_normalized_regret": float(np.nanmean(regrets_final)),
                "std_normalized_regret": float(np.nanstd(regrets_final, ddof=0)),
                "median_normalized_regret": float(np.nanmedian(regrets_final)),
                "mean_best_normalized_regret": float(np.nanmean(regrets_best)),
            }
        )

    if AGGREGATE_RANKING_METRIC == "variant":
        aggregate = sorted(
            aggregate,
            key=lambda r: (
                r["mean_normalized_regret"],
                r["mean_gap_to_variant_floor"],
                -r["variant_success_rate"],
                r["mean_distance_to_nearest_variant_minimum"],
            ),
        )
    else:
        aggregate = sorted(
            aggregate,
            key=lambda r: (
                r["mean_final_loss"],
                r["mean_best_loss_reached"],
                -r["success_rate"],
                r["mean_distance_to_nearest_standard_ackley_minimum"],
            ),
        )

    return aggregate


def best_configuration_per_optimizer(aggregate: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in aggregate:
        optimizer_name = row["optimizer_name"]
        if optimizer_name not in best:
            best[optimizer_name] = row
    return list(best.values())


def write_aggregate_summary_txt(objective_name: str, aggregate: List[Dict[str, Any]], out_path: Path) -> None:
    lines = []
    lines.append("=" * 120)
    lines.append(f"Aggregate statistics: {objective_name}")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"N_RUNS = {N_RUNS}")
    lines.append(f"N_EXPERIMENT_CONFIGS = {len(EXPERIMENT_CONFIGS)}")
    lines.append(f"STANDARD_LEARNING_RATE_GRID = {STANDARD_LEARNING_RATE_GRID}")
    lines.append(f"FRACTIONAL_ORDER_GRID = {FRACTIONAL_ORDER_GRID}")
    lines.append(f"AOFGD_INITIAL_ORDER_GRID = {AOFGD_INITIAL_ORDER_GRID}")
    lines.append(f"SUCCESS_DISTANCE_THRESHOLD = {SUCCESS_DISTANCE_THRESHOLD}")
    lines.append("")
    if aggregate:
        lines.append(
            f"VARIANT_FLOOR_VALUE_USED = "
            f"{aggregate[0].get('variant_floor_value_used', float('nan')):.10f}"
        )
    lines.append(f"AGGREGATE_RANKING_METRIC = {AGGREGATE_RANKING_METRIC}")
    lines.append("")
    lines.append("Ranking rule:")
    lines.append("  mean_final_loss ascending")
    lines.append("  then mean_best_loss_reached ascending")
    lines.append("  then success_rate descending")
    lines.append("  then mean distance to nearest standard Ackley minimum ascending")
    lines.append("")
    lines.append("Runtime statistics:")
    lines.append("  Runtime is measured with time.perf_counter() inside optimize_surface().")
    lines.append("  The measured interval covers the complete optimizer trajectory for one optimizer,")
    lines.append("  including gradient computation, optimizer update, state extraction, and bookkeeping.")
    lines.append("")

    for rank, row in enumerate(aggregate, start=1):
        lines.append(
            f"[{rank:02d}] {row.get('configuration_label', row['optimizer_name'])}\n"
            f"     optimizer_name = {row['optimizer_name']}\n"
            f"     learning_rate = {row.get('learning_rate')}\n"
            f"     fractional_order = {row.get('fractional_order')}\n"
            f"     swept_parameter_name = {row.get('swept_parameter_name')}\n"
            f"     swept_parameter_value = {row.get('swept_parameter_value')}\n"
            f"     n_runs = {row['n_runs']}\n"
            f"     mean_final_loss = {row['mean_final_loss']:.10f}\n"
            f"     std_final_loss = {row['std_final_loss']:.10f}\n"
            f"     median_final_loss = {row['median_final_loss']:.10f}\n"
            f"     min_final_loss = {row['min_final_loss']:.10f}\n"
            f"     max_final_loss = {row['max_final_loss']:.10f}\n"
            f"     mean_best_loss_reached = {row['mean_best_loss_reached']:.10f}\n"
            f"     std_best_loss_reached = {row['std_best_loss_reached']:.10f}\n"
            f"     mean_distance_to_nearest_standard_ackley_minimum = "
            f"{row['mean_distance_to_nearest_standard_ackley_minimum']:.10f}\n"
            f"     std_distance_to_nearest_standard_ackley_minimum = "
            f"{row['std_distance_to_nearest_standard_ackley_minimum']:.10f}\n"
            f"     success_count = {row['success_count']}\n"
            f"     success_rate = {row['success_rate']:.6f}\n"
            f"     mean_steps = {row['mean_steps']:.4f}\n"
            f"     std_steps = {row['std_steps']:.4f}\n"
            f"     mean_runtime_seconds = {row['mean_runtime_seconds']:.10f}\n"
            f"     std_runtime_seconds = {row['std_runtime_seconds']:.10f}\n"
            f"     median_runtime_seconds = {row['median_runtime_seconds']:.10f}\n"
            f"     min_runtime_seconds = {row['min_runtime_seconds']:.10f}\n"
            f"     max_runtime_seconds = {row['max_runtime_seconds']:.10f}\n"
            f"     mean_time_per_step_seconds = {row['mean_time_per_step_seconds']:.10f}\n"
            f"     std_time_per_step_seconds = {row['std_time_per_step_seconds']:.10f}\n"
            f"     mean_initial_distance_to_selected_minimum = "
            f"{row.get('mean_initial_distance_to_selected_minimum', float('nan')):.10f}\n"
            f"     std_initial_distance_to_selected_minimum = "
            f"{row.get('std_initial_distance_to_selected_minimum', float('nan')):.10f}\n"
            f"     mean_distance_to_nearest_variant_minimum = "
            f"{row.get('mean_distance_to_nearest_variant_minimum', float('nan')):.10f}\n"
            f"     std_distance_to_nearest_variant_minimum = "
            f"{row.get('std_distance_to_nearest_variant_minimum', float('nan')):.10f}\n"
            f"     variant_success_count = {row.get('variant_success_count')}\n"
            f"     variant_success_rate = "
            f"{row.get('variant_success_rate', float('nan')):.6f}\n"
            f"     mean_initial_loss = {row.get('mean_initial_loss', float('nan')):.10f}\n"
            f"     mean_gap_to_variant_floor = "
            f"{row.get('mean_gap_to_variant_floor', float('nan')):.10f}\n"
            f"     std_gap_to_variant_floor = "
            f"{row.get('std_gap_to_variant_floor', float('nan')):.10f}\n"
            f"     median_gap_to_variant_floor = "
            f"{row.get('median_gap_to_variant_floor', float('nan')):.10f}\n"
            f"     mean_best_gap_to_variant_floor = "
            f"{row.get('mean_best_gap_to_variant_floor', float('nan')):.10f}\n"
            f"     mean_normalized_regret = "
            f"{row.get('mean_normalized_regret', float('nan')):.10f}\n"
            f"     std_normalized_regret = "
            f"{row.get('std_normalized_regret', float('nan')):.10f}\n"
            f"     median_normalized_regret = "
            f"{row.get('median_normalized_regret', float('nan')):.10f}\n"
            f"     mean_best_normalized_regret = "
            f"{row.get('mean_best_normalized_regret', float('nan')):.10f}\n"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_summary(objective_name: str, results: List[Dict[str, Any]]) -> None:
    print("=" * 100)
    print(f"Summary: {objective_name}")
    print("=" * 100)

    sorted_results = sort_results_by_best_convergence(results)
    for rank, res in enumerate(sorted_results, start=1):
        final_point = res["final_point"]
        print(
            f"[{rank:02d}] {res.get('configuration_label', res['optimizer_name']):45s} "
            f"final_loss={res['final_loss']:.6f} "
            f"steps={res['n_steps']:3d} "
            f"runtime={res['runtime_seconds']:.4f}s "
            f"initial_point=({res['initial_point'][0]:.5f}, {res['initial_point'][1]:.5f}) "
            f"final_point=({final_point[0]:.5f}, {final_point[1]:.5f})"
        )


# =============================================================================
# 8) MAIN
# =============================================================================

def run_single_objective(
    objective_name: str,
    objective_info: Dict[str, Any],
    run_design: List[Dict[str, Any]],
    reference_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    objective_dir = OUTPUT_ROOT / objective_name
    objective_dir.mkdir(parents=True, exist_ok=True)

    if reference_info is None:
        reference_info = get_variant_reference_minima(
            objective_name, objective_info
        )
    write_reference_minima_audit(
        reference_info, objective_info["description"], objective_dir
    )

    print(f"\nComputing grid for: {objective_name}")
    X, Y, Z = compute_surface_grid(objective_info["np_fn"])

    all_objective_results: List[Dict[str, Any]] = []

    for run_info in run_design:
        run_idx = int(run_info["run_idx"])
        run_seed = int(run_info["seed"])

        run_dir = objective_dir / f"run_{run_idx:03d}_seed_{run_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        selected_minimum_index = run_info.get("selected_minimum_index", None)
        selected_minimum = run_info.get("selected_minimum", None)

        if selected_minimum is not None:
            print(
                f"\n[OBJECTIVE: {objective_name}] "
                f"[RUN {run_idx + 1}/{len(run_design)}] "
                f"seed={run_seed} selected_minimum={selected_minimum}"
            )
        else:
            print(
                f"\n[OBJECTIVE: {objective_name}] "
                f"[RUN {run_idx + 1}/{len(run_design)}] "
                f"seed={run_seed}"
            )

        run_results: List[Dict[str, Any]] = []

        for config_idx, config in enumerate(EXPERIMENT_CONFIGS):
            optimizer_name = config["optimizer_name"]
            optimizer_seed = derive_configuration_seed(run_seed, config_idx)

            initial_point = np.array(
                run_info["optimizer_initial_points"][optimizer_name],
                dtype=np.float32,
            )

            start_angle = None
            if "optimizer_start_angles" in run_info:
                start_angle = run_info["optimizer_start_angles"].get(optimizer_name, None)

            start_seed = None
            if "optimizer_start_seeds" in run_info:
                start_seed = run_info["optimizer_start_seeds"].get(optimizer_name, None)

            start_point_accepted = None
            if "optimizer_start_accepted" in run_info:
                start_point_accepted = run_info["optimizer_start_accepted"].get(optimizer_name, None)

            print(
                f"  -> {config['configuration_label']} "
                f"optimizer_seed={optimizer_seed} "
                f"initial={initial_point.tolist()}"
            )

            result = optimize_surface(
                optimizer_name=optimizer_name,
                objective_name=objective_name,
                objective_tf_fn=objective_info["tf_fn"],
                initial_point=initial_point,
                run_seed=optimizer_seed,
                max_steps=MAX_STEPS,
                selected_minimum_index=selected_minimum_index,
                selected_minimum=selected_minimum,
                start_angle=start_angle,
                start_radius=START_RADIUS_AROUND_MINIMUM if START_POINT_MODE in ("common_radius_around_minimum", "radius_around_minimum_per_optimizer") else None,
                start_seed=start_seed,
                start_point_accepted=start_point_accepted,
                reference_info=reference_info,
                learning_rate=config["learning_rate"],
                fractional_order=config["fractional_order"],
                swept_parameter_name=config["swept_parameter_name"],
                swept_parameter_value=config["swept_parameter_value"],
                configuration_label=config["configuration_label"],
            )
            result["run_idx"] = run_idx
            result["base_run_seed"] = run_seed
            result["optimizer_seed"] = optimizer_seed
            result["configuration_index"] = config_idx
            run_results.append(result)
            all_objective_results.append(result)

        run_results = sort_results_by_best_convergence(run_results)

        save_results_json(run_results, run_dir / "run_results.json")
        write_summary_txt(objective_name, objective_info["description"], run_results, run_dir / "summary.txt")

        if SAVE_PER_RUN_PLOTS:
            save_static_trajectory_plot(objective_name, X, Y, Z, run_results, run_dir / "trajectories.png", reference_info=reference_info)
            save_convergence_plot(run_results, run_dir / "convergence.png")
            save_gradient_norm_plot(run_results, run_dir / "gradient_norms.png")
            save_order_plot(run_results, run_dir / "fractional_orders.png")

        if SAVE_PER_RUN_ANIMATIONS:
            save_animation(objective_name, X, Y, Z, run_results, run_dir / f"{objective_name}_optimization.mp4", fps=FPS, reference_info=reference_info)

    aggregate = aggregate_statistics(all_objective_results)
    best_by_optimizer = best_configuration_per_optimizer(aggregate)

    if SAVE_AGGREGATE_JSON:
        save_results_json(all_objective_results, objective_dir / "all_run_results.json")
        save_results_json(all_objective_results, objective_dir / "all_run_results_lr_vderiv_grid.json")
        with open(objective_dir / "aggregate_statistics.json", "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)
        with open(objective_dir / "aggregate_statistics_by_configuration.json", "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)
        with open(objective_dir / "best_configuration_per_optimizer.json", "w", encoding="utf-8") as f:
            json.dump(best_by_optimizer, f, indent=2)

    write_aggregate_summary_txt(objective_name, aggregate, objective_dir / "aggregate_summary.txt")
    write_aggregate_summary_txt(objective_name, aggregate, objective_dir / "aggregate_summary_by_configuration.txt")

    return {
        "objective_name": objective_name,
        "all_run_results": all_objective_results,
        "aggregate_statistics": aggregate,
        "best_configuration_per_optimizer": best_by_optimizer,
        "reference_info": reference_info,
    }


def main() -> None:
    run_design = generate_starting_points()
    save_run_design(run_design)

    all_results: Dict[str, Any] = {}
    all_aggregate_statistics: Dict[str, Any] = {}
    all_best_configurations: Dict[str, Any] = {}
    all_reference_minima: Dict[str, Any] = {}

    print("\nReference minima of each objective variant"
          f" (REFERENCE_MINIMA_SOURCE = {REFERENCE_MINIMA_SOURCE})")
    reference_by_variant: Dict[str, Dict[str, Any]] = {}
    for objective_name, objective_info in OBJECTIVE_VARIANTS.items():
        reference_by_variant[objective_name] = get_variant_reference_minima(
            objective_name, objective_info
        )
        print_reference_minima_audit(reference_by_variant[objective_name])

    for objective_name, objective_info in OBJECTIVE_VARIANTS.items():
        reference_info = reference_by_variant[objective_name]
        objective_result = run_single_objective(
            objective_name, objective_info, run_design, reference_info=reference_info
        )
        all_results[objective_name] = objective_result["all_run_results"]
        all_aggregate_statistics[objective_name] = objective_result["aggregate_statistics"]
        all_best_configurations[objective_name] = objective_result["best_configuration_per_optimizer"]
        all_reference_minima[objective_name] = {
            "is_perturbed": reference_info["is_perturbed"],
            "search_radius": reference_info["search_radius"],
            "source": reference_info.get("source", "computed in this script"),
            "reference_minima": reference_info["reference_minima"].tolist(),
            "reference_values": reference_info["reference_values"],
            "floor_value": reference_info["floor_value"],
            "floor_index": reference_info["floor_index"],
            "audit": reference_info["audit"],
        }

    with open(OUTPUT_ROOT / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open(OUTPUT_ROOT / "all_aggregate_statistics.json", "w", encoding="utf-8") as f:
        json.dump(all_aggregate_statistics, f, indent=2)

    with open(OUTPUT_ROOT / "all_best_configuration_per_optimizer.json", "w", encoding="utf-8") as f:
        json.dump(all_best_configurations, f, indent=2)

    with open(OUTPUT_ROOT / "all_reference_minima.json", "w", encoding="utf-8") as f:
        json.dump(all_reference_minima, f, indent=2)

    print("\nDone.")
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
