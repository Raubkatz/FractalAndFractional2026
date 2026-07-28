#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fractional_himmelblau_optimizer_comparison.py

This script compares TensorFlow optimizers on several Himmelblau-type objective
surfaces, including fractional/fractal perturbation variants.

Multi-run experimental design
-----------------------------
This version extends the single-run comparison to many repeated runs. Each run
uses:

    1. a reproducible base run seed,
    2. a reproducible optimizer-specific seed,
    3. a different starting point for each optimizer,
    4. a fixed-radius starting point around a selected Himmelblau minimum.

The starting-point design is intentionally controlled. For each run, one
reference Himmelblau minimum is selected. For each optimizer, a different random
angle is drawn around that same reference minimum. The starting point is then

    p_0 = m_j + r [cos(theta), sin(theta)],

where m_j is the selected minimum and r is START_RADIUS_AROUND_MINIMUM. This
means that all optimizers start at the same distance from the same minimum
within a run, but not at the same coordinates. This avoids identical trajectories
while keeping the initial difficulty comparable.

The point is accepted only if it lies inside the plotting and optimization
domain [XMIN, XMAX] x [YMIN, YMAX]. If a sampled angle gives a point outside the
domain, the script resamples the angle up to START_POINT_MAX_RESAMPLE_ATTEMPTS
times. If no valid angle is found, it falls back to a clipped point in the
direction of the sampled angle.

This allows later analysis of:

    - mean final loss,
    - standard deviation of final loss,
    - best and worst final loss,
    - mean best loss reached during the trajectory,
    - mean distance to the nearest standard Himmelblau minimum,
    - success rate according to a distance threshold,
    - average number of steps,
    - wall-clock runtime per optimizer run,
    - aggregate runtime statistics across runs,
    - stability across optimizer-specific starting points.

The starting points are generated once from GLOBAL_SEED, RUN_SEEDS, optimizer
indices, and the selected minimum index. They are saved in run_design.json and
then reused for all objective variants. This keeps the experiment reproducible
while giving each optimizer its own controlled starting point.

Output structure
----------------
The script creates:

    fractional_himmelblau_optimizer_comparison_many_runs/
        run_design.json
        all_results.json
        all_aggregate_statistics.json

        <objective_name>/
            aggregate_statistics.json
            aggregate_summary.txt

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

The base Himmelblau function is

    H(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2.

It has four known global minima:

    ( 3.000000,  2.000000)
    (-2.805118,  3.131312)
    (-3.779310, -3.283186)
    ( 3.584428, -1.848126)

This script evaluates two objective variants:

1. Standard Himmelblau
----------------------
    f_0(x, y) = H(x, y)

2. Additive fractional/fractal Himmelblau
-----------------------------------------
    f_add(x, y)
      = H(x, y)
        + lambda * sum_{k=0}^{K} a^k [
              cos(b^k pi x) + cos(b^k pi y)
          ]

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

Required local optimizer files for the active optimizer selection:

    class_b_tf_fractional_optimizers_OHA.py
    class_c_tf_fractional_optimizers_SR.py
    class_minimal_fractional_adaptive_optimizers.py
    class_d_related_work_optimizers.py

Active optimizer selection
--------------------------
The active comparison is restricted to four optimizers per level:

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

from class_e_tf_gen_var_fadam import AdaptiveModesVariableOrderMemoryFAdam

from class_tf_gen_var_frmsprop import AdaptiveRandomVariableOrderMemoryFRMSprop

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

OUTPUT_ROOT = Path("himmelblau_comparison_fin_std_add")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 42
DTYPE = tf.float32
tf.keras.backend.set_floatx("float32")

# Multi-run settings
#N_RUNS = 100
N_RUNS = 40
RUN_SEEDS = [1000 + i for i in range(N_RUNS)]

START_POINT_MODE = "radius_around_minimum_per_optimizer"

# Legacy options kept for compatibility with older start-point modes.
START_POINT_X_RANGE = (-5.5, 5.5)
START_POINT_Y_RANGE = (-5.5, 5.5)

# Controlled radius-based start-point design.
# Each optimizer receives a different point at this exact radius around the
# selected reference minimum.
START_RADIUS_AROUND_MINIMUM = 2.0

# Minimum selection:
#   "cycle"  -> run_idx cycles through the four Himmelblau minima
#   "random" -> selected reproducibly from the run seed
#   integer  -> always use this minimum index
START_MINIMUM_SELECTION_MODE = "cycle"

START_POINT_MAX_RESAMPLE_ATTEMPTS = 1000

SAVE_PER_RUN_PLOTS = True
SAVE_PER_RUN_ANIMATIONS = False
SAVE_AGGREGATE_JSON = True

SUCCESS_DISTANCE_THRESHOLD = 0.3 #0.25 before

# Visualization domain
XMIN, XMAX = -6.0, 6.0
YMIN, YMAX = -6.0, 6.0
GRID_N = 300

# Optimization settings
INITIAL_POINT = np.array([-4.5, 0.5], dtype=np.float32)
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
FRAC_B = 2.5
FRAC_K_MAX = 8
#FRAC_LAMBDA_ADDITIVE = 0.15
FRAC_LAMBDA_ADDITIVE = 0.2

# Known Himmelblau minima
HIMMELBLAU_MINIMA = np.array(
    [
        [3.0, 2.0],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584428, -1.848126],
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

VAR_ORDER_INIT = 0.85
VAR_ORDER_MIN = 0.6
VAR_ORDER_MAX = 1.2
VAR_ORDER_HISTORY_SIZE = 6
VAR_ORDER_NORMALIZE_COEFFS = False
VAR_ORDER_ADAPT_RATE = 0.05
VAR_ORDER_DELTA = 1e-8
VAR_ORDER_EMA_GAMMA = 0.95
VAR_ORDER_TOTAL_SCHEDULE_STEPS = MAX_STEPS
VAR_ORDER_SCHEDULE_EXP_GAMMA = 4.0
VAR_ORDER_STABLE_NU_CAP = 1.0
VAR_ORDER_VARIABILITY_CLIP = 10.0
VAR_ORDER_ORDER_EMA_GAMMA = 0.90
VAR_ORDER_WARMUP_STEPS = 50
VAR_ORDER_HYBRID_FINAL_FRAC_SECOND_MOMENT = 1.0
VAR_ORDER_LOSS_EMA_BETA = 0.95
VAR_ORDER_LR_DECAY_PATIENCE = 50
VAR_ORDER_LR_DECAY_FACTOR = 0.7
VAR_ORDER_MIN_LR_FACTOR = 0.2
VAR_ORDER_LOSS_PLATEAU_TOLERANCE = 1e-4

ARVFRMSPROP_NU_MIN_OFFSET = 0.25
ARVFRMSPROP_NU_MAX_OFFSET = 1.5
ARVFRMSPROP_RANDOM_PERTURB_SCALE = 0.03
ARVFRMSPROP_RANDOM_DECAY = 0.999
ARVFRMSPROP_STABILITY_STEP_SIZE = 0.01
ARVFRMSPROP_STABILITY_EMA_GAMMA = 0.95
ARVFRMSPROP_STABLE_NU_CAP = 1.0
ARVFRMSPROP_WARMUP_STEPS = 50
ARVFRMSPROP_USE_SIGNED_RANDOM_SEARCH = True

LR_SGD = 1e-2
LR_ADAM = 1e-2
LR_RMSPROP = 5e-3
LR_ADAGRAD = 2e-2
LR_ADADELTA = 1.0
LR_ADAMW = 5e-3

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


def select_reference_minimum(run_idx: int, run_seed: int) -> Tuple[int, np.ndarray]:
    """
    Select the reference minimum used for radius-based starting points.
    """
    if START_MINIMUM_SELECTION_MODE == "cycle":
        minimum_idx = int(run_idx % len(HIMMELBLAU_MINIMA))
    elif START_MINIMUM_SELECTION_MODE == "random":
        rng = np.random.default_rng(int(run_seed))
        minimum_idx = int(rng.integers(0, len(HIMMELBLAU_MINIMA)))
    elif isinstance(START_MINIMUM_SELECTION_MODE, int):
        minimum_idx = int(START_MINIMUM_SELECTION_MODE)
        if minimum_idx < 0 or minimum_idx >= len(HIMMELBLAU_MINIMA):
            raise ValueError(
                f"START_MINIMUM_SELECTION_MODE index out of range: {minimum_idx}"
            )
    else:
        raise ValueError(
            f"Unknown START_MINIMUM_SELECTION_MODE: {START_MINIMUM_SELECTION_MODE}"
        )

    return minimum_idx, HIMMELBLAU_MINIMA[minimum_idx].astype(np.float32)


def point_inside_domain(point: np.ndarray) -> bool:
    """
    Check whether a point lies inside the plotting and optimization domain.
    """
    x, y = float(point[0]), float(point[1])
    return (XMIN <= x <= XMAX) and (YMIN <= y <= YMAX)


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

    In the default mode, every optimizer receives a different starting point per
    run. All points for one run are placed at the same radius around the same
    selected Himmelblau minimum. This gives different optimizer trajectories,
    but keeps the starting difficulty comparable.
    """
    rng = np.random.default_rng(GLOBAL_SEED)
    run_design = []

    for run_idx, seed in enumerate(RUN_SEEDS):
        if START_POINT_MODE == "radius_around_minimum_per_optimizer":
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
        "domain": {
            "xmin": XMIN,
            "xmax": XMAX,
            "ymin": YMIN,
            "ymax": YMAX,
        },
        "optimizer_names": OPTIMIZER_NAMES,
        "run_design": run_design,
    }

    with open(OUTPUT_ROOT / "run_design.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# 3) OBJECTIVE FUNCTIONS
# =============================================================================

def himmelblau_base_tf(xy: tf.Tensor) -> tf.Tensor:
    x = tf.cast(xy[0], DTYPE)
    y = tf.cast(xy[1], DTYPE)
    return tf.square(x * x + y - 11.0) + tf.square(x + y * y - 7.0)


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


def fractional_himmelblau_additive_tf(xy: tf.Tensor) -> tf.Tensor:
    x = tf.cast(xy[0], DTYPE)
    y = tf.cast(xy[1], DTYPE)
    base = himmelblau_base_tf(xy)
    frac = fractal_component_tf(x, y)
    return base + tf.cast(FRAC_LAMBDA_ADDITIVE, DTYPE) * frac



def himmelblau_base_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x ** 2 + y - 11.0) ** 2 + (x + y ** 2 - 7.0) ** 2


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


def fractional_himmelblau_additive_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    base = himmelblau_base_numpy(x, y)
    frac = fractal_component_numpy(x, y)
    return base + FRAC_LAMBDA_ADDITIVE * frac



OBJECTIVE_VARIANTS: Dict[str, Dict[str, Any]] = {
    "standard_himmelblau": {
        "tf_fn": himmelblau_base_tf,
        "np_fn": himmelblau_base_numpy,
        "description": "Classical Himmelblau function.",
    },
    "fractional_himmelblau_additive": {
        "tf_fn": fractional_himmelblau_additive_tf,
        "np_fn": fractional_himmelblau_additive_numpy,
        "description": "Himmelblau plus additive multi-scale cosine perturbation.",
    },
}


# =============================================================================
# 4) OPTIMIZER FACTORY
# =============================================================================

def build_optimizer(optimizer_name: str) -> tf.keras.optimizers.Optimizer:
    if optimizer_name == "SGD":
        return tf.keras.optimizers.SGD(lr=LR_SGD, momentum=SGD_MOMENTUM, nesterov=SGD_NESTEROV)

    if optimizer_name == "Adam":
        return tf.keras.optimizers.Adam(lr=LR_ADAM, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON, amsgrad=ADAM_AMSGRAD)

    if optimizer_name == "RMSprop":
        return tf.keras.optimizers.RMSprop(lr=LR_RMSPROP, rho=RMSPROP_RHO, momentum=RMSPROP_MOMENTUM, epsilon=ADAM_EPSILON, centered=RMSPROP_CENTERED)

    if optimizer_name == "Adadelta":
        return tf.keras.optimizers.Adadelta(
            lr=LR_ADADELTA,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
        )

    if optimizer_name == "FSGD":
        return FSGD(lr=LR_SGD, momentum=SGD_MOMENTUM, nesterov=SGD_NESTEROV, vderiv=HERRERA_VDERIV, frac_epsilon=HERRERA_FRAC_EPSILON)

    if optimizer_name == "FAdam":
        return FAdam(lr=LR_ADAM, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON, amsgrad=ADAM_AMSGRAD, vderiv=HERRERA_VDERIV, frac_epsilon=HERRERA_FRAC_EPSILON)

    if optimizer_name == "FRMSprop":
        return FRMSprop(lr=LR_RMSPROP, rho=RMSPROP_RHO, momentum=RMSPROP_MOMENTUM, epsilon=ADAM_EPSILON, centered=RMSPROP_CENTERED, vderiv=HERRERA_VDERIV, frac_epsilon=HERRERA_FRAC_EPSILON)

    if optimizer_name == "FAdagrad":
        return FAdagrad(lr=LR_ADAGRAD, initial_accumulator_value=ADAGRAD_INITIAL_ACCUMULATOR, epsilon=ADAGRAD_EPSILON, vderiv=HERRERA_VDERIV, frac_epsilon=HERRERA_FRAC_EPSILON)

    if optimizer_name == "FAdadelta":
        return FAdadelta(lr=LR_ADADELTA, rho=ADADELTA_RHO, epsilon=ADADELTA_EPSILON, vderiv=HERRERA_VDERIV, frac_epsilon=HERRERA_FRAC_EPSILON)

    if optimizer_name == "FAdamW":
        return FAdamW(lr=LR_ADAMW, weight_decay=ADAMW_WEIGHT_DECAY, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON, amsgrad=ADAM_AMSGRAD, vderiv=HERRERA_VDERIV, frac_epsilon=HERRERA_FRAC_EPSILON)

    if optimizer_name == "MemoryFSGD":
        return MemoryFSGD(lr=LR_SGD, momentum=SGD_MOMENTUM, nesterov=SGD_NESTEROV, vderiv=MEMORY_VDERIV, history_size=MEMORY_HISTORY_SIZE, normalize_coefficients=MEMORY_NORMALIZE_COEFFS)

    if optimizer_name == "MemoryFRMSprop":
        return MemoryFRMSprop(lr=LR_RMSPROP, rho=RMSPROP_RHO, momentum=RMSPROP_MOMENTUM, epsilon=ADAM_EPSILON, centered=RMSPROP_CENTERED, vderiv=MEMORY_VDERIV, history_size=MEMORY_HISTORY_SIZE, normalize_coefficients=MEMORY_NORMALIZE_COEFFS)

    if optimizer_name == "MemoryFAdam":
        return MemoryFAdam(lr=LR_ADAM, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON, amsgrad=ADAM_AMSGRAD, vderiv=MEMORY_VDERIV, history_size=MEMORY_HISTORY_SIZE, normalize_coefficients=MEMORY_NORMALIZE_COEFFS)

    if optimizer_name == "MemoryFAdadelta":
        return MemoryFAdadelta(lr=LR_ADADELTA, rho=ADADELTA_RHO, epsilon=ADADELTA_EPSILON, vderiv=MEMORY_VDERIV, history_size=MEMORY_HISTORY_SIZE, normalize_coefficients=MEMORY_NORMALIZE_COEFFS)

    if optimizer_name == "AdaptiveMemoryFSGD":
        return AdaptiveMemoryFSGD(
            lr=LR_SGD,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            vderiv=MINIMAL_ADAPTIVE_VDERIV,
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
        return AdaptiveMemoryFRMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
            vderiv=MINIMAL_ADAPTIVE_VDERIV,
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
        return AdaptiveMemoryFAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv=MINIMAL_ADAPTIVE_VDERIV,
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
        return AdaptiveMemoryFAdadelta(
            lr=LR_ADADELTA,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
            vderiv=MINIMAL_ADAPTIVE_VDERIV,
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

    if optimizer_name == "AdaptiveModesVariableOrderMemoryFAdam__ema":
        return AdaptiveModesVariableOrderMemoryFAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv_init=VAR_ORDER_INIT,
            nu_min=VAR_ORDER_MIN,
            nu_max=VAR_ORDER_MAX,
            history_size=VAR_ORDER_HISTORY_SIZE,
            normalize_coefficients=VAR_ORDER_NORMALIZE_COEFFS,
            adaptation_mode="ema_smoothed_gradient_variability",
            order_adapt_rate=VAR_ORDER_ADAPT_RATE,
            order_delta=VAR_ORDER_DELTA,
            ema_smoothing_gamma=VAR_ORDER_EMA_GAMMA,
            schedule_type="cosine",
            total_schedule_steps=VAR_ORDER_TOTAL_SCHEDULE_STEPS,
            schedule_exponential_gamma=VAR_ORDER_SCHEDULE_EXP_GAMMA,
            stable_nu_cap=VAR_ORDER_STABLE_NU_CAP,
            variability_clip=VAR_ORDER_VARIABILITY_CLIP,
            order_ema_gamma=VAR_ORDER_ORDER_EMA_GAMMA,
            warmup_steps=VAR_ORDER_WARMUP_STEPS,
            hybrid_final_fractional_second_moment=VAR_ORDER_HYBRID_FINAL_FRAC_SECOND_MOMENT,
            loss_ema_beta=VAR_ORDER_LOSS_EMA_BETA,
            lr_decay_patience=VAR_ORDER_LR_DECAY_PATIENCE,
            lr_decay_factor=VAR_ORDER_LR_DECAY_FACTOR,
            min_lr_factor=VAR_ORDER_MIN_LR_FACTOR,
            loss_plateau_tolerance=VAR_ORDER_LOSS_PLATEAU_TOLERANCE,
        )

    if optimizer_name == "AdaptiveModesVariableOrderMemoryFAdam__schedule":
        return AdaptiveModesVariableOrderMemoryFAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv_init=VAR_ORDER_INIT,
            nu_min=VAR_ORDER_MIN,
            nu_max=VAR_ORDER_MAX,
            history_size=VAR_ORDER_HISTORY_SIZE,
            normalize_coefficients=VAR_ORDER_NORMALIZE_COEFFS,
            adaptation_mode="schedule",
            order_adapt_rate=VAR_ORDER_ADAPT_RATE,
            order_delta=VAR_ORDER_DELTA,
            ema_smoothing_gamma=VAR_ORDER_EMA_GAMMA,
            schedule_type="cosine",
            total_schedule_steps=VAR_ORDER_TOTAL_SCHEDULE_STEPS,
            schedule_exponential_gamma=VAR_ORDER_SCHEDULE_EXP_GAMMA,
            stable_nu_cap=VAR_ORDER_STABLE_NU_CAP,
            variability_clip=VAR_ORDER_VARIABILITY_CLIP,
            order_ema_gamma=VAR_ORDER_ORDER_EMA_GAMMA,
            warmup_steps=VAR_ORDER_WARMUP_STEPS,
            hybrid_final_fractional_second_moment=VAR_ORDER_HYBRID_FINAL_FRAC_SECOND_MOMENT,
            loss_ema_beta=VAR_ORDER_LOSS_EMA_BETA,
            lr_decay_patience=VAR_ORDER_LR_DECAY_PATIENCE,
            lr_decay_factor=VAR_ORDER_LR_DECAY_FACTOR,
            min_lr_factor=VAR_ORDER_MIN_LR_FACTOR,
            loss_plateau_tolerance=VAR_ORDER_LOSS_PLATEAU_TOLERANCE,
        )

    if optimizer_name == "AdaptiveModesVariableOrderMemoryFAdam__hybrid":
        return AdaptiveModesVariableOrderMemoryFAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv_init=VAR_ORDER_INIT,
            nu_min=VAR_ORDER_MIN,
            nu_max=VAR_ORDER_MAX,
            history_size=VAR_ORDER_HISTORY_SIZE,
            normalize_coefficients=VAR_ORDER_NORMALIZE_COEFFS,
            adaptation_mode="hybrid_transition",
            order_adapt_rate=VAR_ORDER_ADAPT_RATE,
            order_delta=VAR_ORDER_DELTA,
            ema_smoothing_gamma=VAR_ORDER_EMA_GAMMA,
            schedule_type="cosine",
            total_schedule_steps=VAR_ORDER_TOTAL_SCHEDULE_STEPS,
            schedule_exponential_gamma=VAR_ORDER_SCHEDULE_EXP_GAMMA,
            stable_nu_cap=VAR_ORDER_STABLE_NU_CAP,
            variability_clip=VAR_ORDER_VARIABILITY_CLIP,
            order_ema_gamma=VAR_ORDER_ORDER_EMA_GAMMA,
            warmup_steps=VAR_ORDER_WARMUP_STEPS,
            hybrid_final_fractional_second_moment=VAR_ORDER_HYBRID_FINAL_FRAC_SECOND_MOMENT,
            loss_ema_beta=VAR_ORDER_LOSS_EMA_BETA,
            lr_decay_patience=VAR_ORDER_LR_DECAY_PATIENCE,
            lr_decay_factor=VAR_ORDER_LR_DECAY_FACTOR,
            min_lr_factor=VAR_ORDER_MIN_LR_FACTOR,
            loss_plateau_tolerance=VAR_ORDER_LOSS_PLATEAU_TOLERANCE,
        )

    if optimizer_name == "AdaptiveRandomVariableOrderMemoryFRMSprop":
        return AdaptiveRandomVariableOrderMemoryFRMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
            vderiv_init=MEMORY_VDERIV,
            nu_min=ARVFRMSPROP_NU_MIN_OFFSET,
            nu_max=ARVFRMSPROP_NU_MAX_OFFSET,
            history_size=MEMORY_HISTORY_SIZE,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
            order_delta=VAR_ORDER_DELTA,
            random_perturb_scale=ARVFRMSPROP_RANDOM_PERTURB_SCALE,
            random_decay=ARVFRMSPROP_RANDOM_DECAY,
            stability_step_size=ARVFRMSPROP_STABILITY_STEP_SIZE,
            stability_ema_gamma=ARVFRMSPROP_STABILITY_EMA_GAMMA,
            stable_nu_cap=ARVFRMSPROP_STABLE_NU_CAP,
            warmup_steps=ARVFRMSPROP_WARMUP_STEPS,
            use_signed_random_search=ARVFRMSPROP_USE_SIGNED_RANDOM_SEARCH,
        )

    # -------------------------------------------------------------------------
    # Related-work optimizers (class_d_related_work_optimizers)
    # -------------------------------------------------------------------------

    if optimizer_name == "AdaGL":
        return AdaGL(
            lr=LR_ADAM,
            alpha=ADAGL_ALPHA,
            history_size=ADAGL_HISTORY_SIZE,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
        )

    if optimizer_name == "FCSGD_GL":
        return FCSGD_GL(
            lr=LR_SGD,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            alpha=FCGL_ALPHA,
            history_size=FCGL_HISTORY_SIZE,
            perturbation_prob=FCGL_PERTURBATION_PROB,
        )

    if optimizer_name == "FCAdam_GL":
        return FCAdam_GL(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            alpha=FCGL_ALPHA,
            history_size=FCGL_HISTORY_SIZE,
            perturbation_prob=FCGL_PERTURBATION_PROB,
        )

    if optimizer_name == "AOFGD_SGD":
        return AOFGD_SGD(
            lr=LR_SGD,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            alpha_init=AOFGD_ALPHA_INIT,
            alpha_min=AOFGD_ALPHA_MIN,
            alpha_max=AOFGD_ALPHA_MAX,
            delta_alpha=AOFGD_DELTA_ALPHA,
            cef_threshold=AOFGD_CEF_THRESHOLD,
            ema_gamma=AOFGD_EMA_GAMMA,
            frac_epsilon=AOFGD_FRAC_EPSILON,
            eps_cef=AOFGD_EPS_CEF,
        )

    if optimizer_name == "AOFGD_Adam":
        return AOFGD_Adam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            alpha_init=AOFGD_ALPHA_INIT,
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

def nearest_himmelblau_minimum(point: np.ndarray) -> Tuple[np.ndarray, float]:
    diffs = HIMMELBLAU_MINIMA - point[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return HIMMELBLAU_MINIMA[idx], float(dists[idx])


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
) -> Dict[str, Any]:
    set_global_seed(run_seed)

    xy = tf.Variable(initial_point.astype(np.float32), dtype=DTYPE, name=f"xy_{optimizer_name}")
    optimizer = build_optimizer(optimizer_name)

    trajectory = []
    losses = []
    grad_norms = []
    orders = []
    lr_factors = []
    mix_values = []

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

        if hasattr(optimizer, "current_mix_values"):
            try:
                curr_mix_values = optimizer.current_mix_values()
                mean_mix = float(np.mean([float(m.numpy()) for m in curr_mix_values]))
            except Exception:
                mean_mix = None
        else:
            mean_mix = None

        orders.append(mean_order)
        lr_factors.append(lr_factor)
        mix_values.append(mean_mix)

        if STOP_IF_GRAD_NORM_BELOW is not None and grad_norm < STOP_IF_GRAD_NORM_BELOW:
            break

    optimization_runtime_seconds = time.perf_counter() - optimization_start_time

    final_xy = np.array(trajectory[-1], dtype=np.float32)
    final_loss = float(objective_tf_fn(tf.constant(final_xy, dtype=DTYPE)).numpy())
    nearest_min, nearest_dist = nearest_himmelblau_minimum(final_xy)

    return {
        "objective_name": objective_name,
        "optimizer_name": optimizer_name,
        "run_seed": int(run_seed),
        "trajectory": trajectory,
        "losses": losses,
        "grad_norms": grad_norms,
        "orders": orders,
        "lr_factors": lr_factors,
        "mix_values": mix_values,
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
        "nearest_standard_himmelblau_minimum": nearest_min.tolist(),
        "distance_to_nearest_standard_himmelblau_minimum": nearest_dist,
        "success_nearest_minimum": bool(nearest_dist <= SUCCESS_DISTANCE_THRESHOLD),
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


def save_static_trajectory_plot(objective_name: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, results: List[Dict[str, Any]], out_path: Path) -> None:
    results = sort_results_by_best_convergence(results)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    contour = ax.contourf(X, Y, Z, levels=CONTOUR_LEVELS)
    plt.colorbar(contour, ax=ax, label="Objective value")

    ax.scatter(HIMMELBLAU_MINIMA[:, 0], HIMMELBLAU_MINIMA[:, 1], marker="*", s=140, c="white", edgecolors="black", linewidths=0.8, label="Standard Himmelblau minima", zorder=5)

    for idx, res in enumerate(results):
        color, linestyle = get_plot_style(idx)
        traj = np.array(res["trajectory"])
        ax.plot(traj[:, 0], traj[:, 1], linewidth=TRAJECTORY_LINEWIDTH, linestyle=linestyle, color=color, label=res["optimizer_name"])
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
        ax.plot(res["losses"], label=res["optimizer_name"], linewidth=CONVERGENCE_LINEWIDTH, linestyle=linestyle, color=color)

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
        ax.plot(res["grad_norms"], label=res["optimizer_name"], linewidth=GRADIENT_LINEWIDTH, linestyle=linestyle, color=color)

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
            ax.plot(clean_orders, label=res["optimizer_name"], linewidth=ORDER_LINEWIDTH, linestyle=linestyle, color=color)

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


def save_animation(objective_name: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, results: List[Dict[str, Any]], out_path: Path, fps: int = FPS) -> None:
    results = sort_results_by_best_convergence(results)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    contour = ax.contourf(X, Y, Z, levels=CONTOUR_LEVELS)
    plt.colorbar(contour, ax=ax, label="Objective value")

    ax.scatter(HIMMELBLAU_MINIMA[:, 0], HIMMELBLAU_MINIMA[:, 1], marker="*", s=140, c="white", edgecolors="black", linewidths=0.8, zorder=5)

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
        name = res["optimizer_name"]
        line, = ax.plot([], [], linewidth=ANIMATION_LINEWIDTH, linestyle=linestyle, color=color, label=name)
        point, = ax.plot([], [], marker="o", markersize=5, color=color)
        line_handles[name] = line
        point_handles[name] = point

    ax.legend(fontsize=8, loc="upper right")

    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def init():
        for res in results:
            name = res["optimizer_name"]
            line_handles[name].set_data([], [])
            point_handles[name].set_data([], [])
        info_text.set_text("")
        return list(line_handles.values()) + list(point_handles.values()) + [info_text]

    def update(frame: int):
        text_lines = [f"step = {frame + 1}"]

        for res in results:
            name = res["optimizer_name"]
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
        nm = res["nearest_standard_himmelblau_minimum"]
        lines.append(
            f"[{rank:02d}] {res['optimizer_name']}\n"
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
            f"     nearest_standard_himmelblau_minimum = ({nm[0]:.8f}, {nm[1]:.8f})\n"
            f"     distance_to_nearest_standard_himmelblau_minimum = {res['distance_to_nearest_standard_himmelblau_minimum']:.10f}\n"
            f"     success_nearest_minimum = {res['success_nearest_minimum']}\n"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def aggregate_statistics(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for res in results:
        grouped.setdefault(res["optimizer_name"], []).append(res)

    aggregate = []

    for optimizer_name, rows in grouped.items():
        final_losses = np.array([r["final_loss"] for r in rows], dtype=np.float64)
        best_losses = np.array([r["best_loss_reached"] for r in rows], dtype=np.float64)
        distances = np.array([r["distance_to_nearest_standard_himmelblau_minimum"] for r in rows], dtype=np.float64)
        steps = np.array([r["n_steps"] for r in rows], dtype=np.float64)
        runtimes = np.array([r["runtime_seconds"] for r in rows], dtype=np.float64)
        time_per_step = np.array([r["mean_time_per_step_seconds"] for r in rows], dtype=np.float64)
        successes = np.array([bool(r["success_nearest_minimum"]) for r in rows], dtype=bool)
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
                "n_runs": int(len(rows)),
                "mean_final_loss": float(np.mean(final_losses)),
                "std_final_loss": float(np.std(final_losses, ddof=0)),
                "median_final_loss": float(np.median(final_losses)),
                "min_final_loss": float(np.min(final_losses)),
                "max_final_loss": float(np.max(final_losses)),
                "mean_best_loss_reached": float(np.mean(best_losses)),
                "std_best_loss_reached": float(np.std(best_losses, ddof=0)),
                "mean_distance_to_nearest_standard_himmelblau_minimum": float(np.mean(distances)),
                "std_distance_to_nearest_standard_himmelblau_minimum": float(np.std(distances, ddof=0)),
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
            }
        )

    aggregate = sorted(
        aggregate,
        key=lambda r: (
            r["mean_final_loss"],
            r["mean_best_loss_reached"],
            -r["success_rate"],
            r["mean_distance_to_nearest_standard_himmelblau_minimum"],
        ),
    )

    return aggregate


def write_aggregate_summary_txt(objective_name: str, aggregate: List[Dict[str, Any]], out_path: Path) -> None:
    lines = []
    lines.append("=" * 120)
    lines.append(f"Aggregate statistics: {objective_name}")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"N_RUNS = {N_RUNS}")
    lines.append(f"SUCCESS_DISTANCE_THRESHOLD = {SUCCESS_DISTANCE_THRESHOLD}")
    lines.append("")
    lines.append("Ranking rule:")
    lines.append("  mean_final_loss ascending")
    lines.append("  then mean_best_loss_reached ascending")
    lines.append("  then success_rate descending")
    lines.append("  then mean distance to nearest standard Himmelblau minimum ascending")
    lines.append("")
    lines.append("Runtime statistics:")
    lines.append("  Runtime is measured with time.perf_counter() inside optimize_surface().")
    lines.append("  The measured interval covers the complete optimizer trajectory for one optimizer,")
    lines.append("  including gradient computation, optimizer update, state extraction, and bookkeeping.")
    lines.append("")

    for rank, row in enumerate(aggregate, start=1):
        lines.append(
            f"[{rank:02d}] {row['optimizer_name']}\n"
            f"     n_runs = {row['n_runs']}\n"
            f"     mean_final_loss = {row['mean_final_loss']:.10f}\n"
            f"     std_final_loss = {row['std_final_loss']:.10f}\n"
            f"     median_final_loss = {row['median_final_loss']:.10f}\n"
            f"     min_final_loss = {row['min_final_loss']:.10f}\n"
            f"     max_final_loss = {row['max_final_loss']:.10f}\n"
            f"     mean_best_loss_reached = {row['mean_best_loss_reached']:.10f}\n"
            f"     std_best_loss_reached = {row['std_best_loss_reached']:.10f}\n"
            f"     mean_distance_to_nearest_standard_himmelblau_minimum = "
            f"{row['mean_distance_to_nearest_standard_himmelblau_minimum']:.10f}\n"
            f"     std_distance_to_nearest_standard_himmelblau_minimum = "
            f"{row['std_distance_to_nearest_standard_himmelblau_minimum']:.10f}\n"
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
            f"[{rank:02d}] {res['optimizer_name']:45s} "
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
) -> Dict[str, Any]:
    objective_dir = OUTPUT_ROOT / objective_name
    objective_dir.mkdir(parents=True, exist_ok=True)

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

        for opt_idx, optimizer_name in enumerate(OPTIMIZER_NAMES):
            optimizer_seed = derive_optimizer_seed(run_seed, opt_idx)

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
                f"  -> {optimizer_name} "
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
                start_radius=START_RADIUS_AROUND_MINIMUM if START_POINT_MODE == "radius_around_minimum_per_optimizer" else None,
                start_seed=start_seed,
                start_point_accepted=start_point_accepted,
            )
            result["run_idx"] = run_idx
            result["base_run_seed"] = run_seed
            result["optimizer_seed"] = optimizer_seed
            run_results.append(result)
            all_objective_results.append(result)

        run_results = sort_results_by_best_convergence(run_results)

        save_results_json(run_results, run_dir / "run_results.json")
        write_summary_txt(objective_name, objective_info["description"], run_results, run_dir / "summary.txt")

        if SAVE_PER_RUN_PLOTS:
            save_static_trajectory_plot(objective_name, X, Y, Z, run_results, run_dir / "trajectories.png")
            save_convergence_plot(run_results, run_dir / "convergence.png")
            save_gradient_norm_plot(run_results, run_dir / "gradient_norms.png")
            save_order_plot(run_results, run_dir / "fractional_orders.png")

        if SAVE_PER_RUN_ANIMATIONS:
            save_animation(objective_name, X, Y, Z, run_results, run_dir / f"{objective_name}_optimization.mp4", fps=FPS)

    aggregate = aggregate_statistics(all_objective_results)

    if SAVE_AGGREGATE_JSON:
        save_results_json(all_objective_results, objective_dir / "all_run_results.json")
        with open(objective_dir / "aggregate_statistics.json", "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)

    write_aggregate_summary_txt(objective_name, aggregate, objective_dir / "aggregate_summary.txt")

    return {
        "objective_name": objective_name,
        "all_run_results": all_objective_results,
        "aggregate_statistics": aggregate,
    }


def main() -> None:
    run_design = generate_starting_points()
    save_run_design(run_design)

    all_results: Dict[str, Any] = {}
    all_aggregate_statistics: Dict[str, Any] = {}

    for objective_name, objective_info in OBJECTIVE_VARIANTS.items():
        objective_result = run_single_objective(objective_name, objective_info, run_design)
        all_results[objective_name] = objective_result["all_run_results"]
        all_aggregate_statistics[objective_name] = objective_result["aggregate_statistics"]

    with open(OUTPUT_ROOT / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open(OUTPUT_ROOT / "all_aggregate_statistics.json", "w", encoding="utf-8") as f:
        json.dump(all_aggregate_statistics, f, indent=2)

    print("\nDone.")
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()