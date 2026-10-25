#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
B_05_activation_optimizer_gradient_probe.py

End-to-end gradient diagnostic for every activation and every optimizer of the
classification study, written for the supplementary appendix.

WHY THIS EXISTS
---------------
B_03 and B_04 answer the reviewer's question about the high-frequency
activation analytically: they measure the value, the derivative, the resolved
scales and the non-finite counts of the activations themselves, on a fixed
sample of pre-activations, without training anything. That establishes what
each activation hands to an optimizer. It does not establish what the
optimizers of this study then do with it, and for the fractional optimizers
that is not a detail, because none of them consumes the raw gradient:

    Herrera-type (class_b)        multiplies the gradient by
                                  f_nu(theta) = (|theta| + eps)^(1-nu)/Gamma(2-nu),
                                  evaluated on the parameter. Near a parameter
                                  that passes through zero this factor grows
                                  without bound: its value at |theta| = eps is
                                  25.8, 564 and 8722 for nu = 1.25, 1.50, 1.75.
                                  An activation that already supplies gradients
                                  of order 1e15 can therefore reach the float32
                                  ceiling once that product is squared into a
                                  second-moment accumulator.

    GL-memory (class_c)           forms a weighted sum over a K-step gradient
                                  history, norm-matches it back to the current
                                  gradient magnitude and falls back to the
                                  plain gradient whenever the memory term is
                                  not a descent direction. Its premise is that
                                  consecutive gradients carry related
                                  information.

    adaptive memory (class_minimal)  mixes the memory term in with a bounded
                                  weight that adapts to observed gradient
                                  variability.

    related work (class_d)        AdaGL adapts a coefficient from consecutive
                                  gradient differences, FCSGD/FCAdam mask the
                                  history at random, AOFGD ratchets its order
                                  from a convergence estimate.

Each of these mechanisms reads something from the gradient sequence. If the
gradient sequence carries no information, as the analysis of the x+sin
activation suggests, then what these optimizers report on that activation is
the behaviour of their fallback paths, not of their mechanisms. That is a
statement the study can make only by measuring the training steps themselves,
which model.fit does not expose.

WHAT THIS SCRIPT MEASURES
-------------------------
For every (activation, truncation, optimizer, setting, seed) cell it trains the
deployed architecture with an explicit training loop and records

    gradient magnitude      mean, median, p95 and maximum global gradient norm
                            over the steps, and the number of steps at which the
                            loss or the gradient is not finite,

    update magnitude        the same statistics for the norm of the parameter
                            change actually applied, which is what the
                            optimizer did rather than what it was given,

    gradient coherence      the cosine similarity between the gradients of
                            consecutive steps. This is the quantity the memory
                            mechanisms depend on: a value near zero means a
                            weighted history has nothing to average,

    accumulator health      the number of non-finite entries in the optimizer's
                            own slot variables, checked periodically, and the
                            step at which the first one appeared. This is where
                            a Herrera factor multiplying an already large
                            gradient would show up,

    family diagnostics      the Herrera factor statistics implied by the final
                            weights, the GL coefficient sum of the memory
                            optimizers, and the realized mixing and stability
                            states of the adaptive optimizers, read through the
                            accessors those classes provide,

    outcome                 final training loss, test accuracy, and the 99th
                            percentile of the pre-activations the first layer
                            produced, which says whether the sampling range
                            used by B_04 was the right one.

TWO PROPERTIES OF THE DEPLOYED SETUP THAT THIS SCRIPT REPRODUCES
----------------------------------------------------------------
The adaptive memory optimizers adapt their mixing weight from two inputs. The
first is a stability score, the relative change between consecutive gradients,
smoothed over steps: below the stability threshold the target mix rises by
adapt_step, above it the target mix falls by the same amount. That input runs
on every step, whatever the training loop looks like. The second is a penalty
that lowers the target mix by loss_mix_decrease whenever the smoothed loss has
stopped improving, and it is supplied from outside through set_current_loss.

The surface experiments call it: A_01 and A_02 feed the loss to the optimizer
from their own training loop. The classification experiments do not, because
B_01 trains with model.fit, which has no hook for it, and the flag the penalty
reads is initialised to zero, so in those runs the second input contributed
exactly nothing while the first worked normally. The same optimizer therefore
ran with both inputs on the surfaces and with one input on the ten
classification datasets.

This script reproduces the classification setting by default, which is the one
its dataset belongs to. FEED_LOSS_TO_ADAPTIVE_OPTIMIZERS switches the surface
setting on, and the difference between the two is itself measurable here.

The stability score is also the quantity that an activation with an incoherent
gradient destroys. If consecutive gradients are effectively independent, the
score sits near sqrt(2), far above the threshold, the mix ratchets down to its
floor, and the adaptive family reduces to the plain optimizer it is built on.
The probe records the realized mix and the gradient coherence side by side, so
that this can be read off rather than assumed.

The uploaded runner carries VDERIV_GRID = [0.75, 1.25, 1.5], three orders,
while the manuscript reports 21 optimizers at four settings each. The four
value grid is used here, as section 0 states; set ORDER_GRID back to three
values to reproduce the uploaded file instead.

COST
----
The full grid is 21 optimizers at four settings, six activations, the
truncation lengths of the review plus each activation's deployed one, and
NETWORK_PROBE_RUNS seeds. The script prints the exact number of runs and a
measured estimate of the remaining time after the first few, and it writes
every finished run to the results file immediately, so an interrupted run
loses nothing: starting it again skips the cells that are already there. Seeds
are the outermost loop, so an early stop still leaves every cell covered once.
Cutting the cost is a matter of shortening ORDER_GRID, TRUNCATIONS_TO_PROBE or
OPTIMIZER_FAMILIES in section 0.

RUNNING
-------
No command line arguments. Requires TensorFlow, scikit-learn and the four
optimizer modules and the activation module of the study in the same directory.

    python3 B_05_activation_optimizer_gradient_probe.py

Outputs
-------
    <OUTPUT_DIR>/run_records.csv                 one row per finished run, written as it goes
    <OUTPUT_DIR>/activation_optimizer_report.txt
    <OUTPUT_DIR>/cell_summary.csv                averaged over seeds
    <OUTPUT_DIR>/activation_optimizer_analysis.json
    <OUTPUT_DIR>/activation_optimizer_tables.tex
    <OUTPUT_DIR>/plots/*.png, *.eps
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# 0) SETTINGS
# =============================================================================

OUTPUT_DIR = Path("activation_optimizer_gradient_probe")
PLOT_DIR = OUTPUT_DIR / "plots"
RECORDS_FILE = OUTPUT_DIR / "run_records.csv"

# -----------------------------------------------------------------------------
# Dataset and architecture. Hyperparameters follow Table 1 of the study.
PROBE_DATASET = "diabetes"
DATASET_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "seeds":    {"task": "multi-class", "n": 64,  "hidden_layers": 2, "batch_size": 32, "epochs": 30},
    "iris":     {"task": "multi-class", "n": 32,  "hidden_layers": 2, "batch_size": 16, "epochs": 30},
    "wine":     {"task": "multi-class", "n": 64,  "hidden_layers": 2, "batch_size": 32, "epochs": 30},
    "glass":    {"task": "multi-class", "n": 64,  "hidden_layers": 2, "batch_size": 32, "epochs": 30},
    "vehicle":  {"task": "multi-class", "n": 128, "hidden_layers": 2, "batch_size": 32, "epochs": 25},
    "diabetes": {"task": "binary",      "n": 64,  "hidden_layers": 1, "batch_size": 32, "epochs": 30},
}

# Same run-number protocol as B_01: consecutive seeds starting at SEED_VAL.
NETWORK_PROBE_RUNS = 3
SEED_VAL = 238974

# -----------------------------------------------------------------------------
# Activations. The four fractal ones are probed at the truncation lengths named
# in the review and at their own deployed length; the two standard ones have no
# truncation parameter.
TRUNCATIONS_TO_PROBE = [8, 16, 24, 32, 64]

ACTIVATION_SPECS: List[Dict[str, Any]] = [
    {"name": "relu", "display": "relu", "family": "standard",
     "term_parameter": None, "deployed_terms": None},
    {"name": "tanh", "display": "tanh", "family": "standard",
     "term_parameter": None, "deployed_terms": None},
    {"name": "modulated_blancmange_curve", "display": "blancmange", "family": "fractal",
     "term_parameter": "n_terms", "deployed_terms": 30},
    {"name": "decaying_cosine_function_tf", "display": "dec. cosine", "family": "fractal",
     "term_parameter": "n_terms", "deployed_terms": 75},
    {"name": "modified_weierstrass_function_tanh", "display": "mod. wei. tanh", "family": "fractal",
     "term_parameter": "n_terms", "deployed_terms": 100},
    {"name": "weierstrass_mandelbrot_function_xpsin", "display": "x+sin", "family": "fractal",
     "term_parameter": "num_terms", "deployed_terms": 100},
]

# -----------------------------------------------------------------------------
# Optimizer grid, mirroring B_01.
OPTIMIZER_FAMILIES: Dict[str, List[str]] = {
    "baseline": ["SGD", "Adam", "RMSprop", "Adadelta"],
    "herrera": ["FSGD", "FAdam", "FRMSprop", "FAdadelta"],
    "memory": ["MemoryFSGD", "MemoryFRMSprop", "MemoryFAdam", "MemoryFAdadelta"],
    "adaptive_memory": [
        "AdaptiveMemoryFSGD", "AdaptiveMemoryFRMSprop",
        "AdaptiveMemoryFAdam", "AdaptiveMemoryFAdadelta",
    ],
    "related_work": ["AdaGL", "FCSGD_GL", "FCAdam_GL", "AOFGD_SGD", "AOFGD_Adam"],
}

INCLUDE_FAMILIES = [
    "baseline", "herrera", "memory", "adaptive_memory", "related_work",
]

# Four settings per optimizer, as in the manuscript: four fractional orders for
# the fractional families, four learning rates for the baselines.
ORDER_GRID = [0.75, 1.25, 1.50, 1.75]

BASELINE_LEARNING_RATE_GRID: Dict[str, List[float]] = {
    "SGD":      [5e-3, 1e-2, 2e-2, 4e-2],
    "Adam":     [5e-4, 1e-3, 2e-3, 4e-3],
    "RMSprop":  [5e-4, 1e-3, 2e-3, 4e-3],
    "Adadelta": [5e-1, 1.0, 2.0, 4.0],
}

# -----------------------------------------------------------------------------
# Optimizer constants, copied from B_01 so that the probe builds exactly the
# optimizers the study ran.
LR_SGD = 1e-2
LR_ADAM = 1e-3
LR_RMSPROP = 1e-3
LR_ADADELTA = 1.0

SGD_MOMENTUM = 0.9
SGD_NESTEROV = False

RMSPROP_RHO = 0.9
RMSPROP_MOMENTUM = 0.0
RMSPROP_CENTERED = False

ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-7
ADAM_AMSGRAD = False

ADADELTA_RHO = 0.95
ADADELTA_EPSILON = 1e-7

HERRERA_FRAC_EPSILON = 1e-6

MEMORY_HISTORY_SIZE = 6
MEMORY_NORMALIZE_COEFFS = False

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

ADAGL_HISTORY_SIZE = 10
FCGL_HISTORY_SIZE = 11
FCGL_PERTURBATION_PROB = 0.5

AOFGD_ALPHA_MIN = 0.5
AOFGD_ALPHA_MAX = 1.5
AOFGD_DELTA_ALPHA = 0.02
AOFGD_CEF_THRESHOLD = 0.05
AOFGD_EMA_GAMMA = 0.95
AOFGD_FRAC_EPSILON = 1e-6
AOFGD_EPS_CEF = 1e-8

GAMMA_EPSILON = 1e-6  # the epsilon of the Herrera factor, for the reported statistics

# The loss penalty of the adaptive optimizers is fed from the training loop.
# A_01 and A_02 feed it; B_01 trains with model.fit and cannot, so in the
# classification runs that one input contributed nothing while the stability
# input worked normally. False reproduces the classification setting, True the
# surface setting.
FEED_LOSS_TO_ADAPTIVE_OPTIMIZERS = False

# -----------------------------------------------------------------------------
# Instrumentation.
SLOT_CHECK_EVERY = 25          # steps between inspections of the optimizer slots
RECORD_STEP_SERIES = False     # write the full per-step norm series of every run
GRAD_NORM_REPORT_THRESHOLD = 1e6
COHERENCE_EPSILON = 1e-30

# -----------------------------------------------------------------------------
# Execution control.
RESUME = True                  # skip cells already present in RECORDS_FILE
PROGRESS_EVERY = 1             # print a line every N finished runs
STOP_AFTER_RUNS = None         # set to an integer to stop early, for a trial

# -----------------------------------------------------------------------------
# Plotting. Style follows the other scripts of the study.
FONT_SCALE = 1.25
DPI_PNG = 400
SAVE_EPS = True

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

ACTIVATION_COLORS: Dict[str, str] = {
    "relu": COLOR_PALETTE_A[0],
    "tanh": COLOR_PALETTE_A[1],
    "modulated_blancmange_curve": COLOR_PALETTE_A[2],
    "decaying_cosine_function_tf": COLOR_PALETTE_A[3],
    "modified_weierstrass_function_tanh": COLOR_PALETTE_A[5],
    "weierstrass_mandelbrot_function_xpsin": COLOR_PALETTE_A[6],
}

FAMILY_COLORS: Dict[str, str] = {
    "baseline": COLOR_PALETTE_A[0],
    "herrera": COLOR_PALETTE_A[1],
    "memory": COLOR_PALETTE_A[2],
    "adaptive_memory": COLOR_PALETTE_A[3],
    "related_work": COLOR_PALETTE_A[6],
}

FAMILY_DISPLAY: Dict[str, str] = {
    "baseline": "standard",
    "herrera": "Herrera",
    "memory": "GL memory",
    "adaptive_memory": "adaptive memory",
    "related_work": "related work",
}

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

RULE = "=" * 120
THIN_RULE = "-" * 120


# =============================================================================
# 1) MODULES OF THE STUDY
# =============================================================================

import tensorflow as tf  # noqa: E402

import fractal_activation_functions as faf  # noqa: E402

from class_b_tf_fractional_optimizers_OHA import (  # noqa: E402
    FSGD, FAdam, FRMSprop, FAdadelta,
)
from class_c_tf_fractional_optimizers_SR import (  # noqa: E402
    MemoryFSGD, MemoryFRMSprop, MemoryFAdam, MemoryFAdadelta,
)
from class_minimal_fractional_adaptive_optimizers import (  # noqa: E402
    AdaptiveMemoryFSGD, AdaptiveMemoryFRMSprop,
    AdaptiveMemoryFAdam, AdaptiveMemoryFAdadelta,
)
from class_d_related_work_optimizers import (  # noqa: E402
    AdaGL, FCSGD_GL, FCAdam_GL, AOFGD_SGD, AOFGD_Adam,
)

from sklearn.datasets import fetch_openml  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # noqa: E402

tf.keras.backend.set_floatx("float32")


# =============================================================================
# 2) ACTIVATIONS
# =============================================================================

class ActivationFactory:
    """
    Builds the Keras-facing callable for one activation at one truncation.

    The fractal activations are taken from the activation module of the study
    and called with their own term parameter, so the function trained with is
    the deployed one and not a reimplementation.
    """

    def __init__(self, specs: Sequence[Dict[str, Any]] = tuple(ACTIVATION_SPECS)) -> None:
        self.specs = [dict(spec) for spec in specs]

    def names(self) -> List[str]:
        return [spec["name"] for spec in self.specs]

    def spec(self, name: str) -> Dict[str, Any]:
        for entry in self.specs:
            if entry["name"] == name:
                return entry
        raise KeyError(name)

    def display(self, name: str) -> str:
        return self.spec(name)["display"]

    def settings_for(self, name: str) -> List[Optional[int]]:
        """Truncations to probe: the review's lengths plus the deployed one."""
        spec = self.spec(name)
        deployed = spec["deployed_terms"]
        if deployed is None:
            return [None]
        selected = sorted(
            {t for t in TRUNCATIONS_TO_PROBE if t <= int(deployed)} | {int(deployed)}
        )
        return [int(t) for t in selected]

    def build(self, name: str, terms: Optional[int]) -> Callable[[Any], Any]:
        if name == "relu":
            return tf.nn.relu
        if name == "tanh":
            return tf.nn.tanh
        spec = self.spec(name)
        function = getattr(faf, name)
        if terms is None:
            return function
        parameter = spec["term_parameter"]

        def wrapped(z, _function=function, _parameter=parameter, _terms=int(terms)):
            return _function(z, **{_parameter: _terms})

        return wrapped


# =============================================================================
# 3) OPTIMIZERS
# =============================================================================

def _construct(factory: Callable[..., Any], **kwargs: Any) -> Any:
    """
    Build an optimizer, tolerating both spellings of the learning rate. B_01
    passes lr=; newer Keras versions accept only learning_rate=. The study's own
    classes accept either, so the first attempt is the one B_01 uses.
    """
    try:
        return factory(**kwargs)
    except (TypeError, ValueError):
        if "lr" not in kwargs:
            raise
        renamed = dict(kwargs)
        renamed["learning_rate"] = renamed.pop("lr")
        return factory(**renamed)


class OptimizerFactory:
    """
    The 21 optimizers of the study, built exactly as in B_01.

    `setting` carries the learning rate for the four standard optimizers and
    the fractional order for the seventeen fractional ones, which is the same
    convention B_01 uses for its `vderiv` field.
    """

    def family_of(self, name: str) -> str:
        for family, members in OPTIMIZER_FAMILIES.items():
            if name in members:
                return family
        raise KeyError(name)

    def names(self) -> List[str]:
        out: List[str] = []
        for family in INCLUDE_FAMILIES:
            out.extend(OPTIMIZER_FAMILIES[family])
        return out

    def settings_for(self, name: str) -> List[float]:
        if self.family_of(name) == "baseline":
            return [float(v) for v in BASELINE_LEARNING_RATE_GRID[name]]
        return [float(v) for v in ORDER_GRID]

    def setting_kind(self, name: str) -> str:
        return "learning_rate" if self.family_of(name) == "baseline" else "order"

    def build(self, name: str, setting: float) -> Any:
        value = float(setting)

        # -------- standard baselines ----------------------------------------
        if name == "SGD":
            return _construct(tf.keras.optimizers.SGD, lr=value,
                              momentum=SGD_MOMENTUM, nesterov=SGD_NESTEROV)
        if name == "Adam":
            return _construct(tf.keras.optimizers.Adam, lr=value,
                              beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2,
                              epsilon=ADAM_EPSILON, amsgrad=ADAM_AMSGRAD)
        if name == "RMSprop":
            return _construct(tf.keras.optimizers.RMSprop, lr=value, rho=RMSPROP_RHO,
                              momentum=RMSPROP_MOMENTUM, epsilon=ADAM_EPSILON,
                              centered=RMSPROP_CENTERED)
        if name == "Adadelta":
            return _construct(tf.keras.optimizers.Adadelta, lr=value,
                              rho=ADADELTA_RHO, epsilon=ADADELTA_EPSILON)

        # -------- Herrera-style (class_b) -----------------------------------
        if name == "FSGD":
            return _construct(FSGD, lr=LR_SGD, momentum=SGD_MOMENTUM,
                              nesterov=SGD_NESTEROV, vderiv=value,
                              frac_epsilon=HERRERA_FRAC_EPSILON)
        if name == "FAdam":
            return _construct(FAdam, lr=LR_ADAM, beta_1=ADAM_BETA_1,
                              beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON,
                              amsgrad=ADAM_AMSGRAD, vderiv=value,
                              frac_epsilon=HERRERA_FRAC_EPSILON)
        if name == "FRMSprop":
            return _construct(FRMSprop, lr=LR_RMSPROP, rho=RMSPROP_RHO,
                              momentum=RMSPROP_MOMENTUM, epsilon=ADAM_EPSILON,
                              centered=RMSPROP_CENTERED, vderiv=value,
                              frac_epsilon=HERRERA_FRAC_EPSILON)
        if name == "FAdadelta":
            return _construct(FAdadelta, lr=LR_ADADELTA, rho=ADADELTA_RHO,
                              epsilon=ADADELTA_EPSILON, vderiv=value,
                              frac_epsilon=HERRERA_FRAC_EPSILON)

        # -------- fixed-order GL memory (class_c) ---------------------------
        if name == "MemoryFSGD":
            return _construct(MemoryFSGD, lr=LR_SGD, momentum=SGD_MOMENTUM,
                              nesterov=SGD_NESTEROV, vderiv=value,
                              history_size=MEMORY_HISTORY_SIZE,
                              normalize_coefficients=MEMORY_NORMALIZE_COEFFS)
        if name == "MemoryFRMSprop":
            return _construct(MemoryFRMSprop, lr=LR_RMSPROP, rho=RMSPROP_RHO,
                              momentum=RMSPROP_MOMENTUM, epsilon=ADAM_EPSILON,
                              centered=RMSPROP_CENTERED, vderiv=value,
                              history_size=MEMORY_HISTORY_SIZE,
                              normalize_coefficients=MEMORY_NORMALIZE_COEFFS)
        if name == "MemoryFAdam":
            return _construct(MemoryFAdam, lr=LR_ADAM, beta_1=ADAM_BETA_1,
                              beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON,
                              amsgrad=ADAM_AMSGRAD, vderiv=value,
                              history_size=MEMORY_HISTORY_SIZE,
                              normalize_coefficients=MEMORY_NORMALIZE_COEFFS)
        if name == "MemoryFAdadelta":
            return _construct(MemoryFAdadelta, lr=LR_ADADELTA, rho=ADADELTA_RHO,
                              epsilon=ADADELTA_EPSILON, vderiv=value,
                              history_size=MEMORY_HISTORY_SIZE,
                              normalize_coefficients=MEMORY_NORMALIZE_COEFFS)

        # -------- adaptive memory (class_minimal) ---------------------------
        adaptive_kwargs = dict(
            vderiv=value,
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
        if name == "AdaptiveMemoryFSGD":
            return _construct(AdaptiveMemoryFSGD, lr=LR_SGD, momentum=SGD_MOMENTUM,
                              nesterov=SGD_NESTEROV, **adaptive_kwargs)
        if name == "AdaptiveMemoryFRMSprop":
            return _construct(AdaptiveMemoryFRMSprop, lr=LR_RMSPROP, rho=RMSPROP_RHO,
                              momentum=RMSPROP_MOMENTUM, epsilon=ADAM_EPSILON,
                              centered=RMSPROP_CENTERED, **adaptive_kwargs)
        if name == "AdaptiveMemoryFAdam":
            return _construct(AdaptiveMemoryFAdam, lr=LR_ADAM, beta_1=ADAM_BETA_1,
                              beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON,
                              amsgrad=ADAM_AMSGRAD, **adaptive_kwargs)
        if name == "AdaptiveMemoryFAdadelta":
            return _construct(AdaptiveMemoryFAdadelta, lr=LR_ADADELTA,
                              rho=ADADELTA_RHO, epsilon=ADADELTA_EPSILON,
                              **adaptive_kwargs)

        # -------- related work (class_d) ------------------------------------
        if name == "AdaGL":
            return _construct(AdaGL, lr=LR_ADAM, alpha=value,
                              history_size=ADAGL_HISTORY_SIZE,
                              beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2,
                              epsilon=ADAM_EPSILON, amsgrad=ADAM_AMSGRAD)
        if name == "FCSGD_GL":
            return _construct(FCSGD_GL, lr=LR_SGD, momentum=SGD_MOMENTUM,
                              nesterov=SGD_NESTEROV, alpha=value,
                              history_size=FCGL_HISTORY_SIZE,
                              perturbation_prob=FCGL_PERTURBATION_PROB)
        if name == "FCAdam_GL":
            return _construct(FCAdam_GL, lr=LR_ADAM, beta_1=ADAM_BETA_1,
                              beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON,
                              amsgrad=ADAM_AMSGRAD, alpha=value,
                              history_size=FCGL_HISTORY_SIZE,
                              perturbation_prob=FCGL_PERTURBATION_PROB)
        if name == "AOFGD_SGD":
            return _construct(AOFGD_SGD, lr=LR_SGD, momentum=SGD_MOMENTUM,
                              nesterov=SGD_NESTEROV, alpha_init=value,
                              alpha_min=AOFGD_ALPHA_MIN,
                              alpha_max=max(AOFGD_ALPHA_MAX, value),
                              delta_alpha=AOFGD_DELTA_ALPHA,
                              cef_threshold=AOFGD_CEF_THRESHOLD,
                              ema_gamma=AOFGD_EMA_GAMMA,
                              frac_epsilon=AOFGD_FRAC_EPSILON,
                              eps_cef=AOFGD_EPS_CEF)
        if name == "AOFGD_Adam":
            return _construct(AOFGD_Adam, lr=LR_ADAM, beta_1=ADAM_BETA_1,
                              beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON,
                              amsgrad=ADAM_AMSGRAD, alpha_init=value,
                              alpha_min=AOFGD_ALPHA_MIN,
                              alpha_max=max(AOFGD_ALPHA_MAX, value),
                              delta_alpha=AOFGD_DELTA_ALPHA,
                              cef_threshold=AOFGD_CEF_THRESHOLD,
                              ema_gamma=AOFGD_EMA_GAMMA,
                              frac_epsilon=AOFGD_FRAC_EPSILON,
                              eps_cef=AOFGD_EPS_CEF)

        raise ValueError(f"unknown optimizer: {name}")


# =============================================================================
# 4) RESULT RECORD
# =============================================================================

@dataclass
class RunRecord:
    dataset: str
    activation: str
    activation_display: str
    activation_family: str
    terms: Optional[int]
    optimizer: str
    optimizer_family: str
    setting_kind: str
    setting: float
    seed: int
    steps: int
    # gradient magnitude
    grad_norm_mean: float
    grad_norm_median: float
    grad_norm_p95: float
    grad_norm_max: float
    steps_above_grad_threshold: int
    # applied update magnitude
    update_norm_mean: float
    update_norm_median: float
    update_norm_max: float
    # coherence of consecutive gradients
    grad_cosine_mean: float
    grad_cosine_median: float
    grad_cosine_abs_mean: float
    # stability
    nonfinite_gradient_steps: int
    nonfinite_loss_steps: int
    nonfinite_update_steps: int
    optimizer_slot_nonfinite: int
    first_slot_nonfinite_step: int
    # family diagnostics
    herrera_factor_median: float
    herrera_factor_max: float
    gl_coefficient_sum: float
    adaptive_mix_mean: float
    adaptive_stability_mean: float
    # outcome
    final_train_loss: float
    test_accuracy: float
    preactivation_p99_abs: float
    seconds: float


RECORD_FIELDS = [f.name for f in fields(RunRecord)]


# =============================================================================
# 5) DATA
# =============================================================================

class DatasetProvider:
    """Loads and splits exactly as B_01 does, and caches the raw arrays."""

    def __init__(self, dataset: str = PROBE_DATASET) -> None:
        self.dataset = dataset
        self._cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def raw(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._cache is None:
            data = fetch_openml(name=self.dataset, version=1, as_frame=True, parser="auto")
            X = np.asarray(data.data, dtype=np.float32)
            y = np.asarray(LabelEncoder().fit_transform(data.target), dtype=np.int32)
            self._cache = (X, y)
            print(f"  [OpenML] loaded {self.dataset}: X={X.shape}, "
                  f"classes={len(np.unique(y))}")
        return self._cache

    def split(self, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, y = self.raw()
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, shuffle=True
        )
        return (
            scaler.fit_transform(X_train).astype(np.float32),
            scaler.transform(X_test).astype(np.float32),
            np.asarray(y_train, np.int32),
            np.asarray(y_test, np.int32),
        )

    def n_classes(self) -> int:
        _, y = self.raw()
        return int(len(np.unique(y)))


# =============================================================================
# 6) MODEL
# =============================================================================

def build_model(
    n_features: int,
    n_outputs: int,
    neurons: int,
    hidden_layers: int,
    activation: Callable[[Any], Any],
) -> Any:
    """The architecture of B_01, with the activation passed in as a callable."""
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    layers: List[Any] = [
        tf.keras.layers.Dense(
            neurons,
            activation=activation,
            kernel_initializer=initializer,
            input_dim=n_features,
        )
    ]
    if hidden_layers == 2:
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Dropout(0.2))
        layers.append(
            tf.keras.layers.Dense(
                neurons * 2, activation=activation, kernel_initializer=initializer
            )
        )
        layers.append(
            tf.keras.layers.Dense(
                n_outputs, activation="softmax", kernel_initializer=initializer
            )
        )
    else:
        layers.append(
            tf.keras.layers.Dense(
                n_outputs, activation="sigmoid", kernel_initializer=initializer
            )
        )
    return tf.keras.Sequential(layers)


# =============================================================================
# 7) INSTRUMENTED TRAINING RUN
# =============================================================================

class InstrumentedRun:
    """
    One training run with per-step instrumentation.

    The step is compiled once per run, because the fractal activations unroll
    into as many operations as they have scales and executing that eagerly at
    every step would dominate the cost. Three quantities need state that
    survives between steps and are therefore kept in variables of their own:
    the previous gradient, for the coherence measurement, and the parameters
    before the update, for the applied update norm.
    """

    def __init__(
        self,
        provider: DatasetProvider,
        activations: ActivationFactory,
        optimizers: OptimizerFactory,
        hyper: Dict[str, Any],
    ) -> None:
        self.provider = provider
        self.activations = activations
        self.optimizers = optimizers
        self.hyper = hyper

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _optimizer_variables(optimizer: Any) -> List[Any]:
        """Slot variables, across the optimizer APIs the study may run under."""
        try:
            variables = optimizer.variables()
        except TypeError:
            variables = optimizer.variables
        except AttributeError:
            variables = getattr(optimizer, "weights", [])
        return list(variables or [])

    def _slot_nonfinite(self, optimizer: Any) -> int:
        total = 0
        for variable in self._optimizer_variables(optimizer):
            try:
                values = variable.numpy()
            except Exception:
                continue
            if np.issubdtype(np.asarray(values).dtype, np.floating):
                total += int(np.count_nonzero(~np.isfinite(values)))
        return total

    @staticmethod
    def _herrera_statistics(
        optimizer_family: str,
        setting: float,
        model: Any,
    ) -> Tuple[float, float]:
        """
        The Herrera factor (|theta| + eps)^(1-nu)/Gamma(2-nu) implied by the
        final weights. Reported for the family that applies it, so that an
        unusually large gradient can be attributed to the factor or to the
        activation.
        """
        if optimizer_family != "herrera":
            return float("nan"), float("nan")
        order = float(setting)
        if abs(2.0 - order) < 1e-12:
            return float("nan"), float("nan")
        gamma_value = math.gamma(2.0 - order)
        factors: List[np.ndarray] = []
        for variable in model.trainable_variables:
            theta = np.abs(variable.numpy()).astype(np.float64) + GAMMA_EPSILON
            factors.append(np.power(theta, 1.0 - order) / gamma_value)
        if not factors:
            return float("nan"), float("nan")
        flat = np.concatenate([f.reshape(-1) for f in factors])
        return float(np.median(flat)), float(np.max(flat))

    @staticmethod
    def _optional_scalar(optimizer: Any, accessor: str) -> float:
        """Read a diagnostic accessor if the optimizer class provides one."""
        method = getattr(optimizer, accessor, None)
        if method is None:
            return float("nan")
        try:
            value = method()
        except Exception:
            return float("nan")
        if isinstance(value, (list, tuple)):
            if not value:
                return float("nan")
            numbers = []
            for entry in value:
                try:
                    numbers.append(float(np.asarray(entry).mean()))
                except Exception:
                    continue
            return float(np.mean(numbers)) if numbers else float("nan")
        try:
            return float(np.asarray(value).mean())
        except Exception:
            return float("nan")

    # ---- the run ------------------------------------------------------------

    def run(
        self,
        activation_name: str,
        terms: Optional[int],
        optimizer_name: str,
        setting: float,
        seed: int,
    ) -> RunRecord:
        started = time.time()
        hyper = self.hyper
        spec = self.activations.spec(activation_name)
        optimizer_family = self.optimizers.family_of(optimizer_name)

        tf.keras.backend.clear_session()
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        X_train, X_test, y_train, y_test = self.provider.split(seed)
        n_classes = self.provider.n_classes()
        n_outputs = n_classes if hyper["task"] == "multi-class" else 1

        model = build_model(
            X_train.shape[1],
            n_outputs,
            int(hyper["n"]),
            int(hyper["hidden_layers"]),
            self.activations.build(activation_name, terms),
        )
        optimizer = self.optimizers.build(optimizer_name, setting)

        if hyper["task"] == "multi-class":
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            targets = y_train.astype(np.int32)
        else:
            loss_fn = tf.keras.losses.BinaryCrossentropy()
            targets = y_train.astype(np.float32).reshape(-1, 1)

        data = (
            tf.data.Dataset.from_tensor_slices((X_train, targets))
            .shuffle(len(X_train), seed=seed)
            .batch(int(hyper["batch_size"]))
        )

        # Build the variables before creating the buffers that mirror them.
        model(X_train[: min(8, len(X_train))], training=False)
        variables = model.trainable_variables
        previous_gradient = [
            tf.Variable(tf.zeros_like(v), trainable=False, name=f"prev_grad_{i}")
            for i, v in enumerate(variables)
        ]
        previous_parameters = [
            tf.Variable(tf.zeros_like(v), trainable=False, name=f"prev_param_{i}")
            for i, v in enumerate(variables)
        ]

        @tf.function
        def train_step(xb, yb):
            with tf.GradientTape() as tape:
                prediction = model(xb, training=True)
                loss = loss_fn(yb, prediction)
            gradients = tape.gradient(loss, variables)

            present, present_previous, present_variables, present_parameters = [], [], [], []
            for gradient, variable, previous, parameter in zip(
                gradients, variables, previous_gradient, previous_parameters
            ):
                if gradient is None:
                    continue
                present.append(gradient)
                present_previous.append(previous)
                present_variables.append(variable)
                present_parameters.append(parameter)

            finite = tf.reduce_all(
                tf.stack([tf.reduce_all(tf.math.is_finite(g)) for g in present])
            )
            norm = tf.linalg.global_norm(present)
            previous_norm = tf.linalg.global_norm(present_previous)
            dot = tf.add_n(
                [
                    tf.reduce_sum(g * p)
                    for g, p in zip(present, present_previous)
                ]
            )
            cosine = dot / (norm * previous_norm + COHERENCE_EPSILON)

            for parameter, variable in zip(present_parameters, present_variables):
                parameter.assign(variable)

            optimizer.apply_gradients(list(zip(present, present_variables)))

            update = tf.linalg.global_norm(
                [
                    variable - parameter
                    for variable, parameter in zip(present_variables, present_parameters)
                ]
            )
            for previous, gradient in zip(present_previous, present):
                previous.assign(gradient)

            return loss, norm, cosine, update, finite

        grad_norms: List[float] = []
        update_norms: List[float] = []
        cosines: List[float] = []
        nonfinite_gradient_steps = 0
        nonfinite_loss_steps = 0
        nonfinite_update_steps = 0
        first_slot_nonfinite_step = -1
        slot_nonfinite = 0
        step_index = 0
        last_loss = float("nan")

        for _ in range(int(hyper["epochs"])):
            for xb, yb in data:
                if FEED_LOSS_TO_ADAPTIVE_OPTIMIZERS and hasattr(optimizer, "set_current_loss"):
                    optimizer.set_current_loss(last_loss if math.isfinite(last_loss) else 0.0)

                loss, norm, cosine, update, finite = train_step(xb, yb)
                step_index += 1

                loss_value = float(loss.numpy())
                norm_value = float(norm.numpy())
                update_value = float(update.numpy())
                cosine_value = float(cosine.numpy())
                last_loss = loss_value

                if not math.isfinite(loss_value):
                    nonfinite_loss_steps += 1
                if not bool(finite.numpy()):
                    nonfinite_gradient_steps += 1
                if math.isfinite(norm_value):
                    grad_norms.append(norm_value)
                if math.isfinite(update_value):
                    update_norms.append(update_value)
                else:
                    nonfinite_update_steps += 1
                # The first step compares against a zero buffer, which is not a
                # measurement of coherence.
                if step_index > 1 and math.isfinite(cosine_value):
                    cosines.append(cosine_value)

                if step_index % SLOT_CHECK_EVERY == 0:
                    current = self._slot_nonfinite(optimizer)
                    if current > slot_nonfinite:
                        slot_nonfinite = current
                        if first_slot_nonfinite_step < 0:
                            first_slot_nonfinite_step = step_index

        final_slots = self._slot_nonfinite(optimizer)
        if final_slots > slot_nonfinite:
            slot_nonfinite = final_slots
            if first_slot_nonfinite_step < 0:
                first_slot_nonfinite_step = step_index

        prediction = model(X_test, training=False).numpy()
        if hyper["task"] == "multi-class":
            accuracy = float(np.mean(np.argmax(prediction, axis=1) == y_test))
        else:
            accuracy = float(
                np.mean((prediction.reshape(-1) >= 0.5).astype(np.int32) == y_test)
            )

        weights = model.layers[0].get_weights()
        pre_activation = X_test @ weights[0] + weights[1]

        herrera_median, herrera_max = self._herrera_statistics(
            optimizer_family, setting, model
        )

        grad_array = np.asarray(grad_norms, dtype=np.float64)
        update_array = np.asarray(update_norms, dtype=np.float64)
        cosine_array = np.asarray(cosines, dtype=np.float64)

        def _statistic(values: np.ndarray, function: Callable[[np.ndarray], Any]) -> float:
            return float(function(values)) if values.size else float("nan")

        return RunRecord(
            dataset=self.provider.dataset,
            activation=activation_name,
            activation_display=spec["display"],
            activation_family=spec["family"],
            terms=None if terms is None else int(terms),
            optimizer=optimizer_name,
            optimizer_family=optimizer_family,
            setting_kind=self.optimizers.setting_kind(optimizer_name),
            setting=float(setting),
            seed=int(seed),
            steps=int(step_index),
            grad_norm_mean=_statistic(grad_array, np.mean),
            grad_norm_median=_statistic(grad_array, np.median),
            grad_norm_p95=_statistic(grad_array, lambda a: np.percentile(a, 95.0)),
            grad_norm_max=_statistic(grad_array, np.max),
            steps_above_grad_threshold=int(
                np.count_nonzero(grad_array > GRAD_NORM_REPORT_THRESHOLD)
            ),
            update_norm_mean=_statistic(update_array, np.mean),
            update_norm_median=_statistic(update_array, np.median),
            update_norm_max=_statistic(update_array, np.max),
            grad_cosine_mean=_statistic(cosine_array, np.mean),
            grad_cosine_median=_statistic(cosine_array, np.median),
            grad_cosine_abs_mean=_statistic(cosine_array, lambda a: np.mean(np.abs(a))),
            nonfinite_gradient_steps=int(nonfinite_gradient_steps),
            nonfinite_loss_steps=int(nonfinite_loss_steps),
            nonfinite_update_steps=int(nonfinite_update_steps),
            optimizer_slot_nonfinite=int(slot_nonfinite),
            first_slot_nonfinite_step=int(first_slot_nonfinite_step),
            herrera_factor_median=herrera_median,
            herrera_factor_max=herrera_max,
            gl_coefficient_sum=self._optional_scalar(optimizer, "gl_coefficient_sum"),
            adaptive_mix_mean=self._optional_scalar(optimizer, "current_mix_values"),
            adaptive_stability_mean=self._optional_scalar(
                optimizer, "current_stability_values"
            ),
            final_train_loss=float(last_loss),
            test_accuracy=accuracy,
            preactivation_p99_abs=float(np.percentile(np.abs(pre_activation), 99.0)),
            seconds=float(time.time() - started),
        )


# =============================================================================
# 8) PROBE DRIVER
# =============================================================================

class ProbeDriver:
    """
    Builds the plan, runs it, and writes every finished run immediately.

    Seeds are the outermost loop, so an interrupted run still covers every cell
    once. A run that fails is recorded in the failure log and does not stop the
    others, because a grid of this size should not be lost to one optimizer
    raising on one setting.
    """

    def __init__(
        self,
        provider: DatasetProvider,
        activations: ActivationFactory,
        optimizers: OptimizerFactory,
    ) -> None:
        self.provider = provider
        self.activations = activations
        self.optimizers = optimizers
        self.hyper = DATASET_HYPERPARAMS[provider.dataset]
        self.runner = InstrumentedRun(provider, activations, optimizers, self.hyper)
        self.failures: List[Dict[str, str]] = []

    # ---- plan ---------------------------------------------------------------

    def plan(self) -> List[Dict[str, Any]]:
        cells: List[Dict[str, Any]] = []
        for run_index in range(NETWORK_PROBE_RUNS):
            seed = SEED_VAL + run_index
            for activation_name in self.activations.names():
                for terms in self.activations.settings_for(activation_name):
                    for optimizer_name in self.optimizers.names():
                        for setting in self.optimizers.settings_for(optimizer_name):
                            cells.append(
                                {
                                    "activation": activation_name,
                                    "terms": terms,
                                    "optimizer": optimizer_name,
                                    "setting": float(setting),
                                    "seed": int(seed),
                                }
                            )
        return cells

    # ---- records ------------------------------------------------------------

    @staticmethod
    def _key(record: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
        terms = record.get("terms")
        terms_text = "" if terms in (None, "", "None") else str(int(float(terms)))
        return (
            str(record["activation"]),
            terms_text,
            str(record["optimizer"]),
            f"{float(record['setting']):.6g}",
            str(int(float(record["seed"]))),
        )

    def completed(self) -> Dict[Tuple[str, str, str, str, str], Dict[str, Any]]:
        if not (RESUME and RECORDS_FILE.exists()):
            return {}
        out: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
        with open(RECORDS_FILE, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    out[self._key(row)] = row
                except Exception:
                    continue
        return out

    @staticmethod
    def _append(record: RunRecord) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        exists = RECORDS_FILE.exists()
        with open(RECORDS_FILE, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(asdict(record))

    # ---- execution ----------------------------------------------------------

    def run(self) -> List[RunRecord]:
        cells = self.plan()
        done = self.completed()
        pending = [cell for cell in cells if self._key(cell) not in done]

        print(f"  plan: {len(cells)} runs "
              f"({len(self.activations.names())} activations, "
              f"{len(self.optimizers.names())} optimizers, "
              f"{NETWORK_PROBE_RUNS} seeds)")
        if done:
            print(f"  {len(done)} already in {RECORDS_FILE}, {len(pending)} to go")

        records: List[RunRecord] = []
        started = time.time()
        for index, cell in enumerate(pending, start=1):
            if STOP_AFTER_RUNS is not None and index > int(STOP_AFTER_RUNS):
                print(f"  stopping after {STOP_AFTER_RUNS} runs as configured")
                break
            try:
                record = self.runner.run(
                    cell["activation"], cell["terms"],
                    cell["optimizer"], cell["setting"], cell["seed"],
                )
            except Exception as exc:  # one failing configuration must not stop the grid
                self.failures.append(
                    {
                        "activation": str(cell["activation"]),
                        "terms": str(cell["terms"]),
                        "optimizer": str(cell["optimizer"]),
                        "setting": str(cell["setting"]),
                        "seed": str(cell["seed"]),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"    [failed] {cell['optimizer']} @ {cell['setting']} on "
                      f"{cell['activation']}: {type(exc).__name__}: {exc}")
                continue

            self._append(record)
            records.append(record)

            if index % PROGRESS_EVERY == 0:
                elapsed = time.time() - started
                rate = elapsed / index
                remaining = rate * (len(pending) - index)
                print(
                    f"    [{index:>5}/{len(pending)}] "
                    f"{record.activation_display:>15} "
                    f"terms={str(record.terms):>5} "
                    f"{record.optimizer:>22} @ {record.setting:<6g} "
                    f"seed={record.seed} "
                    f"acc={record.test_accuracy:.4f} "
                    f"mean|g|={record.grad_norm_mean:.3e} "
                    f"cos={record.grad_cosine_mean:+.3f} "
                    f"slots_nf={record.optimizer_slot_nonfinite} "
                    f"[{record.seconds:.1f}s, eta {remaining / 3600.0:.2f} h]"
                )

        return records

    # ---- loading everything back -------------------------------------------

    def load_all(self) -> List[RunRecord]:
        """Every record on disk, so a resumed run reports the whole grid."""
        if not RECORDS_FILE.exists():
            return []
        out: List[RunRecord] = []
        with open(RECORDS_FILE, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                values: Dict[str, Any] = {}
                for item in fields(RunRecord):
                    raw = row.get(item.name, "")
                    if item.name == "terms":
                        values[item.name] = (
                            None if raw in ("", "None") else int(float(raw))
                        )
                    elif item.type in ("int", int):
                        values[item.name] = int(float(raw)) if raw not in ("", "None") else 0
                    elif item.type in ("float", float):
                        values[item.name] = float(raw) if raw not in ("", "None") else float("nan")
                    else:
                        values[item.name] = raw
                try:
                    out.append(RunRecord(**values))
                except Exception:
                    continue
        return out


# =============================================================================
# 9) AGGREGATION
# =============================================================================

@dataclass
class CellSummary:
    activation: str
    activation_display: str
    terms: Optional[int]
    optimizer: str
    optimizer_family: str
    setting_kind: str
    setting: float
    seeds: int
    grad_norm_mean: float
    grad_norm_max: float
    update_norm_mean: float
    grad_cosine_mean: float
    nonfinite_steps_total: int
    slot_nonfinite_total: int
    herrera_factor_median: float
    gl_coefficient_sum: float
    adaptive_mix_mean: float
    test_accuracy_mean: float
    test_accuracy_std: float
    seconds_mean: float


def summarize(records: Sequence[RunRecord]) -> List[CellSummary]:
    groups: Dict[Tuple[str, Optional[int], str, float], List[RunRecord]] = {}
    for record in records:
        groups.setdefault(
            (record.activation, record.terms, record.optimizer, record.setting), []
        ).append(record)

    def _mean(values: Sequence[float]) -> float:
        array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        return float(np.mean(array)) if array.size else float("nan")

    out: List[CellSummary] = []
    for (activation, terms, optimizer, setting), group in groups.items():
        first = group[0]
        out.append(
            CellSummary(
                activation=activation,
                activation_display=first.activation_display,
                terms=terms,
                optimizer=optimizer,
                optimizer_family=first.optimizer_family,
                setting_kind=first.setting_kind,
                setting=float(setting),
                seeds=len(group),
                grad_norm_mean=_mean([r.grad_norm_mean for r in group]),
                grad_norm_max=float(
                    np.max([r.grad_norm_max for r in group])
                    if group
                    else float("nan")
                ),
                update_norm_mean=_mean([r.update_norm_mean for r in group]),
                grad_cosine_mean=_mean([r.grad_cosine_mean for r in group]),
                nonfinite_steps_total=int(
                    sum(
                        r.nonfinite_gradient_steps
                        + r.nonfinite_loss_steps
                        + r.nonfinite_update_steps
                        for r in group
                    )
                ),
                slot_nonfinite_total=int(
                    sum(r.optimizer_slot_nonfinite for r in group)
                ),
                herrera_factor_median=_mean([r.herrera_factor_median for r in group]),
                gl_coefficient_sum=_mean([r.gl_coefficient_sum for r in group]),
                adaptive_mix_mean=_mean([r.adaptive_mix_mean for r in group]),
                test_accuracy_mean=_mean([r.test_accuracy for r in group]),
                test_accuracy_std=float(
                    np.std([r.test_accuracy for r in group])
                    if len(group) > 1
                    else 0.0
                ),
                seconds_mean=_mean([r.seconds for r in group]),
            )
        )
    return out


# =============================================================================
# 10) PLOTTING
# =============================================================================

class ProbePlotter:
    """Comparison figures for the appendix."""

    def __init__(
        self,
        records: Sequence[RunRecord],
        summaries: Sequence[CellSummary],
        activations: ActivationFactory,
    ) -> None:
        self.records = list(records)
        self.summaries = list(summaries)
        self.activations = activations
        self.activation_order = [
            name for name in activations.names()
            if any(r.activation == name for r in self.records)
        ]
        self.family_order = [
            family for family in INCLUDE_FAMILIES
            if any(r.optimizer_family == family for r in self.records)
        ]

    @staticmethod
    def _save(fig: plt.Figure, out_base: Path) -> None:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_base.with_suffix(".png"), dpi=DPI_PNG, bbox_inches="tight")
        if SAVE_EPS:
            fig.savefig(out_base.with_suffix(".eps"), format="eps", bbox_inches="tight")
        plt.close(fig)

    def _deployed_records(self) -> List[RunRecord]:
        """Only the deployed truncation of each activation."""
        out = []
        for record in self.records:
            spec = self.activations.spec(record.activation)
            if record.terms == spec["deployed_terms"]:
                out.append(record)
        return out

    def _display(self, activation: str) -> str:
        return self.activations.display(activation)

    # ---- figures ------------------------------------------------------------

    def plot_gradient_by_family(self) -> None:
        """Gradient norm and applied update norm, activation by family."""
        records = self._deployed_records()
        if not records:
            return
        fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 1.6 * len(self.activation_order)), 9.0))
        width = 0.8 / max(len(self.family_order), 1)

        for index, family in enumerate(self.family_order):
            gradients, updates = [], []
            for activation in self.activation_order:
                group = [
                    r for r in records
                    if r.activation == activation and r.optimizer_family == family
                ]
                gradients.append(
                    float(np.mean([r.grad_norm_mean for r in group])) if group else np.nan
                )
                updates.append(
                    float(np.mean([r.update_norm_mean for r in group])) if group else np.nan
                )
            positions = (
                np.arange(len(self.activation_order))
                + (index - (len(self.family_order) - 1) / 2) * width
            )
            axes[0].bar(positions, gradients, width=width * 0.9,
                        color=FAMILY_COLORS.get(family, COLOR_PALETTE_A[8]),
                        label=FAMILY_DISPLAY.get(family, family))
            axes[1].bar(positions, updates, width=width * 0.9,
                        color=FAMILY_COLORS.get(family, COLOR_PALETTE_A[8]),
                        label=FAMILY_DISPLAY.get(family, family))

        for ax, title, label in (
            (axes[0], "Gradient the optimizer receives", "mean global gradient norm"),
            (axes[1], "Update the optimizer applies", "mean parameter update norm"),
        ):
            ax.set_yscale("log")
            ax.set_ylabel(label)
            ax.set_title(title)
            ax.grid(True, alpha=0.25, axis="y", which="both")
            ax.set_xticks(np.arange(len(self.activation_order)))
        axes[0].set_xticklabels([])
        axes[1].set_xticklabels(
            [self._display(a) for a in self.activation_order], rotation=20, ha="right"
        )
        # The legend goes above the panels: inside them it would sit on top of
        # the tallest bars, which are the ones the figure is about.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, frameon=False, loc="upper center",
            ncol=len(self.family_order), fontsize=8 * FONT_SCALE,
            bbox_to_anchor=(0.5, 0.945),
        )
        fig.suptitle(
            f"What each optimizer family receives and applies, {PROBE_DATASET}, "
            "deployed truncations",
            y=1.0,
        )
        self._save(fig, PLOT_DIR / "gradient_and_update_by_family")

    def plot_coherence(self) -> None:
        """Cosine similarity of consecutive gradients, the memory premise."""
        records = self._deployed_records()
        if not records:
            return
        fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(self.activation_order)), 5.0))
        width = 0.8 / max(len(self.family_order), 1)
        for index, family in enumerate(self.family_order):
            values = []
            for activation in self.activation_order:
                group = [
                    r for r in records
                    if r.activation == activation and r.optimizer_family == family
                ]
                values.append(
                    float(np.mean([r.grad_cosine_mean for r in group])) if group else np.nan
                )
            positions = (
                np.arange(len(self.activation_order))
                + (index - (len(self.family_order) - 1) / 2) * width
            )
            ax.bar(positions, values, width=width * 0.9,
                   color=FAMILY_COLORS.get(family, COLOR_PALETTE_A[8]),
                   label=FAMILY_DISPLAY.get(family, family))
        ax.axhline(0.0, color=COLOR_PALETTE_A[8], linewidth=1.0)
        ax.set_xticks(np.arange(len(self.activation_order)))
        ax.set_xticklabels(
            [self._display(a) for a in self.activation_order], rotation=20, ha="right"
        )
        ax.set_ylabel("mean cosine of consecutive gradients")
        ax.set_title(
            "Coherence of the gradient sequence, which the memory mechanisms average over"
        )
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(frameon=False, ncol=len(self.family_order), fontsize=8 * FONT_SCALE)
        self._save(fig, PLOT_DIR / "gradient_coherence")

    def plot_truncation_effect(self) -> None:
        """Gradient norm and accuracy against truncation length, per family."""
        series = [
            name for name in self.activation_order
            if self.activations.spec(name)["deployed_terms"] is not None
        ]
        if not series:
            return
        fig, axes = plt.subplots(2, len(series), figsize=(4.2 * len(series), 8.2),
                                 sharex="col")
        if len(series) == 1:
            axes = np.asarray([[axes[0]], [axes[1]]])

        for column, activation in enumerate(series):
            truncations = sorted(
                {r.terms for r in self.records if r.activation == activation and r.terms}
            )
            for family in self.family_order:
                gradients, accuracies = [], []
                for terms in truncations:
                    group = [
                        r for r in self.records
                        if r.activation == activation
                        and r.terms == terms
                        and r.optimizer_family == family
                    ]
                    gradients.append(
                        float(np.mean([r.grad_norm_mean for r in group])) if group else np.nan
                    )
                    accuracies.append(
                        float(np.mean([r.test_accuracy for r in group])) if group else np.nan
                    )
                color = FAMILY_COLORS.get(family, COLOR_PALETTE_A[8])
                axes[0][column].plot(truncations, gradients, marker="o", markersize=4.0,
                                     color=color, label=FAMILY_DISPLAY.get(family, family))
                axes[1][column].plot(truncations, accuracies, marker="o", markersize=4.0,
                                     color=color, label=FAMILY_DISPLAY.get(family, family))
            axes[0][column].set_yscale("log")
            axes[0][column].set_title(self._display(activation))
            axes[0][column].grid(True, alpha=0.25, which="both")
            axes[1][column].grid(True, alpha=0.25)
            axes[1][column].set_xlabel("retained terms")
        axes[0][0].set_ylabel("mean global gradient norm")
        axes[1][0].set_ylabel("test accuracy")
        axes[0][0].legend(frameon=False, fontsize=8 * FONT_SCALE)
        fig.suptitle(f"Effect of the truncation length on training, {PROBE_DATASET}")
        self._save(fig, PLOT_DIR / "truncation_effect")

    def plot_stability_heatmap(self) -> None:
        """Non-finite incidence per activation and optimizer."""
        records = self._deployed_records()
        if not records:
            return
        optimizers: List[str] = []
        for family in self.family_order:
            for name in OPTIMIZER_FAMILIES[family]:
                if any(r.optimizer == name for r in records):
                    optimizers.append(name)
        if not optimizers:
            return

        matrix = np.zeros((len(optimizers), len(self.activation_order)), dtype=np.float64)
        for row, optimizer in enumerate(optimizers):
            for column, activation in enumerate(self.activation_order):
                group = [
                    r for r in records
                    if r.optimizer == optimizer and r.activation == activation
                ]
                if not group:
                    matrix[row, column] = np.nan
                    continue
                incidents = sum(
                    1
                    for r in group
                    if r.nonfinite_gradient_steps
                    or r.nonfinite_loss_steps
                    or r.nonfinite_update_steps
                    or r.optimizer_slot_nonfinite
                )
                matrix[row, column] = incidents / len(group)

        fig, ax = plt.subplots(figsize=(1.5 * len(self.activation_order) + 4.0,
                                        0.32 * len(optimizers) + 2.5))
        image = ax.imshow(matrix, cmap=HEATMAP_CMAP, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(self.activation_order)))
        ax.set_xticklabels(
            [self._display(a) for a in self.activation_order], rotation=20, ha="right"
        )
        ax.set_yticks(np.arange(len(optimizers)))
        ax.set_yticklabels(optimizers, fontsize=7 * FONT_SCALE)
        # A cell that is exactly zero is the expected case, so only the cells
        # that saw something are labelled.
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isfinite(value) and value > 0.0:
                    ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                            fontsize=7 * FONT_SCALE)
        ax.set_title("Fraction of runs with a non-finite loss, gradient, update or slot")
        fig.colorbar(image, ax=ax, shrink=0.85)
        self._save(fig, PLOT_DIR / "stability_heatmap")

    def plot_herrera_factor(self) -> None:
        """The Herrera factor implied by the final weights, against the order."""
        records = [r for r in self._deployed_records() if r.optimizer_family == "herrera"]
        if not records:
            return
        orders = sorted({r.setting for r in records})
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
        for activation in self.activation_order:
            factors, gradients = [], []
            for order in orders:
                group = [
                    r for r in records
                    if r.activation == activation and r.setting == order
                ]
                factors.append(
                    float(np.nanmedian([r.herrera_factor_median for r in group]))
                    if group else np.nan
                )
                gradients.append(
                    float(np.mean([r.grad_norm_mean for r in group])) if group else np.nan
                )
            color = ACTIVATION_COLORS.get(activation, COLOR_PALETTE_A[8])
            axes[0].plot(orders, factors, marker="o", markersize=5.0, color=color,
                         label=self._display(activation))
            axes[1].plot(orders, gradients, marker="o", markersize=5.0, color=color,
                         label=self._display(activation))
        axes[0].set_yscale("log")
        axes[0].set_xlabel("fractional order")
        axes[0].set_ylabel("median Herrera factor at the final weights")
        axes[0].set_title("The factor the Herrera family applies")
        axes[0].grid(True, alpha=0.25, which="both")
        axes[0].legend(frameon=False, fontsize=8 * FONT_SCALE)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("fractional order")
        axes[1].set_ylabel("mean global gradient norm")
        axes[1].set_title("The gradient that results")
        axes[1].grid(True, alpha=0.25, which="both")
        fig.suptitle("Herrera-type optimizers: scaling factor and resulting gradient")
        self._save(fig, PLOT_DIR / "herrera_factor")

    def make_all(self) -> None:
        self.plot_gradient_by_family()
        self.plot_coherence()
        self.plot_truncation_effect()
        self.plot_stability_heatmap()
        self.plot_herrera_factor()


# =============================================================================
# 11) OUTPUT
# =============================================================================

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class ProbeReport:
    """The human-readable report, the tables and the machine record."""

    def __init__(
        self,
        records: Sequence[RunRecord],
        summaries: Sequence[CellSummary],
        activations: ActivationFactory,
        optimizers: OptimizerFactory,
        environment: Dict[str, Any],
        failures: Sequence[Dict[str, str]],
    ) -> None:
        self.records = list(records)
        self.summaries = list(summaries)
        self.activations = activations
        self.optimizers = optimizers
        self.environment = environment
        self.failures = list(failures)
        self.activation_order = [
            name for name in activations.names()
            if any(r.activation == name for r in self.records)
        ]
        self.family_order = [
            family for family in INCLUDE_FAMILIES
            if any(r.optimizer_family == family for r in self.records)
        ]

    # ---- helpers ------------------------------------------------------------

    def _deployed(self) -> List[RunRecord]:
        out = []
        for record in self.records:
            if record.terms == self.activations.spec(record.activation)["deployed_terms"]:
                out.append(record)
        return out

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        return float(np.mean(array)) if array.size else float("nan")

    def _display(self, activation: str) -> str:
        return self.activations.display(activation)

    # ---- tables -------------------------------------------------------------

    def write_tables(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if self.summaries:
            rows = [asdict(s) for s in self.summaries]
            with open(OUTPUT_DIR / "cell_summary.csv", "w", newline="",
                      encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        if self.failures:
            with open(OUTPUT_DIR / "failed_runs.csv", "w", newline="",
                      encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.failures[0].keys()))
                writer.writeheader()
                writer.writerows(self.failures)

    def write_json(self) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "activation_optimizer_analysis.json"
        payload = {
            "format_version": 1,
            "produced_by": "B_05_activation_optimizer_gradient_probe.py",
            "settings": {
                "dataset": PROBE_DATASET,
                "hyperparameters": DATASET_HYPERPARAMS[PROBE_DATASET],
                "runs_per_cell": NETWORK_PROBE_RUNS,
                "seed_val": SEED_VAL,
                "truncations_to_probe": TRUNCATIONS_TO_PROBE,
                "order_grid": ORDER_GRID,
                "baseline_learning_rate_grid": BASELINE_LEARNING_RATE_GRID,
                "optimizer_families": {
                    family: OPTIMIZER_FAMILIES[family] for family in INCLUDE_FAMILIES
                },
                "feed_loss_to_adaptive_optimizers": FEED_LOSS_TO_ADAPTIVE_OPTIMIZERS,
                "slot_check_every": SLOT_CHECK_EVERY,
            },
            "environment": self.environment,
            "cell_summary": [asdict(s) for s in self.summaries],
            "failures": self.failures,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2)
        return path

    def write_latex_table(self) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "activation_optimizer_tables.tex"
        deployed = self._deployed()
        lines: List[str] = [
            "% Produced by B_05_activation_optimizer_gradient_probe.py", ""
        ]
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\begin{tabular}{ll" + "r" * 6 + "}")
        lines.append("\\hline")
        lines.append(
            "activation & family & mean $\\|g\\|$ & mean $\\|\\Delta\\theta\\|$ & "
            "coherence & non-finite & accuracy & runs \\\\"
        )
        lines.append("\\hline")
        for activation in self.activation_order:
            for family in self.family_order:
                group = [
                    r for r in deployed
                    if r.activation == activation and r.optimizer_family == family
                ]
                if not group:
                    continue
                nonfinite = sum(
                    r.nonfinite_gradient_steps + r.nonfinite_loss_steps
                    + r.nonfinite_update_steps + r.optimizer_slot_nonfinite
                    for r in group
                )
                lines.append(
                    f"{self._display(activation)} & {FAMILY_DISPLAY.get(family, family)} & "
                    f"{self._mean([r.grad_norm_mean for r in group]):.2e} & "
                    f"{self._mean([r.update_norm_mean for r in group]):.2e} & "
                    f"{self._mean([r.grad_cosine_mean for r in group]):+.3f} & "
                    f"{nonfinite} & "
                    f"{self._mean([r.test_accuracy for r in group]):.4f} & "
                    f"{len(group)} \\\\"
                )
            lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append(
            "\\caption{What each optimizer family receives and applies on each "
            f"activation at its deployed truncation, on {PROBE_DATASET}, "
            f"averaged over {NETWORK_PROBE_RUNS} seeds and over the four "
            "settings of every optimizer in the family. Coherence is the mean "
            "cosine similarity between the gradients of consecutive steps, "
            "which is the quantity the memory mechanisms average over. The "
            "non-finite column counts steps with a non-finite loss, gradient or "
            "update together with non-finite entries found in the optimizer's "
            "own slot variables.}"
        )
        lines.append("\\label{tab:activation_optimizer_probe}")
        lines.append("\\end{table}")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return path

    # ---- report -------------------------------------------------------------

    def build_report(self) -> str:
        lines: List[str] = []
        deployed = self._deployed()

        lines.append(RULE)
        lines.append(
            "What the optimizers of the study receive from each activation"
        )
        lines.append(RULE)
        lines.append("")
        lines.append(f"dataset        : {PROBE_DATASET} "
                     f"{DATASET_HYPERPARAMS[PROBE_DATASET]}")
        lines.append(f"runs           : {len(self.records)} finished, "
                     f"{NETWORK_PROBE_RUNS} seeds per cell, "
                     f"{len(self.failures)} failed")
        lines.append(f"optimizers     : {len(self.optimizers.names())} across "
                     f"{len(self.family_order)} families, "
                     f"orders {ORDER_GRID} and the learning-rate grid of B_01")
        lines.append(f"activations    : {len(self.activation_order)}, probed at "
                     f"{TRUNCATIONS_TO_PROBE} terms and at their deployed length")
        lines.append(
            "loss injection : set_current_loss "
            + (
                "enabled, as in the surface experiments A_01 and A_02"
                if FEED_LOSS_TO_ADAPTIVE_OPTIMIZERS
                else "disabled, as in the classification experiments B_01; the "
                     "stability input of the adaptive mix is unaffected"
            )
        )
        lines.append(f"environment    : {self.environment}")
        lines.append("")

        lines.append(THIN_RULE)
        lines.append("1) Gradient, update and coherence at the deployed truncations")
        lines.append(THIN_RULE)
        lines.append(
            f"{'activation':>16} {'family':>18} {'mean|g|':>11} {'max|g|':>11} "
            f"{'mean|dtheta|':>13} {'coherence':>10} {'nf steps':>9} "
            f"{'nf slots':>9} {'accuracy':>9} {'runs':>5}"
        )
        for activation in self.activation_order:
            for family in self.family_order:
                group = [
                    r for r in deployed
                    if r.activation == activation and r.optimizer_family == family
                ]
                if not group:
                    continue
                lines.append(
                    f"{self._display(activation):>16} "
                    f"{FAMILY_DISPLAY.get(family, family):>18} "
                    f"{self._mean([r.grad_norm_mean for r in group]):>11.3e} "
                    f"{np.max([r.grad_norm_max for r in group]):>11.3e} "
                    f"{self._mean([r.update_norm_mean for r in group]):>13.3e} "
                    f"{self._mean([r.grad_cosine_mean for r in group]):>+10.3f} "
                    f"{sum(r.nonfinite_gradient_steps + r.nonfinite_loss_steps + r.nonfinite_update_steps for r in group):>9} "
                    f"{sum(r.optimizer_slot_nonfinite for r in group):>9} "
                    f"{self._mean([r.test_accuracy for r in group]):>9.4f} "
                    f"{len(group):>5}"
                )
            lines.append("")
        lines.append(
            "Coherence is the mean cosine similarity between the gradients of "
            "consecutive steps. The memory\nfamilies form a weighted sum over a "
            "gradient history and norm-match it back to the current\nmagnitude, "
            "so a coherence near zero means that what they average carries no "
            "direction and the\nmechanism reduces to its fallback."
        )
        lines.append("")

        lines.append(THIN_RULE)
        lines.append("2) Effect of the truncation length")
        lines.append(THIN_RULE)
        lines.append(
            f"{'activation':>16} {'terms':>6} {'mean|g|':>11} {'mean|dtheta|':>13} "
            f"{'coherence':>10} {'nf steps':>9} {'accuracy':>9} {'runs':>5}"
        )
        for activation in self.activation_order:
            truncations = sorted(
                {r.terms for r in self.records if r.activation == activation},
                key=lambda t: (t is None, t),
            )
            for terms in truncations:
                group = [
                    r for r in self.records
                    if r.activation == activation and r.terms == terms
                ]
                if not group:
                    continue
                lines.append(
                    f"{self._display(activation):>16} {str(terms):>6} "
                    f"{self._mean([r.grad_norm_mean for r in group]):>11.3e} "
                    f"{self._mean([r.update_norm_mean for r in group]):>13.3e} "
                    f"{self._mean([r.grad_cosine_mean for r in group]):>+10.3f} "
                    f"{sum(r.nonfinite_gradient_steps + r.nonfinite_loss_steps + r.nonfinite_update_steps for r in group):>9} "
                    f"{self._mean([r.test_accuracy for r in group]):>9.4f} "
                    f"{len(group):>5}"
                )
            lines.append("")

        herrera = [r for r in deployed if r.optimizer_family == "herrera"]
        if herrera:
            lines.append(THIN_RULE)
            lines.append("3) Herrera-type optimizers: the factor they apply")
            lines.append(THIN_RULE)
            lines.append(
                f"{'activation':>16} {'order':>7} {'factor median':>14} "
                f"{'factor max':>12} {'mean|g|':>11} {'nf steps':>9} "
                f"{'nf slots':>9} {'accuracy':>9}"
            )
            for activation in self.activation_order:
                for order in sorted({r.setting for r in herrera}):
                    group = [
                        r for r in herrera
                        if r.activation == activation and r.setting == order
                    ]
                    if not group:
                        continue
                    lines.append(
                        f"{self._display(activation):>16} {order:>7.2f} "
                        f"{self._mean([r.herrera_factor_median for r in group]):>14.3e} "
                        f"{np.nanmax([r.herrera_factor_max for r in group]):>12.3e} "
                        f"{self._mean([r.grad_norm_mean for r in group]):>11.3e} "
                        f"{sum(r.nonfinite_gradient_steps + r.nonfinite_loss_steps + r.nonfinite_update_steps for r in group):>9} "
                        f"{sum(r.optimizer_slot_nonfinite for r in group):>9} "
                        f"{self._mean([r.test_accuracy for r in group]):>9.4f}"
                    )
                lines.append("")
            lines.append(
                "The factor is (|theta| + eps)^(1-nu)/Gamma(2-nu) evaluated on the "
                "parameters, reported from the\nfinal weights of each run. Its "
                "bound at |theta| = eps is 25.8, 564 and 8722 for nu = 1.25, 1.50 "
                "and\n1.75, so an activation that already supplies a large "
                "gradient can drive a second-moment\naccumulator towards the "
                "float32 ceiling. Whether that happens is what the two non-finite "
                "columns\nreport."
            )
            lines.append("")

        memory = [
            r for r in deployed
            if r.optimizer_family in ("memory", "adaptive_memory")
        ]
        if memory:
            lines.append(THIN_RULE)
            lines.append("4) Memory-based optimizers: what the history contains")
            lines.append(THIN_RULE)
            lines.append(
                f"{'activation':>16} {'family':>18} {'order':>7} {'coherence':>10} "
                f"{'GL coeff sum':>13} {'mix':>8} {'mean|g|':>11} {'accuracy':>9}"
            )
            for activation in self.activation_order:
                for family in ("memory", "adaptive_memory"):
                    for order in sorted({r.setting for r in memory}):
                        group = [
                            r for r in memory
                            if r.activation == activation
                            and r.optimizer_family == family
                            and r.setting == order
                        ]
                        if not group:
                            continue
                        lines.append(
                            f"{self._display(activation):>16} "
                            f"{FAMILY_DISPLAY.get(family, family):>18} "
                            f"{order:>7.2f} "
                            f"{self._mean([r.grad_cosine_mean for r in group]):>+10.3f} "
                            f"{self._mean([r.gl_coefficient_sum for r in group]):>13.4f} "
                            f"{self._mean([r.adaptive_mix_mean for r in group]):>8.4f} "
                            f"{self._mean([r.grad_norm_mean for r in group]):>11.3e} "
                            f"{self._mean([r.test_accuracy for r in group]):>9.4f}"
                        )
                lines.append("")
            lines.append(
                "The GL coefficient sum is the steady-state gain of the memory "
                "filter before norm matching, read\nfrom the optimizer. The mix "
                "column is the realized fractional mixing weight of the adaptive\n"
                "family, bounded by construction at "
                f"{MINIMAL_ADAPTIVE_MAX_MIX}."
            )
            lines.append("")

        if self.failures:
            lines.append(THIN_RULE)
            lines.append("Failed configurations")
            lines.append(THIN_RULE)
            for failure in self.failures[:40]:
                lines.append(
                    f"  {failure['optimizer']} @ {failure['setting']} on "
                    f"{failure['activation']} (terms {failure['terms']}, "
                    f"seed {failure['seed']}): {failure['error']}"
                )
            if len(self.failures) > 40:
                lines.append(f"  ... and {len(self.failures) - 40} more, see failed_runs.csv")
            lines.append("")

        lines.append(RULE)
        lines.append("Conclusions supported by the numbers above")
        lines.append(RULE)
        for text in self.conclusions():
            lines.append(f"  - {text}")
        lines.append("")
        lines.append(RULE)
        return "\n".join(lines)

    def conclusions(self) -> List[str]:
        out: List[str] = []
        deployed = self._deployed()
        if not deployed:
            return ["no finished runs"]

        # gradient magnitude by activation
        magnitudes = []
        for activation in self.activation_order:
            group = [r for r in deployed if r.activation == activation]
            if group:
                magnitudes.append(
                    (self._display(activation), self._mean([r.grad_norm_mean for r in group]))
                )
        if magnitudes:
            magnitudes.sort(key=lambda item: item[1])
            out.append(
                "Averaged over every optimizer and setting, the mean global "
                "gradient norm per step is "
                + ", ".join(f"{name} {value:.3e}" for name, value in magnitudes)
                + "."
            )

        # coherence
        coherences = []
        for activation in self.activation_order:
            group = [r for r in deployed if r.activation == activation]
            if group:
                coherences.append(
                    (self._display(activation), self._mean([r.grad_cosine_mean for r in group]))
                )
        if coherences:
            coherences.sort(key=lambda item: item[1], reverse=True)
            out.append(
                "The mean cosine similarity between consecutive gradients is "
                + ", ".join(f"{name} {value:+.3f}" for name, value in coherences)
                + ". Where it is near zero, a weighted gradient history has no "
                "direction to average and the memory mechanisms of this study "
                "operate on noise."
            )

        # stability incidents
        incidents: List[str] = []
        for activation in self.activation_order:
            for family in self.family_order:
                group = [
                    r for r in deployed
                    if r.activation == activation and r.optimizer_family == family
                ]
                affected = [
                    r for r in group
                    if r.nonfinite_gradient_steps or r.nonfinite_loss_steps
                    or r.nonfinite_update_steps or r.optimizer_slot_nonfinite
                ]
                if affected:
                    incidents.append(
                        f"{self._display(activation)} with {FAMILY_DISPLAY.get(family, family)} "
                        f"({len(affected)} of {len(group)} runs)"
                    )
        if incidents:
            out.append(
                "Non-finite values occurred in " + "; ".join(incidents) +
                ". The activation diagnostics alone produced none at any "
                "truncation, so these arise from the interaction with the "
                "optimizer rather than from the activation by itself."
            )
        else:
            out.append(
                "No run produced a non-finite loss, gradient, update or optimizer "
                "slot entry. The large gradients of the high-frequency activation "
                "stay within the float32 range even after the fractional scaling "
                "factors are applied."
            )

        # update magnitude: what the optimizer does with a huge gradient
        for activation in self.activation_order:
            group = [r for r in deployed if r.activation == activation]
            if not group:
                continue
            gradient = self._mean([r.grad_norm_mean for r in group])
            update = self._mean([r.update_norm_mean for r in group])
            if np.isfinite(gradient) and np.isfinite(update) and update > 0:
                out.append(
                    f"{self._display(activation)}: a mean gradient norm of "
                    f"{gradient:.3e} results in a mean applied update of "
                    f"{update:.3e}, a ratio of {gradient / update:.3e}."
                )

        # truncation effect on accuracy
        for activation in self.activation_order:
            spec = self.activations.spec(activation)
            if spec["deployed_terms"] is None:
                continue
            truncations = sorted(
                {r.terms for r in self.records if r.activation == activation and r.terms}
            )
            if len(truncations) < 2:
                continue
            first = [r for r in self.records
                     if r.activation == activation and r.terms == truncations[0]]
            last = [r for r in self.records
                    if r.activation == activation and r.terms == truncations[-1]]
            if first and last:
                out.append(
                    f"{self._display(activation)}: between {truncations[0]} and "
                    f"{truncations[-1]} terms the mean gradient norm moves from "
                    f"{self._mean([r.grad_norm_mean for r in first]):.3e} to "
                    f"{self._mean([r.grad_norm_mean for r in last]):.3e} and the "
                    f"mean test accuracy from "
                    f"{self._mean([r.test_accuracy for r in first]):.4f} to "
                    f"{self._mean([r.test_accuracy for r in last]):.4f}."
                )

        return out

    def write_report(self) -> Tuple[Path, str]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        text = self.build_report()
        path = OUTPUT_DIR / "activation_optimizer_report.txt"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        return path, text


# =============================================================================
# 12) MAIN
# =============================================================================

def collect_environment() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "matplotlib": matplotlib.__version__,
    }


def main() -> None:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    activations = ActivationFactory()
    optimizers = OptimizerFactory()
    provider = DatasetProvider(PROBE_DATASET)

    activation_settings = sum(
        len(activations.settings_for(name)) for name in activations.names()
    )
    optimizer_settings = sum(
        len(optimizers.settings_for(name)) for name in optimizers.names()
    )
    total = activation_settings * optimizer_settings * NETWORK_PROBE_RUNS

    print("=" * 96)
    print("Gradient probe across the activations and optimizers of the study")
    print("=" * 96)
    print(f"dataset            : {PROBE_DATASET} {DATASET_HYPERPARAMS[PROBE_DATASET]}")
    print(f"activation settings: {activation_settings} "
          f"({len(activations.names())} activations, truncations "
          f"{TRUNCATIONS_TO_PROBE} plus the deployed one)")
    print(f"optimizer settings : {optimizer_settings} "
          f"({len(optimizers.names())} optimizers, four settings each)")
    print(f"seeds              : {NETWORK_PROBE_RUNS}")
    print(f"total runs         : {total}")
    print(f"results file       : {RECORDS_FILE} "
          f"({'resuming' if RESUME else 'resume disabled'})")
    print("")

    driver = ProbeDriver(provider, activations, optimizers)
    driver.run()

    records = driver.load_all()
    summaries = summarize(records)

    print("\nPlotting")
    ProbePlotter(records, summaries, activations).make_all()

    report = ProbeReport(
        records, summaries, activations, optimizers,
        collect_environment(), driver.failures,
    )
    report.write_tables()
    json_path = report.write_json()
    tex_path = report.write_latex_table()
    report_path, text = report.write_report()

    print("")
    print(text)
    print("")
    print(f"Report:       {report_path}")
    print(f"Runs:         {RECORDS_FILE}")
    print(f"JSON record:  {json_path}")
    print(f"LaTeX table:  {tex_path}")
    print(f"Plots:        {PLOT_DIR}")
    print(f"Elapsed:      {(time.time() - started) / 3600.0:.2f} h")


if __name__ == "__main__":
    main()
