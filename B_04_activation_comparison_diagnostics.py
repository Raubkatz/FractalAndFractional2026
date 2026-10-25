#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
B_04_activation_comparison_diagnostics.py

Numerical diagnostic of all six activation functions used in the
classification experiments, written for the supplementary appendix and in
response to the reviewer request for a check on the numerical sensitivity of
the high-frequency activation:

    "The revised discussion of the Weierstrass-Mandelbrot activation would
     benefit from a brief additional numerical check. A limited comparison
     using a few truncation lengths, such as 8, 16, and 24 terms, together
     with simple gradient-magnitude and non-finite-value diagnostics, could
     help determine whether the reported behaviour is sensitive to unresolved
     high-frequency components."

The predecessor of this script, B_03_activation_truncation_sensitivity.py,
answered that question for the x+sin activation alone. This script runs the
same measurements for every activation of the study, so that the behaviour of
the x+sin variant can be stated as a comparison rather than as an isolated
observation, and so that the appendix can show what the other three fractal
activations do differently.

THE SIX ACTIVATIONS
-------------------
Two standard activations, which serve as the control, and four fractal
activations, taken with the parameters at which they were deployed:

    relu                                  max(x, 0)
    tanh                                  tanh(x)
    modulated_blancmange_curve            n = 0..29,  a = 0.75
    decaying_cosine_function_tf           n = 0..74,  c = 0.5, d = 2, zeta = 0.2666
    modified_weierstrass_function_tanh    n = 0..99,  a = 0.5, b = 1.5
    weierstrass_mandelbrot_function_xpsin k = 1..99,  gamma = 0.5, lambda = 2

Each fractal activation is a sum over scales. Scale index m contributes an
oscillation of frequency f_m, an amplitude to the function value, and an
amplitude to the derivative equal to 2*pi*f_m times the value amplitude. The
value amplitudes of all four decay geometrically, so every value series
converges. The derivative ladders behave differently, and that difference is
the subject of the comparison:

    modified Weierstrass-tanh   pi*(a*b)^n = pi*0.75^n   convergent
    modulated blancmange        bounded per scale        marginal
    decaying cosine             zeta*pi*(c*d)^n = const  marginal
    x+sin                       2*pi*2^(k/2)             divergent

WHAT THIS SCRIPT MEASURES
-------------------------
    1) Deployed-setting comparison. Value and derivative statistics of all six
       activations on one fixed sample of pre-activations, with non-finite
       counts, for float32 and float64 pre-activations.

    2) Truncation sweep. Each series activation is evaluated over a grid of
       truncation lengths that includes the lengths named in the review, 8, 16
       and 24 terms, and its own deployed length. Reported per truncation: the
       deviation of the value from the deployed truncation, the derivative
       statistics, the non-finite counts, and the closed-form derivative
       amplitude bound.

    3) Scale structure and phase regimes. For every scale of every series
       activation the script measures the circular concentration of the phase,
       the mean of cos and of |sin|, and the fraction of sample points at which
       the scale lands exactly on a lattice point. This separates scales whose
       oscillation is real from scales that have degenerated, and it does so by
       measurement rather than by assumption.

    4) Input precision probe. Every sample point is shifted by one and by four
       units in the last place of its float32 representation, which is below
       the accuracy with which a network can place that pre-activation.
       Reported: the change of the value, the relative change of the
       derivative, and the fraction of points at which the derivative changes
       sign.

    5) Gradient realizability. The analytic derivative, which is what automatic
       differentiation returns, is compared against the slope the function
       actually realizes between neighbouring float32 inputs, at three step
       sizes. The ratio of the two, and their rank correlation, say whether the
       gradient the optimizer receives describes the function the network
       computes.

    6) Compact network probe (optional, requires TensorFlow). One small
       classification dataset is trained with the deployed architecture, for
       every activation and for several truncation lengths, recording per-step
       global gradient norms, the number of steps with non-finite loss or
       gradients, the reached test accuracy and the observed pre-activation
       range.

    In addition, where TensorFlow is available, every value and derivative of
    the numpy reference is cross-checked against the deployed implementation
    and its automatic derivative.

WHY THIS COMPARISON IS THE POINT
--------------------------------
A fractal activation is usually described by the regularity of its limit. That
description says nothing about what a network trained on it actually receives,
because training consumes the derivative of a truncated sum evaluated in finite
precision. Three properties decide that, and all three are measured here.

First, whether the derivative ladder converges. If it does, as for the modified
Weierstrass-tanh activation, adding scales leaves the derivative bounded and
the truncation length is a detail. If it is marginal, as for the blancmange and
the decaying cosine, the bound grows linearly with the number of scales and
stays of order ten. If it diverges, as for the x+sin variant, the derivative
magnitude is set by the finest retained scale alone and nothing else.

Second, whether a scale is resolved. A scale of frequency f is only meaningful
if the representation error of the pre-activation, about 6e-8 relative in
float32, moves its phase by less than one radian. Scales beyond that limit are
not rough, they are arbitrary.

Third, what an unresolved scale degenerates into. Three of the four fractal
activations use a frequency ladder of base two, and a float32 number times a
power of two is exact, so beyond the mantissa length those scales land exactly
on the lattice of their own period: the oscillation does not become noisy, it
disappears, leaving a fixed value and, for the x+sin variant, a fixed positive
contribution to the derivative. Further out, when the phase product can no
longer be formed accurately, the oscillation returns as a function of the
rounding rather than of the input. The modified Weierstrass-tanh activation
uses b = 1.5, which is not a power of two, and therefore does not show the
lattice degeneracy at all.

The outcome, which the report states with the measured numbers, is that no
activation produces a single non-finite value or derivative, that every value
series is converged well before its deployed truncation, and that the four
fractal activations nevertheless differ by fourteen orders of magnitude in the
gradient they supply, with only the x+sin variant supplying a gradient that
bears no relation to the slope of the function it computes.

RUNNING
-------
No command line arguments. All parameters are set in section 0. TensorFlow is
optional: without it, stages 1 to 5 run from the numpy reference
implementations and stage 6 is skipped; with it, the autodiff cross-check and
the network probe are added. The script expects fractal_activation_functions.py
in the same directory when TensorFlow is used.

    python3 B_04_activation_comparison_diagnostics.py

The numpy stages take a few minutes at the default sample size. The network
probe is the expensive part: it trains
len(NETWORK_PROBE_OPTIMIZERS) * NETWORK_PROBE_RUNS networks per activation and
truncation, and can be switched off with NETWORK_PROBE_ENABLED.

Outputs
-------
    <OUTPUT_DIR>/activation_comparison_report.txt
    <OUTPUT_DIR>/deployed_comparison.csv
    <OUTPUT_DIR>/truncation_sweep.csv
    <OUTPUT_DIR>/scale_structure.csv
    <OUTPUT_DIR>/precision_probe.csv
    <OUTPUT_DIR>/gradient_realizability.csv
    <OUTPUT_DIR>/autodiff_crosscheck.csv            (only with TensorFlow)
    <OUTPUT_DIR>/network_probe.csv                  (only with TensorFlow)
    <OUTPUT_DIR>/activation_comparison_analysis.json
    <OUTPUT_DIR>/activation_comparison_tables.tex
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
from dataclasses import asdict, dataclass, field
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

OUTPUT_DIR = Path("activation_comparison_diagnostics")
PLOT_DIR = OUTPUT_DIR / "plots"

# -----------------------------------------------------------------------------
# Deployed parameters of the activations, taken from
# fractal_activation_functions.py. Changing any of these changes what is being
# measured and no longer describes the experiments of the study.
BLANCMANGE_N_TERMS = 30
BLANCMANGE_A = 0.75

DECAYING_COSINE_C = 0.5
DECAYING_COSINE_D = 2
DECAYING_COSINE_N_TERMS = 75
DECAYING_COSINE_ZETA = 0.2666

MOD_WEIERSTRASS_TANH_A = 0.5
MOD_WEIERSTRASS_TANH_B = 1.5
MOD_WEIERSTRASS_TANH_N_TERMS = 100

XPSIN_GAMMA = 0.5
XPSIN_LAMBDA = 2
XPSIN_NUM_TERMS = 100

# -----------------------------------------------------------------------------
# Truncation grid. Each activation is swept over the entries of this grid that
# do not exceed its deployed number of terms, plus its deployed number itself.
# The first three entries are the lengths named in the review.
TRUNCATION_GRID = [8, 16, 24, 32, 50, 75, 100]
INCLUDE_SHORT_TRUNCATION = 4

# -----------------------------------------------------------------------------
# Sampling of pre-activations. Features are MinMax-scaled to [0, 1] and the
# first layer uses Glorot-uniform weights, so the pre-activations of the
# deployed networks concentrate well inside [-3, 3]. The sample is drawn once
# from a fixed seed and reused by every stage and every activation, so all
# reported numbers refer to the same points.
SAMPLE_RANGE = 3.0
N_SAMPLE_POINTS = 100000
SAMPLE_SEED = 20260224
EXCLUDE_ZERO_RADIUS = 1e-6

# Dense grids used for the curve plots only.
PLOT_WINDOW_WIDE = 3.0
PLOT_WINDOW_NARROW = 0.02
PLOT_GRID_N = 8001

# Precisions compared in stage 1. The sample is quantised to the stated
# precision before evaluation, which is what the network does when it hands a
# float32 tensor to the activation.
SAMPLE_PRECISIONS = ["float32", "float64"]

# -----------------------------------------------------------------------------
# Scale structure. Two statistics describe what the phase p = f*x of a scale
# does over the sample.
#
#   phase concentration  |mean of exp(2*pi*i*p)|, zero for a phase spread
#                        uniformly over the circle, one for a phase that does
#                        not move,
#   lattice moment       the largest of |mean of exp(2*pi*i*k*p)| over
#                        k in PHASE_MOMENT_ORDERS. The higher orders catch a
#                        phase that has collapsed onto a coarse lattice with
#                        several points, such as the two values {0, 1/2} that
#                        occur at the scale where the input mantissa runs out,
#                        which the first moment alone reports as spread.
#   lattice fraction     the fraction of sample points at which p is an exact
#                        integer. This happens either because the input lies on
#                        the lattice of the scale, which is what a frequency
#                        ladder of base two produces, or because the product
#                        f*x has grown so large that it has no fractional part
#                        left in the dtype that formed it.
SCALE_STRUCTURE_ENABLED = True
LATTICE_FRACTION_THRESHOLD = 0.5
PHASE_CONCENTRATION_THRESHOLD = 0.5
PHASE_MOMENT_ORDERS = [1, 2, 4, 8]

# -----------------------------------------------------------------------------
# Input precision probe.
ULP_MULTIPLES = [1.0, 4.0]

# -----------------------------------------------------------------------------
# Gradient realizability. The analytic derivative is compared against the slope
# realized between neighbouring float32 inputs. A step of None means the next
# representable float32 number; a float means a relative step of that size.
REALIZABILITY_ENABLED = True
REALIZABILITY_STEPS: List[Optional[float]] = [None, 1e-5, 1e-3]

# Tolerance within which a truncated value counts as reproducing the deployed
# function, used by the conclusions of the report.
VALUE_CONVERGENCE_TOLERANCE = 1e-3

# -----------------------------------------------------------------------------
# Autodiff cross-check against the deployed TensorFlow implementations.
AUTODIFF_CROSSCHECK_ENABLED = True
AUTODIFF_CROSSCHECK_POINTS = 20000
AUTODIFF_CROSSCHECK_TRUNCATIONS = [8, 16, 24]

# -----------------------------------------------------------------------------
# Compact network probe. Deliberately small: one dataset, a few seeds, the
# truncations named in the review plus the deployed one. Hyperparameters follow
# Table 1 of the study for the chosen dataset.
NETWORK_PROBE_ENABLED = True
NETWORK_PROBE_DATASET = "seeds"
NETWORK_PROBE_TRUNCATIONS = [8, 16, 24]
NETWORK_PROBE_OPTIMIZERS = ["sgd", "adam"]
NETWORK_PROBE_LEARNING_RATE = 0.001
NETWORK_PROBE_RUNS = 3
NETWORK_PROBE_SEED_VAL = 238974
NETWORK_PROBE_GRAD_NORM_REPORT_THRESHOLD = 1e6

NETWORK_PROBE_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "seeds":    {"task": "multi-class", "n": 64,  "hidden_layers": 2, "batch_size": 32, "epochs": 30},
    "iris":     {"task": "multi-class", "n": 32,  "hidden_layers": 2, "batch_size": 16, "epochs": 30},
    "wine":     {"task": "multi-class", "n": 64,  "hidden_layers": 2, "batch_size": 32, "epochs": 30},
    "glass":    {"task": "multi-class", "n": 64,  "hidden_layers": 2, "batch_size": 32, "epochs": 30},
    "diabetes": {"task": "binary",      "n": 64,  "hidden_layers": 1, "batch_size": 32, "epochs": 30},
}

# -----------------------------------------------------------------------------
# Machine precision constants used by the resolution accounting.
EPS_FLOAT32 = float(np.finfo(np.float32).eps)
EPS_FLOAT64 = float(np.finfo(np.float64).eps)
MANTISSA_BITS_FLOAT32 = int(np.finfo(np.float32).nmant)

# -----------------------------------------------------------------------------
# Properties of the deployed implementations that the written formulas do not
# show, read from fractal_activation_functions.py. They are listed in the
# report because they change the numbers.
IMPLEMENTATION_NOTES: List[str] = [
    "weierstrass_mandelbrot_function_xpsin loops over range(1, num_terms), so "
    "num_terms = 100 retains 99 scales, k = 1 to 99. The manuscript writes the "
    "sum as k = 1 to N; the two differ by one scale, that is by a factor "
    "2^(1/2) in the derivative amplitude.",
    "modulated_blancmange_curve divides the accumulated sum by two before "
    "returning it.",
    "decaying_cosine_function_tf carries the unused parameters a and b in its "
    "signature; the ladder is governed by c and d. Its tanh term sits inside "
    "the loop and is therefore added n_terms times, so truncating this "
    "activation changes its non-oscillating part as well.",
    "modulated_blancmange_curve and decaying_cosine_function_tf accumulate in "
    "the dtype of their input, which is float32 inside the networks. "
    "modified_weierstrass_function_tanh and "
    "weierstrass_mandelbrot_function_xpsin cast to float64, accumulate there "
    "and cast the result back to float32.",
    "Every activation returns float32 to the Keras layer, so both the value "
    "and the gradient are rounded to float32 before the optimizer sees them. "
    "The numpy references below reproduce that.",
]

# -----------------------------------------------------------------------------
# Plotting. Style follows the surface and perturbation scripts of the study.
# One colour and one marker per activation, used in every figure.
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

ACTIVATION_MARKERS: Dict[str, str] = {
    "relu": "o",
    "tanh": "s",
    "modulated_blancmange_curve": "^",
    "decaying_cosine_function_tf": "D",
    "modified_weierstrass_function_tanh": "v",
    "weierstrass_mandelbrot_function_xpsin": "P",
}

# The base-two activations share the same scale frequencies, so their curves
# coincide exactly in some panels. Distinct dash patterns keep an overdrawn
# curve visible, which transparency would not do in the eps output.
ACTIVATION_LINESTYLES: Dict[str, Any] = {
    "relu": "solid",
    "tanh": (0, (5, 2)),
    "modulated_blancmange_curve": (0, (1, 1.4)),
    "decaying_cosine_function_tf": (0, (5, 2)),
    "modified_weierstrass_function_tanh": (0, (7, 2, 1.5, 2)),
    "weierstrass_mandelbrot_function_xpsin": "solid",
}

REGIME_COLORS: Dict[str, str] = {
    "oscillating": COLOR_PALETTE_A[0],
    "lattice_locked": COLOR_PALETTE_A[3],
    "rounding_dominated": COLOR_PALETTE_A[6],
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

RULE = "=" * 118
THIN_RULE = "-" * 118


# -----------------------------------------------------------------------------
# Optional TensorFlow import. The numpy stages never depend on it.
TENSORFLOW_AVAILABLE = False
TENSORFLOW_IMPORT_ERROR = ""
ACTIVATION_MODULE_AVAILABLE = False
ACTIVATION_MODULE_IMPORT_ERROR = ""

try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the local install
    TENSORFLOW_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

if TENSORFLOW_AVAILABLE:
    try:
        import fractal_activation_functions as faf

        ACTIVATION_MODULE_AVAILABLE = True
    except Exception as exc:  # pragma: no cover - depends on the local files
        ACTIVATION_MODULE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# =============================================================================
# 1) NUMPY REFERENCE ACTIVATIONS
# =============================================================================

class ActivationReference:
    """
    Numpy mirror of one activation of the study.

    Each subclass reproduces the deployed implementation exactly, including the
    dtype in which it accumulates and the cast to float32 with which it hands
    its result to the Keras layer, and provides the analytic derivative, which
    is what automatic differentiation returns.

    Series activations additionally expose their scale structure: the frequency
    of each scale in cycles per unit of the pre-activation, and an upper bound
    for the contribution of each scale to the derivative. Those two quantities
    are what the resolution accounting and the ladder classification are built
    from.
    """

    name: str = ""
    display_name: str = ""
    family: str = "standard"
    deployed_terms: Optional[int] = None
    term_parameter: str = ""
    index_range_text: str = ""
    accumulation_dtype: str = "float32"
    value_scale_text: str = ""
    derivative_scale_text: str = ""
    deployed_parameter_text: str = ""

    # ---- dtype handling -----------------------------------------------------

    def _dtype(self) -> Any:
        return np.float64 if self.accumulation_dtype == "float64" else np.float32

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        """Cast the pre-activation into the dtype the deployed code works in."""
        return np.asarray(x, dtype=self._dtype())

    @staticmethod
    def _finish(y: np.ndarray) -> np.ndarray:
        """
        Round to float32, as every activation does when it returns to the Keras
        layer, and widen again so that the statistics are computed exactly.
        """
        return np.asarray(y, dtype=np.float32).astype(np.float64)

    # ---- interface ----------------------------------------------------------

    def is_series(self) -> bool:
        return self.deployed_terms is not None

    def value(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    def retained_scales(self, terms: Optional[int] = None) -> int:
        return 0

    def scale_frequencies(self, terms: Optional[int] = None) -> np.ndarray:
        """Frequency of each retained scale in cycles per unit of x."""
        return np.zeros(0, dtype=np.float64)

    def scale_indices(self, terms: Optional[int] = None) -> np.ndarray:
        return np.zeros(0, dtype=np.int64)

    def derivative_scale_amplitudes(self, terms: Optional[int] = None) -> np.ndarray:
        """Upper bound of the contribution of each retained scale to |phi'|."""
        return np.zeros(0, dtype=np.float64)

    def scale_phase(self, x: np.ndarray, frequency: float) -> np.ndarray:
        """Phase of one scale in cycles, as the deployed dtype computes it."""
        return self._prepare(x) * np.asarray(frequency, dtype=self._dtype())

    # ---- derived quantities -------------------------------------------------

    def truncation_grid(self) -> List[Optional[int]]:
        if not self.is_series():
            return [None]
        deployed = int(self.deployed_terms)
        grid = [INCLUDE_SHORT_TRUNCATION] + list(TRUNCATION_GRID)
        selected = sorted({t for t in grid if 0 < t <= deployed} | {deployed})
        return [int(t) for t in selected]

    def derivative_amplitude_bound(self, terms: Optional[int] = None) -> float:
        amplitudes = self.derivative_scale_amplitudes(terms)
        return float(np.sum(amplitudes)) if amplitudes.size else float("nan")

    def derivative_scale_growth_factor(self, terms: Optional[int] = None) -> float:
        """
        Ratio of the derivative amplitudes of the two finest retained scales.
        Above one the ladder diverges, at one it is marginal and its bound grows
        linearly with the number of scales, below one it converges.
        """
        amplitudes = self.derivative_scale_amplitudes(terms)
        if amplitudes.size < 2 or amplitudes[-2] <= 0.0:
            return float("nan")
        return float(amplitudes[-1] / amplitudes[-2])

    def ladder_type(self, terms: Optional[int] = None) -> str:
        if not self.is_series():
            return "no series"
        factor = self.derivative_scale_growth_factor(terms)
        if not np.isfinite(factor):
            return "undetermined"
        if factor > 1.0 + 1e-9:
            return "divergent"
        if factor < 1.0 - 1e-9:
            return "convergent"
        return "marginal"

    def resolved_scale_count(
        self,
        x_abs: float,
        eps: float,
        terms: Optional[int] = None,
    ) -> int:
        """
        Number of retained scales whose phase is resolved at |x| = x_abs in a
        floating point format with machine epsilon eps. A scale of frequency f
        is resolved if the representation error of x, at most eps*|x|, moves the
        phase 2*pi*f*x by less than one radian.
        """
        frequencies = self.scale_frequencies(terms)
        if frequencies.size == 0:
            return 0
        limit = 1.0 / max(2.0 * math.pi * eps * max(x_abs, 1e-300), 1e-300)
        return int(np.count_nonzero(frequencies <= limit))

    def unresolved_derivative_amplitude(
        self,
        x_abs: float,
        eps: float,
        terms: Optional[int] = None,
    ) -> float:
        amplitudes = self.derivative_scale_amplitudes(terms)
        if amplitudes.size == 0:
            return 0.0
        resolved = self.resolved_scale_count(x_abs, eps, terms)
        return float(np.sum(amplitudes[resolved:]))

    def unresolved_derivative_fraction(
        self,
        x_abs: float,
        eps: float,
        terms: Optional[int] = None,
    ) -> float:
        amplitudes = self.derivative_scale_amplitudes(terms)
        total = float(np.sum(amplitudes)) if amplitudes.size else 0.0
        if total <= 0.0 or not np.isfinite(total):
            return float("nan")
        return self.unresolved_derivative_amplitude(x_abs, eps, terms) / total


class ReLU(ActivationReference):
    """phi(x) = max(x, 0). Control, no scales."""

    name = "relu"
    display_name = "relu"
    family = "standard"
    deployed_parameter_text = "none"
    value_scale_text = "-"
    derivative_scale_text = "-"

    def value(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        return self._finish(np.maximum(self._prepare(x), 0.0))

    def derivative(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        return self._finish((self._prepare(x) > 0.0).astype(self._dtype()))


class Tanh(ActivationReference):
    """phi(x) = tanh(x). Control, no scales."""

    name = "tanh"
    display_name = "tanh"
    family = "standard"
    deployed_parameter_text = "none"
    value_scale_text = "-"
    derivative_scale_text = "-"

    def value(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        return self._finish(np.tanh(self._prepare(x)))

    def derivative(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        return self._finish(1.0 - np.tanh(self._prepare(x)) ** 2)


class ModulatedBlancmange(ActivationReference):
    """
    phi(x) = 0.5 * sum_{n=0}^{N-1} tanh(a*2^n*x) * |(2^n*x mod 2) - a*sqrt(|x|)| / 2^n

    Deployed with N = 30 and a = 0.75, accumulated in the dtype of the input.

    The oscillating object of scale n is the sawtooth 2^n*x mod 2, whose period
    in x is 2^(1-n), so its frequency is 2^(n-1) cycles per unit of x.

    The derivative below is the almost-everywhere derivative, which is what
    automatic differentiation returns: the gradient of the modulo with respect
    to its numerator is one and the gradient of the absolute value is the sign
    of its argument.
    """

    name = "modulated_blancmange_curve"
    display_name = "blancmange"
    family = "fractal"
    deployed_terms = BLANCMANGE_N_TERMS
    term_parameter = "n_terms"
    index_range_text = "n = 0 .. n_terms-1"
    accumulation_dtype = "float32"
    value_scale_text = "2^(-n) * |sawtooth - a sqrt|x||"
    derivative_scale_text = "bounded, approx. 0.5 per scale"

    def __init__(self, a: float = BLANCMANGE_A, n_terms: int = BLANCMANGE_N_TERMS) -> None:
        self.a = float(a)
        self.n_terms = int(n_terms)
        self.deployed_terms = int(n_terms)
        self.deployed_parameter_text = f"a = {self.a}, n_terms = {self.n_terms}"

    def retained_scales(self, terms: Optional[int] = None) -> int:
        return int(self.n_terms if terms is None else terms)

    def scale_indices(self, terms: Optional[int] = None) -> np.ndarray:
        return np.arange(self.retained_scales(terms), dtype=np.int64)

    def scale_frequencies(self, terms: Optional[int] = None) -> np.ndarray:
        return np.power(2.0, self.scale_indices(terms).astype(np.float64)) / 2.0

    def derivative_scale_amplitudes(self, terms: Optional[int] = None) -> np.ndarray:
        # Per scale: the sawtooth part contributes |tanh| * |sign| * 1 <= 1 and
        # the modulation part a * sech^2 * |u| <= a * (2 + a*sqrt(|x|)), both
        # halved by the final division. The bound is loose, because sech^2 is
        # exponentially small away from a vanishing neighbourhood of the origin;
        # the measured values in the report are the informative ones.
        count = self.retained_scales(terms)
        per_scale = 0.5 * (1.0 + self.a * (2.0 + self.a * math.sqrt(SAMPLE_RANGE)))
        return np.full(count, per_scale, dtype=np.float64)

    def value(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        xp = self._prepare(x)
        dtype = self._dtype()
        total = np.zeros_like(xp)
        ax = np.asarray(self.a, dtype=dtype) * np.sqrt(np.abs(xp))
        for n in range(self.retained_scales(terms)):
            factor = dtype(2.0 ** n)
            total = total + np.tanh(dtype(self.a) * factor * xp) * np.abs(
                np.mod(xp * factor, dtype(2.0)) - ax
            ) / factor
        return self._finish(total / dtype(2.0))

    def derivative(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        xp = self._prepare(x)
        dtype = self._dtype()
        safe = np.where(np.abs(xp) < dtype(EXCLUDE_ZERO_RADIUS), dtype(EXCLUDE_ZERO_RADIUS), np.abs(xp))
        ax = dtype(self.a) * np.sqrt(safe)
        d_ax = dtype(self.a) * np.sign(xp) / (dtype(2.0) * np.sqrt(safe))
        total = np.zeros_like(xp)
        for n in range(self.retained_scales(terms)):
            factor = dtype(2.0 ** n)
            arg = dtype(self.a) * factor * xp
            tanh_arg = np.tanh(arg)
            sech2 = dtype(1.0) - tanh_arg ** 2
            u = np.mod(xp * factor, dtype(2.0)) - ax
            du = factor - d_ax
            total = total + (
                dtype(self.a) * factor * sech2 * np.abs(u) + tanh_arg * np.sign(u) * du
            ) / factor
        return self._finish(total / dtype(2.0))


class DecayingCosine(ActivationReference):
    """
    phi(x) = sum_{n=0}^{N-1} zeta * ( 0.05*tanh(pi*x)
                                      + c^n * cos(d^n*pi*x) * exp(-|x|/2) * m(x) )

    with m(x) = +1 for x < 0 and -1 otherwise. Deployed with c = 0.5, d = 2,
    zeta = 0.2666 and N = 75, accumulated in the dtype of the input.

    Scale n oscillates as cos(d^n*pi*x), that is at d^n/2 cycles per unit of x.
    Its derivative amplitude is zeta*pi*(c*d)^n, and c*d = 1 exactly, so the
    ladder is marginal: every scale contributes the same amount and the bound
    grows linearly with the number of retained scales.
    """

    name = "decaying_cosine_function_tf"
    display_name = "dec. cosine"
    family = "fractal"
    deployed_terms = DECAYING_COSINE_N_TERMS
    term_parameter = "n_terms"
    index_range_text = "n = 0 .. n_terms-1"
    accumulation_dtype = "float32"
    value_scale_text = "zeta * c^n"
    derivative_scale_text = "zeta * pi * (c*d)^n"

    def __init__(
        self,
        c: float = DECAYING_COSINE_C,
        d: int = DECAYING_COSINE_D,
        n_terms: int = DECAYING_COSINE_N_TERMS,
        zeta: float = DECAYING_COSINE_ZETA,
    ) -> None:
        self.c = float(c)
        self.d = int(d)
        self.n_terms = int(n_terms)
        self.zeta = float(zeta)
        self.deployed_terms = int(n_terms)
        self.deployed_parameter_text = (
            f"c = {self.c}, d = {self.d}, zeta = {self.zeta}, n_terms = {self.n_terms}"
        )

    def retained_scales(self, terms: Optional[int] = None) -> int:
        return int(self.n_terms if terms is None else terms)

    def scale_indices(self, terms: Optional[int] = None) -> np.ndarray:
        return np.arange(self.retained_scales(terms), dtype=np.int64)

    def scale_frequencies(self, terms: Optional[int] = None) -> np.ndarray:
        return np.power(float(self.d), self.scale_indices(terms).astype(np.float64)) / 2.0

    def derivative_scale_amplitudes(self, terms: Optional[int] = None) -> np.ndarray:
        idx = self.scale_indices(terms).astype(np.float64)
        ladder = self.zeta * np.pi * np.power(self.c * float(self.d), idx)
        decay_part = self.zeta * 0.5 * np.power(self.c, idx)
        tanh_part = self.zeta * 0.05 * np.pi
        return ladder + decay_part + tanh_part

    @staticmethod
    def _mirror(x: np.ndarray) -> np.ndarray:
        return np.where(x < 0.0, 1.0, -1.0).astype(x.dtype)

    def value(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        xp = self._prepare(x)
        dtype = self._dtype()
        decay = np.exp(-np.abs(xp) * dtype(0.5))
        mirror = self._mirror(xp)
        tanh_part = np.tanh(dtype(np.pi) * xp)
        total = np.zeros_like(xp)
        for n in range(self.retained_scales(terms)):
            factor = dtype(float(self.d) ** n)
            phase = factor * dtype(np.pi) * xp
            total = total + dtype(self.zeta) * (
                dtype(0.05) * tanh_part
                + dtype(self.c ** n) * np.cos(phase) * decay * mirror
            )
        return self._finish(total)

    def derivative(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        xp = self._prepare(x)
        dtype = self._dtype()
        decay = np.exp(-np.abs(xp) * dtype(0.5))
        mirror = self._mirror(xp)
        sign = np.sign(xp)
        sech2_pi = dtype(1.0) - np.tanh(dtype(np.pi) * xp) ** 2
        total = np.zeros_like(xp)
        for n in range(self.retained_scales(terms)):
            factor = dtype(float(self.d) ** n)
            phase = factor * dtype(np.pi) * xp
            total = total + dtype(self.zeta) * (
                dtype(0.05) * dtype(np.pi) * sech2_pi
                + dtype(self.c ** n) * decay * mirror * (
                    -factor * dtype(np.pi) * np.sin(phase)
                    - dtype(0.5) * sign * np.cos(phase)
                )
            )
        return self._finish(total)


class ModifiedWeierstrassTanh(ActivationReference):
    """
    phi(x) = [ sum_{n=0}^{N-1} (-1)^n a^n cos(b^n*pi*x) ] * exp(-0.75*|x|) + tanh(x)

    Deployed with a = 0.5, b = 1.5 and N = 100, accumulated in float64 and cast
    back to float32.

    Scale n oscillates at b^n/2 cycles per unit of x and its derivative
    amplitude is pi*(a*b)^n = pi*0.75^n, so the ladder converges and the total
    derivative stays bounded however many scales are retained. The base b = 1.5
    is not a power of two, so the scales of this activation never land exactly
    on their own lattice.
    """

    name = "modified_weierstrass_function_tanh"
    display_name = "mod. wei. tanh"
    family = "fractal"
    deployed_terms = MOD_WEIERSTRASS_TANH_N_TERMS
    term_parameter = "n_terms"
    index_range_text = "n = 0 .. n_terms-1"
    accumulation_dtype = "float64"
    value_scale_text = "a^n"
    derivative_scale_text = "pi * (a*b)^n"

    def __init__(
        self,
        a: float = MOD_WEIERSTRASS_TANH_A,
        b: float = MOD_WEIERSTRASS_TANH_B,
        n_terms: int = MOD_WEIERSTRASS_TANH_N_TERMS,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.n_terms = int(n_terms)
        self.deployed_terms = int(n_terms)
        self.deployed_parameter_text = (
            f"a = {self.a}, b = {self.b}, n_terms = {self.n_terms}"
        )

    def retained_scales(self, terms: Optional[int] = None) -> int:
        return int(self.n_terms if terms is None else terms)

    def scale_indices(self, terms: Optional[int] = None) -> np.ndarray:
        return np.arange(self.retained_scales(terms), dtype=np.int64)

    def scale_frequencies(self, terms: Optional[int] = None) -> np.ndarray:
        return np.power(self.b, self.scale_indices(terms).astype(np.float64)) / 2.0

    def derivative_scale_amplitudes(self, terms: Optional[int] = None) -> np.ndarray:
        idx = self.scale_indices(terms).astype(np.float64)
        return np.pi * np.power(self.a * self.b, idx) + 0.75 * np.power(self.a, idx)

    def value(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        xp = self._prepare(x)
        w = np.zeros_like(xp)
        for n in range(self.retained_scales(terms)):
            w = w + ((-1.0) ** n) * (self.a ** n) * np.cos((self.b ** n) * np.pi * xp)
        return self._finish(w * np.exp(-0.75 * np.abs(xp)) + np.tanh(xp))

    def derivative(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        xp = self._prepare(x)
        w = np.zeros_like(xp)
        dw = np.zeros_like(xp)
        for n in range(self.retained_scales(terms)):
            factor = self.b ** n
            phase = factor * np.pi * xp
            sign_n = (-1.0) ** n
            w = w + sign_n * (self.a ** n) * np.cos(phase)
            dw = dw + sign_n * (self.a ** n) * (-factor * np.pi * np.sin(phase))
        decay = np.exp(-0.75 * np.abs(xp))
        return self._finish(
            dw * decay + w * (-0.75 * np.sign(xp)) * decay + (1.0 - np.tanh(xp) ** 2)
        )


class WeierstrassMandelbrotXPlusSin(ActivationReference):
    """
    phi(x) = sum_k 2^(-k*gamma) * (x + sin(2*pi*lambda^k*x))

    Deployed with gamma = 0.5, lambda = 2 and num_terms = 100, accumulated in
    float64 and cast back to float32. The deployed loop is
    for k in range(1, num_terms), so 99 scales are retained, k = 1 to 99.

    Scale k oscillates at lambda^k cycles per unit of x and its derivative
    amplitude is 2*pi*2^(-k*gamma)*lambda^k = 2*pi*2^(k/2), so the ladder
    diverges: the derivative magnitude is set by the finest retained scale.
    """

    name = "weierstrass_mandelbrot_function_xpsin"
    display_name = "x+sin"
    family = "fractal"
    deployed_terms = XPSIN_NUM_TERMS
    term_parameter = "num_terms"
    index_range_text = "k = 1 .. num_terms-1"
    accumulation_dtype = "float64"
    value_scale_text = "2^(-k*gamma)"
    derivative_scale_text = "2*pi*2^(-k*gamma)*lambda^k"

    def __init__(
        self,
        gamma: float = XPSIN_GAMMA,
        lambda_val: int = XPSIN_LAMBDA,
        num_terms: int = XPSIN_NUM_TERMS,
    ) -> None:
        self.gamma = float(gamma)
        self.lambda_val = int(lambda_val)
        self.num_terms = int(num_terms)
        self.deployed_terms = int(num_terms)
        self.deployed_parameter_text = (
            f"gamma = {self.gamma}, lambda = {self.lambda_val}, "
            f"num_terms = {self.num_terms}"
        )

    def scale_indices(self, terms: Optional[int] = None) -> np.ndarray:
        count = int(self.num_terms if terms is None else terms)
        if count < 2:
            return np.zeros(0, dtype=np.int64)
        return np.arange(1, count, dtype=np.int64)

    def retained_scales(self, terms: Optional[int] = None) -> int:
        return int(self.scale_indices(terms).size)

    def scale_frequencies(self, terms: Optional[int] = None) -> np.ndarray:
        return np.power(float(self.lambda_val), self.scale_indices(terms).astype(np.float64))

    def derivative_scale_amplitudes(self, terms: Optional[int] = None) -> np.ndarray:
        idx = self.scale_indices(terms).astype(np.float64)
        if idx.size == 0:
            return np.zeros(0, dtype=np.float64)
        amplitude = np.power(2.0, -idx * self.gamma)
        frequency = np.power(float(self.lambda_val), idx)
        return amplitude * (1.0 + 2.0 * np.pi * frequency)

    def linear_gain(self, terms: Optional[int] = None) -> float:
        idx = self.scale_indices(terms).astype(np.float64)
        return float(np.sum(np.power(2.0, -idx * self.gamma))) if idx.size else 0.0

    def value(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        xp = self._prepare(x)
        total = np.zeros_like(xp)
        for k in self.scale_indices(terms):
            amplitude = 2.0 ** (-int(k) * self.gamma)
            phase = 2.0 * np.pi * float(self.lambda_val ** int(k)) * xp
            total = total + amplitude * (xp + np.sin(phase))
        return self._finish(total)

    def derivative(self, x: np.ndarray, terms: Optional[int] = None) -> np.ndarray:
        xp = self._prepare(x)
        total = np.zeros_like(xp)
        for k in self.scale_indices(terms):
            amplitude = 2.0 ** (-int(k) * self.gamma)
            frequency = float(self.lambda_val ** int(k))
            phase = 2.0 * np.pi * frequency * xp
            total = total + amplitude * (1.0 + 2.0 * np.pi * frequency * np.cos(phase))
        return self._finish(total)


def build_activation_registry() -> List[ActivationReference]:
    """
    The six activations of the classification experiments, in the order in
    which the experiment runner lists them: the two standard activations first,
    then the four fractal ones.
    """
    return [
        ReLU(),
        Tanh(),
        ModulatedBlancmange(),
        DecayingCosine(),
        ModifiedWeierstrassTanh(),
        WeierstrassMandelbrotXPlusSin(),
    ]


ACTIVATION_TF_NAMES: Dict[str, str] = {
    "modulated_blancmange_curve": "modulated_blancmange_curve",
    "decaying_cosine_function_tf": "decaying_cosine_function_tf",
    "modified_weierstrass_function_tanh": "modified_weierstrass_function_tanh",
    "weierstrass_mandelbrot_function_xpsin": "weierstrass_mandelbrot_function_xpsin",
}


# =============================================================================
# 2) RESULT RECORDS
# =============================================================================

@dataclass
class DeployedRow:
    activation: str
    display_name: str
    family: str
    deployed_parameters: str
    deployed_terms: Optional[int]
    index_range: str
    retained_scales: int
    accumulation_dtype: str
    sample_precision: str
    ladder_type: str
    derivative_scale_growth_factor: float
    derivative_amplitude_bound: float
    value_mean_abs: float
    value_max_abs: float
    derivative_mean_abs: float
    derivative_median_abs: float
    derivative_p95_abs: float
    derivative_max_abs: float
    fraction_positive_derivative: float
    nonfinite_values: int
    nonfinite_derivatives: int
    resolved_scales_float32_input: int
    unresolved_scales_float32_input: int
    unresolved_derivative_amplitude: float
    unresolved_derivative_amplitude_fraction: float


@dataclass
class TruncationRow:
    activation: str
    display_name: str
    terms: int
    retained_scales: int
    value_mean_abs: float
    value_max_deviation_from_deployed: float
    value_rms_deviation_from_deployed: float
    derivative_mean_abs: float
    derivative_median_abs: float
    derivative_p95_abs: float
    derivative_max_abs: float
    derivative_amplitude_bound: float
    fraction_positive_derivative: float
    nonfinite_values: int
    nonfinite_derivatives: int
    resolved_scales_float32_input: int
    unresolved_scales_float32_input: int
    oscillating_scales: int
    lattice_locked_scales: int
    rounding_dominated_scales: int


@dataclass
class ScaleRow:
    activation: str
    display_name: str
    scale_index: int
    frequency_cycles_per_unit: float
    derivative_scale_amplitude: float
    mean_cos: float
    mean_abs_sin: float
    phase_concentration: float
    phase_lattice_moment: float
    lattice_fraction: float
    resolved_by_float32_input: bool
    resolved_by_accumulation_dtype: bool
    phase_regime: str


@dataclass
class PrecisionRow:
    activation: str
    display_name: str
    terms: Optional[int]
    retained_scales: int
    ulp_multiple: float
    value_median_abs_change: float
    value_max_abs_change: float
    derivative_median_relative_change: float
    derivative_p95_relative_change: float
    derivative_sign_flip_fraction: float
    nonfinite_after_perturbation: int


@dataclass
class RealizabilityRow:
    activation: str
    display_name: str
    step_label: str
    median_step: float
    analytic_median_abs: float
    realized_median_abs: float
    output_quantization_floor: float
    realized_above_floor: bool
    median_ratio_analytic_to_realized: float
    p95_ratio_analytic_to_realized: float
    rank_correlation: float
    sign_agreement_fraction: float


@dataclass
class AutodiffRow:
    activation: str
    terms: Optional[int]
    n_points: int
    max_abs_value_difference: float
    max_relative_value_difference: float
    median_relative_derivative_difference: float
    max_relative_derivative_difference: float
    nonfinite_tensorflow_values: int
    nonfinite_tensorflow_gradients: int


@dataclass
class NetworkRow:
    dataset: str
    activation: str
    display_name: str
    terms: Optional[int]
    optimizer: str
    seed: int
    steps: int
    grad_norm_mean: float
    grad_norm_median: float
    grad_norm_p95: float
    grad_norm_max: float
    nonfinite_gradient_steps: int
    nonfinite_loss_steps: int
    steps_above_norm_threshold: int
    final_train_loss: float
    test_accuracy: float
    preactivation_p99_abs: float


@dataclass
class StudyResults:
    deployed_rows: List[DeployedRow] = field(default_factory=list)
    truncation_rows: List[TruncationRow] = field(default_factory=list)
    scale_rows: List[ScaleRow] = field(default_factory=list)
    precision_rows: List[PrecisionRow] = field(default_factory=list)
    realizability_rows: List[RealizabilityRow] = field(default_factory=list)
    autodiff_rows: List[AutodiffRow] = field(default_factory=list)
    network_rows: List[NetworkRow] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    sample_summary: Dict[str, Any] = field(default_factory=dict)
    curves: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# =============================================================================
# 3) SAMPLING AND SMALL STATISTICS
# =============================================================================

class PreActivationSample:
    """
    The fixed sample of pre-activations used by every stage and every
    activation.

    The sample is drawn once in float64 and then quantised to the precision in
    which the network would hold it. Quantising before evaluation matters:
    the activations receive a float32 tensor, so the part of each function that
    lives below the float32 spacing of its input is not reachable by the
    network at all, whatever dtype the activation uses internally.
    """

    def __init__(
        self,
        n_points: int = N_SAMPLE_POINTS,
        sample_range: float = SAMPLE_RANGE,
        seed: int = SAMPLE_SEED,
        exclude_zero_radius: float = EXCLUDE_ZERO_RADIUS,
    ) -> None:
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-sample_range, sample_range, size=n_points)
        mask = np.abs(raw) < exclude_zero_radius
        if np.any(mask):
            raw[mask] = exclude_zero_radius
        self.base = np.asarray(raw, dtype=np.float64)
        self.sample_range = float(sample_range)
        self.seed = int(seed)
        self.n_points = int(n_points)

    def at_precision(self, precision: str) -> np.ndarray:
        if precision == "float64":
            return self.base.copy()
        if precision == "float32":
            return self.base.astype(np.float32).astype(np.float64)
        raise ValueError(f"unknown precision: {precision}")

    def ulp(self) -> np.ndarray:
        """Spacing of the float32 grid at each sample point."""
        return np.abs(np.spacing(self.at_precision("float32").astype(np.float32))).astype(
            np.float64
        )

    def median_abs(self) -> float:
        return float(np.median(np.abs(self.base)))

    def summary(self) -> Dict[str, Any]:
        return {
            "n_points": self.n_points,
            "sample_range": self.sample_range,
            "seed": self.seed,
            "mean_abs": float(np.mean(np.abs(self.base))),
            "median_abs": self.median_abs(),
            "max_abs": float(np.max(np.abs(self.base))),
            "float32_spacing_median": float(np.median(self.ulp())),
            "float32_mantissa_bits": MANTISSA_BITS_FLOAT32,
        }


def primary_realizability_step(rows: Sequence["RealizabilityRow"]) -> Optional[str]:
    """
    The step at which the realized slope is above the float32 output
    quantization for every activation, which is the one the comparison can be
    read from. Falls back to the coarsest step if no step qualifies.
    """
    labels: List[str] = []
    for row in rows:
        if row.step_label not in labels:
            labels.append(row.step_label)
    for label in labels:
        group = [row for row in rows if row.step_label == label]
        if group and all(row.realized_above_floor for row in group):
            return label
    return labels[-1] if labels else None


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman rank correlation. Ties are broken by position, which is sufficient
    here because the inputs are continuous measurements.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    a, b = a[finite], b[finite]

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        out = np.empty(values.size, dtype=np.float64)
        out[order] = np.arange(values.size, dtype=np.float64)
        return out

    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    if denominator <= 0.0:
        return float("nan")
    return float(float((ra * rb).sum()) / denominator)


# =============================================================================
# 4) STAGE 1: DEPLOYED-SETTING COMPARISON
# =============================================================================

class DeployedComparison:
    """
    Value and derivative statistics of all six activations at their deployed
    settings, for float32 and for float64 pre-activations. This is the table
    the appendix compares the activations with.
    """

    def __init__(
        self,
        activations: Sequence[ActivationReference],
        sample: PreActivationSample,
        precisions: Sequence[str] = tuple(SAMPLE_PRECISIONS),
    ) -> None:
        self.activations = list(activations)
        self.sample = sample
        self.precisions = list(precisions)

    def run(self) -> List[DeployedRow]:
        rows: List[DeployedRow] = []
        median_abs = self.sample.median_abs()

        for activation in self.activations:
            for precision in self.precisions:
                x = self.sample.at_precision(precision)
                value = activation.value(x)
                derivative = activation.derivative(x)
                retained = activation.retained_scales()
                resolved = activation.resolved_scale_count(median_abs, EPS_FLOAT32)

                rows.append(
                    DeployedRow(
                        activation=activation.name,
                        display_name=activation.display_name,
                        family=activation.family,
                        deployed_parameters=activation.deployed_parameter_text,
                        deployed_terms=activation.deployed_terms,
                        index_range=activation.index_range_text or "-",
                        retained_scales=int(retained),
                        accumulation_dtype=activation.accumulation_dtype,
                        sample_precision=precision,
                        ladder_type=activation.ladder_type(),
                        derivative_scale_growth_factor=activation.derivative_scale_growth_factor(),
                        derivative_amplitude_bound=activation.derivative_amplitude_bound(),
                        value_mean_abs=float(np.mean(np.abs(value))),
                        value_max_abs=float(np.max(np.abs(value))),
                        derivative_mean_abs=float(np.mean(np.abs(derivative))),
                        derivative_median_abs=float(np.median(np.abs(derivative))),
                        derivative_p95_abs=float(np.percentile(np.abs(derivative), 95.0)),
                        derivative_max_abs=float(np.max(np.abs(derivative))),
                        fraction_positive_derivative=float(
                            np.count_nonzero(derivative > 0.0) / derivative.size
                        ),
                        nonfinite_values=int(np.count_nonzero(~np.isfinite(value))),
                        nonfinite_derivatives=int(
                            np.count_nonzero(~np.isfinite(derivative))
                        ),
                        resolved_scales_float32_input=int(resolved),
                        unresolved_scales_float32_input=int(max(0, retained - resolved)),
                        unresolved_derivative_amplitude=(
                            activation.unresolved_derivative_amplitude(
                                median_abs, EPS_FLOAT32
                            )
                        ),
                        unresolved_derivative_amplitude_fraction=(
                            activation.unresolved_derivative_fraction(
                                median_abs, EPS_FLOAT32
                            )
                        ),
                    )
                )

        return rows

    def curves(self) -> Dict[str, Dict[str, Any]]:
        """Dense curves of every activation for the overview figures."""
        out: Dict[str, Dict[str, Any]] = {}
        for activation in self.activations:
            entry: Dict[str, Any] = {"display_name": activation.display_name}
            for label, width in (
                ("wide", PLOT_WINDOW_WIDE),
                ("narrow", PLOT_WINDOW_NARROW),
            ):
                grid = np.linspace(-width, width, PLOT_GRID_N)
                grid = grid.astype(np.float32).astype(np.float64)
                entry[label] = {
                    "x": grid,
                    "value": activation.value(grid),
                    "derivative": activation.derivative(grid),
                }
            out[activation.name] = entry
        return out


# =============================================================================
# 5) STAGE 2: TRUNCATION SWEEP
# =============================================================================

class TruncationSweep:
    """
    Each series activation over its own truncation grid, measured on float32
    pre-activations, which is the situation inside the networks. The deviation
    of the value is taken against the deployed truncation of the same
    activation, so every activation is compared against what the study actually
    ran.
    """

    def __init__(
        self,
        activations: Sequence[ActivationReference],
        sample: PreActivationSample,
    ) -> None:
        self.activations = [a for a in activations if a.is_series()]
        self.sample = sample

    def run(self, scale_rows: Sequence[ScaleRow]) -> List[TruncationRow]:
        rows: List[TruncationRow] = []
        x = self.sample.at_precision("float32")
        median_abs = self.sample.median_abs()

        regimes: Dict[Tuple[str, int], str] = {
            (row.activation, row.scale_index): row.phase_regime for row in scale_rows
        }

        for activation in self.activations:
            deployed_value = activation.value(x)

            for terms in activation.truncation_grid():
                value = activation.value(x, terms)
                derivative = activation.derivative(x, terms)
                deviation = value - deployed_value
                retained = activation.retained_scales(terms)
                indices = activation.scale_indices(terms)

                counts = {"oscillating": 0, "lattice_locked": 0, "rounding_dominated": 0}
                for index in indices:
                    regime = regimes.get((activation.name, int(index)))
                    if regime in counts:
                        counts[regime] += 1

                rows.append(
                    TruncationRow(
                        activation=activation.name,
                        display_name=activation.display_name,
                        terms=int(terms),
                        retained_scales=int(retained),
                        value_mean_abs=float(np.mean(np.abs(value))),
                        value_max_deviation_from_deployed=float(np.max(np.abs(deviation))),
                        value_rms_deviation_from_deployed=float(
                            np.sqrt(np.mean(deviation ** 2))
                        ),
                        derivative_mean_abs=float(np.mean(np.abs(derivative))),
                        derivative_median_abs=float(np.median(np.abs(derivative))),
                        derivative_p95_abs=float(np.percentile(np.abs(derivative), 95.0)),
                        derivative_max_abs=float(np.max(np.abs(derivative))),
                        derivative_amplitude_bound=activation.derivative_amplitude_bound(terms),
                        fraction_positive_derivative=float(
                            np.count_nonzero(derivative > 0.0) / derivative.size
                        ),
                        nonfinite_values=int(np.count_nonzero(~np.isfinite(value))),
                        nonfinite_derivatives=int(
                            np.count_nonzero(~np.isfinite(derivative))
                        ),
                        resolved_scales_float32_input=int(
                            activation.resolved_scale_count(median_abs, EPS_FLOAT32, terms)
                        ),
                        unresolved_scales_float32_input=int(
                            max(
                                0,
                                retained
                                - activation.resolved_scale_count(
                                    median_abs, EPS_FLOAT32, terms
                                ),
                            )
                        ),
                        oscillating_scales=counts["oscillating"],
                        lattice_locked_scales=counts["lattice_locked"],
                        rounding_dominated_scales=counts["rounding_dominated"],
                    )
                )

        return rows


# =============================================================================
# 6) STAGE 3: SCALE STRUCTURE AND PHASE REGIMES
# =============================================================================

class ScaleStructureProbe:
    """
    Per-scale measurement of what the oscillating part of each term actually
    does on float32 pre-activations, for every series activation.

    For scale m of frequency f the probe forms the phase in cycles,
    p = f*x, in the dtype the deployed implementation uses, and records

        phase concentration  |mean of exp(2*pi*i*p)|, zero for a phase spread
                             uniformly over the circle and one for a phase that
                             does not move,
        mean cos, mean |sin| the same information in the form in which the
                             sinusoidal activations use it; a uniform phase
                             gives mean cos = 0 and mean |sin| = 2/pi = 0.6366,
        lattice fraction     the fraction of sample points at which p is an
                             exact integer, which is what happens when the
                             frequency is a power of two and the input is a
                             float32 number.

    Three regimes follow, and which of them a scale falls into is decided from
    these measurements and the two thresholds in section 0:

        oscillating         phase spread over the circle, the scale behaves as
                            intended,
        lattice locked      the phase sits on integers, so the oscillation has
                            collapsed to a constant,
        rounding dominated  the phase is spread again, but the product that
                            produces it can no longer be formed accurately, so
                            it is a function of the rounding and not of the
                            pre-activation.
    """

    def __init__(
        self,
        activations: Sequence[ActivationReference],
        sample: PreActivationSample,
    ) -> None:
        self.activations = [a for a in activations if a.is_series()]
        self.sample = sample

    def run(self) -> List[ScaleRow]:
        rows: List[ScaleRow] = []
        x = self.sample.at_precision("float32")
        median_abs = self.sample.median_abs()

        for activation in self.activations:
            indices = activation.scale_indices()
            frequencies = activation.scale_frequencies()
            amplitudes = activation.derivative_scale_amplitudes()
            eps_internal = (
                EPS_FLOAT64 if activation.accumulation_dtype == "float64" else EPS_FLOAT32
            )

            for position, index in enumerate(indices):
                frequency = float(frequencies[position])
                phase = activation.scale_phase(x, frequency).astype(np.float64)
                angle = 2.0 * np.pi * phase
                cos_values = np.cos(angle)
                sin_values = np.sin(angle)

                concentration = float(
                    math.hypot(float(np.mean(cos_values)), float(np.mean(sin_values)))
                )
                lattice_moment = concentration
                for order in PHASE_MOMENT_ORDERS:
                    if order == 1:
                        continue
                    harmonic = float(order) * angle
                    lattice_moment = max(
                        lattice_moment,
                        float(
                            math.hypot(
                                float(np.mean(np.cos(harmonic))),
                                float(np.mean(np.sin(harmonic))),
                            )
                        ),
                    )
                lattice = float(
                    np.count_nonzero(phase == np.round(phase)) / phase.size
                )

                # A phase that has collapsed onto a coarse lattice is locked,
                # whatever produced the collapse. A phase that is spread while
                # the input sits on the lattice of the scale is spread by the
                # arithmetic and not by the input, which is the rounding
                # dominated case.
                if lattice_moment >= PHASE_CONCENTRATION_THRESHOLD:
                    regime = "lattice_locked"
                elif lattice >= LATTICE_FRACTION_THRESHOLD:
                    regime = "rounding_dominated"
                else:
                    regime = "oscillating"

                rows.append(
                    ScaleRow(
                        activation=activation.name,
                        display_name=activation.display_name,
                        scale_index=int(index),
                        frequency_cycles_per_unit=frequency,
                        derivative_scale_amplitude=float(amplitudes[position]),
                        mean_cos=float(np.mean(cos_values)),
                        mean_abs_sin=float(np.mean(np.abs(sin_values))),
                        phase_concentration=concentration,
                        phase_lattice_moment=lattice_moment,
                        lattice_fraction=lattice,
                        resolved_by_float32_input=bool(
                            2.0 * math.pi * frequency * EPS_FLOAT32 * median_abs < 1.0
                        ),
                        resolved_by_accumulation_dtype=bool(
                            2.0 * math.pi * frequency * eps_internal * median_abs < 1.0
                        ),
                        phase_regime=regime,
                    )
                )

        return rows

    @staticmethod
    def regime_summary(rows: Sequence[ScaleRow]) -> Dict[str, Dict[str, Any]]:
        """Per activation: scale counts, index ranges and amplitude per regime."""
        summary: Dict[str, Dict[str, Any]] = {}
        for activation in sorted({row.activation for row in rows}):
            selected = [row for row in rows if row.activation == activation]
            total = sum(row.derivative_scale_amplitude for row in selected)
            entry: Dict[str, Any] = {}
            for regime in ("oscillating", "lattice_locked", "rounding_dominated"):
                group = [row for row in selected if row.phase_regime == regime]
                amplitude = float(sum(row.derivative_scale_amplitude for row in group))
                entry[regime] = {
                    "n_scales": len(group),
                    "first_scale_index": min(
                        (row.scale_index for row in group), default=None
                    ),
                    "last_scale_index": max(
                        (row.scale_index for row in group), default=None
                    ),
                    "derivative_amplitude": amplitude,
                    "derivative_amplitude_share": (
                        float(amplitude / total) if total > 0 else float("nan")
                    ),
                }
            summary[activation] = entry
        return summary


# =============================================================================
# 7) STAGE 4: INPUT PRECISION PROBE
# =============================================================================

class InputPrecisionProbe:
    """
    Sensitivity of every activation to a shift of the pre-activation that is
    smaller than the accuracy with which a network can represent it.

    Each sample point is moved by `ulp_multiple` units in the last place of its
    float32 representation. A value that comes from a convergent series barely
    moves. A derivative whose series does not converge moves by order one as
    soon as the retained scales exceed what the float32 pre-activation
    resolves, and the fraction of points at which it changes sign approaches
    one half, which is the signature of a phase that the input does not
    determine.

    Series activations are probed over their full truncation grid, the two
    standard activations at their single setting, so that the comparison covers
    the same points.
    """

    def __init__(
        self,
        activations: Sequence[ActivationReference],
        sample: PreActivationSample,
        ulp_multiples: Sequence[float] = tuple(ULP_MULTIPLES),
    ) -> None:
        self.activations = list(activations)
        self.sample = sample
        self.ulp_multiples = [float(m) for m in ulp_multiples]

    def run(self) -> List[PrecisionRow]:
        rows: List[PrecisionRow] = []
        x = self.sample.at_precision("float32")
        spacing = self.sample.ulp()

        for activation in self.activations:
            for terms in activation.truncation_grid():
                value = activation.value(x, terms)
                derivative = activation.derivative(x, terms)
                scale = np.maximum(np.abs(derivative), 1e-12)

                for multiple in self.ulp_multiples:
                    shifted = (x + multiple * spacing).astype(np.float32).astype(np.float64)
                    value_shifted = activation.value(shifted, terms)
                    derivative_shifted = activation.derivative(shifted, terms)

                    relative = np.abs(derivative_shifted - derivative) / scale
                    sign_flips = int(
                        np.count_nonzero(np.sign(derivative_shifted) != np.sign(derivative))
                    )

                    rows.append(
                        PrecisionRow(
                            activation=activation.name,
                            display_name=activation.display_name,
                            terms=None if terms is None else int(terms),
                            retained_scales=int(activation.retained_scales(terms)),
                            ulp_multiple=float(multiple),
                            value_median_abs_change=float(
                                np.median(np.abs(value_shifted - value))
                            ),
                            value_max_abs_change=float(
                                np.max(np.abs(value_shifted - value))
                            ),
                            derivative_median_relative_change=float(np.median(relative)),
                            derivative_p95_relative_change=float(
                                np.percentile(relative, 95.0)
                            ),
                            derivative_sign_flip_fraction=float(
                                sign_flips / max(derivative.size, 1)
                            ),
                            nonfinite_after_perturbation=int(
                                np.count_nonzero(~np.isfinite(value_shifted))
                                + np.count_nonzero(~np.isfinite(derivative_shifted))
                            ),
                        )
                    )

        return rows


# =============================================================================
# 8) STAGE 5: GRADIENT REALIZABILITY
# =============================================================================

class GradientRealizability:
    """
    Comparison of the analytic derivative, which is what automatic
    differentiation returns and what the optimizer consumes, against the slope
    the activation actually realizes between neighbouring float32 inputs.

    For a function a network can work with, the two agree. Where they do not,
    the gradient describes a function other than the one the network computes.
    Three step sizes are used, because the comparison depends on the resolution
    at which it is asked: one unit in the last place of float32, which is the
    finest step a network can address, and two coarser relative steps.

    Both the realized slope and the analytic derivative are evaluated through
    the float32 output cast, so the realized slope has a resolution floor of
    about eps*|phi| divided by the step; values of the same order as that floor
    are noise and the report says so.
    """

    def __init__(
        self,
        activations: Sequence[ActivationReference],
        sample: PreActivationSample,
        steps: Sequence[Optional[float]] = tuple(REALIZABILITY_STEPS),
    ) -> None:
        self.activations = list(activations)
        self.sample = sample
        self.steps = list(steps)

    def run(self) -> List[RealizabilityRow]:
        rows: List[RealizabilityRow] = []
        x = self.sample.at_precision("float32")
        x32 = x.astype(np.float32)

        for activation in self.activations:
            value = activation.value(x)
            analytic = activation.derivative(x)

            for step in self.steps:
                if step is None:
                    shifted32 = np.nextafter(x32, np.float32(np.inf))
                    label = "1 ULP(float32)"
                else:
                    offset = np.asarray(
                        float(step) * np.maximum(np.abs(x), 1.0), dtype=np.float64
                    )
                    shifted32 = (x + offset).astype(np.float32)
                    label = f"relative {step:g}"

                shifted = shifted32.astype(np.float64)
                actual_step = shifted - x
                usable = actual_step != 0.0
                if not np.any(usable):
                    continue

                value_shifted = activation.value(shifted)
                realized = np.full_like(analytic, np.nan)
                realized[usable] = (
                    value_shifted[usable] - value[usable]
                ) / actual_step[usable]

                denominator = np.maximum(np.abs(realized), 1e-30)
                ratio = np.abs(analytic) / denominator
                finite = np.isfinite(ratio) & usable

                # The value passes through the float32 output cast, so the
                # difference quotient cannot resolve a slope smaller than the
                # quantization of the output divided by the step. Rows whose
                # realized slope sits at that floor say nothing about the
                # activation and are marked.
                floor = float(
                    np.median(EPS_FLOAT32 * np.abs(value[finite]))
                    / max(float(np.median(actual_step[usable])), 1e-300)
                )
                realized_median = float(np.median(np.abs(realized[finite])))

                rows.append(
                    RealizabilityRow(
                        activation=activation.name,
                        display_name=activation.display_name,
                        step_label=label,
                        median_step=float(np.median(actual_step[usable])),
                        analytic_median_abs=float(np.median(np.abs(analytic[finite]))),
                        realized_median_abs=realized_median,
                        output_quantization_floor=floor,
                        realized_above_floor=bool(realized_median > 2.0 * floor),
                        median_ratio_analytic_to_realized=float(np.median(ratio[finite])),
                        p95_ratio_analytic_to_realized=float(
                            np.percentile(ratio[finite], 95.0)
                        ),
                        rank_correlation=rank_correlation(
                            analytic[finite], realized[finite]
                        ),
                        sign_agreement_fraction=float(
                            np.count_nonzero(
                                np.sign(analytic[finite]) == np.sign(realized[finite])
                            )
                            / max(int(np.count_nonzero(finite)), 1)
                        ),
                    )
                )

        return rows


# =============================================================================
# 9) AUTODIFF CROSS-CHECK AGAINST THE DEPLOYED IMPLEMENTATIONS
# =============================================================================

class AutodiffCrossCheck:
    """
    Compares the numpy references of section 1 against the deployed TensorFlow
    activations and their automatic derivatives. This is what justifies reading
    the numpy numbers as statements about the experiments.

    For the two activations that accumulate in float32, agreement at scales
    beyond the resolvable range is not expected to be exact: there the result
    depends on how the library reduces a very large argument, so a disagreement
    between numpy and TensorFlow at those scales is itself part of the finding
    and is reported rather than hidden.
    """

    def __init__(
        self,
        activations: Sequence[ActivationReference],
        sample: PreActivationSample,
        truncations: Sequence[int] = tuple(AUTODIFF_CROSSCHECK_TRUNCATIONS),
        n_points: int = AUTODIFF_CROSSCHECK_POINTS,
    ) -> None:
        self.activations = list(activations)
        self.sample = sample
        self.truncations = [int(t) for t in truncations]
        self.n_points = int(n_points)

    def available(self) -> bool:
        return bool(
            AUTODIFF_CROSSCHECK_ENABLED
            and TENSORFLOW_AVAILABLE
            and ACTIVATION_MODULE_AVAILABLE
        )

    @staticmethod
    def _call_tensorflow(
        activation: ActivationReference,
        tensor: Any,
        terms: Optional[int],
    ) -> Any:
        function = getattr(faf, ACTIVATION_TF_NAMES[activation.name])
        if terms is None:
            return function(tensor)
        return function(tensor, **{activation.term_parameter: int(terms)})

    def run(self) -> List[AutodiffRow]:
        if not self.available():
            return []

        rows: List[AutodiffRow] = []
        x32 = self.sample.at_precision("float32")[: self.n_points].astype(np.float32)
        x64 = x32.astype(np.float64)

        for activation in self.activations:
            if activation.name not in ACTIVATION_TF_NAMES:
                continue
            settings: List[Optional[int]] = [None]
            if activation.is_series():
                settings = [t for t in self.truncations if t <= int(activation.deployed_terms)]
                settings.append(int(activation.deployed_terms))

            for terms in settings:
                tensor = tf.Variable(x32, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    out = self._call_tensorflow(activation, tensor, terms)
                    total = tf.reduce_sum(out)
                grad = tape.gradient(total, tensor)

                tf_value = np.asarray(out.numpy(), dtype=np.float64)
                tf_grad = np.asarray(grad.numpy(), dtype=np.float64)
                ref_value = activation.value(x64, terms)
                ref_grad = activation.derivative(x64, terms)

                value_scale = np.maximum(np.abs(ref_value), 1e-12)
                grad_scale = np.maximum(np.abs(ref_grad), 1e-12)

                rows.append(
                    AutodiffRow(
                        activation=activation.name,
                        terms=None if terms is None else int(terms),
                        n_points=int(x32.size),
                        max_abs_value_difference=float(
                            np.max(np.abs(tf_value - ref_value))
                        ),
                        max_relative_value_difference=float(
                            np.max(np.abs(tf_value - ref_value) / value_scale)
                        ),
                        median_relative_derivative_difference=float(
                            np.median(np.abs(tf_grad - ref_grad) / grad_scale)
                        ),
                        max_relative_derivative_difference=float(
                            np.max(np.abs(tf_grad - ref_grad) / grad_scale)
                        ),
                        nonfinite_tensorflow_values=int(
                            np.count_nonzero(~np.isfinite(tf_value))
                        ),
                        nonfinite_tensorflow_gradients=int(
                            np.count_nonzero(~np.isfinite(tf_grad))
                        ),
                    )
                )

        return rows


# =============================================================================
# 10) STAGE 6: COMPACT NETWORK PROBE
# =============================================================================

class NetworkGradientProbe:
    """
    Compact end-to-end check for every activation. One dataset, the
    architecture and the hyperparameters of the study, and a plain training
    loop that records what the optimizer actually receives: the global gradient
    norm at every step, the number of steps at which the loss or the gradient is
    not finite, the reached test accuracy and the range of pre-activations the
    first layer produced.

    The loop is explicit rather than model.fit, because the per-step gradient
    norm is the quantity of interest and a callback cannot see it. The standard
    activations are included as the control: their gradient norms are what a
    working configuration looks like on this dataset.
    """

    def __init__(
        self,
        activations: Sequence[ActivationReference],
        dataset: str = NETWORK_PROBE_DATASET,
        truncations: Sequence[int] = tuple(NETWORK_PROBE_TRUNCATIONS),
        optimizers: Sequence[str] = tuple(NETWORK_PROBE_OPTIMIZERS),
        runs: int = NETWORK_PROBE_RUNS,
        seed_val: int = NETWORK_PROBE_SEED_VAL,
        learning_rate: float = NETWORK_PROBE_LEARNING_RATE,
    ) -> None:
        self.activations = list(activations)
        self.dataset = dataset
        self.truncations = [int(t) for t in truncations]
        self.optimizers = list(optimizers)
        self.runs = int(runs)
        self.seed_val = int(seed_val)
        self.learning_rate = float(learning_rate)

    def available(self) -> bool:
        return bool(
            NETWORK_PROBE_ENABLED
            and TENSORFLOW_AVAILABLE
            and ACTIVATION_MODULE_AVAILABLE
        )

    # ---- data ---------------------------------------------------------------

    @staticmethod
    def _load(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import LabelEncoder

        data = fetch_openml(name=dataset, version=1, as_frame=True, parser="auto")
        X = np.asarray(data.data, dtype=np.float32)
        y = np.asarray(LabelEncoder().fit_transform(data.target), dtype=np.int32)
        return X, y

    @staticmethod
    def _split_scale(
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, shuffle=True
        )
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        return X_train, X_test, np.asarray(y_train, np.int32), np.asarray(y_test, np.int32)

    # ---- model --------------------------------------------------------------

    def _build_model(
        self,
        n_features: int,
        n_outputs: int,
        neurons: int,
        hidden_layers: int,
        activation: Callable[[Any], Any],
    ) -> Any:
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
                    neurons * 2,
                    activation=activation,
                    kernel_initializer=initializer,
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

    def _build_optimizer(self, name: str) -> Any:
        if name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        if name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        if name == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        raise ValueError(f"unknown probe optimizer: {name}")

    def _keras_activation(
        self,
        activation: ActivationReference,
        terms: Optional[int],
    ) -> Callable[[Any], Any]:
        if activation.name == "relu":
            return tf.nn.relu
        if activation.name == "tanh":
            return tf.nn.tanh
        function = getattr(faf, ACTIVATION_TF_NAMES[activation.name])
        if terms is None:
            return function
        parameter = activation.term_parameter

        def wrapped(z, _function=function, _parameter=parameter, _terms=int(terms)):
            return _function(z, **{_parameter: _terms})

        return wrapped

    def _settings_for(self, activation: ActivationReference) -> List[Optional[int]]:
        if not activation.is_series():
            return [None]
        deployed = int(activation.deployed_terms)
        selected = sorted({t for t in self.truncations if t <= deployed} | {deployed})
        return [int(t) for t in selected]

    # ---- loop ---------------------------------------------------------------

    def run(self) -> List[NetworkRow]:
        if not self.available():
            return []

        hyper = NETWORK_PROBE_HYPERPARAMS[self.dataset]
        X, y = self._load(self.dataset)
        n_classes = int(len(np.unique(y)))
        n_outputs = n_classes if hyper["task"] == "multi-class" else 1
        rows: List[NetworkRow] = []

        for activation in self.activations:
            for terms in self._settings_for(activation):
                keras_activation = self._keras_activation(activation, terms)

                for optimizer_name in self.optimizers:
                    for run_index in range(self.runs):
                        # Same run-number protocol as B_01: consecutive seeds
                        # starting at SEED_VAL, re-seeded before every run.
                        seed = self.seed_val + run_index
                        tf.keras.backend.clear_session()
                        os.environ["PYTHONHASHSEED"] = str(seed)
                        random.seed(seed)
                        np.random.seed(seed)
                        tf.random.set_seed(seed)

                        X_train, X_test, y_train, y_test = self._split_scale(X, y, seed)
                        model = self._build_model(
                            X_train.shape[1],
                            n_outputs,
                            int(hyper["n"]),
                            int(hyper["hidden_layers"]),
                            keras_activation,
                        )
                        optimizer = self._build_optimizer(optimizer_name)

                        if hyper["task"] == "multi-class":
                            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
                            y_train_t = y_train.astype(np.int32)
                        else:
                            loss_fn = tf.keras.losses.BinaryCrossentropy()
                            y_train_t = y_train.astype(np.float32).reshape(-1, 1)

                        dataset_tf = (
                            tf.data.Dataset.from_tensor_slices((X_train, y_train_t))
                            .shuffle(len(X_train), seed=seed)
                            .batch(int(hyper["batch_size"]))
                        )

                        # One compiled step per run. The fractal activations
                        # unroll into as many operations as they have scales,
                        # so tracing once instead of executing eagerly at every
                        # step is what keeps this probe compact in wall-clock
                        # time as well.
                        @tf.function
                        def train_step(xb, yb):
                            with tf.GradientTape() as tape:
                                pred = model(xb, training=True)
                                loss = loss_fn(yb, pred)
                            grads = tape.gradient(loss, model.trainable_variables)
                            pairs = [
                                (g, v)
                                for g, v in zip(grads, model.trainable_variables)
                                if g is not None
                            ]
                            present = [g for g, _ in pairs]
                            finite = tf.reduce_all(
                                tf.stack(
                                    [tf.reduce_all(tf.math.is_finite(g)) for g in present]
                                )
                            )
                            norm = tf.linalg.global_norm(present)
                            optimizer.apply_gradients(pairs)
                            return loss, norm, finite

                        norms: List[float] = []
                        nonfinite_gradient_steps = 0
                        nonfinite_loss_steps = 0
                        last_loss = float("nan")

                        for _ in range(int(hyper["epochs"])):
                            for xb, yb in dataset_tf:
                                loss, norm, finite = train_step(xb, yb)
                                loss_value = float(loss.numpy())
                                if not math.isfinite(loss_value):
                                    nonfinite_loss_steps += 1
                                if not bool(finite.numpy()):
                                    nonfinite_gradient_steps += 1
                                    norms.append(float("inf"))
                                else:
                                    norms.append(float(norm.numpy()))
                                last_loss = loss_value

                        norm_array = np.asarray(norms, dtype=np.float64)
                        finite_norms = norm_array[np.isfinite(norm_array)]

                        prediction = model(X_test, training=False).numpy()
                        if hyper["task"] == "multi-class":
                            accuracy = float(
                                np.mean(np.argmax(prediction, axis=1) == y_test)
                            )
                        else:
                            accuracy = float(
                                np.mean(
                                    (prediction.reshape(-1) >= 0.5).astype(np.int32)
                                    == y_test
                                )
                            )

                        weights = model.layers[0].get_weights()
                        pre_activation = X_test @ weights[0] + weights[1]

                        rows.append(
                            NetworkRow(
                                dataset=self.dataset,
                                activation=activation.name,
                                display_name=activation.display_name,
                                terms=None if terms is None else int(terms),
                                optimizer=optimizer_name,
                                seed=int(seed),
                                steps=int(norm_array.size),
                                grad_norm_mean=float(
                                    np.mean(finite_norms) if finite_norms.size else np.nan
                                ),
                                grad_norm_median=float(
                                    np.median(finite_norms) if finite_norms.size else np.nan
                                ),
                                grad_norm_p95=float(
                                    np.percentile(finite_norms, 95.0)
                                    if finite_norms.size
                                    else np.nan
                                ),
                                grad_norm_max=float(
                                    np.max(finite_norms) if finite_norms.size else np.nan
                                ),
                                nonfinite_gradient_steps=int(nonfinite_gradient_steps),
                                nonfinite_loss_steps=int(nonfinite_loss_steps),
                                steps_above_norm_threshold=int(
                                    np.count_nonzero(
                                        finite_norms
                                        > NETWORK_PROBE_GRAD_NORM_REPORT_THRESHOLD
                                    )
                                ),
                                final_train_loss=float(last_loss),
                                test_accuracy=accuracy,
                                preactivation_p99_abs=float(
                                    np.percentile(np.abs(pre_activation), 99.0)
                                ),
                            )
                        )

                        print(
                            f"    [probe] {activation.display_name:>15s} "
                            f"terms={str(terms):>5s} {optimizer_name:>5s} "
                            f"seed={seed} acc={accuracy:.4f} "
                            f"mean|g|={rows[-1].grad_norm_mean:.4e} "
                            f"nonfinite={nonfinite_gradient_steps}"
                        )

        return rows


# =============================================================================
# 11) PLOTTING
# =============================================================================

class ComparisonPlotter:
    """
    Figures for the supplementary appendix.

    One colour and one marker per activation, kept across every figure, the
    same ordering everywhere, logarithmic axes wherever the quantities span
    orders of magnitude, and the two standard activations drawn as reference
    lines in the panels where they have no truncation axis of their own.
    """

    def __init__(
        self,
        activations: Sequence[ActivationReference],
        results: StudyResults,
    ) -> None:
        self.activations = list(activations)
        self.results = results
        self.order = [a.name for a in self.activations]

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _save(fig: plt.Figure, out_base: Path) -> None:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_base.with_suffix(".png"), dpi=DPI_PNG, bbox_inches="tight")
        if SAVE_EPS:
            fig.savefig(out_base.with_suffix(".eps"), format="eps", bbox_inches="tight")
        plt.close(fig)

    def _color(self, name: str) -> str:
        return ACTIVATION_COLORS.get(name, COLOR_PALETTE_A[8])

    def _marker(self, name: str) -> str:
        return ACTIVATION_MARKERS.get(name, "o")

    def _linestyle(self, name: str) -> Any:
        return ACTIVATION_LINESTYLES.get(name, "solid")

    def _display(self, name: str) -> str:
        for activation in self.activations:
            if activation.name == name:
                return activation.display_name
        return name

    def _series_activations(self) -> List[ActivationReference]:
        return [a for a in self.activations if a.is_series()]

    # ---- figures ------------------------------------------------------------

    def plot_function_overview(self) -> None:
        """Value and derivative of every activation, one panel per activation."""
        for quantity, filename, title, logscale in (
            ("value", "overview_values", "Activation values", False),
            ("derivative", "overview_derivatives", "Activation derivatives", True),
        ):
            fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.6), sharex=True)
            for position, activation in enumerate(self.activations):
                ax = axes[position // 3][position % 3]
                entry = self.results.curves.get(activation.name)
                if entry is None:
                    continue
                data = entry["wide"]
                series = data[quantity]
                if logscale:
                    series = np.abs(series)
                ax.plot(
                    data["x"],
                    series,
                    color=self._color(activation.name),
                    linewidth=0.9,
                )
                if logscale:
                    ax.set_yscale("log")
                ax.set_title(activation.display_name)
                ax.grid(True, alpha=0.25, which="both")
                if position // 3 == 1:
                    ax.set_xlabel("pre-activation x")
                if position % 3 == 0:
                    ax.set_ylabel(
                        r"$|\varphi'(x)|$" if logscale else r"$\varphi(x)$"
                    )
            fig.suptitle(f"{title} over the deployed pre-activation range")
            self._save(fig, PLOT_DIR / filename)

    def plot_truncation_comparison(self) -> None:
        """Value convergence and derivative growth against truncation length."""
        rows = self.results.truncation_rows
        if not rows:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))

        for activation in self._series_activations():
            selected = sorted(
                [r for r in rows if r.activation == activation.name],
                key=lambda r: r.terms,
            )
            deployed = int(activation.deployed_terms)
            # The deployed truncation is its own reference, and a deviation of
            # exactly zero means the two agree to the last bit of the float32
            # value; neither can be drawn on a logarithmic axis.
            convergence = [
                r
                for r in selected
                if r.terms != deployed and r.value_max_deviation_from_deployed > 0.0
            ]
            if convergence:
                axes[0].plot(
                    [r.terms for r in convergence],
                    [r.value_max_deviation_from_deployed for r in convergence],
                    marker=self._marker(activation.name),
                    markersize=5.0,
                    color=self._color(activation.name),
                linestyle=self._linestyle(activation.name),
                    label=activation.display_name,
                )
            axes[1].plot(
                [r.terms for r in selected],
                [r.derivative_mean_abs for r in selected],
                marker=self._marker(activation.name),
                markersize=5.0,
                color=self._color(activation.name),
                linestyle=self._linestyle(activation.name),
                label=activation.display_name,
            )

        for activation in self.activations:
            if activation.is_series():
                continue
            reference = [
                r
                for r in self.results.deployed_rows
                if r.activation == activation.name and r.sample_precision == "float32"
            ]
            if not reference:
                continue
            axes[1].axhline(
                reference[0].derivative_mean_abs,
                color=self._color(activation.name),
                linestyle="--",
                linewidth=1.2,
                label=f"{activation.display_name} (no truncation)",
            )

        axes[0].set_yscale("log")
        axes[0].set_xlabel("retained terms")
        axes[0].set_ylabel(r"$\max|\varphi_N - \varphi_{\mathrm{deployed}}|$")
        axes[0].set_title(
            "Value convergence towards the deployed truncation\n"
            "(points at zero deviation are omitted on the logarithmic axis)"
        )
        axes[0].grid(True, alpha=0.25, which="both")
        axes[0].legend(frameon=False)

        axes[1].set_yscale("log")
        axes[1].set_xlabel("retained terms")
        axes[1].set_ylabel(r"mean $|\varphi'|$")
        axes[1].set_title("Derivative magnitude against truncation length")
        axes[1].grid(True, alpha=0.25, which="both")
        axes[1].legend(frameon=False, ncol=2, fontsize=8 * FONT_SCALE)

        fig.suptitle("Truncation sensitivity of the four fractal activations")
        self._save(fig, PLOT_DIR / "truncation_comparison")

    def plot_scale_structure(self) -> None:
        """Derivative ladders and measured phase concentration per scale."""
        rows = self.results.scale_rows
        if not rows:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))

        for activation in self._series_activations():
            selected = sorted(
                [r for r in rows if r.activation == activation.name],
                key=lambda r: r.scale_index,
            )
            if not selected:
                continue
            axes[0].plot(
                [r.scale_index for r in selected],
                [r.derivative_scale_amplitude for r in selected],
                marker=self._marker(activation.name),
                markersize=3.0,
                color=self._color(activation.name),
                linestyle=self._linestyle(activation.name),
                label=activation.display_name,
            )
            axes[1].plot(
                [r.scale_index for r in selected],
                [r.phase_concentration for r in selected],
                marker=self._marker(activation.name),
                markersize=3.0,
                color=self._color(activation.name),
                linestyle=self._linestyle(activation.name),
                label=activation.display_name,
            )

        axes[0].set_yscale("log")
        axes[0].set_xlabel("scale index")
        axes[0].set_ylabel("derivative amplitude of the scale")
        axes[0].set_title("Derivative ladders")
        axes[0].grid(True, alpha=0.25, which="both")
        axes[0].legend(frameon=False)

        axes[1].axhline(
            PHASE_CONCENTRATION_THRESHOLD,
            color=COLOR_PALETTE_A[8],
            linestyle="--",
            linewidth=1.0,
        )
        axes[1].set_ylim(-0.03, 1.05)
        axes[1].set_xlabel("scale index")
        axes[1].set_ylabel("phase concentration")
        axes[1].set_title("Measured phase of each scale on float32 inputs")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(frameon=False)

        fig.suptitle("Scale structure of the fractal activations")
        self._save(fig, PLOT_DIR / "scale_structure")

    def plot_phase_regimes(self) -> None:
        """Where each activation's derivative amplitude sits, by phase regime."""
        rows = self.results.scale_rows
        if not rows:
            return
        series = self._series_activations()
        # One shared logarithmic axis: the point of the figure is that the four
        # activations place their derivative amplitude at entirely different
        # magnitudes, which separate axes would hide.
        fig, axes = plt.subplots(
            1, len(series), figsize=(4.4 * len(series), 4.6), sharey=True
        )
        if len(series) == 1:
            axes = [axes]
        amplitudes = [
            r.derivative_scale_amplitude for r in rows if r.derivative_scale_amplitude > 0
        ]
        bottom = min(amplitudes) / 10.0 if amplitudes else 1e-12
        top = max(amplitudes) * 10.0 if amplitudes else 1.0

        for ax, activation in zip(axes, series):
            selected = sorted(
                [r for r in rows if r.activation == activation.name],
                key=lambda r: r.scale_index,
            )
            for regime, color in REGIME_COLORS.items():
                group = [r for r in selected if r.phase_regime == regime]
                if not group:
                    continue
                ax.bar(
                    [r.scale_index for r in group],
                    [r.derivative_scale_amplitude for r in group],
                    width=0.9,
                    color=color,
                    label=regime.replace("_", " "),
                )
            ax.set_yscale("log")
            ax.set_ylim(bottom, top)
            ax.set_title(activation.display_name)
            ax.set_xlabel("scale index")
            ax.grid(True, alpha=0.25, axis="y", which="major")
        axes[0].set_ylabel("derivative amplitude of the scale")
        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes[1:]:
            extra_handles, extra_labels = ax.get_legend_handles_labels()
            for handle, label in zip(extra_handles, extra_labels):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, 0.02),
        )
        fig.suptitle("Phase regime of every retained scale")
        self._save(fig, PLOT_DIR / "phase_regimes")

    def plot_precision_comparison(self) -> None:
        """Stability of the derivative under a sub-resolution input shift."""
        rows = [
            r
            for r in self.results.precision_rows
            if r.ulp_multiple == ULP_MULTIPLES[0]
        ]
        if not rows:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))

        for activation in self._series_activations():
            selected = sorted(
                [r for r in rows if r.activation == activation.name],
                key=lambda r: (r.terms or 0),
            )
            if not selected:
                continue
            axes[0].plot(
                [r.terms for r in selected],
                [max(r.derivative_median_relative_change, 1e-12) for r in selected],
                marker=self._marker(activation.name),
                markersize=5.0,
                color=self._color(activation.name),
                linestyle=self._linestyle(activation.name),
                label=activation.display_name,
            )
            axes[1].plot(
                [r.terms for r in selected],
                [r.derivative_sign_flip_fraction for r in selected],
                marker=self._marker(activation.name),
                markersize=5.0,
                color=self._color(activation.name),
                linestyle=self._linestyle(activation.name),
                label=activation.display_name,
            )

        for activation in self.activations:
            if activation.is_series():
                continue
            reference = [r for r in rows if r.activation == activation.name]
            if not reference:
                continue
            axes[0].axhline(
                max(reference[0].derivative_median_relative_change, 1e-12),
                color=self._color(activation.name),
                linestyle="--",
                linewidth=1.2,
                label=f"{activation.display_name} (no truncation)",
            )
            axes[1].axhline(
                reference[0].derivative_sign_flip_fraction,
                color=self._color(activation.name),
                linestyle="--",
                linewidth=1.2,
            )

        axes[0].set_yscale("log")
        axes[0].set_ylim(bottom=5e-13)
        axes[0].set_xlabel("retained terms")
        axes[0].set_ylabel("median relative change of the derivative")
        axes[0].set_title(
            "Derivative under a shift of one ULP(float32)\n"
            "(a change of exactly zero is drawn at the lower limit)"
        )
        axes[0].grid(True, alpha=0.25, which="both")
        axes[0].legend(frameon=False, ncol=2, fontsize=8 * FONT_SCALE)

        axes[1].axhline(0.5, color=COLOR_PALETTE_A[8], linestyle=":", linewidth=1.0)
        axes[1].set_ylim(-0.03, 0.62)
        axes[1].set_xlabel("retained terms")
        axes[1].set_ylabel("fraction of sign changes")
        axes[1].set_title("Sign of the derivative under the same shift")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(frameon=False)

        fig.suptitle("Resolution of the derivative at float32 input precision")
        self._save(fig, PLOT_DIR / "precision_comparison")

    def plot_realizability(self) -> None:
        """Analytic derivative against the slope the function realizes."""
        rows = self.results.realizability_rows
        if not rows:
            return
        step_labels = []
        for row in rows:
            if row.step_label not in step_labels:
                step_labels.append(row.step_label)
        primary = primary_realizability_step(rows) or step_labels[0]
        selected = [r for r in rows if r.step_label == primary]
        selected.sort(key=lambda r: self.order.index(r.activation))

        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))
        positions = np.arange(len(selected))
        width = 0.38

        axes[0].bar(
            positions - width / 2,
            [max(r.analytic_median_abs, 1e-12) for r in selected],
            width=width,
            color=COLOR_PALETTE_A[1],
            label="analytic derivative",
        )
        axes[0].bar(
            positions + width / 2,
            [max(r.realized_median_abs, 1e-12) for r in selected],
            width=width,
            color=COLOR_PALETTE_A[3],
            label="realized slope",
        )
        axes[0].set_yscale("log")
        axes[0].set_xticks(positions)
        axes[0].set_xticklabels(
            [r.display_name for r in selected], rotation=20, ha="right"
        )
        axes[0].set_ylabel("median magnitude")
        axes[0].set_title(f"Gradient against realized slope, step {primary}")
        axes[0].grid(True, alpha=0.25, axis="y", which="both")
        axes[0].legend(frameon=False)

        for index, step in enumerate(step_labels):
            group = [r for r in rows if r.step_label == step]
            group.sort(key=lambda r: self.order.index(r.activation))
            # Where the realized slope sits at the float32 output resolution the
            # ratio says nothing about the activation, so it is left out rather
            # than plotted as a spike.
            axes[1].plot(
                np.arange(len(group)),
                [
                    max(r.median_ratio_analytic_to_realized, 1e-12)
                    if r.realized_above_floor
                    else np.nan
                    for r in group
                ],
                marker="osd^"[index % 4],
                markersize=6.0,
                linewidth=1.2,
                color=COLOR_PALETTE_A[index % len(COLOR_PALETTE_A)],
                label=f"step {step}",
            )
        axes[1].axhline(1.0, color=COLOR_PALETTE_A[8], linestyle="--", linewidth=1.0)
        axes[1].set_yscale("log")
        axes[1].set_xticks(np.arange(len(selected)))
        axes[1].set_xticklabels(
            [r.display_name for r in selected], rotation=20, ha="right"
        )
        axes[1].set_ylabel("median ratio analytic / realized")
        axes[1].set_title(
            "Agreement between the two\n"
            "(omitted where the realized slope is below the float32 output "
            "resolution)"
        )
        axes[1].grid(True, alpha=0.25, axis="y", which="both")
        axes[1].legend(frameon=False)

        fig.suptitle("Does the gradient describe the function the network computes")
        self._save(fig, PLOT_DIR / "gradient_realizability")

    def plot_network_probe(self) -> None:
        """Gradient norms and accuracy reached in the compact network probe."""
        rows = self.results.network_rows
        if not rows:
            return
        keys: List[Tuple[str, Optional[int]]] = []
        for row in rows:
            key = (row.activation, row.terms)
            if key not in keys:
                keys.append(key)
        keys.sort(key=lambda k: (self.order.index(k[0]), k[1] if k[1] else 0))
        labels = [
            f"{self._display(name)}" + (f"\n{terms} terms" if terms else "")
            for name, terms in keys
        ]
        optimizers = sorted({row.optimizer for row in rows})

        fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 0.75 * len(keys)), 9.0))
        width = 0.8 / max(len(optimizers), 1)

        for index, optimizer_name in enumerate(optimizers):
            norms, accuracies = [], []
            for name, terms in keys:
                group = [
                    r
                    for r in rows
                    if r.activation == name
                    and r.terms == terms
                    and r.optimizer == optimizer_name
                ]
                norms.append(float(np.mean([r.grad_norm_mean for r in group])) if group else np.nan)
                accuracies.append(
                    float(np.mean([r.test_accuracy for r in group])) if group else np.nan
                )
            positions = np.arange(len(keys)) + (index - (len(optimizers) - 1) / 2) * width
            axes[0].bar(
                positions,
                norms,
                width=width * 0.9,
                color=COLOR_PALETTE_A[index % len(COLOR_PALETTE_A)],
                label=optimizer_name,
            )
            axes[1].bar(
                positions,
                accuracies,
                width=width * 0.9,
                color=COLOR_PALETTE_A[index % len(COLOR_PALETTE_A)],
                label=optimizer_name,
            )

        axes[0].set_yscale("log")
        axes[0].set_ylabel("mean global gradient norm per step")
        axes[0].set_title(f"Gradient magnitude during training, {NETWORK_PROBE_DATASET}")
        axes[0].grid(True, alpha=0.25, axis="y", which="both")
        axes[0].legend(frameon=False)
        axes[0].set_xticks(np.arange(len(keys)))
        axes[0].set_xticklabels([])

        axes[1].set_ylabel("test accuracy")
        axes[1].set_title("Accuracy reached in the same runs")
        axes[1].grid(True, alpha=0.25, axis="y")
        axes[1].set_xticks(np.arange(len(keys)))
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7 * FONT_SCALE)
        axes[1].legend(frameon=False)

        self._save(fig, PLOT_DIR / "network_probe")

    def make_all(self) -> None:
        self.plot_function_overview()
        self.plot_truncation_comparison()
        self.plot_scale_structure()
        self.plot_phase_regimes()
        self.plot_precision_comparison()
        self.plot_realizability()
        self.plot_network_probe()


# =============================================================================
# 12) OUTPUT
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


class ComparisonReport:
    """Writes the human-readable report, the tables and the machine record."""

    def __init__(
        self,
        activations: Sequence[ActivationReference],
        results: StudyResults,
    ) -> None:
        self.activations = list(activations)
        self.results = results
        self.order = [a.name for a in self.activations]

    # ---- tables -------------------------------------------------------------

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Any]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        dict_rows = [asdict(row) for row in rows]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict_rows[0].keys()))
            writer.writeheader()
            writer.writerows(dict_rows)

    def write_tables(self) -> None:
        self._write_csv(OUTPUT_DIR / "deployed_comparison.csv", self.results.deployed_rows)
        self._write_csv(OUTPUT_DIR / "truncation_sweep.csv", self.results.truncation_rows)
        self._write_csv(OUTPUT_DIR / "scale_structure.csv", self.results.scale_rows)
        self._write_csv(OUTPUT_DIR / "precision_probe.csv", self.results.precision_rows)
        self._write_csv(
            OUTPUT_DIR / "gradient_realizability.csv", self.results.realizability_rows
        )
        self._write_csv(OUTPUT_DIR / "autodiff_crosscheck.csv", self.results.autodiff_rows)
        self._write_csv(OUTPUT_DIR / "network_probe.csv", self.results.network_rows)

    def write_json(self) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "activation_comparison_analysis.json"
        payload = {
            "format_version": 1,
            "produced_by": "B_04_activation_comparison_diagnostics.py",
            "activations": [
                {
                    "name": a.name,
                    "display_name": a.display_name,
                    "family": a.family,
                    "deployed_parameters": a.deployed_parameter_text,
                    "deployed_terms": a.deployed_terms,
                    "index_range": a.index_range_text or "-",
                    "retained_scales": a.retained_scales(),
                    "accumulation_dtype": a.accumulation_dtype,
                    "value_scale_amplitude": a.value_scale_text,
                    "derivative_scale_amplitude": a.derivative_scale_text,
                    "ladder_type": a.ladder_type(),
                    "derivative_scale_growth_factor": a.derivative_scale_growth_factor(),
                    "derivative_amplitude_bound": a.derivative_amplitude_bound(),
                }
                for a in self.activations
            ],
            "implementation_notes": IMPLEMENTATION_NOTES,
            "settings": {
                "truncation_grid": TRUNCATION_GRID,
                "include_short_truncation": INCLUDE_SHORT_TRUNCATION,
                "sample_range": SAMPLE_RANGE,
                "n_sample_points": N_SAMPLE_POINTS,
                "sample_seed": SAMPLE_SEED,
                "sample_precisions": SAMPLE_PRECISIONS,
                "ulp_multiples": ULP_MULTIPLES,
                "realizability_steps": [
                    "1 ULP(float32)" if s is None else s for s in REALIZABILITY_STEPS
                ],
                "lattice_fraction_threshold": LATTICE_FRACTION_THRESHOLD,
                "phase_concentration_threshold": PHASE_CONCENTRATION_THRESHOLD,
                "eps_float32": EPS_FLOAT32,
                "eps_float64": EPS_FLOAT64,
                "network_probe_enabled": NETWORK_PROBE_ENABLED,
                "network_probe_dataset": NETWORK_PROBE_DATASET,
                "network_probe_truncations": NETWORK_PROBE_TRUNCATIONS,
                "network_probe_optimizers": NETWORK_PROBE_OPTIMIZERS,
                "network_probe_runs": NETWORK_PROBE_RUNS,
            },
            "environment": self.results.environment,
            "sample": self.results.sample_summary,
            "deployed_comparison": [asdict(r) for r in self.results.deployed_rows],
            "truncation_sweep": [asdict(r) for r in self.results.truncation_rows],
            "scale_structure": [asdict(r) for r in self.results.scale_rows],
            "phase_regime_summary": ScaleStructureProbe.regime_summary(
                self.results.scale_rows
            ),
            "precision_probe": [asdict(r) for r in self.results.precision_rows],
            "gradient_realizability": [
                asdict(r) for r in self.results.realizability_rows
            ],
            "autodiff_crosscheck": [asdict(r) for r in self.results.autodiff_rows],
            "network_probe": [asdict(r) for r in self.results.network_rows],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2)
        return path

    def write_latex_tables(self) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "activation_comparison_tables.tex"
        lines: List[str] = ["% Produced by B_04_activation_comparison_diagnostics.py", ""]

        # Table 1: the six activations at their deployed settings.
        deployed = [
            r for r in self.results.deployed_rows if r.sample_precision == "float32"
        ]
        deployed.sort(key=lambda r: self.order.index(r.activation))
        probe = {
            (r.activation, r.terms): r
            for r in self.results.precision_rows
            if r.ulp_multiple == ULP_MULTIPLES[0]
        }
        primary_step = primary_realizability_step(self.results.realizability_rows)
        realizability = {
            r.activation: r
            for r in self.results.realizability_rows
            if r.step_label == primary_step
        }

        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\begin{tabular}{lrlrrrrr}")
        lines.append("\\hline")
        lines.append(
            "activation & scales & ladder & mean$|\\varphi'|$ & $\\max|\\varphi'|$ & "
            "non-finite & sign flips & analytic/realized \\\\"
        )
        lines.append("\\hline")
        for row in deployed:
            key = (row.activation, row.deployed_terms)
            flip = probe.get(key)
            real = realizability.get(row.activation)
            lines.append(
                f"{row.display_name} & {row.retained_scales} & {row.ladder_type} & "
                f"{row.derivative_mean_abs:.2e} & {row.derivative_max_abs:.2e} & "
                f"{row.nonfinite_values + row.nonfinite_derivatives} & "
                + (f"{flip.derivative_sign_flip_fraction:.3f}" if flip else "--")
                + " & "
                + (f"{real.median_ratio_analytic_to_realized:.2e}" if real else "--")
                + " \\\\"
            )
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append(
            "\\caption{The six activations of the classification experiments at "
            "their deployed settings, measured on "
            f"{N_SAMPLE_POINTS} pre-activations drawn uniformly from "
            f"$[-{SAMPLE_RANGE:g},{SAMPLE_RANGE:g}]$ and quantised to float32. "
            "Non-finite counts cover values and derivatives together. Sign flips "
            "is the fraction of sample points at which the derivative changes "
            "sign when the pre-activation is moved by one unit in the last place "
            "of its float32 representation. The last column is the median ratio "
            "between the analytic derivative and the slope the activation "
            f"realizes between two inputs separated by {primary_step}.}}"
        )
        lines.append("\\label{tab:activation_deployed_comparison}")
        lines.append("\\end{table}")
        lines.append("")

        # Table 2: truncation sweep of the four fractal activations.
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\begin{tabular}{lrrrrrrr}")
        lines.append("\\hline")
        lines.append(
            "activation & terms & scales & "
            "$\\max|\\varphi_N-\\varphi_{\\mathrm{dep}}|$ & mean$|\\varphi'|$ & "
            "non-finite & osc. & degenerate \\\\"
        )
        lines.append("\\hline")
        for activation in self.activations:
            if not activation.is_series():
                continue
            selected = sorted(
                [r for r in self.results.truncation_rows if r.activation == activation.name],
                key=lambda r: r.terms,
            )
            for row in selected:
                lines.append(
                    f"{row.display_name} & {row.terms} & {row.retained_scales} & "
                    f"{row.value_max_deviation_from_deployed:.2e} & "
                    f"{row.derivative_mean_abs:.2e} & "
                    f"{row.nonfinite_values + row.nonfinite_derivatives} & "
                    f"{row.oscillating_scales} & "
                    f"{row.lattice_locked_scales + row.rounding_dominated_scales} \\\\"
                )
            lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append(
            "\\caption{Truncation sweep of the four fractal activations. The "
            "deviation of the value is measured against the deployed truncation "
            "of the same activation. The column osc. counts the retained scales "
            "whose phase is still spread over the circle, and degenerate counts "
            "those that are locked to the lattice of their own period or "
            "dominated by the rounding of the phase product.}"
        )
        lines.append("\\label{tab:activation_truncation_sweep}")
        lines.append("\\end{table}")

        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return path

    # ---- text report --------------------------------------------------------

    def _deployed_rows(self, precision: str = "float32") -> List[DeployedRow]:
        rows = [
            r for r in self.results.deployed_rows if r.sample_precision == precision
        ]
        rows.sort(key=lambda r: self.order.index(r.activation))
        return rows

    def build_report(self) -> str:
        lines: List[str] = []
        results = self.results

        lines.append(RULE)
        lines.append(
            "Numerical comparison of the six activation functions of the "
            "classification experiments"
        )
        lines.append(RULE)
        lines.append("")
        lines.append(
            "Every number below is measured on one common sample of "
            f"{results.sample_summary.get('n_points', 0)} pre-activations drawn "
            f"uniformly from [-{SAMPLE_RANGE:g}, {SAMPLE_RANGE:g}] and quantised "
            "to float32,"
        )
        lines.append(
            "which is the situation inside the networks: features are "
            "MinMax-scaled to [0, 1] and the first layer uses Glorot-uniform "
            "weights."
        )
        lines.append("")
        lines.append(f"sample      : {results.sample_summary}")
        lines.append(f"environment : {results.environment}")
        lines.append("")

        lines.append(THIN_RULE)
        lines.append("0) The activations as deployed")
        lines.append(THIN_RULE)
        lines.append(
            f"{'activation':>16} {'family':>9} {'terms':>6} {'scales':>7} "
            f"{'index range':>22} {'accumulation':>14} {'ladder':>11} {'growth':>8} "
            f"{'bound':>11}"
        )
        for row in self._deployed_rows():
            lines.append(
                f"{row.display_name:>16} {row.family:>9} "
                f"{str(row.deployed_terms):>6} {row.retained_scales:>7} "
                f"{row.index_range:>22} {row.accumulation_dtype:>14} "
                f"{row.ladder_type:>11} "
                + (
                    f"{row.derivative_scale_growth_factor:>8.3f} "
                    if np.isfinite(row.derivative_scale_growth_factor)
                    else f"{'-':>8} "
                )
                + (
                    f"{row.derivative_amplitude_bound:>11.3e}"
                    if np.isfinite(row.derivative_amplitude_bound)
                    else f"{'-':>11}"
                )
            )
        lines.append("")
        for row in self._deployed_rows():
            lines.append(f"  {row.display_name:>16} : {row.deployed_parameters}")
        lines.append("")
        lines.append("Properties of the deployed code that the formulas do not show:")
        for note in IMPLEMENTATION_NOTES:
            wrapped = note
            lines.append(f"  - {wrapped}")
        lines.append("")

        lines.append(THIN_RULE)
        lines.append("1) Value and derivative at the deployed settings")
        lines.append(THIN_RULE)
        lines.append(
            f"{'activation':>16} {'prec':>8} {'mean|phi|':>11} {'max|phi|':>11} "
            f"{'mean|g|':>11} {'median|g|':>11} {'p95|g|':>11} {'max|g|':>11} "
            f"{'frac g>0':>9} {'nf val':>7} {'nf der':>7}"
        )
        for precision in SAMPLE_PRECISIONS:
            for row in self._deployed_rows(precision):
                lines.append(
                    f"{row.display_name:>16} {row.sample_precision:>8} "
                    f"{row.value_mean_abs:>11.3e} {row.value_max_abs:>11.3e} "
                    f"{row.derivative_mean_abs:>11.3e} "
                    f"{row.derivative_median_abs:>11.3e} "
                    f"{row.derivative_p95_abs:>11.3e} "
                    f"{row.derivative_max_abs:>11.3e} "
                    f"{row.fraction_positive_derivative:>9.4f} "
                    f"{row.nonfinite_values:>7} {row.nonfinite_derivatives:>7}"
                )
            lines.append("")
        lines.append(
            "The two precisions differ only in how the pre-activation is "
            "quantised before the activation sees it.\nWhere a row changes "
            "between them, the activation is reading information that a float32 "
            "network does\nnot have."
        )
        lines.append("")

        lines.append(THIN_RULE)
        lines.append("2) Resolution accounting at the median sample magnitude")
        lines.append(THIN_RULE)
        lines.append(
            f"{'activation':>16} {'scales':>7} {'resolved':>9} {'unresolved':>11} "
            f"{'unres. amplitude':>17} {'share':>9}"
        )
        for row in self._deployed_rows():
            if row.retained_scales == 0:
                continue
            lines.append(
                f"{row.display_name:>16} {row.retained_scales:>7} "
                f"{row.resolved_scales_float32_input:>9} "
                f"{row.unresolved_scales_float32_input:>11} "
                f"{row.unresolved_derivative_amplitude:>17.3e} "
                f"{row.unresolved_derivative_amplitude_fraction:>9.2e}"
            )
        lines.append("")
        lines.append(
            "A scale is counted as resolved if the representation error of the "
            "float32 pre-activation moves\nits phase by less than one radian. "
            "The share of the derivative amplitude carried by unresolved "
            "scales\nis not by itself the distinguishing quantity: what "
            "separates the activations is the absolute\namplitude those scales "
            "can contribute."
        )
        lines.append("")

        lines.append(THIN_RULE)
        lines.append("3) Truncation sweep of the fractal activations")
        lines.append(THIN_RULE)
        for activation in self.activations:
            if not activation.is_series():
                continue
            selected = sorted(
                [r for r in results.truncation_rows if r.activation == activation.name],
                key=lambda r: r.terms,
            )
            if not selected:
                continue
            lines.append(
                f"{activation.display_name} "
                f"({activation.term_parameter}, deployed {activation.deployed_terms})"
            )
            lines.append(
                f"  {'terms':>6} {'scales':>7} {'max dev':>11} {'rms dev':>11} "
                f"{'mean|g|':>11} {'max|g|':>11} {'bound':>11} {'frac g>0':>9} "
                f"{'osc':>5} {'lock':>5} {'noise':>6} {'nf':>4}"
            )
            for row in selected:
                lines.append(
                    f"  {row.terms:>6} {row.retained_scales:>7} "
                    f"{row.value_max_deviation_from_deployed:>11.3e} "
                    f"{row.value_rms_deviation_from_deployed:>11.3e} "
                    f"{row.derivative_mean_abs:>11.3e} "
                    f"{row.derivative_max_abs:>11.3e} "
                    f"{row.derivative_amplitude_bound:>11.3e} "
                    f"{row.fraction_positive_derivative:>9.4f} "
                    f"{row.oscillating_scales:>5} {row.lattice_locked_scales:>5} "
                    f"{row.rounding_dominated_scales:>6} "
                    f"{row.nonfinite_values + row.nonfinite_derivatives:>4}"
                )
            lines.append("")
        lines.append(
            "The deviation of the value is measured against the deployed "
            "truncation of the same activation,\nso each block says how much of "
            "its own deployed function a shorter sum reproduces."
        )
        lines.append("")

        if results.scale_rows:
            lines.append(THIN_RULE)
            lines.append("4) Phase regime of every retained scale")
            lines.append(THIN_RULE)
            summary = ScaleStructureProbe.regime_summary(results.scale_rows)
            lines.append(
                f"{'activation':>16} {'regime':>20} {'scales':>7} {'first':>6} "
                f"{'last':>6} {'deriv. amplitude':>18} {'share':>9}"
            )
            for activation in self.activations:
                entry = summary.get(activation.name)
                if not entry:
                    continue
                for regime in ("oscillating", "lattice_locked", "rounding_dominated"):
                    values = entry.get(regime, {})
                    if not values or not values.get("n_scales"):
                        continue
                    lines.append(
                        f"{activation.display_name:>16} {regime:>20} "
                        f"{values['n_scales']:>7} "
                        f"{values['first_scale_index']:>6} "
                        f"{values['last_scale_index']:>6} "
                        f"{values['derivative_amplitude']:>18.3e} "
                        f"{values['derivative_amplitude_share']:>9.2e}"
                    )
                lines.append("")
            lines.append(
                "A uniformly spread phase has concentration zero, mean cos zero "
                "and mean |sin| = 2/pi = 0.63662.\nA scale counts as locked when "
                "its phase has collapsed onto a coarse lattice, which the first "
                "or a\nhigher circular moment detects, and as rounding dominated "
                "when the input lies on the lattice of\nthe scale while the "
                "phase is spread anyway, so that the spread comes from the "
                "arithmetic."
            )
            without_lock = [
                activation.display_name
                for activation in self.activations
                if summary.get(activation.name)
                and not summary[activation.name]
                .get("lattice_locked", {})
                .get("n_scales", 0)
            ]
            with_lock = [
                activation.display_name
                for activation in self.activations
                if summary.get(activation.name)
                and summary[activation.name].get("lattice_locked", {}).get("n_scales", 0)
            ]
            if with_lock:
                lines.append("")
                lines.append(
                    "Locking occurs for " + ", ".join(with_lock) + ". Their "
                    "frequency ladders have base two, and a float32\nnumber times "
                    "a power of two is exact, so beyond the mantissa length of "
                    "the pre-activation their\nscales land on the lattice of "
                    "their own period and stop oscillating."
                )
            if without_lock:
                lines.append("")
                lines.append(
                    "No locking occurs for " + ", ".join(without_lock) + ", whose "
                    "ladder is not a power of two."
                )
            lines.append("")

        lines.append(THIN_RULE)
        lines.append("5) Input precision probe at the deployed settings")
        lines.append(THIN_RULE)
        lines.append(
            f"{'activation':>16} {'ULP':>5} {'med |dval|':>12} {'max |dval|':>12} "
            f"{'med rel dder':>13} {'p95 rel dder':>13} {'sign flips':>11} {'nf':>4}"
        )
        for multiple in ULP_MULTIPLES:
            for activation in self.activations:
                deployed_terms = activation.deployed_terms
                selected = [
                    r
                    for r in results.precision_rows
                    if r.activation == activation.name
                    and r.ulp_multiple == multiple
                    and r.terms == deployed_terms
                ]
                for row in selected:
                    lines.append(
                        f"{row.display_name:>16} {row.ulp_multiple:>5.1f} "
                        f"{row.value_median_abs_change:>12.3e} "
                        f"{row.value_max_abs_change:>12.3e} "
                        f"{row.derivative_median_relative_change:>13.3e} "
                        f"{row.derivative_p95_relative_change:>13.3e} "
                        f"{row.derivative_sign_flip_fraction:>11.4f} "
                        f"{row.nonfinite_after_perturbation:>4}"
                    )
            lines.append("")
        lines.append(
            "A shift of one unit in the last place of the float32 "
            "pre-activation is below the accuracy with\nwhich a network can "
            "place that input. The full dependence on the truncation length is "
            "in\nprecision_probe.csv and in the figure precision_comparison; it "
            "is not monotone for the activations\nwith a base-two ladder, "
            "because a locked scale is stable under the shift while "
            "contributing a\nderivative that the function does not have."
        )
        lines.append("")

        if results.realizability_rows:
            lines.append(THIN_RULE)
            lines.append("6) Gradient realizability")
            lines.append(THIN_RULE)
            primary = primary_realizability_step(results.realizability_rows)
            lines.append(
                f"{'activation':>16} {'step':>16} {'median step':>12} "
                f"{'analytic':>11} {'realized':>11} {'floor':>11} {'above':>6} "
                f"{'ratio':>11} {'p95 ratio':>11} {'rank corr':>10} {'sign agree':>11}"
            )
            steps: List[str] = []
            for row in results.realizability_rows:
                if row.step_label not in steps:
                    steps.append(row.step_label)
            for step in steps:
                for activation in self.activations:
                    selected = [
                        r
                        for r in results.realizability_rows
                        if r.activation == activation.name and r.step_label == step
                    ]
                    for row in selected:
                        lines.append(
                            f"{row.display_name:>16} {row.step_label:>16} "
                            f"{row.median_step:>12.3e} "
                            f"{row.analytic_median_abs:>11.3e} "
                            f"{row.realized_median_abs:>11.3e} "
                            f"{row.output_quantization_floor:>11.3e} "
                            f"{str(row.realized_above_floor):>6} "
                            f"{row.median_ratio_analytic_to_realized:>11.3e} "
                            f"{row.p95_ratio_analytic_to_realized:>11.3e} "
                            f"{row.rank_correlation:>10.4f} "
                            f"{row.sign_agreement_fraction:>11.4f}"
                        )
                lines.append("")
            lines.append(
                "The realized slope is the difference quotient of the activation "
                "between two inputs separated by\nthe stated step, that is the "
                "slope a network could act on at that resolution. A ratio of one "
                "and a\nrank correlation of one mean that the gradient describes "
                "the function the network computes."
            )
            lines.append("")
            lines.append(
                "Both the value and the slope pass through the float32 output "
                "cast, so the difference quotient\ncannot resolve a slope below "
                "the quantization of the output divided by the step, which is "
                "the\ncolumn 'floor'. At a step of one unit in the last place "
                "that floor is of the same order as the\nderivatives of the "
                "well-behaved activations, so their rows there are not "
                "significant and are\nmarked accordingly; the comparison is read "
                f"from the step {primary}, where every activation is\nabove its "
                "floor."
            )
            lines.append("")

        lines.append(THIN_RULE)
        lines.append("7) Cross-check against the deployed TensorFlow implementations")
        lines.append(THIN_RULE)
        if results.autodiff_rows:
            lines.append(
                f"{'activation':>38} {'terms':>6} {'points':>8} {'max |dval|':>12} "
                f"{'max rel val':>12} {'med rel der':>12} {'max rel der':>12} "
                f"{'nf val':>7} {'nf grad':>8}"
            )
            for row in results.autodiff_rows:
                lines.append(
                    f"{row.activation[:38]:>38} {str(row.terms):>6} {row.n_points:>8} "
                    f"{row.max_abs_value_difference:>12.3e} "
                    f"{row.max_relative_value_difference:>12.3e} "
                    f"{row.median_relative_derivative_difference:>12.3e} "
                    f"{row.max_relative_derivative_difference:>12.3e} "
                    f"{row.nonfinite_tensorflow_values:>7} "
                    f"{row.nonfinite_tensorflow_gradients:>8}"
                )
        else:
            lines.append("not run")
            if not TENSORFLOW_AVAILABLE:
                lines.append(f"  TensorFlow unavailable: {TENSORFLOW_IMPORT_ERROR}")
            elif not ACTIVATION_MODULE_AVAILABLE:
                lines.append(
                    f"  fractal_activation_functions unavailable: "
                    f"{ACTIVATION_MODULE_IMPORT_ERROR}"
                )
            if "autodiff_crosscheck_error" in results.environment:
                lines.append(
                    f"  failed: {results.environment['autodiff_crosscheck_error']}"
                )
        lines.append("")

        lines.append(THIN_RULE)
        lines.append(f"8) Compact network probe, {NETWORK_PROBE_DATASET}")
        lines.append(THIN_RULE)
        if results.network_rows:
            lines.append(
                f"{'activation':>16} {'terms':>6} {'optimizer':>10} {'steps':>7} "
                f"{'mean|g|':>11} {'median|g|':>11} {'max|g|':>11} {'nf grad':>8} "
                f"{'nf loss':>8} {'accuracy':>9} {'pre-act p99':>12}"
            )
            keys: List[Tuple[str, Optional[int]]] = []
            for row in results.network_rows:
                key = (row.activation, row.terms)
                if key not in keys:
                    keys.append(key)
            keys.sort(key=lambda k: (self.order.index(k[0]), k[1] if k[1] else 0))
            for name, terms in keys:
                for optimizer_name in sorted({r.optimizer for r in results.network_rows}):
                    group = [
                        r
                        for r in results.network_rows
                        if r.activation == name
                        and r.terms == terms
                        and r.optimizer == optimizer_name
                    ]
                    if not group:
                        continue
                    lines.append(
                        f"{group[0].display_name:>16} {str(terms):>6} "
                        f"{optimizer_name:>10} "
                        f"{int(np.mean([r.steps for r in group])):>7} "
                        f"{np.mean([r.grad_norm_mean for r in group]):>11.3e} "
                        f"{np.mean([r.grad_norm_median for r in group]):>11.3e} "
                        f"{np.max([r.grad_norm_max for r in group]):>11.3e} "
                        f"{int(np.sum([r.nonfinite_gradient_steps for r in group])):>8} "
                        f"{int(np.sum([r.nonfinite_loss_steps for r in group])):>8} "
                        f"{np.mean([r.test_accuracy for r in group]):>9.4f} "
                        f"{np.mean([r.preactivation_p99_abs for r in group]):>12.4f}"
                    )
            lines.append("")
            lines.append(
                f"Averages over {NETWORK_PROBE_RUNS} seeds per cell. The accuracy "
                "column is indicative only: the probe\nexists to show what the "
                "optimizer receives, not to rank the activations, and the study's "
                "own\nclassification results do that with forty seeds per "
                "configuration."
            )
        else:
            lines.append("not run")
            if not NETWORK_PROBE_ENABLED:
                lines.append("  disabled in section 0")
            elif not TENSORFLOW_AVAILABLE:
                lines.append(f"  TensorFlow unavailable: {TENSORFLOW_IMPORT_ERROR}")
            elif not ACTIVATION_MODULE_AVAILABLE:
                lines.append(
                    f"  fractal_activation_functions unavailable: "
                    f"{ACTIVATION_MODULE_IMPORT_ERROR}"
                )
            if "network_probe_error" in results.environment:
                lines.append(f"  failed: {results.environment['network_probe_error']}")
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
        """Statements assembled from the measured values."""
        out: List[str] = []
        deployed = self._deployed_rows()
        if not deployed:
            return ["no measurements available"]

        nonfinite = sum(
            r.nonfinite_values + r.nonfinite_derivatives
            for r in self.results.deployed_rows
        ) + sum(
            r.nonfinite_values + r.nonfinite_derivatives
            for r in self.results.truncation_rows
        )
        out.append(
            f"No activation produced a single non-finite value or derivative at "
            f"any setting: {nonfinite} in all of the deployed and truncated "
            f"evaluations. Truncation length is not a stability problem for any "
            f"of the six."
        )

        series_rows = [r for r in deployed if r.retained_scales > 0]
        if series_rows:
            ordered = sorted(series_rows, key=lambda r: r.derivative_mean_abs)
            out.append(
                "At the deployed settings the mean gradient magnitude differs by "
                f"{ordered[-1].derivative_mean_abs / max(ordered[0].derivative_mean_abs, 1e-300):.2e} "
                f"between the fractal activations, from {ordered[0].derivative_mean_abs:.3e} "
                f"({ordered[0].display_name}) to {ordered[-1].derivative_mean_abs:.3e} "
                f"({ordered[-1].display_name}), while the two standard "
                "activations stay of order one."
            )
            by_ladder: Dict[str, List[str]] = {}
            for row in series_rows:
                by_ladder.setdefault(row.ladder_type, []).append(row.display_name)
            out.append(
                "The derivative ladders separate them: "
                + "; ".join(
                    f"{ladder}: {', '.join(names)}" for ladder, names in by_ladder.items()
                )
                + "."
            )

        # Value convergence per activation.
        for activation in self.activations:
            if not activation.is_series():
                continue
            selected = sorted(
                [
                    r
                    for r in self.results.truncation_rows
                    if r.activation == activation.name
                    and r.terms != activation.deployed_terms
                ],
                key=lambda r: r.terms,
            )
            if not selected:
                continue
            finest = selected[-1]
            deployed_row = [
                r
                for r in self.results.truncation_rows
                if r.activation == activation.name
                and r.terms == activation.deployed_terms
            ]
            gradient_span = (
                f"the mean gradient magnitude over the same sweep runs from "
                f"{selected[0].derivative_mean_abs:.3e} at {selected[0].terms} terms "
                f"to {deployed_row[0].derivative_mean_abs:.3e} at the deployed "
                f"{activation.deployed_terms}"
                if deployed_row
                else ""
            )
            reached = next(
                (
                    row
                    for row in selected
                    if row.value_max_deviation_from_deployed <= VALUE_CONVERGENCE_TOLERANCE
                ),
                None,
            )
            if reached is not None:
                out.append(
                    f"{activation.display_name}: the value reproduces the "
                    f"deployed function to within "
                    f"{VALUE_CONVERGENCE_TOLERANCE:.0e} from {reached.terms} of "
                    f"{activation.deployed_terms} terms on "
                    f"(largest deviation {reached.value_max_deviation_from_deployed:.2e}), "
                    f"while {gradient_span}."
                )
            else:
                out.append(
                    f"{activation.display_name}: the value never comes within "
                    f"{VALUE_CONVERGENCE_TOLERANCE:.0e} of the deployed function "
                    f"over the sweep; at {finest.terms} of "
                    f"{activation.deployed_terms} terms it still differs by "
                    f"{finest.value_max_deviation_from_deployed:.2e}, because for "
                    f"this activation the term count scales its non-oscillating "
                    f"part as well. Meanwhile {gradient_span}."
                )

        # Phase regimes.
        summary = ScaleStructureProbe.regime_summary(self.results.scale_rows)
        degenerate = []
        clean = []
        for activation in self.activations:
            entry = summary.get(activation.name)
            if not entry:
                continue
            locked = entry.get("lattice_locked", {}).get("n_scales", 0)
            noise = entry.get("rounding_dominated", {}).get("n_scales", 0)
            oscillating = entry.get("oscillating", {}).get("n_scales", 0)
            degenerate.append(
                f"{activation.display_name} ({oscillating} oscillating, "
                f"{locked} locked, {noise} rounding dominated)"
            )
            if not locked:
                clean.append(activation.display_name)
        if degenerate or clean:
            shares = []
            for activation in self.activations:
                entry = summary.get(activation.name)
                if not entry:
                    continue
                share = entry.get("lattice_locked", {}).get(
                    "derivative_amplitude_share", 0.0
                ) + entry.get("rounding_dominated", {}).get(
                    "derivative_amplitude_share", 0.0
                )
                shares.append((activation.display_name, float(share)))
            shares.sort(key=lambda item: item[1], reverse=True)
            out.append(
                "Measured scale by scale, the share of the derivative amplitude "
                "carried by scales that no longer oscillate is "
                + ", ".join(f"{name} {share:.3g}" for name, share in shares)
                + "."
            )
        if degenerate:
            out.append(
                "Counted per activation, the scales split as " +
                "; ".join(degenerate) + "."
            )
        if clean:
            out.append(
                "No lattice locking occurs for " + ", ".join(clean) +
                ", whose frequency ladder is not a power of two; its finest "
                "scales are dominated by the rounding of the phase product "
                "instead, at a derivative amplitude that is negligible."
            )

        # Realizability.
        primary_step = primary_realizability_step(self.results.realizability_rows)
        realizability = [
            r for r in self.results.realizability_rows if r.step_label == primary_step
        ]
        if realizability:
            realizability.sort(key=lambda r: self.order.index(r.activation))
            agree = [
                r
                for r in realizability
                if 0.5 <= r.median_ratio_analytic_to_realized <= 2.0
                and r.rank_correlation >= 0.9
            ]
            partial = [
                r
                for r in realizability
                if r not in agree and r.median_ratio_analytic_to_realized <= 5.0
            ]
            disagree = [
                r for r in realizability if r.median_ratio_analytic_to_realized > 5.0
            ]
            if agree:
                out.append(
                    f"Measured at a step of {primary_step}, the gradient "
                    "describes the function the network computes for "
                    + ", ".join(
                        f"{r.display_name} (ratio {r.median_ratio_analytic_to_realized:.2f}, "
                        f"rank correlation {r.rank_correlation:.3f})"
                        for r in agree
                    )
                    + "."
                )
            if partial:
                out.append(
                    "It describes it only loosely for "
                    + ", ".join(
                        f"{r.display_name} (ratio {r.median_ratio_analytic_to_realized:.2f}, "
                        f"rank correlation {r.rank_correlation:.3f})"
                        for r in partial
                    )
                    + ": the almost-everywhere derivative of their non-smooth "
                    "terms overstates the realized slope by about a factor of "
                    "two and tracks it only in part, which is a difference of "
                    "degree and not of kind."
                )
            if disagree:
                out.append(
                    "It does not describe it at all for "
                    + ", ".join(
                        f"{r.display_name} (ratio {r.median_ratio_analytic_to_realized:.2e}, "
                        f"rank correlation {r.rank_correlation:.3f})"
                        for r in disagree
                    )
                    + ", where the analytic derivative and the slope the "
                    "activation realizes over that step are unrelated quantities."
                )

        if self.results.network_rows:
            per_key: Dict[Tuple[str, Optional[int]], List[NetworkRow]] = {}
            for row in self.results.network_rows:
                per_key.setdefault((row.activation, row.terms), []).append(row)
            nonfinite_steps = sum(
                r.nonfinite_gradient_steps + r.nonfinite_loss_steps
                for r in self.results.network_rows
            )
            deployed_cells = {
                key: rows_
                for key, rows_ in per_key.items()
                if key[1] is None
                or key[1]
                == next(
                    (a.deployed_terms for a in self.activations if a.name == key[0]),
                    None,
                )
            }
            if deployed_cells:
                norms = {
                    key: float(np.mean([r.grad_norm_mean for r in rows_]))
                    for key, rows_ in deployed_cells.items()
                }
                lowest = min(norms, key=lambda k: norms[k])
                highest = max(norms, key=lambda k: norms[k])
                out.append(
                    "In the network probe the mean step gradient norm at the "
                    f"deployed settings ranges from {norms[lowest]:.3e} "
                    f"({self._display_name(lowest[0])}) to {norms[highest]:.3e} "
                    f"({self._display_name(highest[0])}), with {nonfinite_steps} "
                    "non-finite steps in total."
                )

        return out

    def _display_name(self, name: str) -> str:
        for activation in self.activations:
            if activation.name == name:
                return activation.display_name
        return name

    def write_report(self) -> Tuple[Path, str]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        text = self.build_report()
        path = OUTPUT_DIR / "activation_comparison_report.txt"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        return path, text


# =============================================================================
# 13) MAIN
# =============================================================================

def collect_environment() -> Dict[str, Any]:
    environment: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "activation_module_available": ACTIVATION_MODULE_AVAILABLE,
    }
    if TENSORFLOW_AVAILABLE:
        environment["tensorflow"] = tf.__version__
    else:
        environment["tensorflow_import_error"] = TENSORFLOW_IMPORT_ERROR
    if TENSORFLOW_AVAILABLE and not ACTIVATION_MODULE_AVAILABLE:
        environment["activation_module_import_error"] = ACTIVATION_MODULE_IMPORT_ERROR
    return environment


def main() -> None:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    activations = build_activation_registry()

    print("=" * 96)
    print("Numerical comparison of the activation functions of the study")
    print("=" * 96)
    for activation in activations:
        print(
            f"  {activation.display_name:>16} : {activation.deployed_parameter_text}"
            + (
                f", {activation.retained_scales()} scales, "
                f"ladder {activation.ladder_type()}"
                if activation.is_series()
                else ""
            )
        )
    if not TENSORFLOW_AVAILABLE:
        print(f"\nTensorFlow unavailable, numpy stages only: {TENSORFLOW_IMPORT_ERROR}")
    elif not ACTIVATION_MODULE_AVAILABLE:
        print(
            "\nfractal_activation_functions.py not importable, numpy stages only: "
            f"{ACTIVATION_MODULE_IMPORT_ERROR}"
        )

    results = StudyResults()
    results.environment = collect_environment()

    print("\nDrawing the pre-activation sample")
    sample = PreActivationSample()
    results.sample_summary = sample.summary()

    print("Stage 1: value and derivative at the deployed settings")
    deployed_stage = DeployedComparison(activations, sample)
    results.deployed_rows = deployed_stage.run()
    results.curves = deployed_stage.curves()

    if SCALE_STRUCTURE_ENABLED:
        print("Stage 3: phase of every scale of every fractal activation")
        results.scale_rows = ScaleStructureProbe(activations, sample).run()

    print("Stage 2: truncation sweep")
    results.truncation_rows = TruncationSweep(activations, sample).run(results.scale_rows)

    print("Stage 4: input precision probe")
    results.precision_rows = InputPrecisionProbe(activations, sample).run()

    if REALIZABILITY_ENABLED:
        print("Stage 5: gradient realizability")
        results.realizability_rows = GradientRealizability(activations, sample).run()

    # The two TensorFlow stages are guarded: a failure there must not discard
    # the stages that already finished. The reason is recorded and reported.
    crosscheck = AutodiffCrossCheck(activations, sample)
    if crosscheck.available():
        print("Cross-check against the deployed TensorFlow implementations")
        try:
            results.autodiff_rows = crosscheck.run()
        except Exception as exc:
            results.environment["autodiff_crosscheck_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    probe = NetworkGradientProbe(activations)
    if probe.available():
        print("Stage 6: compact network probe")
        try:
            results.network_rows = probe.run()
        except Exception as exc:
            results.environment["network_probe_error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    print("Plotting")
    ComparisonPlotter(activations, results).make_all()

    report = ComparisonReport(activations, results)
    report.write_tables()
    json_path = report.write_json()
    tex_path = report.write_latex_tables()
    report_path, text = report.write_report()

    print("")
    print(text)
    print("")
    print(f"Report:       {report_path}")
    print(f"JSON record:  {json_path}")
    print(f"LaTeX tables: {tex_path}")
    print(f"Tables:       {OUTPUT_DIR}")
    print(f"Plots:        {PLOT_DIR}")
    print(f"Elapsed:      {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
