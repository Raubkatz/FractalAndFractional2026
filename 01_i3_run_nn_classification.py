#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_fractional_optimizer_activation_experiments.py

Self-contained TensorFlow experiment runner for testing:
    - standard optimizers,
    - Herrera-style fractional optimizers,
    - explicit memory-based fractional optimizers,
across:
    - standard activations,
    - fractal activations from the uploaded activation module,

on classification datasets.

Main design goals
-----------------
1. No argparse. All settings are defined at the top of the file.
2. Similar result layout to the uploaded reporting script:
       <RESULTS_ROOT>/<dataset>/<config_name>.json
   where each JSON contains:
       dataset, optimizer, activation, vderivs, per-run metrics, etc.
3. Easy activation selection:
   fractal activations are all listed in one place so they can be commented out.
4. Includes training and testing wall-clock timing.
5. Uses a custom training loop so that:
   - all optimizers are handled consistently.
6. Intended for classification benchmarks and optimizer/activation interaction studies.

Why this structure
------------------
The uploaded summary script expects one JSON per configuration under
<root>/<dataset>/*.json, and then extracts:
    dataset, optimizer, activation, vderiv, runs, accuracy
from a "vderivs" list. This script writes results in that format so that the
uploaded summary script can be reused directly afterwards.

Uploaded sources used as design reference
-----------------------------------------
- The report script expects JSON files in <SOURCE_DIR>/<dataset>/*.json and
  computes summaries over fields including dataset, optimizer, activation,
  vderiv, and per-run accuracy results.
- The uploaded fractal activation module exposes several fractal activation
  functions such as modulated Blancmange, modified Weierstrass, and
  Weierstrass-Mandelbrot variants.
- The uploaded optimizer modules provide:
    * Herrera-style fractional optimizers (class_b),
    * memory-based fractional optimizers (class_c),
    * adaptive memory-based fractional optimizers with convex mixing (class_minimal),
    * related-work fractional optimizers: AdaGL, FCSGD_GL, FCAdam_GL,
      AOFGD_SGD, AOFGD_Adam (class_d).

Experimental questions this script supports
-------------------------------------------
1. Do fractional optimizers improve classification performance relative to
   standard baselines?
2. Do fractal activations interact differently with:
   - classical optimizers,
   - Herrera-style fractional optimizers,
   - explicit memory-based fractional optimizers,
   - adaptive memory-based fractional optimizers (class_minimal),
   - related-work fractional optimizers (AdaGL, FCSGD_GL, FCAdam_GL,
     AOFGD_SGD, AOFGD_Adam)?
3. What is the computational overhead in training and evaluation time?

Recommended workflow
--------------------
1. Edit the parameter block at the top.
2. Run the script.
3. Reuse the uploaded summary script on the results folder to create
   text reports of average accuracy, standard deviation, max, and min.

Notes
-----
- This script uses scikit-learn tabular datasets by default so it is
  immediately runnable without additional local data files.
- If you want to plug in your exact previous datasets, edit `load_dataset()`.
- The script uses one common MLP backbone to isolate optimizer / activation effects.
- For fairness, all compared optimizers share the same network architecture,
  seed protocol, split sizes, and training loop.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import sys

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import pandas as pd
from numpy.random import seed
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------------------------------------------------------------------
# IMPORT UPLOADED OPTIMIZER AND ACTIVATION MODULES
# ---------------------------------------------------------------------
# Adjust these import names if your local filenames differ.
# They are aligned with the uploaded files referenced in this conversation.
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

import fractal_activation_functions as faf


# =============================================================================
# 0) GLOBAL EXPERIMENT PARAMETERS
# =============================================================================

# -----------------------------
# Reproducibility
# -----------------------------
GLOBAL_SEED = 238974
SEED_VAL = 238974
RUNS = 40
RUN_SEEDS = list(range(SEED_VAL, SEED_VAL + RUNS))   # same run-number protocol as the older implementation

# -----------------------------
# Output folders
# -----------------------------
RESULTS_ROOT = Path(f"final_results_classificaiton_experiments_40runs")
HISTORY_ROOT = RESULTS_ROOT / "histories"
MODEL_ROOT = RESULTS_ROOT / "saved_models"  # optional; set SAVE_MODELS=False if not needed

SAVE_MODELS = False
SAVE_PER_RUN_HISTORY_JSON = True
SAVE_PER_RUN_PREDICTIONS = False

# -----------------------------
# Datasets to test
# -----------------------------
# Built-in tabular classification datasets for a self-contained experiment.
# Replace or extend this list in `load_dataset()` if you want your earlier datasets.
#
# Extended to the ten public OpenML classification datasets used in
# Raubitzek et al., "Fractals in neural networks: Introducing a new class
# of activation functions" (Commun Nonlinear Sci Numer Simulat, 2026),
# Table 1. Dataset-specific architecture/training hyperparameters (hidden
# width n, number of hidden layers, batch size, epochs) are defined in
# DATASET_HYPERPARAMS below and follow that table exactly.
#DATASETS_TO_RUN = [
#    "diabetes",
#    "climate-model-simulation-crashes",
#    "wine",
#    "glass",
#    "seeds",
#]

DATASETS_TO_RUN = [
    #"diabetes",
    #"climate-model-simulation-crashes",
    #"wine",
    #"glass",
    #"seeds",
    "ionosphere",
    "tic-tac-toe",
    "vertebra-column",
    "iris",
    "vehicle",
]

# -----------------------------
# Dataset-specific hyperparameters
# -----------------------------
# Mirrors Table 1 of Raubitzek et al. (2026): n is the width of the first
# hidden layer; multi-class networks use hidden widths (n, 2n) i.e. two
# hidden layers, while binary-classification networks use a single hidden
# layer of width n.
#
#   Dataset                            Task         n    Hidden layers  Batch  Epochs
#   climate-model-simulation-crashes   binary       32   1               32     30
#   diabetes                           binary       64   1               32     30
#   ionosphere                         binary       128  1               32     30
#   tic-tac-toe                        binary       64   1               32     25
#   vertebra-column                    binary       32   1               16     30
#   glass                              multi-class  64   2               32     30
#   iris                               multi-class  32   2               16     30
#   seeds                              multi-class  64   2               32     30
#   vehicle                            multi-class  128  2               32     25
#   wine                               multi-class  64   2               32     30
DATASET_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "climate-model-simulation-crashes": {"task": "binary",      "n": 32,  "hidden_layers": 1, "batch_size": 32, "epochs": 30},
    "diabetes":                          {"task": "binary",      "n": 64,  "hidden_layers": 1, "batch_size": 32, "epochs": 30},
    "ionosphere":                        {"task": "binary",      "n": 128, "hidden_layers": 1, "batch_size": 32, "epochs": 30},
    "tic-tac-toe":                       {"task": "binary",      "n": 64,  "hidden_layers": 1, "batch_size": 32, "epochs": 25},
    "vertebra-column":                   {"task": "binary",      "n": 32,  "hidden_layers": 1, "batch_size": 16, "epochs": 30},
    "glass":                             {"task": "multi-class", "n": 64,  "hidden_layers": 2, "batch_size": 32, "epochs": 30},
    "iris":                              {"task": "multi-class", "n": 32,  "hidden_layers": 2, "batch_size": 16, "epochs": 30},
    "seeds":                             {"task": "multi-class", "n": 64,  "hidden_layers": 2, "batch_size": 32, "epochs": 30},
    "vehicle":                           {"task": "multi-class", "n": 128, "hidden_layers": 2, "batch_size": 32, "epochs": 25},
    "wine":                              {"task": "multi-class", "n": 64,  "hidden_layers": 2, "batch_size": 32, "epochs": 30},
}


def get_dataset_hidden_units(dataset_name: str) -> List[int]:
    """
    Build the hidden-layer width list for a dataset according to
    DATASET_HYPERPARAMS, following Table 1 of Raubitzek et al. (2026):
    binary tasks use a single hidden layer of width n; multi-class tasks
    use two hidden layers of widths (n, 2n).
    """
    hp = DATASET_HYPERPARAMS.get(dataset_name)
    if hp is None:
        return list(HIDDEN_UNITS)  # fall back to the global default
    n = int(hp["n"])
    if int(hp["hidden_layers"]) == 1:
        return [n]
    return [n, 2 * n]


def get_dataset_batch_size(dataset_name: str) -> int:
    hp = DATASET_HYPERPARAMS.get(dataset_name)
    return int(hp["batch_size"]) if hp is not None else BATCH_SIZE


def get_dataset_epochs(dataset_name: str) -> int:
    hp = DATASET_HYPERPARAMS.get(dataset_name)
    return int(hp["epochs"]) if hp is not None else EPOCHS


# -----------------------------
# OpenML dataset registry
# -----------------------------
# Maps dataset key (as used in DATASETS_TO_RUN) -> (openml numeric id, description).
# IDs follow Raubitzek et al. (2026), Section 3, dataset descriptions.
_OPENML_REGISTRY: Dict[str, Tuple[int, str]] = {
    "climate-model-simulation-crashes": (1467, "climate-model-simulation-crashes"),
    "diabetes":                         (37,   "diabetes"),
    "glass":                            (41,   "glass"),
    "ionosphere":                       (59,   "ionosphere"),
    "iris":                             (61,   "iris"),
    "seeds":                            (1499, "seeds"),
    "tic-tac-toe":                      (50,   "tic-tac-toe"),
    "vehicle":                          (54,   "vehicle"),
    "vertebra-column":                  (1524, "vertebral-column-2clases"),
    "wine":                             (187,  "wine"),
}

# Cache for OpenML datasets to avoid re-downloading within the same run.
_OPENML_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}



def encode_non_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode non-numeric feature columns exactly in the style of the older
    activation-only implementation: each categorical/object value is mapped to
    idx / (number_of_unique_values - 1).
    """
    for column in df.select_dtypes(include=["object", "category"]).columns:
        unique_values = df[column].unique()
        if len(unique_values) <= 1:
            value_to_number = {value: 0.0 for value in unique_values}
        else:
            value_to_number = {
                value: idx / (len(unique_values) - 1)
                for idx, value in enumerate(unique_values)
            }
        df.loc[:, column] = df[column].map(value_to_number)
    return df


def split_scale(X: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split and scale exactly like the older implementation:
        - train_test_split(..., test_size=0.3, random_state=n, shuffle=True)
        - MinMaxScaler(feature_range=(0, 1))
        - no separate validation split
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train, X_test, y_train, y_test = dc(
        train_test_split(X, y, test_size=0.3, random_state=n, shuffle=True)
    )
    X_train = dc(scaler.fit_transform(X_train)).astype(np.float32)
    X_test = dc(scaler.transform(X_test)).astype(np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)
    return X_train, X_test, y_train, y_test


# -----------------------------
# Train/test split
# -----------------------------
TEST_SIZE = 0.30
VAL_SIZE_WITHIN_TRAIN = None  # kept only for result metadata compatibility; the older implementation has no validation split

# -----------------------------
# Training hyperparameters
# -----------------------------
EPOCHS = 40 #usually 40
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 8
SHUFFLE_EACH_EPOCH = True

# -----------------------------
# Network architecture
# -----------------------------
HIDDEN_UNITS = [128, 64]
DROPOUT_RATE = 0.10
USE_BATCH_NORM = False

# -----------------------------
# Activation selection
# -----------------------------
# Standard activations
STANDARD_ACTIVATIONS = {
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
    #"elu": tf.nn.elu,
    #"swish": tf.nn.swish,
}

# Fractal activations — ALL loaded exclusively from fractal_activation_functions (faf).
# Two categories:
#   1. Blancmange / Weierstrass-type (9 fixed-depth + 6 random-depth = 15 functions)
#   2. Cantor-type                    (3 functions: standard, set-indicator, random-depth)
# Comment out any entries you do not want to test.
FRACTAL_ACTIVATIONS = {

    # =========================================================================
    # Category 1 — Blancmange / Weierstrass-type (fixed depth)
    # =========================================================================
    "modulated_blancmange_curve":                           faf.modulated_blancmange_curve,
    "decaying_cosine_function_tf":                          faf.decaying_cosine_function_tf,
    "modified_weierstrass_function_tanh":                   faf.modified_weierstrass_function_tanh,
    #"modified_weierstrass_function_relu":                   faf.modified_weierstrass_function_relu,
    #"weierstrass_mandelbrot_function_xsinsquared":          faf.weierstrass_mandelbrot_function_xsinsquared,
    "weierstrass_mandelbrot_function_xpsin":                faf.weierstrass_mandelbrot_function_xpsin,
    #"weierstrass_mandelbrot_function_relupsin":             faf.weierstrass_mandelbrot_function_relupsin,
    #"weierstrass_mandelbrot_function_tanhpsin":             faf.weierstrass_mandelbrot_function_tanhpsin,
    #"weierstrass_function_tf":                              faf.weierstrass_function_tf,

    # Category 1 — Blancmange / Weierstrass-type (random depth)
    #"random_depth_modified_weierstrass_function_tanh":      faf.random_depth_modified_weierstrass_function_tanh,
    #"random_depth_decaying_cosine_function_tf":             faf.random_depth_decaying_cosine_function_tf,
    #"random_depth_modulated_blancmange_curve":              faf.random_depth_modulated_blancmange_curve,
    #"random_depth_weierstrass_mandelbrot_function_xpsin":   faf.random_depth_weierstrass_mandelbrot_function_xpsin,
    #"random_depth_weierstrass_mandelbrot_function_tanhpsin":faf.random_depth_weierstrass_mandelbrot_function_tanhpsin,
    #"random_depth_weierstrass_function_tf":                 faf.random_depth_weierstrass_function_tf,

    # =========================================================================
    # Category 2 — Cantor-type
    # =========================================================================
    # Standard Devil's Staircase: continuous, monotone, bounded in [0,1]
    #"cantor_function":                                      faf.cantor_function_tf,
    # Random-depth Devil's Staircase: stochastic resolution, same spirit as
    # the random_depth_* Weierstrass functions above
    #"random_depth_cantor_function":                         faf.random_depth_cantor_function_tf,
}


# Choose which activation families to include
INCLUDE_STANDARD_ACTIVATIONS = True
INCLUDE_FRACTAL_ACTIVATIONS = True

# -----------------------------
# Optimizer comparison design
# -----------------------------
# Baseline optimizers
BASELINE_OPTIMIZERS = [
    "SGD",
    "Adam",
    "RMSprop",
    #"Adagrad",
    "Adadelta",
]

# Herrera-style fractional optimizers
HERRERA_OPTIMIZERS = [
    "FSGD",
    "FAdam",
    "FRMSprop",
    #"FAdagrad",
    "FAdadelta",
    #"FAdamW",
]

# Fixed-order memory-based optimizers
MEMORY_OPTIMIZERS = [
    "MemoryFSGD",
    "MemoryFRMSprop",
    "MemoryFAdam",
    "MemoryFAdadelta",
]

INCLUDE_BASELINE_OPTIMIZERS = True
INCLUDE_HERRERA_OPTIMIZERS = True
INCLUDE_MEMORY_OPTIMIZERS = True
# Adaptive memory-based fractional optimizers (class_minimal)
ADAPTIVE_MEMORY_OPTIMIZERS = [
    "AdaptiveMemoryFSGD",
    "AdaptiveMemoryFRMSprop",
    "AdaptiveMemoryFAdam",
    "AdaptiveMemoryFAdadelta",
]

INCLUDE_ADAPTIVE_MEMORY_OPTIMIZERS = True

# Related-work fractional optimizers (class_d)
RELATED_WORK_OPTIMIZERS = [
    "AdaGL",
    "FCSGD_GL",
    "FCAdam_GL",
    "AOFGD_SGD",
    "AOFGD_Adam",
]

INCLUDE_RELATED_WORK_OPTIMIZERS = True

# -----------------------------
# Fractional parameter grids
# -----------------------------
# For standard non-fractional optimizers, vderiv is fixed to 1.0.
#VDERIV_GRID = [float(v) for v in np.round(np.arange(0.1, 2.0, 0.1), decimals=1)]
VDERIV_GRID = [0.75, 1.25, 1.5]
HERRERA_VDERIVS = VDERIV_GRID
MEMORY_VDERIVS = VDERIV_GRID
MEMORY_HISTORY_SIZES = [6]  # you can expand to [4, 6, 8, 10]
MEMORY_NORMALIZE_COEFFS = False


# -----------------------------
# Learning rates and optimizer-specific defaults
# -----------------------------
LR_SGD = 1e-2
LR_ADAM = 1e-3
LR_RMSPROP = 1e-3
LR_ADAGRAD = 1e-2
LR_ADADELTA = 1.0
LR_ADAMW = 1e-3

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

# Herrera-style optimizer extra parameter
HERRERA_FRAC_EPSILON = 1e-6

# -----------------------------
# Adaptive memory optimizer defaults (class_minimal)
# -----------------------------
MINIMAL_ADAPTIVE_VDERIV = 1.25
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

# -----------------------------
# Related-work optimizer defaults (class_d)
# -----------------------------
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

# -----------------------------
# Runtime controls
# -----------------------------
VERBOSE_TRAINING = 1
PRINT_PROGRESS = True
TF_FLOATX = "float32"

# =============================================================================
# 1) REPRODUCIBILITY
# =============================================================================

tf.keras.backend.set_floatx(TF_FLOATX)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =============================================================================
# 2) DATASETS
# =============================================================================

def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load datasets exactly in the style of the older activation-only script:
        fetch_openml(name=data_name, version=1, as_frame=True, parser="auto")
    followed by manual feature encoding and LabelEncoder target encoding.
    """
    if dataset_name not in _OPENML_CACHE:
        data_sk = fetch_openml(name=dataset_name, version=1, as_frame=True, parser="auto")
        X = data_sk.data
        y = data_sk.target

        if isinstance(X, pd.DataFrame):
            X = encode_non_numeric_features(X)

        if isinstance(y, pd.Series) and y.dtype == "object" or "category":
            y = LabelEncoder().fit_transform(y)
        else:
            y = y.to_numpy(dtype="int32")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        _OPENML_CACHE[dataset_name] = (X, y)
        print(
            f"  [OpenML] Loaded {dataset_name}: "
            f"X={X.shape}, classes={len(np.unique(y))}"
        )

    return _OPENML_CACHE[dataset_name]


def prepare_dataset(
    dataset_name: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Prepare the dataset according to the older implementation:
        - load OpenML by dataset name,
        - split once into train/test with test_size=0.3,
        - MinMax-scale features using the train split only,
        - reuse the test split as validation_data during model.fit().
    """
    X, y = dc(load_dataset(dataset_name))
    X_train, X_test, y_train, y_test = split_scale(X, y, seed)

    n_features = X_train.shape[1]
    n_classes = int(len(set(y.tolist() if hasattr(y, "tolist") else y)))

    return {
        "dataset_name": dataset_name,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_test,
        "y_val": y_test,
        "X_test": X_test,
        "y_test": y_test,
        "n_features": n_features,
        "n_classes": n_classes,
        "n_train": len(X_train),
        "n_val": len(X_test),
        "n_test": len(X_test),
    }


# =============================================================================
# 3) MODEL
# =============================================================================

def binary_model(
    n_features: int,
    n_classes: int,
    neurons: int,
    activation: Callable[[tf.Tensor], tf.Tensor],
) -> tf.keras.Model:
    """
    Binary-classification model from the older activation-only implementation.

    Architecture:
        Dense(neurons, activation=activation, GlorotUniform(seed=42), input_dim=n_features)
        Dense(n_classes, activation="sigmoid", GlorotUniform(seed=42))
    """
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                neurons,
                activation=activation,
                kernel_initializer=initializer,
                input_dim=n_features,
            ),
            tf.keras.layers.Dense(
                n_classes,
                activation="sigmoid",
                kernel_initializer=initializer,
            ),
        ]
    )
    return model


def groups_model(
    n_features: int,
    n_classes: int,
    neurons: int,
    activation: Callable[[tf.Tensor], tf.Tensor],
) -> tf.keras.Model:
    """
    Multi-class model from the older activation-only implementation.

    Architecture:
        Dense(neurons, activation=activation, GlorotUniform(seed=42), input_dim=n_features)
        BatchNormalization()
        Dropout(0.2)
        Dense(neurons*2, activation=activation, GlorotUniform(seed=42))
        BatchNormalization()
        Dropout(0.2)
        Dense(n_classes, activation="softmax", GlorotUniform(seed=42))
    """
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                neurons,
                activation=activation,
                kernel_initializer=initializer,
                input_dim=n_features,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                neurons * 2,
                activation=activation,
                kernel_initializer=initializer,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                n_classes,
                activation="softmax",
                kernel_initializer=initializer,
            ),
        ]
    )
    return model


def build_mlp(
    input_dim: int,
    n_classes: int,
    activation_fn: Callable[[tf.Tensor], tf.Tensor],
    activation_name: str,
    hidden_units: Optional[List[int]] = None,
) -> tf.keras.Model:
    """
    Compatibility wrapper used by the newer optimizer/activation pipeline.

    The actual architecture follows the older implementation exactly:
        - binary tasks: binary_model(...)
        - multi-class tasks: groups_model(...)
    The first value of hidden_units is interpreted as the old `neurons` value.
    """
    if hidden_units is None:
        hidden_units = HIDDEN_UNITS

    neurons = int(hidden_units[0])

    if n_classes == 2:
        return binary_model(input_dim, n_classes, neurons, activation_fn)

    if n_classes > 2:
        return groups_model(input_dim, n_classes, neurons, activation_fn)

    raise ValueError(f"Invalid number of classes: {n_classes}")


# =============================================================================
# 4) OPTIMIZER FACTORY
# =============================================================================

def build_optimizer(
    optimizer_name: str,
    vderiv: float = 1.0,
    history_size: int = 6,
    adaptation_mode: str = "ema_smoothed_gradient_variability",
    schedule_type: str = "cosine",
) -> tf.keras.optimizers.Optimizer:
    """
    Create a configured optimizer instance.
    """
    # -----------------------------
    # Standard baselines
    # -----------------------------
    if optimizer_name == "SGD":
        return tf.keras.optimizers.SGD(
            lr=LR_SGD,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
        )

    if optimizer_name == "Adam":
        return tf.keras.optimizers.Adam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
        )

    if optimizer_name == "RMSprop":
        return tf.keras.optimizers.RMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
        )

    if optimizer_name == "Adagrad":
        return tf.keras.optimizers.Adagrad(
            lr=LR_ADAGRAD,
            initial_accumulator_value=ADAGRAD_INITIAL_ACCUMULATOR,
            epsilon=ADAGRAD_EPSILON,
        )

    if optimizer_name == "Adadelta":
        return tf.keras.optimizers.Adadelta(
            lr=LR_ADADELTA,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
        )

    if optimizer_name == "AdamW":
        return tf.keras.optimizers.AdamW(
            lr=LR_ADAMW,
            weight_decay=ADAMW_WEIGHT_DECAY,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
        )

    # -----------------------------
    # Herrera-style fractional
    # -----------------------------
    if optimizer_name == "FSGD":
        return FSGD(
            lr=LR_SGD,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            vderiv=vderiv,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FAdam":
        return FAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv=vderiv,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FRMSprop":
        return FRMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
            vderiv=vderiv,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FAdagrad":
        return FAdagrad(
            lr=LR_ADAGRAD,
            initial_accumulator_value=ADAGRAD_INITIAL_ACCUMULATOR,
            epsilon=ADAGRAD_EPSILON,
            vderiv=vderiv,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FAdadelta":
        return FAdadelta(
            lr=LR_ADADELTA,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
            vderiv=vderiv,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FAdamW":
        return FAdamW(
            lr=LR_ADAMW,
            weight_decay=ADAMW_WEIGHT_DECAY,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv=vderiv,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    # -----------------------------
    # Memory-based fractional
    # -----------------------------
    if optimizer_name == "MemoryFSGD":
        return MemoryFSGD(
            lr=LR_SGD,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
        )

    if optimizer_name == "MemoryFRMSprop":
        return MemoryFRMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
        )

    if optimizer_name == "MemoryFAdam":
        return MemoryFAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
        )

    if optimizer_name == "MemoryFAdadelta":
        return MemoryFAdadelta(
            lr=LR_ADADELTA,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
        )



    # -----------------------------
    # Adaptive memory-based fractional (class_minimal)
    # -----------------------------
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

    # -----------------------------
    # Related-work optimizers (class_d)
    # -----------------------------
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
# 5) TRAINING UTILITIES
# =============================================================================

LOSS_FN = tf.keras.losses.SparseCategoricalCrossentropy()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Metric calculation from the older implementation.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return metrics


def evaluate_model(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Evaluation follows the older implementation:
        - model.evaluate(X_test, y_test)
        - model.predict(X_test)
        - np.argmax predictions
        - macro precision/recall/F1
    """
    eval_start = time.perf_counter()
    test_loss, test_acc = model.evaluate(X, y, batch_size=batch_size, verbose=VERBOSE_TRAINING)
    evaluate_elapsed = time.perf_counter() - eval_start

    predict_start = time.perf_counter()
    y_pred = model.predict(X, batch_size=batch_size, verbose=VERBOSE_TRAINING)
    predict_elapsed = time.perf_counter() - predict_start

    y_pred_classes = np.argmax(y_pred, axis=1)
    metrics = calculate_metrics(y, y_pred_classes)

    return {
        "loss": float(test_loss),
        "accuracy": float(test_acc),
        "f1_macro": float(metrics["f1_score"]),
        "precision_macro": float(metrics["precision"]),
        "recall_macro": float(metrics["recall"]),
        "test_time_seconds": float(predict_elapsed),
        "evaluate_time_seconds": float(evaluate_elapsed),
        "y_true": np.asarray(y),
        "y_pred": y_pred_classes,
    }


def train_one_run(
    dataset_bundle: Dict[str, Any],
    optimizer: tf.keras.optimizers.Optimizer,
    activation_fn: Callable[[tf.Tensor], tf.Tensor],
    activation_name: str,
    run_seed: int,
    hidden_units: Optional[List[int]] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train one run using the older activation-only training setup while keeping
    the newer optimizer/activation experiment pipeline.

    Older setup reproduced here:
        - tf.keras.backend.clear_session() before each run,
        - model built with fixed GlorotUniform(seed=42),
        - train/test split by run number,
        - no separate validation split,
        - validation_data=(X_test, y_test),
        - model.compile(..., loss="sparse_categorical_crossentropy", metrics=["accuracy"]),
        - model.fit(..., epochs=ep, batch_size=batch, validation_data=(X_test, y_test)),
        - no early stopping,
        - evaluate and predict on X_test.
    """
    tf.keras.backend.clear_session()

    if hidden_units is None:
        hidden_units = HIDDEN_UNITS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if epochs is None:
        epochs = EPOCHS

    X_train = dataset_bundle["X_train"]
    y_train = dataset_bundle["y_train"]
    X_test = dataset_bundle["X_test"]
    y_test = dataset_bundle["y_test"]
    n_features = dataset_bundle["n_features"]
    n_classes = dataset_bundle["n_classes"]

    model = build_mlp(
        input_dim=n_features,
        n_classes=n_classes,
        activation_fn=activation_fn,
        activation_name=activation_name,
        hidden_units=hidden_units,
    )

    print(f"Current Seed: {run_seed}")
    seed(run_seed)
    np.random.seed(run_seed)
    tf.random.set_seed(run_seed)

    compile_start = time.perf_counter()
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    compile_elapsed = time.perf_counter() - compile_start

    train_start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=VERBOSE_TRAINING,
    )
    train_elapsed = time.perf_counter() - train_start

    best_training_acc = float(max(history.history.get("accuracy", [0.0])))

    test_metrics = evaluate_model(model, X_test, y_test, batch_size)

    epoch_history: List[Dict[str, float]] = []
    n_epochs_completed = len(history.history.get("loss", []))
    for epoch_idx in range(n_epochs_completed):
        epoch_record = {
            "epoch": int(epoch_idx + 1),
            "train_loss": float(history.history["loss"][epoch_idx]),
            "train_accuracy": float(history.history["accuracy"][epoch_idx]),
            "val_loss": float(history.history["val_loss"][epoch_idx]),
            "val_accuracy": float(history.history["val_accuracy"][epoch_idx]),
        }
        epoch_history.append(epoch_record)

    result = {
        "epochs_completed": int(n_epochs_completed),
        "best_val_loss": float(min(history.history.get("val_loss", [np.inf]))),
        "loss": float(test_metrics["loss"]),
        "best_training_acc": float(best_training_acc),
        "compile_time_seconds": float(compile_elapsed),
        "training_time_seconds": float(train_elapsed),
        "test_time_seconds": float(test_metrics["test_time_seconds"]),
        "evaluate_time_seconds": float(test_metrics["evaluate_time_seconds"]),
        "accuracy": float(test_metrics["accuracy"]),
        "f1_macro": float(test_metrics["f1_macro"]),
        "precision_macro": float(test_metrics["precision_macro"]),
        "recall_macro": float(test_metrics["recall_macro"]),
        "epoch_history": epoch_history,
    }

    if SAVE_PER_RUN_PREDICTIONS:
        result["y_true"] = test_metrics["y_true"].tolist()
        result["y_pred"] = test_metrics["y_pred"].tolist()

    if SAVE_MODELS:
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        save_path = MODEL_ROOT / f"model_seed_{run_seed}_{int(time.time())}.keras"
        model.save(save_path)
        result["saved_model_path"] = str(save_path)

    tf.keras.backend.clear_session()
    os.system('pkill -f "tensorboard"')

    return result


# =============================================================================
# 6) CONFIGURATION ENUMERATION
# =============================================================================

def get_activation_map() -> Dict[str, Callable[[tf.Tensor], tf.Tensor]]:
    acts = {}
    if INCLUDE_STANDARD_ACTIVATIONS:
        acts.update(STANDARD_ACTIVATIONS)
    if INCLUDE_FRACTAL_ACTIVATIONS:
        acts.update(FRACTAL_ACTIVATIONS)
    return acts


def build_experiment_plan() -> List[Dict[str, Any]]:
    """
    Enumerate all experiment configurations.
    """
    configs: List[Dict[str, Any]] = []
    activation_map = get_activation_map()

    for activation_name in activation_map.keys():
        # Standard baselines
        if INCLUDE_BASELINE_OPTIMIZERS:
            for opt_name in BASELINE_OPTIMIZERS:
                configs.append(
                    {
                        "optimizer_name": opt_name,
                        "activation_name": activation_name,
                        "vderiv": 1.0,
                        "history_size": None,
                        "adaptation_mode": None,
                        "schedule_type": None,
                    }
                )

        # Herrera-style
        if INCLUDE_HERRERA_OPTIMIZERS:
            for opt_name in HERRERA_OPTIMIZERS:
                for vderiv in HERRERA_VDERIVS:
                    configs.append(
                        {
                            "optimizer_name": opt_name,
                            "activation_name": activation_name,
                            "vderiv": float(vderiv),
                            "history_size": None,
                            "adaptation_mode": None,
                            "schedule_type": None,
                        }
                    )

        # Memory-based
        if INCLUDE_MEMORY_OPTIMIZERS:
            for opt_name in MEMORY_OPTIMIZERS:
                for vderiv in MEMORY_VDERIVS:
                    for history_size in MEMORY_HISTORY_SIZES:
                        configs.append(
                            {
                                "optimizer_name": opt_name,
                                "activation_name": activation_name,
                                "vderiv": float(vderiv),
                                "history_size": int(history_size),
                                "adaptation_mode": None,
                                "schedule_type": None,
                            }
                        )

        # Adaptive memory-based fractional (class_minimal)
        # These use fixed internal hyperparameters; vderiv is a placeholder (ignored).
        if INCLUDE_ADAPTIVE_MEMORY_OPTIMIZERS:
            for opt_name in ADAPTIVE_MEMORY_OPTIMIZERS:
                configs.append(
                    {
                        "optimizer_name": opt_name,
                        "activation_name": activation_name,
                        "vderiv": 1.0,
                        "history_size": None,
                        "adaptation_mode": None,
                        "schedule_type": None,
                    }
                )

        # Related-work fractional optimizers (class_d)
        # These use fixed internal hyperparameters; vderiv is a placeholder (ignored).
        if INCLUDE_RELATED_WORK_OPTIMIZERS:
            for opt_name in RELATED_WORK_OPTIMIZERS:
                configs.append(
                    {
                        "optimizer_name": opt_name,
                        "activation_name": activation_name,
                        "vderiv": 1.0,
                        "history_size": None,
                        "adaptation_mode": None,
                        "schedule_type": None,
                    }
                )

    return configs


# =============================================================================
# 7) RESULT WRITING
# =============================================================================

def config_to_filename(cfg: Dict[str, Any]) -> str:
    parts = [
        cfg["optimizer_name"],
        cfg["activation_name"],
    ]
    if cfg["history_size"] is not None:
        parts.append(f"h{cfg['history_size']}")
    if cfg["adaptation_mode"] is not None:
        parts.append(cfg["adaptation_mode"])
    if cfg["schedule_type"] is not None:
        parts.append(cfg["schedule_type"])
    parts.append(f"v{str(cfg['vderiv']).replace('.', 'p')}")
    return "__".join(parts) + ".json"


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, float]:
    accs = [r["accuracy"] for r in runs]
    f1s = [r["f1_macro"] for r in runs]
    train_times = [r["training_time_seconds"] for r in runs]
    test_times = [r["test_time_seconds"] for r in runs]
    return {
        "avg_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs, ddof=0)),
        "max_accuracy": float(np.max(accs)),
        "min_accuracy": float(np.min(accs)),
        "avg_f1_macro": float(np.mean(f1s)),
        "avg_training_time_seconds": float(np.mean(train_times)),
        "avg_test_time_seconds": float(np.mean(test_times)),
    }


def write_result_json(
    dataset_name: str,
    cfg: Dict[str, Any],
    run_results: List[Dict[str, Any]],
) -> None:
    """
    Write one JSON file per configuration in a structure compatible with the
    uploaded report script.
    """
    out_dir = RESULTS_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summary = summarize_runs(run_results)

    # Keep the report-compatible schema:
    # data["vderivs"] -> list of objects with vderiv and results
    payload = {
        "dataset": dataset_name,
        "optimizer": cfg["optimizer_name"],
        "activation": cfg["activation_name"],
        "history_size": cfg["history_size"],
        "adaptation_mode": cfg["adaptation_mode"],
        "schedule_type": cfg["schedule_type"],
        "n_runs": len(run_results),
        "test runs": len(run_results),
        "vderivs": [
            {
                "vderiv": float(cfg["vderiv"]),
                "avg accuracy": run_summary["avg_accuracy"],
                "std accuracy": run_summary["std_accuracy"],
                "avg f1 macro": run_summary["avg_f1_macro"],
                "avg training time seconds": run_summary["avg_training_time_seconds"],
                "avg test time seconds": run_summary["avg_test_time_seconds"],
                "results": [
                    {
                        "run_seed": RUN_SEEDS[i],
                        "accuracy": float(r["accuracy"]),
                        "f1_macro": float(r["f1_macro"]),
                        "precision_macro": float(r["precision_macro"]),
                        "recall_macro": float(r["recall_macro"]),
                        "precision": float(r["precision_macro"]),
                        "recall": float(r["recall_macro"]),
                        "f1_score": float(r["f1_macro"]),
                        "loss": float(r.get("loss", np.nan)),
                        "best_training_acc": float(r.get("best_training_acc", np.nan)),
                        "training_time_seconds": float(r["training_time_seconds"]),
                        "test_time_seconds": float(r["test_time_seconds"]),
                        "training time": float(r["training_time_seconds"]),
                        "testing time": float(r["test_time_seconds"]),
                        "epochs_completed": int(r["epochs_completed"]),
                        "best_val_loss": float(r["best_val_loss"]),
                    }
                    for i, r in enumerate(run_results)
                ],
            }
        ],
    }

    if SAVE_PER_RUN_HISTORY_JSON:
        payload["run_histories"] = [
            {
                "run_seed": RUN_SEEDS[i],
                "epoch_history": r["epoch_history"],
            }
            for i, r in enumerate(run_results)
        ]

    out_file = out_dir / config_to_filename(cfg)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[SAVED] {out_file}")


# =============================================================================
# 8) MAIN EXPERIMENT LOOP
# =============================================================================

def run_experiments() -> None:
    set_global_seed(GLOBAL_SEED)

    activation_map = get_activation_map()
    experiment_plan = build_experiment_plan()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Fractional optimizer × fractal activation experiments")
    print("=" * 88)
    print(f"Datasets: {DATASETS_TO_RUN}")
    print(f"Activations: {list(activation_map.keys())}")
    print(f"Total configurations per dataset: {len(experiment_plan)}")
    print(f"Runs per configuration: {len(RUN_SEEDS)}")
    print("=" * 88)

    for dataset_name in DATASETS_TO_RUN:
        print(f"\n[DATASET] {dataset_name}")
        print("-" * 88)

        # Dataset-specific architecture / training hyperparameters
        # (Table 1, Raubitzek et al., 2026); falls back to the global
        # HIDDEN_UNITS / BATCH_SIZE / EPOCHS for any dataset not present
        # in DATASET_HYPERPARAMS.
        dataset_hidden_units = get_dataset_hidden_units(dataset_name)
        dataset_batch_size = get_dataset_batch_size(dataset_name)
        dataset_epochs = get_dataset_epochs(dataset_name)
        print(
            f"  hidden_units={dataset_hidden_units} "
            f"batch_size={dataset_batch_size} "
            f"epochs={dataset_epochs}"
        )

        for cfg_idx, cfg in enumerate(experiment_plan, start=1):
            print(
                f"\n[CONFIG {cfg_idx}/{len(experiment_plan)}] "
                f"optimizer={cfg['optimizer_name']} "
                f"activation={cfg['activation_name']} "
                f"vderiv={cfg['vderiv']} "
                f"history={cfg['history_size']} "
                f"mode={cfg['adaptation_mode']} "
                f"schedule={cfg['schedule_type']}"
            )

            run_results = []

            for run_seed in RUN_SEEDS:
                print(f" [RUN seed={run_seed}]")

                dataset_bundle = prepare_dataset(dataset_name, seed=run_seed)
                activation_fn = activation_map[cfg["activation_name"]]

                optimizer = build_optimizer(
                    optimizer_name=cfg["optimizer_name"],
                    vderiv=cfg["vderiv"],
                    history_size=cfg["history_size"] if cfg["history_size"] is not None else 6,
                    adaptation_mode=cfg["adaptation_mode"] if cfg["adaptation_mode"] else "ema_smoothed_gradient_variability",
                    schedule_type=cfg["schedule_type"] if cfg["schedule_type"] else "cosine",
                )

                result = train_one_run(
                    dataset_bundle=dataset_bundle,
                    optimizer=optimizer,
                    activation_fn=activation_fn,
                    activation_name=cfg["activation_name"],
                    run_seed=run_seed,
                    hidden_units=dataset_hidden_units,
                    batch_size=dataset_batch_size,
                    epochs=dataset_epochs,
                )

                run_results.append(result)

                print(
                    f"   accuracy={result['accuracy']:.5f} "
                    f"f1={result['f1_macro']:.5f} "
                    f"train_time={result['training_time_seconds']:.3f}s "
                    f"test_time={result['test_time_seconds']:.3f}s "
                    f"epochs={result['epochs_completed']}"
                )

            write_result_json(dataset_name, cfg, run_results)

    print("\nDone.")


# =============================================================================
# 9) OPTIONAL: EXPERIMENTAL DESIGN NOTES
# =============================================================================

def print_experimental_design_notes() -> None:
    """
    Print a compact description of the intended comparisons.
    """
    print("\nExperimental design:")
    print("1. Baseline comparison:")
    print("   Compare standard optimizers with Herrera-style and memory-based optimizers.")
    print("2. Activation comparison:")
    print("   Compare standard activations against fractal activations under the same optimizer conditions.")
    print("3. Interaction study:")
    print("   Identify whether certain optimizer families work better with certain fractal activations.")
    print("4. Cost study:")
    print("   Compare average training and test time overheads.")
    print("5. Dataset suite:")
    print("   Ten public OpenML classification datasets following Table 1 of")
    print("   Raubitzek et al. (2026), each with its own architecture width,")
    print("   number of hidden layers, batch size, and epoch count.")


# =============================================================================
# 10) SCRIPT ENTRY
# =============================================================================

if __name__ == "__main__":
    print_experimental_design_notes()
    run_experiments()