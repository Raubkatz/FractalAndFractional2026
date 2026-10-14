#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_fractional_himmelblau_many_runs.py

Analysis script for the multi-run fractional Himmelblau optimizer comparison.

Purpose
-------
This script loads all saved optimizer runs produced by:

    fractional_himmelblau_optimizer_comparison_lr_vderiv_grid.py

or by a compatible script that writes the same result structure.

It reads the individual run results for each objective and optimizer, computes
aggregate statistics, writes machine-readable tables, and writes a human-readable
text report.

Expected input structure
------------------------
The expected results folder is:

    himmelblau_comparison_fin_std_add_3_lr_vderiv_grid/
        run_design.json
        all_results.json
        all_aggregate_statistics.json

        <objective_name>/
            all_run_results.json
            aggregate_statistics.json
            aggregate_summary.txt

            run_000_seed_<seed>/
                run_results.json
                summary.txt
                trajectories.png
                convergence.png
                gradient_norms.png
                fractional_orders.png

The script first tries to load:

    <RESULTS_ROOT>/<objective_name>/all_run_results.json

If that file is missing, it falls back to recursively loading:

    <RESULTS_ROOT>/<objective_name>/run_*/run_results.json

This makes the analysis robust if only the per-run result files are available.

Main analysis outputs
---------------------
The script creates:

    himmelblau_comparison_fin_std_add_3_lr_vderiv_grid_analysis/
        combined_individual_runs.csv
        combined_individual_runs.json
        combined_configuration_statistics.csv
        combined_configuration_statistics.json
        combined_configuration_rank_table.csv
        combined_optimizer_statistics.csv
        combined_optimizer_statistics.json
        combined_objective_winners.csv
        combined_rank_table.csv
        human_readable_report.txt

        <objective_name>/
            individual_runs.csv
            configuration_statistics.csv
            configuration_statistics.json
            configuration_rank_table.csv
            optimizer_statistics.csv
            optimizer_statistics.json
            rank_table.csv
            human_readable_report.txt
            plots/
                mean_final_loss_bar.png
                success_rate_bar.png
                mean_runtime_bar.png
                final_loss_boxplot.png
                runtime_boxplot.png
                mean_best_loss_bar.png
                mean_distance_bar.png

Statistics computed
-------------------
For each objective and optimizer--hyperparameter configuration, the script computes statistics over all runs:

    - final objective value:
        mean, standard deviation, standard error, 95% confidence interval,
        median, quartiles, minimum, maximum

    - best objective value reached during optimization:
        mean, standard deviation, standard error, 95% confidence interval,
        median, quartiles, minimum, maximum

    - distance to the nearest standard Himmelblau minimum:
        mean, standard deviation, standard error, 95% confidence interval,
        median, quartiles, minimum, maximum

    - success rate:
        number of successful runs, success rate,
        binomial standard error,
        approximate 95% confidence interval

    - runtime:
        mean, standard deviation, standard error, 95% confidence interval,
        median, quartiles, minimum, maximum

    - time per step:
        mean, standard deviation, standard error, 95% confidence interval,
        median, quartiles, minimum, maximum

    - number of steps:
        mean, standard deviation, minimum, maximum

Hyperparameter-grid strategy
----------------------------
The run script stores each learning-rate or fractional-order candidate as a
separate configuration_label. This analysis first aggregates every configuration
independently over the loaded runs, then selects the best-ranked configuration
of each base optimizer using the ranking rule below. The existing optimizer-level
tables, plots, winners, and reports therefore compare one selected configuration
per optimizer, while the additional configuration-level CSV/JSON files retain
the complete grid results.

Ranking rule
------------
The main ranking is deliberately conservative and interpretable:

    1. lowest mean_final_loss
    2. lowest mean_best_loss_reached
    3. highest success_rate
    4. lowest mean_distance_to_nearest_standard_himmelblau_minimum
    5. lowest mean_runtime_seconds

This ranking is written to the objective-specific rank tables and the combined
rank table. Lower rank is better.

Notes
-----
- The script does not rerun optimization.
- It only analyzes saved JSON files.
- Missing optional fields are handled with NaN values.
- Denominators are reported explicitly. Loss, distance and runtime means are
  taken over the runs that returned a finite value, and each mean is
  accompanied by the count that was used (<metric>_n) and by the number of
  loaded runs (<metric>_n_total). The success rate is taken over every loaded
  run, so a run that did not return a usable final value counts as a failure
  instead of being dropped. The columns n_runs, n_runs_valid and
  n_runs_invalid report the same information per optimizer, and both the
  objective reports and the combined report print them in a run-validity
  table.
- Runtime analysis is only available if the run script saved runtime_seconds.
- The reported distance and success criterion refer to the minima selected by
  SUCCESS_CRITERION. With the default "variant" they refer to the minima of the
  surface variant that was actually optimized, as resolved by the prior
  perturbation-and-minima stage, so on a perturbed variant success is measured
  on the perturbed objective rather than on the unperturbed one. Both criteria
  are always computed and written out side by side as
  variant_distance_*/variant_success_* and base_distance_*/base_success_*, and
  the objective reports and the combined report print them in a success-criterion
  table. The floor-referenced quantities gap_to_variant_floor and
  normalized_regret are aggregated as well.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 0) USER SETTINGS
# =============================================================================

RESULTS_ROOT = Path("himmelblau_comparison_fin_std_add_3_lr_vderiv_grid")
ANALYSIS_ROOT = Path("himmelblau_comparison_fin_std_add_3_lr_vderiv_grid_analysis")

SUCCESS_DISTANCE_THRESHOLD = 0.3
CONFIDENCE_LEVEL_Z = 1.96

# Which minima the reported distance and success criterion refer to.
#   "variant" : the minima of the surface variant that was actually optimized.
#               These are resolved by the prior perturbation-and-minima stage
#               and stored per run as distance_to_nearest_variant_minimum and
#               success_variant_minimum, so on a perturbed variant the
#               criterion is evaluated on the perturbed objective.
#   "base"    : the minima of the unperturbed Himmelblau surface, which is what
#               earlier versions of this script reported.
# Both criteria are always computed and written out as
# variant_distance_*/variant_success_* and base_distance_*/base_success_*.
# This setting only decides which of the two fills the headline columns
# distance_*/success_*, and therefore which one the ranking rule, the plots and
# the reports use. On a standard variant the two coincide, because there the
# resolved variant minima are the base minima.
SUCCESS_CRITERION = "variant"

if SUCCESS_CRITERION not in {"variant", "base"}:
    raise ValueError('SUCCESS_CRITERION must be "variant" or "base".')

BASE_DISTANCE_COLUMN = "distance_to_nearest_standard_himmelblau_minimum"
VARIANT_DISTANCE_COLUMN = "distance_to_nearest_variant_minimum"

SELECTED_DISTANCE_COLUMN = (
    VARIANT_DISTANCE_COLUMN if SUCCESS_CRITERION == "variant" else BASE_DISTANCE_COLUMN
)

DISTANCE_LABEL = (
    "nearest minimum of the optimized variant"
    if SUCCESS_CRITERION == "variant"
    else "nearest standard Himmelblau minimum"
)

CRITERION_SENTENCE = (
    "A higher success rate means the final point was closer to a minimum of the "
    "surface variant that was actually optimized more often."
    if SUCCESS_CRITERION == "variant"
    else "A higher success rate means the final point was closer to a known "
         "nearest standard Himmelblau minimum more often."
)

# Maximum number of saved runs to load per objective.
# Set to an integer (for example 50) to analyze only the first N saved runs.
# Set to None to load and analyze all available runs.
MAX_RUNS_TO_LOAD: Optional[int] = 100

# Plots
MAKE_PLOTS = True
PLOT_DPI = 200
FIGSIZE_BAR = (12, 7)
FIGSIZE_BOX = (12, 7)
BAR_LABEL_ROTATION = 75
PLOT_FONT_SCALE = 1.18

# Use log scale for loss plots when values are strongly spread out.
USE_LOG_SCALE_FOR_LOSS_PLOTS = True
USE_LOG_SCALE_FOR_RUNTIME_PLOTS = False

# Color palette
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

plt.rcParams.update({
    "font.size": 10 * PLOT_FONT_SCALE,
    "axes.titlesize": 13 * PLOT_FONT_SCALE,
    "axes.labelsize": 12 * PLOT_FONT_SCALE,
    "xtick.labelsize": 6 * PLOT_FONT_SCALE,
    "ytick.labelsize": 10 * PLOT_FONT_SCALE,
    "legend.fontsize": 10 * PLOT_FONT_SCALE,
})


# Plot-only abbreviations. These are used only for axis labels and tick labels
# in figures. Tables, CSV files, and reports keep the original optimizer names.
PLOT_OPTIMIZER_LABELS: Dict[str, str] = {
    "SGD": "SGD",
    "RMSprop": "RMSprop",
    "Adam": "Adam",
    "Adadelta": "Adadelta",
    "FSGD": "FSGD",
    "FRMSprop": "FRMSprop",
    "FAdam": "FAdam",
    "FAdadelta": "FAdadelta",
    "MemoryFSGD": "Mem.\nFSGD",
    "MemoryFRMSprop": "Mem.\nFRMSprop",
    "MemoryFAdam": "Mem.\nFAdam",
    "MemoryFAdadelta": "Mem.\nFAdadelta",
    "AdaptiveMemoryFSGD": "Adapt.\nMem.\nFSGD",
    "AdaptiveMemoryFRMSprop": "Adapt.\nMem.\nFRMSprop",
    "AdaptiveMemoryFAdam": "Adapt.\nMem.\nFAdam",
    "AdaptiveMemoryFAdadelta": "Adapt.\nMem.\nFAdadelta",
    "AdaGL": "AdaGL",
    "FCSGD_GL": "FCSGD\nGL",
    "FCAdam_GL": "FCAdam\nGL",
    "AOFGD_SGD": "AOFGD\nSGD",
    "AOFGD_Adam": "AOFGD\nAdam",
}


def plot_optimizer_label(name: Any) -> str:
    """Return the plot-only abbreviated optimizer label."""
    return PLOT_OPTIMIZER_LABELS.get(str(name), str(name))


def plot_optimizer_labels(names: Iterable[Any]) -> List[str]:
    """Return plot-only abbreviated optimizer labels."""
    return [plot_optimizer_label(name) for name in names]


# Optimizer groups and colours. These match the classification-analysis script
# exactly so that optimizer families have the same visual identity in all
# experiments and figures.
OPTIMIZER_GROUPS: Dict[str, str] = {
    # Group 1 — Standard baselines
    "SGD":                      "Standard",
    "Adam":                     "Standard",
    "RMSprop":                  "Standard",
    "Adadelta":                 "Standard",
    # Group 2 — Herrera-style fractional
    "FSGD":                     "Herrera",
    "FAdam":                    "Herrera",
    "FRMSprop":                 "Herrera",
    "FAdadelta":                "Herrera",
    # Group 3 — GL memory fractional
    "MemoryFSGD":               "GL-Memory",
    "MemoryFRMSprop":           "GL-Memory",
    "MemoryFAdam":              "GL-Memory",
    "MemoryFAdadelta":          "GL-Memory",
    # Group 4 — Adaptive memory fractional
    "AdaptiveMemoryFSGD":       "Adaptive-Memory",
    "AdaptiveMemoryFRMSprop":   "Adaptive-Memory",
    "AdaptiveMemoryFAdam":      "Adaptive-Memory",
    "AdaptiveMemoryFAdadelta":  "Adaptive-Memory",
    # Group 5 — Related-work
    "AdaGL":                    "Related-Work",
    "FCSGD_GL":                 "Related-Work",
    "FCAdam_GL":                "Related-Work",
    "AOFGD_SGD":                "Related-Work",
    "AOFGD_Adam":               "Related-Work",
}

GROUP_ORDER = ["Standard", "Herrera", "GL-Memory", "Adaptive-Memory", "Related-Work"]

GROUP_COLORS: Dict[str, str] = {
    "Standard":        "#2F7F74",
    "Herrera":         "#367DCD",
    "GL-Memory":       "#874BD2",
    "Adaptive-Memory": "#589641",
    "Related-Work":    "#DC6923",
}


def optimizer_group(name: Any) -> str:
    """Return the optimizer family used in the classification experiments."""
    return OPTIMIZER_GROUPS.get(str(name), "Unknown")


def bar_color_by_group(optimizer_names: Iterable[Any]) -> List[str]:
    """Return classification-consistent optimizer-group colours."""
    return [GROUP_COLORS.get(optimizer_group(name), "#888888") for name in optimizer_names]


def add_optimizer_group_legend(ax: plt.Axes, loc: str = "best") -> None:
    """Add the same five-family legend used in the classification analysis."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=GROUP_COLORS[group], label=group) for group in GROUP_ORDER]
    ax.legend(handles=handles, loc=loc, framealpha=0.85)


def format_parameter_for_plot(name: Any, value: Any) -> str:
    """Compact, explicit label for the swept optimizer-specific parameter."""
    name = str(name) if name is not None else ""
    try:
        value_float = float(value)
        value_text = f"{value_float:.4g}" if np.isfinite(value_float) else "nan"
    except Exception:
        value_text = str(value)

    parameter_names = {
        "learning_rate": "lr",
        "vderiv": "ν",
        "fractional_order": "ν",
        "alpha": "α",
        "alpha_init": "α_init",
    }
    return f"{parameter_names.get(name, name or 'param')}={value_text}"


def plot_configuration_label(row: Any) -> str:
    """Plot label containing both optimizer name and its actual swept setting."""
    optimizer = plot_optimizer_label(row.get("optimizer_name", ""))
    parameter = format_parameter_for_plot(
        row.get("swept_parameter_name", ""), row.get("swept_parameter_value", np.nan)
    )
    return f"{optimizer}\n{parameter}"


def mean_scope_text(df: pd.DataFrame, n_col: str = "n_runs") -> str:
    """Describe explicitly what a plotted mean averages over."""
    if n_col in df.columns:
        vals = pd.to_numeric(df[n_col], errors="coerce").dropna().astype(int).unique()
        if len(vals) == 1:
            return f"mean across {int(vals[0])} runs"
    return "mean across repeated runs"


# =============================================================================
# 1) BASIC UTILITIES
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def safe_bool(value: Any) -> bool:
    # A missing or non-finite flag must read as False, and a flag that was
    # round-tripped through a text file must be read by its content and not by
    # the truthiness of the string, because bool("False") is True.
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    if isinstance(value, float) and math.isnan(value):
        return False
    return bool(value)


def get_palette(n: int) -> List[str]:
    colors = []
    for i in range(n):
        colors.append(COLOR_PALETTE_A[i % len(COLOR_PALETTE_A)])
    return colors


def get_plot_filename_prefix(objective_name: Any) -> str:
    objective_name_str = str(objective_name)
    objective_name_lower = objective_name_str.lower()

    if "additive" in objective_name_lower and "himmelblau" in objective_name_lower:
        return "additive_himmelblau_"

    if "standard" in objective_name_lower and "himmelblau" in objective_name_lower:
        return "standard_himmelblau_"

    if objective_name_str:
        return f"{objective_name_str}_"

    return ""


def get_plot_title_prefix(objective_name: Any) -> str:
    objective_name_str = str(objective_name)
    objective_name_lower = objective_name_str.lower()

    if "additive" in objective_name_lower and "himmelblau" in objective_name_lower:
        return "Additive Himmelblau"

    if "standard" in objective_name_lower and "himmelblau" in objective_name_lower:
        return "Standard Himmelblau"

    if objective_name_str:
        return objective_name_str

    return "Himmelblau"


def save_figure_png_and_eps(fig: plt.Figure, out_path: Path) -> None:
    png_path = out_path.with_suffix(".png")
    eps_path = out_path.with_suffix(".eps")
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(eps_path, format="eps", dpi=PLOT_DPI, bbox_inches="tight")


# =============================================================================
# 2) RESULT LOADING
# =============================================================================

def find_objective_dirs(results_root: Path) -> List[Path]:
    objective_dirs = []

    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        has_all_results = (child / "all_run_results.json").exists()
        has_run_dirs = any(p.is_dir() and p.name.startswith("run_") for p in child.iterdir())
        if has_all_results or has_run_dirs:
            objective_dirs.append(child)

    return objective_dirs


def load_objective_results_from_all_file(objective_dir: Path) -> Optional[List[Dict[str, Any]]]:
    all_file = objective_dir / "all_run_results.json"
    if not all_file.exists():
        return None

    data = load_json(all_file)
    if isinstance(data, list):
        return data

    return None


def load_objective_results_from_run_files(objective_dir: Path) -> List[Dict[str, Any]]:
    results = []

    run_files = sorted(objective_dir.glob("run_*/run_results.json"))
    for run_file in run_files:
        data = load_json(run_file)
        if isinstance(data, list):
            results.extend(data)
        elif isinstance(data, dict):
            results.append(data)

    return results


def limit_results_to_max_runs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Limit loaded results to at most MAX_RUNS_TO_LOAD saved runs per objective.

    The limit is applied by base_run_seed/run_seed first so that repeated or
    restarted run_idx numbering cannot cause more than the requested number of
    actual repeated runs to be included. All optimizer/configuration results
    belonging to the same selected seeds are kept together. If seed information
    is unavailable, run_idx is used as a compatibility fallback.

    MAX_RUNS_TO_LOAD = None loads all available runs.
    """
    if MAX_RUNS_TO_LOAD is None:
        return rows

    if not isinstance(MAX_RUNS_TO_LOAD, int) or MAX_RUNS_TO_LOAD < 1:
        raise ValueError("MAX_RUNS_TO_LOAD must be None or an integer >= 1.")

    limited_rows: List[Dict[str, Any]] = []

    objective_names = []
    for row in rows:
        objective_name = str(row.get("objective_name", ""))
        if objective_name not in objective_names:
            objective_names.append(objective_name)

    for objective_name in objective_names:
        objective_rows = [
            row for row in rows
            if str(row.get("objective_name", "")) == objective_name
        ]

        run_seeds = sorted({
            safe_int(row.get("base_run_seed", row.get("run_seed", -1)), -1)
            for row in objective_rows
            if safe_int(row.get("base_run_seed", row.get("run_seed", -1)), -1) >= 0
        })

        if run_seeds:
            selected = set(run_seeds[:MAX_RUNS_TO_LOAD])
            limited_rows.extend(
                row for row in objective_rows
                if safe_int(row.get("base_run_seed", row.get("run_seed", -1)), -1) in selected
            )
            continue

        run_indices = sorted({
            safe_int(row.get("run_idx", -1), -1)
            for row in objective_rows
            if safe_int(row.get("run_idx", -1), -1) >= 0
        })

        if run_indices:
            selected = set(run_indices[:MAX_RUNS_TO_LOAD])
            limited_rows.extend(
                row for row in objective_rows
                if safe_int(row.get("run_idx", -1), -1) in selected
            )
            continue

        limited_rows.extend(objective_rows[:MAX_RUNS_TO_LOAD])

    return limited_rows


def load_all_results(results_root: Path) -> List[Dict[str, Any]]:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    all_results_file = results_root / "all_results.json"
    if all_results_file.exists():
        data = load_json(all_results_file)
        rows = []
        if isinstance(data, dict):
            for objective_name, objective_rows in data.items():
                if isinstance(objective_rows, list):
                    for row in objective_rows:
                        if isinstance(row, dict):
                            row = dict(row)
                            row.setdefault("objective_name", objective_name)
                            rows.append(row)
        if rows:
            return limit_results_to_max_runs(rows)

    rows = []
    objective_dirs = find_objective_dirs(results_root)
    for objective_dir in objective_dirs:
        objective_name = objective_dir.name

        objective_results = load_objective_results_from_all_file(objective_dir)
        if objective_results is None:
            objective_results = load_objective_results_from_run_files(objective_dir)

        for row in objective_results:
            if isinstance(row, dict):
                row = dict(row)
                row.setdefault("objective_name", objective_name)
                rows.append(row)

    return limit_results_to_max_runs(rows)


def normalize_results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    records = []

    for row in results:
        losses = row.get("losses", [])
        grad_norms = row.get("grad_norms", [])
        orders = row.get("orders", [])
        lr_factors = row.get("lr_factors", [])

        loss_array = np.array(losses, dtype=np.float64) if losses else np.array([], dtype=np.float64)
        grad_array = np.array(grad_norms, dtype=np.float64) if grad_norms else np.array([], dtype=np.float64)

        order_values = []
        for value in orders:
            if value is not None:
                order_values.append(safe_float(value))
        order_array = np.array(order_values, dtype=np.float64) if order_values else np.array([], dtype=np.float64)

        lr_factor_values = []
        for value in lr_factors:
            if value is not None:
                lr_factor_values.append(safe_float(value))
        lr_factor_array = np.array(lr_factor_values, dtype=np.float64) if lr_factor_values else np.array([], dtype=np.float64)

        trajectory = row.get("trajectory", [])
        final_point = row.get("final_point", [np.nan, np.nan])
        initial_point = row.get("initial_point", [np.nan, np.nan])

        final_x = safe_float(final_point[0]) if isinstance(final_point, list) and len(final_point) >= 1 else np.nan
        final_y = safe_float(final_point[1]) if isinstance(final_point, list) and len(final_point) >= 2 else np.nan
        initial_x = safe_float(initial_point[0]) if isinstance(initial_point, list) and len(initial_point) >= 1 else np.nan
        initial_y = safe_float(initial_point[1]) if isinstance(initial_point, list) and len(initial_point) >= 2 else np.nan

        runtime_seconds = safe_float(
            row.get("runtime_seconds", row.get("optimization_runtime_seconds", np.nan))
        )
        n_steps = safe_int(row.get("n_steps", len(losses)))
        mean_time_per_step = safe_float(row.get("mean_time_per_step_seconds", np.nan))
        if np.isnan(mean_time_per_step) and not np.isnan(runtime_seconds) and n_steps > 0:
            mean_time_per_step = runtime_seconds / n_steps

        record = {
            "objective_name": row.get("objective_name", ""),
            "optimizer_name": row.get("optimizer_name", ""),
            "configuration_label": row.get(
                "configuration_label", row.get("optimizer_name", "")
            ),
            "learning_rate": safe_float(row.get("learning_rate", np.nan)),
            "fractional_order": safe_float(row.get("fractional_order", np.nan)),
            "swept_parameter_name": row.get("swept_parameter_name", ""),
            "swept_parameter_value": safe_float(
                row.get("swept_parameter_value", np.nan)
            ),
            "configuration_index": safe_int(row.get("configuration_index", -1), -1),
            "run_idx": safe_int(row.get("run_idx", -1), -1),
            "base_run_seed": safe_int(row.get("base_run_seed", row.get("run_seed", -1)), -1),
            "optimizer_seed": safe_int(row.get("optimizer_seed", row.get("run_seed", -1)), -1),
            "run_seed": safe_int(row.get("run_seed", -1), -1),

            "initial_x": initial_x,
            "initial_y": initial_y,
            "final_x": final_x,
            "final_y": final_y,

            "final_loss": safe_float(row.get("final_loss", np.nan)),
            "best_loss_reached": safe_float(row.get("best_loss_reached", np.nan)),
            "first_loss": float(loss_array[0]) if loss_array.size > 0 else np.nan,
            "last_recorded_loss": float(loss_array[-1]) if loss_array.size > 0 else np.nan,

            "loss_improvement_absolute": np.nan,
            "loss_improvement_relative": np.nan,

            "distance_to_nearest_standard_himmelblau_minimum": safe_float(
                row.get("distance_to_nearest_standard_himmelblau_minimum", np.nan)
            ),
            "success_nearest_minimum": safe_bool(row.get("success_nearest_minimum", False)),

            # Minima of the variant that was actually optimized, resolved by the
            # prior perturbation-and-minima stage and stored per run.
            "distance_to_nearest_variant_minimum": safe_float(
                row.get("distance_to_nearest_variant_minimum", np.nan)
            ),
            "success_variant_minimum": safe_bool(row.get("success_variant_minimum", False)),
            "nearest_variant_minimum_index": safe_int(
                row.get("nearest_variant_minimum_index", -1), -1
            ),
            "variant_floor_value": safe_float(row.get("variant_floor_value", np.nan)),
            "gap_to_variant_floor": safe_float(row.get("gap_to_variant_floor", np.nan)),
            "best_gap_to_variant_floor": safe_float(
                row.get("best_gap_to_variant_floor", np.nan)
            ),
            "normalized_regret": safe_float(row.get("normalized_regret", np.nan)),
            "best_normalized_regret": safe_float(row.get("best_normalized_regret", np.nan)),
            "initial_loss": safe_float(row.get("initial_loss", np.nan)),

            "n_steps": n_steps,
            "runtime_seconds": runtime_seconds,
            "optimization_runtime_seconds": safe_float(row.get("optimization_runtime_seconds", runtime_seconds)),
            "mean_time_per_step_seconds": mean_time_per_step,

            "mean_gradient_norm": float(np.mean(grad_array)) if grad_array.size > 0 else np.nan,
            "std_gradient_norm": float(np.std(grad_array, ddof=0)) if grad_array.size > 0 else np.nan,
            "min_gradient_norm": float(np.min(grad_array)) if grad_array.size > 0 else np.nan,
            "max_gradient_norm": float(np.max(grad_array)) if grad_array.size > 0 else np.nan,
            "final_gradient_norm": float(grad_array[-1]) if grad_array.size > 0 else np.nan,

            "mean_fractional_order": float(np.mean(order_array)) if order_array.size > 0 else np.nan,
            "std_fractional_order": float(np.std(order_array, ddof=0)) if order_array.size > 0 else np.nan,
            "min_fractional_order": float(np.min(order_array)) if order_array.size > 0 else np.nan,
            "max_fractional_order": float(np.max(order_array)) if order_array.size > 0 else np.nan,
            "final_fractional_order": float(order_array[-1]) if order_array.size > 0 else np.nan,

            "mean_lr_factor": float(np.mean(lr_factor_array)) if lr_factor_array.size > 0 else np.nan,
            "final_lr_factor": float(lr_factor_array[-1]) if lr_factor_array.size > 0 else np.nan,

            "trajectory_length": len(trajectory),
            "loss_trace_length": len(losses),
            "gradient_trace_length": len(grad_norms),
            "order_trace_length": len(orders),
        }

        if not np.isnan(record["first_loss"]) and not np.isnan(record["final_loss"]):
            record["loss_improvement_absolute"] = record["first_loss"] - record["final_loss"]
            denom = max(abs(record["first_loss"]), 1e-12)
            record["loss_improvement_relative"] = record["loss_improvement_absolute"] / denom

        # Results written before the perturbed-minimum stage existed carry only
        # the base-surface distance. Fall back to it so that such files still
        # analyse, with the two criteria then identical by construction.
        if not np.isfinite(record[VARIANT_DISTANCE_COLUMN]):
            record[VARIANT_DISTANCE_COLUMN] = record[BASE_DISTANCE_COLUMN]
            record["success_variant_minimum"] = record["success_nearest_minimum"]

        # A run counts as valid only if it returned a finite final loss and a
        # finite distance under the reported criterion. An invalid run is kept in
        # the table and counted as a failure under both criteria, so that the
        # success rates and the loss means can be read against stated
        # denominators.
        record["run_is_valid"] = bool(
            np.isfinite(record["final_loss"])
            and np.isfinite(record[SELECTED_DISTANCE_COLUMN])
        )
        if not record["run_is_valid"]:
            record["success_nearest_minimum"] = False
            record["success_variant_minimum"] = False

        records.append(record)

    df = pd.DataFrame.from_records(records)
    return df


# =============================================================================
# 3) STATISTICS
# =============================================================================

def numeric_summary(values: Iterable[float], prefix: str) -> Dict[str, float]:
    # Non-finite entries are excluded from the mean, the standard deviation and
    # the confidence interval. Both the number of entries that were used and the
    # number that were loaded are reported, so the denominator of every mean is
    # explicit and can be compared with the denominator of the success rate.
    arr = np.array(list(values), dtype=np.float64)
    n_total = int(arr.size)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_n_total": n_total,
            f"{prefix}_n_missing": n_total,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_sem": np.nan,
            f"{prefix}_ci95_low": np.nan,
            f"{prefix}_ci95_high": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    sem = float(std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    ci_delta = CONFIDENCE_LEVEL_Z * sem

    return {
        f"{prefix}_n": int(arr.size),
        f"{prefix}_n_total": n_total,
        f"{prefix}_n_missing": n_total - int(arr.size),
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_sem": sem,
        f"{prefix}_ci95_low": mean - ci_delta,
        f"{prefix}_ci95_high": mean + ci_delta,
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q25": float(np.percentile(arr, 25)),
        f"{prefix}_q75": float(np.percentile(arr, 75)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def boolean_summary(values: Iterable[bool], prefix: str) -> Dict[str, float]:
    # The rate is taken over every loaded run, so a run that did not return a
    # usable final value counts as a failure rather than being dropped. The
    # denominator is reported as <prefix>_n.
    arr = np.array([safe_bool(value) for value in values], dtype=bool)

    if arr.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_count": 0,
            f"{prefix}_rate": np.nan,
            f"{prefix}_sem": np.nan,
            f"{prefix}_ci95_low": np.nan,
            f"{prefix}_ci95_high": np.nan,
        }

    rate = float(np.mean(arr))
    count = int(np.sum(arr))
    sem = math.sqrt(rate * (1.0 - rate) / arr.size) if arr.size > 0 else np.nan
    ci_delta = CONFIDENCE_LEVEL_Z * sem

    return {
        f"{prefix}_n": int(arr.size),
        f"{prefix}_count": count,
        f"{prefix}_rate": rate,
        f"{prefix}_sem": float(sem),
        f"{prefix}_ci95_low": float(max(0.0, rate - ci_delta)),
        f"{prefix}_ci95_high": float(min(1.0, rate + ci_delta)),
    }


def compute_optimizer_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = [
        "objective_name",
        "optimizer_name",
        "configuration_label",
        "learning_rate",
        "fractional_order",
        "swept_parameter_name",
        "swept_parameter_value",
    ]
    for group_key, group in df.groupby(group_cols, dropna=False):
        (
            objective_name,
            optimizer_name,
            configuration_label,
            learning_rate,
            fractional_order,
            swept_parameter_name,
            swept_parameter_value,
        ) = group_key

        row: Dict[str, Any] = {
            "objective_name": objective_name,
            "optimizer_name": optimizer_name,
            "configuration_label": configuration_label,
            "learning_rate": learning_rate,
            "fractional_order": fractional_order,
            "swept_parameter_name": swept_parameter_name,
            "swept_parameter_value": swept_parameter_value,
            "n_runs": int(len(group)),
        }

        # Runs that did not return a usable final value are counted here so that
        # the difference between the denominator of the loss means and the
        # denominator of the success rate is visible in every output table.
        if "run_is_valid" in group.columns:
            n_total_runs = int(len(group))
            n_valid_runs = int(np.sum(group["run_is_valid"].astype(bool).values))
            row["n_runs_valid"] = n_valid_runs
            row["n_runs_invalid"] = n_total_runs - n_valid_runs
            row["valid_run_rate"] = (
                float(n_valid_runs / n_total_runs) if n_total_runs > 0 else np.nan
            )

        metrics = [
            ("final_loss", "final_loss"),
            ("best_loss_reached", "best_loss"),
            (BASE_DISTANCE_COLUMN, "base_distance"),
            (VARIANT_DISTANCE_COLUMN, "variant_distance"),
            ("gap_to_variant_floor", "gap_to_variant_floor"),
            ("best_gap_to_variant_floor", "best_gap_to_variant_floor"),
            ("normalized_regret", "normalized_regret"),
            ("best_normalized_regret", "best_normalized_regret"),
            ("runtime_seconds", "runtime_seconds"),
            ("mean_time_per_step_seconds", "time_per_step_seconds"),
            ("n_steps", "steps"),
            ("loss_improvement_absolute", "loss_improvement_absolute"),
            ("loss_improvement_relative", "loss_improvement_relative"),
            ("mean_gradient_norm", "mean_gradient_norm"),
            ("final_gradient_norm", "final_gradient_norm"),
            ("mean_fractional_order", "mean_fractional_order"),
            ("final_fractional_order", "final_fractional_order"),
        ]

        for column, prefix in metrics:
            if column in group.columns:
                row.update(numeric_summary(group[column].values, prefix))

        if "success_nearest_minimum" in group.columns:
            row.update(boolean_summary(group["success_nearest_minimum"].values, "base_success"))

        if "success_variant_minimum" in group.columns:
            row.update(
                boolean_summary(group["success_variant_minimum"].values, "variant_success")
            )

        # The headline columns distance_* and success_* carry the criterion named
        # by SUCCESS_CRITERION, so the ranking rule, the plots and the reports all
        # follow the same choice while both criteria stay in the output.
        selected_prefix = "variant" if SUCCESS_CRITERION == "variant" else "base"
        for source, target in (
            (f"{selected_prefix}_distance", "distance"),
            (f"{selected_prefix}_success", "success"),
        ):
            for key in [k for k in row if k.startswith(source + "_")]:
                row[target + key[len(source):]] = row[key]
        row["success_criterion"] = SUCCESS_CRITERION

        rows.append(row)

    stats_df = pd.DataFrame(rows)
    stats_df = add_rank_columns(stats_df)
    return stats_df



def compute_pooled_statistics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Compute run-level pooled statistics without selecting one hyperparameter
    configuration first. This is deliberately separate from the best-configuration
    ranking so the two views cannot be confused.

    If grouped by optimizer, all saved runs from all swept settings of that
    optimizer are pooled. If grouped by optimizer_group, all optimizers,
    configurations, and runs in that family are pooled.
    """
    rows: List[Dict[str, Any]] = []

    for group_key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: Dict[str, Any] = dict(zip(group_cols, group_key))
        row["n_runs"] = int(len(group))
        if "configuration_label" in group.columns:
            row["n_configurations"] = int(group["configuration_label"].nunique(dropna=False))
        if "optimizer_name" in group.columns:
            row["n_optimizers"] = int(group["optimizer_name"].nunique(dropna=False))

        if "run_is_valid" in group.columns:
            n_total_runs = int(len(group))
            n_valid_runs = int(np.sum(group["run_is_valid"].astype(bool).values))
            row["n_runs_valid"] = n_valid_runs
            row["n_runs_invalid"] = n_total_runs - n_valid_runs
            row["valid_run_rate"] = (
                float(n_valid_runs / n_total_runs) if n_total_runs > 0 else np.nan
            )

        metrics = [
            ("final_loss", "final_loss"),
            ("best_loss_reached", "best_loss"),
            (BASE_DISTANCE_COLUMN, "base_distance"),
            (VARIANT_DISTANCE_COLUMN, "variant_distance"),
            ("gap_to_variant_floor", "gap_to_variant_floor"),
            ("best_gap_to_variant_floor", "best_gap_to_variant_floor"),
            ("normalized_regret", "normalized_regret"),
            ("best_normalized_regret", "best_normalized_regret"),
            ("runtime_seconds", "runtime_seconds"),
            ("mean_time_per_step_seconds", "time_per_step_seconds"),
            ("n_steps", "steps"),
            ("loss_improvement_absolute", "loss_improvement_absolute"),
            ("loss_improvement_relative", "loss_improvement_relative"),
            ("mean_gradient_norm", "mean_gradient_norm"),
            ("final_gradient_norm", "final_gradient_norm"),
            ("mean_fractional_order", "mean_fractional_order"),
            ("final_fractional_order", "final_fractional_order"),
        ]
        for column, prefix in metrics:
            if column in group.columns:
                row.update(numeric_summary(group[column].values, prefix))

        if "success_nearest_minimum" in group.columns:
            row.update(boolean_summary(group["success_nearest_minimum"].values, "base_success"))
        if "success_variant_minimum" in group.columns:
            row.update(boolean_summary(group["success_variant_minimum"].values, "variant_success"))

        selected_prefix = "variant" if SUCCESS_CRITERION == "variant" else "base"
        for source, target in (
            (f"{selected_prefix}_distance", "distance"),
            (f"{selected_prefix}_success", "success"),
        ):
            for key in [k for k in row if k.startswith(source + "_")]:
                row[target + key[len(source):]] = row[key]
        row["success_criterion"] = SUCCESS_CRITERION
        rows.append(row)

    return pd.DataFrame(rows)


def compute_pooled_optimizer_statistics(df: pd.DataFrame) -> pd.DataFrame:
    out = compute_pooled_statistics(df, ["objective_name", "optimizer_name"])
    if not out.empty:
        out["optimizer_group"] = out["optimizer_name"].map(optimizer_group)
    return out


def compute_pooled_optimizer_group_statistics(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["optimizer_group"] = tmp["optimizer_name"].map(optimizer_group)
    return compute_pooled_statistics(tmp, ["objective_name", "optimizer_group"])


def add_rank_columns(stats_df: pd.DataFrame) -> pd.DataFrame:
    ranked_frames = []

    for objective_name, group in stats_df.groupby("objective_name", dropna=False):
        group = group.copy()

        group["rank_final_loss"] = group["final_loss_mean"].rank(method="min", ascending=True)
        group["rank_best_loss"] = group["best_loss_mean"].rank(method="min", ascending=True)
        group["rank_success"] = group["success_rate"].rank(method="min", ascending=False)
        group["rank_distance"] = group["distance_mean"].rank(method="min", ascending=True)
        group["rank_runtime"] = group["runtime_seconds_mean"].rank(method="min", ascending=True)

        group["rank_score"] = (
            group["rank_final_loss"]
            + group["rank_best_loss"]
            + group["rank_success"]
            + group["rank_distance"]
            + group["rank_runtime"]
        )

        group = group.sort_values(
            by=[
                "final_loss_mean",
                "best_loss_mean",
                "success_rate",
                "distance_mean",
                "runtime_seconds_mean",
            ],
            ascending=[True, True, False, True, True],
        ).reset_index(drop=True)

        group["rank_overall"] = np.arange(1, len(group) + 1)
        ranked_frames.append(group)

    return pd.concat(ranked_frames, axis=0, ignore_index=True)


def select_best_configuration_per_optimizer(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best hyperparameter configuration of each optimizer independently
    for every objective, using exactly the same ranking rule as the original
    optimizer-level analysis.

    The configuration-level statistics are ranked first. The best-ranked
    configuration of each optimizer is retained, and the retained rows are then
    ranked again across optimizers so that all existing optimizer-level tables,
    plots, winners, and reports keep their original meaning.
    """
    selected_frames = []

    for objective_name, group in stats_df.groupby("objective_name", dropna=False):
        group = group.sort_values("rank_overall", ascending=True).copy()
        selected = group.drop_duplicates(subset=["optimizer_name"], keep="first").copy()

        # Re-rank the selected best configurations across the base optimizers.
        selected = add_rank_columns(selected)
        selected_frames.append(selected)

    if not selected_frames:
        return stats_df.iloc[0:0].copy()

    return pd.concat(selected_frames, axis=0, ignore_index=True)


def build_rank_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "objective_name",
        "rank_overall",
        "optimizer_name",
        "configuration_label",
        "learning_rate",
        "fractional_order",
        "swept_parameter_name",
        "swept_parameter_value",
        "n_runs",
        "n_runs_valid",
        "n_runs_invalid",
        "valid_run_rate",
        "final_loss_mean",
        "final_loss_n",
        "final_loss_std",
        "final_loss_sem",
        "final_loss_ci95_low",
        "final_loss_ci95_high",
        "best_loss_mean",
        "best_loss_n",
        "best_loss_std",
        "success_rate",
        "success_n",
        "success_count",
        "success_sem",
        "success_ci95_low",
        "success_ci95_high",
        "distance_mean",
        "distance_n",
        "distance_std",
        "success_criterion",
        "variant_success_rate",
        "variant_success_count",
        "variant_success_n",
        "variant_distance_mean",
        "variant_distance_n",
        "base_success_rate",
        "base_success_count",
        "base_success_n",
        "base_distance_mean",
        "base_distance_n",
        "gap_to_variant_floor_mean",
        "normalized_regret_mean",
        "runtime_seconds_mean",
        "runtime_seconds_std",
        "time_per_step_seconds_mean",
        "steps_mean",
    ]

    existing_columns = [col for col in columns if col in stats_df.columns]
    return stats_df[existing_columns].copy()


def build_objective_winners(stats_df: pd.DataFrame) -> pd.DataFrame:
    winners = []
    for objective_name, group in stats_df.groupby("objective_name", dropna=False):
        group = group.sort_values("rank_overall", ascending=True)
        if len(group) > 0:
            winners.append(group.iloc[0].to_dict())
    return pd.DataFrame(winners)


# =============================================================================
# 4) PLOTS
# =============================================================================

def save_bar_plot(
    df: pd.DataFrame,
    value_col: str,
    error_col: Optional[str],
    title: str,
    ylabel: str,
    out_path: Path,
    log_scale: bool = False,
) -> None:
    plot_df = df.sort_values("rank_overall", ascending=True).copy()
    colors = bar_color_by_group(plot_df["optimizer_name"])

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)

    x = np.arange(len(plot_df))
    values = plot_df[value_col].astype(float).values
    errors = plot_df[error_col].astype(float).values if error_col and error_col in plot_df.columns else None

    ax.bar(x, values, yerr=errors, capsize=4, color=colors, edgecolor="white", linewidth=0.5, zorder=3,
           error_kw={"zorder": 4})
    ax.set_xticks(x)
    ax.set_xticklabels(plot_optimizer_labels(plot_df["optimizer_name"].values), rotation=BAR_LABEL_ROTATION, ha="center")
    ax.set_title(f"{title} ({mean_scope_text(plot_df)}; best conf. per optimizer)")
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    add_optimizer_group_legend(ax)

    if log_scale:
        positive_values = values[values > 0]
        if positive_values.size > 0:
            ax.set_yscale("log")

    fig.tight_layout()
    save_figure_png_and_eps(fig, out_path)
    plt.close(fig)


def save_boxplot(
    df: pd.DataFrame,
    value_col: str,
    stats_df: pd.DataFrame,
    title: str,
    ylabel: str,
    out_path: Path,
    log_scale: bool = False,
) -> None:
    ordered_rows = stats_df.sort_values("rank_overall", ascending=True)
    data = []
    labels = []

    for _, stats_row in ordered_rows.iterrows():
        optimizer = stats_row["optimizer_name"]
        configuration_label = stats_row.get("configuration_label", optimizer)

        mask = df["optimizer_name"] == optimizer
        if "configuration_label" in df.columns:
            mask = mask & (df["configuration_label"] == configuration_label)

        values = df.loc[mask, value_col].dropna().astype(float).values
        if values.size > 0:
            data.append(values)
            labels.append(plot_configuration_label(stats_row))

    if not data:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_BOX)
    bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
    for patch, (_, stats_row) in zip(bp["boxes"], ordered_rows.iterrows()):
        patch.set_facecolor(GROUP_COLORS.get(optimizer_group(stats_row["optimizer_name"]), "#888888"))
        patch.set_alpha(0.85)
    ax.set_title(f"{title} (selected best conf. per optimizer over repeated runs)")
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", rotation=BAR_LABEL_ROTATION)
    add_optimizer_group_legend(ax)

    if log_scale:
        all_values = np.concatenate(data)
        if np.any(all_values > 0):
            ax.set_yscale("log")

    fig.tight_layout()
    save_figure_png_and_eps(fig, out_path)
    plt.close(fig)



def save_pooled_optimizer_bar_plot(
    pooled_df: pd.DataFrame,
    value_col: str,
    error_col: Optional[str],
    title: str,
    ylabel: str,
    out_path: Path,
    log_scale: bool = False,
) -> None:
    """Plot every optimizer after pooling all of its swept settings and runs."""
    if pooled_df.empty or value_col not in pooled_df.columns:
        return
    plot_df = pooled_df.sort_values(value_col, ascending=(value_col != "success_rate")).copy()
    colors = bar_color_by_group(plot_df["optimizer_name"])
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    x = np.arange(len(plot_df))
    values = plot_df[value_col].astype(float).values
    errors = plot_df[error_col].astype(float).values if error_col and error_col in plot_df.columns else None
    ax.bar(x, values, yerr=errors, capsize=4, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_optimizer_labels(plot_df["optimizer_name"]), rotation=BAR_LABEL_ROTATION, ha="center")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    add_optimizer_group_legend(ax)
    if log_scale:
        positive_values = values[values > 0]
        if positive_values.size > 0:
            ax.set_yscale("log")
    fig.tight_layout()
    save_figure_png_and_eps(fig, out_path)
    plt.close(fig)


def save_optimizer_group_bar_plot(
    group_df: pd.DataFrame,
    value_col: str,
    error_col: Optional[str],
    title: str,
    ylabel: str,
    out_path: Path,
    log_scale: bool = False,
) -> None:
    """Plot pooled optimizer-family statistics using classification colours."""
    if group_df.empty or value_col not in group_df.columns:
        return
    order_map = {name: i for i, name in enumerate(GROUP_ORDER)}
    plot_df = group_df.copy()
    plot_df["_order"] = plot_df["optimizer_group"].map(order_map).fillna(99)
    plot_df = plot_df.sort_values("_order")
    colors = [GROUP_COLORS.get(g, "#888888") for g in plot_df["optimizer_group"]]
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    x = np.arange(len(plot_df))
    values = plot_df[value_col].astype(float).values
    errors = plot_df[error_col].astype(float).values if error_col and error_col in plot_df.columns else None
    ax.bar(x, values, yerr=errors, capsize=4, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["optimizer_group"], rotation=25, ha="center")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    if log_scale:
        positive_values = values[values > 0]
        if positive_values.size > 0:
            ax.set_yscale("log")
    fig.tight_layout()
    save_figure_png_and_eps(fig, out_path)
    plt.close(fig)


def save_top20_configuration_bar_plot(
    configuration_stats: pd.DataFrame,
    value_col: str,
    error_col: Optional[str],
    title: str,
    ylabel: str,
    out_path: Path,
    log_scale: bool = False,
    top_n: int = 20,
) -> None:
    """Plot the top ranked configurations and show the actual swept parameter."""
    if configuration_stats.empty or value_col not in configuration_stats.columns:
        return
    plot_df = configuration_stats.sort_values("rank_overall", ascending=True).head(top_n).copy()
    plot_df["optimizer_group"] = plot_df["optimizer_name"].map(optimizer_group)
    colors = bar_color_by_group(plot_df["optimizer_name"])
    labels = [plot_configuration_label(row) for _, row in plot_df.iterrows()]
    fig, ax = plt.subplots(figsize=(FIGSIZE_BAR[0] * 1.15, FIGSIZE_BAR[1]))
    x = np.arange(len(plot_df))
    values = plot_df[value_col].astype(float).values
    errors = plot_df[error_col].astype(float).values if error_col and error_col in plot_df.columns else None
    ax.bar(x, values, yerr=errors, capsize=4, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=BAR_LABEL_ROTATION,
        ha="center",
        fontsize=8 * PLOT_FONT_SCALE,
    )
    ax.tick_params(axis="x", pad=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    add_optimizer_group_legend(ax)
    if log_scale:
        positive_values = values[values > 0]
        if positive_values.size > 0:
            ax.set_yscale("log")
    fig.tight_layout()
    save_figure_png_and_eps(fig, out_path)
    plt.close(fig)


def make_objective_plots(
    objective_df: pd.DataFrame,
    objective_stats: pd.DataFrame,
    objective_configuration_stats: pd.DataFrame,
    pooled_optimizer_stats: pd.DataFrame,
    pooled_group_stats: pd.DataFrame,
    plots_dir: Path,
) -> None:
    ensure_dir(plots_dir)

    objective_name = objective_stats["objective_name"].iloc[0] if len(objective_stats) > 0 else ""
    plot_prefix = get_plot_filename_prefix(objective_name)
    plot_title_prefix = get_plot_title_prefix(objective_name)

    save_bar_plot(
        objective_stats,
        value_col="final_loss_mean",
        error_col="final_loss_sem",
        title=f"{plot_title_prefix}: Mean final loss by optimizer",
        ylabel="Mean final loss",
        out_path=plots_dir / f"{plot_prefix}mean_final_loss_bar.png",
        log_scale=USE_LOG_SCALE_FOR_LOSS_PLOTS,
    )

    save_bar_plot(
        objective_stats,
        value_col="best_loss_mean",
        error_col="best_loss_sem",
        title=f"{plot_title_prefix}: Mean best loss reached by optimizer",
        ylabel="Mean best loss reached",
        out_path=plots_dir / f"{plot_prefix}mean_best_loss_bar.png",
        log_scale=USE_LOG_SCALE_FOR_LOSS_PLOTS,
    )

    save_bar_plot(
        objective_stats,
        value_col="success_rate",
        error_col="success_sem",
        title=f"{plot_title_prefix}: Success rate by optimizer",
        ylabel="Success rate",
        out_path=plots_dir / f"{plot_prefix}success_rate_bar.png",
        log_scale=False,
    )

    save_bar_plot(
        objective_stats,
        value_col="distance_mean",
        error_col="distance_sem",
        title=f"{plot_title_prefix}: Mean distance to {DISTANCE_LABEL}",
        ylabel="Mean distance",
        out_path=plots_dir / f"{plot_prefix}mean_distance_bar.png",
        log_scale=False,
    )

    save_bar_plot(
        objective_stats,
        value_col="runtime_seconds_mean",
        error_col="runtime_seconds_sem",
        title=f"{plot_title_prefix}: Mean runtime by optimizer",
        ylabel="Mean runtime [s]",
        out_path=plots_dir / f"{plot_prefix}mean_runtime_bar.png",
        log_scale=USE_LOG_SCALE_FOR_RUNTIME_PLOTS,
    )

    save_boxplot(
        objective_df,
        value_col="final_loss",
        stats_df=objective_stats,
        title=f"{plot_title_prefix}: Final loss distribution by optimizer",
        ylabel="Final loss",
        out_path=plots_dir / f"{plot_prefix}final_loss_boxplot.png",
        log_scale=USE_LOG_SCALE_FOR_LOSS_PLOTS,
    )

    save_boxplot(
        objective_df,
        value_col="runtime_seconds",
        stats_df=objective_stats,
        title=f"{plot_title_prefix}: Runtime distribution by optimizer",
        ylabel="Runtime [s]",
        out_path=plots_dir / f"{plot_prefix}runtime_boxplot.png",
        log_scale=USE_LOG_SCALE_FOR_RUNTIME_PLOTS,
    )


    # -------------------------------------------------------------------------
    # Additional transparent aggregation views.
    # These do NOT replace the original best-configuration-per-optimizer plots.
    # -------------------------------------------------------------------------
    pooled_scope = "all parameters and runs pooled"

    save_pooled_optimizer_bar_plot(
        pooled_optimizer_stats,
        value_col="final_loss_mean",
        error_col="final_loss_sem",
        title=f"{plot_title_prefix}: Mean final loss by optimizer ({pooled_scope})",
        ylabel="Pooled mean final loss",
        out_path=plots_dir / f"{plot_prefix}all_configurations_pooled_mean_final_loss_by_optimizer.png",
        log_scale=USE_LOG_SCALE_FOR_LOSS_PLOTS,
    )
    save_pooled_optimizer_bar_plot(
        pooled_optimizer_stats,
        value_col="success_rate",
        error_col="success_sem",
        title=f"{plot_title_prefix}: Success rate by optimizer ({pooled_scope})",
        ylabel="Pooled success rate",
        out_path=plots_dir / f"{plot_prefix}all_configurations_pooled_success_rate_by_optimizer.png",
        log_scale=False,
    )
    save_pooled_optimizer_bar_plot(
        pooled_optimizer_stats,
        value_col="runtime_seconds_mean",
        error_col="runtime_seconds_sem",
        title=f"{plot_title_prefix}: Mean runtime by optimizer ({pooled_scope})",
        ylabel="Pooled mean runtime [s]",
        out_path=plots_dir / f"{plot_prefix}all_configurations_pooled_mean_runtime_by_optimizer.png",
        log_scale=USE_LOG_SCALE_FOR_RUNTIME_PLOTS,
    )

    save_optimizer_group_bar_plot(
        pooled_group_stats,
        value_col="final_loss_mean",
        error_col="final_loss_sem",
        title=f"{plot_title_prefix}: Mean final loss by optimizer group",
        ylabel="Pooled mean final loss",
        out_path=plots_dir / f"{plot_prefix}optimizer_group_pooled_mean_final_loss.png",
        log_scale=USE_LOG_SCALE_FOR_LOSS_PLOTS,
    )
    save_optimizer_group_bar_plot(
        pooled_group_stats,
        value_col="success_rate",
        error_col="success_sem",
        title=f"{plot_title_prefix}: Success rate by optimizer group",
        ylabel="Pooled success rate",
        out_path=plots_dir / f"{plot_prefix}optimizer_group_pooled_success_rate.png",
        log_scale=False,
    )

    # Top 20 configuration-level views. The tick label explicitly states the
    # learning rate / fractional order / alpha / alpha_init of every bar.
    save_top20_configuration_bar_plot(
        objective_configuration_stats,
        value_col="final_loss_mean",
        error_col="final_loss_sem",
        title=f"{plot_title_prefix}: Top 20 configurations — mean final loss across runs",
        ylabel="Mean final loss",
        out_path=plots_dir / f"{plot_prefix}top20_configurations_mean_final_loss.png",
        log_scale=USE_LOG_SCALE_FOR_LOSS_PLOTS,
    )
    save_top20_configuration_bar_plot(
        objective_configuration_stats,
        value_col="best_loss_mean",
        error_col="best_loss_sem",
        title=f"{plot_title_prefix}: Top 20 configurations — mean best loss across runs",
        ylabel="Mean best loss reached",
        out_path=plots_dir / f"{plot_prefix}top20_configurations_mean_best_loss.png",
        log_scale=USE_LOG_SCALE_FOR_LOSS_PLOTS,
    )
    save_top20_configuration_bar_plot(
        objective_configuration_stats,
        value_col="success_rate",
        error_col="success_sem",
        title=f"{plot_title_prefix}: Top 20 configurations — success rate across runs",
        ylabel="Success rate",
        out_path=plots_dir / f"{plot_prefix}top20_configurations_success_rate.png",
        log_scale=False,
    )
    save_top20_configuration_bar_plot(
        objective_configuration_stats,
        value_col="distance_mean",
        error_col="distance_sem",
        title=f"{plot_title_prefix}: Top 20 configurations — mean distance across runs",
        ylabel="Mean distance",
        out_path=plots_dir / f"{plot_prefix}top20_configurations_mean_distance.png",
        log_scale=False,
    )
    save_top20_configuration_bar_plot(
        objective_configuration_stats,
        value_col="runtime_seconds_mean",
        error_col="runtime_seconds_sem",
        title=f"{plot_title_prefix}: Top 20 configurations — mean runtime across runs",
        ylabel="Mean runtime [s]",
        out_path=plots_dir / f"{plot_prefix}top20_configurations_mean_runtime.png",
        log_scale=USE_LOG_SCALE_FOR_RUNTIME_PLOTS,
    )


# =============================================================================
# 5) REPORTING
# =============================================================================

def format_float(value: Any, digits: int = 6) -> str:
    try:
        value = float(value)
        if np.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    except Exception:
        return "nan"


def dataframe_to_text_table(df: pd.DataFrame, columns: List[str], max_rows: Optional[int] = None) -> str:
    tmp = df[columns].copy()
    if max_rows is not None:
        tmp = tmp.head(max_rows)

    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: format_float(x, 6))

    return tmp.to_string(index=False)


def write_objective_report(
    objective_name: str,
    objective_df: pd.DataFrame,
    objective_stats: pd.DataFrame,
    objective_configuration_stats: pd.DataFrame,
    pooled_optimizer_stats: pd.DataFrame,
    pooled_group_stats: pd.DataFrame,
    rank_table: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = []
    lines.append("=" * 120)
    lines.append(f"Optimizer analysis report for objective: {objective_name}")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"Number of individual optimizer runs: {len(objective_df)}")
    lines.append(f"Number of optimizers: {objective_stats['optimizer_name'].nunique()}")
    if "configuration_label" in objective_df.columns:
        lines.append(
            f"Number of optimizer--hyperparameter configurations: "
            f"{objective_df['configuration_label'].nunique()}"
        )
    lines.append("")

    lines.append("Ranking rule:")
    lines.append("  1. lowest mean final loss")
    lines.append("  2. lowest mean best loss reached")
    lines.append("  3. highest success rate")
    lines.append(f"  4. lowest mean distance to {DISTANCE_LABEL}")
    lines.append("  5. lowest mean runtime")
    lines.append("")
    lines.append("Aggregation / averaging used in this report:")
    lines.append("  - Configuration statistics: each learning-rate/fractional-order setting is kept separate and each metric is averaged across repeated runs only.")
    lines.append("  - Best-configuration optimizer ranking: the best-ranked configuration of each base optimizer is selected; configurations are NOT averaged together for this ranking.")
    lines.append("  - Pooled optimizer tables/plots: all runs from all swept settings of one optimizer are pooled explicitly; these outputs are named 'all configurations pooled'.")
    lines.append("  - Optimizer-group tables/plots: all optimizers, settings, and runs of a family are pooled explicitly.")
    lines.append("")

    lines.append("Main ranking table:")
    lines.append("-" * 120)
    rank_cols = [
        "rank_overall",
        "optimizer_name",
        "configuration_label",
        "swept_parameter_name",
        "swept_parameter_value",
        "n_runs",
        "final_loss_mean",
        "final_loss_std",
        "best_loss_mean",
        "success_rate",
        "distance_mean",
        "runtime_seconds_mean",
        "time_per_step_seconds_mean",
    ]
    rank_cols = [c for c in rank_cols if c in rank_table.columns]
    lines.append(dataframe_to_text_table(rank_table, rank_cols))
    lines.append("")

    lines.append("Run validity and denominators:")
    lines.append("-" * 120)
    lines.append("Loss, distance and runtime means use only the runs that returned a finite value.")
    lines.append("The success rate uses every loaded run, so a run without a usable final value counts as a failure.")
    lines.append("n_runs is the number of loaded runs, n_runs_valid the number with a finite final loss and a finite distance.")
    lines.append("")
    validity_cols = [
        "rank_overall",
        "optimizer_name",
        "n_runs",
        "n_runs_valid",
        "n_runs_invalid",
        "final_loss_n",
        "distance_n",
        "runtime_seconds_n",
        "success_n",
        "success_count",
        "success_rate",
    ]
    validity_cols = [c for c in validity_cols if c in objective_stats.columns]
    lines.append(dataframe_to_text_table(objective_stats.sort_values("rank_overall"), validity_cols))
    lines.append("")

    lines.append("Success criterion:")
    lines.append("-" * 120)
    lines.append(f"Reported criterion: {SUCCESS_CRITERION}.")
    lines.append("variant_* refers to the minima of the surface variant that was actually optimized.")
    lines.append("base_* refers to the minima of the unperturbed surface.")
    lines.append("On a standard variant the two coincide, because there the variant minima are the base minima.")
    lines.append("")
    criterion_cols = [
        "rank_overall",
        "optimizer_name",
        "n_runs",
        "variant_success_rate",
        "variant_success_count",
        "base_success_rate",
        "base_success_count",
        "variant_distance_mean",
        "base_distance_mean",
        "gap_to_variant_floor_mean",
        "normalized_regret_mean",
    ]
    criterion_cols = [c for c in criterion_cols if c in objective_stats.columns]
    lines.append(dataframe_to_text_table(objective_stats.sort_values("rank_overall"), criterion_cols))
    lines.append("")

    lines.append("Detailed uncertainty table:")
    lines.append("-" * 120)
    uncertainty_cols = [
        "rank_overall",
        "optimizer_name",
        "final_loss_mean",
        "final_loss_sem",
        "final_loss_ci95_low",
        "final_loss_ci95_high",
        "best_loss_mean",
        "best_loss_sem",
        "success_rate",
        "success_sem",
        "success_ci95_low",
        "success_ci95_high",
        "runtime_seconds_mean",
        "runtime_seconds_sem",
        "runtime_seconds_ci95_low",
        "runtime_seconds_ci95_high",
    ]
    uncertainty_cols = [c for c in uncertainty_cols if c in objective_stats.columns]
    lines.append(dataframe_to_text_table(objective_stats.sort_values("rank_overall"), uncertainty_cols))
    lines.append("")

    lines.append("Best optimizer summary:")
    lines.append("-" * 120)
    best = objective_stats.sort_values("rank_overall").iloc[0]
    lines.append(f"Best optimizer: {best['optimizer_name']}")
    if "configuration_label" in best.index:
        lines.append(f"Best configuration: {best['configuration_label']}")
        lines.append(
            f"Swept parameter: {best.get('swept_parameter_name')} = "
            f"{format_float(best.get('swept_parameter_value'), 8)}"
        )
    lines.append(f"Mean final loss: {format_float(best.get('final_loss_mean'), 10)}")
    lines.append(f"Standard deviation final loss: {format_float(best.get('final_loss_std'), 10)}")
    lines.append(f"Mean best loss reached: {format_float(best.get('best_loss_mean'), 10)}")
    lines.append(f"Success rate: {format_float(best.get('success_rate'), 6)}")
    lines.append(f"Mean distance to {DISTANCE_LABEL}: {format_float(best.get('distance_mean'), 10)}")
    lines.append(f"Mean runtime seconds: {format_float(best.get('runtime_seconds_mean'), 10)}")
    lines.append("")

    lines.append("Full optimizer list — selected best configuration for every optimizer:")
    lines.append("-" * 120)
    full_optimizer_cols = [
        "rank_overall", "optimizer_name", "optimizer_group", "configuration_label",
        "learning_rate", "fractional_order", "swept_parameter_name", "swept_parameter_value",
        "n_runs", "final_loss_mean", "final_loss_std", "best_loss_mean",
        "success_rate", "distance_mean", "runtime_seconds_mean", "time_per_step_seconds_mean",
    ]
    detailed_optimizer_stats = objective_stats.copy()
    detailed_optimizer_stats["optimizer_group"] = detailed_optimizer_stats["optimizer_name"].map(optimizer_group)
    full_optimizer_cols = [c for c in full_optimizer_cols if c in detailed_optimizer_stats.columns]
    lines.append(dataframe_to_text_table(detailed_optimizer_stats.sort_values("rank_overall"), full_optimizer_cols))
    lines.append("")

    lines.append("Pooled optimizer overview — all parameter configurations and runs:")
    lines.append("-" * 120)
    pooled_optimizer_cols = [
        "optimizer_name", "optimizer_group", "n_configurations", "n_runs",
        "final_loss_mean", "final_loss_std", "best_loss_mean", "success_rate",
        "distance_mean", "runtime_seconds_mean", "time_per_step_seconds_mean",
    ]
    pooled_optimizer_cols = [c for c in pooled_optimizer_cols if c in pooled_optimizer_stats.columns]
    pooled_optimizer_sort = pooled_optimizer_stats.sort_values(["final_loss_mean", "best_loss_mean"], ascending=[True, True])
    lines.append(dataframe_to_text_table(pooled_optimizer_sort, pooled_optimizer_cols))
    lines.append("")

    lines.append("Optimizer-group overview — all optimizers, parameter configurations, and runs pooled:")
    lines.append("-" * 120)
    pooled_group_cols = [
        "optimizer_group", "n_optimizers", "n_configurations", "n_runs",
        "final_loss_mean", "final_loss_std", "best_loss_mean", "success_rate",
        "distance_mean", "runtime_seconds_mean", "time_per_step_seconds_mean",
    ]
    pooled_group_cols = [c for c in pooled_group_cols if c in pooled_group_stats.columns]
    group_order_map = {g: i for i, g in enumerate(GROUP_ORDER)}
    pooled_group_print = pooled_group_stats.copy()
    pooled_group_print["_group_order"] = pooled_group_print["optimizer_group"].map(group_order_map).fillna(99)
    pooled_group_print = pooled_group_print.sort_values("_group_order")
    lines.append(dataframe_to_text_table(pooled_group_print, pooled_group_cols))
    lines.append("")

    lines.append("Complete configuration list — every evaluated optimizer/parameter setting:")
    lines.append("-" * 120)
    complete_config = objective_configuration_stats.copy()
    complete_config["optimizer_group"] = complete_config["optimizer_name"].map(optimizer_group)
    config_cols = [
        "rank_overall", "optimizer_name", "optimizer_group", "configuration_label",
        "learning_rate", "fractional_order", "swept_parameter_name", "swept_parameter_value",
        "n_runs", "final_loss_mean", "final_loss_std", "best_loss_mean",
        "success_rate", "distance_mean", "runtime_seconds_mean", "time_per_step_seconds_mean",
    ]
    config_cols = [c for c in config_cols if c in complete_config.columns]
    lines.append(dataframe_to_text_table(complete_config.sort_values("rank_overall"), config_cols))
    lines.append("")

    lines.append("Interpretation notes:")
    lines.append("-" * 120)
    lines.append("A lower final loss means the optimizer ended in a better point on average.")
    lines.append("A lower best loss reached means the optimizer found a good point at least once during its trajectory.")
    lines.append(CRITERION_SENTENCE)
    lines.append("A lower runtime means the optimizer was faster for the same maximum number of steps.")
    lines.append("The runtime includes the TensorFlow gradient calculation, optimizer update, state extraction, and bookkeeping.")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_combined_report(
    full_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    configuration_stats_df: pd.DataFrame,
    pooled_optimizer_stats_df: pd.DataFrame,
    pooled_group_stats_df: pd.DataFrame,
    winners_df: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = []
    lines.append("=" * 120)
    lines.append("Combined optimizer analysis report")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"Results root: {RESULTS_ROOT}")
    lines.append(f"Analysis root: {ANALYSIS_ROOT}")
    lines.append(f"Total individual optimizer runs: {len(full_df)}")
    lines.append(f"Objectives: {full_df['objective_name'].nunique()}")
    lines.append(f"Optimizers: {full_df['optimizer_name'].nunique()}")
    if "configuration_label" in full_df.columns:
        lines.append(
            f"Optimizer--hyperparameter configurations: "
            f"{full_df['configuration_label'].nunique()}"
        )
    lines.append("")
    lines.append("Aggregation / averaging:")
    lines.append("  - Best-configuration tables select one configuration per optimizer after run-level aggregation; they do not average the four swept settings together.")
    lines.append("  - Outputs named 'pooled' explicitly average over all runs from all swept settings of the named optimizer or optimizer group.")
    lines.append("")

    lines.append("Objective winners:")
    lines.append("-" * 120)
    winner_cols = [
        "objective_name",
        "optimizer_name",
        "configuration_label",
        "swept_parameter_name",
        "swept_parameter_value",
        "n_runs",
        "final_loss_mean",
        "best_loss_mean",
        "success_rate",
        "distance_mean",
        "runtime_seconds_mean",
    ]
    winner_cols = [c for c in winner_cols if c in winners_df.columns]
    lines.append(dataframe_to_text_table(winners_df, winner_cols))
    lines.append("")

    lines.append("Combined ranking table:")
    lines.append("-" * 120)
    rank_cols = [
        "objective_name",
        "rank_overall",
        "optimizer_name",
        "configuration_label",
        "swept_parameter_name",
        "swept_parameter_value",
        "n_runs",
        "final_loss_mean",
        "final_loss_std",
        "best_loss_mean",
        "success_rate",
        "distance_mean",
        "runtime_seconds_mean",
        "time_per_step_seconds_mean",
    ]
    rank_cols = [c for c in rank_cols if c in stats_df.columns]
    lines.append(dataframe_to_text_table(stats_df.sort_values(["objective_name", "rank_overall"]), rank_cols))
    lines.append("")

    lines.append("Run validity and denominators:")
    lines.append("-" * 120)
    lines.append("Loss, distance and runtime means use only the runs that returned a finite value.")
    lines.append("The success rate uses every loaded run, so a run without a usable final value counts as a failure.")
    if "run_is_valid" in full_df.columns:
        n_total_runs = int(len(full_df))
        n_invalid_runs = int(np.sum(~full_df["run_is_valid"].astype(bool).values))
        lines.append(
            f"Loaded runs: {n_total_runs}. Runs without a usable final value: {n_invalid_runs}."
        )
    lines.append("")
    validity_cols = [
        "objective_name",
        "rank_overall",
        "optimizer_name",
        "n_runs",
        "n_runs_valid",
        "n_runs_invalid",
        "final_loss_n",
        "distance_n",
        "runtime_seconds_n",
        "success_n",
        "success_count",
        "success_rate",
    ]
    validity_cols = [c for c in validity_cols if c in stats_df.columns]
    lines.append(
        dataframe_to_text_table(
            stats_df.sort_values(["objective_name", "rank_overall"]), validity_cols
        )
    )
    lines.append("")

    lines.append("Success criterion:")
    lines.append("-" * 120)
    lines.append(f"Reported criterion: {SUCCESS_CRITERION}.")
    lines.append("variant_* refers to the minima of the surface variant that was actually optimized.")
    lines.append("base_* refers to the minima of the unperturbed surface.")
    lines.append("On a standard variant the two coincide, because there the variant minima are the base minima.")
    lines.append("")
    criterion_cols = [
        "objective_name",
        "rank_overall",
        "optimizer_name",
        "n_runs",
        "variant_success_rate",
        "variant_success_count",
        "base_success_rate",
        "base_success_count",
        "variant_distance_mean",
        "base_distance_mean",
        "gap_to_variant_floor_mean",
        "normalized_regret_mean",
    ]
    criterion_cols = [c for c in criterion_cols if c in stats_df.columns]
    lines.append(
        dataframe_to_text_table(
            stats_df.sort_values(["objective_name", "rank_overall"]), criterion_cols
        )
    )
    lines.append("")

    lines.append("Full selected optimizer list for every objective:")
    lines.append("-" * 120)
    selected_all = stats_df.copy()
    selected_all["optimizer_group"] = selected_all["optimizer_name"].map(optimizer_group)
    selected_cols = [
        "objective_name", "rank_overall", "optimizer_name", "optimizer_group",
        "configuration_label", "swept_parameter_name", "swept_parameter_value",
        "n_runs", "final_loss_mean", "best_loss_mean", "success_rate",
        "distance_mean", "runtime_seconds_mean",
    ]
    selected_cols = [c for c in selected_cols if c in selected_all.columns]
    lines.append(dataframe_to_text_table(selected_all.sort_values(["objective_name", "rank_overall"]), selected_cols))
    lines.append("")

    lines.append("Pooled optimizer overview for every objective — all parameter settings and runs:")
    lines.append("-" * 120)
    pooled_cols = [
        "objective_name", "optimizer_name", "optimizer_group", "n_configurations", "n_runs",
        "final_loss_mean", "best_loss_mean", "success_rate", "distance_mean", "runtime_seconds_mean",
    ]
    pooled_cols = [c for c in pooled_cols if c in pooled_optimizer_stats_df.columns]
    lines.append(dataframe_to_text_table(
        pooled_optimizer_stats_df.sort_values(["objective_name", "final_loss_mean"]), pooled_cols
    ))
    lines.append("")

    lines.append("Complete configuration list for every objective:")
    lines.append("-" * 120)
    all_configs = configuration_stats_df.copy()
    all_configs["optimizer_group"] = all_configs["optimizer_name"].map(optimizer_group)
    all_config_cols = [
        "objective_name", "rank_overall", "optimizer_name", "optimizer_group",
        "configuration_label", "learning_rate", "fractional_order",
        "swept_parameter_name", "swept_parameter_value", "n_runs",
        "final_loss_mean", "best_loss_mean", "success_rate", "distance_mean", "runtime_seconds_mean",
    ]
    all_config_cols = [c for c in all_config_cols if c in all_configs.columns]
    lines.append(dataframe_to_text_table(all_configs.sort_values(["objective_name", "rank_overall"]), all_config_cols))
    lines.append("")

    lines.append("Runtime interpretation:")
    lines.append("-" * 120)
    lines.append("Runtime is reported per optimizer trajectory and aggregated over runs.")
    lines.append("Use mean_time_per_step_seconds when runs stop after different numbers of steps.")
    lines.append("Use runtime_seconds_mean when all runs use the same maximum number of steps.")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# 6) MAIN
# =============================================================================

def main() -> None:
    ensure_dir(ANALYSIS_ROOT)

    print("=" * 100)
    print("Fractional Himmelblau optimizer analysis")
    print("=" * 100)
    print(f"Loading results from: {RESULTS_ROOT}")

    raw_results = load_all_results(RESULTS_ROOT)
    if not raw_results:
        raise RuntimeError(
            "No result rows were found. Check RESULTS_ROOT and the expected folder structure."
        )

    full_df = normalize_results_to_dataframe(raw_results)
    if full_df.empty:
        raise RuntimeError("Loaded results, but the normalized DataFrame is empty.")

    print(f"Loaded individual optimizer runs: {len(full_df)}")

    configuration_stats_df = compute_optimizer_statistics(full_df)
    configuration_rank_table = build_rank_table(configuration_stats_df)

    stats_df = select_best_configuration_per_optimizer(configuration_stats_df)
    stats_df["optimizer_group"] = stats_df["optimizer_name"].map(optimizer_group)
    configuration_stats_df["optimizer_group"] = configuration_stats_df["optimizer_name"].map(optimizer_group)
    rank_table = build_rank_table(stats_df)
    winners_df = build_objective_winners(stats_df)

    # Explicit pooled views. These are separate from the best-configuration ranking.
    pooled_optimizer_stats_df = compute_pooled_optimizer_statistics(full_df)
    pooled_group_stats_df = compute_pooled_optimizer_group_statistics(full_df)

    full_df.to_csv(ANALYSIS_ROOT / "combined_individual_runs.csv", index=False)
    full_df.to_json(ANALYSIS_ROOT / "combined_individual_runs.json", orient="records", indent=2)

    configuration_stats_df.to_csv(
        ANALYSIS_ROOT / "combined_configuration_statistics.csv", index=False
    )
    configuration_stats_df.to_json(
        ANALYSIS_ROOT / "combined_configuration_statistics.json",
        orient="records",
        indent=2,
    )
    configuration_rank_table.to_csv(
        ANALYSIS_ROOT / "combined_configuration_rank_table.csv", index=False
    )

    stats_df.to_csv(ANALYSIS_ROOT / "combined_optimizer_statistics.csv", index=False)
    stats_df.to_json(ANALYSIS_ROOT / "combined_optimizer_statistics.json", orient="records", indent=2)

    rank_table.to_csv(ANALYSIS_ROOT / "combined_rank_table.csv", index=False)
    winners_df.to_csv(ANALYSIS_ROOT / "combined_objective_winners.csv", index=False)
    pooled_optimizer_stats_df.to_csv(
        ANALYSIS_ROOT / "combined_optimizer_statistics_all_configurations_pooled.csv", index=False
    )
    pooled_group_stats_df.to_csv(
        ANALYSIS_ROOT / "combined_optimizer_group_statistics_all_configurations_pooled.csv", index=False
    )

    for objective_name, objective_df in full_df.groupby("objective_name", dropna=False):
        objective_dir = ANALYSIS_ROOT / str(objective_name)
        ensure_dir(objective_dir)

        objective_configuration_stats = configuration_stats_df.loc[
            configuration_stats_df["objective_name"] == objective_name
        ].copy()
        objective_configuration_rank_table = build_rank_table(
            objective_configuration_stats
        )

        objective_stats = stats_df.loc[stats_df["objective_name"] == objective_name].copy()
        objective_rank_table = build_rank_table(objective_stats)
        objective_pooled_optimizer_stats = pooled_optimizer_stats_df.loc[
            pooled_optimizer_stats_df["objective_name"] == objective_name
        ].copy()
        objective_pooled_group_stats = pooled_group_stats_df.loc[
            pooled_group_stats_df["objective_name"] == objective_name
        ].copy()

        objective_df.to_csv(objective_dir / "individual_runs.csv", index=False)
        objective_configuration_stats.to_csv(
            objective_dir / "configuration_statistics.csv", index=False
        )
        objective_configuration_rank_table.to_csv(
            objective_dir / "configuration_rank_table.csv", index=False
        )
        save_json(
            objective_configuration_stats.to_dict(orient="records"),
            objective_dir / "configuration_statistics.json",
        )
        objective_stats.to_csv(objective_dir / "optimizer_statistics.csv", index=False)
        objective_pooled_optimizer_stats.to_csv(
            objective_dir / "optimizer_statistics_all_configurations_pooled.csv", index=False
        )
        objective_pooled_group_stats.to_csv(
            objective_dir / "optimizer_group_statistics_all_configurations_pooled.csv", index=False
        )
        objective_rank_table.to_csv(objective_dir / "rank_table.csv", index=False)
        save_json(objective_stats.to_dict(orient="records"), objective_dir / "optimizer_statistics.json")

        write_objective_report(
            objective_name=str(objective_name),
            objective_df=objective_df,
            objective_stats=objective_stats,
            objective_configuration_stats=objective_configuration_stats,
            pooled_optimizer_stats=objective_pooled_optimizer_stats,
            pooled_group_stats=objective_pooled_group_stats,
            rank_table=objective_rank_table,
            out_path=objective_dir / "human_readable_report.txt",
        )

        if MAKE_PLOTS:
            make_objective_plots(
                objective_df=objective_df,
                objective_stats=objective_stats,
                objective_configuration_stats=objective_configuration_stats,
                pooled_optimizer_stats=objective_pooled_optimizer_stats,
                pooled_group_stats=objective_pooled_group_stats,
                plots_dir=objective_dir / "plots",
            )

    write_combined_report(
        full_df=full_df,
        stats_df=stats_df,
        configuration_stats_df=configuration_stats_df,
        pooled_optimizer_stats_df=pooled_optimizer_stats_df,
        pooled_group_stats_df=pooled_group_stats_df,
        winners_df=winners_df,
        out_path=ANALYSIS_ROOT / "human_readable_report.txt",
    )

    print("")
    print("Done.")
    print(f"Analysis output written to: {ANALYSIS_ROOT}")


if __name__ == "__main__":
    main()