#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_fractional_himmelblau_many_runs.py

Analysis script for the multi-run fractional Himmelblau optimizer comparison.

Purpose
-------
This script loads all saved optimizer runs produced by:

    fractional_himmelblau_optimizer_comparison_many_runs_with_time.py

or by a compatible script that writes the same result structure.

It reads the individual run results for each objective and optimizer, computes
aggregate statistics, writes machine-readable tables, and writes a human-readable
text report.

Expected input structure
------------------------
The expected results folder is:

    fractional_himmelblau_optimizer_comparison_many_runs/
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

    fractional_himmelblau_optimizer_analysis/
        combined_individual_runs.csv
        combined_individual_runs.json
        combined_optimizer_statistics.csv
        combined_optimizer_statistics.json
        combined_objective_winners.csv
        combined_rank_table.csv
        human_readable_report.txt

        <objective_name>/
            individual_runs.csv
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
For each objective and optimizer, the script computes statistics over all runs:

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
- Runtime analysis is only available if the run script saved runtime_seconds.
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

RESULTS_ROOT = Path("himmelblau_comparison_fin_std_add")
ANALYSIS_ROOT = Path("himmelblau_comparison_fin_std_add_analysis")

SUCCESS_DISTANCE_THRESHOLD = 0.25
CONFIDENCE_LEVEL_Z = 1.96

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
    "xtick.labelsize": 10 * PLOT_FONT_SCALE,
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
            return rows

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

    return rows


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

        records.append(record)

    df = pd.DataFrame.from_records(records)
    return df


# =============================================================================
# 3) STATISTICS
# =============================================================================

def numeric_summary(values: Iterable[float], prefix: str) -> Dict[str, float]:
    arr = np.array(list(values), dtype=np.float64)
    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        return {
            f"{prefix}_n": 0,
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
    arr = np.array(list(values), dtype=bool)

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

    group_cols = ["objective_name", "optimizer_name"]
    for (objective_name, optimizer_name), group in df.groupby(group_cols, dropna=False):
        row: Dict[str, Any] = {
            "objective_name": objective_name,
            "optimizer_name": optimizer_name,
            "n_runs": int(len(group)),
        }

        metrics = [
            ("final_loss", "final_loss"),
            ("best_loss_reached", "best_loss"),
            ("distance_to_nearest_standard_himmelblau_minimum", "distance"),
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
            row.update(boolean_summary(group["success_nearest_minimum"].values, "success"))

        rows.append(row)

    stats_df = pd.DataFrame(rows)
    stats_df = add_rank_columns(stats_df)
    return stats_df


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


def build_rank_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "objective_name",
        "rank_overall",
        "optimizer_name",
        "n_runs",
        "final_loss_mean",
        "final_loss_std",
        "final_loss_sem",
        "final_loss_ci95_low",
        "final_loss_ci95_high",
        "best_loss_mean",
        "best_loss_std",
        "success_rate",
        "success_sem",
        "success_ci95_low",
        "success_ci95_high",
        "distance_mean",
        "distance_std",
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
    colors = get_palette(len(plot_df))

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)

    x = np.arange(len(plot_df))
    values = plot_df[value_col].astype(float).values
    errors = plot_df[error_col].astype(float).values if error_col and error_col in plot_df.columns else None

    ax.bar(x, values, yerr=errors, capsize=4, color=colors, zorder=3,
           error_kw={"zorder": 4})
    ax.set_xticks(x)
    ax.set_xticklabels(plot_optimizer_labels(plot_df["optimizer_name"].values), rotation=BAR_LABEL_ROTATION, ha="center")
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


def save_boxplot(
    df: pd.DataFrame,
    value_col: str,
    stats_df: pd.DataFrame,
    title: str,
    ylabel: str,
    out_path: Path,
    log_scale: bool = False,
) -> None:
    ordered_optimizers = stats_df.sort_values("rank_overall", ascending=True)["optimizer_name"].tolist()
    data = []
    labels = []

    for optimizer in ordered_optimizers:
        values = df.loc[df["optimizer_name"] == optimizer, value_col].dropna().astype(float).values
        if values.size > 0:
            data.append(values)
            labels.append(plot_optimizer_label(optimizer))

    if not data:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_BOX)
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", rotation=BAR_LABEL_ROTATION)

    if log_scale:
        all_values = np.concatenate(data)
        if np.any(all_values > 0):
            ax.set_yscale("log")

    fig.tight_layout()
    save_figure_png_and_eps(fig, out_path)
    plt.close(fig)


def make_objective_plots(objective_df: pd.DataFrame, objective_stats: pd.DataFrame, plots_dir: Path) -> None:
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
        title=f"{plot_title_prefix}: Mean distance to nearest standard Himmelblau minimum",
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
    lines.append("")

    lines.append("Ranking rule:")
    lines.append("  1. lowest mean final loss")
    lines.append("  2. lowest mean best loss reached")
    lines.append("  3. highest success rate")
    lines.append("  4. lowest mean distance to nearest standard Himmelblau minimum")
    lines.append("  5. lowest mean runtime")
    lines.append("")

    lines.append("Main ranking table:")
    lines.append("-" * 120)
    rank_cols = [
        "rank_overall",
        "optimizer_name",
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
    lines.append(f"Mean final loss: {format_float(best.get('final_loss_mean'), 10)}")
    lines.append(f"Standard deviation final loss: {format_float(best.get('final_loss_std'), 10)}")
    lines.append(f"Mean best loss reached: {format_float(best.get('best_loss_mean'), 10)}")
    lines.append(f"Success rate: {format_float(best.get('success_rate'), 6)}")
    lines.append(f"Mean distance to nearest minimum: {format_float(best.get('distance_mean'), 10)}")
    lines.append(f"Mean runtime seconds: {format_float(best.get('runtime_seconds_mean'), 10)}")
    lines.append("")

    lines.append("Interpretation notes:")
    lines.append("-" * 120)
    lines.append("A lower final loss means the optimizer ended in a better point on average.")
    lines.append("A lower best loss reached means the optimizer found a good point at least once during its trajectory.")
    lines.append("A higher success rate means the final point was closer to a known standard Himmelblau minimum more often.")
    lines.append("A lower runtime means the optimizer was faster for the same maximum number of steps.")
    lines.append("The runtime includes the TensorFlow gradient calculation, optimizer update, state extraction, and bookkeeping.")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_combined_report(
    full_df: pd.DataFrame,
    stats_df: pd.DataFrame,
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
    lines.append("")

    lines.append("Objective winners:")
    lines.append("-" * 120)
    winner_cols = [
        "objective_name",
        "optimizer_name",
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

    stats_df = compute_optimizer_statistics(full_df)
    rank_table = build_rank_table(stats_df)
    winners_df = build_objective_winners(stats_df)

    full_df.to_csv(ANALYSIS_ROOT / "combined_individual_runs.csv", index=False)
    full_df.to_json(ANALYSIS_ROOT / "combined_individual_runs.json", orient="records", indent=2)

    stats_df.to_csv(ANALYSIS_ROOT / "combined_optimizer_statistics.csv", index=False)
    stats_df.to_json(ANALYSIS_ROOT / "combined_optimizer_statistics.json", orient="records", indent=2)

    rank_table.to_csv(ANALYSIS_ROOT / "combined_rank_table.csv", index=False)
    winners_df.to_csv(ANALYSIS_ROOT / "combined_objective_winners.csv", index=False)

    for objective_name, objective_df in full_df.groupby("objective_name", dropna=False):
        objective_dir = ANALYSIS_ROOT / str(objective_name)
        ensure_dir(objective_dir)

        objective_stats = stats_df.loc[stats_df["objective_name"] == objective_name].copy()
        objective_rank_table = build_rank_table(objective_stats)

        objective_df.to_csv(objective_dir / "individual_runs.csv", index=False)
        objective_stats.to_csv(objective_dir / "optimizer_statistics.csv", index=False)
        objective_rank_table.to_csv(objective_dir / "rank_table.csv", index=False)
        save_json(objective_stats.to_dict(orient="records"), objective_dir / "optimizer_statistics.json")

        write_objective_report(
            objective_name=str(objective_name),
            objective_df=objective_df,
            objective_stats=objective_stats,
            rank_table=objective_rank_table,
            out_path=objective_dir / "human_readable_report.txt",
        )

        if MAKE_PLOTS:
            make_objective_plots(
                objective_df=objective_df,
                objective_stats=objective_stats,
                plots_dir=objective_dir / "plots",
            )

    write_combined_report(
        full_df=full_df,
        stats_df=stats_df,
        winners_df=winners_df,
        out_path=ANALYSIS_ROOT / "human_readable_report.txt",
    )

    print("")
    print("Done.")
    print(f"Analysis output written to: {ANALYSIS_ROOT}")


if __name__ == "__main__":
    main()