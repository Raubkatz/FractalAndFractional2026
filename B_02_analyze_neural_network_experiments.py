#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_nn_classification_results.py

Full analysis script for the multi-run fractional optimizer × fractal activation
classification experiments produced by:

    01_run_nn_classifications_21_optimizers.py

Mirrors the structure of the Himmelblau surface analysis but adapted for
classification metrics (accuracy, F1, precision, recall, epochs, timing).

Expected input structure
------------------------
    results_fractional_activation_optimizer_study_21/
        <dataset_name>/
            <optimizer>__<activation>__v<vderiv>.json   (one per NN config)
            <optimizer>__<activation>__h<h>__v<vderiv>.json

Each NN JSON schema:
    {
        "dataset": str,
        "optimizer": str,
        "activation": str,
        "history_size": int|null,
        "n_runs": int,
        "vderivs": [
            {
                "vderiv": float,
                "avg accuracy": float,
                "std accuracy": float,
                "avg f1 macro": float,
                "avg training time seconds": float,
                "avg test time seconds": float,
                "results": [
                    {
                        "run_seed": int, "accuracy": float, "f1_macro": float,
                        "precision_macro": float, "recall_macro": float,
                        "training_time_seconds": float, "test_time_seconds": float,
                        "epochs_completed": int, "best_val_loss": float
                    }, ...
                ]
            }
        ],
        "run_histories": [...]   // optional
    }

Main outputs
------------
    <ANALYSIS_ROOT>/
        combined_individual_runs.csv
        combined_config_statistics.csv
        combined_rank_table.csv
        combined_optimizer_statistics.csv        # collapsed across activations & vderivs
        combined_activation_statistics.csv       # collapsed across optimizers & vderivs
        combined_optimizer_group_statistics.csv  # by optimizer group
        human_readable_report_all_datasets.txt

        <dataset_name>/
            individual_runs.csv
            config_statistics.csv
            rank_table.csv
            optimizer_statistics.csv
            activation_statistics.csv
            optimizer_group_statistics.csv
            vderiv_sensitivity.csv
            human_readable_report_<dataset_name>.txt
            plots/
                accuracy_bar_ranked_<dataset_name>.png / .eps
                accuracy_boxplot_<dataset_name>.png / .eps
                f1_bar_ranked_<dataset_name>.png / .eps
                training_time_bar_<dataset_name>.png / .eps
                epochs_bar_<dataset_name>.png / .eps
                optimizer_activation_heatmap_accuracy_<dataset_name>.png / .eps
                optimizer_activation_heatmap_f1_<dataset_name>.png / .eps
                optimizer_group_accuracy_<dataset_name>.png / .eps
                vderiv_sensitivity_accuracy_<dataset_name>.png / .eps
                vderiv_sensitivity_f1_<dataset_name>.png / .eps
                accuracy_vs_time_scatter_<dataset_name>.png / .eps
                accuracy_distribution_kde_<dataset_name>.png / .eps
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.text import Text

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 0) USER SETTINGS
# =============================================================================

#RESULTS_ROOT = Path("results_fractional_activation_optimizer_study_21_50iterations_reduced")
#ANALYSIS_ROOT = Path("nn_classification_analysis_21_50iterations_reduced")

#RESULTS_ROOT = Path("final_results_classificaiton_experiments_40runs")
#ANALYSIS_ROOT = Path("analysis_final_nn_40runs")

RESULTS_ROOT = Path("compiled_results_all_experiments")
ANALYSIS_ROOT = Path("compiled_analysis_October2026")



# Ranking primary metric direction (higher is better)
RANK_BY = "accuracy"           # "accuracy" or "f1_macro"

if RANK_BY not in {"accuracy", "f1_macro"}:
    raise ValueError('RANK_BY must be either "accuracy" or "f1_macro".')

RANK_MEAN_COL = f"{RANK_BY}_mean"
RANK_SECONDARY_MEAN_COL = "f1_macro_mean" if RANK_BY == "accuracy" else "accuracy_mean"
RANK_DISPLAY_NAME = "mean accuracy" if RANK_BY == "accuracy" else "mean macro-F1"

# CI level
CI_Z = 1.96

# Plot settings
MAKE_PLOTS = True
FONT_SCALE = 2.3  # 1.0 = original font sizes; >1 enlarges all plot fonts; <1 reduces them
PLOT_DPI = 180
FIGSIZE_BAR      = (16, 7)
FIGSIZE_WIDE_BAR = (22, 8)
FIGSIZE_BOX      = (18, 7)
FIGSIZE_HEAT     = (22, 14)
FIGSIZE_SCATTER  = (14, 9)
FIGSIZE_KDE      = (14, 7)
BAR_ROT          = 80
HEAT_CMAP        = "RdYlGn"

# Optimizer groups (matching the 21 used in the experiment)
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

# Activation families
FRACTAL_ACTIVATIONS = {
    "modulated_blancmange_curve",
    "decaying_cosine_function_tf",
    "modified_weierstrass_function_tanh",
    "weierstrass_mandelbrot_function_xpsin",
}

STANDARD_ACTIVATIONS = {"relu", "tanh"}

# Additional activation-family analysis: exclude only the x+sin
# Weierstrass-Mandelbrot activation. The original analyses remain unchanged.
ACTIVATION_FAMILY_EXCLUDE_XPSIN = "weierstrass_mandelbrot_function_xpsin"


# Plot-only abbreviations. These are used only for axis labels, tick labels,
# legends, and annotations in figures. Tables, CSV files, and reports keep the
# original optimizer and activation names.
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

PLOT_ACTIVATION_LABELS: Dict[str, str] = {
    "relu": "ReLU",
    "tanh": "tanh",
    "modulated_blancmange_curve": "mod.\nblancmange",
    "decaying_cosine_function_tf": "dec.\ncosine",
    "modified_weierstrass_function_tanh": "mod.\nwei.\ntanh",
    "weierstrass_mandelbrot_function_xpsin": "wei.\nx+sin",
}


def plot_optimizer_label(name: Any) -> str:
    """Return the plot-only abbreviated optimizer label."""
    return PLOT_OPTIMIZER_LABELS.get(str(name), str(name))


def plot_activation_label(name: Any) -> str:
    """Return the plot-only abbreviated activation label."""
    return PLOT_ACTIVATION_LABELS.get(str(name), str(name))


def plot_optimizer_labels(names: Iterable[Any]) -> List[str]:
    """Return plot-only abbreviated optimizer labels."""
    return [plot_optimizer_label(name) for name in names]


def plot_activation_labels(names: Iterable[Any]) -> List[str]:
    """Return plot-only abbreviated activation labels."""
    return [plot_activation_label(name) for name in names]


# =============================================================================
# 1) UTILITIES
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_dataset_suffix(path: Path, dataset_name: str) -> Path:
    """Append _<dataset_name> immediately before the file extension."""
    return path.with_name(f"{path.stem}_{dataset_name}{path.suffix}")


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


def fmt(value: Any, digits: int = 5) -> str:
    try:
        v = float(value)
        if np.isnan(v):
            return "nan"
        return f"{v:.{digits}f}"
    except Exception:
        return str(value)


def fmt_pct(value: Any) -> str:
    try:
        return f"{float(value)*100:.2f}%"
    except Exception:
        return "nan"


def activation_family(name: str) -> str:
    if name in FRACTAL_ACTIVATIONS:
        return "Fractal"
    if name in STANDARD_ACTIVATIONS:
        return "Standard"
    return "Other"


def optimizer_group(name: str) -> str:
    return OPTIMIZER_GROUPS.get(name, "Unknown")


def bar_color_by_group(optimizer_names: List[str]) -> List[str]:
    palette = [
        "#589641", "#367DCD", "#874BD2", "#DC6923", "#E1AA14",
        "#2896A0", "#BE372D", "#2F7F74", "#C05A8A", "#7B5E3A",
        "#445FAB", "#AB4466", "#6DAB44", "#C0982A", "#3A9ABE",
        "#D45E2B", "#6A44AB", "#AB6A44", "#44AB8A", "#AB4498",
        "#7CAB44",
    ]
    colors = []
    counter: Dict[str, int] = {}
    for name in optimizer_names:
        group = OPTIMIZER_GROUPS.get(name, "Unknown")
        colors.append(GROUP_COLORS.get(group, "#888888"))
    return colors


# =============================================================================
# 2) DATA LOADING
# =============================================================================

def find_result_files(results_root: Path) -> List[Path]:
    """Return the neural-network result files."""
    nn_files: List[Path] = []
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    for dataset_dir in sorted(results_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for f in sorted(dataset_dir.glob("*.json")):
            nn_files.append(f)
    return nn_files


def load_nn_results(nn_files: List[Path]) -> List[Dict[str, Any]]:
    """
    Parse all NN result JSON files into flat per-run records.
    One record per (dataset, optimizer, activation, history_size, vderiv, run_seed).
    """
    records: List[Dict[str, Any]] = []

    for path in nn_files:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"  [WARN] Could not load {path}: {e}")
            continue

        dataset = data.get("dataset", path.parent.name)
        optimizer = data.get("optimizer", "")
        activation = data.get("activation", "")
        history_size = data.get("history_size", None)

        vderiv_list = data.get("vderivs", [])
        if not isinstance(vderiv_list, list):
            vderiv_list = []

        for vderiv_entry in vderiv_list:
            vderiv = safe_float(vderiv_entry.get("vderiv", 1.0))
            run_results = vderiv_entry.get("results", [])
            if not isinstance(run_results, list):
                continue

            for run in run_results:
                records.append(
                    {
                        "dataset":                 dataset,
                        "optimizer":               optimizer,
                        "activation":              activation,
                        "activation_family":       activation_family(activation),
                        "optimizer_group":         optimizer_group(optimizer),
                        "history_size":            safe_int(history_size, -1) if history_size is not None else None,
                        "vderiv":                  vderiv,
                        "run_seed":                safe_int(run.get("run_seed", -1)),
                        "accuracy":                safe_float(run.get("accuracy")),
                        "f1_macro":                safe_float(run.get("f1_macro")),
                        "precision_macro":         safe_float(run.get("precision_macro")),
                        "recall_macro":            safe_float(run.get("recall_macro")),
                        "training_time_seconds":   safe_float(run.get("training_time_seconds")),
                        "test_time_seconds":       safe_float(run.get("test_time_seconds")),
                        "epochs_completed":        safe_int(run.get("epochs_completed", -1)),
                        "best_val_loss":           safe_float(run.get("best_val_loss")),
                        "source_file":             str(path),
                    }
                )

    return records


# =============================================================================
# 3) STATISTICS
# =============================================================================

def numeric_summary(values: np.ndarray, prefix: str) -> Dict[str, float]:
    arr = values[~np.isnan(values)]
    n = int(arr.size)
    if n == 0:
        keys = ["n", "mean", "std", "sem", "ci95_low", "ci95_high",
                "median", "q25", "q75", "min", "max"]
        return {f"{prefix}_{k}": np.nan if k != "n" else 0 for k in keys}

    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem  = std / math.sqrt(n) if n > 1 else 0.0
    delta = CI_Z * sem

    return {
        f"{prefix}_n":         n,
        f"{prefix}_mean":      mean,
        f"{prefix}_std":       std,
        f"{prefix}_sem":       sem,
        f"{prefix}_ci95_low":  mean - delta,
        f"{prefix}_ci95_high": mean + delta,
        f"{prefix}_median":    float(np.median(arr)),
        f"{prefix}_q25":       float(np.percentile(arr, 25)),
        f"{prefix}_q75":       float(np.percentile(arr, 75)),
        f"{prefix}_min":       float(np.min(arr)),
        f"{prefix}_max":       float(np.max(arr)),
    }


def compute_config_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-config statistics.
    Config = (dataset, optimizer, activation, history_size, vderiv).
    """
    group_cols = ["dataset", "optimizer", "activation", "history_size", "vderiv",
                  "optimizer_group", "activation_family"]
    rows = []

    for keys, grp in df.groupby(group_cols, dropna=False):
        row: Dict[str, Any] = dict(zip(group_cols, keys))
        row["n_runs"] = len(grp)

        for col in ["accuracy", "f1_macro", "precision_macro", "recall_macro",
                    "training_time_seconds", "test_time_seconds", "epochs_completed", "best_val_loss"]:
            if col in grp.columns:
                row.update(numeric_summary(grp[col].values.astype(float), col))

        rows.append(row)

    out = pd.DataFrame(rows)
    out = _rank_configs(out)
    return out


def _rank_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank_overall (per dataset) by the selected primary ranking metric."""
    frames = []
    for dataset, grp in df.groupby("dataset", dropna=False):
        grp = grp.copy()
        grp = grp.sort_values(
            [RANK_MEAN_COL, RANK_SECONDARY_MEAN_COL, "training_time_seconds_mean"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        grp["rank_overall"] = np.arange(1, len(grp) + 1)
        frames.append(grp)
    if not frames:
        return df
    return pd.concat(frames, ignore_index=True)


def compute_optimizer_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse across activations and vderivs — best config per optimizer per dataset.
    Returns one row per (dataset, optimizer), using the config with the highest
    value of the selected primary ranking metric.
    """
    config_stats = compute_config_statistics(df)
    rows = []

    for (dataset, optimizer), grp in config_stats.groupby(["dataset", "optimizer"], dropna=False):
        # Best config = highest selected primary metric
        best = grp.sort_values(
            [RANK_MEAN_COL, RANK_SECONDARY_MEAN_COL, "training_time_seconds_mean"],
            ascending=[False, False, True],
        ).iloc[0]
        rows.append(best.to_dict())

    out = pd.DataFrame(rows)
    out = _rank_optimizers(out)
    return out


def _rank_optimizers(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for dataset, grp in df.groupby("dataset", dropna=False):
        grp = grp.copy()
        grp = grp.sort_values(
            [RANK_MEAN_COL, RANK_SECONDARY_MEAN_COL, "training_time_seconds_mean"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        grp["rank_optimizer"] = np.arange(1, len(grp) + 1)
        frames.append(grp)
    if not frames:
        return df
    return pd.concat(frames, ignore_index=True)


def compute_activation_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Best config per (dataset, activation) — collapsed across optimizers and vderivs."""
    config_stats = compute_config_statistics(df)
    rows = []

    for (dataset, activation), grp in config_stats.groupby(["dataset", "activation"], dropna=False):
        best = grp.sort_values(
            [RANK_MEAN_COL, RANK_SECONDARY_MEAN_COL, "training_time_seconds_mean"],
            ascending=[False, False, True],
        ).iloc[0]
        rows.append(best.to_dict())

    out = pd.DataFrame(rows)
    return out


def compute_optimizer_group_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (dataset, optimizer_group)."""
    rows = []
    for (dataset, group), grp in df.groupby(["dataset", "optimizer_group"], dropna=False):
        row: Dict[str, Any] = {"dataset": dataset, "optimizer_group": group}
        row["n_configs"] = grp["optimizer"].nunique() if "optimizer" in grp.columns else np.nan
        row["n_runs"] = len(grp)
        for col in ["accuracy", "f1_macro", "training_time_seconds"]:
            if col in grp.columns:
                row.update(numeric_summary(grp[col].values.astype(float), col))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_vderiv_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For Herrera and GL-Memory optimizers (which have a vderiv sweep),
    compute accuracy_mean per (dataset, optimizer, vderiv).
    """
    mask = df["optimizer_group"].isin(["Herrera", "GL-Memory"])
    sub = df[mask].copy()

    rows = []
    for keys, grp in sub.groupby(["dataset", "optimizer", "vderiv"], dropna=False):
        row = {"dataset": keys[0], "optimizer": keys[1], "vderiv": keys[2]}
        row["n_runs"] = len(grp)
        row.update(numeric_summary(grp["accuracy"].values.astype(float), "accuracy"))
        row.update(numeric_summary(grp["f1_macro"].values.astype(float), "f1_macro"))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 3b) ACTIVATION-FOCUSED STATISTICS
# =============================================================================

def compute_activation_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank activations by the selected primary metric per dataset, collapsed across ALL optimizers.
    One row per (dataset, activation) — the primary activation-focused view.
    """
    rows = []
    for (dataset, activation), grp in df.groupby(["dataset", "activation"], dropna=False):
        row: Dict[str, Any] = {
            "dataset":            dataset,
            "activation":         activation,
            "activation_family":  activation_family(activation),
            "n_runs":             len(grp),
            "n_optimizers":       int(grp["optimizer"].nunique()),
        }
        row.update(numeric_summary(grp["accuracy"].values.astype(float), "accuracy"))
        row.update(numeric_summary(grp["f1_macro"].values.astype(float), "f1_macro"))
        row.update(numeric_summary(grp["training_time_seconds"].values.astype(float), "training_time_seconds"))
        rows.append(row)

    out = pd.DataFrame(rows)

    # Rank per dataset by the selected primary metric descending
    frames = []
    for dataset, grp in out.groupby("dataset", dropna=False):
        grp = grp.copy()
        grp = grp.sort_values(
            [RANK_MEAN_COL, RANK_SECONDARY_MEAN_COL],
            ascending=[False, False],
        ).reset_index(drop=True)
        grp["rank_activation"] = np.arange(1, len(grp) + 1)
        frames.append(grp)
    return pd.concat(frames, ignore_index=True) if frames else out


def compute_optgroup_x_activation_family(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulation: optimizer_group × activation_family → mean accuracy.
    Collapsed across individual optimizers and activations within each cell.
    """
    rows = []
    for (dataset, opt_group, act_family), grp in df.groupby(
        ["dataset", "optimizer_group", "activation_family"], dropna=False
    ):
        row: Dict[str, Any] = {
            "dataset":            dataset,
            "optimizer_group":    opt_group,
            "activation_family":  act_family,
            "n_runs":             len(grp),
            "n_optimizers":       int(grp["optimizer"].nunique()),
            "n_activations":      int(grp["activation"].nunique()),
        }
        row.update(numeric_summary(grp["accuracy"].values.astype(float), "accuracy"))
        row.update(numeric_summary(grp["f1_macro"].values.astype(float), "f1_macro"))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_fractal_vs_standard(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Direct fractal vs standard comparison per (dataset, optimizer_group).

    Returns
    -------
    fvs_df : pd.DataFrame
        Long-format: one row per (dataset, optimizer_group, activation_family).
    fvs_gains : pd.DataFrame
        Wide-format gain table: one row per (dataset, optimizer_group) with
        fractal_acc, standard_acc, and fractal_gain = fractal_acc − standard_acc.
    """
    sub = df[df["activation_family"].isin(["Fractal", "Standard"])].copy()

    rows = []
    for (dataset, opt_group, act_family), grp in sub.groupby(
        ["dataset", "optimizer_group", "activation_family"], dropna=False
    ):
        row: Dict[str, Any] = {
            "dataset":           dataset,
            "optimizer_group":   opt_group,
            "activation_family": act_family,
            "n_runs":            len(grp),
        }
        row.update(numeric_summary(grp["accuracy"].values.astype(float), "accuracy"))
        rows.append(row)

    fvs_df = pd.DataFrame(rows)

    # Compute gain per (dataset, optimizer_group)
    gain_rows = []
    for (dataset, opt_group), grp in fvs_df.groupby(["dataset", "optimizer_group"], dropna=False):
        frac = grp[grp["activation_family"] == "Fractal"]
        std  = grp[grp["activation_family"] == "Standard"]
        if frac.empty or std.empty:
            continue
        fractal_acc  = float(frac["accuracy_mean"].values[0])
        standard_acc = float(std["accuracy_mean"].values[0])
        gain_rows.append(
            {
                "dataset":         dataset,
                "optimizer_group": opt_group,
                "fractal_acc":     fractal_acc,
                "standard_acc":    standard_acc,
                "fractal_gain":    fractal_acc - standard_acc,
            }
        )

    fvs_gains = pd.DataFrame(gain_rows)
    return fvs_df, fvs_gains


def build_optimizer_activation_heatmap(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    For a single dataset, build an optimizer × activation pivot of mean accuracy.
    Uses the best vderiv/history_size config per (optimizer, activation).
    """
    sub = df[df["dataset"] == dataset].copy()
    config_stats = compute_config_statistics(sub)

    # Best config per (optimizer, activation), selected by the primary ranking metric
    best_rows = []
    for (optimizer, activation), grp in config_stats.groupby(["optimizer", "activation"], dropna=False):
        best = grp.sort_values(
            [RANK_MEAN_COL, RANK_SECONDARY_MEAN_COL],
            ascending=[False, False],
        ).iloc[0]
        best_rows.append({"optimizer": optimizer, "activation": activation, "accuracy_mean": best["accuracy_mean"]})

    if not best_rows:
        return pd.DataFrame()

    pivot_df = pd.DataFrame(best_rows)
    pivot = pivot_df.pivot(index="optimizer", columns="activation", values="accuracy_mean")

    # Sort optimizers by mean accuracy across activations
    pivot["__mean__"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("__mean__", ascending=False).drop(columns="__mean__")
    return pivot


# =============================================================================
# 4) PLOTS
# =============================================================================

def _scale_figure_fonts(fig: plt.Figure) -> None:
    """Scale every Matplotlib text element in the figure by FONT_SCALE."""
    if FONT_SCALE == 1.0:
        return
    for text_obj in fig.findobj(match=Text):
        text_obj.set_fontsize(text_obj.get_fontsize() * FONT_SCALE)


def _send_grid_behind_artists(fig: plt.Figure) -> None:
    """Force grid lines behind plotted artists such as bars."""
    for ax in fig.axes:
        ax.set_axisbelow(True)
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            gridline.set_zorder(0)


def _finish_fig(fig: plt.Figure, path: Path) -> None:
    dataset_name = path.parent.parent.name if path.parent.name == "plots" else "all_datasets"
    path = append_dataset_suffix(path, dataset_name)
    _send_grid_behind_artists(fig)
    _scale_figure_fonts(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".eps"), format="eps", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_bar(opt_stats: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    """Ranked mean accuracy bar chart, coloured by optimizer group."""
    sub = opt_stats[opt_stats["dataset"] == dataset].sort_values("rank_optimizer").copy()
    if sub.empty:
        return

    colors = bar_color_by_group(sub["optimizer"].tolist())
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE_BAR)
    x = np.arange(len(sub))
    bars = ax.bar(x, sub["accuracy_mean"], yerr=sub["accuracy_sem"], capsize=3,
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_optimizer_labels(sub["optimizer"]), rotation=BAR_ROT, ha="center", fontsize=8)
    ax.set_title(f"[{dataset}]  Mean accuracy by optimizer (best config per optimizer)", fontsize=12)
    ax.set_ylabel("Mean accuracy ± SEM")
    ax.set_ylim(bottom=max(0, float(sub["accuracy_mean"].min()) - 0.05))
    ax.grid(True, axis="y", alpha=0.3)

    # Legend for groups
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.8)

    _finish_fig(fig, out_dir / "accuracy_bar_ranked.png")


def plot_accuracy_boxplot(df: pd.DataFrame, opt_stats: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    """Accuracy box plots, ranked by mean accuracy."""
    sub_df    = df[df["dataset"] == dataset].copy()
    sub_stats = opt_stats[opt_stats["dataset"] == dataset].sort_values("rank_optimizer")
    if sub_df.empty or sub_stats.empty:
        return

    ordered = sub_stats["optimizer"].tolist()
    data = []
    labels = []
    colors = []
    for opt in ordered:
        vals = sub_df[sub_df["optimizer"] == opt]["accuracy"].dropna().values
        if vals.size > 0:
            data.append(vals)
            labels.append(plot_optimizer_label(opt))
            colors.append(GROUP_COLORS.get(optimizer_group(opt), "#888888"))

    if not data:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE_BAR)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black", markersize=4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.tick_params(axis="x", rotation=BAR_ROT, labelsize=8)
    ax.set_title(f"[{dataset}]  Accuracy distribution by optimizer", fontsize=12)
    ax.set_ylabel("Accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    _finish_fig(fig, out_dir / "accuracy_boxplot.png")


def plot_f1_bar(opt_stats: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    sub = opt_stats[opt_stats["dataset"] == dataset].sort_values("rank_optimizer").copy()
    if sub.empty:
        return

    colors = bar_color_by_group(sub["optimizer"].tolist())
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE_BAR)
    x = np.arange(len(sub))
    ax.bar(x, sub["f1_macro_mean"], yerr=sub["f1_macro_sem"], capsize=3,
           color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_optimizer_labels(sub["optimizer"]), rotation=BAR_ROT, ha="center", fontsize=8)
    ax.set_title(f"[{dataset}]  Mean macro-F1 by optimizer (best config)", fontsize=12)
    ax.set_ylabel("Mean F1-macro ± SEM")
    ax.set_ylim(bottom=max(0, float(sub["f1_macro_mean"].min()) - 0.05))
    ax.grid(True, axis="y", alpha=0.3)
    _finish_fig(fig, out_dir / "f1_bar_ranked.png")


def plot_training_time_bar(opt_stats: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    sub = opt_stats[opt_stats["dataset"] == dataset].sort_values("rank_optimizer").copy()
    if sub.empty:
        return

    colors = bar_color_by_group(sub["optimizer"].tolist())
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE_BAR)
    x = np.arange(len(sub))
    ax.bar(x, sub["training_time_seconds_mean"], yerr=sub["training_time_seconds_sem"],
           capsize=3, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_optimizer_labels(sub["optimizer"]), rotation=BAR_ROT, ha="center", fontsize=8)
    ax.set_title(f"[{dataset}]  Mean training time by optimizer [s]", fontsize=12)
    ax.set_ylabel("Mean training time [s] ± SEM")
    ax.grid(True, axis="y", alpha=0.3)
    _finish_fig(fig, out_dir / "training_time_bar.png")


def plot_epochs_bar(opt_stats: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    sub = opt_stats[opt_stats["dataset"] == dataset].sort_values("rank_optimizer").copy()
    if "epochs_completed_mean" not in sub.columns:
        return
    sub = sub.dropna(subset=["epochs_completed_mean"])
    if sub.empty:
        return

    colors = bar_color_by_group(sub["optimizer"].tolist())
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE_BAR)
    x = np.arange(len(sub))
    ax.bar(x, sub["epochs_completed_mean"], yerr=sub["epochs_completed_sem"],
           capsize=3, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_optimizer_labels(sub["optimizer"]), rotation=BAR_ROT, ha="center", fontsize=8)
    ax.set_title(f"[{dataset}]  Mean epochs completed (early stopping)", fontsize=12)
    ax.set_ylabel("Mean epochs ± SEM")
    ax.grid(True, axis="y", alpha=0.3)
    _finish_fig(fig, out_dir / "epochs_bar.png")


def plot_heatmap(pivot: pd.DataFrame, title: str, out_path: Path) -> None:
    if pivot.empty:
        return

    data = pivot.values.astype(float)
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    fig, ax = plt.subplots(figsize=FIGSIZE_HEAT)
    im = ax.imshow(data, aspect="auto", cmap=HEAT_CMAP, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(plot_activation_labels(pivot.columns), rotation=65, ha="center", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(plot_optimizer_labels(pivot.index), fontsize=8)
    ax.set_title(title, fontsize=11)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if (val - vmin) / max(vmax - vmin, 1e-9) < 0.35 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    _finish_fig(fig, out_path)


def plot_optimizer_group(grp_stats: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    sub = grp_stats[grp_stats["dataset"] == dataset].copy()
    sub["group_order"] = sub["optimizer_group"].map(
        {g: i for i, g in enumerate(GROUP_ORDER)}
    ).fillna(99)
    sub = sub.sort_values("group_order")
    if sub.empty:
        return

    colors = [GROUP_COLORS.get(g, "#888888") for g in sub["optimizer_group"]]
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    x = np.arange(len(sub))
    ax.bar(x, sub["accuracy_mean"], yerr=sub["accuracy_sem"], capsize=4,
           color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["optimizer_group"], rotation=30, ha="center")
    ax.set_title(f"[{dataset}]  Mean accuracy by optimizer group", fontsize=12)
    ax.set_ylabel("Mean accuracy ± SEM")
    ax.set_ylim(bottom=max(0, float(sub["accuracy_mean"].min()) - 0.05))
    ax.grid(True, axis="y", alpha=0.3)
    _finish_fig(fig, out_dir / "optimizer_group_accuracy.png")


def plot_vderiv_sensitivity(vderiv_df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    sub = vderiv_df[vderiv_df["dataset"] == dataset].copy()
    if sub.empty:
        return

    optimizers = sub["optimizer"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(optimizers)))

    # Accuracy
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    for opt, color in zip(optimizers, colors):
        s = sub[sub["optimizer"] == opt].sort_values("vderiv")
        ax.plot(s["vderiv"], s["accuracy_mean"], marker="o", label=plot_optimizer_label(opt), color=color)
        ax.fill_between(s["vderiv"],
                        s["accuracy_mean"] - s["accuracy_sem"],
                        s["accuracy_mean"] + s["accuracy_sem"],
                        alpha=0.15, color=color)
    ax.set_xlabel("vderiv (fractional order)")
    ax.set_ylabel("Mean accuracy ± SEM")
    ax.set_title(f"[{dataset}]  vderiv sensitivity — accuracy", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    _finish_fig(fig, out_dir / "vderiv_sensitivity_accuracy.png")

    # F1
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    for opt, color in zip(optimizers, colors):
        s = sub[sub["optimizer"] == opt].sort_values("vderiv")
        ax.plot(s["vderiv"], s["f1_macro_mean"], marker="o", label=plot_optimizer_label(opt), color=color)
        ax.fill_between(s["vderiv"],
                        s["f1_macro_mean"] - s["f1_macro_sem"],
                        s["f1_macro_mean"] + s["f1_macro_sem"],
                        alpha=0.15, color=color)
    ax.set_xlabel("vderiv (fractional order)")
    ax.set_ylabel("Mean F1-macro ± SEM")
    ax.set_title(f"[{dataset}]  vderiv sensitivity — F1-macro", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    _finish_fig(fig, out_dir / "vderiv_sensitivity_f1.png")


def plot_accuracy_vs_time(opt_stats: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    """Scatter: mean accuracy vs mean training time, coloured by group."""
    sub = opt_stats[opt_stats["dataset"] == dataset].dropna(
        subset=["accuracy_mean", "training_time_seconds_mean"]
    ).copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)
    for group in GROUP_ORDER:
        grp = sub[sub["optimizer_group"] == group]
        if grp.empty:
            continue
        ax.scatter(grp["training_time_seconds_mean"], grp["accuracy_mean"],
                   label=group, color=GROUP_COLORS[group], s=80, alpha=0.85, zorder=3)
        for _, row in grp.iterrows():
            ax.annotate(plot_optimizer_label(row["optimizer"]),
                        (row["training_time_seconds_mean"], row["accuracy_mean"]),
                        fontsize=6, alpha=0.7,
                        xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel("Mean training time [s]")
    ax.set_ylabel("Mean accuracy")
    ax.set_title(f"[{dataset}]  Accuracy vs training time (Pareto view)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _finish_fig(fig, out_dir / "accuracy_vs_time_scatter.png")


def plot_accuracy_kde(df: pd.DataFrame, opt_stats: pd.DataFrame,
                      dataset: str, out_dir: Path) -> None:
    """KDE of accuracy distributions for the top optimizers per group."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return

    sub_df = df[df["dataset"] == dataset].copy()
    sub_stats = opt_stats[opt_stats["dataset"] == dataset].sort_values("rank_optimizer")

    # Best optimizer per group
    shown_optimizers: List[str] = []
    for group in GROUP_ORDER:
        g = sub_stats[sub_stats["optimizer_group"] == group]
        if not g.empty:
            shown_optimizers.append(g.iloc[0]["optimizer"])

    if not shown_optimizers:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_KDE)
    x_grid = np.linspace(0, 1, 500)

    for opt in shown_optimizers:
        vals = sub_df[sub_df["optimizer"] == opt]["accuracy"].dropna().values
        if vals.size < 3:
            continue
        kde = gaussian_kde(vals, bw_method="scott")
        color = GROUP_COLORS.get(optimizer_group(opt), "#888888")
        ax.plot(x_grid, kde(x_grid), label=plot_optimizer_label(opt), color=color, linewidth=2)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.1, color=color)

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Density")
    ax.set_title(f"[{dataset}]  Accuracy KDE — best optimizer per group", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _finish_fig(fig, out_dir / "accuracy_distribution_kde.png")


def plot_cross_dataset_summary(opt_stats: pd.DataFrame, out_dir: Path) -> None:
    """
    Cross-dataset: for each optimizer, plot mean accuracy across datasets
    (averaged across datasets, weighted equally).
    """
    if "optimizer" not in opt_stats.columns or opt_stats.empty:
        return

    summary = (
        opt_stats.groupby("optimizer")
        .agg(
            mean_accuracy=("accuracy_mean", "mean"),
            std_accuracy=("accuracy_mean", "std"),
            mean_f1=("f1_macro_mean", "mean"),
            n_datasets=("dataset", "nunique"),
            optimizer_group=("optimizer_group", "first"),
        )
        .reset_index()
        .sort_values("mean_accuracy" if RANK_BY == "accuracy" else "mean_f1", ascending=False)
    )

    colors = bar_color_by_group(summary["optimizer"].tolist())
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE_BAR)
    x = np.arange(len(summary))
    ax.bar(x, summary["mean_accuracy"], yerr=summary["std_accuracy"],
           capsize=3, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_optimizer_labels(summary["optimizer"]), rotation=BAR_ROT, ha="center", fontsize=8)
    ax.set_title("Cross-dataset mean accuracy (averaged over all datasets)", fontsize=12)
    ax.set_ylabel("Mean accuracy (cross-dataset mean)")
    ax.grid(True, axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.8)
    _finish_fig(fig, out_dir / "cross_dataset_optimizer_summary.png")


def plot_cross_dataset_group_heatmap(opt_stats: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: optimizer_group × dataset → mean accuracy."""
    if opt_stats.empty:
        return

    grp_stats = (
        opt_stats.groupby(["dataset", "optimizer_group"])
        .agg(accuracy_mean=("accuracy_mean", "mean"))
        .reset_index()
    )
    pivot = grp_stats.pivot(index="optimizer_group", columns="dataset", values="accuracy_mean")
    # Reorder rows
    pivot = pivot.reindex([g for g in GROUP_ORDER if g in pivot.index])

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2), 6))
    data = pivot.values.astype(float)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    im = ax.imshow(data, aspect="auto", cmap=HEAT_CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="center", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Optimizer group × dataset — mean accuracy (best optimizer per group)", fontsize=11)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                tc = "white" if (val - vmin) / max(vmax - vmin, 1e-9) < 0.4 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=tc)
    plt.colorbar(im, ax=ax, fraction=0.02)
    _finish_fig(fig, out_dir / "cross_dataset_group_heatmap.png")


# =============================================================================
# 4b) ACTIVATION-FOCUSED PLOTS
# =============================================================================

def plot_activation_bar_ranked(act_ranking: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    """Ranked mean accuracy per activation (collapsed across all optimizers)."""
    sub = act_ranking[act_ranking["dataset"] == dataset].sort_values("rank_activation").copy()
    if sub.empty:
        return

    colors = ["#589641" if r == "Fractal" else "#2F7F74" if r == "Standard" else "#888888"
              for r in sub["activation_family"]]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE_BAR)
    x = np.arange(len(sub))
    ax.bar(x, sub["accuracy_mean"], yerr=sub["accuracy_sem"], capsize=3,
           color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_activation_labels(sub["activation"]), rotation=BAR_ROT, ha="center", fontsize=8)
    ax.set_title(f"[{dataset}]  Mean accuracy by activation (all optimizers)", fontsize=12)
    ax.set_ylabel("Mean accuracy ± SEM")
    ax.set_ylim(bottom=max(0, float(sub["accuracy_mean"].min()) - 0.05))
    ax.grid(True, axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#589641", label="Fractal"), Patch(facecolor="#2F7F74", label="Standard")]
    ax.legend(handles=handles, fontsize=9)
    _finish_fig(fig, out_dir / "activation_bar_ranked.png")


def plot_activation_boxplot_chart(df: pd.DataFrame, act_ranking: pd.DataFrame,
                                  dataset: str, out_dir: Path) -> None:
    """Accuracy box plots per activation, ranked by mean accuracy."""
    sub_df    = df[df["dataset"] == dataset].copy()
    sub_stats = act_ranking[act_ranking["dataset"] == dataset].sort_values("rank_activation")
    if sub_df.empty or sub_stats.empty:
        return

    data, labels, colors = [], [], []
    for act in sub_stats["activation"]:
        vals = sub_df[sub_df["activation"] == act]["accuracy"].dropna().values
        if vals.size > 0:
            data.append(vals)
            labels.append(plot_activation_label(act))
            fam = activation_family(act)
            colors.append("#589641" if fam == "Fractal" else "#2F7F74" if fam == "Standard" else "#888888")

    if not data:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE_BAR)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black", markersize=4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.tick_params(axis="x", rotation=BAR_ROT, labelsize=8)
    ax.set_title(f"[{dataset}]  Accuracy distribution by activation", fontsize=12)
    ax.set_ylabel("Accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    _finish_fig(fig, out_dir / "activation_boxplot.png")


def plot_optgroup_x_activation_family_heatmap(grp_x_act: pd.DataFrame,
                                               dataset: str, out_dir: Path) -> None:
    """Heatmap: optimizer_group × activation_family → mean accuracy."""
    sub = grp_x_act[grp_x_act["dataset"] == dataset].copy()
    if sub.empty:
        return

    pivot = sub.pivot(index="optimizer_group", columns="activation_family", values="accuracy_mean")
    pivot = pivot.reindex([g for g in GROUP_ORDER if g in pivot.index])
    if pivot.empty:
        return

    data = pivot.values.astype(float)
    vmin, vmax = np.nanmin(data), np.nanmax(data)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 3), 6))
    im = ax.imshow(data, aspect="auto", cmap=HEAT_CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title(f"[{dataset}]  Optimizer group × activation family — mean accuracy", fontsize=12)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                tc = "white" if (val - vmin) / max(vmax - vmin, 1e-9) < 0.4 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=11, color=tc, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.02)
    _finish_fig(fig, out_dir / "optgroup_x_activation_family_heatmap.png")


def plot_fractal_vs_standard_by_optgroup(fvs_gains: pd.DataFrame, dataset: str,
                                          out_dir: Path) -> None:
    """
    Two-panel: (left) grouped bar of fractal vs standard accuracy per optimizer group;
               (right) fractal gain (Δ = fractal − standard) per optimizer group.
    """
    if fvs_gains.empty or "dataset" not in fvs_gains.columns:
        sub = fvs_gains
    else:
        sub = fvs_gains[fvs_gains["dataset"] == dataset].copy()
    if sub.empty:
        return

    sub = sub.copy()
    sub["_ord"] = sub["optimizer_group"].map(
        {g: i for i, g in enumerate(GROUP_ORDER)}
    ).fillna(99)
    sub = sub.sort_values("_ord")

    x = np.arange(len(sub))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: absolute comparison
    ax1.bar(x - width / 2, sub["fractal_acc"],  width, label="Fractal",  color="#589641", alpha=0.85)
    ax1.bar(x + width / 2, sub["standard_acc"], width, label="Standard", color="#2F7F74", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["optimizer_group"], rotation=28, ha="center")
    ax1.set_title(f"[{dataset}]  Fractal vs Standard — accuracy by optimizer group", fontsize=11)
    ax1.set_ylabel("Mean accuracy")
    ax1.legend(fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)

    # Right: gain
    bar_colors = ["#589641" if g >= 0 else "#BE372D" for g in sub["fractal_gain"]]
    ax2.bar(x, sub["fractal_gain"], color=bar_colors, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub["optimizer_group"], rotation=28, ha="center")
    ax2.set_title(f"[{dataset}]  Fractal gain (Fractal − Standard) per optimizer group", fontsize=11)
    ax2.set_ylabel("Accuracy gain")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"[{dataset}]  Fractal vs Standard activation analysis", fontsize=13, fontweight="bold")
    _finish_fig(fig, out_dir / "fractal_vs_standard_by_optgroup.png")


def plot_activation_family_x_optgroup_lines(grp_x_act: pd.DataFrame, dataset: str,
                                             out_dir: Path) -> None:
    """
    Line plot: for each activation family (Fractal / Standard), plot mean accuracy
    across optimizer groups — shows interaction between family and optimizer group.
    """
    sub = grp_x_act[grp_x_act["dataset"] == dataset].copy()
    if sub.empty:
        return

    act_families = [f for f in ["Fractal", "Standard"] if f in sub["activation_family"].values]
    x_labels = [g for g in GROUP_ORDER if g in sub["optimizer_group"].values]
    if not x_labels:
        return

    x = np.arange(len(x_labels))
    colors_map = {"Fractal": "#589641", "Standard": "#2F7F74"}

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    for fam in act_families:
        vals, sems = [], []
        for grp_name in x_labels:
            m = sub[(sub["optimizer_group"] == grp_name) & (sub["activation_family"] == fam)]
            vals.append(float(m["accuracy_mean"].values[0]) if not m.empty else np.nan)
            sems.append(float(m["accuracy_sem"].values[0])  if not m.empty else np.nan)
        vals = np.array(vals, dtype=float)
        sems = np.array(sems, dtype=float)
        color = colors_map.get(fam, "#888888")
        ax.plot(x, vals, marker="o", label=fam, color=color, linewidth=2.5, markersize=9)
        ax.fill_between(x, vals - sems, vals + sems, alpha=0.15, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=22, ha="center")
    ax.set_title(f"[{dataset}]  Activation family across optimizer groups", fontsize=12)
    ax.set_ylabel("Mean accuracy ± SEM")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _finish_fig(fig, out_dir / "activation_family_x_optgroup_lines.png")


def plot_activation_family_x_optgroup_lines_without_xpsin(
    grp_x_act: pd.DataFrame,
    dataset: str,
    out_dir: Path,
) -> None:
    """
    Additional line plot identical to plot_activation_family_x_optgroup_lines(),
    but based on statistics computed after excluding
    weierstrass_mandelbrot_function_xpsin.
    """
    sub = grp_x_act[grp_x_act["dataset"] == dataset].copy()
    if sub.empty:
        return

    act_families = [f for f in ["Fractal", "Standard"] if f in sub["activation_family"].values]
    x_labels = [g for g in GROUP_ORDER if g in sub["optimizer_group"].values]
    if not x_labels:
        return

    x = np.arange(len(x_labels))
    colors_map = {"Fractal": "#589641", "Standard": "#2F7F74"}

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    for fam in act_families:
        vals, sems = [], []
        for grp_name in x_labels:
            m = sub[(sub["optimizer_group"] == grp_name) & (sub["activation_family"] == fam)]
            vals.append(float(m["accuracy_mean"].values[0]) if not m.empty else np.nan)
            sems.append(float(m["accuracy_sem"].values[0])  if not m.empty else np.nan)
        vals = np.array(vals, dtype=float)
        sems = np.array(sems, dtype=float)
        color = colors_map.get(fam, "#888888")
        ax.plot(x, vals, marker="o", label=fam, color=color, linewidth=2.5, markersize=9)
        ax.fill_between(x, vals - sems, vals + sems, alpha=0.15, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=22, ha="center")
    ax.set_title(
        f"[{dataset}]  Activation family across optimizer groups "
        f"(without wei. x+sin)",
        fontsize=12,
    )
    ax.set_ylabel("Mean accuracy ± SEM")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _finish_fig(
        fig,
        out_dir / "activation_family_x_optgroup_lines_without_xpsin.png",
    )


def plot_all_optimizers_x_activations_heatmap(df: pd.DataFrame, dataset: str,
                                               out_dir: Path) -> None:
    """
    Full heatmap: every optimizer × every activation → mean accuracy.
    Optimizers sorted by best accuracy across activations (row); activations by
    overall mean (column). Colour-coded by accuracy magnitude.
    """
    sub = df[df["dataset"] == dataset].copy()
    if sub.empty:
        return

    pivot_data = (
        sub.groupby(["optimizer", "activation"])["accuracy"]
        .mean()
        .reset_index()
    )
    if pivot_data.empty:
        return

    pivot = pivot_data.pivot(index="optimizer", columns="activation", values="accuracy")

    # Sort optimizers descending by max accuracy
    pivot["_best"] = pivot.max(axis=1)
    pivot = pivot.sort_values("_best", ascending=False).drop(columns="_best")

    # Sort activations descending by mean
    act_means = pivot.mean(axis=0)
    pivot = pivot[act_means.sort_values(ascending=False).index]

    data = pivot.values.astype(float)
    vmin, vmax = np.nanmin(data), np.nanmax(data)

    fig, ax = plt.subplots(figsize=(FIGSIZE_HEAT[0], FIGSIZE_HEAT[1] * 1.10))
    im = ax.imshow(data, aspect="auto", cmap=HEAT_CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(plot_activation_labels(pivot.columns), rotation=65, ha="center", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(plot_optimizer_labels(pivot.index), fontsize=7)
    ax.tick_params(axis="y", pad=7)
    ax.set_title(f"[{dataset}]  All optimizers × all activations — mean accuracy", fontsize=11)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                tc = "white" if (val - vmin) / max(vmax - vmin, 1e-9) < 0.35 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=6.5, color=tc)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    _finish_fig(fig, out_dir / "all_optimizers_x_activations_heatmap.png")


def plot_cross_dataset_activation_summary(act_ranking: pd.DataFrame, out_dir: Path) -> None:
    """Cross-dataset ranked bar: mean accuracy per activation averaged over all datasets."""
    if act_ranking.empty:
        return

    summary = (
        act_ranking.groupby(["activation", "activation_family"])
        .agg(
            mean_acc=("accuracy_mean", "mean"),
            std_acc=("accuracy_mean", "std"),
            mean_f1=("f1_macro_mean", "mean"),
        )
        .reset_index()
        .sort_values("mean_acc" if RANK_BY == "accuracy" else "mean_f1", ascending=False)
    )
    if summary.empty:
        return

    colors = ["#589641" if f == "Fractal" else "#2F7F74" if f == "Standard" else "#888888"
              for f in summary["activation_family"]]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE_BAR)
    x = np.arange(len(summary))
    ax.bar(x, summary["mean_acc"], yerr=summary["std_acc"], capsize=3, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_activation_labels(summary["activation"]), rotation=BAR_ROT, ha="center", fontsize=8)
    ax.set_title("Cross-dataset mean accuracy by activation (averaged over all datasets)", fontsize=12)
    ax.set_ylabel("Mean accuracy (cross-dataset)")
    ax.grid(True, axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#589641", label="Fractal"), Patch(facecolor="#2F7F74", label="Standard")]
    ax.legend(handles=handles, fontsize=9)
    _finish_fig(fig, out_dir / "cross_dataset_activation_summary.png")


def plot_cross_dataset_optgroup_x_actfamily_heatmap(grp_x_act: pd.DataFrame,
                                                      out_dir: Path) -> None:
    """Cross-dataset heatmap: optimizer_group × activation_family averaged over all datasets."""
    if grp_x_act.empty:
        return

    cross = (
        grp_x_act.groupby(["optimizer_group", "activation_family"])
        .agg(accuracy_mean=("accuracy_mean", "mean"))
        .reset_index()
    )
    pivot = cross.pivot(index="optimizer_group", columns="activation_family", values="accuracy_mean")
    pivot = pivot.reindex([g for g in GROUP_ORDER if g in pivot.index])
    if pivot.empty:
        return

    data = pivot.values.astype(float)
    vmin, vmax = np.nanmin(data), np.nanmax(data)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 3), 6))
    im = ax.imshow(data, aspect="auto", cmap=HEAT_CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)
    ax.set_title("Cross-dataset: optimizer group × activation family — mean accuracy", fontsize=12)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                tc = "white" if (val - vmin) / max(vmax - vmin, 1e-9) < 0.4 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=12, color=tc, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.02)
    _finish_fig(fig, out_dir / "cross_dataset_optgroup_x_actfamily_heatmap.png")


# =============================================================================
# 5b) ACTIVATION-FOCUSED REPORTS
# =============================================================================

def write_activation_report(
    dataset: str,
    df: pd.DataFrame,
    act_ranking: pd.DataFrame,
    grp_x_act: pd.DataFrame,
    fvs_df: pd.DataFrame,
    fvs_gains: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write a dedicated activation-function analysis report for one dataset."""
    lines = []
    W = 120

    lines.append("=" * W)
    lines.append(f"Activation function analysis report — dataset: {dataset}")
    lines.append("=" * W)
    lines.append("")

    sub_df = df[df["dataset"] == dataset]
    lines.append(f"  Activations tested: {sub_df['activation'].nunique()}")
    lines.append(f"  Optimizers tested:  {sub_df['optimizer'].nunique()}")
    lines.append(f"  Total runs:         {len(sub_df)}")
    lines.append("")

    # ---- 1. Activation ranking ----
    lines.append(f"1. Activation ranking ({RANK_DISPLAY_NAME} across ALL optimizers, ranked desc)")
    lines.append("-" * W)
    sub_act = act_ranking[act_ranking["dataset"] == dataset].sort_values("rank_activation")
    if not sub_act.empty:
        cols = ["rank_activation", "activation", "activation_family", "n_optimizers",
                "accuracy_mean", "accuracy_std", "accuracy_sem",
                "accuracy_ci95_low", "accuracy_ci95_high",
                "f1_macro_mean", "f1_macro_std"]
        cols = [c for c in cols if c in sub_act.columns]
        lines.append(_df_to_text(sub_act[cols]))
    lines.append("")

    # ---- 2. Activation family summary ----
    lines.append("2. Activation family summary (Fractal vs Standard — all optimizers)")
    lines.append("-" * W)
    fam_summary = (
        sub_df.groupby("activation_family")
        .agg(
            n_activations=("activation", "nunique"),
            n_runs=("accuracy", "count"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            acc_sem=("accuracy", lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0),
            f1_mean=("f1_macro", "mean"),
        )
        .reset_index()
        .sort_values("acc_mean" if RANK_BY == "accuracy" else "f1_mean", ascending=False)
    )
    lines.append(_df_to_text(fam_summary))
    lines.append("")
    if len(fam_summary) >= 2:
        frac = fam_summary[fam_summary["activation_family"] == "Fractal"]
        std  = fam_summary[fam_summary["activation_family"] == "Standard"]
        if not frac.empty and not std.empty:
            delta = float(frac["acc_mean"].values[0]) - float(std["acc_mean"].values[0])
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"  → Overall fractal gain over standard: {sign}{fmt(delta)}  "
                f"({'Fractal better' if delta > 0 else 'Standard better'})"
            )
    lines.append("")

    # ---- 3. Optimizer group × activation family ----
    lines.append("3. Optimizer group × activation family — mean accuracy")
    lines.append("-" * W)
    sub_gxa = grp_x_act[grp_x_act["dataset"] == dataset].copy()
    if not sub_gxa.empty:
        cols = ["optimizer_group", "activation_family", "n_runs", "n_activations",
                "accuracy_mean", "accuracy_std", "accuracy_sem", "f1_macro_mean"]
        cols = [c for c in cols if c in sub_gxa.columns]
        lines.append(_df_to_text(sub_gxa.sort_values(["optimizer_group", "activation_family"])[cols]))
    lines.append("")

    # ---- 4. Fractal gain per optimizer group ----
    lines.append("4. Fractal activation gain (Fractal − Standard) per optimizer group")
    lines.append("-" * W)
    if not fvs_gains.empty and "dataset" in fvs_gains.columns:
        sub_gains = fvs_gains[fvs_gains["dataset"] == dataset].sort_values("fractal_gain", ascending=False)
    elif not fvs_gains.empty:
        sub_gains = fvs_gains.sort_values("fractal_gain", ascending=False)
    else:
        sub_gains = pd.DataFrame()

    if sub_gains.empty:
        lines.append("  (No fractal/standard comparison available)")
    else:
        for _, row in sub_gains.iterrows():
            sign   = "+" if row["fractal_gain"] >= 0 else ""
            better = "▲ FRACTAL better" if row["fractal_gain"] > 0.001 else (
                     "▼ STANDARD better" if row["fractal_gain"] < -0.001 else "≈ TIE")
            lines.append(
                f"  {row['optimizer_group']:<25s}  "
                f"fractal={fmt(row['fractal_acc'])}  "
                f"standard={fmt(row['standard_acc'])}  "
                f"gain={sign}{fmt(row['fractal_gain'])}  {better}"
            )
        mean_gain = float(sub_gains["fractal_gain"].mean())
        n_better  = int((sub_gains["fractal_gain"] > 0).sum())
        n_total   = len(sub_gains)
        lines.append("")
        lines.append(
            f"  Summary: mean gain = {fmt(mean_gain)}  |  "
            f"{n_better}/{n_total} groups benefit from fractal activations"
        )
    lines.append("")

    # ---- 5. Best activation per optimizer group ----
    lines.append("5. Best activation per optimizer group")
    lines.append("-" * W)
    for grp in GROUP_ORDER:
        grp_sub = sub_df[sub_df["optimizer_group"] == grp]
        if grp_sub.empty:
            continue
        act_means = grp_sub.groupby("activation")[RANK_BY].mean().sort_values(ascending=False)
        if act_means.empty:
            continue
        best_act = act_means.index[0]
        best_value = float(act_means.iloc[0])
        fam = activation_family(best_act)
        metric_label = "acc" if RANK_BY == "accuracy" else "f1"
        lines.append(
            f"  {grp:<25s}  best={best_act:<45s}  {metric_label}={fmt(best_value)}  family={fam}"
        )
    lines.append("")

    # ---- 6. Best optimizer group per activation ----
    lines.append("6. Best optimizer group per activation")
    lines.append("-" * W)
    act_list = sub_act["activation"].tolist() if not sub_act.empty else sub_df["activation"].unique().tolist()
    for act in act_list:
        act_sub = sub_df[sub_df["activation"] == act]
        if act_sub.empty:
            continue
        grp_means = act_sub.groupby("optimizer_group")[RANK_BY].mean().sort_values(ascending=False)
        if grp_means.empty:
            continue
        best_grp = grp_means.index[0]
        best_value = float(grp_means.iloc[0])
        fam = activation_family(act)
        metric_label = "acc" if RANK_BY == "accuracy" else "f1"
        lines.append(
            f"  {act:<45s} ({fam:<9s})  "
            f"best_group={best_grp:<25s}  {metric_label}={fmt(best_value)}"
        )
    lines.append("")

    # ---- 7. Fractal vs Standard per individual activation ----
    lines.append("7. Individual fractal activation performance vs standard baseline mean")
    lines.append("-" * W)
    std_mean = float(sub_df[sub_df["activation_family"] == "Standard"]["accuracy"].mean()) \
               if "Standard" in sub_df["activation_family"].values else np.nan
    fractal_acts = sub_act[sub_act["activation_family"] == "Fractal"].sort_values(
        "accuracy_mean", ascending=False
    )
    if not fractal_acts.empty and not np.isnan(std_mean):
        lines.append(f"  Standard activation mean accuracy: {fmt(std_mean)}")
        lines.append("")
        for _, row in fractal_acts.iterrows():
            delta = row["accuracy_mean"] - std_mean
            sign  = "+" if delta >= 0 else ""
            lines.append(
                f"  Rank {int(row['rank_activation']):2d}  {row['activation']:<45s}  "
                f"acc={fmt(row['accuracy_mean'])}  Δvs_std={sign}{fmt(delta)}  "
                f"std_dev={fmt(row['accuracy_std'])}"
            )
    lines.append("")

    lines.append("=" * W)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_combined_activation_report(
    df: pd.DataFrame,
    act_ranking: pd.DataFrame,
    grp_x_act: pd.DataFrame,
    fvs_gains: pd.DataFrame,
    out_path: Path,
) -> None:
    """Cross-dataset activation analysis combined report."""
    lines = []
    W = 120

    lines.append("=" * W)
    lines.append("Combined activation analysis — fractal vs standard × optimizer groups")
    lines.append("=" * W)
    lines.append("")
    lines.append(f"  Results root:   {RESULTS_ROOT}")
    lines.append(f"  Datasets:       {df['dataset'].nunique()}  ({', '.join(sorted(df['dataset'].unique()))})")
    lines.append(f"  Activations:    {df['activation'].nunique()}")
    lines.append(f"  Fractal:        {df[df['activation_family']=='Fractal']['activation'].nunique()}")
    lines.append(f"  Standard:       {df[df['activation_family']=='Standard']['activation'].nunique()}")
    lines.append("")

    # ---- Cross-dataset activation ranking ----
    lines.append("Cross-dataset activation ranking (mean of per-dataset accuracy means)")
    lines.append("-" * W)
    cross_act = (
        act_ranking.groupby(["activation", "activation_family"])
        .agg(
            avg_acc=("accuracy_mean", "mean"),
            std_acc=("accuracy_mean", "std"),
            avg_f1=("f1_macro_mean", "mean"),
            n_datasets=("dataset", "nunique"),
        )
        .reset_index()
        .sort_values("avg_acc" if RANK_BY == "accuracy" else "avg_f1", ascending=False)
        .reset_index(drop=True)
    )
    cross_act.index = cross_act.index + 1
    cross_act.index.name = "rank"
    lines.append(cross_act.to_string())
    lines.append("")

    # ---- Activation family summary ----
    lines.append("Activation family summary — cross-dataset")
    lines.append("-" * W)
    fam_cross = (
        df.groupby("activation_family")
        .agg(
            n_activations=("activation", "nunique"),
            n_runs=("accuracy", "count"),
            avg_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std"),
            avg_f1=("f1_macro", "mean"),
        )
        .reset_index()
        .sort_values("avg_acc" if RANK_BY == "accuracy" else "avg_f1", ascending=False)
    )
    lines.append(fam_cross.to_string(index=False))
    lines.append("")

    # ---- Fractal gain summary ----
    if not fvs_gains.empty:
        lines.append("Fractal gain summary per optimizer group (averaged across datasets)")
        lines.append("-" * W)
        gain_summary = (
            fvs_gains.groupby("optimizer_group")
            .agg(
                mean_gain=("fractal_gain", "mean"),
                std_gain=("fractal_gain", "std"),
                n_datasets=("dataset", "nunique"),
                pct_positive=("fractal_gain", lambda x: float((x > 0).mean())),
            )
            .reset_index()
            .sort_values("mean_gain", ascending=False)
        )
        for _, row in gain_summary.iterrows():
            sign   = "+" if row["mean_gain"] >= 0 else ""
            better = "▲ Fractal better" if row["mean_gain"] > 0.001 else (
                     "▼ Standard better" if row["mean_gain"] < -0.001 else "≈ TIE")
            lines.append(
                f"  {row['optimizer_group']:<25s}  "
                f"mean_gain={sign}{fmt(row['mean_gain'])}  "
                f"std={fmt(row['std_gain'])}  "
                f"pct_positive={row['pct_positive']:.0%}  "
                f"{better}"
            )
        lines.append("")

    # ---- Optimizer group × activation family (cross-dataset) ----
    lines.append("Optimizer group × activation family — cross-dataset mean accuracy")
    lines.append("-" * W)
    if not grp_x_act.empty:
        cross_gxa = (
            grp_x_act.groupby(["optimizer_group", "activation_family"])
            .agg(avg_acc=("accuracy_mean", "mean"), std_acc=("accuracy_mean", "std"))
            .reset_index()
            .sort_values(["optimizer_group", "avg_acc"], ascending=[True, False])
        )
        lines.append(cross_gxa.to_string(index=False))
    lines.append("")

    # ---- Per-dataset fractal gain overview ----
    if not fvs_gains.empty and "dataset" in fvs_gains.columns:
        lines.append("Per-dataset fractal gain overview (all optimizer groups combined)")
        lines.append("-" * W)
        per_dataset = (
            fvs_gains.groupby("dataset")
            .agg(mean_gain=("fractal_gain", "mean"), n_groups_positive=("fractal_gain", lambda x: int((x > 0).sum())))
            .reset_index()
            .sort_values("mean_gain", ascending=False)
        )
        for _, row in per_dataset.iterrows():
            sign = "+" if row["mean_gain"] >= 0 else ""
            lines.append(
                f"  {row['dataset']:<35s}  "
                f"mean_gain={sign}{fmt(row['mean_gain'])}  "
                f"groups_where_fractal_helps={int(row['n_groups_positive'])}"
            )
        lines.append("")

    lines.append("=" * W)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# =============================================================================
# 6) REPORTS
# =============================================================================

def write_dataset_report(
    dataset: str,
    df: pd.DataFrame,
    config_stats: pd.DataFrame,
    opt_stats: pd.DataFrame,
    grp_stats: pd.DataFrame,
    vderiv_df: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = []
    W = 120

    lines.append("=" * W)
    lines.append(f"Classification results — dataset: {dataset}")
    lines.append("=" * W)
    lines.append("")

    sub_df = df[df["dataset"] == dataset]
    n_configs = config_stats[config_stats["dataset"] == dataset]["optimizer"].nunique() \
                if not config_stats.empty else "?"
    n_runs    = len(sub_df)

    lines.append(f"  Individual runs:   {n_runs}")
    lines.append(f"  Optimizers tested: {sub_df['optimizer'].nunique()}")
    lines.append(f"  Activations:       {sub_df['activation'].nunique()}")
    lines.append(f"  Configs (unique):  {len(config_stats[config_stats['dataset'] == dataset])}")
    lines.append("")

    # Optimizer ranking
    lines.append(f"Optimizer ranking (best config per optimizer, ranked by {RANK_DISPLAY_NAME} desc)")
    lines.append("-" * W)
    nn = opt_stats[opt_stats["dataset"] == dataset].sort_values("rank_optimizer")
    rank_cols = ["rank_optimizer", "optimizer", "optimizer_group",
                 "accuracy_mean", "accuracy_std", "accuracy_sem",
                 "accuracy_ci95_low", "accuracy_ci95_high",
                 "f1_macro_mean", "f1_macro_std",
                 "training_time_seconds_mean", "epochs_completed_mean",
                 "vderiv", "activation"]
    rank_cols = [c for c in rank_cols if c in nn.columns]
    lines.append(_df_to_text(nn[rank_cols]))
    lines.append("")

    # Best config per group
    lines.append("Best optimizer per group")
    lines.append("-" * W)
    for grp in GROUP_ORDER:
        g = nn[nn["optimizer_group"] == grp]
        if g.empty:
            continue
        best = g.iloc[0]
        lines.append(
            f"  {grp:<22s}  {best['optimizer']:<40s}  "
            f"acc={fmt(best['accuracy_mean'])} ± {fmt(best['accuracy_std'], 5)}  "
            f"f1={fmt(best['f1_macro_mean'])}  "
            f"time={fmt(best['training_time_seconds_mean'], 3)}s  "
            f"vderiv={fmt(best['vderiv'], 2)}  "
            f"activation={best['activation']}"
        )
    lines.append("")

    # Optimizer group summary
    lines.append("Optimizer group summary")
    lines.append("-" * W)
    grp_sub = grp_stats[grp_stats["dataset"] == dataset].copy()
    if not grp_sub.empty:
        gcols = ["optimizer_group", "n_runs",
                 "accuracy_mean", "accuracy_std", "accuracy_sem",
                 "f1_macro_mean", "training_time_seconds_mean"]
        gcols = [c for c in gcols if c in grp_sub.columns]
        lines.append(_df_to_text(
            grp_sub[gcols].sort_values(RANK_MEAN_COL, ascending=False)
        ))
    lines.append("")

    # Activation family summary
    lines.append("Activation family summary")
    lines.append("-" * W)
    act_summary = (
        sub_df.groupby("activation_family")
        .agg(
            n_runs=("accuracy", "count"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            f1_mean=("f1_macro", "mean"),
        )
        .reset_index()
        .sort_values("acc_mean" if RANK_BY == "accuracy" else "f1_mean", ascending=False)
    )
    lines.append(_df_to_text(act_summary))
    lines.append("")

    # vderiv sensitivity summary
    vd_sub = vderiv_df[vderiv_df["dataset"] == dataset]
    if not vd_sub.empty:
        lines.append("vderiv sensitivity (Herrera & GL-Memory optimizers)")
        lines.append("-" * W)
        vcols = ["optimizer", "vderiv", "n_runs",
                 "accuracy_mean", "accuracy_std", "f1_macro_mean"]
        vcols = [c for c in vcols if c in vd_sub.columns]
        lines.append(_df_to_text(vd_sub.sort_values(["optimizer", "vderiv"])[vcols]))
        lines.append("")

    # Top-5 full config breakdown
    lines.append(f"Top-5 configurations (by {RANK_DISPLAY_NAME}, all configs)")
    lines.append("-" * W)
    cs = config_stats[config_stats["dataset"] == dataset].sort_values(
        [RANK_MEAN_COL, RANK_SECONDARY_MEAN_COL],
        ascending=[False, False],
    ).head(5)
    top_cols = ["optimizer", "activation", "vderiv", "history_size",
                "accuracy_mean", "accuracy_std", "f1_macro_mean",
                "training_time_seconds_mean", "epochs_completed_mean"]
    top_cols = [c for c in top_cols if c in cs.columns]
    lines.append(_df_to_text(cs[top_cols]))
    lines.append("")

    lines.append("=" * W)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_combined_report(
    df: pd.DataFrame,
    opt_stats: pd.DataFrame,
    grp_stats: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = []
    W = 120

    lines.append("=" * W)
    lines.append("Combined analysis report — fractional optimizer × fractal activation classification")
    lines.append("=" * W)
    lines.append("")
    lines.append(f"  Results root:   {RESULTS_ROOT}")
    lines.append(f"  Analysis root:  {ANALYSIS_ROOT}")
    lines.append(f"  Datasets:       {df['dataset'].nunique()}  "
                 f"({', '.join(sorted(df['dataset'].unique()))})")
    lines.append(f"  Optimizers:     {df['optimizer'].nunique()}")
    lines.append(f"  Activations:    {df['activation'].nunique()}")
    lines.append(f"  Total runs:     {len(df)}")
    lines.append("")


    # Cross-dataset rankings
    lines.append(f"Cross-dataset optimizer ranking (mean of per-dataset best-config {RANK_DISPLAY_NAME})")
    lines.append("-" * W)
    cross = (
        opt_stats.groupby("optimizer")
        .agg(
            avg_acc=("accuracy_mean", "mean"),
            std_acc=("accuracy_mean", "std"),
            avg_f1=("f1_macro_mean", "mean"),
            avg_time=("training_time_seconds_mean", "mean"),
            n_datasets=("dataset", "nunique"),
            optimizer_group=("optimizer_group", "first"),
        )
        .reset_index()
        .sort_values("avg_acc" if RANK_BY == "accuracy" else "avg_f1", ascending=False)
        .reset_index(drop=True)
    )
    cross.index = cross.index + 1
    cross.index.name = "rank"
    lines.append(cross.to_string())
    lines.append("")

    # Group summary across datasets
    lines.append("Optimizer group summary across all datasets")
    lines.append("-" * W)
    grp_cross = (
        opt_stats.groupby("optimizer_group")
        .agg(
            avg_acc=("accuracy_mean", "mean"),
            std_acc=("accuracy_mean", "std"),
            avg_f1=("f1_macro_mean", "mean"),
            avg_time=("training_time_seconds_mean", "mean"),
            n_optimizers=("optimizer", "nunique"),
        )
        .reset_index()
        .sort_values("avg_acc" if RANK_BY == "accuracy" else "avg_f1", ascending=False)
    )
    lines.append(grp_cross.to_string(index=False))
    lines.append("")

    # Activation family summary
    lines.append("Activation family summary across all datasets")
    lines.append("-" * W)
    act_cross = (
        df.groupby("activation_family")
        .agg(
            n_runs=("accuracy", "count"),
            avg_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std"),
            avg_f1=("f1_macro", "mean"),
        )
        .reset_index()
        .sort_values("avg_acc" if RANK_BY == "accuracy" else "avg_f1", ascending=False)
    )
    lines.append(act_cross.to_string(index=False))
    lines.append("")

    lines.append("=" * W)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _df_to_text(df: pd.DataFrame, float_digits: int = 5) -> str:
    tmp = df.copy().reset_index(drop=True)
    for col in tmp.select_dtypes(include=[float, np.floating]).columns:
        tmp[col] = tmp[col].apply(lambda x: fmt(x, float_digits))
    return tmp.to_string(index=False)


# =============================================================================
# 7) MAIN
# =============================================================================

def main() -> None:
    ensure_dir(ANALYSIS_ROOT)

    print("=" * 100)
    print("NN Classification experiment analysis  —  21 fractional optimizers")
    print("=" * 100)
    print(f"Primary ranking metric: {RANK_BY}")
    print(f"Loading results from: {RESULTS_ROOT}")

    # ---- Load ----
    nn_files = find_result_files(RESULTS_ROOT)
    print(f"  Found {len(nn_files)} NN result files.")

    nn_records = load_nn_results(nn_files)

    if not nn_records:
        raise RuntimeError(
            f"No NN results found under {RESULTS_ROOT}. "
            "Check that the experiment has been run and the directory exists."
        )

    df = pd.DataFrame(nn_records)
    print(f"  Individual NN runs:      {len(df)}")
    print(f"  Datasets found:          {df['dataset'].unique().tolist()}")
    print(f"  Optimizers found:        {df['optimizer'].nunique()}")
    print(f"  Activations found:       {df['activation'].nunique()}")
    print()

    # ---- Compute statistics ----
    print("Computing statistics ...")
    config_stats  = compute_config_statistics(df)
    opt_stats     = compute_optimizer_statistics(df)
    act_stats     = compute_activation_statistics(df)
    grp_stats     = compute_optimizer_group_statistics(df)
    vderiv_df     = compute_vderiv_sensitivity(df)

    # Activation-focused statistics
    print("Computing activation-focused statistics ...")
    act_ranking       = compute_activation_ranking(df)
    grp_x_act         = compute_optgroup_x_activation_family(df)
    df_without_xpsin  = df[df["activation"] != ACTIVATION_FAMILY_EXCLUDE_XPSIN].copy()
    grp_x_act_without_xpsin = compute_optgroup_x_activation_family(df_without_xpsin)
    fvs_df, fvs_gains = compute_fractal_vs_standard(df)

    # ---- Save combined CSVs ----
    print("Saving combined tables ...")
    df.to_csv(ANALYSIS_ROOT / "combined_individual_runs.csv", index=False)
    config_stats.to_csv(ANALYSIS_ROOT / "combined_config_statistics.csv", index=False)
    opt_stats.to_csv(ANALYSIS_ROOT / "combined_optimizer_statistics.csv", index=False)
    act_stats.to_csv(ANALYSIS_ROOT / "combined_activation_statistics.csv", index=False)
    grp_stats.to_csv(ANALYSIS_ROOT / "combined_optimizer_group_statistics.csv", index=False)
    vderiv_df.to_csv(ANALYSIS_ROOT / "combined_vderiv_sensitivity.csv", index=False)
    act_ranking.to_csv(ANALYSIS_ROOT / "combined_activation_ranking.csv", index=False)
    grp_x_act.to_csv(ANALYSIS_ROOT / "combined_optgroup_x_activation_family.csv", index=False)
    grp_x_act_without_xpsin.to_csv(
        ANALYSIS_ROOT / "combined_optgroup_x_activation_family_without_xpsin.csv",
        index=False,
    )
    fvs_df.to_csv(ANALYSIS_ROOT / "combined_fractal_vs_standard.csv", index=False)
    fvs_gains.to_csv(ANALYSIS_ROOT / "combined_fractal_gains.csv", index=False)

    # ---- Combined plots ----
    if MAKE_PLOTS:
        print("Generating combined plots ...")
        plot_cross_dataset_summary(opt_stats, ANALYSIS_ROOT)
        plot_cross_dataset_group_heatmap(opt_stats, ANALYSIS_ROOT)
        plot_cross_dataset_activation_summary(act_ranking, ANALYSIS_ROOT)
        plot_cross_dataset_optgroup_x_actfamily_heatmap(grp_x_act, ANALYSIS_ROOT)

    # ---- Combined reports ----
    print("Writing combined reports ...")
    write_combined_report(
        df, opt_stats, grp_stats,
        append_dataset_suffix(ANALYSIS_ROOT / "human_readable_report.txt", "all_datasets"),
    )
    write_combined_activation_report(
        df, act_ranking, grp_x_act, fvs_gains,
        append_dataset_suffix(ANALYSIS_ROOT / "activation_analysis_report.txt", "all_datasets"),
    )

    # ---- Per-dataset analysis ----
    datasets = sorted(df["dataset"].unique())
    print(f"\nPer-dataset analysis ({len(datasets)} datasets):")

    for dataset in datasets:
        print(f"  [{dataset}] ...", end="", flush=True)

        d_dir = ANALYSIS_ROOT / dataset
        ensure_dir(d_dir)
        plots_dir = d_dir / "plots"
        if MAKE_PLOTS:
            ensure_dir(plots_dir)

        # Subset stats
        d_config = config_stats[config_stats["dataset"] == dataset]
        d_opt    = opt_stats[opt_stats["dataset"] == dataset]
        d_grp    = grp_stats[grp_stats["dataset"] == dataset]
        d_vd     = vderiv_df[vderiv_df["dataset"] == dataset]

        # Save per-dataset CSVs
        d_config.to_csv(d_dir / "config_statistics.csv", index=False)
        d_opt.to_csv(d_dir / "optimizer_statistics.csv", index=False)
        d_grp.to_csv(d_dir / "optimizer_group_statistics.csv", index=False)
        d_vd.to_csv(d_dir / "vderiv_sensitivity.csv", index=False)

        # Activation-focused per-dataset CSVs
        act_ranking[act_ranking["dataset"] == dataset].to_csv(
            d_dir / "activation_ranking.csv", index=False)
        grp_x_act[grp_x_act["dataset"] == dataset].to_csv(
            d_dir / "optgroup_x_activation_family.csv", index=False)
        grp_x_act_without_xpsin[grp_x_act_without_xpsin["dataset"] == dataset].to_csv(
            d_dir / "optgroup_x_activation_family_without_xpsin.csv", index=False)
        if not fvs_gains.empty and "dataset" in fvs_gains.columns:
            fvs_gains[fvs_gains["dataset"] == dataset].to_csv(
                d_dir / "fractal_vs_standard_gains.csv", index=False)

        # Heatmap pivot
        heat_pivot = build_optimizer_activation_heatmap(df, dataset)
        if not heat_pivot.empty:
            heat_pivot.to_csv(d_dir / "optimizer_activation_heatmap.csv")

        # Per-dataset reports (optimizer-focused + activation-focused)
        write_dataset_report(
            dataset=dataset,
            df=df,
            config_stats=config_stats,
            opt_stats=opt_stats,
            grp_stats=grp_stats,
            vderiv_df=vderiv_df,
            out_path=append_dataset_suffix(d_dir / "human_readable_report.txt", dataset),
        )
        write_activation_report(
            dataset=dataset,
            df=df,
            act_ranking=act_ranking,
            grp_x_act=grp_x_act,
            fvs_df=fvs_df,
            fvs_gains=fvs_gains,
            out_path=append_dataset_suffix(d_dir / "activation_analysis_report.txt", dataset),
        )

        # Plots
        if MAKE_PLOTS:
            # ---- Optimizer-focused plots ----
            plot_accuracy_bar(d_opt, dataset, plots_dir)
            plot_accuracy_boxplot(df, d_opt, dataset, plots_dir)
            plot_f1_bar(d_opt, dataset, plots_dir)
            plot_training_time_bar(d_opt, dataset, plots_dir)
            plot_epochs_bar(d_opt, dataset, plots_dir)
            plot_optimizer_group(grp_stats, dataset, plots_dir)
            plot_vderiv_sensitivity(vderiv_df, dataset, plots_dir)
            plot_accuracy_vs_time(d_opt, dataset, plots_dir)
            plot_accuracy_kde(df, d_opt, dataset, plots_dir)
            if not heat_pivot.empty:
                plot_heatmap(
                    heat_pivot,
                    title=f"[{dataset}]  Optimizer × Activation — mean accuracy (best vderiv per config)",
                    out_path=plots_dir / "optimizer_activation_heatmap_accuracy.png",
                )
            # ---- Activation-focused plots ----
            plot_activation_bar_ranked(act_ranking, dataset, plots_dir)
            plot_activation_boxplot_chart(df, act_ranking, dataset, plots_dir)
            plot_optgroup_x_activation_family_heatmap(grp_x_act, dataset, plots_dir)
            plot_activation_family_x_optgroup_lines(grp_x_act, dataset, plots_dir)
            plot_activation_family_x_optgroup_lines_without_xpsin(
                grp_x_act_without_xpsin, dataset, plots_dir
            )
            plot_all_optimizers_x_activations_heatmap(df, dataset, plots_dir)
            d_gains = (fvs_gains[fvs_gains["dataset"] == dataset]
                       if not fvs_gains.empty and "dataset" in fvs_gains.columns
                       else pd.DataFrame())
            if not d_gains.empty:
                plot_fractal_vs_standard_by_optgroup(d_gains, dataset, plots_dir)

        print(" done.")

    print()
    print("=" * 100)
    print("Analysis complete.")
    print(f"Results written to: {ANALYSIS_ROOT.resolve()}")
    print("=" * 100)


if __name__ == "__main__":
    main()