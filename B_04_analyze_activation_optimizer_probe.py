#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
B_04_analyze_activation_optimizer_probe.py

Analysis and plotting stage for the gradient probe of
B_03_activation_optimizer_gradient_probe.py.

This script trains nothing and imports neither TensorFlow nor the optimizer
modules of the study. It reads the records that the probe wrote, one row per
finished run, and rebuilds every table and every figure from them. Running it
is therefore cheap and repeatable: the expensive part happened once, in the
probe, and everything below is a view of that file.

    B_03_activation_optimizer_gradient_probe.py   writes  <PROBE_DIR>/run_records.csv
    B_04_analyze_activation_optimizer_probe.py    reads   that file and writes its
                                                          own report, tables and figures

WHAT IT PRODUCES
----------------
The five figures of the probe, reproduced from the same numbers with the
spacing and alignment tightened:

    1  gradient_and_update_by_family   what each optimizer family receives and
                                       what it applies, per activation
    2  gradient_coherence              cosine similarity of consecutive
                                       gradients, the quantity the memory
                                       mechanisms average over
    3  truncation_effect               gradient magnitude and accuracy against
                                       the number of retained terms
    4  stability_heatmap               fraction of runs with a non-finite loss,
                                       gradient, update or optimizer slot
    5  herrera_factor                  the factor the Herrera family applies and
                                       the gradient that results

and eight further figures adjacent to those, which the saved records support
without any new computation:

    6  gradient_norm_heatmap           every optimizer against every activation
    7  accuracy_heatmap                the same grid, by test accuracy
    8  damping_ratio                   how much of the gradient each family
                                       discards between receiving and applying
    9  accuracy_vs_gradient            accuracy against gradient magnitude, one
                                       point per configuration
    10 coherence_vs_truncation         where the gradient sequence loses its
                                       coherence as scales are added
    11 setting_sensitivity             accuracy and gradient magnitude against
                                       the fractional order, or against the
                                       learning rate for the standard family
    12 adaptive_state                  the realized mixing weight of the
                                       adaptive family against the coherence it
                                       adapts to, and the GL coefficient sum of
                                       the fixed-order family
    13 run_distribution                the spread over configurations and seeds,
                                       and the wall-clock cost per activation

and the tables, as text in the report, as CSV, and as LaTeX for the appendix:
the deployed comparison by family, the truncation sweep, the Herrera factor
table, the memory-family table, a per-optimizer table over all 21 optimizers, a
setting-sensitivity table, the best configuration per activation, a stability
incident table and a runtime table.

READING THE RESULTS
-------------------
Three quantities carry most of the information and are worth keeping apart.

The gradient norm is what the activation hands the optimizer. The update norm
is what the optimizer then applies to the parameters. Their ratio says how much
of the gradient the optimizer discards, which is where Adam-type rescaling and
the fractional factors become visible.

The coherence is the cosine similarity between the gradients of consecutive
steps. The GL-memory family sums a weighted gradient history and norm-matches
it back, the adaptive family mixes such a sum in with a bounded weight, and
both depend on that number being away from zero. Where it is near zero the
history has no direction to average, the descent safeguard of those optimizers
rejects the memory term, and what the family reports is its fallback path. The
adaptive family also adapts its mixing weight from the relative change between
consecutive gradients, so figure 12 shows the mechanism reacting to the
coherence of figure 2 directly.

Non-finite values are reported from four places at once: the loss, the
gradient, the applied update and the optimizer's own slot variables. The slot
column is the one that catches an accumulator overflowing, which is what a
large gradient multiplied by a Herrera factor and then squared would do.

RUNNING
-------
No command line arguments. Point PROBE_DIR at the directory that
B_03_activation_optimizer_gradient_probe.py wrote and run

    python3 B_04_analyze_activation_optimizer_probe.py

Outputs
-------
    <OUTPUT_DIR>/activation_optimizer_analysis_report.txt
    <OUTPUT_DIR>/cell_summary.csv
    <OUTPUT_DIR>/optimizer_summary.csv
    <OUTPUT_DIR>/setting_sensitivity.csv
    <OUTPUT_DIR>/best_configurations.csv
    <OUTPUT_DIR>/stability_incidents.csv
    <OUTPUT_DIR>/runtime_summary.csv
    <OUTPUT_DIR>/analysis_tables.tex
    <OUTPUT_DIR>/plots/*.png, *.eps
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# 0) SETTINGS
# =============================================================================

# Where the probe wrote its records, and where this script writes its own outputs.
PROBE_DIR = Path("activation_optimizer_gradient_probe")
RECORDS_FILE = PROBE_DIR / "run_records.csv"
ANALYSIS_JSON = PROBE_DIR / "activation_optimizer_analysis.json"

OUTPUT_DIR = Path("activation_optimizer_analysis")
PLOT_DIR = OUTPUT_DIR / "plots"

# -----------------------------------------------------------------------------
# Display order and naming. These mirror B_03_activation_optimizer_gradient_probe.py
# so that the figures of the two scripts can be placed next to each other.
ACTIVATION_ORDER = [
    "relu",
    "tanh",
    "modulated_blancmange_curve",
    "decaying_cosine_function_tf",
    "modified_weierstrass_function_tanh",
    "weierstrass_mandelbrot_function_xpsin",
]

ACTIVATION_DISPLAY: Dict[str, str] = {
    "relu": "relu",
    "tanh": "tanh",
    "modulated_blancmange_curve": "blancmange",
    "decaying_cosine_function_tf": "dec. cosine",
    "modified_weierstrass_function_tanh": "mod. wei. tanh",
    "weierstrass_mandelbrot_function_xpsin": "x+sin",
}

# The deployed number of terms of each activation, used to select the rows that
# describe the configuration the study actually ran. Activations without a
# truncation parameter carry None.
DEPLOYED_TERMS: Dict[str, Optional[int]] = {
    "relu": None,
    "tanh": None,
    "modulated_blancmange_curve": 30,
    "decaying_cosine_function_tf": 75,
    "modified_weierstrass_function_tanh": 100,
    "weierstrass_mandelbrot_function_xpsin": 100,
}

FAMILY_ORDER = ["baseline", "herrera", "memory", "adaptive_memory", "related_work"]

FAMILY_DISPLAY: Dict[str, str] = {
    "baseline": "standard",
    "herrera": "Herrera",
    "memory": "GL memory",
    "adaptive_memory": "adaptive memory",
    "related_work": "related work",
}

OPTIMIZER_ORDER: Dict[str, List[str]] = {
    "baseline": ["SGD", "Adam", "RMSprop", "Adadelta"],
    "herrera": ["FSGD", "FAdam", "FRMSprop", "FAdadelta"],
    "memory": ["MemoryFSGD", "MemoryFRMSprop", "MemoryFAdam", "MemoryFAdadelta"],
    "adaptive_memory": [
        "AdaptiveMemoryFSGD", "AdaptiveMemoryFRMSprop",
        "AdaptiveMemoryFAdam", "AdaptiveMemoryFAdadelta",
    ],
    "related_work": ["AdaGL", "FCSGD_GL", "FCAdam_GL", "AOFGD_SGD", "AOFGD_Adam"],
}

# The bound of the adaptive mixing weight, quoted in the report.
ADAPTIVE_MAX_MIX = 0.30

# -----------------------------------------------------------------------------
# Analysis settings.
TOP_CONFIGURATIONS_PER_ACTIVATION = 5
HEATMAP_ANNOTATE_LIMIT = 200       # annotate cells while the grid stays readable
SCATTER_POINT_SIZE = 26.0
DISTRIBUTION_SHOWFLIERS = False

# -----------------------------------------------------------------------------
# Plotting. Style follows the other scripts of the study. Every figure uses the
# constrained layout engine with explicit padding, so that panels, labels and
# legends keep the same spacing across figures of different sizes.
FONT_SCALE = 1.25
DPI_PNG = 400
SAVE_EPS = True

LAYOUT_H_PAD = 0.16
LAYOUT_W_PAD = 0.16
LAYOUT_HSPACE = 0.10
LAYOUT_WSPACE = 0.10

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

ACTIVATION_LINESTYLES: Dict[str, Any] = {
    "relu": "solid",
    "tanh": (0, (5, 2)),
    "modulated_blancmange_curve": (0, (1, 1.4)),
    "decaying_cosine_function_tf": (0, (5, 2)),
    "modified_weierstrass_function_tanh": (0, (7, 2, 1.5, 2)),
    "weierstrass_mandelbrot_function_xpsin": "solid",
}

FAMILY_COLORS: Dict[str, str] = {
    "baseline": COLOR_PALETTE_A[0],
    "herrera": COLOR_PALETTE_A[1],
    "memory": COLOR_PALETTE_A[2],
    "adaptive_memory": COLOR_PALETTE_A[3],
    "related_work": COLOR_PALETTE_A[6],
}

FAMILY_MARKERS: Dict[str, str] = {
    "baseline": "o",
    "herrera": "s",
    "memory": "^",
    "adaptive_memory": "D",
    "related_work": "P",
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
    "axes.grid": False,
    "savefig.dpi": DPI_PNG,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

RULE = "=" * 124
THIN_RULE = "-" * 124


# =============================================================================
# 1) LOADING
# =============================================================================

@dataclass
class RunRecord:
    """One finished run, as the probe wrote it."""

    dataset: str = ""
    activation: str = ""
    activation_display: str = ""
    activation_family: str = ""
    terms: Optional[int] = None
    optimizer: str = ""
    optimizer_family: str = ""
    setting_kind: str = ""
    setting: float = float("nan")
    seed: int = 0
    steps: int = 0
    grad_norm_mean: float = float("nan")
    grad_norm_median: float = float("nan")
    grad_norm_p95: float = float("nan")
    grad_norm_max: float = float("nan")
    steps_above_grad_threshold: int = 0
    update_norm_mean: float = float("nan")
    update_norm_median: float = float("nan")
    update_norm_max: float = float("nan")
    grad_cosine_mean: float = float("nan")
    grad_cosine_median: float = float("nan")
    grad_cosine_abs_mean: float = float("nan")
    nonfinite_gradient_steps: int = 0
    nonfinite_loss_steps: int = 0
    nonfinite_update_steps: int = 0
    optimizer_slot_nonfinite: int = 0
    first_slot_nonfinite_step: int = -1
    herrera_factor_median: float = float("nan")
    herrera_factor_max: float = float("nan")
    gl_coefficient_sum: float = float("nan")
    adaptive_mix_mean: float = float("nan")
    adaptive_stability_mean: float = float("nan")
    final_train_loss: float = float("nan")
    test_accuracy: float = float("nan")
    preactivation_p99_abs: float = float("nan")
    seconds: float = float("nan")

    @property
    def nonfinite_total(self) -> int:
        return int(
            self.nonfinite_gradient_steps
            + self.nonfinite_loss_steps
            + self.nonfinite_update_steps
            + self.optimizer_slot_nonfinite
        )

    @property
    def damping_ratio(self) -> float:
        """Gradient norm divided by the applied update norm."""
        if not np.isfinite(self.grad_norm_mean) or not np.isfinite(self.update_norm_mean):
            return float("nan")
        if self.update_norm_mean <= 0.0:
            return float("nan")
        return float(self.grad_norm_mean / self.update_norm_mean)


class RecordLoader:
    """
    Reads the records file.

    The loader goes by column name, coerces each field to the type the record
    declares, and tolerates both extra columns and missing ones. Anything it
    had to substitute is reported rather than passed over silently.
    """

    INT_FIELDS = {
        "seed", "steps", "steps_above_grad_threshold", "nonfinite_gradient_steps",
        "nonfinite_loss_steps", "nonfinite_update_steps", "optimizer_slot_nonfinite",
        "first_slot_nonfinite_step",
    }
    STR_FIELDS = {
        "dataset", "activation", "activation_display", "activation_family",
        "optimizer", "optimizer_family", "setting_kind",
    }

    def __init__(self, path: Path = RECORDS_FILE) -> None:
        self.path = Path(path)
        self.missing_columns: List[str] = []
        self.extra_columns: List[str] = []
        self.skipped_rows = 0

    def load(self) -> List[RunRecord]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"no records at {self.path}. Run "
                f"B_05_activation_optimizer_gradient_probe.py first, or point "
                f"PROBE_DIR at the directory that holds run_records.csv."
            )

        expected = {item.name for item in fields(RunRecord)}
        records: List[RunRecord] = []

        with open(self.path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = set(reader.fieldnames or [])
            self.missing_columns = sorted(expected - header)
            self.extra_columns = sorted(header - expected)

            for row in reader:
                try:
                    records.append(self._convert(row))
                except Exception:
                    self.skipped_rows += 1

        return records

    def _convert(self, row: Dict[str, str]) -> RunRecord:
        values: Dict[str, Any] = {}
        for item in fields(RunRecord):
            raw = (row.get(item.name) or "").strip()
            if item.name == "terms":
                values[item.name] = None if raw in ("", "None", "nan") else int(float(raw))
            elif item.name in self.STR_FIELDS:
                values[item.name] = raw
            elif item.name in self.INT_FIELDS:
                values[item.name] = int(float(raw)) if raw not in ("", "None") else 0
            else:
                values[item.name] = (
                    float(raw) if raw not in ("", "None") else float("nan")
                )
        return RunRecord(**values)

    def provenance(self) -> Dict[str, Any]:
        """The settings the probe recorded, if its JSON record is next to the CSV."""
        if not ANALYSIS_JSON.exists():
            return {}
        try:
            with open(ANALYSIS_JSON, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {}
        return {
            "settings": payload.get("settings", {}),
            "environment": payload.get("environment", {}),
            "failures": len(payload.get("failures", []) or []),
        }


# =============================================================================
# 2) AGGREGATION
# =============================================================================

def _mean(values: Iterable[float]) -> float:
    array = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)], dtype=np.float64
    )
    return float(np.mean(array)) if array.size else float("nan")


def _std(values: Iterable[float]) -> float:
    array = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)], dtype=np.float64
    )
    return float(np.std(array)) if array.size > 1 else 0.0


def _max(values: Iterable[float]) -> float:
    array = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)], dtype=np.float64
    )
    return float(np.max(array)) if array.size else float("nan")


@dataclass
class CellSummary:
    """One (activation, terms, optimizer, setting) cell, averaged over seeds."""

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
    damping_ratio: float
    grad_cosine_mean: float
    nonfinite_total: int
    slot_nonfinite_total: int
    herrera_factor_median: float
    gl_coefficient_sum: float
    adaptive_mix_mean: float
    adaptive_stability_mean: float
    test_accuracy_mean: float
    test_accuracy_std: float
    seconds_mean: float


@dataclass
class OptimizerSummary:
    """One optimizer, averaged over its settings and seeds, per activation."""

    optimizer: str
    optimizer_family: str
    activation: str
    activation_display: str
    settings: int
    runs: int
    grad_norm_mean: float
    update_norm_mean: float
    damping_ratio: float
    grad_cosine_mean: float
    nonfinite_total: int
    runs_with_nonfinite: int
    test_accuracy_mean: float
    test_accuracy_std: float
    best_setting: float
    best_setting_accuracy: float


class Aggregator:
    """Groups the records into the views the tables and figures need."""

    def __init__(self, records: Sequence[RunRecord]) -> None:
        self.records = list(records)

    # ---- selection ----------------------------------------------------------

    def activations(self) -> List[str]:
        present = {r.activation for r in self.records}
        ordered = [name for name in ACTIVATION_ORDER if name in present]
        ordered.extend(sorted(present - set(ordered)))
        return ordered

    def families(self) -> List[str]:
        present = {r.optimizer_family for r in self.records}
        ordered = [name for name in FAMILY_ORDER if name in present]
        ordered.extend(sorted(present - set(ordered)))
        return ordered

    def optimizers(self) -> List[str]:
        present = {r.optimizer for r in self.records}
        ordered: List[str] = []
        for family in self.families():
            for name in OPTIMIZER_ORDER.get(family, []):
                if name in present:
                    ordered.append(name)
        ordered.extend(sorted(present - set(ordered)))
        return ordered

    def truncations(self, activation: str) -> List[Optional[int]]:
        values = {r.terms for r in self.records if r.activation == activation}
        return sorted(values, key=lambda t: (t is None, t))

    def deployed(self) -> List[RunRecord]:
        """
        The rows that describe the deployed configuration: for each activation
        the largest truncation present, which is its deployed length unless the
        probe was restricted to shorter ones.
        """
        out: List[RunRecord] = []
        for activation in self.activations():
            expected = DEPLOYED_TERMS.get(activation, None)
            present = [t for t in self.truncations(activation) if t is not None]
            if expected is not None and expected in present:
                target: Optional[int] = expected
            elif present:
                target = max(present)
            else:
                target = None
            out.extend(
                r for r in self.records
                if r.activation == activation and r.terms == target
            )
        return out

    def deployed_terms_present(self, activation: str) -> Optional[int]:
        rows = [r for r in self.deployed() if r.activation == activation]
        return rows[0].terms if rows else None

    def select(
        self,
        records: Optional[Sequence[RunRecord]] = None,
        activation: Optional[str] = None,
        family: Optional[str] = None,
        optimizer: Optional[str] = None,
        terms: Optional[int] = "any",  # type: ignore[assignment]
        setting: Optional[float] = None,
    ) -> List[RunRecord]:
        source = self.records if records is None else records
        out = []
        for record in source:
            if activation is not None and record.activation != activation:
                continue
            if family is not None and record.optimizer_family != family:
                continue
            if optimizer is not None and record.optimizer != optimizer:
                continue
            if terms != "any" and record.terms != terms:
                continue
            if setting is not None and not math.isclose(
                record.setting, setting, rel_tol=1e-9, abs_tol=1e-12
            ):
                continue
            out.append(record)
        return out

    # ---- summaries ----------------------------------------------------------

    def cell_summaries(self) -> List[CellSummary]:
        groups: Dict[Tuple[str, Optional[int], str, float], List[RunRecord]] = {}
        for record in self.records:
            groups.setdefault(
                (record.activation, record.terms, record.optimizer, record.setting), []
            ).append(record)

        out: List[CellSummary] = []
        for (activation, terms, optimizer, setting), group in groups.items():
            first = group[0]
            out.append(
                CellSummary(
                    activation=activation,
                    activation_display=ACTIVATION_DISPLAY.get(
                        activation, first.activation_display or activation
                    ),
                    terms=terms,
                    optimizer=optimizer,
                    optimizer_family=first.optimizer_family,
                    setting_kind=first.setting_kind,
                    setting=float(setting),
                    seeds=len(group),
                    grad_norm_mean=_mean(r.grad_norm_mean for r in group),
                    grad_norm_max=_max(r.grad_norm_max for r in group),
                    update_norm_mean=_mean(r.update_norm_mean for r in group),
                    damping_ratio=_mean(r.damping_ratio for r in group),
                    grad_cosine_mean=_mean(r.grad_cosine_mean for r in group),
                    nonfinite_total=int(sum(r.nonfinite_total for r in group)),
                    slot_nonfinite_total=int(
                        sum(r.optimizer_slot_nonfinite for r in group)
                    ),
                    herrera_factor_median=_mean(r.herrera_factor_median for r in group),
                    gl_coefficient_sum=_mean(r.gl_coefficient_sum for r in group),
                    adaptive_mix_mean=_mean(r.adaptive_mix_mean for r in group),
                    adaptive_stability_mean=_mean(
                        r.adaptive_stability_mean for r in group
                    ),
                    test_accuracy_mean=_mean(r.test_accuracy for r in group),
                    test_accuracy_std=_std([r.test_accuracy for r in group]),
                    seconds_mean=_mean(r.seconds for r in group),
                )
            )
        out.sort(
            key=lambda s: (
                self.activations().index(s.activation)
                if s.activation in self.activations() else 99,
                s.terms if s.terms is not None else -1,
                self.optimizers().index(s.optimizer)
                if s.optimizer in self.optimizers() else 99,
                s.setting,
            )
        )
        return out

    def optimizer_summaries(self, records: Optional[Sequence[RunRecord]] = None) -> List[OptimizerSummary]:
        source = self.deployed() if records is None else list(records)
        out: List[OptimizerSummary] = []
        for optimizer in self.optimizers():
            for activation in self.activations():
                group = [
                    r for r in source
                    if r.optimizer == optimizer and r.activation == activation
                ]
                if not group:
                    continue
                by_setting: Dict[float, List[RunRecord]] = {}
                for record in group:
                    by_setting.setdefault(record.setting, []).append(record)
                accuracies = {
                    setting: _mean(r.test_accuracy for r in rows)
                    for setting, rows in by_setting.items()
                }
                best_setting = max(
                    accuracies,
                    key=lambda s: (
                        accuracies[s] if np.isfinite(accuracies[s]) else -np.inf
                    ),
                )
                out.append(
                    OptimizerSummary(
                        optimizer=optimizer,
                        optimizer_family=group[0].optimizer_family,
                        activation=activation,
                        activation_display=ACTIVATION_DISPLAY.get(activation, activation),
                        settings=len(by_setting),
                        runs=len(group),
                        grad_norm_mean=_mean(r.grad_norm_mean for r in group),
                        update_norm_mean=_mean(r.update_norm_mean for r in group),
                        damping_ratio=_mean(r.damping_ratio for r in group),
                        grad_cosine_mean=_mean(r.grad_cosine_mean for r in group),
                        nonfinite_total=int(sum(r.nonfinite_total for r in group)),
                        runs_with_nonfinite=int(
                            sum(1 for r in group if r.nonfinite_total > 0)
                        ),
                        test_accuracy_mean=_mean(r.test_accuracy for r in group),
                        test_accuracy_std=_std([r.test_accuracy for r in group]),
                        best_setting=float(best_setting),
                        best_setting_accuracy=float(accuracies[best_setting]),
                    )
                )
        return out


# =============================================================================
# 3) PLOT HELPERS
# =============================================================================

class FigureBuilder:
    """
    Shared figure construction, so that every figure in this script has the
    same margins, the same padding between panels and the same legend
    placement. The constrained layout engine does the alignment; the padding
    constants in section 0 set how much room it leaves.
    """

    @staticmethod
    def create(
        nrows: int = 1,
        ncols: int = 1,
        figsize: Tuple[float, float] = (12.0, 5.0),
        **kwargs: Any,
    ) -> Tuple[plt.Figure, Any]:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, layout="constrained", **kwargs
        )
        engine = fig.get_layout_engine()
        if engine is not None:
            try:
                engine.set(
                    h_pad=LAYOUT_H_PAD, w_pad=LAYOUT_W_PAD,
                    hspace=LAYOUT_HSPACE, wspace=LAYOUT_WSPACE,
                )
            except Exception:
                pass
        return fig, axes

    @staticmethod
    def legend_below(
        fig: plt.Figure,
        handles: Sequence[Any],
        labels: Sequence[str],
        ncol: int,
    ) -> None:
        """A single figure legend under the panels, outside the axes."""
        if not handles:
            return
        try:
            fig.legend(
                handles, labels, loc="outside lower center", ncol=ncol,
                frameon=False, fontsize=8.5 * FONT_SCALE,
            )
        except (TypeError, ValueError):
            fig.legend(
                handles, labels, loc="lower center", ncol=ncol,
                frameon=False, fontsize=8.5 * FONT_SCALE,
                bbox_to_anchor=(0.5, -0.04),
            )

    @staticmethod
    def legend_above(
        fig: plt.Figure,
        handles: Sequence[Any],
        labels: Sequence[str],
        ncol: int,
    ) -> None:
        if not handles:
            return
        try:
            fig.legend(
                handles, labels, loc="outside upper center", ncol=ncol,
                frameon=False, fontsize=8.5 * FONT_SCALE,
            )
        except (TypeError, ValueError):
            fig.legend(
                handles, labels, loc="upper center", ncol=ncol,
                frameon=False, fontsize=8.5 * FONT_SCALE,
                bbox_to_anchor=(0.5, 1.04),
            )

    @staticmethod
    def grid(ax: plt.Axes, axis: str = "both", which: str = "major") -> None:
        ax.grid(True, alpha=0.25, axis=axis, which=which, linewidth=0.7)
        ax.set_axisbelow(True)

    @staticmethod
    def category_labels(
        ax: plt.Axes,
        labels: Sequence[str],
        rotation: float = 20.0,
        fontsize: Optional[float] = None,
    ) -> None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(
            labels,
            rotation=rotation,
            ha="right" if rotation else "center",
            rotation_mode="anchor" if rotation else None,
            fontsize=fontsize if fontsize else 9 * FONT_SCALE,
        )

    @staticmethod
    def save(fig: plt.Figure, out_base: Path) -> None:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_base.with_suffix(".png"), dpi=DPI_PNG)
        if SAVE_EPS:
            fig.savefig(out_base.with_suffix(".eps"), format="eps")
        plt.close(fig)


def safe_log_floor(values: Sequence[float], factor: float = 10.0) -> float:
    """A positive lower limit for a logarithmic axis that has zeros in it."""
    positive = [v for v in values if np.isfinite(v) and v > 0.0]
    return (min(positive) / factor) if positive else 1e-12


# =============================================================================
# 4) FIGURES
# =============================================================================

class ProbeFigures:
    """
    The five figures of the probe and eight adjacent ones, all from the saved
    records.
    """

    def __init__(self, aggregator: Aggregator) -> None:
        self.data = aggregator
        self.records = aggregator.records
        self.cells = aggregator.cell_summaries()
        self.activation_order = aggregator.activations()
        self.family_order = aggregator.families()
        self.optimizer_order = aggregator.optimizers()
        self.dataset = self.records[0].dataset if self.records else ""

    # ---- helpers ------------------------------------------------------------

    def _display(self, activation: str) -> str:
        return ACTIVATION_DISPLAY.get(activation, activation)

    def _activation_labels(self) -> List[str]:
        return [self._display(a) for a in self.activation_order]

    def _family_handles(self) -> Tuple[List[Any], List[str]]:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS.get(f, COLOR_PALETTE_A[8]))
            for f in self.family_order
        ]
        labels = [FAMILY_DISPLAY.get(f, f) for f in self.family_order]
        return handles, labels

    def _grouped_means(
        self,
        records: Sequence[RunRecord],
        attribute: str,
        reducer: Any = _mean,
    ) -> Dict[str, List[float]]:
        """attribute averaged per family, in activation order."""
        out: Dict[str, List[float]] = {}
        for family in self.family_order:
            row: List[float] = []
            for activation in self.activation_order:
                group = [
                    r for r in records
                    if r.activation == activation and r.optimizer_family == family
                ]
                row.append(
                    reducer(getattr(r, attribute) for r in group) if group else float("nan")
                )
            out[family] = row
        return out

    # ---- 1) gradient and update by family -----------------------------------

    def gradient_and_update_by_family(self) -> None:
        records = self.data.deployed()
        if not records:
            return
        gradients = self._grouped_means(records, "grad_norm_mean")
        updates = self._grouped_means(records, "update_norm_mean")

        fig, axes = FigureBuilder.create(
            2, 1, figsize=(max(10.0, 1.7 * len(self.activation_order)), 8.6),
            sharex=True,
        )
        width = 0.8 / max(len(self.family_order), 1)
        for index, family in enumerate(self.family_order):
            positions = (
                np.arange(len(self.activation_order))
                + (index - (len(self.family_order) - 1) / 2) * width
            )
            color = FAMILY_COLORS.get(family, COLOR_PALETTE_A[8])
            axes[0].bar(positions, gradients[family], width=width * 0.9, color=color)
            axes[1].bar(positions, updates[family], width=width * 0.9, color=color)

        for ax, title, label, series in (
            (axes[0], "Gradient the optimizer receives",
             r"mean global gradient norm $\|g\|$", gradients),
            (axes[1], "Update the optimizer applies",
             r"mean update norm $\|\Delta\theta\|$", updates),
        ):
            ax.set_yscale("log")
            ax.set_ylabel(label)
            ax.set_title(title)
            ax.set_ylim(bottom=safe_log_floor(
                [v for row in series.values() for v in row]
            ))
            FigureBuilder.grid(ax, axis="y", which="both")
        FigureBuilder.category_labels(axes[1], self._activation_labels())

        handles, labels = self._family_handles()
        FigureBuilder.legend_below(fig, handles, labels, ncol=len(self.family_order))
        fig.suptitle(
            f"What each optimizer family receives and applies, {self.dataset}, "
            "deployed truncations"
        )
        FigureBuilder.save(fig, PLOT_DIR / "gradient_and_update_by_family")

    # ---- 2) coherence -------------------------------------------------------

    def gradient_coherence(self) -> None:
        records = self.data.deployed()
        if not records:
            return
        coherence = self._grouped_means(records, "grad_cosine_mean")

        fig, ax = FigureBuilder.create(
            figsize=(max(9.0, 1.6 * len(self.activation_order)), 5.4)
        )
        width = 0.8 / max(len(self.family_order), 1)
        for index, family in enumerate(self.family_order):
            positions = (
                np.arange(len(self.activation_order))
                + (index - (len(self.family_order) - 1) / 2) * width
            )
            ax.bar(
                positions, coherence[family], width=width * 0.9,
                color=FAMILY_COLORS.get(family, COLOR_PALETTE_A[8]),
            )
        ax.axhline(0.0, color="#444444", linewidth=1.0)
        FigureBuilder.category_labels(ax, self._activation_labels())
        ax.set_ylabel(r"mean $\cos(g_t, g_{t-1})$")
        ax.set_title(
            "Coherence of the gradient sequence, which the memory mechanisms average over"
        )
        FigureBuilder.grid(ax, axis="y")

        handles, labels = self._family_handles()
        FigureBuilder.legend_below(fig, handles, labels, ncol=len(self.family_order))
        FigureBuilder.save(fig, PLOT_DIR / "gradient_coherence")

    # ---- 3) truncation effect -----------------------------------------------

    def truncation_effect(self) -> None:
        series = [
            a for a in self.activation_order
            if len([t for t in self.data.truncations(a) if t is not None]) > 1
        ]
        if not series:
            return
        fig, axes = FigureBuilder.create(
            2, len(series), figsize=(3.9 * len(series), 8.2), sharex="col",
        )
        axes = np.atleast_2d(axes)
        if axes.shape[0] == 1:
            axes = axes.reshape(2, -1)

        for column, activation in enumerate(series):
            truncations = [t for t in self.data.truncations(activation) if t is not None]
            for family in self.family_order:
                gradients, accuracies = [], []
                for terms in truncations:
                    group = self.data.select(
                        activation=activation, family=family, terms=terms
                    )
                    gradients.append(
                        _mean(r.grad_norm_mean for r in group) if group else float("nan")
                    )
                    accuracies.append(
                        _mean(r.test_accuracy for r in group) if group else float("nan")
                    )
                color = FAMILY_COLORS.get(family, COLOR_PALETTE_A[8])
                marker = FAMILY_MARKERS.get(family, "o")
                axes[0][column].plot(truncations, gradients, marker=marker,
                                     markersize=4.5, linewidth=1.3, color=color)
                axes[1][column].plot(truncations, accuracies, marker=marker,
                                     markersize=4.5, linewidth=1.3, color=color)
            axes[0][column].set_yscale("log")
            axes[0][column].set_title(self._display(activation))
            FigureBuilder.grid(axes[0][column], which="both")
            FigureBuilder.grid(axes[1][column])
            axes[1][column].set_xlabel("retained terms")
            axes[0][column].set_xticks(truncations)
            axes[1][column].set_xticks(truncations)
        axes[0][0].set_ylabel(r"mean $\|g\|$")
        axes[1][0].set_ylabel("test accuracy")

        handles, labels = self._family_handles()
        FigureBuilder.legend_below(fig, handles, labels, ncol=len(self.family_order))
        fig.suptitle(f"Effect of the truncation length on training, {self.dataset}")
        FigureBuilder.save(fig, PLOT_DIR / "truncation_effect")

    # ---- 4) stability heatmap -----------------------------------------------

    def stability_heatmap(self) -> None:
        records = self.data.deployed()
        if not records:
            return
        matrix = np.full((len(self.optimizer_order), len(self.activation_order)), np.nan)
        for row, optimizer in enumerate(self.optimizer_order):
            for column, activation in enumerate(self.activation_order):
                group = [
                    r for r in records
                    if r.optimizer == optimizer and r.activation == activation
                ]
                if not group:
                    continue
                matrix[row, column] = sum(
                    1 for r in group if r.nonfinite_total > 0
                ) / len(group)

        fig, ax = FigureBuilder.create(
            figsize=(1.45 * len(self.activation_order) + 4.6,
                     0.34 * len(self.optimizer_order) + 2.8)
        )
        image = ax.imshow(matrix, cmap=HEATMAP_CMAP, aspect="auto", vmin=0.0, vmax=1.0)
        self._annotate_heatmap(ax, matrix, fmt="{:.2f}", skip_zero=True)
        self._label_heatmap(ax, "Fraction of runs with a non-finite value")
        bar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
        bar.set_label("fraction of runs")
        FigureBuilder.save(fig, PLOT_DIR / "stability_heatmap")

    # ---- 5) Herrera factor --------------------------------------------------

    def herrera_factor(self) -> None:
        records = [
            r for r in self.data.deployed() if r.optimizer_family == "herrera"
        ]
        if not records:
            return
        orders = sorted({r.setting for r in records})
        fig, axes = FigureBuilder.create(1, 2, figsize=(13.2, 5.2))

        for activation in self.activation_order:
            factors, gradients = [], []
            for order in orders:
                group = [
                    r for r in records
                    if r.activation == activation
                    and math.isclose(r.setting, order, rel_tol=1e-9)
                ]
                factors.append(
                    _mean(r.herrera_factor_median for r in group) if group else float("nan")
                )
                gradients.append(
                    _mean(r.grad_norm_mean for r in group) if group else float("nan")
                )
            color = ACTIVATION_COLORS.get(activation, COLOR_PALETTE_A[8])
            style = ACTIVATION_LINESTYLES.get(activation, "solid")
            marker = ACTIVATION_MARKERS.get(activation, "o")
            axes[0].plot(orders, factors, marker=marker, markersize=5.0,
                         color=color, linestyle=style, label=self._display(activation))
            axes[1].plot(orders, gradients, marker=marker, markersize=5.0,
                         color=color, linestyle=style, label=self._display(activation))

        axes[0].set_yscale("log")
        axes[0].set_xlabel("fractional order")
        axes[0].set_ylabel("median Herrera factor at the final weights")
        axes[0].set_title("The factor the Herrera family applies")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("fractional order")
        axes[1].set_ylabel(r"mean $\|g\|$")
        axes[1].set_title("The gradient that results")
        for ax in axes:
            ax.set_xticks(orders)
            FigureBuilder.grid(ax, which="both")

        handles, labels = axes[0].get_legend_handles_labels()
        FigureBuilder.legend_below(fig, handles, labels, ncol=min(len(labels), 6))
        fig.suptitle("Herrera-type optimizers: scaling factor and resulting gradient")
        FigureBuilder.save(fig, PLOT_DIR / "herrera_factor")

    # ---- 6) and 7) heatmaps over the whole grid -----------------------------

    def _annotate_heatmap(
        self,
        ax: plt.Axes,
        matrix: np.ndarray,
        fmt: str,
        skip_zero: bool = False,
    ) -> None:
        if matrix.size > HEATMAP_ANNOTATE_LIMIT:
            return
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if not np.isfinite(value):
                    continue
                if skip_zero and value <= 0.0:
                    continue
                ax.text(
                    column, row, fmt.format(value), ha="center", va="center",
                    fontsize=7 * FONT_SCALE, color="#1a1a1a",
                )

    def _label_heatmap(self, ax: plt.Axes, title: str) -> None:
        ax.set_xticks(np.arange(len(self.activation_order)))
        ax.set_xticklabels(
            self._activation_labels(), rotation=20, ha="right",
            rotation_mode="anchor",
        )
        ax.set_yticks(np.arange(len(self.optimizer_order)))
        ax.set_yticklabels(self.optimizer_order, fontsize=7.5 * FONT_SCALE)
        ax.set_title(title)
        # Thin separators between the optimizer families, so the blocks read as
        # blocks rather than as twenty-one unrelated rows.
        boundaries: List[float] = []
        seen = None
        for index, optimizer in enumerate(self.optimizer_order):
            family = next(
                (f for f, members in OPTIMIZER_ORDER.items() if optimizer in members),
                None,
            )
            if seen is not None and family != seen:
                boundaries.append(index - 0.5)
            seen = family
        for boundary in boundaries:
            ax.axhline(boundary, color="#ffffff", linewidth=2.0)
        ax.set_xticks(np.arange(-0.5, len(self.activation_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(self.optimizer_order), 1), minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=0.5)
        ax.tick_params(which="minor", length=0)

    def gradient_norm_heatmap(self) -> None:
        records = self.data.deployed()
        if not records:
            return
        matrix = np.full((len(self.optimizer_order), len(self.activation_order)), np.nan)
        for row, optimizer in enumerate(self.optimizer_order):
            for column, activation in enumerate(self.activation_order):
                group = [
                    r for r in records
                    if r.optimizer == optimizer and r.activation == activation
                ]
                if group:
                    value = _mean(r.grad_norm_mean for r in group)
                    matrix[row, column] = (
                        math.log10(value) if np.isfinite(value) and value > 0 else np.nan
                    )

        fig, ax = FigureBuilder.create(
            figsize=(1.45 * len(self.activation_order) + 4.8,
                     0.34 * len(self.optimizer_order) + 2.8)
        )
        image = ax.imshow(matrix, cmap=HEATMAP_CMAP, aspect="auto")
        self._annotate_heatmap(ax, matrix, fmt="{:.1f}")
        self._label_heatmap(
            ax, "Mean global gradient norm per step, $\\log_{10}$"
        )
        bar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
        bar.set_label(r"$\log_{10}\|g\|$")
        FigureBuilder.save(fig, PLOT_DIR / "gradient_norm_heatmap")

    def accuracy_heatmap(self) -> None:
        records = self.data.deployed()
        if not records:
            return
        matrix = np.full((len(self.optimizer_order), len(self.activation_order)), np.nan)
        for row, optimizer in enumerate(self.optimizer_order):
            for column, activation in enumerate(self.activation_order):
                group = [
                    r for r in records
                    if r.optimizer == optimizer and r.activation == activation
                ]
                if group:
                    matrix[row, column] = _mean(r.test_accuracy for r in group)

        fig, ax = FigureBuilder.create(
            figsize=(1.45 * len(self.activation_order) + 4.8,
                     0.34 * len(self.optimizer_order) + 2.8)
        )
        image = ax.imshow(matrix, cmap=HEATMAP_CMAP, aspect="auto")
        self._annotate_heatmap(ax, matrix, fmt="{:.3f}")
        self._label_heatmap(ax, "Mean test accuracy over the four settings and the seeds")
        bar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
        bar.set_label("test accuracy")
        FigureBuilder.save(fig, PLOT_DIR / "accuracy_heatmap")

    # ---- 8) damping ---------------------------------------------------------

    def damping_ratio(self) -> None:
        records = self.data.deployed()
        if not records:
            return
        ratios = self._grouped_means(records, "damping_ratio")

        fig, ax = FigureBuilder.create(
            figsize=(max(9.0, 1.6 * len(self.activation_order)), 5.4)
        )
        width = 0.8 / max(len(self.family_order), 1)
        for index, family in enumerate(self.family_order):
            positions = (
                np.arange(len(self.activation_order))
                + (index - (len(self.family_order) - 1) / 2) * width
            )
            ax.bar(
                positions, ratios[family], width=width * 0.9,
                color=FAMILY_COLORS.get(family, COLOR_PALETTE_A[8]),
            )
        ax.set_yscale("log")
        ax.set_ylim(bottom=safe_log_floor([v for row in ratios.values() for v in row]))
        FigureBuilder.category_labels(ax, self._activation_labels())
        ax.set_ylabel(r"mean $\|g\| \, / \, \|\Delta\theta\|$")
        ax.set_title(
            "How much of the gradient each family discards between receiving and applying it"
        )
        FigureBuilder.grid(ax, axis="y", which="both")

        handles, labels = self._family_handles()
        FigureBuilder.legend_below(fig, handles, labels, ncol=len(self.family_order))
        FigureBuilder.save(fig, PLOT_DIR / "damping_ratio")

    # ---- 9) accuracy against gradient magnitude ------------------------------

    def accuracy_vs_gradient(self) -> None:
        cells = [
            c for c in self.cells
            if c.terms == self.data.deployed_terms_present(c.activation)
        ]
        if not cells:
            return
        fig, axes = FigureBuilder.create(1, 2, figsize=(13.4, 5.4))

        for activation in self.activation_order:
            selected = [c for c in cells if c.activation == activation]
            if not selected:
                continue
            color = ACTIVATION_COLORS.get(activation, COLOR_PALETTE_A[8])
            marker = ACTIVATION_MARKERS.get(activation, "o")
            axes[0].scatter(
                [c.grad_norm_mean for c in selected],
                [c.test_accuracy_mean for c in selected],
                s=SCATTER_POINT_SIZE, color=color, marker=marker,
                edgecolors="none", label=self._display(activation),
            )
            axes[1].scatter(
                [c.grad_cosine_mean for c in selected],
                [c.test_accuracy_mean for c in selected],
                s=SCATTER_POINT_SIZE, color=color, marker=marker,
                edgecolors="none", label=self._display(activation),
            )

        axes[0].set_xscale("log")
        axes[0].set_xlabel(r"mean $\|g\|$ of the configuration")
        axes[0].set_ylabel("mean test accuracy")
        axes[0].set_title("Accuracy against gradient magnitude")
        axes[1].set_xlabel(r"mean $\cos(g_t, g_{t-1})$ of the configuration")
        axes[1].set_ylabel("mean test accuracy")
        axes[1].set_title("Accuracy against gradient coherence")
        for ax in axes:
            FigureBuilder.grid(ax, which="both")

        handles, labels = axes[0].get_legend_handles_labels()
        FigureBuilder.legend_below(fig, handles, labels, ncol=min(len(labels), 6))
        fig.suptitle(
            "One point per optimizer and setting, at the deployed truncations"
        )
        FigureBuilder.save(fig, PLOT_DIR / "accuracy_vs_gradient")

    # ---- 10) coherence against truncation ------------------------------------

    def coherence_vs_truncation(self) -> None:
        series = [
            a for a in self.activation_order
            if len([t for t in self.data.truncations(a) if t is not None]) > 1
        ]
        if not series:
            return
        fig, axes = FigureBuilder.create(1, 2, figsize=(13.4, 5.4))

        for activation in series:
            truncations = [t for t in self.data.truncations(activation) if t is not None]
            coherence, damping = [], []
            for terms in truncations:
                group = self.data.select(activation=activation, terms=terms)
                coherence.append(
                    _mean(r.grad_cosine_mean for r in group) if group else float("nan")
                )
                damping.append(
                    _mean(r.damping_ratio for r in group) if group else float("nan")
                )
            color = ACTIVATION_COLORS.get(activation, COLOR_PALETTE_A[8])
            style = ACTIVATION_LINESTYLES.get(activation, "solid")
            marker = ACTIVATION_MARKERS.get(activation, "o")
            axes[0].plot(truncations, coherence, marker=marker, markersize=5.0,
                         color=color, linestyle=style, label=self._display(activation))
            axes[1].plot(truncations, damping, marker=marker, markersize=5.0,
                         color=color, linestyle=style, label=self._display(activation))

        # The two standard activations have no truncation axis; they are the
        # reference the others are read against.
        for activation in self.activation_order:
            if activation in series:
                continue
            group = self.data.select(activation=activation)
            if not group:
                continue
            axes[0].axhline(
                _mean(r.grad_cosine_mean for r in group),
                color=ACTIVATION_COLORS.get(activation, COLOR_PALETTE_A[8]),
                linestyle=(0, (2, 2)), linewidth=1.2,
                label=f"{self._display(activation)} (no truncation)",
            )

        axes[0].axhline(0.0, color="#444444", linewidth=0.9)
        axes[0].set_xlabel("retained terms")
        axes[0].set_ylabel(r"mean $\cos(g_t, g_{t-1})$")
        axes[0].set_title("Coherence against the number of retained terms")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("retained terms")
        axes[1].set_ylabel(r"mean $\|g\| \, / \, \|\Delta\theta\|$")
        axes[1].set_title("Damping against the number of retained terms")
        for ax in axes:
            FigureBuilder.grid(ax, which="both")

        handles, labels = axes[0].get_legend_handles_labels()
        FigureBuilder.legend_below(fig, handles, labels, ncol=min(len(labels), 4))
        FigureBuilder.save(fig, PLOT_DIR / "coherence_vs_truncation")

    # ---- 11) setting sensitivity --------------------------------------------

    def setting_sensitivity(self) -> None:
        records = self.data.deployed()
        families = [f for f in self.family_order if any(
            r.optimizer_family == f for r in records
        )]
        if not families:
            return
        fig, axes = FigureBuilder.create(
            2, len(families), figsize=(3.5 * len(families), 8.4),
            sharey="row",
        )
        axes = np.atleast_2d(axes)
        if axes.shape[0] == 1:
            axes = axes.reshape(2, -1)

        for column, family in enumerate(families):
            kind = next(
                (r.setting_kind for r in records if r.optimizer_family == family), ""
            )
            family_records = [r for r in records if r.optimizer_family == family]

            # For the fractional families the setting is the same quantity for
            # every member, so it can be plotted directly. The standard family
            # carries a learning rate, and each of its four optimizers has a
            # grid of its own; plotting those raw would interleave three
            # different scales on one axis. They are therefore placed by their
            # position within their own optimizer's grid.
            if kind == "learning_rate":
                ranks: Dict[Tuple[str, float], int] = {}
                for optimizer in {r.optimizer for r in family_records}:
                    own = sorted({
                        r.setting for r in family_records if r.optimizer == optimizer
                    })
                    for position, value in enumerate(own, start=1):
                        ranks[(optimizer, value)] = position
                positions = sorted(set(ranks.values()))
                key = lambda record: ranks.get((record.optimizer, record.setting))
            else:
                positions = sorted({r.setting for r in family_records})
                key = lambda record: record.setting

            for activation in self.activation_order:
                accuracies, gradients = [], []
                for position in positions:
                    group = [
                        r for r in family_records
                        if r.activation == activation and key(r) == position
                    ]
                    accuracies.append(
                        _mean(r.test_accuracy for r in group) if group else float("nan")
                    )
                    gradients.append(
                        _mean(r.grad_norm_mean for r in group) if group else float("nan")
                    )
                color = ACTIVATION_COLORS.get(activation, COLOR_PALETTE_A[8])
                style = ACTIVATION_LINESTYLES.get(activation, "solid")
                marker = ACTIVATION_MARKERS.get(activation, "o")
                axes[0][column].plot(positions, accuracies, marker=marker, markersize=4.5,
                                     color=color, linestyle=style,
                                     label=self._display(activation))
                axes[1][column].plot(positions, gradients, marker=marker, markersize=4.5,
                                     color=color, linestyle=style)
            axes[0][column].set_title(FAMILY_DISPLAY.get(family, family))
            axes[1][column].set_yscale("log")
            axes[1][column].set_xlabel(
                "learning rate, lowest to highest\nof each optimizer's own grid"
                if kind == "learning_rate" else "fractional order"
            )
            for ax in (axes[0][column], axes[1][column]):
                ax.set_xticks(positions)
                if kind == "learning_rate":
                    ax.set_xticklabels([str(p) for p in positions])
            FigureBuilder.grid(axes[0][column])
            FigureBuilder.grid(axes[1][column], which="both")
        axes[0][0].set_ylabel("mean test accuracy")
        axes[1][0].set_ylabel(r"mean $\|g\|$")

        handles, labels = axes[0][0].get_legend_handles_labels()
        FigureBuilder.legend_below(fig, handles, labels, ncol=min(len(labels), 6))
        fig.suptitle("Sensitivity to the setting, by optimizer family")
        FigureBuilder.save(fig, PLOT_DIR / "setting_sensitivity")

    # ---- 12) adaptive state --------------------------------------------------

    def adaptive_state(self) -> None:
        adaptive = [
            r for r in self.data.deployed()
            if r.optimizer_family == "adaptive_memory"
            and np.isfinite(r.adaptive_mix_mean)
        ]
        memory = [
            r for r in self.data.deployed()
            if r.optimizer_family == "memory" and np.isfinite(r.gl_coefficient_sum)
        ]
        if not adaptive and not memory:
            return
        fig, axes = FigureBuilder.create(1, 2, figsize=(13.4, 5.4))

        for activation in self.activation_order:
            selected = [r for r in adaptive if r.activation == activation]
            if selected:
                axes[0].scatter(
                    [r.grad_cosine_mean for r in selected],
                    [r.adaptive_mix_mean for r in selected],
                    s=SCATTER_POINT_SIZE,
                    color=ACTIVATION_COLORS.get(activation, COLOR_PALETTE_A[8]),
                    marker=ACTIVATION_MARKERS.get(activation, "o"),
                    edgecolors="none", label=self._display(activation),
                )
        axes[0].axhline(ADAPTIVE_MAX_MIX, color="#444444", linestyle=(0, (4, 3)),
                        linewidth=1.1)
        axes[0].annotate(
            f"bound {ADAPTIVE_MAX_MIX:g}", xy=(0.02, ADAPTIVE_MAX_MIX),
            xycoords=("axes fraction", "data"), va="bottom",
            fontsize=8 * FONT_SCALE, color="#444444",
        )
        axes[0].set_xlabel(r"mean $\cos(g_t, g_{t-1})$")
        axes[0].set_ylabel("realized fractional mixing weight")
        axes[0].set_title("Adaptive family: the mix it settles on")
        FigureBuilder.grid(axes[0])

        if memory:
            orders = sorted({r.setting for r in memory})
            # The coefficient sum is a property of the fractional order and the
            # history length alone, so every activation produces the same curve.
            # Drawing one line says that; drawing six would hide five of them
            # underneath the sixth.
            sums, spreads = [], []
            for order in orders:
                group = [
                    r for r in memory
                    if math.isclose(r.setting, order, rel_tol=1e-9)
                ]
                sums.append(_mean(r.gl_coefficient_sum for r in group))
                spreads.append(_std([r.gl_coefficient_sum for r in group]))
            axes[1].errorbar(
                orders, sums, yerr=spreads, marker="o", markersize=5.5,
                color=COLOR_PALETTE_A[2], linewidth=1.4, capsize=3.0,
                label="all activations",
            )
            axes[1].axhline(0.0, color="#444444", linewidth=0.9)
            axes[1].set_xticks(orders)
            axes[1].annotate(
                "identical across activations:\nthe sum depends on the order and\n"
                "the history length only",
                xy=(0.03, 0.06), xycoords="axes fraction", va="bottom",
                fontsize=8 * FONT_SCALE, color="#444444",
            )
        axes[1].set_xlabel("fractional order")
        axes[1].set_ylabel("GL coefficient sum")
        axes[1].set_title("Fixed-order family: gain of the memory filter")
        FigureBuilder.grid(axes[1])

        handles, labels = axes[0].get_legend_handles_labels()
        FigureBuilder.legend_below(fig, handles, labels, ncol=min(len(labels), 6))
        fig.suptitle("State the memory mechanisms arrive at")
        FigureBuilder.save(fig, PLOT_DIR / "adaptive_state")

    # ---- 13) distributions ----------------------------------------------------

    def run_distribution(self) -> None:
        records = self.data.deployed()
        if not records:
            return
        fig, axes = FigureBuilder.create(1, 2, figsize=(13.4, 5.6))

        accuracy_groups, runtime_groups, labels, colors = [], [], [], []
        for activation in self.activation_order:
            group = [r for r in records if r.activation == activation]
            if not group:
                continue
            accuracy_groups.append(
                [r.test_accuracy for r in group if np.isfinite(r.test_accuracy)]
            )
            runtime_groups.append(
                [r.seconds for r in group if np.isfinite(r.seconds)]
            )
            labels.append(self._display(activation))
            colors.append(ACTIVATION_COLORS.get(activation, COLOR_PALETTE_A[8]))

        for ax, groups, ylabel, title, logscale in (
            (axes[0], accuracy_groups, "test accuracy",
             "Spread over all optimizers, settings and seeds", False),
            (axes[1], runtime_groups, "seconds per run",
             "Wall-clock cost of one run", True),
        ):
            boxes = ax.boxplot(
                groups, patch_artist=True, showfliers=DISTRIBUTION_SHOWFLIERS,
                widths=0.6, medianprops={"color": "#1a1a1a", "linewidth": 1.4},
            )
            for patch, color in zip(boxes["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_edgecolor("#1a1a1a")
                patch.set_linewidth(0.8)
            # A logarithmic axis only where the values actually span decades;
            # over a narrow range it produces ticks that are harder to read
            # than the plain ones.
            flat = [v for group in groups for v in group if np.isfinite(v) and v > 0]
            if logscale and flat and max(flat) / min(flat) >= 10.0:
                ax.set_yscale("log")
                logscale = True
            else:
                logscale = False
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=20, ha="right", rotation_mode="anchor")
            FigureBuilder.grid(ax, axis="y", which="both" if logscale else "major")

        fig.suptitle(f"Distribution of outcomes per activation, {self.dataset}")
        FigureBuilder.save(fig, PLOT_DIR / "run_distribution")

    # ---- driver -------------------------------------------------------------

    def make_all(self) -> List[str]:
        produced: List[str] = []
        for name, method in (
            ("gradient_and_update_by_family", self.gradient_and_update_by_family),
            ("gradient_coherence", self.gradient_coherence),
            ("truncation_effect", self.truncation_effect),
            ("stability_heatmap", self.stability_heatmap),
            ("herrera_factor", self.herrera_factor),
            ("gradient_norm_heatmap", self.gradient_norm_heatmap),
            ("accuracy_heatmap", self.accuracy_heatmap),
            ("damping_ratio", self.damping_ratio),
            ("accuracy_vs_gradient", self.accuracy_vs_gradient),
            ("coherence_vs_truncation", self.coherence_vs_truncation),
            ("setting_sensitivity", self.setting_sensitivity),
            ("adaptive_state", self.adaptive_state),
            ("run_distribution", self.run_distribution),
        ):
            method()
            if (PLOT_DIR / f"{name}.png").exists():
                produced.append(name)
        return produced


# =============================================================================
# 5) TABLES AND REPORT
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
    return value


class AnalysisReport:
    """Every table the probe printed, plus the ones this script adds."""

    def __init__(
        self,
        aggregator: Aggregator,
        figures: Sequence[str],
        provenance: Dict[str, Any],
        loader_notes: Dict[str, Any],
    ) -> None:
        self.data = aggregator
        self.records = aggregator.records
        self.cells = aggregator.cell_summaries()
        self.optimizer_rows = aggregator.optimizer_summaries()
        self.figures = list(figures)
        self.provenance = provenance
        self.loader_notes = loader_notes
        self.activation_order = aggregator.activations()
        self.family_order = aggregator.families()
        self.optimizer_order = aggregator.optimizers()
        self.dataset = self.records[0].dataset if self.records else ""

    def _display(self, activation: str) -> str:
        return ACTIVATION_DISPLAY.get(activation, activation)

    def _column_width(self) -> int:
        """Column width for the activation columns of the wide tables."""
        longest = max(
            (len(self._display(a)) for a in self.activation_order), default=12
        )
        return max(12, longest + 1)

    # ---- csv ---------------------------------------------------------------

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def best_configurations(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for activation in self.activation_order:
            terms = self.data.deployed_terms_present(activation)
            selected = [
                c for c in self.cells
                if c.activation == activation and c.terms == terms
            ]
            selected.sort(
                key=lambda c: (
                    c.test_accuracy_mean if np.isfinite(c.test_accuracy_mean) else -np.inf
                ),
                reverse=True,
            )
            for rank, cell in enumerate(selected[:TOP_CONFIGURATIONS_PER_ACTIVATION], 1):
                out.append(
                    {
                        "activation": self._display(activation),
                        "terms": terms,
                        "rank": rank,
                        "optimizer": cell.optimizer,
                        "family": FAMILY_DISPLAY.get(
                            cell.optimizer_family, cell.optimizer_family
                        ),
                        "setting_kind": cell.setting_kind,
                        "setting": cell.setting,
                        "accuracy_mean": cell.test_accuracy_mean,
                        "accuracy_std": cell.test_accuracy_std,
                        "grad_norm_mean": cell.grad_norm_mean,
                        "coherence": cell.grad_cosine_mean,
                        "seeds": cell.seeds,
                    }
                )
        return out

    def setting_sensitivity_rows(self) -> List[Dict[str, Any]]:
        records = self.data.deployed()
        out: List[Dict[str, Any]] = []
        for family in self.family_order:
            settings = sorted({r.setting for r in records if r.optimizer_family == family})
            for setting in settings:
                for activation in self.activation_order:
                    group = [
                        r for r in records
                        if r.optimizer_family == family
                        and r.activation == activation
                        and math.isclose(r.setting, setting, rel_tol=1e-9)
                    ]
                    if not group:
                        continue
                    out.append(
                        {
                            "family": FAMILY_DISPLAY.get(family, family),
                            "setting_kind": group[0].setting_kind,
                            "setting": setting,
                            "activation": self._display(activation),
                            "runs": len(group),
                            "grad_norm_mean": _mean(r.grad_norm_mean for r in group),
                            "update_norm_mean": _mean(r.update_norm_mean for r in group),
                            "coherence": _mean(r.grad_cosine_mean for r in group),
                            "nonfinite_total": sum(r.nonfinite_total for r in group),
                            "accuracy_mean": _mean(r.test_accuracy for r in group),
                            "accuracy_std": _std([r.test_accuracy for r in group]),
                        }
                    )
        return out

    def stability_rows(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for record in self.records:
            if record.nonfinite_total <= 0:
                continue
            out.append(
                {
                    "activation": self._display(record.activation),
                    "terms": record.terms,
                    "optimizer": record.optimizer,
                    "family": FAMILY_DISPLAY.get(
                        record.optimizer_family, record.optimizer_family
                    ),
                    "setting": record.setting,
                    "seed": record.seed,
                    "nonfinite_loss_steps": record.nonfinite_loss_steps,
                    "nonfinite_gradient_steps": record.nonfinite_gradient_steps,
                    "nonfinite_update_steps": record.nonfinite_update_steps,
                    "optimizer_slot_nonfinite": record.optimizer_slot_nonfinite,
                    "first_slot_nonfinite_step": record.first_slot_nonfinite_step,
                    "grad_norm_max": record.grad_norm_max,
                    "test_accuracy": record.test_accuracy,
                }
            )
        return out

    def runtime_rows(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for activation in self.activation_order:
            for terms in self.data.truncations(activation):
                group = self.data.select(activation=activation, terms=terms)
                if not group:
                    continue
                total = sum(r.seconds for r in group if np.isfinite(r.seconds))
                out.append(
                    {
                        "activation": self._display(activation),
                        "terms": terms,
                        "runs": len(group),
                        "seconds_mean": _mean(r.seconds for r in group),
                        "seconds_max": _max(r.seconds for r in group),
                        "hours_total": total / 3600.0,
                        "steps_mean": _mean(float(r.steps) for r in group),
                    }
                )
        return out

    def write_tables(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._write_csv(OUTPUT_DIR / "cell_summary.csv", [asdict(c) for c in self.cells])
        self._write_csv(
            OUTPUT_DIR / "optimizer_summary.csv", [asdict(o) for o in self.optimizer_rows]
        )
        self._write_csv(OUTPUT_DIR / "setting_sensitivity.csv", self.setting_sensitivity_rows())
        self._write_csv(OUTPUT_DIR / "best_configurations.csv", self.best_configurations())
        self._write_csv(OUTPUT_DIR / "stability_incidents.csv", self.stability_rows())
        self._write_csv(OUTPUT_DIR / "runtime_summary.csv", self.runtime_rows())

    def write_latex(self) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "analysis_tables.tex"
        deployed = self.data.deployed()
        lines: List[str] = [
            "% Produced by B_06_analyze_activation_optimizer_probe.py", ""
        ]

        # Table 1: activation by family.
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\begin{tabular}{ll" + "r" * 6 + "}")
        lines.append("\\hline")
        lines.append(
            "activation & family & mean $\\|g\\|$ & mean $\\|\\Delta\\theta\\|$ & "
            "damping & coherence & non-finite & accuracy \\\\"
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
                lines.append(
                    f"{self._display(activation)} & "
                    f"{FAMILY_DISPLAY.get(family, family)} & "
                    f"{_mean(r.grad_norm_mean for r in group):.2e} & "
                    f"{_mean(r.update_norm_mean for r in group):.2e} & "
                    f"{_mean(r.damping_ratio for r in group):.2e} & "
                    f"{_mean(r.grad_cosine_mean for r in group):+.3f} & "
                    f"{sum(r.nonfinite_total for r in group)} & "
                    f"{_mean(r.test_accuracy for r in group):.4f} \\\\"
                )
            lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append(
            "\\caption{What each optimizer family receives and applies on each "
            f"activation at its deployed truncation, on {self.dataset}. Damping "
            "is the ratio of the two preceding columns. Coherence is the mean "
            "cosine similarity between the gradients of consecutive steps. The "
            "non-finite column counts steps with a non-finite loss, gradient or "
            "update together with non-finite entries in the optimizer's own slot "
            "variables.}"
        )
        lines.append("\\label{tab:probe_family_comparison}")
        lines.append("\\end{table}")
        lines.append("")

        # Table 2: the best configuration per activation.
        best = self.best_configurations()
        if best:
            lines.append("\\begin{table}[htbp]")
            lines.append("\\centering")
            lines.append("\\small")
            lines.append("\\begin{tabular}{llllrrr}")
            lines.append("\\hline")
            lines.append(
                "activation & rank & optimizer & family & setting & accuracy & "
                "mean $\\|g\\|$ \\\\"
            )
            lines.append("\\hline")
            current = None
            for row in best:
                if current is not None and row["activation"] != current:
                    lines.append("\\hline")
                current = row["activation"]
                lines.append(
                    f"{row['activation']} & {row['rank']} & {row['optimizer']} & "
                    f"{row['family']} & {row['setting']:g} & "
                    f"{row['accuracy_mean']:.4f} $\\pm$ {row['accuracy_std']:.4f} & "
                    f"{row['grad_norm_mean']:.2e} \\\\"
                )
            lines.append("\\hline")
            lines.append("\\end{tabular}")
            lines.append(
                "\\caption{The "
                f"{TOP_CONFIGURATIONS_PER_ACTIVATION} highest-ranked "
                "configurations per activation at the deployed truncation, by "
                "mean test accuracy over the seeds. The ranking is descriptive: "
                "with this many seeds per cell the differences between "
                "neighbouring ranks are not resolved.}"
            )
            lines.append("\\label{tab:probe_best_configurations}")
            lines.append("\\end{table}")

        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return path

    def write_json(self) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "analysis_summary.json"
        payload = {
            "format_version": 1,
            "produced_by": "B_06_analyze_activation_optimizer_probe.py",
            "source_records": str(RECORDS_FILE),
            "records_loaded": len(self.records),
            "loader_notes": self.loader_notes,
            "probe_provenance": self.provenance,
            "figures": self.figures,
            "cell_summary": [asdict(c) for c in self.cells],
            "optimizer_summary": [asdict(o) for o in self.optimizer_rows],
            "best_configurations": self.best_configurations(),
            "setting_sensitivity": self.setting_sensitivity_rows(),
            "runtime_summary": self.runtime_rows(),
            "stability_incidents": self.stability_rows(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2)
        return path

    # ---- text report --------------------------------------------------------

    def build_report(self) -> str:
        lines: List[str] = []
        deployed = self.data.deployed()

        lines.append(RULE)
        lines.append("Analysis of the activation and optimizer gradient probe")
        lines.append(RULE)
        lines.append("")
        lines.append(f"source         : {RECORDS_FILE}")
        lines.append(f"records        : {len(self.records)} runs on "
                     f"{self.dataset}, {len(self.cells)} configurations")
        lines.append("activations    : "
                     + ", ".join(
                         f"{self._display(a)} "
                         f"({', '.join(str(t) for t in self.data.truncations(a))})"
                         for a in self.activation_order
                     ))
        lines.append(f"optimizers     : {len(self.optimizer_order)} across "
                     f"{len(self.family_order)} families")
        if self.provenance.get("settings"):
            settings = self.provenance["settings"]
            lines.append(f"probe settings : orders {settings.get('order_grid')}, "
                         f"truncations {settings.get('truncations_to_probe')}, "
                         f"{settings.get('runs_per_cell')} seeds, "
                         f"loss injection "
                         f"{settings.get('feed_loss_to_adaptive_optimizers')}")
        if self.loader_notes.get("missing_columns"):
            lines.append(f"missing columns: {self.loader_notes['missing_columns']}")
        if self.loader_notes.get("extra_columns"):
            lines.append(f"extra columns  : {self.loader_notes['extra_columns']}")
        if self.loader_notes.get("skipped_rows"):
            lines.append(f"skipped rows   : {self.loader_notes['skipped_rows']}")
        lines.append(f"figures        : {len(self.figures)} written to {PLOT_DIR}")
        lines.append("")

        # ---- 1
        lines.append(THIN_RULE)
        lines.append("1) Gradient, update, damping and coherence at the deployed truncations")
        lines.append(THIN_RULE)
        lines.append(
            f"{'activation':>16} {'family':>18} {'mean|g|':>11} {'max|g|':>11} "
            f"{'mean|dtheta|':>13} {'damping':>11} {'coherence':>10} "
            f"{'nf steps':>9} {'nf slots':>9} {'accuracy':>9} {'runs':>5}"
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
                    f"{_mean(r.grad_norm_mean for r in group):>11.3e} "
                    f"{_max(r.grad_norm_max for r in group):>11.3e} "
                    f"{_mean(r.update_norm_mean for r in group):>13.3e} "
                    f"{_mean(r.damping_ratio for r in group):>11.3e} "
                    f"{_mean(r.grad_cosine_mean for r in group):>+10.3f} "
                    f"{sum(r.nonfinite_gradient_steps + r.nonfinite_loss_steps + r.nonfinite_update_steps for r in group):>9} "
                    f"{sum(r.optimizer_slot_nonfinite for r in group):>9} "
                    f"{_mean(r.test_accuracy for r in group):>9.4f} "
                    f"{len(group):>5}"
                )
            lines.append("")

        # ---- 2
        lines.append(THIN_RULE)
        lines.append("2) Effect of the truncation length")
        lines.append(THIN_RULE)
        lines.append(
            f"{'activation':>16} {'terms':>6} {'mean|g|':>11} {'mean|dtheta|':>13} "
            f"{'damping':>11} {'coherence':>10} {'nf steps':>9} {'accuracy':>9} "
            f"{'runs':>5}"
        )
        for activation in self.activation_order:
            for terms in self.data.truncations(activation):
                group = self.data.select(activation=activation, terms=terms)
                if not group:
                    continue
                lines.append(
                    f"{self._display(activation):>16} {str(terms):>6} "
                    f"{_mean(r.grad_norm_mean for r in group):>11.3e} "
                    f"{_mean(r.update_norm_mean for r in group):>13.3e} "
                    f"{_mean(r.damping_ratio for r in group):>11.3e} "
                    f"{_mean(r.grad_cosine_mean for r in group):>+10.3f} "
                    f"{sum(r.nonfinite_total for r in group):>9} "
                    f"{_mean(r.test_accuracy for r in group):>9.4f} "
                    f"{len(group):>5}"
                )
            lines.append("")

        # ---- 3
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
                        if r.activation == activation
                        and math.isclose(r.setting, order, rel_tol=1e-9)
                    ]
                    if not group:
                        continue
                    lines.append(
                        f"{self._display(activation):>16} {order:>7.2f} "
                        f"{_mean(r.herrera_factor_median for r in group):>14.3e} "
                        f"{_max(r.herrera_factor_max for r in group):>12.3e} "
                        f"{_mean(r.grad_norm_mean for r in group):>11.3e} "
                        f"{sum(r.nonfinite_gradient_steps + r.nonfinite_loss_steps + r.nonfinite_update_steps for r in group):>9} "
                        f"{sum(r.optimizer_slot_nonfinite for r in group):>9} "
                        f"{_mean(r.test_accuracy for r in group):>9.4f}"
                    )
                lines.append("")

        # ---- 4
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
                f"{'GL coeff sum':>13} {'mix':>8} {'stability':>10} {'mean|g|':>11} "
                f"{'accuracy':>9}"
            )
            for activation in self.activation_order:
                for family in ("memory", "adaptive_memory"):
                    for order in sorted({r.setting for r in memory}):
                        group = [
                            r for r in memory
                            if r.activation == activation
                            and r.optimizer_family == family
                            and math.isclose(r.setting, order, rel_tol=1e-9)
                        ]
                        if not group:
                            continue
                        lines.append(
                            f"{self._display(activation):>16} "
                            f"{FAMILY_DISPLAY.get(family, family):>18} "
                            f"{order:>7.2f} "
                            f"{_mean(r.grad_cosine_mean for r in group):>+10.3f} "
                            f"{_mean(r.gl_coefficient_sum for r in group):>13.4f} "
                            f"{_mean(r.adaptive_mix_mean for r in group):>8.4f} "
                            f"{_mean(r.adaptive_stability_mean for r in group):>10.4f} "
                            f"{_mean(r.grad_norm_mean for r in group):>11.3e} "
                            f"{_mean(r.test_accuracy for r in group):>9.4f}"
                        )
                lines.append("")
            lines.append(
                "The mixing weight is bounded by construction at "
                f"{ADAPTIVE_MAX_MIX:g}, and the adaptive family moves it from the "
                "relative change between\nconsecutive gradients: a coherence near "
                "zero drives that change towards its ceiling, which lowers the mix "
                "towards\nits floor and leaves the plain optimizer underneath."
            )
            lines.append("")

        # ---- 5 per optimizer
        lines.append(THIN_RULE)
        lines.append("5) Every optimizer against every activation, deployed truncations")
        lines.append(THIN_RULE)
        width = self._column_width()
        lines.append(
            f"{'optimizer':>24} {'family':>18} "
            + " ".join(
                f"{self._display(a):>{width}}" for a in self.activation_order
            )
        )
        for optimizer in self.optimizer_order:
            rows = [o for o in self.optimizer_rows if o.optimizer == optimizer]
            if not rows:
                continue
            family = rows[0].optimizer_family
            accuracy_cells = []
            for activation in self.activation_order:
                entry = next((o for o in rows if o.activation == activation), None)
                accuracy_cells.append(
                    f"{entry.test_accuracy_mean:>{width}.4f}" if entry
                    else f"{'-':>{width}}"
                )
            lines.append(
                f"{optimizer:>24} {FAMILY_DISPLAY.get(family, family):>18} "
                + " ".join(accuracy_cells)
            )
        lines.append("")
        lines.append("  mean test accuracy over the four settings and the seeds; "
                     "the gradient magnitudes of the same grid are in\n"
                     "  optimizer_summary.csv and in the figure gradient_norm_heatmap.")
        lines.append("")

        # ---- 6 setting sensitivity
        lines.append(THIN_RULE)
        lines.append("6) Sensitivity to the setting")
        lines.append(THIN_RULE)
        lines.append(
            f"{'family':>18} {'setting':>9} "
            + " ".join(
                f"{self._display(a):>{width}}" for a in self.activation_order
            )
        )
        for family in self.family_order:
            settings = sorted({r.setting for r in deployed if r.optimizer_family == family})
            for setting in settings:
                cells = []
                for activation in self.activation_order:
                    group = [
                        r for r in deployed
                        if r.optimizer_family == family
                        and r.activation == activation
                        and math.isclose(r.setting, setting, rel_tol=1e-9)
                    ]
                    cells.append(
                        f"{_mean(r.test_accuracy for r in group):>{width}.4f}"
                        if group else f"{'-':>{width}}"
                    )
                lines.append(
                    f"{FAMILY_DISPLAY.get(family, family):>18} {setting:>9g} "
                    + " ".join(cells)
                )
            lines.append("")
        lines.append("  mean test accuracy; for the standard family the setting is "
                     "the learning rate, for the others the fractional order.")
        lines.append("")

        # ---- 7 best configurations
        best = self.best_configurations()
        if best:
            lines.append(THIN_RULE)
            lines.append(
                f"7) The {TOP_CONFIGURATIONS_PER_ACTIVATION} highest-ranked "
                "configurations per activation"
            )
            lines.append(THIN_RULE)
            lines.append(
                f"{'activation':>16} {'rank':>5} {'optimizer':>24} {'family':>18} "
                f"{'setting':>9} {'accuracy':>18} {'mean|g|':>11} {'coherence':>10}"
            )
            for row in best:
                lines.append(
                    f"{row['activation']:>16} {row['rank']:>5} "
                    f"{row['optimizer']:>24} {row['family']:>18} "
                    f"{row['setting']:>9g} "
                    f"{row['accuracy_mean']:>11.4f} +- {row['accuracy_std']:.4f} "
                    f"{row['grad_norm_mean']:>11.3e} {row['coherence']:>+10.3f}"
                )
            lines.append("")
            lines.append(
                "  Descriptive ranking only: with this many seeds per cell the "
                "differences between neighbouring ranks are\n  not resolved, and "
                "the study's own classification results use forty seeds per "
                "configuration."
            )
            lines.append("")

        # ---- 8 stability incidents
        incidents = self.stability_rows()
        lines.append(THIN_RULE)
        lines.append("8) Runs with a non-finite loss, gradient, update or slot entry")
        lines.append(THIN_RULE)
        if not incidents:
            lines.append("  none in any run")
        else:
            lines.append(
                f"{'activation':>16} {'terms':>6} {'optimizer':>24} {'setting':>9} "
                f"{'seed':>8} {'loss':>6} {'grad':>6} {'upd':>6} {'slots':>8} "
                f"{'first step':>11} {'max|g|':>11}"
            )
            for row in incidents[:60]:
                lines.append(
                    f"{row['activation']:>16} {str(row['terms']):>6} "
                    f"{row['optimizer']:>24} {row['setting']:>9g} {row['seed']:>8} "
                    f"{row['nonfinite_loss_steps']:>6} "
                    f"{row['nonfinite_gradient_steps']:>6} "
                    f"{row['nonfinite_update_steps']:>6} "
                    f"{row['optimizer_slot_nonfinite']:>8} "
                    f"{row['first_slot_nonfinite_step']:>11} "
                    f"{row['grad_norm_max']:>11.3e}"
                )
            if len(incidents) > 60:
                lines.append(f"  ... and {len(incidents) - 60} more, "
                             f"see stability_incidents.csv")
        lines.append("")

        # ---- 9 runtime
        lines.append(THIN_RULE)
        lines.append("9) Wall-clock cost")
        lines.append(THIN_RULE)
        lines.append(
            f"{'activation':>16} {'terms':>6} {'runs':>6} {'s per run':>11} "
            f"{'max s':>9} {'hours':>8} {'steps':>8}"
        )
        for row in self.runtime_rows():
            lines.append(
                f"{row['activation']:>16} {str(row['terms']):>6} {row['runs']:>6} "
                f"{row['seconds_mean']:>11.2f} {row['seconds_max']:>9.1f} "
                f"{row['hours_total']:>8.2f} {row['steps_mean']:>8.0f}"
            )
        total_hours = sum(row["hours_total"] for row in self.runtime_rows())
        lines.append("")
        lines.append(f"  total measured compute: {total_hours:.2f} h over "
                     f"{len(self.records)} runs")
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
        deployed = self.data.deployed()
        if not deployed:
            return ["no finished runs"]

        magnitudes = [
            (self._display(a), _mean(
                r.grad_norm_mean for r in deployed if r.activation == a
            ))
            for a in self.activation_order
        ]
        magnitudes = [(n, v) for n, v in magnitudes if np.isfinite(v)]
        if magnitudes:
            magnitudes.sort(key=lambda item: item[1])
            out.append(
                "Averaged over every optimizer and setting, the mean global "
                "gradient norm per step is "
                + ", ".join(f"{n} {v:.3e}" for n, v in magnitudes)
                + "."
            )

        damping = [
            (self._display(a), _mean(
                r.damping_ratio for r in deployed if r.activation == a
            ))
            for a in self.activation_order
        ]
        damping = [(n, v) for n, v in damping if np.isfinite(v)]
        if damping:
            out.append(
                "The ratio between the gradient received and the update applied is "
                + ", ".join(f"{n} {v:.3e}" for n, v in damping)
                + ", so the optimizers absorb most of the difference in gradient "
                "magnitude before it reaches the parameters."
            )

        coherences = [
            (self._display(a), _mean(
                r.grad_cosine_mean for r in deployed if r.activation == a
            ))
            for a in self.activation_order
        ]
        coherences = [(n, v) for n, v in coherences if np.isfinite(v)]
        if coherences:
            coherences.sort(key=lambda item: item[1], reverse=True)
            out.append(
                "The mean cosine similarity between consecutive gradients is "
                + ", ".join(f"{n} {v:+.3f}" for n, v in coherences)
                + "."
            )

        # the adaptive mix against coherence
        adaptive = [
            r for r in deployed
            if r.optimizer_family == "adaptive_memory" and np.isfinite(r.adaptive_mix_mean)
        ]
        if adaptive:
            pairs = [
                (
                    self._display(a),
                    _mean(r.grad_cosine_mean for r in adaptive if r.activation == a),
                    _mean(r.adaptive_mix_mean for r in adaptive if r.activation == a),
                )
                for a in self.activation_order
            ]
            pairs = [p for p in pairs if np.isfinite(p[2])]
            if pairs:
                out.append(
                    "The adaptive family settles at a mixing weight of "
                    + ", ".join(f"{n} {mix:.3f} (coherence {coh:+.3f})"
                                for n, coh, mix in pairs)
                    + f", against a bound of {ADAPTIVE_MAX_MIX:g}."
                )

        incidents = self.stability_rows()
        if incidents:
            grouped: Dict[Tuple[str, str], int] = {}
            for row in incidents:
                grouped[(row["activation"], row["family"])] = (
                    grouped.get((row["activation"], row["family"]), 0) + 1
                )
            out.append(
                "Non-finite values occurred in "
                + "; ".join(
                    f"{activation} with {family} ({count} runs)"
                    for (activation, family), count in sorted(
                        grouped.items(), key=lambda item: -item[1]
                    )
                )
                + "."
            )
        else:
            out.append(
                "No run produced a non-finite loss, gradient, update or optimizer "
                "slot entry."
            )

        for activation in self.activation_order:
            truncations = [t for t in self.data.truncations(activation) if t is not None]
            if len(truncations) < 2:
                continue
            first = self.data.select(activation=activation, terms=truncations[0])
            last = self.data.select(activation=activation, terms=truncations[-1])
            if first and last:
                out.append(
                    f"{self._display(activation)}: between {truncations[0]} and "
                    f"{truncations[-1]} terms the mean gradient norm moves from "
                    f"{_mean(r.grad_norm_mean for r in first):.3e} to "
                    f"{_mean(r.grad_norm_mean for r in last):.3e}, the coherence "
                    f"from {_mean(r.grad_cosine_mean for r in first):+.3f} to "
                    f"{_mean(r.grad_cosine_mean for r in last):+.3f}, and the mean "
                    f"test accuracy from {_mean(r.test_accuracy for r in first):.4f} "
                    f"to {_mean(r.test_accuracy for r in last):.4f}."
                )

        return out

    def write_report(self) -> Tuple[Path, str]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        text = self.build_report()
        path = OUTPUT_DIR / "activation_optimizer_analysis_report.txt"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        return path, text


# =============================================================================
# 6) MAIN
# =============================================================================

def main() -> None:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("Analysis of the activation and optimizer gradient probe")
    print("=" * 96)
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, "
          f"matplotlib {matplotlib.__version__} on {platform.platform()}")
    print(f"reading {RECORDS_FILE}")

    loader = RecordLoader()
    records = loader.load()
    loader_notes = {
        "missing_columns": loader.missing_columns,
        "extra_columns": loader.extra_columns,
        "skipped_rows": loader.skipped_rows,
    }
    if not records:
        raise SystemExit(f"{RECORDS_FILE} holds no usable rows")

    print(f"  {len(records)} runs on {records[0].dataset}")
    if loader.missing_columns:
        print(f"  columns missing from the file: {loader.missing_columns}")
    if loader.extra_columns:
        print(f"  columns in the file this script does not use: {loader.extra_columns}")
    if loader.skipped_rows:
        print(f"  rows skipped as unreadable: {loader.skipped_rows}")

    aggregator = Aggregator(records)
    print(f"  {len(aggregator.activations())} activations, "
          f"{len(aggregator.optimizers())} optimizers, "
          f"{len(aggregator.cell_summaries())} configurations")

    print("\nPlotting")
    figures = ProbeFigures(aggregator).make_all()
    for name in figures:
        print(f"  {name}")

    report = AnalysisReport(
        aggregator, figures, loader.provenance(), loader_notes
    )
    report.write_tables()
    json_path = report.write_json()
    tex_path = report.write_latex()
    report_path, text = report.write_report()

    print("")
    print(text)
    print("")
    print(f"Report:       {report_path}")
    print(f"Tables:       {OUTPUT_DIR}")
    print(f"JSON summary: {json_path}")
    print(f"LaTeX tables: {tex_path}")
    print(f"Plots:        {PLOT_DIR} ({len(figures)} figures, png and eps)")
    print(f"Elapsed:      {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
