# 🌀 Fractional Optimizers Meet Fractal Activation Functions

Official code repository for the paper:

> **Fractional Optimizers Meet Fractal Activation Functions: An Empirical Study of Multi-Scale Optimization in Neural Networks**

**Authors:** Sebastian Raubitzek, Georg Goldenits, Sebastian Schrittwieser, Philip König, and Kevin Mallinger

---

## 📖 Overview

This repository studies the interaction between **fractional optimizers** and **fractal activation functions** in neural-network training.

The experiments are divided into two parts:

1. **Controlled loss-surface optimization** on standard and fractally perturbed Ackley and Himmelblau functions.
2. **Neural-network classification** on ten public OpenML datasets using 21 optimizers, six activation functions, and 40 repeated runs per configuration.

The full classification experiment contains **222 optimizer–activation–order configurations per dataset**, corresponding to **88,800 neural-network training runs** across all ten datasets.

---

## ✨ Main Components

- 🧮 Classical and fractional TensorFlow optimizers
- 🌀 Grünwald–Letnikov gradient-memory methods
- ⚖️ Adaptive memory-based fractional optimizers
- 🧠 Weierstrass- and Blancmange-type activation functions
- 🗺️ Ackley and Himmelblau optimization experiments
- 🧪 Repeated neural-network classification experiments
- 📊 Statistical analysis, rankings, tables, and plots
- 🔁 Reproducible random seeds and experimental settings

---

## 🗂️ Repository Structure

```text
FractalAndFractional2026/
│
├── README.md
├── Fractal_And_Fractional.pdf
├── template.tex
│
├── fractal_activation_functions.py
├── class_b_tf_fractional_optimizers_OHA.py
├── class_c_tf_fractional_optimizers_SR.py
├── class_minimal_fractional_adaptive_optimizers.py
├── class_d_related_work_optimizers.py
│
├── A_01_himmelblau_loss_surface_optimization.py
├── A_02_ackley_loss_surface_optimization.py
├── A_03_analyze_himmelblau.py
├── A_04_analyze_ackley.py
├── A_05_plot_loss_surfaces.py
│
├── B_01_run_nn_classification.py
└── B_02_analyze_neural_network_experiments.py
```

> Rename uploaded files to the clean names shown above before execution so that all Python imports resolve correctly.

---

## 📄 File Descriptions

### Manuscript

#### `Fractal_And_Fractional.pdf`

Complete paper describing the mathematical background, optimizer constructions, fractal activation functions, experiments, results, and conclusions.

#### `template.tex`

LaTeX source of the manuscript.

---

### Activation Functions

#### `fractal_activation_functions.py`

Provides the TensorFlow implementations of the activation functions used in the experiments.

The final classification suite contains:

- `relu`
- `tanh`
- `modulated_blancmange_curve`
- `decaying_cosine_function_tf`
- `modified_weierstrass_function_tanh`
- `weierstrass_mandelbrot_function_xpsin`

---

### Optimizer Implementations

#### `class_b_tf_fractional_optimizers_OHA.py`

Implements the Herrera-type fractional optimizers used in the final experiments:

- `FSGD`
- `FAdam`
- `FRMSprop`
- `FAdadelta`

These methods locally rescale the current gradient using the fractional order but do not introduce explicit gradient memory.

#### `class_c_tf_fractional_optimizers_SR.py`

Implements the explicit Grünwald–Letnikov memory-based optimizers:

- `MemoryFSGD`
- `MemoryFRMSprop`
- `MemoryFAdam`
- `MemoryFAdadelta`

These optimizers construct the update direction from the current and previous gradients.

#### `class_minimal_fractional_adaptive_optimizers.py`

Implements the adaptive memory-based optimizers introduced in the paper:

- `AdaptiveMemoryFSGD`
- `AdaptiveMemoryFRMSprop`
- `AdaptiveMemoryFAdam`
- `AdaptiveMemoryFAdadelta`

The fractional-memory direction is mixed with the ordinary gradient through an adaptive trust coefficient.

#### `class_d_related_work_optimizers.py`

Implements the representative related-work methods:

- `AdaGL`
- `FCSGD_GL`
- `FCAdam_GL`
- `AOFGD_SGD`
- `AOFGD_Adam`

---

### Loss-Surface Experiments

#### `A_01_himmelblau_loss_surface_optimization.py`

Runs repeated optimization experiments on the standard and additively perturbed Himmelblau surfaces using the final optimizer suite.

#### `A_02_ackley_loss_surface_optimization.py`

Runs the corresponding experiments on the standard and additively perturbed Ackley surfaces.

#### `A_03_analyze_himmelblau.py`

Analyzes the Himmelblau runs and produces optimizer statistics, rankings, reports, and plots.

#### `A_04_analyze_ackley.py`

Analyzes the Ackley runs and produces the corresponding statistical summaries and visualizations.

#### `A_05_plot_loss_surfaces.py`

Creates contour visualizations of the benchmark surfaces and their fractally perturbed variants.

---

### Neural-Network Experiments

#### `B_01_run_nn_classification.py`

Runs the complete neural-network classification experiment.

The script:

- loads ten public OpenML datasets;
- applies the dataset-specific network settings;
- evaluates six activation functions;
- evaluates 21 optimizers;
- sweeps the fractional orders where applicable;
- performs 40 repeated runs;
- records accuracy, macro-F1, precision, recall, timing, and training histories;
- writes one JSON result file per configuration.

#### `B_02_analyze_neural_network_experiments.py`

Analyzes the classification results.

The script computes:

- configuration-level statistics;
- optimizer-specific rankings;
- activation rankings;
- optimizer-group comparisons;
- fractional-order sensitivity;
- fractal-versus-standard activation comparisons;
- cross-dataset summaries;
- publication-ready PNG and EPS plots;
- CSV and human-readable reports.

---

## ⚙️ Final Experimental Suite

### Standard Optimizers

- `SGD`
- `Adam`
- `RMSprop`
- `Adadelta`

### Herrera-Type Fractional Optimizers

- `FSGD`
- `FAdam`
- `FRMSprop`
- `FAdadelta`

### Grünwald–Letnikov Memory Optimizers

- `MemoryFSGD`
- `MemoryFRMSprop`
- `MemoryFAdam`
- `MemoryFAdadelta`

### Adaptive Memory Optimizers

- `AdaptiveMemoryFSGD`
- `AdaptiveMemoryFRMSprop`
- `AdaptiveMemoryFAdam`
- `AdaptiveMemoryFAdadelta`

### Related-Work Optimizers

- `AdaGL`
- `FCSGD_GL`
- `FCAdam_GL`
- `AOFGD_SGD`
- `AOFGD_Adam`

### Fractional Orders

The fixed-order fractional methods are evaluated with:

```text
ν = 0.75
ν = 1.25
ν = 1.50
```

---

## 📚 Classification Datasets

The classification experiments use:

- Climate-Model Simulation Crashes
- Diabetes
- Glass Identification
- Ionosphere
- Iris
- Seeds
- Tic-Tac-Toe Endgame
- Vehicle Silhouettes
- Vertebral Column
- Wine Recognition

Binary datasets use one hidden layer. Multi-class datasets use two hidden layers with dataset-specific widths, batch sizes, and epoch counts.

---

## 💻 Requirements

```text
python==3.8.19
numpy==1.23.5
scipy==1.10.1
pandas==2.0.3
scikit-learn==1.3.2
tensorflow==2.10.0
matplotlib==3.7.5
seaborn==0.13.2

conda create -n fractal-fractional python=3.8.19
conda activate fractal-fractional

pip install \
    numpy==1.23.5 \
    scipy==1.10.1 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    tensorflow==2.10.0 \
    matplotlib==3.7.5 \
    seaborn==0.13.2
```

---

## 📌 Main Findings

- Fractal activations frequently occur among the strongest optimizer-specific configurations.
- Herrera-type fractional optimizers are highly competitive in classification.
- Explicit gradient memory is more effective on deterministic perturbed surfaces than during minibatch training.
- Adaptive memory improves the robustness of direct memory substitution.
- No single fractional order is optimal across all datasets and optimizers.
- Activation function, optimizer formulation, and fractional order should be selected jointly.
- Repeated-run evaluation is necessary because many leading configurations have overlapping uncertainty ranges.

---

