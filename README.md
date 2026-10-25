# 🌀 Fractional Optimizers Meet Fractal Activation Functions

Official code repository for the paper:

> **Fractional Optimizers Meet Fractal Activation Functions: An Empirical Study of Multi-Scale Optimization in Neural Networks**

**Author of this Repository:** Sebastian Raubitzek

---

## 📖 Overview

This repository studies the interaction between **fractional optimizers** and **fractal activation functions** in neural-network training.

The experiments are divided into two parts:

1. **Controlled loss-surface optimization** on standard and additively perturbed Ackley and Himmelblau functions.
2. **Neural-network classification** on ten public OpenML datasets using 21 optimizers, six activation functions, and 40 repeated runs per configuration.

The final classification experiment contains **432 optimizer–activation–parameter configurations per dataset**, corresponding to **17,280 neural-network training runs per dataset and 172,800 training runs across all ten datasets**.

The final loss-surface experiment evaluates **84 optimizer–parameter configurations per surface variant**, with **100 repeated runs per configuration**. Across standard Ackley, additive Ackley, standard Himmelblau, and additive Himmelblau, this corresponds to **33,600 surface-optimization runs**.

---

## ✨ Main Components

- 🧮 Classical and fractional TensorFlow optimizers
- 🌀 Grünwald–Letnikov gradient-memory methods
- ⚖️ Adaptive memory-based fractional optimizers
- 🧠 Weierstrass- and Blancmange-type activation functions
- 🗺️ Ackley and Himmelblau optimization experiments
- 🔎 Dedicated perturbation and perturbed-minimum determination
- 🧪 Repeated neural-network classification experiments
- 📊 Statistical analysis, rankings, tables, and plots
- 🔬 Analytical investigation of the Grünwald–Letnikov memory mechanism
- 🔍 Activation and optimizer gradient diagnostics
- 🔁 Reproducible random seeds, shared starting points, and experimental settings

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
├── A_00a_himmelblau_perturbation_and_minima.py
├── A_00b_ackley_perturbation_and_minima.py
├── A_01_himmelblau_loss_surface_optimization.py
├── A_02_ackley_loss_surface_optimization.py
├── A_03_analyze_himmelblau.py
├── A_04_analyze_ackley.py
│
├── B_01_run_nn_classification.py
├── B_02_analyze_neural_network_experiments.py
├── B_03_activation_optimizer_gradient_probe.py
├── B_04_activation_comparison_diagnostics.py
│
└── D_01_memory_degeneracy_kernel_analysis.py
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

#### `A_00a_himmelblau_perturbation_and_minima.py`

Defines the standard and additively perturbed Himmelblau surfaces, determines the minima of the perturbed surface, and stores the resolved minima together with a fingerprint of the perturbation for use by the optimization experiment.

#### `A_00b_ackley_perturbation_and_minima.py`

Performs the corresponding perturbation and perturbed-minimum analysis for the Ackley surface and stores the reference information required by the optimization experiment.

#### `A_01_himmelblau_loss_surface_optimization.py`

Runs repeated optimization experiments on the standard and additively perturbed Himmelblau surfaces using the final optimizer suite.

For the perturbed variant, the script loads the reference minima produced by `A_00a_himmelblau_perturbation_and_minima.py` and verifies that the perturbation definition agrees with the stored fingerprint.

#### `A_02_ackley_loss_surface_optimization.py`

Runs the corresponding experiments on the standard and additively perturbed Ackley surfaces.

For the perturbed variant, the script loads the reference minima produced by `A_00b_ackley_perturbation_and_minima.py` and verifies the perturbation definition before running the experiment.

#### `A_03_analyze_himmelblau.py`

Analyzes the Himmelblau runs and produces configuration-level and optimizer-level statistics, rankings, human-readable reports, CSV and JSON outputs, and plots.

#### `A_04_analyze_ackley.py`

Analyzes the Ackley runs and produces the corresponding statistical summaries, rankings, reports, and visualizations.

---

### Neural-Network Experiments

#### `B_01_run_nn_classification.py`

Runs the complete neural-network classification experiment.

The script:

- loads ten public OpenML datasets;
- applies the dataset-specific network settings;
- evaluates six activation functions;
- evaluates 21 optimizers;
- sweeps the optimizer-specific parameter settings;
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
- activation-family comparisons;
- fractional-order sensitivity;
- optimizer–activation interaction analyses;
- cross-dataset summaries;
- publication-ready PNG and EPS plots;
- CSV and human-readable reports.

#### `B_03_activation_optimizer_gradient_probe.py`

Runs an end-to-end gradient diagnostic across the activation functions and optimizer families. It records gradient and update magnitudes, consecutive-gradient coherence, optimizer-state stability, family-specific fractional diagnostics, and training outcomes.

#### `B_04_activation_comparison_diagnostics.py`

Performs numerical diagnostics of all six activation functions, including truncation sensitivity, derivative magnitude, finite-precision behaviour, resolved scale structure, gradient realizability, and an optional compact network probe.

---

### Memory-Kernel Analysis

#### `D_01_memory_degeneracy_kernel_analysis.py`

Analyzes the Grünwald–Letnikov memory pipeline independently of the training experiments.

The script studies the GL coefficients, DC gain, persistent-gradient limit, descent safeguard, norm matching, adaptive mixing, safeguard activation, and the conditions under which the memory contribution differs from the ordinary gradient.

---

## 🔗 Experimental Workflow

The loss-surface experiments follow the sequence:

```text
A_00a_himmelblau_perturbation_and_minima.py
        │
        ▼
A_01_himmelblau_loss_surface_optimization.py
        │
        ▼
A_03_analyze_himmelblau.py


A_00b_ackley_perturbation_and_minima.py
        │
        ▼
A_02_ackley_loss_surface_optimization.py
        │
        ▼
A_04_analyze_ackley.py
```

The perturbation scripts determine the perturbed reference minima before optimization. The optimization scripts verify the corresponding perturbation fingerprints and generate the repeated-run results. The analysis scripts then process the stored results without rerunning optimization.

The neural-network workflow is:

```text
fractal_activation_functions.py
        │
        ├── class_b_tf_fractional_optimizers_OHA.py
        ├── class_c_tf_fractional_optimizers_SR.py
        ├── class_minimal_fractional_adaptive_optimizers.py
        └── class_d_related_work_optimizers.py
        │
        ▼
B_01_run_nn_classification.py
        │
        ▼
B_02_analyze_neural_network_experiments.py
```

The additional diagnostic scripts complement the main classification analysis:

```text
fractal_activation_functions.py
        │
        ├──> B_04_activation_comparison_diagnostics.py
        │
        └──> B_03_activation_optimizer_gradient_probe.py
                    ▲
                    │
        optimizer implementation modules

class_c_tf_fractional_optimizers_SR.py
        │
class_minimal_fractional_adaptive_optimizers.py
        │
        ▼
D_01_memory_degeneracy_kernel_analysis.py
```

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

---

## ⚙️ Parameter Grids

Each optimizer is evaluated under four optimizer-specific parameter settings.

### Standard Optimizers

The standard optimizers are evaluated at four learning rates.

For the classification experiments:

```text
SGD:
0.005, 0.010, 0.020, 0.040

Adam:
0.0005, 0.001, 0.002, 0.004

RMSprop:
0.0005, 0.001, 0.002, 0.004

Adadelta:
0.5, 1.0, 2.0, 4.0
```

The surface experiments use the corresponding four-value optimizer-specific learning-rate grids defined in the respective experiment scripts.

### Fractional Orders

The Herrera, GL-Memory, and Adaptive-Memory methods are evaluated with:

```text
ν = 0.75
ν = 1.25
ν = 1.50
ν = 1.75
```

The corresponding order-related parameters of the related-work optimizers are evaluated over the same four-setting design.

AOFGD adapts its order during optimization and is evaluated using four initial-order settings.

---

## 🗺️ Surface-Experiment Design

Each surface is evaluated in two variants:

- standard;
- additive finite Weierstrass-type perturbation.

The same 21 optimizers are evaluated on every variant.

Each optimizer contributes four parameter settings:

```text
21 optimizers
× 4 parameter settings
= 84 configurations
```

Each configuration is evaluated over 100 repeated runs:

```text
84 configurations
× 100 runs
= 8,400 runs per surface variant
```

Across all four variants:

```text
4 × 8,400
= 33,600 surface-optimization runs
```

The starting points are generated before optimization and shared across optimizer configurations within each repeated run. This ensures that the compared optimizers start from identical coordinates for a given run.

The perturbed variants use the resolved minima generated by the corresponding `A_00` scripts. Success and distance statistics can therefore be evaluated with respect to the surface variant that was actually optimized.

---

## 🧠 Classification-Experiment Design

The final classification suite combines:

- 21 optimizers;
- six activation functions;
- four optimizer-specific parameter settings;
- ten datasets;
- 40 repeated runs per configuration.

The final evaluated experiment contains:

```text
432 optimizer–activation–parameter configurations per dataset
× 40 repeated runs
= 17,280 training runs per dataset
```

Across ten datasets:

```text
17,280 × 10
= 172,800 neural-network training runs
```

The optimizer-specific parameter search is balanced so that every optimizer contributes four candidate settings.

The analysis keeps several views separate:

- individual configurations;
- selected best configuration per optimizer;
- activation-level results;
- optimizer-group aggregates;
- activation-family aggregates;
- fractional-order sensitivity;
- optimizer–activation interactions;
- cross-dataset summaries.

This makes it possible to distinguish the strongest observed configuration from the broader average behaviour of an optimizer or activation family.

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

## 🔬 Additional Diagnostics

### Activation Diagnostics

`B_04_activation_comparison_diagnostics.py` compares all six activation functions under the numerical conditions used by the study.

The analysis includes:

- deployed activation values and derivatives;
- float32 and float64 comparisons;
- truncation-length sensitivity;
- derivative-amplitude growth;
- retained and resolved scales;
- phase and lattice behaviour;
- sensitivity to floating-point perturbations;
- comparison between analytical gradients and numerically realized slopes;
- non-finite value and gradient checks;
- an optional compact neural-network probe.

### Activation–Optimizer Gradient Probe

`B_03_activation_optimizer_gradient_probe.py` extends the activation diagnostics to the actual optimizer interactions.

The script records:

- global gradient norms;
- parameter-update norms;
- gradient coherence between consecutive steps;
- non-finite losses and gradients;
- optimizer-slot stability;
- Herrera scaling factors;
- GL-memory properties;
- adaptive-memory mixing and stability values;
- final training loss;
- test accuracy;
- observed pre-activation ranges.

This connects the numerical properties of the activation functions with the mechanisms used by the different optimizer families.

### Memory-Kernel Diagnostic

`D_01_memory_degeneracy_kernel_analysis.py` studies the memory mechanism independently of network training.

It examines:

- GL coefficient sequences;
- truncated DC gain;
- persistent-gradient behaviour;
- descent-safeguard activation;
- norm matching;
- adaptive convex mixing;
- gradient-persistence regimes;
- the transverse component contributed by the memory term.

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
```

Create the environment with:

```bash
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

## ▶️ Recommended Execution Order

### Loss-Surface Experiments

```bash
python A_00a_himmelblau_perturbation_and_minima.py
python A_00b_ackley_perturbation_and_minima.py

python A_01_himmelblau_loss_surface_optimization.py
python A_02_ackley_loss_surface_optimization.py

python A_03_analyze_himmelblau.py
python A_04_analyze_ackley.py
```

### Neural-Network Classification

```bash
python B_01_run_nn_classification.py
python B_02_analyze_neural_network_experiments.py
```

### Additional Diagnostics

```bash
python B_04_activation_comparison_diagnostics.py
python B_03_activation_optimizer_gradient_probe.py
python D_01_memory_degeneracy_kernel_analysis.py
```

The analysis scripts operate on saved experimental records and do not rerun the main optimization or classification experiments.

---
