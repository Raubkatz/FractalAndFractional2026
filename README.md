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

The final classification experiment contains **504 optimizer–activation–parameter configurations per dataset** (21 optimizers × 6 activations × 4 parameter settings), corresponding to **20,160 neural-network training runs per dataset and 201,600 training runs across all ten datasets**.

The final loss-surface experiment evaluates **84 optimizer–parameter configurations per surface variant**, with **100 repeated runs per configuration**. Across standard Ackley, additive Ackley, standard Himmelblau, and additive Himmelblau, this corresponds to **33,600 surface-optimization runs**.

All settings are defined at the top of each script, and every script is configured with the settings of the experiments reported in the article.

This repository contains the original experimental framework. Updated optimizer and activation-function implementations for **Keras 3 (TensorFlow 2.21)** and **PyTorch 2.15** are maintained separately in [Fractal_Fractional_Base](https://github.com/Raubkatz/Fractal_Fractional_Base). The companion repository is not a substitute for the experiment scripts and analysis workflow documented here.

---

## 📚 Attribution, Original Publications, and Code Sources

The optimizers and activation functions used in this repository originate from the publications identified below. The references acknowledge the underlying methods and mathematical constructions; they do not imply that the implementation files were copied directly from the original authors' code.

| Components in this repository | Original publications | Source and attribution |
|---|---|---|
| `class_b_tf_fractional_optimizers_OHA.py`: `FSGD`, `FAdam`, `FRMSprop`, `FAdadelta` | Herrera-Alcántara and co-authors [[H1]](#ref-h1), [[H2]](#ref-h2) | Herrera-style fractional gradient scaling. The original TensorFlow–Keras work identifies [ia.azc.uam.mx](http://ia.azc.uam.mx/) as a source-code location; the subsequent article describes the original authors' PyTorch work. These papers are the methodological sources. |
| `class_d_related_work_optimizers.py`: `AdaGL` | Chen, Zhang, and Mu [[R1]](#ref-r1) | Implementation of the published adaptive Grünwald–Letnikov optimizer. |
| `class_d_related_work_optimizers.py`: `FCSGD_GL`, `FCAdam_GL` | Zhou, Zhao, and Huang [[R2]](#ref-r2) | Implementation of the published short-memory fractional methods. |
| `class_d_related_work_optimizers.py`: `AOFGD_SGD`, `AOFGD_Adam` | Xiang et al. [[R3]](#ref-r3) | Implementation of the adaptive-order fractional-gradient approach described in the cited preprint. |
| `class_c_tf_fractional_optimizers_SR.py`: `MemoryF*` | Present study [[O1]](#ref-o1); Grünwald–Letnikov foundations [[G1]](#ref-g1), [[G2]](#ref-g2) | Study-specific gradient-memory implementations incorporating norm matching and a descent safeguard. |
| `class_minimal_fractional_adaptive_optimizers.py`: `AdaptiveMemoryF*` | Present study [[O1]](#ref-o1) | Adaptive mixing of the ordinary gradient and a fractional-memory direction, as described in the article. |
| `fractal_activation_functions.py` | Predecessor activation-function study [[O2]](#ref-o2); historical sources [[F1]–[F4]](#fractal-functions-mathematical-origins) | The activation functions originate from the predecessor study. Its separate code repository is [Fractal_Activation_Functions](https://github.com/Raubkatz/Fractal_Activation_Functions). |
| Standard baseline optimizers | Original optimizer literature [[B1]–[B10]](#base-optimizers) | Standard optimization methods supplied by the TensorFlow/Keras framework and their documented implementations. |

**Original-author code and reimplementations.** References [[H1]](#ref-h1) and [[H2]](#ref-h2) document Herrera-Alcántara's original implementations. For AdaGL, FCSGD_GL/FCAdam_GL, and AOFGD, the cited papers are the primary methodological references; the preceding implementation README does not identify direct original-author code downloads. Implementation-specific choices should therefore be checked against the relevant paper rather than assumed to reproduce every undocumented detail of an author's original code.

**Updated implementations.** The companion repository [Fractal_Fractional_Base](https://github.com/Raubkatz/Fractal_Fractional_Base) provides the updated optimizer and activation-function implementations for **Keras 3 (TensorFlow 2.21)** and **PyTorch 2.15**, including additional optimizer families and activation variants. It also documents the independently implemented Cantor-based activations attributed to Chipera, Du, and Tsapara [[C1]](#ref-c1). Those additional activations are distinct from the six-activation classification suite used in this repository's paper experiments.

**Citation guidance.** Cite the original publication of each method used, together with [[O1]](#ref-o1) for the GL-memory and adaptive-memory implementations or the experimental comparison, and [[O2]](#ref-o2) for the fractal activation functions. The original experiment code is provided here; the companion Keras 3/PyTorch code is maintained in the [updated repository](https://github.com/Raubkatz/Fractal_Fractional_Base).

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
├── requirements.txt
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
├── B_04_analyze_activation_optimizer_probe.py
├── B_04_activation_comparison_diagnostics.py
│
├── D_01_memory_degeneracy_kernel_analysis.py
└── D_02_memory_degeneracy_runtime_probe.py
```

> Keep the Python module filenames consistent with those shown above so that the imports resolve correctly. All scripts are run from the repository root; they import the optimizer and activation modules from the same directory and write their outputs into subfolders of the working directory.

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

The classification suite contains:

- `relu`
- `tanh`
- `modulated_blancmange_curve` (a = 0.75, 30 terms)
- `decaying_cosine_function_tf` (c = 0.5, d = 2, ζ = 0.2666, 75 terms)
- `modified_weierstrass_function_tanh` (a = 0.5, b = 1.5, 100 terms)
- `weierstrass_mandelbrot_function_xpsin` (γ = 0.5, λ = 2, `num_terms = 100`, i.e. scales k = 1, …, 99)

The fractal activations are used with the default parameters of the module, which are those of the predecessor study [[O2]](#ref-o2). The module also contains further activations (other Weierstrass-type variants, random-depth variants, and Cantor-based variants) that are not part of the experiments of the article.

---

### Optimizer Implementations

All optimizers use the `optimizer_v2` API of `tf.keras` in TensorFlow 2.10 and accept the learning rate as `lr=` or `learning_rate=`.

#### `class_b_tf_fractional_optimizers_OHA.py`

Implements the Herrera-type fractional optimizers used in the experiments:

- `FSGD`
- `FAdam`
- `FRMSprop`
- `FAdadelta`

These methods locally rescale the current gradient by the Caputo-type factor (|θ| + ε)^(1−ν) / Γ(2−ν) with ε = 10⁻⁶ but do not introduce explicit gradient memory. `FAdagrad` and `FAdamW` are provided in addition.

#### `class_c_tf_fractional_optimizers_SR.py`

Implements the explicit Grünwald–Letnikov memory-based optimizers:

- `MemoryFSGD`
- `MemoryFRMSprop`
- `MemoryFAdam`
- `MemoryFAdadelta`

These optimizers construct the update direction from the current and previous gradients (history length K = 6, unnormalized coefficients), with norm matching, a descent safeguard, and raw-gradient second moments. The memory-based extensions `MemoryFAdagrad`, `MemoryFAdamW`, `MemoryFNadam`, and `MemoryFAdamax` are provided in addition.

#### `class_minimal_fractional_adaptive_optimizers.py`

Implements the adaptive memory-based optimizers introduced in the paper:

- `AdaptiveMemoryFSGD`
- `AdaptiveMemoryFRMSprop`
- `AdaptiveMemoryFAdam`
- `AdaptiveMemoryFAdadelta`

The fractional-memory direction (history length K = 4, normalized coefficients) is mixed with the ordinary gradient through an adaptive trust coefficient λ_t ∈ [0, 0.30], initialized at 0.10. The extensions `AdaptiveMemoryFAdagrad`, `AdaptiveMemoryFAdamW`, `AdaptiveMemoryFNadam`, and `AdaptiveMemoryFAdamax` are provided in addition.

#### `class_d_related_work_optimizers.py`

Implements the representative related-work methods:

- `AdaGL` (K = 10)
- `FCSGD_GL` (K = 11, Bernoulli retention probability p = 0.5)
- `FCAdam_GL` (K = 11, p = 0.5)
- `AOFGD_SGD`
- `AOFGD_Adam`

---

### Loss-Surface Experiments

#### `A_00a_himmelblau_perturbation_and_minima.py`

Defines the standard and additively perturbed Himmelblau surfaces, determines the minima of the perturbed surface, and stores the resolved minima together with a fingerprint of the perturbation for use by the optimization experiment. Output: `himmelblau_perturbation_and_minima/`.

#### `A_00b_ackley_perturbation_and_minima.py`

Performs the corresponding perturbation and perturbed-minimum analysis for the Ackley surface and stores the reference information required by the optimization experiment. Output: `ackley_perturbation_and_minima/`.

#### `A_01_himmelblau_loss_surface_optimization.py`

Runs repeated optimization experiments on the standard and additively perturbed Himmelblau surfaces using the full optimizer suite: 21 optimizers × 4 settings = 84 configurations, 100 runs each, 250 steps per run.

For the perturbed variant, the script loads the reference minima produced by `A_00a_himmelblau_perturbation_and_minima.py` and refuses to run unless the perturbation definition agrees with the stored fingerprint. Output: `himmelblau_comparison_fin_std_add_3_lr_vderiv_grid/`.

#### `A_02_ackley_loss_surface_optimization.py`

Runs the corresponding experiments on the standard and additively perturbed Ackley surfaces.

For the perturbed variant, the script loads the reference minima produced by `A_00b_ackley_perturbation_and_minima.py` and verifies the perturbation definition before running the experiment. Output: `ackley_comparison_fin_std_add_5_lr_vderiv_grid/`.

#### `A_03_analyze_himmelblau.py`

Analyzes the Himmelblau runs and produces configuration-level and optimizer-level statistics, rankings, pooled optimizer and optimizer-group views, human-readable reports, CSV and JSON outputs, and plots. Output: `himmelblau_comparison_fin_std_add_3_lr_vderiv_grid_analysis/`.

#### `A_04_analyze_ackley.py`

Analyzes the Ackley runs and produces the corresponding statistical summaries, rankings, reports, and visualizations. Output: `ackley_comparison_fin_std_add_5_lr_vderiv_grid_analysis/`.

---

### Neural-Network Experiments

#### `B_01_run_nn_classification.py`

Runs the complete neural-network classification experiment.

The script:

- loads the ten public OpenML datasets by name;
- applies the dataset-specific network settings;
- evaluates six activation functions;
- evaluates 21 optimizers;
- sweeps the four optimizer-specific parameter settings;
- performs 40 repeated runs;
- records accuracy, macro-F1, precision, recall, timing, completed epochs, best validation loss, and training histories;
- writes one JSON result file per configuration to `compiled_results_all_experiments/<dataset>/`.

#### `B_02_analyze_neural_network_experiments.py`

Analyzes the classification results stored in `compiled_results_all_experiments/` and writes to `compiled_analysis_October2026/`.

The script computes:

- configuration-level statistics (mean, standard deviation, standard error, 95% confidence interval, median, quartiles, extremes);
- optimizer-specific rankings, based on the best configuration of each optimizer per dataset (highest mean accuracy, ties resolved by mean macro-F1 and then by shorter mean training time);
- activation rankings;
- optimizer-group comparisons;
- activation-family comparisons, including a view without the Weierstrass–Mandelbrot x+sin activation;
- fractional-order sensitivity;
- optimizer–activation interaction analyses;
- cross-dataset summaries;
- publication-ready PNG and EPS plots;
- CSV and human-readable reports.

The activation-level, optimizer-group, activation-family, and order-sensitivity views pool all result records found in the input folder.

#### `B_03_activation_optimizer_gradient_probe.py`

Runs the end-to-end gradient diagnostic of the appendix on gradient-level diagnostics: the `diabetes` dataset with the classification architecture (one hidden layer of 64 units, batch size 32, 30 epochs, 510 optimization steps per run), the six activations, each fractal activation at the truncation lengths 8, 16, 24, 32 and 64 that do not exceed its deployed length plus the deployed length, and all 21 optimizers at their four settings, with three seeds. This gives 2016 configurations and 6048 runs. It records gradient and update magnitudes, consecutive-gradient coherence, non-finite losses, gradients, updates and optimizer-slot entries, family-specific fractional diagnostics, and training outcomes. Finished runs are written immediately to `activation_optimizer_gradient_probe/run_records.csv`, so an interrupted probe can be resumed.

#### `B_04_analyze_activation_optimizer_probe.py`

Rebuilds every table and figure of the gradient probe from `activation_optimizer_gradient_probe/run_records.csv` without training, and adds further views (gradient-norm and accuracy heatmaps over the whole grid, damping ratios, setting sensitivity, adaptive-memory state, runtime). Output: `activation_optimizer_analysis/`.

#### `B_04_activation_comparison_diagnostics.py`

Performs numerical diagnostics of all six activation functions, including truncation sensitivity, derivative magnitude, finite-precision behaviour, resolved scale structure, gradient realizability, and an optional compact network probe. TensorFlow is optional; without it, the NumPy stages run and the network probe is skipped. Output: `activation_comparison_diagnostics/`.

---

### Memory-Kernel Analysis

#### `D_01_memory_degeneracy_kernel_analysis.py`

Analyzes the Grünwald–Letnikov memory pipeline independently of the training experiments and reproduces the appendix on the response of the truncated memory kernel. It requires only NumPy and Matplotlib.

The script studies the GL coefficients, DC gain, persistent-gradient limit, descent safeguard, norm matching, adaptive mixing, safeguard activation, and the conditions under which the memory contribution differs from the ordinary gradient, for every order of the grid ν ∈ {0.75, 1.25, 1.50, 1.75}. Output: `degeneracy_analysis_kernel/`.

#### `D_02_memory_degeneracy_runtime_probe.py`

Records the same quantities per step and per parameter tensor during training, from instrumented subclasses of the optimizer classes used in the experiments, verified against them by trajectory comparison. It additionally records the realized gradient-change ratio ρ_t of each step, so that the measured gradient sequences can be placed on the axis of the kernel analysis.

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
        │      compiled_results_all_experiments/
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
                    ▲           │      activation_optimizer_gradient_probe/run_records.csv
                    │           ▼
                    │    B_04_analyze_activation_optimizer_probe.py
                    │
        optimizer implementation modules

class_c_tf_fractional_optimizers_SR.py
        │
class_minimal_fractional_adaptive_optimizers.py
        │
        ├──> D_01_memory_degeneracy_kernel_analysis.py
        └──> D_02_memory_degeneracy_runtime_probe.py
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

The standard optimizers are evaluated at four learning rates, 0.5, 1, 2, and 4 times their reference learning rate.

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

For the surface experiments:

```text
SGD and Adam:
0.005, 0.01, 0.02, 0.04

RMSprop:
0.0025, 0.005, 0.01, 0.02

Adadelta:
0.5, 1.0, 2.0, 4.0
```

### Fractional Orders

The Herrera, GL-Memory, and Adaptive-Memory methods are evaluated with:

```text
ν = 0.75
ν = 1.25
ν = 1.50
ν = 1.75
```

AdaGL, FCSGD_GL, and FCAdam_GL use the same four values as their parameter α.

AOFGD adapts its order during optimization and is evaluated at four initial orders α_init:

```text
Classification:  α_init ∈ {0.75, 1.25, 1.50, 1.75},  α_max = max(1.5, α_init)
Surfaces:        α_init ∈ {0.75, 1.00, 1.25, 1.50},  α_max = 1.5
```

with α_min = 0.5, Δα = 0.02, convergence-evaluation threshold 0.05, and EMA factor 0.95.

The fractional optimizers keep the reference learning rate of their base optimizer: 10⁻² (SGD-based), 10⁻³ (Adam- and RMSprop-based), and 1.0 (Adadelta-based) in the classification experiments, and 0.01 (SGD- and Adam-based), 0.005 (RMSprop-based), and 1.0 (Adadelta-based) in the surface experiments.

### Fixed Parameters

```text
SGD momentum:                 0.9, no Nesterov
RMSprop:                      ρ = 0.9, no momentum, not centered
Adam:                         β1 = 0.9, β2 = 0.999, no AMSGrad
Adadelta:                     ρ = 0.95
Numerical stabilizer:         1e-7
Herrera factor ε:             1e-6
GL-Memory:                    K = 6, unnormalized coefficients
Adaptive-Memory:              K = 4, normalized coefficients, λ ∈ [0, 0.30], λ0 = 0.10,
                              adaptation step 0.01, mixing EMA 0.90, stability EMA 0.95,
                              stability threshold 0.25, stability clip 10,
                              loss EMA 0.95, loss-worsening tolerance 1e-4,
                              loss-triggered mixing decrease 0.02, no warm-up
AdaGL:                        K = 10
FCSGD_GL / FCAdam_GL:         K = 11, p = 0.5
```

---

## 🗺️ Surface-Experiment Design

Each surface is evaluated in two variants:

- standard;
- additive finite Weierstrass-type perturbation,
  f_add(x, y) = f(x, y) + α_add Σ_{k=0}^{J} a_P^k [cos(b_P^k π x) + cos(b_P^k π y)],
  with α_add = 0.2, a_P = 0.55, J = 8, and b_P = 2.4 (Ackley) or b_P = 2.5 (Himmelblau).

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

Each run takes 250 optimization steps. The starting points are generated before optimization and shared across optimizer configurations within each repeated run. Each starting point lies at a fixed radius from a reference minimum (2.0 on Himmelblau, cycling through the four minima; 2.5 on Ackley, around the global minimum), with the angles of all runs sharing a reference minimum spread evenly over the full circle. This ensures that the compared optimizers start from identical coordinates for a given run, while the set of runs covers all directions around each minimum.

The perturbed variants use the resolved minima generated by the corresponding `A_00` scripts. Success and distance statistics are therefore evaluated with respect to the surface variant that was actually optimized; the distance to the nearest base-surface minimum is recorded alongside. A run counts as a success if its final point lies within a distance of 0.3 of a minimum. The analysis also reports the gap to the lowest objective value over the resolved minima of each variant and the corresponding normalized regret.

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
21 optimizers × 6 activation functions × 4 parameter settings
= 504 optimizer–activation–parameter configurations per dataset

504 configurations × 40 repeated runs
= 20,160 training runs per dataset
```

Across ten datasets:

```text
20,160 × 10
= 201,600 neural-network training runs
```

The optimizer-specific parameter search is balanced so that every optimizer contributes four candidate settings.

`B_01_run_nn_classification.py` is configured with this complete grid: all ten datasets, all six activations, all 21 optimizers, the four learning rates of each standard optimizer, and the four order-related settings of each fractional optimizer. The 40 run seeds are the consecutive integers 238974, …, 239013. The tested learning rate (standard optimizers) or order-related parameter (fractional optimizers) is stored in the `vderiv` field of each result file.

Each run uses a new shuffled 70/30 split of the dataset with the run seed as random state; the features are min–max scaled on the training part. The held-out part is used both as validation data during training and as the test partition. Training runs for the full epoch budget without early stopping. All networks use Glorot-uniform initialization with a fixed initializer seed.

| Dataset | Task | Width *n* | Hidden layers | Batch size | Epochs |
|---|---|---|---|---|---|
| climate-model-simulation-crashes | binary | 32 | 1 | 32 | 30 |
| diabetes | binary | 64 | 1 | 32 | 30 |
| ionosphere | binary | 128 | 1 | 32 | 30 |
| tic-tac-toe | binary | 64 | 1 | 32 | 25 |
| vertebra-column | binary | 32 | 1 | 16 | 30 |
| glass | multi-class | 64 | 2 (*n*, 2*n*) | 32 | 30 |
| iris | multi-class | 32 | 2 (*n*, 2*n*) | 16 | 30 |
| seeds | multi-class | 64 | 2 (*n*, 2*n*) | 32 | 30 |
| vehicle | multi-class | 128 | 2 (*n*, 2*n*) | 32 | 25 |
| wine | multi-class | 64 | 2 (*n*, 2*n*) | 32 | 30 |

Multi-class networks apply batch normalization and dropout (0.2) after each hidden layer; binary networks consist of one dense hidden layer followed by the output layer.

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

**Archiving:** Neither GitHub repository is assigned a separate archival DOI.

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

`B_03_activation_optimizer_gradient_probe.py` extends the activation diagnostics to the actual optimizer interactions, and `B_04_analyze_activation_optimizer_probe.py` rebuilds its tables and figures from the saved records.

The probe records:

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

`D_01_memory_degeneracy_kernel_analysis.py` studies the memory mechanism independently of network training, and `D_02_memory_degeneracy_runtime_probe.py` records the same quantities during training.

They examine:

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

pip install -r requirements.txt
```

The optimizer classes use the `optimizer_v2` API of `tf.keras`, which is the default optimizer API of TensorFlow 2.10. The OpenML datasets are downloaded on first use; saving animations of the surface runs (disabled by default) additionally requires FFmpeg.

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
python B_04_analyze_activation_optimizer_probe.py
python D_01_memory_degeneracy_kernel_analysis.py
python D_02_memory_degeneracy_runtime_probe.py
```

The analysis scripts operate on saved experimental records and do not rerun the main optimization or classification experiments.

### Output Folders

| Script | Output folder |
|---|---|
| `A_00a_himmelblau_perturbation_and_minima.py` | `himmelblau_perturbation_and_minima/` |
| `A_00b_ackley_perturbation_and_minima.py` | `ackley_perturbation_and_minima/` |
| `A_01_himmelblau_loss_surface_optimization.py` | `himmelblau_comparison_fin_std_add_3_lr_vderiv_grid/` |
| `A_02_ackley_loss_surface_optimization.py` | `ackley_comparison_fin_std_add_5_lr_vderiv_grid/` |
| `A_03_analyze_himmelblau.py` | `himmelblau_comparison_fin_std_add_3_lr_vderiv_grid_analysis/` |
| `A_04_analyze_ackley.py` | `ackley_comparison_fin_std_add_5_lr_vderiv_grid_analysis/` |
| `B_01_run_nn_classification.py` | `compiled_results_all_experiments/` |
| `B_02_analyze_neural_network_experiments.py` | `compiled_analysis_October2026/` |
| `B_03_activation_optimizer_gradient_probe.py` | `activation_optimizer_gradient_probe/` |
| `B_04_analyze_activation_optimizer_probe.py` | `activation_optimizer_analysis/` |
| `B_04_activation_comparison_diagnostics.py` | `activation_comparison_diagnostics/` |
| `D_01_memory_degeneracy_kernel_analysis.py` | `degeneracy_analysis_kernel/` |

---

## References

- <a id="ref-h1"></a>**[H1]** O. Herrera-Alcántara. Fractional Derivative Gradient-Based Optimizers for Neural Networks and Human Activity Recognition. *Applied Sciences* **12**(18), 9264 (2022). [doi:10.3390/app12189264](https://doi.org/10.3390/app12189264)
- <a id="ref-h2"></a>**[H2]** O. Herrera-Alcántara, J. R. Castelán-Aguilar. Fractional Gradient Optimizers for PyTorch: Enhancing GAN and BERT. *Fractal and Fractional* **7**(7), 500 (2023). [doi:10.3390/fractalfract7070500](https://doi.org/10.3390/fractalfract7070500)
- <a id="ref-r1"></a>**[R1]** S. Chen, C. Zhang, H. Mu. An Adaptive Learning Rate Deep Learning Optimizer Using Long and Short-Term Gradients Based on G–L Fractional-Order Derivative. *Neural Processing Letters* **56**, 106 (2024). [doi:10.1007/s11063-024-11571-7](https://doi.org/10.1007/s11063-024-11571-7)
- <a id="ref-r2"></a>**[R2]** X. Zhou, C. Zhao, Y. Huang. A Deep Learning Optimizer Based on Grünwald–Letnikov Fractional Order Definition. *Mathematics* **11**(2), 316 (2023). [doi:10.3390/math11020316](https://doi.org/10.3390/math11020316)
- <a id="ref-r3"></a>**[R3]** K. Xiang, C. Zhao, Q. Su, X. Zhou, J. Ye, Y. Huang. AOFGD: Adaptive Order Fractional Gradient Descent Method. SSRN preprint, not peer-reviewed (2025). [doi:10.2139/ssrn.5717167](https://doi.org/10.2139/ssrn.5717167) · [ssrn.com/abstract=5717167](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5717167)
- <a id="ref-c1"></a>**[C1]** R. Chipera, J. Du, I. Tsapara. Fractal and Chaotic Activation Functions in Echo State Networks: Preprocessing Topology Governs the Echo State Property. arXiv preprint (2025). [arXiv:2512.14675](https://arxiv.org/abs/2512.14675)
- <a id="ref-o1"></a>**[O1]** S. Raubitzek, G. Goldenits, S. Schrittwieser, P. König, K. Mallinger. Fractional Optimizers Meet Fractal Activation Functions: An Empirical Study of Multi-Scale Optimization in Neural Networks. *Fractal and Fractional* (2026, accepted). [arXiv:2608.14636](https://arxiv.org/abs/2608.14636) · [code](https://github.com/Raubkatz/FractalAndFractional2026)
- <a id="ref-o2"></a>**[O2]** S. Raubitzek, T. Kietreiber, S. Eresheim, A. Buchelt, K. Mallinger. Fractals in neural networks: Introducing a new class of activation functions. *Communications in Nonlinear Science and Numerical Simulation* **161**, 110064 (2026). [doi:10.1016/j.cnsns.2026.110064](https://doi.org/10.1016/j.cnsns.2026.110064) · [code](https://github.com/Raubkatz/Fractal_Activation_Functions)

**Grünwald–Letnikov background**

- <a id="ref-g1"></a>**[G1]** I. Podlubny. *Fractional Differential Equations.* Mathematics in Science and Engineering, vol. 198. Academic Press, San Diego (1999).
- <a id="ref-g2"></a>**[G2]** Y. Wei, Y. Chen, S. Cheng, Y. Wang. A Note on Short Memory Principle of Fractional Calculus. *Fractional Calculus and Applied Analysis* **20**, 1382–1404 (2017). [doi:10.1515/fca-2017-0073](https://doi.org/10.1515/fca-2017-0073)

<a id="base-optimizers"></a>**Base optimizers**

- <a id="ref-b1"></a>**[B1]** SGD: H. Robbins, S. Monro. A Stochastic Approximation Method. *Annals of Mathematical Statistics* **22**, 400–407 (1951). [doi:10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586)
- <a id="ref-b2"></a>**[B2]** Momentum: B. T. Polyak. Some Methods of Speeding up the Convergence of Iteration Methods. *USSR Computational Mathematics and Mathematical Physics* **4**, 1–17 (1964). [doi:10.1016/0041-5553(64)90137-5](https://doi.org/10.1016/0041-5553(64)90137-5)
- <a id="ref-b3"></a>**[B3]** Nesterov momentum: Y. Nesterov. A method for solving the convex programming problem with convergence rate O(1/k²). *Soviet Mathematics Doklady* **27**, 372–376 (1983); in deep learning: I. Sutskever, J. Martens, G. Dahl, G. Hinton. On the importance of initialization and momentum in deep learning. *Proc. ICML*, PMLR **28**(3), 1139–1147 (2013). [proceedings.mlr.press/v28/sutskever13](https://proceedings.mlr.press/v28/sutskever13.html)
- <a id="ref-b4"></a>**[B4]** Adam and Adamax: D. P. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. *ICLR* (2015). [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- <a id="ref-b5"></a>**[B5]** AMSGrad: S. J. Reddi, S. Kale, S. Kumar. On the Convergence of Adam and Beyond. *ICLR* (2018). [openreview.net/forum?id=ryQu7f-RZ](https://openreview.net/forum?id=ryQu7f-RZ)
- <a id="ref-b6"></a>**[B6]** RMSprop: T. Tieleman, G. Hinton. Lecture 6.5 — RMSProp: Divide the gradient by a running average of its recent magnitude. *Neural Networks for Machine Learning*, Coursera (2012). [lecture slides](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- <a id="ref-b7"></a>**[B7]** Adadelta: M. D. Zeiler. ADADELTA: An Adaptive Learning Rate Method. arXiv (2012). [arXiv:1212.5701](https://arxiv.org/abs/1212.5701)
- <a id="ref-b8"></a>**[B8]** Adagrad: J. Duchi, E. Hazan, Y. Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. *Journal of Machine Learning Research* **12**, 2121–2159 (2011). [jmlr.org (PDF)](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
- <a id="ref-b9"></a>**[B9]** AdamW: I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. *ICLR* (2019). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- <a id="ref-b10"></a>**[B10]** Nadam: T. Dozat. Incorporating Nesterov Momentum into Adam. Stanford CS229 project report (2015); ICLR 2016 Workshop track. [report (PDF)](http://cs229.stanford.edu/proj2015/054_report.pdf) · [ICLR workshop paper (PDF)](https://openreview.net/pdf/OM0jvwB8jIp57ZJjtNEZ.pdf)

<a id="fractal-functions-mathematical-origins"></a>**Fractal functions: mathematical origins**

- <a id="ref-f1"></a>**[F1]** Weierstrass function: K. Weierstraß. Über continuirliche Functionen eines reellen Arguments, die für keinen Werth des letzteren einen bestimmten Differentialquotienten besitzen (lecture, 1872). Reprinted in R. Siegmund-Schultze (ed.), *Ausgewählte Kapitel aus der Funktionenlehre*, Teubner-Archiv zur Mathematik 9, Springer, Vienna (1988), pp. 190–193.
- <a id="ref-f2"></a>**[F2]** Blancmange (Takagi) function: T. Takagi. A Simple Example of the Continuous Function without Derivative. *Proc. Phys.-Math. Soc. Japan Ser. II* (Tokyo Sugaku-Butsurigakkwai Hokoku) **1**, F176–F177 (1903). [doi:10.11429/subutsuhokoku1901.1.F176](https://doi.org/10.11429/subutsuhokoku1901.1.F176)
- <a id="ref-f3"></a>**[F3]** Weierstrass–Mandelbrot function: M. V. Berry, Z. V. Lewis. On the Weierstrass–Mandelbrot Fractal Function. *Proc. R. Soc. Lond. A* **370**, 459–484 (1980). [doi:10.1098/rspa.1980.0044](https://doi.org/10.1098/rspa.1980.0044)
- <a id="ref-f4"></a>**[F4]** B. B. Mandelbrot. *The Fractal Geometry of Nature.* W. H. Freeman, San Francisco (1982).
- <a id="ref-f5"></a>**[F5]** Cantor set and Cantor function: G. Cantor. De la puissance des ensembles parfaits de points. *Acta Mathematica* **4**, 381–392 (1884).

**Software**

- <a id="ref-s1"></a>**[S1]** F. Chollet et al. Keras (2015–). [keras.io](https://keras.io) · [github.com/keras-team/keras](https://github.com/keras-team/keras)
- <a id="ref-s2"></a>**[S2]** M. Abadi et al. TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems. arXiv (2016). [arXiv:1603.04467](https://arxiv.org/abs/1603.04467) · [tensorflow.org](https://www.tensorflow.org)
- <a id="ref-s3"></a>**[S3]** F. Pedregosa et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research* **12**, 2825–2830 (2011). [jmlr.org/papers/v12/pedregosa11a](https://jmlr.org/papers/v12/pedregosa11a.html)
- <a id="ref-s4"></a>**[S4]** A. Paszke et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems* **32** (NeurIPS 2019), 8024–8035. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703) · [pytorch.org](https://pytorch.org)

**Data**

- <a id="ref-d1"></a>**[D1]** Iris data: R. A. Fisher. The use of multiple measurements in taxonomic problems. *Annals of Eugenics* **7**(2), 179–188 (1936). [doi:10.1111/j.1469-1809.1936.tb02137.x](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x) · OpenML dataset 61: [openml.org/d/61](https://www.openml.org/d/61)

### BibTeX

```bibtex
@article{herrera2022fractional,
  author  = {Herrera-Alc{\'a}ntara, Oscar},
  title   = {Fractional Derivative Gradient-Based Optimizers for Neural Networks and Human Activity Recognition},
  journal = {Applied Sciences},
  volume  = {12},
  number  = {18},
  pages   = {9264},
  year    = {2022},
  doi     = {10.3390/app12189264}
}

@article{herrera2023pytorch,
  author  = {Herrera-Alc{\'a}ntara, Oscar and Castel{\'a}n-Aguilar, Josu{\'e} R.},
  title   = {Fractional Gradient Optimizers for {PyTorch}: Enhancing {GAN} and {BERT}},
  journal = {Fractal and Fractional},
  volume  = {7},
  number  = {7},
  pages   = {500},
  year    = {2023},
  doi     = {10.3390/fractalfract7070500}
}

@article{chen2024adagl,
  author  = {Chen, Shuang and Zhang, Changlun and Mu, Haibing},
  title   = {An Adaptive Learning Rate Deep Learning Optimizer Using Long and Short-Term Gradients Based on {G--L} Fractional-Order Derivative},
  journal = {Neural Processing Letters},
  volume  = {56},
  number  = {2},
  pages   = {106},
  year    = {2024},
  doi     = {10.1007/s11063-024-11571-7}
}

@article{zhou2023gl,
  author  = {Zhou, Xiaojun and Zhao, Chunna and Huang, Yaqun},
  title   = {A Deep Learning Optimizer Based on {Gr{\"u}nwald--Letnikov} Fractional Order Definition},
  journal = {Mathematics},
  volume  = {11},
  number  = {2},
  pages   = {316},
  year    = {2023},
  doi     = {10.3390/math11020316}
}

@misc{xiang2025aofgd,
  author       = {Xiang, Kemeng and Zhao, Chunna and Su, Qian and Zhou, Xiaojun and Ye, Junjie and Huang, Yaqun},
  title        = {{AOFGD}: Adaptive Order Fractional Gradient Descent Method},
  howpublished = {SSRN preprint, abstract ID 5717167},
  year         = {2025},
  doi          = {10.2139/ssrn.5717167}
}

@misc{chipera2025fractal,
  author        = {Chipera, Rae and Du, Jenny and Tsapara, Irene},
  title         = {Fractal and Chaotic Activation Functions in Echo State Networks: Preprocessing Topology Governs the Echo State Property},
  year          = {2025},
  eprint        = {2512.14675},
  archivePrefix = {arXiv}
}

@article{raubitzek2026fractional,
  author  = {Raubitzek, Sebastian and Goldenits, Georg and Schrittwieser, Sebastian and K{\"o}nig, Philip and Mallinger, Kevin},
  title   = {Fractional Optimizers Meet Fractal Activation Functions: An Empirical Study of Multi-Scale Optimization in Neural Networks},
  journal = {Fractal and Fractional},
  year    = {2026},
  note    = {Accepted. Preprint: arXiv:2608.14636. Code: https://github.com/Raubkatz/FractalAndFractional2026}
}

@article{raubitzek2026fractals,
  author  = {Raubitzek, Sebastian and Kietreiber, Tobias and Eresheim, Sebastian and Buchelt, Alexander and Mallinger, Kevin},
  title   = {Fractals in neural networks: Introducing a new class of activation functions},
  journal = {Communications in Nonlinear Science and Numerical Simulation},
  volume  = {161},
  pages   = {110064},
  year    = {2026},
  doi     = {10.1016/j.cnsns.2026.110064}
}

@inproceedings{paszke2019pytorch,
  author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and K{\"o}pf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
  title     = {{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
  booktitle = {Advances in Neural Information Processing Systems 32},
  pages     = {8024--8035},
  year      = {2019}
}
```

## Citation

Please cite the original publication for every optimizer or activation function used: Herrera-Alcántara [[H1]](#ref-h1), [[H2]](#ref-h2); Chen, Zhang, and Mu [[R1]](#ref-r1); Zhou, Zhao, and Huang [[R2]](#ref-r2); Xiang et al. [[R3]](#ref-r3); and, where Cantor-based activations from the companion repository are used, Chipera, Du, and Tsapara [[C1]](#ref-c1). Cite [[O1]](#ref-o1) for the study's GL-memory and adaptive-memory optimizers and experimental comparison, and [[O2]](#ref-o2) for the fractal activation functions.

**Original experimental repository:** [FractalAndFractional2026](https://github.com/Raubkatz/FractalAndFractional2026)  
**Updated Keras 3 / PyTorch implementations:** [Fractal_Fractional_Base](https://github.com/Raubkatz/Fractal_Fractional_Base)  
**Predecessor fractal-activation source:** [Fractal_Activation_Functions](https://github.com/Raubkatz/Fractal_Activation_Functions)
