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
│
└── D_01_memory_degeneracy_kernel_analysis.py
