"""
class_d_related_work_optimizers.py

TensorFlow / Keras legacy-API implementations of the three published
fractional optimizer families most closely related to the Adaptive
Memory-Fractional framework developed in this project.

These implementations follow the same API style as the other class files
in this project (class_b and class_c), using:
    - subclass tf.keras.optimizers.Optimizer
    - _set_hyper(...) / _get_hyper(...)
    - _create_slots(...)
    - _resource_apply_dense(...)
    - _resource_apply_sparse(...)
    - lr=... as legacy learning-rate argument

Purpose
-------
This file exists so that the three related-work optimizers can be compared
directly against the optimizers in class_b, class_c, and the adaptive
memory class in the same experiment runners (01_run_nn_classifications.py
and 01bg_optimize_fractal_surface_himmelblau_requested_16_optimizers.py).

Implemented optimizers
----------------------
Group 1: AdaGL (Chen, Zhang & Mu, 2024)
    - AdaGL

Group 2: FCSGD_GL / FCAdam_GL (Zhou, Zhao & Huang, 2023)
    - FCSGD_GL
    - FCAdam_GL

Group 3: AOFGD family (Xiang, Zhao, Su, Zhou, Ye & Huang, 2025)
    - AOFGD_SGD
    - AOFGD_Adam

Mathematical definitions
------------------------

--- AdaGL ---

GL fractional gradient (K=10 terms, order alpha):
    g_frac_t = sum_{j=0}^{K-1} (-1)^j * C(alpha, j) * g_{t-j}
where
    C(alpha, j) = Gamma(alpha+1) / (Gamma(j+1) * Gamma(alpha-j+1))

Equivalently, using the recursive formulation (avoids Gamma for j>=1):
    w_0 = 1
    w_j = (1 - (alpha+1)/j) * w_{j-1},  j >= 1
    g_frac_t = sum_{j=0}^{K-1} w_j * g_{t-j}

Step-size control coefficient (element-wise softsign function):
    C_t = 1.1 - 0.5 / (1 + |g_t - g_{t-1}|)
    C_t in [0.6, 1.1)

Adam moments using fractional gradient:
    m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_frac_t
    v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_frac_t^2

Bias-corrected update:
    m_hat_t = m_t / (1 - beta_1^t)
    v_hat_t = v_t / (1 - beta_2^t)

Final update:
    theta_t = theta_{t-1} - eta * C_t * m_hat_t / (sqrt(v_hat_t) + eps)

--- FCSGD_GL / FCAdam_GL ---

GL fractional gradient with stochastic perturbation (Zhou et al., 2023):
    g_frac_t = g_t + sum_{i=1}^{10} c[i-1] * w_i * g_{t-i}
where
    c[i-1] ~ Bernoulli(perturbation_prob)  (random 0 or 1 at each step)
    w_0 = 1
    w_i = (1 - (alpha+1)/i) * w_{i-1},  i >= 1

The formula sums over 10 historical terms (i=1 to 10) plus the current
gradient g_t, for 11 total terms. The default history_size=11 matches this.
Users who prefer fewer terms may reduce history_size; the sum then runs
from i=1 to history_size-1.

Note on secondary sources: The Mu et al. (AdaGL, 2024) paper's description
of Zhou et al.'s recursion contains two transcription errors — it states
w_0 = 0 (should be w_0 = 1, as proven in Zhou et al.'s own Theorem 1) and
uses (i+1) in the denominator (should be i). The implementation here uses
the correct standard GL recursion w_k = (1 - (alpha+1)/k) * w_{k-1},
w_0 = 1, which is the form that avoids the Gamma function as intended by
Zhou et al.

FCSGD_GL update:
    theta_t = theta_{t-1} - eta * g_frac_t

FCAdam_GL update (fractional gradient enters BOTH moments):
    m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_frac_t
    v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_frac_t^2
    theta_t = theta_{t-1} - eta * m_hat_t / (sqrt(v_hat_t) + eps)

Key difference from AdaGL: no step-size control coefficient C_t,
and the second moment also uses the fractional gradient (not the raw gradient).

--- AOFGD family ---

Adaptive-order Caputo-type fractional gradient (Xiang et al., 2025).

The Caputo-type gradient modification (Herrera-style, but with time-varying
order):
    g_frac_t = g_t * f_{alpha_t}(theta_t)
where
    f_{alpha}(theta) = (|theta| + eps)^(1 - alpha) / Gamma(2 - alpha)

Convergence Evaluation Factor (CEF):
    CEF_t = ||g_t||_2 / (||g_{t-1}||_2 + eps_cef)

The CEF measures how much the gradient norm changed. Values close to 1
indicate stable gradient magnitude; values far from 1 indicate rapid change.

Adaptive order rule:
    If CEF_t > 1 + cef_threshold (gradient growing):
        alpha_t = alpha_{t-1} - delta_alpha  (increase memory)
    If CEF_t < 1 - cef_threshold (gradient shrinking):
        alpha_t = alpha_{t-1} + delta_alpha  (reduce memory)
    Otherwise:
        alpha_t = alpha_{t-1}

    alpha_t = clip(alpha_t, alpha_min, alpha_max)

EMA smoothing of the order (optional, improves stability):
    alpha_smooth_t = ema_gamma * alpha_smooth_{t-1} + (1-ema_gamma) * alpha_t

AOFGD_SGD update:
    theta_t = theta_{t-1} - eta * g_frac_t

AOFGD_Adam update (raw gradient for second moment, fractional for first):
    m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_frac_t
    v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
    theta_t = theta_{t-1} - eta * m_hat_t / (sqrt(v_hat_t) + eps)

Note: Keeping the raw gradient in the second moment is consistent with the
interpretation in the Adaptive Memory-Fractional framework developed in this
project. It prevents the adaptive order from also destabilizing the variance
estimate.

Key difference from the Adaptive Memory-Fractional framework:
- AOFGD adapts the fractional order alpha_t at each step.
- The gradient is always fractional (there is no lambda_t=0 fallback to
  the classical optimizer).
- The adaptation mechanism targets the derivative order, not the memory
  contribution weight.

References
----------
1. Chen, S.; Zhang, C.; Mu, H. An Adaptive Learning Rate Deep Learning
   Optimizer Using Long and Short-Term Gradients Based on G-L Fractional-
   Order Derivative. Neural Processing Letters 2024, 56, 106.
   https://doi.org/10.1007/s11063-024-11571-7

2. Zhou, X.; Zhao, C.; Huang, Y. A Deep Learning Optimizer Based on
   Grunwald-Letnikov Fractional Order Definition. Mathematics 2023, 11(2),
   316. https://doi.org/10.3390/math11020316

3. Xiang, K.; Zhao, C.; Su, Q.; Zhou, X.; Ye, J.; Huang, Y. AOFGD:
   Adaptive Order Fractional Gradient Descent Method. Preprint, 2025.
   https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5717167

Usage example
-------------
    from class_d_related_work_optimizers import AdaGL, FCAdam_GL, AOFGD_Adam

    optimizer = AdaGL(lr=1e-3, alpha=0.9, history_size=10)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
    model.fit(x, y, epochs=10)

Notes
-----
- Sparse gradients are densified in _resource_apply_sparse for simplicity.
- The second moment in AOFGD_Adam uses the raw gradient (not the fractional
  gradient) for numerical robustness.
- In FCAdam_GL, the second moment uses the fractional gradient, matching the
  original paper.
- The Bernoulli perturbation coefficient in FCSGD_GL / FCAdam_GL is applied
  per-element and per-step, as described in Zhou et al. (2023). The
  perturbation_prob parameter controls the probability that each historical
  term is included. Setting perturbation_prob=1.0 disables perturbation and
  recovers a plain GL memory optimizer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import tensorflow as tf


# =============================================================================
# Shared GL coefficient helper
# =============================================================================


def _compute_gl_coefficients(alpha: float, n_terms: int) -> List[float]:
    """
    Compute the short-memory Grunwald-Letnikov coefficient sequence.

    Recursion:
        w_0 = 1
        w_k = (1 - (alpha + 1) / k) * w_{k-1},  k >= 1

    For alpha numerically equal to 1.0, returns [1, 0, ..., 0].
    This is the standard GL recursion used throughout this project.
    """
    if abs(alpha - 1.0) < 1e-12:
        return [1.0] + [0.0] * (n_terms - 1)
    coeffs = [1.0]
    for k in range(1, n_terms):
        coeffs.append((1.0 - (alpha + 1.0) / k) * coeffs[-1])
    return coeffs


# =============================================================================
# Group 1: AdaGL
# Chen, Zhang & Mu, Neural Processing Letters 2024
# =============================================================================


class AdaGL(tf.keras.optimizers.Optimizer):
    """
    AdaGL: Adaptive Learning Rate Deep Learning Optimizer Using Long and
    Short-Term Gradients Based on G-L Fractional-Order Derivative.

    Reference: Chen, Zhang & Mu (2024), Neural Processing Letters, 56, 106.

    Algorithm:
        1. Compute G-L fractional gradient g_frac_t using history of length K.
        2. Compute step-size control coefficient C_t from short-term gradient
           change (element-wise softsign function).
        3. Update Adam moments using g_frac_t for BOTH first and second moments.
        4. Apply bias-corrected Adam update scaled by C_t.

    The step-size control coefficient is:
        C_t = 1.1 - 0.5 / (1 + |g_t - g_{t-1}|),   C_t in [0.6, 1.1)

    When |g_t - g_{t-1}| is small (stable), C_t -> 0.6 (smaller step).
    When |g_t - g_{t-1}| is large (unstable/sharp region), C_t -> 1.1
    (larger step to escape).

    Key differences from the Adaptive Memory-Fractional framework:
    - No mixing coefficient lambda_t; g_frac_t always replaces g_t entirely.
    - No fallback to the classical optimizer (lambda_t = 0 does not exist).
    - Second moment uses g_frac_t, not the raw gradient.
    - Step-size modulation via C_t acts on the learning rate, not on
      gradient direction.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        alpha: float = 0.9,
        history_size: int = 10,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        coeff_epsilon: float = 1e-12,
        name: str = "AdaGL",
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        learning_rate : float
            Base learning rate eta.
        lr : float, optional
            Legacy alias for learning_rate.
        alpha : float
            Fractional order for the G-L gradient, in (0, 2).
        history_size : int
            Number of past gradient steps to retain. The original paper
            uses 10 terms.
        beta_1, beta_2 : float
            Adam first- and second-moment decay rates.
        epsilon : float
            Numerical stability constant in the Adam denominator.
        amsgrad : bool
            Whether to use the AMSGrad variant.
        coeff_epsilon : float
            Small constant to guard against degenerate coefficient sums.
        """
        super(AdaGL, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 < float(alpha) < 2.0):
            raise ValueError("`alpha` must satisfy 0 < alpha < 2.")
        if not isinstance(history_size, int) or history_size < 1:
            raise ValueError("`history_size` must be an integer >= 1.")
        if not (0.0 <= beta_1 < 1.0):
            raise ValueError("`beta_1` must satisfy 0 <= beta_1 < 1.")
        if not (0.0 <= beta_2 < 1.0):
            raise ValueError("`beta_2` must satisfy 0 <= beta_2 < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.alpha = float(alpha)
        self.history_size = int(history_size)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.amsgrad = bool(amsgrad)
        self.coeff_epsilon = float(coeff_epsilon)

        self._gl_coeffs = _compute_gl_coefficients(self.alpha, self.history_size)

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    def _history_slot_names(self) -> List[str]:
        return [f"grad_hist_{i}" for i in range(self.history_size)]

    def _create_slots(self, var_list):
        for var in var_list:
            for sn in self._history_slot_names():
                self.add_slot(var, sn)
            self.add_slot(var, "prev_grad")
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            if self.amsgrad:
                self.add_slot(var, "vhat")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_dense(self, grad, var):
        if isinstance(grad, tf.IndexedSlices):
            return tf.scatter_nd(
                tf.expand_dims(grad.indices, 1),
                grad.values,
                tf.shape(var),
            )
        return grad

    def _get_history(self, var):
        slots = [self.get_slot(var, sn) for sn in self._history_slot_names()]
        return tf.stack(slots, axis=-1)

    def _assign_history(self, var, new_history):
        ops = []
        for i, sn in enumerate(self._history_slot_names()):
            ops.append(
                self.get_slot(var, sn).assign(
                    new_history[..., i], use_locking=self._use_locking
                )
            )
        return tf.group(*ops)

    def _gl_fractional_gradient(self, raw_grad, var):
        """
        Compute the GL fractional gradient and update the history buffer.

        Returns (new_history, g_frac).
        """
        dtype = var.dtype.base_dtype
        history = self._get_history(var)

        # Shift history: prepend current gradient, drop oldest
        new_history = tf.concat(
            [tf.expand_dims(raw_grad, axis=-1), history[..., :-1]],
            axis=-1,
        )

        if abs(self.alpha - 1.0) < 1e-12 or self.history_size == 1:
            return new_history, raw_grad

        coeffs = tf.constant(self._gl_coeffs, dtype=dtype)
        reshape = [1] * len(var.shape) + [self.history_size]
        coeffs = tf.reshape(coeffs, reshape)
        g_frac = tf.reduce_sum(new_history * coeffs, axis=-1)
        return new_history, g_frac

    def _step_size_coefficient(self, raw_grad, prev_grad, dtype):
        """
        Element-wise step-size control coefficient C_t.

        C_t = 1.1 - 0.5 / (1 + |g_t - g_{t-1}|)

        Domain: C_t in [0.6, 1.1). When consecutive gradients are very
        different, C_t approaches 1.1, allowing a slightly larger step. When
        gradients are stable, C_t approaches 0.6, applying mild compression.
        """
        diff = tf.abs(tf.cast(raw_grad, dtype) - tf.cast(prev_grad, dtype))
        c = tf.cast(1.1, dtype) - tf.cast(0.5, dtype) / (
            tf.cast(1.0, dtype) + diff
        )
        return c

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)
        b1 = tf.cast(self.beta_1, dtype)
        b2 = tf.cast(self.beta_2, dtype)
        eps = tf.cast(self.epsilon, dtype)
        step = tf.cast(self.iterations + 1, dtype)

        raw_grad = self._to_dense(grad, var)
        prev_grad = self.get_slot(var, "prev_grad")
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # Step-size coefficient uses the raw gradient difference
        c_t = self._step_size_coefficient(raw_grad, prev_grad, dtype)

        # GL fractional gradient
        new_history, g_frac = self._gl_fractional_gradient(raw_grad, var)

        # Update moments with fractional gradient (both m and v)
        new_m = b1 * m + (1.0 - b1) * g_frac
        new_v = b2 * v + (1.0 - b2) * tf.square(g_frac)

        m_t = m.assign(new_m, use_locking=self._use_locking)
        v_t = v.assign(new_v, use_locking=self._use_locking)

        b1_pow = tf.pow(b1, step)
        b2_pow = tf.pow(b2, step)
        m_hat = new_m / (1.0 - b1_pow)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            new_vhat = tf.maximum(vhat, new_v)
            vhat_t = vhat.assign(new_vhat, use_locking=self._use_locking)
            v_hat = new_vhat / (1.0 - b2_pow)
        else:
            vhat_t = tf.no_op()
            v_hat = new_v / (1.0 - b2_pow)

        # C_t scales the learning rate element-wise
        var_update = var.assign_sub(
            lr_t * c_t * m_hat / (tf.sqrt(v_hat) + eps),
            use_locking=self._use_locking,
        )

        hist_op = self._assign_history(var, new_history)
        prev_op = prev_grad.assign(raw_grad, use_locking=self._use_locking)

        return tf.group(var_update, hist_op, prev_op, m_t, v_t, vhat_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        dense = tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)
        )
        return self._resource_apply_dense(dense, var, apply_state=apply_state)

    def get_config(self):
        cfg = super(AdaGL, self).get_config()
        cfg.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "alpha": self.alpha,
                "history_size": self.history_size,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "coeff_epsilon": self.coeff_epsilon,
            }
        )
        return cfg


# =============================================================================
# Group 2: FCSGD_GL and FCAdam_GL
# Zhou, Zhao & Huang, Mathematics 2023
# =============================================================================


class _FCGLBase:
    """
    Shared GL gradient construction for FCSGD_GL and FCAdam_GL.

    Implements the stochastic perturbation design from Zhou et al. (2023):
        g_frac_t = g_t + sum_{i=1}^{10} c[i-1] * w_i * g_{t-i}
    where c[i-1] ~ Bernoulli(perturbation_prob).

    The sum covers 10 historical terms (i=1 to 10), plus the current gradient
    g_t, for 11 total terms. With the default history_size=11, new_history has
    shape [..., 11]: index 0 holds g_t and indices 1-10 hold g_{t-1}...g_{t-10}.
    Taking new_history[..., 1:] then gives exactly the 10 historical slots, and
    self._gl_coeffs[1:] gives w_1...w_{10}. The result is:
        g_frac_t = g_t + sum_{i=1}^{10} c[i-1] * w_i * g_{t-i}
    matching Zhou et al. (2023) Eq. 10.

    GL coefficient recursion (standard GL, avoids Gamma function):
        w_0 = 1
        w_k = (1 - (alpha+1)/k) * w_{k-1},  k >= 1

    Note: The Mu et al. (AdaGL, 2024) paper's secondary description of Zhou
    et al.'s formula contains two transcription errors: it states w_0=0
    (Zhou et al.'s own Theorem 1 proof states w_0=1) and writes (i+1) in
    the denominator (the correct Gamma-eliminating recursion uses i). This
    implementation uses the correct formula from the original Zhou et al. paper.

    Setting perturbation_prob=1.0 disables stochastic perturbation and
    recovers a deterministic GL memory gradient. Setting it to 0.5 matches
    the spirit of the original paper (random inclusion of historical terms).
    """

    def _init_fcgl_args(
        self,
        alpha: float = 0.9,
        history_size: int = 10,
        perturbation_prob: float = 0.5,
        coeff_epsilon: float = 1e-12,
    ) -> None:
        if not (0.0 < float(alpha) < 2.0):
            raise ValueError("`alpha` must satisfy 0 < alpha < 2.")
        if not isinstance(history_size, int) or history_size < 1:
            raise ValueError("`history_size` must be an integer >= 1.")
        if not (0.0 <= perturbation_prob <= 1.0):
            raise ValueError("`perturbation_prob` must be in [0, 1].")

        self.alpha = float(alpha)
        self.history_size = int(history_size)
        self.perturbation_prob = float(perturbation_prob)
        self.coeff_epsilon = float(coeff_epsilon)

        # Full coefficient vector w_0..w_{K-1}
        self._gl_coeffs = _compute_gl_coefficients(self.alpha, self.history_size)

    def _history_slot_names(self) -> List[str]:
        return [f"grad_hist_{i}" for i in range(self.history_size)]

    def _create_history_slots(self, var_list):
        for var in var_list:
            for sn in self._history_slot_names():
                self.add_slot(var, sn)

    def _to_dense(self, grad, var):
        if isinstance(grad, tf.IndexedSlices):
            return tf.scatter_nd(
                tf.expand_dims(grad.indices, 1),
                grad.values,
                tf.shape(var),
            )
        return grad

    def _get_history(self, var):
        return tf.stack(
            [self.get_slot(var, sn) for sn in self._history_slot_names()],
            axis=-1,
        )

    def _assign_history(self, var, new_history):
        ops = []
        for i, sn in enumerate(self._history_slot_names()):
            ops.append(
                self.get_slot(var, sn).assign(
                    new_history[..., i], use_locking=self._use_locking
                )
            )
        return tf.group(*ops)

    def _fcgl_gradient(self, raw_grad, var):
        """
        Compute the stochastically perturbed GL fractional gradient.

        g_frac_t = g_t + sum_{i=1}^{history_size-1} c[i-1] * w_i * g_{t-i}

        With the default history_size=11 this gives 10 historical terms
        (i=1 to 10), matching the formula in Zhou et al. (2023) Eq. 10.
        The perturbation mask c is sampled fresh each call from Bernoulli(p).
        """
        dtype = var.dtype.base_dtype
        history = self._get_history(var)

        # Shift history
        new_history = tf.concat(
            [tf.expand_dims(raw_grad, axis=-1), history[..., :-1]],
            axis=-1,
        )

        if abs(self.alpha - 1.0) < 1e-12 or self.history_size == 1:
            return new_history, raw_grad

        # w_0 = 1 corresponds to the current gradient (already included as g_t).
        # Historical terms start at index 1.
        past_history = new_history[..., 1:]  # shape: var_shape + [K-1]
        past_coeffs = self._gl_coeffs[1:]    # w_1 .. w_{K-1}

        # Bernoulli perturbation mask: shape [K-1], broadcast over var dims
        n_past = self.history_size - 1
        mask = tf.cast(
            tf.less(
                tf.random.uniform([n_past], dtype=tf.float32),
                tf.constant(self.perturbation_prob, dtype=tf.float32),
            ),
            dtype=dtype,
        )  # each entry is 1 with prob p, 0 otherwise

        coeffs_tensor = tf.constant(past_coeffs, dtype=dtype)  # [K-1]
        effective_coeffs = mask * coeffs_tensor                 # [K-1]

        # Reshape for broadcast
        reshape = [1] * len(var.shape) + [n_past]
        effective_coeffs = tf.reshape(effective_coeffs, reshape)

        # Weighted sum of past gradients
        past_contribution = tf.reduce_sum(past_history * effective_coeffs, axis=-1)
        g_frac = raw_grad + past_contribution

        return new_history, g_frac

    def _fcgl_config(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "history_size": self.history_size,
            "perturbation_prob": self.perturbation_prob,
            "coeff_epsilon": self.coeff_epsilon,
        }


class FCSGD_GL(_FCGLBase, tf.keras.optimizers.Optimizer):
    """
    FCSGD_GL: Fractional Calculus SGD based on the G-L definition.

    Reference: Zhou, Zhao & Huang, Mathematics 2023, 11(2), 316.

    The gradient is replaced by a stochastically perturbed GL fractional
    gradient. The update rule is otherwise plain SGD (with optional momentum).

    Update:
        g_frac_t = g_t + sum_{i=1}^{10} c[i-1] * w_i * g_{t-i}
        theta_{t+1} = theta_t - eta * g_frac_t

    The default history_size=11 stores 11 gradient slots: the current gradient
    at index 0 and 10 historical gradients at indices 1-10, giving the 10-term
    historical sum in the formula above (matching Zhou et al. (2023) Eq. 10).

    The stochastic perturbation (Bernoulli mask on historical terms) is the
    key novelty from Zhou et al. (2023) that distinguishes this from a plain
    GL memory gradient. Setting perturbation_prob=1.0 removes the stochasticity.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        lr: float = None,
        momentum: float = 0.0,
        nesterov: bool = False,
        alpha: float = 0.9,
        history_size: int = 11,
        perturbation_prob: float = 0.5,
        coeff_epsilon: float = 1e-12,
        name: str = "FCSGD_GL",
        **kwargs: Any,
    ):
        super(FCSGD_GL, self).__init__(name, **kwargs)
        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= momentum < 1.0):
            raise ValueError("`momentum` must satisfy 0 <= momentum < 1.")

        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self._init_fcgl_args(
            alpha=alpha,
            history_size=history_size,
            perturbation_prob=perturbation_prob,
            coeff_epsilon=coeff_epsilon,
        )

    def _create_slots(self, var_list):
        self._create_history_slots(var_list)
        if self.momentum > 0.0:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)
        raw_grad = self._to_dense(grad, var)

        new_history, g_frac = self._fcgl_gradient(raw_grad, var)
        hist_op = self._assign_history(var, new_history)

        if self.momentum > 0.0:
            mu = tf.cast(self.momentum, dtype)
            m = self.get_slot(var, "momentum")
            new_m = mu * m - lr_t * g_frac
            m_op = m.assign(new_m, use_locking=self._use_locking)
            if self.nesterov:
                var_op = var.assign_add(
                    mu * new_m - lr_t * g_frac, use_locking=self._use_locking
                )
            else:
                var_op = var.assign_add(new_m, use_locking=self._use_locking)
            return tf.group(var_op, hist_op, m_op)

        var_op = var.assign_sub(lr_t * g_frac, use_locking=self._use_locking)
        return tf.group(var_op, hist_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        dense = tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)
        )
        return self._resource_apply_dense(dense, var, apply_state=apply_state)

    def get_config(self):
        cfg = super(FCSGD_GL, self).get_config()
        cfg.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                **self._fcgl_config(),
            }
        )
        return cfg


class FCAdam_GL(_FCGLBase, tf.keras.optimizers.Optimizer):
    """
    FCAdam_GL: Fractional Calculus Adam based on the G-L definition.

    Reference: Zhou, Zhao & Huang, Mathematics 2023, 11(2), 316.

    The GL fractional gradient (with Bernoulli perturbation) enters BOTH
    the first and second Adam moment estimates. This is the key structural
    difference from the Adaptive Memory-Fractional FAdam, where only the
    first moment uses the fractional/mixed gradient while the second moment
    always uses the raw gradient.

    Update:
        g_frac_t = g_t + sum_{i=1}^{10} c[i-1] * w_i * g_{t-i}
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_frac_t
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_frac_t^2
        theta_{t+1} = theta_t - eta * m_hat_t / (sqrt(v_hat_t) + eps)

    The default history_size=11 stores 11 gradient slots: the current gradient
    at index 0 and 10 historical gradients at indices 1-10, giving the 10-term
    historical sum in the formula above (matching Zhou et al. (2023) Eq. 10).

    There is no step-size control coefficient (unlike AdaGL).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        alpha: float = 0.9,
        history_size: int = 11,
        perturbation_prob: float = 0.5,
        coeff_epsilon: float = 1e-12,
        name: str = "FCAdam_GL",
        **kwargs: Any,
    ):
        super(FCAdam_GL, self).__init__(name, **kwargs)
        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= beta_1 < 1.0):
            raise ValueError("`beta_1` must satisfy 0 <= beta_1 < 1.")
        if not (0.0 <= beta_2 < 1.0):
            raise ValueError("`beta_2` must satisfy 0 <= beta_2 < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.amsgrad = bool(amsgrad)
        self._init_fcgl_args(
            alpha=alpha,
            history_size=history_size,
            perturbation_prob=perturbation_prob,
            coeff_epsilon=coeff_epsilon,
        )

    def _create_slots(self, var_list):
        self._create_history_slots(var_list)
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            if self.amsgrad:
                self.add_slot(var, "vhat")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)
        b1 = tf.cast(self.beta_1, dtype)
        b2 = tf.cast(self.beta_2, dtype)
        eps = tf.cast(self.epsilon, dtype)
        step = tf.cast(self.iterations + 1, dtype)

        raw_grad = self._to_dense(grad, var)
        new_history, g_frac = self._fcgl_gradient(raw_grad, var)
        hist_op = self._assign_history(var, new_history)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # Both moments use the fractional gradient (original paper design)
        new_m = b1 * m + (1.0 - b1) * g_frac
        new_v = b2 * v + (1.0 - b2) * tf.square(g_frac)

        m_t = m.assign(new_m, use_locking=self._use_locking)
        v_t = v.assign(new_v, use_locking=self._use_locking)

        b1_pow = tf.pow(b1, step)
        b2_pow = tf.pow(b2, step)
        m_hat = new_m / (1.0 - b1_pow)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            new_vhat = tf.maximum(vhat, new_v)
            vhat_t = vhat.assign(new_vhat, use_locking=self._use_locking)
            v_hat = new_vhat / (1.0 - b2_pow)
        else:
            vhat_t = tf.no_op()
            v_hat = new_v / (1.0 - b2_pow)

        var_op = var.assign_sub(
            lr_t * m_hat / (tf.sqrt(v_hat) + eps),
            use_locking=self._use_locking,
        )
        return tf.group(var_op, hist_op, m_t, v_t, vhat_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        dense = tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)
        )
        return self._resource_apply_dense(dense, var, apply_state=apply_state)

    def get_config(self):
        cfg = super(FCAdam_GL, self).get_config()
        cfg.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                **self._fcgl_config(),
            }
        )
        return cfg


# =============================================================================
# Group 3: AOFGD family
# Xiang, Zhao, Su, Zhou, Ye & Huang, SSRN Preprint 2025
# =============================================================================


class _AOFGDBase:
    """
    Shared adaptive-order Caputo-type gradient construction for AOFGD.

    Implements an adaptive fractional order alpha_t based on a Convergence
    Evaluation Factor (CEF) derived from the ratio of consecutive gradient
    norms. The Caputo-type gradient modification follows the Herrera-style
    construction:

        f_{alpha}(theta) = (|theta| + eps_frac)^(1 - alpha) / Gamma(2 - alpha)
        g_frac_t = g_t * f_{alpha_t}(theta_t)

    The order alpha_t evolves according to:
        CEF_t = ||g_t||_2 / (||g_{t-1}||_2 + eps_cef)

        If CEF_t > 1 + cef_threshold (gradient growing):
            alpha_t = alpha_{t-1} - delta_alpha  [increase memory]
        If CEF_t < 1 - cef_threshold (gradient shrinking):
            alpha_t = alpha_{t-1} + delta_alpha  [reduce memory]
        Else:
            alpha_t = alpha_{t-1}

        alpha_t = clip(alpha_t, alpha_min, alpha_max)

    An EMA smoother is applied to alpha_t for stability:
        alpha_smooth_t = ema_gamma * alpha_smooth_{t-1} + (1-ema_gamma) * alpha_t

    The effective order used for the gradient computation is alpha_smooth_t.
    """

    def _init_aofgd_args(
        self,
        alpha_init: float = 0.9,
        alpha_min: float = 0.5,
        alpha_max: float = 1.5,
        delta_alpha: float = 0.02,
        cef_threshold: float = 0.05,
        ema_gamma: float = 0.95,
        frac_epsilon: float = 1e-6,
        eps_cef: float = 1e-8,
    ) -> None:
        if not (0.0 < float(alpha_init) < 2.0):
            raise ValueError("`alpha_init` must satisfy 0 < alpha_init < 2.")
        if not (0.0 < float(alpha_min) <= float(alpha_init)):
            raise ValueError("`alpha_min` must satisfy 0 < alpha_min <= alpha_init.")
        if not (float(alpha_init) <= float(alpha_max) < 2.0):
            raise ValueError("`alpha_max` must satisfy alpha_init <= alpha_max < 2.")
        if delta_alpha < 0.0:
            raise ValueError("`delta_alpha` must be >= 0.")
        if not (0.0 <= ema_gamma < 1.0):
            raise ValueError("`ema_gamma` must satisfy 0 <= ema_gamma < 1.")
        if frac_epsilon <= 0.0:
            raise ValueError("`frac_epsilon` must be > 0.")
        if eps_cef <= 0.0:
            raise ValueError("`eps_cef` must be > 0.")

        self.alpha_init = float(alpha_init)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.delta_alpha = float(delta_alpha)
        self.cef_threshold = float(cef_threshold)
        self.ema_gamma = float(ema_gamma)
        self.frac_epsilon = float(frac_epsilon)
        self.eps_cef = float(eps_cef)

        # Precompute gamma(2 - alpha) for the initial order (updated dynamically)
        self._gamma_cache = {}

    def _gamma_factor(self, alpha_val: float) -> float:
        """Return Gamma(2 - alpha) with a small cache."""
        if alpha_val not in self._gamma_cache:
            self._gamma_cache[alpha_val] = float(math.gamma(2.0 - alpha_val))
        return self._gamma_cache[alpha_val]

    def _create_aofgd_slots(self, var_list):
        """Create slots for the current alpha state and previous gradient norm."""
        for var in var_list:
            # Store current (smoothed) fractional order as a scalar slot
            self.add_slot(
                var,
                "alpha_smooth",
                initializer=tf.keras.initializers.Constant(self.alpha_init),
            )
            # Store previous gradient L2 norm (scalar stored as var-shaped fill).
            # Initialized to a small positive value so that CEF = curr/prev is
            # close to 1.0 on the first step rather than diverging to +inf
            # (which would happen with the default zero initialization and cause
            # a spurious forced alpha decrease before any real gradient has been
            # observed).
            self.add_slot(
                var,
                "prev_grad_norm",
                initializer=tf.keras.initializers.Constant(1e-4),
            )

    def _to_dense(self, grad, var):
        if isinstance(grad, tf.IndexedSlices):
            return tf.scatter_nd(
                tf.expand_dims(grad.indices, 1),
                grad.values,
                tf.shape(var),
            )
        return grad

    def _scalar_from_slot(self, slot, dtype):
        return tf.cast(tf.reduce_mean(slot), dtype)

    def _fill_like(self, scalar, var):
        return tf.ones_like(var, dtype=var.dtype.base_dtype) * tf.cast(
            scalar, var.dtype.base_dtype
        )

    def _compute_caputo_gradient(self, raw_grad, var, alpha_val_scalar):
        """
        Apply the Caputo-type Herrera-style scaling with the current alpha.

        f_{alpha}(theta) = (|theta| + eps)^(1 - alpha) / Gamma(2 - alpha)
        g_frac = g * f_{alpha}(theta)

        alpha_val_scalar is a TF scalar tensor (the current smoothed order).
        """
        dtype = var.dtype.base_dtype
        eps = tf.cast(self.frac_epsilon, dtype)
        exponent = 1.0 - alpha_val_scalar          # scalar tensor
        # Gamma(2 - alpha): compute in Python as scalar; approximate with
        # the value at alpha_init for numerical stability (order changes slowly)
        # For fully dynamic computation, we use tf.math.lgamma.
        log_gamma = tf.math.lgamma(
            tf.cast(2.0, dtype) - tf.cast(alpha_val_scalar, dtype)
        )
        gamma_val = tf.exp(log_gamma)

        factor = tf.pow(tf.abs(var) + eps, tf.cast(exponent, dtype)) / gamma_val
        return raw_grad * factor

    def _update_alpha(self, raw_grad, var, dtype):
        """
        Update the fractional order alpha based on the CEF.

        Returns (new_alpha_smooth, alpha_update_op).
        The caller is responsible for executing the update op.
        """
        prev_norm_slot = self.get_slot(var, "prev_grad_norm")
        alpha_slot = self.get_slot(var, "alpha_smooth")

        alpha_curr = self._scalar_from_slot(alpha_slot, dtype)
        prev_norm = self._scalar_from_slot(prev_norm_slot, dtype)

        # Convergence evaluation factor
        curr_norm = tf.cast(
            tf.sqrt(tf.reduce_sum(tf.square(raw_grad)) + 1e-30), dtype
        )
        cef = curr_norm / (prev_norm + tf.cast(self.eps_cef, dtype))

        thr = tf.cast(self.cef_threshold, dtype)
        d_alpha = tf.cast(self.delta_alpha, dtype)

        # Ratchet rule
        new_alpha_raw = tf.cond(
            cef > tf.cast(1.0, dtype) + thr,
            true_fn=lambda: alpha_curr - d_alpha,
            false_fn=lambda: tf.cond(
                cef < tf.cast(1.0, dtype) - thr,
                true_fn=lambda: alpha_curr + d_alpha,
                false_fn=lambda: alpha_curr,
            ),
        )
        new_alpha_raw = tf.clip_by_value(
            new_alpha_raw,
            tf.cast(self.alpha_min, dtype),
            tf.cast(self.alpha_max, dtype),
        )

        # EMA smoothing
        gamma = tf.cast(self.ema_gamma, dtype)
        new_alpha_smooth = gamma * alpha_curr + (1.0 - gamma) * new_alpha_raw

        # Write back
        alpha_op = alpha_slot.assign(
            self._fill_like(new_alpha_smooth, var),
            use_locking=self._use_locking,
        )
        norm_op = prev_norm_slot.assign(
            self._fill_like(curr_norm, var),
            use_locking=self._use_locking,
        )
        return new_alpha_smooth, tf.group(alpha_op, norm_op)

    def current_alpha_values(self) -> List[tf.Tensor]:
        """Return the current smoothed fractional order for each variable."""
        return [
            tf.reduce_mean(w)
            for w in self.weights
            if "alpha_smooth" in w.name
        ]

    def _aofgd_config(self) -> Dict[str, Any]:
        return {
            "alpha_init": self.alpha_init,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "delta_alpha": self.delta_alpha,
            "cef_threshold": self.cef_threshold,
            "ema_gamma": self.ema_gamma,
            "frac_epsilon": self.frac_epsilon,
            "eps_cef": self.eps_cef,
        }


class AOFGD_SGD(_AOFGDBase, tf.keras.optimizers.Optimizer):
    """
    AOFGD_SGD: Adaptive Order Fractional Gradient Descent with SGD base.

    Reference: Xiang et al., SSRN Preprint 2025.

    The fractional order alpha_t adapts at each iteration based on the
    Convergence Evaluation Factor (CEF = ||g_t|| / ||g_{t-1}||). The gradient
    is always fractional; there is no lambda_t=0 fallback to plain SGD. This
    is the key difference from the Adaptive Memory-Fractional framework.

    Update:
        CEF_t = ||g_t|| / ||g_{t-1}||
        alpha_t adapted via ratchet + EMA smoothing
        g_frac_t = g_t * (|theta_t| + eps)^(1 - alpha_t) / Gamma(2 - alpha_t)
        theta_{t+1} = theta_t - eta * g_frac_t
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        lr: float = None,
        momentum: float = 0.0,
        nesterov: bool = False,
        alpha_init: float = 0.9,
        alpha_min: float = 0.5,
        alpha_max: float = 1.5,
        delta_alpha: float = 0.02,
        cef_threshold: float = 0.05,
        ema_gamma: float = 0.95,
        frac_epsilon: float = 1e-6,
        eps_cef: float = 1e-8,
        name: str = "AOFGD_SGD",
        **kwargs: Any,
    ):
        super(AOFGD_SGD, self).__init__(name, **kwargs)
        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= momentum < 1.0):
            raise ValueError("`momentum` must satisfy 0 <= momentum < 1.")

        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self._init_aofgd_args(
            alpha_init=alpha_init,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            delta_alpha=delta_alpha,
            cef_threshold=cef_threshold,
            ema_gamma=ema_gamma,
            frac_epsilon=frac_epsilon,
            eps_cef=eps_cef,
        )

    def _create_slots(self, var_list):
        self._create_aofgd_slots(var_list)
        if self.momentum > 0.0:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)
        raw_grad = self._to_dense(grad, var)

        # Adapt the fractional order
        alpha_smooth, alpha_op = self._update_alpha(raw_grad, var, dtype)

        # Apply Caputo-type fractional scaling
        g_frac = self._compute_caputo_gradient(raw_grad, var, alpha_smooth)

        if self.momentum > 0.0:
            mu = tf.cast(self.momentum, dtype)
            m = self.get_slot(var, "momentum")
            new_m = mu * m - lr_t * g_frac
            m_op = m.assign(new_m, use_locking=self._use_locking)
            if self.nesterov:
                var_op = var.assign_add(
                    mu * new_m - lr_t * g_frac, use_locking=self._use_locking
                )
            else:
                var_op = var.assign_add(new_m, use_locking=self._use_locking)
            return tf.group(var_op, alpha_op, m_op)

        var_op = var.assign_sub(lr_t * g_frac, use_locking=self._use_locking)
        return tf.group(var_op, alpha_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        dense = tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)
        )
        return self._resource_apply_dense(dense, var, apply_state=apply_state)

    def get_config(self):
        cfg = super(AOFGD_SGD, self).get_config()
        cfg.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                **self._aofgd_config(),
            }
        )
        return cfg


class AOFGD_Adam(_AOFGDBase, tf.keras.optimizers.Optimizer):
    """
    AOFGD_Adam: Adaptive Order Fractional Gradient Descent with Adam base.

    Reference: Xiang et al., SSRN Preprint 2025.

    The fractional order alpha_t adapts at each iteration via the CEF. The
    Caputo-type fractional gradient enters the first moment. The second moment
    uses the raw gradient for numerical stability. This mirrors the design
    choice made in the Adaptive Memory-Fractional FAdam and differs from
    FCAdam_GL (which uses the fractional gradient for both moments).

    Update:
        CEF_t = ||g_t|| / ||g_{t-1}||
        alpha_t adapted via ratchet + EMA smoothing
        g_frac_t = g_t * (|theta_t| + eps)^(1 - alpha_t) / Gamma(2 - alpha_t)
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_frac_t
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2       [raw gradient]
        theta_{t+1} = theta_t - eta * m_hat_t / (sqrt(v_hat_t) + eps_adam)

    The raw-gradient second moment keeps the variance estimate numerically
    stable even when alpha_t changes rapidly.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        alpha_init: float = 0.9,
        alpha_min: float = 0.5,
        alpha_max: float = 1.5,
        delta_alpha: float = 0.02,
        cef_threshold: float = 0.05,
        ema_gamma: float = 0.95,
        frac_epsilon: float = 1e-6,
        eps_cef: float = 1e-8,
        name: str = "AOFGD_Adam",
        **kwargs: Any,
    ):
        super(AOFGD_Adam, self).__init__(name, **kwargs)
        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= beta_1 < 1.0):
            raise ValueError("`beta_1` must satisfy 0 <= beta_1 < 1.")
        if not (0.0 <= beta_2 < 1.0):
            raise ValueError("`beta_2` must satisfy 0 <= beta_2 < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.amsgrad = bool(amsgrad)
        self._init_aofgd_args(
            alpha_init=alpha_init,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            delta_alpha=delta_alpha,
            cef_threshold=cef_threshold,
            ema_gamma=ema_gamma,
            frac_epsilon=frac_epsilon,
            eps_cef=eps_cef,
        )

    def _create_slots(self, var_list):
        self._create_aofgd_slots(var_list)
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            if self.amsgrad:
                self.add_slot(var, "vhat")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)
        b1 = tf.cast(self.beta_1, dtype)
        b2 = tf.cast(self.beta_2, dtype)
        eps = tf.cast(self.epsilon, dtype)
        step = tf.cast(self.iterations + 1, dtype)

        raw_grad = self._to_dense(grad, var)

        # Adapt the fractional order
        alpha_smooth, alpha_op = self._update_alpha(raw_grad, var, dtype)

        # Fractional gradient for first moment
        g_frac = self._compute_caputo_gradient(raw_grad, var, alpha_smooth)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # First moment uses fractional gradient
        new_m = b1 * m + (1.0 - b1) * g_frac
        # Second moment uses raw gradient for stability
        new_v = b2 * v + (1.0 - b2) * tf.square(raw_grad)

        m_t = m.assign(new_m, use_locking=self._use_locking)
        v_t = v.assign(new_v, use_locking=self._use_locking)

        b1_pow = tf.pow(b1, step)
        b2_pow = tf.pow(b2, step)
        m_hat = new_m / (1.0 - b1_pow)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            new_vhat = tf.maximum(vhat, new_v)
            vhat_t = vhat.assign(new_vhat, use_locking=self._use_locking)
            v_hat = new_vhat / (1.0 - b2_pow)
        else:
            vhat_t = tf.no_op()
            v_hat = new_v / (1.0 - b2_pow)

        var_op = var.assign_sub(
            lr_t * m_hat / (tf.sqrt(v_hat) + eps),
            use_locking=self._use_locking,
        )
        return tf.group(var_op, alpha_op, m_t, v_t, vhat_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        dense = tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)
        )
        return self._resource_apply_dense(dense, var, apply_state=apply_state)

    def get_config(self):
        cfg = super(AOFGD_Adam, self).get_config()
        cfg.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                **self._aofgd_config(),
            }
        )
        return cfg


# =============================================================================
# Export list
# =============================================================================

__all__ = [
    # Group 1: AdaGL
    "AdaGL",
    # Group 2: FCSGD_GL / FCAdam_GL
    "FCSGD_GL",
    "FCAdam_GL",
    # Group 3: AOFGD
    "AOFGD_SGD",
    "AOFGD_Adam",
]


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("Running smoke tests for class_d_related_work_optimizers.py")
    print("=" * 60)

    tf.random.set_seed(42)

    x = tf.random.normal((128, 10))
    y = tf.random.uniform((128,), minval=0, maxval=3, dtype=tf.int32)

    def build_model():
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(10,)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )

    configs = [
        (
            "AdaGL",
            AdaGL(lr=1e-3, alpha=0.9, history_size=10),
        ),
        (
            "FCSGD_GL",
            FCSGD_GL(lr=1e-2, alpha=0.9, history_size=11, perturbation_prob=0.5),
        ),
        (
            "FCAdam_GL",
            FCAdam_GL(lr=1e-3, alpha=0.9, history_size=11, perturbation_prob=0.5),
        ),
        (
            "AOFGD_SGD",
            AOFGD_SGD(
                lr=1e-2,
                alpha_init=0.9,
                alpha_min=0.5,
                alpha_max=1.5,
                delta_alpha=0.02,
                cef_threshold=0.05,
            ),
        ),
        (
            "AOFGD_Adam",
            AOFGD_Adam(
                lr=1e-3,
                alpha_init=0.9,
                alpha_min=0.5,
                alpha_max=1.5,
                delta_alpha=0.02,
                cef_threshold=0.05,
            ),
        ),
    ]

    for name, opt in configs:
        model = build_model()
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        hist = model.fit(x, y, epochs=2, batch_size=32, verbose=0)
        final_loss = hist.history["loss"][-1]
        final_acc = hist.history["accuracy"][-1]
        print(f"  {name:20s}  loss={final_loss:.4f}  acc={final_acc:.4f}  OK")

    print("=" * 60)
    print("All smoke tests passed.")
