"""
class_minimal_fractional_adaptive_optimizers.py  (corrected and extended)

Legacy TensorFlow / Keras implementations of adaptive memory-based
fractional optimizers.

=============================================================================
ANALYSIS OF THE ORIGINAL IMPLEMENTATION AND CORRECTIONS APPLIED
=============================================================================

This family was already designed around the two main mathematical fixes
that the plain MemoryF* family lacked: the fractional gradient was
norm-matched to the raw gradient, and all squared-gradient accumulators
used the raw gradient by default. Both mechanisms are kept unchanged.
The following corrections and improvements are applied on top, mirroring
the corrected memory_fractional_optimizers file:

(1) DESCENT SAFEGUARD (new, default on).
    For orders nu > 1 (the experiments sweep nu in {1.25, 1.5}) the
    truncated GL kernel has a NEGATIVE coefficient sum, so under
    near-constant gradients the fractional gradient points uphill. The
    convex mixing (1-lambda) g + lambda g_hat bounds the damage as long as
    lambda_t < 1/2, but the memory component still pulls against descent.
    FIX: if <g^(nu), g_t> <= 0 for a variable, the raw gradient replaces
    the fractional gradient for that variable in that step (before norm
    matching). The memory term is used exactly when it is
    descent-compatible.

(2) O(K) SLOT WRITES PER STEP (efficiency bug).
    The original stored K per-lag history slots and re-wrote ALL of them
    every step (shift-by-one via K assigns plus stack/concat).
    FIX: a single history slot of shape [K] + var.shape used as a CIRCULAR
    buffer; exactly one row is written per step (scatter_nd_update at
    position t mod K), and the GL coefficients are rotated with tf.gather
    instead of moving the data.

(3) SCALAR STATE STORED AS FULL TENSORS (memory bug).
    The mixing coefficient lambda_t and the smoothed stability state are
    scalars per variable, but were stored as full var-shaped slots filled
    with a constant (doubling optimizer memory and adding full-tensor
    writes every step).
    FIX: both are now true scalar slots (shape []). The public inspection
    methods current_mix_values() and current_stability_values() behave as
    before.

(4) KERAS-EXACT ADAM FORMULATION.
    AdaptiveMemoryFAdam now folds the bias correction into the learning
    rate and places epsilon outside the uncorrected sqrt, exactly as the
    Keras Adam baseline does. At nu = 1 (where g_eff = g_t exactly,
    independently of lambda_t) the optimizer is then bit-compatible with
    Keras Adam.

    NOTE: slot layout changes (one "grad_history" slot instead of K
    "grad_history_i" slots; scalar "frac_mix"/"stability_state"); old
    checkpoints of these optimizers are not restorable. Constructor
    arguments, training code, set_current_loss(), and the inspection
    methods are unchanged.

Main idea (unchanged)
---------------------
    1. Keep the fractional order fixed.
    2. Compute a short-memory fractional gradient
           g_t^(nu) = sum_{k=0}^{K-1} c_k^(nu) g_{t-k}.
    3. Descent safeguard, then match the fractional-gradient norm to the
       raw-gradient norm.
    4. Mix raw and fractional gradients:
           g_eff = (1 - lambda_t) g_t + lambda_t g_hat_t^(nu).
    5. Adapt only lambda_t (stability ratchet, EMA, clipping, warm-up,
       loss-aware decrease), not the fractional order.

lambda_t = 0 gives exactly the base optimizer; lambda_t > 0 adds
fractional-memory smoothing.

Implemented optimizers
----------------------
    AdaptiveMemoryFSGD          (SGD, optional momentum / Nesterov)
    AdaptiveMemoryFRMSprop      (RMSprop, optional momentum / centered)
    AdaptiveMemoryFAdam         (Adam, optional AMSGrad)
    AdaptiveMemoryFAdadelta     (Adadelta)
    -- new in this version --
    AdaptiveMemoryFAdagrad      (Adagrad)
    AdaptiveMemoryFAdamW        (AdamW, decoupled weight decay)
    AdaptiveMemoryFNadam        (Nadam, Nesterov-Adam)
    AdaptiveMemoryFAdamax       (Adamax, infinity-norm second moment)

Recommended defaults (unchanged)
--------------------------------
Neural networks:  vderiv = 0.85-0.9, history_size = 3-4,
                  initial_mix = 0.10, max_mix = 0.30, adapt_step = 0.01,
                  stability_threshold = 0.25.
Difficult analytic surfaces: history_size = 4-6, max_mix = 0.40,
                  adapt_step = 0.015.

Usage (unchanged)
-----------------
    optimizer = AdaptiveMemoryFAdam(lr=1e-3, vderiv=0.9, history_size=4)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
    model.fit(x, y, epochs=10)
    # optional, from a custom loop: optimizer.set_current_loss(loss_value)
"""

from __future__ import annotations

from typing import Any, Dict, List

import tensorflow as tf

# ---------------------------------------------------------------------------
# Base-class selection (optimizer_v2 API with _set_hyper etc.), as in the
# corrected memory_fractional_optimizers file.
# ---------------------------------------------------------------------------
if hasattr(tf.keras.optimizers.Optimizer, "_set_hyper"):
    _KerasOptimizerBase = tf.keras.optimizers.Optimizer
else:  # pragma: no cover - depends on installed TF/Keras version
    try:
        _KerasOptimizerBase = tf.keras.optimizers.legacy.Optimizer
        if not hasattr(_KerasOptimizerBase, "_set_hyper"):
            raise ImportError
    except Exception as _exc:
        raise ImportError(
            "No optimizer_v2-style Keras optimizer base class found. "
            "These optimizers require the legacy Keras optimizer API "
            "(tf.keras.optimizers.Optimizer with _set_hyper, or "
            "tf.keras.optimizers.legacy.Optimizer / the tf_keras package)."
        ) from _exc


# =============================================================================
# Shared adaptive memory-fractional helper base
# =============================================================================


class _AdaptiveMemoryFractionalBase:
    """
    Shared helper functionality for adaptive memory-based fractional
    optimizers with a circular gradient-history buffer.
    """

    _CLASSICAL_TOL = 1e-12

    # ------------------------------------------------------------------
    # Init and coefficients
    # ------------------------------------------------------------------

    def _init_adaptive_memory_fractional_args(
        self,
        vderiv: float = 0.9,
        history_size: int = 4,
        normalize_coefficients: bool = True,
        coeff_epsilon: float = 1e-12,
        initial_mix: float = 0.10,
        min_mix: float = 0.0,
        max_mix: float = 0.30,
        adapt_step: float = 0.01,
        mix_ema_gamma: float = 0.90,
        stability_ema_gamma: float = 0.95,
        stability_threshold: float = 0.25,
        stability_clip: float = 10.0,
        norm_match_fractional_gradient: bool = True,
        use_raw_gradient_for_second_moment: bool = True,
        loss_ema_beta: float = 0.95,
        loss_worsen_tolerance: float = 1e-4,
        loss_mix_decrease: float = 0.02,
        warmup_steps: int = 0,
        descent_check: bool = True,
    ) -> None:
        if not isinstance(vderiv, (float, int)):
            raise TypeError("`vderiv` must be a float.")
        if not (0.0 < float(vderiv) < 2.0):
            raise ValueError("`vderiv` must satisfy 0 < vderiv < 2.")
        if not isinstance(history_size, int) or history_size < 1:
            raise ValueError("`history_size` must be an integer >= 1.")
        if coeff_epsilon <= 0.0:
            raise ValueError("`coeff_epsilon` must be > 0.")

        if not (0.0 <= min_mix <= initial_mix <= max_mix <= 1.0):
            raise ValueError(
                "Mix values must satisfy 0 <= min_mix <= initial_mix <= max_mix <= 1."
            )
        if adapt_step < 0.0:
            raise ValueError("`adapt_step` must be >= 0.")
        if not (0.0 <= mix_ema_gamma < 1.0):
            raise ValueError("`mix_ema_gamma` must satisfy 0 <= mix_ema_gamma < 1.")
        if not (0.0 <= stability_ema_gamma < 1.0):
            raise ValueError(
                "`stability_ema_gamma` must satisfy 0 <= stability_ema_gamma < 1."
            )
        if stability_threshold < 0.0:
            raise ValueError("`stability_threshold` must be >= 0.")
        if stability_clip <= 0.0:
            raise ValueError("`stability_clip` must be > 0.")
        if not (0.0 <= loss_ema_beta < 1.0):
            raise ValueError("`loss_ema_beta` must satisfy 0 <= loss_ema_beta < 1.")
        if loss_worsen_tolerance < 0.0:
            raise ValueError("`loss_worsen_tolerance` must be >= 0.")
        if loss_mix_decrease < 0.0:
            raise ValueError("`loss_mix_decrease` must be >= 0.")
        if warmup_steps < 0:
            raise ValueError("`warmup_steps` must be >= 0.")

        self.vderiv = float(vderiv)
        self.history_size = int(history_size)
        self.normalize_coefficients = bool(normalize_coefficients)
        self.coeff_epsilon = float(coeff_epsilon)

        self.initial_mix = float(initial_mix)
        self.min_mix = float(min_mix)
        self.max_mix = float(max_mix)
        self.adapt_step = float(adapt_step)
        self.mix_ema_gamma = float(mix_ema_gamma)
        self.stability_ema_gamma = float(stability_ema_gamma)
        self.stability_threshold = float(stability_threshold)
        self.stability_clip = float(stability_clip)

        self.norm_match_fractional_gradient = bool(norm_match_fractional_gradient)
        self.use_raw_gradient_for_second_moment = bool(
            use_raw_gradient_for_second_moment
        )

        self.loss_ema_beta = float(loss_ema_beta)
        self.loss_worsen_tolerance = float(loss_worsen_tolerance)
        self.loss_mix_decrease = float(loss_mix_decrease)
        self.warmup_steps = int(warmup_steps)
        self.descent_check = bool(descent_check)

        self._fractional_coefficients = self._compute_fractional_coefficients(
            self.vderiv,
            self.history_size,
            self.normalize_coefficients,
        )

        self._current_loss_value = None
        self._loss_ema_value = None
        self._best_loss_ema_value = None
        self._loss_worse_value = 0.0

    def _compute_fractional_coefficients(
        self,
        nu: float,
        n_terms: int,
        normalize: bool = True,
    ) -> List[float]:
        """
        Short-memory Grunwald-Letnikov coefficients.

        c_0 = 1
        c_k = (1 - (nu + 1)/k) c_{k-1},  k >= 1

        Classical-limit convention: if nu is numerically 1, [1, 0, ..., 0].
        """
        if abs(nu - 1.0) < self._CLASSICAL_TOL:
            return [1.0] + [0.0] * (n_terms - 1)

        coeffs = [1.0]
        for k in range(1, n_terms):
            coeffs.append((1.0 - (nu + 1.0) / k) * coeffs[-1])

        if normalize:
            denom = sum(abs(c) for c in coeffs)
            if denom < self.coeff_epsilon:
                denom = 1.0
            coeffs = [c / denom for c in coeffs]

        return coeffs

    def gl_coefficient_sum(self) -> float:
        """Diagnostic: DC gain of the (possibly normalized) GL kernel."""
        return float(sum(self._fractional_coefficients))

    @property
    def _is_classical(self) -> bool:
        return (
            abs(self.vderiv - 1.0) < self._CLASSICAL_TOL
            or self.history_size == 1
        )

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------

    def _to_dense_gradient(self, grad, var):
        """Convert sparse gradients to dense tensors."""
        if isinstance(grad, tf.IndexedSlices):
            return tf.scatter_nd(
                indices=tf.expand_dims(grad.indices, axis=1),
                updates=grad.values,
                shape=tf.shape(var),
            )
        return grad

    def _global_l2_norm(self, x, dtype):
        x = tf.cast(x, dtype)
        return tf.sqrt(tf.reduce_sum(tf.square(x)))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _add_adaptive_memory_slots(self, var) -> None:
        if not self._is_classical:
            self.add_slot(
                var,
                "grad_history",
                initializer="zeros",
                shape=[self.history_size] + var.shape.as_list(),
            )
        self.add_slot(var, "prev_raw_grad")
        self.add_slot(
            var,
            "frac_mix",
            initializer=tf.keras.initializers.Constant(self.initial_mix),
            shape=[],
        )
        self.add_slot(var, "stability_state", initializer="zeros", shape=[])

    # ------------------------------------------------------------------
    # Loss interface (unchanged)
    # ------------------------------------------------------------------

    def set_current_loss(self, loss_value):
        """
        Optional loss injection from a custom training loop.

        The optimizer does not require this, but if the loop calls it, the
        fractional mix is reduced when the smoothed loss becomes worse than
        the best smoothed loss observed so far.
        """
        try:
            loss_scalar = float(
                tf.keras.backend.get_value(tf.cast(loss_value, tf.float32))
            )
        except Exception:
            loss_scalar = float(loss_value)

        self._current_loss_value = loss_scalar

        if self._loss_ema_value is None:
            self._loss_ema_value = loss_scalar
            self._best_loss_ema_value = loss_scalar
            self._loss_worse_value = 0.0
            return

        self._loss_ema_value = (
            self.loss_ema_beta * self._loss_ema_value
            + (1.0 - self.loss_ema_beta) * loss_scalar
        )

        threshold = self.loss_worsen_tolerance * max(
            1.0, abs(self._best_loss_ema_value)
        )

        if self._loss_ema_value < self._best_loss_ema_value - threshold:
            self._best_loss_ema_value = self._loss_ema_value
            self._loss_worse_value = 0.0
        elif self._loss_ema_value > self._best_loss_ema_value + threshold:
            self._loss_worse_value = 1.0
        else:
            self._loss_worse_value = 0.0

    # ------------------------------------------------------------------
    # Fractional gradient (circular buffer) and adaptive mix
    # ------------------------------------------------------------------

    def _fractional_gradient(self, raw_grad, var):
        """
        Write g_t into the circular buffer and return the fractional
        gradient after the descent safeguard and norm matching.

        Returns (frac_grad, history_update_op).
        """
        dtype = var.dtype.base_dtype

        if self._is_classical:
            return raw_grad, tf.no_op()

        k = self.history_size
        hist = self.get_slot(var, "grad_history")

        pos = tf.cast(
            tf.math.floormod(tf.cast(self.iterations, tf.int64), k), tf.int32
        )
        hist_op = hist.scatter_nd_update(
            indices=tf.reshape(pos, [1, 1]),
            updates=tf.expand_dims(raw_grad, axis=0),
        )

        with tf.control_dependencies([hist_op]):
            coeffs = tf.constant(self._fractional_coefficients, dtype=dtype)
            lags = tf.math.floormod(pos - tf.range(k, dtype=tf.int32), k)
            c_perm = tf.gather(coeffs, lags)
            c_shape = [k] + [1] * len(var.shape)
            frac_grad = tf.reduce_sum(
                hist * tf.reshape(c_perm, c_shape), axis=0
            )

        if self.descent_check:
            dot = tf.reduce_sum(frac_grad * raw_grad)
            keep = tf.cast(dot > 0.0, dtype)
            frac_grad = keep * frac_grad + (1.0 - keep) * raw_grad

        if self.norm_match_fractional_gradient:
            raw_norm = self._global_l2_norm(raw_grad, dtype)
            frac_norm = self._global_l2_norm(frac_grad, dtype)
            scale = raw_norm / (frac_norm + tf.cast(self.coeff_epsilon, dtype))
            frac_grad = frac_grad * scale

        return frac_grad, hist_op

    def _stability_score(self, raw_grad, prev_raw_grad, dtype):
        diff_norm = self._global_l2_norm(raw_grad - prev_raw_grad, dtype)
        prev_norm = self._global_l2_norm(prev_raw_grad, dtype)
        score = diff_norm / (prev_norm + tf.cast(self.coeff_epsilon, dtype))
        return tf.minimum(score, tf.cast(self.stability_clip, dtype))

    def _warmup_max_mix(self, dtype):
        if self.warmup_steps <= 0:
            return tf.cast(self.max_mix, dtype)
        step = tf.cast(self.iterations + 1, dtype)
        warmup = tf.cast(self.warmup_steps, dtype)
        fraction = tf.clip_by_value(step / warmup, 0.0, 1.0)
        return tf.cast(self.max_mix, dtype) * fraction

    def _adaptive_effective_gradient(self, grad, var):
        """
        Main shared adaptive step (identical control flow to the original;
        the fractional gradient now comes from the corrected circular
        buffer with descent safeguard).

        Returns:
            raw_grad
            eff_grad
            second_moment_grad
            update_ops
        """
        dtype = var.dtype.base_dtype

        raw_grad = self._to_dense_gradient(grad, var)
        prev_raw_grad = self.get_slot(var, "prev_raw_grad")
        mix_state = self.get_slot(var, "frac_mix")
        stability_state = self.get_slot(var, "stability_state")

        mix_scalar = tf.cast(mix_state, dtype)
        stability_scalar = tf.cast(stability_state, dtype)

        frac_grad, hist_op = self._fractional_gradient(raw_grad, var)

        score = self._stability_score(raw_grad, prev_raw_grad, dtype)

        gamma_stab = tf.cast(self.stability_ema_gamma, dtype)
        new_stability = gamma_stab * stability_scalar + (1.0 - gamma_stab) * score

        stable = tf.cast(
            new_stability <= tf.cast(self.stability_threshold, dtype), dtype
        )

        step = tf.cast(self.adapt_step, dtype)
        target_mix = tf.where(
            stable > 0.5,
            mix_scalar + step,
            mix_scalar - step,
        )

        loss_worse = tf.cast(self._loss_worse_value, dtype)
        target_mix = target_mix - loss_worse * tf.cast(self.loss_mix_decrease, dtype)

        max_mix_t = self._warmup_max_mix(dtype)
        min_mix_t = tf.cast(self.min_mix, dtype)
        target_mix = tf.clip_by_value(target_mix, min_mix_t, max_mix_t)

        gamma_mix = tf.cast(self.mix_ema_gamma, dtype)
        new_mix = gamma_mix * mix_scalar + (1.0 - gamma_mix) * target_mix
        new_mix = tf.clip_by_value(new_mix, min_mix_t, max_mix_t)

        eff_grad = (1.0 - new_mix) * raw_grad + new_mix * frac_grad

        if self.use_raw_gradient_for_second_moment:
            second_moment_grad = raw_grad
        else:
            second_moment_grad = eff_grad

        prev_op = prev_raw_grad.assign(raw_grad, use_locking=self._use_locking)
        mix_op = mix_state.assign(
            tf.cast(new_mix, mix_state.dtype), use_locking=self._use_locking
        )
        stability_op = stability_state.assign(
            tf.cast(new_stability, stability_state.dtype),
            use_locking=self._use_locking,
        )

        update_ops = tf.group(hist_op, prev_op, mix_op, stability_op)

        return raw_grad, eff_grad, second_moment_grad, update_ops

    # ------------------------------------------------------------------
    # Inspection (unchanged behaviour)
    # ------------------------------------------------------------------

    def current_mix_values(self) -> List[tf.Tensor]:
        values = []
        for w in self.weights:
            if "frac_mix" in w.name:
                values.append(tf.reduce_mean(tf.identity(w)))
        return values

    def current_stability_values(self) -> List[tf.Tensor]:
        values = []
        for w in self.weights:
            if "stability_state" in w.name:
                values.append(tf.reduce_mean(tf.identity(w)))
        return values

    def _adaptive_memory_fractional_config(self) -> Dict[str, Any]:
        return {
            "vderiv": self.vderiv,
            "history_size": self.history_size,
            "normalize_coefficients": self.normalize_coefficients,
            "coeff_epsilon": self.coeff_epsilon,
            "initial_mix": self.initial_mix,
            "min_mix": self.min_mix,
            "max_mix": self.max_mix,
            "adapt_step": self.adapt_step,
            "mix_ema_gamma": self.mix_ema_gamma,
            "stability_ema_gamma": self.stability_ema_gamma,
            "stability_threshold": self.stability_threshold,
            "stability_clip": self.stability_clip,
            "norm_match_fractional_gradient":
                self.norm_match_fractional_gradient,
            "use_raw_gradient_for_second_moment":
                self.use_raw_gradient_for_second_moment,
            "loss_ema_beta": self.loss_ema_beta,
            "loss_worsen_tolerance": self.loss_worsen_tolerance,
            "loss_mix_decrease": self.loss_mix_decrease,
            "warmup_steps": self.warmup_steps,
            "descent_check": self.descent_check,
        }


# Shared keyword plumbing: every optimizer below forwards these to
# _init_adaptive_memory_fractional_args.
_ADAPTIVE_KWARG_NAMES = (
    "vderiv", "history_size", "normalize_coefficients", "coeff_epsilon",
    "initial_mix", "min_mix", "max_mix", "adapt_step", "mix_ema_gamma",
    "stability_ema_gamma", "stability_threshold", "stability_clip",
    "norm_match_fractional_gradient", "use_raw_gradient_for_second_moment",
    "loss_ema_beta", "loss_worsen_tolerance", "loss_mix_decrease",
    "warmup_steps", "descent_check",
)


def _split_adaptive_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Pop the adaptive-memory kwargs out of a kwargs dict."""
    return {k: kwargs.pop(k) for k in list(kwargs) if k in _ADAPTIVE_KWARG_NAMES}


# =============================================================================
# AdaptiveMemoryFSGD
# =============================================================================


class AdaptiveMemoryFSGD(_AdaptiveMemoryFractionalBase, _KerasOptimizerBase):
    """
    Adaptive memory-fractional SGD.

    Uses adaptive fractional mixing in the SGD update direction.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        lr: float = None,
        momentum: float = 0.0,
        nesterov: bool = False,
        name: str = "AdaptiveMemoryFSGD",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
        super(AdaptiveMemoryFSGD, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= momentum < 1.0):
            raise ValueError("`momentum` must satisfy 0 <= momentum < 1.")

        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def _create_slots(self, var_list):
        for var in var_list:
            self._add_adaptive_memory_slots(var)
            if self.momentum > 0.0:
                self.add_slot(var, "momentum")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)

        _, eff_grad, _, adaptive_ops = self._adaptive_effective_gradient(grad, var)

        if self.momentum > 0.0:
            mu = tf.cast(self.momentum, dtype)
            m = self.get_slot(var, "momentum")
            new_m = mu * m - lr_t * eff_grad
            m_op = m.assign(new_m, use_locking=self._use_locking)
            if self.nesterov:
                var_op = var.assign_add(
                    mu * new_m - lr_t * eff_grad,
                    use_locking=self._use_locking,
                )
            else:
                var_op = var.assign_add(new_m, use_locking=self._use_locking)
            return tf.group(var_op, m_op, adaptive_ops)

        var_op = var.assign_sub(lr_t * eff_grad, use_locking=self._use_locking)
        return tf.group(var_op, adaptive_ops)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse = tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)
        )
        return self._resource_apply_dense(sparse, var, apply_state=apply_state)

    def get_config(self):
        config = super(AdaptiveMemoryFSGD, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# AdaptiveMemoryFRMSprop
# =============================================================================


class AdaptiveMemoryFRMSprop(_AdaptiveMemoryFractionalBase, _KerasOptimizerBase):
    """
    Adaptive memory-fractional RMSprop.

    Adaptive fractional mixing in the numerator; the squared-gradient
    accumulator uses the raw gradient by default.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-7,
        centered: bool = False,
        name: str = "AdaptiveMemoryFRMSprop",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
        super(AdaptiveMemoryFRMSprop, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= rho < 1.0):
            raise ValueError("`rho` must satisfy 0 <= rho < 1.")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("`momentum` must satisfy 0 <= momentum < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.rho = float(rho)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.centered = bool(centered)

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def _create_slots(self, var_list):
        for var in var_list:
            self._add_adaptive_memory_slots(var)
            self.add_slot(var, "velocity")
            if self.momentum > 0.0:
                self.add_slot(var, "momentum")
            if self.centered:
                self.add_slot(var, "avg_grad")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)
        rho_t = tf.cast(self.rho, dtype)
        eps_t = tf.cast(self.epsilon, dtype)

        _, eff_grad, second_grad, adaptive_ops = (
            self._adaptive_effective_gradient(grad, var)
        )

        velocity = self.get_slot(var, "velocity")
        new_velocity = rho_t * velocity + (1.0 - rho_t) * tf.square(second_grad)
        velocity_op = velocity.assign(new_velocity, use_locking=self._use_locking)

        if self.centered:
            avg_grad = self.get_slot(var, "avg_grad")
            new_avg = rho_t * avg_grad + (1.0 - rho_t) * second_grad
            avg_op = avg_grad.assign(new_avg, use_locking=self._use_locking)
            denom = tf.sqrt(new_velocity - tf.square(new_avg) + eps_t)
        else:
            avg_op = tf.no_op()
            denom = tf.sqrt(new_velocity + eps_t)

        if self.momentum > 0.0:
            mu = tf.cast(self.momentum, dtype)
            mom = self.get_slot(var, "momentum")
            new_mom = mu * mom + lr_t * eff_grad / denom
            mom_op = mom.assign(new_mom, use_locking=self._use_locking)
            var_op = var.assign_sub(new_mom, use_locking=self._use_locking)
            return tf.group(var_op, adaptive_ops, velocity_op, avg_op, mom_op)

        var_op = var.assign_sub(
            lr_t * eff_grad / denom, use_locking=self._use_locking
        )
        return tf.group(var_op, adaptive_ops, velocity_op, avg_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse = tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)
        )
        return self._resource_apply_dense(sparse, var, apply_state=apply_state)

    def get_config(self):
        config = super(AdaptiveMemoryFRMSprop, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "rho": self.rho,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "centered": self.centered,
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# AdaptiveMemoryFAdam
# =============================================================================


class AdaptiveMemoryFAdam(_AdaptiveMemoryFractionalBase, _KerasOptimizerBase):
    """
    Adaptive memory-fractional Adam.

    Adaptive fractional mixing in the first moment; the second moment uses
    the raw gradient by default. Keras-exact Adam formulation, so at
    nu = 1 the optimizer is bit-compatible with the Keras Adam baseline.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        name: str = "AdaptiveMemoryFAdam",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
        super(AdaptiveMemoryFAdam, self).__init__(name, **kwargs)

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

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def _create_slots(self, var_list):
        for var in var_list:
            self._add_adaptive_memory_slots(var)
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            if self.amsgrad:
                self.add_slot(var, "vhat")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)
        b1 = tf.cast(self.beta_1, dtype)
        b2 = tf.cast(self.beta_2, dtype)
        eps_t = tf.cast(self.epsilon, dtype)
        step = tf.cast(self.iterations + 1, dtype)

        _, eff_grad, second_grad, adaptive_ops = (
            self._adaptive_effective_gradient(grad, var)
        )

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        new_m = b1 * m + (1.0 - b1) * eff_grad
        new_v = b2 * v + (1.0 - b2) * tf.square(second_grad)

        m_op = m.assign(new_m, use_locking=self._use_locking)
        v_op = v.assign(new_v, use_locking=self._use_locking)

        # Keras-exact formulation (bias correction folded into lr).
        b1_pow = tf.pow(b1, step)
        b2_pow = tf.pow(b2, step)
        lr_corr = lr_t * tf.sqrt(1.0 - b2_pow) / (1.0 - b1_pow)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            new_vhat = tf.maximum(vhat, new_v)
            vhat_op = vhat.assign(new_vhat, use_locking=self._use_locking)
            denom_v = new_vhat
        else:
            vhat_op = tf.no_op()
            denom_v = new_v

        var_op = var.assign_sub(
            lr_corr * new_m / (tf.sqrt(denom_v) + eps_t),
            use_locking=self._use_locking,
        )
        return tf.group(var_op, adaptive_ops, m_op, v_op, vhat_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse = tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)
        )
        return self._resource_apply_dense(sparse, var, apply_state=apply_state)

    def get_config(self):
        config = super(AdaptiveMemoryFAdam, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# AdaptiveMemoryFAdadelta
# =============================================================================


class AdaptiveMemoryFAdadelta(_AdaptiveMemoryFractionalBase, _KerasOptimizerBase):
    """
    Adaptive memory-fractional Adadelta.

    Adaptive fractional mixing in the update direction; the accumulators
    use the raw gradient / raw-driven update by default.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        lr: float = None,
        rho: float = 0.95,
        epsilon: float = 1e-7,
        name: str = "AdaptiveMemoryFAdadelta",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
        super(AdaptiveMemoryFAdadelta, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= rho < 1.0):
            raise ValueError("`rho` must satisfy 0 <= rho < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.rho = float(rho)
        self.epsilon = float(epsilon)

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def _create_slots(self, var_list):
        for var in var_list:
            self._add_adaptive_memory_slots(var)
            self.add_slot(var, "accumulated_grad")
            self.add_slot(var, "accumulated_delta")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)
        rho_t = tf.cast(self.rho, dtype)
        eps_t = tf.cast(self.epsilon, dtype)

        _, eff_grad, second_grad, adaptive_ops = (
            self._adaptive_effective_gradient(grad, var)
        )

        acc_g = self.get_slot(var, "accumulated_grad")
        acc_d = self.get_slot(var, "accumulated_delta")

        new_acc_g = rho_t * acc_g + (1.0 - rho_t) * tf.square(second_grad)
        acc_g_op = acc_g.assign(new_acc_g, use_locking=self._use_locking)

        rms_delta = tf.sqrt(acc_d + eps_t)
        rms_grad = tf.sqrt(new_acc_g + eps_t)
        delta = -(rms_delta / rms_grad) * eff_grad

        new_acc_d = rho_t * acc_d + (1.0 - rho_t) * tf.square(delta)
        acc_d_op = acc_d.assign(new_acc_d, use_locking=self._use_locking)

        var_op = var.assign_add(lr_t * delta, use_locking=self._use_locking)
        return tf.group(var_op, adaptive_ops, acc_g_op, acc_d_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse = tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)
        )
        return self._resource_apply_dense(sparse, var, apply_state=apply_state)

    def get_config(self):
        config = super(AdaptiveMemoryFAdadelta, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "rho": self.rho,
                "epsilon": self.epsilon,
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# AdaptiveMemoryFAdagrad  (new)
# =============================================================================


class AdaptiveMemoryFAdagrad(_AdaptiveMemoryFractionalBase, _KerasOptimizerBase):
    """
    Adaptive memory-fractional Adagrad.

    Update:
        acc_t = acc_{t-1} + s_t^2      (s_t = raw gradient by default)
        theta_{t+1} = theta_t - lr * g_eff_t / (sqrt(acc_t) + eps)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-7,
        name: str = "AdaptiveMemoryFAdagrad",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
        super(AdaptiveMemoryFAdagrad, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if initial_accumulator_value < 0.0:
            raise ValueError("`initial_accumulator_value` must be >= 0.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.initial_accumulator_value = float(initial_accumulator_value)
        self.epsilon = float(epsilon)

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def _create_slots(self, var_list):
        init = tf.keras.initializers.Constant(self.initial_accumulator_value)
        for var in var_list:
            self._add_adaptive_memory_slots(var)
            self.add_slot(var, "accumulator", initializer=init)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)
        eps_t = tf.cast(self.epsilon, dtype)

        _, eff_grad, second_grad, adaptive_ops = (
            self._adaptive_effective_gradient(grad, var)
        )

        acc = self.get_slot(var, "accumulator")
        new_acc = acc + tf.square(second_grad)
        acc_op = acc.assign(new_acc, use_locking=self._use_locking)

        var_op = var.assign_sub(
            lr_t * eff_grad / (tf.sqrt(new_acc) + eps_t),
            use_locking=self._use_locking,
        )
        return tf.group(var_op, adaptive_ops, acc_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse = tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)
        )
        return self._resource_apply_dense(sparse, var, apply_state=apply_state)

    def get_config(self):
        config = super(AdaptiveMemoryFAdagrad, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "initial_accumulator_value": self.initial_accumulator_value,
                "epsilon": self.epsilon,
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# AdaptiveMemoryFAdamW  (new)
# =============================================================================


class AdaptiveMemoryFAdamW(AdaptiveMemoryFAdam):
    """
    Adaptive memory-fractional AdamW: AdaptiveMemoryFAdam plus decoupled
    weight decay,
        theta_{t+1} = theta_t - lr * (Adam step) - lr * weight_decay * theta_t.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        weight_decay: float = 0.004,
        name: str = "AdaptiveMemoryFAdamW",
        **kwargs: Any,
    ):
        if weight_decay < 0.0:
            raise ValueError("`weight_decay` must be >= 0.")
        super(AdaptiveMemoryFAdamW, self).__init__(
            learning_rate=learning_rate, lr=lr, name=name, **kwargs
        )
        self.weight_decay = float(weight_decay)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)
        adam_op = super(AdaptiveMemoryFAdamW, self)._resource_apply_dense(
            grad, var, apply_state=apply_state
        )
        if self.weight_decay > 0.0:
            wd_t = tf.cast(self.weight_decay, dtype)
            with tf.control_dependencies([adam_op]):
                wd_op = var.assign_sub(
                    lr_t * wd_t * var, use_locking=self._use_locking
                )
            return tf.group(adam_op, wd_op)
        return adam_op

    def get_config(self):
        config = super(AdaptiveMemoryFAdamW, self).get_config()
        config.update({"weight_decay": self.weight_decay})
        return config


# =============================================================================
# AdaptiveMemoryFNadam  (new)
# =============================================================================


class AdaptiveMemoryFNadam(_AdaptiveMemoryFractionalBase, _KerasOptimizerBase):
    """
    Adaptive memory-fractional Nadam (Nesterov-accelerated Adam).

    Update (Dozat's formulation without the momentum-product schedule):
        m_t   = beta_1 m_{t-1} + (1 - beta_1) g_eff_t
        v_t   = beta_2 v_{t-1} + (1 - beta_2) s_t^2
        m_bar = beta_1 m_t / (1 - beta_1^{t+1})
                + (1 - beta_1) g_eff_t / (1 - beta_1^t)
        theta_{t+1} = theta_t - lr * m_bar / (sqrt(v_t/(1-beta_2^t)) + eps)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        name: str = "AdaptiveMemoryFNadam",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
        super(AdaptiveMemoryFNadam, self).__init__(name, **kwargs)

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

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def _create_slots(self, var_list):
        for var in var_list:
            self._add_adaptive_memory_slots(var)
            self.add_slot(var, "m")
            self.add_slot(var, "v")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)
        b1 = tf.cast(self.beta_1, dtype)
        b2 = tf.cast(self.beta_2, dtype)
        eps_t = tf.cast(self.epsilon, dtype)
        step = tf.cast(self.iterations + 1, dtype)

        _, eff_grad, second_grad, adaptive_ops = (
            self._adaptive_effective_gradient(grad, var)
        )

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        new_m = b1 * m + (1.0 - b1) * eff_grad
        new_v = b2 * v + (1.0 - b2) * tf.square(second_grad)

        m_op = m.assign(new_m, use_locking=self._use_locking)
        v_op = v.assign(new_v, use_locking=self._use_locking)

        b1_pow = tf.pow(b1, step)
        b2_pow = tf.pow(b2, step)

        m_bar = (
            b1 * new_m / (1.0 - b1_pow * b1)
            + (1.0 - b1) * eff_grad / (1.0 - b1_pow)
        )
        v_hat = new_v / (1.0 - b2_pow)

        var_op = var.assign_sub(
            lr_t * m_bar / (tf.sqrt(v_hat) + eps_t),
            use_locking=self._use_locking,
        )
        return tf.group(var_op, adaptive_ops, m_op, v_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse = tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)
        )
        return self._resource_apply_dense(sparse, var, apply_state=apply_state)

    def get_config(self):
        config = super(AdaptiveMemoryFNadam, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# AdaptiveMemoryFAdamax  (new)
# =============================================================================


class AdaptiveMemoryFAdamax(_AdaptiveMemoryFractionalBase, _KerasOptimizerBase):
    """
    Adaptive memory-fractional Adamax (infinity-norm Adam).

    Update:
        m_t = beta_1 m_{t-1} + (1 - beta_1) g_eff_t
        u_t = max(beta_2 u_{t-1}, |s_t|)
        theta_{t+1} = theta_t - lr / (1 - beta_1^t) * m_t / (u_t + eps)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        name: str = "AdaptiveMemoryFAdamax",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
        super(AdaptiveMemoryFAdamax, self).__init__(name, **kwargs)

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

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def _create_slots(self, var_list):
        for var in var_list:
            self._add_adaptive_memory_slots(var)
            self.add_slot(var, "m")
            self.add_slot(var, "u")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(dtype), dtype)
        b1 = tf.cast(self.beta_1, dtype)
        b2 = tf.cast(self.beta_2, dtype)
        eps_t = tf.cast(self.epsilon, dtype)
        step = tf.cast(self.iterations + 1, dtype)

        _, eff_grad, second_grad, adaptive_ops = (
            self._adaptive_effective_gradient(grad, var)
        )

        m = self.get_slot(var, "m")
        u = self.get_slot(var, "u")

        new_m = b1 * m + (1.0 - b1) * eff_grad
        new_u = tf.maximum(b2 * u, tf.abs(second_grad))

        m_op = m.assign(new_m, use_locking=self._use_locking)
        u_op = u.assign(new_u, use_locking=self._use_locking)

        b1_pow = tf.pow(b1, step)
        var_op = var.assign_sub(
            lr_t / (1.0 - b1_pow) * new_m / (new_u + eps_t),
            use_locking=self._use_locking,
        )
        return tf.group(var_op, adaptive_ops, m_op, u_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse = tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)
        )
        return self._resource_apply_dense(sparse, var, apply_state=apply_state)

    def get_config(self):
        config = super(AdaptiveMemoryFAdamax, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# Export list
# =============================================================================

__all__ = [
    "AdaptiveMemoryFSGD",
    "AdaptiveMemoryFRMSprop",
    "AdaptiveMemoryFAdam",
    "AdaptiveMemoryFAdadelta",
    "AdaptiveMemoryFAdagrad",
    "AdaptiveMemoryFAdamW",
    "AdaptiveMemoryFNadam",
    "AdaptiveMemoryFAdamax",
]


# =============================================================================
# Smoke tests
# =============================================================================

if __name__ == "__main__":
    print("Smoke tests for corrected class_minimal_fractional_adaptive_optimizers.py")
    print("=" * 74)

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

    common = dict(vderiv=0.9, history_size=4, initial_mix=0.10,
                  max_mix=0.30, adapt_step=0.01)

    configs = [
        ("AdaptiveMemoryFSGD", AdaptiveMemoryFSGD(lr=1e-2, **common)),
        ("AdaptiveMemoryFSGD nu=1.5",
         AdaptiveMemoryFSGD(lr=1e-2, vderiv=1.5, history_size=4,
                            initial_mix=0.10, max_mix=0.30, adapt_step=0.01)),
        ("AdaptiveMemoryFRMSprop", AdaptiveMemoryFRMSprop(lr=1e-3, **common)),
        ("AdaptiveMemoryFAdam", AdaptiveMemoryFAdam(lr=1e-3, **common)),
        ("AdaptiveMemoryFAdadelta", AdaptiveMemoryFAdadelta(lr=1.0, **common)),
        ("AdaptiveMemoryFAdagrad", AdaptiveMemoryFAdagrad(lr=1e-2, **common)),
        ("AdaptiveMemoryFAdamW",
         AdaptiveMemoryFAdamW(lr=1e-3, weight_decay=0.004, **common)),
        ("AdaptiveMemoryFNadam", AdaptiveMemoryFNadam(lr=1e-3, **common)),
        ("AdaptiveMemoryFAdamax", AdaptiveMemoryFAdamax(lr=1e-3, **common)),
        ("AdaptiveMemoryFAdam nu=1 (classical)",
         AdaptiveMemoryFAdam(lr=1e-3, vderiv=1.0, history_size=4)),
    ]

    for name, opt in configs:
        model = build_model()
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        hist = model.fit(x, y, epochs=12, batch_size=32, verbose=0)
        losses = hist.history["loss"]
        decreasing = losses[-1] < losses[0]
        mixes = opt.current_mix_values()
        mean_mix = float(tf.reduce_mean(tf.stack(mixes))) if mixes else float("nan")
        print(
            f"  {name:38s} loss {losses[0]:.4f} -> {losses[-1]:.4f}"
            f"  (decreasing={decreasing}, mean lambda={mean_mix:.3f})"
        )
        assert decreasing, f"{name}: loss did not decrease"

    print("=" * 74)
    print("All smoke tests passed.")
