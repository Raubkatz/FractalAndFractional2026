import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tnp # start using tnp instead of numpy or math library
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.experimental import numpy as tnp


# Custom activation function: Revised Modulated Blancmange Curve
def modulated_blancmange_curve(x, n_terms=30, a=0.75):
    y = tf.zeros_like(x)
    for n in range(n_terms):
        factor = 2 ** n
        modulation = tf.tanh(a * factor * x)  # Modulating factor using tanh
        ax = a * tf.sqrt(tf.abs(x))
        y += modulation * tf.abs(x * factor % 2 - 1 * ax) / factor
    return y / 2


# Blended function with tanh and cosine components with exponential decay using TensorFlow
def decaying_cosine_function_tf(x, a=0.5, b=3, c=0.5, d=2, n_terms=75,  zeta=0.2666):
    # Mirrored function using TensorFlow
    def mirrored_function_tf(x):
        return tf.where(x < 0, tf.ones_like(x), -tf.ones_like(x))

    #if not (0 < a < 1):
    #    raise ValueError("Parameter 'a' must be in the interval (0, 1).")

    #if b <= 0:
    #    raise ValueError("Parameter 'b' must be positive.")

    #if c <= 0:
    #    raise ValueError("Parameter 'c' must be positive.")

    #if d <= 0:
    #    raise ValueError("Parameter 'd' must be positive.")

    #if n_terms <= 0:
    #    raise ValueError("Parameter 'n_terms' must be a positive integer.")

    # Initialize the output tensor
    w = tf.zeros_like(x)

    for n in range(n_terms):
        # Exponentially decaying cosine part
        decay = tf.exp(-tf.abs(x) * 0.5)
        # Sum the components: tanh and decaying cosine with mirrored behavior
        w += zeta * (0.05 * tf.tanh(tnp.pi * x) + (c ** n) * tf.cos((d ** n) * tnp.pi * x) * decay * mirrored_function_tf(x))

    return w


# Custom activation function: Modulated Weierstrass function using tnp
def modified_weierstrass_function_tanh(x, a=0.5, b=1.5, n_terms=100):
    #if not (0 < a < 1):
    #    raise ValueError("Parameter 'a' must be in the interval (0, 1).")

    #if b % 2 == 0 or b <= 0:
    #    raise ValueError("Parameter 'b' must be a positive odd integer.")

    #if n_terms <= 0:
    #    raise ValueError("Parameter 'n_terms' must be a positive integer.")

    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Calculate the modified Weierstrass function in float64
    w = tf.zeros_like(x, dtype=tf.float64)
    for n in range(n_terms):
        w += ((-1) ** n) * (a ** n) * tf.cos((b ** n) * tnp.pi * x)

    # Blend with the hyperbolic tangent function, using tnp.exp and tnp.abs
    combined_function = w * tnp.exp(-0.75 * tnp.abs(x)) + tnp.tanh(x)

    # Finally, convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(combined_function, tf.float32)


# Custom activation function: Modulated Weierstrass function with ReLU using tnp
def modified_weierstrass_function_relu(x, a=0.5, b=3, n_terms=100):
    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Calculate the modified Weierstrass function in float64
    w = tf.zeros_like(x, dtype=tf.float64)
    for n in range(n_terms):
        w += ((-1) ** n) * (a ** n) * tf.cos((b ** n) * tnp.pi * x)

    # Blend with the ReLU function, using tnp.exp and tnp.abs
    combined_function = w * tnp.exp(-0.75 * tnp.abs(x)) + tf.nn.relu(x)  # Replaced tanh with ReLU

    # Finally, convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(combined_function, tf.float32)


def weierstrass_mandelbrot_function_xsinsquared(x, gamma=0.5, lambda_val=2, num_terms=100):
    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Initialize the function value to zeros
    M_x = tf.zeros_like(x, dtype=tf.float64)

    # Compute the Weierstrass-Mandelbrot function
    for k in range(1, num_terms):
        term = (2**(-k * gamma)) * x * (tnp.sin(2 * tnp.pi * lambda_val**k * x) ** 2)
        M_x += term

    # Convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(M_x, tf.float32)


def weierstrass_mandelbrot_function_xpsin(x, gamma=0.5, lambda_val=2, num_terms=100):
    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Initialize the function value to zeros
    M_x = tf.zeros_like(x, dtype=tf.float64)

    # Compute the Weierstrass-Mandelbrot function
    for k in range(1, num_terms):
        term = (2**(-k * gamma)) * (x + (tnp.sin(2 * tnp.pi * lambda_val**k * x)))
        M_x += term

    # Convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(M_x, tf.float32)

def weierstrass_mandelbrot_function_relupsin(x, gamma=0.5, lambda_val=2, num_terms=100):
    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Initialize the function value to zeros
    M_x = tf.zeros_like(x, dtype=tf.float64)

    # Compute the Weierstrass-Mandelbrot function with ReLU(x) + sin(...)
    for k in range(1, num_terms):
        relu_term = tf.nn.relu(x)
        sin_term = tnp.sin(2 * tnp.pi * lambda_val**k * x)
        term = (2**(-k * gamma)) * (relu_term + sin_term)
        M_x += term

    # Convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(M_x, tf.float32)

def weierstrass_mandelbrot_function_tanhpsin(x, gamma=0.5, lambda_val=2, num_terms=100):
    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Initialize the function value to zeros
    M_x = tf.zeros_like(x, dtype=tf.float64)

    # Compute the Weierstrass-Mandelbrot function with ReLU(x) + sin(...)
    for k in range(1, num_terms):
        tanh_term = tnp.tanh(x)
        sin_term = tnp.sin(2 * tnp.pi * lambda_val**k * x)
        term = (2**(-k * gamma)) * (tanh_term + sin_term)
        M_x += term

    # Convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(M_x, tf.float32)


# Custom activation function: Classical Weierstrass function using TensorFlow
def weierstrass_function_tf(x, gamma=0.5, lambda_val=2, num_terms=100):
    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Initialize the function value to zeros
    W_x = tf.zeros_like(x, dtype=tf.float64)

    # Compute the Weierstrass function
    for k in range(1, num_terms):
        term = (2**(-k * gamma)) * tnp.sin(2 * tnp.pi * lambda_val**k * x)
        W_x += term

    # Convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(W_x, tf.float32)


# ============================================================
# Random-depth fractal activation functions
# TensorFlow-compatible function-style implementations
# ============================================================

def _sample_random_depth(min_terms=10, max_terms=50):
    """
    Sample one random integer depth for one activation call.

    Returns
    -------
    tf.Tensor
        Scalar int32 tensor in [min_terms, max_terms].
    """
    return tf.random.uniform(
        shape=[],
        minval=int(min_terms),
        maxval=int(max_terms) + 1,
        dtype=tf.int32,
    )


def random_depth_modified_weierstrass_function_tanh(
        x,
        a=0.5,
        b=1.5,
        min_terms=10,
        max_terms=50
    ):
    """
    Random-depth version of modified_weierstrass_function_tanh.

    Same usage pattern as:

        modified_weierstrass_function_tanh(x, n_terms=100)

    but n_terms is sampled randomly on each function call.
    """
    depth = _sample_random_depth(min_terms, max_terms)

    x = tf.cast(x, tf.float64)
    w = tf.zeros_like(x, dtype=tf.float64)

    for n in range(int(max_terms)):
        active = tf.cast(n < depth, tf.float64)
        w += active * ((-1) ** n) * (a ** n) * tf.cos((b ** n) * tnp.pi * x)

    combined_function = w * tnp.exp(-0.75 * tnp.abs(x)) + tnp.tanh(x)

    return tf.cast(combined_function, tf.float32)


def random_depth_decaying_cosine_function_tf(
        x,
        a=0.5,
        b=3,
        c=0.5,
        d=2,
        min_terms=10,
        max_terms=50,
        zeta=0.2666
    ):
    """
    Random-depth version of decaying_cosine_function_tf.

    Same usage pattern as:

        decaying_cosine_function_tf(x, n_terms=75)

    but n_terms is sampled randomly on each function call.
    """
    depth = _sample_random_depth(min_terms, max_terms)

    def mirrored_function_tf(x):
        return tf.where(x < 0, tf.ones_like(x), -tf.ones_like(x))

    x = tf.cast(x, tf.float32)
    w = tf.zeros_like(x, dtype=tf.float32)

    for n in range(int(max_terms)):
        active = tf.cast(n < depth, tf.float32)

        decay = tf.exp(-tf.abs(x) * 0.5)

        w += active * zeta * (
            0.05 * tf.tanh(tnp.pi * x)
            + (c ** n)
            * tf.cos((d ** n) * tnp.pi * x)
            * decay
            * mirrored_function_tf(x)
        )

    return w


def random_depth_modulated_blancmange_curve(
        x,
        min_terms=10,
        max_terms=50,
        a=0.75
    ):
    """
    Random-depth version of modulated_blancmange_curve.

    Same usage pattern as:

        modulated_blancmange_curve(x, n_terms=30)

    but n_terms is sampled randomly on each function call.
    """
    depth = _sample_random_depth(min_terms, max_terms)

    x = tf.cast(x, tf.float32)
    y = tf.zeros_like(x, dtype=tf.float32)

    for n in range(int(max_terms)):
        active = tf.cast(n < depth, tf.float32)

        factor = 2 ** n
        modulation = tf.tanh(a * factor * x)
        ax = a * tf.sqrt(tf.abs(x) + 1e-8)

        y += active * modulation * tf.abs(x * factor % 2 - 1 * ax) / factor

    return y / 2


def random_depth_weierstrass_mandelbrot_function_xpsin(
        x,
        gamma=0.5,
        lambda_val=2,
        min_terms=10,
        max_terms=50
    ):
    """
    Random-depth version of weierstrass_mandelbrot_function_xpsin.

    This variant performed competitively in the paper and is less extreme
    than the ReLU-based Weierstrass-Mandelbrot variant.
    """
    depth = _sample_random_depth(min_terms, max_terms)

    x = tf.cast(x, tf.float64)
    M_x = tf.zeros_like(x, dtype=tf.float64)

    for k in range(1, int(max_terms)):
        active = tf.cast(k < depth, tf.float64)

        term = (2 ** (-k * gamma)) * (
            x + tnp.sin(2 * tnp.pi * lambda_val ** k * x)
        )

        M_x += active * term

    return tf.cast(M_x, tf.float32)


def random_depth_weierstrass_mandelbrot_function_tanhpsin(
        x,
        gamma=0.5,
        lambda_val=2,
        min_terms=10,
        max_terms=50
    ):
    """
    Random-depth version of weierstrass_mandelbrot_function_tanhpsin.

    Same structure as the fixed-depth version, but the number of active
    series terms is sampled on each function call.
    """
    depth = _sample_random_depth(min_terms, max_terms)

    x = tf.cast(x, tf.float64)
    M_x = tf.zeros_like(x, dtype=tf.float64)

    for k in range(1, int(max_terms)):
        active = tf.cast(k < depth, tf.float64)

        tanh_term = tnp.tanh(x)
        sin_term = tnp.sin(2 * tnp.pi * lambda_val ** k * x)

        term = (2 ** (-k * gamma)) * (tanh_term + sin_term)

        M_x += active * term

    return tf.cast(M_x, tf.float32)


def random_depth_weierstrass_function_tf(
        x,
        gamma=0.5,
        lambda_val=2,
        min_terms=10,
        max_terms=50
    ):
    """
    Random-depth version of the plain Weierstrass activation.

    This is mainly useful as a control activation.
    The paper suggests that the plain Weierstrass function alone is
    less useful than its modulated variants.
    """
    depth = _sample_random_depth(min_terms, max_terms)

    x = tf.cast(x, tf.float64)
    W_x = tf.zeros_like(x, dtype=tf.float64)

    for k in range(1, int(max_terms)):
        active = tf.cast(k < depth, tf.float64)

        term = (2 ** (-k * gamma)) * tnp.sin(
            2 * tnp.pi * lambda_val ** k * x
        )

        W_x += active * term

    return tf.cast(W_x, tf.float32)


# ============================================================
# Cantor-based activation functions
# Three distinct implementations of the Cantor construction
# ============================================================

def cantor_function_tf(x, depth=10):
    """
    Cantor function activation — the Devil's Staircase (Implementation 1: Standard).

    Implements the recursive construction from Chiperaa et al. (2025), Eq. 10,
    with depth d=10 and sigmoid preprocessing.

    Maps R -> [0, 1]:
        f_C(x) = c_depth(sigmoid(x))

    Recursive definition:
        c_0(y) = y
        c_d(y) = 0.5 * c_{d-1}(3y)            if y in [0, 1/3]
               = 0.5                             if y in (1/3, 2/3)
               = 0.5 + 0.5 * c_{d-1}(3y - 2)   if y in [2/3, 1]

    Key properties (per Chiperaa et al., 2025):
    - Continuous everywhere but differentiable nowhere (zero derivative a.e.)
    - Strictly monotonically increasing: maps [0, 1] -> [0, 1]
    - Not globally Lipschitz; Hölder continuous (exponent ≈ log2/log3 ≈ 0.631)
    - Maintains Echo State Property up to spectral radius ρ ≈ 10 in ESNs
    - Fastest convergence among all activations tested (median 6.1 vs 15.6 for tanh)

    Implementation: iterative loop of fixed depth, avoids Python recursion limits.
    Gradients flow primarily through the sigmoid preprocessing term; the iterative
    masking operations are differentiable with respect to x.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of any shape and float dtype.
    depth : int
        Recursion depth. Higher depth approaches the true Cantor function more
        closely but adds computation. Default 10 is sufficient for float32.

    Returns
    -------
    tf.Tensor
        Same shape and dtype as x, values in [0, 1].
    """
    y = tf.math.sigmoid(x)          # map R -> (0, 1)

    # Iterative Cantor computation.  At each level we maintain:
    #   accumulated : value built up from higher-level thirds already resolved
    #   scale       : contribution weight at this level (halves each step)
    #   current_y   : rescaled argument for this recursion level
    #   active      : 1 for points still recursing; 0 once frozen at a plateau
    accumulated = tf.zeros_like(y)
    scale       = tf.ones_like(y)
    current_y   = y
    active      = tf.ones_like(y)

    third      = tf.cast(1.0 / 3.0, y.dtype)
    two_thirds = tf.cast(2.0 / 3.0, y.dtype)

    for _ in range(depth):
        in_left   = active * tf.cast(current_y <= third,                                y.dtype)
        in_middle = active * tf.cast((current_y > third) & (current_y < two_thirds),   y.dtype)
        in_right  = active * tf.cast(current_y >= two_thirds,                          y.dtype)

        # Middle third: function is 0.5 at this level — freeze these points.
        accumulated = accumulated + in_middle * 0.5 * scale
        active      = active - in_middle

        # Right third: contribute 0.5 * scale, then recurse into (3y - 2).
        accumulated = accumulated + in_right * 0.5 * scale

        # Rescale for the next level.
        current_y = in_left * (current_y * 3.0) + in_right * (current_y * 3.0 - 2.0)
        scale     = scale * 0.5

    # Base case c_0(y) = y: remaining active points contribute scale * current_y.
    accumulated = accumulated + active * scale * current_y

    return accumulated


def cantor_set_activation_tf(x, depth=10):
    """
    Cantor set indicator activation (Implementation 2: Cantor set / discontinuous).

    Implements f_S(x) = 1_{C_d}(sigmoid(x)) from Chiperaa et al. (2025), Eq. 11.

    Maps R -> {0, 1}:
        Returns 1  if sigmoid(x) is (approximately) in the depth-d Cantor set.
        Returns 0  if sigmoid(x) falls in a removed middle-third interval.

    The depth-d Cantor set is constructed by removing the open middle third from
    [0, 1] iteratively d times:
        C_d = { y ∈ [0, 1] : for all k in {0,...,d-1},
                floor(3^k * y mod 3) ≠ 1 }
    which is equivalent to: no ternary digit of y (in positions 1 to d) equals 1.

    Key properties:
    - Discontinuous everywhere (jumps between 0 and 1 at measure-zero points)
    - Binary output {0, 1}: acts as a stochastic-looking gate on the pre-activation
    - Uncountably infinite points in the Cantor set, yet the set has Lebesgue measure 0
    - All points NOT in the set have output 0; Cantor-set points output 1
    - Distinct from cantor_function_tf: that function is continuous and monotone,
      this one is binary and highly discontinuous

    Implementation: iterative ternary-digit extraction — at each step, check if
    the current digit (in base 3) is 1 (= middle third), and accumulate the
    indicator product. Points that hit a middle third at any level get output 0.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of any shape and float dtype.
    depth : int
        Approximation depth. depth=10 gives 3^10 = 59,049 resolved sub-intervals.

    Returns
    -------
    tf.Tensor
        Same shape and dtype as x, values in {0.0, 1.0}.
    """
    y = tf.math.sigmoid(x)          # map R -> (0, 1)

    # Start assuming every point is in the Cantor set (indicator = 1).
    # Iteratively zero out points whose ternary digit at this level is 1
    # (meaning they fall in a removed middle-third interval at that level).
    in_cantor = tf.ones_like(y)
    current_y = y

    third      = tf.cast(1.0 / 3.0, y.dtype)
    two_thirds = tf.cast(2.0 / 3.0, y.dtype)

    for _ in range(depth):
        # Ternary digit at this level:
        #   current_y in [0,   1/3) → digit 0 (left  third)
        #   current_y in [1/3, 2/3) → digit 1 (middle third — REMOVED)
        #   current_y in [2/3, 1  ) → digit 2 (right third)
        is_middle = tf.cast((current_y >= third) & (current_y < two_thirds), y.dtype)

        # Points in the middle third are excluded from the Cantor set.
        in_cantor = in_cantor * (1.0 - is_middle)

        # Rescale for the next digit:
        # Left third  [0,   1/3) → [0, 1) via 3y
        # Right third [2/3, 1  ) → [0, 1) via 3y - 2
        # Middle: value doesn't matter (already zeroed out above)
        current_y = tf.where(
            current_y < third,
            current_y * 3.0,
            tf.where(
                current_y >= two_thirds,
                current_y * 3.0 - 2.0,
                current_y,              # middle: keep as-is (will be excluded next)
            ),
        )

    return in_cantor


def random_depth_cantor_function_tf(x, min_depth=5, max_depth=15):
    """
    Random-depth Cantor function activation (Implementation 3: stochastic depth).

    Same mathematical construction as cantor_function_tf (Devil's Staircase),
    but the recursion depth is sampled uniformly at random on each call, in the
    same spirit as the other random_depth_* functions in this module.

    Maps R -> [0, 1]:
        f_C(x) = c_{depth}(sigmoid(x)),   depth ~ Uniform{min_depth, ..., max_depth}

    This introduces stochastic variation into the activation's resolution:
    - Low depth (≈ min_depth): coarser staircase, fewer plateaus
    - High depth (≈ max_depth): finer staircase, more plateaus, closer to the
      true Cantor function
    - Randomness across forward passes acts as implicit regularisation, similar
      to how dropout randomises which units are active

    The depth is sampled once per forward pass (a single scalar), not per
    neuron, so all elements of the input tensor share the same depth in one
    call. This is consistent with how _sample_random_depth is used elsewhere.

    Implementation uses a gate mask `step_gate` at each step of the fixed-length
    loop (max_depth iterations): steps beyond the sampled depth are masked to
    zero contribution, leaving the state unchanged until the base case.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of any shape and float dtype.
    min_depth : int
        Minimum recursion depth (inclusive).
    max_depth : int
        Maximum recursion depth (inclusive). Determines loop length.

    Returns
    -------
    tf.Tensor
        Same shape and dtype as x, values in [0, 1].
    """
    depth = _sample_random_depth(min_terms=min_depth, max_terms=max_depth)

    y = tf.math.sigmoid(x)

    accumulated = tf.zeros_like(y)
    scale       = tf.ones_like(y)
    current_y   = y
    active      = tf.ones_like(y)

    third      = tf.cast(1.0 / 3.0, y.dtype)
    two_thirds = tf.cast(2.0 / 3.0, y.dtype)

    for step in range(max_depth):
        # Gate: 1.0 if this step is within the sampled depth, 0.0 otherwise.
        # Once step >= depth, all remaining iterations are no-ops.
        step_gate = tf.cast(step < depth, y.dtype)

        in_left   = active * tf.cast(current_y <= third,                                y.dtype)
        in_middle = active * tf.cast((current_y > third) & (current_y < two_thirds),   y.dtype)
        in_right  = active * tf.cast(current_y >= two_thirds,                          y.dtype)

        # Apply contributions only for active steps (gated by step_gate).
        accumulated = accumulated + step_gate * in_middle * 0.5 * scale
        active      = active      - step_gate * in_middle   # deactivate middle points
        accumulated = accumulated + step_gate * in_right  * 0.5 * scale

        # Rescale current_y for the next level (no-op when step_gate = 0).
        new_y     = in_left * (current_y * 3.0) + in_right * (current_y * 3.0 - 2.0)
        current_y = step_gate * new_y + (1.0 - step_gate) * current_y

        # Scale halves on active steps; stays constant on gated-out steps.
        scale     = scale * (step_gate * 0.5 + (1.0 - step_gate) * 1.0)

    # Base case: remaining active (not-yet-frozen) points contribute scale * current_y.
    accumulated = accumulated + active * scale * current_y

    return accumulated
