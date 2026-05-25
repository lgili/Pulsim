"""Pulsim — discrete-time control building blocks.

These are pure Python stateful classes that mirror v1's
`control.hpp` C++ helpers (PIController, PIDController,
Comparator, etc.). Combine them with the v2
`step_observer(t, x)` callback to close the loop around a v2
plant without touching the C++ kernel.

The canonical pattern:

    import pulsim as p
    from pulsim.control import PIController

    pi = PIController(Kp=0.5, Ki=2000.0, output_min=0.05, output_max=0.95)
    duty = [pi.output]   # mutable container the switch_fn reads

    def observe(t, x):
        v_out = x[vout_idx]
        duty[0] = pi.update(setpoint=v_ref, measured=v_out, dt=dt)

    def switch_fn(t):
        # mask is computed using the latest duty value
        ...

    p.simulate(builder, t_end=..., dt=..., switch_fn=switch_fn,
               step_observer=observe)

Every controller integrates with the trapezoidal rule (matching
v1's `PIController` exactly) so substep-state-correction in the
v2 kernel doesn't break controller numerics.

All controllers are stateful — instantiate ONE per loop and call
`reset()` between back-to-back simulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


__all__ = [
    "PIController",
    "PIDController",
    "Comparator",
    "RateLimiter",
    "SampleHold",
    "FirstOrderLowPass",
    "LookupTable1D",
    "MovingAverageFilter",
    # Math blocks (Phase A.1)
    "Gain",
    "Sum",
    "Subtract",
    "MathBlock",
    # Standalone control blocks (Phase A.1)
    "Integrator",
    "Differentiator",
    "TransferFunction",
    "StateMachine",
    "OpAmp",
    # Signal-shaping blocks (Phase A.1)
    "Limiter",
    "DelayBlock",
    # Modulation blocks (Phase A.1)
    "PwmGenerator",
    "SpaceVectorModulator",
    # Transform blocks (Phase A.1)
    "ClarkeTransform",
    "InverseClarkeTransform",
    "ParkTransform",
    "InverseParkTransform",
    # Synchronization (Phase A.1)
    "PLL",
    # Routing (Phase A.1)
    "SignalMux",
    "SignalDemux",
    # Auto-tuning helpers (Bode-based loop shaping).
    "tune_pi_from_bode",
    "loop_gain",
    "phase_margin_from_loop",
    "gain_margin_from_loop",
]


# =============================================================================
# PI controller — trapezoidal integration + conditional anti-windup
# =============================================================================

@dataclass
class PIController:
    """Discrete-time proportional + integral controller.

    Mirrors v1's `pulsim::v1::PIController`:

    - Trapezoidal integration of the error: I_k = I_{k-1} + Ki·dt·(e_k + e_{k-1})/2
    - Anti-windup: when the output saturates AND the integrator would
      push it further into saturation, the integrator is HELD (not
      decremented). Standard conditional anti-windup, robust to
      small numerical noise.
    - Hard output clamp to [output_min, output_max].

    Parameters
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        Integral gain (continuous-time convention; the discretization
        is done internally via dt).
    output_min, output_max : float
        Output clamp limits. Use `(-math.inf, math.inf)` to disable.
    integrator_state : float, default 0
        Initial value of the integrator (typically 0; for warm-start
        set to the expected steady-state output).
    """

    Kp: float = 1.0
    Ki: float = 0.0
    output_min: float = -math.inf
    output_max: float = math.inf
    integrator_state: float = 0.0
    # Internal: previous error for trapezoidal integration.
    _prev_error: float = field(default=0.0, repr=False)
    # Internal: latest output (after clamp), for convenience.
    output: float = field(default=0.0, repr=False)

    def reset(self, integrator_state: float = 0.0) -> None:
        """Zero the integrator and forget the previous error."""
        self.integrator_state = integrator_state
        self._prev_error = 0.0
        self.output = 0.0

    def update(
        self,
        *,
        setpoint: float,
        measured: float,
        dt: float,
    ) -> float:
        """Compute the next control output. Returns the clamped value."""
        error = setpoint - measured

        # Trapezoidal integration: integrator += Ki·dt·(e + e_prev)/2
        new_int = (
            self.integrator_state
            + self.Ki * dt * 0.5 * (error + self._prev_error)
        )

        # Provisional output (P + I terms).
        u_unclamped = self.Kp * error + new_int

        # Anti-windup: only accept the integrator update if it
        # doesn't push the output further into saturation.
        if u_unclamped > self.output_max and error > 0:
            # Would saturate high AND integrator pushing it higher
            # → freeze integrator at its old value.
            new_int = self.integrator_state
            u_unclamped = self.Kp * error + new_int
        elif u_unclamped < self.output_min and error < 0:
            new_int = self.integrator_state
            u_unclamped = self.Kp * error + new_int

        # Hard clamp on output.
        u_clamped = max(self.output_min, min(self.output_max, u_unclamped))

        # Commit.
        self.integrator_state = new_int
        self._prev_error = error
        self.output = u_clamped
        return u_clamped


# =============================================================================
# PID controller — adds backward-difference D with IIR filter
# =============================================================================

@dataclass
class PIDController:
    """Discrete-time PI + filtered D.

    Same I path as :class:`PIController`. The derivative is filtered
    by a single-pole IIR (alpha ∈ [0, 1]):
        D_k = alpha · ((e_k − e_{k-1}) / dt) + (1 − alpha) · D_{k-1}

    `alpha = 1` means raw derivative (no filtering, very noisy).
    `alpha → 0` means heavy filtering (smooth but laggy).
    Typical value: 0.1–0.3 for tame plants, 0.05 for noisy ones.

    Anti-windup logic identical to PIController.
    """

    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    derivative_alpha: float = 0.1   # IIR coefficient on D
    output_min: float = -math.inf
    output_max: float = math.inf
    integrator_state: float = 0.0
    derivative_state: float = field(default=0.0, repr=False)
    _prev_error: float = field(default=0.0, repr=False)
    output: float = field(default=0.0, repr=False)

    def reset(self, integrator_state: float = 0.0) -> None:
        self.integrator_state = integrator_state
        self.derivative_state = 0.0
        self._prev_error = 0.0
        self.output = 0.0

    def update(
        self,
        *,
        setpoint: float,
        measured: float,
        dt: float,
    ) -> float:
        error = setpoint - measured

        new_int = (
            self.integrator_state
            + self.Ki * dt * 0.5 * (error + self._prev_error)
        )

        # Filtered derivative (backward-difference + IIR).
        d_raw = (error - self._prev_error) / dt if dt > 0 else 0.0
        new_deriv = (
            self.derivative_alpha * d_raw
            + (1.0 - self.derivative_alpha) * self.derivative_state
        )

        u_unclamped = self.Kp * error + new_int + self.Kd * new_deriv

        # Anti-windup (only on I, not D — D doesn't accumulate).
        if u_unclamped > self.output_max and error > 0:
            new_int = self.integrator_state
            u_unclamped = self.Kp * error + new_int + self.Kd * new_deriv
        elif u_unclamped < self.output_min and error < 0:
            new_int = self.integrator_state
            u_unclamped = self.Kp * error + new_int + self.Kd * new_deriv

        u_clamped = max(self.output_min, min(self.output_max, u_unclamped))

        self.integrator_state = new_int
        self.derivative_state = new_deriv
        self._prev_error = error
        self.output = u_clamped
        return u_clamped


# =============================================================================
# Comparator with hysteresis (Schmitt trigger)
# =============================================================================

@dataclass
class Comparator:
    """Schmitt trigger / hysteretic comparator.

    Output flips HIGH when `input > threshold + hysteresis/2` and
    LOW when `input < threshold − hysteresis/2`. Stays in the
    same state inside the deadband.

    With `hysteresis = 0` this is a plain comparator (a single
    threshold). Use a non-zero band to prevent chatter when the
    input is noisy.
    """

    threshold: float = 0.0
    hysteresis: float = 0.0           # full band, NOT half
    output_high: float = 1.0
    output_low: float = 0.0
    state: bool = False               # True = HIGH

    def reset(self, state: bool = False) -> None:
        self.state = state

    def update(self, *, input_value: float) -> float:
        half_band = 0.5 * self.hysteresis
        if self.state:
            # Currently HIGH → flip LOW only if we cross the lower bound.
            if input_value < self.threshold - half_band:
                self.state = False
        else:
            # Currently LOW → flip HIGH only if we cross the upper bound.
            if input_value > self.threshold + half_band:
                self.state = True
        return self.output_high if self.state else self.output_low


# =============================================================================
# Rate limiter — asymmetric slew
# =============================================================================

@dataclass
class RateLimiter:
    """Limit how fast a signal can change.

    Allows independent positive and negative slew limits (e.g. a
    soft-start that ramps up at 1 V/ms but can drop instantly).
    """

    slew_up:   float = math.inf       # max +d/dt per second
    slew_down: float = math.inf       # max −d/dt per second
    state: float = 0.0

    def reset(self, state: float = 0.0) -> None:
        self.state = state

    def update(self, *, target: float, dt: float) -> float:
        delta = target - self.state
        max_up   = self.slew_up   * dt
        max_down = self.slew_down * dt
        if   delta >  max_up:   delta = max_up
        elif delta < -max_down: delta = -max_down
        self.state += delta
        return self.state


# =============================================================================
# Sample & hold — periodic ZOH
# =============================================================================

@dataclass
class SampleHold:
    """Zero-order hold sampled at a fixed period.

    Output is held constant for `sample_period` seconds, then
    snaps to the most recent input. Models a sampled digital
    controller updating once per switching cycle.
    """

    sample_period: float = 1e-3
    state: float = 0.0
    _last_sample_t: float = field(default=-math.inf, repr=False)

    def reset(self, state: float = 0.0) -> None:
        self.state = state
        self._last_sample_t = -math.inf

    def update(self, *, input_value: float, t: float) -> float:
        if t - self._last_sample_t >= self.sample_period:
            self.state = input_value
            self._last_sample_t = t
        return self.state


# =============================================================================
# First-order low-pass — exponential smoothing
# =============================================================================

@dataclass
class FirstOrderLowPass:
    """y_k = y_{k-1} + (dt / τ) · (x_k − y_{k-1}).

    Single-pole IIR filter with continuous-time time constant τ.
    Useful for conditioning a noisy feedback signal before feeding
    it to a controller.
    """

    tau: float = 1e-3
    state: float = 0.0

    def reset(self, state: float = 0.0) -> None:
        self.state = state

    def update(self, *, input_value: float, dt: float) -> float:
        if self.tau <= 0:
            self.state = input_value
        else:
            alpha = dt / (self.tau + dt)
            self.state += alpha * (input_value - self.state)
        return self.state


# =============================================================================
# Moving-average filter (exponential moving average — name kept from v1)
# =============================================================================

@dataclass
class MovingAverageFilter:
    """First-order exponential moving average — `y_k = α·y_{k-1} + (1-α)·x_k`.

    Despite the name (kept for v1 parity), this is an exponential filter,
    NOT a windowed boxcar average. `alpha` is the carry-over coefficient
    in [0, 1):
      - α = 0 → no filtering (output = input)
      - α → 1 → very slow filtering (long memory)
    """

    alpha: float = 0.9
    state: float = 0.0

    def reset(self, state: float = 0.0) -> None:
        self.state = state

    def update(self, *, input_value: float) -> float:
        self.state = self.alpha * self.state + (1.0 - self.alpha) * input_value
        return self.state


# =============================================================================
# Phase A.1 — math blocks
# =============================================================================

@dataclass
class Gain:
    """y = k · x. Stateless."""
    k: float = 1.0

    def reset(self) -> None:
        pass

    def update(self, *, x: float) -> float:
        return self.k * x


@dataclass
class Sum:
    """N-input weighted sum: y = Σ w_i · x_i.

    `weights` is a tuple of N coefficients. Pass `inputs` as a tuple
    of the same length to `update`. Default weights of all 1.0 give
    a plain sum.
    """
    weights: tuple = (1.0, 1.0)

    def reset(self) -> None:
        pass

    def update(self, *, inputs) -> float:
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Sum: expected {len(self.weights)} inputs, "
                f"got {len(inputs)}")
        return sum(w * x for w, x in zip(self.weights, inputs))


@dataclass
class Subtract:
    """y = a − b. Stateless."""

    def reset(self) -> None:
        pass

    def update(self, *, a: float, b: float) -> float:
        return a - b


@dataclass
class MathBlock:
    """Generic 1- or 2-input math op selected by string.

    `op` ∈ {"add", "sub", "mul", "div", "abs", "neg", "sqrt", "pow2"}.
    `add/sub/mul/div` need both `a` and `b`. `abs/neg/sqrt/pow2` use
    only `a`.
    """
    op: str = "add"

    def reset(self) -> None:
        pass

    def update(self, *, a: float, b: float = 0.0) -> float:
        op = self.op.lower()
        if op == "add":  return a + b
        if op == "sub":  return a - b
        if op == "mul":  return a * b
        if op == "div":  return a / b if b != 0.0 else 0.0
        if op == "abs":  return abs(a)
        if op == "neg":  return -a
        if op == "sqrt": return math.sqrt(a) if a >= 0 else 0.0
        if op == "pow2": return a * a
        raise ValueError(f"MathBlock: unknown op {self.op!r}")


# =============================================================================
# Phase A.1 — standalone control blocks
# =============================================================================

@dataclass
class Integrator:
    """y_k = clamp(y_{k-1} + gain · dt · x_k, output_min, output_max).

    Conditional anti-windup (freezes when output saturates and input
    pushes further into saturation). Forward-Euler discretization to
    keep the input-output relationship causal.
    """
    gain: float = 1.0
    output_min: float = -math.inf
    output_max: float = math.inf
    state: float = 0.0

    def reset(self, state: float = 0.0) -> None:
        self.state = state

    def update(self, *, x: float, dt: float) -> float:
        new = self.state + self.gain * dt * x
        if new > self.output_max and x > 0:
            new = self.state          # anti-windup hold
        elif new < self.output_min and x < 0:
            new = self.state
        new = max(self.output_min, min(self.output_max, new))
        self.state = new
        return new


@dataclass
class Differentiator:
    """Filtered backward-difference derivative.

    Raw: d_raw = (x_k − x_{k-1}) / dt
    IIR: d_k = α · d_raw + (1 − α) · d_{k-1}
    """
    alpha: float = 0.2
    _prev_x: float = field(default=0.0, repr=False)
    state: float = 0.0

    def reset(self, state: float = 0.0) -> None:
        self.state = state
        self._prev_x = 0.0

    def update(self, *, x: float, dt: float) -> float:
        d_raw = (x - self._prev_x) / dt if dt > 0 else 0.0
        self.state = self.alpha * d_raw + (1.0 - self.alpha) * self.state
        self._prev_x = x
        return self.state


@dataclass
class TransferFunction:
    """Discrete-time transfer function in direct form II.

    H(z) = Σ b_k · z^-k  /  (1 + Σ a_k · z^-k)

    `num` and `den` are sequences of coefficients in INCREASING powers
    of z^-1: `num[0]` multiplies the current input, `num[1]` the input
    one step ago, etc. `den[0]` is the implicit 1 (do NOT include it).
    """
    num: tuple = (1.0,)
    den: tuple = ()
    _x_hist: list = field(default_factory=list, repr=False)
    _y_hist: list = field(default_factory=list, repr=False)

    def reset(self) -> None:
        self._x_hist = []
        self._y_hist = []

    def update(self, *, x: float) -> float:
        self._x_hist.insert(0, x)
        # Keep only what we need.
        if len(self._x_hist) > len(self.num):
            self._x_hist = self._x_hist[:len(self.num)]
        # Compute the numerator contribution.
        num_part = sum(b * (self._x_hist[i] if i < len(self._x_hist) else 0.0)
                        for i, b in enumerate(self.num))
        # Compute the denominator (feedback) contribution.
        den_part = sum(a * (self._y_hist[i] if i < len(self._y_hist) else 0.0)
                        for i, a in enumerate(self.den))
        y = num_part - den_part
        self._y_hist.insert(0, y)
        if len(self._y_hist) > len(self.den):
            self._y_hist = self._y_hist[:max(1, len(self.den))]
        return y


@dataclass
class StateMachine:
    """Mealy state machine — mode ∈ {"toggle", "level", "sr_latch"}.

    - "toggle"   : output flips every time `trigger > threshold`
    - "level"    : output = high_value if input > threshold else low_value
    - "sr_latch" : two-input latch (S=set, R=reset, both via threshold)
    """
    mode: str = "level"
    threshold: float = 0.5
    high_value: float = 1.0
    low_value: float = 0.0
    state: float = 0.0
    _trigger_prev: float = field(default=0.0, repr=False)

    def reset(self, state: float = 0.0) -> None:
        self.state = state
        self._trigger_prev = 0.0

    def update(self, *, trigger: float = 0.0, set_: float = 0.0,
                 reset_: float = 0.0) -> float:
        mode = self.mode.lower()
        if mode == "toggle":
            rising_edge = (self._trigger_prev <= self.threshold
                            and trigger > self.threshold)
            if rising_edge:
                self.state = (self.low_value if self.state == self.high_value
                                else self.high_value)
            self._trigger_prev = trigger
        elif mode == "level":
            self.state = (self.high_value if trigger > self.threshold
                            else self.low_value)
        elif mode == "sr_latch":
            if set_ > self.threshold:
                self.state = self.high_value
            elif reset_ > self.threshold:
                self.state = self.low_value
        else:
            raise ValueError(f"StateMachine: unknown mode {self.mode!r}")
        return self.state


@dataclass
class OpAmp:
    """Saturating ideal op-amp — y = clamp(gain · (in_pos − in_neg), rails).

    Stateless. Use this when you want a pure software op-amp (no
    branch-current unknown). The circuit-side VCVS-based op-amp
    (`add_op_amp_ideal`) is preferred when the op-amp is part of an
    electrical feedback network.
    """
    gain: float = 1e5
    rail_min: float = -math.inf
    rail_max: float = math.inf

    def reset(self) -> None:
        pass

    def update(self, *, in_pos: float, in_neg: float) -> float:
        u = self.gain * (in_pos - in_neg)
        return max(self.rail_min, min(self.rail_max, u))


# =============================================================================
# Phase A.1 — signal-shaping blocks
# =============================================================================

@dataclass
class Limiter:
    """Hard clamp: y = clamp(x, min, max). Stateless."""
    output_min: float = -math.inf
    output_max: float = math.inf

    def reset(self) -> None:
        pass

    def update(self, *, x: float) -> float:
        return max(self.output_min, min(self.output_max, x))


@dataclass
class DelayBlock:
    """FIFO-buffer delay: outputs the input from `samples` updates ago.

    The first `samples` calls return `initial_value` (default 0).
    """
    samples: int = 1
    initial_value: float = 0.0
    _buf: list = field(default_factory=list, repr=False)

    def reset(self, initial: float | None = None) -> None:
        v = self.initial_value if initial is None else initial
        self._buf = [v] * self.samples

    def update(self, *, x: float) -> float:
        if len(self._buf) != self.samples:
            self.reset()
        out = self._buf.pop(0)
        self._buf.append(x)
        return out


# =============================================================================
# Phase A.1 — modulation blocks
# =============================================================================

@dataclass
class PwmGenerator:
    """Naturally-sampled PWM modulator — outputs 0/1 based on whether
    a sawtooth carrier exceeds the input duty command.

    Returns 1.0 while `phase ∈ [0, duty·T)` and 0.0 for the rest of
    the cycle. The block is purely software (no SwitchStateMask) — use
    the output as a control signal, drive a `add_voltage_source`'s
    overlay, or feed it into a higher-level switch_fn.
    """
    frequency: float = 100e3
    phase: float = 0.0          # cycle-relative offset [s]
    state: float = 0.0

    def reset(self, state: float = 0.0) -> None:
        self.state = state

    def update(self, *, duty: float, t: float) -> float:
        T = 1.0 / self.frequency if self.frequency > 0 else math.inf
        if not math.isfinite(T) or T == 0:
            self.state = 0.0
            return 0.0
        phase01 = math.fmod((t - self.phase) / T, 1.0)
        if phase01 < 0:
            phase01 += 1.0
        self.state = 1.0 if phase01 < duty else 0.0
        return self.state


@dataclass
class SpaceVectorModulator:
    """Space-vector PWM (SVM) for a 3-leg VSI.

    Input: αβ-frame reference vector (`v_alpha`, `v_beta`).
    Output: three duty cycles `d_a`, `d_b`, `d_c` for the upper-leg
    switches (lower-leg = 1 − d).

    Uses the classical 7-segment SVM ordering with the zero-vector
    centered (symmetric V0-V7-V0 split). Modulation index up to ≈
    1.155 (15 % over linear SPWM) before overmodulation.

    Returns a tuple `(d_a, d_b, d_c)`.
    """
    v_dc: float = 1.0
    state: tuple = (0.5, 0.5, 0.5)

    def reset(self, state: tuple = (0.5, 0.5, 0.5)) -> None:
        self.state = state

    def update(self, *, v_alpha: float, v_beta: float) -> tuple:
        # Sector identification (1..6).
        v3 = math.sqrt(3.0)
        # SVM duty equations — classic min/max-injection form.
        u_a = v_alpha
        u_b = (-v_alpha + v3 * v_beta) / 2.0
        u_c = (-v_alpha - v3 * v_beta) / 2.0
        # Min/max injection for zero-sequence centering.
        u_max = max(u_a, u_b, u_c)
        u_min = min(u_a, u_b, u_c)
        u_zero = 0.5 * (u_max + u_min)
        # Final duties (0..1).
        v_dc = self.v_dc if self.v_dc > 0 else 1.0
        d_a = 0.5 + (u_a - u_zero) / v_dc
        d_b = 0.5 + (u_b - u_zero) / v_dc
        d_c = 0.5 + (u_c - u_zero) / v_dc
        # Clamp to [0, 1] (overmodulation handling).
        d_a = max(0.0, min(1.0, d_a))
        d_b = max(0.0, min(1.0, d_b))
        d_c = max(0.0, min(1.0, d_c))
        self.state = (d_a, d_b, d_c)
        return self.state


# =============================================================================
# Phase A.1 — transform blocks (Clarke, Park, inverses)
# =============================================================================

@dataclass
class ClarkeTransform:
    """abc → αβ0 (power-invariant convention).

    α = (2/3)·(a − b/2 − c/2)
    β = (2/3)·(√3/2)·(b − c)
    0 = (1/3)·(a + b + c)
    """

    def reset(self) -> None:
        pass

    def update(self, *, a: float, b: float, c: float) -> tuple:
        v3 = math.sqrt(3.0)
        alpha = (2.0 / 3.0) * (a - 0.5 * b - 0.5 * c)
        beta  = (2.0 / 3.0) * (v3 / 2.0) * (b - c)
        zero  = (1.0 / 3.0) * (a + b + c)
        return (alpha, beta, zero)


@dataclass
class InverseClarkeTransform:
    """αβ0 → abc (inverse of `ClarkeTransform`)."""

    def reset(self) -> None:
        pass

    def update(self, *, alpha: float, beta: float,
                 zero: float = 0.0) -> tuple:
        v3 = math.sqrt(3.0)
        a = alpha + zero
        b = -0.5 * alpha + (v3 / 2.0) * beta + zero
        c = -0.5 * alpha - (v3 / 2.0) * beta + zero
        return (a, b, c)


@dataclass
class ParkTransform:
    """αβ → dq with rotor angle θ.

    d =  α·cos(θ) + β·sin(θ)
    q = -α·sin(θ) + β·cos(θ)
    """

    def reset(self) -> None:
        pass

    def update(self, *, alpha: float, beta: float,
                 theta: float) -> tuple:
        c, s = math.cos(theta), math.sin(theta)
        d =  alpha * c + beta * s
        q = -alpha * s + beta * c
        return (d, q)


@dataclass
class InverseParkTransform:
    """dq → αβ with rotor angle θ (inverse of `ParkTransform`)."""

    def reset(self) -> None:
        pass

    def update(self, *, d: float, q: float,
                 theta: float) -> tuple:
        c, s = math.cos(theta), math.sin(theta)
        alpha = d * c - q * s
        beta  = d * s + q * c
        return (alpha, beta)


# =============================================================================
# Phase A.1 — phase-locked loop
# =============================================================================

@dataclass
class PLL:
    """Single-phase / αβ-frame PLL.

    Estimates the angle and frequency of a sinusoidal input. Internal
    structure: phase-detector (cross-product) → PI controller → VCO.

    Input: αβ pair (`v_alpha`, `v_beta`) from a Clarke transform of
    the measured signal.
    Output: estimated angle `theta_hat` in [0, 2π) and estimated
    frequency `omega_hat`.

    Defaults are tuned for 50/60 Hz AC mains. For motor sensorless
    control, set `f_nominal` to the expected motor electrical freq.
    """
    f_nominal: float = 60.0          # initial guess [Hz]
    Kp: float = 100.0
    Ki: float = 1000.0
    theta_hat: float = 0.0
    omega_hat: float = field(default=2.0 * math.pi * 60.0, repr=False)
    _integ: float = field(default=0.0, repr=False)

    def reset(self) -> None:
        self.theta_hat = 0.0
        self.omega_hat = 2.0 * math.pi * self.f_nominal
        self._integ = 0.0

    def update(self, *, v_alpha: float, v_beta: float,
                 dt: float) -> tuple:
        # Cross-product phase detector: rotate measured by -θ_hat.
        # Locked when q-axis = 0.
        c, s = math.cos(self.theta_hat), math.sin(self.theta_hat)
        q = -v_alpha * s + v_beta * c
        # PI controller drives q → 0.
        self._integ += self.Ki * dt * q
        domega = self.Kp * q + self._integ
        self.omega_hat = 2.0 * math.pi * self.f_nominal + domega
        self.theta_hat += self.omega_hat * dt
        # Wrap angle.
        self.theta_hat = math.fmod(self.theta_hat, 2.0 * math.pi)
        if self.theta_hat < 0:
            self.theta_hat += 2.0 * math.pi
        return (self.theta_hat, self.omega_hat)


# =============================================================================
# Phase A.1 — routing blocks (mux / demux)
# =============================================================================

@dataclass
class SignalMux:
    """Selects one of N inputs based on `selector` (integer index).

    Out-of-range selectors return the LAST input. Pass `inputs` as a
    tuple to each `update()`.
    """

    def reset(self) -> None:
        pass

    def update(self, *, selector: int, inputs) -> float:
        idx = max(0, min(len(inputs) - 1, int(selector)))
        return inputs[idx]


@dataclass
class SignalDemux:
    """Broadcasts one input to N outputs.

    `update()` returns a tuple of length `n_outputs` where ALL entries
    equal the input — useful as a "fan-out" hint inside YAML chains
    (Python users can just reuse the variable).
    """
    n_outputs: int = 2

    def reset(self) -> None:
        pass

    def update(self, *, x: float) -> tuple:
        return tuple(x for _ in range(self.n_outputs))


# =============================================================================
# 1-D lookup table (linear interpolation)
# =============================================================================

@dataclass
class LookupTable1D:
    """Sorted x-array → linearly interpolated y. Used for
    pre-computed compensator schedules, soft-start ramps, etc."""

    xs: tuple = ()
    ys: tuple = ()

    def update(self, *, x: float) -> float:
        if not self.xs:
            return 0.0
        if x <= self.xs[0]:
            return self.ys[0]
        if x >= self.xs[-1]:
            return self.ys[-1]
        # Linear interp between bracketing points.
        for i in range(len(self.xs) - 1):
            if self.xs[i] <= x < self.xs[i + 1]:
                t = (x - self.xs[i]) / (self.xs[i + 1] - self.xs[i])
                return self.ys[i] + t * (self.ys[i + 1] - self.ys[i])
        return self.ys[-1]


# =============================================================================
# Auto-tuning — Bode-based loop shaping
# =============================================================================

def _interp_complex_at_freq(freqs, H, f_target):
    """Linear-interp |H| (dB) and ∠H (deg) at f_target.
    Returns (|H|_linear, phase_deg).
    """
    import math as _math
    import numpy as _np
    freqs = _np.asarray(freqs, dtype=float)
    H = _np.asarray(H, dtype=complex)
    if f_target <= freqs[0]:
        return abs(H[0]), _math.degrees(_np.angle(H[0]))
    if f_target >= freqs[-1]:
        return abs(H[-1]), _math.degrees(_np.angle(H[-1]))
    # Bracket.
    idx = int(_np.searchsorted(freqs, f_target))
    f0, f1 = freqs[idx - 1], freqs[idx]
    mag0, mag1 = abs(H[idx - 1]), abs(H[idx])
    ph0 = _math.degrees(_np.angle(H[idx - 1]))
    ph1 = _math.degrees(_np.angle(H[idx]))
    # Phase unwrap across the bracket: if jump > 180°, undo it.
    if ph1 - ph0 >  180.0: ph1 -= 360.0
    if ph1 - ph0 < -180.0: ph1 += 360.0
    # Log-linear in freq, linear in dB / deg.
    log_f0 = _math.log10(f0)
    log_f1 = _math.log10(f1)
    log_ft = _math.log10(f_target)
    alpha = (log_ft - log_f0) / (log_f1 - log_f0)
    mag = 10 ** ((1 - alpha) * _math.log10(mag0)
                  + alpha * _math.log10(mag1))
    phase = (1 - alpha) * ph0 + alpha * ph1
    return mag, phase


def tune_pi_from_bode(
    freqs,
    H,
    *,
    f_crossover: float,
    phase_margin_deg: float = 60.0,
    output_min: float = -math.inf,
    output_max: float = math.inf,
) -> dict:
    """Auto-tune a PI controller from a measured plant Bode.

    Given the plant frequency response `H(jω)` sampled at `freqs`,
    choose Kp and Ki such that the closed-loop crossover happens at
    `f_crossover` Hz with the requested `phase_margin_deg`.

    Math (loop-shaping for PI):

        Plant gain/phase at ω_c:  Mg = |G(jω_c)|, φ_g = ∠G(jω_c)
        Required PI phase at ω_c: φ_pi = phase_margin - 180° - φ_g
        For a pure PI: φ_pi must lie in (-90°, 0°). If it doesn't,
        the design is infeasible (the plant phase at ω_c is too
        far from -180° + PM); the function falls back to a
        conservative rule-of-thumb and emits a warning.

        From C(s) = Kp · (1 + 1/(s·τi)):
            ∠C(jω_c) = arctan(-1/(ω_c·τi)) = φ_pi
            → τi = -1 / (ω_c · tan(φ_pi))
            |C(jω_c)| = Kp / cos(φ_pi)
        For unity loop gain at crossover:
            Kp · |G(jω_c)| / cos(φ_pi) = 1
            → Kp = cos(φ_pi) / Mg
            → Ki = Kp / τi

    Parameters
    ----------
    freqs
        Plant Bode frequencies (Hz), array-like.
    H
        Plant complex frequency response sampled at `freqs`.
    f_crossover
        Target loop-gain crossover frequency (Hz).
    phase_margin_deg
        Target phase margin (degrees). Default 60° — robust
        general-purpose choice.
    output_min, output_max
        Forwarded to the returned PIController so it's ready to
        use.

    Returns
    -------
    dict with keys:
        Kp, Ki                 — gains
        controller             — a `PIController` instance ready to use
        achieved_pm_deg        — the phase margin the controller delivers
        crossover_hz           — `f_crossover` echoed back
        plant_mag_at_crossover — |G(jω_c)|
        plant_phase_at_crossover_deg — ∠G(jω_c)
        warnings               — list[str] of design notes
    """
    import math as _math

    warnings: list[str] = []
    omega_c = 2.0 * _math.pi * f_crossover
    Mg, phi_g = _interp_complex_at_freq(freqs, H, f_crossover)

    # Required PI phase contribution to hit the target PM.
    phi_pi_deg = phase_margin_deg - 180.0 - phi_g

    # Feasibility: a pure PI can only add phase in (-90°, 0°).
    if phi_pi_deg >= 0.0 or phi_pi_deg <= -90.0:
        warnings.append(
            f"Pure-PI design infeasible at f_c={f_crossover:.1f} Hz "
            f"with PM={phase_margin_deg}° (need φ_pi={phi_pi_deg:.1f}° "
            f"∉ (-90°, 0°)). Falling back to a conservative rule of "
            f"thumb (Kp = 1/Mg, ω_z = ω_c/10)."
        )
        Kp = 1.0 / Mg
        Ki = Kp * omega_c / 10.0
        # Compute the achieved PM at the original target (likely
        # not exactly PM_target).
        # |L(jω_c)| = Kp·sqrt(1 + (Ki/(Kp·ω_c))²)·Mg
        # ∠L(jω_c) = arctan(-Ki/(Kp·ω_c)) + φ_g
        wt = Ki / (Kp * omega_c)
        pm_achieved = 180.0 + _math.degrees(
            _math.atan(-wt) + _math.radians(phi_g)
        )
    else:
        phi_pi_rad = _math.radians(phi_pi_deg)
        tau_i = -1.0 / (omega_c * _math.tan(phi_pi_rad))
        Kp = _math.cos(phi_pi_rad) / Mg
        Ki = Kp / tau_i
        pm_achieved = phase_margin_deg  # by construction

    controller = PIController(
        Kp=Kp, Ki=Ki,
        output_min=output_min, output_max=output_max,
    )

    return {
        "Kp": Kp,
        "Ki": Ki,
        "controller": controller,
        "achieved_pm_deg": pm_achieved,
        "crossover_hz": f_crossover,
        "plant_mag_at_crossover": Mg,
        "plant_phase_at_crossover_deg": phi_g,
        "warnings": warnings,
    }


def loop_gain(freqs, H_plant, Kp: float, Ki: float):
    """Compute L(jω) = C(jω) · G(jω) where C is a PI controller.

    Returns an array of complex L values matching `freqs`.
    """
    import numpy as _np
    omega = 2.0 * _np.pi * _np.asarray(freqs, dtype=float)
    C = Kp + Ki / (1j * omega)
    H = _np.asarray(H_plant, dtype=complex)
    return C * H


def phase_margin_from_loop(freqs, L) -> float:
    """Phase margin in degrees from loop gain L(jω).

    PM = 180° + ∠L(jω_c)  where ω_c is the **last** frequency
    at which |L| crosses 1 from above. Picking the last
    crossing (= the highest-frequency 0-dB crossover before
    rolloff) is the convention that matters for stability:
    earlier crossings may exist due to local peaks in the
    plant Bode (e.g. an LC resonance), but only the last one
    determines how close the loop is to encircling −1.
    Returns +inf if |L| never crosses 1.
    """
    import math as _math
    import numpy as _np
    freqs = _np.asarray(freqs, dtype=float)
    L = _np.asarray(L, dtype=complex)
    mag = _np.abs(L)
    # Find ALL down-crossings of |L| = 1, take the highest-freq one.
    crossed = mag[:-1] >= 1.0
    next_below = mag[1:] < 1.0
    crossings = _np.where(crossed & next_below)[0]
    if len(crossings) == 0:
        return float("inf")
    i = int(crossings[-1])    # last crossing
    # Interpolate the crossing freq in log scale.
    log_f0, log_f1 = _math.log10(freqs[i]), _math.log10(freqs[i + 1])
    log_m0, log_m1 = _math.log10(mag[i]), _math.log10(mag[i + 1])
    alpha = (0.0 - log_m0) / (log_m1 - log_m0)   # mag=1 → log_mag=0
    log_fc = (1 - alpha) * log_f0 + alpha * log_f1
    # Interpolate phase at f_c.
    ph0 = _math.degrees(_np.angle(L[i]))
    ph1 = _math.degrees(_np.angle(L[i + 1]))
    if ph1 - ph0 >  180.0: ph1 -= 360.0
    if ph1 - ph0 < -180.0: ph1 += 360.0
    log_fc_lin = (log_fc - log_f0) / (log_f1 - log_f0)
    phase_at_xover = (1 - log_fc_lin) * ph0 + log_fc_lin * ph1
    return 180.0 + phase_at_xover


def gain_margin_from_loop(freqs, L) -> float:
    """Gain margin in dB from loop gain L(jω).

    GM = -20·log10(|L(jω_180)|) where ω_180 is the first frequency
    at which ∠L = -180°. Returns +inf if no -180° crossing.
    """
    import math as _math
    import numpy as _np
    freqs = _np.asarray(freqs, dtype=float)
    L = _np.asarray(L, dtype=complex)
    # Unwrap phase before searching.
    phase = _np.unwrap(_np.angle(L))
    phase_deg = _np.degrees(phase)
    # Find the lowest freq at which phase crosses -180°.
    above = phase_deg[:-1] > -180.0
    next_below = phase_deg[1:] <= -180.0
    crossings = _np.where(above & next_below)[0]
    if len(crossings) == 0:
        return float("inf")
    i = int(crossings[0])
    # Interpolate the crossing.
    alpha = (-180.0 - phase_deg[i]) / (phase_deg[i + 1] - phase_deg[i])
    log_m0, log_m1 = _math.log10(abs(L[i])), _math.log10(abs(L[i + 1]))
    log_m180 = (1 - alpha) * log_m0 + alpha * log_m1
    return -20.0 * log_m180


# ---------------------------------------------------------------------------
# add-python-closed-loop-helper (v1.5) — PI + PWM bound to a switch
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClosedLoop:
    """Output of :func:`bind_pi_to_switch` — packages the per-loop
    callbacks plus their history records into a single immutable
    handle.

    Why frozen? The fields hold *references* the simulator needs; the
    history lists are append-only, populated by the captured closure
    inside the factory. Freezing the dataclass prevents accidental
    reassignment while leaving the lists mutable.

    Fields
    ------
    switch_fn
        Callable ``t -> SwitchStateMask`` that toggles the bound
        switch's bit according to the PWM phase test
        ``(t mod T_PWM) / T_PWM < duty``.
    step_observer
        Callable ``(t, x)`` that throttles to one PI update per PWM
        period (``T_PWM = 1 / freq``), reads the measured signal via
        the user-supplied lambda, calls ``pi.update(setpoint=…,
        measured=…, dt=T_PWM)``, and pushes the new duty into the
        shared state ``switch_fn`` reads on its next invocation.
    duty_history
        List of ``(t_seconds, duty_unit)`` tuples appended once per
        PI tick. Convert to numpy with
        ``np.asarray(loop.duty_history).T`` to plot.
    error_history
        List of ``(t_seconds, error_unit)`` tuples in the same
        cadence as ``duty_history``.
    """
    switch_fn: "Callable[[float], object]"  # noqa: F821 — SwitchStateMask
    step_observer: "Callable[[float, object], None]"
    duty_history: "list[tuple[float, float]]"
    error_history: "list[tuple[float, float]]"


def bind_pi_to_switch(
    builder,
    *,
    pi: PIController,
    measured: "Callable[[object], float]",
    setpoint: float,
    switch,
    freq: float,
    t_start: float = 0.0,
) -> ClosedLoop:
    """Return a :class:`ClosedLoop` that wires a PI controller's output
    into a PWM-driven switch's duty.

    Replaces the ~30 lines of closure-with-mutable-list boilerplate
    every closed-loop script reinvents.

    Parameters
    ----------
    builder
        Populated :class:`CircuitBuilder`. Used to resolve
        ``switch`` (when given by name) and to size the
        ``SwitchStateMask``.
    pi
        :class:`PIController` instance. Its ``update`` is called
        once per PWM period.
    measured
        Callable ``state_vec -> float`` extracting the feedback
        signal. Typical: ``lambda x: x[builder.node_id_of("vout")]``.
    setpoint
        Reference value passed straight to ``pi.update``.
    switch
        Either a device name (resolved via
        ``builder.switch_index_of``) or an integer bit position.
    freq
        PWM carrier frequency in Hz. The PI is throttled to one
        update per ``T_PWM = 1 / freq``.
    t_start
        Simulation start time, used to align the throttle's first
        tick. Defaults to 0.

    Examples
    --------
    A buck closed loop on V_in=12 V, V_ref=5 V::

        pi = ps.PIController(Kp=0.08, Ki=40.0, output_min=0.05,
                              output_max=0.95)
        loop = ps.control.bind_pi_to_switch(
            builder,
            pi=pi,
            measured=lambda x: x[builder.node_id_of("vout")],
            setpoint=5.0,
            switch="Q1",
            freq=10e3,
        )
        res = ps.simulate(builder, t_end=20e-3, dt=2e-6,
                          switch_fn=loop.switch_fn,
                          step_observer=loop.step_observer)
    """
    # Import lazily so `pulsim.control` doesn't depend on the C++
    # binding at module-import time (test fixtures sometimes import
    # control.py without the kernel).
    from . import SwitchStateMask as _Mask

    if freq <= 0:
        raise ValueError(f"freq must be > 0 Hz; got {freq}")

    T_PWM = 1.0 / float(freq)

    # Resolve `switch` to an integer bit position. Accept either a
    # device name or an integer index — the name path goes through
    # `add-python-named-lookups`' `switch_index_of`.
    if isinstance(switch, str):
        idx = int(builder.switch_index_of(switch))
    else:
        idx = int(switch)

    n_switches = int(builder.graph.num_switches)
    if idx < 0 or idx >= n_switches:
        raise ValueError(
            f"switch index {idx} out of range for "
            f"num_switches={n_switches}"
        )

    # Shared mutable state — lives in this closure. The factory hides
    # the `[0.5]` / `[t_start - T_PWM]` lists callers used to write
    # by hand.
    duty_state = [0.5]
    last_tick = [t_start - T_PWM]  # forces immediate first update
    duty_history: "list[tuple[float, float]]" = []
    error_history: "list[tuple[float, float]]" = []

    def switch_fn(t: float):
        mask = _Mask(n_switches)
        phase = (t % T_PWM) / T_PWM
        if phase < duty_state[0]:
            mask.set(idx, True)
        return mask

    def step_observer(t: float, x) -> None:
        # Throttle: skip until at least one T_PWM has elapsed since
        # the last update. Prevents the loop from chasing PWM ripple.
        if t - last_tick[0] < T_PWM:
            return
        last_tick[0] = t
        m = float(measured(x))
        error = setpoint - m
        new_duty = pi.update(setpoint=setpoint, measured=m, dt=T_PWM)
        duty_state[0] = float(new_duty)
        duty_history.append((t, duty_state[0]))
        error_history.append((t, error))

    return ClosedLoop(
        switch_fn=switch_fn,
        step_observer=step_observer,
        duty_history=duty_history,
        error_history=error_history,
    )


def bind_pi_to_duty_callable(
    builder,
    *,
    pi: PIController,
    measured: "Callable[[object], float]",
    setpoint: float,
    freq: float,
    t_start: float = 0.0,
):
    """Like :func:`bind_pi_to_switch` but returns the controller as a
    ``(duty_get, step_observer, duty_history)`` triple instead of a
    switch-bound loop.

    Useful when one PI drives multiple switches (e.g., a half-bridge
    complementary pair where the high-side gets ``duty`` and the
    low-side gets ``1 - duty``)::

        duty_get, observer, history = ps.control.bind_pi_to_duty_callable(
            builder, pi=pi, measured=..., setpoint=..., freq=10e3,
        )

        def switch_fn(t):
            mask = ps.SwitchStateMask(builder.num_switches)
            phase = (t % T_PWM) / T_PWM
            d = duty_get()
            mask.set(builder.switch_index_of("Q_HIGH"), phase < d)
            mask.set(builder.switch_index_of("Q_LOW"),  phase >= d)
            return mask

    Returns
    -------
    duty_get : Callable[[], float]
        Zero-arg function returning the current duty (read by the
        custom ``switch_fn``).
    step_observer : Callable[[float, np.ndarray], None]
        Same throttled PI loop as :func:`bind_pi_to_switch` produces.
    duty_history : list[tuple[float, float]]
        Append-only history of ``(t, duty)`` updates.
    """
    if freq <= 0:
        raise ValueError(f"freq must be > 0 Hz; got {freq}")

    T_PWM = 1.0 / float(freq)
    duty_state = [0.5]
    last_tick = [t_start - T_PWM]
    duty_history: "list[tuple[float, float]]" = []

    def duty_get() -> float:
        return duty_state[0]

    def step_observer(t: float, x) -> None:
        if t - last_tick[0] < T_PWM:
            return
        last_tick[0] = t
        new_duty = pi.update(
            setpoint=setpoint,
            measured=float(measured(x)),
            dt=T_PWM,
        )
        duty_state[0] = float(new_duty)
        duty_history.append((t, duty_state[0]))

    return duty_get, step_observer, duty_history
