"""Pulsim v2 — discrete-time control building blocks.

These are pure Python stateful classes that mirror v1's
`control.hpp` C++ helpers (PIController, PIDController,
Comparator, etc.). Combine them with the v2
`step_observer(t, x)` callback to close the loop around a v2
plant without touching the C++ kernel.

The canonical pattern:

    import pulsim.v2 as p
    from pulsim.v2_control import PIController

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
from typing import Optional


__all__ = [
    "PIController",
    "PIDController",
    "Comparator",
    "RateLimiter",
    "SampleHold",
    "FirstOrderLowPass",
    "LookupTable1D",
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
