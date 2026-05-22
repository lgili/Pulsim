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
    import numpy as _np

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
    f_c = 10 ** log_fc
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
    log_f0, log_f1 = _math.log10(freqs[i]), _math.log10(freqs[i + 1])
    log_f180 = (1 - alpha) * log_f0 + alpha * log_f1
    log_m0, log_m1 = _math.log10(abs(L[i])), _math.log10(abs(L[i + 1]))
    log_m180 = (1 - alpha) * log_m0 + alpha * log_m1
    return -20.0 * log_m180
