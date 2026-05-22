"""Pulsim v2 — harmonic-balance / harmonic-analysis helpers.

Closes Phase-2 gap #2 from the v1→v2 retirement punch list.

Engineering perspective
-----------------------
"Harmonic Balance" in textbooks is a frequency-domain solver that
iterates on Fourier coefficients. In practice, what a power-
electronics engineer actually wants from "HB" is the **harmonic
spectrum of the steady-state operating point** — DC, fundamental,
2nd, 3rd, ... so they can:

* compute THD (total harmonic distortion),
* check whether ripple complies with a spec (e.g. ripple voltage at
  the switching frequency vs at line-frequency multiples),
* identify which harmonic dominates the converter loss.

This module ships two functions:

* :func:`analyze_harmonics(result, *, period, n_harmonics, ...)` —
  post-process any :class:`SimulationResult` that ALREADY ran long
  enough for the natural transient to die out. FFT every state
  variable over one period. Returns amplitudes + phases + THD.

* :func:`run_harmonic_balance(builder, *, period, ...)` — convenience
  wrapper. Calls :func:`run_periodic_shooting` to find the
  steady-state ``x*``, runs **one clean period** from ``x*``, FFTs.
  The full "find SS + measure spectrum" pipeline in one call.

For a true frequency-domain Newton (multi-tone or
non-perturbatively-nonlinear systems) — that's a future addition.
The shooting-based path here covers ≥ 80 % of power-electronics
use cases at a fraction of the implementation cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


__all__ = [
    "HarmonicAnalysisResult",
    "analyze_harmonics",
    "run_harmonic_balance",
]


@dataclass
class HarmonicAnalysisResult:
    """Spectrum of every state variable over one steady-state period.

    Shapes
    ------
    Let ``H`` = ``n_harmonics + 1`` (DC + ``n_harmonics`` line tones)
    and ``N`` = ``state_size``.

    Attributes
    ----------
    frequencies
        Length ``H`` array — ``[0, f0, 2 f0, …, n_harmonics · f0]``
        in Hz.
    amplitudes
        Shape ``(N, H)``. ``amplitudes[i, 0]`` is the DC value of
        state ``i``; ``amplitudes[i, k]`` (k ≥ 1) is the peak
        amplitude (not RMS) at harmonic ``k``. For real signals,
        ``A_k = 2 · |X_k|`` where ``X_k`` is the complex DFT bin.
    phases_deg
        Shape ``(N, H)``. Phase of each harmonic in degrees,
        wrapped to ``(-180, 180]``. ``phases_deg[i, 0]`` is 0 by
        convention (DC has no phase).
    thd
        Shape ``(N,)``. Total harmonic distortion, ratio (not %).
        Computed as ``sqrt(Σ_{k≥2} A_k²) / A_1``. ``NaN`` for
        states where ``A_1`` is below ``min_fundamental`` (default
        1e-12) — protects against divide-by-zero on quiet states.
    complex_coefficients
        Shape ``(N, H)``. Full complex DFT bins (``X_k``).
    state_labels
        Optional per-state label list of length ``N``.
    period
        Period over which the FFT was taken (s) — copied through
        from the caller's input.
    """
    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases_deg: np.ndarray
    thd: np.ndarray
    complex_coefficients: np.ndarray
    state_labels: List[str] = field(default_factory=list)
    period: float = 0.0


def _extract_last_period(result, period: float):
    """Slice the simulation result down to the LAST full period.
    Returns (t_arr, x_arr) where x_arr has shape (n_samples,
    state_size). Resamples uniformly via interpolation to keep the
    FFT exact even when the kernel ran with variable dt."""
    times = np.asarray(result.times, dtype=np.float64)
    states_raw = result.states
    if hasattr(states_raw, "shape"):
        states = np.asarray(states_raw, dtype=np.float64)
    else:
        states = np.asarray(
            [list(v) for v in states_raw], dtype=np.float64)

    t_end = float(times[-1])
    t_start = t_end - period
    # Allow a 1-sample (or 1 ppm) tolerance — when the user runs
    # exactly ``t_end = period`` floating-point can leave t_start
    # slightly below times[0].
    tol = max(period * 1e-6, (times[-1] - times[0]) / max(times.size - 1, 1))
    if t_start < float(times[0]) - tol:
        raise ValueError(
            f"analyze_harmonics: simulation window "
            f"[{times[0]:.6g}, {t_end:.6g}] is shorter than one "
            f"period ({period:.6g}). Either run longer or pass a "
            f"shorter period.")
    t_start = max(t_start, float(times[0]))

    mask = times >= t_start
    return times[mask], states[mask]


def analyze_harmonics(result,
                          *,
                          period: float,
                          n_harmonics: int = 20,
                          n_samples_per_period: int = 1024,
                          state_labels: Optional[Sequence[str]] = None,
                          min_fundamental: float = 1e-12,
                          ) -> HarmonicAnalysisResult:
    """Extract per-harmonic amplitude + phase from a steady-state
    :class:`SimulationResult`.

    The function takes the LAST ``period`` worth of samples, resamples
    them uniformly via linear interpolation (so the result is exact
    even when the kernel used variable dt), then FFTs each state
    variable.

    Parameters
    ----------
    result
        A :class:`SimulationResult` whose tail has already converged
        to steady state. Length must be ``≥ period``.
    period
        Fundamental period (s). For PWM converters ``T = 1 / f_pwm``;
        for line-driven ``T = 1 / f_line``.
    n_harmonics
        How many harmonics to report beyond DC. Default 20.
        Must satisfy ``2 · n_harmonics + 1 ≤ n_samples_per_period``
        (Nyquist).
    n_samples_per_period
        Uniform resample resolution. Must be ≥ ``2 · n_harmonics``
        per Nyquist. Default 1024 — overkill but cheap.
    state_labels
        Optional labels parallel to the state vector. Default
        ``[f"x[{i}]" for i in range(N)]``.
    min_fundamental
        Below this absolute amplitude, the THD for that state is
        reported as ``NaN`` (the fundamental is below numerical
        noise; the THD ratio is meaningless).

    Returns
    -------
    HarmonicAnalysisResult
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")
    if n_samples_per_period < 2 * n_harmonics + 1:
        raise ValueError(
            f"n_samples_per_period={n_samples_per_period} too small "
            f"for n_harmonics={n_harmonics} (need ≥ "
            f"{2 * n_harmonics + 1})")

    t_slice, x_slice = _extract_last_period(result, period)
    if t_slice.size < 2:
        raise ValueError(
            "result has too few samples in the last period")

    state_size = x_slice.shape[1]
    if state_labels is None:
        state_labels = [f"x[{i}]" for i in range(state_size)]
    elif len(state_labels) != state_size:
        raise ValueError(
            f"state_labels has {len(state_labels)} entries but "
            f"state vector has {state_size}")

    # Resample on a uniform grid covering EXACTLY one period.
    # Endpoint=False so the period boundary doesn't double-count.
    t_uniform = t_slice[0] + np.linspace(
        0.0, period, n_samples_per_period, endpoint=False)
    # Linear interpolation per state column.
    x_uniform = np.empty(
        (n_samples_per_period, state_size), dtype=np.float64)
    for i in range(state_size):
        x_uniform[:, i] = np.interp(t_uniform, t_slice, x_slice[:, i])

    # FFT along time axis (axis 0). rfft for real input.
    X = np.fft.rfft(x_uniform, axis=0) / n_samples_per_period
    # X[0]   = DC mean
    # X[k]   = (a_k - j·b_k) / 2 for k ≥ 1 (numpy convention with the
    #          1/N normalisation applied above). Peak amplitude is
    #          2 · |X[k]|.

    H = n_harmonics + 1
    if X.shape[0] < H:
        raise RuntimeError(
            "FFT returned fewer bins than requested harmonics — "
            "increase n_samples_per_period")

    coeffs = X[:H, :].T  # shape (state_size, H)

    amplitudes = np.abs(coeffs).copy()
    amplitudes[:, 1:] *= 2.0   # peak amplitude convention

    phases_deg = np.degrees(np.angle(coeffs))
    phases_deg[:, 0] = 0.0     # DC has no phase

    # THD = sqrt(Σ_{k≥2} A_k²) / A_1.
    fund = amplitudes[:, 1]
    upper = amplitudes[:, 2:]
    thd = np.where(
        np.abs(fund) > min_fundamental,
        np.sqrt(np.sum(upper ** 2, axis=1)) / fund,
        np.nan)

    f0 = 1.0 / period
    freqs = np.arange(H, dtype=np.float64) * f0

    return HarmonicAnalysisResult(
        frequencies=freqs,
        amplitudes=amplitudes,
        phases_deg=phases_deg,
        thd=thd,
        complex_coefficients=coeffs,
        state_labels=list(state_labels),
        period=float(period),
    )


def run_harmonic_balance(builder,
                              *,
                              period: float,
                              dt: float,
                              n_harmonics: int = 20,
                              n_samples_per_period: int = 1024,
                              switch_fn=None,
                              b_extra_fn=None,
                              state_labels: Optional[Sequence[str]] = None,
                              shooting_tol: float = 1e-6,
                              shooting_max_iters: int = 20,
                              **simulate_kwargs,
                              ) -> HarmonicAnalysisResult:
    """One-call harmonic balance: find steady-state via
    :func:`run_periodic_shooting`, then extract the harmonic
    spectrum via :func:`analyze_harmonics`.

    The full "do HB for me" pipeline. For a buck converter on a
    100 kHz PWM you'd typically write::

        result = p.run_harmonic_balance(
            b, period=10e-6, dt=100e-9,
            switch_fn=sw_fn,
            n_harmonics=10)
        print(f"V_out DC:        {result.amplitudes[Vout_idx, 0]:.3f} V")
        print(f"V_out 1st harm:  {result.amplitudes[Vout_idx, 1]*1e3:.1f} mV")
        print(f"V_out THD:       {result.thd[Vout_idx]*100:.2f} %")

    Parameters
    ----------
    builder
        Populated CircuitBuilder.
    period
        Fundamental period (s).
    dt
        Time step for the underlying simulation.
    n_harmonics, n_samples_per_period
        Forwarded to :func:`analyze_harmonics`.
    switch_fn, b_extra_fn
        Forwarded to the underlying ``simulate`` call. Must be
        periodic with ``period``.
    state_labels
        Optional per-state labels.
    shooting_tol, shooting_max_iters
        Tolerance + iteration cap for the underlying periodic
        shooting Newton.
    **simulate_kwargs
        Forwarded to ``simulate``.

    Returns
    -------
    HarmonicAnalysisResult
    """
    from . import v2 as _v2

    # 1) Find steady state via periodic shooting.
    ss = _v2.run_periodic_shooting(
        builder,
        t_period=period, dt=dt,
        switch_fn=switch_fn,
        b_extra_fn=b_extra_fn,
        tol=shooting_tol,
        max_iterations=shooting_max_iters,
        **simulate_kwargs)

    if not ss.converged:
        raise RuntimeError(
            f"run_harmonic_balance: periodic shooting failed to "
            f"converge (||G||_∞ = {ss.final_residual:.3e} after "
            f"{ss.n_iterations} iterations). Try a tighter "
            f"shooting_tol, larger shooting_max_iters, or pre-run "
            f"a transient to seed a closer x_initial.")

    # 2) Run ONE clean period from x_steady_state with
    #    enough samples for the FFT.
    res = _v2.simulate(
        builder,
        t_end=period, dt=dt,
        switch_fn=switch_fn,
        b_extra_fn=b_extra_fn,
        initial_state=ss.x_steady_state.tolist(),
        **simulate_kwargs)

    # 3) FFT.
    return analyze_harmonics(
        res,
        period=period,
        n_harmonics=n_harmonics,
        n_samples_per_period=n_samples_per_period,
        state_labels=state_labels)
