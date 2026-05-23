"""Pulsim v2 — Frequency Response Analyzer (FRA).

Closes Phase-2 gap #3 from the v1→v2 retirement punch list: a
swept-sine identification helper that complements the linearised
MNA-based ``run_ac_sweep``.

When to use FRA vs AC sweep
---------------------------
``run_ac_sweep`` linearises the small-signal model around a DC OP
and runs a one-shot frequency-domain solve. It's fast and exact
for **linear** circuits, and a good approximation for switching
converters when their averaged model is well-behaved.

``run_fra`` injects a real sine in the time domain, runs the
**full nonlinear** simulation, and recovers the gain/phase by
FFT. It's slower (one simulation per frequency point) but correct
even when the system has:

* hard switching with the switches included (not averaged);
* large-signal effects (slew limits, current limits, saturation);
* a controller that's part of the response (closed-loop Bode);
* nonlinearities the AC linearisation misses (PWL diodes off the
  conducting branch, BH-loop hysteresis if present).

Algorithm
---------
For each test frequency f:

1. Run for ``periods_settle`` periods at f to discard any
   transient excited by the perturbation onset.
2. Run for ``periods_measure`` periods while recording both the
   injected and measured signals at the kernel's dt.
3. Compute the DFT bin at exactly f in each signal. The ratio
   gives H(f); 20·log10|H| is the gain in dB and ∠H is the phase.

The injected signal is a current source applied via ``b_extra_fn``
at the user-specified state index (typically a node voltage's KCL
equation). The measured signal is read from the state vector at
another user-specified index. The user can stack multiple
measurements (the result has one (mag, phase) per measurement
channel).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np


__all__ = ["FraResult", "run_fra"]


@dataclass
class FraResult:
    """Output of :func:`run_fra`.

    Attributes
    ----------
    frequencies
        Array of test frequencies (Hz), same as the input.
    magnitude_db
        20 · log10 |H(f)| for each frequency. Shape:
        ``(n_measurements, n_frequencies)``.
    phase_deg
        Phase of H(f) in degrees, wrapped to ``(-180, 180]``.
        Same shape as ``magnitude_db``.
    H
        Complex transfer function value per measurement and frequency.
        Same shape as ``magnitude_db``.
    measurement_labels
        Names matching the ``measurements`` argument.
    """
    frequencies: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray
    H: np.ndarray
    measurement_labels: List[str]


def _dft_bin(signal: np.ndarray, dt: float, f: float) -> complex:
    """Compute the DFT at exactly frequency f from a uniformly
    sampled signal (no FFT, since we don't need the full spectrum
    — just one bin). Standard ``Σ x_k · exp(-j 2π f k dt) · dt``."""
    n = signal.size
    if n == 0:
        return complex(0.0, 0.0)
    k = np.arange(n, dtype=np.float64)
    phase = -2.0 * np.pi * f * k * dt
    return complex(np.sum(signal * np.exp(1j * phase)) * dt)


def run_fra(builder,
                *,
                frequencies: Sequence[float],
                injection_state_idx: int,
                injection_amplitude: float,
                measurements: Sequence[int],
                measurement_labels: Optional[Sequence[str]] = None,
                dt: Optional[float] = None,
                samples_per_period: int = 200,
                periods_settle: int = 3,
                periods_measure: int = 3,
                switch_fn: Optional[Callable] = None,
                step_observer: Optional[Callable] = None,
                base_b_extra_fn: Optional[Callable] = None,
                initial_state: Optional[np.ndarray] = None,
                **simulate_kwargs,
                ) -> FraResult:
    """Swept-sine FRA.

    Parameters
    ----------
    builder
        Populated CircuitBuilder.
    frequencies
        Array of test frequencies (Hz). One simulation per point.
    injection_state_idx
        Index in the state vector where the perturbation current
        ``injection_amplitude · sin(2π f t)`` is injected via
        ``b_extra_fn``. Typically the KCL row of the node being
        perturbed (for v2, this matches the state index of a node-
        voltage entry — use :meth:`builder.node_id_of` to resolve).
    injection_amplitude
        Peak amplitude of the injection current (A). Pick small
        enough to keep the perturbation linear (≤ 1% of the DC
        operating-point current at the same node is a reasonable
        starting point) and large enough that the response is well
        above numerical noise.
    measurements
        State indices to measure simultaneously. The FFT bin at f
        is computed for each — useful for single-input multi-output
        Bode (e.g. control-to-output AND input-to-output on the same
        injection).
    measurement_labels
        Optional names parallel to ``measurements``. Default
        ``"meas_<idx>"``.
    dt
        Simulation step. If ``None``, set automatically so that
        ``samples_per_period`` samples cover one period of the
        HIGHEST frequency.
    samples_per_period
        Used only when ``dt`` is auto. Default 200 (≫ Nyquist).
    periods_settle, periods_measure
        Number of full periods to skip (settle) and record (measure)
        per frequency. Defaults 3 + 3. Increase ``periods_measure``
        if a frequency falls near a strong harmonic of the switching
        ripple — averaging across more periods rejects the ripple.
    switch_fn, step_observer, base_b_extra_fn, initial_state
        Forwarded to ``simulate``. ``base_b_extra_fn`` is added on
        top of the swept sine injection, so closed-loop converters
        can be measured while the controller is still injecting its
        own forcing.
    **simulate_kwargs
        Forwarded to :func:`simulate`.

    Returns
    -------
    FraResult

    Notes
    -----
    For a closed-loop converter Bode this is the natural tool. For
    open-loop linearised analysis prefer :func:`run_ac_sweep` —
    it's a single MNA solve and orders of magnitude faster.
    """
    import pulsim as _v2  # late import (circular)

    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.size == 0:
        raise ValueError("run_fra: frequencies cannot be empty")
    if injection_amplitude <= 0:
        raise ValueError("run_fra: injection_amplitude must be > 0")

    if measurement_labels is None:
        measurement_labels = [f"meas_{i}" for i in measurements]
    else:
        measurement_labels = [str(s) for s in measurement_labels]
        if len(measurement_labels) != len(measurements):
            raise ValueError(
                "measurement_labels length must equal measurements")

    state_size = builder.pool.state_size(builder.graph)
    for m in measurements:
        if not (0 <= int(m) < state_size):
            raise ValueError(
                f"measurement index {m} out of range "
                f"[0, {state_size})")
    if not (0 <= injection_state_idx < state_size):
        raise ValueError(
            f"injection_state_idx {injection_state_idx} out of "
            f"range [0, {state_size})")

    # Auto-resolve dt.
    f_max = float(freqs.max())
    dt_eff = float(dt) if dt is not None else \
        1.0 / (f_max * samples_per_period)

    n_meas = len(measurements)
    H = np.empty((n_meas, freqs.size), dtype=np.complex128)

    for i_f, f in enumerate(freqs):
        T = 1.0 / f
        t_settle = periods_settle * T
        t_record = periods_measure * T
        t_total = t_settle + t_record

        omega = 2.0 * np.pi * f
        amp = injection_amplitude

        # b_extra adds the perturbation. If user supplied a base,
        # compose them.
        if base_b_extra_fn is None:
            def b_extra_plain(t, _amp=amp, _omega=omega,
                                _idx=injection_state_idx,
                                _sz=state_size):
                out = [0.0] * _sz
                out[_idx] = _amp * np.sin(_omega * t)
                return out
            b_extra = b_extra_plain
        else:
            def b_extra_composed(t, _amp=amp, _omega=omega,
                                    _idx=injection_state_idx,
                                    _base=base_b_extra_fn):
                base = list(_base(t))
                base[_idx] = base[_idx] + _amp * np.sin(_omega * t)
                return base
            b_extra = b_extra_composed

        # Capture only the measurement window via step_observer.
        rec_t: List[float] = []
        rec_inj: List[float] = []
        rec_meas: List[List[float]] = [[] for _ in measurements]

        def obs(t, x, _t0=t_settle, _amp=amp, _omega=omega,
                  _idxs=list(measurements)):
            if t < _t0:
                return
            rec_t.append(float(t))
            rec_inj.append(float(_amp * np.sin(_omega * t)))
            for j, idx in enumerate(_idxs):
                rec_meas[j].append(float(x[idx]))
            if step_observer is not None:
                step_observer(t, x)

        _v2.simulate(builder,
                       t_end=t_total, dt=dt_eff,
                       switch_fn=switch_fn,
                       step_observer=obs,
                       b_extra_fn=b_extra,
                       initial_state=(
                           initial_state.tolist()
                           if initial_state is not None else None),
                       **simulate_kwargs)

        if not rec_t:
            raise RuntimeError(
                f"run_fra: no samples captured for f={f} Hz — check "
                f"t_settle vs t_end")

        inj_arr = np.asarray(rec_inj, dtype=np.float64)
        # DFT of injection at f. Phase is sin → ∠ = -π/2.
        Inj = _dft_bin(inj_arr, dt_eff, f)
        if abs(Inj) < 1e-30:
            raise RuntimeError(
                f"run_fra: injection DFT bin is zero at f={f} Hz")
        for j in range(n_meas):
            m_arr = np.asarray(rec_meas[j], dtype=np.float64)
            M = _dft_bin(m_arr, dt_eff, f)
            H[j, i_f] = M / Inj

    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-30))
    phase_deg = np.degrees(np.angle(H))

    return FraResult(
        frequencies=freqs,
        magnitude_db=magnitude_db,
        phase_deg=phase_deg,
        H=H,
        measurement_labels=measurement_labels,
    )
