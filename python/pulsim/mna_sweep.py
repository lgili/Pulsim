"""Pulsim v2 — fast frequency sweep via impulse response + FFT.

The swept-sine `run_ac_sweep` is robust but expensive: each test
frequency takes its own multi-cycle transient. For LINEAR (or
linearised-at-operating-point) plants, the same Bode can be obtained
in O(1) transient by exciting the plant with an *impulse* and FFTing
the result. That's mathematically equivalent to:

  ẋ = A·x + B·u(t),   y = C·x
  ↓
  H(jω) = C·(jωI − A)⁻¹·B

which is the MNA-linearisation that `core/include/pulsim/ac/
mna_sweep.hpp` (planned) would expose at the C++ level. This Python
implementation reaches the same numerical result by sampling the
plant's impulse response and applying the discrete Fourier
transform — no kernel rebuild needed.

Limitations
-----------

* The plant is treated as LTI around the DC operating point. For
  switching converters, average the duty cycle (or hold switches at
  their on-state) before calling this function — running with a
  time-varying duty leaks switching ripple into the FFT.
* The frequency grid is set by the FFT: f_grid = k · fs / N for
  k = 0 .. N/2, so the maximum resolved frequency is fs/2 and the
  minimum is fs/N (rule of thumb: simulate for `n_cycles_lowest_f`
  cycles of the lowest-frequency component you care about).
* For non-LTI sources (PWM, switched diodes), the impulse response
  also includes the switching harmonics — filter the relevant
  frequency band when interpreting the result.

Typical usage
-------------

    builder = build_rc_filter()
    res = p.run_mna_sweep(builder,
                             v_in_branch_id=...,
                             output_idx=builder.node_id_of("vout"),
                             dt=1e-6, t_end=20e-3,
                             freqs=np.logspace(1, 4, 40))
    p.plot_bode(res, title="RC filter — MNA sweep")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


__all__ = [
    "MnaSweepResult",
    "run_mna_sweep",
]


@dataclass
class MnaSweepResult:
    """Outcome of an impulse-FFT frequency sweep.

    Mirrors `AcSweepResult` for plotting compatibility.
    """
    freqs: np.ndarray
    H: np.ndarray
    mag_dB: np.ndarray
    phase_deg: np.ndarray
    # Raw impulse response for debugging.
    times: np.ndarray
    y_impulse: np.ndarray


def _wrap_phase_deg(deg: np.ndarray) -> np.ndarray:
    """Wrap phase to (-180, 180]."""
    return np.mod(deg + 180.0, 360.0) - 180.0


def run_mna_sweep(builder,
                    *,
                    output_idx: int,
                    dt: float | None = None,
                    t_end: float | None = None,
                    freqs: Optional[Sequence[float]] = None,
                    v_in_branch_id: Optional[int] = None,
                    impulse_amplitude: float = 1.0,
                    start_from_dc_op: bool = True,
                    switch_fn=None,
                    enable_nonlinear_refresh: Optional[bool] = None,
                    use_kernel: bool = True,
                    verbose: bool = False,
                    ) -> MnaSweepResult:
    """Fast frequency sweep via impulse response + FFT.

    Parameters
    ----------
    builder
        A populated CircuitBuilder.
    output_idx
        State-vector index of the output node (target of H).
    dt
        Solver timestep. Determines the Nyquist frequency: f_max =
        1/(2·dt). Must be ≤ 0.5 / max(freqs).
    t_end
        Total simulation horizon. Determines the frequency
        resolution: Δf = 1/t_end. Set ≥ 10/min(freqs) for clean
        low-frequency response.
    freqs
        If provided, the result H is interpolated onto this grid
        (linear interpolation of magnitude and unwrapped phase).
        If None, returns the raw FFT bin frequencies.
    v_in_branch_id
        Optional ID of the input voltage source that the impulse
        will excite. If None, the impulse is applied to the FIRST
        constraint row via b_extra_fn (this matches `run_ac_sweep`'s
        default `excite_fn`).
    impulse_amplitude
        Strength of the unit impulse. Larger amplitudes improve SNR
        but risk leaving the linear regime around x_op.
    start_from_dc_op
        Start the simulation from the DC operating point.
    switch_fn
        If provided, passed through to `simulate(...)`. Useful for
        SMPS plants where the duty cycle is held at the average
        operating point.

    Returns
    -------
    MnaSweepResult
        Bode pair + raw impulse response.
    """
    # Lazy-import to avoid circular dependency at module load time.
    import pulsim as _v2

    # ---------------------------------------------------------------
    # Fast path: direct C++ kernel MNA sweep.
    # ---------------------------------------------------------------
    if use_kernel and freqs is not None and v_in_branch_id is not None:
        try:
            from . import _pulsim as _k  # type: ignore[import-not-found]
            mask = _k.SwitchStateMask(builder.graph.num_switches)
            src_idx = builder.pool.branch_var_id_for_source(
                v_in_branch_id, builder.graph)
            freqs_arr = np.asarray(freqs, dtype=float)
            res = _k.run_mna_sweep_kernel(
                builder.graph, builder.pool, mask,
                list(map(float, freqs_arr)),
                int(src_idx), int(output_idx))
            H = np.asarray(res.H, dtype=complex)
            mag_dB = 20.0 * np.log10(np.maximum(np.abs(H), 1e-300))
            phase_deg = _wrap_phase_deg(np.degrees(np.angle(H)))
            if verbose:
                print(f"  run_mna_sweep [kernel]: {len(freqs_arr)} "
                       f"frequencies via Eigen complex SparseLU")
            return MnaSweepResult(
                freqs=freqs_arr, H=H,
                mag_dB=mag_dB, phase_deg=phase_deg,
                times=np.array([]), y_impulse=np.array([]),
            )
        except Exception as exc:
            if verbose:
                print(f"  run_mna_sweep: kernel path failed "
                       f"({type(exc).__name__}), falling back to "
                       f"impulse-FFT")

    # ---------------------------------------------------------------
    # Fallback: impulse-FFT (LTI-equivalent via time-domain).
    # ---------------------------------------------------------------
    if dt is None or t_end is None:
        raise ValueError(
            "run_mna_sweep: kernel path was disabled or unavailable, "
            "but `dt` and `t_end` weren't provided for the impulse-"
            "FFT fallback. Pass them explicitly.")

    n_samples = int(round(t_end / dt))
    if n_samples < 16:
        raise ValueError(
            f"t_end/dt = {n_samples} too few samples for a useful "
            f"FFT; need at least 16.")

    state_size = builder.pool.state_size(builder.graph)
    # Build the impulse b_extra_fn — a few samples of strong perturbation,
    # then zero. The integrated charge ≈ impulse_amplitude (units of
    # V·s for voltage perturbation, A·s for current). We spread the
    # impulse over `n_pulse` timesteps for two reasons:
    #   1. avoids missing it due to off-by-one at the boundary
    #      (simulate() evaluates b_extra at t = t_new, so the t=0
    #      sample is never hit by a single-step impulse);
    #   2. low-pass-filters the impulse so its spectrum is flat
    #      out to ≈ 1/(2·n_pulse·dt) (still WAY above any frequency
    #      of practical interest).
    n_pulse = 4
    impulse_width = n_pulse * dt
    impulse_height = impulse_amplitude / impulse_width

    # If user gave a specific input source branch, perturb its
    # constraint row; otherwise, just perturb state-vector entry 0.
    if v_in_branch_id is not None:
        src_id = builder.pool.branch_var_id_for_source(
            v_in_branch_id, builder.graph)
    else:
        src_id = 0

    def b_extra_fn(t):
        out = [0.0] * state_size
        if t <= impulse_width + 0.5 * dt:
            out[src_id] = -impulse_height
        return out

    res = _v2.simulate(
        builder,
        t_end=t_end, dt=dt, t_start=0.0,
        b_extra_fn=b_extra_fn,
        switch_fn=switch_fn,
        start_from_dc_op=start_from_dc_op,
        enable_nonlinear_refresh=enable_nonlinear_refresh,
        max_event_iterations=8,
    )

    times = np.asarray(res.times)
    states = np.asarray(res.states)
    y_raw = states[:, output_idx]
    # Subtract DC offset (the impulse only EXCITES around the OP).
    y_op = states[0, output_idx]
    y_impulse = y_raw - y_op

    # FFT — single-sided spectrum.
    Y = np.fft.rfft(y_impulse) * dt        # rfft normalisation
    f_bins = np.fft.rfftfreq(n_samples, d=dt)
    # H(f) = FFT(y) / FFT(δ) — but our excitation isn't a true delta,
    # it's a rectangular pulse of width W = impulse_width starting at
    # t = 0. Its FFT is `impulse_amplitude · sinc(π·f·W) ·
    # exp(-j·π·f·W)`. Divide it out to recover the LTI transfer
    # function. This correction makes the result flat all the way to
    # the FFT's Nyquist limit, instead of rolling off at ~1/(2·W).
    sinc_arg = np.pi * f_bins * impulse_width
    # Use np.sinc which is normalized: np.sinc(x) = sin(πx)/(πx).
    sinc_term = np.sinc(f_bins * impulse_width)
    delay_term = np.exp(-1j * sinc_arg)
    # Guard against divide-by-zero at the sinc zeros (f = k/W).
    pulse_spectrum = impulse_amplitude * sinc_term * delay_term
    safe_mask = np.abs(pulse_spectrum) > 1e-9
    H_raw = np.zeros_like(Y)
    H_raw[safe_mask] = Y[safe_mask] / pulse_spectrum[safe_mask]
    # Replace nans/infs with the nearest safe neighbour (just for
    # plot/interpolation robustness — won't be hit if user picks a
    # reasonable impulse_width).
    if not np.all(safe_mask):
        good_idx = np.where(safe_mask)[0]
        if len(good_idx) > 0:
            last_good = good_idx[-1]
            H_raw[~safe_mask] = H_raw[last_good]

    if freqs is not None:
        freqs_arr = np.asarray(freqs, dtype=float)
        # Linear interpolation in log-space → on-grid magnitude and
        # unwrapped phase.
        mag = np.abs(H_raw)
        phase = np.unwrap(np.angle(H_raw))
        # Avoid f=0 in log interpolation.
        mask = f_bins > 0
        mag_interp = np.interp(np.log(freqs_arr),
                                 np.log(f_bins[mask]),
                                 mag[mask])
        phase_interp = np.interp(np.log(freqs_arr),
                                   np.log(f_bins[mask]),
                                   phase[mask])
        H = mag_interp * np.exp(1j * phase_interp)
        freqs_out = freqs_arr
    else:
        H = H_raw
        freqs_out = f_bins

    mag_dB = 20.0 * np.log10(np.maximum(np.abs(H), 1e-300))
    phase_deg = _wrap_phase_deg(np.degrees(np.angle(H)))

    if verbose:
        print(f"  run_mna_sweep: n_samples={n_samples}, "
               f"f_max={1/(2*dt):.1f} Hz, "
               f"Δf={1/t_end:.3f} Hz, "
               f"impulse_amplitude={impulse_amplitude}")

    return MnaSweepResult(
        freqs=freqs_out, H=H,
        mag_dB=mag_dB, phase_deg=phase_deg,
        times=times, y_impulse=y_impulse,
    )
