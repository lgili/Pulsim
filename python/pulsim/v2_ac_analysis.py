"""Pulsim v2 — small-signal AC analysis via swept-sine excitation.

Pure-Python implementation that reuses the existing transient
solver: at each test frequency f, run the plant for several
cycles with a small sine perturbation, then extract the
complex transfer function H(jω) via single-frequency DFT.

This is the standard "frequency response analyzer" technique
used by commercial network analyzers (HP / Venable / Bode 100).
Advantages over MNA-matrix linearization:

  * Works for ANY plant the v2 kernel can simulate, including
    nonlinear devices (smooth-blend diode, SH1 MOSFET, IGBT,
    saturable inductor) — the DFT picks up the linear response
    around the DC operating point automatically.
  * No new kernel code needed.
  * Naturally averages over PWM ripple if you simulate for
    enough cycles.

Disadvantage: slower than direct linearisation — each test
frequency requires its own transient. For a 50-point Bode
sweep on a typical SMPS, expect ~30 seconds of wall clock.

Typical usage:

    import pulsim.v2 as p
    import numpy as np

    builder = build_my_circuit()
    freqs = np.logspace(1, 5, 40)   # 10 Hz → 100 kHz

    def excite(t, eps, freq):
        # Inject `eps · sin(2π·f·t)` on the V_in source's
        # constraint row. Returns a b_extra vector.
        out = [0.0] * builder.pool.state_size(builder.graph)
        src_id = builder.pool.branch_var_id_for_source(
            v_in_branch_id, builder.graph)
        out[src_id] = -eps * math.sin(2 * math.pi * freq * t)
        return out

    result = p.run_ac_sweep(
        builder, freqs=freqs,
        excite_fn=excite,
        output_idx=builder.node_id_of("vout"),
        dt=1e-7, perturbation_amplitude=0.1,
        cycles_per_point=20,
        settling_cycles=10,
    )
    # result.mag_dB, result.phase_deg ready to plot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import v2 as p


__all__ = [
    "AcSweepResult",
    "run_ac_sweep",
    "extract_phasor",
    "plot_bode",
]


@dataclass
class AcSweepResult:
    """Outcome of one frequency sweep.

    Parallel arrays of length `len(freqs)`. `H` is complex
    transfer function; `mag_dB` / `phase_deg` are the
    extracted Bode pair.
    """

    freqs: np.ndarray
    H: np.ndarray                  # complex, shape (N,)
    mag_dB: np.ndarray             # 20·log10(|H|)
    phase_deg: np.ndarray          # wrapped to (-180, 180]
    # Optional metadata for downstream verification.
    cycles_per_point: int = 0
    settling_cycles: int = 0
    perturbation_amplitude: float = 0.0


def extract_phasor(
    times: np.ndarray,
    samples: np.ndarray,
    frequency: float,
) -> complex:
    """Single-frequency DFT — returns the complex Fourier
    coefficient of `samples(t)` at the given frequency.

    Sliding the cosine and sine references over the sampled
    window gives `(A_c, A_s)` such that
        samples(t) ≈ DC + A_c·cos(2π·f·t) + A_s·sin(2π·f·t)
    Then `phasor = A_c + j·A_s` (or rather A_c - j·A_s — see
    the formula below — depending on convention).

    Convention: signal = A·cos(ωt + φ) ⇒ phasor = A·exp(jφ).
    """
    omega = 2 * math.pi * frequency
    # Numerical integration (rectangle) of x · exp(-jωt) /
    # T_window gives the phasor amplitude/2 (single-sided).
    T = times[-1] - times[0]
    if T <= 0:
        return 0.0 + 0.0j
    integrand = samples * np.exp(-1j * omega * times)
    # Trapezoidal integral, divided by T/2 (one-sided amplitude).
    integral = np.trapezoid(integrand, x=times)
    return 2.0 * integral / T


def _wrap_phase_deg(deg: np.ndarray) -> np.ndarray:
    """Wrap phase to (-180, 180]."""
    out = np.mod(deg + 180.0, 360.0) - 180.0
    return out


def run_ac_sweep(
    builder,
    *,
    freqs: Sequence[float],
    excite_fn: Callable[[float, float, float], list],
    output_idx: int,
    dt: float | None = None,
    dt_fn: Callable[[float], float] | None = None,
    samples_per_cycle: int = 200,
    dt_max: float = 1e-4,
    perturbation_amplitude: float = 0.01,
    cycles_per_point: int = 20,
    settling_cycles: int = 10,
    switch_fn=None,
    switch_fn_factory: Callable[[float, float], Callable] | None = None,
    extra_b_extra_fn: Callable[[float], list] | None = None,
    enable_nonlinear_refresh: bool | None = None,
    start_from_dc_op: bool = True,
    verbose: bool = True,
) -> AcSweepResult:
    """Sweep frequency and build an AC frequency response.

    For each `f` in `freqs`:
      1. Simulate `settling_cycles + cycles_per_point` cycles
         with a perturbation `excite_fn(t, eps, f)` overlaid.
      2. Drop the first `settling_cycles` (transient).
      3. DFT the remaining samples at frequency `f` to get the
         complex amplitude of the OUTPUT and the INPUT
         perturbation.
      4. H(jω) = Y/U.

    Parameters
    ----------
    builder
        Populated CircuitBuilder.
    freqs
        Sequence of test frequencies, Hz.
    excite_fn
        Signature: `(t, eps, freq) -> list[float]`. Returns the
        b_extra vector to add at time t. `eps` is the amplitude
        passed in; `freq` is the current test frequency.
        Typically returns `[0]*N` with one entry set to a sine
        perturbation on the input source's branch-current row.
    output_idx
        State-vector index of the node whose Bode response is
        being measured (typically the output node).
    dt
        Solver step size.
    perturbation_amplitude
        ε passed to `excite_fn`. Should be small (~1 % of
        nominal source value) to stay in the linear regime.
    cycles_per_point
        How many full sine cycles to integrate for the DFT.
        20 is a good default; the DFT noise floor is ~1/N.
    settling_cycles
        Cycles to discard at the start of each frequency
        (lets the plant reach steady state).
    switch_fn
        Optional. Passed through to `simulate(...)`.
    extra_b_extra_fn
        Optional extra `b_extra(t)` overlay (e.g. a DC bias).
        Summed with the AC perturbation each step.
    enable_nonlinear_refresh
        Passed through to `simulate`. Auto-detected if None.
    start_from_dc_op
        Seed each per-frequency simulation from the DC
        operating point so the plant starts in steady state.
        Default True.

    Returns
    -------
    AcSweepResult
    """
    freqs = np.asarray(freqs, dtype=float)
    H = np.zeros(len(freqs), dtype=complex)

    for i, f in enumerate(freqs):
        T_cycle = 1.0 / f
        t_end = (settling_cycles + cycles_per_point) * T_cycle

        # Pick the per-frequency dt. Priority order:
        #   1. Caller-supplied dt_fn(f)
        #   2. Caller-supplied fixed dt
        #   3. Auto: T_cycle / samples_per_cycle, capped at dt_max
        if dt_fn is not None:
            dt_local = dt_fn(f)
        elif dt is not None:
            dt_local = dt
        else:
            dt_local = min(T_cycle / samples_per_cycle, dt_max)

        # Build the per-step b_extra: AC perturbation + any
        # user-supplied DC overlay.
        def b_extra_fn(t, f_local=f):
            ac = excite_fn(t, perturbation_amplitude, f_local)
            if extra_b_extra_fn is None:
                return ac
            extra = extra_b_extra_fn(t)
            # Element-wise sum.
            return [a + b for a, b in zip(ac, extra)]

        # Pick switch_fn per-frequency if a factory was supplied.
        sw_fn = (switch_fn_factory(perturbation_amplitude, f)
                  if switch_fn_factory is not None
                  else switch_fn)

        res = p.simulate(
            builder,
            t_end=t_end, dt=dt_local,
            switch_fn=sw_fn,
            b_extra_fn=b_extra_fn,
            start_from_dc_op=start_from_dc_op,
            enable_nonlinear_refresh=enable_nonlinear_refresh,
        )

        times = np.asarray(res.times)
        states = np.array([s[output_idx] for s in res.states])

        # Drop settling region.
        k_settle = int(settling_cycles * T_cycle / dt_local)
        t_w = times[k_settle:]
        y_w = states[k_settle:]

        # Subtract DC mean before DFT (cleaner phasor).
        y_w_ac = y_w - y_w.mean()
        Y = extract_phasor(t_w, y_w_ac, f)
        # The input phasor is just (perturbation_amplitude) · 1
        # by construction (sin → −j amplitude in the exp(-jωt)
        # convention, so divide and rotate accordingly).
        # Excitation: ε·sin(ωt) = (ε/(2j)) · (exp(jωt) − exp(-jωt))
        # In our 2·integral/T convention, the phasor of sin(ωt)
        # is -j·ε. So H = Y / (-j·ε).
        U = -1j * perturbation_amplitude
        H[i] = Y / U

        if verbose:
            mag = 20 * math.log10(abs(H[i])) if abs(H[i]) > 0 else float("-inf")
            print(f"  f = {f:>10.2f} Hz  |  "
                  f"|H| = {abs(H[i]):>9.4f}  ({mag:>+6.2f} dB)  "
                  f"∠H = {math.degrees(np.angle(H[i])):>+7.2f}°")

    mag_dB = 20 * np.log10(np.abs(H))
    phase_deg = _wrap_phase_deg(np.degrees(np.angle(H)))

    return AcSweepResult(
        freqs=freqs, H=H,
        mag_dB=mag_dB, phase_deg=phase_deg,
        cycles_per_point=cycles_per_point,
        settling_cycles=settling_cycles,
        perturbation_amplitude=perturbation_amplitude,
    )


def plot_bode(result: AcSweepResult, *, title: str = "Bode",
               compare_analytical=None, ax=None):
    """Render a Bode plot. `compare_analytical(freqs) → complex`
    overlays an analytical reference if supplied."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, (ax_mag, ax_phase) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True)
    else:
        ax_mag, ax_phase = ax

    ax_mag.semilogx(result.freqs, result.mag_dB,
                     label="measured", lw=1.4)
    ax_phase.semilogx(result.freqs, result.phase_deg,
                       label="measured", lw=1.4)

    if compare_analytical is not None:
        H_ref = compare_analytical(result.freqs)
        mag_ref = 20 * np.log10(np.abs(H_ref))
        ph_ref = _wrap_phase_deg(np.degrees(np.angle(H_ref)))
        ax_mag.semilogx(result.freqs, mag_ref,
                         ls="--", lw=0.8, label="analytical")
        ax_phase.semilogx(result.freqs, ph_ref,
                            ls="--", lw=0.8, label="analytical")

    ax_mag.set_ylabel("|H| [dB]"); ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.set_title(title)
    ax_mag.legend(loc="best")

    ax_phase.set_xlabel("frequency [Hz]")
    ax_phase.set_ylabel("∠H [deg]")
    ax_phase.grid(True, which="both", alpha=0.3)
    ax_phase.legend(loc="best")
    return ax_mag, ax_phase
