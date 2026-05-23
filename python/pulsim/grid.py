"""Pulsim — three-phase grid helpers (Phase C.2).

Building blocks for grid-connected circuits: rectifiers, motor
drives, PV inverters, grid-tie converters, harmonic analyzers, etc.

The core abstraction is a balanced 3-phase voltage source connected
to three phase nodes (`a`, `b`, `c`) with a shared neutral. Each
phase is an instance of v2's `add_sine_voltage_source` with the
correct phase offset (0°, −120°, +120° for positive sequence).

Sequence-component math is the standard symmetrical-components
decomposition:
   v_0 = (v_a + v_b + v_c) / 3                       (zero sequence)
   v_p = (v_a + α·v_b + α²·v_c) / 3                  (positive)
   v_n = (v_a + α²·v_b + α·v_c) / 3                  (negative)
where α = e^{j·2π/3}. Each sequence is COMPLEX-valued — its real /
imaginary parts encode the magnitude and phase of that sequence
component.

Typical usage — 3-φ diode rectifier with grid:

    b = p.CircuitBuilder()
    p.add_three_phase_grid(b, V_rms=230.0, f_Hz=50.0,
                              phase_nodes=("a", "b", "c"),
                              neutral_node="n")
    p.add_three_phase_line_impedance(b, R=0.5, L=2e-3,
                                          src=("a", "b", "c"),
                                          dst=("a1", "b1", "c1"))
    # … add a rectifier between (a1, b1, c1) and (vdc+, vdc-) …
    res = p.simulate(b, t_end=100e-3, dt=1e-5)
    times, ia, ib, ic = ...   # extract from result
    v0, vp, vn = p.sequence_components(va, vb, vc)
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


__all__ = [
    "add_three_phase_grid",
    "add_three_phase_line_impedance",
    "sequence_components",
    "instantaneous_power_3phase",
    "voltage_unbalance_factor",
]


# =============================================================================
# Builder helpers
# =============================================================================

def add_three_phase_grid(builder,
                            *,
                            V_rms: float,
                            f_Hz: float,
                            phase_nodes: Sequence[str] = ("a", "b", "c"),
                            neutral_node: str = "n",
                            phase_sequence: str = "abc",
                            name_prefix: str = "Vgrid",
                            ) -> None:
    """Add a balanced 3-phase sinusoidal voltage source.

    Three `add_sine_voltage_source` calls, one per phase, all
    referenced to the same neutral. Phase amplitude is
    ``V_amp = V_rms · √2`` (line-to-neutral peak).

    Parameters
    ----------
    builder
        Target CircuitBuilder.
    V_rms
        Line-to-neutral RMS voltage in volts (e.g. 230 V for EU mains).
    f_Hz
        Grid frequency (e.g. 50 or 60 Hz).
    phase_nodes
        Names of the three phase output nodes. Default ("a", "b", "c").
    neutral_node
        Name of the neutral / star-point node. Default "n".
    phase_sequence
        ``"abc"`` (positive sequence, default) or ``"acb"``
        (negative sequence — useful for testing motor-drive
        sequence detection).
    name_prefix
        Used to name the three sources: ``{prefix}_a``,
        ``{prefix}_b``, ``{prefix}_c``.
    """
    if len(phase_nodes) != 3:
        raise ValueError("phase_nodes must have exactly 3 entries")
    if phase_sequence not in ("abc", "acb"):
        raise ValueError("phase_sequence must be 'abc' or 'acb'")

    V_amp = float(V_rms) * math.sqrt(2.0)

    # Phase offsets in radians (positive sequence by default).
    if phase_sequence == "abc":
        offsets_deg = (0.0, -120.0, +120.0)
    else:  # acb
        offsets_deg = (0.0, +120.0, -120.0)
    offsets_rad = tuple(math.radians(d) for d in offsets_deg)

    for k, (node, off) in enumerate(zip(phase_nodes, offsets_rad)):
        builder.add_sine_voltage_source(
            f"{name_prefix}_{('a','b','c')[k]}",
            node, neutral_node,
            v_dc=0.0, v_amplitude=V_amp,
            frequency=float(f_Hz), phase=off,
        )


def add_three_phase_line_impedance(builder,
                                       *,
                                       R: float,
                                       L: float,
                                       src: Sequence[str],
                                       dst: Sequence[str],
                                       name_prefix: str = "Zline",
                                       ) -> None:
    """Add per-phase line impedance (R in series with L) between two
    sets of nodes.

    Models distribution-line impedance, transformer leakage, or
    cable inductance. Same R, L on each phase — for unbalanced
    studies, instantiate three single-phase pairs manually.

    Parameters
    ----------
    R, L
        Per-phase series resistance (Ω) and inductance (H).
    src
        3-element sequence of source-side node names.
    dst
        3-element sequence of destination-side node names.
    """
    if len(src) != 3 or len(dst) != 3:
        raise ValueError("src and dst must have exactly 3 entries")
    for k in range(3):
        mid = f"{name_prefix}_mid_{('a','b','c')[k]}"
        builder.add_resistor(
            f"{name_prefix}_R_{('a','b','c')[k]}",
            src[k], mid, float(R))
        builder.add_inductor(
            f"{name_prefix}_L_{('a','b','c')[k]}",
            mid, dst[k], float(L))


# =============================================================================
# Sequence-component math (post-processing on arrays)
# =============================================================================

def sequence_components(v_a, v_b, v_c, *,
                            t=None, frequency=None,
                            ) -> Tuple[complex, complex, complex]:
    """Symmetrical-components decomposition (Fortescue) at a single
    grid frequency.

    Two operating modes:

    * **Phasor mode** (default, when `t` and `frequency` are given):
      single-frequency DFT extracts the complex phasor at
      `frequency` for each phase, then applies the standard
      Fortescue matrix. Returns three COMPLEX SCALARS — the
      0/positive/negative-sequence phasors at the test frequency.
      This is what you want for utility-grid analysis (VUF,
      sequence-component IDs, etc.).

    * **Static-input mode** (when v_a/v_b/v_c are already complex
      phasors): just applies Fortescue. Each input must be a single
      complex scalar OR a 3-element array of complex phasors.

    Parameters
    ----------
    v_a, v_b, v_c
        Per-phase instantaneous (real-valued) signal traces, OR
        precomputed complex phasors (one per phase).
    t
        Time vector (s), same length as the v_x arrays. Required for
        the DFT path.
    frequency
        Test frequency (Hz). Required for the DFT path.

    Returns
    -------
    (V_0, V_p, V_n)
        Complex phasors. ``|V_p|`` gives the positive-sequence
        amplitude; ``np.angle(V_p)`` gives its phase. For a balanced
        positive-sequence input ``|V_p| ≈ amplitude`` and
        ``|V_n|, |V_0| ≈ 0``.

    Notes
    -----
    Fortescue matrix with ``α = exp(j·2π/3)``:

        ⎡V_0⎤   1   ⎡1  1   1 ⎤   ⎡V_a⎤
        ⎢V_p⎥ = ─ · ⎢1  α   α²⎥ · ⎢V_b⎥
        ⎣V_n⎦   3   ⎣1  α²  α ⎦   ⎣V_c⎦
    """
    # Phasor extraction path.
    if t is not None and frequency is not None:
        Va = _extract_phasor(t, v_a, frequency)
        Vb = _extract_phasor(t, v_b, frequency)
        Vc = _extract_phasor(t, v_c, frequency)
    else:
        Va = complex(np.asarray(v_a).item())
        Vb = complex(np.asarray(v_b).item())
        Vc = complex(np.asarray(v_c).item())

    alpha = np.exp(1j * 2.0 * np.pi / 3.0)
    alpha_sq = alpha * alpha
    V0 = (Va + Vb + Vc) / 3.0
    Vp = (Va + alpha * Vb + alpha_sq * Vc) / 3.0
    Vn = (Va + alpha_sq * Vb + alpha * Vc) / 3.0
    return V0, Vp, Vn


def _extract_phasor(t, x, frequency: float) -> complex:
    """Single-frequency DFT — returns the complex amplitude of x(t)
    at the given frequency."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    T = float(t[-1] - t[0])
    if T <= 0:
        return 0.0 + 0.0j
    omega = 2.0 * np.pi * float(frequency)
    integrand = x * np.exp(-1j * omega * t)
    integral = np.trapezoid(integrand, x=t)
    # Single-sided amplitude convention: V_amp · e^{jφ}.
    return complex(2.0 * integral / T)


def instantaneous_power_3phase(v_a, v_b, v_c,
                                   i_a, i_b, i_c) -> np.ndarray:
    """Instantaneous three-phase active power: P(t) = Σ v·i.

    Returns the total power delivered by the 3-phase source at each
    time sample (sum of per-phase products).
    """
    return (np.asarray(v_a) * np.asarray(i_a) +
              np.asarray(v_b) * np.asarray(i_b) +
              np.asarray(v_c) * np.asarray(i_c))


def voltage_unbalance_factor(v_a, v_b, v_c, *,
                                 t, frequency: float) -> float:
    """IEEE 1159 voltage unbalance factor:
       VUF = |V_neg| / |V_pos|  (in %).

    `t` and `frequency` are required so the phasor extraction picks
    out the correct grid-frequency component (rejecting harmonics,
    DC offsets, switching ripple, …). A VUF above 2 % is considered
    unacceptable on most utility grids.
    """
    _, Vp, Vn = sequence_components(v_a, v_b, v_c,
                                       t=t, frequency=frequency)
    return float(abs(Vn) / max(abs(Vp), 1e-30))
