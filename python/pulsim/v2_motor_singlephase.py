"""Pulsim v2 — single-phase induction motor (steady-state).

Closes Phase-2 gap #6 from the v1→v2 retirement punch list. Models
the **permanent-split capacitor (PSC)** single-phase induction
motor — the topology in fridge compressors, room AC indoor units,
ceiling fans, and most domestic appliances.

What this module is
-------------------
A revolving-field (cross-field) steady-state analyser. Given
nameplate-style equivalent-circuit parameters + supply voltage +
load, returns the operating point: slip, mechanical speed, torque,
stator currents (main + aux), input power, output power,
efficiency, and power factor. Plus a torque-speed sweep for
characteristic curves.

Why steady-state only
---------------------
For PSC motors the steady-state operating point is what designers
spend 80 % of their time on:
* Cap sizing (which C_run gives best PF at design point?).
* Torque margin against load at rated voltage.
* Locked-rotor torque (s = 1).
* Efficiency map across the operating range.

Dynamic (start-up transient) simulation requires the full dq model
with cross-coupling and is genuinely complex — left for a future
addition. For most appliance applications the running-state
operating point captures the relevant engineering insight.

Theory recap
------------
For a single-phase machine the pulsating MMF decomposes into two
counter-rotating fields of equal amplitude (Forward and Backward).
Each behaves like a balanced 3-phase machine at half the voltage.

For PSC with main + aux + series capacitor + turns ratio
``a = N_aux / N_main``:

* Main and aux currents :math:`I_m, I_a` are complex.
* Symmetrical components transform them into forward (+) and
  backward (−) sequence currents:
  :math:`I_+ = (I_m + j\\,a\\,I_a)/2`,
  :math:`I_- = (I_m - j\\,a\\,I_a)/2`.
* Forward sequence sees impedance ``Z_+(s)`` with slip ``s``.
* Backward sequence sees ``Z_-(s)`` with slip ``2 - s``.

Slip equation: :math:`s = (\\omega_s - \\omega_m) / \\omega_s` with
:math:`\\omega_s = 2 \\pi f / (p/2)` (pole pairs).

Air-gap power per field:
:math:`P_{g\\pm} = |I_\\pm|^2 \\cdot \\Re(Z_\\pm \\text{ rotor part})`.

Torque: :math:`T_{em} = (P_{g+} - P_{g-}) / \\omega_s`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


__all__ = [
    "SinglePhaseIMParams",
    "PSCOperatingPoint",
    "compute_psc_operating_point",
    "psc_torque_speed_curve",
    "reference_psc_motor",
]


# =============================================================================
# Parameter struct + reference values
# =============================================================================

@dataclass
class SinglePhaseIMParams:
    """Equivalent-circuit parameters of a single-phase PSC motor.

    All values referred to the main winding. Two flavours of
    parameters live here:

    * **Electrical** (per-winding R + X at supply frequency, plus
      magnetising X_m and rotor R_r, X_r referred to stator).
    * **Mechanical** (pole pairs, inertia, viscous friction).
    * **Topology** (turns ratio of aux to main, run capacitor in µF).

    Naming convention: ``_m`` for main winding, ``_a`` for auxiliary.

    Attributes
    ----------
    R_main, X_main
        Stator main-winding resistance (Ω) and reactance at
        nominal frequency (Ω).
    R_aux, X_aux
        Stator auxiliary-winding R + X. Often R_aux > R_main and
        X_aux > X_main because the aux has more turns / smaller
        gauge.
    R_rotor, X_rotor
        Rotor R + X referred to stator (main winding).
    X_m
        Magnetising reactance at supply frequency (Ω).
    turns_ratio
        ``N_aux / N_main`` (typically 1.0–1.5 for PSC).
    C_run_uF
        Run-capacitor in series with aux (µF). For starting torque
        you might add a larger start cap in parallel; this module
        models the running condition with C_run.
    pole_pairs
        Pole pairs (1 for a 2-pole motor, 2 for a 4-pole, …).
    J
        Rotor moment of inertia (kg·m²).
    B
        Viscous friction coefficient (N·m·s/rad).
    """
    # Electrical
    R_main: float
    X_main: float
    R_aux: float
    X_aux: float
    R_rotor: float
    X_rotor: float
    X_m: float
    # Topology
    turns_ratio: float = 1.0
    C_run_uF: float = 10.0
    # Mechanical
    pole_pairs: int = 2
    J: float = 1e-4
    B: float = 1e-4


# Reference parameter sets for common appliance motors. Sources:
# textbook examples / typical nameplate-to-circuit fits.
_REFERENCE_MOTORS = {
    "psc_1_4_hp_4pole": SinglePhaseIMParams(
        R_main=2.0, X_main=2.6,
        R_aux=7.0, X_aux=8.0,
        R_rotor=4.5, X_rotor=2.6,
        X_m=66.8,
        turns_ratio=1.18, C_run_uF=15.0,
        pole_pairs=2, J=2e-3, B=5e-4),
    "psc_1_2_hp_4pole": SinglePhaseIMParams(
        R_main=1.0, X_main=1.5,
        R_aux=4.5, X_aux=4.0,
        R_rotor=2.8, X_rotor=1.5,
        X_m=42.0,
        turns_ratio=1.25, C_run_uF=20.0,
        pole_pairs=2, J=5e-3, B=8e-4),
    "psc_compressor_120w": SinglePhaseIMParams(  # fridge compressor
        R_main=4.5, X_main=5.2,
        R_aux=12.0, X_aux=10.0,
        R_rotor=6.5, X_rotor=4.5,
        X_m=80.0,
        turns_ratio=1.4, C_run_uF=5.0,
        pole_pairs=1, J=1e-4, B=1e-4),
}


def reference_psc_motor(name: str) -> SinglePhaseIMParams:
    """Pre-tuned PSC-motor parameter sets for common sizes.

    Available: ``psc_1_4_hp_4pole``, ``psc_1_2_hp_4pole``,
    ``psc_compressor_120w``.
    """
    try:
        return _REFERENCE_MOTORS[name]
    except KeyError:
        avail = ", ".join(sorted(_REFERENCE_MOTORS))
        raise KeyError(
            f"unknown PSC motor reference {name!r}; available: {avail}")


# =============================================================================
# Operating point
# =============================================================================

@dataclass
class PSCOperatingPoint:
    """Output of :func:`compute_psc_operating_point`.

    All quantities at the supply frequency / voltage given.

    Attributes
    ----------
    slip
        ``s = (ω_s − ω_m) / ω_s`` (dimensionless).
    omega_mech_rad_s
        Mechanical angular velocity (rad/s).
    speed_rpm
        Mechanical speed (RPM).
    torque_em_Nm
        Electromagnetic torque produced (N·m). Positive in the
        direction of rotation.
    I_main, I_aux
        Stator complex currents (A, phasor in stator-main reference
        frame). Magnitudes give RMS values.
    P_in_W
        Electrical input power (W).
    P_mech_W
        Mechanical output power (W) = T_em · ω_mech minus friction.
    efficiency
        ``P_mech / P_in`` (ratio).
    power_factor
        ``cos(angle of total stator current vs voltage)``.
    Z_total_main
        Impedance seen at main winding (Ω, complex).
    Z_total_aux
        Impedance seen at aux winding including the run capacitor
        and main coupling (Ω, complex).
    """
    slip: float
    omega_mech_rad_s: float
    speed_rpm: float
    torque_em_Nm: float
    I_main: complex
    I_aux: complex
    P_in_W: float
    P_mech_W: float
    efficiency: float
    power_factor: float
    Z_total_main: complex = field(default=0.0 + 0j)
    Z_total_aux: complex = field(default=0.0 + 0j)


def _parallel(z1: complex, z2: complex) -> complex:
    """Parallel impedance of two complex numbers."""
    if z1 == 0 or z2 == 0:
        return 0 + 0j
    return (z1 * z2) / (z1 + z2)


def _field_impedances(params: SinglePhaseIMParams,
                          s: float) -> tuple:
    """Return (Z_forward, Z_backward) — the per-sequence impedances
    seen by the forward (+) and backward (−) revolving-field
    components for slip ``s``.

    Each sequence "sees" a balanced-3φ-like circuit at half the
    voltage. Conventionally: magnetising X_m/2 in parallel with
    (R_rotor / (2·s_eff)) + j(X_rotor / 2), where ``s_eff = s`` for
    forward and ``s_eff = 2 - s`` for backward.
    """
    # Forward (+)
    s_eff_plus = max(s, 1e-9)  # avoid /0 at sync speed
    Z_r_plus = (params.R_rotor / s_eff_plus) + 1j * params.X_rotor
    Z_plus = _parallel(1j * params.X_m, Z_r_plus)

    # Backward (−) — slip seen by the rotor is (2 − s)
    s_eff_minus = 2.0 - s
    Z_r_minus = (params.R_rotor / s_eff_minus) + 1j * params.X_rotor
    Z_minus = _parallel(1j * params.X_m, Z_r_minus)

    return Z_plus, Z_minus


def compute_psc_operating_point(params: SinglePhaseIMParams,
                                       *,
                                       V_supply: float,
                                       f_supply: float,
                                       slip: float | None = None,
                                       speed_rpm: float | None = None,
                                       ) -> PSCOperatingPoint:
    """Compute the steady-state operating point of a PSC motor.

    Either ``slip`` or ``speed_rpm`` must be given to specify
    the operating point.

    Parameters
    ----------
    params
        :class:`SinglePhaseIMParams`.
    V_supply
        RMS supply voltage (V).
    f_supply
        Supply frequency (Hz).
    slip
        Operating slip (dimensionless, in ``[0, 1]`` for motoring).
        ``s = 0`` is synchronous speed; ``s = 1`` is locked rotor.
    speed_rpm
        Alternative way to specify the operating point — mechanical
        speed in RPM. Mutually exclusive with ``slip``.

    Returns
    -------
    PSCOperatingPoint
    """
    if (slip is None) == (speed_rpm is None):
        raise ValueError(
            "compute_psc_operating_point: pass EXACTLY one of "
            "slip= or speed_rpm=")
    omega_e = 2.0 * np.pi * f_supply
    # Sync speed in mechanical rad/s.
    omega_s_mech = omega_e / params.pole_pairs
    if speed_rpm is not None:
        # Convert RPM to mech rad/s, then slip.
        omega_m = (float(speed_rpm) * 2.0 * np.pi) / 60.0
        s = 1.0 - omega_m / omega_s_mech
    else:
        assert slip is not None
        s = float(slip)
        omega_m = omega_s_mech * (1.0 - s)

    # Forward/backward field impedances.
    Z_plus, Z_minus = _field_impedances(params, s)

    # Stator total impedance for the main winding (without aux).
    Z_main_stator = params.R_main + 1j * params.X_main
    Z_aux_stator = params.R_aux + 1j * params.X_aux

    # Capacitor reactance (negative imaginary).
    omega_e_actual = 2.0 * np.pi * f_supply
    X_C = -1.0 / (omega_e_actual * params.C_run_uF * 1e-6)
    Z_cap = 1j * X_C  # X_C is negative → Z_cap is negative imag

    # Aux branch impedance: stator aux + capacitor.
    Z_aux_branch = Z_aux_stator + Z_cap

    # For PSC the symmetrical-component decomposition relates the
    # main and aux currents through the turns ratio ``a``. Set up
    # the linear 2×2 system in (I_m, I_a) and solve:
    #
    #   V_main = I_m·Z_main_stator + I_+·Z_+ + I_-·Z_-
    #   V_aux  = I_a·Z_aux_branch + j·a·(I_+·Z_+ − I_-·Z_-)
    #
    # with
    #   I_+ = (I_m + j·a·I_a) / 2
    #   I_- = (I_m − j·a·I_a) / 2
    #
    # Both windings are connected to the same supply V (V_main =
    # V_aux = V_supply in PSC). Solving yields a closed-form
    # expression for (I_m, I_a).

    a = params.turns_ratio
    V = float(V_supply) + 0j

    # Sequence-frame decomposition for a 2-phase machine with main
    # on the d-axis and aux on the q-axis 90° ahead. The convention
    # that gives FORWARD starting torque (positive aux MMF lagging
    # main by 90° in space → equivalent to ``e^(-jπ/2) = -j``):
    #
    #   I_+ = (I_m − j·a·I_a) / 2
    #   I_- = (I_m + j·a·I_a) / 2
    #
    # If a particular motor wires the aux 180° flipped, pass a
    # negative ``turns_ratio`` to flip back.
    Z11 = Z_main_stator + 0.5 * (Z_plus + Z_minus)
    Z12 = -0.5j * a * (Z_plus - Z_minus)
    Z21 = 0.5j * a * (Z_plus - Z_minus)
    Z22 = Z_aux_branch + 0.5 * (a ** 2) * (Z_plus + Z_minus)

    # Solve the 2×2 system [Z11 Z12; Z21 Z22] · [I_m; I_a] = [V; V].
    det = Z11 * Z22 - Z12 * Z21
    if abs(det) < 1e-20:
        raise RuntimeError(
            "compute_psc_operating_point: singular coupling matrix")
    I_m = (V * Z22 - V * Z12) / det
    I_a = (V * Z11 - V * Z21) / det

    # Sequence currents (matching the new sign convention above).
    I_plus  = 0.5 * (I_m - 1j * a * I_a)
    I_minus = 0.5 * (I_m + 1j * a * I_a)

    # Air-gap power per sequence — Re part of the rotor circuit
    # contribution. Rotor branch is the parallel-arm of Z_+ that's
    # NOT the magnetising branch.
    def rotor_real_power(I_seq: complex, s_eff: float) -> float:
        """Power dissipated in the rotor R for one field. Equals
        |I_r|² · R_r/s_eff where I_r is the rotor current in that
        field. We compute via voltage division: rotor current = the
        sequence current times (Z_m // (rotor)) divided by rotor
        impedance — easier: P = |I_seq|² · Re(Z_seq rotor part).
        We approximate the rotor air-gap power as
        |I_seq|² · Re(Z_seq) − |I_seq|² · 0 (the magnetising part
        is purely reactive, so Re cancels). That's accurate when
        X_m >> R_r — typical for normal-saturation operation."""
        Z_r = (params.R_rotor / s_eff) + 1j * params.X_rotor
        Z_seq = _parallel(1j * params.X_m, Z_r)
        return float(abs(I_seq) ** 2 * Z_seq.real)

    P_g_plus  = rotor_real_power(I_plus,  max(s, 1e-9))
    P_g_minus = rotor_real_power(I_minus, 2.0 - s)

    # Torque: (P_g+ - P_g-) / ω_s_mech (mechanical sync speed).
    T_em = (P_g_plus - P_g_minus) / omega_s_mech

    # Input power: Re{V · conj(I_m + a·I_a)}.
    I_total_stator = I_m + a * I_a   # current drawn from supply
    P_in = float((V * I_total_stator.conjugate()).real)

    # Output mechanical power (rotor): T_em · ω_m − friction.
    P_mech_em = T_em * omega_m
    P_friction = params.B * omega_m * omega_m
    P_mech = P_mech_em - P_friction

    eta = (P_mech / P_in) if P_in > 0 else 0.0

    # Power factor — angle between V and total stator current.
    pf = float(np.cos(np.angle(I_total_stator)))

    return PSCOperatingPoint(
        slip=s,
        omega_mech_rad_s=omega_m,
        speed_rpm=float(omega_m * 60.0 / (2.0 * np.pi)),
        torque_em_Nm=float(T_em),
        I_main=I_m,
        I_aux=I_a,
        P_in_W=P_in,
        P_mech_W=float(P_mech),
        efficiency=float(eta),
        power_factor=pf,
        Z_total_main=Z11,
        Z_total_aux=Z22,
    )


def psc_torque_speed_curve(params: SinglePhaseIMParams,
                                *,
                                V_supply: float,
                                f_supply: float,
                                slip_array: Sequence[float] | None = None,
                                n_points: int = 100,
                                ) -> List[PSCOperatingPoint]:
    """Sweep slip from 1 (locked rotor) to ~0 (sync) and return the
    operating point at each value. Use for plotting T-s and η-s
    curves.

    Parameters
    ----------
    params, V_supply, f_supply
        As for :func:`compute_psc_operating_point`.
    slip_array
        Optional explicit array of slip values. If ``None``, sweep
        ``n_points`` log-spaced values from 1.0 down to 0.001 (so
        the cusp near sync is well-resolved).
    n_points
        Number of points when ``slip_array`` is ``None``.

    Returns
    -------
    list of PSCOperatingPoint
        Parallel to the input ``slip_array``.
    """
    if slip_array is None:
        # Linear from 1 to small, dense near s=0 where the slope is
        # steep (pull-out is at low slip).
        slips = np.concatenate([
            np.linspace(1.0, 0.1, n_points // 2),
            np.geomspace(0.1, 1e-3, n_points - n_points // 2),
        ])
    else:
        slips = np.asarray(slip_array, dtype=float)
    out = []
    for s in slips:
        op = compute_psc_operating_point(
            params, V_supply=V_supply, f_supply=f_supply,
            slip=float(s))
        out.append(op)
    return out
