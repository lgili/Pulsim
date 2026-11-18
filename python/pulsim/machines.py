"""MNA-native electrical machines built from existing kernel elements.

``CircuitBuilder.add_dc_machine`` (openspec ``add-mna-native-machines``,
step 1) puts a DC machine INSIDE the circuit matrix:

    a_pos ─ R_a ─ L_a ─ e ─[ secondary ]─ a_neg       v_e − v_a_neg = K·ω
                           ideal transformer, n = K
    omega ─[ primary ]─ gnd                           torque K·i into omega
    omega ─ C = J ─ gnd,  omega ─ R = 1/B ─ gnd,  I = T_load out of omega

The speed ω [rad/s] is the voltage of ``omega_node`` and torque [N·m]
is current, so ``J·dω/dt = K·i − B·ω − T_load`` is the node's KCL. No
quantity is taken from the previous step: the machine is second order
on ``pwl``, runs unchanged on ``trbdf2`` and appears in the DC
operating point (inductor short, inertia open, so
``i = T_load/K`` and ``ω = (V − R_a·i)/K`` when ``B = 0``). Power is
conserved by construction: ``K·ω·i`` leaves the armature and enters
the shaft.

It replaces the forward-Euler observer of ``pulsim.add_dc_motor``
(first order: 13.6 % speed error at dt = 100 µs on the
``add-mna-native-machines`` probe, against 0.48 % here).

Sub-elements (only those with a non-zero value are created):
``{name}.ra`` (resistor), ``{name}.la`` (inductor, its current is the
armature current), ``{name}.em`` (ideal transformer), ``{name}.j``
(capacitor), ``{name}.b`` (resistor ``1/B``), ``{name}.tl`` (current
source).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ._pulsim import CircuitBuilder

__all__ = ["install", "pm_harmonics_from_waveform", "trapezoid_harmonics"]


def _check(name: str, label: str, value: float, *, positive: bool) -> float:
    v = float(value)
    ok = math.isfinite(v) and (v > 0.0 if positive else v >= 0.0)
    if not ok:
        rule = "> 0" if positive else ">= 0"
        raise ValueError(
            f"add_dc_machine({name!r}): {label} must be finite and "
            f"{rule} (got {value!r}).")
    return v


def add_dc_machine(self, name: str, a_pos: str, a_neg: str,
                   omega_node: str, *, R_a: float, L_a: float,
                   K: Optional[float] = None, J: float, B: float = 0.0,
                   T_load: float = 0.0, theta_node: Optional[str] = None,
                   i0: Optional[float] = None, omega0: float = 0.0,
                   Ke: Optional[float] = None, Kt: Optional[float] = None
                   ) -> "CircuitBuilder":
    """Add a permanent-magnet DC machine solved inside the matrix.

    Parameters [SI]: ``R_a`` [Ω, ≥ 0], ``L_a`` [H, > 0], ``K``
    [V·s/rad = N·m/A, > 0], ``J`` [kg·m², > 0], ``B`` [N·m·s/rad,
    ≥ 0], ``T_load`` [N·m, constant, opposing positive speed]. ``i0``
    is the initial armature current [A] and ``omega0`` the initial
    speed [rad/s]. ``omega_node`` is created if new; read the speed
    with ``result.v(omega_node)`` and the armature current with
    ``result.i(f"{name}.la")``. A speed-dependent load is any element
    you connect to ``omega_node`` (torque = current out of it).

    ``Ke`` / ``Kt`` are the legacy ``add_dc_motor`` spellings: in SI
    units they are the same constant, so a call with ``Kt != Ke`` is
    refused (the coupling would not conserve power).

    ``theta_node`` (optional) gets the shaft angle [rad]: a unit VCCS
    ``{name}.th`` from the speed node into a 1 F capacitor ``{name}.ct``,
    so ``v(theta_node) = ∫ω dt`` (starting at 0).
    """
    given = {k: v for k, v in (("K", K), ("Ke", Ke), ("Kt", Kt))
             if v is not None}
    if not given:
        raise ValueError(f"add_dc_machine({name!r}): K is required "
                         "(V·s/rad, equal to the torque constant N·m/A).")
    values = {float(v) for v in given.values()}
    if len(values) > 1:
        raise ValueError(
            f"add_dc_machine({name!r}): got different constants "
            f"{given}. In SI units the back-EMF constant Ke [V·s/rad] "
            "and the torque constant Kt [N·m/A] are the same number "
            "(power K·ω·i leaves the armature and enters the shaft); "
            "a machine with Kt != Ke would create or destroy energy. "
            "Pass one K; convert datasheet units (e.g. V/krpm: "
            "K = Ke_V_per_krpm · 60 / (2π·1000)).")
    k = _check(name, "K", values.pop(), positive=True)
    r_a = _check(name, "R_a", R_a, positive=False)
    l_a = _check(name, "L_a", L_a, positive=True)
    j = _check(name, "J", J, positive=True)
    b = _check(name, "B", B, positive=False)
    t_load = float(T_load)
    if not math.isfinite(t_load):
        raise ValueError(f"add_dc_machine({name!r}): T_load must be "
                         f"finite (got {T_load!r}).")

    l_from = f"{name}.ra_la" if r_a > 0.0 else a_pos
    emf = f"{name}.emf"
    if r_a > 0.0:
        self.add_resistor(f"{name}.ra", a_pos, l_from, r_a)
    self.add_inductor(f"{name}.la", l_from, emf, l_a, i0)
    # Secondary = armature side (v_s = K·v_p), primary = speed node:
    # the reflected primary current K·i is the torque into omega.
    self.add_ideal_transformer(f"{name}.em", omega_node, "gnd",
                               emf, a_neg, k)
    self.add_capacitor(f"{name}.j", omega_node, "gnd", j, float(omega0))
    if b > 0.0:
        self.add_resistor(f"{name}.b", omega_node, "gnd", 1.0 / b)
    if t_load != 0.0:
        # A current source pushes I out of n_pos: n_pos = gnd draws
        # T_load out of the speed node.
        self.add_current_source(f"{name}.tl", "gnd", omega_node, t_load)
    if theta_node is not None:
        # θ = ∫ω dt: a unit VCCS from the speed into a 1 F capacitor.
        self.add_vccs(f"{name}.th", omega_node, "gnd", theta_node, "gnd",
                      1.0)
        self.add_capacitor(f"{name}.ct", theta_node, "gnd", 1.0, 0.0)
    return self


def trapezoid_harmonics(k_e: float, pole_pairs: int, *,
                        flat_top_deg: float = 120.0,
                        n_harmonics: int = 25) -> list:
    """PM-flux harmonic series ``[(h, psi_h, phi_h), ...]`` of a
    trapezoidal back-EMF (``add-mna-native-machines`` design, BLDC).

    ``k_e`` [V·s/rad] is the phase back-EMF on the flat top per unit
    MECHANICAL speed; the flat top is ``flat_top_deg`` electrical degrees
    wide and the ramps are straight. With ramp half-width
    ``gamma = (180° − flat_top)/2``, odd ``h`` up to ``n_harmonics``::

        psi_h = (k_e/pp)·4·sin(h·gamma)/(pi·gamma·h³),   phi_h = 0

    so ``lambda_pm,a(theta_e) = sum psi_h·cos(h·theta_e)`` and the
    back-EMF ``-pp·omega·dlambda/dtheta_e`` is the truncated trapezoid
    (RMS deviation 0.40 % of the flat top at 25 harmonics, 120°).
    """
    k_e, pp = float(k_e), int(pole_pairs)
    if not (math.isfinite(k_e) and k_e > 0.0) or pp < 1:
        raise ValueError("trapezoid_harmonics: k_e must be > 0 and "
                         "pole_pairs a positive integer")
    if not 0.0 <= flat_top_deg < 180.0:
        raise ValueError("trapezoid_harmonics: flat_top_deg must be in "
                         "[0, 180)")
    gamma = math.radians(180.0 - flat_top_deg) / 2.0
    out = []
    for h in range(1, int(n_harmonics) + 1, 2):
        s = (math.sin(h * gamma) / (h * gamma)) if gamma > 0 else 1.0
        out.append((float(h), (k_e / pp) * 4.0 * s / (math.pi * h * h),
                    0.0))
    return out


def pm_harmonics_from_waveform(theta_e, emf, omega_e: float, *,
                               n_harmonics: int = 25) -> list:
    """PM-flux series ``[(h, psi_h, phi_h), ...]`` from one electrical
    period of a measured phase-a back-EMF.

    ``theta_e`` are ``N`` uniformly spaced electrical angles covering
    one period (``[0, 2π)``, endpoint excluded), ``emf`` the phase-a
    back-EMF [V] at those angles, recorded at the constant ELECTRICAL
    speed ``omega_e`` [rad/s] with the motor convention ``e = dλ/dt``.
    Returns the harmonics ``1..n_harmonics`` (the DC term is dropped:
    a constant flux has no back-EMF and no torque).
    """
    import numpy as np

    th = np.asarray(theta_e, dtype=float)
    e = np.asarray(emf, dtype=float)
    n = th.size
    if e.size != n or n < 2 * n_harmonics + 2:
        raise ValueError("pm_harmonics_from_waveform: theta_e and emf "
                         "must have the same length, >= 2·n_harmonics + 2")
    if not (math.isfinite(omega_e) and omega_e != 0.0):
        raise ValueError("pm_harmonics_from_waveform: omega_e must be "
                         "finite and non-zero")
    c = np.fft.rfft(e / omega_e) / n * 2.0           # dλ/dθ_e coefficients
    out = []
    for h in range(1, int(n_harmonics) + 1):
        ch = c[h] * np.exp(-1j * h * th[0])          # phase at theta = 0
        amp = abs(ch)
        if amp == 0.0:
            continue
        # dλ/dθ = amp·cos(hθ + β) = −h·psi·sin(hθ + φ), φ = β − π/2.
        out.append((float(h), float(amp / h),
                    float(np.angle(ch) - math.pi / 2.0)))
    return out


def add_bldc_mna(self, name: str, phase_a: str, phase_b: str, phase_c: str,
                 neutral: str, omega_node: str, theta_node: str, *,
                 R_s: float, L_s: float, k_e: float, pole_pairs: int,
                 J: float, B: float = 0.0, T_load: float = 0.0,
                 flat_top_deg: float = 120.0, n_harmonics: int = 25,
                 omega0: float = 0.0, theta0: float = 0.0
                 ) -> "CircuitBuilder":
    """Add an MNA-native BLDC: ``add_pmsm_mna`` (round rotor,
    ``L_d = L_q = L_s``) with the PM flux of a trapezoidal back-EMF
    (:func:`trapezoid_harmonics`). Back-EMF and torque come from the
    same series and its exact derivatives, so the torque is consistent
    with the back-EMF (co-energy). ``k_e`` [V·s/rad]: phase back-EMF on
    the flat top per unit mechanical speed."""
    harm = trapezoid_harmonics(k_e, pole_pairs, flat_top_deg=flat_top_deg,
                               n_harmonics=n_harmonics)
    return self.add_pmsm_mna(
        name, phase_a, phase_b, phase_c, neutral, omega_node, theta_node,
        R_s=R_s, L_d=L_s, L_q=L_s, psi_pm=harm[0][1],
        pole_pairs=pole_pairs, J=J, B=B, T_load=T_load, omega0=omega0,
        theta0=theta0, harmonics=harm)


def install(builder_cls) -> None:
    """Patch ``add_dc_machine`` / ``add_bldc_mna`` onto ``builder_cls``
    (idempotent)."""
    for fn in (add_dc_machine, add_bldc_mna):
        if not hasattr(builder_cls, fn.__name__):
            fn.__qualname__ = f"CircuitBuilder.{fn.__name__}"
            setattr(builder_cls, fn.__name__, fn)
