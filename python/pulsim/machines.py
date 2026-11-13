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

__all__ = ["install"]


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

    ``theta_node`` needs a voltage-controlled current source, which the
    kernel does not have yet (``add-controlled-source-set``); it is
    refused by name.
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
    if theta_node is not None:
        raise NotImplementedError(
            f"add_dc_machine({name!r}): theta_node needs a "
            "voltage-controlled current source (openspec "
            "add-controlled-source-set), which the kernel does not "
            "have yet. Integrate result.v(omega_node) instead.")

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
    return self


def install(builder_cls) -> None:
    """Patch ``add_dc_machine`` onto ``builder_cls`` (idempotent)."""
    if not hasattr(builder_cls, "add_dc_machine"):
        add_dc_machine.__qualname__ = "CircuitBuilder.add_dc_machine"
        builder_cls.add_dc_machine = add_dc_machine
