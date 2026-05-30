"""Pulsim — electromechanical motor models (Phase D).

Pure-Python motor helpers built on top of v2's R/L/voltage-source
primitives plus the step_observer + b_extra_fn pair. Each motor is
implemented as:

  * **Electrical side** — armature/phase RL + dummy "0 V" voltage
    sources that the observer modulates at runtime to inject the
    back-EMF.
  * **Mechanical side** — Python closure carrying (ω, θ) state. At
    each step the observer:
      1. Reads phase currents from the state vector
      2. Computes electromagnetic torque T_em
      3. Integrates ``J · dω/dt = T_em − T_load − B · ω`` (forward
         Euler — adequate at v2's typical sub-µs dt)
      4. Updates θ = ∫ ω dt
      5. Sets the next-step back-EMF amplitudes / phases

Three motor flavours ship in Phase D:
  * `add_dc_motor` — single-loop armature with V_bemf = Ke · ω.
  * `add_pmsm` — 3-φ PMSM with sinusoidal back-EMF.
  * `add_bldc` — 3-φ BLDC with trapezoidal back-EMF.

Each helper returns a ``MotorState`` dataclass carrying the live
mechanical state, so the user can:

    motor = p.add_dc_motor(builder, ...)
    obs, b_extra = p.make_dc_motor_observer(builder, motor, ...)
    res = p.simulate(builder, ..., step_observer=obs,
                         b_extra_fn=b_extra)
    print(f"final ω = {motor.omega_rad_s} rad/s")
    print(f"final θ = {motor.theta_rad} rad")

The Mechanical base does forward-Euler at the simulation `dt`. For
plants where the mechanical time constant is much longer than the
electrical one (motor inertia >> winding L/R), Euler is plenty
accurate. For coupled high-bandwidth designs, switch to the
adaptive driver (Phase B.1) and the observer integrates at its own
finer pace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional


__all__ = [
    "Mechanical",
    "DcMotor",
    "PMSM",
    "BLDC",
    "InductionMotor",
    "MotorObserverBundle",
    "add_dc_motor",
    "make_dc_motor_observer",
    "add_pmsm",
    "make_pmsm_observer",
    "add_bldc",
    "make_bldc_observer",
    "add_induction_motor",
    "make_induction_motor_observer",
    "im_parameters_from_nameplate",
]


# =============================================================================
# Motor observer bundle — exposes per-step traces (T2.2)
# =============================================================================

class MotorObserverBundle:
    """Bundles a motor's ``(step_observer, b_extra_fn)`` pair together
    with per-step trace buffers for ω, θ, i_d/i_q, T_em.

    Why this exists (GUI integration findings T2.2). The C++ PMSM /
    BLDC / DC handles only expose topology metadata
    (``neutral_node``, branch ids). The rotor state ω/θ lives inside
    the observer's closure and was unreachable from a
    :class:`SimulationResult`. Every motor study needs a speed/angle
    plot — wrapping the observer in a throwaway probe was the
    documented workaround. This bundle fixes that.

    Backward compatibility. The bundle is *callable* (it IS the
    step_observer) and *iterable* — existing call sites can keep
    using::

        obs, b_extra = p.make_pmsm_observer(b, motor, dt=DT)

    and `obs` is a normal step_observer Callable[[float, Sequence],
    None]. New callers can keep the bundle reference and read::

        bundle = p.make_pmsm_observer(b, motor, dt=DT)
        res = p.simulate(..., step_observer=bundle,
                          b_extra_fn=bundle.b_extra_fn)
        plt.plot(bundle.times, bundle.omega_rad_s)
        plt.plot(bundle.times, bundle.theta_rad)

    Or via :class:`SimulationResult`::

        res.signal(f"{bundle.name}.omega")   # rotor speed [rad/s]
        res.signal(f"{bundle.name}.theta")   # rotor angle [rad]
        res.signal(f"{bundle.name}.T_em")    # electromagnetic torque [Nm]
        res.signal(f"{bundle.name}.i_d")     # d-axis current [A] (3φ only)
        res.signal(f"{bundle.name}.i_q")     # q-axis current [A] (3φ only)

    The auto-attach happens inside :func:`pulsim.simulate` — the user
    doesn't need to wire it manually.

    Parameters
    ----------
    name
        Trace prefix for :meth:`attach_to_result`. Defaults to
        ``"M1"`` — match the motor's ``name=`` kwarg.
    has_dq
        Whether to populate the ``i_d``/``i_q`` lists. True for 3-phase
        motors (PMSM / BLDC / IM); False for the DC motor (which has
        ``i_a`` instead).
    """

    __slots__ = (
        "_inner_step",
        "b_extra_fn",
        "name",
        "has_dq",
        "times",
        "omega_rad_s",
        "theta_rad",
        "T_em",
        "i_d",
        "i_q",
        "i_a",
        "i_b",
        "i_c",
    )

    def __init__(
        self,
        inner_step,
        b_extra_fn,
        *,
        name: str = "M1",
        has_dq: bool = True,
    ) -> None:
        self._inner_step = inner_step
        self.b_extra_fn = b_extra_fn
        self.name = name
        self.has_dq = has_dq
        self.times: list = []
        self.omega_rad_s: list = []
        self.theta_rad: list = []
        self.T_em: list = []
        # 3φ-only buffers (always present so the slots layout is
        # uniform; left empty for DC motor).
        self.i_d: list = []
        self.i_q: list = []
        self.i_a: list = []
        self.i_b: list = []
        self.i_c: list = []

    def __call__(self, t, x):
        # `_inner_step` is the per-motor closure that updates the
        # `mech` state AND pushes into our buffers. We forward the
        # call verbatim; the closure has captured our `self`.
        return self._inner_step(t, x)

    def __iter__(self):
        # Legacy unpacking:  obs, b_extra = make_pmsm_observer(...)
        yield self
        yield self.b_extra_fn

    def __getitem__(self, idx):
        return (self, self.b_extra_fn)[idx]

    # -------- Trace publishing ---------------------------------------
    def to_dict(self):
        """Return a ``{trace_name: numpy.ndarray}`` snapshot of the
        accumulated buffers. Pulls in numpy lazily so import-time cost
        stays low."""
        import numpy as np
        out = {
            f"{self.name}.t":       np.asarray(self.times, dtype=float),
            f"{self.name}.omega":   np.asarray(self.omega_rad_s, dtype=float),
            f"{self.name}.theta":   np.asarray(self.theta_rad, dtype=float),
            f"{self.name}.T_em":    np.asarray(self.T_em, dtype=float),
        }
        if self.has_dq:
            out[f"{self.name}.i_d"] = np.asarray(self.i_d, dtype=float)
            out[f"{self.name}.i_q"] = np.asarray(self.i_q, dtype=float)
            out[f"{self.name}.i_a"] = np.asarray(self.i_a, dtype=float)
            out[f"{self.name}.i_b"] = np.asarray(self.i_b, dtype=float)
            out[f"{self.name}.i_c"] = np.asarray(self.i_c, dtype=float)
        else:
            # DC motor — armature current under the "i_a" key.
            out[f"{self.name}.i_a"] = np.asarray(self.i_a, dtype=float)
        return out

    def attach_to_result(self, result) -> None:
        """Stash this bundle's traces on `result` so
        ``result.signal('M1.omega')`` resolves.

        Idempotent: callable multiple times. The traces are taken from
        the bundle's current buffers (so call this AFTER the simulation
        finishes).
        """
        traces = getattr(result, "_motor_traces", None)
        if traces is None:
            traces = {}
            try:
                result._motor_traces = traces
            except AttributeError:  # pragma: no cover — defensive
                return
        traces.update(self.to_dict())


# =============================================================================
# Mechanical base — shared inertia / friction / load model
# =============================================================================

@dataclass
class Mechanical:
    """Rotational mechanical model: J · dω/dt = T_em − T_load − B · ω.

    Carries live ``omega_rad_s`` and ``theta_rad`` state plus the
    physical parameters. `T_load_Nm` can be a constant or a callable
    ``(t, omega) -> Nm`` for speed-dependent loads (fans, pumps).
    """
    J_kgm2: float                                  # rotor inertia
    B_Nms_per_rad: float = 0.0                    # viscous friction
    T_load_Nm: float = 0.0                        # constant load (Nm)
    T_load_fn: Optional[Callable[[float, float], float]] = None

    omega_rad_s: float = 0.0
    theta_rad: float = 0.0

    def reset(self) -> None:
        self.omega_rad_s = 0.0
        self.theta_rad = 0.0

    def integrate(self, t: float, T_em_Nm: float, dt: float) -> None:
        """Forward-Euler update of (ω, θ) by one timestep."""
        T_load = self.T_load_Nm
        if self.T_load_fn is not None:
            T_load = float(self.T_load_fn(t, self.omega_rad_s))
        domega = ((T_em_Nm - T_load -
                     self.B_Nms_per_rad * self.omega_rad_s) /
                    max(self.J_kgm2, 1e-30)) * dt
        self.omega_rad_s += domega
        self.theta_rad += self.omega_rad_s * dt


# =============================================================================
# 1. DC motor
# =============================================================================

@dataclass
class DcMotor:
    """A DC motor instance. Holds parameters + a `Mechanical` block.

    Branch indices populated by :func:`add_dc_motor` are stored here
    so the observer can find the armature current row.
    """
    R_a_ohm: float
    L_a_H: float
    Ke_V_s_per_rad: float                # back-EMF constant
    Kt_Nm_per_A: float                   # torque constant
    mech: Mechanical
    # Filled in by add_dc_motor:
    armature_node: str = ""
    bemf_pos_node: str = ""
    bemf_neg_node: str = ""
    bemf_source_branch_id: int = -1
    inductor_branch_id: int = -1


def add_dc_motor(builder,
                    *,
                    name: str = "M1",
                    armature_pos: str,
                    armature_neg: str,
                    R_a: float,
                    L_a: float,
                    Ke: float,
                    Kt: Optional[float] = None,
                    J: float,
                    B: float = 0.0,
                    T_load: float = 0.0,
                    T_load_fn=None,
                    ) -> DcMotor:
    """Add a DC motor to the builder.

    Topology added (between `armature_pos` and `armature_neg`):

        armature_pos ── R_a ── L_a ── (V_bemf=0) ── armature_neg

    The 0V "dummy" voltage source represents the back-EMF; the
    observer modulates its value to ``Ke · ω`` at every step via
    ``b_extra_fn``.

    Parameters
    ----------
    name
        Used to prefix the added device names.
    armature_pos, armature_neg
        Node names of the two armature terminals.
    R_a, L_a
        Armature winding resistance (Ω) and inductance (H).
    Ke
        Back-EMF constant in V·s/rad (V per rad/s). For DC motors
        in SI units, ``Kt = Ke`` numerically; this is required if
        you don't supply Kt explicitly.
    Kt
        Torque constant in Nm/A. Defaults to `Ke`.
    J
        Rotor inertia (kg·m²).
    B
        Viscous-friction coefficient (Nm·s/rad).
    T_load
        Constant load torque (Nm). For variable loads, pass
        `T_load_fn(t, omega) -> Nm`.
    """
    if Kt is None:
        Kt = Ke
    mech = Mechanical(J_kgm2=J, B_Nms_per_rad=B,
                          T_load_Nm=T_load, T_load_fn=T_load_fn)
    mid = f"{name}_mid"
    bemf_mid = f"{name}_bemf"

    builder.add_resistor(f"{name}_R", armature_pos, mid, float(R_a))
    builder.add_inductor(f"{name}_L", mid, bemf_mid, float(L_a))
    # Back-EMF "dummy" source (0V). Observer will modulate via b_extra.
    # Record the branch id (next branch index = current num_branches).
    bemf_id = builder.graph.num_branches
    builder.add_voltage_source(f"{name}_Ebemf", bemf_mid,
                                  armature_neg, 0.0)
    inductor_id = bemf_id - 1   # the inductor was the previous branch

    motor = DcMotor(
        R_a_ohm=R_a, L_a_H=L_a, Ke_V_s_per_rad=Ke,
        Kt_Nm_per_A=Kt, mech=mech,
        armature_node=armature_pos,
        bemf_pos_node=bemf_mid, bemf_neg_node=armature_neg,
        bemf_source_branch_id=bemf_id,
        inductor_branch_id=inductor_id,
    )
    return motor


def make_dc_motor_observer(builder, motor: DcMotor, *, dt: float,
                              name: str = "M1") -> "MotorObserverBundle":
    """Build a :class:`MotorObserverBundle` for a DC motor.

    The observer:
      1. Reads i_armature from the state vector.
      2. Computes T_em = Kt · i.
      3. Integrates Mechanical by one step (forward-Euler).
      4. Stashes the next-step back-EMF voltage in a closure.

    The b_extra_fn injects the stashed back-EMF into the source's
    constraint row at every step.

    The bundle exposes ``times``, ``omega_rad_s``, ``theta_rad``,
    ``T_em`` and ``i_a`` (armature current) — populated by every
    step_observer call. After :func:`pulsim.simulate`, the same
    traces are reachable via :meth:`SimulationResult.signal`
    (e.g. ``res.signal("M1.omega")``). See
    :func:`make_pmsm_observer` for the full bundle contract;
    note: DC motors don't define dq currents, so the ``i_d``/``i_q``
    buffers stay empty (``has_dq=False``).

    Backward compatibility. Existing call sites that do
    ``obs, b_extra = make_dc_motor_observer(...)`` keep working —
    the bundle iterates as a 2-tuple.

    Parameters
    ----------
    builder
        The CircuitBuilder that already has the motor added.
    motor
        The `DcMotor` returned by :func:`add_dc_motor`.
    dt
        Mechanical integration step. Typically equal to the
        simulation dt.
    name
        Trace prefix for :meth:`MotorObserverBundle.attach_to_result`.
    """
    state_size = builder.pool.state_size(builder.graph)
    i_idx = builder.pool.branch_var_id_for_inductor(
        motor.inductor_branch_id, builder.graph)
    src_idx = builder.pool.branch_var_id_for_source(
        motor.bemf_source_branch_id, builder.graph)

    bemf = {"V": 0.0}
    bundle = MotorObserverBundle(
        inner_step=None, b_extra_fn=None,
        name=name, has_dq=False,
    )

    def step_observer(t, x):
        i_a = float(x[i_idx])
        T_em = motor.Kt_Nm_per_A * i_a
        motor.mech.integrate(t, T_em, dt)
        bemf["V"] = motor.Ke_V_s_per_rad * motor.mech.omega_rad_s

        bundle.times.append(float(t))
        bundle.omega_rad_s.append(float(motor.mech.omega_rad_s))
        bundle.theta_rad.append(float(motor.mech.theta_rad))
        bundle.T_em.append(float(T_em))
        bundle.i_a.append(i_a)

    def b_extra_fn(t):
        out = [0.0] * state_size
        # Convention: constraint row reads (V_from − V_to) − V_source = 0.
        # Adding +V_extra to b shifts V_source by −V_extra, i.e. the
        # source value becomes (0 − (−V_bemf)) = V_bemf. So inject
        # −V_bemf to get the right sign.
        out[src_idx] = -bemf["V"]
        return out

    bundle._inner_step = step_observer
    bundle.b_extra_fn = b_extra_fn
    return bundle


# =============================================================================
# 2. PMSM — 3-phase synchronous motor with sinusoidal back-EMF
# =============================================================================

@dataclass
class PMSM:
    """3-phase permanent-magnet synchronous motor.

    Three armature windings (Y-connected to a floating neutral).
    Each phase has R_s + L_s + a phase-dependent back-EMF source.
    The observer tracks electrical angle θ_e = pp · θ_m and drives
    the three back-EMF sources with sinusoids 120° apart.
    """
    R_s_ohm: float
    L_s_H: float
    psi_pm_Wb: float                     # PM flux linkage [Wb]
    pole_pairs: int                       # pp
    mech: Mechanical
    # Filled in by add_pmsm:
    phase_branch_ids: tuple = field(default_factory=tuple)   # 3 inductor ids
    bemf_source_ids:  tuple = field(default_factory=tuple)   # 3 source ids
    neutral_node: str = ""


def add_pmsm(builder,
                *,
                name: str = "M1",
                phase_nodes,
                neutral_node: str,
                R_s: float,
                L_s: float,
                psi_pm: float,
                pole_pairs: int,
                J: float,
                B: float = 0.0,
                T_load: float = 0.0,
                T_load_fn=None,
                ) -> PMSM:
    """Add a 3-phase PMSM. Each phase: R_s + L_s + back-EMF source
    in series between `phase_nodes[k]` and `neutral_node`.

    Parameters
    ----------
    phase_nodes
        3-element sequence of phase terminal node names.
    neutral_node
        Star-point neutral.
    R_s, L_s
        Per-phase stator resistance + inductance.
    psi_pm
        PM flux linkage in Wb. Back-EMF peak = pp · ψ_pm · ω_m.
    pole_pairs
        Number of pole pairs (pp). ω_e = pp · ω_m.
    J, B, T_load, T_load_fn
        Mechanical parameters (see :func:`add_dc_motor`).
    """
    if len(phase_nodes) != 3:
        raise ValueError("phase_nodes must have 3 entries")
    mech = Mechanical(J_kgm2=J, B_Nms_per_rad=B,
                          T_load_Nm=T_load, T_load_fn=T_load_fn)

    phase_branch_ids = []
    bemf_source_ids = []
    for k, p_node in enumerate(phase_nodes):
        mid_r = f"{name}_R_mid_{('a','b','c')[k]}"
        mid_l = f"{name}_L_mid_{('a','b','c')[k]}"
        # Skip R if zero — keeps the graph simpler. Otherwise:
        builder.add_resistor(f"{name}_Rs_{('a','b','c')[k]}",
                                p_node, mid_r, float(R_s))
        ind_id = builder.graph.num_branches
        builder.add_inductor(f"{name}_Ls_{('a','b','c')[k]}",
                                mid_r, mid_l, float(L_s))
        phase_branch_ids.append(ind_id)
        # Back-EMF source (0V initially; observer modulates).
        bemf_id = builder.graph.num_branches
        builder.add_voltage_source(
            f"{name}_E_{('a','b','c')[k]}",
            mid_l, neutral_node, 0.0)
        bemf_source_ids.append(bemf_id)

    motor = PMSM(
        R_s_ohm=R_s, L_s_H=L_s, psi_pm_Wb=psi_pm,
        pole_pairs=pole_pairs, mech=mech,
        phase_branch_ids=tuple(phase_branch_ids),
        bemf_source_ids=tuple(bemf_source_ids),
        neutral_node=neutral_node,
    )
    return motor


def _make_3phase_motor_observer(builder, motor, *, dt: float,
                                    waveform: str,
                                    name: str = "M1"):
    """Shared implementation for PMSM (sinusoidal) and BLDC
    (trapezoidal) back-EMF observers.

    Returns a :class:`MotorObserverBundle` carrying per-step traces
    (ω, θ, T_em, i_a/i_b/i_c, i_d/i_q).
    """
    state_size = builder.pool.state_size(builder.graph)
    phase_idx = tuple(
        builder.pool.branch_var_id_for_inductor(bid, builder.graph)
        for bid in motor.phase_branch_ids)
    src_idx = tuple(
        builder.pool.branch_var_id_for_source(sid, builder.graph)
        for sid in motor.bemf_source_ids)

    bemf = {"v": (0.0, 0.0, 0.0)}
    # T2.2: trace publishing. ``bundle`` is the callable we return.
    # We build it first as a forward declaration, then point its
    # _inner_step at the closure below. The closure captures
    # ``bundle`` so it can push samples.
    bundle = MotorObserverBundle(
        inner_step=None,   # set after closure is defined
        b_extra_fn=None,   # set after closure is defined
        name=name,
        has_dq=True,
    )

    def bemf_shape(theta_e: float, k: int) -> float:
        """Back-EMF shape for phase k (k=0,1,2 → a,b,c).

        Convention matched to the FOC chain's Park transform:
          * d-axis aligned with rotor PM flux
          * q-axis 90° ahead of d in the direction of rotation
          * Positive i_q with i_d=0 must produce POSITIVE torque

        With the chain's Park matrix
            [d]   [ cos θ   sin θ]   [α]
            [q] = [−sin θ   cos θ] · [β]
        and inverse Clarke α=i_a, β=(i_b−i_c)/√3, a positive i_q
        appears in phase A as i_a = −sin(θ_e)·i_q. For the power
        balance P_em = Σ e·i to be POSITIVE in motor mode, the
        BEMF in phase A must be e_a = −sin(θ_e)·E_peak (i.e. the
        OPPOSITE sign of the naive +sin form).
        """
        offset = -2.0 * math.pi / 3.0 * k    # 0, −120°, +120° for k=0,1,2
        if waveform == "sinusoidal":
            return -math.sin(theta_e + offset)
        # Trapezoidal: same sign convention but with a 120° flat top.
        s = -math.sin(theta_e + offset)
        # 120°-flat-top: f(θ) = clip(2 sin(θ), -1, +1) gives a
        # roughly trapezoidal shape with flat 60° regions.
        return max(-1.0, min(1.0, 2.0 * s))

    sqrt3 = math.sqrt(3.0)

    def step_observer(t, x):
        i_a = float(x[phase_idx[0]])
        i_b = float(x[phase_idx[1]])
        i_c = float(x[phase_idx[2]])
        omega_m = motor.mech.omega_rad_s
        theta_e = motor.mech.theta_rad * motor.pole_pairs
        E_peak = motor.pole_pairs * motor.psi_pm_Wb * omega_m

        e_a = E_peak * bemf_shape(theta_e, 0)
        e_b = E_peak * bemf_shape(theta_e, 1)
        e_c = E_peak * bemf_shape(theta_e, 2)
        bemf["v"] = (e_a, e_b, e_c)

        # Torque from instantaneous power balance:
        # P_mech = ω_m · T_em = e_a·i_a + e_b·i_b + e_c·i_c
        P_em = e_a * i_a + e_b * i_b + e_c * i_c
        if abs(omega_m) > 1e-6:
            T_em = P_em / omega_m
        else:
            # At standstill, use the limit torque formula:
            # T_em ≈ pp · ψ_pm · (i_a · shape_a + …) / ω_m → indeterminate.
            # Use the dq-frame formula proxy with ω→0.
            T_em = motor.pole_pairs * motor.psi_pm_Wb * (
                bemf_shape(theta_e, 0) * i_a +
                bemf_shape(theta_e, 1) * i_b +
                bemf_shape(theta_e, 2) * i_c)
        motor.mech.integrate(t, T_em, dt)

        # T2.2: publish traces. Clarke (amplitude-invariant) → Park.
        # Matches the convention used in `pulsim.motor_helpers.FOC`:
        #   α = (2/3)·(i_a − ½ i_b − ½ i_c)
        #   β = (1/√3)·(i_b − i_c)
        #   [d; q] = [cos θ, sin θ; −sin θ, cos θ] · [α; β]
        i_alpha = (2.0 / 3.0) * (i_a - 0.5 * i_b - 0.5 * i_c)
        i_beta = (i_b - i_c) / sqrt3
        cos_t = math.cos(theta_e)
        sin_t = math.sin(theta_e)
        i_d_now = i_alpha * cos_t + i_beta * sin_t
        i_q_now = -i_alpha * sin_t + i_beta * cos_t

        bundle.times.append(float(t))
        bundle.omega_rad_s.append(float(motor.mech.omega_rad_s))
        bundle.theta_rad.append(float(motor.mech.theta_rad))
        bundle.T_em.append(float(T_em))
        bundle.i_a.append(i_a)
        bundle.i_b.append(i_b)
        bundle.i_c.append(i_c)
        bundle.i_d.append(float(i_d_now))
        bundle.i_q.append(float(i_q_now))

    def b_extra_fn(t):
        out = [0.0] * state_size
        for k in range(3):
            out[src_idx[k]] = -bemf["v"][k]
        return out

    bundle._inner_step = step_observer
    bundle.b_extra_fn = b_extra_fn
    return bundle


def make_pmsm_observer(builder, motor: PMSM, *, dt: float,
                          name: str = "M1") -> "MotorObserverBundle":
    """Build a :class:`MotorObserverBundle` for a PMSM with sinusoidal
    back-EMF.

    The bundle is a callable step_observer that also exposes
    per-step trace buffers (``times``, ``omega_rad_s``, ``theta_rad``,
    ``T_em``, ``i_d``, ``i_q``, ``i_a``/``i_b``/``i_c``). It still
    unpacks as ``(step_observer, b_extra_fn)`` for backward
    compatibility::

        # Legacy (works unchanged):
        obs, b_extra = p.make_pmsm_observer(b, motor, dt=DT)
        res = p.simulate(b, ..., step_observer=obs, b_extra_fn=b_extra)

        # New: keep the bundle for trace access.
        bundle = p.make_pmsm_observer(b, motor, dt=DT, name="M1")
        res = p.simulate(b, ..., step_observer=bundle,
                          b_extra_fn=bundle.b_extra_fn)
        # After simulate, traces are auto-attached to res:
        plt.plot(res.signal("M1.omega"))
        plt.plot(bundle.omega_rad_s)   # same data, raw list

    Parameters
    ----------
    name
        Trace prefix. Default ``"M1"`` matches :func:`add_pmsm`'s
        default device name; override when running multiple motors
        in the same circuit.
    """
    return _make_3phase_motor_observer(builder, motor, dt=dt,
                                            waveform="sinusoidal",
                                            name=name)


# =============================================================================
# 3. BLDC — 3-phase with trapezoidal back-EMF
# =============================================================================

@dataclass
class BLDC(PMSM):
    """Brushless-DC motor — same structure as PMSM but trapezoidal
    back-EMF instead of sinusoidal. Typically driven by 120°-block
    commutation (3-phase, 6-step) rather than continuous SVM."""


def add_bldc(builder,
                *,
                name: str = "M1",
                phase_nodes,
                neutral_node: str,
                R_s: float,
                L_s: float,
                psi_pm: float,
                pole_pairs: int,
                J: float,
                B: float = 0.0,
                T_load: float = 0.0,
                T_load_fn=None,
                ) -> BLDC:
    """Add a 3-phase BLDC motor. Same topology as PMSM but the
    observer drives the back-EMF with a trapezoidal shape."""
    pmsm = add_pmsm(builder, name=name,
                       phase_nodes=phase_nodes,
                       neutral_node=neutral_node,
                       R_s=R_s, L_s=L_s, psi_pm=psi_pm,
                       pole_pairs=pole_pairs,
                       J=J, B=B, T_load=T_load, T_load_fn=T_load_fn)
    # Upgrade the PMSM dataclass to a BLDC (same fields).
    return BLDC(
        R_s_ohm=pmsm.R_s_ohm, L_s_H=pmsm.L_s_H,
        psi_pm_Wb=pmsm.psi_pm_Wb, pole_pairs=pmsm.pole_pairs,
        mech=pmsm.mech,
        phase_branch_ids=pmsm.phase_branch_ids,
        bemf_source_ids=pmsm.bemf_source_ids,
        neutral_node=pmsm.neutral_node,
    )


def make_bldc_observer(builder, motor: BLDC, *, dt: float,
                          name: str = "M1") -> "MotorObserverBundle":
    """Build a :class:`MotorObserverBundle` for a BLDC with trapezoidal
    back-EMF. See :func:`make_pmsm_observer` for the bundle API."""
    return _make_3phase_motor_observer(builder, motor, dt=dt,
                                            waveform="trapezoidal",
                                            name=name)


# =============================================================================
# 4. Three-phase squirrel-cage Induction Motor (dq stationary-frame model)
# =============================================================================
#
# State-space — 4 internal rotor / mechanical states maintained by the
# observer + 3 stator-current states already living in the MNA state
# vector (the per-phase σLs branch currents).
#
#   dψ_rα/dt = −(1/Tr)·ψ_rα + (Lm/Tr)·i_sα − ω_e·ψ_rβ
#   dψ_rβ/dt = −(1/Tr)·ψ_rβ + (Lm/Tr)·i_sβ + ω_e·ψ_rα
#   Tr      = Lr / Rr           (rotor time constant)
#   ω_e     = pp · ω_m          (electrical angular frequency)
#
#   T_em = (3/2) · pp · (Lm/Lr) · (i_sβ·ψ_rα − i_sα·ψ_rβ)
#
# The induced stator EMF reflected from the rotor flux is
#   e_α = (Lm/Lr) · dψ_rα/dt
#   e_β = (Lm/Lr) · dψ_rβ/dt
# These are injected into the b_extra map via the three "dummy"
# voltage sources at each phase terminal (same trick as PMSM/BLDC).
#
# Topology — per phase k ∈ {a,b,c}:
#
#   phase_node ── R_s ── (Rmid_k) ── σLs ── (Lmid_k) ── E_k ── neutral
#                                            ↑
#                                            └── modulated by observer
#
# where σLs = (1 − Lm² / (Ls·Lr)) · Ls is the stator transient
# inductance — the part of stator inductance NOT shared with the
# rotor circuit through the mutual coupling. Using σLs at the
# topology level lets us put the mutual-coupling dynamics entirely
# inside the observer (Lm/Lr·dψ_r/dt term) without an explicit
# coupled-inductor in the MNA matrix.

@dataclass
class InductionMotor:
    """Three-phase squirrel-cage induction motor — dq stationary-frame
    model. Created by :func:`add_induction_motor`."""
    # Electrical parameters (Y-equivalent, per phase, rotor referred to stator).
    R_s_ohm: float                                 # stator resistance
    L_s_H: float                                   # stator self-inductance
    R_r_ohm: float                                 # rotor resistance (referred)
    L_r_H: float                                   # rotor self-inductance (referred)
    L_m_H: float                                   # mutual inductance (referred)
    pole_pairs: int
    mech: Mechanical
    # Filled in by add_induction_motor:
    phase_branch_ids: tuple = field(default_factory=tuple)   # 3 inductor ids
    bemf_source_ids:  tuple = field(default_factory=tuple)   # 3 source ids
    neutral_node: str = ""
    # Live observer state (snapshotted by the observer for diagnostics).
    psi_r_alpha_Wb: float = 0.0
    psi_r_beta_Wb:  float = 0.0
    last_T_em_Nm:   float = 0.0
    last_slip:      float = 0.0

    @property
    def sigma(self) -> float:
        """Leakage factor σ = 1 − Lm² / (Ls·Lr) ∈ (0, 1)."""
        return 1.0 - (self.L_m_H ** 2) / (self.L_s_H * self.L_r_H)

    @property
    def L_sigma_s_H(self) -> float:
        """Stator transient inductance σ·Ls."""
        return self.sigma * self.L_s_H

    @property
    def rotor_time_constant_s(self) -> float:
        """Tr = Lr / Rr — rotor electrical time constant."""
        return self.L_r_H / self.R_r_ohm


def add_induction_motor(builder,
                            *,
                            name: str = "M1",
                            phase_nodes,
                            neutral_node: str,
                            R_s: float,
                            L_s: float,
                            R_r: float,
                            L_r: float,
                            L_m: float,
                            pole_pairs: int,
                            J: float,
                            B: float = 0.0,
                            T_load: float = 0.0,
                            T_load_fn: Optional[
                                Callable[[float, float], float]] = None,
                            ) -> InductionMotor:
    """Add a 3-phase squirrel-cage induction motor.

    Each phase contributes ``R_s + σ·L_s`` in series with a modulated
    "back-EMF" source whose voltage is computed by
    :func:`make_induction_motor_observer` from the rotor-flux dynamics.

    Parameters
    ----------
    phase_nodes
        3-element sequence of phase terminal node names.
    neutral_node
        Star-point neutral.
    R_s, L_s
        Stator per-phase resistance and self-inductance.
    R_r, L_r
        Rotor per-phase resistance and self-inductance — values
        REFERRED to the stator side (turns-ratio squared already
        applied). Match the standard IEEE locked-rotor / no-load
        equivalent-circuit definition.
    L_m
        Mutual inductance (referred). Constraint: ``L_m² < L_s · L_r``
        (otherwise the leakage factor σ would be ≤ 0, unphysical).
    pole_pairs
        Number of pole pairs (pp). ω_e = pp · ω_m.
    J, B, T_load, T_load_fn
        Mechanical parameters (see :func:`add_dc_motor`).

    Returns
    -------
    motor : :class:`InductionMotor`
        Dataclass handle to pass to
        :func:`make_induction_motor_observer`.

    Notes
    -----
    * Use :func:`im_parameters_from_nameplate` if you only have
      datasheet values (rated voltage / power / efficiency / cos φ)
      and need the equivalent-circuit R/L set automatically.
    * Numerical caveat: at exactly synchronous speed (slip = 0) the
      rotor flux dynamics decay-only with no driving term — set a
      non-zero initial rotor flux (via the ``Mechanical`` initial
      conditions or a brief startup excitation) for steady-state
      operation tests.
    """
    if len(phase_nodes) != 3:
        raise ValueError("phase_nodes must have 3 entries")
    if L_m * L_m >= L_s * L_r:
        raise ValueError(
            f"Unphysical leakage: L_m²={L_m**2:.3e} >= "
            f"L_s·L_r={L_s*L_r:.3e}. Adjust the mutual inductance "
            f"so the leakage factor σ = 1 − Lm²/(Ls·Lr) stays in (0, 1).")
    sigma = 1.0 - (L_m * L_m) / (L_s * L_r)
    L_sigma_s = sigma * L_s     # stator transient inductance

    mech = Mechanical(J_kgm2=J, B_Nms_per_rad=B,
                          T_load_Nm=T_load, T_load_fn=T_load_fn)

    phase_branch_ids = []
    bemf_source_ids = []
    for k, p_node in enumerate(phase_nodes):
        mid_r = f"{name}_R_mid_{('a','b','c')[k]}"
        mid_l = f"{name}_L_mid_{('a','b','c')[k]}"
        builder.add_resistor(f"{name}_Rs_{('a','b','c')[k]}",
                                p_node, mid_r, float(R_s))
        ind_id = builder.graph.num_branches
        builder.add_inductor(f"{name}_Lsig_{('a','b','c')[k]}",
                                mid_r, mid_l, float(L_sigma_s))
        phase_branch_ids.append(ind_id)
        bemf_id = builder.graph.num_branches
        builder.add_voltage_source(
            f"{name}_E_{('a','b','c')[k]}",
            mid_l, neutral_node, 0.0)
        bemf_source_ids.append(bemf_id)

    return InductionMotor(
        R_s_ohm=R_s, L_s_H=L_s, R_r_ohm=R_r, L_r_H=L_r, L_m_H=L_m,
        pole_pairs=pole_pairs, mech=mech,
        phase_branch_ids=tuple(phase_branch_ids),
        bemf_source_ids=tuple(bemf_source_ids),
        neutral_node=neutral_node,
    )


def make_induction_motor_observer(builder,
                                       motor: InductionMotor,
                                       *,
                                       dt: float):
    """Build a (step_observer, b_extra_fn) pair for an induction motor.

    The observer maintains the rotor flux (ψ_rα, ψ_rβ) as internal
    state and updates the mechanical (ω_m, θ_m) by forward-Euler each
    simulation step.  The induced stator EMF is computed from the
    instantaneous rotor-flux derivative and injected into each phase's
    dummy voltage source via ``b_extra``.
    """
    state_size = builder.pool.state_size(builder.graph)
    phase_idx = tuple(
        builder.pool.branch_var_id_for_inductor(bid, builder.graph)
        for bid in motor.phase_branch_ids)
    src_idx = tuple(
        builder.pool.branch_var_id_for_source(sid, builder.graph)
        for sid in motor.bemf_source_ids)

    inv_Tr = motor.R_r_ohm / motor.L_r_H      # 1 / Tr
    Lm_over_Lr = motor.L_m_H / motor.L_r_H
    pp = motor.pole_pairs
    sqrt3_over_2 = math.sqrt(3.0) / 2.0

    # Mutable closure state — observer's "extra" state beyond the
    # MNA state vector. Stored as a dict for cheap rebinding.
    state = {
        "psi_r_alpha": 0.0,
        "psi_r_beta":  0.0,
        "e_alpha":     0.0,
        "e_beta":      0.0,
    }

    def step_observer(t, x):
        i_a = float(x[phase_idx[0]])
        i_b = float(x[phase_idx[1]])
        i_c = float(x[phase_idx[2]])
        # Amplitude-invariant Clarke 3φ → αβ (drops zero-sequence;
        # OK because we treat the motor as 3-wire Y-connected).
        i_alpha = (2.0 / 3.0) * (i_a - 0.5 * i_b - 0.5 * i_c)
        i_beta = (2.0 / 3.0) * sqrt3_over_2 * (i_b - i_c)

        omega_m = motor.mech.omega_rad_s
        omega_e = pp * omega_m

        psi_ra = state["psi_r_alpha"]
        psi_rb = state["psi_r_beta"]

        # Rotor flux dynamics in the STATIONARY αβ frame:
        # dψ_rα/dt = −ψ_rα/Tr + (Lm/Tr)·i_sα − ω_e·ψ_rβ
        # dψ_rβ/dt = −ψ_rβ/Tr + (Lm/Tr)·i_sβ + ω_e·ψ_rα
        dpsi_ra = (-inv_Tr * psi_ra
                       + (motor.L_m_H * inv_Tr) * i_alpha
                       - omega_e * psi_rb)
        dpsi_rb = (-inv_Tr * psi_rb
                       + (motor.L_m_H * inv_Tr) * i_beta
                       + omega_e * psi_ra)

        # Induced stator EMF reflected from rotor: e = (Lm/Lr)·dψ_r/dt.
        # This is the "back-EMF" the observer feeds to the dummy
        # voltage sources at each phase terminal.
        e_alpha = Lm_over_Lr * dpsi_ra
        e_beta  = Lm_over_Lr * dpsi_rb

        # Electromagnetic torque (amplitude-invariant Clarke convention):
        # T_em = (3/2)·pp·(Lm/Lr)·(i_sβ·ψ_rα − i_sα·ψ_rβ)
        T_em = (1.5 * pp * Lm_over_Lr
                  * (i_beta * psi_ra - i_alpha * psi_rb))

        # Forward-Euler update of internal observer states.
        state["psi_r_alpha"] = psi_ra + dpsi_ra * dt
        state["psi_r_beta"]  = psi_rb + dpsi_rb * dt
        state["e_alpha"]      = e_alpha
        state["e_beta"]       = e_beta

        # Mechanical integration.
        motor.mech.integrate(t, T_em, dt)

        # Diagnostics snapshot (caller can read these between sim steps).
        motor.psi_r_alpha_Wb = state["psi_r_alpha"]
        motor.psi_r_beta_Wb  = state["psi_r_beta"]
        motor.last_T_em_Nm    = T_em
        # Slip = (ω_sync − ω_m) / ω_sync — undefined when ω_sync = 0;
        # report 1.0 (full slip) by convention in that case.
        if abs(omega_e) > 1e-6:
            # We don't know ω_sync directly without the source frequency,
            # so report fractional slip vs the instantaneous rotor-flux
            # rotation frequency, which equals ω_e at steady state.
            omega_psi = math.atan2(state["psi_r_beta"],
                                       state["psi_r_alpha"])
            del omega_psi   # placeholder for a future angular-velocity diff
            motor.last_slip = max(
                0.0,
                min(1.0,
                      (omega_e - omega_e) / max(abs(omega_e), 1e-9)))
        else:
            motor.last_slip = 1.0

    def b_extra_fn(t):
        out = [0.0] * state_size
        # Inverse Clarke αβ → abc (amplitude-invariant).
        e_alpha = state["e_alpha"]
        e_beta  = state["e_beta"]
        e_a = e_alpha
        e_b = -0.5 * e_alpha + sqrt3_over_2 * e_beta
        e_c = -0.5 * e_alpha - sqrt3_over_2 * e_beta
        # Sign convention: voltage source row in the residual already
        # carries +V_source; the observer must inject −V to drive the
        # EMF in the loop direction. Same convention as PMSM.
        out[src_idx[0]] = -e_a
        out[src_idx[1]] = -e_b
        out[src_idx[2]] = -e_c
        return out

    return step_observer, b_extra_fn


def im_parameters_from_nameplate(*,
                                       P_rated_W: float,
                                       V_LL_V: float,
                                       f_Hz: float,
                                       eta: float = 0.85,
                                       cos_phi: float = 0.85,
                                       pole_pairs: int = 2,
                                       inertia_TC_s: float = 0.5,
                                       ) -> dict:
    """Convert datasheet info to an :class:`InductionMotor` parameter
    starting point. The result is a ``dict`` of keyword arguments
    suitable for :func:`add_induction_motor` (plus a diagnostics
    sub-dict).

    This is a heuristic — accurate impedance values require a
    locked-rotor + no-load test or a manufacturer-provided
    equivalent-circuit. The heuristic splits the per-phase impedance
    using NEMA-design B rules of thumb (R_s : R_r : X_σs : X_σr : X_m
    ≈ 0.10 : 0.05 : 0.08 : 0.08 : 0.95 of the rated impedance).

    Parameters
    ----------
    P_rated_W
        Nameplate output power.
    V_LL_V
        Line-to-line RMS voltage (Y-connection assumed).
    f_Hz
        Supply frequency.
    eta
        Estimated efficiency (default 0.85). Used to back-compute
        the input current from output power.
    cos_phi
        Estimated power factor (default 0.85).
    pole_pairs
        Number of pole pairs (default 2 — i.e. a 4-pole machine).
    inertia_TC_s
        Approximate mechanical time constant (J·ω_sync / T_rated)
        used to back out a plausible J. Default 0.5 s.

    Returns
    -------
    params : dict
        Kwargs: ``R_s``, ``L_s``, ``R_r``, ``L_r``, ``L_m``,
        ``pole_pairs``, ``J``. Plus ``_diagnostics`` carrying the
        derived intermediate quantities (rated current, rated
        torque, synchronous speed).

    Examples
    --------
    >>> kw = im_parameters_from_nameplate(P_rated_W=4000.0,
    ...                                          V_LL_V=400.0,
    ...                                          f_Hz=50.0)
    >>> # Append topology arguments and call add_induction_motor.
    >>> # motor = add_induction_motor(builder, name="M1",
    >>> #                                phase_nodes=("a", "b", "c"),
    >>> #                                neutral_node="n",
    >>> #                                T_load=0.0,
    >>> #                                **{k: v for k, v in kw.items()
    >>> #                                    if not k.startswith("_")})
    """
    if P_rated_W <= 0 or V_LL_V <= 0 or f_Hz <= 0:
        raise ValueError("P_rated_W, V_LL_V, f_Hz must be positive.")
    omega_e = 2.0 * math.pi * f_Hz
    omega_s = omega_e / pole_pairs                # synchronous mech speed
    # Per-phase Y voltage and rated stator current.
    V_phase = V_LL_V / math.sqrt(3.0)
    I_rated = P_rated_W / (math.sqrt(3.0) * V_LL_V * eta * cos_phi)
    Z_phase = V_phase / I_rated                   # rated per-phase impedance
    # NEMA-B heuristic split.
    R_s = 0.10 * Z_phase * cos_phi
    R_r = 0.05 * Z_phase * cos_phi
    X_sigma_s = 0.08 * Z_phase
    X_sigma_r = 0.08 * Z_phase
    X_m = 0.95 * Z_phase
    L_sigma_s = X_sigma_s / omega_e
    L_sigma_r = X_sigma_r / omega_e
    L_m = X_m / omega_e
    L_s = L_sigma_s + L_m
    L_r = L_sigma_r + L_m
    # Inertia from desired mechanical time constant.
    T_rated = P_rated_W / omega_s
    J = max(1e-6, inertia_TC_s * T_rated / max(omega_s, 1e-3))
    return {
        "R_s": R_s, "L_s": L_s,
        "R_r": R_r, "L_r": L_r, "L_m": L_m,
        "pole_pairs": pole_pairs,
        "J": J,
        "_diagnostics": {
            "I_rated_A": I_rated,
            "T_rated_Nm": T_rated,
            "omega_sync_rad_s": omega_s,
            "Z_phase_ohm": Z_phase,
        },
    }
