"""Pulsim v2 — electromechanical motor models (Phase D).

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
    "add_dc_motor",
    "make_dc_motor_observer",
    "add_pmsm",
    "make_pmsm_observer",
    "add_bldc",
    "make_bldc_observer",
]


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


def make_dc_motor_observer(builder, motor: DcMotor, *, dt: float):
    """Build a (step_observer, b_extra_fn) pair for a DC motor.

    The observer:
      1. Reads i_armature from the state vector.
      2. Computes T_em = Kt · i.
      3. Integrates Mechanical by one step (forward-Euler).
      4. Stashes the next-step back-EMF voltage in a closure.

    The b_extra_fn injects the stashed back-EMF into the source's
    constraint row at every step.

    Parameters
    ----------
    builder
        The CircuitBuilder that already has the motor added.
    motor
        The `DcMotor` returned by :func:`add_dc_motor`.
    dt
        Mechanical integration step. Typically equal to the
        simulation dt.
    """
    state_size = builder.pool.state_size(builder.graph)
    i_idx = builder.pool.branch_var_id_for_inductor(
        motor.inductor_branch_id, builder.graph)
    src_idx = builder.pool.branch_var_id_for_source(
        motor.bemf_source_branch_id, builder.graph)

    bemf = {"V": 0.0}

    def step_observer(t, x):
        i_a = float(x[i_idx])
        T_em = motor.Kt_Nm_per_A * i_a
        motor.mech.integrate(t, T_em, dt)
        bemf["V"] = motor.Ke_V_s_per_rad * motor.mech.omega_rad_s

    def b_extra_fn(t):
        out = [0.0] * state_size
        # Convention: constraint row reads (V_from − V_to) − V_source = 0.
        # Adding +V_extra to b shifts V_source by −V_extra, i.e. the
        # source value becomes (0 − (−V_bemf)) = V_bemf. So inject
        # −V_bemf to get the right sign.
        out[src_idx] = -bemf["V"]
        return out

    return step_observer, b_extra_fn


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
                                    waveform: str):
    """Shared implementation for PMSM (sinusoidal) and BLDC
    (trapezoidal) back-EMF observers."""
    state_size = builder.pool.state_size(builder.graph)
    phase_idx = tuple(
        builder.pool.branch_var_id_for_inductor(bid, builder.graph)
        for bid in motor.phase_branch_ids)
    src_idx = tuple(
        builder.pool.branch_var_id_for_source(sid, builder.graph)
        for sid in motor.bemf_source_ids)

    bemf = {"v": (0.0, 0.0, 0.0)}

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

    def b_extra_fn(t):
        out = [0.0] * state_size
        for k in range(3):
            out[src_idx[k]] = -bemf["v"][k]
        return out

    return step_observer, b_extra_fn


def make_pmsm_observer(builder, motor: PMSM, *, dt: float):
    """Build a (step_observer, b_extra_fn) pair for a PMSM with
    sinusoidal back-EMF."""
    return _make_3phase_motor_observer(builder, motor, dt=dt,
                                            waveform="sinusoidal")


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


def make_bldc_observer(builder, motor: BLDC, *, dt: float):
    """Build a (step_observer, b_extra_fn) pair for a BLDC with
    trapezoidal back-EMF."""
    return _make_3phase_motor_observer(builder, motor, dt=dt,
                                            waveform="trapezoidal")
