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

    Saliency support (T2.1). For surface-mounted PM motors
    ``Ld_H == Lq_H == L_s_H``; for IPM / salient-pole rotors
    ``Ld_H ≠ Lq_H`` and the model captures the dq-frame reluctance
    torque ``T_rel = (3/2)·pp·(Ld−Lq)·i_d·i_q``. The abc topology
    is built around the AVERAGE inductance ``L_s_H = (Ld+Lq)/2`` so
    the high-frequency electrical impedance is correct on average;
    full anisotropic di/dt coupling lives only in the dq frame and
    is not modelled here. For most IPM control studies (FOC,
    flux-weakening, MTPA) the average-L + reluctance-torque model
    captures the dominant behaviour.
    """
    R_s_ohm: float
    L_s_H: float                          # average inductance (Ld+Lq)/2
    psi_pm_Wb: float                     # PM flux linkage [Wb]
    pole_pairs: int                       # pp
    mech: Mechanical
    # T2.1 — saliency parameters. Default to L_s for surface-PM.
    Ld_H: float = 0.0
    Lq_H: float = 0.0
    # T2.1 — initial dq-axis currents (seed via inverse Park into the
    # phase inductors' i0= before the kernel runs).
    i_d_init_A: float = 0.0
    i_q_init_A: float = 0.0
    theta_init_rad: float = 0.0
    # Filled in by add_pmsm:
    phase_branch_ids: tuple = field(default_factory=tuple)   # 3 inductor ids
    bemf_source_ids:  tuple = field(default_factory=tuple)   # 3 source ids
    neutral_node: str = ""

    @property
    def is_salient(self) -> bool:
        """True when ``Ld_H ≠ Lq_H`` (within float-precision tolerance)."""
        return abs(self.Ld_H - self.Lq_H) > 1e-12 * max(
            abs(self.Ld_H), abs(self.Lq_H), 1e-30)


def _inverse_park_to_abc(i_d: float, i_q: float, theta_e: float
                            ) -> "tuple[float, float, float]":
    """Inverse Park + amplitude-invariant inverse Clarke from
    ``(i_d, i_q)`` at electrical angle ``theta_e`` to phase
    currents ``(i_a, i_b, i_c)``. Used to seed initial conditions on
    the phase inductors when ``i_d_init`` / ``i_q_init`` are
    non-zero.

    Convention matches the FOC chain and the
    :class:`MotorObserverBundle` dq computation:

    * Park: ``[d; q] = [cos θ, sin θ; −sin θ, cos θ] · [α; β]``
    * Inverse Park: ``[α; β] = [cos θ, −sin θ; sin θ, cos θ] · [d; q]``
    * Inverse Clarke (amplitude-invariant): ``i_a = α``,
      ``i_b = −½α + (√3/2)β``, ``i_c = −½α − (√3/2)β``.
    """
    cos_t = math.cos(theta_e)
    sin_t = math.sin(theta_e)
    i_alpha = i_d * cos_t - i_q * sin_t
    i_beta = i_d * sin_t + i_q * cos_t
    half = 0.5
    sqrt3_half = math.sqrt(3.0) / 2.0
    i_a = i_alpha
    i_b = -half * i_alpha + sqrt3_half * i_beta
    i_c = -half * i_alpha - sqrt3_half * i_beta
    return (i_a, i_b, i_c)


def add_pmsm(builder,
                *,
                name: str = "M1",
                phase_nodes,
                neutral_node: str,
                R_s: float,
                L_s: Optional[float] = None,
                Ld: Optional[float] = None,
                Lq: Optional[float] = None,
                psi_pm: float,
                pole_pairs: int,
                J: float,
                B: float = 0.0,
                T_load: float = 0.0,
                T_load_fn=None,
                i_d_init: float = 0.0,
                i_q_init: float = 0.0,
                theta_init: float = 0.0,
                ) -> PMSM:
    """Add a 3-phase PMSM. Each phase: R_s + L_s + back-EMF source
    in series between ``phase_nodes[k]`` and ``neutral_node``.

    Parameters
    ----------
    phase_nodes
        3-element sequence of phase terminal node names.
    neutral_node
        Star-point neutral.
    R_s
        Per-phase stator resistance (Ω).
    L_s, Ld, Lq
        Stator inductance specification. Use either ``L_s=`` (surface
        PM / non-salient, single value) OR ``Ld=, Lq=`` (salient /
        IPM, distinct dq-axis values). Combining both raises
        ``ValueError``. When ``Ld != Lq`` the abc topology uses the
        average ``L_s = (Ld + Lq) / 2``; the observer publishes the
        reluctance torque ``T_rel = (3/2)·pp·(Ld−Lq)·i_d·i_q`` on
        top of the magnet torque ``(3/2)·pp·ψ_pm·i_q``.

        Caveat: the di/dt anisotropy along dq is NOT modelled in
        abc — the high-frequency response uses the average L only.
        For most IPM control work (FOC, flux-weakening, MTPA) the
        reluctance-torque term is the dominant saliency effect and
        this approximation is adequate. Full dq-frame reformulation
        is a v2 follow-up.
    psi_pm
        PM flux linkage in Wb. Back-EMF peak = pp · ψ_pm · ω_m.
    pole_pairs
        Number of pole pairs (pp). ω_e = pp · ω_m.
    J, B, T_load, T_load_fn
        Mechanical parameters (see :func:`add_dc_motor`).
    i_d_init, i_q_init
        Initial dq-axis stator currents (A). Inverse-Park'd to abc
        at electrical angle ``pp · theta_init`` and seeded into the
        three phase inductors via ``i0=``. Useful for starting a
        simulation from a non-zero FOC operating point (e.g. an MTPA
        steady-state) without waiting for a 5-τ electrical startup
        transient.
    theta_init
        Initial mechanical rotor angle (rad). Default 0.0. The
        observer's ``mech.theta_rad`` is initialised to this value
        so ``θ_e(t=0) = pp · theta_init``.
    """
    if len(phase_nodes) != 3:
        raise ValueError("phase_nodes must have 3 entries")

    # ----- T2.1: resolve inductance specification ----------------------
    if L_s is not None and (Ld is not None or Lq is not None):
        raise ValueError(
            "add_pmsm: pass either `L_s=` (single stator inductance for "
            "surface-PM) OR `Ld=, Lq=` (salient/IPM), not both.")
    if (Ld is None) != (Lq is None):
        raise ValueError(
            "add_pmsm: `Ld` and `Lq` must be set together (both or "
            "neither).")
    if Ld is not None and Lq is not None:
        Ld_H = float(Ld)
        Lq_H = float(Lq)
        L_s_avg = 0.5 * (Ld_H + Lq_H)
    elif L_s is not None:
        L_s_avg = float(L_s)
        Ld_H = L_s_avg
        Lq_H = L_s_avg
    else:
        raise ValueError(
            "add_pmsm: provide either `L_s=` or `Ld=, Lq=` — no "
            "stator inductance specified.")

    mech = Mechanical(J_kgm2=J, B_Nms_per_rad=B,
                          T_load_Nm=T_load, T_load_fn=T_load_fn)
    # Seed the mechanical angle so θ_e(t=0) = pp · theta_init.
    mech.theta_rad = float(theta_init)

    # Inverse-Park the dq ICs to abc once for the t=0 seed.
    theta_e0 = float(pole_pairs) * float(theta_init)
    i_a0, i_b0, i_c0 = _inverse_park_to_abc(
        float(i_d_init), float(i_q_init), theta_e0)
    i_abc_init = (i_a0, i_b0, i_c0)

    phase_branch_ids = []
    bemf_source_ids = []
    for k, p_node in enumerate(phase_nodes):
        mid_r = f"{name}_R_mid_{('a','b','c')[k]}"
        mid_l = f"{name}_L_mid_{('a','b','c')[k]}"
        # Skip R if zero — keeps the graph simpler. Otherwise:
        builder.add_resistor(f"{name}_Rs_{('a','b','c')[k]}",
                                p_node, mid_r, float(R_s))
        ind_id = builder.graph.num_branches
        # T2.1: seed i0= only when the IC is non-trivial. Passing
        # i0=0.0 explicitly is equivalent but pollutes the kernel's
        # initial_state vector with zeros that the auto-default already
        # produces.
        ind_kwargs = {}
        if abs(i_abc_init[k]) > 1e-15:
            ind_kwargs["i0"] = i_abc_init[k]
        builder.add_inductor(f"{name}_Ls_{('a','b','c')[k]}",
                                mid_r, mid_l, L_s_avg,
                                **ind_kwargs)
        phase_branch_ids.append(ind_id)
        # Back-EMF source (0V initially; observer modulates).
        bemf_id = builder.graph.num_branches
        builder.add_voltage_source(
            f"{name}_E_{('a','b','c')[k]}",
            mid_l, neutral_node, 0.0)
        bemf_source_ids.append(bemf_id)

    motor = PMSM(
        R_s_ohm=R_s, L_s_H=L_s_avg, psi_pm_Wb=psi_pm,
        pole_pairs=pole_pairs, mech=mech,
        Ld_H=Ld_H, Lq_H=Lq_H,
        i_d_init_A=float(i_d_init), i_q_init_A=float(i_q_init),
        theta_init_rad=float(theta_init),
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

        # T2.1 — saliency reluctance torque
        # ``T_rel = (3/2) · pp · (L_d − L_q) · i_d · i_q``. Only fires
        # on salient rotors (Ld_H != Lq_H). The PMSM dataclass sets
        # both to the same value for surface-PM, so this branch is
        # cheap (a single subtract + abs compare) and zero-effect.
        if (waveform == "sinusoidal"
                and hasattr(motor, "is_salient")
                and motor.is_salient):
            # Clarke (amplitude-invariant) + Park to compute i_d, i_q.
            i_alpha = (2.0 / 3.0) * (i_a - 0.5 * i_b - 0.5 * i_c)
            i_beta = (i_b - i_c) / math.sqrt(3.0)
            cos_t = math.cos(theta_e)
            sin_t = math.sin(theta_e)
            i_d_now = i_alpha * cos_t + i_beta * sin_t
            i_q_now = -i_alpha * sin_t + i_beta * cos_t
            T_rel = (1.5 * motor.pole_pairs
                     * (motor.Ld_H - motor.Lq_H)
                     * i_d_now * i_q_now)
            T_em += T_rel

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
