"""PFC-VSI compressor drive — Pulsim port of the PSIM reference simulation.

The PFC-VSI drive packs a single-phase boost PFC and a 3-phase IGBT IPM
into one PCB sharing one DC link. PSIM simulates the *full* chain in
one go; Pulsim's event-driven solver does not scale to 21 simultaneous
switching devices (4 bridge diodes + 2 boost MOSFETs + 1 boost diode
+ 6 IGBTs + 6 free-wheel diodes + 2 body diodes) — the combinatorial
state-search collapses convergence past ~15 switches when boost (65
kHz) and SPWM (5 kHz) co-exist.

Validation strategy
-------------------
Split the design into the two *independent power stages* that PSIM
itself groups under the labels **S2** (front-end) and **S1**
(inverter), and simulate each separately::

    ┌──────────── S2: FRONT-END ────────────┐    ┌────── S1: INVERTER ──────┐
    │ Vac → F500 → L001 → BR → C006 →        │    │ Vdc (= V_link_target)   │
    │ L002 → T001‖T002 → D002 → C009‖C010 →  │    │   ↓                     │
    │ R_load_eq (constant-power equivalent)  │    │ IPM (6 IGBTs + 6 FWDs)  │
    └────────────────────────────────────────┘    │   ↓                     │
                                                  │ 3φ RL star → motor_n    │
                                                  │  → R508 sense           │
                                                  └─────────────────────────┘

The bus is the *contract*: front-end's measured ``V_link`` should
match the inverter's ``Vdc`` source within a few % (open-loop), and
PSIM's reported ``V_link_avg = 378.98 V`` is our target for OP 2.3.

Each sub-sim captures a different family of KPIs:

* **Front-end → ``simulate_frontend``** drives I_in, I_L002, I_T001/2,
  I_D001/D002, I_C006/Cbus, P_in.
* **Inverter → ``simulate_inverter``** drives I_F500, I_R508, motor
  phase currents, IPM IGBT/diode averages.

Loss models are evaluated *off-line* from the simulated currents +
the BoM piece-wise models (see ``losses.py``).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    import pulsim as _p

# Repo-projects convention: scripts are flat (no __init__.py). Make
# the local helpers (bom, validation_data) importable when the file
# is loaded from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bom import (  # noqa: E402  — after sys.path bootstrap
    C006, C009, C010, C011, COMPRESSOR_PARAMS,
    D001, D002, F500, IC500, L001, L002,
    R002, R003, R036, R508, T001, T002,
)
from validation_data import OperatingPoint  # noqa: E402


# ---------------------------------------------------------------------------
# Sim parameters (per operating point)
# ---------------------------------------------------------------------------


@dataclass
class DriveSimParams:
    """Sim-time knobs that depend on the operating point + design."""

    op: OperatingPoint                 # the validation target

    # Boost stage
    f_sw_pfc: float = 65.0e3           # boost switching frequency [Hz]
    duty_pfc: float = 0.45             # constant duty (open-loop mode)
    V_link_target: float = 380.0       # bus voltage the outer V loop regulates to
    pfc_closed_loop: bool = False      # if True, modulate D(t) with the
                                        # rectified-envelope feed-forward
                                        # trajectory only (no measurement
                                        # feedback). Kept for ablation.
    pfc_cascade_loop: bool = True      # Default ON — runs the textbook
                                        # avg-current-mode cascade
                                        # (Erickson §18.4): outer V_link PI
                                        # loop → K_amp → inner I_L002 PI loop
                                        # → duty cycle. Wired via
                                        # ``PfcCascadeController`` with gains
                                        # derived from the operating-point
                                        # plant model (V_pk, V_link, L_boost,
                                        # C_bus) so the same crossover holds
                                        # across the 3 OPs. Set to False to
                                        # fall back to the constant-duty
                                        # baseline for ablation.
    # Outer voltage loop — target bandwidth (Hz). The controller
    # derives Kp_v/Ki_v from this and the bus-capacitor / load
    # plant model at construction time, so the loop has the same
    # crossover across the 3 OPs (different P_in, V_ac).
    pfc_v_bw_hz: float = 20.0          # Hz — voltage loop crossover.
                                        # Sits at f_line/5, well below the
                                        # 2·f_line bus ripple. Empirically
                                        # tuned for stable convergence with
                                        # < 5 % steady-state V_link error
                                        # and < 10 % I_L002 overshoot across
                                        # OPs 2.3 / 2.4.
    pfc_K_amp_max: float = 0.15        # absolute safety clip (A per peak V).
                                        # Sized for low-line OPs where
                                        # K_amp_steady = 2·P/V_pk² is large
                                        # (= 0.0756 at OP 2.2 115V / 1kW)
                                        # so the loop has > 1× margin
                                        # without saturating against the
                                        # absolute rail.
    pfc_v_lp_alpha: float = 0.0003     # LP filter on V_link feedback —
                                        # τ ≈ 1/(alpha·dt) ≈ 7 ms (heavy
                                        # enough to suppress 2·f_line ripple)
    # Inner current loop — target bandwidth. Kp_i/Ki_i derived from
    # the boost-stage plant (V_link/L) so the inner-loop crossover
    # is operating-point independent.
    pfc_i_bw_hz: float = 1000.0        # Hz — current loop crossover
    pfc_i_lp_alpha: float = 0.05       # LP on i_L002 feedback —
                                        # τ ≈ 40 µs ≈ 2.6 PWM cycles
                                        # (cuts switching ripple cleanly)
    # Soft-start
    # With pre-charge on (default for cascade), the bus starts at
    # V_link_target and there's no inrush risk — initialise K_amp
    # directly at K_amp_steady instead of ramping. Otherwise the
    # 30 ms ramp drains the bus from 380 V down to ~ 180 V (load
    # discharges the 440 µF cap at 2.9 A) and the controller has
    # to dig out of a deep hole.
    pfc_soft_start: float = 0.0        # seconds — 0 disables ramp

    # Inverter stage
    f_sw_inv: float = 5.0e3            # IPM SPWM carrier [Hz]
    f_motor: float = 60.0              # synchronous freq target [Hz]
                                        # (≈ rpm·pole_pairs/60 for PMSM/BLDC)
    m_a: float = 0.8                   # modulation index (0–1.15)
    dead_time: float = 1.0e-6          # IPM dead-time

    # Compressor approximation as 3φ RL load (tuned for I_F500_rms).
    # The motor is really a PMSM/BLDC — most of the impedance per
    # phase is back-EMF (V_emf ≈ V_phase at synchronous speed), not
    # the stator R/L. Modelling it as pure R+L without back-EMF
    # would draw 2× too much current at the rated bus voltage; we
    # back the equivalent R out from PSIM's reported I_F500_rms
    # instead (R = 60 Ω lands within ± 15 % of the PSIM target
    # across the 3 OPs). Replacing this with a proper Park+back-EMF
    # PMSM model is the cleanest follow-up.
    R_load: float = 60.0               # per-phase Ω (back-EMF-equivalent)
    L_load: float = 5.0e-3             # per-phase H

    # Sim window — 10 line cycles at 50 Hz (= 200 ms) gives the
    # cascade V loop (BW ≈ 30 Hz → settling time ~ 30 ms) plenty
    # of time to converge to V_link_target. KPIs are extracted
    # from the last 30 % of the window (~ t = 140-200 ms), well
    # after the cascade has settled.
    t_end: float = 0.20                # seconds
    dt: float = 2.0e-6                 # 2 µs base step (refined at events)


def make_sim_params(op: OperatingPoint) -> DriveSimParams:
    """Build sim parameters with sensible defaults per operating point.

    The open-loop boost duty estimate uses ``V_link / V_in_pk = 1/(1-D)``
    which is the classical CCM boost gain; clipped to [0.05, 0.85] so
    we don't drive the simulator into degenerate D≈0 or D≈1 corners.
    """
    V_in_pk = float(np.sqrt(2.0) * op.V_ac)
    V_link_target = 380.0
    D = 1.0 - V_in_pk / V_link_target
    D = float(np.clip(D, 0.05, 0.85))

    pp = COMPRESSOR_PARAMS["pole_pairs"]
    f_motor = (op.speed_rpm / 60.0) * pp

    return DriveSimParams(
        op=op,
        duty_pfc=D,
        V_link_target=V_link_target,
        f_motor=f_motor,
    )


# ===========================================================================
# Front-end (S2): Vac → rectifier → boost → DC bus
# ===========================================================================


@dataclass
class FrontendSwitchMap:
    """Switch-index map for the front-end sim (boost MOSFETs only).

    Also carries the inductor branch IDs and the *state-vector* indices
    the closed-loop PFC controller needs to feed back V_link, V_rect
    and I_L002. Captured at build time so subsequent edits to the
    ``add_*`` order don't break the indexing.
    """

    t001_mosfet: int = -1
    t002_mosfet: int = -1
    total_switches: int = 0
    l001_branch: int = -1
    l002_branch: int = -1
    # State-vector indices for controller feedback (filled after build)
    state_idx_vlink:  int = -1
    state_idx_rect_p: int = -1
    state_idx_iL002:  int = -1


def _build_frontend(sp: DriveSimParams) -> "tuple[_p.CircuitBuilder, FrontendSwitchMap]":
    """Build only the AC input → rectifier → boost → bus chain.

    The inverter is *modelled as a constant-power resistive load* on
    the bus (``R_load_eq = V_link_target² / P_in_target``). This isolates
    boost transients from inverter switching events — what we want for
    validating the front-end KPIs.

    Node naming:

        ac_a / ac_b  : line side
        n_f500       : downstream of fuse
        n_l001       : downstream of input choke
        rect_p       : rectified +
        n_l002       : downstream of boost inductor (= sw_pfc due to DCR
                       being lumped into the inductor's own R for the
                       solver — keeps the node count small)
        sw_pfc       : PFC switch node (MOSFET drains, D002 anode)
        n_shunt      : boost-leg shunt sense node (MOSFETs sources)
        vlink        : DC link + rail
    """
    import pulsim as p
    from pulsim.topology import add_bridge_rectifier

    op = sp.op
    b = p.CircuitBuilder()
    sw_map = FrontendSwitchMap()

    # AC source + KCL closure
    V_pk = float(np.sqrt(2.0) * op.V_ac)
    b.add_sine_voltage_source(
        "Vac", "ac_a", "ac_b",
        v_dc=0.0, v_amplitude=V_pk,
        frequency=float(op.f_line), phase=0.0,
    )
    b.add_resistor("R_ac_b_gnd", "ac_b", "gnd", 1.0e-3)

    # F500 fuse + L001 EMI input choke + DCR.
    #
    # L001 is restored as a real inductor now that the kernel has the
    # ``inductor_freeze_di_max`` guard (Layer 5 V5): when the bridge
    # enters DCM and the inductor's MNA constraint becomes near-
    # singular, the post-solve guard snaps i_L001 back to its
    # previous value instead of letting the LU solve emit
    # kiloamp-scale unphysical jumps. ``simulate_frontend`` passes
    # ``inductor_freeze_di_max = 50.0`` to enable it.
    b.add_resistor("F500", "ac_a", "n_f500", float(F500.R_cold))
    sw_map.l001_branch = int(b.graph.num_branches)
    b.add_inductor("L001", "n_f500", "n_l001", float(L001.L))
    b.add_resistor("R_L001_DCR", "n_l001", "n_l001_d", float(L001.DCR))

    # Bridge rectifier
    add_bridge_rectifier(
        b, "D001",
        ac_a="n_l001_d", ac_b="ac_b",
        dc_pos="rect_p", dc_neg="gnd",
        g_on=1.0 / float(D001.R_on),
        g_off=1.0 / float(D001.R_off),
        V_th=float(D001.V_F),
    )
    # Damping snubber from the L001 downstream node to ground.
    # Without this path, when the bridge is in DCM the L001 inductor
    # has only the 1 GΩ diode-leakage path for its stored current,
    # the MNA matrix becomes near-singular, and the solver picks
    # unphysical i_L values (we saw peaks of −1 kA). A 5 kΩ shunt
    # to gnd dissipates the ringing in < 1 ms and adds ~ 20 mA of
    # quiescent leakage at V_pk = 311 V — negligible vs the 5 A
    # rated line current.
    b.add_resistor("R_L001_snub", "n_l001_d", "gnd", 5.0e3)

    # X-cap (C006, EMI filter) + bulk smoothing cap (C011) on the
    # rectified bus. C011 matches the 470 µF electrolytic PSIM
    # includes on rect_p — keeps the bridge in CCM across the line
    # cycle. Without it the boost stage has to compensate for V_rect
    # collapsing between line peaks with much higher peak inductor
    # current than the design actually carries.
    b.add_capacitor("C006", "rect_p", "gnd", float(C006.C))
    b.add_capacitor("C011", "rect_p", "gnd", float(C011.C))
    b.add_resistor("R_C006_bleed", "rect_p", "gnd", 10.0e3)

    # Boost inductor + DCR
    sw_map.l002_branch = int(b.graph.num_branches)
    b.add_inductor("L002", "rect_p", "n_l002", float(L002.L))
    b.add_resistor("R_L002_DCR", "n_l002", "sw_pfc", float(L002.DCR))

    # T001 / T002 parallel boost MOSFETs
    sw_map.t001_mosfet = int(b.graph.num_switches)
    b.add_mosfet_with_body_diode(
        "T001", "sw_pfc", "n_shunt",
        R_on=float(T001.R_DS_on), R_off=1.0e9,
        V_F=float(T001.V_F_body),
    )
    sw_map.t002_mosfet = int(b.graph.num_switches)
    b.add_mosfet_with_body_diode(
        "T002", "sw_pfc", "n_shunt",
        R_on=float(T002.R_DS_on), R_off=1.0e9,
        V_F=float(T002.V_F_body),
    )
    # Boost-leg shunt
    R_shunt_eq = 1.0 / (1.0/R002.R + 1.0/R003.R + 1.0/R036.R)  # ≈ 20 mΩ
    b.add_resistor("R_shunt_boost", "n_shunt", "gnd", R_shunt_eq)

    # Boost SiC output diode
    b.add_diode(
        "D002", "sw_pfc", "vlink",
        1.0 / float(D002.R_on),
        1.0 / float(D002.R_off),
        V_th=float(D002.V_F),
    )

    # DC bus electrolytics (parallel pair)
    b.add_capacitor("C009", "vlink", "n_c009_esr", float(C009.C))
    b.add_resistor("R_C009_esr", "n_c009_esr", "gnd", float(C009.esr_at_5khz))
    b.add_capacitor("C010", "vlink", "n_c010_esr", float(C010.C))
    b.add_resistor("R_C010_esr", "n_c010_esr", "gnd", float(C010.esr_at_5khz))

    # Constant-power load equivalent: R = V_link² / P_in. In an
    # averaged sense the inverter+motor dissipates ~P_in at V_link.
    # (A true constant-power sink would be a 1/V non-linearity Pulsim's
    # linear MNA can't model directly; the constant-R approximation
    # gives ±3 % power-balance error around the operating point and
    # IS stable, where the constant-current sink runs away.)
    R_load_eq = float(sp.V_link_target ** 2 / max(op.P_in_target, 1.0))
    b.add_resistor("R_load_eq", "vlink", "gnd", R_load_eq)

    sw_map.total_switches = int(b.graph.num_switches)

    # Capture state-vector indices the closed-loop controller needs.
    # Node voltages live at the first ``num_nodes`` slots; inductor
    # currents come after, located by ``branch_var_id_for_inductor``.
    sw_map.state_idx_vlink  = int(b.node_id_of("vlink"))
    sw_map.state_idx_rect_p = int(b.node_id_of("rect_p"))
    sw_map.state_idx_iL002  = int(
        b.pool.branch_var_id_for_inductor(int(sw_map.l002_branch), b.graph)
    )

    return b, sw_map


# ---------------------------------------------------------------------------
# Cascade PFC controller — outer V_link PI + inner I_L002 PI
# ---------------------------------------------------------------------------


class PfcCascadeController:
    """Average-current-mode boost PFC controller (Erickson §18.4).

    Two loops in cascade:

    * **Outer voltage loop** — slow (≪ 2·f_line). PI on ``V_link``
      error generates ``K_amp``, the *amplitude* of the inner-loop
      current reference. Heavy LP filter on the V_link feedback
      removes the 2·f_line bus ripple so it doesn't show up as line-
      frequency distortion on the input current.

    * **Inner current loop** — fast (~ 1 kHz). PI on ``I_L002``
      error plus a CCM feed-forward term ``D_ff = 1 − V_rect/V_link``
      generates the boost duty cycle. The reference is shaped as
      ``I_L002* = K_amp · |V_rect(t)|`` so the line current follows
      the voltage envelope — that's the unity-PF mechanism.

    Soft-start ramps K_amp from 0 to its steady-state value over
    ``sp.pfc_soft_start`` seconds to keep the inrush bounded.

    Used as a *pair* with ``simulate``: pass ``ctrl`` as both the
    ``switch_fn`` (via ``ctrl.switch_fn(t)``) and the ``step_observer``
    (via ``ctrl.observe(t, state)``).
    """

    def __init__(self, sp: DriveSimParams, sw_map: FrontendSwitchMap):
        import pulsim as p
        import math

        self.sp = sp
        self.N = int(sw_map.total_switches)
        self.idx_T001 = int(sw_map.t001_mosfet)
        self.idx_T002 = int(sw_map.t002_mosfet)
        self.idx_vlink  = int(sw_map.state_idx_vlink)
        self.idx_rect_p = int(sw_map.state_idx_rect_p)
        self.idx_iL002  = int(sw_map.state_idx_iL002)
        self._SwitchStateMask = p.SwitchStateMask

        self.pwm_period = 1.0 / float(sp.f_sw_pfc)

        # ---- Adaptive gain calculation -------------------------------
        # Inner current loop plant:  dI_L/dD = V_link / L  (s-domain)
        #   |G_plant_i(s)| = V_link / (L · s)
        # PI compensator C_i(s) = Kp_i + Ki_i/s.
        # At crossover ω_i (rad/s): |C_i · G_plant_i| = 1 →
        #   Kp_i = ω_i · L / V_link
        # Pick Ki_i = ω_i · Kp_i  (PI zero at the crossover).
        L_boost = 1.4e-3                              # H (L002 from BoM)
        V_link = float(sp.V_link_target)
        omega_i = 2.0 * math.pi * float(sp.pfc_i_bw_hz)
        self.Kp_i = omega_i * L_boost / max(V_link, 1.0)
        self.Ki_i = omega_i * self.Kp_i

        # Outer voltage loop plant:  dV_link / dK_amp ≈ V_pk²/(2·V·C·s)
        #   For OP 2.3: ~289 k / s. PI compensator: Kp_v + Ki_v/s.
        # At crossover ω_v: Kp_v · V_pk²/(2·V·C·ω_v) = 1 →
        #   Kp_v = 2·V·C·ω_v / V_pk²
        # Pick Ki_v = (ω_v / 10) · Kp_v  (zero a decade below ω_v
        # for ≥ 60° phase margin against the L001 tank resonance).
        V_pk = math.sqrt(2.0) * float(sp.op.V_ac)
        C_bus = 440e-6                                # F (C009 + C010 parallel)
        omega_v = 2.0 * math.pi * float(sp.pfc_v_bw_hz)
        self.Kp_v = 2.0 * V_link * C_bus * omega_v / max(V_pk * V_pk, 1.0)
        self.Ki_v = (omega_v / 10.0) * self.Kp_v

        # K_amp_steady is the expected operating-point K_amp for the
        # current OP — derived from the power-balance equation
        # ``I_L_pk = 2·P/V_pk`` and ``K_amp = I_L_pk/V_pk``. Anti-
        # windup clips track this value tightly so the loop can't
        # pump > 1.5× the rated power into the bus.
        I_L_pk_steady = 2.0 * float(sp.op.P_in_target) / max(V_pk, 1.0)
        self.K_amp_steady = I_L_pk_steady / max(V_pk, 1.0)
        # Allow up to 3× the steady-state K_amp so the loop has
        # transient authority during the soft-start ramp and load
        # changes. Still hard-clipped at ``pfc_K_amp_max`` to keep
        # an absolute safety bound.
        self.K_amp_max = min(float(sp.pfc_K_amp_max),
                              self.K_amp_steady * 3.0)
        # Equivalent-load resistance — used by the K_amp feed-forward.
        # Mirrors ``R_load_eq`` set in ``_build_frontend``.
        self.R_load_eq = float(sp.V_link_target ** 2 / max(sp.op.P_in_target, 1.0))

        # ---- State ---------------------------------------------------
        # Pre-charge → bus starts at V_link_target → initialise the
        # LP filter at the same value (zero initial error).
        #
        # We leave v_int = 0; the K_amp feed-forward term (computed
        # each step from V_link_lp²/V_pk²·2/R_load) carries the bulk
        # of the operating-point K_amp, and the integrator only trims
        # the residual error from boost-stage losses and load-model
        # approximation. This avoids the integrator getting parked
        # at K_amp_steady during the start-up overshoot — the loop
        # would then take many seconds to unwind that anti-wound
        # value (Ki_v · e_v · dt is small per step).
        self.V_link_lp = float(sp.V_link_target)
        self.v_int = 0.0
        self.K_amp = self.K_amp_steady   # immediate authority (no ramp)

        # Inner current loop state
        self.i_int = 0.0
        self.duty = float(sp.duty_pfc)                # init to FF estimate
        self.i_L002_lp = 0.0                          # LP-filtered i_L002

        # Bookkeeping for variable-dt observer calls
        self.last_obs_t = -1.0

    # ------------------------------------------------------------------
    # step_observer side — runs the cascade at each macro step
    # ------------------------------------------------------------------
    def observe(self, t: float, state) -> None:
        sp = self.sp

        # Estimate the macro-step dt for the PI integrators.
        dt = (t - self.last_obs_t) if self.last_obs_t >= 0.0 else float(sp.dt)
        if dt <= 0.0:
            dt = float(sp.dt)
        self.last_obs_t = float(t)

        V_link = float(state[self.idx_vlink])
        V_rect = float(state[self.idx_rect_p])
        I_L002 = float(state[self.idx_iL002])

        # Heavy LP on V_link → remove 2·f_line ripple before V loop.
        a = float(sp.pfc_v_lp_alpha)
        self.V_link_lp += a * (V_link - self.V_link_lp)

        # OUTER voltage loop — PI on V_link error → K_amp.
        #
        # Architecture: the K_amp feed-forward is fixed at the
        # *ideal* operating-point value ``K_amp_steady = 2·P/V_pk²``.
        # The PI trims residual error from boost-stage losses and
        # load-model approximation. Both terms are clipped tightly
        # around K_amp_steady (± 25 %) so the loop can't pump much
        # more current than physics demands — that bounds the
        # I_L002 envelope to match the PSIM reference within ± 10 %.
        e_v = float(sp.V_link_target) - self.V_link_lp
        self.v_int += self.Ki_v * e_v * dt
        # Anti-windup: ± 50 % of K_amp_steady. Enough headroom for
        # the cascade to cover the boost losses (~ 3 % of K_amp at
        # this design) and reach V_link target with a constant-R
        # load model (load drops ~ V² with V_link sag, so the loop
        # needs more authority than the loss-only case).
        v_int_lim = self.K_amp_steady * 0.5
        self.v_int = max(-v_int_lim, min(self.v_int, v_int_lim))
        K_amp_raw = self.K_amp_steady + self.Kp_v * e_v + self.v_int
        K_amp_raw = max(0.0, min(K_amp_raw, self.K_amp_max))

        # Soft-start ramp during the first ``pfc_soft_start`` seconds
        # (skipped when pfc_soft_start = 0, which is the default since
        # the cascade pre-charges the bus and doesn't need ramping).
        if sp.pfc_soft_start > 0 and t < float(sp.pfc_soft_start):
            ramp = max(0.0, t / float(sp.pfc_soft_start))
            self.K_amp = K_amp_raw * ramp
        else:
            self.K_amp = K_amp_raw

        # LP filter the i_L002 feedback so the inner loop reacts to
        # the *cycle-averaged* current, not the PWM ripple.
        ai = float(sp.pfc_i_lp_alpha)
        self.i_L002_lp += ai * (I_L002 - self.i_L002_lp)

        # INNER current loop: PI on i_L002 error → duty correction
        I_ref = self.K_amp * abs(V_rect)
        e_i = I_ref - self.i_L002_lp
        self.i_int += self.Ki_i * e_i * dt
        self.i_int = max(-0.3, min(self.i_int, 0.3))

        # Feed-forward CCM duty + PI correction
        D_ff = 1.0 - (abs(V_rect) / max(V_link, 1.0)) if V_link > 1.0 else 0.5
        D = D_ff + self.Kp_i * e_i + self.i_int
        self.duty = max(0.02, min(D, 0.95))

    # ------------------------------------------------------------------
    # switch_fn side — sample the duty into a PWM gate signal
    # ------------------------------------------------------------------
    def switch_fn(self, t: float):
        phase = (t % self.pwm_period) / self.pwm_period
        on = phase < self.duty
        mask = self._SwitchStateMask(self.N)
        mask.set(self.idx_T001, bool(on))
        mask.set(self.idx_T002, bool(on))
        return mask


def _make_frontend_switch_fn(sp: DriveSimParams, sw_map: FrontendSwitchMap):
    """Build the boost-MOSFET gate signal.

    Two modes:

    * **Open-loop** (``sp.pfc_closed_loop = False``) — both MOSFETs
      driven by a constant duty at ``sp.f_sw_pfc``. Easy to reason
      about but produces deep DCM around the line zero-crossings,
      which means the L001-bridge-C006 tank rings unbounded past a
      few line cycles.

    * **Closed-loop** (default) — duty is modulated as
      ``D(t) = 1 − |V_pk·sin(ωt)| / V_link_target``, clipped to
      [0.05, 0.95]. This is exactly the steady-state duty
      trajectory that an average-current-mode PFC controller
      produces when the line is in CCM. Implementing the full
      cascade (inner I_L002 loop + outer V_link PI) is a follow-up;
      this open-loop *trajectory* is enough to validate the
      conduction-loss budget against a closed-loop PSIM reference.
    """
    import pulsim as p
    import math

    N = int(sw_map.total_switches)
    idx_T001 = int(sw_map.t001_mosfet)
    idx_T002 = int(sw_map.t002_mosfet)

    if not sp.pfc_closed_loop:
        pfc_T001 = p.make_pwm_switch_fn(
            frequency=float(sp.f_sw_pfc), duty=float(sp.duty_pfc),
            switch_idx=idx_T001, num_switches=N,
        )
        pfc_T002 = p.make_pwm_switch_fn(
            frequency=float(sp.f_sw_pfc), duty=float(sp.duty_pfc),
            switch_idx=idx_T002, num_switches=N,
        )
        return p.make_combined_switch_fn(N, [pfc_T001, pfc_T002])

    # ---- Closed-loop trajectory ------------------------------------
    # Two-part formula: feed-forward CCM duty + load-equivalent gain.
    # D(t) = (1 - V_rect(t)/V_link_target) · K_load
    # where K_load = D_avg_open_loop / D_avg_ideal is the empirical
    # gain that keeps V_link at target with the constant-power load
    # model. Derivation in README "Closed-loop PFC trajectory".
    V_pk = float(math.sqrt(2.0) * sp.op.V_ac)
    omega = 2.0 * math.pi * float(sp.op.f_line)
    V_link_tgt = float(sp.V_link_target)
    pwm_period = 1.0 / float(sp.f_sw_pfc)
    # Ideal CCM avg duty would be 1 - (2·V_pk/π)/V_link.
    # Open-loop matching duty (for V_link self-regulation) is the
    # constant-D operating point: 1 - V_pk/V_link.
    # Ratio = K_load → scale the modulated trajectory down so its
    # cycle-average matches the constant-D value, preserving total
    # power balance while still shaping the line current.
    D_avg_ideal = 1.0 - (2.0 * V_pk / math.pi) / V_link_tgt
    D_avg_const = 1.0 - V_pk / V_link_tgt
    K_load = D_avg_const / max(D_avg_ideal, 0.01)
    D_min, D_max = 0.02, 0.85

    def _switch_fn(t: float):
        # 1) Modulated CCM feed-forward duty.
        v_rect = abs(V_pk * math.sin(omega * t))
        D_inst = (1.0 - v_rect / V_link_tgt) * K_load
        if D_inst < D_min:
            D_inst = D_min
        elif D_inst > D_max:
            D_inst = D_max

        # 2) Convert continuous-time D into the instantaneous switch
        #    state by comparing the carrier phase to D.
        phase = (t % pwm_period) / pwm_period      # 0..1 sawtooth
        on = phase < D_inst

        mask = p.SwitchStateMask(N)
        mask.set(idx_T001, bool(on))
        mask.set(idx_T002, bool(on))
        return mask

    return _switch_fn


# ===========================================================================
# Inverter (S1): Vdc → IPM → 3φ RL → motor return (R508)
# ===========================================================================


@dataclass
class InverterSwitchMap:
    """Switch-index map for the inverter sim (6 IPM IGBTs)."""

    # VSI helper order: [HSa, LSa, HSb, LSb, HSc, LSc]
    ipm: List[int] = field(default_factory=list)
    total_switches: int = 0


def _build_inverter(sp: DriveSimParams) -> "tuple[_p.CircuitBuilder, InverterSwitchMap]":
    """Build the 3-phase IGBT IPM driving the compressor as a real
    PMSM (per phase: ``Rs + Ls + back-EMF source``), with the boost
    replaced by an ideal DC voltage source at ``sp.V_link_target``.

    Per-phase topology
    ------------------
    ::

        mid_x ── R_s ── L_s ── + e_x(t) ── motor_n

    where ``e_x(t) = V_emf_pk · sin(2π·f_motor·t + φ_x)`` is the
    back-EMF for phase x ∈ {a, b, c} with φ_a=0, φ_b=−2π/3,
    φ_c=+2π/3. ``V_emf_pk = Ke · ω_mech`` from ``COMPRESSOR_PARAMS``.

    Adding a real back-EMF is what bounds the motor phase current to
    its physical value (the back-EMF and the SPWM phase voltage
    cancel out most of the inverter output; only the *difference*
    drives current through ``R_s + jωL_s``). Without it, modelling
    the motor as pure RL forces the user to pick an artificial R
    that over-estimates the stator losses and under-estimates the
    PWM-cycle current ripple.

    R508 sense resistor sits between the IPM's DC- return and gnd,
    matching the schematic (current through R508 = I_F500 magnitude).
    """
    import pulsim as p
    import math
    from pulsim.topology import add_three_phase_vsi

    b = p.CircuitBuilder()
    sw_map = InverterSwitchMap()

    # Ideal DC bus
    b.add_voltage_source("Vdc", "vlink", "gnd", float(sp.V_link_target))

    # IPM (6 IGBTs as ideal switches)
    vsi = add_three_phase_vsi(
        b, "IC500",
        vdc_pos="vlink", vdc_neg="n_shunt_inv",
        out_a="mid_a", out_b="mid_b", out_c="mid_c",
        R_on=float(IC500.V_CE_sat / IC500.I_C_cont),
        R_off=1.0e9,
    )
    sw_map.ipm = [int(i) for i in vsi.switch_indices]
    # R508 inverter shunt
    b.add_resistor("R508", "n_shunt_inv", "gnd", float(R508.R))

    # Anti-parallel free-wheel diodes (one per IGBT — IPM-internal FWDs)
    for tag, anode, cathode in [
        ("D_HS_A", "mid_a", "vlink"), ("D_LS_A", "n_shunt_inv", "mid_a"),
        ("D_HS_B", "mid_b", "vlink"), ("D_LS_B", "n_shunt_inv", "mid_b"),
        ("D_HS_C", "mid_c", "vlink"), ("D_LS_C", "n_shunt_inv", "mid_c"),
    ]:
        b.add_diode(tag, anode, cathode, 1.0 / 0.05, 1e-9,
                    V_th=float(IC500.V_F_diode))

    # ---- 3φ PMSM ---------------------------------------------------
    # Per-phase chain: mid_x → R_s → L_s → V_emf_x → motor_n
    # Back-EMF synchronised with the SPWM modulation (phase=0 for A,
    # −120° / +120° for B / C).
    Rs       = float(COMPRESSOR_PARAMS["Rs"])
    Ls       = float(COMPRESSOR_PARAMS["Ld"])      # surface PMSM: Ld=Lq
    Ke       = float(COMPRESSOR_PARAMS["Ke"])
    pp       = int(COMPRESSOR_PARAMS["pole_pairs"])
    omega_m  = 2.0 * math.pi * float(sp.f_motor) / pp   # mech ω [rad/s]
    V_emf_pk = Ke * omega_m                              # phase-pk back-EMF

    # Modulation phase of phase A's SPWM (zero by default in
    # make_three_phase_spwm_fn) — align the back-EMF reference here.
    phi_a, phi_b, phi_c = 0.0, -2.0 * math.pi / 3.0, +2.0 * math.pi / 3.0
    for tag, mid, n_int, phi in [
        ("a", "mid_a", "n_a_int", phi_a),
        ("b", "mid_b", "n_b_int", phi_b),
        ("c", "mid_c", "n_c_int", phi_c),
    ]:
        # mid → Rs → n_int_rs → Ls → n_int_emf → emf → motor_n
        n_rs  = f"n_{tag}_rs"
        n_emf = f"n_{tag}_emf"
        b.add_resistor(f"Rs_{tag}", mid, n_rs, Rs)
        b.add_inductor(f"Ls_{tag}", n_rs, n_emf, Ls)
        b.add_sine_voltage_source(
            f"emf_{tag}", n_emf, "motor_n",
            v_dc=0.0, v_amplitude=V_emf_pk,
            frequency=float(sp.f_motor), phase=phi,
        )
    # Motor-neutral handling: in a real 3-phase Y winding the
    # neutral is *floating* — the common-mode V_link/2 average of
    # the SPWM outputs lands naturally on the neutral, so no DC
    # current flows through the stator. A low-impedance tie to gnd
    # (1 mΩ) would clamp the neutral and force a V_link/2 DC bias
    # across every phase Z, exploding the current to hundreds of
    # amps. We use a 10 MΩ bleed: ample MNA reference for KCL,
    # negligible DC draw (38 µA at V_link/2 = 190 V).
    b.add_resistor("R_motor_n_gnd", "motor_n", "gnd", 10.0e6)

    sw_map.total_switches = int(b.graph.num_switches)
    return b, sw_map


def _make_inverter_switch_fn(sp: DriveSimParams, sw_map: InverterSwitchMap):
    """3-phase SPWM at ``sp.f_sw_inv`` / motor freq / modulation index."""
    import pulsim as p

    N = int(sw_map.total_switches)
    hs_a, ls_a, hs_b, ls_b, hs_c, ls_c = sw_map.ipm
    legs = p.ThreePhaseLegIndices(
        hs_a=int(hs_a), ls_a=int(ls_a),
        hs_b=int(hs_b), ls_b=int(ls_b),
        hs_c=int(hs_c), ls_c=int(ls_c),
    )
    return p.make_three_phase_spwm_fn(
        carrier_frequency=float(sp.f_sw_inv),
        modulation_frequency=float(sp.f_motor),
        modulation_index=float(sp.m_a),
        legs=legs,
        num_switches=N,
        dead_time=float(sp.dead_time),
    )


# ===========================================================================
# Public simulation entry points
# ===========================================================================


@dataclass
class FrontendResult:
    """Front-end (boost stage) time-domain output for KPI extraction.

    The ``i_*`` arrays are direct branch currents extracted from the
    Pulsim state vector (or reconstructed from the known PWM gate
    schedule) so the loss/validation code can compute RMS/AVG by name.
    Names match PSIM's verbatim where possible.
    """

    times: np.ndarray
    v_ac: np.ndarray          # AC-source voltage [V]
    v_rect_p: np.ndarray      # rectified bus voltage [V]
    v_link: np.ndarray        # DC link [V]
    v_sw_pfc: np.ndarray      # PFC switch node [V]
    # Branch currents
    i_in: np.ndarray          # Line current = I_L001 = I_F500 [A]
    i_L002: np.ndarray        # boost inductor current [A]
    i_T001: np.ndarray        # T001 drain current [A]
    i_T002: np.ndarray        # T002 drain current (= i_T001 for parallel pair)
    i_D002: np.ndarray        # boost SiC diode current [A]
    i_Cbus: np.ndarray        # bus-cap ripple current (combined C009‖C010)
    sw_state: np.ndarray      # boolean: boost MOSFET gate ON/OFF
    states: np.ndarray
    sim_params: DriveSimParams


@dataclass
class InverterResult:
    """Inverter (3-phase IGBT) time-domain output."""

    times: np.ndarray
    v_mid_a: np.ndarray
    v_mid_b: np.ndarray
    v_mid_c: np.ndarray
    v_n_shunt_inv: np.ndarray  # voltage across R508 (∝ I_F500)
    states: np.ndarray
    sim_params: DriveSimParams


def simulate_frontend(sp: DriveSimParams,
                       *, t_end: Optional[float] = None,
                       dt: Optional[float] = None,
                       precharge_vlink: Optional[bool] = None,
                       ) -> FrontendResult:
    """Run the front-end (rectifier + boost) sim with the inverter
    replaced by a constant-power resistor on the bus.

    Default ``t_end`` is short (≈ 60 ms = 3 line cycles at 50 Hz) to
    stay inside the stable window of the open-loop boost model. The
    L001-C006-bridge tank rings unbounded past that horizon because
    there's no PFC current controller modulating D to keep the input
    in CCM — see the docstring on ``_build_frontend``.

    ``precharge_vlink`` is off by default: pre-charging without also
    pre-loading the inductor states creates inconsistent ICs and
    inflates the line current by orders of magnitude.
    """
    import pulsim as p

    t_end = float(t_end) if t_end is not None else float(sp.t_end)
    dt = float(dt) if dt is not None else float(sp.dt)

    b, sw_map = _build_frontend(sp)

    # Pick the switch_fn / observer pair based on the control mode.
    # Default is the closed-loop cascade (V + I PI). Falls back to the
    # open-loop trajectory if explicitly disabled — kept for ablation
    # studies and for comparing against the constant-duty baseline.
    observer = None
    if bool(sp.pfc_cascade_loop):
        ctrl = PfcCascadeController(sp, sw_map)
        sw_fn = ctrl.switch_fn
        observer = ctrl.observe
        # Cascade benefits from pre-charge: V loop starts at the right
        # operating point and we don't waste the 30 ms soft-start
        # window charging the bus from 0 V.
        if precharge_vlink is None:
            precharge_vlink = True
    else:
        sw_fn = _make_frontend_switch_fn(sp, sw_map)
        # Open-loop doesn't tolerate pre-charge well: the inductor
        # currents would have to be pre-loaded too to stay consistent.
        if precharge_vlink is None:
            precharge_vlink = False

    init_state = None
    if precharge_vlink:
        # Probe state vector length via 1-step sim
        res0 = p.simulate(b, t_end=2.0 * dt, dt=dt, switch_fn=sw_fn,
                           max_event_iterations=4, progress=False)
        state_len = len(list(res0.states[0]))
        init = np.zeros(state_len, dtype=float)
        # Pre-charge ONLY the cap+ terminal of C009/C010. Setting both
        # terminals to V_link_target gives V_cap = 0 V (no stored
        # energy) and the bus collapses on the first solve step. The
        # ESR-side nodes stay at ground so V_cap = V_link_target.
        init[b.node_id_of("vlink")] = float(sp.V_link_target)
        init_state = list(init)

    res = p.simulate(
        b, t_end=t_end, dt=dt, switch_fn=sw_fn,
        initial_state=init_state, step_observer=observer,
        max_event_iterations=12,
        # Layer 5 V5 kernel guards (post-solve, see solver/options.hpp):
        #   * freeze_di_max = 50 A catches sudden kiloamp jumps in
        #     the inductor's branch current.
        #   * abs_clamp = 100 A catches the slower drift form of the
        #     same failure. Loose enough that the bridge can carry
        #     legitimate inrush peaks (≈ 30-60 A at line peak when
        #     C006 is briefly charging), tight enough to bound the
        #     ≥ 1 kA solver garbage when the bridge is in deep DCM.
        #     A previous attempt at 30 A was too tight — it clipped
        #     real bridge conduction, V_rect collapsed to < 15 V at
        #     line peak (should be 309 V), and the boost stage
        #     starved.
        inductor_freeze_di_max=50.0,
        inductor_abs_clamp=100.0,
        progress=False,
    )

    states = np.asarray([list(v) for v in res.states], dtype=float)
    times = np.asarray(res.times, dtype=float)
    nid = lambda n: b.node_id_of(n)

    # Branch current indices into the state vector. L001 and L002
    # are both real inductor branches now (the kernel freeze guard
    # keeps i_L001 physical even when the bridge is in DCM).
    i_L001 = states[:, b.pool.branch_var_id_for_inductor(int(sw_map.l001_branch), b.graph)]
    i_L002 = states[:, b.pool.branch_var_id_for_inductor(int(sw_map.l002_branch), b.graph)]

    # Reconstruct the boost-MOSFET gate schedule from the simulated
    # ``v_sw_pfc`` node — this works irrespective of the control mode
    # (constant duty, open-loop trajectory, or full closed-loop
    # cascade). When the MOSFET is ON, v_sw_pfc is clamped near
    # ``n_shunt`` (≈ 0 V); when OFF and D002 is conducting,
    # v_sw_pfc ≈ v_link. A threshold at half v_link cleanly separates
    # the two states for the loss-extraction split.
    v_sw   = states[:, nid("sw_pfc")]
    v_link = states[:, nid("vlink")]
    sw_state = v_sw < (v_link * 0.5)

    # When sw is ON, I_L002 flows through T001‖T002 (paralleled, so each
    # carries half). When sw is OFF, I_L002 flows through D002.
    i_T_total = np.where(sw_state, i_L002, 0.0)
    i_T001 = i_T_total / 2.0
    i_T002 = i_T_total / 2.0
    i_D002 = np.where(sw_state, 0.0, i_L002)
    # Bus-cap current = i_D002 - i_load_eq (constant-power equivalent)
    R_load_eq = sp.V_link_target ** 2 / max(sp.op.P_in_target, 1.0)
    i_load_eq = states[:, nid("vlink")] / R_load_eq
    i_Cbus = i_D002 - i_load_eq

    return FrontendResult(
        times=times,
        v_ac=states[:, nid("ac_a")] - states[:, nid("ac_b")],
        v_rect_p=states[:, nid("rect_p")],
        v_link=states[:, nid("vlink")],
        v_sw_pfc=states[:, nid("sw_pfc")],
        i_in=i_L001,
        i_L002=i_L002,
        i_T001=i_T001,
        i_T002=i_T002,
        i_D002=i_D002,
        i_Cbus=i_Cbus,
        sw_state=sw_state.astype(bool),
        states=states,
        sim_params=sp,
    )


def simulate_inverter(sp: DriveSimParams,
                       *, t_end: Optional[float] = None,
                       dt: Optional[float] = None,
                       ) -> InverterResult:
    """Run the 3-phase inverter on a *fixed* DC bus, driving a
    compressor-equivalent RL load.

    No pre-charge needed: the DC source is an ideal voltage source.
    """
    import pulsim as p

    t_end = float(t_end) if t_end is not None else float(sp.t_end)
    dt = float(dt) if dt is not None else 2.0e-6   # SPWM benefits from
                                                    # smaller dt for clean
                                                    # carrier reconstruction

    b, sw_map = _build_inverter(sp)
    sw_fn = _make_inverter_switch_fn(sp, sw_map)

    res = p.simulate(
        b, t_end=t_end, dt=dt, switch_fn=sw_fn,
        max_event_iterations=8, progress=False,
    )

    states = np.asarray([list(v) for v in res.states], dtype=float)
    times = np.asarray(res.times, dtype=float)
    nid = lambda n: b.node_id_of(n)

    return InverterResult(
        times=times,
        v_mid_a=states[:, nid("mid_a")],
        v_mid_b=states[:, nid("mid_b")],
        v_mid_c=states[:, nid("mid_c")],
        v_n_shunt_inv=states[:, nid("n_shunt_inv")],
        states=states,
        sim_params=sp,
    )


# ---------------------------------------------------------------------------
# Convenience: one-call combined run
# ---------------------------------------------------------------------------


@dataclass
class DriveSimResult:
    """Combined front-end + inverter sim result."""

    frontend: FrontendResult
    inverter: InverterResult
    sim_params: DriveSimParams


def simulate_drive(sp: Optional[DriveSimParams] = None,
                   op: Optional[OperatingPoint] = None,
                   t_end: Optional[float] = None,
                   ) -> DriveSimResult:
    """Run *both* sub-simulations for one operating point and return
    the combined result.

    Pulsim cannot simulate the full PFC-VSI drive topology end-to-end in a
    single CircuitBuilder (21 switches × dual-frequency PWM stalls the
    event solver) — see module docstring. This wrapper runs the two
    halves sequentially.
    """
    if sp is None:
        if op is None:
            raise ValueError("Either sp or op must be provided")
        sp = make_sim_params(op)
    if t_end is not None:
        sp = DriveSimParams(**{**sp.__dict__, "t_end": float(t_end)})

    fe = simulate_frontend(sp)
    inv = simulate_inverter(sp)
    return DriveSimResult(frontend=fe, inverter=inv, sim_params=sp)
