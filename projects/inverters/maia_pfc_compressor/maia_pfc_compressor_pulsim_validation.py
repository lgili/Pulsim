"""Maia PFC FR — Pulsim port of the PSIM reference simulation.

The Maia drive packs a single-phase boost PFC and a 3-phase IGBT IPM
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
    C006, C009, C010, COMPRESSOR_PARAMS,
    D001, D002, F500, IC500, L001, L002,
    R002, R003, R036, R508, T001, T002,
)
from validation_data import OperatingPoint  # noqa: E402


# ---------------------------------------------------------------------------
# Sim parameters (per operating point)
# ---------------------------------------------------------------------------


@dataclass
class MaiaSimParams:
    """Sim-time knobs that depend on the operating point + design."""

    op: OperatingPoint                 # the validation target

    # Boost stage
    f_sw_pfc: float = 65.0e3           # boost switching frequency [Hz]
    duty_pfc: float = 0.45             # constant duty for open-loop run
    V_link_target: float = 380.0       # bus voltage the PFC loop would regulate to

    # Inverter stage
    f_sw_inv: float = 5.0e3            # IPM SPWM carrier [Hz]
    f_motor: float = 60.0              # synchronous freq target [Hz]
                                        # (≈ rpm·pole_pairs/60 for PMSM/BLDC)
    m_a: float = 0.8                   # modulation index (0–1.15)
    dead_time: float = 1.0e-6          # IPM dead-time

    # Compressor approximation as 3φ RL load (tuned for I_F500_rms)
    R_load: float = 60.0               # per-phase Ω
    L_load: float = 5.0e-3             # per-phase H

    # Sim window — 3 line cycles at 50 Hz (= 60 ms) sits inside the
    # open-loop stability horizon of the front-end (see
    # ``_build_frontend`` notes); 2 ms for the inverter is enough for
    # a fundamental cycle at 120 Hz with steady-state currents.
    t_end: float = 0.04                # seconds
    dt: float = 2.0e-6                 # 2 µs base step (refined at events)


def make_sim_params(op: OperatingPoint) -> MaiaSimParams:
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

    return MaiaSimParams(
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
    """Switch-index map for the front-end sim (boost MOSFETs only)."""

    t001_mosfet: int = -1
    t002_mosfet: int = -1
    total_switches: int = 0


def _build_frontend(sp: MaiaSimParams) -> "tuple[_p.CircuitBuilder, FrontendSwitchMap]":
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

    # F500 fuse + L001 input choke + L001 DCR
    # NB: this open-loop sim is stable for ≈ 60 ms; the L001-C006-bridge
    # tank rings unbounded after that because there's no PFC current
    # controller modulating D to keep the input in CCM. The KPIs are
    # therefore extracted from a 20–40 ms window before drift kicks in
    # (see ``simulate_frontend`` defaults).
    b.add_resistor("F500", "ac_a", "n_f500", float(F500.R_cold))
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

    # X-cap on rectified bus
    b.add_capacitor("C006", "rect_p", "gnd", float(C006.C))

    # Boost inductor + DCR
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

    # Constant-power load equivalent: R = V_link² / P_in
    # In an averaged sense the inverter+motor dissipates ~P_in at V_link.
    R_load_eq = float(sp.V_link_target ** 2 / max(op.P_in_target, 1.0))
    b.add_resistor("R_load_eq", "vlink", "gnd", R_load_eq)

    sw_map.total_switches = int(b.graph.num_switches)
    return b, sw_map


def _make_frontend_switch_fn(sp: MaiaSimParams, sw_map: FrontendSwitchMap):
    """PWM the two boost MOSFETs in lock-step at ``sp.f_sw_pfc / sp.duty_pfc``."""
    import pulsim as p

    N = int(sw_map.total_switches)
    pfc_T001 = p.make_pwm_switch_fn(
        frequency=float(sp.f_sw_pfc), duty=float(sp.duty_pfc),
        switch_idx=int(sw_map.t001_mosfet), num_switches=N,
    )
    pfc_T002 = p.make_pwm_switch_fn(
        frequency=float(sp.f_sw_pfc), duty=float(sp.duty_pfc),
        switch_idx=int(sw_map.t002_mosfet), num_switches=N,
    )
    return p.make_combined_switch_fn(N, [pfc_T001, pfc_T002])


# ===========================================================================
# Inverter (S1): Vdc → IPM → 3φ RL → motor return (R508)
# ===========================================================================


@dataclass
class InverterSwitchMap:
    """Switch-index map for the inverter sim (6 IPM IGBTs)."""

    # VSI helper order: [HSa, LSa, HSb, LSb, HSc, LSc]
    ipm: List[int] = field(default_factory=list)
    total_switches: int = 0


def _build_inverter(sp: MaiaSimParams) -> "tuple[_p.CircuitBuilder, InverterSwitchMap]":
    """Build the 3-phase IGBT IPM driving the compressor as a 3φ star
    RL load, with the boost replaced by an ideal DC voltage source at
    ``sp.V_link_target``.

    R508 sense resistor sits between the IPM's DC- return and ``gnd``,
    matching the schematic (current through R508 = I_F500 magnitude).
    """
    import pulsim as p
    from pulsim.topology import add_three_phase_vsi, add_three_phase_rl_load

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

    # Compressor as 3φ star RL
    add_three_phase_rl_load(
        b, "Motor",
        node_a="mid_a", node_b="mid_b", node_c="mid_c",
        node_neutral="motor_n",
        R=float(sp.R_load), L=float(sp.L_load),
        topology="star",
    )
    b.add_resistor("R_motor_n_gnd", "motor_n", "gnd", 1.0e-3)

    sw_map.total_switches = int(b.graph.num_switches)
    return b, sw_map


def _make_inverter_switch_fn(sp: MaiaSimParams, sw_map: InverterSwitchMap):
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
    Pulsim state vector — they match PSIM's signal names verbatim so
    the loss/validation code can compute RMS/AVG by name.
    """

    times: np.ndarray
    v_ac: np.ndarray          # AC-source voltage [V]
    v_rect_p: np.ndarray      # rectified bus voltage [V]
    v_link: np.ndarray        # DC link [V]
    v_sw_pfc: np.ndarray      # PFC switch node [V]
    # Branch currents
    i_in: np.ndarray          # Line current = I_L001 = I_F500 [A]
    i_L002: np.ndarray        # boost inductor current [A]
    states: np.ndarray
    sim_params: MaiaSimParams


@dataclass
class InverterResult:
    """Inverter (3-phase IGBT) time-domain output."""

    times: np.ndarray
    v_mid_a: np.ndarray
    v_mid_b: np.ndarray
    v_mid_c: np.ndarray
    v_n_shunt_inv: np.ndarray  # voltage across R508 (∝ I_F500)
    states: np.ndarray
    sim_params: MaiaSimParams


def simulate_frontend(sp: MaiaSimParams,
                       *, t_end: Optional[float] = None,
                       dt: Optional[float] = None,
                       precharge_vlink: bool = False,
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
    sw_fn = _make_frontend_switch_fn(sp, sw_map)

    init_state = None
    if precharge_vlink:
        # Probe state vector length via 1-step sim
        res0 = p.simulate(b, t_end=2.0 * dt, dt=dt, switch_fn=sw_fn,
                           max_event_iterations=4, progress=False)
        state_len = len(list(res0.states[0]))
        init = np.zeros(state_len, dtype=float)
        for n in ("vlink", "n_c009_esr", "n_c010_esr"):
            try:
                init[b.node_id_of(n)] = float(sp.V_link_target)
            except Exception:
                pass
        init_state = list(init)

    res = p.simulate(
        b, t_end=t_end, dt=dt, switch_fn=sw_fn,
        initial_state=init_state,
        max_event_iterations=12, progress=False,
    )

    states = np.asarray([list(v) for v in res.states], dtype=float)
    times = np.asarray(res.times, dtype=float)
    nid = lambda n: b.node_id_of(n)

    # Branch current indices into the state vector. Pulsim packs
    # source/inductor currents past the node-voltage block, and
    # ``branch_var_id_for_inductor`` returns the absolute state index.
    # L001 is the 4th add_* call → branch id 3; L002 is the 11th → id 10.
    i_L001 = states[:, b.pool.branch_var_id_for_inductor(3, b.graph)]
    i_L002 = states[:, b.pool.branch_var_id_for_inductor(10, b.graph)]

    return FrontendResult(
        times=times,
        v_ac=states[:, nid("ac_a")] - states[:, nid("ac_b")],
        v_rect_p=states[:, nid("rect_p")],
        v_link=states[:, nid("vlink")],
        v_sw_pfc=states[:, nid("sw_pfc")],
        i_in=i_L001,
        i_L002=i_L002,
        states=states,
        sim_params=sp,
    )


def simulate_inverter(sp: MaiaSimParams,
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
class MaiaSimResult:
    """Combined front-end + inverter sim result."""

    frontend: FrontendResult
    inverter: InverterResult
    sim_params: MaiaSimParams


def simulate_maia(sp: Optional[MaiaSimParams] = None,
                   op: Optional[OperatingPoint] = None,
                   t_end: Optional[float] = None,
                   ) -> MaiaSimResult:
    """Run *both* sub-simulations for one operating point and return
    the combined result.

    Pulsim cannot simulate the full Maia topology end-to-end in a
    single CircuitBuilder (21 switches × dual-frequency PWM stalls the
    event solver) — see module docstring. This wrapper runs the two
    halves sequentially.
    """
    if sp is None:
        if op is None:
            raise ValueError("Either sp or op must be provided")
        sp = make_sim_params(op)
    if t_end is not None:
        sp = MaiaSimParams(**{**sp.__dict__, "t_end": float(t_end)})

    fe = simulate_frontend(sp)
    inv = simulate_inverter(sp)
    return MaiaSimResult(frontend=fe, inverter=inv, sim_params=sp)
