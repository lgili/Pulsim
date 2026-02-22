"""GUI component parity smoke and behavioral checks."""

from __future__ import annotations

import textwrap
from time import perf_counter

import pytest

import pulsim as ps


GUI_COMPONENT_SMOKE_CASES: list[tuple[str, str]] = [
    (
        "BJT_NPN",
        """
- type: BJT_NPN
  name: QN1
  nodes: [base, in, out]
  beta: 80
""",
    ),
    (
        "BJT_PNP",
        """
- type: BJT_PNP
  name: QP1
  nodes: [base, in, out]
  beta: 80
""",
    ),
    (
        "THYRISTOR",
        """
- type: THYRISTOR
  name: SCR1
  nodes: [gate, in, 0]
  gate_threshold: 1.0
  holding_current: 0.05
  latch_current: 0.1
""",
    ),
    (
        "TRIAC",
        """
- type: TRIAC
  name: TRI1
  nodes: [gate, in, 0]
  gate_threshold: 1.0
  holding_current: 0.05
  latch_current: 0.1
""",
    ),
    (
        "SWITCH",
        """
- type: SWITCH
  name: S1
  nodes: [in, sw]
  g_on: 1e4
  g_off: 1e-9
""",
    ),
    (
        "FUSE",
        """
- type: FUSE
  name: F1
  nodes: [in, fuse_out]
  rating: 5
  blow_i2t: 2
""",
    ),
    (
        "CIRCUIT_BREAKER",
        """
- type: CIRCUIT_BREAKER
  name: B1
  nodes: [in, breaker_out]
  trip_current: 5
  trip_time: 1e-3
""",
    ),
    (
        "RELAY",
        """
- type: RELAY
  name: K1
  nodes: [coil_p, coil_n, com, no, nc]
  pickup_voltage: 6
  dropout_voltage: 3
  contact_resistance: 0.05
  off_resistance: 1e9
""",
    ),
    (
        "OP_AMP",
        """
- type: OP_AMP
  name: A1
  nodes: [in, out, ctrl]
  open_loop_gain: 1e5
  rail_low: -12
  rail_high: 12
""",
    ),
    (
        "COMPARATOR",
        """
- type: COMPARATOR
  name: CMP1
  nodes: [in, out, cmp_out]
  threshold: 0.0
  hysteresis: 0.2
  high: 5.0
  low: 0.0
""",
    ),
    (
        "PI_CONTROLLER",
        """
- type: PI_CONTROLLER
  name: PI1
  nodes: [in, out, pi_out]
  kp: 1.0
  ki: 10.0
  output_min: 0.0
  output_max: 1.0
""",
    ),
    (
        "PID_CONTROLLER",
        """
- type: PID_CONTROLLER
  name: PID1
  nodes: [in, out, pid_out]
  kp: 1.0
  ki: 10.0
  kd: 0.1
  output_min: 0.0
  output_max: 1.0
""",
    ),
    (
        "MATH_BLOCK",
        """
- type: MATH_BLOCK
  name: MATH1
  nodes: [in, out]
  operation: add
""",
    ),
    (
        "PWM_GENERATOR",
        """
- type: PWM_GENERATOR
  name: PWM1
  nodes: [ctrl]
  frequency: 10000
  duty: 0.4
""",
    ),
    (
        "INTEGRATOR",
        """
- type: INTEGRATOR
  name: INT1
  nodes: [in, out]
  output_min: -2.0
  output_max: 2.0
""",
    ),
    (
        "DIFFERENTIATOR",
        """
- type: DIFFERENTIATOR
  name: DIFF1
  nodes: [in, out]
  alpha: 0.5
""",
    ),
    (
        "LIMITER",
        """
- type: LIMITER
  name: LIM1
  nodes: [in, out]
  min: -1.0
  max: 1.0
""",
    ),
    (
        "RATE_LIMITER",
        """
- type: RATE_LIMITER
  name: RL1
  nodes: [in, out]
  rising_rate: 10
  falling_rate: 10
""",
    ),
    (
        "HYSTERESIS",
        """
- type: HYSTERESIS
  name: HYS1
  nodes: [in, out]
  threshold: 0.0
  hysteresis: 0.5
  high: 1.0
  low: -1.0
""",
    ),
    (
        "LOOKUP_TABLE",
        """
- type: LOOKUP_TABLE
  name: LUT1
  nodes: [in, out]
  x: [0, 1, 2]
  y: [0, 10, 20]
""",
    ),
    (
        "TRANSFER_FUNCTION",
        """
- type: TRANSFER_FUNCTION
  name: TF1
  nodes: [in, out]
  num: [0.5, 0.5]
  den: [1.0, -0.5]
""",
    ),
    (
        "DELAY_BLOCK",
        """
- type: DELAY_BLOCK
  name: DEL1
  nodes: [in, out]
  delay: 1e-6
""",
    ),
    (
        "SAMPLE_HOLD",
        """
- type: SAMPLE_HOLD
  name: SH1
  nodes: [in, out]
  sample_period: 1e-6
""",
    ),
    (
        "STATE_MACHINE",
        """
- type: STATE_MACHINE
  name: SM1
  nodes: [ctrl]
  mode: toggle
  threshold: 0.5
""",
    ),
    (
        "SATURABLE_INDUCTOR",
        """
- type: SATURABLE_INDUCTOR
  name: LSAT1
  nodes: [in, sat]
  inductance: 1m
  saturation_current: 2
  saturation_inductance: 100u
""",
    ),
    (
        "COUPLED_INDUCTOR",
        """
- type: COUPLED_INDUCTOR
  name: K1
  nodes: [p1, p2, s1, s2]
  l1: 1m
  l2: 2m
  coupling: 0.95
""",
    ),
    (
        "SNUBBER_RC",
        """
- type: SNUBBER_RC
  name: SN1
  nodes: [in, out]
  resistance: 100
  capacitance: 1n
""",
    ),
    (
        "VOLTAGE_PROBE",
        """
- type: VOLTAGE_PROBE
  name: VP1
  nodes: [out, 0]
""",
    ),
    (
        "CURRENT_PROBE",
        """
- type: CURRENT_PROBE
  name: IP1
  nodes: [in, 0]
  target_component: V1
""",
    ),
    (
        "POWER_PROBE",
        """
- type: POWER_PROBE
  name: PP1
  nodes: [in, 0]
  target_component: V1
""",
    ),
    (
        "ELECTRICAL_SCOPE",
        """
- type: ELECTRICAL_SCOPE
  name: SCOPE_E
  nodes: [in, out, 0]
""",
    ),
    (
        "THERMAL_SCOPE",
        """
- type: THERMAL_SCOPE
  name: SCOPE_T
  nodes: [in]
""",
    ),
    (
        "SIGNAL_MUX",
        """
- type: SIGNAL_MUX
  name: MUX1
  nodes: [in, out, ctrl]
  select_index: 1
""",
    ),
    (
        "SIGNAL_DEMUX",
        """
- type: SIGNAL_DEMUX
  name: DMX1
  nodes: [in, out_a, out_b]
""",
    ),
]


def _build_component_yaml(component_entry: str) -> str:
    base = textwrap.dedent(
        """
        schema: pulsim-v1
        version: 1
        simulation:
          tstart: 0
          tstop: 2e-6
          dt: 1e-6
        components:
          - type: voltage_source
            name: V1
            nodes: [in, 0]
            waveform: {type: dc, value: 5}
          - type: resistor
            name: R1
            nodes: [in, out]
            value: 1k
        """
    ).strip()
    entry = textwrap.indent(textwrap.dedent(component_entry).strip(), "  ")
    return f"{base}\n{entry}\n"


@pytest.mark.parametrize("component_type,component_entry", GUI_COMPONENT_SMOKE_CASES)
def test_gui_component_smoke_circuits_parse_without_unsupported_type(
    component_type: str, component_entry: str
) -> None:
    parser = ps.YamlParser()
    circuit, _ = parser.load_string(_build_component_yaml(component_entry))

    unsupported = [msg for msg in parser.errors if "Unsupported component type" in msg]
    assert unsupported == [], f"{component_type}: {unsupported}"
    assert parser.errors == [], f"{component_type}: {parser.errors}"
    assert circuit.num_devices() + circuit.num_virtual_components() >= 1


def test_gui_component_catalog_regression_gate_has_no_unsupported_types() -> None:
    unsupported_types: list[str] = []

    for component_type, component_entry in GUI_COMPONENT_SMOKE_CASES:
        parser = ps.YamlParser()
        parser.load_string(_build_component_yaml(component_entry))
        if any("Unsupported component type" in msg for msg in parser.errors):
            unsupported_types.append(component_type)

    assert unsupported_types == []


def test_family_behavioral_reference_coverage() -> None:
    circuit = ps.Circuit()
    n_ctrl = circuit.add_node("ctrl")
    n_in = circuit.add_node("in")
    gnd = circuit.ground()

    circuit.add_voltage_source("V1", n_in, gnd, 5.0)
    circuit.add_switch("F_SW", n_in, gnd, True, 1e3, 1e-9)
    circuit.add_virtual_component(
        "fuse",
        "F_EVT",
        [n_in, gnd],
        {"g_on": 1e3, "blow_i2t": 0.5, "initial_closed": 1.0},
        {"target_component": "F_SW"},
    )
    circuit.add_virtual_component(
        "pi_controller",
        "PI1",
        [n_in, gnd, n_ctrl],
        {"kp": 0.0, "ki": 1.0, "output_min": 0.0, "output_max": 1.0},
        {},
    )
    circuit.add_virtual_component("voltage_probe", "VP1", [n_in, gnd], {}, {})

    x = [0.0] * circuit.system_size()
    x[n_in] = 5.0
    circuit.execute_mixed_domain_step(x, 0.0)
    step = circuit.execute_mixed_domain_step(x, 1e-3)

    assert step.channel_values["F_EVT.state"] == 0.0
    assert 0.0 <= step.channel_values["PI1"] <= 1.0
    assert step.channel_values["VP1"] == 5.0


def test_switching_family_behavioral_reference() -> None:
    circuit = ps.Circuit()
    n_gate = circuit.add_node("gate")
    n_main = circuit.add_node("main")
    gnd = circuit.ground()

    circuit.add_switch("SCR_SW", n_main, gnd, False, 1e3, 1e-9)
    circuit.add_virtual_component(
        "thyristor",
        "SCR_EVT",
        [n_gate, n_main, gnd],
        {"gate_threshold": 1.0, "holding_current": 0.1, "latch_current": 0.5, "g_on": 1e3},
        {"target_component": "SCR_SW"},
    )

    x = [0.0] * circuit.system_size()
    x[n_gate] = 2.0
    x[n_main] = 1.0
    on_step = circuit.execute_mixed_domain_step(x, 1e-6)
    assert on_step.channel_values["SCR_EVT.state"] == 1.0

    x[n_gate] = 0.0
    x[n_main] = 5e-5
    off_step = circuit.execute_mixed_domain_step(x, 2e-6)
    assert off_step.channel_values["SCR_EVT.state"] == 0.0


def test_magnetic_family_behavioral_reference() -> None:
    circuit = ps.Circuit()
    n_p = circuit.add_node("p")
    n_s = circuit.add_node("s")
    gnd = circuit.ground()
    circuit.add_inductor("K1__L1", n_p, gnd, 1e-3, 0.0)
    circuit.add_inductor("K1__L2", n_s, gnd, 4e-3, 0.0)
    circuit.add_virtual_component(
        "coupled_inductor",
        "K1",
        [n_p, gnd, n_s, gnd],
        {"l1": 1e-3, "l2": 4e-3, "coupling": 0.9},
        {"target_component_1": "K1__L1", "target_component_2": "K1__L2"},
    )

    br1 = circuit.num_nodes()
    br2 = circuit.num_nodes() + 1
    x = [0.0] * circuit.system_size()
    x[n_p] = 1.0
    x[n_s] = -1.0
    x[br1] = 0.8
    x[br2] = -0.4
    jacobian, _ = circuit.assemble_jacobian(x)
    assert abs(jacobian[br1][br2]) > 1e-6
    assert abs(jacobian[br2][br1]) > 1e-6


def test_mixed_domain_smoke_runtime_has_bounded_overhead() -> None:
    base = ps.Circuit()
    n_in = base.add_node("in")
    gnd = base.ground()
    base.add_voltage_source("V1", n_in, gnd, 5.0)
    base.add_resistor("R1", n_in, gnd, 100.0)

    mixed = ps.Circuit()
    n_m_in = mixed.add_node("in")
    n_m_ctrl = mixed.add_node("ctrl")
    gnd_m = mixed.ground()
    mixed.add_voltage_source("V1", n_m_in, gnd_m, 5.0)
    mixed.add_resistor("R1", n_m_in, gnd_m, 100.0)
    mixed.add_virtual_component(
        "pi_controller",
        "PI1",
        [n_m_in, gnd_m, n_m_ctrl],
        {"kp": 0.5, "ki": 5.0, "output_min": 0.0, "output_max": 1.0},
        {},
    )
    mixed.add_virtual_component("voltage_probe", "VP1", [n_m_in, gnd_m], {}, {})
    mixed.add_virtual_component("state_machine", "SM1", [n_m_ctrl], {"threshold": 0.5}, {"mode": "toggle"})

    steps = 2000
    x_base = [0.0] * base.system_size()
    x_mixed = [0.0] * mixed.system_size()
    x_base[n_in] = 5.0
    x_mixed[n_m_in] = 5.0

    t0 = perf_counter()
    for i in range(steps):
        base.execute_mixed_domain_step(x_base, i * 1e-6)
    base_elapsed = perf_counter() - t0

    t1 = perf_counter()
    for i in range(steps):
        mixed.execute_mixed_domain_step(x_mixed, i * 1e-6)
    mixed_elapsed = perf_counter() - t1

    # Guardrail: mixed-domain bookkeeping should stay within one order of magnitude
    # for this small synthetic case.
    assert mixed_elapsed <= max(base_elapsed * 10.0, 0.5)
