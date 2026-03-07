"""Regression coverage for the full canonical backend component catalog."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pulsim as ps
import pytest


@dataclass(frozen=True)
class ComponentCase:
    category: str
    canonical_type: str
    component_entry: str
    support_components: str = ""


COMPONENT_CASES: tuple[ComponentCase, ...] = (
    ComponentCase(
        category="passive",
        canonical_type="resistor",
        support_components="""
- type: resistor
  name: RLOAD
  nodes: [vout, 0]
  value: 100
""",
        component_entry="""
- type: resistor
  name: RTEST
  nodes: [vin, vout]
  value: 10
""",
    ),
    ComponentCase(
        category="passive",
        canonical_type="capacitor",
        support_components="""
- type: resistor
  name: RLOAD
  nodes: [vout, 0]
  value: 100
""",
        component_entry="""
- type: capacitor
  name: CTEST
  nodes: [vin, vout]
  value: 1u
  ic: 0
""",
    ),
    ComponentCase(
        category="passive",
        canonical_type="inductor",
        support_components="""
- type: resistor
  name: RLOAD
  nodes: [vout, 0]
  value: 100
""",
        component_entry="""
- type: inductor
  name: LTEST
  nodes: [vin, vout]
  value: 1m
  ic: 0
""",
    ),
    ComponentCase(
        category="sources",
        canonical_type="voltage_source",
        support_components="""
- type: resistor
  name: RSRC
  nodes: [vsrc, 0]
  value: 100
""",
        component_entry="""
- type: voltage_source
  name: VTEST
  nodes: [vsrc, 0]
  waveform: {type: dc, value: 2}
""",
    ),
    ComponentCase(
        category="sources",
        canonical_type="current_source",
        support_components="""
- type: resistor
  name: RISRC
  nodes: [isrc, 0]
  value: 10
""",
        component_entry="""
- type: current_source
  name: ITEST
  nodes: [isrc, 0]
  value: 0.1
""",
    ),
    ComponentCase(
        category="switching_power",
        canonical_type="diode",
        support_components="""
- type: resistor
  name: RLOAD
  nodes: [vout, 0]
  value: 100
""",
        component_entry="""
- type: diode
  name: D1
  nodes: [vin, vout]
  g_on: 1e3
  g_off: 1e-9
""",
    ),
    ComponentCase(
        category="switching_power",
        canonical_type="switch",
        support_components="""
- type: resistor
  name: RLOAD
  nodes: [vsw, 0]
  value: 100
""",
        component_entry="""
- type: switch
  name: S1
  nodes: [vin, vsw]
  g_on: 1e3
  g_off: 1e-9
""",
    ),
    ComponentCase(
        category="switching_power",
        canonical_type="vcswitch",
        support_components="""
- type: resistor
  name: RLOAD
  nodes: [vsw, 0]
  value: 100
- type: voltage_source
  name: VCTRL
  nodes: [ctrl, 0]
  waveform: {type: dc, value: 2}
""",
        component_entry="""
- type: vcswitch
  name: VS1
  nodes: [ctrl, vin, vsw]
  v_threshold: 1.0
  g_on: 1e3
  g_off: 1e-9
""",
    ),
    ComponentCase(
        category="switching_power",
        canonical_type="mosfet",
        support_components="""
- type: voltage_source
  name: VDD
  nodes: [vdd, 0]
  waveform: {type: dc, value: 5}
- type: resistor
  name: RPULL
  nodes: [vdd, main]
  value: 10
- type: voltage_source
  name: VG
  nodes: [gate, 0]
  waveform: {type: dc, value: 10}
""",
        component_entry="""
- type: mosfet
  name: M1
  nodes: [gate, main, 0]
  vth: 2.0
  kp: 0.1
""",
    ),
    ComponentCase(
        category="switching_power",
        canonical_type="igbt",
        support_components="""
- type: voltage_source
  name: VDD
  nodes: [vdd, 0]
  waveform: {type: dc, value: 5}
- type: resistor
  name: RPULL
  nodes: [vdd, main]
  value: 10
- type: voltage_source
  name: VG
  nodes: [gate, 0]
  waveform: {type: dc, value: 10}
""",
        component_entry="""
- type: igbt
  name: Q1
  nodes: [gate, main, 0]
  vth: 2.0
  g_on: 1e3
  g_off: 1e-9
""",
    ),
    ComponentCase(
        category="magnetics",
        canonical_type="transformer",
        support_components="""
- type: resistor
  name: RSEC
  nodes: [vsec, 0]
  value: 100
""",
        component_entry="""
- type: transformer
  name: T1
  nodes: [vin, 0, vsec, 0]
  turns_ratio: 1.0
""",
    ),
    ComponentCase(
        category="switching_power",
        canonical_type="snubber_rc",
        support_components="""
- type: resistor
  name: RLOAD
  nodes: [vsn, 0]
  value: 100
""",
        component_entry="""
- type: snubber_rc
  name: SN1
  nodes: [vin, vsn]
  resistance: 100
  capacitance: 1n
""",
    ),
    ComponentCase(
        category="switching_power",
        canonical_type="bjt_npn",
        support_components="""
- type: voltage_source
  name: VDD
  nodes: [vdd, 0]
  waveform: {type: dc, value: 5}
- type: resistor
  name: RPULL
  nodes: [vdd, main]
  value: 10
- type: voltage_source
  name: VB
  nodes: [base, 0]
  waveform: {type: dc, value: 5}
""",
        component_entry="""
- type: bjt_npn
  name: QN1
  nodes: [base, main, 0]
  beta: 80
""",
    ),
    ComponentCase(
        category="switching_power",
        canonical_type="bjt_pnp",
        support_components="""
- type: voltage_source
  name: VDD
  nodes: [vdd, 0]
  waveform: {type: dc, value: 5}
- type: resistor
  name: RPULL
  nodes: [vdd, main]
  value: 10
- type: voltage_source
  name: VB
  nodes: [base, 0]
  waveform: {type: dc, value: 5}
""",
        component_entry="""
- type: bjt_pnp
  name: QP1
  nodes: [base, main, 0]
  beta: 80
""",
    ),
    ComponentCase(
        category="switching_power",
        canonical_type="thyristor",
        support_components="""
- type: voltage_source
  name: VG
  nodes: [gate, 0]
  waveform: {type: dc, value: 2}
""",
        component_entry="""
- type: thyristor
  name: SCR1
  nodes: [gate, vin, 0]
  gate_threshold: 1.0
  holding_current: 0.05
  latch_current: 0.1
""",
    ),
    ComponentCase(
        category="switching_power",
        canonical_type="triac",
        support_components="""
- type: voltage_source
  name: VG
  nodes: [gate, 0]
  waveform: {type: dc, value: 2}
""",
        component_entry="""
- type: triac
  name: TRI1
  nodes: [gate, vin, 0]
  gate_threshold: 1.0
  holding_current: 0.05
  latch_current: 0.1
""",
    ),
    ComponentCase(
        category="protection",
        canonical_type="fuse",
        support_components="""
- type: resistor
  name: RLOAD
  nodes: [vf, 0]
  value: 100
""",
        component_entry="""
- type: fuse
  name: F1
  nodes: [vin, vf]
  rating: 5
  blow_i2t: 2
""",
    ),
    ComponentCase(
        category="protection",
        canonical_type="circuit_breaker",
        support_components="""
- type: resistor
  name: RLOAD
  nodes: [vb, 0]
  value: 100
""",
        component_entry="""
- type: circuit_breaker
  name: B1
  nodes: [vin, vb]
  trip_current: 5
  trip_time: 1e-3
""",
    ),
    ComponentCase(
        category="protection",
        canonical_type="relay",
        support_components="""
- type: voltage_source
  name: VCOIL
  nodes: [coil_p, 0]
  waveform: {type: dc, value: 12}
- type: resistor
  name: RCOM
  nodes: [com, 0]
  value: 1k
- type: resistor
  name: RNO
  nodes: [no, 0]
  value: 1k
- type: resistor
  name: RNC
  nodes: [nc, 0]
  value: 1k
""",
        component_entry="""
- type: relay
  name: K1
  nodes: [coil_p, 0, com, no, nc]
  pickup_voltage: 6
  dropout_voltage: 3
  contact_resistance: 0.05
  off_resistance: 1e9
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="op_amp",
        support_components="""
- type: voltage_source
  name: VREF
  nodes: [vref, 0]
  waveform: {type: dc, value: 1}
- type: voltage_source
  name: VFB
  nodes: [vfb, 0]
  waveform: {type: dc, value: 0.5}
- type: resistor
  name: ROUT
  nodes: [op_out, 0]
  value: 1k
""",
        component_entry="""
- type: op_amp
  name: A1
  nodes: [vref, vfb, op_out]
  open_loop_gain: 1e5
  rail_low: -12
  rail_high: 12
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="comparator",
        support_components="""
- type: voltage_source
  name: VREF
  nodes: [vref, 0]
  waveform: {type: dc, value: 2}
- type: resistor
  name: ROUT
  nodes: [cmp_out, 0]
  value: 1k
""",
        component_entry="""
- type: comparator
  name: CMP1
  nodes: [vin, vref, cmp_out]
  threshold: 0.0
  hysteresis: 0.2
  high: 5.0
  low: 0.0
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="pi_controller",
        component_entry="""
- type: pi_controller
  name: PI1
  nodes: [vref, vin, pi_out]
  kp: 1.0
  ki: 10.0
  output_min: 0.0
  output_max: 1.0
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="pid_controller",
        component_entry="""
- type: pid_controller
  name: PID1
  nodes: [vref, vin, pid_out]
  kp: 1.0
  ki: 10.0
  kd: 0.1
  output_min: 0.0
  output_max: 1.0
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="gain",
        component_entry="""
- type: gain
  name: GAIN1
  nodes: [vin, gain_out]
  gain: 2.0
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="sum",
        component_entry="""
- type: sum
  name: SUM1
  nodes: [vin, sum_out]
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="subtraction",
        component_entry="""
- type: subtraction
  name: SUB1
  nodes: [vin, sub_out]
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="math_block",
        component_entry="""
- type: math_block
  name: MATH1
  nodes: [vin, math_out]
  operation: add
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="pwm_generator",
        support_components="""
- type: voltage_source
  name: VCTRL
  nodes: [ctrl, 0]
  waveform: {type: dc, value: 0.4}
""",
        component_entry="""
- type: pwm_generator
  name: PWM1
  nodes: [ctrl]
  frequency: 10000
  duty: 0.4
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="integrator",
        component_entry="""
- type: integrator
  name: INT1
  nodes: [vin, int_out]
  output_min: -1.0
  output_max: 1.0
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="differentiator",
        component_entry="""
- type: differentiator
  name: DIFF1
  nodes: [vin, diff_out]
  alpha: 0.5
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="limiter",
        component_entry="""
- type: limiter
  name: LIM1
  nodes: [vin, lim_out]
  min: -1.0
  max: 1.0
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="rate_limiter",
        component_entry="""
- type: rate_limiter
  name: RL1
  nodes: [vin, rl_out]
  rising_rate: 10
  falling_rate: 10
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="hysteresis",
        component_entry="""
- type: hysteresis
  name: HYS1
  nodes: [vin, hys_out]
  threshold: 0.0
  hysteresis: 0.5
  high: 1.0
  low: -1.0
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="lookup_table",
        component_entry="""
- type: lookup_table
  name: LUT1
  nodes: [vin, lut_out]
  x: [0, 1, 2]
  y: [0, 10, 20]
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="transfer_function",
        component_entry="""
- type: transfer_function
  name: TF1
  nodes: [vin, tf_out]
  num: [0.5, 0.5]
  den: [1.0, -0.5]
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="delay_block",
        component_entry="""
- type: delay_block
  name: DEL1
  nodes: [vin, del_out]
  delay: 1e-6
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="sample_hold",
        component_entry="""
- type: sample_hold
  name: SH1
  nodes: [vin, sh_out]
  sample_period: 1e-6
""",
    ),
    ComponentCase(
        category="control",
        canonical_type="state_machine",
        component_entry="""
- type: state_machine
  name: SM1
  nodes: [vin]
  mode: toggle
  threshold: 0.5
""",
    ),
    ComponentCase(
        category="magnetics",
        canonical_type="saturable_inductor",
        support_components="""
- type: resistor
  name: RLOAD
  nodes: [vsat, 0]
  value: 100
""",
        component_entry="""
- type: saturable_inductor
  name: LSAT1
  nodes: [vin, vsat]
  inductance: 1m
  saturation_current: 2
  saturation_inductance: 100u
""",
    ),
    ComponentCase(
        category="magnetics",
        canonical_type="coupled_inductor",
        support_components="""
- type: resistor
  name: RSEC
  nodes: [vsec, 0]
  value: 100
""",
        component_entry="""
- type: coupled_inductor
  name: K1
  nodes: [vin, 0, vsec, 0]
  l1: 1m
  l2: 2m
  coupling: 0.95
""",
    ),
    ComponentCase(
        category="instrumentation",
        canonical_type="voltage_probe",
        component_entry="""
- type: voltage_probe
  name: VP1
  nodes: [vin, 0]
""",
    ),
    ComponentCase(
        category="instrumentation",
        canonical_type="current_probe",
        component_entry="""
- type: current_probe
  name: IP1
  nodes: [vin, 0]
  target_component: VBASE
""",
    ),
    ComponentCase(
        category="instrumentation",
        canonical_type="power_probe",
        component_entry="""
- type: power_probe
  name: PP1
  nodes: [vin, 0]
  target_component: VBASE
""",
    ),
    ComponentCase(
        category="instrumentation",
        canonical_type="electrical_scope",
        component_entry="""
- type: electrical_scope
  name: SCOPE_E
  nodes: [vin, 0]
""",
    ),
    ComponentCase(
        category="instrumentation",
        canonical_type="thermal_scope",
        component_entry="""
- type: thermal_scope
  name: SCOPE_T
  nodes: [vin]
""",
    ),
    ComponentCase(
        category="signal_routing",
        canonical_type="signal_mux",
        support_components="""
- type: voltage_source
  name: VCTRL
  nodes: [ctrl, 0]
  waveform: {type: dc, value: 0}
""",
        component_entry="""
- type: signal_mux
  name: MUX1
  nodes: [vin, mux_out, ctrl]
  select_index: 0
""",
    ),
    ComponentCase(
        category="signal_routing",
        canonical_type="signal_demux",
        component_entry="""
- type: signal_demux
  name: DMX1
  nodes: [vin, out_a, out_b]
""",
    ),
)

EXPECTED_CANONICAL_TYPES = {
    "resistor",
    "capacitor",
    "inductor",
    "voltage_source",
    "current_source",
    "diode",
    "switch",
    "vcswitch",
    "mosfet",
    "igbt",
    "transformer",
    "snubber_rc",
    "bjt_npn",
    "bjt_pnp",
    "thyristor",
    "triac",
    "fuse",
    "circuit_breaker",
    "relay",
    "op_amp",
    "comparator",
    "pi_controller",
    "pid_controller",
    "gain",
    "sum",
    "subtraction",
    "math_block",
    "pwm_generator",
    "integrator",
    "differentiator",
    "limiter",
    "rate_limiter",
    "hysteresis",
    "lookup_table",
    "transfer_function",
    "delay_block",
    "sample_hold",
    "state_machine",
    "saturable_inductor",
    "coupled_inductor",
    "voltage_probe",
    "current_probe",
    "power_probe",
    "electrical_scope",
    "thermal_scope",
    "signal_mux",
    "signal_demux",
}


def _build_component_yaml(case: ComponentCase) -> str:
    base = textwrap.dedent(
        """
        schema: pulsim-v1
        version: 1
        simulation:
          tstart: 0
          tstop: 3e-6
          dt: 1e-6
          enable_events: true
        components:
          - type: voltage_source
            name: VBASE
            nodes: [vin, 0]
            waveform: {type: dc, value: 5}
          - type: resistor
            name: RBASE
            nodes: [vin, 0]
            value: 1k
        """
    ).strip()

    blocks = [base]
    if case.support_components.strip():
        blocks.append(textwrap.indent(textwrap.dedent(case.support_components).strip(), "  "))
    blocks.append(textwrap.indent(textwrap.dedent(case.component_entry).strip(), "  "))
    return "\n".join(blocks) + "\n"


def _run_case_transient(case: ComponentCase) -> tuple[list[str], ps.Circuit | None, ps.SimulationResult | None]:
    parser = ps.YamlParser()
    circuit, options = parser.load_string(_build_component_yaml(case))
    if parser.errors:
        return parser.errors, None, None
    options.adaptive_timestep = False
    options.dt_min = options.dt
    options.dt_max = options.dt
    options.formulation_mode = ps.FormulationMode.ProjectedWrapper
    options.direct_formulation_fallback = True
    result = ps.Simulator(circuit, options).run_transient(circuit.initial_state())
    return [], circuit, result


def _node_value(circuit: ps.Circuit, result: ps.SimulationResult, node_name: str) -> float:
    return float(result.states[-1][circuit.get_node(node_name)])


def _channel_last(result: ps.SimulationResult, channel_name: str) -> float:
    assert channel_name in result.virtual_channels
    values = [float(v) for v in result.virtual_channels[channel_name]]
    assert len(values) == len(result.time)
    return values[-1]


def _assert_component_expected_behavior(
    case: ComponentCase,
    circuit: ps.Circuit,
    result: ps.SimulationResult,
) -> None:
    ctype = case.canonical_type

    if ctype == "resistor":
        assert _node_value(circuit, result, "vout") == pytest.approx(4.545, rel=0.05)
    elif ctype == "capacitor":
        vout = _node_value(circuit, result, "vout")
        assert 4.7 <= vout <= 5.0
    elif ctype == "inductor":
        vout = _node_value(circuit, result, "vout")
        assert 0.8 <= vout <= 1.4
    elif ctype == "voltage_source":
        assert _node_value(circuit, result, "vsrc") == pytest.approx(2.0, rel=0.001)
    elif ctype == "current_source":
        assert _node_value(circuit, result, "isrc") == pytest.approx(1.0, rel=0.01)
    elif ctype == "diode":
        assert _node_value(circuit, result, "vout") >= 4.5
    elif ctype == "switch":
        assert _node_value(circuit, result, "vsw") <= 1e-3
    elif ctype == "vcswitch":
        assert _node_value(circuit, result, "vsw") >= 4.5
    elif ctype == "mosfet":
        assert _node_value(circuit, result, "main") <= 1.0
    elif ctype == "igbt":
        assert _node_value(circuit, result, "main") <= 0.1
    elif ctype == "transformer":
        assert _node_value(circuit, result, "vsec") == pytest.approx(5.0, rel=0.1)
    elif ctype == "snubber_rc":
        vsn = _node_value(circuit, result, "vsn")
        assert 2.0 <= vsn <= 3.2
    elif ctype == "bjt_npn":
        assert _node_value(circuit, result, "main") <= 1.5
    elif ctype == "bjt_pnp":
        assert _node_value(circuit, result, "main") >= 4.5
    elif ctype == "thyristor":
        assert _channel_last(result, "SCR1.trigger") == pytest.approx(1.0, abs=1e-12)
        assert _channel_last(result, "SCR1.state") == pytest.approx(1.0, abs=1e-12)
    elif ctype == "triac":
        assert _channel_last(result, "TRI1.trigger") == pytest.approx(1.0, abs=1e-12)
        assert _channel_last(result, "TRI1.state") == pytest.approx(1.0, abs=1e-12)
    elif ctype == "fuse":
        assert _channel_last(result, "F1.i2t") >= 0.0
        assert _channel_last(result, "F1.state") == pytest.approx(1.0, abs=1e-12)
    elif ctype == "circuit_breaker":
        assert _channel_last(result, "B1.trip_timer") == pytest.approx(0.0, abs=1e-12)
        assert _channel_last(result, "B1.state") == pytest.approx(1.0, abs=1e-12)
    elif ctype == "relay":
        assert _channel_last(result, "K1.state") == pytest.approx(1.0, abs=1e-12)
        assert _channel_last(result, "K1.no_state") == pytest.approx(1.0, abs=1e-12)
        assert _channel_last(result, "K1.nc_state") == pytest.approx(0.0, abs=1e-12)
    elif ctype == "op_amp":
        assert _channel_last(result, "A1") == pytest.approx(12.0, abs=1e-12)
    elif ctype == "comparator":
        assert _channel_last(result, "CMP1") == pytest.approx(5.0, abs=1e-12)
    elif ctype == "pi_controller":
        assert _channel_last(result, "PI1") == pytest.approx(0.0, abs=1e-12)
    elif ctype == "pid_controller":
        assert _channel_last(result, "PID1") == pytest.approx(0.0, abs=1e-12)
    elif ctype == "gain":
        assert _channel_last(result, "GAIN1") == pytest.approx(10.0, abs=1e-12)
    elif ctype == "sum":
        assert _channel_last(result, "SUM1") == pytest.approx(5.0, abs=1e-12)
    elif ctype == "subtraction":
        assert _channel_last(result, "SUB1") == pytest.approx(5.0, abs=1e-12)
    elif ctype == "math_block":
        assert _channel_last(result, "MATH1") == pytest.approx(5.0, abs=1e-12)
    elif ctype == "pwm_generator":
        assert _channel_last(result, "PWM1.duty") == pytest.approx(0.4, abs=1e-12)
        assert _channel_last(result, "PWM1") in (0.0, 1.0)
    elif ctype == "integrator":
        val = _channel_last(result, "INT1")
        assert 0.0 < val < 1.0
    elif ctype == "differentiator":
        assert _channel_last(result, "DIFF1") == pytest.approx(0.0, abs=1e-12)
    elif ctype == "limiter":
        assert _channel_last(result, "LIM1") == pytest.approx(1.0, abs=1e-12)
    elif ctype == "rate_limiter":
        assert _channel_last(result, "RL1") == pytest.approx(5.0, rel=0.01)
    elif ctype == "hysteresis":
        assert _channel_last(result, "HYS1") == pytest.approx(1.0, abs=1e-12)
    elif ctype == "lookup_table":
        assert _channel_last(result, "LUT1") == pytest.approx(20.0, abs=1e-12)
    elif ctype == "transfer_function":
        assert _channel_last(result, "TF1") == pytest.approx(9.0625, rel=1e-9)
    elif ctype == "delay_block":
        assert _channel_last(result, "DEL1") == pytest.approx(5.0, abs=1e-12)
    elif ctype == "sample_hold":
        assert _channel_last(result, "SH1") == pytest.approx(5.0, abs=1e-12)
    elif ctype == "state_machine":
        assert _channel_last(result, "SM1") == pytest.approx(1.0, abs=1e-12)
    elif ctype == "saturable_inductor":
        l_eff = _channel_last(result, "LSAT1.l_eff")
        i_est = _channel_last(result, "LSAT1.i_est")
        assert 1e-5 <= l_eff <= 1e-2
        assert i_est > 0.0
    elif ctype == "coupled_inductor":
        assert _channel_last(result, "K1.k") == pytest.approx(0.95, abs=1e-12)
        assert _channel_last(result, "K1.mutual") > 0.0
    elif ctype == "voltage_probe":
        assert _channel_last(result, "VP1") == pytest.approx(5.0, abs=1e-12)
    elif ctype == "current_probe":
        assert abs(_channel_last(result, "IP1")) == pytest.approx(0.005, rel=0.01)
    elif ctype == "power_probe":
        assert abs(_channel_last(result, "PP1")) == pytest.approx(0.025, rel=0.01)
    elif ctype == "electrical_scope":
        assert _channel_last(result, "SCOPE_E") == pytest.approx(2.5, rel=0.01)
    elif ctype == "thermal_scope":
        assert _channel_last(result, "SCOPE_T") == pytest.approx(5.0, abs=1e-12)
    elif ctype == "signal_mux":
        assert _channel_last(result, "MUX1") == pytest.approx(5.0, abs=1e-12)
    elif ctype == "signal_demux":
        assert _channel_last(result, "DMX1") == pytest.approx(5.0, abs=1e-12)
    else:
        raise AssertionError(f"Missing behavioral assertion for {ctype}")


@pytest.mark.parametrize("case", COMPONENT_CASES, ids=lambda case: case.canonical_type)
def test_supported_component_catalog_parses_without_unsupported_type(case: ComponentCase) -> None:
    parser = ps.YamlParser()
    circuit, _ = parser.load_string(_build_component_yaml(case))

    unsupported = [msg for msg in parser.errors if "Unsupported component type" in msg]
    assert unsupported == [], f"{case.canonical_type}: {unsupported}"
    assert parser.errors == [], f"{case.canonical_type}: {parser.errors}"
    assert circuit.num_devices() + circuit.num_virtual_components() >= 1


@pytest.mark.parametrize("case", COMPONENT_CASES, ids=lambda case: case.canonical_type)
def test_supported_component_catalog_runtime_smoke(case: ComponentCase) -> None:
    parse_errors, _, result = _run_case_transient(case)
    assert parse_errors == [], f"{case.canonical_type}: {parse_errors}"
    assert result is not None
    assert result.success
    assert len(result.time) >= 2, case.canonical_type


@pytest.mark.parametrize("case", COMPONENT_CASES, ids=lambda case: case.canonical_type)
def test_supported_component_catalog_matches_expected_behavior(case: ComponentCase) -> None:
    parse_errors, circuit, result = _run_case_transient(case)
    assert parse_errors == [], f"{case.canonical_type}: {parse_errors}"
    assert circuit is not None
    assert result is not None
    assert result.success
    _assert_component_expected_behavior(case, circuit, result)


def test_supported_component_catalog_is_complete_and_unique() -> None:
    listed_types = [case.canonical_type for case in COMPONENT_CASES]

    assert len(listed_types) == len(set(listed_types))
    assert len(listed_types) == 47
    assert set(listed_types) == EXPECTED_CANONICAL_TYPES


def test_supported_components_docs_table_matches_catalog() -> None:
    doc_path = Path(__file__).resolve().parents[2] / "docs" / "supported-components-catalog.md"
    doc_content = doc_path.read_text(encoding="utf-8")
    doc_types = set(re.findall(r"`([a-z_]+)`", doc_content))

    assert EXPECTED_CANONICAL_TYPES.issubset(doc_types)
