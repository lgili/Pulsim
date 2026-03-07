"""Analytical electrothermal regression tests for thermal-enabled component types."""

from __future__ import annotations

import math
import statistics
import textwrap
from dataclasses import dataclass

import pulsim as ps
import pytest


AMBIENT_C = 25.0
RTH = 2.0
CTH = 1e-3


@dataclass(frozen=True)
class ThermalAnalyticCase:
    component_type: str
    component_name: str
    expected_power_w: float
    components_yaml: str


THERMAL_ANALYTIC_CASES: tuple[ThermalAnalyticCase, ...] = (
    ThermalAnalyticCase(
        component_type="resistor",
        component_name="R1",
        expected_power_w=2.5,  # V^2 / R = 5^2 / 10
        components_yaml="""
- type: voltage_source
  name: V1
  nodes: [n, 0]
  waveform: {type: dc, value: 5}
- type: resistor
  name: R1
  nodes: [n, 0]
  value: 10
  thermal:
    enabled: true
    rth: 2
    cth: 1e-3
""",
    ),
    ThermalAnalyticCase(
        component_type="diode",
        component_name="D1",
        expected_power_w=1.25,  # g_on * V^2 = 0.05 * 5^2
        components_yaml="""
- type: voltage_source
  name: V1
  nodes: [n, 0]
  waveform: {type: dc, value: 5}
- type: diode
  name: D1
  nodes: [n, 0]
  g_on: 0.05
  g_off: 1e-9
  thermal:
    enabled: true
    rth: 2
    cth: 1e-3
""",
    ),
    ThermalAnalyticCase(
        component_type="mosfet",
        component_name="M1",
        expected_power_w=1.25,  # forced-on branch: kp * Vds^2 = 0.05 * 5^2
        components_yaml="""
- type: voltage_source
  name: VDS
  nodes: [d, 0]
  waveform: {type: dc, value: 5}
- type: voltage_source
  name: VG
  nodes: [g, 0]
  waveform: {type: dc, value: 0}
- type: mosfet
  name: M1
  nodes: [g, d, 0]
  vth: 2
  kp: 0.05
  lambda: 0
  thermal:
    enabled: true
    rth: 2
    cth: 1e-3
- type: pwm_generator
  name: PWM_M1
  nodes: [g]
  frequency: 1000
  duty: 1.0
  target_component: M1
""",
    ),
    ThermalAnalyticCase(
        component_type="igbt",
        component_name="Q1",
        expected_power_w=1.25,  # forced-on branch: g_on * Vce^2 = 0.05 * 5^2
        components_yaml="""
- type: voltage_source
  name: VCE
  nodes: [c, 0]
  waveform: {type: dc, value: 5}
- type: voltage_source
  name: VG
  nodes: [g, 0]
  waveform: {type: dc, value: 0}
- type: igbt
  name: Q1
  nodes: [g, c, 0]
  vth: 2
  g_on: 0.05
  g_off: 1e-9
  thermal:
    enabled: true
    rth: 2
    cth: 1e-3
- type: pwm_generator
  name: PWM_Q1
  nodes: [g]
  frequency: 1000
  duty: 1.0
  target_component: Q1
""",
    ),
    ThermalAnalyticCase(
        component_type="bjt_npn",
        component_name="QN1",
        expected_power_w=0.5,  # surrogate kp=beta*1e-3 => 0.02*5^2 when forced-on
        components_yaml="""
- type: voltage_source
  name: VC
  nodes: [c, 0]
  waveform: {type: dc, value: 5}
- type: voltage_source
  name: VB
  nodes: [b, 0]
  waveform: {type: dc, value: 0}
- type: bjt_npn
  name: QN1
  nodes: [b, c, 0]
  beta: 20
  thermal:
    enabled: true
    rth: 2
    cth: 1e-3
- type: pwm_generator
  name: PWM_QN1
  nodes: [b]
  frequency: 1000
  duty: 1.0
  target_component: QN1
""",
    ),
    ThermalAnalyticCase(
        component_type="bjt_pnp",
        component_name="QP1",
        expected_power_w=0.5,  # surrogate kp=beta*1e-3 => 0.02*|V|^2 when forced-on
        components_yaml="""
- type: voltage_source
  name: VE
  nodes: [e, 0]
  waveform: {type: dc, value: 5}
- type: voltage_source
  name: VB
  nodes: [b, 0]
  waveform: {type: dc, value: 0}
- type: bjt_pnp
  name: QP1
  nodes: [b, 0, e]
  beta: 20
  thermal:
    enabled: true
    rth: 2
    cth: 1e-3
- type: pwm_generator
  name: PWM_QP1
  nodes: [b]
  frequency: 1000
  duty: 1.0
  target_component: QP1
""",
    ),
)

EXPECTED_THERMAL_ENABLED_TYPES = {
    "resistor",
    "diode",
    "mosfet",
    "igbt",
    "bjt_npn",
    "bjt_pnp",
}


def _build_yaml(case: ThermalAnalyticCase) -> str:
    simulation = textwrap.dedent(
        """
        schema: pulsim-v1
        version: 1
        simulation:
          tstart: 0
          tstop: 5e-4
          dt: 1e-6
          enable_losses: true
          thermal:
            enabled: true
            ambient: 25
            policy: loss_only
            default_rth: 2
            default_cth: 1e-3
        components:
        """
    ).strip()
    body = textwrap.indent(textwrap.dedent(case.components_yaml).strip(), "  ")
    return f"{simulation}\n{body}\n"


def _expected_thermal_trace_from_constant_power(
    power_w: float,
    time: list[float],
    ambient_c: float,
    rth: float,
    cth: float,
) -> list[float]:
    trace = [ambient_c]
    current = ambient_c
    for i in range(1, len(time)):
        dt = float(time[i]) - float(time[i - 1])
        if cth <= 0.0:
            current = ambient_c + power_w * rth
        else:
            tau = max(rth * cth, 1e-12)
            delta = current - ambient_c
            delta_dot = (power_w * rth - delta) / tau
            current = ambient_c + delta + dt * delta_dot
        trace.append(current)
    return trace


def _assert_close(value: float, expected: float, *, rel: float = 1e-9, abs_: float = 1e-9) -> None:
    assert math.isclose(value, expected, rel_tol=rel, abs_tol=abs_), (
        f"value={value} expected={expected} rel={rel} abs={abs_}"
    )


def test_thermal_analytic_cases_cover_all_supported_thermal_types() -> None:
    covered = {case.component_type for case in THERMAL_ANALYTIC_CASES}
    assert covered == EXPECTED_THERMAL_ENABLED_TYPES


@pytest.mark.parametrize("case", THERMAL_ANALYTIC_CASES, ids=lambda case: case.component_type)
def test_thermal_response_matches_discrete_rc_theory(case: ThermalAnalyticCase) -> None:
    parser = ps.YamlParser()
    circuit, options = parser.load_string(_build_yaml(case))
    assert parser.errors == [], parser.errors

    options.adaptive_timestep = False
    options.dt_min = options.dt
    options.dt_max = options.dt

    result = ps.Simulator(circuit, options).run_transient(circuit.initial_state())
    assert result.success

    channel = f"T({case.component_name})"
    assert channel in result.virtual_channels
    trace = [float(v) for v in result.virtual_channels[channel]]
    time = [float(t) for t in result.time]
    assert len(trace) == len(time)
    assert len(trace) >= 2
    assert all(curr >= prev - 1e-12 for prev, curr in zip(trace, trace[1:]))

    expected_trace = _expected_thermal_trace_from_constant_power(
        case.expected_power_w, time, AMBIENT_C, RTH, CTH
    )
    assert len(expected_trace) == len(trace)

    expected_final = expected_trace[-1]
    expected_peak = max(expected_trace)
    expected_avg = statistics.fmean(expected_trace)

    _assert_close(trace[-1], expected_final)
    _assert_close(max(trace), expected_peak)
    _assert_close(statistics.fmean(trace), expected_avg)

    component_rows = {item.component_name: item for item in result.component_electrothermal}
    assert case.component_name in component_rows
    component_row = component_rows[case.component_name]
    assert component_row.thermal_enabled
    _assert_close(float(component_row.average_power), case.expected_power_w)
    _assert_close(float(component_row.final_temperature), trace[-1])
    _assert_close(float(component_row.peak_temperature), max(trace))
    _assert_close(float(component_row.average_temperature), statistics.fmean(trace))

    summary_rows = {item.device_name: item for item in result.thermal_summary.device_temperatures}
    assert case.component_name in summary_rows
    summary_row = summary_rows[case.component_name]
    _assert_close(float(summary_row.final_temperature), trace[-1])
    _assert_close(float(summary_row.peak_temperature), max(trace))
    _assert_close(float(summary_row.average_temperature), statistics.fmean(trace))
