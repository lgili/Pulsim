from __future__ import annotations

import math

import pulsim as ps


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _expected_l_eff(i_est: float, params: dict[str, float], *, state: float | None = None) -> float:
    l_unsat = max(abs(params.get("inductance", 1e-3)), 1e-12)
    i_sat = max(abs(params.get("saturation_current", 1.0)), 1e-12)
    l_sat_raw = abs(params.get("saturation_inductance", l_unsat * 0.2))
    l_sat = _clamp(l_sat_raw, 1e-12, l_unsat)
    exponent = _clamp(params.get("saturation_exponent", 2.0), 1.0, 8.0)
    ratio = math.pow(abs(i_est) / i_sat, exponent)
    l_eff = l_sat + (l_unsat - l_sat) / (1.0 + ratio)

    if state is not None:
        band = max(params.get("hysteresis_band", 0.0), 0.0)
        strength = _clamp(abs(params.get("hysteresis_strength", 0.15)), 0.0, 0.95)
        direction = 1.0 if i_est > band else -1.0 if i_est < -band else 0.0
        multiplier = 1.0 - strength * state * direction
        l_eff = _clamp(l_eff * multiplier, l_sat, l_unsat)

    return max(l_eff, 1e-12)


def _expected_core_loss_series(
    time: list[float],
    i_equiv_signed: list[float],
    params: dict[str, float],
    *,
    h_state: list[float] | None = None,
) -> list[float]:
    core_loss_k = max(params.get("core_loss_k", 0.0), 0.0)
    if core_loss_k <= 0.0:
        return [0.0 for _ in time]
    core_loss_alpha = _clamp(params.get("core_loss_alpha", 2.0), 0.0, 8.0)
    core_loss_freq_coeff = max(params.get("core_loss_freq_coeff", 0.0), 0.0)
    hysteresis_loss_coeff = _clamp(abs(params.get("hysteresis_loss_coeff", 0.2)), 0.0, 50.0)
    band = max(params.get("hysteresis_band", 0.0), 0.0)

    out: list[float] = []
    prev_i: float | None = None
    prev_t: float | None = None
    for idx, (t, i_signed) in enumerate(zip(time, i_equiv_signed)):
        i_equiv = abs(i_signed)
        freq_multiplier = 1.0
        if core_loss_freq_coeff > 0.0:
            if prev_i is None or prev_t is None:
                prev_i = max(params.get("magnetic_i_equiv_init", i_equiv), 0.0)
                prev_t = t
            else:
                dt = t - prev_t
                if dt > 0.0 and math.isfinite(dt):
                    di_dt = (i_equiv - prev_i) / dt
                    additive = core_loss_freq_coeff * abs(di_dt)
                    if math.isfinite(additive) and additive > 0.0:
                        freq_multiplier += additive
                prev_i = i_equiv
                prev_t = t

        hysteresis_multiplier = 1.0
        if h_state is not None:
            state = h_state[idx]
            direction = 1.0 if i_signed > band else -1.0 if i_signed < -band else 0.0
            mismatch = 0.5 if direction == 0.0 else 0.5 * (1.0 - state * direction)
            hysteresis_multiplier += hysteresis_loss_coeff * mismatch

        core_loss = core_loss_k * math.pow(i_equiv, core_loss_alpha) * freq_multiplier * hysteresis_multiplier
        out.append(max(core_loss, 0.0))

    return out


def test_magnetic_core_saturation_matches_effective_inductance_formula() -> None:
    parser = ps.YamlParser()
    circuit, opts = parser.load_string(
        """
schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 1.5e-3
  dt: 2e-6
  step_mode: fixed
components:
  - type: voltage_source
    name: Vin
    nodes: [vin, 0]
    waveform:
      type: sine
      amplitude: 6
      frequency: 1200
      offset: 0
  - type: resistor
    name: R1
    nodes: [vin, n1]
    value: 2
  - type: saturable_inductor
    name: Lsat
    nodes: [n1, 0]
    inductance: 1m
    saturation_current: 1.2
    saturation_inductance: 150u
    saturation_exponent: 2.2
    magnetic_core:
      enabled: true
      model: saturation
      core_loss_k: 0.0
"""
    )
    assert parser.errors == [], parser.errors
    opts.newton_options.num_nodes = int(circuit.num_nodes())
    opts.newton_options.num_branches = int(circuit.num_branches())
    result = ps.Simulator(circuit, opts).run_transient(circuit.initial_state())
    assert result.success

    i_est = [float(v) for v in result.virtual_channels["Lsat.i_est"]]
    l_eff = [float(v) for v in result.virtual_channels["Lsat.l_eff"]]
    params = {
        "inductance": 1e-3,
        "saturation_current": 1.2,
        "saturation_inductance": 150e-6,
        "saturation_exponent": 2.2,
    }

    max_rel = 0.0
    for i_val, l_val in zip(i_est, l_eff):
        expected = _expected_l_eff(i_val, params)
        rel = abs(l_val - expected) / max(expected, 1e-12)
        max_rel = max(max_rel, rel)

    assert max_rel <= 1e-9


def test_magnetic_core_hysteresis_loss_matches_expected_multiplier() -> None:
    parser = ps.YamlParser()
    circuit, opts = parser.load_string(
        """
schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 2.5e-3
  dt: 2e-6
  step_mode: fixed
components:
  - type: voltage_source
    name: Vin
    nodes: [vin, 0]
    waveform:
      type: sine
      amplitude: 8
      frequency: 900
      offset: 0
  - type: resistor
    name: R1
    nodes: [vin, n1]
    value: 1.5
  - type: saturable_inductor
    name: Lsat
    nodes: [n1, 0]
    inductance: 1m
    saturation_current: 1.0
    saturation_inductance: 200u
    magnetic_core:
      enabled: true
      model: hysteresis
      hysteresis_band: 0.05
      hysteresis_strength: 0.2
      hysteresis_loss_coeff: 0.4
      hysteresis_state_init: 1.0
      core_loss_k: 0.15
      core_loss_alpha: 2.0
      core_loss_freq_coeff: 0.0
"""
    )
    assert parser.errors == [], parser.errors
    opts.newton_options.num_nodes = int(circuit.num_nodes())
    opts.newton_options.num_branches = int(circuit.num_branches())
    result = ps.Simulator(circuit, opts).run_transient(circuit.initial_state())
    assert result.success

    time = [float(t) for t in result.time]
    i_est = [float(v) for v in result.virtual_channels["Lsat.i_est"]]
    h_state = [float(v) for v in result.virtual_channels["Lsat.h_state"]]
    core_loss = [float(v) for v in result.virtual_channels["Lsat.core_loss"]]

    params = {
        "core_loss_k": 0.15,
        "core_loss_alpha": 2.0,
        "core_loss_freq_coeff": 0.0,
        "hysteresis_band": 0.05,
        "hysteresis_loss_coeff": 0.4,
        "magnetic_i_equiv_init": 0.0,
    }

    expected = _expected_core_loss_series(time, i_est, params, h_state=h_state)
    max_rel = 0.0
    for actual, exp in zip(core_loss, expected):
        rel = abs(actual - exp) / max(exp, 1e-12)
        max_rel = max(max_rel, rel)

    assert max_rel <= 1e-9

    energy_actual = 0.0
    energy_expected = 0.0
    for idx in range(1, len(time)):
        dt = time[idx] - time[idx - 1]
        energy_actual += core_loss[idx] * dt
        energy_expected += expected[idx] * dt

    assert abs(energy_actual - energy_expected) <= 1e-10


def test_magnetic_core_loss_summary_matches_channel_reduction() -> None:
    parser = ps.YamlParser()
    circuit, opts = parser.load_string(
        """
schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 1e-3
  dt: 2e-6
  step_mode: fixed
  enable_losses: true
components:
  - type: voltage_source
    name: Vin
    nodes: [vin, 0]
    waveform: {type: dc, value: 10}
  - type: resistor
    name: R1
    nodes: [vin, n1]
    value: 2
  - type: saturable_inductor
    name: Lsat
    nodes: [n1, 0]
    inductance: 1m
    saturation_current: 1.0
    saturation_inductance: 200u
    magnetic_core:
      enabled: true
      model: saturation
      loss_policy: loss_summary
      core_loss_k: 0.2
      core_loss_alpha: 2.0
"""
    )
    assert parser.errors == [], parser.errors
    opts.newton_options.num_nodes = int(circuit.num_nodes())
    opts.newton_options.num_branches = int(circuit.num_branches())
    result = ps.Simulator(circuit, opts).run_transient(circuit.initial_state())
    assert result.success

    core_loss = [float(v) for v in result.virtual_channels["Lsat.core_loss"]]
    duration = float(result.time[-1] - result.time[0])
    total_energy = 0.0
    for idx in range(1, len(result.time)):
        dt = float(result.time[idx] - result.time[idx - 1])
        total_energy += core_loss[idx] * dt
    avg_power = total_energy / duration if duration > 0.0 else 0.0

    rows = {row.device_name: row for row in result.loss_summary.device_losses}
    row = rows["Lsat.core"]
    assert abs(row.total_energy - total_energy) <= 1e-10
    assert abs(row.average_power - avg_power) <= 1e-10


def test_magnetic_core_channel_order_and_allocation_are_deterministic() -> None:
    netlist = """
schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 8e-4
  dt: 2e-6
  step_mode: fixed
components:
  - type: voltage_source
    name: Vin
    nodes: [vin, 0]
    waveform: {type: dc, value: 10}
  - type: resistor
    name: R1
    nodes: [vin, n1]
    value: 2
  - type: saturable_inductor
    name: Lsat
    nodes: [n1, 0]
    inductance: 1m
    saturation_current: 1.0
    saturation_inductance: 200u
    magnetic_core:
      enabled: true
      model: saturation
      core_loss_k: 0.2
      core_loss_alpha: 2.0
      core_loss_freq_coeff: 1e-4
"""

    def _run() -> ps.SimulationResult:
        parser = ps.YamlParser()
        circuit, opts = parser.load_string(netlist)
        assert parser.errors == [], parser.errors
        opts.newton_options.num_nodes = int(circuit.num_nodes())
        opts.newton_options.num_branches = int(circuit.num_branches())
        return ps.Simulator(circuit, opts).run_transient(circuit.initial_state())

    result_a = _run()
    result_b = _run()
    assert list(result_a.virtual_channels.keys()) == list(result_b.virtual_channels.keys())
    assert result_a.backend_telemetry.virtual_channel_reallocations == 0
    assert result_b.backend_telemetry.virtual_channel_reallocations == 0
