"""Smoke tests for ``Circuit.add_three_phase_source`` (Phase-28 follow-up).

These tests verify the pybind11 bindings work end-to-end: building a circuit,
running a transient, and inspecting the three line-to-neutral waveforms for
the expected phase relationships and unbalance scaling.

Path setup is inherited from conftest.py — same pattern as test_api.py.
"""

import math

import pytest

import pulsim as ps


_V_LL_RMS = 400.0
_FREQUENCY_HZ = 50.0
# V_ph_peak = V_LL_RMS * sqrt(2) / sqrt(3)
_V_PH_PEAK = _V_LL_RMS * math.sqrt(2.0) / math.sqrt(3.0)


def _build_circuit(*, params: "ps.ThreePhaseSourceParams") -> tuple["ps.Circuit", int, int, int]:
    circuit = ps.Circuit()
    n_a = circuit.add_node("A")
    n_b = circuit.add_node("B")
    n_c = circuit.add_node("C")
    circuit.add_three_phase_source(
        "Vgrid", n_a, n_b, n_c, ps.Circuit.ground(), params
    )
    circuit.add_resistor("Ra", n_a, ps.Circuit.ground(), 100.0)
    circuit.add_resistor("Rb", n_b, ps.Circuit.ground(), 100.0)
    circuit.add_resistor("Rc", n_c, ps.Circuit.ground(), 100.0)
    return circuit, n_a, n_b, n_c


def _run(circuit: "ps.Circuit") -> "ps.TransientResult":
    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 40e-3  # 2 cycles at 50 Hz
    opts.dt = 50e-6
    opts.dt_min = 1e-9
    opts.dt_max = 50e-6
    opts.adaptive_timestep = False
    opts.enable_bdf_order_control = False
    opts.newton_options.num_nodes = circuit.num_nodes()
    opts.newton_options.num_branches = circuit.num_branches()

    sim = ps.Simulator(circuit, opts)
    result = sim.run_transient()
    assert result.success, f"Sim failed: {result.message}"
    return result


def _sample_at(times, states, node_idx: int, target_time: float) -> float:
    """Linear-interp the node voltage at ``target_time``."""
    n = len(times)
    if n == 0:
        return 0.0
    if target_time <= times[0]:
        return float(states[0][node_idx])
    if target_time >= times[-1]:
        return float(states[-1][node_idx])
    for i in range(1, n):
        if times[i] >= target_time:
            t0, t1 = times[i - 1], times[i]
            alpha = (target_time - t0) / (t1 - t0)
            v0 = float(states[i - 1][node_idx])
            v1 = float(states[i][node_idx])
            return v0 * (1.0 - alpha) + v1 * alpha
    return float(states[-1][node_idx])


class TestThreePhaseSource:
    """End-to-end binding + transient smoke tests."""

    def test_params_class_is_aggregate(self):
        """ThreePhaseSourceParams should be default-constructible from Python."""
        params = ps.ThreePhaseSourceParams()
        assert params.line_to_line_voltage_rms == pytest.approx(400.0)
        assert params.frequency_hz == pytest.approx(50.0)
        assert params.phase_a_deg == pytest.approx(0.0)
        assert params.positive_sequence is True
        assert params.unbalance_factor == pytest.approx(0.0)

    def test_params_fields_are_writable(self):
        params = ps.ThreePhaseSourceParams()
        params.line_to_line_voltage_rms = 230.0
        params.frequency_hz = 60.0
        params.phase_a_deg = 30.0
        params.positive_sequence = False
        params.unbalance_factor = 0.05
        assert params.line_to_line_voltage_rms == pytest.approx(230.0)
        assert params.frequency_hz == pytest.approx(60.0)
        assert params.phase_a_deg == pytest.approx(30.0)
        assert params.positive_sequence is False
        assert params.unbalance_factor == pytest.approx(0.05)

    def test_balanced_positive_sequence_reaches_peak(self):
        """Each leg should reach V_peak at its expected quarter-period."""
        params = ps.ThreePhaseSourceParams()
        params.line_to_line_voltage_rms = _V_LL_RMS
        params.frequency_hz = _FREQUENCY_HZ
        params.positive_sequence = True

        circuit, na, nb, nc = _build_circuit(params=params)
        result = _run(circuit)

        period = 1.0 / _FREQUENCY_HZ
        v_a_peak = _sample_at(result.time, result.states, na, period / 4.0)
        v_b_peak = _sample_at(
            result.time, result.states, nb, period / 4.0 + period / 3.0
        )
        v_c_peak = _sample_at(
            result.time, result.states, nc, period / 4.0 + 2.0 * period / 3.0
        )

        margin = 0.03 * _V_PH_PEAK
        assert v_a_peak == pytest.approx(_V_PH_PEAK, abs=margin)
        assert v_b_peak == pytest.approx(_V_PH_PEAK, abs=margin)
        assert v_c_peak == pytest.approx(_V_PH_PEAK, abs=margin)

    def test_negative_sequence_swaps_phase_b(self):
        """Flipping ``positive_sequence`` should flip phase B sign at t=T."""
        period = 1.0 / _FREQUENCY_HZ
        margin = 0.05 * _V_PH_PEAK

        params_pos = ps.ThreePhaseSourceParams()
        params_pos.positive_sequence = True
        circuit_pos, _, nb_pos, _ = _build_circuit(params=params_pos)
        result_pos = _run(circuit_pos)
        v_b_pos = _sample_at(result_pos.time, result_pos.states, nb_pos, period)

        params_neg = ps.ThreePhaseSourceParams()
        params_neg.positive_sequence = False
        circuit_neg, _, nb_neg, _ = _build_circuit(params=params_neg)
        result_neg = _run(circuit_neg)
        v_b_neg = _sample_at(result_neg.time, result_neg.states, nb_neg, period)

        # Phase B at t=T:
        #   pos seq:  V_peak * sin(2π - 2π/3) = -√3/2 * V_peak
        #   neg seq:  V_peak * sin(2π + 2π/3) = +√3/2 * V_peak
        expected_pos = _V_PH_PEAK * math.sin(-2.0 * math.pi / 3.0)
        expected_neg = _V_PH_PEAK * math.sin(+2.0 * math.pi / 3.0)
        assert v_b_pos == pytest.approx(expected_pos, abs=margin)
        assert v_b_neg == pytest.approx(expected_neg, abs=margin)
        assert v_b_pos * v_b_neg < 0.0  # Opposite signs confirm sequence swap

    def test_unbalance_factor_scales_b_and_c(self):
        """``unbalance_factor=u`` keeps A nominal, B at (1-u)·V, C at (1+u)·V."""
        u = 0.1
        params = ps.ThreePhaseSourceParams()
        params.unbalance_factor = u

        circuit, na, nb, nc = _build_circuit(params=params)
        result = _run(circuit)

        period = 1.0 / _FREQUENCY_HZ
        v_a_peak = _sample_at(result.time, result.states, na, period / 4.0)
        v_b_peak = _sample_at(
            result.time, result.states, nb, period / 4.0 + period / 3.0
        )
        v_c_peak = _sample_at(
            result.time, result.states, nc, period / 4.0 + 2.0 * period / 3.0
        )

        margin = 0.03 * _V_PH_PEAK
        assert v_a_peak == pytest.approx(_V_PH_PEAK, abs=margin)
        assert v_b_peak == pytest.approx(_V_PH_PEAK * (1.0 - u), abs=margin)
        assert v_c_peak == pytest.approx(_V_PH_PEAK * (1.0 + u), abs=margin)

    def test_convenience_overload(self):
        """The (V_LL_RMS, f) convenience overload should work without params."""
        circuit = ps.Circuit()
        n_a = circuit.add_node("A")
        n_b = circuit.add_node("B")
        n_c = circuit.add_node("C")
        circuit.add_three_phase_source(
            "Vgrid", n_a, n_b, n_c, ps.Circuit.ground(), 400.0, 50.0
        )
        # 3 branch indices reserved (one per leg)
        assert circuit.num_branches() == 3
