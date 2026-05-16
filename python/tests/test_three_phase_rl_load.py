"""Smoke tests for ``Circuit.add_three_phase_rl_load``.

Mirrors the Catch2 suite — verifies Star, Delta, and unbalance modes
through the pybind boundary against analytical impedance.
"""
from __future__ import annotations

import math

import pytest

import pulsim as ps


_PI = math.pi


def _run_transient(circuit: "ps.Circuit", tstop: float = 0.2,
                   dt: float = 50e-6) -> "ps.TransientResult":
    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = tstop
    opts.dt = dt
    opts.dt_min = 1e-9
    opts.dt_max = dt
    opts.adaptive_timestep = False
    opts.enable_bdf_order_control = False
    opts.newton_options.num_nodes = circuit.num_nodes()
    opts.newton_options.num_branches = circuit.num_branches()
    sim = ps.Simulator(circuit, opts)
    return sim.run_transient()


def _rms(states, var_idx: int, t: list[float], t_window_start: float) -> float:
    sum_sq = 0.0
    count = 0
    for i, t_i in enumerate(t):
        if t_i < t_window_start:
            continue
        v = states[i][var_idx]
        sum_sq += v * v
        count += 1
    if count == 0:
        return 0.0
    return math.sqrt(sum_sq / count)


class TestThreePhaseRLLoadBindings:

    def test_params_defaults(self):
        p = ps.ThreePhaseRLLoadParams()
        assert p.resistance_per_phase == pytest.approx(10.0)
        assert p.inductance_per_phase == pytest.approx(1e-3)
        assert p.topology == ps.ThreePhaseLoadTopology.Star
        assert p.unbalance_factor == pytest.approx(0.0)

    def test_params_writable(self):
        p = ps.ThreePhaseRLLoadParams()
        p.resistance_per_phase = 30.0
        p.inductance_per_phase = 50e-3
        p.topology = ps.ThreePhaseLoadTopology.Delta
        p.unbalance_factor = 0.1
        assert p.resistance_per_phase == pytest.approx(30.0)
        assert p.inductance_per_phase == pytest.approx(50e-3)
        assert p.topology == ps.ThreePhaseLoadTopology.Delta
        assert p.unbalance_factor == pytest.approx(0.1)

    def test_star_balanced_matches_analytical(self):
        """Star load: I_line = V_LN / |Z| at line frequency."""
        circuit = ps.Circuit()
        na = circuit.add_node("A")
        nb = circuit.add_node("B")
        nc = circuit.add_node("C")

        # 400 V_LL_RMS / 50 Hz
        src_p = ps.ThreePhaseSourceParams()
        src_p.line_to_line_voltage_rms = 400.0
        src_p.frequency_hz = 50.0
        circuit.add_three_phase_source(
            "Vsrc", na, nb, nc, ps.Circuit.ground(), src_p
        )

        ld_p = ps.ThreePhaseRLLoadParams()
        ld_p.resistance_per_phase = 30.0
        ld_p.inductance_per_phase = 50e-3
        ld_p.topology = ps.ThreePhaseLoadTopology.Star
        circuit.add_three_phase_rl_load(
            "RL", na, nb, nc, ps.Circuit.ground(), ld_p
        )

        result = _run_transient(circuit)
        assert result.success, f"Sim failed: {result.message}"

        # Analytical I_line for Y connection:
        v_ln_rms = 400.0 / math.sqrt(3.0)
        x_l = 2 * _PI * 50.0 * 50e-3
        z_mag = math.sqrt(30.0**2 + x_l**2)
        i_line_expected = v_ln_rms / z_mag

        # Branch order: 3 source branches + L_A, L_B, L_C
        n_nodes = circuit.num_nodes()
        i_la = _rms(result.states, n_nodes + 3, result.time, 0.05)

        assert i_la == pytest.approx(i_line_expected, rel=0.05), (
            f"I_a={i_la:.3f} A vs expected {i_line_expected:.3f} A"
        )

    def test_delta_balanced_matches_analytical(self):
        """Delta load: I_branch = V_LL / |Z|."""
        circuit = ps.Circuit()
        na = circuit.add_node("A")
        nb = circuit.add_node("B")
        nc = circuit.add_node("C")

        src_p = ps.ThreePhaseSourceParams()
        src_p.line_to_line_voltage_rms = 400.0
        src_p.frequency_hz = 50.0
        circuit.add_three_phase_source(
            "Vsrc", na, nb, nc, ps.Circuit.ground(), src_p
        )

        ld_p = ps.ThreePhaseRLLoadParams()
        ld_p.resistance_per_phase = 30.0
        ld_p.inductance_per_phase = 50e-3
        ld_p.topology = ps.ThreePhaseLoadTopology.Delta
        circuit.add_three_phase_rl_load(
            "RL", na, nb, nc, ps.Circuit.ground(), ld_p
        )

        result = _run_transient(circuit)
        assert result.success

        x_l = 2 * _PI * 50.0 * 50e-3
        z_mag = math.sqrt(30.0**2 + x_l**2)
        i_branch_expected = 400.0 / z_mag

        n_nodes = circuit.num_nodes()
        i_lab = _rms(result.states, n_nodes + 3, result.time, 0.05)

        assert i_lab == pytest.approx(i_branch_expected, rel=0.05)

    def test_convenience_overload(self):
        """Explicit (R, L) overload should default to Star."""
        circuit = ps.Circuit()
        na = circuit.add_node("A")
        nb = circuit.add_node("B")
        nc = circuit.add_node("C")
        circuit.add_three_phase_source(
            "Vsrc", na, nb, nc, ps.Circuit.ground(), 400.0, 50.0
        )
        circuit.add_three_phase_rl_load(
            "RL", na, nb, nc, ps.Circuit.ground(),
            resistance_per_phase=30.0,
            inductance_per_phase=50e-3,
        )
        # 3 sine sources + 3 inductors = 6 branches
        assert circuit.num_branches() == 6

    def test_unbalance_scales_phase_currents(self):
        """unbalance_factor=u → |Z_b|=Z·(1-u), |Z_c|=Z·(1+u)."""
        circuit = ps.Circuit()
        na = circuit.add_node("A")
        nb = circuit.add_node("B")
        nc = circuit.add_node("C")
        src_p = ps.ThreePhaseSourceParams()
        src_p.line_to_line_voltage_rms = 400.0
        src_p.frequency_hz = 50.0
        circuit.add_three_phase_source(
            "Vsrc", na, nb, nc, ps.Circuit.ground(), src_p
        )
        ld_p = ps.ThreePhaseRLLoadParams()
        ld_p.resistance_per_phase = 30.0
        ld_p.inductance_per_phase = 50e-3
        ld_p.topology = ps.ThreePhaseLoadTopology.Star
        ld_p.unbalance_factor = 0.2
        circuit.add_three_phase_rl_load(
            "RL", na, nb, nc, ps.Circuit.ground(), ld_p
        )

        result = _run_transient(circuit)
        assert result.success

        n_nodes = circuit.num_nodes()
        i_a = _rms(result.states, n_nodes + 3, result.time, 0.05)
        i_b = _rms(result.states, n_nodes + 4, result.time, 0.05)
        i_c = _rms(result.states, n_nodes + 5, result.time, 0.05)

        assert i_b > i_a, "Phase B (lower Z) should have higher current"
        assert i_c < i_a, "Phase C (higher Z) should have lower current"
        assert i_b / i_a == pytest.approx(1.0 / 0.8, rel=0.05)
        assert i_c / i_a == pytest.approx(1.0 / 1.2, rel=0.05)
