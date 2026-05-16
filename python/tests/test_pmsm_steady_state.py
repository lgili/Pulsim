"""Smoke tests for ``Circuit.add_pmsm_steady_state``.

Verifies basic PMSM behaviour at fixed rotor speed through the pybind
boundary. Mirrors the Catch2 suite — zero-back-EMF degeneration,
back-EMF amplitude scaling with ω_e.
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
    return ps.Simulator(circuit, opts).run_transient()


def _rms(states, var_idx: int, t: list[float], t_window_start: float) -> float:
    sum_sq = 0.0
    count = 0
    for i, t_i in enumerate(t):
        if t_i < t_window_start:
            continue
        v = states[i][var_idx]
        sum_sq += v * v
        count += 1
    return math.sqrt(sum_sq / count) if count else 0.0


class TestPmsmSteadyStateBindings:

    def test_params_defaults(self):
        p = ps.PmsmSteadyStateParams()
        assert p.R_s == pytest.approx(0.5)
        assert p.L_s == pytest.approx(2e-3)
        assert p.lambda_pm == pytest.approx(0.1)
        assert p.omega_electrical == pytest.approx(2 * _PI * 50.0, rel=1e-4)
        assert p.positive_sequence is True

    def test_params_writable(self):
        p = ps.PmsmSteadyStateParams()
        p.R_s = 0.3
        p.L_s = 5e-3
        p.lambda_pm = 0.15
        p.omega_electrical = 2 * _PI * 60.0
        p.phase_a_offset_deg = 15.0
        p.positive_sequence = False
        assert p.R_s == pytest.approx(0.3)
        assert p.L_s == pytest.approx(5e-3)
        assert p.lambda_pm == pytest.approx(0.15)
        assert p.positive_sequence is False

    def test_zero_back_emf_matches_rl_load(self):
        """λ_pm = 0 → PMSM degenerates to passive 3-phase RL."""
        circuit = ps.Circuit()
        na, nb, nc = (circuit.add_node(x) for x in "ABC")
        f_grid = 50.0
        omega_e = 2 * _PI * f_grid

        src = ps.ThreePhaseSourceParams()
        src.line_to_line_voltage_rms = 220.0
        src.frequency_hz = f_grid
        circuit.add_three_phase_source(
            "Vgrid", na, nb, nc, ps.Circuit.ground(), src
        )

        pmsm = ps.PmsmSteadyStateParams()
        pmsm.R_s = 0.5
        pmsm.L_s = 2e-3
        pmsm.lambda_pm = 0.0
        pmsm.omega_electrical = omega_e
        circuit.add_pmsm_steady_state(
            "M1", na, nb, nc, ps.Circuit.ground(), pmsm
        )

        result = _run_transient(circuit)
        assert result.success

        v_ln = 220.0 / math.sqrt(3.0)
        z = math.sqrt(0.5**2 + (omega_e * 2e-3) ** 2)
        i_expected = v_ln / z

        # PMSM inductor of phase A is at branch index n_nodes + 3
        # (after 3 source branches).
        n_nodes = circuit.num_nodes()
        i_a = _rms(result.states, n_nodes + 3, result.time, 0.05)
        assert i_a == pytest.approx(i_expected, rel=0.05)

    def test_back_emf_scales_with_omega(self):
        """Back-EMF amplitude = ω_e · λ_pm — doubling ω_e doubles emf."""
        def measure_emf_peak(omega_e: float) -> float:
            circuit = ps.Circuit()
            na, nb, nc = (circuit.add_node(x) for x in "ABC")
            pmsm = ps.PmsmSteadyStateParams()
            pmsm.R_s = 0.5
            pmsm.L_s = 2e-3
            pmsm.lambda_pm = 0.1
            pmsm.omega_electrical = omega_e
            circuit.add_pmsm_steady_state(
                "M1", na, nb, nc, ps.Circuit.ground(), pmsm
            )
            # High-impedance test load to give the solver a closed loop
            for n, name in zip((na, nb, nc), ("a", "b", "c")):
                circuit.add_resistor(f"R_{name}", n, ps.Circuit.ground(), 10000.0)
            result = _run_transient(circuit, tstop=0.1)
            assert result.success
            return max(
                abs(s[na])
                for s, t_i in zip(result.states, result.time)
                if t_i > 0.04
            )

        peak_50 = measure_emf_peak(2 * _PI * 50.0)
        peak_100 = measure_emf_peak(2 * _PI * 100.0)
        assert peak_100 / peak_50 == pytest.approx(2.0, rel=0.05)

    def test_convenience_overload(self):
        """Explicit (R_s, L_s, λ_pm, ω_e) overload works."""
        circuit = ps.Circuit()
        na, nb, nc = (circuit.add_node(x) for x in "ABC")
        circuit.add_pmsm_steady_state(
            "M1", na, nb, nc, ps.Circuit.ground(),
            R_s=0.5, L_s=2e-3,
            lambda_pm=0.1, omega_electrical=2 * _PI * 50.0,
        )
        # Inductors + back-EMF sine sources contribute 6 branches.
        assert circuit.num_branches() == 6
