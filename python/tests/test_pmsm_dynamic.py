"""Smoke tests for the dynamic PMSM device-variant (``Circuit.add_pmsm``).

Mirrors the Catch2 suite in ``core/tests/test_pmsm_dynamic.cpp`` but
through the pybind boundary. The goal is to verify:

  - The Python wrapper exposes ``add_pmsm`` and the ``pmsm_*`` accessors.
  - The device participates in the MNA solve (zero-magnet → RL load).
  - State accessors return finite numbers after a successful transient.
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


class TestPmsmDynamicBindings:
    """Verify the pybind11 surface area for the dynamic PMSM device."""

    def test_params_struct_exposed(self):
        params = ps.PmsmParams()
        params.Rs = 0.5
        params.Ld = 2e-3
        params.Lq = 2e-3
        params.psi_pm = 0.1
        params.pole_pairs = 2
        params.J = 1e-3
        params.b_friction = 1e-4
        params.omega_init = 100.0
        # Round-trip — reads back what we wrote.
        assert params.Rs == pytest.approx(0.5)
        assert params.psi_pm == pytest.approx(0.1)
        assert params.pole_pairs == 2
        assert params.omega_init == pytest.approx(100.0)

    def test_add_pmsm_method_present(self):
        circuit = ps.Circuit()
        assert hasattr(circuit, "add_pmsm")
        assert hasattr(circuit, "set_pmsm_tau_load")
        assert hasattr(circuit, "pmsm_omega")
        assert hasattr(circuit, "pmsm_theta")
        assert hasattr(circuit, "pmsm_i_d")
        assert hasattr(circuit, "pmsm_i_q")
        assert hasattr(circuit, "pmsm_tau_em")
        assert hasattr(circuit, "pmsm_i_a")
        assert hasattr(circuit, "pmsm_i_b")
        assert hasattr(circuit, "pmsm_i_c")


class TestPmsmDynamicBehavior:
    """End-to-end transient checks through the pybind layer."""

    def test_zero_magnet_acts_as_3_phase_rl_load(self):
        """With ψ_pm=0 the device has no back-EMF — just R+L per phase."""
        circuit = ps.Circuit()
        na, nb, nc = (circuit.add_node(x) for x in "ABC")

        f_grid = 50.0
        omega_e = 2 * _PI * f_grid
        Rs = 0.5
        Ls = 2e-3
        V_LL = 220.0

        src = ps.ThreePhaseSourceParams()
        src.line_to_line_voltage_rms = V_LL
        src.frequency_hz = f_grid
        circuit.add_three_phase_source(
            "Vgrid", na, nb, nc, ps.Circuit.ground(), src
        )

        params = ps.PmsmParams()
        params.Rs = Rs
        params.Ld = Ls
        params.Lq = Ls
        params.psi_pm = 0.0
        params.pole_pairs = 2
        params.J = 1e-3
        params.b_friction = 1e-4
        circuit.add_pmsm(
            "M1", na, nb, nc, ps.Circuit.ground(), params
        )

        result = _run_transient(circuit)
        assert result.success

        # Analytical: V_LN_rms = V_LL/√3, Z = √(R² + (ωL)²).
        v_ln = V_LL / math.sqrt(3.0)
        z = math.sqrt(Rs**2 + (omega_e * Ls) ** 2)
        i_expected = v_ln / z

        # PMSM phase A branch sits at branch_index 3 (after the 3 source
        # branches reserved by add_three_phase_source).
        n_nodes = circuit.num_nodes()
        states = result.states
        t = result.time
        sum_sq = 0.0
        count = 0
        for i, t_i in enumerate(t):
            if t_i < 0.1:
                continue
            v = states[i][n_nodes + 3]
            sum_sq += v * v
            count += 1
        i_a_rms = math.sqrt(sum_sq / count)

        assert i_a_rms == pytest.approx(i_expected, rel=0.07)

    def test_spin_down_under_load(self):
        """ω_init=100, τ_load=0.1, no electrical drive → ω drops over 0.2s."""
        circuit = ps.Circuit()
        na, nb, nc = (circuit.add_node(x) for x in "ABC")

        params = ps.PmsmParams()
        params.Rs = 0.5
        params.Ld = params.Lq = 2e-3
        params.psi_pm = 0.0
        params.pole_pairs = 2
        params.J = 1e-3
        params.b_friction = 5e-4
        params.omega_init = 100.0
        circuit.add_pmsm("M1", na, nb, nc, ps.Circuit.ground(), params)

        # Closure resistors (high-Z) so the solver always has a path.
        for n, name in zip((na, nb, nc), "abc"):
            circuit.add_resistor(f"R_{name}", n, ps.Circuit.ground(), 1e6)

        circuit.set_pmsm_tau_load("M1", 0.1)

        result = _run_transient(circuit, tstop=0.2, dt=100e-6)
        assert result.success

        omega_end = circuit.pmsm_omega("M1")
        # Meaningful drop from 100 rad/s.
        assert omega_end < 100.0
        assert (100.0 - omega_end) > 10.0
        assert math.isfinite(circuit.pmsm_theta("M1"))
        assert math.isfinite(circuit.pmsm_tau_em("M1"))

    def test_dq_currents_are_finite_after_balanced_drive(self):
        """Balanced 3-phase drive produces finite dq/torque telemetry."""
        circuit = ps.Circuit()
        na, nb, nc = (circuit.add_node(x) for x in "ABC")

        src = ps.ThreePhaseSourceParams()
        src.line_to_line_voltage_rms = 30.0
        src.frequency_hz = 50.0
        circuit.add_three_phase_source(
            "Vgrid", na, nb, nc, ps.Circuit.ground(), src
        )

        params = ps.PmsmParams()
        params.Rs = 0.5
        params.Ld = params.Lq = 2e-3
        params.psi_pm = 0.1
        params.pole_pairs = 2
        params.J = 5e-3
        params.b_friction = 1e-3
        params.omega_init = 2 * _PI * 50.0 / params.pole_pairs
        circuit.add_pmsm(
            "M1", na, nb, nc, ps.Circuit.ground(), params
        )

        result = _run_transient(circuit, tstop=0.1, dt=50e-6)
        assert result.success

        for accessor in (
            circuit.pmsm_i_d,
            circuit.pmsm_i_q,
            circuit.pmsm_tau_em,
            circuit.pmsm_omega,
            circuit.pmsm_theta,
            circuit.pmsm_i_a,
            circuit.pmsm_i_b,
            circuit.pmsm_i_c,
        ):
            value = accessor("M1")
            assert math.isfinite(value), f"{accessor.__name__} returned {value}"

        # The motor must produce a non-zero dq projection (otherwise the
        # device is a silent no-op).
        i_d = circuit.pmsm_i_d("M1")
        i_q = circuit.pmsm_i_q("M1")
        assert (i_d * i_d + i_q * i_q) > 1e-6

    def test_convenience_overload(self):
        """Explicit (Rs, Ld, Lq, ψ, p, J, b) overload constructs same device."""
        circuit = ps.Circuit()
        na, nb, nc = (circuit.add_node(x) for x in "ABC")
        circuit.add_pmsm(
            "M1", na, nb, nc, ps.Circuit.ground(),
            0.5, 2e-3, 2e-3, 0.1, 2, 1e-3, 1e-4,
        )
        # Smoke check — device registered.
        assert circuit.num_devices() == 1
        # State accessors return finite defaults (omega_init=0 in convenience).
        assert circuit.pmsm_omega("M1") == pytest.approx(0.0)
