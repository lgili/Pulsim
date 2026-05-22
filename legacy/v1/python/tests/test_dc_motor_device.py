"""Smoke tests for ``Circuit.add_dc_motor`` (Track 2 device-variant integration).

Mirrors the C++ Catch2 suite in ``core/tests/test_dc_motor_device.cpp``:
verifies the analytical steady-state speed and loaded torque relations are
reachable from Python through the pybind11 bindings.
"""
from __future__ import annotations

import math

import pytest

import pulsim as ps


def _default_params() -> "ps.DcMotorParams":
    p = ps.DcMotorParams()
    p.R_a = 0.5
    p.L_a = 10e-3
    p.K_e = 0.05
    p.K_t = 0.05
    p.J = 1e-4
    p.b = 1e-5
    p.i_a_init = 0.0
    p.omega_init = 0.0
    p.theta_init = 0.0
    return p


def _expected_omega_ss(p: "ps.DcMotorParams", v_a: float, tau_load: float) -> float:
    return (v_a * p.K_t - tau_load * p.R_a) / (p.K_t * p.K_e + p.b * p.R_a)


def _run(circuit: "ps.Circuit", tstop: float, dt: float) -> "ps.TransientResult":
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
    result = sim.run_transient()
    assert result.success, f"Sim failed: {result.message}"
    return result


class TestDcMotorBindings:

    def test_params_default_values(self):
        p = ps.DcMotorParams()
        # Defaults from the math header
        assert p.R_a == pytest.approx(1.0)
        assert p.L_a == pytest.approx(1e-3)
        assert p.K_e == pytest.approx(0.05)
        assert p.K_t == pytest.approx(0.05)
        assert p.J == pytest.approx(1e-4)
        assert p.b == pytest.approx(1e-5)

    def test_params_fields_writable(self):
        p = ps.DcMotorParams()
        p.R_a = 0.25
        p.L_a = 5e-3
        p.K_e = 0.08
        assert p.R_a == pytest.approx(0.25)
        assert p.L_a == pytest.approx(5e-3)
        assert p.K_e == pytest.approx(0.08)

    def test_add_dc_motor_with_params(self):
        circuit = ps.Circuit()
        n_arm = circuit.add_node("arm")
        params = _default_params()
        circuit.add_dc_motor("M1", n_arm, ps.Circuit.ground(), params)
        # Motor reserves one branch row for the armature current.
        assert circuit.num_branches() >= 1

    def test_add_dc_motor_convenience_overload(self):
        circuit = ps.Circuit()
        n_arm = circuit.add_node("arm")
        circuit.add_dc_motor("M2", n_arm, ps.Circuit.ground(),
                             R_a=0.5, L_a=1e-2, K_e=0.05,
                             K_t=0.05, J=1e-4, b=1e-5)
        assert circuit.num_branches() >= 1

    def test_unknown_motor_omega_returns_nan(self):
        circuit = ps.Circuit()
        result = circuit.motor_omega("does_not_exist")
        assert math.isnan(result)

    def test_no_load_step_response_reaches_analytical_speed(self):
        circuit = ps.Circuit()
        n_arm = circuit.add_node("arm")
        v_a = 12.0
        params = _default_params()
        circuit.add_voltage_source("Va", n_arm, ps.Circuit.ground(), v_a)
        circuit.add_dc_motor("M1", n_arm, ps.Circuit.ground(), params)

        _run(circuit, tstop=1.0, dt=100e-6)
        omega = circuit.motor_omega("M1")
        omega_expected = _expected_omega_ss(params, v_a, 0.0)

        # 1% tolerance on the analytical no-load steady-state.
        assert omega == pytest.approx(omega_expected, rel=0.01), (
            f"ω={omega} rad/s, expected ~{omega_expected} rad/s"
        )

    def test_loaded_motor_speed_drops_with_tau_load(self):
        circuit = ps.Circuit()
        n_arm = circuit.add_node("arm")
        v_a = 12.0
        tau_load = 0.01  # 10 mN·m
        params = _default_params()
        circuit.add_voltage_source("Va", n_arm, ps.Circuit.ground(), v_a)
        circuit.add_dc_motor("M1", n_arm, ps.Circuit.ground(), params)
        circuit.set_motor_tau_load("M1", tau_load)

        _run(circuit, tstop=1.0, dt=100e-6)
        omega_loaded = circuit.motor_omega("M1")
        omega_expected = _expected_omega_ss(params, v_a, tau_load)

        assert omega_loaded == pytest.approx(omega_expected, rel=0.015)
        # Loaded speed must be strictly lower than no-load.
        omega_no_load = _expected_omega_ss(params, v_a, 0.0)
        assert omega_loaded < omega_no_load

        # Steady-state armature current matches load torque / K_t.
        i_a = circuit.motor_i_a("M1")
        assert i_a == pytest.approx(tau_load / params.K_t, abs=0.05)

    def test_rotor_angle_integrates_speed(self):
        circuit = ps.Circuit()
        n_arm = circuit.add_node("arm")
        params = _default_params()
        circuit.add_voltage_source("Va", n_arm, ps.Circuit.ground(), 12.0)
        circuit.add_dc_motor("M1", n_arm, ps.Circuit.ground(), params)

        _run(circuit, tstop=0.5, dt=50e-6)
        theta = circuit.motor_theta("M1")
        omega = circuit.motor_omega("M1")

        # Sanity: positive ω drives positive θ. θ < ω·tstop (since ω ramps up).
        assert theta > 0.0
        assert theta < omega * 0.5
