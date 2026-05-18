"""Smoke tests for realistic diode loss + thermal (Phase 1, Pulsim 0.10.0a6).

Validates the pybind11 surface area for the V_F0 + R_d linear-fit diode
model, the loss accumulator (P_avg / peak_P / E_total), the simple
R_th_ja thermal binding, and the bridge-rectifier helper. Mirrors the
Catch2 suite in core/tests/test_diode_loss_thermal.cpp.
"""
from __future__ import annotations

import math

import pytest

import pulsim as ps


def _run_transient(circuit, tstop=20e-3, dt=100e-6):
    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = tstop
    opts.dt = dt
    opts.dt_min = 1e-9
    opts.dt_max = dt
    opts.adaptive_timestep = False
    opts.enable_bdf_order_control = False
    # simplify-and-harden-numerical-surface §11: Auto resolves to Ideal in
    # v0.11. The V_F0 + R_d loss/thermal contract only validates on the
    # Behavioral diode stamp; PWL Ideal uses a unit-conductance switch and
    # would produce spurious test values. Pin Behavioral.
    opts.switching_mode = ps.SwitchingMode.Behavioral
    opts.newton_options.num_nodes = circuit.num_nodes()
    opts.newton_options.num_branches = circuit.num_branches()
    return ps.Simulator(circuit, opts).run_transient()


class TestRealisticDiodeBindings:
    """Verify the pybind11 surface for the new diode loss/thermal model."""

    def test_realistic_diode_params_exposed(self):
        p = ps.RealisticDiodeParams()
        p.V_F0 = 0.497   # GBU2506 MCC tangent fit @ T_j=125 °C
        p.R_d = 0.0108
        p.V_F0_tc = -2e-3
        p.R_th_ja = 25.0
        p.T_amb = 60.0
        assert p.V_F0 == pytest.approx(0.497)
        assert p.R_d == pytest.approx(0.0108)
        assert p.V_F0_tc == pytest.approx(-2e-3)
        assert p.R_th_ja == pytest.approx(25.0)
        assert p.T_amb == pytest.approx(60.0)

    def test_circuit_diode_methods_present(self):
        c = ps.Circuit()
        for name in (
            "diode_average_power", "diode_peak_power", "diode_total_energy",
            "diode_junction_temperature",
            "diode_steady_state_junction_temperature",
            "diode_last_current", "diode_last_voltage",
            "diode_conduction_time", "set_diode_T_j", "reset_diode_loss",
            "add_bridge_rectifier",
        ):
            assert hasattr(c, name), f"Circuit missing method {name!r}"


class TestRealisticDiodeBehavior:
    """End-to-end verification through the pybind boundary."""

    def test_loss_matches_VF_times_I_for_dc_steady_state(self):
        c = ps.Circuit()
        n_a = c.add_node("anode")
        n_l = c.add_node("load")
        gnd = ps.Circuit.ground()

        p = ps.RealisticDiodeParams()
        p.V_F0 = 0.7
        p.R_d = 0.01
        p.V_F0_tc = 0.0          # disable thermal feedback for clean math
        p.g_on = 100.0
        p.R_th_ja = 25.0
        p.T_amb = 25.0

        I_target = 5.0
        R_load = 1.0
        v_src = p.V_F0 + (p.R_d + R_load) * I_target

        c.add_voltage_source("Vsrc", n_a, gnd, v_src)
        c.add_diode("D1", n_a, n_l, p)
        c.add_resistor("Rload", n_l, gnd, R_load)

        r = _run_transient(c)
        assert r.success

        i_d = c.diode_last_current("D1")
        v_d = c.diode_last_voltage("D1")
        p_avg = c.diode_average_power("D1")
        p_expected = (p.V_F0 + p.R_d * I_target) * I_target
        assert i_d == pytest.approx(I_target, rel=0.10)
        assert p_avg == pytest.approx(p_expected, rel=0.15)
        # Steady-state T_j derivation is consistent with the formula.
        t_j = c.diode_steady_state_junction_temperature("D1")
        assert t_j == pytest.approx(p.T_amb + p_avg * p.R_th_ja, rel=0.01)

    def test_bridge_rectifier_composes_4_diodes(self):
        c = ps.Circuit()
        ac_a = c.add_node("ac_a")
        ac_b = c.add_node("ac_b")
        dc_p = c.add_node("dc_p")
        dc_n = c.add_node("dc_n")
        c.add_bridge_rectifier("BR", ac_a, ac_b, dc_p, dc_n, 0.7, 0.01)
        # 4 diodes appended.
        assert c.num_devices() == 4
        # Each diode is reachable by its mangled name.
        for suffix in ("__D1", "__D2", "__D3", "__D4"):
            n = "BR" + suffix
            assert c.diode_average_power(n) == pytest.approx(0.0)

    def test_bridge_rectifier_full_chain_produces_per_diode_telemetry(self):
        """AC source → 4-diode bridge → RC filter. Validates per-diode P_avg
        and T_j accessors return finite numbers after a short transient."""
        c = ps.Circuit()
        ac_a = c.add_node("ac_a")
        dc_p = c.add_node("dc_p")
        dc_n = c.add_node("dc_n")
        gnd = ps.Circuit.ground()

        # 100 V_rms, 50 Hz AC source between ac_a and gnd.
        sine = ps.SineParams()
        sine.amplitude = 100.0 * math.sqrt(2.0)
        sine.frequency = 50.0
        sine.offset = 0.0
        sine.phase = 0.0
        c.add_sine_voltage_source("Vac", ac_a, gnd, sine)

        # GBU2506 MCC tangent fit at T_j=125 °C.
        p = ps.RealisticDiodeParams()
        p.V_F0 = 0.497
        p.R_d = 0.0108
        p.V_F0_tc = 0.0
        p.R_th_ja = 15.0
        p.T_amb = 60.0
        p.g_on = 1.0 / p.R_d

        c.add_bridge_rectifier("BR", ac_a, gnd, dc_p, dc_n, p)
        c.add_capacitor("Cbus", dc_p, dc_n, 220e-6)
        c.add_resistor("Rload", dc_p, dc_n, 200.0)
        c.add_resistor("Rgnd", dc_n, gnd, 1e6)

        r = _run_transient(c, tstop=0.04, dt=50e-6)
        assert r.success

        for suffix in ("__D1", "__D2", "__D3", "__D4"):
            n = "BR" + suffix
            p_avg = c.diode_average_power(n)
            t_j = c.diode_steady_state_junction_temperature(n)
            assert math.isfinite(p_avg), f"{n}: P_avg not finite"
            assert math.isfinite(t_j), f"{n}: T_j not finite"
            assert t_j >= p.T_amb       # only self-heating raises T_j

    def test_fixed_point_thermal_iteration_converges(self):
        """Mimics the powerStage workflow: simulate → read T_j → push T_j
        back via set_diode_T_j → re-simulate. Should converge in ≤5 passes
        for a moderate-current diode."""
        c = ps.Circuit()
        n_a = c.add_node("anode")
        n_l = c.add_node("load")
        gnd = ps.Circuit.ground()

        p = ps.RealisticDiodeParams()
        p.V_F0 = 0.7
        p.R_d = 0.01
        p.V_F0_tc = -2e-3
        p.g_on = 100.0
        p.R_th_ja = 25.0
        p.T_amb = 60.0

        I_target = 5.0
        R_load = 1.0
        v_src = p.V_F0 + (p.R_d + R_load) * I_target
        c.add_voltage_source("Vsrc", n_a, gnd, v_src)
        c.add_diode("D1", n_a, n_l, p)
        c.add_resistor("Rload", n_l, gnd, R_load)

        t_j_prev = p.T_amb
        t_j_curr = p.T_amb
        p_avg = 0.0
        passes = 0
        for passes in range(6):
            r = _run_transient(c, tstop=20e-3, dt=100e-6)
            assert r.success
            t_j_prev = t_j_curr
            # Read both *before* resetting the accumulator so the
            # self-consistency assertion below holds.
            t_j_curr = c.diode_steady_state_junction_temperature("D1")
            p_avg = c.diode_average_power("D1")
            c.set_diode_T_j("D1", t_j_curr)
            c.reset_diode_loss("D1")
            if abs(t_j_curr - t_j_prev) < 0.5:
                break

        assert passes <= 5
        assert t_j_curr > p.T_amb
        # Self-consistency: T_j = T_amb + P_avg · R_th_ja within 5 %.
        assert t_j_curr == pytest.approx(
            p.T_amb + p_avg * p.R_th_ja, rel=0.05
        )
