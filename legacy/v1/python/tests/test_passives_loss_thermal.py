"""Smoke tests for Capacitor / Resistor / Inductor loss + thermal
(Phase 3 of inverter-bridge-losses, Pulsim 0.10.0a9).

Verifies the pybind boundary for ResistorParams / CapacitorParams /
InductorParams plus the matching Circuit accessors. Mirrors the
Catch2 suite in `core/tests/test_passives_loss_thermal.cpp`.
"""
from __future__ import annotations

import math

import pytest

import pulsim as ps


def _run(circuit, tstop=10e-3, dt=100e-6):
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


class TestPassivesBindings:
    """Verify Params classes + Circuit accessors are exposed."""

    def test_resistor_params_exposed(self):
        p = ps.ResistorParams()
        p.resistance = 100.0
        p.TCR = 4e-3
        p.R_th_ja = 50.0
        p.T_amb = 25.0
        assert p.resistance == pytest.approx(100.0)
        assert p.TCR == pytest.approx(4e-3)
        assert p.R_th_ja == pytest.approx(50.0)

    def test_capacitor_params_exposed(self):
        p = ps.CapacitorParams()
        p.capacitance = 330e-6
        p.ESR = 0.5
        p.ESR_tc = 0.01
        p.R_th_ja = 8.0
        assert p.capacitance == pytest.approx(330e-6)
        assert p.ESR == pytest.approx(0.5)
        assert p.R_th_ja == pytest.approx(8.0)

    def test_inductor_params_exposed(self):
        p = ps.InductorParams()
        p.inductance = 200e-6
        p.DCR = 0.05
        p.R_th_ja = 12.0
        assert p.inductance == pytest.approx(200e-6)
        assert p.DCR == pytest.approx(0.05)

    def test_circuit_methods_present(self):
        c = ps.Circuit()
        for prefix in ("capacitor", "resistor", "inductor"):
            for suffix in (
                "_average_power", "_peak_power", "_total_energy",
                "_junction_temperature",
                "_steady_state_junction_temperature", "_last_current",
            ):
                assert hasattr(c, prefix + suffix), f"missing {prefix + suffix}"
        for prefix in ("capacitor", "resistor", "inductor"):
            assert hasattr(c, f"set_{prefix}_T_j")
            assert hasattr(c, f"reset_{prefix}_loss")


class TestPassivesBehavior:
    """End-to-end Python validation of the loss accumulator."""

    def test_resistor_loss_tracked_without_thermal_feedback(self):
        # Unified pipeline (Pulsim 0.10.0a12+): the per-device loss
        # accumulator runs always. `R_th_ja == 0` only disables T_j
        # feedback (the resistance is taken as the static value).
        c = ps.Circuit()
        n = c.add_node("n")
        c.add_voltage_source("Vs", n, ps.Circuit.ground(), 10.0)
        c.add_resistor("R1", n, ps.Circuit.ground(), 100.0)
        r = _run(c)
        assert r.success
        # P = V²/R = 100/100 = 1 W (no thermal feedback).
        assert c.resistor_average_power("R1") == pytest.approx(1.0, rel=0.05)

    def test_resistor_with_thermal_dissipates_VV_over_R(self):
        c = ps.Circuit()
        n = c.add_node("n")
        c.add_voltage_source("Vs", n, ps.Circuit.ground(), 10.0)
        rp = ps.ResistorParams()
        rp.resistance = 10.0
        rp.TCR = 0.0
        rp.R_th_ja = 2.0
        rp.T_amb = 25.0
        c.add_resistor("R1", n, ps.Circuit.ground(), rp)
        r = _run(c)
        assert r.success
        # P = V²/R = 100/10 = 10 W
        assert c.resistor_average_power("R1") == pytest.approx(10.0, rel=0.05)
        # T_j = T_amb + P · R_th_ja = 25 + 10·2 = 45 °C
        t_j = c.resistor_steady_state_junction_temperature("R1")
        assert t_j == pytest.approx(45.0, rel=0.05)

    def test_inductor_DCR_loss(self):
        # V → L → R_load → GND. Steady-state I = V/R_load.
        c = ps.Circuit()
        a = c.add_node("a")
        b = c.add_node("b")
        c.add_voltage_source("Vs", a, ps.Circuit.ground(), 5.0)
        lp = ps.InductorParams()
        lp.inductance = 1e-3
        lp.DCR = 0.05
        lp.DCR_tc = 0.0
        lp.R_th_ja = 3.0
        lp.T_amb = 25.0
        c.add_inductor("L1", a, b, lp)
        c.add_resistor("Rload", b, ps.Circuit.ground(), 1.0)
        r = _run(c, tstop=50e-3, dt=100e-6)
        assert r.success
        # I_steady = 5/1 = 5 A. P = 25·0.05 = 1.25 W.
        assert c.inductor_average_power("L1") == pytest.approx(1.25, rel=0.10)

    def test_capacitor_legacy_mode_zero_loss(self):
        c = ps.Circuit()
        n = c.add_node("n")
        c.add_voltage_source("Vs", n, ps.Circuit.ground(), 5.0)
        c.add_capacitor("C1", n, ps.Circuit.ground(), 1e-6)
        r = _run(c)
        assert r.success
        assert c.capacitor_average_power("C1") == pytest.approx(0.0)


class TestPowerStageCapMatch:
    """A focused sanity check that mirrors the user's notebook setup:
    330 µF/400 V bulk cap on a single AC source. Verifies the
    accumulator produces a finite, non-trivial dissipation."""

    def test_330uf_cap_with_ESR(self):
        c = ps.Circuit()
        ac = c.add_node("ac")
        gnd = ps.Circuit.ground()
        # 100 V_rms / 60 Hz AC source (matches the inverter_550W
        # operating point your notebook uses).
        sine = ps.SineParams()
        sine.amplitude = 100.0 * math.sqrt(2.0)
        sine.frequency = 60.0
        sine.offset = 0.0
        sine.phase = 0.0
        c.add_sine_voltage_source("Vac", ac, gnd, sine)

        cp = ps.CapacitorParams()
        cp.capacitance = 330e-6
        cp.ESR = 0.05      # typical generic 330 µF/400 V AlElCap ESR
        cp.ESR_tc = 0.0
        cp.R_th_ja = 8.0
        cp.T_amb = 60.0
        c.add_capacitor("Cbulk", ac, gnd, cp)

        r = _run(c, tstop=50e-3, dt=20e-6)
        assert r.success
        p_avg = c.capacitor_average_power("Cbulk")
        t_j   = c.capacitor_steady_state_junction_temperature("Cbulk")
        assert math.isfinite(p_avg)
        assert p_avg > 0.0
        assert math.isfinite(t_j)
        assert t_j > cp.T_amb
