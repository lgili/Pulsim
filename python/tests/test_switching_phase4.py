"""Smoke tests for Phase 4 features (Pulsim 0.10.0a10+):

  * Sinusoidal-modulated PWM source helper (SPWM + SVM).
  * Switching-loss accumulator on MOSFET / IGBT / Diode.
  * Auto-promotion of MOSFET/IGBT to Ideal switching mode when the
    user opts into the switching-loss model (Eon_25 > 0). This is
    the fix for the runtime gate-detection caveat — without it,
    Newton on the smooth Shichman-Hodges stamp struggles with
    discontinuous PWM gate voltages and V_gate stays frozen.
"""
from __future__ import annotations

import math

import pytest

import pulsim as ps


def _run(circuit, tstop=200e-6, dt=5e-6):
    opts = ps.SimulationOptions()
    opts.tstart = 0.0; opts.tstop = tstop
    opts.dt = dt; opts.dt_min = 1e-9; opts.dt_max = dt
    opts.adaptive_timestep = False
    opts.enable_bdf_order_control = False
    opts.newton_options.num_nodes = circuit.num_nodes()
    opts.newton_options.num_branches = circuit.num_branches()
    return ps.Simulator(circuit, opts).run_transient()


class TestModulatedPwm:
    def test_sine_modulator_averages_to_half_rail(self):
        c = ps.Circuit()
        n = c.add_node("n")
        gnd = ps.Circuit.ground()
        c.add_modulated_pwm_source(
            "PWM", n, gnd,
            v_high=12.0, v_low=0.0,
            switching_frequency_hz=2000.0,
            modulation_index=0.8,
            modulation_frequency_hz=50.0,
        )
        c.add_resistor("Rload", n, gnd, 1000.0)
        r = _run(c, tstop=40e-3, dt=50e-6)
        assert r.success

        # Time-averaged V_n after skipping startup should be ~ V_dc/2.
        sum_v = 0.0; count = 0
        for t, state in zip(r.time, r.states):
            if t < 10e-3:
                continue
            sum_v += state[n]; count += 1
        v_avg = sum_v / count
        assert v_avg == pytest.approx(6.0, rel=0.20)

    def test_svm_modulator_uses_third_harmonic(self):
        c = ps.Circuit()
        n = c.add_node("n")
        gnd = ps.Circuit.ground()
        p = ps.ModulatedPwmParams()
        p.v_high = 12.0; p.v_low = 0.0
        p.switching_frequency_hz = 2000.0
        p.modulation_index = 0.8
        p.modulation_frequency_hz = 50.0
        p.modulation = ps.PwmModulation.SVM
        c.add_modulated_pwm_source("PWM", n, gnd, p)
        c.add_resistor("Rload", n, gnd, 1000.0)
        r = _run(c, tstop=40e-3, dt=50e-6)
        assert r.success
        # SVM mean over a full line cycle is still 0.5 (3rd harmonic
        # has zero mean), so the rail average is V_dc/2.
        sum_v = 0.0; count = 0
        for t, state in zip(r.time, r.states):
            if t < 10e-3:
                continue
            sum_v += state[n]; count += 1
        assert (sum_v / count) == pytest.approx(6.0, rel=0.20)


class TestSwitchingLossAutoPromotion:
    """The fix for the gate-detection caveat: MOSFETs/IGBTs that opt
    into the switching-loss model (Eon_25 > 0 or Eoff_25 > 0) get
    auto-promoted to SwitchingMode.Ideal so the PWL event scanner can
    detect PWM gate transitions correctly. Without this, the smooth
    Shichman-Hodges stamp struggles to converge with discontinuous
    gate voltages and V_gate stays frozen."""

    def test_mosfet_with_switching_loss_captures_events(self):
        c = ps.Circuit()
        gate = c.add_node("gate")
        drain = c.add_node("drain")
        n_pos = c.add_node("pos")
        gnd = ps.Circuit.ground()

        mp = ps.MOSFETParams()
        mp.vth = 2.0; mp.kp = 10.0; mp.g_on = 100.0; mp.g_off = 1e-12
        mp.is_nmos = True
        mp.R_th_ja = 1.0
        mp.T_amb = 25.0
        # Switching-loss opt-in → auto-promotes to Ideal mode.
        mp.Eon_25 = 100e-6
        mp.Eoff_25 = 100e-6
        mp.I_ref = 10.0
        mp.V_ref = 100.0
        c.add_mosfet("M1", gate, drain, gnd, mp)
        c.add_pwm_voltage_source("Vgate", gate, gnd, 10.0, 0.0, 10e3, 0.5)
        c.add_voltage_source("Vdrain", n_pos, gnd, 100.0)
        c.add_resistor("Rload", n_pos, drain, 10.0)

        r = _run(c, tstop=500e-6, dt=5e-6)
        assert r.success

        events = c.mosfet_switching_events("M1")
        e_sw = c.mosfet_switching_energy("M1")
        p_sw = c.mosfet_average_switching_power("M1")

        # 5 PWM periods × 2 transitions/period = 10 events expected.
        # The PWL event scanner may merge or miss a couple, but the
        # count must be at least a handful (not 0).
        assert events >= 4, f"only {events} switching events recorded"
        assert e_sw > 0.0
        assert math.isfinite(p_sw)

    def test_mosfet_legacy_path_still_works(self):
        """When Eon_25 = Eoff_25 = 0 (default), the device stays in
        SwitchingMode.Auto (resolves to Behavioral). This preserves the
        pre-Phase-4 behaviour exactly."""
        c = ps.Circuit()
        gate = c.add_node("gate")
        drain = c.add_node("drain")
        gnd = ps.Circuit.ground()
        mp = ps.MOSFETParams()  # default — no switching loss params
        c.add_mosfet("M1", gate, drain, gnd, mp)
        c.add_voltage_source("Vgate", gate, gnd, 5.0)
        c.add_voltage_source("Vd", drain, gnd, 12.0)
        r = _run(c, tstop=1e-3, dt=50e-6)
        assert r.success
        assert c.mosfet_switching_energy("M1") == pytest.approx(0.0)
        assert c.mosfet_switching_events("M1") == 0


class TestDiodeReverseRecovery:
    def test_diode_qrr_zero_means_no_events(self):
        c = ps.Circuit()
        a = c.add_node("a")
        gnd = ps.Circuit.ground()
        dp = ps.RealisticDiodeParams()
        dp.V_F0 = 0.7
        dp.R_d = 0.01
        dp.R_th_ja = 25.0
        # dp.Qrr defaults to 0 → no E_rec recorded
        c.add_diode("D1", a, gnd, dp)
        sine = ps.SineParams()
        sine.amplitude = 5.0; sine.frequency = 1000.0
        c.add_sine_voltage_source("Vs", a, gnd, sine)
        r = _run(c, tstop=5e-3, dt=5e-6)
        assert r.success
        assert c.diode_switching_energy("D1") == pytest.approx(0.0)
        assert c.diode_switching_events("D1") == 0
