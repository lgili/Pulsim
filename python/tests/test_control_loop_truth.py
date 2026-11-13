"""The control helpers report the loop they build, and tick on schedule.

Found auditing examples/scripts/run_auto_tune_buck.py, which printed
"achieved PM = 60.00°" for a loop whose measured margin was 37.9°:

* ``tune_pi_from_bode`` returned ``achieved_pm_deg = phase_margin_deg``
  "by construction" at ``f_crossover``. On a buck (Q = 3.4) asked for
  60° at 2 kHz, |L| rises above 1 again at the LC peak and the last
  0 dB crossing, the one that sets stability, has 6.0°.
* ``phase_margin_from_loop`` read the wrapped angle, so a loop lagging
  205° at crossover (unstable) reported +335° instead of -25°.
* ``bind_pi_to_switch``'s self-throttle (``t - last >= T``) skipped a
  tick whenever a step landed a floating-point hair before k*T:
  2978 updates instead of 3000 over 30 ms at dt = T/100.
"""
from __future__ import annotations

import numpy as np
import pytest

import pulsim as p

L_H, C_F, R_OHM, V_IN = 100e-6, 47e-6, 5.0, 24.0
F = np.logspace(1, 4.6, 3000)


def _buck_gvd(f):
    s = 2j * np.pi * np.asarray(f)
    return V_IN / (L_H * C_F * s**2 + L_H / R_OHM * s + 1)


def test_tuner_reports_the_margin_of_the_loop_it_returns():
    G = _buck_gvd(F)
    t = p.tune_pi_from_bode(F, G, f_crossover=2000.0, phase_margin_deg=60.0)
    real = p.phase_margin_from_loop(F, p.loop_gain(F, G, t["Kp"], t["Ki"]))
    assert t["achieved_pm_deg"] == pytest.approx(real, abs=1e-9)
    assert real == pytest.approx(6.03, abs=0.05)   # was reported as 60.00
    assert any("crosses 0 dB" in w for w in t["warnings"])
    assert len(t["loop_crossovers_hz"]) == 2


def test_tuner_single_crossover_design_keeps_its_target():
    G = _buck_gvd(F)
    t = p.tune_pi_from_bode(F, G, f_crossover=300.0, phase_margin_deg=88.0)
    assert t["achieved_pm_deg"] == pytest.approx(88.0, abs=0.05)
    assert t["warnings"] == []


def test_phase_margin_of_a_loop_lagging_past_180_is_negative():
    f = np.logspace(1, 4, 2000)
    s = 2j * np.pi * f
    w_c = 2 * np.pi * 1e3
    # Integrator crossing at 1 kHz plus a delay worth 115° there: lag 205°.
    L = (w_c / s) * np.exp(-s * np.radians(115.0) / w_c)
    assert p.phase_margin_from_loop(f, L) == pytest.approx(-25.0, abs=0.1)


def test_bind_pi_to_switch_self_throttle_ticks_once_per_period():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_resistor("R", "sw", "gnd", 10.0)
    dt, f_pwm, t_end = 1e-7, 100e3, 3e-3
    loop = p.bind_pi_to_switch(
        b, pi=p.PIController(Kp=0.0, Ki=0.0, output_min=0.5, output_max=0.5),
        measured=lambda x: 0.0, setpoint=0.0, switch="S", freq=f_pwm)
    # The user-composed observer pattern: the loop's throttle decides.
    p.simulate(b, t_end=t_end, dt=dt, switch_fn=loop.switch_fn,
               step_observer=lambda t, x: loop.step_observer(t, x))
    ticks = np.array([t for t, _ in loop.duty_history])
    assert len(ticks) == round(t_end * f_pwm) + 1     # k = 0 .. 300 inclusive
    np.testing.assert_allclose(np.diff(ticks), 1 / f_pwm, rtol=1e-6)
