"""esdirk43 x a thermal network x a stateful kind (integration batch 14p).

A Lauritzen diode in the same circuit as a switch whose loss port heats an
RC junction: on integrator='esdirk43' the junction follows TR-BDF2 at tight
tolerance (the heat booked agrees) and the diode keeps its closed-form
reverse recovery. A loss port on the Lauritzen diode itself is refused by
name (it has none, on either integrator).
"""
import warnings

import numpy as np
import pytest

import pulsim as p

from .test_esdirk43_stateful_kinds import TD, _laur, _laur_err

T_ON, T_AMB = 5e-6, 25.0


def _circuit():
    b = _laur()
    b.add_voltage_source("V2", "in2", "gnd", 10.0)
    b.add_switch("S1", "in2", "x", 10.0, 1e-9)
    b.add_resistor("RL", "x", "gnd", 5.0)
    b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
    b.add_resistor("R1", "tj", "amb", 2.0)
    b.add_capacitor("C1", "tj", "amb", 2e-6)
    b.set_device_losses("S1", junction="tj")
    n, k = b.graph.num_switches, b.switch_index_of("S1")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t >= T_ON)
        return m
    return b, fn


def _run(**kw):
    b, fn = _circuit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, TD + 5e-6, switch_fn=fn, engine="trbdf2",
                          rtol=1e-10, atol=1e-12, **kw)


def test_a_junction_next_to_a_lauritzen_diode_agrees_on_both_integrators():
    ra, rb = _run(), _run(integrator="esdirk43")
    g = np.linspace(1e-6, TD + 5e-6, 2001)
    tja = np.interp(g, ra.times, ra.v("tj"))
    tjb = np.interp(g, rb.times, rb.v("tj"))
    rise = float(np.max(tja)) - T_AMB
    assert rise > 0.5
    # measured 4.7e-6 of the rise (linear interpolation of both grids)
    assert np.max(np.abs(tja - tjb)) < 5e-5 * rise
    # measured 1.8e-12 relative
    assert rb.heat_injected == pytest.approx(ra.heat_injected, rel=1e-9)
    # the diode's recovery stays on its closed form (TR-BDF2 is 2e-1 there)
    assert _laur_err(rb) < 1e-5


def test_a_loss_port_on_a_lauritzen_diode_is_refused_by_name():
    # refused when the port is declared, before any engine is chosen
    b = _laur()
    b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
    b.add_resistor("R1", "tj", "amb", 2.0)
    with pytest.raises(ValueError, match="LauritzenDiode has no loss port"):
        b.set_device_losses("D", junction="tj")
