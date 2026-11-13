"""A loop of voltage sources and inductors has no DC operating point.

``compute_dc_op`` returned 2.4e13 A for a 48 V source across two series
inductors (the inductors' 1e-12 ohm DC regularisation), and
``ll16_flyback_closed_loop`` started its transient from that state on
every engine (openspec/changes/add-newton-roundoff-acceptance/probes/
ll16_dc_op.py). The loop is found structurally and refused by name.
"""

import numpy as np
import pytest

import pulsim as p


def _series_lls(r=None):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 48.0)
    b.add_inductor("L1", "vin", "d", 2e-6)
    if r is None:
        b.add_inductor("L2", "d", "gnd", 158e-6)
    else:
        b.add_inductor("L2", "d", "e", 158e-6)
        b.add_resistor("R", "e", "gnd", r)
    return b


def _refusal(b, **kw):
    with pytest.raises(p.UnsupportedError) as ei:
        p.compute_dc_op(b, **kw)
    return ei.value


def test_source_across_series_inductors_is_refused_by_name():
    e = _refusal(_series_lls())
    assert e.feature == "dc_operating_point"
    assert e.device == "V"
    assert "L1" in str(e) and "L2" in str(e) and "48 V" in str(e)
    assert "zero state" in str(e) and "winding resistance" in str(e)


def test_with_series_resistance_the_current_is_v_over_r():
    b = _series_lls(r=4.0)
    x = np.asarray(p.compute_dc_op(b))
    names = b.state_var_names()
    assert x[names.index("I(L2)")] == pytest.approx(12.0, rel=1e-9)
    assert x[names.index("V(d)")] == pytest.approx(48.0, rel=1e-9)


def test_loops_without_emf_are_not_refused():
    # Inductors in parallel (no source in the loop) and a pulse source
    # that is 0 V at t = 0 across an inductor: finite (zero) solutions.
    b = p.CircuitBuilder()
    b.add_current_source("I", "gnd", "a", 1.0)
    b.add_inductor("La", "a", "gnd", 1e-3)
    b.add_inductor("Lb", "a", "gnd", 2e-3)
    b.add_pulse_voltage_source("Vp", "c", "gnd", 0.0, 5.0, 1e-6, 1e-6)
    b.add_inductor("Lc", "c", "gnd", 1e-3)
    x = np.asarray(p.compute_dc_op(b))
    assert np.all(np.isfinite(x)) and np.abs(x).max() < 10.0


def test_loop_through_coupled_windings_is_refused():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_transformer("T", "a", "gnd", "s", "gnd", 1e-3, 4e-3, 0.98)
    b.add_resistor("R", "s", "gnd", 10.0)
    e = _refusal(b)
    assert e.device == "V"


def test_loop_through_an_ideal_transformer_is_refused_only_when_closed():
    # Secondary shorted by an inductor: the primary is a short too, and a
    # source through an inductor across it is a loop.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_inductor("Lk", "a", "p", 1e-6)
    b.add_ideal_transformer("X", "p", "gnd", "s", "gnd", 2.0)
    b.add_inductor("Ls", "s", "gnd", 1e-3)
    e = _refusal(b)
    assert "X" in str(e) and "Ls" in str(e)
    # The same source through a resistor: legitimate, primary at 0 V.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_resistor("R", "a", "p", 5.0)
    b.add_ideal_transformer("X", "p", "gnd", "s", "gnd", 2.0)
    b.add_inductor("Ls", "s", "gnd", 1e-3)
    x = np.asarray(p.compute_dc_op(b))
    assert x[b.state_var_names().index("V(p)")] == pytest.approx(0.0, abs=1e-9)
    # A leakage inductor and a resistive load: the source sets v_p = 10 V.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_inductor("Lk", "a", "p", 1e-6)
    b.add_ideal_transformer("X", "p", "gnd", "s", "gnd", 2.0)
    b.add_resistor("Rl", "s", "gnd", 8.0)
    x = np.asarray(p.compute_dc_op(b))
    assert x[b.state_var_names().index("V(s)")] == pytest.approx(20.0, rel=1e-9)


@pytest.mark.parametrize("strategy", ["auto", "naive", "gmin_step", "source_step"])
def test_every_strategy_refuses(strategy):
    _refusal(_series_lls(), strategy=strategy)


def test_simulate_start_from_dc_op_lets_the_refusal_through():
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(_series_lls(), t_end=1e-6, dt=1e-8, engine="pwl",
                   start_from_dc_op=True)
    assert ei.value.engine == "pwl"
    assert ei.value.feature == "dc_operating_point"
