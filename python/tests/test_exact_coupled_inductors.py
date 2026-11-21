"""Coupled inductors on the exact engine: the extractor applies the full inductance matrix
(L_ii self, L_ij = M) on the inductor rows (add-exact-propagator-strategy DECISIONS, coupled
inductors)."""
import numpy as np
import pytest

import pulsim as p

ENGINES = ["dsed", "exact"]


def _rectifier(k=0.95):
    # sine -> Rp -> 1:1 transformer (k = 0.95, leakage) -> diode -> C || R
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 20.0, 1e3)
    b.add_resistor("Rp", "in", "p1", 1.0)
    b.add_transformer("T", "p1", "gnd", "s1", "gnd", 1e-3, 1e-3, k)
    b.add_diode("D", "s1", "out", 1e3, 1e-9)
    b.add_capacitor("C", "out", "gnd", 1e-5)
    b.add_resistor("RL", "out", "gnd", 100.0)
    return b


@pytest.mark.parametrize("engine", ENGINES)
def test_a_transformer_fed_rectifier_matches_pwl_at_fine_dt(engine):
    r = p.simulate(_rectifier(), 5e-3, engine=engine, dt=1e-6)
    ref = p.simulate(_rectifier(), 5e-3, engine="pwl", dt=6.25e-8)
    t, v = np.asarray(r.times), np.asarray(r.v("out"))
    v_ref = np.interp(t, np.asarray(ref.times), np.asarray(ref.v("out")))
    scale = np.abs(v_ref).max()
    assert scale > 5.0                       # the rectifier conducts
    # pwl itself at dt = 250 ns sits 1.4e-6 (relative rms) from this reference
    assert np.sqrt(np.mean((v - v_ref) ** 2)) / scale < 1e-5


@pytest.mark.parametrize("engine", ENGINES)
def test_a_unity_coupling_is_refused(engine):
    # |k| = 1: the inductance matrix is singular, the winding currents are not independent
    with pytest.raises(p.UnsupportedError):
        p.simulate(_rectifier(k=1.0), 1e-4, engine=engine, dt=1e-6)


def _open_secondary(k):
    # the secondary winding is open (an inductor cut-set): the dependent-state reduction
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("Rp", "in", "p1", 1.0)
    b.add_transformer("T", "p1", "gnd", "s1", "gnd", 1e-3, 1e-3, k)
    b.add_resistor("Rs", "s1", "x", 1.0)
    return b


@pytest.mark.parametrize("engine", ENGINES)
def test_a_unity_coupling_on_a_cut_set_is_refused_by_name(engine):
    # the general path takes the same positive-definite criterion as the structural one
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(_open_secondary(1.0), 1e-4, engine=engine, dt=1e-6)
    assert "not positive definite" in str(ei.value)
    r = p.simulate(_open_secondary(0.9), 1e-4, engine=engine, dt=1e-6)
    # the open secondary carries no current; the primary charges as an RL of L = 1 mH
    t = np.asarray(r.times)
    i_p = np.asarray(r.v("in")) - np.asarray(r.v("p1"))      # through Rp = 1 ohm
    np.testing.assert_allclose(i_p, 1.0 - np.exp(-t / 1e-3), rtol=1e-6, atol=1e-9)
