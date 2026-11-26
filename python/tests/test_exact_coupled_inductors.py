"""Coupled inductors on the exact engine: the extractor applies the full inductance matrix
(L_ii self, L_ij = M) on the inductor rows (add-exact-propagator-strategy DECISIONS, coupled
inductors)."""
import warnings

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


def _coupled_step(k=0.9, L=1e-3, R=1.0):
    # 10 V step on the primary; each winding in series with R; the secondary loop closed
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "p1", R)
    b.add_transformer("T", "p1", "gnd", "s1", "gnd", L, L, k)
    b.add_resistor("R2", "s1", "gnd", R)
    return b


@pytest.mark.parametrize("engine", ENGINES)
def test_two_coupled_inductors_follow_the_closed_form(engine):
    # tasks 7.2: i_1,2 = 5 [(1 - e^{-t/tau+}) +- (1 - e^{-t/tau-})], tau+- = (1 +- k) L / R
    # = 1.9 ms, 0.1 ms (sum and difference modes of L1 = L2 = 1 mH, k = 0.9, R = 1 ohm)
    r = p.simulate(_coupled_step(), 5e-3, engine=engine, dt=1e-5)
    t = np.asarray(r.times)
    assert t.size > 400
    tp, tm = 1.9e-3, 0.1e-3
    i1 = 5.0 * ((1 - np.exp(-t / tp)) + (1 - np.exp(-t / tm)))
    i2 = 5.0 * ((1 - np.exp(-t / tp)) - (1 - np.exp(-t / tm)))   # into the dotted terminal
    i_r1 = np.asarray(r.v("in")) - np.asarray(r.v("p1"))
    i_r2 = np.asarray(r.v("s1"))                                  # out of the dotted terminal
    assert np.abs(i_r1 - i1).max() < 1e-10
    assert np.abs(-i_r2 - i2).max() < 1e-10


def _flyback():
    # 24 V, 100 kHz, D = 0.4; 1:1 windings of 100 uH, k = 0.98; RC snubber on the switch
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 24.0)
    b.add_transformer("T", "vin", "sw", "gnd", "s", 100e-6, 100e-6, 0.98)
    b.add_switch("S", "sw", "gnd", 1e3, 1e-9)
    b.add_resistor("Rsn", "sw", "sn", 100.0)
    b.add_capacitor("Csn", "sn", "gnd", 1e-9)
    b.add_diode("D", "s", "out", 1e3, 1e-9)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("RL", "out", "gnd", 10.0)
    fn = p.make_pwm_switch_fn(1e5, 0.4, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    return b, fn


def _at_instants(r, ts):
    # the sample at each instant (the right limit where an edge stores two): both engines land
    # on every gate edge, so no interpolation error enters the comparison
    t, v = np.asarray(r.times), np.asarray(r.v("out"))
    return np.array([v[np.where(np.abs(t - s) < 1e-13)[0][-1]] for s in ts])


def test_flyback_agrees_with_trbdf2():
    # tasks 7.2: the truth is trbdf2 at rtol 1e-9 (an independent engine), at the 99 period
    # starts of 1 ms. Measured 3.2e-7 relative; trbdf2's own error: 1.0e-6 at rtol 1e-8,
    # 7.8e-8 at 1e-10 (it converges onto the exact run)
    b, fn = _flyback()
    r = p.simulate(b, 1e-3, engine="exact", switch_fn=fn)
    b, fn = _flyback()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # trbdf2's h-floor note at the edges
        q = p.simulate(b, 1e-3, engine="trbdf2", switch_fn=fn, rtol=1e-9, atol=1e-12)
    ts = np.arange(1, 100) * 1e-5
    v, v_q = _at_instants(r, ts), _at_instants(q, ts)
    scale = np.abs(v_q).max()
    assert scale > 5.0                       # the flyback transfers energy
    assert np.abs(v - v_q).max() / scale < 1e-5
