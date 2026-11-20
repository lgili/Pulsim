"""Bergeron transmission line (openspec add-realistic-passives section 5, D18).

A lossless line of surge impedance Z0 and delay tau is, at each port,
v - Z0*i = (the other end's outgoing wave w = v + Z0*i, tau ago). Truths
(closed forms, independent of the engine under test; D18 "Closed forms"):

* matched source, open end, 1 V step: incident 1/2 V, far end 0 before tau
  and exactly 1 V after;
* source 10 ohm on 50 ohm, open end: V+ = 5/6, Gamma_s = -2/3, far end
  after (2n+1) tau = 2 V+ sum_{j<=n} Gamma_s^j = 5/3, 5/9, 35/27 V;
  shorted end: far-end current 2V+/Z0 sum_{j<=n} (-Gamma_s)^j =
  100/3, 500/9, 1900/27 mA;
* energy: what entered both ports equals what is in flight,
  (1/(4 Z0)) * integral over the last tau of (w_a^2 + w_b^2);
* lossy line (R/4 - R/2 - R/4), matched both ends: Z0/(2 Z0 + R) at DC.

On the fixed engine with tau a multiple of dt the far end is the stored
sample (no interpolation), so the line between resistors is exact to
rounding (1e-12 here: no libm on the path).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import pulsim as p

Z0, TAU = 50.0, 100e-9
T_EDGE = 10e-9


def _line(r_s, *, load=None, short=False, dt_edge=1e-12, R=0.0, model="bergeron"):
    """1 V step at T_EDGE behind r_s into a line; the far end open, loaded
    with `load` ohms or shorted by a 0 V source 'Vs'."""
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "s", "gnd", 0.0, 1.0, T_EDGE, 1.0, 0.0,
                               dt_edge, dt_edge)
    b.add_resistor("Rs", "s", "k", r_s)
    b.add_transmission_line("T", "k", "gnd", "m", "gnd", Z0=Z0, delay=TAU,
                            R=R, model=model)
    if load is not None:
        b.add_resistor("RL", "m", "gnd", load)
    if short:
        b.add_voltage_source("Vs", "m", "gnd", 0.0)
    return b


def _at(r, node, t):
    return float(np.interp(t, np.asarray(r.times), np.asarray(r.v(node))))


# ---------------------------------------------------------------------------
# fixed engine (pwl), tau = k*dt: exact
# ---------------------------------------------------------------------------

def test_matched_step_arrives_after_exactly_tau():
    dt = 1e-9   # tau = 100 dt; the edge lies in (10, 11] ns
    r = p.simulate(_line(Z0), t_end=4 * TAU, dt=dt, engine="pwl")
    t, vm, vk = np.asarray(r.times), np.asarray(r.v("m")), np.asarray(r.v("k"))
    before = t <= T_EDGE + TAU + 1e-15
    after = t >= T_EDGE + TAU + dt - 1e-15
    assert np.all(vm[before] == 0.0)
    np.testing.assert_allclose(vm[after], 1.0, rtol=0, atol=1e-12)
    # the near end: 1/2 V until the reflection returns at 2 tau, then 1 V
    np.testing.assert_allclose(vk[(t > T_EDGE + 2 * dt) & (t < T_EDGE + 2 * TAU)],
                               0.5, atol=1e-12)
    assert vm.max() <= 1.0 + 1e-12   # no overshoot


@pytest.mark.parametrize("n, v", [(0, 5 / 3), (1, 5 / 9), (2, 35 / 27)])
def test_open_end_lattice(n, v):
    r = p.simulate(_line(10.0), t_end=7 * TAU, dt=1e-9, engine="pwl")
    t = T_EDGE + (2 * n + 1) * TAU + TAU / 2   # mid plateau
    assert _at(r, "m", t) == pytest.approx(v, rel=1e-12)


@pytest.mark.parametrize("n, i_ma", [(0, 100 / 3), (1, 500 / 9), (2, 1900 / 27)])
def test_shorted_end_current_staircase(n, i_ma):
    r = p.simulate(_line(10.0, short=True), t_end=7 * TAU, dt=1e-9, engine="pwl")
    t = T_EDGE + (2 * n + 1) * TAU + TAU / 2
    tt = np.asarray(r.times)
    i_far = float(np.interp(t, tt, -np.asarray(r.i("T.b"))))   # out of the line
    assert i_far * 1e3 == pytest.approx(i_ma, rel=1e-12)
    assert np.max(np.abs(np.asarray(r.v("m")))) == 0.0


def test_energy_in_equals_energy_in_flight():
    # mismatched both ends: waves keep travelling and decaying
    dt = 1e-9
    r = p.simulate(_line(10.0, load=200.0), t_end=6 * TAU, dt=dt, engine="pwl")
    t = np.asarray(r.times)
    va, vb = np.asarray(r.v("k")), np.asarray(r.v("m"))
    ia, ib = np.asarray(r.i("T")), np.asarray(r.i("T.b"))
    pw = va * ia + vb * ib
    e_in = float(np.sum(0.5 * (pw[1:] + pw[:-1]) * np.diff(t)))
    w2 = (va + Z0 * ia) ** 2 + (vb + Z0 * ib) ** 2
    k = int(round(TAU / dt))
    stored = float(np.sum(0.5 * (w2[-k:] + w2[-k - 1:-1]) * dt)) / (4 * Z0)
    assert e_in > 1e-12
    assert stored == pytest.approx(e_in, rel=1e-12)


def test_a_lossless_line_is_a_wire_at_dc_and_starts_without_a_wave():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "s", "gnd", 1.0)
    b.add_resistor("Rs", "s", "k", 10.0)
    b.add_transmission_line("T", "k", "gnd", "m", "gnd", Z0=Z0, delay=TAU,
                            model="bergeron")
    b.add_resistor("RL", "m", "gnd", 40.0)
    r = p.simulate(b, t_end=5 * TAU, dt=1e-9, engine="pwl", start_from_dc_op=True)
    vm = np.asarray(r.v("m"))
    np.testing.assert_allclose(vm, 0.8, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(r.i("T")), 0.02, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(r.i("T.b")), -0.02, rtol=1e-9)


# ---------------------------------------------------------------------------
# fixed engine, tau not a multiple of dt: interpolated far end
# ---------------------------------------------------------------------------

def test_non_integer_ratio_plateaus_exact_front_within_one_step():
    dt = TAU / 100.5
    r = p.simulate(_line(10.0), t_end=6 * TAU, dt=dt, engine="pwl")
    for n, v in [(0, 5 / 3), (1, 5 / 9), (2, 35 / 27)]:
        assert _at(r, "m", T_EDGE + (2 * n + 1) * TAU + TAU / 2) == pytest.approx(v, rel=1e-12)
    # the front: zero up to tau after the edge's step, the plateau one step later
    t, vm = np.asarray(r.times), np.asarray(r.v("m"))
    assert np.all(vm[t <= T_EDGE + TAU - dt] == 0.0)
    assert np.all(np.abs(vm[(t >= T_EDGE + TAU + 2 * dt) & (t <= 3 * TAU)] - 5 / 3) < 1e-12)


def _sine_far_error(ratio):
    """Matched line, sine behind Z0 into Z0: v_far = sin(w(t - tau))/2."""
    f = 1 / (8 * TAU)
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "s", "gnd", 0.0, 1.0, f)
    b.add_resistor("Rs", "s", "k", Z0)
    b.add_transmission_line("T", "k", "gnd", "m", "gnd", Z0=Z0, delay=TAU,
                            model="bergeron")
    b.add_resistor("RL", "m", "gnd", Z0)
    r = p.simulate(b, t_end=10 * TAU, dt=TAU / ratio, engine="pwl")
    t, vm = np.asarray(r.times), np.asarray(r.v("m"))
    keep = t > 2 * TAU
    exact = 0.5 * np.sin(2 * np.pi * f * (t[keep] - TAU))
    return float(np.max(np.abs(vm[keep] - exact)))


def test_interpolation_is_second_order_and_exact_on_the_grid():
    e = [_sine_far_error(x) for x in (25.5, 50.5, 100.5)]   # theta = 1/2 each
    orders = [np.log2(e[k] / e[k + 1]) for k in range(2)]
    assert all(1.8 < q < 2.2 for q in orders), (e, orders)
    # theta (1 - theta) (w dt)^2 / 2 per pass, D18 (1): the amplitude loss bound
    w_dt = 2 * np.pi / (8 * 100.5)
    assert e[2] <= 0.5 * 0.25 * w_dt ** 2 / 2 * 1.05
    assert _sine_far_error(100.0) < 1e-12   # tau = 100 dt: no interpolation


def test_restart_rule_and_event_split_keep_the_plateaus():
    for kw in ({"discontinuity_restart": True},
               {"discontinuity_restart": True, "enable_substep_state_correction": True}):
        r = p.simulate(_line(10.0, dt_edge=0.3e-9), t_end=6 * TAU, dt=1e-9,
                       engine="pwl", **kw)
        for n, v in [(0, 5 / 3), (1, 5 / 9), (2, 35 / 27)]:
            assert _at(r, "m", T_EDGE + (2 * n + 1) * TAU + TAU / 2) == pytest.approx(v, rel=1e-12), kw


# ---------------------------------------------------------------------------
# lossy line, resume, snapshot, refusals
# ---------------------------------------------------------------------------

def test_lossy_line_dc_attenuation():
    r = p.simulate(_line(Z0, load=Z0, R=1.0), t_end=60 * TAU, dt=1e-9, engine="pwl")
    assert _at(r, "m", 60 * TAU) == pytest.approx(Z0 / (2 * Z0 + 1.0), rel=1e-6)
    names = {c["name"] for c in _line(Z0, load=Z0, R=1.0).components()}
    assert {"T.ra", "T.rm", "T.rb", "T", "T.b", "T.h1.b", "T.h2"} <= names


def test_resume_equals_the_uninterrupted_run():
    full = p.simulate(_line(10.0, load=200.0), t_end=6 * TAU, dt=1e-9, engine="pwl")
    a = p.simulate(_line(10.0, load=200.0), t_end=2.35 * TAU, dt=1e-9, engine="pwl")
    snap = pickle.loads(pickle.dumps(a.final_snapshot))
    assert len(snap.line_state) > 0
    # bounded: tau/dt + 2 samples at most
    assert snap.line_state[3] <= TAU / 1e-9 + 2
    b = p.simulate(_line(10.0, load=200.0), t_end=6 * TAU, dt=1e-9, engine="pwl",
                   resume_from=snap)
    tf, vf = np.asarray(full.times), np.asarray(full.v("m"))
    tb, vb = np.asarray(b.times), np.asarray(b.v("m"))
    np.testing.assert_allclose(np.interp(tb, tf, vf), vb, rtol=0, atol=1e-12)


def test_snapshot_without_a_line_pickles_as_before():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "s", "gnd", 1.0)
    b.add_resistor("R", "s", "gnd", 1.0)
    b.add_capacitor("C", "s", "gnd", 1e-6)
    s = p.simulate(b, t_end=1e-6, dt=1e-7, engine="pwl").final_snapshot
    assert len(s.__getstate__()) == 19
    assert list(s.line_state) == []


def test_refusals_by_name():
    with pytest.raises(p.UnsupportedError, match="'T'") as ei:
        p.simulate(_line(Z0), t_end=TAU, dt=2 * TAU, engine="pwl")
    assert (ei.value.feature, ei.value.device) == ("dt", "T")
    with pytest.raises(ValueError, match="R <= Z0|exceeds Z0"):
        _line(Z0, R=60.0)
    with pytest.raises(ValueError, match="sections"):
        p.CircuitBuilder().add_transmission_line(
            "T", "k", "gnd", "m", "gnd", Z0=Z0, delay=TAU, model="bergeron",
            sections=4)


# ---------------------------------------------------------------------------
# AC sweep: the exact two-port (D18 (5))
# ---------------------------------------------------------------------------

def _sweep(b, inj, out, freqs):
    from pulsim._pulsim import run_mna_sweep_kernel
    r = run_mna_sweep_kernel(b.graph, b.pool, p.SwitchStateMask(0), list(freqs),
                             b.node_id_of(inj), b.node_id_of(out))
    return np.asarray(r.H)


def _loaded(r_load):
    b = p.CircuitBuilder()
    b.add_transmission_line("T", "k", "gnd", "m", "gnd", Z0=Z0, delay=TAU,
                            model="bergeron")
    b.add_resistor("RL", "m", "gnd", r_load)
    return b


def test_ac_matched_line_is_a_pure_delay():
    f = np.array([0.0, 1e5, 1.3e6, 2.5e6, 7.7e6, 5e7])
    z_in = _sweep(_loaded(Z0), "k", "k", f)
    z_tr = _sweep(_loaded(Z0), "k", "m", f)
    np.testing.assert_allclose(z_in, Z0, rtol=1e-9)
    np.testing.assert_allclose(z_tr, Z0 * np.exp(-2j * np.pi * f * TAU), rtol=1e-9, atol=1e-9)


def test_ac_quarter_wave_transformer():
    f_q = 1 / (4 * TAU)   # omega tau = pi/2
    for r_l in (10.0, 200.0):
        z = _sweep(_loaded(r_l), "k", "k", [f_q])[0]
        assert z.real == pytest.approx(Z0 ** 2 / r_l, rel=1e-9)
        assert abs(z.imag) < 1e-9 * abs(z)
    # half wave (sin = 0, singular in Y form) reflects the load itself
    z = _sweep(_loaded(10.0), "k", "k", [1 / (2 * TAU)])[0]
    assert z.real == pytest.approx(10.0, rel=1e-9)


def test_quarter_wave_transient_steady_state_matches_the_sweep():
    # sine current 1 mA into the near end at f = 1/(4 tau), load 10 ohm:
    # |v_k| = Z0^2/R_L * 1 mA = 0.25 V once the reflections have died out
    f_q = 1 / (4 * TAU)
    b = _loaded(10.0)
    b.add_resistor("Rp", "k", "gnd", 1e6)   # keeps the zero-frequency path
    b.add_sine_current_source("I", "gnd", "k", 0.0, 1e-3, f_q)
    r = p.simulate(b, t_end=400 * TAU, dt=TAU / 50, engine="pwl")
    t, v = np.asarray(r.times), np.asarray(r.v("k"))
    keep = t >= t[-1] - 20 / f_q
    w = 2 * np.pi * f_q * t[keep]
    a = np.column_stack([np.sin(w), np.cos(w), np.ones_like(w)])
    c = np.linalg.lstsq(a, v[keep], rcond=None)[0]
    z_ac = _sweep(b, "k", "k", [f_q])[0] * 1e-3
    assert np.hypot(c[0], c[1]) == pytest.approx(abs(z_ac), rel=1e-6)
