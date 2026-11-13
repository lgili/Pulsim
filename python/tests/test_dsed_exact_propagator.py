"""add-exact-propagator-strategy, first slice — the dsed exact stepper
against 50-digit references, and the robustness gaps it closes.

Circuit of the audit (``numerics/p2_buck.py``): 24 V, 100 uH, 47 uF,
g_on = 1e3, g_off = 1e-9, 100 kHz. References from
``openspec/changes/add-exact-propagator-strategy/probes/
references_50_digits.py``. Every case below either aborted or was less
accurate on the untouched kernel (``probes/dsed_baseline.py``).
"""

import warnings

import numpy as np
import pytest

import pulsim as p

VIN, L0, C0, GON, GOFF, T = 24.0, 100e-6, 47e-6, 1e3, 1e-9, 1e-5


def _buck(R, vth, *, L=L0, D=0.5, t0=0.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF, vth)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", C0)
    b.add_resistor("RL", "vout", "gnd", R)
    fn = p.make_pwm_switch_fn(1 / T, D, b.switch_index_of("S"),
                              b.graph.num_switches, t0)
    return b, fn


def _run(b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, engine="dsed", **kw)


def test_ccm_buck_matches_the_50_digit_reference():
    b, fn = _buck(1.0, 1e-6)
    r = _run(b, t_end=40 * T, switch_fn=fn, dt=T / 10)
    assert abs(np.asarray(r.i("L"))[-1] - 11.8015131102418) < 1e-10
    assert abs(np.asarray(r.v("vout"))[-1] - 12.2288664641139) < 1e-10


@pytest.mark.parametrize("vth", [1e-6, 0.0])
def test_dcm_orbit_and_turn_off_instant(vth):
    # V_th = 0 aborted on the chatter guard at t = 0.996 ms before.
    nper = 4000
    b, fn = _buck(50.0, vth, D=0.3)
    r = _run(b, t_end=nper * T, switch_fn=fn)
    t = np.asarray(r.times)
    k = int(np.argmin(abs(t - (nper - 1) * T)))
    assert abs(np.asarray(r.v("vout"))[k] - 8.99624708728943) < 1e-9
    off = r.event_times[(r.event_types == 2) & (r.event_times > (nper - 1) * T)]
    tau = (off[0] - (nper - 1) * T) / T
    # At t = 40 ms one ulp of t is 6.9e-18 s = 6.9e-13 T: the bound
    # max(4u|t|, 1e-12 h) of the time bracket is 3.5e-12 T here.
    assert abs(tau - 0.799830698444468) < 5e-12


def _rest(vth):
    b, _ = _buck(50.0, vth)
    n = b.graph.num_switches
    return _run(b, t_end=1e-6, switch_fn=lambda t: p.SwitchStateMask(n))


def test_zero_threshold_diode_at_rest_settles():
    # "did not settle after 64 rounds at t = 0.000000" before.
    a, c = _rest(0.0), _rest(1e-9)
    np.testing.assert_allclose(np.asarray(a.states)[-1],
                               np.asarray(c.states)[-1], rtol=0, atol=1e-12)
    # A rounding-level ON crossing may be located near t = 0; the zero
    # band then keeps the diode OFF: it never conducts.
    assert not np.asarray(a.switch_states).any()


def _vend(L):
    b, fn = _buck(50.0, 1e-6, L=L, D=0.3)
    return float(np.asarray(_run(b, t_end=40 * T, switch_fn=fn).v("vout"))[-1])


@pytest.mark.parametrize("delta", [1e-5, -1e-5, 1e-4, -1e-4, 1e-2, -1e-2])
def test_dcm_idle_mode_does_not_chatter(delta):
    # Four of these six aborted with "D1 is chattering" before.
    assert 13.30 < _vend(L0 * (1 + delta)) < 13.32


def test_finite_difference_in_L_is_clean():
    d = 1e-5 * L0
    fd = (_vend(L0 + d) - _vend(L0 - d)) / (2 * d)
    assert abs(fd / 1758.1254 - 1) < 1e-6


@pytest.mark.parametrize("vth", [0.0, 1e-9])
def test_gate_held_off_then_dcm(vth):
    b, fn = _buck(50.0, vth, D=0.3, t0=2e-6)
    r = _run(b, t_end=2e-6 + 40 * T, switch_fn=fn)
    assert abs(np.asarray(r.v("vout"))[-1] - 13.309497) < 5e-7


def _rlc(R, L=100e-6, C=100e-9):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "a", R)
    b.add_inductor("L", "a", "o", L)
    b.add_capacitor("C", "o", "gnd", C)
    return b


def test_critically_damped_rlc_is_not_taken_on_tier_1():
    # R = 2*sqrt(L/C) exactly: the untouched kernel accepted a basis
    # with kappa ~ 1e9 on isInvertible() and was 1.1e-6 off.
    L, C = 100e-6, 100e-9
    h = 2 * np.pi * np.sqrt(L * C)
    r = _run(_rlc(2 * np.sqrt(L / C)), t_end=h)
    # 50-digit mpmath reference from x0 = 0 (same doubles).
    import mpmath as mp
    mp.mp.dps = 40
    R = 2 * np.sqrt(L / C)
    M = mp.matrix([[-mp.mpf(R) / L, -1 / mp.mpf(L), 10 / mp.mpf(L)],
                   [1 / mp.mpf(C), 0, 0], [0, 0, 0]])
    x = mp.expm(M * mp.mpf(h)) * mp.matrix([0, 0, 1])
    assert abs(np.asarray(r.v("o"))[-1] - float(x[1])) < 1e-10 * 10
    assert abs(np.asarray(r.i("L"))[-1] - float(x[0])) * np.sqrt(L / C) < 1e-9
    tiers = [m["tier"] for m in r.exact["masks"]]
    assert tiers == [2]
    assert "kappa2" in r.exact["masks"][0]["reason"]


def test_series_diode_from_rest_conducts():
    # A 10 V step through L and a series diode into C || R. At t = 0
    # v_D = 0 < V_on; the slow-manifold projection then puts the full
    # 10 V across the diode, whose ON predicate starts past its
    # crossing and never fired: the untouched kernel kept the diode OFF
    # for the whole run and read v_C = 3.3e-6 V. The settle now runs
    # again on the projected state.
    for vth in (1e-3, 0.0):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 10.0)
        b.add_inductor("L", "in", "a", 10e-6)
        b.add_diode("D", "a", "o", GON, GOFF, vth)
        b.add_capacitor("C", "o", "gnd", 100e-9)
        b.add_resistor("R", "o", "gnd", 1e3)
        r = _run(b, t_end=40e-6)
        assert abs(np.asarray(r.v("o"))[-1] - 13.7267583) < 1e-6
        # The diode stops at the current zero, near pi*sqrt(LC).
        off = r.event_times[r.event_types == 2]
        assert len(off) == 1 and abs(off[0] - 3.1617924e-6) < 1e-12


def test_double_crossing_inside_one_step_is_seen():
    # LC tank, v_t = 10 sin(w t), w = 1e6 rad/s, and a diode into a
    # 9.5 V rail through 1 kOhm: v_D > 0 only for w t in (1.253, 1.889),
    # 0.64 us inside one 10 us step whose ends both read v_D < 0. The
    # scan step pi/(4 w) = 0.79 us finds the window.
    # Integration batch 4: with V_th = 0 the run was silently wrong after
    # the turn-on (D ON at -18 mA at 4.25 us; pwl at dt = 0.1 ns has D OFF
    # there) and the diode-law guard now names it; with V_th = 1e-3 both
    # windows are right (1.2536-1.8877 us and 7.537-8.170 us, pwl ends 8.170).
    def build(v_th):
        b = p.CircuitBuilder()
        b.add_capacitor("C", "t", "gnd", 100e-9)
        b.add_inductor("L", "t", "gnd", 10e-6, i0=-1.0)
        b.add_diode("D", "t", "k", GON, GOFF, v_th)
        b.add_resistor("R", "k", "r", 1e3)
        b.add_voltage_source("Vr", "r", "gnd", 9.5)
        return b
    # Batch 6a: the kernel guard does NOT catch this one (it repairs one
    # event and returns v(k) = -5.18 V at 10 us; trbdf2 9.50 V); the
    # stored-sample guard does, which is why both are kept.
    from pulsim._dsed_dispatch import DiodeModeInconsistentError
    with pytest.raises(DiodeModeInconsistentError):
        _run(build(0.0), t_end=10e-6)
    r = _run(build(1e-3), t_end=10e-6)
    on = r.event_times[r.event_types == 1]
    assert len(on) == 2
    assert abs(on[0] - np.arcsin(0.9501) / 1e6) < 1e-12
    assert list(r.event_types) == [1, 2, 1, 2]
    assert abs(r.event_times[1] - 1.88772e-6) < 1e-11


def test_ten_megahertz_duty():
    # Period 1e-7 s: the absolute 1e-10 tolerance was 1e-3 of it.
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF, 1e-6)
    b.add_inductor("L", "sw", "vout", 1e-6)
    b.add_capacitor("C", "vout", "gnd", 1e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_pwm_switch_fn(1e7, 0.3, b.switch_index_of("S"),
                              b.graph.num_switches)
    r = _run(b, t_end=2e-6, switch_fn=fn, dt=1e-8)
    gate = np.asarray(r.switch_states)[:, 0]
    t = np.asarray(r.times)
    # Gate edges land exactly; the realised on-time per period:
    on = [t[i - 1] for i in range(1, len(t)) if gate[i - 1] and not gate[i]]
    up = [t[i - 1] for i in range(1, len(t)) if not gate[i - 1] and gate[i]]
    duty = (on[-1] - up[-1]) / 1e-7
    assert abs(duty - 0.3) < 1e-9


def _sync_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_switch("Q2", "sw", "gnd", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_dead_time_pwm_pair_fn(
        100e3, 0.5, b.switch_index_of("Q1"), b.switch_index_of("Q2"),
        b.graph.num_switches, 0.0)
    return b, fn


def test_counters_on_the_synchronous_buck():
    # perf row 22: 2000 periods, two exact steps per period, one
    # eigendecomposition per visited mask, no Pade.
    b, fn = _sync_buck()
    r = _run(b, t_end=20e-3, switch_fn=fn)
    assert abs(r.n_exact_steps - 4000) <= 1
    assert r.exact["n_eig"] == len(r.exact["masks"]) == 2
    assert r.exact["n_pade"] == 0
    assert all(m["tier"] == 1 for m in r.exact["masks"])
    assert abs(np.asarray(r.v("vout"))[-1] - 11.998771666) < 1e-8


def test_predicate_evaluations_per_event_are_bounded():
    b, fn = _buck(50.0, 1e-6, D=0.3)
    r = _run(b, t_end=200 * T, switch_fn=fn)
    located = np.count_nonzero(r.event_types == 2)
    assert located >= 170
    # Two evaluations per step (the armed predicate at both ends) plus
    # the root search; measured 8.3 per located event, gate 40 (3.5).
    assert r.exact["n_predicate_evals"] <= 2 * r.n_exact_steps + 40 * located


# ---- D5: builder initial conditions (refused on dsed before) ----------


def test_inductor_initial_current_decays_exactly():
    b = p.CircuitBuilder()
    b.add_inductor("L", "a", "gnd", 1e-3, i0=100.0)
    b.add_resistor("R", "a", "gnd", 1.0)
    r = _run(b, t_end=5e-3, dt=1e-4)
    t = np.asarray(r.times)
    np.testing.assert_allclose(np.asarray(r.i("L")), 100 * np.exp(-t / 1e-3),
                               rtol=0, atol=1e-10 * 100)


def test_floating_capacitor_initial_voltage():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R1", "in", "a", 1e3)
    b.add_capacitor("C", "a", "b", 1e-6, c0=5.0)
    b.add_resistor("R2", "b", "gnd", 1e3)
    r = _run(b, t_end=1e-3)
    v = np.asarray(r.v("a")) - np.asarray(r.v("b"))
    assert abs(v[0] - 5.0) < 1e-12
    tr = p.simulate(b, t_end=1e-3, engine="trbdf2", rtol=1e-9, atol=1e-12)
    vt = np.asarray(tr.v("a")) - np.asarray(tr.v("b"))
    assert abs(v[-1] - vt[-1]) < 1e-6


def test_inconsistent_parallel_capacitors_are_refused_by_name():
    # Parallel capacitors (1 uF at 10 V, 3 uF at 2 V) are a capacitor
    # loop; the LTI extractor refuses every capacitor loop, so the
    # charge-conserving start (4.0 V) is not reachable through
    # simulate() yet. It must be a named refusal, never a wrong start.
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1e-6, c0=10.0)
    b.add_capacitor("C2", "a", "gnd", 3e-6)
    b.set_initial("C2", 2.0)
    b.add_resistor("R", "a", "gnd", 1e3)
    with pytest.raises(p.UnsupportedError, match="capacitors form a cycle"):
        _run(b, t_end=1e-4)


def test_consistent_initial_conditions_report_no_adjustment():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "s", "gnd", 5.0)
    b.add_resistor("R", "s", "a", 1e3)
    b.add_capacitor("C", "a", "gnd", 1e-6, c0=2.0)
    r = _run(b, t_end=1e-4)
    assert r.initial_condition_adjustment == 0.0
    # RC charge from 2 V towards 5 V, tau = 1 ms
    t = np.asarray(r.times)
    ref = 5.0 - 3.0 * np.exp(-t / 1e-3)
    assert np.max(np.abs(np.asarray(r.v("a")) - ref)) < 1e-10


# Integration batch 4: the zero-band settle can keep a diode in a mode its
# law forbids; check_diode_modes turns that silent answer into a named error.
def _boost(v_th, R):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("S", "sw", "gnd", 1e3, 1e-9)
    b.add_diode("D1", "sw", "vout", 1e3, 1e-9, v_th)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", R)
    fn = p.make_pwm_switch_fn(1e5, 0.5, b.switch_index_of("S"), b.graph.num_switches)
    return b, fn


@pytest.mark.parametrize("v_th,R,v_end", [(0.0, 50.0, 24.5295), (1e-3, 50.0, 24.5296),
                                          (0.0, 500.0, 45.4835)])
def test_boost_with_a_wrong_settled_mode_is_repaired_or_named(v_th, R, v_end):
    # The settle alone holds D1 ON at -1.07 A (R = 50, V_th = 0; 0.76 V low at
    # R = 500). Batch 4: the stored-sample guard named it. Since the
    # complementarity guard (batch 6a) the kernel repairs those events and
    # the run lands on trbdf2 rtol 1e-10 (end v(vout) to 1e-4, measured
    # <= 1e-5); with the kernel guard off the stored-sample guard still
    # names it. Re-pin: raise -> repaired run.
    import pulsim._dsed_dispatch as DD
    b, fn = _boost(v_th, R)
    r = p.simulate(b, 3e-3, engine="dsed", switch_fn=fn)
    assert r.diagnostics["mode_guard"]["repairs"] > 0
    assert np.asarray(r.v("vout"))[-1] == pytest.approx(v_end, abs=1e-4)
    b, fn = _boost(v_th, R)
    DD.MODE_GUARD = False
    try:
        with pytest.raises(DD.DiodeModeInconsistentError, match="D1"):
            p.simulate(b, 3e-3, engine="dsed", switch_fn=fn)
    finally:
        DD.MODE_GUARD = True


def test_boost_with_a_threshold_band_runs_and_agrees():
    b, fn = _boost(1e-2, 50.0)
    r = p.simulate(b, 3e-3, engine="dsed", switch_fn=fn)
    b, fn = _boost(1e-2, 50.0)
    ref = p.simulate(b, 3e-3, engine="trbdf2", switch_fn=fn, rtol=1e-10, atol=1e-13)
    # on dsed's own samples (its sparse output is not interpolated across edges)
    v = np.interp(np.asarray(r.times), np.asarray(ref.times), np.asarray(ref.v("vout")))
    assert np.abs(v - np.asarray(r.v("vout"))).max() < 1e-3


def test_half_wave_rl_rectifier_with_a_zero_band_diode_is_repaired_or_named():
    # Re-pin (batch 6a): the kernel guard repairs the one wrong event; the
    # stored-sample guard still names it with the kernel guard off.
    import pulsim._dsed_dispatch as DD

    def build():
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
        b.add_diode("D", "in", "a", 1e3, 1e-9)
        b.add_resistor("R", "a", "b", 10.0)
        b.add_inductor("L", "b", "gnd", 1e-3)
        return b
    r = p.simulate(build(), 3e-3, engine="dsed")
    assert r.diagnostics["mode_guard"]["repairs"] == 1
    DD.MODE_GUARD = False
    try:
        with pytest.raises(DD.DiodeModeInconsistentError, match="'D' is ON while carrying"):
            p.simulate(build(), 3e-3, engine="dsed")
    finally:
        DD.MODE_GUARD = True
