"""add-exact-propagator-strategy group 13, report-only slice (D11): the segment-eligibility
classifier (`pulsim.exact_route`, `result.exact['segments']`) and `engine='exact'` as the
canonical name of the event-driven exact engine, with 'dsed' its alias."""

import functools
import warnings

import numpy as np
import pytest

import pulsim as p
from tests.oracles import lcp_pulsim_bridge as B

GON, GOFF, T = 1e3, 1e-9, 1e-5


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF, 0.0)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("R", "vout", "gnd", 50.0)
    fn = p.make_pwm_switch_fn(1 / T, 0.3, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    return b, fn, 40 * T


def _boost():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("S", "sw", "gnd", GON, GOFF)
    b.add_diode("D1", "sw", "out", GON, GOFF, 0.0)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 50.0)
    fn = p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    return b, fn, 40 * T


def _family(name, t_end):
    def make():
        return B.FAMILIES[name].builder(0.0), None, t_end
    return make


def _three_phase_example():
    lc = p.load_yaml_file("examples/three_phase_diode_rectifier.yaml")
    return getattr(lc, "builder", lc), None, 0.02


EXACT = {
    "buck": _buck,
    "boost": _boost,
    "single-phase bridge": _family("bridge1_vf0", 0.04),
    "three-phase bridge, C filter": _family("bridge3_cfilter", 0.04),
    "examples/three_phase_diode_rectifier": _three_phase_example,
}


@pytest.mark.parametrize("case", list(EXACT))
def test_converters_and_rectifiers_classify_exact(case):
    b, fn, t_end = EXACT[case]()
    rep = p.exact_route(b, t_end, switch_fn=fn)
    assert rep["exact"] and rep["reason"] is None
    segs = rep["segments"]
    assert segs and all(s["exact"] and s["reason"] is None and s["tier"] in (1, 2) for s in segs)
    assert sum(s["n_segments"] for s in segs) >= 2          # the run commutated


def test_shockley_rectifier_is_not_exact_device():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 50.0)
    b.add_shockley_diode("Dsh", "a", "o")
    b.add_capacitor("C", "o", "gnd", 100e-6)
    b.add_resistor("R", "o", "gnd", 100.0)
    rep = p.exact_route(b, 0.02)
    assert (rep["exact"], rep["reason"]) == (False, "device")
    (row,) = rep["segments"]
    assert (row["device"], row["kind"]) == ("Dsh", "ShockleyDiode")


def test_coupled_inductors_route_exact():
    # the extraction takes the inductance matrix (add-exact-propagator-strategy, coupled inductors)
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 100.0, 50.0)
    b.add_transformer("T", "a", "gnd", "s", "r", 10e-3, 2.5e-3, 0.99)
    b.add_resistor("Rr", "r", "gnd", 1e6)
    b.add_diode("D1", "s", "o", GON, GOFF)
    b.add_capacitor("C", "o", "r", 100e-6)
    b.add_resistor("R", "o", "r", 100.0)
    rep = p.exact_route(b, 0.02)
    assert (rep["exact"], rep["reason"]) == (True, None)


def test_inductor_cut_set_runs_and_agrees_with_pwl():
    """The three-phase overlap family: a DC current source in series with the line inductors.
    The settle picked a mode its law forbids at 126 us (on mode_selection='lcp' too) and the run
    was refused by name; since exact engine session 11 decisions at a located diode event are read on the candidate mode's slow manifold (its parasitic transient settled first)
    (and a dependent-state jump at such an event is not an impulse), and the run agrees
    with pwl+split at dt 1e-7 over 20 ms (pwl itself moves 6.3e-3 V from dt 2e-7 to 1e-7)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(B.FAMILIES["bridge3_overlap"].builder(0.0), t_end=0.02, engine="exact")
        q = p.simulate(B.FAMILIES["bridge3_overlap"].builder(0.0), t_end=0.02, engine="pwl", dt=1e-7,
                       discontinuity_restart=True, enable_substep_state_correction=True)
    tg = np.linspace(0.0, 0.02, 4001)

    def bus(x):
        return np.interp(tg, x.times, np.asarray(x.v("p")) - np.asarray(x.v("n")))
    assert np.abs(bus(r) - bus(q)).max() <= 3e-2
    for ph in "abc":
        ir = np.interp(tg, r.times, np.asarray(r.i("L" + ph)))
        iq = np.interp(tg, q.times, np.asarray(q.i("L" + ph)))
        assert np.abs(ir - iq).max() <= 2e-5


def test_floating_capacitor_loop_routes_exact():
    # C1 || C2 between two non-ground nodes: a capacitor loop, reduced by the general path
    # (one capacitor a link); the run follows 10 (1 - exp(-t / 6 ms)) to 3e-14
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_resistor("R1", "a", "x", 1e3)
    b.add_capacitor("C1", "x", "y", 1e-6)
    b.add_capacitor("C2", "x", "y", 2e-6)
    b.add_resistor("R2", "y", "gnd", 1e3)
    rep = p.exact_route(b, 1e-3)
    assert (rep["exact"], rep["reason"]) == (True, None)


def test_pwm_voltage_source_routes_exact():
    """Task 16.1: a PWM source is level + ramp between corners, each corner a time event, so its
    masks are exact (it was refused as 'source' before) and the run takes only exact steps."""
    b = p.CircuitBuilder()
    b.add_pwm_voltage_source("Vg", "in", "gnd", 5.0, 0.0, 1e4, 0.5)
    b.add_resistor("R", "in", "a", 10.0)
    b.add_diode("D", "a", "o", GON, GOFF)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    rep = p.exact_route(b, 3e-4)
    assert (rep["exact"], rep["reason"]) == (True, None)
    assert all(row["exact"] for row in rep["segments"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=3e-4, engine="exact")
    assert r.n_exact_steps == len(r.times) - 1


def test_segments_ride_on_the_result():
    b, fn, t_end = _buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=t_end, engine="exact", switch_fn=fn)
    segs = r.exact["segments"]
    assert {len(s["mask"]) for s in segs} == {b.graph.num_switches}
    assert len(segs) == len(r.exact["masks"])


@pytest.mark.parametrize("make", [_buck, _family("bridge3_cfilter", 0.02)])
def test_exact_and_dsed_are_one_engine(make):
    runs = []
    for eng in ("exact", "dsed"):
        b, fn, t_end = make()
        kw = {} if fn is None else {"switch_fn": fn}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runs.append(p.simulate(b, t_end=t_end, engine=eng, **kw))
    a, d = runs
    assert np.array_equal(np.asarray(a.times), np.asarray(d.times))
    assert np.array_equal(np.asarray(a.states), np.asarray(d.states))
    assert a.engine_used == d.engine_used == "dsed"     # one engine; its name in results (DECISIONS)


def test_exact_refusals_name_the_engine_column():
    # a refusal raised under engine='exact' carries the same fields as under 'dsed'
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    for eng in ("exact", "dsed"):
        with pytest.raises(p.UnsupportedError) as ei:
            p.simulate(b, t_end=1e-3, engine=eng, store_every=4)
        assert (ei.value.engine, ei.value.feature) == ("dsed", "store_every")


def test_capability_check_takes_the_exact_name():
    # the kernel's per-engine device check accepts 'exact' as the dsed column
    from pulsim import _pulsim as k
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 50.0)
    b.add_shockley_diode("Dsh", "a", "o")
    b.add_resistor("R", "o", "gnd", 100.0)
    for eng in ("exact", "dsed"):
        with pytest.raises(p.UnsupportedError) as ei:
            k._require_supported(b.graph, b.pool, eng)
        assert (ei.value.engine, ei.value.device) == ("dsed", "Dsh")


def test_a_run_without_a_tier_record_is_not_reported_exact():
    # integrator='bdf2' keeps no exact-tier record: the route is unknown, not "exact"
    with pytest.raises(ValueError, match="no exact-tier record"):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "a", "gnd", 1.0)
        b.add_resistor("R", "a", "o", 1e3)
        b.add_capacitor("C", "o", "gnd", 1e-6)
        p.exact_route(b, 1e-3, integrator="bdf2")


# ---------------------------------------------------------------------------------------------
# Task 13.2: engine='exact' refuses a segment it cannot step exactly, unless allowed
# ---------------------------------------------------------------------------------------------
def _rc_with_b_extra():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    b.add_resistor("R", "a", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    zero = [0.0] * b.pool.state_size(b.graph)
    return b, (lambda _t: zero)


@pytest.mark.parametrize("engine", ["exact", "dsed"])
def test_a_b_extra_fn_segment_is_refused_by_rule_unless_allowed(engine):
    b, fn = _rc_with_b_extra()
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, t_end=1e-3, engine=engine, b_extra_fn=fn)
    assert (ei.value.engine, ei.value.feature, ei.value.device) == ("dsed", "source", "")
    assert "allow_implicit_segments=True" in str(ei.value)
    b, fn = _rc_with_b_extra()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=1e-3, engine=engine, b_extra_fn=fn, allow_implicit_segments=True)
    assert r.n_exact_steps == 0 and r.n_rk45_steps + r.n_bdf2_steps > 0
    assert np.asarray(r.v("o"))[-1] == pytest.approx(5.0 * (1 - np.exp(-1.0)), rel=1e-5)


def _paralleled_bridges(n):
    b = p.CircuitBuilder()
    for k in range(n):
        for ph, ang in (("a", 0.0), ("b", -120.0), ("c", 120.0)):
            b.add_sine_voltage_source(f"V{k}{ph}", f"s{k}{ph}", "gnd", 0.0, 100.0, 50.0,
                                      np.deg2rad(ang + 6.0 * k))
            b.add_inductor(f"L{k}{ph}", f"s{k}{ph}", f"x{k}{ph}", 0.2e-3)
            b.add_diode(f"Dt{k}{ph}", f"x{k}{ph}", "p", GON, GOFF, 0.0)
            b.add_diode(f"Db{k}{ph}", "n", f"x{k}{ph}", GON, GOFF, 0.0)
    b.add_capacitor("C", "p", "n", 2e-3)
    b.add_resistor("R", "p", "n", 5.0)
    b.add_resistor("Rg", "n", "gnd", 1e6)
    return b


@pytest.mark.parametrize("mode", [None, "lcp"])
def test_ten_paralleled_bridges_agree_with_pwl(mode):
    """Ten three-phase bridges paralleled on one bus, the bus held to ground by 1 Mohm. Slice 7
    refused this run (an ON diode at -0.30 A, 0.87 ms) and blamed the mode selection; the cause
    was the reduction: G_aa's condition (1 kS diodes against the 1 uS tie) cost the plain Schur
    complement 0.55 of A's -100 /s bus entry, so the trajectory drifted (2e-4 V at 0.1 ms) and
    the predicate of Dt0a read 3e-5 V of noise, keeping it ON 4.6 us past its current zero.
    The double-double refined reduction (exact engine 8) runs it on either selection, within
    the 5-bank test's 1e-3 V of pwl+split.
    Strict since exact engine session 16: 11 of 11 runs perturbed at 1e-15 run on both selections
    (the law check reads, at a located diode event, the settled state the decision read; the
    un-relaxed g_off spike of 1e-13 s refused 1 in 11 before)."""
    kw = {} if mode is None else {"mode_selection": mode}
    t_end = 0.02
    tg = np.linspace(0.0, t_end, 2001)

    def bus(r):
        return np.interp(tg, r.times, np.asarray(r.v("p")) - np.asarray(r.v("n")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(_paralleled_bridges(10), t_end=t_end, engine="exact", **kw)
        ref = _ten_bridges_pwl()
    assert np.abs(bus(r) - np.interp(tg, *ref)).max() <= 1e-3


@functools.cache
def _ten_bridges_pwl():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = p.simulate(_paralleled_bridges(10), t_end=0.02, engine="pwl", dt=1e-7,
                       discontinuity_restart=True, enable_substep_state_correction=True)
    return np.asarray(q.times), np.asarray(q.v("p")) - np.asarray(q.v("n"))


def test_ten_paralleled_bridges_first_mode_matches_a_50_digit_exponential():
    """The first mode (every bank: Dt_a, Db_b, Dt_c ON) held from rest to 0.1 ms, against the
    exponential (mpmath, 50 digits) of the augmented system built from the circuit's KCL written
    out by hand (states i_L and the bus; n = R_g·sum(i)), not from the reduction under test
    (openspec/changes/add-exact-propagator-strategy/probes/ten_banks_50_digits.py 10): bus 11.725547151164887 V,
    i(L0a) 0.12972284951463489 A. The plain reduction was 2.1e-4 V / 5.5e-6 A off; the ON
    diode Dt0a must read v = i/g_on from the recovery map (was 3e-5 V off)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(_paralleled_bridges(10), t_end=1e-4, engine="exact")
    bus = float(np.asarray(r.v("p"))[-1] - np.asarray(r.v("n"))[-1])
    il = float(np.asarray(r.i("L0a"))[-1])
    assert r.times[-1] == pytest.approx(1e-4, abs=1e-15)
    assert bus == pytest.approx(11.725547151164887, abs=1e-7)
    assert il == pytest.approx(0.12972284951463489, abs=1e-8)
    vd = float(np.asarray(r.v("x0a"))[-1] - np.asarray(r.v("p"))[-1])
    assert vd == pytest.approx(il / GON, abs=1e-8)


def _tied_bridge(vf, k):
    rng = np.random.default_rng(k)

    def f(v):
        return v * (1 + (rng.uniform(-1, 1) * 1e-15 if k else 0))
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, f(325.0), 50.0)
    b.add_resistor("Rs", "s", "a", f(0.5))
    b.add_inductor("Ls", "a", "ac1", f(1e-3))
    for nm, an, ca in (("D1", "ac1", "dp"), ("D2", "gnd", "dp"), ("D3", "dn", "ac1"), ("D4", "dn", "gnd")):
        b.add_diode(nm, an, ca, GON, GOFF, 0.0, vf)
    b.add_capacitor("C", "dp", "dn", f(470e-6))
    b.add_resistor("RL", "dp", "dn", f(100.0))
    b.add_resistor("Rtie", "dn", "gnd", f(1e6))
    return b


@pytest.mark.parametrize("vf", [0.0, 0.7])
def test_a_diode_at_its_own_current_zero_is_decided_on_the_slow_manifold(vf):
    """Exact engine session 11: a diode leaving at its located current zero is decided on the OFF
    mode's slow manifold, after its parasitic transient (L*g_off) is settled, not by the OFF
    mode's value at the root, which reads the root's leftover current through g_off as kV. With the refined reduction the tied bridge
    with a 0.7 V drop was refused on every run (D1 OFF forward-biased by 1.1 V at 23.6 ms).
    Five runs perturbed at 1e-15: every one runs, with one commutation sequence."""
    runs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(5):
            r = p.simulate(_tied_bridge(vf, k), t_end=0.04, engine="exact")
            runs.append((list(r.event_types), np.asarray(r.event_times)))
    types0, times0 = runs[0]
    for types, times in runs[1:]:
        assert types == types0
        assert np.abs(times - times0).max() <= 1e-9


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "auto"])
def test_allow_implicit_segments_refused_off_the_exact_engine(engine):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    b.add_resistor("R", "a", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    kw = {"dt": 1e-5} if engine == "pwl" else {}
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, t_end=1e-4, engine=engine, allow_implicit_segments=True, **kw)
    assert ei.value.feature == "allow_implicit_segments"


def test_auto_would_route_is_reported_and_read_by_nothing():
    """Task 13.3, report only: result.exact['auto_would_route'] is True when every segment is
    exact (the buck), False with a numeric segment (a b_extra_fn, allowed); 'exact' and 'dsed'
    agree, and engine='auto' still routes the buck to the fixed / TR-BDF2 engines."""
    b, fn, t_end = _buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=t_end, engine="exact", switch_fn=fn)
        b2, fn2, _ = _buck()
        r2 = p.simulate(b2, t_end=t_end, engine="dsed", switch_fn=fn2)
        b3, fn3, _ = _buck()
        ra = p.simulate(b3, t_end=t_end, switch_fn=fn3)
    assert r.exact["auto_would_route"] is True and r2.exact["auto_would_route"] is True
    assert ra.engine_used != "dsed"
    b, fx = _rc_with_b_extra()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=1e-3, engine="exact", b_extra_fn=fx, allow_implicit_segments=True)
    assert r.exact["auto_would_route"] is False
