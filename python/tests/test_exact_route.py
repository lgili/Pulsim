"""add-exact-propagator-strategy group 13, report-only slice (D11): the segment-eligibility
classifier (`pulsim.exact_route`, `result.exact['segments']`) and `engine='exact'` as the
canonical name of the event-driven exact engine, with 'dsed' its alias."""

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


def test_coupled_inductors_are_not_exact_extraction():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 100.0, 50.0)
    b.add_transformer("T", "a", "gnd", "s", "r", 10e-3, 2.5e-3, 0.99)
    b.add_resistor("Rr", "r", "gnd", 1e6)
    b.add_diode("D1", "s", "o", GON, GOFF)
    b.add_capacitor("C", "o", "r", 100e-6)
    b.add_resistor("R", "o", "r", 100.0)
    rep = p.exact_route(b, 0.02)
    assert (rep["exact"], rep["reason"]) == (False, "extraction")


def test_inductor_cut_set_is_not_exact_extraction():
    # the three-phase overlap family: a DC current source in series with the line inductors
    # (an inductor cut-set) -- dsed refuses its extraction today, so it is not exact-eligible
    rep = p.exact_route(B.FAMILIES["bridge3_overlap"].builder(0.0), 0.02)
    assert (rep["exact"], rep["reason"]) == (False, "extraction")


def test_floating_capacitor_loop_is_not_exact_extraction():
    # C1 || C2 between two non-ground nodes: a capacitor loop the extractor cannot reduce
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_resistor("R1", "a", "x", 1e3)
    b.add_capacitor("C1", "x", "y", 1e-6)
    b.add_capacitor("C2", "x", "y", 2e-6)
    b.add_resistor("R2", "y", "gnd", 1e3)
    rep = p.exact_route(b, 1e-3)
    assert (rep["exact"], rep["reason"]) == (False, "extraction")


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
def test_a_stiff_near_defective_mode_is_refused_as_cost(mode):
    """Ten three-phase bridges paralleled on one bus: the mode at rest (all 30 diodes ON) has no
    exact step (kappa2(V) 2.3e6, ||A*h|| 1.5e6) and the numeric fallback made no progress past
    1 us ("no meaningful progress"). engine='exact' now names it at its first segment
    (add-complementarity-mode-selection DECISIONS #25)."""
    kw = {} if mode is None else {"mode_selection": mode}
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(_paralleled_bridges(10), t_end=0.02, engine="exact", **kw)
    assert (ei.value.engine, ei.value.feature) == ("dsed", "cost")
    assert "near-defective" in str(ei.value)


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
