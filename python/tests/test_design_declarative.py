"""Declarative KPIs and constraints, and `evaluate_many(backend=
'process')` (add-design-optimization-workflow, slice 5 item 2). The
declarative specs hash by value, pickle, and are computed by the
evaluator on the route's own footing (the kernel's closed forms on the
pss route; the samples with right limits on a sampled route); the
process backend gives the thread records; an evaluator that cannot
reach a worker is refused by name before one starts. Values and
counters, never wall time."""

import concurrent.futures
import functools
import pickle

import numpy as np
import pytest

import pulsim as p
from pulsim.design import Constraint, Evaluator, Kpi, SteadySpec
from pulsim.design import _mean, _node_col, _p2p, _with_right_limits

T = 1e-5


def buck(d):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "x", d["L"])
    b.add_resistor("RLs", "x", "vout", 0.01)
    b.add_capacitor("C", "vout", "gnd", d["C"])
    b.add_resistor("RL", "vout", "gnd", d["R"])
    sf = p.make_pwm_switch_fn(1 / T, 0.4, b.switch_index_of("S"),
                              b.graph.num_switches)
    return b, sf


def base_circuit():
    return buck(dict(L=100e-6, C=47e-6, R=1.0))[0].build()


def schedule(d):
    # S is bit 0 of two (the diode D1 is bit 1): a width-1 mask is refused
    return p.make_pwm_switch_fn(1 / T, 0.4, 0, 2)


def ripple_x(per):
    """The callable twin of Kpi.ripple('x'), written by hand."""
    t, v = _with_right_limits(per.result, _node_col(per.builder, "x"))
    return _p2p(t, v, per.t_from, per.t_to)


def mean_x(per):
    t, v = _with_right_limits(per.result, _node_col(per.builder, "x"))
    return _mean(t, v, per.t_from, per.t_to)


DESIGNS = [dict(L=L, C=C, R=R) for L in (22e-6, 100e-6)
           for C in (22e-6, 47e-6) for R in (1.0, 50.0)]
SPECS = [Kpi.efficiency(), Kpi.ripple("x"), Kpi.ripple("vout"),
         Kpi.mean("x"), Kpi.i_peak("L"), Kpi.i_rms("L"), Kpi.ripple_i("L")]
CONSTRAINTS = [Constraint.max("ripple_v", 0.05),
               Constraint.min("efficiency", 0.5)]


def _ev(kpis=(), constraints=None, **kw):
    kw.setdefault("steady", SteadySpec("pss"))
    return Evaluator(buck, period=T, loads=("RL",), output="vout",
                     kpis=kpis, constraints=constraints, **kw)


def test_a_spec_that_names_a_default_is_that_default():
    for d in DESIGNS[:4]:
        r = _ev(SPECS, CONSTRAINTS).evaluate(d)
        assert r.status in ("ok", "infeasible"), r.reason
        k = r.kpis
        assert k["ripple_v.vout"] == k["ripple_v"]
        for name in ("efficiency", "i_peak.L", "i_rms.L", "ripple_i.L"):
            assert name in k
        assert r.violations == _ev(SPECS, {"ripple_v": (None, 0.05),
                                           "efficiency": (0.5, None)}
                                   ).evaluate(d).violations


def test_on_a_sampled_route_the_specs_are_the_callables_bit_for_bit():
    kw = dict(engine="pwl", options={"dt": T / 200},
              steady=SteadySpec("pss"))
    spec = _ev([Kpi.ripple("x"), Kpi.mean("x")], **kw)
    lam = _ev([Kpi("rx", ripple_x), Kpi("mx", mean_x)], **kw)
    for d in DESIGNS[:4]:
        a, b = spec.evaluate(d), lam.evaluate(d)
        assert a.kpis["ripple_v.x"] == b.kpis["rx"]
        assert a.kpis["v_avg.x"] == b.kpis["mx"]


def test_on_the_kernel_route_the_specs_are_the_closed_forms():
    # the callable reads 2000 samples of the orbit, the spec the
    # kernel's closed-form extrema and integral: they agree to the
    # trapezoid's O(h^2); the mean of x equals v_out + i_L R_Ls exactly
    spec = _ev([Kpi.ripple("x"), Kpi.mean("x")])
    lam = _ev([Kpi("rx", ripple_x), Kpi("mx", mean_x)])
    for d in DESIGNS:
        a, b = spec.evaluate(d), lam.evaluate(d)
        assert a.counters["kpi_path"] == "kernel"
        assert a.kpis["ripple_v.x"] == pytest.approx(b.kpis["rx"], rel=1e-5)
        assert a.kpis["v_avg.x"] == pytest.approx(b.kpis["mx"], rel=1e-9)
        i_avg = a.kpis["v_out"] / d["R"]      # the mean capacitor current is 0
        assert a.kpis["v_avg.x"] == pytest.approx(
            a.kpis["v_out"] + 0.01 * i_avg, rel=1e-9)


def test_specs_hash_by_value():
    a = _ev([Kpi.ripple("x")], [Constraint.max("ripple_v", 0.05)])
    b = _ev([Kpi.ripple("x")], {"ripple_v": (None, 0.05)})
    c = _ev([Kpi.ripple("vout")], [Constraint.max("ripple_v", 0.05)])
    assert a.design_id(DESIGNS[0]) == b.design_id(DESIGNS[0])
    assert a.design_id(DESIGNS[0]) != c.design_id(DESIGNS[0])
    assert pickle.loads(pickle.dumps(Kpi.ripple("x"))) == Kpi.ripple("x")
    assert hash(Constraint.max("k", 1)) == hash(Constraint("k", None, 1.0))


@pytest.mark.parametrize("form", ["factory", "circuit"])
def test_the_process_backend_gives_the_thread_records(form):
    if form == "factory":
        ev = _ev([Kpi.ripple("x"), Kpi.i_peak("L")], CONSTRAINTS)
    else:
        ev = Evaluator(circuit=base_circuit, fields={"L": "L.L", "C": "C.C",
                                                     "R": "RL.R"},
                       schedule=schedule, period=T, loads=("RL",),
                       output="vout", kpis=[Kpi.ripple("x")],
                       constraints=CONSTRAINTS, steady=SteadySpec("pss"))
    ds = DESIGNS + DESIGNS[:3]
    thr = ev.evaluate_many(ds, workers=1, backend="thread")
    prc = ev.evaluate_many(ds, workers=2, backend="process")
    # ids are made in the workers: the spec hashes the same there
    assert prc == thr
    assert [r.design_id for r in prc] == [r.design_id for r in thr]


def test_a_circuit_factory_must_rebuild_the_same_circuit():
    ev = Evaluator(circuit=base_circuit, fields={"L": "L.L"},
                   schedule=schedule, period=T, steady=SteadySpec("pss"))
    clone = pickle.loads(pickle.dumps(ev))
    assert clone.evaluate(DESIGNS[0] | {"L": 50e-6}) == \
        ev.evaluate(DESIGNS[0] | {"L": 50e-6})
    st = ev.__getstate__()
    st["_circuit_hash"] = "0" * 16
    bad = Evaluator.__new__(Evaluator)
    with pytest.raises(RuntimeError, match="deterministic"):
        bad.__setstate__(st)


def test_a_circuit_reaches_process_workers():
    """A Circuit pickles as its YAML text (add-immutable-circuit-ir 4.3),
    so the circuit form runs on the process backend as on threads."""
    ev = Evaluator(circuit=base_circuit(), fields={"L": "L.L"},
                   schedule=schedule, period=T, steady=SteadySpec("pss"))
    ds = [DESIGNS[0] | {"L": L} for L in (50e-6, 80e-6)]
    assert ev.evaluate_many(ds, workers=2, backend="process") == \
        ev.evaluate_many(ds, workers=1)


def test_unpicklable_parts_are_refused_by_name_before_a_worker_starts(
        monkeypatch):
    started = []
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor",
                        lambda *a, **k: started.append(1))
    cases = [
        (_ev([Kpi("vx", lambda per: 0.0)]), "kpis['vx']"),
        (Evaluator(buck, period=lambda d: T, loads=("RL",)), "period"),
        # a Circuit pickles; the switch_fn object beside it does not
        (Evaluator(circuit=base_circuit(), fields={"L": "L.L"},
                   switch_fn=schedule({}), period=T), "switch_fn"),
    ]
    for ev, name in cases:
        with pytest.raises(p.UnsupportedError) as ei:
            ev.evaluate_many(DESIGNS, workers=2, backend="process")
        assert ei.value.feature == "backend"
        assert ei.value.device == name
    ev = _ev()
    with pytest.raises(p.UnsupportedError) as ei:
        ev.evaluate_many(DESIGNS + [dict(DESIGNS[0], tag=lambda: 0)],
                         workers=2, backend="process")
    assert ei.value.device == f"designs[{len(DESIGNS)}]"
    assert not started
    # the thread backend takes the same callables
    r = _ev([Kpi("vx", lambda per: 1.5)]).evaluate_many(
        DESIGNS[:4], workers=2, backend="thread")
    assert all(x.kpis["vx"] == 1.5 for x in r)


def test_constraint_constructors_and_misuse():
    assert Constraint.between("k", 1, 2) == Constraint("k", 1.0, 2.0)
    with pytest.raises(TypeError, match="Constraint objects"):
        _ev([], [("ripple_v", (None, 1.0))])
    with pytest.raises(ValueError, match="bounded twice"):
        _ev([], [Constraint.max("ripple_v", 1), Constraint.min("ripple_v", 0)])
    with pytest.raises(ValueError, match="given twice"):
        _ev([Kpi.ripple("x"), Kpi.ripple("x")])
    with pytest.raises(KeyError, match="no such node"):
        _ev([Kpi.ripple("nope")]).evaluate(DESIGNS[0])
    with pytest.raises(KeyError, match="loads="):
        Evaluator(buck, period=T, kpis=[Kpi.efficiency()],
                  steady=SteadySpec("pss")).evaluate(DESIGNS[0])
    with pytest.raises(TypeError, match="computed by the Evaluator"):
        Kpi.ripple("x").fn(None)
    assert functools.reduce(lambda a, k: a and isinstance(k, Kpi), SPECS, True)
    assert np.isfinite(_ev([Kpi.ripple("gnd")]).evaluate(DESIGNS[0])
                       .kpis["ripple_v.gnd"])
