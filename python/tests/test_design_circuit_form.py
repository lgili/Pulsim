"""`Evaluator(circuit=...)` on the immutable circuit IR (design task
7.2): a design is `with_params` / `with_value` of one shared base
circuit; the records equal the factory form's; the counters prove what
is shared (symbolic analyses per batch, state labels, kernel
observables) and bound the Python work per evaluation."""

import cProfile
import pstats
import sys
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim.design import Evaluator, Kpi, SteadySpec

from . import _design_reference as ref
from .test_design_evaluation import CCM, T, buck

DESIGNS = [dict(CCM, L=L, C=C) for L in (60e-6, 100e-6, 220e-6)
           for C in (22e-6, 47e-6)]


def base():
    b, _ = buck(CCM)
    return b, b.build()


def sched(d):
    b, _ = base()
    return p.make_pwm_switch_fn(1 / d["T"], d["D"], b.switch_index_of("S"),
                                b.graph.num_switches)


_B, _C = base()
_SF = p.make_pwm_switch_fn(1 / T, CCM["D"], _B.switch_index_of("S"),
                           _B.graph.num_switches)


def _pair(**kw):
    kw.setdefault("steady", SteadySpec(method="pss"))
    common = dict(period=T, loads=("RL",), output="vout",
                  constraints={"ripple_v": (None, 0.05)}, **kw)
    return (Evaluator(buck, **common),
            Evaluator(circuit=_C, fields={"L": "L.L", "C": "C.C"},
                      switch_fn=_SF, **common))


def _same(r1, r2):
    for f in ("status", "kpis", "losses", "violations", "t_j", "reason",
              "refusal", "counters", "grads", "design"):
        assert getattr(r1, f) == getattr(r2, f), f


@pytest.mark.parametrize("engine,opts,method", [
    ("dsed", {}, "pss"), ("exact", {}, "transient"),
    ("pwl", {"dt": T / 100}, "pss"), ("pwl", {"dt": T / 100}, "transient"),
    ("trbdf2", {}, "transient")])
def test_the_circuit_form_gives_the_factory_records(engine, opts, method):
    fac, cir = _pair(engine=engine, options=opts,
                     steady=SteadySpec(method=method))
    for d in DESIGNS[:3]:
        _same(fac.evaluate(d), cir.evaluate(d))


def test_threads_share_the_base_circuit_unchanged():
    _, cir = _pair()
    h = _C.value_hash()
    one = cir.evaluate_many(DESIGNS * 3, workers=1)
    eight = cir.evaluate_many(DESIGNS * 3, workers=8)
    assert one == eight
    assert _C.value_hash() == h
    assert {r.status for r in one} == {"ok", "infeasible"}


@pytest.fixture
def caches(monkeypatch):
    made = []
    real = p.PwlStateSpaceCache

    def spy(*a, **k):
        c = real(*a, **k)
        made.append(c)
        return c

    monkeypatch.setattr(p, "PwlStateSpaceCache", spy)
    return made


def _analyses(made):
    return sum(int(c.metrics().symbolic_analyses) for c in made)


def test_one_symbolic_analysis_per_batch(caches):
    st = SteadySpec(method="transient", min_periods=8, max_periods=8,
                    tol=1.0)
    fac, _ = _pair(engine="pwl", options={"dt": T / 100}, steady=st)
    # a fresh base: the plan is shared by every circuit of its structure
    # for its lifetime, so an earlier test's analysis would be reused
    cir = Evaluator(circuit=base()[1], fields={"L": "L.L", "C": "C.C"},
                    switch_fn=_SF, period=T, loads=("RL",), output="vout",
                    constraints={"ripple_v": (None, 0.05)}, engine="pwl",
                    options={"dt": T / 100}, steady=st)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = fac.evaluate_many(DESIGNS, workers=3)
        n_fac = _analyses(caches)
        caches.clear()
        b = cir.evaluate_many(DESIGNS, workers=3)
    assert [r.kpis for r in a] == [r.kpis for r in b]
    assert n_fac >= len(DESIGNS)        # one per run of the factory form
    assert _analyses(caches) == 1       # one for the whole batch


def test_the_exact_route_shares_labels_and_observables(monkeypatch):
    # the exact stepper extracts dense per-mask state spaces (no symbolic
    # analysis to share): what it shares per structure is the state
    # labels and the kernel observables -- each computed once
    pss = sys.modules["pulsim.periodic_steady_state"]
    import pulsim._period_kpis as pk
    pss._LABELS.clear()
    n = {"labels": 0, "spec": 0}
    real_l, real_s = pss._labels, pk.spec_for

    def lab(*a):
        n["labels"] += 1
        return real_l(*a)

    def spec(*a):
        n["spec"] += 1
        return real_s(*a)
    monkeypatch.setattr(pss, "_labels", lab)
    monkeypatch.setattr(pk, "spec_for", spec)
    fac, cir = _pair()
    cir.evaluate_many(DESIGNS, workers=1, backend="thread")
    assert n == {"labels": 1, "spec": 1}
    n.update(labels=0, spec=0)
    fac.evaluate_many(DESIGNS, workers=1)
    assert n == {"labels": len(DESIGNS), "spec": len(DESIGNS)}


def test_a_value_the_observables_read_is_resolved_per_design(monkeypatch):
    import pulsim._period_kpis as pk
    calls = []
    real = pk.spec_for
    monkeypatch.setattr(pk, "spec_for", lambda *a: calls.append(1) or real(*a))
    ev = Evaluator(circuit=_C, fields={"L": "L.L", "V": "Vin.V"},
                   switch_fn=_SF, period=T, loads=("RL",), output="vout",
                   steady=SteadySpec(method="pss"))
    r1, r2 = (ev.evaluate(dict(CCM, V=v)) for v in (ref.VIN, 2 * ref.VIN))
    assert len(calls) == 2
    assert r2.kpis["v_out"] == pytest.approx(2 * r1.kpis["v_out"], rel=1e-9)


def test_python_calls_per_evaluation_are_bounded():
    # 751 per evaluation of the buck Pareto before the circuit form
    # (cProfile); the bound is structural: no factory, no per-record
    # spec or label extraction, no function-level imports
    _, cir = _pair()
    cir.evaluate_many(DESIGNS[:2], workers=1)
    pr = cProfile.Profile()
    pr.enable()
    for d in DESIGNS:
        cir.evaluate(d)
    pr.disable()
    per = pstats.Stats(pr).total_calls / len(DESIGNS)
    assert per < 500, per


def test_schedule_per_design_and_parameters():
    b = p.CircuitBuilder()
    Lp = b.param("Lbase", 100e-6, "H")
    b.add_voltage_source("Vin", "vin", "gnd", ref.VIN)
    b.add_switch("S", "vin", "sw", ref.GON, ref.GOFF)
    b.add_diode("D1", "gnd", "sw", ref.GON, ref.GOFF)
    b.add_inductor("L", "sw", "x", Lp)
    b.add_resistor("RLs", "x", "vout", CCM["R_L"])
    b.add_capacitor("C", "vout", "gnd", CCM["C"])
    b.add_resistor("RL", "vout", "gnd", CCM["R"])
    calls = []

    def schedule(d):
        calls.append(d["D"])
        return p.make_pwm_switch_fn(1 / T, d["D"], b.switch_index_of("S"),
                                    b.graph.num_switches)
    ev = Evaluator(circuit=b.build(), fields={"L": "Lbase"},
                   schedule=schedule, period=T, loads=("RL",),
                   output="vout", steady=SteadySpec(method="pss"))
    fac = Evaluator(buck, period=T, loads=("RL",), output="vout",
                    steady=SteadySpec(method="pss"))
    for D in (0.3, 0.5):
        d = dict(CCM, L=150e-6, D=D)
        r = ev.evaluate(d)
        assert r.kpis == fac.evaluate(d).kpis
    assert calls == [0.3, 0.5]


def test_ids_cover_the_base_circuit_by_value():
    _, cir = _pair()
    other = Evaluator(circuit=_C.with_value("RL.R", 2.0),
                      fields={"L": "L.L", "C": "C.C"}, switch_fn=_SF,
                      period=T, loads=("RL",), output="vout",
                      constraints={"ripple_v": (None, 0.05)},
                      steady=SteadySpec(method="pss"))
    assert cir.design_id(CCM) != other.design_id(CCM)
    assert cir.design_id(CCM) == _pair()[1].design_id(CCM)


def test_configuration_errors_name_the_problem():
    with pytest.raises(TypeError, match="exactly one"):
        Evaluator(buck, circuit=_C, period=T)
    with pytest.raises(TypeError, match="exactly one"):
        Evaluator(period=T)
    with pytest.raises(ValueError, match="L1"):
        Evaluator(circuit=_C, fields={"L": "L1.L"}, period=T)
    with pytest.raises(ValueError, match="neither a parameter"):
        Evaluator(circuit=_C, fields={"L": "Lbase"}, period=T)
    with pytest.raises(TypeError, match="go with circuit"):
        Evaluator(buck, fields={"L": "L.L"}, period=T)
    with pytest.raises(TypeError, match="not both"):
        Evaluator(circuit=_C, switch_fn=_SF, schedule=sched, period=T)
    with pytest.raises(TypeError, match="pulsim.Circuit"):
        Evaluator(circuit=_B, period=T)
    from pulsim.design import ThermalSpec
    with pytest.raises(ValueError, match="temperature_dependent"):
        Evaluator(circuit=_C, period=T, thermal=ThermalSpec(
            r_th={"S": 1.0}, temperature_dependent=True))
    _, cir = _pair()
    with pytest.raises(KeyError, match="'C'"):
        cir.evaluate({k: v for k, v in CCM.items() if k != "C"})
    # the circuit pickles (IR group 4); the native schedule does not:
    # refused by name before any worker starts
    with pytest.raises(p.UnsupportedError, match="switch_fn does not pickle"):
        cir.evaluate_many(DESIGNS, workers=2, backend="process")


def test_user_kpis_run_on_the_circuit_form():
    fac, cir = _pair(kpis=[Kpi("v_mid", lambda per: per.average(
        np.asarray(per.result.states)[:, 0]))])
    _same(fac.evaluate(CCM), cir.evaluate(CCM))


def test_periodic_steady_state_takes_a_circuit():
    a = p.periodic_steady_state(_B, period=T, switch_fn=_SF)
    b = p.periodic_steady_state(_C.with_value("L.L", CCM["L"]), period=T,
                                switch_fn=_SF)
    np.testing.assert_array_equal(a.x0, b.x0)
    assert a.state_labels == b.state_labels
