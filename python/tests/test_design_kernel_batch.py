"""The kernel batch evaluator (add-design-optimization-workflow DECISIONS
26): on the circuit form with declarative KPIs, `evaluate_many` runs
every design in one GIL-free kernel call. Its records EQUAL the Python
path's (bit for bit, counters included); every typed outcome is the
Python path's; the counters prove no Python runs per design."""

import importlib.util
import math
import pathlib
import sys

import pytest

import pulsim as p
import pulsim.design as design
from pulsim import _design_kernel as dk
from pulsim.design import Constraint, Evaluator, Kpi, SteadySpec

from .test_design_circuit_form import DESIGNS, _B, _C, _SF
from .test_design_evaluation import T

EXAMPLE = (pathlib.Path(__file__).resolve().parents[2] / "examples"
           / "optimization" / "buck_pareto.py")


@pytest.fixture(scope="module")
def bp():
    spec = importlib.util.spec_from_file_location("buck_pareto", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["buck_pareto"] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("buck_pareto", None)


def _cir(circuit=_C, **kw):
    kw.setdefault("steady", SteadySpec(method="pss"))
    kw.setdefault("kpis", [Kpi.ripple("x"), Kpi.formula(
        "f", "v_out * efficiency / max(i_peak.L, 1e-3) + loss.S")])
    kw.setdefault("constraints", {"ripple_v": (None, 0.05)})
    return Evaluator(circuit=circuit, fields={"L": "L.L", "C": "C.C"},
                     switch_fn=_SF, period=T, loads=("RL",), output="vout",
                     **kw)


def _python(ev, ds):
    return [ev.evaluate(d) for d in ds]


def test_the_buck_pareto_records_equal_the_python_path(bp):
    ds = bp.latin_hypercube(32)
    for name, ev in bp.make_evaluator().items():
        sub = [d for d in ds if d["part"] == name]
        recs = ev.evaluate_many(sub, workers=4)
        assert ev.last_batch["kernel"] == len(sub)
        assert ev.last_batch["python"] == 0
        assert recs == _python(ev, sub)          # every field but wall_s
        assert all("batch" not in r.counters for r in recs)


def _tables():
    import numpy as np
    from pulsim.loss_tables import LossTable
    v, i, tj = [10.0, 24.0, 48.0], [0.0, 5.0, 20.0], [25.0, 125.0]
    e = np.fromfunction(lambda a, b, c: 1e-6 * (1 + a) * (1 + 2 * b)
                        * (1 + 0.1 * c), (3, 3, 2))
    return {"S": {"E_on_table": LossTable(v, i, tj, e),
                  "E_off_table": LossTable(v, i, tj, 0.5 * e), "Tj": 90.0}}


@pytest.mark.parametrize("cycles", [1, 9])
def test_a_diode_circuit_and_a_multi_edge_period_equal_the_python_path(
        cycles):
    # the CCM buck's diode (mode guard on); a period of nine switching
    # cycles sums nine turn-on energies (numpy's pairwise sum from 8)
    ev = _cir(loss_options={"switch_specs": _tables()})
    ev.period = cycles * T
    recs = ev.evaluate_many(DESIGNS, workers=3)
    assert recs == _python(ev, DESIGNS)
    assert ev.last_batch["kernel"] == len(DESIGNS)
    assert all(r.losses["S"]["switching"] > 0 for r in recs)


def test_counters_no_python_per_design_one_analysis_nogil():
    c = _B.build()                       # a fresh structure: no analysis yet
    ev = _cir(c)
    ev.evaluate_many(DESIGNS, workers=4)
    st = ev.last_batch
    assert st["kernel"] == len(DESIGNS) and st["python"] == 0
    assert st["python_calls_in_kernel"] == 0
    assert st["nogil"] is True
    assert st["symbolic_analyses"] <= 1      # one per structure, shared
    ev.evaluate_many(DESIGNS, workers=4)
    assert ev.last_batch["symbolic_analyses"] == 0
    assert ev.last_batch["periodicity_checks"] == 0   # cached per pattern


def test_threads_processes_and_order_give_equal_records(bp, monkeypatch):
    ev = _cir(kpis=[Kpi.ripple("x")])
    one = ev.evaluate_many(DESIGNS * 2, workers=1)
    eight = ev.evaluate_many(DESIGNS * 2, workers=8)
    assert one == eight
    assert [r.design for r in one] == DESIGNS * 2
    assert ev.evaluate_many(DESIGNS, workers=2, backend="thread") == one[:6]
    # processes need a picklable evaluator: the example's (a circuit
    # factory and formula KPIs), imported by name in a spawned worker
    monkeypatch.syspath_prepend(str(EXAMPLE.parent))
    ds = bp.latin_hypercube(16)
    ev = bp.make_evaluator()["SYN-B"]
    sub = [d for d in ds if d["part"] == "SYN-B"]
    assert ev.evaluate_many(sub, workers=2, backend="process") == \
        ev.evaluate_many(sub, workers=4)
    assert ev.last_batch["kernel"] == len(sub)


def test_a_design_the_kernel_does_not_finish_is_the_python_paths_record(
        monkeypatch):
    ev = _cir()
    # energy balance failing everywhere (tolerance below rounding)
    monkeypatch.setitem(design._ENERGY_RTOL, "exact", 1e-30)
    ev.__dict__.pop("_kplan", None)
    r = ev.evaluate_many(DESIGNS[:4], workers=2)
    assert {x.status for x in r} == {"failed"}
    assert all("energy_balance" in x.reason for x in r)
    assert all(x.counters["batch"] == "python: kernel: energy balance"
               for x in r)
    assert ev.last_batch["python_reasons"] == {
        "kernel: energy balance": 4}


def test_non_finite_and_infeasible_and_budget_outcomes():
    ev = _cir(kpis=[Kpi.formula("bad", "v_out / (efficiency - efficiency)")])
    r = ev.evaluate_many(DESIGNS[:4], workers=2)
    assert {x.status for x in r} == {"failed"}
    assert r[0].reason == "non-finite KPI: bad"
    assert all(x.counters["batch"] == "python: kernel: a non-finite KPI"
               for x in r)
    assert [dict_free(x) for x in r] == _python(ev, DESIGNS[:4])
    # infeasible with the excess, from the kernel
    ev = _cir(constraints={"ripple_v": (None, 1e-6)}, kpis=[])
    r = ev.evaluate_many(DESIGNS[:4], workers=2)
    assert {x.status for x in r} == {"infeasible"}
    assert all(math.isclose(x.violations["ripple_v"],
                            x.kpis["ripple_v"] - 1e-6) for x in r)
    assert ev.last_batch["kernel"] == 4
    # an orbit the budget does not reach: failed by the Python path
    ev = _cir(steady=SteadySpec(method="pss", max_period_runs=1))
    r = ev.evaluate_many(DESIGNS[:4], workers=2)
    assert {x.status for x in r} == {"failed"}
    assert all(x.counters["batch"] == "python: kernel: orbit: budget"
               for x in r)
    assert [dict_free(x) for x in r] == _python(ev, DESIGNS[:4])


def dict_free(rec):
    """The record without the batch marker."""
    from dataclasses import replace
    c = {k: v for k, v in rec.counters.items() if k != "batch"}
    return replace(rec, counters=design._fz(c))


def test_a_refused_design_keeps_its_four_fields():
    # one design's schedule is an opaque Python callable: that design
    # takes the Python path, which refuses it by name
    def sched(d):
        return (lambda t: _SF(t)) if d["L"] == 60e-6 else _SF
    ev = Evaluator(circuit=_C, fields={"L": "L.L", "C": "C.C"},
                   schedule=sched, period=T, loads=("RL",), output="vout",
                   steady=SteadySpec(method="pss"))
    r = ev.evaluate_many(DESIGNS, workers=2)
    bad = [x for x in r if x.design["L"] == 60e-6]
    assert {x.status for x in bad} == {"refused"}
    assert all(len(x.refusal) == 4 and x.refusal[1] == "switch_fn"
               for x in bad)
    assert all(x.counters["batch"].startswith("python: a schedule")
               for x in bad)
    good = [x for x in r if x.design["L"] != 60e-6]
    assert {x.status for x in good} <= {"ok", "infeasible"}
    assert all("batch" not in x.counters for x in good)


def test_what_the_plan_cannot_express_takes_the_python_path_by_name():
    ev = _cir(kpis=[Kpi("u", lambda per: 1.0)])
    r = ev.evaluate_many(DESIGNS[:4], workers=2)
    assert all(x.counters["batch"] == "python: KPI 'u' is a Python callable"
               for x in r)
    assert ev.last_batch["kernel"] == 0
    with pytest.raises(ValueError, match="no kernel plan: KPI 'u'"):
        ev.evaluate_many(DESIGNS[:4], backend="kernel")


def test_a_periodicity_mismatch_stops_the_kernel_batch(monkeypatch):
    real = p.simulate

    def sim(*a, **kw):
        r = real(*a, **kw)
        r.states_reduced = [x * (1 + 1e-6) for x in r.states_reduced]
        return r
    monkeypatch.setattr(p, "simulate", sim)
    with pytest.raises(design.PeriodicityError, match="kernel batch"):
        _cir().evaluate_many(DESIGNS, workers=2)


def test_a_design_error_propagates_with_the_design_index():
    ev = _cir()
    with pytest.raises(KeyError):
        ev.evaluate_many(DESIGNS + [{"L": 1e-4}], workers=2)


def test_formula_parses_names_and_ieee_division():
    f = dk.Formula("max(a, b.c) ** 2 / -x + abs(-1) + sqrt(4) + inf * 0")
    assert f.names == ["a", "b.c", "x"]
    assert math.isnan(f.evaluate({"a": 1.0, "b.c": 3.0, "x": 2.0}.get))
    assert dk.Formula("1 / x").evaluate({"x": 0.0}.get) == math.inf
    assert dk.Formula("-1 / x").evaluate({"x": 0.0}.get) == -math.inf
    for bad in ("a if b else c", "a < b", "f(x)", "x[0]", "'s'"):
        with pytest.raises(ValueError, match="Kpi.formula"):
            dk.Formula(bad)
    import pickle
    k = Kpi.formula("y", "2 * v_out")
    assert pickle.loads(pickle.dumps(k)) == k


@pytest.mark.skipif(sys.version_info < (3, 12),
                    reason="sum() is a plain loop before 3.12")
def test_psum_is_the_builtin_float_sum():
    import random
    rng = random.Random(7)
    for n in range(0, 40):
        xs = [rng.uniform(-1, 1) * 10 ** rng.randint(-8, 8)
              for _ in range(n)]
        assert dk._psum(xs) == float(sum(xs))


def test_constraint_objects_and_formula_kpis_run_in_the_kernel():
    ev = _cir(constraints=[Constraint.max("f", 1e9)])
    r = ev.evaluate_many(DESIGNS[:3])
    assert ev.last_batch["kernel"] == 3
    assert r == _python(ev, DESIGNS[:3])


def test_changing_kpis_or_constraints_after_a_batch_is_refused():
    """The kernel plan and the design ids froze kpis= and constraints=
    when the evaluator was made: mutating them later is refused by name,
    never a batch that reads the old ones."""
    ev = _cir()
    ev.evaluate_many(DESIGNS[:1])
    ev.constraints["ripple_v"] = (None, 0.5)
    with pytest.raises(ValueError, match="constraints="):
        ev.evaluate_many(DESIGNS[:1])
    ev = _cir()
    ev.evaluate_many(DESIGNS[:1])
    ev.kpis = ev.kpis[:1]
    with pytest.raises(ValueError, match="kpis="):
        ev.evaluate_many(DESIGNS[:1])
