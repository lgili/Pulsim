"""`pulsim.design`: the evaluation contract (add-design-optimization-workflow,
tasks 1-2). The KPI oracle is a numpy matrix-exponential reference of
the same periodic orbit (`_design_reference.py`); the failure tests pin
the status mapping of design D3: a refusal, a non-convergence and a
violated bound are typed records, never a default number, and any
other exception propagates.
"""

from pathlib import Path

import numpy as np
import pytest

import pulsim as p
from pulsim.design import (EvaluationRecord, Evaluator, Kpi, SteadySpec,
                           as_constraints, as_objectives, evaluate)

from . import _design_reference as ref

T = 1e-5
CCM = dict(L=100e-6, C=47e-6, R=1.0, R_L=0.01, T=T, D=0.5)


def buck(d):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", ref.VIN)
    b.add_switch("S", "vin", "sw", ref.GON, ref.GOFF)
    b.add_diode("D1", "gnd", "sw", ref.GON, ref.GOFF)
    b.add_inductor("L", "sw", "x", d["L"])
    b.add_resistor("RLs", "x", "vout", d["R_L"])
    b.add_capacitor("C", "vout", "gnd", d["C"])
    b.add_resistor("RL", "vout", "gnd", d["R"])
    sf = p.make_pwm_switch_fn(1 / d["T"], d["D"], b.switch_index_of("S"),
                              b.graph.num_switches)
    return b, sf


def shockley_buck(d):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_shockley_diode("D1", "gnd", "sw")
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 5.0)
    return b, p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                                   b.graph.num_switches)


def _ev(**kw):
    kw.setdefault("steady", SteadySpec(method="pss"))
    return Evaluator(buck, period=T, loads=("RL",), output="vout", **kw)


def test_the_tested_pulsim_is_this_tree():
    assert Path(p.__file__).resolve().is_relative_to(
        Path(__file__).resolve().parents[1])


# ---------------------------------------------------------------------
# 1.3 the oracle
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def ccm():
    return _ev().evaluate(CCM), ref.kpis(CCM)


def test_ccm_buck_kpis_match_the_matrix_exponential_reference(ccm):
    rec, R = ccm
    assert rec.status == "ok", rec.reason
    k = rec.kpis
    # measured 2e-12 (efficiency), 8e-15 (loss), 6e-11 (ripple)
    for name in ("efficiency", "p_loss", "ripple_v", "p_in", "p_out"):
        assert k[name] == pytest.approx(R[name], rel=1e-8), name
    assert rec.losses["S"]["conduction"] == pytest.approx(R["p_s"], rel=1e-8)
    assert rec.losses["D1"]["conduction"] == pytest.approx(R["p_d"], rel=1e-8)
    assert rec.losses["RLs"]["resistive"] == pytest.approx(R["p_rl"],
                                                           rel=1e-8)
    assert "RL" not in rec.losses           # the load is the output
    # sanity: the reference itself against the textbook numbers
    assert R["efficiency"] == pytest.approx(0.98911738798, rel=1e-9)
    assert R["p_in"] == pytest.approx(R["p_out"] + R["p_loss"], rel=1e-12)


def test_ripple_agrees_with_the_small_ripple_formula(ccm):
    rec, _ = ccm
    D, f = CCM["D"], 1 / T
    dv = (1 - D) * rec.kpis["v_out"] / (8 * CCM["L"] * CCM["C"] * f * f)
    assert rec.kpis["ripple_v"] == pytest.approx(dv, rel=0.02)   # 1.1 %


def test_record_shape_counters_and_provenance(ccm):
    rec, _ = ccm
    assert isinstance(rec, EvaluationRecord)
    assert rec.counters["steady_method"] == "pss"
    assert rec.counters["route"] == "pss/dsed"
    assert 1 <= rec.counters["pss_period_runs"] <= 8
    assert rec.counters["python_callbacks"] == 0
    assert rec.provenance["pulsim"] == p.__version__
    assert len(rec.provenance["build"]) == 16
    assert len(rec.design_id) == 64 and rec.seed == int(rec.design_id, 16) % 2**32
    assert rec.violations == {} and rec.refusal is None and rec.reason is None
    with pytest.raises(TypeError):
        rec.kpis["efficiency"] = 1.0       # records are immutable
    # the semiconductor stresses are there
    for n in ("S", "D1"):
        assert {f"i_peak.{n}", f"i_rms.{n}", f"v_peak.{n}"} <= set(rec.kpis)
    assert rec.kpis["v_peak.S"] == pytest.approx(24.0, rel=1e-3)


def test_design_id_depends_on_design_and_options_only():
    a, b = _ev(), _ev()
    assert a.design_id(CCM) == b.design_id(dict(reversed(CCM.items())))
    assert a.design_id(CCM) != a.design_id({**CCM, "C": 48e-6})
    assert a.design_id(CCM) != _ev(samples_per_period=500).design_id(CCM)

    def law(k):
        return lambda x: k * x
    # two closures of one factory are two designs; the same capture is one
    assert a.design_id({"f": law(1.0)}) != a.design_id({"f": law(2.0)})
    assert a.design_id({"f": law(1.0)}) == a.design_id({"f": law(1.0)})
    assert a.design_id({"v": np.array([1.0, 2.0])}) == \
        a.design_id({"v": [1.0, 2.0]})


def test_user_kpi_is_time_weighted():
    def mean_vout(per):
        return per.average(per.result.v("vout"))
    rec = _ev(kpis=[Kpi("mean_vout", mean_vout)]).evaluate(CCM)
    assert rec.kpis["mean_vout"] == pytest.approx(rec.kpis["v_out"],
                                                  rel=1e-12)


# ---------------------------------------------------------------------
# 2. failures are records
# ---------------------------------------------------------------------

def test_refusal_is_a_record_with_the_four_fields():
    rec = Evaluator(shockley_buck, period=T, loads=("RL",),
                    engine="dsed").evaluate({})
    assert rec.status == "refused"
    assert rec.refusal == ("dsed", "", "D1", "ShockleyDiode")
    assert "D1" in rec.reason
    assert rec.kpis == {} and rec.losses == {}
    # 'auto' routed to the transient BEFORE running: pss refused it
    assert rec.counters["steady_method"] == "transient"
    assert "ShockleyDiode" in rec.counters["steady_route_reason"]


def test_pss_refusal_is_not_followed_by_a_transient():
    rec = Evaluator(shockley_buck, period=T, loads=("RL",),
                    steady=SteadySpec(method="pss")).evaluate({})
    assert rec.status == "refused"
    assert rec.refusal[0] == "periodic_steady_state"
    assert rec.counters["simulations"] == 0


def test_not_settled_is_failed_with_the_reason():
    # DCM buck: Floquet multiplier 0.989, ~1250 periods to 1e-6.
    ev = Evaluator(buck, period=T, loads=("RL",), output="vout",
                   engine="dsed", steady=SteadySpec(
                       method="transient", min_periods=8, max_periods=32))
    rec = ev.evaluate({**CCM, "R": 50.0, "D": 0.3})
    assert rec.status == "failed"
    assert rec.reason.startswith("not_settled")
    assert rec.counters["periods"] == 32
    assert rec.counters["settle_error"] > 1e-6
    assert rec.kpis == {}


def test_pss_non_convergence_is_failed():
    rec = _ev(steady=SteadySpec(method="pss", max_period_runs=1)).evaluate(
        {**CCM, "R": 50.0, "D": 0.3})
    assert rec.status == "failed"
    assert "no periodic orbit found" in rec.reason
    assert rec.kpis == {}


def test_kernel_runtime_error_is_failed_and_other_errors_propagate(
        monkeypatch):
    def aborted(*a, **k):
        raise RuntimeError("did not settle after 64 rounds")
    monkeypatch.setattr(p, "simulate", aborted)
    monkeypatch.setattr(p._pulsim, "dsed_period_map", aborted)   # pss route
    rec = _ev().evaluate(CCM)
    assert rec.status == "failed"
    assert "did not settle" in rec.reason

    def bug(*a, **k):
        raise TypeError("a bug, not a design outcome")
    monkeypatch.setattr(p, "simulate", bug)
    monkeypatch.setattr(p._pulsim, "dsed_period_map", bug)
    with pytest.raises(TypeError, match="a bug"):
        _ev().evaluate(CCM)


def test_not_implemented_and_recursion_errors_propagate(monkeypatch):
    for exc in (NotImplementedError, RecursionError):
        def stub(*a, exc=exc, **k):
            raise exc("a stub or a bug")
        monkeypatch.setattr(p, "simulate", stub)
        monkeypatch.setattr(p._pulsim, "dsed_period_map", stub)
        with pytest.raises(exc):
            _ev().evaluate(CCM)


def test_an_unstable_orbit_is_failed_not_ok(monkeypatch):
    import dataclasses
    import sys
    pss = sys.modules["pulsim.periodic_steady_state"]   # the module
    real = pss.periodic_steady_state

    def unstable(*a, **k):
        return dataclasses.replace(real(*a, **k), stable=False,
                                   unstable_multipliers=[(0, 1.3 + 0j, "real")])
    monkeypatch.setattr(pss, "periodic_steady_state", unstable)
    rec = _ev(steady=SteadySpec(method="pss")).evaluate(CCM)
    assert rec.status == "failed"
    assert rec.reason.startswith("unstable_orbit")
    assert rec.kpis == {}


def test_build_errors_propagate():
    def broken(d):
        raise ZeroDivisionError("in the user's build")
    with pytest.raises(ZeroDivisionError):
        evaluate(broken, {}, period=T)


def test_ripple_bound_exceeded_is_infeasible_with_the_excess_in_volts():
    rec = _ev(constraints={"ripple_v": (None, 0.12)}).evaluate(
        {**CCM, "C": 4.8e-6})
    assert rec.status == "infeasible"
    r = rec.kpis["ripple_v"]
    assert r == pytest.approx(0.15, abs=0.005)
    assert rec.violations["ripple_v"] == pytest.approx(r - 0.12, rel=1e-12)
    assert rec.violations["ripple_v"] == pytest.approx(0.03, abs=0.005)
    ok = _ev(constraints={"ripple_v": (None, 0.12)}).evaluate(CCM)
    assert ok.status == "ok" and ok.violations["ripple_v"] == 0.0


def test_adapters_penalise_refused_and_failed_only(ccm):
    rec, _ = ccm
    names = ["efficiency", "p_loss"]
    obj = as_objectives(rec, names, maximise=["efficiency"])
    assert np.all(np.isfinite(obj)) and obj[0] == -rec.kpis["efficiency"]
    refused = Evaluator(shockley_buck, period=T, engine="dsed").evaluate({})
    failed = _ev(steady=SteadySpec(method="pss", max_period_runs=1)
                 ).evaluate({**CCM, "R": 50.0, "D": 0.3})
    for r in (refused, failed):
        assert list(as_objectives(r, names, penalty=7.0)) == [7.0, 7.0]
        assert list(as_constraints(r, ["ripple_v"], penalty=7.0)) == [7.0]
    inf = _ev(constraints={"ripple_v": (None, 0.12)}).evaluate(
        {**CCM, "C": 4.8e-6})
    g = as_constraints(inf, ["ripple_v"])
    assert g[0] == inf.violations["ripple_v"] > 0
    assert np.all(np.isfinite(as_objectives(inf, names)))
    with pytest.raises(KeyError):       # a misspelt name is not "feasible"
        as_constraints(inf, ["ripple"])


def test_bad_configuration_is_rejected_up_front():
    with pytest.raises(ValueError, match="dsed exact stepper"):
        Evaluator(buck, period=T, engine="pwl", steady=SteadySpec("pss"))
    with pytest.raises(ValueError, match="min_periods"):
        Evaluator(buck, period=T, steady=SteadySpec(min_periods=4))
    with pytest.raises(ValueError, match="name of a default KPI"):
        _ev(kpis=[Kpi("efficiency", lambda per: 1.0)]).evaluate(CCM)
    with pytest.raises(KeyError, match="names no KPI"):
        _ev(constraints={"nope": (0, 1)}).evaluate(CCM)


def test_a_run_simulate_only_warns_about_is_failed_not_ok():
    """simulate WARNS (once per call site) for an implausible node
    voltage; an optimiser loop never sees it. Before integration batch
    6b this design came back `ok` with v_peak.S = 1.19e10 V (trbdf2)."""
    def opening(d):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", 48.0)
        b.add_inductor("L", "vin", "sw", 1e-3)
        b.add_switch("S", "sw", "gnd", 1e3, 1e-12)    # no freewheel path
        return b, p.make_pwm_switch_fn(1e4, 0.5, 0, 1)
    with pytest.warns(UserWarning):
        rec = Evaluator(opening, period=1e-4, engine="trbdf2", steady=SteadySpec(
            method="transient", min_periods=8, max_periods=8)).evaluate({})
    assert rec.status == "failed", rec
    assert rec.reason.startswith("implausible_voltage: node sw")
    assert rec.kpis == {}


def test_a_trbdf2_missed_commutation_is_failed_not_ok():
    """The floating bridge of test_trbdf2_robustness (rtol 2e-5): trbdf2
    holds a diode against its rule and only warns; design maps it to
    failed."""
    from pulsim import design
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 325.0, 50.0)
    b.add_resistor("Rs", "s", "a", 1e-3)
    p.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="gnd", dc_pos="dp",
                           dc_neg="dn")
    b.add_inductor("Lf", "dp", "dq", 1e-3)
    b.add_capacitor("C", "dq", "dn", 470e-6)
    b.add_resistor("RL", "dq", "dn", 1e3)
    with pytest.warns(RuntimeWarning, match="commutation rule forbids"):
        res = p.simulate(b, t_end=0.06, engine="trbdf2", rtol=2e-5, atol=2e-8)
    with pytest.raises(design._Failed, match="missed_commutation: .*BR__D4"):
        design._raise_if_suspect(res, b)


def test_exact_is_the_dsed_engine_for_the_evaluator():
    for steady in (SteadySpec(method="pss"), SteadySpec(method="auto")):
        a = Evaluator(buck, period=T, loads=("RL",), engine="exact",
                      steady=steady).evaluate(CCM)
        b = Evaluator(buck, period=T, loads=("RL",), engine="dsed",
                      steady=steady).evaluate(CCM)
        assert a.status == "ok" and a == b
    with pytest.raises(ValueError, match="'exact' \\(alias 'dsed'\\)"):
        Evaluator(buck, period=T, engine="spice")


def test_a_guard_held_trace_or_an_undersampled_gate_is_failed(monkeypatch):
    """Batch-6b review: two more warnings that mean the number may be
    wrong. A guard-held inductor trace is the configured limit; a gate
    pulse narrower than trbdf2's switch_fn sampling may have hidden
    others. The explicit pss route refuses an unextractable circuit
    (as the auto route does) instead of raising a raw ValueError."""
    from types import SimpleNamespace as NS

    from pulsim import design
    run = NS(_implausible_voltage=None, inductor_guard_actions=[object()],
             engine_used="pwl")
    with pytest.raises(design._Failed, match="inductor_guard"):
        design._raise_if_suspect(run, None)
    run = NS(_implausible_voltage=None, inductor_guard_actions=[],
             engine_used="trbdf2",
             _trbdf2_stats={"n_gate_undersampled": 1, "gate_pulse_min": 2e-6})
    with pytest.raises(design._Failed, match="gate_undersampled: a 2e-06 s"):
        design._raise_if_suspect(run, None)

    import sys
    pss = sys.modules["pulsim.periodic_steady_state"]   # the name is the function

    # the refusal is typed at its source (_Solver), so make the
    # extraction itself fail
    def no_extract(*a, **k):
        raise RuntimeError("state space extraction failed (test)")
    monkeypatch.setattr(pss, "_labels", no_extract)
    rec = _ev(steady=SteadySpec(method="pss")).evaluate(CCM)
    assert rec.status == "refused" and rec.refusal[1] == "state_space", rec


def test_accepted_with_error_steps_are_counted():
    """trbdf2 forced accepts and Newton roundoff accepts stay ``ok`` but
    are counted in the record (DECISIONS 5), so a filter can see them."""
    from types import SimpleNamespace as NS

    from pulsim import design
    run = NS(_implausible_voltage=None, inductor_guard_actions=[],
             engine_used="pwl", n_accept=10, newton_roundoff_accepts=2,
             _trbdf2_stats={"n_forced_accepts": 3})
    counters = {"simulations": 0}
    design.Evaluator._count(run, counters, None)
    assert counters["forced_accepts"] == 3
    assert counters["newton_roundoff_accepts"] == 2
    run.newton_roundoff_accepts, run._trbdf2_stats = 0, {}
    counters = {"simulations": 0}
    design.Evaluator._count(run, counters, None)
    assert "forced_accepts" not in counters
    assert "newton_roundoff_accepts" not in counters
    run.resolution = NS(lte_max=0.5, ringing=[], unestimated_fraction=0.0)
    design.Evaluator._count(run, counters, None)
    assert counters["resolution_warnings"] == 1
