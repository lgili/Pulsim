"""`pulsim.design`: steady-state routes and batch evaluation
(add-design-optimization-workflow, tasks 4 and 5).

Counters decide (route, period runs, periods simulated); no wall clock.
The DCM orbit is the 50-digit reference of
`add-periodic-steady-state-shooting` (`_pss_reference.py`).
"""

import numpy as np
import pytest

import pulsim as p
from pulsim.design import Evaluator, Kpi, SteadySpec

from . import _pss_reference as pref

T = pref.T


def dcm_buck(d):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", pref.VIN)
    b.add_switch("S", "vin", "sw", pref.GON, pref.GOFF)
    b.add_diode("D1", "gnd", "sw", pref.GON, pref.GOFF)
    b.add_inductor("L", "sw", "vout", d.get("L", 100e-6))
    b.add_capacitor("C", "vout", "gnd", d.get("C", 47e-6))
    b.add_resistor("RL", "vout", "gnd", d.get("R", 50.0))
    sf = p.make_pwm_switch_fn(1 / T, d.get("D", 0.3), b.switch_index_of("S"),
                              b.graph.num_switches)
    return b, sf


def lauritzen_buck(d):
    """A Lauritzen (charge-storage) diode: periodic_steady_state refuses
    it (a stateful device), the variable-step engine runs it."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_lauritzen_diode("D1", "gnd", "sw", tau=1e-7, T_M=1e-8)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 5.0)
    return b, p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                                   b.graph.num_switches)


def _v_c0(per):
    """v_C at the section (the start of the KPI period)."""
    return float(np.interp(per.t_from, per.result.times,
                           per.result.v("vout")))


V_C0 = Kpi("v_c0", _v_c0)


def _dcm(**kw):
    return Evaluator(dcm_buck, period=T, loads=("RL",), output="vout",
                     kpis=[V_C0], **kw)


# ---------------------------------------------------------------------
# 4. steady state
# ---------------------------------------------------------------------

def test_pss_reaches_the_dcm_orbit():
    rec = _dcm(steady=SteadySpec(method="pss")).evaluate({})
    assert rec.status == "ok", rec.reason
    assert abs(rec.kpis["v_c0"] - pref.DCM_ORBIT[1]) <= 1e-9
    assert rec.counters["route"] == "pss/dsed"
    assert rec.counters["pss_period_runs"] <= 8      # 6 today
    assert "periods" not in rec.counters


@pytest.mark.parametrize("tol", [1e-4, 1e-6])
def test_transient_reaches_the_dcm_orbit_within_the_settle_tolerance(tol):
    """At 32 periods the start-up overshoot makes the last half contract
    fast (r = 0.19) while each period moves 1.1 % (rho = 0.989); an
    estimate from the halves alone stopped there, 51 % off."""
    rec = _dcm(engine="dsed", steady=SteadySpec(method="transient",
                                                tol=tol)).evaluate({})
    assert rec.status == "ok", rec.reason
    assert rec.counters["route"] == "transient/dsed"
    assert rec.counters["settle_error"] <= tol
    # the tolerance is relative to the state's magnitude (~9 V)
    assert abs(rec.kpis["v_c0"] - pref.DCM_ORBIT[1]) <= tol * 9.0
    # multiplier 0.989: 16 periods are far from enough, the run doubled
    assert rec.counters["periods"] >= 1024
    assert rec.counters["simulations"] >= 7


def test_auto_picks_pss_when_it_applies_and_transient_when_it_refuses():
    a = _dcm().evaluate({})
    assert a.counters["steady_method"] == "pss"
    r = Evaluator(lauritzen_buck, period=T, output="vout").evaluate({})
    assert r.status == "ok", r.reason
    assert r.counters["steady_method"] == "transient"
    assert "LauritzenDiode" in r.counters["steady_route_reason"]
    assert r.counters["route"] == "transient/trbdf2"
    assert r.counters["settle_error"] <= 1e-6
    assert r.counters["periods"] <= 2048          # 1024 today
    # its power is not in the loss summary: no loss total is claimed
    assert "p_loss" not in r.kpis and "efficiency" not in r.kpis
    assert r.counters["losses_not_covered"] == "D1"
    assert r.kpis["ripple_v"] > 0


def test_an_uncovered_dissipator_refuses_the_efficiency():
    """Summed without it, the efficiency read 0.99989: the diode's
    1.36 W of the 28.4 W input was missing."""
    r = Evaluator(lauritzen_buck, period=T, loads=("RL",),
                  output="vout").evaluate({})
    assert r.status == "refused"
    assert r.refusal == ("design.evaluate", "losses", "D1",
                         "lauritzen_diode")
    assert r.kpis == {}


def test_pss_failure_does_not_fall_back_to_a_transient():
    rec = _dcm(steady=SteadySpec(method="auto", max_period_runs=1)
               ).evaluate({})
    assert rec.status == "failed"
    assert rec.counters["steady_method"] == "pss"
    assert "periods" not in rec.counters


# ---------------------------------------------------------------------
# 5. batch, determinism, thread safety
# ---------------------------------------------------------------------

def _designs(n):
    rng = np.random.default_rng(1234)
    return [dict(L=float(rng.uniform(20e-6, 200e-6)),
                 C=float(rng.uniform(10e-6, 100e-6)),
                 R=float(rng.uniform(1.0, 60.0)),
                 D=float(rng.uniform(0.2, 0.7))) for _ in range(n)]


def _strip(r):
    return {k: getattr(r, k) for k in r.__dataclass_fields__ if k != "wall_s"}


def test_64_designs_on_1_and_8_workers_give_equal_records():
    ev = _dcm(steady=SteadySpec(method="pss"))
    ds = _designs(64)
    one = ev.evaluate_many(ds, workers=1)
    eight = ev.evaluate_many(ds, workers=8)
    assert len(one) == len(eight) == 64
    assert [r.design for r in one] == ds          # input order
    assert one == eight                           # D1 equality: no wall_s
    for a, b in zip(one, eight):
        assert _strip(a) == _strip(b)
    assert {r.status for r in one} == {"ok"}


def test_two_evaluations_of_one_design_are_equal():
    ev = _dcm(steady=SteadySpec(method="pss"))
    d = _designs(1)[0]
    a, b = ev.evaluate(d), ev.evaluate(d)
    assert a == b and a.wall_s >= 0 and _strip(a) == _strip(b)


def maybe_shockley(d):
    b, sf = dcm_buck(d)
    if d.get("shockley"):
        b.add_shockley_diode("D2", "gnd", "sw")
    return b, sf


def test_a_refusing_design_does_not_stop_the_batch():
    ev = Evaluator(maybe_shockley, period=T, loads=("RL",), engine="dsed")
    ds = _designs(6)
    ds[4]["shockley"] = True
    recs = ev.evaluate_many(ds, workers=4)
    assert [r.status for r in recs].count("refused") == 1
    assert recs[4].status == "refused"
    assert recs[4].refusal == ("dsed", "", "D2", "ShockleyDiode")
    assert all(r.status == "ok" for i, r in enumerate(recs) if i != 4)


def test_three_identical_refusals_stop_the_batch():
    ev = Evaluator(maybe_shockley, period=T, loads=("RL",), engine="dsed")
    ds = [{**d, "shockley": True} for d in _designs(5)]
    with pytest.raises(p.UnsupportedError, match="first three") as ei:
        ev.evaluate_many(ds, workers=2)
    assert (ei.value.engine, ei.value.device) == ("dsed", "D2")


def test_a_mid_run_mode_refusal_does_not_stop_the_batch(monkeypatch):
    """Batch-6b review: the kernel's 'consistent_mode' refusal is met at
    a state that depends on the design, so three of them in a row are
    three records, not a configuration verdict."""
    from pulsim._refusal import unsupported

    def refuse(self, design, did, counters):
        raise unsupported("dsed", "consistent_mode", "mode selection at t = 1 s",
                          device="D1", kind="Diode")
    monkeypatch.setattr(Evaluator, "_evaluate", refuse)
    recs = Evaluator(dcm_buck, period=T).evaluate_many(_designs(4), workers=1)
    assert [r.status for r in recs] == ["refused"] * 4


def test_a_bug_in_a_batch_propagates():
    def buggy(d):
        if d["D"] > 0.5:
            raise ZeroDivisionError("user bug")
        return dcm_buck(d)
    ev = Evaluator(buggy, period=T, loads=("RL",))
    ds = [dict(D=0.3), dict(D=0.3), dict(D=0.3), dict(D=0.3), dict(D=0.6)]
    with pytest.raises(ZeroDivisionError):
        ev.evaluate_many(ds, workers=2)


def test_process_backend_gives_the_thread_records():
    ev = _dcm(steady=SteadySpec(method="pss"))
    ds = _designs(5)
    assert ev.evaluate_many(ds, workers=2, backend="process") == \
        ev.evaluate_many(ds, workers=1)
