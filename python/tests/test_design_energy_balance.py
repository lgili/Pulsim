"""The evaluator's energy-balance guard: on every record with KPIs,
``P_in = sum P_avg + dE/T`` over the steady period, from the same
integrals as the KPIs. The record carries the electrical residual and
the table (datasheet) losses, which lie outside the circuit's balance;
a residual above the route's tolerance is ``failed``."""

import pulsim as p
import pulsim._period_kpis as pk
from pulsim.design import Evaluator, SteadySpec

from .test_design_evaluation import CCM, T, buck

SW = {"S": {"E_on_ref": 2e-6, "E_off_ref": 3e-6, "V_ref": 24.0,
            "I_ref": 5.0}}


def _ev(**kw):
    kw.setdefault("steady", SteadySpec(method="pss"))
    return Evaluator(buck, period=T, loads=("RL",), output="vout", **kw)


def test_the_exact_record_balances_and_carries_the_table_losses():
    r = _ev(loss_options={"switch_specs": SW}).evaluate(CCM)
    assert r.status == "ok" and r.counters["kpi_path"] == "kernel"
    assert abs(r.counters["energy_residual"]) <= 1e-10   # 1e-8 fails
    assert r.counters["p_table"] == r.losses["S"]["switching"] > 0
    # the table losses are outside the electrical balance, not in it
    plain = _ev().evaluate(CCM)
    assert plain.counters["p_table"] == 0.0
    assert plain.counters["energy_residual"] == r.counters["energy_residual"]


def test_the_transient_route_counts_the_stored_energy(monkeypatch):
    # DCM: the window ends a settle tolerance short of the orbit; without
    # dE/T the residual is 4.8e-6 and the record fails
    dcm = dict(CCM, L=20e-6, R=50.0)
    ev = _ev(engine="dsed", steady=SteadySpec(method="transient"))
    r = ev.evaluate(dcm)
    assert r.status == "ok"
    assert abs(r.counters["energy_residual"]) <= 1e-8
    monkeypatch.setattr(p.design, "_stored_power", lambda *a: 0.0)
    assert ev.evaluate(dcm).status == "failed"


def test_a_loss_integral_off_by_1e_6_is_failed_by_name(monkeypatch):
    real = pk.loss_rows

    def off(b, kp, opts):
        rows = real(b, kp, opts)
        rows["RL"] = dict(rows["RL"], P_avg=rows["RL"]["P_avg"] * (1 + 1e-6))
        return rows

    monkeypatch.setattr(pk, "loss_rows", off)
    r = _ev().evaluate(CCM)
    assert r.status == "failed" and r.kpis == {}
    assert r.reason.startswith("energy_balance: ")
    assert "exact route" in r.reason


def test_no_balance_when_a_source_is_not_measured():
    def two(d):
        b, sf = buck(d)
        b.add_voltage_source("Vaux", "aux", "gnd", 5.0)
        b.add_resistor("Raux", "aux", "gnd", 10.0)
        return b, sf

    r = Evaluator(two, period=T, loads=("RL",), sources=("Vin",),
                  steady=SteadySpec(method="pss")).evaluate(CCM)
    assert r.status == "ok"
    assert "energy_residual" not in r.counters
    full = Evaluator(two, period=T, loads=("RL",),
                     steady=SteadySpec(method="pss")).evaluate(CCM)
    assert abs(full.counters["energy_residual"]) <= 1e-10


def test_a_stepped_route_is_held_to_its_own_tolerance():
    # trbdf2's variable steps: above the exact routes' 1e-8 (DECISIONS 29:
    # stepped stays 1e-1, first order at gate edges: -3.9e-2 at T/25)
    r = _ev(engine="trbdf2", steady=SteadySpec(method="transient")
            ).evaluate(CCM)
    assert r.status == "ok"
    assert 1e-8 < abs(r.counters["energy_residual"]) < 1e-2
    # pss on the fixed step: each side of a gate edge integrated alone
    r = _ev(engine="pwl", options={"dt": T / 200}).evaluate(CCM)
    assert r.status == "ok" and abs(r.counters["energy_residual"]) < 1e-8
    assert p.design._ENERGY_RTOL == {"exact": 1e-8, "sampled": 1e-8,
                                     "stepped": 1e-1}


def test_a_coarser_sampling_widens_the_sampled_tolerance_as_1_over_n2():
    # DCM buck, dsed transient: 1.3e-7 at 200 samples per period (the
    # rows integrate products of per-step lines), within 1e-8 (2000/200)^2
    dcm = dict(CCM, L=20e-6, R=50.0)
    r = _ev(engine="dsed", samples_per_period=200,
            steady=SteadySpec(method="transient")).evaluate(dcm)
    assert r.status == "ok"
    assert 1e-8 < abs(r.counters["energy_residual"]) <= 1e-6
