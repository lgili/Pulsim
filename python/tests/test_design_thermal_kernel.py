"""Event losses on the periodic orbit and the kernel junction fixed point
(add-in-kernel-event-losses, the exact-route part;
add-design-optimization-workflow DECISIONS 28).

The loss tables are read at every commutation of the steady period by
one rule -- turn-on: blocking ``v`` before, load ``i`` after; turn-off
the reverse -- and ``ThermalSpec(mode='kernel')`` solves
``T_j = T_amb + R_th P(T_j)`` on that fixed orbit. Every table here is
SYNTHETIC (made up for the test), as are Pulsim's shipped parts."""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

import pulsim as p
from pulsim import _period_kpis as pk
from pulsim._result_views import switch_bits_by_branch
from pulsim.design import Evaluator, SteadySpec, ThermalSpec
from pulsim.loss_tables import LossTable

T = 1e-5
ALPHA = 0.004
T_AMB = 40.0


def buck(d):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "x", 100e-6)
    b.add_resistor("RLs", "x", "vout", 0.01)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 1.0)
    return b, p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                                   b.graph.num_switches)


# E = 2 uJ (1+a)(1+b)(1 + 0.4 c) on tj = [25, 125]: linear in T_j with
# slope ALPHA per K from 25 C, extrapolated beyond 125 C
_E = np.fromfunction(lambda a, b, c: 2e-6 * (1 + a) * (1 + b)
                     * (1 + ALPHA * 100 * c), (3, 3, 2))
TABLE = LossTable([10.0, 24.0, 48.0], [0.0, 10.0, 20.0], [25.0, 125.0], _E)
SPECS = {"switch_specs": {"S": {"E_on_table": TABLE, "E_off_table": TABLE,
                                "Tj": 25.0}}}


def _ev(r_th=None, **th):
    return Evaluator(buck, period=T, loads=("RL",), output="vout",
                     steady=SteadySpec(method="pss"), loss_options=SPECS,
                     thermal=None if r_th is None else ThermalSpec(
                         r_th={"S": r_th}, t_amb=T_AMB, t_limit=1e9, **th))


@pytest.fixture(scope="module")
def p25():
    """(conduction, switching at 25 C) of the switch, W."""
    r = _ev().evaluate({})
    return r.losses["S"]["conduction"], r.losses["S"]["switching"]


@pytest.mark.parametrize("r_th", [2.0, 20.0])
def test_the_kernel_fixed_point_is_the_closed_form(p25, r_th):
    # P(T) = Pc + Ps (1 + ALPHA (T - 25)):
    # T* = (T_amb + R (Pc + Ps (1 - 25 ALPHA))) / (1 - R Ps ALPHA)
    pc, ps = p25
    t_star = ((T_AMB + r_th * (pc + ps * (1 - 25 * ALPHA)))
              / (1 - r_th * ps * ALPHA))
    rec = _ev(r_th).evaluate({})
    assert rec.status == "ok", rec.reason
    assert rec.counters["thermal_mode"] == "kernel"       # 'auto' -> kernel
    assert abs(rec.t_j["S"] - t_star) < 0.01               # measured 1e-13
    assert rec.counters["thermal_residual"] < 1e-9
    # the record's loss is the tables' loss AT T_j
    assert rec.t_j["S"] == pytest.approx(
        T_AMB + r_th * sum(rec.losses["S"].values()), abs=1e-9)
    # the damped loop stops within its tol (0.1 K) of the same point
    fp = _ev(r_th, mode="fixed_point").evaluate({})
    assert fp.counters["thermal_mode"] == "fixed_point"
    assert abs(fp.t_j["S"] - rec.t_j["S"]) < 0.1


def test_runaway_is_infeasible_with_its_slope(p25):
    _, ps = p25
    rec = _ev(1.2 / (ps * ALPHA)).evaluate({})       # R_th P0 alpha = 1.2
    assert rec.status == "infeasible"
    assert rec.reason.startswith("thermal_runaway: S")
    assert "dP/dT_j = 1.2 " in rec.reason
    assert rec.violations["thermal_runaway"] == pytest.approx(0.2, rel=1e-9)
    assert rec.kpis == {}


def test_the_limit_and_the_mode_refusals():
    rec = Evaluator(buck, period=T, loads=("RL",), loss_options=SPECS,
                    steady=SteadySpec(method="pss"), thermal=ThermalSpec(
                        r_th={"S": 20.0}, t_amb=T_AMB, t_limit=50.0)
                    ).evaluate({})
    assert rec.status == "infeasible" and "thermal_limit" in rec.violations
    with pytest.raises(ValueError, match="mode='kernel'"):
        Evaluator(buck, period=T, thermal=ThermalSpec(
            r_th={"S": 1.0}, mode="kernel", loss_at=lambda d, p_, t: p_))
    # a sampled route: kernel mode refused by name, auto -> fixed point
    ev = Evaluator(buck, period=T, loads=("RL",), loss_options=SPECS,
                   steady=SteadySpec(method="transient"),
                   thermal=ThermalSpec(r_th={"S": 2.0}, mode="kernel"))
    rec = ev.evaluate({})
    assert rec.status == "refused" and "exact steady-state" in rec.reason


def test_a_linear_conduction_table_heats_the_junction(p25):
    # V_on = r i (1 + ALPHA (T - 25)): heat r i_rms^2 (1 + ALPHA (T-25))
    r = 2e-3
    tab = {"i": [0.0, 10.0], "tj": [25.0, 125.0],
           "v_on": [[0.0, 0.0], [10 * r, 10 * r * (1 + 100 * ALPHA)]]}
    plain = _ev()
    src, sf = plain._build({})
    run, _, _, b2 = plain._steady(src, sf, T, {"simulations": 0})
    h25 = r * pk.loss_rows(b2, run, {})["S"]["i_rms"] ** 2
    plain = plain.evaluate({})
    ev = Evaluator(buck, period=T, loads=("RL",), output="vout",
                   steady=SteadySpec(method="pss"), thermal=ThermalSpec(
                       r_th={"S": 5.0}, t_amb=T_AMB, conduction={"S": tab}))
    rec = ev.evaluate({})
    assert rec.status == "ok", rec.reason
    # the heat at T_j, and T_j = T_amb + R (P_cond_table)
    t = rec.t_j["S"]
    heat = rec.kpis["p_cond_table.S"]
    assert t == pytest.approx(T_AMB + 5.0 * heat, abs=1e-9)
    assert heat == pytest.approx(h25 * (1 + ALPHA * (t - 25)), rel=1e-6)
    # the circuit's own conduction loss stays in the record (balance)
    assert rec.losses["S"]["conduction"] == pytest.approx(
        plain.losses["S"]["conduction"], rel=1e-12)
    # three current knots: not a period integral, refused by name
    bad = dict(tab, i=[0.0, 5.0, 10.0], v_on=[[0, 0], [5 * r, 5 * r],
                                              [10 * r, 10 * r]])
    ev = Evaluator(buck, period=T, loads=("RL",),
                   steady=SteadySpec(method="pss"), thermal=ThermalSpec(
                       r_th={"S": 5.0}, conduction={"S": bad}))
    rec = ev.evaluate({})
    assert rec.status == "refused" and "two current knots" in rec.reason
    with pytest.raises(ValueError, match="kernel fixed point only"):
        Evaluator(buck, period=T, thermal=ThermalSpec(
            r_th={"S": 1.0}, conduction={"S": tab}, mode="fixed_point"))


EXAMPLE = (pathlib.Path(__file__).resolve().parents[2] / "examples"
           / "optimization" / "buck_pareto.py")


def test_kernel_and_fixed_point_agree_on_the_buck_examples_best_design():
    """Task 7.1: on the example's best design (highest efficiency among
    the feasible of 32), both modes within 0.1 K."""
    spec = importlib.util.spec_from_file_location("bp_thermal", EXAMPLE)
    bp = importlib.util.module_from_spec(spec)
    sys.modules["bp_thermal"] = bp
    try:
        spec.loader.exec_module(bp)
        ds = bp.latin_hypercube(32)
        best = None
        for name, ev in bp.make_evaluator().items():
            for r in ev.evaluate_many([d for d in ds if d["part"] == name]):
                if r.status == "ok" and (best is None or r.kpis["efficiency"]
                                         > best.kpis["efficiency"]):
                    best = r
        r_jc = float(bp.PARTS[best.design["part"]].params["R_th_jc"])
        t = {}
        for mode in ("kernel", "fixed_point"):
            th = ThermalSpec(r_th={"Q1": r_jc + 2.0, "Q2": r_jc + 2.0},
                             t_amb=bp.T_AMB, mode=mode)
            ev = bp.make_evaluator(thermal=th)[best.design["part"]]
            rec = ev.evaluate(best.design)
            assert rec.status == "ok", rec.reason
            assert rec.counters["thermal_mode"] == mode
            t[mode] = rec.t_j
        for d in ("Q1", "Q2"):
            assert abs(t["kernel"][d] - t["fixed_point"][d]) < 0.1
        assert t["kernel"]["Q1"] > bp.T_AMB + 1.0      # the tables moved
    finally:
        sys.modules.pop("bp_thermal", None)


def _vsi(d):
    """10 kHz SPWM, 1 us dead time, 600 V, star R-L load whose neutral
    returns through 1 uF || 1 kOhm (a slow common-mode path)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", 600.0)
    v = p.add_three_phase_vsi(b, "INV", vdc_pos="dcp", vdc_neg="gnd",
                              out_a="a", out_b="b", out_c="c")
    for ph in "abc":
        b.add_resistor(f"R{ph}", ph, f"m{ph}", 10.0)
        b.add_inductor(f"L{ph}", f"m{ph}", "n", 5e-3)
    b.add_capacitor("Cn", "n", "gnd", 1e-6)
    b.add_resistor("Rn", "n", "gnd", 1e3)
    fn = p.make_three_phase_spwm_fn(
        10e3, 50.0, 0.8, p.ThreePhaseLegIndices(*v.switch_indices),
        b.graph.num_switches, 1e-6)
    return b, fn


def test_a_three_phase_inverter_at_125C_equals_device_loss_summary():
    """Fixed T_j = 125 C. Kernel E_on / E_off of every switch equal
    device_loss_summary's on a settled dsed period within 1e-9 (measured
    1.8e-13); every diode's recovery count equals its turn-off count."""
    tp = 0.02
    v, i, tj = [100.0, 600.0, 800.0], [0.0, 20.0, 60.0], [25.0, 125.0, 150.0]
    e = np.fromfunction(lambda a, b, c: 1e-4 * (1 + a) * (1 + 0.5 * b)
                        * (1 + 0.2 * c), (3, 3, 3))
    on, off, rec_t = (LossTable(v, i, tj, e), LossTable(v, i, tj, 0.8 * e),
                      LossTable(v, i, tj, 0.3 * e))
    b, _ = _vsi({})
    kinds = {c["name"]: c["kind"] for c in b.components()}
    sw = [n for n, k in kinds.items() if k == "switch"]
    di = [n for n, k in kinds.items() if k == "diode"]
    specs = {n: {"E_on_table": on, "E_off_table": off, "Tj": 125.0}
             for n in sw}
    specs.update({n: {"E_off_table": rec_t, "Tj": 125.0} for n in di})
    ev = Evaluator(_vsi, period=tp, loads=("Ra", "Rb", "Rc"),
                   steady=SteadySpec(method="pss"),
                   loss_options={"switch_specs": specs})
    rec = ev.evaluate({})
    assert rec.status == "ok", rec.reason
    assert all(rec.losses[n]["recovery"] > 0 for n in di)
    # the kernel period behind the record, and its loss rows
    src, sf = ev._build({})
    run, _, _, b2 = ev._steady(src, sf, tp, {"simulations": 0})
    rows = pk.loss_rows(b2, run, ev.loss_options)
    for n in sw + di:
        cat = "switching" if n in sw else "recovery"
        assert rows[n]["P_sw_avg"] == rec.losses[n][cat]
    # a settled sampled period, from the orbit, read off the edges
    pss = p.periodic_steady_state(b2, period=tp, switch_fn=sf, rtol=1e-12)
    res = p.simulate(b2, t_end=3 * tp, switch_fn=sf, engine="dsed",
                     initial_state=pss.snapshot, rtol=1e-11, atol=1e-14)
    w0 = tp + 0.0023456            # not on an edge
    dls = {x["name"]: x for x in p.device_loss_summary(
        b2, res, switch_specs={n: specs[n] for n in sw}, t_from=w0,
        t_to=w0 + tp) if "name" in x}
    for n in sw:
        for k in ("E_sw_on_total", "E_sw_off_total"):
            assert rows[n][k] == pytest.approx(dls[n][k], rel=1e-9)
        assert rows[n]["n_turn_on_events"] == dls[n]["n_turn_on_events"]
        assert rows[n]["n_turn_off_events"] == dls[n]["n_turn_off_events"]
    tt = np.asarray(res.times)
    ss = np.asarray(res.switch_states).astype(bool)
    bits = switch_bits_by_branch(b2)
    bid = {c["name"]: c["branch_id"] for c in b2.components()}
    for n in di:
        col = ss[:, bits[bid[n]]]
        k = np.flatnonzero(col[:-1] & ~col[1:]) + 1
        n_off = int(np.count_nonzero((tt[k] > w0) & (tt[k] <= w0 + tp)))
        assert n_off > 100
        assert rows[n]["n_turn_off_events"] == n_off
    # the sampled summary refuses a diode's table by name (it would
    # have ignored it)
    with pytest.raises(ValueError, match="names diode"):
        p.device_loss_summary(b2, res, switch_specs={di[0]: specs[di[0]]})


def test_a_diode_recovery_table_runs_in_the_kernel_batch():
    rec_t = LossTable([10.0, 24.0], [0.0, 20.0], [25.0, 125.0],
                      np.full((2, 2, 2), 1e-6))
    from .test_design_circuit_form import DESIGNS, _C, _SF
    ev = Evaluator(circuit=_C, fields={"L": "L.L", "C": "C.C"},
                   switch_fn=_SF, period=T, loads=("RL",), output="vout",
                   steady=SteadySpec(method="pss"),
                   loss_options={"switch_specs": {"D1": {
                       "E_off_table": rec_t, "Tj": 90.0}}})
    recs = ev.evaluate_many(DESIGNS, workers=2)
    assert ev.last_batch["kernel"] == len(DESIGNS)
    assert recs == [ev.evaluate(d) for d in DESIGNS]
    assert all(r.losses["D1"]["recovery"] > 0 for r in recs)
    with pytest.raises(ValueError, match="E_off_table"):
        Evaluator(circuit=_C, fields={"L": "L.L", "C": "C.C"},
                  switch_fn=_SF, period=T, loads=("RL",),
                  steady=SteadySpec(method="pss"),
                  loss_options={"switch_specs": {"D1": {
                      "E_on_table": rec_t, "Tj": 90.0}}}).evaluate(DESIGNS[0])
