"""The kernel period map and its integrals (add-design-optimization-
workflow, kernel item 3; core/include/pulsim/dsed/period_integrals.hpp).

Checked: the kernel's closed-form integrals and extrema equal the Python
closed forms of `pulsim.sensitivity` on the same orbit within 1e-12;
the evaluator takes the kernel path with no Python call in the period
runs and every period run outside the GIL (counters, no wall clock);
the loss rows agree with the sampled `device_loss_summary` route; the
switch edge at the section is counted; 1 and 8 workers give identical
records."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import pulsim as p
from pulsim import _period_kpis as pk

from .test_design_evaluation import CCM, T, _ev, buck

EXAMPLE = (Path(__file__).resolve().parents[2] / "examples" / "optimization"
           / "buck_pareto.py")


def _dcm():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 50.0)
    sf = p.make_pwm_switch_fn(1e5, 0.3, b.switch_index_of("S"),
                              b.graph.num_switches)
    return b, sf


@pytest.mark.parametrize("case", ["ccm", "dcm"])
def test_kernel_integrals_equal_the_python_closed_forms(case):
    b, sf = buck(CCM) if case == "ccm" else _dcm()
    spec = pk.spec_for(b, "vout", {})
    assert isinstance(spec, pk.Spec), spec
    r = p.periodic_steady_state(b, period=T, switch_fn=sf,
                                sensitivities=["L.L"], _observables=spec)
    d = r._period
    assert d["integrals_ok"], d["integrals_reason"]
    assert d["python_callbacks"] == 0 and d["nogil"]
    kp = pk.Period(spec, d, T)
    for sig, key in (("vout", ("node", "vout")), ("L", ("i", "L"))):
        scale = max(abs(r.kpi(sig, "peak").value),
                    abs(r.kpi(sig, "valley").value))
        want = {"avg": kp.mean(key), "rms": kp.rms(key),
                "peak": kp.st[kp._j(key), 2], "valley": kp.st[kp._j(key), 4]}
        for kind, got in want.items():
            ref = r.kpi(sig, kind).value
            # measured <= 3e-15 of the signal's scale
            assert abs(got - ref) <= 1e-12 * scale, (case, sig, kind, got, ref)
    # M: the kernel's product of step propagators and saltations against
    # a central difference of the map itself. Measured: CCM 6e-16 from
    # the Python segment product; DCM 2.9e-11 from the central
    # difference, where the Python product is 5.0e-10 off.
    s = r._ctx
    F = np.empty((s.n, s.n))
    for j in range(s.n):
        e = np.zeros(s.n)
        e[j] = 1e-4 * max(1.0, abs(r.x0[j]))
        F[:, j] = (s.run(r.x0 + e).x_end - s.run(r.x0 - e).x_end) / (2 * e[j])
    assert np.abs(r.monodromy - F).max() <= 1e-9 * np.abs(F).max()
    # the independent path: simulate (event projection on) from x0
    # returns to x0 (the kernel verification run is the Newton map)
    x_back = np.asarray(r.orbit.states_reduced[-1], dtype=float)
    scale = np.abs(np.asarray(r.orbit.states_reduced)).max(axis=0)
    assert np.all(np.abs(x_back - r.x0) <= 1e-8 * (scale + 1e-30))


def test_evaluator_kernel_path_counters_and_kpis(monkeypatch):
    rec = _ev().evaluate(CCM)
    assert rec.status == "ok", rec.reason
    c = rec.counters
    assert c["kpi_path"] == "kernel"
    assert c["python_callbacks"] == 0
    # every period run (Newton iterates + verification) left the GIL
    assert c["nogil_period_runs"] == c["pss_period_runs"] > 0
    # the waveform is never simulated without a user KPI
    assert c["simulations"] == c["pss_period_runs"]
    # against the sampled route on the same orbit (2000 samples/period:
    # the trapezoid's O(h^2) is the difference)
    monkeypatch.setattr(pk, "spec_for", lambda *a: "forced sampled (test)")
    rs = _ev().evaluate(CCM)
    assert rs.counters["kpi_path"] == "sampled: forced sampled (test)"
    for k, v in rec.kpis.items():
        assert v == pytest.approx(rs.kpis[k], rel=1e-6, abs=1e-9), k


def test_a_declined_option_takes_the_sampled_route():
    ev = _ev(loss_options={"conduction_specs": {"D1": {"V_f0": 0.5,
                                                       "r_on": 0.01}}})
    rec = ev.evaluate(CCM)
    assert rec.status == "ok", rec.reason
    assert rec.counters["kpi_path"].startswith("sampled: loss options")


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location("buck_pareto", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["buck_pareto"] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("buck_pareto", None)


def test_the_section_edge_is_switched_and_both_routes_agree(ex, monkeypatch):
    """The PWM turn-on sits AT the section t = 0. A one-period sampled
    run starts in the on state and never saw it (E_on was missing); the
    kernel reads it from x(T-) -> x(0+), the sampled route from a
    two-period run."""
    d = dict(f_sw=200e3, L=10e-6, C=100e-6, part="SYN-A")
    ev = ex.make_evaluator()["SYN-A"]
    rec = ev.evaluate(d)
    assert rec.counters["kpi_path"] == "kernel"
    b, sf = ex.build(d)
    spec = pk.spec_for(b, "vout", dict(ev.loss_options))
    r = p.periodic_steady_state(b, period=1 / d["f_sw"], switch_fn=sf,
                                _observables=spec)
    rows = pk.loss_rows(b, pk.Period(spec, r._period, 1 / d["f_sw"]),
                        dict(ev.loss_options))
    q1 = rows["Q1"]
    assert (q1["n_turn_on_events"], q1["n_turn_off_events"]) == (1, 1)
    assert q1["E_sw_on_total"] > 0 and q1["E_sw_off_total"] > 0
    monkeypatch.setattr(pk, "spec_for", lambda *a: "forced sampled (test)")
    rs = ev.evaluate(d)
    sw_k = rec.losses["Q1"]["switching"]
    sw_s = rs.losses["Q1"]["switching"]
    assert sw_k == pytest.approx(sw_s, rel=1e-6)
    assert sw_k == pytest.approx(q1["P_sw_avg"], rel=1e-12)
    for k in ("efficiency", "p_loss", "v_out", "ripple_v", "volume"):
        assert rec.kpis[k] == pytest.approx(rs.kpis[k], rel=1e-6), k


def test_one_and_eight_workers_give_identical_records(ex):
    designs = ex.latin_hypercube(16, seed=3)
    a = ex.evaluate_all(designs, workers=1)
    b = ex.evaluate_all(designs, workers=8)
    assert [r.status for r in a] == [r.status for r in b]
    for ra, rb in zip(a, b):
        assert ra.kpis == rb.kpis and ra.losses == rb.losses
        assert ra.counters.get("kpi_path") == rb.counters.get("kpi_path")
    assert any(r.counters.get("kpi_path") == "kernel" for r in a)


def test_user_kpi_reads_a_lazily_simulated_period():
    from pulsim.design import Kpi
    seen = []

    def k(per):
        seen.append(per.t_to - per.t_from)
        return per.average(per.result.v("vout"))
    rec = _ev(kpis=[Kpi("v_mean", k)]).evaluate(CCM)
    assert rec.counters["kpi_path"] == "kernel"
    assert rec.kpis["v_mean"] == pytest.approx(rec.kpis["v_out"], rel=1e-9)
    assert seen == [pytest.approx(T)]          # the second of two periods
    # the waveform run is counted and checked like any other
    assert rec.counters["simulations"] == rec.counters["pss_period_runs"] + 1


def test_kernel_map_refuses_time_varying_sources():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 1.0, 1e3, 0.0)
    b.add_resistor("R", "a", "n", 1.0)
    b.add_capacitor("C", "n", "gnd", 1e-6)
    from pulsim import _pulsim
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(1e-3)
    ad = _pulsim._NativeCircuitBuilderAdapter(b, cache, None)
    sf = lambda t: p.SwitchStateMask(0)  # noqa: E731
    with pytest.raises(ValueError, match="time-varying sources"):
        _pulsim.dsed_period_map(ad, sf, np.zeros(1), 1e-3)



def test_eight_threads_on_one_evaluator_give_the_serial_records():
    """dsed_period_map releases the GIL: 8 threads x 200 different
    designs through ONE shared Evaluator equal the serial records
    exactly (batch 6c stress check; no static/thread_local state in
    period_integrals.hpp, the native PWM is read through const calls)."""
    from concurrent.futures import ThreadPoolExecutor
    rng = np.random.default_rng(6)
    designs = [dict(CCM, L=float(rng.uniform(20e-6, 200e-6)),
                    C=float(rng.uniform(10e-6, 100e-6)),
                    R=float(rng.uniform(0.5, 50.0)),
                    D=float(rng.uniform(0.2, 0.8))) for _ in range(1600)]
    ev = _ev()
    serial = [ev.evaluate(d) for d in designs]
    with ThreadPoolExecutor(8) as pool:
        par = list(pool.map(ev.evaluate, designs, chunksize=200))
    assert sum(r.counters.get("kpi_path") == "kernel" for r in serial) > 1000
    for a, b in zip(serial, par):
        assert (a.status, a.design_id, a.reason) == (b.status, b.design_id,
                                                     b.reason)
        assert a.kpis == b.kpis and a.losses == b.losses
        assert a.counters == b.counters
