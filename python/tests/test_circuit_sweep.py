"""add-immutable-circuit-ir groups 5 and 10: sweeps over one Circuit.

``sweep(circuit, params=...)`` and ``monte_carlo(circuit, ...)`` must
equal the factory form bit for bit, and every point must share the
circuit's symbolic analysis: ONE analysis for a 100-point sweep, where
the factory form does one per point (counters decide; times are not
asserted).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p


def _buck(b, L, R):
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", R)
    return b


def factory(L=100e-6, R=5.0):
    return _buck(p.CircuitBuilder(), L, R)


def parametric():
    b = p.CircuitBuilder()
    return _buck(b, b.param("L", 100e-6, "H"), b.param("R", 5.0, "Ohm")).build()


def _kw():
    b = factory()
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"),
                              b.graph.num_switches)
    return dict(t_end=1e-4, dt=1e-7, engine="pwl", switch_fn=fn)


def _kpi(res, _params):
    return {"v": float(res.states[-1][-1]), "i": float(res.states[-1][0]),
            "n": float(len(res.times))}


@pytest.fixture
def caches(monkeypatch):
    """Every PWL cache simulate() builds, for the counters."""
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


VALUES = {"L": list(np.geomspace(50e-6, 500e-6, 10)), "R": [2.0, 5.0, 10.0]}


def test_sweep_over_a_circuit_equals_the_factory_form_bit_for_bit(caches):
    kw = _kw()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = p.sweep(factory, params=VALUES, kpi_fn=_kpi, verbose=False, **kw)
        n_ref = _analyses(caches)
        caches.clear()
        got = p.sweep(parametric(), params=VALUES, kpi_fn=_kpi,
                      verbose=False, **kw)
    assert ref.n_simulations == got.n_simulations == 30
    for k in ("v", "i", "n"):
        np.testing.assert_array_equal(got.kpis[k], ref.kpis[k])
    # The counter gate: one analysis per point before, one in all now.
    assert n_ref == 30
    assert _analyses(caches) == 1


def test_device_field_keys_use_with_value(caches):
    kw = _kw()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = p.sweep(factory, params={"R": [2.0, 8.0]}, kpi_fn=_kpi,
                      verbose=False, **kw)
        got = p.sweep(factory().build(), params={"RL.R": [2.0, 8.0]},
                      kpi_fn=_kpi, verbose=False, **kw)
    np.testing.assert_array_equal(got.kpis["v"], ref.kpis["v"])


def test_monte_carlo_threads_share_the_circuit():
    kw = _kw()
    dist = {"L": lambda r: r.uniform(50e-6, 500e-6),
            "R": lambda r: r.uniform(2.0, 10.0)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = p.monte_carlo(factory, distributions=dist, kpi_fn=_kpi,
                            n_samples=12, seed=7, verbose=False, **kw)
        c = parametric()
        h = c.value_hash()
        got = p.monte_carlo(c, distributions=dist, kpi_fn=_kpi,
                            n_samples=12, seed=7, parallel=True,
                            n_workers=4, verbose=False, **kw)
    assert got.backend == "kernel" and got.failed == []
    np.testing.assert_array_equal(got.params["L"], ref.params["L"])
    for k in ("v", "i", "n"):
        np.testing.assert_array_equal(got.kpis[k], ref.kpis[k])
    assert c.value_hash() == h


def test_a_circuit_of_another_structure_does_not_share(caches):
    kw = _kw()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = parametric()
        p.simulate(a, **kw)
        p.simulate(a.with_params(L=200e-6), **kw)
        assert _analyses(caches) == 1
        b = factory()
        b.add_resistor("Rx", "vout", "gnd", 1e3)   # another structure
        p.simulate(b.build(), **kw)
    assert _analyses(caches) == 2


# ---- task 5.2: the kernel path ---------------------------------------------
def _states_by_point(store):
    def kpi(res, params):
        store[tuple(params.values())] = np.asarray(res.states).copy()
        return {"n": float(len(res.times))}
    return kpi


def _per_point(circuit, field, values, kw):
    return [np.asarray(p.simulate(circuit.with_value(field, v), **kw).states)
            for v in values]


@pytest.mark.parametrize("parallel", [False, True])
def test_kernel_path_runs_no_python_per_point(monkeypatch, caches, parallel):
    """Counters: the template point is the only Python run_transient call,
    one symbolic analysis and one preflight for the whole sweep, and every
    point bit-identical to simulate(circuit.with_value(...))."""
    kw = _kw()
    c = factory().build()
    values = list(np.geomspace(50e-6, 500e-6, 12))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = _per_point(factory().build(), "L.L", values, kw)  # own plan
        caches.clear()
        calls = []
        real = p.run_transient
        monkeypatch.setattr(p, "run_transient",
                            lambda *a, **k: calls.append(1) or real(*a, **k))
        got = {}
        r = p.sweep(c, params={"L.L": values}, kpi_fn=_states_by_point(got),
                    parallel=parallel, n_workers=4, verbose=False, **kw)
    assert r.backend == "kernel" and r.failed == []
    assert len(calls) == 1                       # point 0 only
    assert r.counters["kernel_points"] == len(values) - 1
    assert _analyses(caches) + r.counters["symbolic_analyses"] == 1
    assert r.counters["preflights"] == 1
    for v, x in zip(values, ref):
        np.testing.assert_array_equal(got[(v,)], x)


def test_kernel_path_regularised_and_initial_conditions():
    """A floating node (preflight ties) and a capacitor IC: the kernel
    points equal simulate() of each point, ties and plan included."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    b.add_resistor("R", "a", "b", 10.0)
    b.add_capacitor("C", "b", "gnd", 1e-6)
    b.set_initial("C", 2.0)
    b.add_capacitor("Cf", "b", "f", 1e-6)      # f floats at DC
    b.add_capacitor("Cg", "f", "gnd", 1e-6)
    c = b.build()
    kw = dict(t_end=2e-5, dt=1e-6, engine="pwl")
    values = [5.0, 10.0, 20.0, 40.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = [p.simulate(c.with_value("R.R", v), **kw) for v in values]
        plans = {}

        def kpi(res, params):
            plans[params["R.R"]] = (np.asarray(res.states).copy(),
                                    res.plan.ties, res.circuit.value_hash())
            return {}
        r = p.sweep(c, params={"R.R": values}, kpi_fn=kpi, parallel=True,
                    n_workers=3, verbose=False, **kw)
    assert r.backend == "kernel" and r.counters["preflights"] == 1
    assert ref[0].plan.ties    # the circuit does need a tie
    for v, x in zip(values, ref):
        st, ties, h = plans[v]
        np.testing.assert_array_equal(st, np.asarray(x.states))
        assert ties == x.plan.ties and h == x.circuit.value_hash()


def test_a_structure_change_runs_the_preflight_again():
    """with_value of a branch-form R_on to 0 changes the structure: that
    point is regularised by itself (one more preflight), still equal to
    simulate()."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    b.add_ideal_switch("S", "a", "b", R_on=1e-3, g_off=1e-9)
    b.add_resistor("R", "b", "gnd", 10.0)
    c = b.build()
    m = p.SwitchStateMask(c.num_switches)
    m.set(0, True)
    kw = dict(t_end=5e-6, dt=1e-6, engine="pwl",
              switch_fn=p.make_constant_switch_fn(m))
    values = [1e-3, 0.0, 2e-3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = _per_point(c, "S.R_on", values, kw)
        got = {}
        r = p.sweep(c, params={"S.R_on": values}, kpi_fn=_states_by_point(got),
                    verbose=False, **kw)
    assert r.backend == "kernel" and r.failed == []
    assert r.counters["preflights"] == 2          # the base, then R_on = 0
    for v, x in zip(values, ref):
        np.testing.assert_array_equal(got[(v,)], x)


def test_a_python_hook_keeps_the_python_path():
    kw = _kw()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.sweep(parametric(), params={"L": [1e-4, 2e-4]}, kpi_fn=_kpi,
                    verbose=False, step_observer=lambda t, x: None, **kw)
    assert r.backend == "serial" and r.counters["kernel_points"] == 0


def test_a_kpi_that_simulates_does_not_change_the_replayed_run():
    """Review finding: the recipe of point 0 was recorded while its kpi_fn
    ran, so a simulate() inside the kpi overwrote dt / t_end for every
    kernel point."""
    kw = _kw()
    c = factory().build()
    values = [1e-4, 2e-4, 3e-4]

    def kpi(res, params):
        p.simulate(factory(), t_end=1e-6, dt=1e-8, engine="pwl")
        return {"n": float(len(res.times))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.sweep(c, params={"L.L": values}, kpi_fn=kpi, verbose=False, **kw)
    assert r.backend == "kernel"
    assert list(r.kpis["n"]) == [1001.0] * 3


def test_a_failing_point_is_recorded_as_on_the_python_path():
    kw = _kw()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.sweep(factory().build(), params={"RL.R": [5.0, -1.0, 8.0]},
                    kpi_fn=_kpi, verbose=False, **kw)
        with pytest.raises(ValueError) as exc:
            factory().build().with_value("RL.R", -1.0)
    assert r.backend == "kernel"
    assert r.failed == [(1, str(exc.value))]
    assert r.counters["kernel_points"] == 1
