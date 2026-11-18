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
    assert got.backend == "thread" and got.failed == []
    np.testing.assert_array_equal(got.params["L"], ref.params["L"])
    for k in ("v", "i", "n"):
        np.testing.assert_array_equal(got.kpis[k], ref.kpis[k])
    assert c.value_hash() == h


def test_process_backend_is_refused_for_a_circuit():
    with pytest.raises(ValueError, match="backend='process' cannot carry a pulsim.Circuit"):
        p.sweep(parametric(), params={"L": [1e-4]}, kpi_fn=_kpi,
                parallel=True, backend="process", verbose=False, **_kw())


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
