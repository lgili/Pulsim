"""The Newton shooting loop in the kernel (``dsed_periodic_orbit``).

add-design-optimization-workflow DECISIONS 18: the loop of
``periodic_steady_state`` moved to ``core/include/pulsim/dsed/
periodic_orbit.hpp`` and runs with the GIL released. These tests hold it
to the Python loop it replaces (kept behind ``_python_loop=True``): the
same orbit, monodromy, multipliers, period-run and iteration counts and
period integrals on the CCM and DCM bucks and the boost of the change,
and the same typed outcomes. Counters, never wall time.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import pulsim as p
from pulsim._period_kpis import Spec, spec_for

from . import _pss_reference as ref

T = ref.T
pss_mod = sys.modules["pulsim.periodic_steady_state"]


def test_pulsim_is_imported_from_this_tree():
    root = Path(__file__).resolve().parents[2]
    assert Path(p.__file__).resolve().is_relative_to(root)
    assert Path(p._pulsim.__file__).resolve().is_relative_to(root)


def _buck(R, D):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", ref.VIN)
    b.add_switch("S", "vin", "sw", ref.GON, ref.GOFF)
    b.add_diode("D1", "gnd", "sw", ref.GON, ref.GOFF)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", R)
    return b, p.make_pwm_switch_fn(1 / T, D, b.switch_index_of("S"),
                                   b.graph.num_switches)


def _boost(R, v_th):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("S", "sw", "gnd", ref.GON, ref.GOFF)
    b.add_diode("D1", "sw", "vout", ref.GON, ref.GOFF, v_th)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", R)
    return b, p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                                   b.graph.num_switches)


def _negative_rc():
    """dv/dt = (g - 1/R) v / C + V/(R C), g = 3 mS: an unstable fixed
    point, mu = exp(2000 T)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("Ra", "a", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    b.add_vccs("G", "o", "gnd", "o", "gnd", 3e-3)
    return b, None


CASES = {
    "buck CCM": (lambda: _buck(1.0, 0.5), 11.9877289409),
    "buck DCM": (lambda: _buck(50.0, 0.3), ref.DCM_ORBIT[1]),
    "boost CCM": (lambda: _boost(20.0, 0.0), 24.0536294841),
    # 36.59440084 V on both loops (the slice brief quoted 36.5943997644,
    # 1.07e-6 V lower; test_periodic_steady_state pins 36.594401 +- 1e-5)
    "boost DCM": (lambda: _boost(500.0, 0.0), 36.5944008379),
    "boost CCM, V_th 1 mV": (lambda: _boost(20.0, 1e-3), 24.0536294841),
    "boost DCM, V_th 1 mV": (lambda: _boost(500.0, 1e-3), 36.5944008379),
}


def _solve(make, python_loop):
    b, sf = make()
    spec = spec_for(b, None, {})
    return p.periodic_steady_state(
        b, period=T, switch_fn=sf, _python_loop=python_loop,
        _observables=spec if isinstance(spec, Spec) else None)


@pytest.mark.parametrize("name", list(CASES))
def test_kernel_loop_equals_the_python_loop(name):
    make, v_out = CASES[name]
    k, q = _solve(make, False), _solve(make, True)
    assert k._ctx.loop == "kernel" and q._ctx.loop == "python"
    lab = k.state_labels
    assert k.x0[lab.index("v(vout)")] == pytest.approx(v_out, abs=1e-8)
    scale = float(np.abs(q._ctx.orbit_traj.scale).max())
    assert np.abs(k.x0 - q.x0).max() <= 1e-12 * scale
    # the same Newton path, run for run
    assert k.n_period_runs == q.n_period_runs
    assert k.n_engine_steps == q.n_engine_steps
    assert len(k.residual_history) == len(q.residual_history)
    assert k.pattern_history == q.pattern_history
    assert [e for e, _ in k.pattern] == [e for e, _ in q.pattern]
    assert np.abs(k.monodromy - q.monodromy).max() <= 1e-12
    assert np.abs(np.sort_complex(k.multipliers)
                  - np.sort_complex(q.multipliers)).max() <= 1e-12
    assert k.stable and q.stable
    # the verification period's integrals and extrema
    sk = np.asarray(k._period["stats"])
    sq = np.asarray(q._period["stats"])
    fin = np.isfinite(sq)
    assert (np.isfinite(sk) == fin).all()
    assert np.abs(sk[fin] - sq[fin]).max() <= 1e-10 * max(
        1.0, np.abs(sq[fin]).max())
    # every period ran with the GIL released, no Python callback
    c = k._ctx
    assert c.n_nogil_runs == k.n_period_runs and c.n_python_callbacks == 0


def test_the_loop_does_not_call_the_python_period_map(monkeypatch):
    """Removing the kernel loop would route every Newton period through
    ``dsed_period_map`` from Python: refuse that call and the kernel
    route must still solve."""
    def no(*a, **k):
        raise AssertionError("the Newton loop ran a period from Python")
    monkeypatch.setattr(p._pulsim, "dsed_period_map", no)
    b, sf = _buck(50.0, 0.3)
    r = p.periodic_steady_state(b, period=T, switch_fn=sf)
    assert r._ctx.loop == "kernel"
    assert r.x0[r.state_labels.index("v(vout)")] == pytest.approx(
        ref.DCM_ORBIT[1], rel=1e-9)


def test_an_unstable_orbit_is_returned_on_both_loops():
    k = _solve(_negative_rc, False)
    q = _solve(_negative_rc, True)
    mu = np.exp(2000.0 * T)
    for r in (k, q):
        assert not r.stable
        assert r.multipliers[0].real == pytest.approx(mu, rel=1e-12)
        assert r.x0[0] == pytest.approx(-0.5, rel=1e-12)  # V/(1 - g R)
        assert r.unstable_multipliers[0][2] == "fold"
    assert k.n_period_runs == q.n_period_runs


@pytest.mark.parametrize("python_loop", [False, True])
def test_typed_outcomes_are_kept(python_loop):
    b, sf = _buck(50.0, 0.3)
    with pytest.raises(p.ConvergenceError, match="max_period_runs=2.*Residual"):
        p.periodic_steady_state(b, period=T, switch_fn=sf, max_period_runs=2,
                                _python_loop=python_loop)
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_inductor("Lx", "a", "m", 1e-3)
    b.add_capacitor("Cx", "m", "gnd", 1e-6)
    with pytest.raises(p.ConvergenceError, match=r"not isolated.*(Cx.*Lx|Lx.*Cx)"):
        p.periodic_steady_state(b, period=T, _python_loop=python_loop)


def test_the_kernel_status_carries_no_orbit_when_not_converged():
    """The raw binding: with the budget spent after the first period the
    status is 'budget' and there is no verification period to read KPIs
    from (the Python wrapper raises ConvergenceError on it)."""
    b, sf = _buck(50.0, 0.3)
    s = pss_mod._Solver(b, T, sf)
    d = s._orbit(s.ad, s.sf, np.zeros(2), T, [], [], 1e-9, 0, 5)
    assert d["status"] == "budget" and d["n_period_runs"] == 1
    assert d["period"] is None


def test_loss_table_total_equals_the_numpy_sum():
    from pulsim.loss_tables import LossTable
    rng = np.random.default_rng(1)
    t = LossTable([24.0, 48.0, 80.0], [0.0, 10.0, 100.0], [25.0, 125.0],
                  rng.random((3, 3, 2)) * 1e-5)
    for _ in range(500):
        n = int(rng.integers(1, 6))
        v, i = rng.uniform(-20, 120, n), rng.uniform(-50, 200, n)
        tj = rng.uniform(0, 200, n)
        assert t.total(v, i, tj) == float(np.sum(t(v, i, tj)))
    assert np.isnan(t.total([np.nan], [1.0], [25.0]))
