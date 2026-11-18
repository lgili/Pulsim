"""update-trbdf2-factor-reuse parts 1-2: counter regression (no wall clock).

The event table is bounded by bytes, not by 8 entries: on the VSI every
mask is built once, and a tight budget evicts without changing a bit.
"""

import warnings

import numpy as np
import pulsim


def _vsi():
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", 600.0)
    v = pulsim.add_three_phase_vsi(b, "INV", vdc_pos="dcp", vdc_neg="gnd",
                                   out_a="a", out_b="b", out_c="c")
    for ph in "abc":
        b.add_resistor(f"R{ph}", ph, f"m{ph}", 10.0)
        b.add_inductor(f"L{ph}", f"m{ph}", "n", 5e-3)
    legs = pulsim.ThreePhaseLegIndices(*v.switch_indices)
    fn = pulsim.make_three_phase_spwm_fn(10e3, 50.0, 0.8, legs,
                                         b.graph.num_switches, 1e-6)
    return b, fn


def _run(monkeypatch, budget=None):
    caches = []
    orig = pulsim.PwlStateSpaceCache

    def wrap(*a, **k):
        c = orig(*a, **k)
        if budget is not None:
            c.set_segment_budget_bytes(budget)
        caches.append(c)
        return c

    monkeypatch.setitem(pulsim.__dict__, "PwlStateSpaceCache", wrap)
    b, fn = _vsi()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pulsim.simulate(b, t_end=10e-3, engine="trbdf2", switch_fn=fn)
    (cache,) = caches
    return res, cache


def test_vsi_trbdf2_every_mask_built_once(monkeypatch):
    _, cache = _run(monkeypatch)
    m = cache.metrics()
    assert m.event_evictions == 0
    # No eviction and no erase: every build is still resident, so the
    # builds are exactly the distinct masks solved.
    assert m.event_builds == cache.num_event_entries()
    assert m.event_builds > 8  # the old count cap was 8


def test_tight_budget_evicts_without_changing_the_result(monkeypatch):
    ref, _ = _run(monkeypatch)
    res, cache = _run(monkeypatch, budget=4096)
    assert cache.metrics().event_evictions > 0
    assert np.array_equal(np.asarray(res.times), np.asarray(ref.times))
    assert np.array_equal(np.asarray(res.states), np.asarray(ref.states))


def _buck_calls(R, D, nper=200):
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", R)
    n, k, T, t0 = b.graph.num_switches, b.switch_index_of("S"), 1e-5, 2e-6
    calls = [0]

    def fn(t):
        calls[0] += 1
        m = pulsim.SwitchStateMask(n)
        ph = (t - t0) / T
        m.set(k, bool(t >= t0 - 1e-13 and ph - np.floor(ph + 1e-9) < D - 1e-9))
        return m
    r = pulsim.simulate(b, t_end=t0 + nper * T, engine="trbdf2", switch_fn=fn)
    return r, calls[0] / nper


def test_opaque_gate_calls_per_period_on_the_buck():
    """switch_fn is the buck's cost (99 % of engine time). The edge
    bisection now probes the predicted edge first, and once a pulse
    width is known a step takes ceil(2h/w) samples (min 1, was 4):
    CCM 88.7 -> 38.7 calls per period, DCM 154.7 -> 74.4."""
    r, ccm = _buck_calls(1.0, 0.5)
    assert ccm < 65   # adaptive: 38.7 measured, 88.7 before
    st = r._trbdf2_stats
    assert st["n_factor_hits"] > st["n_factor_refactors"] > 0
    assert st["n_factor_builds"] >= 2
    _, dcm = _buck_calls(50.0, 0.3)
    assert dcm < 120  # 74.4 measured, 154.7 before
