"""v2.0 Phase 1: lazy segment cache LRU byte budget (Python surface).

The C++ suite covers eviction mechanics; here we pin the Python
bindings: budget getter/setter, byte estimate, and the event-entry
counter.
"""

import pulsim


def _four_switch_builder():
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    for i in range(4):
        b.add_switch(f"S{i}", "vin", "vout", 1e-3, 1e9)  # R_on, R_off
    b.add_resistor("Rload", "vout", "gnd", 10.0)
    return b


def test_budget_default_and_setter():
    b = _four_switch_builder()
    cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
    assert cache.segment_budget_bytes() == 1 << 30  # 1 GiB default
    cache.set_segment_budget_bytes(0)
    assert cache.segment_budget_bytes() == 0
    cache.set_segment_budget_bytes(4096)
    assert cache.segment_budget_bytes() == 4096


def test_cache_bytes_grows_with_builds():
    b = _four_switch_builder()
    cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(0.0)
    assert cache.segment_cache_bytes() == 0
    # Eager build factorises all 2^4 masks.
    cache.build(0.0)
    assert cache.num_built_segments() == 16
    assert cache.segment_cache_bytes() > 0


def test_num_event_entries_starts_empty():
    b = _four_switch_builder()
    cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(0.0)
    assert cache.num_event_entries() == 0


def test_eviction_is_observable_and_correct_from_python():
    """Adversarial-review finding F10: the Python tests were
    getter-only — no eviction, no solve — and CacheMetrics was not
    bound at all, so the entire LRU could have been deleted with the
    Python suite still green. Drive a real run under a tiny budget
    and check both the telemetry and the waveform."""
    import numpy as np

    def run(budget_bytes):
        b = _four_switch_builder()
        cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
        cache.build_lazy(1e-6)
        opts = pulsim.SimulationOptions(t_start=0.0, t_end=2e-4, dt=1e-6)

        def sw(t):
            m = pulsim.SwitchStateMask(4)
            k = int(t / 1e-6)
            for bit in range(4):
                m.set(bit, ((k >> bit) & 1) != 0)
            return m

        if budget_bytes is not None:
            # Prime one segment so the budget can be sized from a
            # real measurement, then keep room for ~2.
            pulsim.run_transient(cache, b.graph, b.pool,
                                 pulsim.SimulationOptions(
                                     t_start=0.0, t_end=1e-6, dt=1e-6), sw)
            cache.set_segment_budget_bytes(
                2 * cache.segment_cache_bytes())
        res = pulsim.run_transient(cache, b.graph, b.pool, opts, sw)
        return cache, res

    bounded, res_bounded = run(budget_bytes=True)
    free, res_free = run(budget_bytes=None)

    # Telemetry is visible from Python now, and eviction really ran.
    assert bounded.metrics().segment_evictions > 0
    assert free.metrics().segment_evictions == 0
    assert bounded.num_built_segments() < free.num_built_segments()
    assert "evictions=" in repr(bounded.metrics())

    # ...and the bounded run is numerically identical to the free one.
    np.testing.assert_allclose(
        np.asarray(res_bounded.states),
        np.asarray(res_free.states), rtol=0, atol=1e-12)


def test_event_solver_metrics_are_visible():
    b = _four_switch_builder()
    cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(1e-6)
    assert cache.metrics().event_builds == 0
    assert cache.metrics().event_hits == 0


def test_result_total_bytes_covers_more_than_states():
    b = _four_switch_builder()
    res = pulsim.simulate(b, t_end=1e-4, dt=1e-6)
    assert res.total_bytes > res.states_bytes   # + times + diagnostics


def test_store_every_via_solver_options_bundle():
    # The SolverOptions bundle must be equivalent to the flat kwarg
    # (review finding store-every-unreachable-from-every-public-
    # entry-point).
    b = _four_switch_builder()
    flat = pulsim.simulate(b, t_end=1e-4, dt=1e-6, store_every=10)
    bundled = pulsim.simulate(
        b, t_end=1e-4, dt=1e-6,
        solver=pulsim.SolverOptions(store_every=10))
    assert flat.num_steps() == bundled.num_steps()
    assert bundled.num_steps() == 11
