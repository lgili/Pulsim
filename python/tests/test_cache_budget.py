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
