"""Exact engine D4: a diode's predicate is read from a dedicated row.

The row of v(from) - v(to) is formed once per mode from the refined
recovery map (the difference taken before rounding), so its rounding
scale S is the size of the quantity itself, not of two node rows that
cancel against each other through 1/g_off-sized entries.
"""

import numpy as np
import pulsim as p
from pulsim import _pulsim


def _bridge():
    # one three-phase bank on a floating bus (1 MOhm to ground)
    b = p.CircuitBuilder()
    for ph, ang in (("a", 0.0), ("b", -120.0), ("c", 120.0)):
        b.add_sine_voltage_source(f"V{ph}", f"s{ph}", "gnd", 0.0, 100.0, 50.0, np.deg2rad(ang))
        b.add_inductor(f"L{ph}", f"s{ph}", f"x{ph}", 0.2e-3)
        b.add_diode(f"Dt{ph}", f"x{ph}", "p", 1e3, 1e-9, 0.0)
        b.add_diode(f"Db{ph}", "n", f"x{ph}", 1e3, 1e-9, 0.0)
    b.add_capacitor("C", "p", "n", 2e-3)
    b.add_resistor("R", "p", "n", 5.0)
    b.add_resistor("Rg", "n", "gnd", 1e6)
    return b


def test_an_on_diode_row_is_the_size_of_its_current():
    b = _bridge()
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(1e-6)
    ad = _pulsim._NativeCircuitBuilderAdapter(b, cache, None)
    names = [b.graph.branch_name(k) for k in range(b.graph.num_branches)]
    br = b.graph.branches[names.index("Dba")]
    fr, to = int(br["from_"]), int(br["to"])
    m = _pulsim.SwitchStateMask(b.graph.num_switches)
    m.set(b.switch_index_of("Dba"), True)
    m.set(b.switch_index_of("Dtb"), True)
    ad.set_mask(m)
    rng = np.random.default_rng(0)
    u = np.finfo(float).eps / 2
    for _ in range(5):
        x = rng.uniform(-10.0, 10.0, ad.n_state) * np.r_[30.0, np.ones(ad.n_state - 1)]
        s_row, n = ad._node_pair_bound_terms(fr, to, 1e-3, x)[:2]
        s_old = (ad._node_pair_bound_terms(fr, -1, 1e-3, x)[0]
                 + ad._node_pair_bound_terms(to, -1, 1e-3, x)[0])
        v_row = ad.node_pair_voltage(fr, to, 1e-3, x)
        v_old = ad.node_pair_voltage(fr, -1, 1e-3, x) - ad.node_pair_voltage(to, -1, 1e-3, x)
        # the same quantity, within the old reading's own rounding bound
        assert abs(v_row - v_old) <= n * u * s_old
        # the row's terms: an ON diode's voltage is its current over g_on
        assert s_row < 1e-3 * s_old
        assert s_row >= abs(v_row)


def _full_bridge_dc():
    import sys
    from pathlib import Path
    bench = Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path[:0] = [str(bench), str(bench / "tools")]
    import v1_reader
    return v1_reader.load(bench / "circuits/switching/isolated/full_bridge_dc.yaml")


def test_a_pair_row_bound_covers_the_assembled_conductances():
    # full_bridge_dc, Q3 Q4 D3 D4 ON, i_L = 0: v(load_n) - v(vcc) is
    # -24*2e-6/(400 + 2e-6) exactly; the row is exact for the ASSEMBLED
    # 400 + 2e-6 S and read -1.06e-15 V off it, which its own terms (S ~ 2.4e-7)
    # called significant and decided D3's mode on
    from fractions import Fraction as F
    b, _, _ = _full_bridge_dc().build()
    g = b.graph
    ix = {n["name"]: k for k, n in enumerate(g.nodes)}
    cache = p.PwlStateSpaceCache(g, b.pool)
    cache.build_lazy(1e-6)
    ad = _pulsim._NativeCircuitBuilderAdapter(b, cache, None)
    m = _pulsim.SwitchStateMask(g.num_switches)
    for q in ("Q3", "Q4", "D3", "D4"):
        m.set(b.switch_index_of(q), True)
    ad.set_mask(m)
    x = np.zeros(ad.n_state)
    exact = float(-F(24) * 2 * F(1e-6) / (F(400) + 2 * F(1e-6)))
    v = ad.node_pair_voltage(ix["load_n"], ix["vcc"], 0.0, x)
    s, n, _, _, data, _, _ = ad._node_pair_bound_terms(ix["load_n"], ix["vcc"], 0.0, x)
    u = np.finfo(float).eps / 2
    # the row's own terms do not cover it; the assembled data's adjoint term
    # u*sum_r (2k_r - 1)|w_r||G_r,:||x_full| does, with no tuned factor
    assert abs(v - exact) > n * u * s
    assert abs(v - exact) <= n * u * s + data
    # and it is a first-order bound, not a blanket: measured 45x the error it covers
    assert data < 100 * abs(v - exact)


def test_full_bridge_dc_turns_both_freewheel_diodes_off_at_the_zero():
    # the exact truth (benchmarks/tools/exact_truth.py): D3 and D4 carry the same
    # current and turn OFF together at 14.8964700582364510 us; dsed kept D3 ON
    # there (an exact-zero decision on the row's data error) and broke its law
    # 0.2 us later (DiodeModeInconsistentError, 791 samples)
    import warnings
    case = _full_bridge_dc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = case.run("dsed")
    t = np.asarray(r.times)
    s = np.asarray(r.switch_states).astype(bool)
    b, _, _ = case.build()
    k = np.flatnonzero((t > 14.9e-6) & (t < 20e-6))
    assert k.size > 0
    for d in ("D3", "D4"):
        assert not s[k, b.switch_index_of(d)].any(), d
    et = np.asarray(r.event_times)
    assert np.abs(et - 14.8964700582364510e-6).min() < 1e-15
