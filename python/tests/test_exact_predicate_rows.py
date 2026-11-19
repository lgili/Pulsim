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
        s_row, n, _, _ = ad._node_pair_bound_terms(fr, to, 1e-3, x)
        s_old = (ad._node_pair_bound_terms(fr, -1, 1e-3, x)[0]
                 + ad._node_pair_bound_terms(to, -1, 1e-3, x)[0])
        v_row = ad.node_pair_voltage(fr, to, 1e-3, x)
        v_old = ad.node_pair_voltage(fr, -1, 1e-3, x) - ad.node_pair_voltage(to, -1, 1e-3, x)
        # the same quantity, within the old reading's own rounding bound
        assert abs(v_row - v_old) <= n * u * s_old
        # the row's terms: an ON diode's voltage is its current over g_on
        assert s_row < 1e-3 * s_old
        assert s_row >= abs(v_row)
