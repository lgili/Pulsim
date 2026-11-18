"""``compute_dc_op(strategy='pseudo_trans')`` converges on MNA circuits.

The pseudo-transient rung added (1/dt)·I to EVERY row of the Jacobian.
On a voltage source's branch row that is a negative resistance in the
loop, not a capacitor, and the iterate was repelled: a 10 V,
1 kΩ / 2 kΩ / 3 kΩ divider ended at ||F|| = 1.446e84 after 500
iterations. The shift now sits on the node (KCL) rows only — a 1 F
capacitor from each node to ground — and the branch rows stay exact.
"""
from __future__ import annotations

import pytest

import pulsim as p


@pytest.mark.parametrize("with_cap", [False, True])
def test_pseudo_transient_solves_a_divider(with_cap):
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "n1", "gnd", 10.0)
    b.add_resistor("R1", "n1", "n2", 1000.0)
    b.add_resistor("R2", "n2", "n3", 2000.0)
    b.add_resistor("R3", "n3", "gnd", 3000.0)
    if with_cap:
        b.add_capacitor("C1", "n2", "gnd", 1e-6)
    x = p.compute_dc_op(b, strategy="pseudo_trans")
    assert x[b.node_id_of("n2")] == pytest.approx(10.0 * 5.0 / 6.0, rel=1e-9)
    assert x[b.node_id_of("n3")] == pytest.approx(5.0, rel=1e-9)


def test_pseudo_transient_solves_a_diode_load():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "a", "gnd", 5.0)
    b.add_resistor("R1", "a", "k", 1000.0)
    b.add_shockley_diode("D1", "k", "gnd")
    x_pt = p.compute_dc_op(b, strategy="pseudo_trans")
    x_ref = p.compute_dc_op(b, strategy="auto")
    assert x_pt[b.node_id_of("k")] == pytest.approx(x_ref[b.node_id_of("k")], abs=1e-6)
    assert 0.3 < x_pt[b.node_id_of("k")] < 0.9
