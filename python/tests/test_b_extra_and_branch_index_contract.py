"""Two contracts the examples audit found misread, pinned so they cannot
drift silently (both are documented, not defects):

* ``b_extra`` adds to the residual F = J·x + b. On a voltage source's
  row an entry e gives v(+) − v(−) = V − e; three induction-motor
  scripts wrote +v and drove the machine with −v. On a node row it draws
  a current e out of the node.
* ``pool.branch_var_id_for_*`` returns ``num_nodes + k``: a node added
  afterwards moves it. The MMC dq script cached (26, 28, 30) before the
  build finished; the real slots were (28, 29, 30).
"""
from __future__ import annotations

import pytest

import pulsim as p


def test_b_extra_is_a_residual_addend():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 0.0)
    b.add_resistor("Ra", "a", "gnd", 1.0)
    b.add_resistor("Rn", "n", "gnd", 1.0)
    n = b.pool.state_size(b.graph)
    ks = b.pool.branch_var_id_for_source(b.branch_id_of("V"), b.graph)
    kn = b.node_id_of("n")

    def bx(_t):
        out = [0.0] * n
        out[ks] = 10.0
        out[kn] = 2.0
        return out

    r = p.simulate(b, t_end=1e-5, dt=1e-6, b_extra_fn=bx)
    assert r.v("a")[-1] == pytest.approx(-10.0, abs=1e-9)
    assert r.v("n")[-1] == pytest.approx(-2.0, abs=1e-9)


def test_branch_index_moves_when_a_node_is_added():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "a", "gnd", 1.0)
    b.add_inductor("L1", "a", "x", 1e-3)
    b.add_resistor("R1", "x", "gnd", 1.0)
    lid = b.branch_id_of("L1")
    early = b.pool.branch_var_id_for_inductor(lid, b.graph)
    assert early == b.graph.num_nodes + 1  # after the source's current
    b.add_resistor("R2", "y", "gnd", 1.0)  # one more node
    late = b.pool.branch_var_id_for_inductor(lid, b.graph)
    assert late == early + 1
    r = p.simulate(b, t_end=20e-3, dt=1e-5)
    assert r.states[-1][late] == pytest.approx(r.i("L1")[-1], rel=1e-9)
    assert r.i("L1")[-1] == pytest.approx(1.0, rel=1e-3)
