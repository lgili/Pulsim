"""SPICE current-source sign.

SPICE ``Iname n+ n- value`` drives ``value`` from ``n+`` THROUGH the source
to ``n-``, so it leaves the source at ``n-``: ``I1 0 a DC 1`` with
``R1 a 0 1`` gives v(a) = +1 V (ngspice agrees). The kernel's
``add_current_source(name, from, to, I)`` pushes ``I`` out of ``from``
into the circuit, so the importer must pass the SPICE nodes swapped.
Measured before the fix: v(a) = -1.000 V.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulsim as p


def _v_a(net: str) -> float:
    b = p.spice_to_builder(net)
    res = p.simulate(b, 1e-6, 1e-7, engine="pwl")
    return float(np.asarray(res.v("a"))[-1])


def test_current_into_node_a_gives_positive_voltage():
    assert _v_a("* sign\nI1 0 a DC 1\nR1 a 0 1\n.end\n") == pytest.approx(1.0, rel=1e-9)


def test_current_out_of_node_a_gives_negative_voltage():
    assert _v_a("* sign\nI1 a 0 DC 2\nR1 a 0 1\n.end\n") == pytest.approx(-2.0, rel=1e-9)
