"""F-Z-13: under the restart rule an in-loop recorder's right-limit rows (the
x(0+) row of an inconsistent start, an event's right side) read a capacitor's
current as the post-jump network's: the companion current of the right
limit's own solve, so KCL holds at the capacitor's nodes. Before, the row
carried the restart step's stage-quadrature current: i(Cf) = 1.109 A on the
rectifier's start row (KCL at dp off by 1.1 A), i(C) = 4.19e-4 A on the buck's
(0 is right). The integral-type feed (the quadrature, ∫i_C = C·Δv) is not
touched (test_recorder_device_readings)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

from tests import test_recorder_device_readings as T

EPS = np.finfo(float).eps
GAMMA = 1.0 - 1.0 / math.sqrt(2.0)
DT = 1e-6


def _run(circ, split):
    b, fn, t_end, kcl, caps = T._CIRCUITS[circ]()
    devs = sorted({t.lstrip("-") for ts in kcl.values() for t in ts})
    vs = sorted({f"v({n})" for n, _ in caps.values()})
    extra = {} if fn is None else {"switch_fn": fn}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end, recorders=[p.Recorder("r", [f"i({d})" for d in devs] + vs)],
                         engine="pwl", dt=DT, discontinuity_restart=True,
                         enable_substep_state_correction=split, **extra)
    return res, kcl, caps


# the right limit's solve is backward Euler of ε = 1e-3·γ·dt: its companion
# current (C/ε)·(v − u) carries an ulp of v as C·eps·|v|/ε; a node sums n terms
def _tol(cols, a, caps, scale):
    cap = sum(c / (1e-3 * GAMMA * DT) * np.max(np.abs(a[f"v({n})"])) for n, c in caps.values())
    return 8 * len(cols) * EPS * (scale + cap)


@pytest.mark.parametrize("circ,split", [("rectifier", False), ("buck", False), ("buck", True)])
def test_every_event_row_closes_kcl_at_the_capacitor(circ, split):
    res, kcl, caps = _run(circ, split)
    a = res.recorded["r"]
    assert a.event_t.size >= 5 and a.event_t[0] == 0.0     # the x(0+) row is there
    (cap,) = caps
    for node, terms in kcl.items():
        if not any(x.lstrip("-") == cap for x in terms):
            continue
        cols = [(-1.0 if x[0] == "-" else 1.0, f"i({x.lstrip('-')})") for x in terms]
        scale = max(np.max(np.abs(a[q])) for _, q in cols)
        se = np.abs(sum(sg * a.event_right[q] for sg, q in cols))
        # measured: 3.3e-12 A (rectifier), 7.7e-11 / 8.9e-11 A (buck); before 1.1 A / 4.19e-4 A
        assert se.max() <= _tol(cols, a, caps, scale), (node, a.event_t[np.argmax(se)], se.max())


def test_the_rectifier_start_row_with_the_split():
    res, kcl, caps = _run("rectifier", True)
    a = res.recorded["r"]
    assert a.event_t[0] == 0.0
    i_cf = a.event_right["i(Cf)"][0]
    i_in = a.event_right["i(D1)"][0] + a.event_right["i(D2)"][0] - a.event_right["i(RL)"][0]
    assert abs(i_cf - i_in) <= 1e-9 * abs(i_in)        # measured: 0.0046008 A both
    assert abs(i_cf - 1.109) > 1.0


def test_a_capacitor_switched_onto_a_source_reads_its_closed_form():
    # C at 2 V switched onto 10 V through R at t = 0: i_C(0+) = (V - v_C)/R
    V, R, C, VC0 = 10.0, 1e3, 1e-6, 2.0
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", V)
    b.add_resistor("R1", "in", "out", R)
    b.add_capacitor("C1", "out", "gnd", C, VC0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, 1e-4, recorders=[p.Recorder("r", ["i(C1)", "i(R1)", "v(out)"])],
                         engine="pwl", dt=DT, discontinuity_restart=True)
    a = res.recorded["r"]
    assert a.event_t.size == 1 and a.event_t[0] == 0.0
    i0 = (V - VC0) / R
    tol = 8 * EPS * (i0 + C / (1e-3 * GAMMA * DT) * VC0)
    assert abs(a.event_right["i(C1)"][0] - i0) <= tol
    assert abs(a.event_right["i(R1)"][0] - i0) <= tol
