"""Task 9.3 of add-compiled-control-graph: C blocks on trbdf2 and on the
exact strategy.

trbdf2 lands its steps on the block's tick instants (the control graph's
variable-step plan), so a C block there samples v(kT) and holds from kT:
the state-time law, v[10] = 0.643545722992 for the P loop of
docs/control-graph.md (pwl runs the label law, w_10(h)). The exact
strategy (engine="exact" / "dsed") has no b_extra input and no observer:
it refuses a C block by name.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

TS, TAU = 1e-4, 1e-3


def _v_state(k):
    w = 0.0
    a = math.exp(-TS / TAU)
    for _ in range(k):
        w = a * w + (1.0 - a) * 2.0 * (1.0 - w)
    return w


def _rc_with_p_block(**kw):
    b = p.CircuitBuilder()
    b.add_resistor("R", "u", "v", 1e3)
    b.add_capacitor("C", "v", "gnd", 1e-6)

    def step(t, dt, inp, out, state):
        out[0] = 2.0 * (1.0 - inp[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        h = p.add_c_block(b, inputs=[("v", "v")],
                          outputs=[("v", "u", "gnd")], dt=TS, fn=step, **kw)
    return b, h


def test_state_time_closed_form():
    assert _v_state(10) == pytest.approx(0.643545722992, abs=1e-12)


def test_c_block_on_trbdf2_runs_the_state_time_law():
    b, h = _rc_with_p_block()
    r = p.simulate(b, 11 * TS, engine="trbdf2", rtol=1e-9, atol=1e-12)
    t, v = np.asarray(r.times), np.asarray(r.v("v"))
    # just before the tick at 10 T the capacitor holds v[10]
    for k in (1, 5, 10):
        i = int(np.searchsorted(t, k * TS - 1e-12)) - 1
        vk = np.interp(k * TS, t[i:i + 2], v[i:i + 2])
        assert abs(vk - _v_state(k)) < 1e-7, (k, vk, _v_state(k))
    assert h.n_fires == 11


def test_c_block_on_trbdf2_agrees_with_the_fixed_engine():
    # pwl samples v(kT - h) (label time); the two meet as h -> 0
    b1, _ = _rc_with_p_block()
    b2, _ = _rc_with_p_block()
    r1 = p.simulate(b1, 2e-3, dt=1e-7)
    r2 = p.simulate(b2, 2e-3, engine="trbdf2")
    v1 = np.interp(2e-3 - 5e-5, r1.times, r1.v("v"))
    v2 = np.interp(2e-3 - 5e-5, r2.times, r2.v("v"))
    assert abs(v1 - v2) < 1e-4
    assert abs(v2 - 2.0 / 3.0) < 2e-3


@pytest.mark.parametrize("engine", ("exact", "dsed"))
def test_c_block_on_the_exact_strategy_is_refused_by_name(engine):
    b, _ = _rc_with_p_block()
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, 1e-3, engine=engine)
    assert ei.value.feature == "c_blocks"
    assert "C block" in str(ei.value) or "c_blocks" in str(ei.value)
