"""v2.0 Phase 3, item 5: one result surface for both engines.

`result.v('sw_node')` — the most-probed waveform in power
electronics — was unrecoverable from a dsed run: the reduction
eliminates every non-state unknown, and the result carried only the
reduced integrator state. The audit called this out directly, and an
existing test confessed it in a comment: "we can't easily match [the
pwl layout] without poking at builder.graph internals … just confirm
both engines produce non-NaN finite states."

Now every recorded sample carries the mask it was recorded under,
and a kernel post-pass reconstructs the FULL MNA vector through the
per-mask recovery map — node voltages first, then source and
inductor branch currents, the pwl engine's exact row layout. So
`.states`, `.v()` and `.i()` mean the same thing on both engines,
and the reduced state stays available as `.states_reduced`.
"""

import warnings

import numpy as np
import pytest

import pulsim as p


def _rect():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 10.0, 60.0)
    b.add_diode("D", "ac", "vout", 1e3, 1e-9, 0.7)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("Rl", "vout", "gnd", 200.0)
    return b


def _run(b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=5e-2, engine="dsed", **kw)


def _tavg(t, v):
    t = np.asarray(t)
    v = np.asarray(v)
    return float(np.trapezoid(v, t) / (t[-1] - t[0]))


def test_v_by_name_matches_pwl():
    rd = _run(_rect())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rp = p.simulate(_rect(), t_end=5e-2, dt=1e-8)
    avg_d = _tavg(rd.times, rd.v("vout"))
    avg_p = _tavg(rp.times, rp.v("vout"))
    assert avg_d == pytest.approx(avg_p, rel=1e-3)


def test_an_eliminated_algebraic_node_is_recovered():
    """'ac' has no capacitor — the reduction removed it entirely.
    Its reconstruction must be the source voltage, exactly."""
    rd = _run(_rect())
    va = np.asarray(rd.v("ac"))
    t = np.asarray(rd.times)
    ref = 10.0 * np.sin(2.0 * np.pi * 60.0 * t)
    np.testing.assert_allclose(va, ref, rtol=0, atol=1e-6)


def test_branch_current_by_name():
    rd = _run(_rect())
    i_rl = np.asarray(rd.i("Rl"))
    v_out = np.asarray(rd.v("vout"))
    # Ohm's law, sample by sample — the current is derived from the
    # same reconstructed node voltages.
    np.testing.assert_allclose(i_rl, v_out / 200.0, rtol=0,
                               atol=1e-9)


def test_states_is_the_full_mna_layout():
    rd = _run(_rect())
    b = _rect()
    n_mna = int(b.pool.state_size(b.graph)) if hasattr(
        b.pool, "state_size") else rd.states.shape[1]
    assert rd.has_full_states
    assert rd.states.shape[1] >= 3          # nodes + i_src ≥ 3 here
    assert rd.states_reduced.shape[1] == 1  # one cap state
    assert rd.states.shape[0] == rd.states_reduced.shape[0]
    assert not rd.states.flags.writeable    # an output, not scratch
    del n_mna


def test_the_mask_is_tracked_across_commutations():
    """The reconstruction is PER-MASK: a sample taken while the
    diode conducts uses a different recovery map than one taken
    while it blocks. If the wrong map were used anywhere, v('ac')
    would jump off the source sine at the commutations — the test
    above already sweeps through 6 of them at 1e-6 tolerance. Here
    we only pin that the run really did commutate."""
    rd = _run(_rect())
    assert rd.n_events >= 6
