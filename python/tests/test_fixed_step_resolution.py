"""add-fixed-step-resolution-diagnostics: a fixed step that does not
resolve the circuit says so.

`res.resolution` (pwl only) holds a per-step truncation estimate of the
linear capacitors and inductors, normalised by each variable's running
peak, and the runs of >= 4 sign-alternating steps of any state entry
(Nyquist ringing). One `pulsim.ResolutionWarning` per run when
`lte_max > 1e-2` or a ringing record exists. Read-only: states are bit
for bit the same with `resolution_diagnostics=False`. Removing the
diagnostics fails every test here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

TAU = 1e-3
F = 100e3
T = 1 / F


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "out", 1e3)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    return b


def _sim(b, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = p.simulate(b, engine="pwl", **kw)
    return res, [x for x in w if issubclass(x.category, p.ResolutionWarning)]


def _names(b, res):
    nm = b.state_var_names()
    return {nm[r.var]: r.length for r in sorted(res.resolution.ringing,
                                                 key=lambda r: r.length)}


@pytest.mark.parametrize("ratio, lte, warns", [
    (10, 0.347, True),       # prototype 0.35
    (4, 0.178, True),        # 0.18
    (1, 2.67e-2, True),      # 2.7e-2
    (0.1, 3.27e-4, False),   # 3.3e-4
    (0.01, 3.33e-6, False),  # 3.3e-6
])
def test_rc_truncation_estimate(ratio, lte, warns):
    b = _rc()
    dt = ratio * TAU
    res, w = _sim(b, t_end=max(30 * dt, 0.02), dt=dt)
    rd = res.resolution
    assert rd.lte_max == pytest.approx(lte, rel=1e-2)
    assert b.graph.branch_name(rd.lte_branch) == "C"
    assert rd.n_branches_covered == 1
    assert len(rd.lte) == len(res.times)
    assert float(np.max(rd.lte)) == pytest.approx(rd.lte_max, rel=1e-12)
    assert len(w) == (1 if warns else 0)


def test_rc_at_ten_tau_is_flagged_once_with_the_variable_named():
    """p13: dt = 10 tau overshoots to 11.11 V and is 1.67 V off the exact
    curve; before this change, with no warning at all."""
    b = _rc()
    res, w = _sim(b, t_end=30 * 10 * TAU, dt=10 * TAU)
    v = np.asarray(res.v("out"))
    t = np.asarray(res.times)
    assert v.max() == pytest.approx(11.111111, rel=1e-6)
    assert np.max(abs(v - 10 * (1 - np.exp(-t / TAU)))) == pytest.approx(1.666, rel=1e-3)
    assert len(w) == 1
    msg = str(w[0].message)
    assert "C:" in msg and "Is(V): 29 sign-alternating" in msg and "Reduce dt" in msg
    # Ringing: V(out) 8 alternating steps, the source current 29.
    assert _names(b, res) == {"V(out)": 8, "Is(V)": 29}


def _buck(R, L, N, nper=200, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", R)
    fn = p.make_pwm_switch_fn(F, 0.4, b.switch_index_of("Q1"), b.graph.num_switches)
    res, w = _sim(b, t_end=nper * T, dt=T / N, switch_fn=fn, **kw)
    return b, res, w


@pytest.mark.parametrize("N, lte", [(20, 1.38e-4), (100, 2.74e-5), (400, 6.83e-6)])
def test_ccm_buck_is_silent(N, lte):
    _, res, w = _buck(2.0, 100e-6, N)
    assert not w
    assert res.resolution.lte_max == pytest.approx(lte, rel=2e-2)
    assert res.resolution.lte_max < 1e-3
    assert not res.resolution.ringing


def test_dcm_idle_ringing_is_flagged_without_the_restart_rule():
    """The default engine's post-event alternation in the DCM idle
    interval is real (those samples are wrong); the restart rule removes
    it and the warning with it."""
    b, res, w = _buck(50.0, 10e-6, 100)
    assert len(w) == 1
    ring = _names(b, res)
    assert ring["V(sw)"] == 52                 # prototype: a run of 49
    b, res, w = _buck(50.0, 10e-6, 100, discontinuity_restart=True)
    assert not w and not res.resolution.ringing
    assert res.resolution.lte_max < 1e-5       # 7.4e-7


def _cap_loop(dt, **kw):
    """A 1 nF node hard-switched through a 1 mOhm switch (tau_on = 1 ps,
    far below dt), discharged through 10 Ohm (10 ns)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_capacitor("C", "sw", "gnd", 1e-9)
    b.add_resistor("R", "sw", "gnd", 10.0)
    fn = p.make_pwm_switch_fn(1e6, 0.5, 0, 1)
    res, w = _sim(b, t_end=5e-6, dt=dt, switch_fn=fn, **kw)
    return b, res, w


def test_capacitor_loop_rings_without_the_restart_rule():
    b, res, w = _cap_loop(1e-9)
    assert len(w) == 1
    assert _names(b, res)["Is(V)"] == 500
    assert res.resolution.lte_max == pytest.approx(0.661, rel=1e-2)
    # With the rule this loop is silent here (the prototype's variant
    # kept a run of 517; DECISIONS.md).
    _, res, w = _cap_loop(1e-9, discontinuity_restart=True)
    assert not w and not res.resolution.ringing


def _sat(**kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "m", 1.0)
    b.add_saturable_inductor("Ls", "m", "gnd", 1e-3, 2.0, 1e-4)
    b.add_capacitor("C", "m", "gnd", 1e-6)
    return p.simulate(b, t_end=2e-3, dt=2e-6, engine="pwl", **kw)


@pytest.mark.parametrize("make", [
    lambda **kw: _buck(50.0, 10e-6, 100, nper=40, **kw)[1],   # linear, diodes
    lambda **kw: _buck(50.0, 10e-6, 100, nper=40, discontinuity_restart=True,
                       enable_substep_state_correction=True, **kw)[1],
    _sat,                                                   # Newton path
])
def test_states_are_bit_identical_with_the_diagnostics_off(make):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = make()
        off = make(resolution_diagnostics=False)
    assert on.resolution is not None and off.resolution is None
    a, b = np.asarray(on.states), np.asarray(off.states)
    assert a.tobytes() == b.tobytes()
    assert np.asarray(on.times).tobytes() == np.asarray(off.times).tobytes()


def test_off_means_no_record_and_no_warning():
    res, w = _sim(_rc(), t_end=0.3, dt=10 * TAU, resolution_diagnostics=False)
    assert res.resolution is None and not w
    sol = p.SolverOptions(resolution_diagnostics=False)
    res, w = _sim(_rc(), t_end=0.3, dt=10 * TAU, solver=sol)
    assert res.resolution is None and not w


def test_variable_step_engine_has_no_record():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(_rc(), t_end=5e-3, engine="trbdf2")
    assert getattr(res, "resolution", None) is None


def test_store_every_keeps_the_maximum_between_samples():
    b = _rc()
    full, _ = _sim(b, t_end=0.02, dt=TAU)
    dec, _ = _sim(_rc(), t_end=0.02, dt=TAU, store_every=4)
    lf, ld = np.asarray(full.resolution.lte), np.asarray(dec.resolution.lte)
    assert len(ld) == len(dec.times) == 6
    for m in range(1, len(ld)):
        assert ld[m] == np.max(lf[4 * m - 3:4 * m + 1])
