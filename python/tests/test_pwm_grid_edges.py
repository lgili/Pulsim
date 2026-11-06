"""Grid-exact PWM edges and DSED gate-edge landing.

Two defects, one cause: the built-in PWM generators did not know where
their own edges were.

* Fixed grid (``engine='pwl'``). ``make_pwm_switch_fn`` compared
  ``fmod(t / T, 1) < duty`` in floating point, so an edge exactly on a
  grid point flipped a step early or late at random: 49/50/51 on-steps
  per period at 100 kHz, D = 0.5, dt = 1e-7. The duty jitter kept the
  guide's buck ringing (67 mV pp of ripple instead of 17 mV).
* Variable step (``engine='dsed'``). With no ``next_edge_after`` the
  scheduler sampled the switch_fn every dt_max/10 (1 us by default) and
  moved every edge onto that grid. A synchronous buck at 100 kHz,
  D = 0.25, 100 ns dead time read 14.28 V against 11.49 V; its body
  diodes never conducted and the low-side switch dissipated 1.27 W
  against 7.2 mW.

The helpers now return a :class:`pulsim.SwitchSchedule` (grid-exact
edges + analytic ``next_edge_after``), and DSED bisects the edges of a
plain callable inside each poll window.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p


# ---------------------------------------------------------------------------
# P1 — fixed grid
# ---------------------------------------------------------------------------

GRID_CASES = [
    (100e3, 0.50, 1e-7),
    (100e3, 0.25, 1e-7),
    (100e3, 0.50, 2e-7),
    (50e3, 0.30, 1e-7),
    (200e3, 0.37, 5e-8),
    (20e3, 0.60, 2.5e-7),
]


def _on_counts(fn, bit, f, dt, periods=200, t0=0.0):
    n = round(1 / (f * dt))
    return {sum(bool(fn(t0 + k * dt).get(bit)) for k in range(j * n, (j + 1) * n))
            for j in range(periods)}


@pytest.mark.parametrize("f,duty,dt", GRID_CASES)
def test_helper_on_steps_are_exact(f, duty, dt):
    want = round(duty / (f * dt))
    sw = p.make_pwm_switch_fn(f, duty, 0, 1)
    assert _on_counts(sw, 0, f, dt) == {want}
    assert _on_counts(sw, 0, f, dt, periods=30, t0=1000 / f) == {want}
    native = p.NativePwm2Switch(1 / f, duty, 2)
    assert _on_counts(native, 0, f, dt) == {want}


@pytest.mark.parametrize("f,duty,dt", GRID_CASES[:4])
def test_recorded_duty_equals_d_on_the_fixed_engine(f, duty, dt):
    """The duty read from ``switch_states`` is D exactly in every
    period when D*T/dt is a whole number."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_switch("S", "a", "b", 1e3, 1e-9)
    b.add_resistor("R", "b", "gnd", 10.0)
    bit = b.switch_index_of("S")
    sw = p.make_pwm_switch_fn(f, duty, bit, b.graph.num_switches)
    res = p.simulate(b, t_end=60 / f, dt=dt, switch_fn=sw, engine="pwl")
    rec = np.asarray(res.switch_states)[:, bit].astype(int)
    n = round(1 / (f * dt))
    # Row k is the mask of the step that ended at t_k.
    per = {int(rec[j * n + 1:(j + 1) * n + 1].sum())
           for j in range((len(rec) - 1) // n)}
    assert per == {round(duty * n)}


def test_dead_time_pair_on_steps_are_exact():
    f, dt = 100e3, 1e-7
    for duty, td, hs, ls in ((0.5, 2e-7, 48, 48), (0.25, 3e-7, 22, 72)):
        sw = p.make_dead_time_pwm_pair_fn(f, duty, 0, 1, 2, td)
        assert _on_counts(sw, 0, f, dt) == {hs}
        assert _on_counts(sw, 1, f, dt) == {ls}


def test_control_pwm_generator_is_grid_exact():
    from pulsim.control import PwmGenerator
    g = PwmGenerator(frequency=100e3)
    n_on = {sum(g.update(duty=0.25, t=k * 1e-7) for k in range(j * 100, j * 100 + 100))
            for j in range(200)}
    assert n_on == {25.0}


def test_helpers_return_a_switch_schedule_with_edges():
    sw = p.make_dead_time_pwm_pair_fn(100e3, 0.25, 0, 1, 2, 1e-7)
    assert isinstance(sw, p.SwitchSchedule)
    assert sw.has_edges
    edges, t = [], 0.0
    for _ in range(4):
        t = sw.next_edge_after(t)
        edges.append(t)
    assert edges == pytest.approx([2.4e-6, 2.5e-6, 9.9e-6, 10e-6], rel=1e-12)
    combo = p.make_combined_switch_fn(2, [sw, p.make_pwm_switch_fn(30e3, 0.2, 1, 2)])
    assert combo.has_edges
    mixed = p.make_combined_switch_fn(2, [sw, lambda t: p.SwitchStateMask(2)])
    assert not mixed.has_edges
    assert math.isinf(mixed.next_edge_after(0.0))


# ---------------------------------------------------------------------------
# P2 — DSED synchronous buck with dead time
# ---------------------------------------------------------------------------

F, D, TD = 100e3, 0.25, 100e-9
T_END, T_FROM = 2e-3, 1.5e-3


def _sync_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_mosfet("QH", "vin", "sw", R_on=10e-3)
    b.add_mosfet("QL", "sw", "gnd", R_on=10e-3)
    b.add_inductor("L", "sw", "vout", 22e-6)
    b.add_capacitor("C", "vout", "gnd", 100e-6)
    b.add_resistor("RL", "vout", "gnd", 1.2)
    return b


def _pwm(b):
    return p.make_dead_time_pwm_pair_fn(
        F, D, b.switch_index_of("QH"), b.switch_index_of("QL"),
        b.graph.num_switches, TD)


def _summary(b, res):
    t = np.asarray(res.times)
    v = p.time_average(t, np.asarray(res.v("vout")), t_from=T_FROM)
    s = np.asarray(res.switch_states)
    w = t[1:] >= T_FROM
    dtk = np.diff(t)[w]
    frac = {n: float(np.sum(s[1:, b.switch_index_of(n)][w] * dtk) / np.sum(dtk))
            for n in ("QH", "QH_body", "QL", "QL_body")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = {e["name"]: e["P_avg"] for e in p.device_loss_summary(b, res)
                if e["name"] in frac}
    return v, frac, loss


@pytest.fixture(scope="module")
def pwl_reference():
    b = _sync_buck()
    res = p.simulate(b, t_end=T_END, dt=5e-9, switch_fn=_pwm(b), engine="pwl")
    return _summary(b, res)


def _check_against_reference(got, ref):
    v, frac, loss = got
    v_ref, frac_ref, loss_ref = ref
    assert v == pytest.approx(v_ref, rel=2e-3)
    # Dead time: the LS body diode carries the current while both
    # switches are off (76 % on the reference, where it then shares
    # the LS on-time); before the fix it never turned on (0.0000).
    for name in ("QH", "QL", "QL_body"):
        assert frac[name] == pytest.approx(frac_ref[name], abs=3e-3), name
    assert frac["QH_body"] < 1e-3
    # Before: QL 1.27 W, QL_body 0 W against 7.2 mW / 74 mW.
    for name in ("QH", "QL", "QL_body"):
        assert loss[name] == pytest.approx(loss_ref[name], rel=0.03), name


def test_dsed_sync_buck_matches_pwl_with_a_schedule(pwl_reference):
    b = _sync_buck()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # no polling warning
        res = p.simulate(b, t_end=T_END, switch_fn=_pwm(b), engine="dsed")
    _check_against_reference(_summary(b, res), pwl_reference)


def test_dsed_bisects_the_edges_of_a_plain_callable(pwl_reference):
    """A plain callable (no next_edge_after) is polled every 1 us and
    each change is bisected to the edge: the 100 ns dead time
    survives."""
    b = _sync_buck()
    sched = _pwm(b)
    plain = lambda t: sched(t)  # noqa: E731
    with pytest.warns(UserWarning, match="shorter than 1e-06 s"):
        res = p.simulate(b, t_end=T_END, switch_fn=plain, engine="dsed")
    _check_against_reference(_summary(b, res), pwl_reference)


def test_trbdf2_sync_buck_matches_pwl_with_a_schedule(pwl_reference):
    b = _sync_buck()
    res = p.simulate(b, t_end=T_END, dt=1e-6, switch_fn=_pwm(b),
                     engine="trbdf2")
    _check_against_reference(_summary(b, res), pwl_reference)


def test_dsed_without_switch_fn_does_not_warn_about_polling():
    """The default all-closed mask is constant; the polling warning
    spoke of "this plain Python switch_fn" the user never passed."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_capacitor("C", "b", "gnd", 1e-6)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        p.simulate(b, t_end=1e-5, engine="dsed")
    assert not [w for w in rec if "next_edge_after" in str(w.message)]
