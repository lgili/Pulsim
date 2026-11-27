"""Event LOCATION under the fixed engine's event split (findings F-SB-2 and
F-SB-4 of the restart-rule scoreboard, notes in
openspec/changes/update-number-changing-release/fixes/split-event-location.md).

Every instant is checked against a truth the engine does not produce: a
threshold crossing written down by hand, the closed-form onset of a diode
fed by a sine, or the mpmath engine (benchmarks/tools/exact_truth.py).
"""

from __future__ import annotations

import dataclasses
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
RESTART = dict(discontinuity_restart=True)
SPLIT = dict(discontinuity_restart=True, enable_substep_state_correction=True)


def _flips(res, k):
    """(instant, entered state) at each change of switch ``k`` in
    ``with_event_limits(res)``: the instant is the row before the change."""
    v = p.with_event_limits(res)
    t = np.asarray(v.times)
    s = np.asarray(v.switch_states)[:, k]
    return [(float(t[j]), bool(s[j + 1])) for j in np.flatnonzero(s[1:] != s[:-1])]


# ---- F-SB-2: a gate edge inside a step -------------------------------------

def _gated(edge, h, *, opaque=False, **kw):
    # vcswitch_resistor's minimal form: the control ramps 0 -> 10 V in 1 ns from
    # 20 us, the 5 V threshold is crossed at its middle: the edge is ``edge``.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_switch("S", "vin", "out", 1000.0, 1e-9)
    b.add_resistor("R", "out", "gnd", 100.0)
    k = b.switch_index_of("S")
    fn = p.make_pwm_switch_fn(20e3, 0.6, k, b.graph.num_switches, edge)
    if opaque:
        sched = fn
        fn = lambda t: sched(t)  # noqa: E731  (no next_edge_after: opaque)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=30e-6, engine="pwl", dt=h, switch_fn=fn, **kw)
    return r, [f[0] for f in _flips(r, k) if f[1]]


H = 1e-7
EDGES = [20.0005e-6, 20.03e-6, 20.0995e-6]       # 0.005 h, 0.3 h, 0.995 h into a step


@pytest.mark.parametrize("opaque", [False, True], ids=["oracle", "opaque"])
@pytest.mark.parametrize("edge", EDGES)
def test_the_split_places_a_gate_edge_at_its_instant(edge, opaque):
    r, on = _gated(edge, H, opaque=opaque, **SPLIT)
    assert len(on) == 1
    # located to split_snap*dt = 1e-6 h (oracle: exact up to the snap)
    assert abs(on[0] - edge) <= 1e-3 * H, (on[0] - edge) / H
    c = r.counters
    if opaque:
        assert c["n_gate_events_bisected"] == 2 and c["unlocated_gate_steps"] == 0
    else:
        assert c["n_gate_events_bisected"] == 0 and c["n_gate_events"] == 2


@pytest.mark.parametrize("edge", EDGES[:2])
def test_restart_without_the_split_takes_plain_pwls_side(edge):
    """Without the split an event sits on the grid by definition: the step
    that contains the edge carries the new mask (the grid point BEFORE the
    edge), as plain pwl does."""
    t_prev = np.floor(edge / H) * H
    _, plain = _gated(edge, H)
    _, rest = _gated(edge, H, **RESTART)
    assert rest == pytest.approx(plain, abs=1e-6 * H)
    assert rest[0] == pytest.approx(t_prev, abs=1e-6 * H)


def test_restart_takes_an_edge_within_dt_over_100_of_a_grid_point_there():
    # the gate is read at t_k - 0.5001 dt: an edge 0.995 h in is taken at t_k (the nearer point)
    _, rest = _gated(EDGES[2], H, **RESTART)
    assert rest[0] == pytest.approx(20.1e-6, abs=1e-6 * H)


def test_restart_keeps_an_edge_on_the_grid_exact():
    _, rest = _gated(20.0e-6, H, **RESTART)
    assert rest[0] == pytest.approx(20.0e-6, abs=1e-6 * H)


# ---- F-SB-4: a diode turn-on behind a curved signal --------------------------

W, R, C, V0, A = 2 * np.pi * 5e4, 100.0, 10e-6, 9.8, 10.0


def _onset_truth():
    # D OFF from t = 0 (in = 0 < out = V0): out = V0 exp(-t/RC) (g_off's
    # 1e-9 S moves it by < 3e-11 V), on when A sin(wt) = out.
    import mpmath as mp
    mp.mp.dps = 40
    return float(mp.findroot(lambda t: A * mp.sin(W * t) - V0 * mp.e ** (-t / (R * C)),
                             (mp.mpf("3e-6"), mp.mpf("5e-6")), solver="anderson"))


def _onset(dt):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, A, 5e4)
    b.add_diode("D1", "in", "out", 1e3, 1e-9)
    b.add_resistor("R", "out", "gnd", R)
    b.add_capacitor("C", "out", "gnd", C, V0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=8e-6, engine="pwl", dt=dt, **SPLIT)
    return r, [f[0] for f in _flips(r, b.switch_index_of("D1")) if f[1]][0]


def test_a_diode_onset_behind_a_sine_is_located():
    te = _onset_truth()                       # 4.29825101649e-6
    d = []
    for dt in (1e-6, 5e-7, 2.5e-7):
        r, t_on = _onset(dt)
        d.append((t_on - te) / dt)
        assert r.counters["split_locator_trials"] > 0
    # the chord alone: +0.203, +0.078, +0.030 h
    assert all(abs(x) <= 1e-3 for x in d), d


def _netlist(rel, t_end):
    sys.path[:0] = [str(BENCH / "tools"), str(BENCH)]
    import v1_reader
    return dataclasses.replace(v1_reader.load(BENCH / rel), t_end=t_end)


def test_diode_rectifier_turn_on_is_located():
    case = _netlist("circuits/diodes/diode_rectifier.yaml", 3e-5)
    b, _, _ = case.build()
    k = b.switch_index_of("D1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = case.run("pwl", **SPLIT)
    on = [f[0] for f in _flips(r, k) if f[1] and f[0] > 1e-5]
    truth = 2.4375423059672187e-05            # exact_truth (mpmath), D1+ after the first discharge
    assert abs(on[0] - truth) <= 1e-3 * case.dt, (on[0] - truth) / case.dt


W2 = 2 * np.pi * 50e3
LOOPS2 = {1: (-7.845461538461539, -4.279175963571965), 2: (9.915404682274248, -1.7637933521060212)}


def _two_diodes():
    """Two independent loops, Vk 10*sin(w*t + phi_k) -> Dk -> 1 ohm -> E_k.
    Dk turns on when 10*sin(w*t + phi_k) = E_k (closed form):
    t_k = (asin(E_k/10) - phi_k)/w: D1 10.75 us, D2 10.20 us, both in the
    step [10, 11] us at dt = 1 us; D1's signal is convex, D2's concave, so
    the chords order them wrongly (D1 10.71 us, D2 10.90 us)."""
    b = p.CircuitBuilder()
    for k, (e, ph) in LOOPS2.items():
        b.add_sine_voltage_source(f"V{k}", f"s{k}", "gnd", 0.0, 10.0, 50e3, phase=ph)
        b.add_diode(f"D{k}", f"s{k}", f"m{k}", 1e3, 1e-9)
        b.add_resistor(f"R{k}", f"m{k}", f"e{k}", 1.0)
        b.add_voltage_source(f"E{k}", f"e{k}", "gnd", e)
    return b, p.simulate(b, t_end=12e-6, dt=1e-6, engine="pwl", **SPLIT)


def _truth2(k):
    e, ph = LOOPS2[k]
    return (np.arcsin(e / 10) - ph) / W2


def test_two_independent_diodes_in_one_step_the_earlier_is_located():
    """Only the earliest CHORD's crosser used to be refined: D1 to 10.75 us,
    and D2 taken at D1's instant (+0.55 h), held OFF since 10.20 us, with no
    counter. Every other crosser is now checked at the refined instant: D2
    is located at its truth and D1 stays OFF through it (its own turn-on
    lies in the sub-step after the split; see the strict xfail below)."""
    b, r = _two_diodes()
    v = p.with_event_limits(r)
    t = np.asarray(v.times)
    k2 = b.switch_index_of("D2")
    (got,) = [x for x, on in _flips(r, k2) if on and 10e-6 < x < 11e-6]
    assert got == pytest.approx(_truth2(2), abs=1e-4 * 1e-6)
    (j,) = [j for j in np.flatnonzero(t[1:] == np.nextafter(t[:-1], np.inf)) if 10e-6 < t[j] < 11e-6]
    assert abs(np.asarray(v.i("D1"))[j + 1]) < 1e-6   # D1 off at D2's right limit
    assert r.counters["split_locator_giveups"] == 0


@pytest.mark.xfail(strict=True, reason=(
    "open (restart line STATUS, item 2): a crossing in the sub-step AFTER a located "
    "event is not located (the locator runs on the trapezoidal interval 0 only): D1 "
    "is taken at the step end 11 us: measured 2026-11-27, the nearest sample is "
    "2.5e-7 s (0.25 dt) from its truth 10.75 us, bound 1e-10 s"))
def test_two_independent_diodes_in_one_step_the_later_is_located_too():
    b, r = _two_diodes()
    v = p.with_event_limits(r)
    t = np.asarray(v.times)
    assert np.any(np.abs(t - _truth2(1)) < 1e-4 * 1e-6)
