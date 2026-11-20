"""``t_start`` is an ABSOLUTE time label (fix/core-tstart-time-base).

Every time-dependent input is read at absolute t on every engine, so a run on
[t_start, t_end] started from the state at t_start equals the tail of the run
on [0, t_end]. The exact engine runs a fresh ``t_start`` run on its own clock
from 0: it read the gate (and its edge oracle, the start mask, ``b_extra_fn``)
on that clock, t_start early, while pwl / trbdf2 read them at absolute t (a
two-switch leg with t_start = half a period booked its high side conducting,
1.8e-4 J, against 1.8e-8 J on trbdf2). ``simulate()`` now shifts the drive
onto the engine's clock where the plan hands it over (``_run_dsed``), and
relabels every reported time. What cannot be shifted from Python is refused
by name: the circuit's own time-varying sources on the exact engine (the
native adapter evaluates them), controllers with t_start on the exact engine.

Two kinds of tests:

* the time base of each input, made observable on a STATELESS circuit (the
  output at t is the input at t): f(absolute t), never f(t - t_start);
* the property, on three circuits (a leg with a loss port, a sine-fed
  rectifier, a PWM buck with a control tick) and every engine: (a) resume
  from the snapshot at T_MID and (b) a fresh run at t_start = T_MID from the
  snapshot's state both match the uninterrupted run's tail (states, loss
  energies and events, recorder window / period statistics).
"""

from __future__ import annotations

import math
import pickle
import warnings

import numpy as np
import pytest

import pulsim as p

F = 1e4                      # every gate / source period is 1e-4 s
TS, T = 3e-5, 2e-4           # the time-base probes: t_start off every edge

ENGINES = {
    "pwl": dict(engine="pwl", dt=1e-7),
    "trbdf2": dict(engine="trbdf2", rtol=1e-8),
    "esdirk43": dict(engine="trbdf2", integrator="esdirk43", rtol=1e-8),
    "dsed": dict(engine="dsed"),
    "auto": dict(engine="auto", rtol=1e-8),
}
EXACT = ("dsed",)


def _sim(b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, **kw)


# ------------------------------------------------- the time base of each input

def _square(t, duty=0.3):
    return ((t * F) % 1.0 < duty) * 1.0


def _leg_r():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "dcp", "gnd", 1.0)
    b.add_switch("HS", "dcp", "a", 1e3, 1e-9)
    b.add_resistor("R", "a", "gnd", 1.0)
    return b


class _Gate:
    """A Python gate, duty 0.3, with or without an edge oracle."""

    def __init__(self, oracle):
        if oracle:
            self.next_edge_after = self._nea

    def __call__(self, t):
        m = p.SwitchStateMask(1)
        m.set(0, bool(_square(t)))
        return m

    @staticmethod
    def _nea(t):
        k = math.floor(t * F)
        return next(c for c in (k / F, (k + 0.3) / F, (k + 1) / F, (k + 1.3) / F) if c > t)


def _input(name):
    """(builder, simulate kwargs, the output v(a) as a function of absolute t)."""
    b, kw = p.CircuitBuilder(), {}
    if name == "sine_source":
        b.add_sine_voltage_source("V", "a", "gnd", 0.0, 1.0, F, 0.3)
        return b, kw, lambda t: np.sin(2 * np.pi * F * t + 0.3)
    if name == "pulse_source":
        b.add_pulse_voltage_source("V", "a", "gnd", 0.0, 1.0, 2e-5, 4e-5, 1e-4)
        return b, kw, lambda t: ((((t - 2e-5) % 1e-4) < 4e-5) & (t >= 2e-5)) * 1.0
    if name == "pwm_source":
        b.add_pwm_voltage_source("V", "a", "gnd", 1.0, 0.0, F, 0.3)
        return b, kw, _square
    if name == "pwl_source":
        b.add_pwl_current_source("I", "gnd", "a", [0.0, 1e-4, 2e-4], [0.0, 1.0, 0.0])
        b.add_resistor("R", "a", "gnd", 1.0)
        return b, kw, lambda t: -np.interp(t, [0.0, 1e-4, 2e-4], [0.0, 1.0, 0.0])
    if name == "b_extra_fn":   # tau = 1 ps: v(a) is the injection
        b.add_resistor("R", "a", "gnd", 1.0)
        b.add_capacitor("C", "a", "gnd", 1e-12)
        n = b.pool.state_size(b.graph)

        def bx(t):
            v = np.zeros(n)
            v[0] = math.sin(2 * np.pi * F * t)
            return v
        return b, dict(b_extra_fn=bx), lambda t: -np.sin(2 * np.pi * F * t)
    b = _leg_r()
    if name == "schedule":       # a SwitchSchedule: the native edge oracle
        return b, dict(switch_fn=p.make_pwm_switch_fn(F, 0.3, 0, 1)), _square
    return b, dict(switch_fn=_Gate(name == "python_gate_oracle")), _square


INPUTS = ("schedule", "python_gate", "python_gate_oracle", "b_extra_fn",
          "sine_source", "pulse_source", "pwm_source", "pwl_source")
TIMED_SOURCES = {"sine_source": "V", "pulse_source": "V", "pwm_source": "V",
                 "pwl_source": "I"}
DSED_ROUTES = {"dsed_rk45": dict(engine="dsed", integrator="rk45"), "dsed_auto": dict(engine="dsed", integrator="auto"),
               "dsed_bdf2": dict(engine="dsed", integrator="bdf2"),
               "exact": dict(engine="exact")}


@pytest.mark.parametrize("route", [*ENGINES, *DSED_ROUTES])
@pytest.mark.parametrize("name", INPUTS)
def test_every_input_is_read_at_absolute_time(name, route):
    b, kw, f = _input(name)
    opts = {**ENGINES, **DSED_ROUTES}[route]
    exact = opts["engine"] in ("dsed", "exact")
    if exact and name in TIMED_SOURCES:
        # the native adapter evaluates the circuit's sources on the engine's
        # clock from 0: refused by name, never shifted silently
        with pytest.raises(p.UnsupportedError, match="t_start") as ei:
            _sim(b, t_end=T, t_start=TS, **opts, **kw)
        assert ei.value.feature == "t_start" and ei.value.device == TIMED_SOURCES[name]
        return
    if exact and name == "b_extra_fn" and route != "dsed_bdf2":
        pytest.skip("b_extra_fn has no exact step on the event-driven integrators "
                    "(refused by name at any t_start)")
    r = _sim(b, t_end=T, t_start=TS, **opts, **kw)
    t, v = np.asarray(r.times), np.asarray(r.v("a"))
    # after the start transient, off the edges (a sample on an edge is a limit)
    def smooth(u):
        return np.abs(f(u + 1e-9) - f(u - 1e-9)) < 0.5
    ok = (t > TS + 1e-8) & smooth(t) & smooth(t - TS)
    assert ok.sum() >= 10
    assert np.abs(v[ok] - f(t[ok])).max() < 1e-3          # absolute t
    assert np.abs(v[ok] - f(t[ok] - TS)).max() > 0.2      # (the probe tells the two apart)


def test_exact_engine_reports_reinit_records_at_absolute_time():
    """An ideal switch closing on a capacitor loop at 7e-5 s: the record's t
    (and the event time) is 7e-5 with and without t_start."""
    class Close:
        def __call__(self, t):
            m = p.SwitchStateMask(1)
            m.set(0, t >= 7e-5)
            return m

        def next_edge_after(self, t):
            return 7e-5 if t < 7e-5 else math.inf

    for ts in (0.0, 3e-5):
        b = p.CircuitBuilder()
        b.add_capacitor("C1", "x", "gnd", 1e-6, 10.0)
        b.add_capacitor("C2", "y", "gnd", 2e-6, 2.0)
        b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
        b.add_resistor("R", "x", "gnd", 1e3)
        r = _sim(b, t_end=1e-4, t_start=ts, engine="dsed", switch_fn=Close())
        (rec,) = r.reinit_records
        assert rec.t == pytest.approx(7e-5, abs=1e-15)
        assert r.times[rec.sample] == pytest.approx(7e-5, abs=1e-15)
        np.testing.assert_allclose(r.event_times, [7e-5], rtol=0, atol=1e-15)


def test_exact_engine_with_controllers_and_t_start_is_refused_by_name():
    b = _leg_r()
    with pytest.raises(p.UnsupportedError, match="t_start != 0"):
        _sim(b, t_end=T, t_start=TS, engine="dsed", switch_fn=_Gate(True),
             controller_period=1e-4, step_observer=lambda t, x: None)


def test_exact_engine_fresh_t_start_run_writes_no_snapshot_by_name():
    """Its controller block holds engine-clock times: no snapshot, and why."""
    b, kw, _ = _input("schedule")
    r = _sim(b, t_end=T, t_start=TS, engine="dsed", **kw)
    assert not r.final_snapshot.valid
    assert "t_start" in r.final_snapshot.invalid_reason
    with pytest.raises(p.UnsupportedError):
        _sim(b, t_end=2 * T, engine="dsed", resume_from=r.final_snapshot, **kw)


# ------------------------------------------------------------- the property

T_END = 0.975e-3


def _tab(e):
    return p.LossTable(v_axis=[0.0, 600.0], i_axis=[0.0, 50.0], tj_axis=[25.0],
                       energy=np.array([[[0.0], [0.0]], [[0.0], [e]]]))


def _leg(port=True):
    """600 V leg, R-L load, a 10 kHz complementary pair, a loss port on HS."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", 600.0)
    b.add_switch("HS", "dcp", "a", 1e3, 1e-9)
    b.add_switch("LS", "a", "gnd", 1e3, 1e-9)
    b.add_resistor("Rl", "a", "m", 10.0)
    b.add_inductor("L", "m", "gnd", 1e-3)
    if port:
        b.set_device_losses("HS", Tj=25.0, E_on=_tab(4e-3), E_off=_tab(3.5e-3))
    fn = p.make_dead_time_pwm_pair_fn(F, 0.5, b.switch_index_of("HS"),
                                      b.switch_index_of("LS"), b.graph.num_switches, 0.0)
    return b, dict(switch_fn=fn), ("i(L)", "v(a)")


def _rectifier():
    """A 10 kHz sine source through a diode into an RC load."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "s", "gnd", 0.0, 100.0, F)
    b.add_resistor("Rs", "s", "k", 1.0)
    b.add_diode("D", "k", "out", 1e3, 1e-9)
    b.add_capacitor("C", "out", "gnd", 2e-6)
    b.add_resistor("R", "out", "gnd", 50.0)
    return b, {}, ("v(out)", "i(Rs)")


class _PGate:
    """A buck gate on at (k + 0.25)/F for duty/F; the control tick sets the
    duty from v(out) (a P law: no controller state but the duty)."""

    def __init__(self, b):
        self.k_out = next(n["id"] for n in b.graph.nodes if n["name"] == "out")
        self.q, self.n = b.switch_index_of("Q"), b.graph.num_switches
        self.duty = 0.5

    def __call__(self, t):
        ph = (t * F) % 1.0
        m = p.SwitchStateMask(self.n)
        m.set(self.q, 0.25 <= ph < 0.25 + self.duty)
        return m

    def next_edge_after(self, t):
        k = math.floor(t * F)
        cands = [(j + o) / F for j in (k, k + 1) for o in (0.25, 0.25 + self.duty)]
        return min(c for c in cands if c > t)

    def tick(self, t, x):
        self.duty = min(0.9, max(0.1, 0.5 + 0.05 * (12.0 - float(x[self.k_out]))))


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e3, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-4)
    b.add_capacitor("C", "out", "gnd", 2e-5)
    b.add_resistor("R", "out", "gnd", 5.0)
    g = _PGate(b)
    return b, dict(switch_fn=g, step_observer=g.tick, controller_period=1e-4), ("v(out)", "i(L)")


# T_MID: off every edge for the leg and the rectifier; on the buck's tick grid
# (a fresh run's ticks start at t_start, the documented contract)
CIRCUITS = {"leg": (_leg, 0.43e-3), "rectifier": (_rectifier, 0.43e-3), "buck": (_buck, 0.4e-3)}


def _run(circuit, opts, t_end, rec=True, built=None, **kw):
    """``built``: the (builder, drive, probes) of an earlier run, so a resume
    continues its Python controller objects (the snapshot holds the circuit)."""
    make, t_mid = CIRCUITS[circuit] if isinstance(circuit, str) else circuit
    b, kw0, probes = built or make()
    recs = [p.Recorder("r", list(probes), keep="none", windows=[(0.5e-3, 0.9e-3)],
                       period=1e-4)] if rec and t_end > 0.5e-3 else None
    r = _sim(b, t_end=t_end, **opts, **kw0, recorders=recs, **kw)
    r._built = (b, kw0, probes)
    return r, probes


def _losses(r):
    out = {}
    for x in getattr(r, "_device_losses", None) or []:
        out[x.device] = (x.E_cond_electrical, x.E_on, x.E_off, x.n_on, x.n_off)
    return out


def _tail_events(r, dev, t0):
    ev = r.loss_events(dev)
    ev = ev[ev["t"] > t0 + 1e-12]
    return np.asarray(ev["t"], float), np.asarray(ev["E"], float)


# Agreement of a run from T_MID with the uninterrupted run's tail, relative to
# the circuit's scale (states, loss energies; event instants to 1e-12 s). The
# exact engine: to rounding. trbdf2 / esdirk43 at rtol 1e-8: the two runs take
# different steps. pwl at dt = 1e-7: a fresh start has no trapezoid history
# (one first-order start step). A resume on pwl and on the exact engine is
# bit-identical in the state.
TOL = {"dsed": 1e-10, "trbdf2": 1e-5, "esdirk43": 1e-5, "auto": 1e-5, "pwl": 2e-3}
# The recorder's window / period statistics are quadratures (and extrema) over
# each run's own samples: two runs on different grids differ by that sampling.
STAT_TOL = {"dsed": 1e-8, "trbdf2": 1e-3, "esdirk43": 1e-3, "auto": 1e-3, "pwl": 2e-3}


def _assert_tail(engine, how, full, first, got, probes, t_mid, circuit):
    tol, stol = TOL[engine], STAT_TOL[engine]
    # not the buck on pwl: resumed exactly on a tick instant, the fixed engine's
    # tick at label t reads x(t - dt) in the uninterrupted run and the
    # snapshot's x(t) after a resume (pre-existing, not a time base: the
    # labels agree; fixes/tstart-time-base.md)
    bitwise = how == "resume" and (engine == "dsed" or (engine == "pwl" and circuit != "buck"))
    xs, xg = np.asarray(full.states[-1]), np.asarray(got.states[-1])
    scale = max(1.0, float(np.abs(np.asarray(full.states)).max()))
    assert got.times[-1] == pytest.approx(full.times[-1], abs=1e-15)
    if bitwise:
        assert xg.tobytes() == xs.tobytes()
    else:
        assert np.abs(xg - xs).max() <= tol * scale, np.abs(xg - xs).max() / scale
    # loss energies: a resume continues the tracker, a fresh run books the tail
    lf, l1, lg = _losses(full), _losses(first), _losses(got)
    assert set(lg) == set(lf)
    for dev in lf:
        want = lf[dev] if how == "resume" else tuple(
            a - c for a, c in zip(lf[dev], l1[dev]))
        assert lg[dev][3:] == want[3:]                               # n_on, n_off
        np.testing.assert_allclose(lg[dev][:3], want[:3], rtol=tol, atol=tol * 1e-3)
        tf, ef = _tail_events(full, dev, t_mid)
        tg, eg = _tail_events(got, dev, t_mid)
        np.testing.assert_allclose(tg, tf, rtol=0, atol=1e-12)        # absolute instants
        np.testing.assert_allclose(eg, ef, rtol=tol, atol=0)
    # recorder statistics over an absolute window and absolute periods
    rf, rg = full.recorded["r"], got.recorded["r"]
    for pr in probes:
        for f in ("avg", "rms", "min", "max"):
            a, c = rf.windows[pr][f], rg.windows[pr][f]
            np.testing.assert_allclose(c, a, rtol=stol, atol=stol * scale, err_msg=f"{pr} {f}")
        keep = rf.periods[pr]["t0"] >= t_mid - 1e-9
        np.testing.assert_allclose(rg.periods[pr]["t0"][rg.periods[pr]["t0"] >= t_mid - 1e-9],
                                   rf.periods[pr]["t0"][keep], rtol=0, atol=1e-12)
        np.testing.assert_allclose(rg.periods[pr]["avg"][rg.periods[pr]["t0"] >= t_mid - 1e-9],
                                   rf.periods[pr]["avg"][keep], rtol=stol, atol=stol * scale)


@pytest.mark.parametrize("engine", list(ENGINES))
@pytest.mark.parametrize("circuit", list(CIRCUITS))
def test_resume_and_fresh_t_start_match_the_uninterrupted_tail(circuit, engine):
    opts = ENGINES[engine]
    t_mid = CIRCUITS[circuit][1]
    full, probes = _run(circuit, opts, T_END)
    first, _ = _run(circuit, opts, t_mid)
    snap = pickle.loads(pickle.dumps(first.final_snapshot))
    # (a) resume
    got, _ = _run(circuit, opts, T_END, resume_from=snap, built=first._built)
    _assert_tail(engine, "resume", full, first, got, probes, t_mid, circuit)
    # (b) a fresh run at t_start = t_mid from the snapshot's state
    fresh = dict(t_start=t_mid, initial_state=np.asarray(snap.x))
    if engine in EXACT and circuit == "rectifier":
        with pytest.raises(p.UnsupportedError, match="time-varying source") as ei:
            _run(circuit, opts, T_END, **fresh)
        assert ei.value.device == "Vs"
        return
    if engine in EXACT and circuit == "buck":
        with pytest.raises(p.UnsupportedError, match="t_start != 0"):
            _run(circuit, opts, T_END, **fresh)
        return
    got, _ = _run(circuit, opts, T_END, **fresh)
    _assert_tail(engine, "fresh", full, first, got, probes, t_mid, circuit)


@pytest.mark.parametrize("integrator", ["auto", "bdf2"])
def test_exact_engine_other_integrators_fresh_t_start_matches_the_tail(integrator):
    """The per-mode 'auto' and the fixed-step 'bdf2' integrators (a leg with
    no loss port and no diode reaches them): a fresh run at T_MID from the
    state at T_MID reads the gate on the absolute clock."""
    opts = dict(engine="dsed", integrator=integrator)
    circ = (lambda: _leg(port=False), 0.43e-3)
    full, probes = _run(circ, opts, T_END)
    first, _ = _run(circ, opts, 0.43e-3)
    got, _ = _run(circ, opts, T_END, t_start=0.43e-3,
                  initial_state=np.asarray(first.states[-1]))
    scale = float(np.abs(np.asarray(full.states)).max())
    tol = 1e-10 if integrator == "auto" else 1e-6   # bdf2: a fixed grid from each start
    assert np.abs(np.asarray(got.states[-1]) - np.asarray(full.states[-1])).max() <= tol * scale
    for pr in probes:
        np.testing.assert_allclose(got.recorded["r"].windows[pr]["avg"],
                                   full.recorded["r"].windows[pr]["avg"], rtol=tol, atol=tol * scale)
