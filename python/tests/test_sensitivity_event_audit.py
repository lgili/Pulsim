"""add-parameter-sensitivities group 6, the audit (DECISIONS 16): every cell
event kind × device class × engine configuration that the code distinguishes
ends in exactly one of two outcomes —

- a NUMBER that matches a truth independent of the simulator (central
  differences of scipy Radau at rtol 1e-12 with event functions, within
  1e-5 relative), or
- a refusal BY NAME (``UnsupportedError`` whose message names the reason).

There is no third outcome: an exception of another type, or a number off
its truth, fails the cell. The expected outcome of each cell is pinned in
``EXPECTED``: a refusal that starts returning a number fails here until the
table is updated (and the number is then checked against its truth).

Plain pwl (no restart) starts first order from an inconsistent initial
state (the source steps at t = 0): its state carries an O(dt) error of its
own (1e-4 here), so its numbers are checked as the derivative of its own
discrete map — central differences of the same scheme, 1e-6 — instead.
"""

import functools
import math
import warnings

import numpy as np
import pytest

import pulsim as p

from .test_sensitivity_events import (C0, D0, GOFF, GON, GS_OFF, GS_ON, R0, R2, T_END_ON,
                                      TAU_ON, TP, V, VS, VTH, _cstep, _diode_on,
                                      _pulse_closed, _pulse_rc, _truth_on)

scipy_integrate = pytest.importorskip("scipy.integrate")

# the Newton device: a Shockley junction at node c (forward in A/B, where
# it carries a few % of the current; reverse in C, where v_c reaches 10 V)
IS, NN, VT, GMIN = 1e-8, 10.0, 0.025852, 1e-12
DP = 0.3                                    # duty of the B cases
SA, SF, SC, SR2, SRL = 10.0, 1e3, 10e-6, 20.0, 100.0   # C: half-wave rectifier

ENGINES = {
    "pwl": ("pwl", {}),
    "pwl+restart": ("pwl", dict(discontinuity_restart=True)),
    "pwl+split": ("pwl", dict(discontinuity_restart=True,
                              enable_substep_state_correction=True)),
    "trbdf2": ("trbdf2", dict(rtol=1e-10, atol=1e-13)),
    "esdirk43": ("trbdf2", dict(rtol=1e-10, atol=1e-13, integrator="esdirk43")),
}
# kind: (parameter, t_end, pwl dt). A: a PWL diode turn-on (state event) and
# no event (t_end before it); B: one PWM pulse, its gate edges fixed (R.R)
# or moving with the duty; C: a sine-fed rectifier, turn-on AND turn-off
# with a time-varying source (the ḃ term of ẋ⁻; the series R2 keeps the
# turn-on's jump alive to t_end: with ḃ's sign flipped this cell is 11 %
# off). The "_in_step_1" kinds put the event inside the first step of a
# coarse plain-pwl run (the gate baseline and the first stage's mode).
KINDS = {
    "no_event": ("R.R", 0.5 * TAU_ON, 1e-8),
    "diode_turn_on": ("R.R", T_END_ON, 1e-8),
    "fixed_gate_edge": ("R.R", 0.8 * TP, 1e-7),
    "duty_edge": ("S.duty", 0.8 * TP, 1e-7),
    "diode_on_off_sine": ("RL.R", 1.15e-3, 1e-7),
    "diode_turn_on_in_step_1": ("R.R", 3e-4, 1e-4),
    "duty_edge_in_step_1": ("S.duty", 0.8 * TP, 1e-5),
}
STEP_1 = {"diode_turn_on_in_step_1": DP, "duty_edge_in_step_1": 0.005}   # kind: duty
DEVICES = ("linear", "newton", "stateful")

# refusal reasons, by the words of their messages
NOT_LOCATED = "not located"
STAIRCASE = "staircase"
NEWTON = "Newton device"
STATEFUL = "its own state history"
LANDING = "interpolated or ESDIRK diode landing"
FLOOR = "conditioning floor"


def _expected(kind, dev, eng):
    if dev == "stateful":
        return STATEFUL
    if kind in STEP_1:
        return STAIRCASE if kind == "duty_edge_in_step_1" else NOT_LOCATED
    if kind in ("no_event", "fixed_gate_edge"):
        return None
    if eng in ("pwl", "pwl+restart"):
        return STAIRCASE if kind == "duty_edge" else NOT_LOCATED
    if eng == "esdirk43" and kind != "duty_edge":
        return LANDING
    if dev == "newton":
        return NEWTON
    if kind == "diode_on_off_sine" and eng == "trbdf2":
        return FLOOR      # the turn-off is flipped at t, not landed
    return None


EXPECTED = {(k, d, e): _expected(k, d, e) for k in KINDS for d in DEVICES for e in ENGINES
            if k not in STEP_1 or e in ("pwl", "pwl+restart")}


def _shock(v):
    return IS * math.expm1(v / (NN * VT)) + GMIN * v


def _circuit(kind, dev, R=R0, D=None, RL=SRL):
    sf = None
    D = STEP_1.get(kind, DP) if D is None else D
    if kind in ("no_event", "diode_turn_on", "diode_turn_on_in_step_1"):
        b = _diode_on(R=R)
    elif kind == "diode_on_off_sine":
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "in", "gnd", 0.0, SA, SF)
        b.add_diode("D", "in", "k", GON, GOFF, V_th=VTH)
        b.add_resistor("R2", "k", "c", SR2)
        b.add_capacitor("C", "c", "gnd", SC)
        b.add_resistor("RL", "c", "gnd", RL)
    else:
        b = _pulse_rc(R=R)
        sf = p.make_pwm_switch_fn(1 / TP, D, b.switch_index_of("S"),
                                  b.graph.num_switches)
    if dev == "newton":
        if kind == "diode_on_off_sine":
            b.add_shockley_diode("Dx", "gnd", "c", IS, NN, VT)
        else:
            b.add_shockley_diode("Dx", "c", "gnd", IS, NN, VT)
    elif dev == "stateful":
        b.add_saturable_inductor("Ls", "c", "m", 1e-3, 1.0)
        b.add_resistor("Rm", "m", "gnd", 1e6)
    return b, sf


def _ivp(f, t0, t1, y, events=None):
    return scipy_integrate.solve_ivp(f, (t0, t1), y, events=events, method="Radau",
                                     rtol=1e-12, atol=1e-15)


@functools.lru_cache(maxsize=None)
def _truth_v(kind, dev, t_end, R=R0, D=DP, RL=SRL):
    """v_c(t_end) of the circuit's ODE, events located by Radau's root finder."""
    if kind == "diode_on_off_sine":
        sh = (lambda v: -_shock(-v)) if dev == "newton" else (lambda v: 0.0)

        def vin(s):
            return SA * math.sin(2 * math.pi * SF * s)

        def f(g):
            return lambda s, y: [((vin(s) - y[0]) * g / (1 + g * SR2) - y[0] / RL
                                  - sh(y[0])) / SC]

        def ev_on(s, y):   # the diode's own voltage reaches V_th
            return (vin(s) - y[0]) / (1 + GOFF * SR2) - VTH

        def ev_off(s, y):   # its current reaches zero
            return vin(s) - y[0]

        ev_on.terminal, ev_on.direction = True, 1
        ev_off.terminal, ev_off.direction = True, -1
        t0, y, g = 0.0, [0.0], GOFF
        while True:
            s = _ivp(f(g), t0, t_end, y, ev_on if g == GOFF else ev_off)
            if s.status == 0:
                return float(s.y[0, -1])
            t0, y = float(s.t_events[0][0]), s.y_events[0][0]
            g = GON if g == GOFF else GOFF
    sh = _shock if dev == "newton" else (lambda v: 0.0)
    if kind in ("no_event", "diode_turn_on"):
        def f(g):
            return lambda s, y: [((V - y[0]) / R - y[0] / (1 / g + R2) - sh(y[0])) / C0]

        def ev(s, y):
            return y[0] / (1 + GOFF * R2) - VTH   # the diode's own voltage

        ev.terminal, ev.direction = True, 1
        s1 = _ivp(f(GOFF), 0.0, t_end, [0.0], ev)
        if s1.status == 0:
            return float(s1.y[0, -1])
        s2 = _ivp(f(GON), float(s1.t_events[0][0]), t_end, s1.y_events[0][0])
        return float(s2.y[0, -1])

    def f(g):
        return lambda s, y: [((VS - y[0]) / (R + 1 / g) - sh(y[0])) / C0]

    t1 = D * TP
    s1 = _ivp(f(GS_ON), 0.0, t1, [0.0])
    return float(_ivp(f(GS_OFF), t1, t_end, s1.y[:, -1]).y[0, -1])


def _truth(kind, dev, t_end, name):
    key = {"R.R": "R", "S.duty": "D", "RL.R": "RL"}[name]
    x0 = {"R": R0, "D": DP, "RL": SRL}[key]
    h = 1e-4 * x0
    return (_truth_v(kind, dev, t_end, **{key: x0 + h})
            - _truth_v(kind, dev, t_end, **{key: x0 - h})) / (2 * h)


def _run(kind, dev, eng, sens=True, **over):
    name, t_end, dt = KINDS[kind]
    engine, kw = ENGINES[eng]
    kw = dict(kw)
    if engine == "pwl":
        kw["dt"] = dt
    if dev == "newton":
        kw.update(tol_newton_dx=1e-12, tol_newton_res=1e-12)
    b, sf = _circuit(kind, dev, **over)
    if sens:
        kw["sensitivities"] = [name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, engine=engine, switch_fn=sf, **kw)


@pytest.mark.parametrize("kind,dev,eng", list(EXPECTED))
def test_every_cell_is_a_true_number_or_a_refusal_by_name(kind, dev, eng):
    name = KINDS[kind][0]
    want = EXPECTED[(kind, dev, eng)]
    try:
        r = _run(kind, dev, eng)
    except p.UnsupportedError as e:
        assert want is not None, f"refused, the table expects a number: {e}"
        assert e.feature in ("sensitivities", "grazing_event"), e.feature
        assert want in str(e), str(e)
        return
    assert want is None, f"a number came back where the table expects a refusal ({want})"
    # the run is the same bits without sensitivities, with the same factor
    # traffic (5.4 / 5.5; pwl's linear factors come from `resident_factor`,
    # which never builds one); one sensitivity solve per committed stage
    r0 = _run(kind, dev, eng, False)
    assert np.array_equal(np.asarray(r0.times), np.asarray(r.times))
    assert np.array_equal(np.asarray(r0.states), np.asarray(r.states))
    for k in ("n_factorizations", "n_solves", "n_accept", "n_reject"):
        assert getattr(r0.stats, k) == getattr(r.stats, k), k
    assert r0.newton.factorizations == r.newton.factorizations
    c = r.sensitivity.counters
    rows = len(r.sensitivity.times) - 1
    assert rows <= c["n_sens_solves"] <= 5 * rows + 8, (c, rows)
    if dev == "linear":
        assert c["n_sens_iterations"] == 0, c
    else:   # modified Newton on the stage's factor: about one refinement each
        assert c["n_sens_solves"] <= c["n_sens_iterations"] <= 2 * c["n_sens_solves"], c
    s = float(r.sens("c", name)[-1])
    t_last = float(r.sensitivity.times[-1])
    if eng == "pwl":
        key = {"R.R": "R", "RL.R": "RL"}[name]
        x0 = {"R": R0, "RL": SRL}[key]
        h = 1e-4 * x0
        ref = (float(np.asarray(_run(kind, dev, eng, False, **{key: x0 + h}).v("c"))[-1])
               - float(np.asarray(_run(kind, dev, eng, False, **{key: x0 - h}).v("c"))[-1])
               ) / (2 * h)
        assert s == pytest.approx(ref, rel=1e-6), (s, ref)
        return
    truth = _truth(kind, dev, t_last, name)
    assert s == pytest.approx(truth, rel=1e-5), (s, truth)


# The event AT t_end (DECISIONS 16, the reviewer's finding): an event within
# the time resolution of the run's end, its jump never applied, read the left
# limit (dv/dD 0.000 against 7.38; dv/dR -6.7e-4 against +4.4e-3). The pwl
# grid is made to end at t_end (dt = t_end / n), so the event lies in its last
# step. Each cell is a refusal by name or the truth; the same event one and a
# half steps before t_end stays a number wherever the engine gives one.
NOT_APPLIED = "whose jump in the derivative was not applied"
END_EVENTS = {"duty_edge": ("S.duty", D0 * TP, 1e-7), "diode_turn_on": ("R.R", TAU_ON, 1e-8)}


def _end_expected(kind, where, eng):
    if eng == "pwl":
        # plain pwl's own trajectory (first order from the start, 1e-4 V
        # off) has not crossed yet: no event in its run, the number is
        # that of its own map (h = 1e-4 R would step over its crossing)
        return None if (kind, where) == ("diode_turn_on", "at_t_end") else (
            STAIRCASE if kind == "duty_edge" else NOT_LOCATED)
    if eng == "pwl+restart":
        return NOT_APPLIED if (kind, where) == ("duty_edge", "at_t_end") else (
            STAIRCASE if kind == "duty_edge" else NOT_LOCATED)
    if eng == "esdirk43" and kind == "diode_turn_on":
        return LANDING
    return NOT_APPLIED if where == "at_t_end" else None


@pytest.mark.parametrize("kind", list(END_EVENTS))
@pytest.mark.parametrize("where", ["at_t_end", "before_t_end"])
@pytest.mark.parametrize("eng", list(ENGINES))
def test_an_event_at_t_end_is_refused_by_name_or_true(kind, where, eng):
    name, tau, dt = END_EVENTS[kind]
    t_end = tau * (1 + 1e-11) if where == "at_t_end" else tau + 1.5 * dt
    engine, kw = ENGINES[eng]
    kw = dict(kw)
    if engine == "pwl":
        kw["dt"] = t_end / round(t_end / dt)
    want = _end_expected(kind, where, eng)

    def run(R=R0, sens=True):
        if kind == "duty_edge":
            b = _pulse_rc(R=R)
            sf = p.make_pwm_switch_fn(1 / TP, D0, b.switch_index_of("S"), b.graph.num_switches)
        else:
            b, sf = _diode_on(R=R), None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return p.simulate(b, t_end, engine=engine, switch_fn=sf,
                              **kw, **({"sensitivities": [name]} if sens else {}))

    try:
        r = run()
    except p.UnsupportedError as e:
        assert want is not None, f"refused, the table expects a number: {e}"
        assert want in str(e), str(e)
        return
    assert want is None, f"a number came back where the table expects a refusal ({want})"
    s = float(r.sens("c", name)[-1])
    t_last = float(r.sensitivity.times[-1])
    assert t_last > tau
    if eng == "pwl":   # the derivative of its own map
        h = 1e-6 * R0
        ref = (float(np.asarray(run(R0 + h, False).v("c"))[-1])
               - float(np.asarray(run(R0 - h, False).v("c"))[-1])) / (2 * h)
        assert s == pytest.approx(ref, rel=1e-6), (s, ref)
        return
    truth = (_cstep(lambda d: _pulse_closed(R0, C0, d, t_last), D0) if kind == "duty_edge"
             else _truth_on(name, t_last))
    assert s == pytest.approx(truth, rel=1e-5), (s, truth)
