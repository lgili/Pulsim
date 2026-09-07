"""The fixed engine is GLOBALLY FIRST ORDER from an inconsistent start.

The engine advertises trapezoidal companions and delivers second
order only when the run happens to start consistent. At t_start the
companion history carries just half of each device's state:
`seed_from_dc_op` (pwl/history_state.hpp:396) sets an inductor's
v_prev and a capacitor's i_prev to ZERO, because the state vector
does not carry them. Stepping trapezoidally from that is not
"trapezoidal with a small error" — the companion source is wrong by
(h/2L)·v_0, an O(h) LOCAL error, so the whole run is globally first
order. RL decay, R = 1 ohm, L = 1 mH, i0 = 5 A, at 5 tau against
5·e^-5:

    dt      rel err      dt/(2 tau)
    4e-6    +1.997e-03   2.000e-03
    2e-6    +9.993e-04   1.000e-03
    1e-6    +4.998e-04   5.000e-04
    5e-7    +2.500e-04   2.500e-04

exactly dt/(2 tau), falling first order — the same on an RC from c0,
and on either driven from rest by a DC step (1.35e-05 at dt = 4e-6).

THE DECISIVE CONTROL, which passes below: a sine-driven RL from rest,
where v_L(0) = 0 makes the zero seeding accidentally CORRECT, is
second order. The stepper is fine; the start is not.

Two fixes were measured and one was rejected, which is why this is
pinned rather than fixed:

* A genuine backward-Euler first step (local error O(h^2), so the run
  stays second order) can be had with no new solver: the trapezoidal
  companion assembled at 2h IS the backward-Euler companion for a
  step of h. It works — every case above became second order, the
  DC-step cases improving 1500x — but it is WRONG for any device
  whose history commit integrates over the elapsed time. Passing 2h
  makes a charge-based Coss integrate twice the interval: measured,
  test_nonlinear_capacitor.py::test_charge_is_conserved_exactly went
  from its documented half-step charge offset to a full-step one
  (4.8 % at sample 11). It also re-scales the dt-retry ladder, so
  every deliberate-failure test changed behaviour (21 tests in all).
* The right fix is CONSISTENT INITIAL CONDITIONS: solve once at
  t_start with the inductor currents pinned at i0 and the capacitor
  voltages at v0, then seed v_prev / i_prev from that solution. It
  changes no elapsed-time bookkeeping, so it disturbs neither the
  retry ladder nor the stateful devices; it also fixes sample 0,
  which today reports v = 0 V on a node whose consistent value is
  -5 V, and it would make the charge test's offset vanish entirely
  (its q_ref is I·(t - h/2) precisely because of this defect). It
  needs a solve with capacitors as voltage sources, which the MNA
  assembly does not currently offer — a feature, not a patch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p

TAU = 1e-3
DTS = (4e-6, 2e-6, 1e-6, 5e-7)


def _order(values, exacts):
    """Measured convergence order between successive halvings."""
    rels = [abs(v / e - 1.0) for v, e in zip(values, exacts)]
    return [math.log(a / b) / math.log(2.0) for a, b in zip(rels, rels[1:])], rels


def _rl_ic(dt):
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    b.add_inductor("L", "a", "gnd", 1e-3, i0=5.0)
    return float(np.asarray(p.simulate(b, t_end=5 * TAU, dt=dt, engine="pwl").i("L"))[-1])


def _rc_ic(dt):
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1000.0)
    b.add_capacitor("C", "a", "gnd", 1e-6, c0=5.0)
    return float(np.asarray(p.simulate(b, t_end=5 * TAU, dt=dt, engine="pwl").v("a"))[-1])


def _rl_step(dt):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_inductor("L", "m", "gnd", 1e-3)
    return float(np.asarray(p.simulate(b, t_end=5 * TAU, dt=dt, engine="pwl").i("L"))[-1])


def _rc_step(dt):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "s", "gnd", 1.0)
    b.add_resistor("R", "s", "a", 1000.0)
    b.add_capacitor("C", "a", "gnd", 1e-6)
    return float(np.asarray(p.simulate(b, t_end=5 * TAU, dt=dt, engine="pwl").v("a"))[-1])


def _rl_sine(dt):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0, v_amplitude=1.0,
                              frequency=200.0, phase=0.0)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_inductor("L", "m", "gnd", 1e-3)
    return float(np.asarray(p.simulate(b, t_end=5 * TAU, dt=dt, engine="pwl").i("L"))[-1])


def _rl_sine_exact():
    w, R, L, T = 2 * math.pi * 200.0, 1.0, 1e-3, 5 * TAU
    Z2 = R * R + (w * L) ** 2
    def i_ss(t):
        return (R * math.sin(w * t) - w * L * math.cos(w * t)) / Z2
    return i_ss(T) - i_ss(0.0) * math.exp(-T / (L / R))


DECAY = 5.0 * math.exp(-5.0)
RISE = 1.0 - math.exp(-5.0)

CASES = [
    # name, run, analytic value at t_end
    ("RL from i0 (no source)", _rl_ic, DECAY),
    ("RC from c0 (no source)", _rc_ic, DECAY),
    ("RL driven from rest by a DC step", _rl_step, RISE),
    ("RC driven from rest by a DC step", _rc_step, RISE),
    ("RL driven by a sine from rest", _rl_sine, _rl_sine_exact()),
]


@pytest.mark.xfail(strict=True,
                   reason="the fixed engine is globally FIRST order from an "
                          "inconsistent start: the companion history seeds an "
                          "inductor's v_prev and a capacitor's i_prev to zero. "
                          "See the module docstring for the measured table and "
                          "for why the cheap backward-Euler start was rejected.")
@pytest.mark.parametrize("name,run,exact", CASES[:4], ids=[c[0] for c in CASES[:4]])
def test_the_fixed_engine_is_second_order_from_every_start(name, run, exact):
    vals = [run(dt) for dt in DTS]
    orders, rels = _order(vals, [exact] * len(DTS))
    assert all(o > 1.85 for o in orders), (name, orders, rels)
    # Not just the slope: the constant must be small too. First order
    # put the coarsest step at 2e-3 relative on the decay cases.
    assert rels[0] < 1e-5, (name, rels)


def test_the_first_step_is_half_a_backward_euler_step():
    """What the first step actually solves today: i_1 = i_0 / (1 +
    h·R/(2L)) — HALF a backward-Euler step, because the trapezoidal
    conductance is used with a history that has no v_prev in it. A
    genuine backward-Euler start would give i_0 / (1 + h·R/L) and an
    O(h^2) local error; this has an O(h) one, and that is the whole
    defect. Pinned so the fix is visible when it lands."""
    h, R, L, i0 = 1e-6, 1.0, 1e-3, 5.0
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", R)
    b.add_inductor("L", "a", "gnd", L, i0=i0)
    i = np.asarray(p.simulate(b, t_end=4 * h, dt=h, engine="pwl").i("L"))
    assert i[0] == pytest.approx(i0, rel=1e-15)
    assert i[1] == pytest.approx(i0 / (1.0 + h * R / (2.0 * L)), rel=1e-12)
    # Step 2 onward IS a clean trapezoidal recursion — the history is
    # consistent from t_1 on. Only the first step is wrong, and it
    # leaves a permanent multiplicative offset on everything after it.
    a = h * R / (2.0 * L)
    assert i[2] == pytest.approx(i[1] * (1.0 - a) / (1.0 + a), rel=1e-12)


def test_the_control_a_consistent_start_is_second_order():
    """The decisive control. A sine source is 0 V at t = 0, so the
    zero-seeded v_prev is the RIGHT value and nothing is inconsistent
    — and there the engine is cleanly second order. This is what
    proves the stepper is fine and the START is the defect."""
    vals = [_rl_sine(dt) for dt in DTS]
    orders, rels = _order(vals, [_rl_sine_exact()] * len(DTS))
    assert all(o > 1.9 for o in orders), (orders, rels)
    assert rels[0] < 1e-6, rels


def test_a_resumed_run_keeps_its_exact_first_step():
    """The snapshot carries the whole history, so a resumed run starts
    consistent and must NOT take the backward-Euler step — that would
    make `resume_from` inexact, which is the property it exists for."""
    def build():
        b = p.CircuitBuilder()
        b.add_resistor("R", "a", "gnd", 1.0)
        b.add_inductor("L", "a", "gnd", 1e-3, i0=5.0)
        b.add_capacitor("C", "a", "gnd", 1e-9)
        return b

    dt = 2e-6
    full = p.simulate(build(), t_end=1e-3, dt=dt, engine="pwl")
    first = p.simulate(build(), t_end=2e-4, dt=dt, engine="pwl")
    second = p.simulate(build(), t_end=1e-3, dt=dt, engine="pwl",
                        resume_from=first.final_snapshot)
    tf = np.asarray(full.times)
    xf = np.asarray(full.states)[tf >= second.times[0] - 0.5 * dt]
    xs = np.asarray(second.states)
    n = min(len(xf), len(xs))
    err = float(np.abs(xf[:n] - xs[:n]).max())
    assert err <= 1e-9 * float(np.abs(xf[:n]).max()), err
