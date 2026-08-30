"""Periodic steady state by shooting on the monodromy map (F.2).

Getting a converter to steady state costs a long transient today
— and you must guess how long. The one-period map is affine while
the switching pattern holds, so the periodic orbit is ONE linear
solve, `(I - Phi) h* = c`, with no shooting iteration.

Two things this file pins that were learned by measuring:

* The map's state is the COMPANION HISTORY, not the MNA vector.
  A monodromy built on `x` converges to the fixed point of a
  different dynamical system — one that resets its history every
  period. It is stable to 1e-15 across six periods and still
  0.14% wrong, which is precisely what makes it dangerous.
* A gate schedule whose duty edge lands on a different STEP in
  different periods makes the system not T-periodic at all, so
  no exact orbit exists. The one-period residual cannot see it
  (that period is self-consistent); the multi-period drift can,
  and it is reported by name rather than returned as a number.
"""

import time

import numpy as np
import pytest

import pulsim as p

T = 1e-5          # 100 kHz
DT = 2e-8         # 500 steps per period
NSTEP = int(round(T / DT))
NON = int(round(0.42 * NSTEP))


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "vout", 220e-6)
    b.add_capacitor("Cout", "vout", "gnd", 220e-6)
    b.add_resistor("R_L", "vout", "gnd", 8.0)
    return b


def _grid_gate(b):
    """A duty edge pinned to a STEP INDEX: exactly T-periodic."""
    n = b.graph.num_switches
    q = b.switch_index_of("Q1")

    def sf(t):
        m = p.SwitchStateMask(n)
        m.set(q, (int(round(t / DT)) % NSTEP) < NON)
        return m

    return sf


def _float_gate(b):
    """The common idiom, whose edge jitters by a step."""
    n = b.graph.num_switches
    q = b.switch_index_of("Q1")

    def sf(t):
        m = p.SwitchStateMask(n)
        m.set(q, (t / T) % 1.0 < 0.42)
        return m

    return sf


def test_orbit_is_a_fixed_point_and_stays_one():
    b = _buck()
    r = p.steady_state(b, period=T, dt=DT, switch_fn=_grid_gate(b))
    assert r.residual < 1e-8
    assert r.drift < 1e-8
    assert 0.0 < r.floquet_radius < 1.0     # a stable orbit
    # 100 periods from it must not move.
    far = p.simulate(b, t_end=r.snapshot.t + 100 * T, dt=DT,
                      switch_fn=_grid_gate(b),
                      resume_from=r.snapshot)
    h_far = np.asarray(list(far.final_snapshot.history))
    assert np.abs(h_far - r.history).max() < 1e-8


def test_brute_force_converges_toward_the_orbit():
    """The honest comparison. Settling is SLOW here — the Floquet
    radius is 0.997, a 350-period time constant — so a transient
    of any affordable length has not arrived yet. What can be
    asserted is that it keeps moving toward `h*`, which is also
    the argument that `h*` is the better answer: the full 80 ms
    reference (10 s of compute, 8000 periods) lands on
    5.03897 .. 5.03973, and `h*` = 5.03929 sits inside it.
    """
    b = _buck()
    t0 = time.perf_counter()
    r = p.steady_state(b, period=T, dt=DT, switch_fn=_grid_gate(b))
    t_ss = time.perf_counter() - t0
    v_star = r.history[2]                    # v_prev of Cout

    gaps = []
    for t_end in (2e-3, 4e-3, 8e-3):
        ref = p.simulate(_buck(), t_end=t_end, dt=DT,
                          switch_fn=_grid_gate(b))
        vv = np.asarray(ref.v("vout"))
        gaps.append(abs(float(vv[-1]) - v_star))
    assert gaps[0] > gaps[1] > gaps[2]        # converging toward it
    assert gaps[2] < 0.05                      # and getting close

    # And it costs a handful of one-period runs, not thousands.
    assert r.n_period_runs < 20
    assert t_ss < 1.0


def test_the_snapshot_starts_a_run_in_steady_state():
    b = _buck()
    r = p.steady_state(b, period=T, dt=DT, switch_fn=_grid_gate(b))
    run = p.simulate(b, t_end=r.snapshot.t + 5 * T, dt=DT,
                      switch_fn=_grid_gate(b),
                      resume_from=r.snapshot)
    v = np.asarray(run.v("vout"))
    t = np.asarray(run.times)
    first = v[t < t[0] + T]
    last = v[t > t.max() - T]
    # No settling: the first period's band already equals the
    # last's, which is the whole point.
    assert abs(first.mean() - last.mean()) < 1e-6


def test_a_schedule_that_is_not_grid_periodic_is_named():
    """The one-period residual is tiny and the answer is still
    wrong, because the SYSTEM is not T-periodic. Silence here
    would be a confident wrong number."""
    b = _buck()
    with pytest.raises(ValueError, match="not exactly"):
        p.steady_state(b, period=T, dt=DT, switch_fn=_float_gate(b))


def test_the_one_period_answer_can_still_be_asked_for():
    b = _buck()
    r = p.steady_state(b, period=T, dt=DT,
                        switch_fn=_float_gate(b), verify=False)
    assert np.isfinite(r.history).all()
    assert np.isnan(r.drift)


def test_a_circuit_with_no_affine_orbit_is_refused():
    """A diode whose conduction window moves under the probe makes
    the one-period map non-affine, so there is no fixed point to
    find — and saying so is the only honest output."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 10.0,
                               1.0 / T, 0.0)
    b.add_diode("D", "ac", "out", 1e3, 1e-9, 0.7)
    b.add_capacitor("Co", "out", "gnd", 1e-6)
    b.add_resistor("Rl", "out", "gnd", 100.0)
    n = b.graph.num_switches

    def sf(t):
        return p.SwitchStateMask(n)

    # The pattern check cannot see it here — `commutation_events`
    # comes back EMPTY for this circuit even though the diode
    # plainly conducts and blocks — so the residual backstop is
    # what refuses: one period from the "answer" moves it by 1.2,
    # which is not a fixed point of anything.
    with pytest.raises(ValueError, match="no periodic orbit"):
        p.steady_state(b, period=T, dt=DT, switch_fn=sf,
                        probe_scale=50.0)


def test_argument_errors_name_what_is_wrong():
    b = _buck()
    with pytest.raises(ValueError, match="period must be"):
        p.steady_state(b, period=0.0, dt=DT)
    with pytest.raises(ValueError, match="dt must be"):
        p.steady_state(b, period=T, dt=0.0)
    with pytest.raises(ValueError, match="caricature"):
        p.steady_state(b, period=T, dt=T / 2)

    static = p.CircuitBuilder()
    static.add_voltage_source("V", "a", "gnd", 5.0)
    static.add_resistor("R", "a", "gnd", 1.0)
    with pytest.raises(ValueError, match="no dynamic devices"):
        p.steady_state(static, period=T, dt=DT)
