"""A run's complete state, and resuming from it exactly (F.3).

`initial_state=` restores the MNA vector and then INVENTS the
companion history from it — `seed_from_dc_op` gives a capacitor
i_prev = 0 and an inductor v_prev = 0. Those are not the state:
each dynamic device carries an INDEPENDENT (v, i) pair. So
resuming from x alone does not reproduce the run, silently.

Measured on a minimal RLC: continuous 2T versus T-then-resume
differ by 2.3e-4 with `initial_state=`, and by exactly 0 with
`resume_from=`.

This is also what a steady-state / shooting method needs: the
one-period map is a function of the FULL state, so a monodromy
built on x alone converges to the fixed point of a different
dynamical system — one that resets its companion history every
period. (Caught exactly that way: the x-only map's fixed point
was stable to 1e-15 across six periods and still disagreed with
the converged brute-force answer by 0.14%.)
"""

import numpy as np
import pytest

import pulsim as p

DT = 1e-7


def _rlc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R", "in", "n1", 1e3)
    b.add_capacitor("C", "n1", "gnd", 1e-6)
    b.add_inductor("L", "n1", "gnd", 1e-3)
    return b


def test_resume_reproduces_a_continuous_run_exactly():
    full = p.simulate(_rlc(), t_end=2e-4, dt=DT, engine="pwl")
    first = p.simulate(_rlc(), t_end=1e-4, dt=DT, engine="pwl")
    second = p.simulate(_rlc(), t_end=2e-4, dt=DT, engine="pwl",
                         resume_from=first.final_snapshot)

    x_cont = np.asarray(full.states)[-1]
    x_split = np.asarray(second.states)[-1]
    assert np.array_equal(x_cont, x_split)


def test_initial_state_alone_does_not(and_that_is_the_point=None):
    """The defect this exists to fix, pinned so it cannot be
    mistaken for a resume."""
    full = p.simulate(_rlc(), t_end=2e-4, dt=DT, engine="pwl")
    first = p.simulate(_rlc(), t_end=1e-4, dt=DT, engine="pwl")
    x_mid = np.asarray(first.states)[-1].copy()
    second = p.simulate(_rlc(), t_end=1e-4, dt=DT, engine="pwl",
                         initial_state=x_mid)
    err = float(np.abs(np.asarray(full.states)[-1]
                        - np.asarray(second.states)[-1]).max())
    # It is close — which is exactly why it went unnoticed — but
    # it is not the same run.
    assert 1e-6 < err < 1e-2


def test_snapshot_carries_the_companion_history():
    res = p.simulate(_rlc(), t_end=1e-4, dt=DT, engine="pwl")
    snap = res.final_snapshot
    assert snap.valid
    assert snap.t == pytest.approx(1e-4)
    assert len(snap.x) == 4
    # 2 reals per dynamic device (C and L): the (v_prev, i_prev)
    # pairs that x alone cannot supply.
    assert len(snap.history) == 4
    assert any(abs(v) > 0 for v in snap.history)


def test_resume_carries_diode_state():
    """Diode on-bits are solver-owned; a mask cannot rebuild
    them, so a resume that dropped them would restart every diode
    from its default and re-derive a different commutation."""
    def rect():
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 10.0,
                                   50.0, 0.0)
        b.add_diode("D", "ac", "out", 1e3, 1e-9, 0.7)
        b.add_capacitor("Co", "out", "gnd", 100e-6)
        b.add_resistor("Rl", "out", "gnd", 100.0)
        return b

    full = p.simulate(rect(), t_end=30e-3, dt=1e-6, engine="pwl")
    first = p.simulate(rect(), t_end=15e-3, dt=1e-6, engine="pwl")
    assert len(first.final_snapshot.diode_on) == 1
    # t_start comes from the snapshot, so the sine source keeps
    # its phase — resuming continues TIME, not just state.
    second = p.simulate(rect(), t_end=30e-3, dt=1e-6, engine="pwl",
                         resume_from=first.final_snapshot)
    assert second.times[0] == pytest.approx(15e-3)
    # Machine precision rather than bit-identical: the resumed run
    # computes t = t_start + k*dt from t_start = 15e-3, so the
    # source's sine argument differs in its last bits. That is a
    # floating-point consequence of resuming at a non-zero time,
    # not a state that failed to carry.
    assert np.abs(np.asarray(full.states)[-1]
                   - np.asarray(second.states)[-1]).max() < 1e-12


def test_many_short_segments_equal_one_long_run():
    """The 'run 1 s, save the point, start every study there'
    workflow, which had no correct form before."""
    full = p.simulate(_rlc(), t_end=5e-4, dt=DT, engine="pwl")
    snap = None
    for k in range(1, 6):
        r = p.simulate(_rlc(), t_end=k * 1e-4, dt=DT, engine="pwl",
                        resume_from=snap)
        snap = r.final_snapshot
    assert np.array_equal(np.asarray(full.states)[-1],
                           np.asarray(r.states)[-1])


def test_snapshot_from_another_circuit_is_refused():
    small = p.simulate(_rlc(), t_end=1e-5, dt=DT, engine="pwl")
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R", "in", "gnd", 1e3)
    with pytest.raises(Exception, match="different circuit"):
        p.simulate(b, t_end=1e-5, dt=DT, engine="pwl",
                    resume_from=small.final_snapshot)
