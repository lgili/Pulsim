"""Charge-based Coss, and the ZVS verdict it changes (audit C.1).

A grep for `coss|qrr|tail|miller` over the kernel used to return
nothing: every device was static I-V, so the resonant transition
did not exist in the waveforms and ZVS of an LLC, DAB or PSFB was
unsimulable.

The thing a linear capacitor cannot stand in for is CHARGE. What
decides whether a half-bridge reaches zero volts inside its dead
time is `Q(V) = ∫C dv`, not the small-signal `C` a datasheet
quotes at the operating point. For a planar junction
(C0 = 2 nF, V0 = 25 V, m = 0.5) at 400 V those differ by 1.61x —
which is the difference between a design that reads as clean ZVS
and one that is hard-switching with 17 µJ per edge.
"""

import numpy as np
import pytest

import pulsim as p

C0, V0, M = 2000e-12, 25.0, 0.5


def _q_closed(v, C0=C0, V0=V0, m=M):
    e = 1.0 - m
    return C0 * V0 / e * ((1.0 + v / V0) ** e - 1.0)


def _c_of(v, C0=C0, V0=V0, m=M):
    return C0 / (1.0 + v / V0) ** m


def _ramp(cap_kind, i_src=6.0, t_end=80e-9, dt=2e-11):
    """Charge the device from 0 V with a constant current.

    A constant current into a capacitor is the cleanest probe
    there is: the voltage reached after time t satisfies
    Q(v) = I·t exactly, whatever C(v) does, so the answer can be
    checked against the closed form rather than against another
    simulation.
    """
    b = p.CircuitBuilder()
    # from->to drains `to`, so charging node "n" means n->gnd
    # (established against a linear capacitor, which gives the
    # same sign — the model's stamp was right; the first version
    # of this fixture was not).
    b.add_current_source("Isrc", "n", "gnd", i_src)
    if cap_kind == "nonlinear":
        b.add_nonlinear_capacitor("Coss", "n", "gnd", C0, V0, M)
    elif cap_kind == "linear_datasheet":
        # A linear cap at the datasheet value quoted for 400 V —
        # the substitution this test exists to discredit.
        b.add_capacitor("Coss", "n", "gnd", _c_of(400.0))
    else:
        b.add_capacitor("Coss", "n", "gnd", C0)
    b.add_resistor("Rleak", "n", "gnd", 1e12)
    res = p.simulate(b, t_end=t_end, dt=dt, engine="pwl")
    return np.asarray(res.times), np.asarray(res.v("n"))


def test_charge_is_conserved_exactly():
    """Q(v(t)) must equal I·t at every sample — the companion is
    written on charge precisely so this holds however sharply
    C(v) varies."""
    i_src, dt = 6.0, 2e-11
    t, v = _ramp("nonlinear", i_src=i_src, dt=dt)
    q_sim = _q_closed(v)
    # The exact statement, not a loose one: the trapezoidal
    # companion starts from i = 0 while the source is already
    # delivering, so it is short by exactly half a step of charge
    # for the whole run — Q(v(t)) = I·(t − h/2), forever, with no
    # accumulating drift. (That half-step shows up as a 1/(2k)
    # relative error at sample k, which is how it was identified:
    # 0.25 at k = 2, 0.167 at k = 3.)
    q_ref = i_src * (t - dt / 2.0)
    ok = t > 10 * dt
    err = np.abs(q_sim[ok] - q_ref[ok]) / q_ref[ok]
    assert err.max() < 1e-6, err.max()


def test_a_linear_datasheet_cap_gets_the_dead_time_wrong():
    """The measurement that motivates the device.

    Charging at 6 A, the charge-accurate device needs 52 ns to
    reach 400 V; a linear cap at Coss(400 V) claims 32 ns. Trust
    the linear one and the switch turns on with the node still
    around 200 V.
    """
    i_src = 6.0
    t_nl, v_nl = _ramp("nonlinear", i_src=i_src)
    t_lin, v_lin = _ramp("linear_datasheet", i_src=i_src)

    def t_to(t, v, target):
        k = int(np.argmax(v >= target))
        assert v[k] >= target, "never reached the target"
        return float(t[k])

    t_true = t_to(t_nl, v_nl, 400.0)
    t_claimed = t_to(t_lin, v_lin, 400.0)
    assert t_true == pytest.approx(52e-9, rel=0.05)
    assert t_claimed == pytest.approx(32e-9, rel=0.05)
    assert t_true / t_claimed == pytest.approx(1.61, rel=0.05)

    # And the consequence, stated the way a designer meets it: at
    # the dead time the linear model endorses, where is the node?
    v_at_claimed = float(np.interp(t_claimed, t_nl, v_nl))
    assert 180.0 < v_at_claimed < 240.0        # measured ~209 V


def test_m_zero_is_exactly_a_linear_capacitor():
    """The model reduces, so the A/B is the same device."""
    b = p.CircuitBuilder()
    b.add_current_source("I", "n", "gnd", 1e-3)
    b.add_nonlinear_capacitor("C", "n", "gnd", 1e-9, 10.0, 0.0)
    b.add_resistor("R", "n", "gnd", 1e12)
    nl = p.simulate(b, t_end=1e-6, dt=1e-9, engine="pwl")

    b2 = p.CircuitBuilder()
    b2.add_current_source("I", "n", "gnd", 1e-3)
    b2.add_capacitor("C", "n", "gnd", 1e-9)
    b2.add_resistor("R", "n", "gnd", 1e12)
    lin = p.simulate(b2, t_end=1e-6, dt=1e-9, engine="pwl")

    assert np.abs(np.asarray(nl.v("n"))
                   - np.asarray(lin.v("n"))).max() < 1e-9


def test_it_runs_on_the_variable_step_engine():
    """The resonant transition is exactly what wants an adaptive
    step, so the Coss carries the BDF2 CHARGE history term through
    TR-BDF2's second stage: `(c1·Q(v) + c2·Q_γ + c3·Q_n)/h`, not
    the trapezoidal one. The conductance is shared — `c1/h = 2/γh`
    is the identity the whole method rests on — which is exactly
    what makes the wrong version look healthy: same matrix, same
    sparsity, converging Newton, wrong answer.

    Charge conservation is the check that can tell them apart.
    """
    b = p.CircuitBuilder()
    b.add_current_source("I", "n", "gnd", 6.0)
    b.add_nonlinear_capacitor("Coss", "n", "gnd", C0, V0, M)
    b.add_resistor("R", "n", "gnd", 1e12)
    res = p.simulate(b, t_end=60e-9, engine="trbdf2")
    assert res.engine_used == "trbdf2"
    t = np.asarray(res.times)
    v = np.asarray(res.v("n"))
    ok = t > t[-1] * 0.2
    err = np.abs(_q_closed(v[ok]) - 6.0 * t[ok]) / (6.0 * t[ok])
    assert err.max() < 5e-3, err.max()


def test_no_dt_needed_for_a_resonant_transition():
    """The payoff: `engine='auto'` picks the variable engine for a
    Coss circuit now, so nobody has to guess a step fine enough
    for a 50 ns transition."""
    b = p.CircuitBuilder()
    b.add_current_source("I", "n", "gnd", 6.0)
    b.add_nonlinear_capacitor("Coss", "n", "gnd", C0, V0, M)
    b.add_resistor("R", "n", "gnd", 1e12)
    res = p.simulate(b, t_end=60e-9)          # no dt, no engine
    assert res.engine_used == "trbdf2"
    v = np.asarray(res.v("n"))
    # 6 A for 60 ns is 360 nC; Q(400 V) is 312 nC, so it clears
    # the rail inside the window.
    assert v.max() > 400.0


def test_degenerate_parameters_are_refused():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "n", "gnd", 1.0)
    for kwargs, what in (
        (dict(C0=0.0, V0=25.0), "C0"),
        (dict(C0=1e-9, V0=0.0), "V0"),
        (dict(C0=1e-9, V0=25.0, m=1.0), "m must be"),
        (dict(C0=1e-9, V0=25.0, m=-0.1), "m must be"),
    ):
        with pytest.raises(Exception, match=what):
            b.add_nonlinear_capacitor("Cbad", "n", "gnd", **kwargs)
