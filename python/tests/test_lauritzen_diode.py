"""A diode that actually recovers (audit C.1).

Every diode Pulsim had was static I-V — the PWL `add_diode`, the
smooth-blend nonlinear one, and the exponential `add_shockley_diode`
all compute current from the present voltage alone. A static law
CANNOT recover, because recovery is stored charge leaving the
device, and a static law stores nothing.

Measured on a double-pulse test (400 V rail, 20 A clamped
inductive load, 50 nH commutation loop, low-side switch turning
on) before this model existed:

    diode                reverse peak     Q_rr
    add_diode (PWL)        0.00000 A        0
    add_shockley_diode     0.00000 A        0

The 20 A commutates straight to zero. A real 600 V fast-recovery
Si part at that di/dt sweeps out several microcoulombs first,
peaking 15-30 A NEGATIVE — and that current flows through the
turning-on SWITCH, where it usually dominates turn-on loss. A
simulator reporting zero is not slightly optimistic about
hard-switched efficiency; it is silent about the largest term.

The Lauritzen-Mattsson model makes charge the state:

    i    = (q_E - q_M) / T_M
    dq_M = (q_E - q_M) / T_M - q_M / tau
    dt

so forcing the current negative leaves q_M unable to follow, and
i = -q_M/T_M is the reverse spike.
"""

import numpy as np
import pytest

import pulsim as p

TAU = 1e-7      # carrier lifetime — how long recovery lasts
T_M = 1e-8      # transit time — how hard the peak is


# ---------------------------------------------------------------
# The DC curve must be unchanged: only the dynamics are new.
# ---------------------------------------------------------------

def _dc_current(add, v_fwd, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "a", "gnd", v_fwd)
    b.add_resistor("R1", "a", "k", 1.0)
    add(b, **kw)
    res = p.simulate(b, t_end=1e-6, dt=1e-7, engine="pwl")
    return -float(np.asarray(res.i("V1"))[-1])


def test_the_dc_curve_matches_a_shockley_junction():
    """In steady state dq_M/dt = 0, so this reduces to an ordinary
    Shockley diode with I_S scaled by tau/(tau + T_M). If that
    were not true the model would be a different device at DC,
    and every conduction-loss number would move."""
    scale = TAU / (TAU + T_M)
    for v in (1.0, 5.0, 20.0):
        i_l = _dc_current(
            lambda b: b.add_lauritzen_diode(
                "D", "k", "gnd", tau=TAU, T_M=T_M), v)
        i_s = _dc_current(
            lambda b: b.add_shockley_diode(
                "D", "k", "gnd", I_S=1e-12 * scale), v)
        assert i_l == pytest.approx(i_s, rel=1e-3), (v, i_l, i_s)


def test_forward_drop_still_rises_with_current():
    """The exponential is still an exponential."""
    v_f = []
    for v_src in (1.0, 10.0, 100.0):
        b = p.CircuitBuilder()
        b.add_voltage_source("V1", "a", "gnd", v_src)
        b.add_resistor("R1", "a", "k", 1.0)
        b.add_lauritzen_diode("D", "k", "gnd", tau=TAU, T_M=T_M)
        res = p.simulate(b, t_end=1e-6, dt=1e-7, engine="pwl")
        v_f.append(float(np.asarray(res.v("k"))[-1]))
    assert v_f[0] < v_f[1] < v_f[2]
    # ~60 mV per decade, so two decades is ~120 mV.
    assert 0.08 < v_f[2] - v_f[0] < 0.20, v_f


# ---------------------------------------------------------------
# The point of the model.
# ---------------------------------------------------------------

def _double_pulse(add_diode, t_end=1.5e-6, dt=2e-10):
    """Low-side switch under test, freewheeling diode to the rail.
    20 A circulates while the switch is off; turning it on
    commutates the diode."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 400.0)
    b.add_inductor("Lload", "dc", "sw", 200e-6, i0=20.0)
    b.add_inductor("Lstray", "d2", "dc", 50e-9, i0=20.0)
    add_diode(b)
    b.add_switch("S1", "sw", "gnd", 1e3, 1e-9)
    b.add_resistor("Rsnub", "sw", "gnd", 1e5)
    n, i_s1 = b.graph.num_switches, b.switch_index_of("S1")

    def sf(t):
        m = p.SwitchStateMask(n)
        if t >= 1e-6:
            m.set(i_s1, True)
        return m

    res = p.simulate(b, t_end=t_end, dt=dt, engine="pwl",
                     switch_fn=sf)
    t = np.asarray(res.times)
    i = np.asarray(res.i("Lstray"))
    return t, i


def test_the_static_diodes_do_not_recover_at_all():
    """The baseline this model exists to fix — pinned so it cannot
    be mistaken for a modelling subtlety later."""
    for add in (
        lambda b: b.add_diode("D1", "sw", "d2", 1e3, 1e-9, 0.7),
        lambda b: b.add_shockley_diode("D1", "sw", "d2"),
    ):
        _, i = _double_pulse(add)
        assert i.min() > -1e-3, i.min()


def test_it_recovers():
    """The property: a reverse current spike that a static law
    cannot produce."""
    _, i = _double_pulse(
        lambda b: b.add_lauritzen_diode("D1", "sw", "d2",
                                         tau=TAU, T_M=T_M))
    assert i.min() < -1.0, i.min()


def test_recovered_charge_is_the_charge_that_was_stored():
    """Q_rr is not a free parameter — it is the stored charge, and
    at a steady forward current that is i*tau. The integral of the
    reverse current must come back to it."""
    t, i = _double_pulse(
        lambda b: b.add_lauritzen_diode("D1", "sw", "d2",
                                         tau=TAU, T_M=T_M))
    neg = np.minimum(i, 0.0)
    q_rr = -np.trapezoid(neg, t)
    q_stored = 20.0 * TAU
    assert q_rr == pytest.approx(q_stored, rel=0.5), (q_rr,
                                                       q_stored)


def test_the_peak_follows_the_square_root_of_di_dt():
    """The strongest check available, and the one that says the
    dynamics are right rather than merely non-zero.

    Classical recovery theory gives I_rrm = sqrt(2*Q_stored*di/dt)
    when all the stored charge leaves through the terminal. Here
    di/dt is set by the rail over the commutation loop, so
    sweeping the loop inductance sweeps di/dt over two decades:

        L_loop   di/dt        I_rrm     sqrt(2*Q*di/dt)   ratio
          50 nH  8000 A/us   106 A          179 A          0.59
         400 nH  1000 A/us    42 A           63 A          0.66
           4 uH   100 A/us   8.6 A           20 A          0.43

    The ratio sits near a constant because the model also lets
    charge leave by RECOMBINATION, which the idealised formula
    ignores; it falls at low di/dt because a slower recovery gives
    recombination longer to work. Both are the physics, and the
    same effect shows up as Q_rr < I_F*tau.

    (This also settled a scare: a reverse peak of 106 A on a 20 A
    commutation looked absurd until the sweep showed it was the
    FIXTURE — a 50 nH loop across 400 V is 8000 A/us, far past any
    real gate drive.)
    """
    peaks = []
    for l_loop in (4e-6, 1e-6, 2e-7):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vdc", "dc", "gnd", 400.0)
        b.add_inductor("Lload", "dc", "sw", 200e-6, i0=20.0)
        b.add_inductor("Lstray", "d2", "dc", l_loop, i0=20.0)
        b.add_lauritzen_diode("D1", "sw", "d2", tau=TAU, T_M=T_M)
        b.add_switch("S1", "sw", "gnd", 1e3, 1e-9)
        b.add_resistor("Rsnub", "sw", "gnd", 1e5)
        n, k = b.graph.num_switches, b.switch_index_of("S1")

        def sf(t, k=k, n=n):
            m = p.SwitchStateMask(n)
            if t >= 1e-6:
                m.set(k, True)
            return m

        res = p.simulate(b, t_end=3e-6, dt=2e-10, engine="pwl",
                         switch_fn=sf)
        peaks.append(-float(np.asarray(res.i("Lstray")).min()))

    # Each step is 4x / 5x the di/dt, so the peak must roughly
    # double — not quadruple (that would be linear in di/dt) and
    # not stay put (that would be no dynamics at all).
    for lo, hi in zip(peaks, peaks[1:]):
        assert 1.4 < hi / lo < 3.2, peaks


def test_a_longer_lifetime_recovers_more_charge():
    """`tau` is the knob the datasheet gives you."""
    q = []
    for tau in (5e-8, 2e-7):
        t, i = _double_pulse(
            lambda b, tau=tau: b.add_lauritzen_diode(
                "D1", "sw", "d2", tau=tau, T_M=T_M))
        q.append(-np.trapezoid(np.minimum(i, 0.0), t))
    assert q[1] > q[0] * 2.0, q


# ---------------------------------------------------------------
# Refusals — a wrong parameter must be named, not absorbed.
# ---------------------------------------------------------------

@pytest.mark.parametrize("kw,frag", [
    ({"tau": 0.0}, "tau"),
    ({"T_M": 0.0}, "T_M"),
    ({"tau": 1e-9, "T_M": 1e-8}, "T_M must be smaller"),
    ({"I_S": 0.0}, "I_S"),
    ({"n": 0.0}, "n"),
])
def test_bad_parameters_are_refused_by_name(kw, frag):
    b = p.CircuitBuilder()
    with pytest.raises(Exception, match=frag):
        b.add_lauritzen_diode("D", "a", "gnd", **kw)


def test_a_schottky_is_not_a_tiny_tau():
    """The refusal names the alternative rather than letting a
    degenerate lifetime stand in for a device with no stored
    charge."""
    b = p.CircuitBuilder()
    with pytest.raises(Exception, match="Schottky"):
        b.add_lauritzen_diode("D", "a", "gnd", tau=0.0)
