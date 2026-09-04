"""The saturable inductor's flux must close.

Flux linkage is a function of current alone: lambda = Lambda(i), with
Lambda(i) = integral from 0 to i of L(u) du. So for a device driven by
a voltage of zero mean,

    Lambda(t) = Lambda(0) + integral of v dt

is exactly periodic, and therefore so is the current. A pure inductor
across a zero-mean source has no mechanism to accumulate DC — not over
ten cycles and not over ten thousand.

The stamp advances the flux as L(i_new)*(i_new - i_old): a
right-endpoint rectangle rule for d(Lambda), exact only while L is
constant across a step. It is not a symmetric error. The stamp solves
for the current increment given the voltage,

    delta_i = h*(v_new + v_old) / (2*L(i_new)),

so on the ASCENDING leg L(i_new) is the smallest L of the interval and
delta_i comes out too large, while on the DESCENDING leg L(i_new) is
the largest and |delta_i| comes out too small. Both biases push the
current OUTWARD. The error rectifies: it ratchets, it does not cancel.

Measured on the old stamp, 1 kHz sine of zero mean, L_0 = 1 mH,
I_sat = 5 A, at FIVE THOUSAND steps per cycle:

    cycles       10       40      160      400
    DC current  63.6 A   72.3 A  104.7 A  145.5 A

on a device whose saturation current is 5 A, from a source with no DC
at all. The drift is first order in dt (0.562 / 0.283 / 0.141 / 0.070
A per cycle at dt = 4e-7 … 5e-8, ratios 1.99 / 2.00 / 2.00) and
unbounded in time.

A LINEAR inductor in the identical circuit drifts by 7e-15 A/cycle —
machine noise. That control matters: the fixed-step engine has a
first-order first-step artifact of its own, and it is not what this
test measures.
"""

import numpy as np

import pulsim as p

L0, ISAT, LRES = 1e-3, 5.0, 5e-5
F, VAMP = 1e3, 50.0


def _run(dt, *, saturable, n_cycles):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0,
                              v_amplitude=VAMP, frequency=F, phase=0.0)
    if saturable:
        b.add_saturable_inductor("Ls", "a", "gnd", L0, ISAT, LRES)
    else:
        b.add_inductor("Ls", "a", "gnd", L0)
    res = p.simulate(b, t_end=n_cycles / F, dt=dt, engine="pwl")
    return np.asarray(res.times), np.asarray(res.i("Ls"))


def _cycle_dc(t, i, k):
    """Mean current over drive cycle k — its DC content."""
    m = (t >= k / F) & (t < (k + 1) / F)
    assert m.sum() > 10, "cycle window too thin to average"
    return float(i[m].mean())


def test_a_linear_inductor_does_not_drift():
    """The control. Lambda = L*i makes the stamp's rule exact, so any
    drift here would be the ENGINE, not the flux rule — and the
    saturable numbers below would mean nothing."""
    t, i = _run(2e-7, saturable=False, n_cycles=40)
    first, last = _cycle_dc(t, i, 0), _cycle_dc(t, i, 39)
    assert abs(last - first) / 40 < 1e-9, (first, last)


def test_the_saturable_inductor_does_not_ratchet():
    """The regression. Zero-mean drive, so the DC content of the
    current must not grow."""
    t, i = _run(2e-7, saturable=True, n_cycles=40)
    first, last = _cycle_dc(t, i, 0), _cycle_dc(t, i, 39)
    # It must actually be saturating, or this proves nothing.
    assert np.abs(i).max() > 5 * ISAT, np.abs(i).max()
    drift = abs(last - first) / 40
    # Measured after the fix: 3.2e-10 A per cycle, and the per-cycle
    # DC is bit-identical across cycles 0, 1, 2 and 39. The bound
    # leaves four orders of headroom over that and still fails the
    # old stamp by five.
    assert drift < 1e-7 * abs(first), (first, last, drift)


def test_the_drift_does_not_grow_without_bound():
    """Ten times the cycles must not mean ten times the error. On the
    old stamp this went 63.6 A -> 145.5 A between 10 and 400 cycles."""
    short = _cycle_dc(*_run(2e-7, saturable=True, n_cycles=10), 9)
    long_ = _cycle_dc(*_run(2e-7, saturable=True, n_cycles=200), 199)
    assert abs(long_ - short) < 1e-2 * abs(short), (short, long_)


def test_flux_returns_when_the_current_does():
    """Path independence, stated directly: bring the device deep into
    saturation and back to the same current, and the flux -- the
    integral of v dt -- must return to where it started."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0,
                              v_amplitude=VAMP, frequency=F, phase=0.0)
    b.add_saturable_inductor("Ls", "a", "gnd", L0, ISAT, LRES)
    res = p.simulate(b, t_end=8 / F, dt=1e-7, engine="pwl")
    t = np.asarray(res.times)
    i = np.asarray(res.i("Ls"))
    # One full drive period, from the end of the first cycle.
    m = (t >= 4 / F) & (t <= 5 / F)
    peak = float(np.abs(i[m]).max())
    assert peak > 5 * ISAT
    # Same current at both ends of a closed excursion. Measured
    # against the PEAK, not against the endpoint value: at this phase
    # the current is legitimately ~zero, so a relative comparison of
    # the two endpoints compares two numbers that both ought to
    # vanish. On the rectangle-rule stamp the gap was 1.4e-2 A on a
    # 190 A peak; with the flux difference it is 3e-11 A.
    assert abs(i[m][-1] - i[m][0]) < 1e-8 * peak, (i[m][0], i[m][-1], peak)
