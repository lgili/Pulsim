"""Sampled-data frequency response from the monodromy (F.2).

A Bode plot costs one transient per point today — `run_fra` needs
2.9 s per point on a 100 kHz buck, so 20 points is 58 s and a
real 100-point sweep is five minutes. The audit calls it the
control engineer's daily bottleneck.

Around the periodic orbit the sampled dynamics are linear, and
`Phi` is already computed by `steady_state`. So the whole sweep is
`H(z) = C(zI - Phi)^-1 B + D` — a handful of one-period runs
TOTAL, and the cost stops depending on the number of points.

Measured: 100 points in 36 ms (21 period runs), against 57.7 s
for 20 points, and it matches the analytic buck to 0.7%.
"""

import time

import numpy as np
import pytest

import pulsim as p

T = 1e-5
DT = 2e-8
NSTEP = int(round(T / DT))
VIN, LH, CF, RL = 12.0, 220e-6, 220e-6, 8.0
D0 = 0.42


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "vout", LH)
    b.add_capacitor("Cout", "vout", "gnd", CF)
    b.add_resistor("R_L", "vout", "gnd", RL)
    return b


def _gate_of(b):
    """duty -> switch_fn, with the edge pinned to a STEP INDEX so
    the schedule is exactly T-periodic (see test_steady_state)."""
    n = b.graph.num_switches
    q = b.switch_index_of("Q1")

    def of(duty):
        non = int(round(duty * NSTEP))

        def sf(t):
            m = p.SwitchStateMask(n)
            m.set(q, (int(round(t / DT)) % NSTEP) < non)
            return m

        return sf

    return of


def _analytic(f):
    """Ideal buck control-to-output: Vin / (1 + sL/R + s^2 LC)."""
    s = 2j * np.pi * np.asarray(f)
    return VIN / (1.0 + s * LH / RL + s ** 2 * LH * CF)


def test_matches_the_analytic_buck_through_resonance():
    b = _buck()
    f = np.logspace(1.5, np.log10(5e3), 80)
    fr = p.frequency_response(b, period=T, dt=DT,
                               switch_fn_of=_gate_of(b),
                               output="vout", frequencies=f, u0=D0)
    ref = _analytic(f)
    err = np.abs(np.abs(fr.H) - np.abs(ref)) / np.abs(ref)
    assert err.max() < 0.02          # measured 0.007
    # DC gain is Vin per unit duty for an ideal buck.
    assert abs(fr.H[0]) == pytest.approx(VIN, rel=0.02)
    # The Q = 8 LC resonance lands where it should.
    f0 = 1.0 / (2 * np.pi * np.sqrt(LH * CF))
    assert f[int(np.argmax(np.abs(fr.H)))] == pytest.approx(
        f0, rel=0.1)
    # Phase rolls through -180 after the double pole.
    assert fr.phase_deg[-1] < -150.0


def test_cost_does_not_grow_with_the_number_of_points():
    """The property that makes this usable: the period runs are
    spent once, and each extra frequency is a small linear solve.

    ASSERT THE MECHANISM, NOT THE STOPWATCH. `n_period_runs`
    being equal is the exact statement — it is what "the cost
    stops depending on how many points you ask for" MEANS, and it
    is deterministic. An earlier version of this test also
    demanded `t_big < t_small * 2.0`, which measured the CI
    runner's noise rather than this code: it came back 2.2x on a
    shared macOS box and failed a PR that had not touched this
    module. Wall clock is kept only as a coarse shape check — 40x the points
    must not cost anything like 40x the time — with best-of-3
    timing and a margin wide enough that only a real complexity
    regression can trip it.
    """
    b = _buck()
    of = _gate_of(b)

    def timed(n_points):
        best = float("inf")
        out = None
        for _ in range(3):
            t0 = time.perf_counter()
            out = p.frequency_response(
                b, period=T, dt=DT, switch_fn_of=of,
                output="vout",
                frequencies=np.logspace(2, 4, n_points), u0=D0)
            best = min(best, time.perf_counter() - t0)
        return out, best

    small, t_small = timed(10)
    big, t_big = timed(400)

    # The mechanism, exactly.
    assert small.n_period_runs == big.n_period_runs

    # The shape: 40x the points, nowhere near 40x the cost.
    assert t_big < t_small * 8.0, (t_small, t_big)


def test_reuses_a_steady_state_when_given_one():
    b = _buck()
    of = _gate_of(b)
    ss = p.steady_state(b, period=T, dt=DT, switch_fn=of(D0))
    fr = p.frequency_response(b, period=T, dt=DT, switch_fn_of=of,
                               output="vout",
                               frequencies=np.logspace(2, 3, 5),
                               u0=D0, steady=ss)
    assert fr.steady is ss
    assert np.allclose(fr.Phi, np.asarray(ss.monodromy))


def test_above_nyquist_is_refused():
    """A sampled model says nothing past 1/2T — the response
    aliases, and returning numbers there would be fiction."""
    b = _buck()
    with pytest.raises(ValueError, match="Nyquist"):
        p.frequency_response(b, period=T, dt=DT,
                              switch_fn_of=_gate_of(b),
                              output="vout",
                              frequencies=[1e3, 1.0 / T], u0=D0)


def test_an_invisible_perturbation_is_refused():
    """On a fixed grid the gate moves in whole steps, so a du
    below dt/period does not move it at all — and B is then not
    zero but WORSE: the steady state's own residual divided by a
    tiny du. Measured |B| = 0.76 out of a 7.6e-10 residual at
    du = 1e-9 — noise wearing the shape of a response. The guard
    compares the two schedules directly, so there is no threshold
    to tune."""
    b = _buck()
    with pytest.raises(ValueError, match="does not move the gate"):
        p.frequency_response(b, period=T, dt=DT,
                              switch_fn_of=_gate_of(b),
                              output="vout",
                              frequencies=[1e3], u0=D0,
                              du=1e-9)


def test_argument_errors_name_what_is_wrong():
    b = _buck()
    of = _gate_of(b)
    with pytest.raises(ValueError, match="no frequencies"):
        p.frequency_response(b, period=T, dt=DT, switch_fn_of=of,
                              output="vout", frequencies=[])
    with pytest.raises(ValueError, match="must be positive"):
        p.frequency_response(b, period=T, dt=DT, switch_fn_of=of,
                              output="vout", frequencies=[-1.0])


def test_switch_masks_compare_by_value():
    """Found while writing the guard above: `SwitchStateMask.__eq__`
    was not bound, so Python fell back to IDENTITY and two masks
    with the same bits compared False. Anything diffing schedules
    silently saw every pair as different."""
    a = p.SwitchStateMask(3)
    b = p.SwitchStateMask(3)
    assert a == b
    assert not (a != b)
    b.set(1, True)
    assert a != b
    assert not (a == b)
    assert len({a, b}) == 2          # and hashable, consistently
