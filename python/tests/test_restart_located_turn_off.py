"""A diode turn-off inside a restart step's second stage is located (restart line, item 1).

The 1 nH boost (review rev_n item 4): the gate opens S, the inductor's current
charges Co through D and reaches 0 inside the restart step's stage 2. SDIRK's
stage 2 starts from x0* = x0 + c (x1 - x0), c = (1 - gamma)/gamma = 2.414: stage
1's charging slope extrapolated past the turn-off, and the OFF solve keeps it
(the 2.1.0 release: v_out +7.8 % here, +22 % on the reviewer's circuit, every
dt that puts the turn-off in stage 2). The step is cut at the located turn-off
and restarted there.

Truth (closed form): before T_E, i_L = V/R_on (steady), v_out = V_c0 e^(-t/RC)
(D reverse-biased); after it, the linear (i, v) system of the ON diode,
L di/dt = V - R_on i - v, C dv/dt = i - v/R, as an eigen-decomposition, until
i = 0 (bisection), then v e^(-(t - t0)/RC).
"""

import warnings

import numpy as np
import pytest

import pulsim as p

L, C, R, V, RON, I0, VC0, T_E = 1e-9, 1e-6, 20.0, 12.0, 0.01, 1200.0, 30.0, 2e-6
PROBES = (T_E + 1e-7, T_E + 4e-7)


def _truth(ts):
    v_e = VC0 * np.exp(-T_E / (R * C))
    A = np.array([[-RON / L, -1 / L], [1 / C, -1 / (R * C)]])
    xp = -np.linalg.solve(A, np.array([V / L, 0.0]))
    w, Q = np.linalg.eig(A)
    k = np.linalg.solve(Q, np.array([I0, v_e]) - xp)

    def on(s):
        return (xp + Q @ (k * np.exp(w * s))).real
    lo, hi = 0.0, 1e-7
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if on(m)[0] > 0 else (lo, m)
    s0 = 0.5 * (lo + hi)
    v0 = on(s0)[1]
    return s0, np.array([on(t - T_E)[1] if t - T_E < s0 else
                         v0 * np.exp(-(t - T_E - s0) / (R * C)) for t in ts])


def _boost(dt, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V)
    b.add_inductor("Lc", "in", "sw", L, I0)
    b.add_switch("S", "sw", "gnd", 1 / RON, 1e-9)
    b.add_diode("Dd", "sw", "out", 1 / RON, 1e-9)
    b.add_capacitor("Co", "out", "gnd", C, VC0)
    b.add_resistor("R", "out", "gnd", R)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t < T_E)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=T_E + 4e-7 + 1e-12, dt=dt, switch_fn=fn, engine="pwl",
                       discontinuity_restart=True, **kw)
    t = np.asarray(r.times)
    return r, t, np.interp(PROBES, t, np.asarray(r.v("out")))


# SDIRK(2,2)'s local error constant on y' = lambda*y: R(z) - e^z = C3 z^3 + ...
C3 = 0.0404
OMEGA = 1 / np.sqrt(L * C)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("dt", [1e-7, 5e-8, 2.5e-8, 1e-8])
def test_a_turn_off_inside_a_restart_stage_is_located(dt, split):
    """Bound (second order at dt): the ON phase (t_on = 33.9 ns) on steps of h_on =
    min(dt, t_on), each with the local error C3 (omega h_on)^3 of the swing's amplitude
    A = I0 sqrt(L/C): |err| <= C3 (omega h_on)^2 (omega t_on) A / v. 2.1.0 / 5c2bca28
    (the stage-2 pseudo-state kept): +7.8e-2 at 0.1 us and +8.8e-2 at 50 ns, 2.0 and 2.3
    times the bound; located: 8.2e-3 / 8.2e-3 / 3.7e-3 / 8.5e-4."""
    t_on, want = _truth(PROBES)
    r, _t, got = _boost(dt, enable_substep_state_correction=split)
    assert got[0] > 45.0   # D conducted: Co charged from 27.1 V
    h_on = min(dt, t_on)
    bound = C3 * (OMEGA * h_on) ** 2 * (OMEGA * t_on) * I0 * np.sqrt(L / C) / want[0]
    err = np.abs(got / want - 1)
    assert np.all(err <= bound), (err, bound)
    if not split:   # (with the split a trapezoidal interval may locate it)
        assert r.counters["n_restart_cuts"] >= 1
    assert [x for x in r.reinit_records if x.t > 1e-6] == []   # no residual cut as a jump


def test_the_cut_converges_at_second_order():
    """The same bound at small dt, where it is the asymptotic one: 2.5 ns (omega h 0.079)."""
    t_on, want = _truth(PROBES)
    _r, _t, got = _boost(2.5e-9)
    bound = C3 * (OMEGA * 2.5e-9) ** 2 * (OMEGA * t_on) * I0 * np.sqrt(L / C) / want[0]
    assert np.all(np.abs(got / want - 1) <= bound)


def test_without_a_turn_off_in_the_step_nothing_is_cut():
    """Control: the same gate edge (a restart step) with nothing to commutate (V = 0,
    i_L = 0: D stays reverse-biased): no cut, v_out the RC decay (closed form)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_inductor("Lc", "in", "sw", L, 0.0)
    b.add_switch("S", "sw", "gnd", 1 / RON, 1e-9)
    b.add_diode("Dd", "sw", "out", 1 / RON, 1e-9)
    b.add_capacitor("Co", "out", "gnd", C, 400.0)
    b.add_resistor("R", "out", "gnd", R)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t < T_E)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=T_E + 4e-7 + 1e-12, dt=1e-7, switch_fn=fn, engine="pwl",
                       discontinuity_restart=True)
    t, v = np.asarray(r.times), np.asarray(r.v("out"))
    assert r.counters["n_restart_cuts"] == 0
    np.testing.assert_allclose(v, 400.0 * np.exp(-t / (R * C)), rtol=1e-6)
