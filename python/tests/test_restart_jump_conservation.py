"""The restart rule's jump conserves what the event conserves (fixes/jump-conservation.md).

A switch closing C1 (charged) onto C2 –R2– C3, the node set {n1, n2, n3} floating: two fast
modes (S's 1 mOhm and R2) move C2 at once. The rule's per-state a = d'/d is the one-mode
series of the jump and took a different multiple of d per state: the island's charge was
off by up to 3.4 % and its end voltage +0.05 .. +1.44 V. The jump is now the vector series
of the probe's second differences (a Krylov sequence of the stage map), which lies in their
span and keeps every charge / flux each stage keeps.

Truths are closed forms (linear) or mpmath (junction laws), derived here, not read from
the engine: the island's charge Q0 is conserved, every capacitor ends at v_inf with
Q(v_inf) summed = Q0, and the record is the energy the fast modes dissipate,
E(before) - E(after) (Tellegen: with all three at one voltage and the charge conserved,
sum of the integral of (v+ - v) dq is -dE).
"""

from __future__ import annotations

import warnings

import mpmath as mp
import numpy as np
import pytest

import pulsim as p

mp.mp.dps = 30
DT = 1e-8
V1 = 100.0
T_OPEN, T_CLOSE = 1e-6, 2e-6
LAWS = {"C1": (1e-9, 25.0, 0.5), "C2": (2e-9, 10.0, 1 / 3), "C3": (1e-9, 10.0, 0.5)}
NODE = {"C1": "n1", "C2": "n2", "C3": "n3"}


def _law(name, nl):
    c0, v0, m = LAWS[name]
    if not nl:
        return lambda v: mp.mpf(c0)
    return lambda v: mp.mpf(c0) / (1 + mp.mpf(v) / v0) ** m


def _q(name, v, nl):
    c0, v0, m = LAWS[name]
    if not nl:
        return mp.mpf(c0) * v
    return mp.mpf(c0) * v0 / (1 - m) * ((1 + mp.mpf(v) / v0) ** (1 - m) - 1)


def _w(name, v, nl):
    c = _law(name, nl)
    return mp.quad(lambda u: u * c(u), [0, v])


def _island(R2, nl, split):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vs", "vs", "gnd", 0.0, V1, 0.0, 1.0, 0.0, 1e-7, 0.0)
    b.add_resistor("Rs", "vs", "a", 10.0)
    b.add_switch("Ssrc", "a", "n1", 1e3, 1e-12)
    b.add_switch("S", "n1", "n2", 1e3, 1e-12)
    b.add_resistor("R2", "n2", "n3", R2)
    for n, (c0, v0, m) in LAWS.items():
        if nl:
            b.add_nonlinear_capacitor(n, NODE[n], "gnd", c0, v0, m)
        else:
            b.add_capacitor(n, NODE[n], "gnd", c0)
    ns, ka, kb = b.graph.num_switches, b.switch_index_of("Ssrc"), b.switch_index_of("S")

    def fn(t):
        mk = p.SwitchStateMask(ns)
        mk.set(ka, t < T_OPEN - 1e-15)
        mk.set(kb, t >= T_CLOSE - 1e-15)
        return mk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=3e-6, dt=DT, switch_fn=fn, engine="pwl",
                          discontinuity_restart=True, enable_substep_state_correction=split)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("nl", [False, True])
@pytest.mark.parametrize("R2", [1e-6, 0.05, 0.4, 0.8, 1e9])
def test_an_island_of_two_fast_modes_keeps_its_charge(R2, nl, split):
    """R2 = 0.05 / 0.4 / 0.8 Ohm: tau2 = R2 * 0.75 nF = 37.5 ps .. 0.6 ns << gamma*dt = 2.9 ns,
    both modes fast. R2 -> 0 (1 uOhm): one mode, C2 || C3. R2 -> inf (1 GOhm, tau2 = 0.75 s):
    one mode, C3 untouched (it stays at 0 V over the run)."""
    r = _island(R2, nl, split)
    t = np.asarray(r.times)
    vs = {n: np.asarray(r.v(NODE[n])) for n in LAWS}
    kb = int(np.searchsorted(t, T_CLOSE - 1e-15))   # the stored sample at 2 us: the left limit
    assert t[kb] == pytest.approx(T_CLOSE)
    assert vs["C1"][kb] == pytest.approx(V1, rel=1e-6)
    assert abs(vs["C2"][kb]) < 1e-6 and abs(vs["C3"][kb]) < 1e-6
    both = R2 < 1.0
    names = list(LAWS) if both else ["C1", "C2"]
    # from the state before the event (the source charged C1 through Rs for 100 tau)
    q0 = sum(_q(n, vs[n][kb], nl) for n in names)
    vinf = float(mp.findroot(lambda v: sum(_q(n, v, nl) for n in names) - q0, 30))
    e_loss = float(sum(_w(n, vs[n][kb], nl) - _w(n, vinf, nl) for n in names))

    def qsum(i):
        return float(sum(_q(n, vs[n][i], nl) for n in LAWS))

    # the island's charge over the event's step: to rounding plus what Ssrc's OFF leakage
    # carries in one step (1e-12 S x 100 V x dt = 1e-18 C, measured 7e-19 C); a junction
    # law's stages are Newton solves of the charge (tolerance 1e-9); over the rest of the
    # run the steps' own KCL rounding and that leakage
    leak = 1e-12 * V1 * DT
    assert abs(qsum(kb + 1) - qsum(kb)) < (1e-9 if nl else 1e-12) * float(q0) + leak
    assert abs(qsum(len(t) - 1) - qsum(kb)) < 1e-7 * float(q0)
    for n in names:   # after the step (both modes dead) and at the end
        assert vs[n][kb + 1] == pytest.approx(vinf, abs=1e-6 * V1), (n, vs[n][kb + 1], vinf)
        assert vs[n][-1] == pytest.approx(vinf, abs=1e-6 * V1)
    if not both:
        assert abs(vs["C3"][-1]) < 1e-3
    recs = [x for x in r.reinit_records if x.t == pytest.approx(T_CLOSE)]
    assert len(recs) == 1 and sorted(recs[0].states) == sorted(names), list(r.reinit_records)
    # linear: the series is exact for two modes (measured 1e-13); a junction law's stage
    # map is not linear, the series fits its secant: 9e-8 / 2.4e-6 at R2 = 0.4 / 0.8 (was
    # +1.7 % with the per-state a)
    assert recs[0].dW == pytest.approx(e_loss, rel=1e-5 if nl else 1e-9)


def _dual(Rm, split):
    """The flux dual: L1 (a -> gnd, i0 = 10 A) circulates through S (a -> gnd, closed); L2
    a -> c, L3 c -> gnd, Rm c -> gnd. S opens at 1 us. KVL around L1, L2, L3 (no other
    element): v_L2 + v_L3 = v_L1, so phi = L2 i2 + L3 i3 - L1 i1 is conserved."""
    b = p.CircuitBuilder()
    b.add_inductor("L1", "a", "gnd", 1e-6, 10.0)
    b.add_switch("S", "a", "gnd", 1e3, 1e-9)
    b.add_inductor("L2", "a", "c", 2e-6, 0.0)
    b.add_inductor("L3", "c", "gnd", 1e-6, 0.0)
    b.add_resistor("Rm", "c", "gnd", Rm)
    ns, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        mk = p.SwitchStateMask(ns)
        mk.set(k, t < T_OPEN - 1e-15)
        return mk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=2e-6, dt=DT, switch_fn=fn, engine="pwl",
                          discontinuity_restart=True, enable_substep_state_correction=split)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("Rm", [1e9, 2e4, 2500.0, 1250.0, 1e-6])
def test_an_inductor_loop_of_two_fast_modes_keeps_its_flux(Rm, split):
    """Rm = 2e4 / 2500 / 1250 Ohm: tau2 = 0.75 uH / Rm = 37.5 ps .. 0.6 ns (the island's
    dual). After both modes one loop current i_m flows (i_Rm = 0): i1 = -i2 = -i3 =
    L1 i1(0-)/(L1 + L2 + L3). Rm -> 0: L3 shorted (tau 0.75 s), i1 = -i2 = L1 i1/(L1 + L2)."""
    L1, L2, L3 = 1e-6, 2e-6, 1e-6
    r = _dual(Rm, split)
    t = np.asarray(r.times)
    i1, i2, i3 = (np.asarray(r.i(n)) for n in ("L1", "L2", "L3"))
    k = int(np.searchsorted(t, T_OPEN - 1e-15))
    phi = L2 * i2 + L3 * i3 - L1 * i1
    assert abs(phi[-1] - phi[k]) < 1e-12 * L1 * abs(i1[k]) + 1e-15 * (L2 + L3) * abs(i1[k])
    both = Rm > 1.0
    # after: i2 = -i1 (S open), and i3 = i2 (both modes) or i3 = i3(0-) (L3 shorted)
    if both:
        im = -phi[k] / (L1 + L2 + L3)
        after = {"L1": im, "L2": -im, "L3": -im}
    else:
        im = (L3 * i3[k] - phi[k]) / (L1 + L2)
        after = {"L1": im, "L2": -im, "L3": i3[k]}
    for n, i in zip(("L1", "L2", "L3"), (i1, i2, i3)):
        assert i[k + 1] == pytest.approx(after[n], abs=1e-6 * abs(i1[k])), n
        assert i[-1] == pytest.approx(after[n], abs=1e-6 * abs(i1[k])), n
    ls = {"L1": L1, "L2": L2, "L3": L3}
    e_loss = sum(0.5 * ls[n] * (i[k] ** 2 - after[n] ** 2)
                 for n, i in zip(("L1", "L2", "L3"), (i1, i2, i3)))
    recs = [x for x in r.reinit_records if x.t == pytest.approx(T_OPEN)]
    assert len(recs) == 1, list(r.reinit_records)
    assert recs[0].dW == pytest.approx(e_loss, rel=1e-6)


def _clamp(nl, split, i_src=10.0, C=2e-9):
    """A current source pulls I from node n; Sl (1 mOhm) holds n at -I R_on = -0.01 V; Dl
    (gnd -> n, V_f 0.7 V, 1 mOhm) clamps n at -0.71 V. Sl opens at 1 us: C carries I and
    n falls 0.7 V in C*0.7/I = 140 ps (finite rate), then Dl takes the current."""
    b = p.CircuitBuilder()
    b.add_current_source("I", "gnd", "n", i_src)
    b.add_switch("Sl", "n", "gnd", 1e3, 1e-9)
    b.add_diode("Dl", "gnd", "n", 1e3, 1e-9, 0.0, 0.7)
    if nl:
        b.add_nonlinear_capacitor("C", "n", "gnd", C, 25.0, 0.5)
    else:
        b.add_capacitor("C", "n", "gnd", C)
    ns, k = b.graph.num_switches, b.switch_index_of("Sl")

    def fn(t):
        mk = p.SwitchStateMask(ns)
        mk.set(k, t < T_OPEN - 1e-15)
        return mk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=2e-6, dt=DT, switch_fn=fn, engine="pwl",
                          discontinuity_restart=True, enable_substep_state_correction=split)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("nl", [False, True])
def test_a_fast_move_that_a_diode_stops_inside_stage_1_is_no_jump(nl, split):
    """The move ends at theta = 140 ps inside stage 1 (gamma h = 2.9 ns) when Dl turns on: a
    state event the step does not locate, not an impulse — the source does the work, no
    resistor dissipates it. It was recorded as 1/2 C 0.7^2 = 4.9e-10 J. (The t = 0 record
    is real: C starts at 0 V against Sl's -0.01 V through 1 mOhm, tau 2 ps.)"""
    r = _clamp(nl, split)
    t, v = np.asarray(r.times), np.asarray(r.v("n"))
    k = int(np.searchsorted(t, T_OPEN - 1e-15))
    assert [x.t for x in r.reinit_records if x.t > 0] == [], list(r.reinit_records)
    assert v[k] == pytest.approx(-0.01, abs=1e-9)
    assert np.allclose(v[k + 1:], -0.71, atol=1e-9), v[k + 1:k + 4]


def test_a_jump_whose_stage_1_also_turns_a_diode_is_recorded():
    """A guard on the start-mode allowance: S (1 mOhm) closes C1 (10 nF at 5 V) onto n (C2 1 nF
    at 0 V, Dl n -> gnd with V_f 0.7 V, 1 mOhm). In the start mode (Dl off) the two share in
    tau = 0.9 ps — a jump of that mode — and then Dl clamps both at its drop: stage 1 turns
    Dl on, the event is still an impulse. The record carries C1 and C2."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vs", "gnd", 5.0)
    b.add_switch("Sc", "vs", "m", 1e3, 1e-12)
    b.add_capacitor("C1", "m", "gnd", 1e-8)
    b.add_switch("S", "m", "n", 1e3, 1e-12)
    b.add_capacitor("C2", "n", "gnd", 1e-9)
    b.add_diode("Dl", "n", "gnd", 1e3, 1e-9, 0.0, 0.7)
    ns, kc, ks = b.graph.num_switches, b.switch_index_of("Sc"), b.switch_index_of("S")

    def fn(t):
        mk = p.SwitchStateMask(ns)
        mk.set(kc, t < T_OPEN - 1e-15)
        mk.set(ks, t >= T_CLOSE - 1e-15)
        return mk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=3e-6, dt=DT, switch_fn=fn, engine="pwl",
                       discontinuity_restart=True)
    recs = [x for x in r.reinit_records if x.t == pytest.approx(T_CLOSE)]
    assert len(recs) == 1 and "C1" in recs[0].states, list(r.reinit_records)
    t = np.asarray(r.times)
    k = int(np.searchsorted(t, T_CLOSE - 1e-15))
    v = np.asarray(r.v("m"))
    assert v[k] == pytest.approx(5.0, rel=1e-6)
    assert 0.69 < v[k + 1] < 0.72, v[k + 1]
