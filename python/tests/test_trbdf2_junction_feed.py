"""add-in-kernel-event-losses DECISIONS 16/17: trbdf2 with a junction fed by
the in-kernel loss tracker.

The heat power used to change at every step boundary while the capacitors'
history held the previous step's current: TR-BDF2's estimate of that jump
(0.138·E/C) did not shrink with h, every step was rejected down to h_lo and
forced, and the six-junction VSI exhausted max_steps at 20 µs. Now the power
changes at the step start with its right limit in the history, and the
conduction power at the start is delivered over the step (no lag for a
constant loss). Truths: the closed forms of an RC, a Foster and a Cauer
network under a constant loss; the heat ledger. Step / reject counts are
upper bounds with platform margin (before the fix the first two runs abort).
Loss data here is synthetic.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

from . import test_device_losses_variable_step as rl
from .test_device_losses_kernel import _switch_names, _vsi

T_AMB = 25.0
_RNG = np.random.default_rng(7)
_AX = dict(v_axis=[0.0, 400.0, 800.0], i_axis=[0.0, 30.0, 60.0], tj_axis=[25.0, 150.0])
E_ON = p.LossTable(**_AX, energy=_RNG.uniform(1e-3, 5e-3, (3, 3, 2)))
E_OFF = p.LossTable(**_AX, energy=_RNG.uniform(1e-3, 5e-3, (3, 3, 2)))


def _sim(b, t_end, fn, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, switch_fn=fn, engine="trbdf2", atol=1e-12, **kw)


def _booked(r, names):
    s = 0.0
    for n in names:
        e = r.device_energy(n)
        s += e["E_on"] + e["E_off"] + e["E_cond_electrical"] + e["E_impulse"]
    return s


def _vsi_case():
    """The reported case: the SPWM VSI, one RC junction per switch with
    temperature-dependent tables (test_exact_junction_loop's _vsi_case)."""
    b, fn = _vsi()
    names = _switch_names(b)
    b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
    for k, n in enumerate(names):
        b.add_resistor(f"Rth{k}", f"tj{k}", "amb", 2.0)
        b.add_capacitor(f"Cth{k}", f"tj{k}", "amb", 2e-4)
        b.set_device_losses(n, junction=f"tj{k}", E_on=E_ON, E_off=E_OFF)
    return b, fn, names


def test_the_vsi_with_six_junctions_runs_at_tight_tolerance():
    b, fn, names = _vsi_case()
    r = _sim(b, 2.03e-3, fn, rtol=1e-9)        # aborted at 20 µs before
    s = r.stats
    assert s.n_accept <= 26_000, s.n_accept       # 12 160 on the branch, 12 618 on 14n (2x margin)
    assert s.n_reject <= 400, s.n_reject          # 172
    assert s.n_forced_accepts <= 3, s.n_forced_accepts   # 1 (the start)
    for k in range(len(names)):
        assert np.asarray(r.v(f"tj{k}")).max() > T_AMB + 1.0, k
    assert r.heat_injected + r.heat_pending == pytest.approx(_booked(r, names), rel=1e-9)


T_ON = 10e-6
V, RON, RL = 10.0, 1e-2, 1.0
P = (V / (RON + RL)) ** 2 * RON     # the switch's conduction loss once closed (W)


def _held_switch(net):
    """A switch closing at T_ON into a resistor (no electrical state: a
    constant loss), its heat into a thermal network; the smallest circuit
    that stalled was the "rc" one."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V)
    b.add_switch("S1", "in", "x", 1.0 / RON, 1e-9)
    b.add_resistor("R", "x", "gnd", RL)
    b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
    if net == "rc":
        b.add_resistor("R1", "tj", "amb", 2.0)
        b.add_capacitor("C1", "tj", "amb", 2e-4)
    elif net == "foster":   # series R||C stages: both capacitors carry the heat step
        b.add_resistor("R1", "tj", "n1", 1.2)
        b.add_capacitor("C1", "tj", "n1", 5e-5)
        b.add_resistor("R2", "n1", "amb", 0.8)
        b.add_capacitor("C2", "n1", "amb", 1e-3)
    else:                   # cauer: capacitors to ambient, resistors in series
        b.add_capacitor("C1", "tj", "amb", 5e-5)
        b.add_resistor("R1", "tj", "n1", 1.2)
        b.add_capacitor("C2", "n1", "amb", 1e-3)
        b.add_resistor("R2", "n1", "amb", 0.8)
    b.set_device_losses("S1", junction="tj")
    n, k = b.graph.num_switches, b.switch_index_of("S1")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t >= T_ON)
        return m
    return b, fn


def _closed_form(net, t):
    t = np.maximum(np.asarray(t) - T_ON, 0.0)
    if net == "rc":
        return T_AMB + P * 2.0 * (1 - np.exp(-t / 4e-4))
    if net == "foster":
        return T_AMB + P * (1.2 * (1 - np.exp(-t / 6e-5)) + 0.8 * (1 - np.exp(-t / 8e-4)))
    c = np.diag([5e-5, 1e-3])     # C θ' = -G θ + P e1, θ(0) = 0
    g = np.array([[1 / 1.2, -1 / 1.2], [-1 / 1.2, 1 / 1.2 + 1 / 0.8]])
    w, q = np.linalg.eig(-np.linalg.solve(c, g))
    th_inf = np.linalg.solve(g, np.array([P, 0.0]))
    a = np.linalg.solve(q, -th_inf)
    return T_AMB + th_inf[0] + (q[0] * a * np.exp(np.outer(t, w))).sum(axis=1).real


def test_the_smallest_stalling_circuit_runs():
    """One switch, constant conduction into one RC junction: 20 570 forced
    accepts at rtol 1e-6 and an abort at 1e-8 before."""
    b, fn = _held_switch("rc")
    r = _sim(b, 2e-3, fn, rtol=1e-8)
    s = r.stats
    assert s.n_accept <= 2_200, s.n_accept        # 1065 measured (2x: platform margin)
    assert s.n_reject <= 20, s.n_reject           # 2
    assert s.n_forced_accepts <= 2, s.n_forced_accepts   # 0


@pytest.mark.parametrize("integrator", ["trbdf2", "esdirk43"])
@pytest.mark.parametrize("net", ["rc", "foster", "cauer"])
def test_a_thermal_network_under_a_constant_loss_follows_its_closed_form(net, integrator):
    """The electrical side is frozen (a constant loss after the edge), so
    T_j is the network's own response: within 2e-7 of the rise at rtol 1e-10
    (2.6e-8 / 2.2e-8 / 1.7e-8 measured on trbdf2, <= 1.1e-8 on esdirk43);
    the heat booked is the heat delivered."""
    b, fn = _held_switch(net)
    kw = {} if integrator == "trbdf2" else {"integrator": "esdirk43"}
    r = _sim(b, 2e-3, fn, rtol=1e-10, **kw)
    t = np.asarray(r.times)
    tj = np.asarray(r.v("tj"))
    keep = (t > 0) & (np.abs(t - T_ON) > 1e-15)   # the edge's row is either limit
    truth = _closed_form(net, t[keep])
    rise = truth.max() - T_AMB
    assert rise > 1.0
    np.testing.assert_allclose(tj[keep], truth, rtol=0, atol=2e-7 * rise)
    assert r.heat_injected + r.heat_pending == pytest.approx(_booked(r, ["S1"]), rel=1e-9)
    assert r.stats.n_forced_accepts <= 2


def test_the_rc_network_holds_the_heat_it_was_given():
    """C·θ(t_end) + ∫θ/R dt (the network's own energy) = heat_injected."""
    b, fn = _held_switch("rc")
    r = _sim(b, 2e-3, fn, rtol=1e-10)
    t = np.asarray(r.times)[1:]
    th = np.asarray(r.v("tj"))[1:] - T_AMB
    network = 2e-4 * th[-1] + np.trapezoid(th / 2.0, t)
    assert network == pytest.approx(r.heat_injected, rel=1e-6)


def _chopper_truth(times, r_th=5.0, c_th=1e-4):
    """T_j of the RL chopper's S1 into one RC node: gs·v_S1² in each mode
    (i_L = A + B e^{-bt}: three exponentials) plus each commutation's
    k·v·i as an impulse E/C (test_exact_junction_loop's closed form)."""
    tau = r_th * c_th
    out = np.full(len(times), np.nan)
    i, t, s_on, th, k = 0.0, 0.0, True, 0.0, 0

    def seg(th0, T, terms):
        v = th0 * math.exp(-T / tau)
        for c, q in terms:
            v += c / c_th * (math.exp(-q * T) - math.exp(-T / tau)) / (1.0 / tau - q)
        return v

    while t < rl.T_END:
        T = min(rl.DUTY / rl.FSW if s_on else (1 - rl.DUTY) / rl.FSW, rl.T_END - t)
        gs, _, c0, c1, a, bb = rl._mode(s_on)
        A, B = a / bb, i - a / bb
        al, be = (rl.VIN - c0) - c1 * A, c1 * B           # v_S1 = al - be e^{-bb s}
        terms = [(gs * al * al, 0.0), (-2 * gs * al * be, bb), (gs * be * be, 2 * bb)]
        while k < len(times) and times[k] <= t + T:
            out[k] = seg(th, times[k] - t, terms)
            k += 1
        th = seg(th, T, terms)
        i = A + B * math.exp(-bb * T)
        t += T
        if t >= rl.T_END - 1e-15:
            break
        (vsl, isl), _ = rl._devices(s_on, i)
        (vsr, isr), _ = rl._devices(not s_on, i)
        th += (rl.KOFF * abs(vsr) * abs(isl) if s_on else rl.KON * abs(vsl) * abs(isr)) / c_th
        s_on = not s_on
    return out + T_AMB


def test_the_switching_chopper_junction_follows_its_closed_form():
    """Commutation energies plus a varying conduction loss: T_j within
    1.5e-3 of the rise at rtol 1e-8 (2.6e-4 measured; each event's energy
    reaches the network over the first step after the edge, a first-order
    offset the estimator does not see). It aborted at 1e-8 before."""
    b, fn = rl._rl_chopper(thermal=(5.0, 1e-4))
    r = _sim(b, rl.T_END, fn, rtol=1e-8)
    t = np.asarray(r.times)
    tj = np.asarray(r.v("tj"))
    edges = np.arange(0, 2 * rl.NPER + 2) * (0.5 / rl.FSW)
    off = (t > 0) & (np.min(np.abs(t[:, None] - edges[None, :]), axis=1) > 1e-12)
    truth = _chopper_truth(t[off])
    rise = truth.max() - T_AMB
    assert rise > 1.0
    np.testing.assert_allclose(tj[off], truth, rtol=0, atol=1.5e-3 * rise)
