"""The restart rule on a saturable core: its state is the FLUX (restart line, item N).

An event conserves a core's flux linkage, not its current — the twin of a
nonlinear capacitor's charge (F-S-6). The rule's jump test, the jump itself
(lambda0+ = lambda0 + J, i0+ from the law's inverse) and the record's energy
(the integral of (lambda(i) - lambda(i0)) di) are in flux for the stateless
laws; the hysteretic law (no inverse) is refused by name.

Truths: closed forms of the atan law lambda(i) = L_r i + (L_0 - L_r) I_s atan(i / I_s),
inverted here by bisection (independent of the kernel's Newton inverse).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

L0, IS, LR = 1e-3, 0.02, 1e-4
T_E, DT = 5e-6, 1e-7


def _lam(i, l0=L0, i_s=IS, lr=LR):
    return lr * i + (l0 - lr) * i_s * math.atan(i / i_s)


def _stored(i, l0=L0, i_s=IS, lr=LR):
    """The energy a core holds at current i: the integral of i dlambda = i lambda(i) - int lambda di."""
    big = lr * i * i / 2 + (l0 - lr) * i_s * (i * math.atan(i / i_s) - i_s / 2 * math.log1p((i / i_s) ** 2))
    return i * _lam(i, l0, i_s, lr) - big


def _inv(f, target, lo=-1e4, hi=1e4):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _run(b, t_end, split, switch=None):
    kw = {}
    if switch is not None:
        n, k = b.graph.num_switches, b.switch_index_of(switch)

        def fn(t):
            m = p.SwitchStateMask(n)
            m.set(k, t < T_E)
            return m
        kw["switch_fn"] = fn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=t_end, dt=DT, engine="pwl", discontinuity_restart=True,
                          enable_substep_state_correction=split, **kw)


@pytest.mark.parametrize("split", [False, True])
def test_a_core_leaving_saturation_at_a_source_reversal_is_not_a_jump(split):
    """+10 V across a core (1 mH, I_sat 10 mA, L_r 10 nH) for 1.6 us, then -10 V.
    Truth: lambda(t) is the source's integral (KVL, exact), i = lambda^-1: the current
    falls from 30 A to 0.4 A inside the reversal's first stage while the flux moves by
    10 V * 29 ns. Read in current that was 'the event moves saturable_inductor Ls by
    -23.6 A in an instant' (refused, PR head 7188a9e1); in flux nothing jumps."""
    l0, i_s, lr, v, t1 = 1e-3, 0.01, 1e-8, 10.0, 1.6e-6
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "a", "gnd", v, -v, t1, 1.0)
    b.add_saturable_inductor("Ls", "a", "gnd", l0, i_s, lr)
    r = _run(b, 3.5e-6, split)
    assert list(r.reinit_records) == []
    t, i = np.asarray(r.times), np.asarray(r.i("Ls"))
    flux = np.where(t <= t1, v * t, v * t1 - v * (t - t1))
    truth = np.array([_inv(lambda u: _lam(u, l0, i_s, lr), x) for x in flux])
    np.testing.assert_allclose(i, truth, rtol=1e-6, atol=1e-8)   # seen: 1.5e-10 A of 30 A


@pytest.mark.parametrize("split", [False, True])
def test_a_saturated_core_cut_by_its_switch_is_recorded_with_its_stored_energy(split):
    """10 V -S(1 mohm / 1e-9 S)- core, opened at 5 us with the core saturated (0.233 A,
    I_sat 20 mA). The current has no path but g_off: it collapses in L g_off ~ 1e-13 s and
    the path takes what the core held, W = i lambda(i) - int lambda di at lambda = 10 V * 5 us
    (closed form; the switch's 1 mohm drop moves it by 2e-5). Was refused by name."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", 1e3, 1e-9)
    b.add_saturable_inductor("Lsat", "a", "gnd", L0, IS, LR)
    r = _run(b, 1e-5, split, switch="S")
    (rec,) = r.reinit_records
    i0 = _inv(_lam, 10.0 * T_E)
    assert rec.kind == "cut_set" and list(rec.states) == ["Lsat"]
    assert rec.dW == pytest.approx(_stored(i0), rel=1e-4)
    assert 0.5 * L0 * i0 * i0 > 5 * rec.dW          # not the linear 1/2 L i^2 (27 uJ for 3.6 uJ)
    t, i = np.asarray(r.times), np.asarray(r.i("Lsat"))
    assert i[t <= T_E][-1] == pytest.approx(i0, rel=1e-4)
    assert np.all(np.abs(i[t > T_E + DT / 2]) < 1e-6)   # the leakage, 1e-8 A


@pytest.mark.parametrize("split", [False, True])
def test_a_core_and_a_linear_inductor_forced_in_series_share_their_flux(split):
    """10 V -> core -> node m -> L2 200 uH -> gnd, S shorts L2 until 5 us. At the opening
    the core's current i1 must pass L2 (0 A): the cut set conserves lambda(i1) + L2 * 0 =
    lambda(i) + L2 i (closed form, bisection). After it the series pair integrates 10 V.
    The loss is W(i1) - W(i) - 1/2 L2 i^2."""
    l2 = 2e-4
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_saturable_inductor("Ls", "in", "m", L0, IS, LR)
    b.add_inductor("L2", "m", "gnd", l2)
    b.add_switch("S", "m", "gnd", 1e3, 1e-9)
    r = _run(b, 1e-5, split, switch="S")
    (rec,) = r.reinit_records
    assert rec.kind == "cut_set" and sorted(rec.states) == ["L2", "Ls"]
    i1 = _inv(_lam, 10.0 * T_E)

    def both(i):
        return _lam(i) + l2 * i
    ic = _inv(both, _lam(i1))
    assert rec.dW == pytest.approx(_stored(i1) - _stored(ic) - 0.5 * l2 * ic * ic, rel=1e-3)
    t = np.asarray(r.times)
    m = t > T_E + DT / 2
    truth = np.array([_inv(both, _lam(i1) + 10.0 * (x - T_E)) for x in t[m]])
    np.testing.assert_allclose(np.asarray(r.i("Ls"))[m], truth, rtol=1e-5)   # seen: 4e-9 A
    np.testing.assert_allclose(np.asarray(r.i("L2"))[m], truth, rtol=1e-5)


def test_a_hysteretic_core_cut_by_its_switch_is_refused_by_name():
    """The Jiles-Atherton law's lambda(i) depends on its (H, M) path: no inverse, so its
    jump is not carried; the refusal names the core and the law."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", 1e3, 1e-9)
    b.add_hysteretic_core_inductor("Lh", "a", "gnd", 100.0, 1e-4, 0.1, 0.0, 4.0e5,
                                   2000.0, 1e-4, 0.2, 400.0)
    with pytest.raises(p.UnsupportedError, match="saturable_inductor 'Lh'.*hysteretic") as ei:
        _run(b, 1e-5, False, switch="S")
    assert ei.value.device == "Lh" and ei.value.feature == "impulse"


def _clamp(l1, l2, v, losses=False):
    """V -S- a ; L1 a-gnd (a core: (L_0, I_sat, L_r); a float: linear) ; clamp gnd -D-> b -L2- a."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", v)
    b.add_switch("S", "in", "a", 1e3, 1e-9)
    if isinstance(l1, tuple):
        b.add_saturable_inductor("L1", "a", "gnd", *l1)
    else:
        b.add_inductor("L1", "a", "gnd", l1)
    b.add_diode("D", "gnd", "b", 1e3, 1e-9)
    b.add_inductor("L2", "b", "a", l2)
    if losses:
        tab = p.LossTable(v_axis=[0.0, 1e3], i_axis=[0.0, 20.0], tj_axis=[25.0, 150.0],
                          energy=np.array([[[0.0, 0.0], [20.0, 20.0]], [[0.0, 0.0], [20.0, 20.0]]]))
        b.set_device_losses("D", Tj=100.0, E_on=tab)
    return b


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("l2", [2e-7, 5e-7])
def test_a_core_the_series_carries_without_the_flag_keeps_its_share_of_the_record(l2, split):
    """A saturated core (1 mH, 2 mA, 0.5 uH; 0.73 A) clamped through D and L2 when S opens.
    The clamp turns on inside stage 1, so the core's flux move L2 * i+ is under the
    allowance and the vector series carries it unflagged; in saturation that small flux
    is half the current. Truth (flux conservation from the run's own left value,
    bisection): dW = W(i0) - W(i+) - 1/2 L2 i+^2. The record held L2's share only: 49 %
    missing at L2 = 0.5 uH, 28 % at 0.2 uH (fresh review after item N)."""
    core, dt = (1e-3, 0.002, 5e-7), 7e-7
    b = _clamp(core, l2, 1.0)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t < 5 * dt)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=9 * dt, dt=dt, engine="pwl", switch_fn=fn, discontinuity_restart=True,
                       enable_substep_state_correction=split)
    t, i = np.asarray(r.times), np.asarray(r.i("L1"))
    i0 = i[np.searchsorted(t, 5.5 * dt) - 1]
    ic = _inv(lambda u: _lam(u, *core) + l2 * u, _lam(i0, *core))
    truth = _stored(i0, *core) - _stored(ic, *core) - 0.5 * l2 * ic * ic
    (rec,) = r.reinit_records
    assert rec.dW == pytest.approx(truth, rel=2e-3)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("l1", [1e-3, 3e-4, 1e-4])
def test_the_kernel_reads_a_clamp_turn_on_after_the_recorded_collapse(l1, split):
    """Linear twin, L2 = 200 uH: at the opening L1's current i0 is shared, i+ = L1 i0 /
    (L1 + L2) (flux conservation, closed form) — the current the clamp diode picks up
    (E_on linear in i books the current). The right-limit row holds the state BEFORE the
    recorded collapse (L2 at the leakage); the kernel's edge must read the regular part's
    start: at L1 = 0.1 mH (D not named in the record) it read 1e-8 A (fresh review)."""
    b = _clamp(l1, 2e-4, 10.0, losses=True)
    r = _run(b, 8e-6, split, switch="S")
    t = np.asarray(r.times)
    i0 = np.asarray(r.i("L1"))[np.searchsorted(t, T_E + DT / 2) - 1]
    k = r.device_energy("D")
    assert k["n_on"] == 1
    assert k["E_on"] == pytest.approx(l1 * i0 / (l1 + 2e-4), rel=2e-3)
