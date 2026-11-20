"""Release findings, groups D and E (and F-6-1): the variable-step engine
and the control tick. Each test reproduces a finding of
openspec/changes/update-number-changing-release (FINDINGS.md) with the
release behaviour turned on EXPLICITLY on the PR line, against an
independent truth; the trace evidence is in
openspec/changes/update-number-changing-release/fixes/group-de.md."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

sys.path.insert(0, str(Path(__file__).parent))

# ---- F-4-5: the loss port next to a nonlinear C_oss on trbdf2 ----------------
# test_loss_port_seams' circuit: a 50 kHz buck, a conductance-form switch
# (g_on = 1e3 S) with C_oss = C0/(1 + v/V0)^0.5 across it, a branch-form
# freewheel diode, E_on = E_off = 1e-4 J (one table knot).
C0, V0, R_ON, G_OFF = 1e-9, 25.0, 1e-3, 1e-9


def _e_oss(v):
    """∫_0^v u·C(u) du for C(u) = C0/(1 + u/V0)^0.5, closed form: the
    energy C_oss holds at v, all of it dissipated in the channel when the
    switch turns on into it (tau = R_on·C_oss ≈ 0.6 ps)."""
    f = lambda w: (2.0 / 3.0) * w ** 1.5 - 2.0 * w ** 0.5   # noqa: E731
    return C0 * V0 ** 2 * (f(1.0 + v / V0) - f(1.0))


def _coss_buck(loss=True):
    from test_loss_port_seams import _stateful_with_loss
    return _stateful_with_loss(loss, True)


def _run_coss(t_end=4e-4, **kw):
    b, fn = _coss_buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, switch_fn=fn, **kw)


def _conduction_truth(r):
    """The channel's energy from the run's own electrical trajectory (which
    every engine resolves; the junction does not feed back): R_on·∫i_L² over
    each ON interval + g_off·∫v² over each OFF interval + E_oss(v) at each
    turn-on, v the switch voltage just before it."""
    t = np.asarray(r.times)
    i_l = np.asarray(r.i("L"))
    v = np.asarray(r.v("vin")) - np.asarray(r.v("sw"))
    e = 0.0
    for k in range(20):                      # 50 kHz, D = 0.4, 0.4 ms
        a, c, z = k * 2e-5, k * 2e-5 + 8e-6, (k + 1) * 2e-5
        s = (t > a + 1e-9) & (t < c)
        tt = np.r_[a, t[s], c]
        e += R_ON * np.trapezoid(np.interp(tt, t, i_l) ** 2, tt)
        s = (t > c) & (t < z)
        tt = np.r_[c, t[s], z]
        e += G_OFF * np.trapezoid(np.interp(tt, t, v) ** 2, tt)
        if k > 0:
            e += _e_oss(v[np.searchsorted(t, a - 1e-12) - 1])
    return e


@pytest.mark.parametrize("kw", [
    dict(rtol=1e-6, atol=1e-9),              # the finding's call
    dict(),                                  # the defaults
    dict(rtol=1e-8, atol=1e-11),
    dict(integrator="esdirk43"),
])
def test_coss_turn_on_energy_is_e_oss_on_the_variable_step_engine(kw):
    """F-4-5: 7.40 J (T_j 6806 °C) where the truth is 4.04e-3 J. Cause: the
    TR-BDF2 error estimate had no row for the nonlinear capacitor's charge,
    so the step after a turn-on (1 ns) never resolved the 0.6 ps channel /
    C_oss discharge and the loss port's quadrature read its start power
    (48 V · 48 kA) over the whole step. Fixed by the q rows of
    update-trbdf2-error-control (ecb07507): the controller resolves the
    discharge and the port books E_oss."""
    r = _run_coss(engine="trbdf2", **kw)
    d = r._device_losses[0]
    assert d.n_on == 20 and d.n_off == 20
    assert d.E_on == pytest.approx(20 * 1e-4, rel=1e-12)
    assert d.E_off == pytest.approx(20 * 1e-4, rel=1e-12)
    truth = _conduction_truth(r)
    # E_oss is 29 % of it. Measured +2.1 % (rtol 1e-8) .. +4.4 % (defaults),
    # esdirk43 +4.1 %: the first step after each turn-on explains 0.8 %
    # (group-de.md), the rest is NOT traced. The finding was +1.7e5.
    assert d.E_cond_electrical == pytest.approx(truth, rel=5e-2)
    # every joule booked reaches the junction (or is pending at t_end)
    assert r.heat_injected + r.heat_pending == pytest.approx(
        d.E_on + d.E_off + d.E_cond_electrical, rel=1e-9)
    assert float(np.asarray(r.v("tj")).max()) < 30.0


def test_coss_loss_port_energy_does_not_depend_on_rtol():
    heats = [_run_coss(engine="trbdf2", rtol=rt, atol=rt * 1e-3).heat_injected
             for rt in (1e-3, 1e-5, 1e-7)]
    # the spread is the first step after each turn-on (0.7 uJ of 4 mJ;
    # it was 6.56 J .. 7.40 J)
    assert max(heats) - min(heats) < 1e-3 * max(heats)


def test_coss_turn_on_energy_on_one_edge_is_e_oss():
    """One turn-on at t = 20 us (the second), resolved: the channel's energy
    over [19.9, 21] us is E_oss(v_left) + R_on·∫i_L² to the first step's
    error."""
    kw = dict(engine="trbdf2", rtol=1e-6, atol=1e-9)
    a, b = _run_coss(1.99e-5, **kw), _run_coss(2.1e-5, **kw)
    t = np.asarray(b.times)
    v = np.asarray(b.v("vin")) - np.asarray(b.v("sw"))
    i_l = np.asarray(b.i("L"))
    j = np.searchsorted(t, 2e-5) - 1
    on = t > 2e-5 + 1e-9
    truth = _e_oss(v[j]) + R_ON * (np.trapezoid(i_l[on] ** 2, t[on])
                                   + i_l[j] ** 2 * (t[on][0] - 2e-5))
    de = (b._device_losses[0].E_cond_electrical
          - a._device_losses[0].E_cond_electrical)
    assert de == pytest.approx(truth, rel=3e-2)


@pytest.mark.xfail(strict=True, reason=(
    "release finding F-4-5 (fixed-engine part): the fixed engine at dt = 1e-7 "
    "collapses the 0.6 ps channel / C_oss discharge inside one step; the energy "
    "C_oss held (E_oss, 29 % of the channel's conduction energy here) is "
    "dissipated by the step's numerical damping and booked nowhere (2.92e-5 J "
    "against 4.17e-5 J). Needs an impulsive booking of an unresolved RC "
    "collapse (the restart rule's ReinitRecord covers ideal devices only)"))
def test_coss_turn_on_energy_on_the_fixed_engine():
    r = _run_coss(engine="pwl", dt=1e-7)
    assert r._device_losses[0].E_cond_electrical == pytest.approx(
        _conduction_truth(r), rel=5e-2)


# ---- F-4-7: control_tick='state' and the run's last tick ---------------------
def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "out", 10.0)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    return b


DT = 1e-6


def _fires(engine, tick, period, offset, n_steps):
    b = _rc()
    ch = p.MixedDomainBlockChain(rate=p.Rate(period, offset=offset))
    ch.add("i", p.Integrator(), inputs=dict(x=1.0, dt="dt"), output="y")
    kw = {"dt": DT} if engine in ("pwl", "dsed") else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.simulate(b, n_steps * DT, engine=engine, control_tick=tick,
                   step_observer=ch.make_step_observer(b), **kw)
    return int(ch._cxx_chain.tick_counts()[1])


CASES = [(1.0, 0.0), (2.0, 0.0), (2.5, 0.0), (3.0, 1.0)]   # (period, offset) / dt


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed"])
@pytest.mark.parametrize("period,offset", CASES)
def test_both_tick_modes_fire_the_same_instants(engine, period, offset):
    """F-4-7: under control_tick='state' the fixed engine dropped the tick at
    t_end - dt (n steps, period dt: n - 1 fires). The count identity: ticks
    at t0 + offset + k*T, every one before t_end fires — on the fixed engine
    every one whose state the run reaches (tau <= t_end - dt: a tick inside
    the last step reads a state the run does not have, and a resumed run
    takes it)."""
    for n in range(1, 12):
        t_end = n * DT
        if engine == "pwl":
            want = math.floor((t_end - DT - offset * DT) / (period * DT) + 1e-9) + 1
        else:
            want = math.ceil((t_end - offset * DT) / (period * DT) - 1e-9)
        want = max(want, 0)
        got = [_fires(engine, m, period * DT, offset * DT, n)
               for m in ("label", "state")]
        assert got == [want, want], (n, got, want)


@pytest.mark.parametrize("tick", ["label", "state"])
@pytest.mark.parametrize("period", [1.0, 2.5])
def test_a_split_run_fires_the_uninterrupted_runs_ticks(tick, period):
    """A tick inside the fixed engine's last step (2.5 dt at t_end = 3 dt) is
    the resumed run's first: a split at any step fires the same count."""
    def run(splits):
        b = _rc()
        ch = p.MixedDomainBlockChain(rate=p.Rate(period * DT))
        ch.add("i", p.Integrator(), inputs=dict(x=1.0, dt="dt"), output="y")
        obs, snap = ch.make_step_observer(b), None
        for n in splits:
            snap = p.simulate(b, n * DT, dt=DT, control_tick=tick,
                              step_observer=obs, resume_from=snap).final_snapshot
        return int(ch._cxx_chain.tick_counts()[1])
    full = run([11])
    assert full == math.ceil(11 / period - 1e-9)
    assert [run([n, 11]) for n in (3, 4, 5, 6)] == [full] * 4


def test_the_c_block_fires_on_every_tick_before_t_end_under_state_ticks(tmp_path):
    """F-4-7's call: a C block of period dt over ten steps fires ten times."""
    import test_control_graph_export_c as X
    from pulsim.c_export import export_c
    so = X._build(tmp_path, export_c(X._pi_pwm(p.Rate(DT)), X._rc(),
                                     outputs=["y"]), "pipwm10")
    b = X._rc()
    h = p.add_c_block(b, inputs=[("v", "out")], outputs=[("v", "sw", "gnd")],
                      dt=DT, lib=str(so))
    p.simulate(b, 10 * DT, dt=DT, control_tick="state")
    assert h.n_fires == 10


# ---- F-6-1: steady_state's default probe on the restart engine ---------------
def test_the_default_probe_finds_the_orbit_to_1e_8_on_the_restart_engine():
    """F-6-1: the columns carry the period map's round-off over the probe
    (1.9e-8 at the default probe on the restart engine). One chord step on
    the verify run's residual takes the answer to the map's floor."""
    import test_steady_state as S
    b = S._buck()
    r = p.steady_state(b, period=S.T, dt=S.DT, switch_fn=S._grid_gate(b),
                       discontinuity_restart=True)
    assert r.residual < 1e-10
    assert r.drift < 1e-10
    # the same orbit at a probe 1000x larger (the columns' error 1000x smaller)
    big = p.steady_state(S._buck(), period=S.T, dt=S.DT,
                         switch_fn=S._grid_gate(b), probe_scale=0.1,
                         discontinuity_restart=True)
    np.testing.assert_allclose(r.history, big.history, rtol=0, atol=1e-10)


def test_the_default_probe_on_the_legacy_engine_reaches_the_floor():
    import test_steady_state as S
    b = S._buck()
    r = p.steady_state(b, period=S.T, dt=S.DT, switch_fn=S._grid_gate(b))
    assert r.residual < 1e-12 and r.drift < 1e-12   # was 7.6e-10 / 7.2e-10
    assert r.n_period_runs == 4 + 2 + 2 + 8        # + the chord's one run


# ---- F-3-4: trbdf2, branch-form switch AND diode, snubbed saturable flyback --
def _flyback_branch(form_s, form_d):
    from test_saturable_transformer import CORE
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_switch("Q1", "sw", "gnd", 1e3, 1e-9, form=form_s)
    b.add_resistor("Rsn", "sw", "sn1", 47.0)
    b.add_capacitor("Csn", "sn1", "gnd", 2.2e-9)
    b.add_saturable_transformer("T1", "vin", "sw", "sec_neg", "sec_anode",
                                L_leak_p=5e-6, **CORE)
    b.add_diode("D1", "sec_anode", "vout", 1e3, 1e-9, V_th=0.7, form=form_d)
    b.add_capacitor("Cout", "vout", "sec_neg", 100e-6)
    b.add_resistor("R_L", "vout", "sec_neg", 5.0)
    b.add_resistor("Rgnd", "sec_neg", "gnd", 1e-6)
    return b


def _flyback_dc(form_s, form_d, **kw):
    b = _flyback_branch(form_s, form_d)
    q, n, period = b.switch_index_of("Q1"), b.graph.num_switches, 1e-5

    def sf(t):
        m = p.SwitchStateMask(n)
        m.set(q, (t % period) / period < 0.4)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=1e-3, switch_fn=sf, **kw)
    t, v = np.asarray(r.times), np.asarray(r.v("vout"))
    m = t > 0.8e-3
    return float(np.trapezoid(v[m], t[m]) / (t[m][-1] - t[m][0]))


_PAIRS = [("conductance", "conductance"), ("branch", "branch"),
          ("conductance", "branch"), ("branch", "conductance")]


@pytest.mark.parametrize("forms,integrator", [
    *[(f, "esdirk43") for f in _PAIRS], *[(f, "trbdf2") for f in _PAIRS[:3]],
    pytest.param(("branch", "conductance"), "trbdf2", marks=pytest.mark.xfail(strict=True, reason=(
        "release finding F-3-4: TR-BDF2 takes no L-stable restart after a diode "
        "event when the circuit has a stateful kind (sdirk_possible = !hs.any(); "
        "the saturable transformer), so the step after each of the 47 landings "
        "reads the pre-event derivative slot of the leakage inductor T1.lp next "
        "to the all-inductor node T1.pm: the estimate grows as h (err 16295 at "
        "3.3e-9 s, 1632 at 3.3e-10) and every landing collapses h to 1e-13 .. "
        "1e-14 s, where Newton's absolute dx test stalls on the 2L/h rows "
        "(abort at the 46th, t = 0.47 ms; the other form pairs survive the same "
        "collapse by luck). Needs the SDIRK restart for stateful kinds on the "
        "TR-BDF2 stage pair (esdirk43 has it: all four pairs run, 15.31034 V)"))),
])
def test_branch_form_flyback_runs_on_trbdf2(forms, integrator):
    """F-3-4: SimulationAborted (Newton at h_min, t = 0.41 ms) with BOTH
    devices in the branch form (b9e46778); with the TR-BDF2 q rows merged
    (ecb07507) that pair runs and branch switch / conductance diode aborts
    instead (t = 0.47 ms). Truth: the fixed engine at dt = 1e-8, 15.3095 V
    (test_saturable_transformer); esdirk43 15.31034 V on all four pairs."""
    kw = {"integrator": "esdirk43"} if integrator == "esdirk43" else {}
    dc = _flyback_dc(*forms, engine="trbdf2", rtol=1e-6, atol=1e-9, **kw)
    assert dc == pytest.approx(15.3095, rel=2e-4)
