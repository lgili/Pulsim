"""The fixed engine's restart rule (`discontinuity_restart=True`) on the
devices and features integrated next to it (integration batch 3).

Each device is driven through an on-grid discontinuity (a source jump
or a gate edge). Without the rule the trapezoid is first order there
(Richardson ratio of successive differences ~2 at dt halving); with it
the three SDIRK(2,2) steps make it second order (~4). Where an exact
solution exists the error is checked against it. Numbers measured
2026-11-13 on the integration branch.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.linalg import expm

import pulsim as p
from pulsim.passives import RacLadder

ON = dict(discontinuity_restart=True)
GON, GOFF = 1e3, 1e-9
TE = 2e-6


def _sim(b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, engine="pwl", **kw)


def _at(res, sig, tq):
    t = np.asarray(res.times)
    return float(np.asarray(sig)[int(np.argmin(abs(t - tq)))])


def _ratios(fn, dts, **kw):
    out = [fn(dt, **kw) for dt in dts]
    y = [v for v, _ in out]
    d = [a - b for a, b in zip(y, y[1:])]
    return [a / b for a, b in zip(d, d[1:])], y, out[0][1]


def _switch_on_at(b, name, te):
    n, k = b.graph.num_switches, b.switch_index_of(name)

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t >= te - 1e-15)
        return m
    return fn


# --- waveform current sources ----------------------------------------------------
def _pulse_i_rc(dt, **kw):
    """1 A pulse current (zero rise) out of 'a' into 1 kOhm || 1 nF."""
    b = p.CircuitBuilder()
    b.add_pulse_current_source("I", "gnd", "a", 0.0, 1.0, TE, 50e-6, 0.0, 0.0, 0.0)
    b.add_resistor("R", "a", "gnd", 1e3)
    b.add_capacitor("C", "a", "gnd", 1e-9)
    r = _sim(b, t_end=TE + 4e-6 + 1e-12, dt=dt, **kw)
    return _at(r, r.v("a"), TE + 1e-6), r.n_restart_steps


def test_a_pulse_current_source_jump_restarts():
    """The pulse current source is the pulse voltage source's trapezoid:
    its zero-rise edges are jumps (sources/source_jumps.hpp)."""
    dts = (1e-7, 5e-8, 2.5e-8, 1.25e-8)
    exact = -1e3 * (1 - math.exp(-1.0))
    on, y_on, n = _ratios(_pulse_i_rc, dts, **ON)
    assert n == 6                                    # start + edge (3 before the fix)
    assert all(r >= 3.5 for r in on), on             # 3.61 3.81
    assert abs(y_on[0] - exact) < 0.3                # 0.260 V
    off, y_off, _ = _ratios(_pulse_i_rc, dts)
    assert all(abs(r - 2) < 0.1 for r in off), off   # 1.95 1.98
    assert abs(y_off[0] - exact) > 17.0              # 17.8 V


def _table_i_rc(dt, **kw):
    b = p.CircuitBuilder()
    b.add_pwl_current_source("I", "gnd", "a", [0.0, TE, TE + 1e-6, 1.0], [0.0, 0.0, 1.0, 1.0])
    b.add_resistor("R", "a", "gnd", 1e3)
    b.add_capacitor("C", "a", "gnd", 1e-9)
    r = _sim(b, t_end=TE + 4e-6 + 1e-12, dt=dt, **kw)
    return _at(r, r.v("a"), TE + 2e-6), r.n_restart_steps


def _sine_i_rc(dt, **kw):
    b = p.CircuitBuilder()
    b.add_sine_current_source("I", "gnd", "a", 0.0, 1.0, 100e3)
    b.add_resistor("R", "a", "gnd", 1e3)
    b.add_capacitor("C", "a", "gnd", 1e-9)
    r = _sim(b, t_end=6e-6 + 1e-12, dt=dt, **kw)
    return _at(r, r.v("a"), 5e-6), r.n_restart_steps


@pytest.mark.parametrize("fn", [_table_i_rc, _sine_i_rc], ids=["table", "sine"])
def test_continuous_current_sources_only_restart_at_the_start(fn):
    """A table source is continuous by validation (a periodic table must
    end where it starts) and a sine has no jump: only the start window."""
    dts = (1e-7, 5e-8, 2.5e-8)
    on, y_on, n = _ratios(fn, dts, **ON)
    off, y_off, _ = _ratios(fn, dts)
    assert n == 3
    assert abs(on[0] - 4) < 0.1 and abs(off[0] - 4) < 0.1, (on, off)
    assert abs(y_on[-1] - y_off[-1]) < 2e-5 * abs(y_off[-1])


# --- the controlled-source kinds -------------------------------------------------
def _controlled(dt, **kw):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vs", "a", "gnd", 0.0, 1.0, TE, 50e-6, 0.0, 0.0, 0.0)
    b.add_resistor("Ra", "a", "b", 100.0)
    b.add_voltage_source("Vsense", "b", "c", 0.0)
    b.add_capacitor("Cc", "c", "gnd", 1e-8)
    b.add_vccs("G1", "a", "gnd", "g", "gnd", 1e-3)
    b.add_resistor("Rg", "g", "gnd", 1e3)
    b.add_capacitor("Cg", "g", "gnd", 1e-9)
    b.add_cccs("F1", "Vsense", "f", "gnd", 2.0)
    b.add_resistor("Rf", "f", "gnd", 50.0)
    b.add_inductor("Lf", "f", "gnd", 50e-6)
    b.add_ccvs("H1", "Vsense", "h", "gnd", 30.0)
    b.add_resistor("Rh", "h", "h2", 10.0)
    b.add_inductor("Lh", "h2", "gnd", 10e-6)
    b.add_vcvs("E1", "c", "gnd", "e", "gnd", 2.0)
    b.add_resistor("Re", "e", "e2", 10.0)
    b.add_capacitor("Ce", "e2", "gnd", 1e-7)
    b.add_gyrator("Y1", "g", "gnd", "y", "gnd", 0.01)
    b.add_capacitor("Cy", "y", "gnd", 1e-8)
    b.add_resistor("Ry", "y", "gnd", 1e3)
    r = _sim(b, t_end=TE + 2e-6 + 1e-12, dt=dt, **kw)
    tq = TE + 1e-6
    return (_at(r, r.v("e2"), tq) + _at(r, r.v("y"), tq) + _at(r, r.i("Lf"), tq)
            + _at(r, r.i("Lh"), tq) + _at(r, r.v("g"), tq)), r.n_restart_steps


def test_controlled_sources_through_a_restart():
    dts = (1e-7, 5e-8, 2.5e-8, 1.25e-8)
    on, _, n = _ratios(_controlled, dts, **ON)
    assert n == 6 and all(r >= 3.5 for r in on), on       # 3.59 3.81
    off, _, _ = _ratios(_controlled, dts)
    assert all(r < 2.7 for r in off), off                 # 2.61 2.35


# --- charge-law capacitors: Q = I t from the first step ----------------------------
_LAWS = {
    "junction": lambda: p.ChargeLaw.junction(2000e-12, 25.0, 0.5),
    "table": lambda: p.ChargeLaw.table(p.ChargeTable.from_capacitance(
        [-10.0, 0.0, 5.0, 20.0, 100.0, 400.0, 500.0],
        [3e-9, 2e-9, 1.2e-9, 5e-10, 1.5e-10, 6e-11, 6e-11])),
    "vdmos_cgd": lambda: p.ChargeLaw.vdmos_cgd(1e-9, 50e-12, 0.5),
}


@pytest.mark.parametrize("law", list(_LAWS))
def test_charge_laws_integrate_q_equals_i_t(law):
    """The SDIRK stage extrapolates each charge through the law itself
    (NonlinearCapacitor::charge), so the table and vdmos_cgd laws hold
    Q(v(t)) = I t like the junction law."""
    lw = _LAWS[law]()
    rel = {}
    for name, kw in (("off", {}), ("on", ON)):
        b = p.CircuitBuilder()
        b.add_current_source("Isrc", "n", "gnd", 6.0)
        b.add_nonlinear_capacitor("Coss", "n", "gnd", lw)
        b.add_resistor("Rleak", "n", "gnd", 1e12)
        r = _sim(b, t_end=40e-9, dt=2e-11, **kw)
        t, v = np.asarray(r.times), np.asarray(r.v("n"))
        q = np.array([lw.charge(x) for x in v])
        rel[name] = np.abs(q[1:] - 6.0 * t[1:]) / (6.0 * t[1:])
    assert rel["on"].max() < 1e-8, rel["on"].max()        # <= 4.5e-10
    assert rel["off"][0] == pytest.approx(0.5, abs=1e-9)  # the half-step start


# --- add_power_mosfet -------------------------------------------------------------
def _pmos(c_gd, c_ds):
    def run(dt, **kw):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vdc", "vdc", "gnd", 40.0)
        b.add_resistor("RL", "vdc", "d", 4.0)
        b.add_power_mosfet("M", "d", "gnd", "drv", K=1.43, V_T=3.0, C_gs=1e-9,
                           C_gd=c_gd, C_ds=c_ds, R_g=10.0)
        b.add_pulse_voltage_source("Vg", "drv", "gnd", 0.0, 12.0, 100e-9, 1.0, 0.0, 0.0, 0.0)
        r = _sim(b, t_end=200e-9 + 1e-15, dt=dt, **kw)
        return _at(r, r.v("d"), 130e-9), r.n_restart_steps
    return run


@pytest.mark.parametrize("c_gd, c_ds", [
    (100e-12, 200e-12),
    (p.ChargeLaw.vdmos_cgd(300e-12, 30e-12, 0.5), p.ChargeLaw.junction(500e-12, 25.0, 0.5)),
], ids=["linear", "vdmos_cgd"])
def test_power_mosfet_turn_on_through_a_restart(c_gd, c_ds):
    dts = (2.5e-10, 1.25e-10, 6.25e-11, 3.125e-11)
    on, _, n = _ratios(_pmos(c_gd, c_ds), dts, **ON)
    assert n == 6 and all(3.5 <= r <= 4.6 for r in on), on   # 4.08 4.01 / 3.99 4.01
    off, _, _ = _ratios(_pmos(c_gd, c_ds), dts)
    assert all(abs(r - 2) < 0.15 for r in off), off           # 1.99 2.00


# --- add_dc_machine ----------------------------------------------------------------
R_, L_, K_, J_ = 0.5, 1e-3, 0.1, 1e-5


def _dcm(dt, **kw):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "a", "gnd", 0.0, 10.0, 1e-3, 1.0, 0.0, 0.0, 0.0)
    b.add_dc_machine("M", "a", "gnd", "w", R_a=R_, L_a=L_, K=K_, J=J_)
    r = _sim(b, t_end=3e-3 + 1e-12, dt=dt, **kw)
    return _at(r, r.v("w"), 2e-3), r.n_restart_steps


def test_dc_machine_through_a_restart():
    """Ideal transformer + capacitor inertia: speed 1 ms after a 10 V
    step against the exact solution of the 2x2 linear system."""
    A = np.array([[-R_ / L_, -K_ / L_, 10.0 / L_], [K_ / J_, 0, 0], [0, 0, 0]])
    exact = float((expm(A * 1e-3) @ np.array([0.0, 0.0, 1.0]))[1])
    dts = (2e-5, 1e-5, 5e-6)
    on = [_dcm(dt, **ON) for dt in dts]
    err_on = [v - exact for v, _ in on]
    assert on[0][1] == 6
    assert abs(err_on[0]) < 3e-3                                   # 2.59e-3
    assert all(a / b > 3.8 for a, b in zip(err_on, err_on[1:]))    # 3.94 3.97
    err_off = [_dcm(dt)[0] - exact for dt in dts]
    assert abs(err_off[0]) > 0.6                                   # 0.663 rad/s
    assert all(abs(a / b - 2) < 0.05 for a, b in zip(err_off, err_off[1:]))


# --- ESR / ESL composite and the R_ac ladder ---------------------------------------
_LADDER = RacLadder(R_dc=10e-3, L_hf=10e-6 - 143.24e-9, R_k=(90e-3,), L_k=(143.24e-9,))


def _esr_esl(dt, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_resistor("R", "a", "o", 1.0)
    b.add_capacitor("C", "o", "gnd", 1e-6, esr=0.1, esl=20e-9)
    r = _sim(b, t_end=3e-6 + 1e-12, dt=dt, switch_fn=_switch_on_at(b, "S", 1e-6), **kw)
    return _at(r, r.v("o"), 1.5e-6), r.n_restart_steps


def _rac(dt, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_resistor("R", "a", "o", 1.0)
    b.add_inductor("L", "o", "gnd", 10e-6, rac=_LADDER)
    r = _sim(b, t_end=3e-6 + 1e-12, dt=dt, switch_fn=_switch_on_at(b, "S", 1e-6), **kw)
    return _at(r, r.i("R"), 1.5e-6), r.n_restart_steps


def _retired_diode(dt, **kw):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "in", "gnd", 5.0, -5.0, 1e-6, 1.0, 0.0, 0.0, 0.0)
    b.add_resistor("R", "in", "a", 10.0)
    b.add_nonlinear_diode("D", "a", "o", p.IdealDiodeParams(), allow_non_passive=True)
    b.add_resistor("Ro", "o", "c", 10.0)
    b.add_capacitor("C", "c", "gnd", 1e-7)
    r = _sim(b, t_end=3e-6 + 1e-12, dt=dt, **kw)
    return _at(r, r.v("c"), 1.5e-6), r.n_restart_steps


@pytest.mark.parametrize("fn, dts", [
    (_esr_esl, (2e-8, 1e-8, 5e-9, 2.5e-9)),          # on 3.85 3.93, off 1.99 1.99
    (_rac, (2e-8, 1e-8, 5e-9, 2.5e-9)),              # on 3.85 3.93, off 2.00 2.00
    (_retired_diode, (1e-8, 5e-9, 2.5e-9, 1.25e-9)),  # on 3.83 3.92, off 2.00 2.00
], ids=["esr_esl", "rac_ladder", "retired_diode"])
def test_composites_through_a_restart(fn, dts):
    on, _, n = _ratios(fn, dts, **ON)
    assert n == 6 and all(3.5 <= r <= 4.6 for r in on), on
    off, _, _ = _ratios(fn, dts)
    assert all(abs(r - 2) < 0.1 for r in off), off


# --- controller ticks with the rule on ---------------------------------------------
def test_controller_ticks_are_unchanged_and_the_gate_moves_one_step():
    """With the rule the gate of step (t_{k-1}, t_k] is switch_fn(t_{k-1}):
    tick instants, counts and jitter are the flag-off ones; the recorded
    gate edges move one sample later."""
    def run(**kw):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "in", "gnd", 24.0)
        b.add_switch("Q", "in", "sw", 1e2, 1e-9)
        b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
        b.add_inductor("L", "sw", "out", 1e-3)
        b.add_capacitor("C", "out", "gnd", 47e-6)
        b.add_resistor("R", "out", "gnd", 10.0)
        lp = p.control.bind_pi_to_switch(
            b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05, output_max=0.95),
            measured=lambda x, i=b.node_id_of("out"): x[i], setpoint=12.0,
            switch="Q", freq=20e3)
        calls = []
        r = _sim(b, t_end=2e-3, dt=1e-6, closed_loops=[lp],
                 step_observer=lambda t, x: calls.append(t),
                 controller_period=50e-6, **kw)
        return r, lp, calls
    r0, l0, c0 = run()
    r1, l1, c1 = run(**ON)
    assert list(r0.control_tick_counts) == list(r1.control_tick_counts) == [40, 40]
    assert c0 == c1 and [d[0] for d in l0.duty_history] == [d[0] for d in l1.duty_history]
    assert r0.control_tick_jitter_max == r1.control_tick_jitter_max == 0.0
    e0 = np.flatnonzero(np.diff(np.asarray(r0.switch_states)[:, 0].astype(int)))
    e1 = np.flatnonzero(np.diff(np.asarray(r1.switch_states)[:, 0].astype(int)))
    assert e1[0] == e0[0] + 1                  # 12 -> 13 (same duty)


# --- snapshot consumers ---------------------------------------------------------------
def _buck_window_at_boundary():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e3, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    n, k = b.graph.num_switches, b.switch_index_of("Q")

    def fn(t):   # on for steps 99..135 of every 100: an edge 1 step before the boundary
        m = p.SwitchStateMask(n)
        j = int(round(t / 1e-7)) % 100
        m.set(k, j >= 99 or j < 36)
        return m
    return b, fn


def test_steady_state_carries_an_open_restart_window():
    """The period map's state includes restart_steps_left; dropping it in
    _snapshot_like made the orbit drift 2.03e-5 (flag off 2.65e-10)."""
    b, fn = _buck_window_at_boundary()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss = p.steady_state(b, period=1e-5, dt=1e-7, switch_fn=fn, **ON)
    assert ss.snapshot.restart_steps_left == 2
    assert ss.drift < 1e-8, ss.drift          # 2.73e-10
    assert ss.residual < 1e-8, ss.residual    # 2.31e-10


def test_steady_state_without_dt_refuses_the_flag():
    """Without dt it is periodic_steady_state on dsed, which takes no
    simulate option (and dsed refuses the rule)."""
    b, fn = _buck_window_at_boundary()
    with pytest.raises(ValueError, match="discontinuity_restart"):
        p.steady_state(b, period=1e-5, switch_fn=fn, **ON)


def test_a_parameter_refactor_reaches_the_restart_steps():
    """The restart stages solve on the cache's event table, which
    refactor_parametric clears: R 1 -> 100 Ohm at 50 us, the next edge
    restarts toward 0.1 A (a private restart cache ramped toward 10 A)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", 1e3, 1e-9)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_inductor("L", "m", "gnd", 10e-6)
    b.add_resistor("Rb", "a", "gnd", 100.0)
    n, k, rid = b.graph.num_switches, b.switch_index_of("S"), b.branch_id_of("R")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, (int(round(t / 1e-9)) // 20000) % 2 == 0)
        return m

    def hook(cache):
        done = []

        def obs(t, x):
            if not done and t >= 50e-6 - 1e-12:
                cache.refactor_parametric(rid, 100.0)
                done.append(t)
        return obs, None
    r = _sim(b, t_end=90e-6, dt=1e-7, switch_fn=fn, on_cache=hook, **ON)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    j = int(np.searchsorted(t, 80e-6 - 1e-12))
    assert np.all(i[j:j + 6] <= 0.1 + 1e-6), i[j:j + 6]   # 0 0.065 0.088 0.096 ...


def test_sweep_path_aware_passes_the_flag():
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "in", "gnd", 0.0, 10.0, 2e-6, 50e-6, 0.0, 0.0, 0.0)
    b.add_resistor("R_load", "in", "a", 1e3)
    b.add_capacitor("C", "a", "gnd", 1e-9)
    seen = []
    p.sweep_path_aware(b, params={"R_load": [1e3, 2e3]},
                       kpi_fn=lambda res, *_: seen.append(res.n_restart_steps) or {},
                       t_end=6e-6, dt=1e-7, verbose=False, **ON)
    assert seen == [6, 6]


# --- conformance of the rule's keywords ----------------------------------------------
@pytest.mark.parametrize("engine", ["trbdf2", "dsed"])
def test_next_jump_after_is_a_no_op_on_the_variable_step_engines(engine):
    """`b_extra_fn.next_jump_after` feeds the fixed engine's restart
    window; trbdf2 and dsed control the step on their own error estimate
    and ignore the attribute (results bit-identical)."""
    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 1.0)
        b.add_resistor("R", "in", "a", 1e3)
        b.add_capacitor("C", "a", "gnd", 1e-9)
        return b
    n = build().pool.state_size(build().graph)
    ia = build().node_id_of("a")

    def bx(t):
        v = np.zeros(n)
        v[ia] = 1e-3 if t >= 2e-6 else 0.0
        return v

    def declared(t):
        return bx(t)
    declared.next_jump_after = lambda t: 2e-6 if t < 2e-6 else math.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = p.simulate(build(), t_end=6e-6, engine=engine, b_extra_fn=bx)
        r1 = p.simulate(build(), t_end=6e-6, engine=engine, b_extra_fn=declared)
    assert np.array_equal(np.asarray(r0.times), np.asarray(r1.times))
    assert np.array_equal(np.asarray(r0.states), np.asarray(r1.states))
