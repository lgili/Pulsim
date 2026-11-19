"""Integration batch 12: the gated PWL devices (IGBT knee, thyristor latch;
add-pwl-igbt-and-thyristor) at the seams with the other merged tracks.

A: a latched thyristor resumed from a snapshot (device interface: flat state,
   mask) equals the uninterrupted run.
B: an IGBT with a loss port: the in-kernel conduction heat carries the knee.
C: the circuit IR: with_value on V_ce0 / I_H, and the value hash tells a
   thyristor from a plain diode.
D: engine='auto' never routes a gated circuit to the exact engine.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

F = 60.0
T = 1.0 / F


def _half_wave_rl(I_H=0.0):
    # RL load (tau = 1 ms) fired at 30 deg by a 100 us pulse: the latch
    # carries the current past the gate edge and the voltage zero.
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 100.0, F)
    b.add_thyristor("T", "a", "k", 1e-3, 1e9, I_H=I_H)
    b.add_resistor("R", "k", "m", 10.0)
    b.add_inductor("L", "m", "gnd", 10e-3)
    sw = p.make_phase_control_fn(F, math.radians(30.0), 100e-6,
                                 [b.switch_index_of("T")], b.graph.num_switches)
    return b, sw


def _sim(b, t_end, engine, sw, **kw):
    if engine == "pwl":
        kw.setdefault("dt", 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return p.simulate(b, t_end, engine=engine, switch_fn=sw, **kw)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_a_latched_thyristor_survives_a_resume(engine):
    b, sw = _half_wave_rl()
    t_mid = 6e-3   # gate pulse ended at 1.49 ms; the latch alone conducts
    full = _sim(b, T, engine, sw)
    first = _sim(b, t_mid, engine, sw)
    i_mid = float(np.asarray(first.i("T"))[-1])
    assert i_mid > 0.5   # control: latched, gate off, at the split
    second = _sim(b, T, engine, sw, t_start=t_mid,
                  resume_from=first.final_snapshot)
    t2, i2 = np.asarray(second.times), np.asarray(second.i("T"))
    t, i = np.asarray(full.times), np.asarray(full.i("T"))
    # still conducting right after the resume (the latch was not lost) ...
    assert i2[(t2 > t_mid) & (t2 < t_mid + 1e-3)].min() > 0.5
    # ... and the resumed waveform is the uninterrupted one
    peak = float(np.abs(i).max())
    tol = 1e-9 if engine == "pwl" else 1e-4
    assert np.interp(t2, t, i) == pytest.approx(i2, abs=tol * peak)


def test_igbt_loss_port_conduction_heat_carries_the_knee():
    # Chopper into a 10 A current sink: P_cond = D (V_ce0 I + R_on I^2).
    D, Idc, Vce0, Ron = 0.4, 10.0, 0.9, 10e-3
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "p", "gnd", 100.0)
    b.add_igbt("Q", "p", "sw", Ron, 1e9, V_ce0=Vce0, reverse_blocking=True)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9, 0.0, 0.0)
    b.add_current_source("Io", "gnd", "sw", Idc)
    b.set_device_losses("Q", Tj=25.0)
    sw = p.make_pwm_switch_fn(10e3, D, b.switch_index_of("Q"),
                              b.graph.num_switches)
    r = _sim(b, 1e-3, "pwl", sw)
    e_cond = r.device_energy("Q")["E_cond_electrical"]
    want = D * (Vce0 * Idc + Ron * Idc * Idc) * 1e-3
    assert e_cond == pytest.approx(want, rel=2e-3)
    assert e_cond > 1.5 * D * Ron * Idc * Idc * 1e-3   # the knee is in it


def test_ir_with_value_and_hash_on_gated_devices():
    b, sw = _half_wave_rl(I_H=0.05)
    c = b.build()
    # a thyristor and a diode with the same numbers hash apart
    bd = p.CircuitBuilder()
    bd.add_sine_voltage_source("V", "a", "gnd", 0.0, 100.0, F)
    bd.add_diode("T", "a", "k", 1e3, 1e-9, 0.0, 0.0)
    bd.add_resistor("R", "k", "m", 10.0)
    bd.add_inductor("L", "m", "gnd", 10e-3)
    assert bd.build().value_hash() != c.value_hash()
    c2 = c.with_value("R.R", 20.0)
    assert {d["name"]: d["params"] for d in c2.components()}["T"]["gate"] == 2
    bi = p.CircuitBuilder()
    bi.add_voltage_source("Vdc", "p", "gnd", 100.0)
    bi.add_igbt("Q", "p", "sw", 10e-3, 1e9, V_ce0=0.9, reverse_blocking=True)
    bi.add_resistor("Rl", "sw", "gnd", 10.0)
    ci = bi.build()
    ci2 = ci.with_value("Q.V_f", 1.5)   # V_ce0 is the stored V_f
    assert ci2.value_hash() != ci.value_hash()
    m = p.SwitchStateMask(bi.graph.num_switches)
    m.set(bi.switch_index_of("Q"), True)
    v = [float(np.asarray(_sim(cc, 1e-5, "pwl", lambda _t: m).v("sw"))[-1])
         for cc in (ci, ci2)]
    # (100 - V_ce0) * 10 / (10 + 0.01): the knee moved by 0.6 V
    assert v[0] - v[1] == pytest.approx(0.6 * 10 / 10.01, rel=1e-6)


def test_auto_never_routes_a_gated_circuit_to_exact():
    b, sw = _half_wave_rl()
    r = _sim(b, 2e-3, "auto", sw)
    assert r.engine_used not in ("exact", "dsed")
    assert float(np.abs(np.asarray(r.i("T"))).max()) > 0.5   # control: it fired


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_pss_shooting_refuses_a_latching_device(engine):
    # each period run starts from x alone: a latch fired before t = 0 would
    # be cut off and the orbit would be another circuit's (batch-12 review)
    b, sw = _half_wave_rl()
    kw = {"dt": 1e-6} if engine == "pwl" else {}
    with pytest.raises(p.UnsupportedError, match="'T'") as ei:
        p.periodic_steady_state(b, period=T, switch_fn=sw, engine=engine, **kw)
    assert ei.value.feature == "gated_diode"


def test_gated_commutations_are_recorded_under_the_restart_rule():
    # the record compares with the last OUTER step's stamp: the restart
    # stages re-stamp in between and hid every IGBT turn-off (review)
    def run(**kw):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vdc", "p", "gnd", 100.0)
        b.add_igbt("Q", "p", "sw", 10e-3, 1e9, V_ce0=0.9, reverse_blocking=True)
        b.add_diode("D", "gnd", "sw", 1e3, 1e-9, 0.0, 0.0)
        b.add_resistor("Rl", "sw", "gnd", 10.0)
        sw = p.make_pwm_switch_fn(10e3, 0.4, b.switch_index_of("Q"),
                                  b.graph.num_switches)
        r = _sim(b, 1e-3, "pwl", sw, **kw)
        bid = b.graph.branch_index("Q") if hasattr(b.graph, "branch_index") \
            else [c["branch_id"] for c in b.components() if c["name"] == "Q"][0]
        # edges before the last period (the restart rule also records the
        # one landing on t_end)
        return [(round(e.t_estimated / 1e-6), e.new_state)
                for e in r.commutation_events
                if e.branch_id == bid and e.t_estimated < 9.5e-4]

    plain = run()
    assert len(plain) == 20   # control: both edges of 10 periods, at t_prev
    # the restart rule takes each edge at its instant, one step later
    assert run(discontinuity_restart=True) == [(k + 1 if k else 0, s)
                                               for k, s in plain]


# ---------------------------------------------------------------------
# E: a declared gate edge under the restart rule (scr_halfwave_r).
# make_phase_control_fn declares its edges (next_edge_after). The split
# reads them and lands the edge at its instant; the restart rule alone
# samples every gate at the step START (design D4) and reads no oracle,
# so an edge inside a step fires one step late.
# ---------------------------------------------------------------------
_A_SCR = math.radians(60.0)


def _scr_halfwave_error(**kw):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 1e3)
    b.add_thyristor("T", "a", "k", 1e-2, 1e9, V_f=0.7)
    b.add_resistor("R", "k", "gnd", 10.0)
    sw = p.make_phase_control_fn(1e3, _A_SCR, 1e-4, [b.switch_index_of("T")],
                                 b.graph.num_switches)
    assert math.isfinite(sw.next_edge_after(0.0))   # the edges are declared
    r = _sim(b, 3e-3, "pwl", sw, dt=1e-6, **kw)
    t, v = np.asarray(r.times), np.asarray(r.v("k"))
    th = np.mod(2 * np.pi * 1e3 * t, 2 * np.pi)
    on = (th > _A_SCR + 1e-12) & (th <= np.pi - math.asin(0.07))
    ref = np.where(on, (10 * np.sin(th) - 0.7) * 10 / (10 + 1e-2),
                   10 * np.sin(th) * 10 / (10 + 1e9))
    return float(np.abs(v - ref)[1:].max()) / 10.0


def test_the_split_fires_a_declared_gate_edge_on_time():
    # control for the reproducer below (measured 8e-16)
    assert _scr_halfwave_error(discontinuity_restart=True,
                               enable_substep_state_correction=True) < 1e-9


@pytest.mark.xfail(strict=True, reason=(
    "owner: the event-queue track (add-unified-event-queue). Under "
    "discontinuity_restart alone the fixed engine samples the gate at the "
    "step start and does not read the schedule's declared edges: the "
    "60 deg edge fires one 1 us step late (0.80 of the 10 V span at that "
    "sample). The split lands it (test above)."))
def test_the_restart_rule_fires_a_declared_gate_edge_on_time():
    assert _scr_halfwave_error(discontinuity_restart=True) < 1e-9


def _rect(alpha_deg, width):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 100.0, F)
    b.add_thyristor("T", "a", "k", 1e-3, 1e9)
    b.add_resistor("R", "k", "gnd", 10.0)
    sw = p.make_phase_control_fn(F, math.radians(alpha_deg), width,
                                 [b.switch_index_of("T")], b.graph.num_switches)
    return b, sw


_EDGE = 30.0 / 360.0 * T + 100e-6     # the 30 deg pulse's falling edge


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
@pytest.mark.parametrize("alpha,width,t_mid", [
    (90.0, 100e-6, 60.0 / 360.0 * T),   # gate off, blocking forward
    (30.0, 100e-6, _EDGE),              # the gate edge on the seam
    (0.0, 0.9 * T, 0.75 * T),           # gate on, unlatched, reverse
])
def test_a_gated_resume_is_the_uninterrupted_run(engine, alpha, width, t_mid):
    # add-pwl-igbt-and-thyristor D13: the snapshot carries the gate each
    # device was decided under. Split on the 1 us grid; compared away
    # from the current steps (a 1 us grid shift there is not a defect).
    t_mid = round(t_mid * 1e6) * 1e-6
    b, sw = _rect(alpha, width)
    full = _sim(b, 1.5 * T, engine, sw)
    first = _sim(b, t_mid, engine, sw)
    second = _sim(b, 1.5 * T, engine, sw, t_start=t_mid,
                  resume_from=first.final_snapshot)
    t, i = np.asarray(full.times), np.asarray(full.i("T"))
    t2, i2 = np.asarray(second.times), np.asarray(second.i("T"))
    peak = float(np.abs(i).max())
    assert peak > 5.0   # control: it fired
    jumps = t[1:][np.abs(np.diff(i)) > 0.05 * peak]
    keep = np.array([np.all(np.abs(x - jumps) > 2e-5) for x in t2])
    assert keep.sum() > 0.9 * keep.size
    tol = 1e-9 if engine == "pwl" else 1e-4
    assert np.interp(t2[keep], t, i) == pytest.approx(i2[keep], abs=tol * peak)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_the_snapshot_mask_carries_the_gate_bit(engine):
    # conduction bit, then the gate bit: latched with the gate off after
    # the pulse, latched with the gate on inside it
    b, sw = _half_wave_rl()
    assert list(_sim(b, 6e-3, engine, sw).final_snapshot.mask) == [True, False]
    inside = 30.0 / 360.0 * T + 50e-6
    assert list(_sim(b, inside, engine, sw).final_snapshot.mask) == [True, True]


def test_holding_current_is_a_with_value_field():
    b, sw = _half_wave_rl(I_H=0.05)
    c = b.build()
    c2 = c.with_value("T.I_H", 0.2)
    assert {d["name"]: d["params"] for d in c2.components()}["T"]["I_H"] == 0.2
    assert c2.value_hash() != c.value_hash()
    with pytest.raises(ValueError, match="I_H"):
        c.with_value("T.I_H", -1.0)
    # a plain diode has no I_H
    bd = p.CircuitBuilder()
    bd.add_voltage_source("V", "a", "gnd", 1.0)
    bd.add_diode("D", "a", "gnd", 1e3, 1e-9, 0.0, 0.0)
    with pytest.raises((ValueError, KeyError), match="I_H"):
        bd.build().with_value("D.I_H", 0.2)


# ---------------------------------------------------------------------
# F: the result records the commanded gate beside the stamped state
# (task 2.2), and the missed-turn-on check reads it.
# ---------------------------------------------------------------------
@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_the_result_records_the_commanded_gate(engine):
    b, sw = _half_wave_rl()
    r = _sim(b, T, engine, sw)
    j = b.switch_index_of("T")
    g = np.asarray(r.gate_states)[:, j]
    c = np.asarray(r.switch_states)[:, j]
    t = np.asarray(r.times)
    t_on = 30.0 / 360.0 * T
    t_off = t_on + 100e-6
    away = (np.abs(t - t_on) > 2e-6) & (np.abs(t - t_off) > 2e-6) & (t > 0)
    assert np.array_equal(g[away], ((t > t_on) & (t < t_off))[away])
    # distinct from the stamped state: latched after the pulse, gate off
    assert (c & ~g).sum() > 10
    from pulsim import _diode_state
    traces = _diode_state.diode_state_traces(b, r)
    assert [tr.n_rule_violations for tr in traces.values()] == [0]


def test_a_circuit_without_a_gated_device_has_no_gate_record():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 1e3)
    b.add_diode("D", "a", "k", 1e3, 1e-9, 0.0, 0.0)
    b.add_resistor("R", "k", "gnd", 10.0)
    r = p.simulate(b, 1e-3, dt=1e-6, engine="pwl")
    assert r.gate_states is None
    assert r.switch_states is not None   # control: the stamped record exists


class _Forced:
    """A result whose recorded masks are replaced (the check reads only
    the records): a missed turn-on cannot be provoked from a correct run."""

    def __init__(self, res, j, stamped=None, gate=None):
        self._res = res
        self.switch_states = np.array(res.switch_states)
        self.gate_states = np.array(res.gate_states)
        if stamped is not None:
            self.switch_states[:, j] = stamped
        if gate is not None:
            self.gate_states[:, j] = gate

    def __getattr__(self, name):
        if name.startswith("_"):   # not the original's cached traces
            raise AttributeError(name)
        return getattr(self._res, name)


def test_the_missed_turn_on_check_reads_the_gate():
    # a thyristor recorded OFF while its gate is on and it is forward-
    # biased is a missed turn-on; with the gate recorded off it is not
    b, sw = _half_wave_rl()
    r = _sim(b, T, "pwl", sw)
    j = b.switch_index_of("T")
    from pulsim import _diode_state

    def violations(res):
        return [tr.n_rule_violations
                for tr in _diode_state.diode_state_traces(b, res).values()]

    assert violations(r) == [0]   # control: the real run is clean
    gate_on = np.ones(len(r.times), dtype=bool)
    assert violations(_Forced(r, j, stamped=False, gate=gate_on))[0] > 0
    assert violations(_Forced(r, j, stamped=False,
                              gate=np.zeros_like(gate_on))) == [0]
