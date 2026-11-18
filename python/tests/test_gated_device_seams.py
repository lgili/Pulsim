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
    # I_H is not a field of the value table: with_value refuses it by name
    # (it must not rebuild the device as a plain diode)
    with pytest.raises((ValueError, KeyError), match="I_H"):
        c.with_value("T.I_H", 0.2)
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
