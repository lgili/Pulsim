"""add-pwl-igbt-and-thyristor: the PWL IGBT knee, the kernel thyristor latch,
GTO, TRIAC, the phase-control schedule and the line-commutated bridge.

Every number is an analytic reference (ideal-converter formulas); every mean
is a time-weighted integral over an exact window (``_wmean``), never
``.mean()`` over an adaptive grid. Every blocking assertion has a control
that proves the same device conducts.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim import _pulsim as k


def _wmean(t, v, t0, t1):
    """Time average of the piecewise-linear trace v(t) over [t0, t1]."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    tt = np.concatenate(([t0], t[(t > t0) & (t < t1)], [t1]))
    return float(np.trapezoid(np.interp(tt, t, v), tt) / (t1 - t0))


def _wrms(t, v, t0, t1):
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    tt = np.concatenate(([t0], t[(t > t0) & (t < t1)], [t1]))
    vv = np.interp(tt, t, v)
    # exact for a piecewise-linear v: mean of v^2 over each segment
    a, b = vv[:-1], vv[1:]
    return math.sqrt(float(np.sum((a * a + a * b + b * b) / 3 * np.diff(tt)))
                     / (t1 - t0))


class _Mask:
    """A constant mask with the given bits on (and an edge oracle)."""

    def __init__(self, n, on=()):
        self.m = p.SwitchStateMask(n)
        for i in on:
            self.m.set(i, True)

    def __call__(self, _t):
        return self.m

    def next_edge_after(self, _t):
        return math.inf


def _sim(b, t_end, engine, sw, dt=1e-5, **kw):
    if engine == "pwl":
        kw["dt"] = dt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return p.simulate(b, t_end, engine=engine, switch_fn=sw, **kw)


# ---------------------------------------------------------------------
# IGBT
# ---------------------------------------------------------------------

def _igbt_forced(i_dc, gate=True):
    b = p.CircuitBuilder()
    b.add_current_source("I", "c", "gnd", i_dc)  # out of c, through Q
    b.add_resistor("Rp", "c", "gnd", 1e6)     # keeps c defined when blocking
    b.add_igbt("Q", "c", "gnd", 10e-3, 1e9, V_ce0=0.9, reverse_blocking=True)
    sw = _Mask(b.graph.num_switches, [b.switch_index_of("Q")] if gate else [])
    return b, sw


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_igbt_forward_conduction_shows_the_knee(engine):
    b, sw = _igbt_forced(10.0)
    r = _sim(b, 1e-4, engine, sw)
    v = np.asarray(r.v("c"))
    i_rp = v[-1] / 1e6
    # v_CE = V_ce0 + R_on * i, i = 10 A minus the 1 MOhm probe's 1 uA.
    assert v[-1] == pytest.approx(0.9 + 10e-3 * (10.0 - i_rp), abs=1e-9)
    assert r.i("Q")[-1] == pytest.approx(10.0 - i_rp, abs=1e-9)


def test_igbt_dc_operating_point_shows_the_knee():
    b, sw = _igbt_forced(10.0)
    r = _sim(b, 1e-5, "pwl", sw, start_from_dc_op=True)
    v0 = np.asarray(r.v("c"))[0]
    assert v0 == pytest.approx(0.9 + 10e-3 * (10.0 - v0 / 1e6), abs=1e-12)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_igbt_blocks_reverse_and_gate_off_forward(engine):
    def circuit(V, gate):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "c", "gnd", V)
        b.add_igbt("Q", "c", "gnd", 10e-3, 1e9, V_ce0=0.9,
                   reverse_blocking=True)
        on = [b.switch_index_of("Q")] if gate else []
        return b, _Mask(b.graph.num_switches, on)

    # Control: +5 V with the gate on conducts (5 - 0.9)/10 mOhm = 410 A.
    b, sw = circuit(5.0, True)
    assert _sim(b, 1e-5, engine, sw).i("Q")[-1] == pytest.approx(410.0, rel=1e-9)
    # Reverse, gate on: R_off only.
    b, sw = circuit(-100.0, True)
    assert _sim(b, 1e-5, engine, sw).i("Q")[-1] == pytest.approx(-1e-7, rel=1e-9)
    # Forward, gate off: R_off only.
    b, sw = circuit(100.0, False)
    assert _sim(b, 1e-5, engine, sw).i("Q")[-1] == pytest.approx(1e-7, rel=1e-9)


def test_igbt_knee_without_reverse_blocking_is_refused_by_name():
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match=r"Q1.*V_ce0.*reverse_blocking"):
        b.add_igbt("Q1", "c", "e", V_ce0=0.9, reverse_blocking=False)
    with pytest.raises(ValueError, match="Q2"):
        b.add_igbt("Q2", "c", "e", V_ce0=-0.1, reverse_blocking=True)


def test_igbt_default_warns_and_stays_the_old_switch():
    def vsi(new):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vdc", "p", "gnd", 400.0)
        for ph in "abc":
            for hi, (c, e) in ((True, ("p", ph)), (False, (ph, "gnd"))):
                name = f"Q{ph}{'h' if hi else 'l'}"
                if new:
                    b.add_igbt(name, c, e, reverse_blocking=False)
                else:
                    b.add_switch(name, c, e, 1 / 10e-3, 1 / 1e9)
            b.add_resistor("R" + ph, ph, "n" + ph, 5.0)
            b.add_inductor("L" + ph, "n" + ph, "s", 2e-3)
        return b

    with pytest.warns(FutureWarning, match="reverse_blocking"):
        p.CircuitBuilder().add_igbt("Q", "c", "e")
    b_old, b_new = vsi(False), vsi(True)
    sw = p.make_three_phase_spwm_fn(
        10e3, 50.0, 0.8,
        p.ThreePhaseLegIndices(*[b_old.switch_index_of(f"Q{x}{s}")
                                 for x in "abc" for s in "hl"]),
        b_old.graph.num_switches, 1e-6)
    r0 = _sim(b_old, 2e-3, "pwl", sw, dt=1e-6)
    r1 = _sim(b_new, 2e-3, "pwl", sw, dt=1e-6)
    assert np.array_equal(np.asarray(r0.states), np.asarray(r1.states))
    assert np.array_equal(np.asarray(r0.switch_states),
                          np.asarray(r1.switch_states))
    l0 = p.device_loss_summary(b_old, r0, switch_fn=sw)
    l1 = p.device_loss_summary(b_new, r1, switch_fn=sw)
    assert [d.get("P_avg") for d in l0] == [d.get("P_avg") for d in l1]
    assert any(d.get("P_avg") for d in l0)


def test_igbt_loss_summary_includes_the_knee():
    b2 = p.CircuitBuilder()
    b2.add_current_source("I", "c", "gnd", 10.0)
    b2.add_igbt("Q", "c", "gnd", 10e-3, 1e9, V_ce0=0.9, reverse_blocking=True)
    sw2 = _Mask(1, [0])
    r = _sim(b2, 1e-4, "pwl", sw2)
    row = [d for d in p.device_loss_summary(b2, r, switch_fn=sw2, t_from=1e-5)
           if d["name"] == "Q"][0]
    # V_ce0 * I + R_on * I^2 = 9 + 1 W, not the R_on * I^2 = 1 W of a switch.
    assert row["P_avg"] == pytest.approx(10.0, rel=1e-9)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_igbt_held_at_the_knee_never_conducts_backwards(engine):
    # A sine of 0.9 V +- 1 mV around the knee through 1 Ohm, gate on.
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "s", "gnd", 0.9, 1e-3, 1e3)
    b.add_resistor("R", "s", "c", 1.0)
    b.add_igbt("Q", "c", "gnd", 10e-3, 1e9, V_ce0=0.9, reverse_blocking=True)
    r = _sim(b, 3e-3, engine, _Mask(1, [0]), dt=1e-6)
    i = np.asarray(r.i("Q"))
    assert i.max() > 0.5e-3          # control: it conducts above the knee
    assert i.min() >= -1e-9


def test_igbt_chopper_conduction_loss_closed_form():
    # IGBT + freewheel diode chopper into a large inductor (constant 10 A):
    # P_cond = D * (V_ce0 * Idc + R_on * Idc^2).
    D, Idc, Vce0, Ron = 0.4, 10.0, 0.9, 10e-3
    # The load is a 10 A current source (an ideal large inductor).
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "p", "gnd", 100.0)
    b.add_igbt("Q", "p", "sw", Ron, 1e9, V_ce0=Vce0, reverse_blocking=True)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9, 0.0, 0.0)
    b.add_current_source("Io", "gnd", "sw", Idc)   # draws Idc out of sw
    sw = p.make_pwm_switch_fn(10e3, D, b.switch_index_of("Q"),
                              b.graph.num_switches)
    r = _sim(b, 1e-3, "pwl", sw, dt=1e-6)
    row = [d for d in p.device_loss_summary(b, r, switch_fn=sw, t_from=1e-4,
                                            t_to=9e-4)
           if d["name"] == "Q"][0]
    assert row["P_avg"] == pytest.approx(D * (Vce0 * Idc + Ron * Idc * Idc),
                                         rel=1e-3)
    assert row["P_avg"] > 1.5 * D * Ron * Idc * Idc   # the knee is in it


# ---------------------------------------------------------------------
# Thyristor
# ---------------------------------------------------------------------

VM, R, F = 100.0, 10.0, 60.0
T = 1.0 / F


def _half_wave(alpha_deg, width=1e-3, I_H=0.0, L=None):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, VM, F)
    b.add_thyristor("T", "a", "k", 1e-3, 1e9, I_H=I_H)
    if L:
        b.add_resistor("R", "k", "m", R)
        b.add_inductor("L", "m", "gnd", L)
    else:
        b.add_resistor("R", "k", "gnd", R)
    sw = p.make_phase_control_fn(F, math.radians(alpha_deg), width,
                                 [b.switch_index_of("T")],
                                 b.graph.num_switches)
    return b, sw


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
@pytest.mark.parametrize("alpha", [30.0, 90.0])
def test_half_wave_rectifier_off_the_grid(engine, alpha):
    b, sw = _half_wave(alpha)
    r = _sim(b, 3 * T, engine, sw, dt=10e-6)
    t, v = np.asarray(r.times), np.asarray(r.v("k"))
    ana = VM / (2 * math.pi) * (1 + math.cos(math.radians(alpha)))
    assert _wmean(t, v, 2 * T, 3 * T) == pytest.approx(ana, rel=5e-3)
    assert v.max() > 0.4 * VM        # control: it conducts
    # never in reverse: at most the R_off leakage, VM * R / R_off = 1e-6 V
    assert v.min() >= -1.01 * VM * R / 1e9


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_latch_outlives_the_gate_pulse(engine):
    # RL load (tau = 1 ms), fired at 30 deg by a 100 us pulse: it keeps
    # conducting past the gate edge, through the voltage zero, until the
    # current zero.
    b, sw = _half_wave(30.0, width=100e-6, L=10e-3)
    r = _sim(b, T, engine, sw, dt=1e-6)
    t, i = np.asarray(r.times), np.asarray(r.i("T"))
    t_fire = 30.0 / 360.0 * T
    on = t[i > 1e-3]
    assert t_fire - 2e-6 < on.min() < t_fire + 5e-6   # the gate, not 0 deg, fired it
    assert on.max() > T / 2 + 1e-4   # conducted past the voltage zero
    # it stopped at a current zero and stayed off for the rest of the cycle
    after = (t > on.max() + 1e-5)
    assert np.all(np.abs(i[after]) < 1e-6)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_holding_current(engine):
    # A current source ramping from 1 A to 0 between 0.1 and 1 ms drives
    # the thyristor (a 1 kOhm bleed keeps the node defined when it is off).
    b = p.CircuitBuilder()
    b.add_pwl_current_source("I", "a", "gnd", [0.0, 1e-4, 1e-3],
                             [1.0, 1.0, 0.0])
    b.add_resistor("Rb", "a", "gnd", 1e3)
    b.add_thyristor("T", "a", "gnd", 1e-3, 1e9, I_H=0.05)
    idx = b.switch_index_of("T")

    def gate_until(t_off):
        def fn(t):
            m = p.SwitchStateMask(1)
            m.set(idx, t < t_off)
            return m
        return fn

    # Gate removed at 50 us: off at the first state with i <= 0.05 A.
    r = _sim(b, 1e-3, engine, gate_until(5e-5), dt=1e-6)
    t, i = np.asarray(r.times), np.asarray(r.i("T"))
    assert i[(t > 1e-4) & (t < 5e-4)].min() > 0.5   # control: latched
    # the source current falls through 0.05 A at t = 1e-4 + 0.95 * 9e-4
    t_h = 1e-4 + 0.95 * 9e-4
    assert np.all(np.abs(i[t > t_h + 2e-6]) < 1e-6)
    assert np.all(i[t < t_h - 2e-6][t[t < t_h - 2e-6] > 1e-4] > 0.05)
    # Gate held on: conducts down to zero.
    r = _sim(b, 1e-3, engine, gate_until(1.0), dt=1e-6)
    t, i = np.asarray(r.times), np.asarray(r.i("T"))
    assert i[(t > t_h + 1e-5) & (t < 1e-3 - 1e-5)].min() > 0.0


# ---------------------------------------------------------------------
# Six-pulse bridge
# ---------------------------------------------------------------------

VLL, ID = 400.0, 20.0
W = 2 * math.pi * F


def _bridge(alpha_deg, Ls=0.0):
    vm = VLL * math.sqrt(2) / math.sqrt(3)
    b = p.CircuitBuilder()
    for ph, ang in (("a", 0.0), ("b", -2 * math.pi / 3),
                    ("c", 2 * math.pi / 3)):
        node = ph + "s" if Ls else ph
        b.add_sine_voltage_source("V" + ph, node, "gnd", 0.0, vm, F, ang)
        if Ls:
            b.add_inductor("L" + ph, node, ph, Ls)
    # firing order: T1 a+, T2 c-, T3 b+, T4 a-, T5 c+, T6 b-
    for name, a, kk in (("T1", "a", "p"), ("T2", "n", "c"), ("T3", "b", "p"),
                        ("T4", "n", "a"), ("T5", "c", "p"), ("T6", "n", "b")):
        b.add_thyristor(name, a, kk)
    b.add_current_source("Id", "n", "p", ID)
    idx = [b.switch_index_of(f"T{j}") for j in range(1, 7)]
    sw = p.make_phase_control_fn(F, math.radians(alpha_deg), 0.33 * T, idx,
                                 b.graph.num_switches, math.pi / 6)
    return b, sw


def _bridge_mean(alpha, engine, Ls=0.0):
    b, sw = _bridge(alpha, Ls)
    r = _sim(b, 3 * T, engine, sw, dt=1e-6)
    t = np.asarray(r.times)
    v = np.asarray(r.v("p")) - np.asarray(r.v("n"))
    return _wmean(t, v, 2 * T, 3 * T)


def _bridge_ref(alpha, Ls=0.0):
    return (3 * math.sqrt(2) / math.pi * VLL * math.cos(math.radians(alpha))
            - 3 / math.pi * W * Ls * ID)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
@pytest.mark.parametrize("alpha,Ls", [(0, 0.0), (30, 0.0), (60, 0.0),
                                      (30, 1e-3)])
def test_six_pulse_bridge_firing_angle_and_overlap(engine, alpha, Ls):
    got = _bridge_mean(alpha, engine, Ls)
    # analytic within 0.5 % (spec); measured within 2e-4 (the R_on drops)
    assert got == pytest.approx(_bridge_ref(alpha, Ls), rel=1e-3)


def test_six_pulse_bridge_engines_agree():
    a = _bridge_mean(30, "pwl")
    b = _bridge_mean(30, "trbdf2")
    assert abs(a - b) < 1e-3 * abs(a)


# ---------------------------------------------------------------------
# GTO, TRIAC
# ---------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_gto_forced_turn_off_hands_the_current_to_the_diode(engine):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "p", "gnd", 100.0)
    b.add_gto("G", "p", "sw", 1e-3, 1e9, V_f=1.0)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9, 0.0, 0.0)
    b.add_inductor("L", "sw", "o", 10e-3)
    b.add_resistor("R", "o", "gnd", 10.0)
    idx = b.switch_index_of("G")

    def fn(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(idx, t < 5e-3)
        return m

    r = _sim(b, 6e-3, engine, fn, dt=1e-6)
    t = np.asarray(r.times)
    ig, id_ = np.asarray(r.i("G")), np.asarray(r.i("D"))
    before = (t > 4.5e-3) & (t < 5e-3 - 1e-6)
    after = (t > 5e-3 + 2e-6) & (t < 5.2e-3)
    assert ig[before].min() > 9.0            # control: ~ (100-1)/10 A
    assert np.abs(ig[after]).max() < 1e-6    # turned off by the gate
    assert id_[after].min() > 8.0            # the diode took the load


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_triac_phase_control_rms(engine):
    alpha = 30.0
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, VM, F)
    b.add_triac("X", "a", "k")
    b.add_resistor("R", "k", "gnd", R)
    idx = b.switch_indices_of("X")
    assert len(idx) == 2
    # fwd fired at alpha, rev at alpha + 180 deg: a 2-pulse schedule.
    sw = p.make_phase_control_fn(F, math.radians(alpha), 1e-3, idx,
                                 b.graph.num_switches)
    r = _sim(b, 3 * T, engine, sw, dt=10e-6)
    t, v = np.asarray(r.times), np.asarray(r.v("k"))
    a = math.radians(alpha)
    ref = VM / math.sqrt(2) * math.sqrt(1 - a / math.pi
                                        + math.sin(2 * a) / (2 * math.pi))
    assert _wrms(t, v, 2 * T, 3 * T) == pytest.approx(ref, rel=5e-3)
    assert abs(_wmean(t, v, 2 * T, 3 * T)) < 0.5   # symmetric


# ---------------------------------------------------------------------
# Engine column, YAML, deprecations
# ---------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["dsed", "exact", "bdf1"])
def test_unsupported_engines_refuse_by_name(engine):
    b, sw = _half_wave(30.0)
    with pytest.raises(p.UnsupportedError) as ei:
        if engine == "bdf1":
            k.run_transient_bdf1(
                b, k.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5), sw)
        else:
            _sim(b, 1e-3, engine, sw)
    assert ei.value.device == "T"
    assert ei.value.feature == "gated_diode"


def test_gated_capability_table():
    caps = k._gated_capabilities()
    assert set(caps) == {"GateHeld", "Latching"}
    for row in caps.values():
        assert row["pwl"] and row["trbdf2"]
        assert not row["dsed"] and not row["bdf1"]


def test_yaml_types_and_igbt_keys(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text(
        "circuit:\n"
        "  devices:\n"
        "    - {type: sine_voltage_source, name: V, from: a, to: gnd,"
        " v_dc: 0, v_amplitude: 100, frequency: 60}\n"
        "    - {type: thyristor, name: T, anode: a, cathode: k, I_H: 0.0}\n"
        "    - {type: gto, name: G, anode: a, cathode: k2, V_f: 1.0}\n"
        "    - {type: triac, name: X, t1: a, t2: k3}\n"
        "    - {type: igbt, name: Q, collector: a, emitter: k4, V_ce0: 0.9}\n"
        "    - {type: resistor, name: R, from: k, to: gnd, R: 10}\n"
        "    - {type: resistor, name: R2, from: k2, to: gnd, R: 10}\n"
        "    - {type: resistor, name: R3, from: k3, to: gnd, R: 10}\n"
        "    - {type: resistor, name: R4, from: k4, to: gnd, R: 10}\n")
    b = p.load_yaml_file(str(y))
    b = getattr(b, "builder", b)
    comps = {c["name"]: c for c in b.components()}
    assert comps["T"]["params"]["gate"] == 2
    assert comps["G"]["params"]["gate"] == 1
    assert comps["Q"]["params"]["V_f"] == 0.9
    assert len(b.switch_indices_of("X")) == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text("circuit:\n  devices:\n"
                   "    - {type: thyristor, name: T9, anode: a, cathode: k,"
                   " V_ce0: 1}\n")
    with pytest.raises(Exception, match="T9"):
        p.load_yaml_file(str(bad))
    typo = tmp_path / "typo.yaml"
    typo.write_text("circuit:\n  devices:\n"
                    "    - {type: igbt, name: Q9, collector: a, emitter: k, Vce0: 1}\n")
    with pytest.raises(Exception, match="Q9.*Vce0|Vce0.*Q9"):
        p.load_yaml_file(str(typo))


def test_pwl_commutation_records_follow_the_gate():
    # Each gate fire and each current-zero turn-off is one record, at the
    # step it takes effect (not the v - V_on crossing of an ungated diode).
    b, sw = _half_wave(30.0, width=100e-6)
    r = _sim(b, 2 * T, "pwl", sw, dt=1e-6)
    bid = b.branch_id_of("T")
    ev = [(e.new_state, e.t_estimated) for e in r.commutation_events if e.branch_id == bid]
    assert [s for s, _ in ev] == [True, False, True, False]
    t_fire = 30.0 / 360.0 * T
    assert abs(ev[0][1] - t_fire) < 2e-6 and abs(ev[2][1] - (T + t_fire)) < 2e-6


def test_python_closure_is_deprecated():
    with pytest.warns(DeprecationWarning, match="add_thyristor"):
        p.make_thyristor_switch_fn(num_switches=1, switch_idx=0,
                                   gate_fn=lambda t: True,
                                   current_probe=lambda t, x: 0.0)


def test_inductive_cut_without_freewheel_names_the_igbt():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "p", "gnd", 100.0)
    b.add_igbt("Q1", "p", "sw", V_ce0=0.9, reverse_blocking=True)
    b.add_inductor("L", "sw", "o", 1e-3)
    b.add_resistor("R", "o", "gnd", 10.0)
    sw = p.make_pwm_switch_fn(10e3, 0.5, 0, 1)
    with pytest.warns(UserWarning, match=r"'Q1'.*freewheel"):
        p.simulate(b, 2e-4, engine="pwl", dt=1e-7, switch_fn=sw)


def test_mode_selection_is_refused_with_a_gated_device():
    # the complementarity port knows no gate: it would turn a gate-off IGBT on
    b, sw = _igbt_forced(10.0)
    with pytest.raises(p.UnsupportedError, match="Q") as ei:
        p.simulate(b, 1e-5, engine="pwl", dt=1e-6, switch_fn=sw, mode_selection="lcp")
    assert ei.value.feature == "mode_selection"


def test_a_pickled_circuit_keeps_the_gate():
    import pickle
    b, _ = _half_wave(30.0, I_H=0.05)
    c = b.build()
    c2 = pickle.loads(pickle.dumps(c))
    assert c2.value_hash() == c.value_hash() and c2.structure_hash() == c.structure_hash()
    gates = {d["name"]: d["params"].get("gate") for d in c2.components()}
    assert gates["T"] == 2


# ---------------------------------------------------------------------
# Per-site blocking tests (task 4.4, pwl and trbdf2): a gate held OFF
# under forward bias passes only g_off·v at every stored sample, at the
# decision sites a ramp reaches mid-step (pwl sub-step watcher / event
# iteration, trbdf2 crossing signal) and at a hard gate turn-off on a step
# boundary with an inductive current (the edge in `combine`, trbdf2's
# boundary settle). Each has a control in which the same device conducts.
# The exact engine refuses gated devices by name (task 4.3).
# ---------------------------------------------------------------------

G_OFF = 1e-9


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_gate_off_device_never_turns_on_under_a_rising_forward_voltage(engine):
    # 10 V, 1 kHz sine through 1 kOhm: v_CE crosses V_th = 0.9 V inside a
    # step (dt = 7 us is no divisor of the crossing time)
    def circuit(gate):
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
        b.add_resistor("R", "in", "c", 1e3)
        b.add_igbt("Q", "c", "gnd", 10e-3, 1.0 / G_OFF, V_ce0=0.9,
                   reverse_blocking=True)
        on = [b.switch_index_of("Q")] if gate else []
        return b, _Mask(b.graph.num_switches, on)

    b, sw = circuit(True)
    r = _sim(b, 1e-3, engine, sw, dt=7e-6)
    assert np.max(np.asarray(r.i("Q"))) > 9e-3   # control: (10 − 0.9)/1 kOhm
    b, sw = circuit(False)
    r = _sim(b, 1e-3, engine, sw, dt=7e-6)
    v = np.asarray(r.v("c"))
    i = np.asarray(r.i("Q"))
    assert np.max(v) > 9.9                         # forward bias reached
    assert np.all(np.abs(i) <= G_OFF * np.abs(v) * (1 + 1e-9) + 1e-15)
    col = b.switch_index_of("Q")
    assert not np.any(np.asarray(r.switch_states)[:, col])


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_gate_turned_off_on_a_step_boundary_blocks_the_inductive_current(engine):
    # 100 V into an RL load (10 Ohm, 1 mH) through the IGBT, freewheel
    # diode across the load. Gate on for 200 us, off after: v_CE jumps to
    # about 100 V (plus the diode's i/g_on while it freewheels) and the IGBT
    # current to g_off·v_CE, while the load current decays through the
    # diode (control: ~8.6 A before).
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "dc", "gnd", 100.0)
    b.add_igbt("Q", "dc", "c", 10e-3, 1.0 / G_OFF, V_ce0=0.9,
               reverse_blocking=True)
    b.add_resistor("RL", "c", "m", 10.0)
    b.add_inductor("L", "m", "gnd", 1e-3)
    b.add_diode("Dfw", "gnd", "c", 1e3, 1e-9)    # g_on, g_off
    j, n = b.switch_index_of("Q"), b.graph.num_switches
    t_off = 200e-6

    class Gate:
        def __call__(self, t):
            m = p.SwitchStateMask(n)
            m.set(j, t < t_off)
            return m

        def next_edge_after(self, t):
            return t_off if t < t_off else math.inf

    r = _sim(b, 400e-6, engine, Gate(), dt=1e-6)
    t = np.asarray(r.times)
    i = np.asarray(r.i("Q"))
    v_ce = 100.0 - np.asarray(r.v("c"))
    before = (t > 150e-6) & (t < t_off)
    # every sample decided under the gate off (the trbdf2 edge sample at
    # t_off + 6e-13 s still carries the left limit, decided with it on)
    after = ~np.asarray(r.gate_states)[:, j]
    assert np.all(t[after] >= t_off) and np.sum(after) > 50
    assert np.min(i[before]) > 5.0                 # control: conducting
    assert np.min(v_ce[after]) > 99.0             # forward bias, gate off
    assert np.all(np.abs(i[after]) <= G_OFF * v_ce[after] * (1 + 1e-9) + 1e-15)
    assert np.asarray(r.i("L"))[after][0] > 5.0   # the load current freewheels
