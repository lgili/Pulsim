"""An event's right limit is a STATE VALUE (finding F-SB-1, notes
openspec/changes/update-number-changing-release/fixes/restart-right-limit.md).

Under the restart rule the fixed engine used to store, as the right limit of
every restart step, the start of the step's quadrature line
2·w1·X1 + (1 − 2·w1)·X2. Now ``right_limit_*`` / ``inner_events`` hold
x(t_e+): each capacitor voltage / inductor current continuous, or moved by an
ideal element's jump (the record's), every other unknown solved in the
event's mode there — and only where something happens. Truths: closed forms
derived here and the exact engine (``engine='dsed'``), which records one
right limit per event with state values.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

ROOT = Path(__file__).resolve().parents[2]
RULE = dict(engine="pwl", discontinuity_restart=True)
T_E = 2.25e-6   # the gate edge; pwl samples the gate and sees it at 2.5 us
DT = 0.5e-6


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _gate(b, name, on_before):
    k = b.switch_index_of(name)

    def fn(t):
        m = p.SwitchStateMask(1)
        m.set(k, (t < T_E) if on_before else (t >= T_E))
        return m
    return fn


def _limits(r):
    """(t, left, right) of each right-limit row of ``with_event_limits``."""
    v = p.with_event_limits(r)
    t = np.asarray(v.times)
    k = np.flatnonzero(t[1:] == np.nextafter(t[:-1], np.inf))
    return v, t, k


@pytest.mark.parametrize("split", [False, True])
def test_a_closing_records_one_right_limit_and_the_state_is_continuous(split):
    # 10 V -> ideal switch -> 1 ohm -> 1 uF from 0 V: at the closing the
    # capacitor voltage is continuous; v_a = 10 V and i = (10 - v_C)/R jump
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "c", 1.0)
    b.add_capacitor("C", "c", "gnd", 1e-6, 0.0)
    r = _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=_gate(b, "S", False),
               enable_substep_state_correction=split, **RULE)
    t = np.asarray(r.times)
    # one row, at the (sampled) edge: none at t = 0, none at the window's
    # later steps 3.0 / 3.5 us
    assert t[np.asarray(r.right_limit_index)] == pytest.approx([2.5e-6], rel=1e-12)
    v, tv, k = _limits(r)
    vc, va = np.asarray(v.v("c")), np.asarray(v.v("a"))
    (j,) = k
    assert vc[j] == 0.0
    assert abs(vc[j + 1] - vc[j]) <= 1e-12 * 10.0
    assert va[j] == 0.0 and va[j + 1] == pytest.approx(10.0, rel=1e-12)


@pytest.mark.parametrize("engine", ["pwl", "dsed"])
def test_a_capacitor_switched_onto_a_source_jumps_by_its_record(engine):
    # 10 V -> ideal switch -> C 1 uF (from 2 V) || R 1 kohm: at the closing
    # v_C jumps to the source, 10 V; the record moves q = C·J
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "c", 0.0, 0.0)
    b.add_capacitor("C", "c", "gnd", 1e-6, 2.0)
    b.add_resistor("R", "c", "gnd", 1e3)
    kw = RULE if engine == "pwl" else dict(engine="dsed")
    r = _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=_gate(b, "S", False), **kw)
    (rec,) = r.reinit_records
    v, t, k = _limits(r)
    (j,) = k
    vc = np.asarray(v.v("c"))
    # left: the RC discharge from 2 V up to the (sampled) edge
    assert vc[j] == pytest.approx(2.0 * np.exp(-t[j] / 1e-3), rel=1e-6)
    assert vc[j + 1] == pytest.approx(10.0, rel=1e-12)
    assert vc[j + 1] - vc[j] == pytest.approx(abs(rec.lambda_) / 1e-6, rel=1e-9)


@pytest.mark.parametrize("engine", ["pwl", "dsed"])
def test_an_inductor_current_cut_jumps_to_zero_by_its_record(engine):
    # 10 V -> ideal switch -> L 10 uH -> R 1 ohm; the switch opens: i_L -> 0
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_inductor("L", "a", "m", 10e-6)
    b.add_resistor("R", "m", "gnd", 1.0)
    kw = RULE if engine == "pwl" else dict(engine="dsed")
    r = _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=_gate(b, "S", True), **kw)
    (rec,) = r.reinit_records
    v, t, k = _limits(r)
    (j,) = k
    il = np.asarray(v.i("L"))
    # left: the RL rise 10·(1 − e^{−t/τ}), τ = 10 us, at the (sampled) edge
    # (pwl's trapezoid at dt = τ/20: O((dt/τ)²) of it)
    assert il[j] == pytest.approx(10.0 * (1.0 - np.exp(-t[j] / 1e-5)), rel=1e-3)
    assert abs(il[j + 1]) <= 1e-12
    assert il[j] - il[j + 1] == pytest.approx(abs(rec.lambda_) / 10e-6, rel=1e-9)


def test_a_conductance_form_cut_keeps_the_models_right_limit_and_warns():
    """The impulse rule records the collapse of the 1 ps mode L·g_off; the
    right limit is the model's own at t_e+ (i_L continuous, v_sw = i_L/g_off,
    as the exact engine's: test_release_findings_a F-5-10) and the voltage
    guard reads it on the fixed engine too."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 48.0)
    b.add_inductor("L", "in", "sw", 1e-3)
    b.add_switch("S", "sw", "gnd", 1e3, 1e-9)

    def sf(t):
        m = p.SwitchStateMask(1)
        m.set(0, t < T_E)
        return m
    with pytest.warns(UserWarning, match="largest voltage"):
        r = p.simulate(b, t_end=6e-6, dt=DT, switch_fn=sf, **RULE)
    assert len(r.reinit_records) == 1
    v, t, k = _limits(r)
    (j,) = k
    il, vsw = np.asarray(v.i("L")), np.asarray(v.v("sw"))
    # the limit is taken over ε = 1e-9·γh against the 1 ps mode: O((ε/τ)²)
    assert il[j + 1] == pytest.approx(il[j], rel=1e-6)
    assert vsw[j + 1] == pytest.approx(il[j] / 1e-9, rel=1e-6)


def test_ll09_right_limits_hold_the_continuous_states(monkeypatch):
    """ll09 (half-bridge leg): the investigation's right row read V(mid)
    48.98 V at 0.924 ms where the truth is 99.99 V on both sides. Its one state
    is i(Lload), and V(mid) is a function of it alone (the Rmid divider): both
    are continuous at every right limit (each edge also carries a record: the
    conductance-form switch's unresolved mode, whose collapse it reports)."""
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks" / "tools"))
    pytest.importorskip("yaml")
    import v1_reader

    case = v1_reader.load(ROOT / "benchmarks/circuits/switching/inverter/"
                          "ll09_half_bridge_inverter_leg.yaml")
    r = _quiet(case.run, "pwl", discontinuity_restart=True)
    assert len(r.right_limit_index) > 0
    v, t, k = _limits(r)
    il, vm = np.asarray(v.i("Lload")), np.asarray(v.v("mid"))
    assert np.max(np.abs(il[k + 1] - il[k])) <= 1e-12 * np.max(np.abs(il))
    assert np.max(np.abs(vm[k + 1] - vm[k])) <= 1e-12 * np.max(np.abs(vm))


def _half_wave_rl():
    # diode_half_wave_rl_load: 10 V 50 Hz -> D (g_on 200, g_off 1 uS) -> 10 ohm
    # -> 1 mH; D turns off when i_L decays to zero near 10.1 ms
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 50.0)
    b.add_diode("D1", "in", "out", 200.0, 1e-6)
    b.add_resistor("R1", "out", "nl", 10.0)
    b.add_inductor("L1", "nl", "gnd", 1e-3, 0.0)
    return b


@pytest.mark.parametrize("dt", [20e-6, 30e-6])   # the crossing lands on b / inside the step
def test_a_located_turn_off_is_continuous_at_its_right_limit(dt):
    """At a LOCATED crossing where only V_f = 0 diodes flip and nothing else
    happens, the ON and OFF laws meet (i = 0, v_D = 0): nothing jumps, the
    right limit is the left (no solve, no mode forced). The located current
    is zero only to the locator's tolerance (here ~26 uA); read through the
    OFF mode's 1/g_off it was a 25.6 V spike (the scoreboard cell 9.6e-4 ->
    5.5e-2)."""
    r = _quiet(p.simulate, _half_wave_rl(), t_end=12e-3, dt=dt,
               enable_substep_state_correction=True, **RULE)
    v, t, k = _limits(r)
    k = k[np.abs(t[k] - 10.1e-3) < 2 * dt]
    assert len(k) == 1
    (j,) = k
    vo, il = np.asarray(v.v("out")), np.asarray(v.i("L1"))
    v_in = 10.0 * np.sin(2 * np.pi * 50.0 * t[j])
    span = 20.0
    assert abs(vo[j + 1] - vo[j]) <= 1e-6 * span
    assert abs(vo[j + 1] - v_in) <= 1e-6 * span
    assert il[j + 1] == pytest.approx(il[j], rel=1e-9, abs=1e-15)   # the state, x0
    assert r.n_right_limit_fallbacks == 0


def _cap_share(engine):
    # C1 1 uF at 10 V || R1 1 kohm  -- ideal S (closes at T_E) --  C2 1 uF at 0 V || R2 500 ohm
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1e-6, 10.0)
    b.add_resistor("R1", "a", "gnd", 1e3)
    b.add_ideal_switch("S", "a", "c", 0.0, 0.0)
    b.add_capacitor("C2", "c", "gnd", 1e-6, 0.0)
    b.add_resistor("R2", "c", "gnd", 500.0)
    kw = RULE if engine == "pwl" else dict(engine="dsed")
    return _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=_gate(b, "S", False), **kw)


def test_a_constraint_current_after_a_charge_sharing_jump_is_the_regular_one():
    """D4: the ideal switch ties two capacitors, both jump to the shared
    voltage v+; the moved states meet the constraint only to the probe's
    tolerance, and an ε solve reads that mismatch as C·δv/ε through the
    switch (68 A here). The right limit is the regular current after the
    impulse: i_S = C2·dv/dt + v+/R2 with dv/dt = −v+(1/R1 + 1/R2)/(C1 + C2)
    (2.5 mA at v+ = 5 V); the exact engine gives the same."""
    r = _cap_share("pwl")
    v, t, k = _limits(r)
    (j,) = k
    vp = np.asarray(v.v("a"))[j + 1]
    i_s = np.asarray(v.i("S"))[j + 1]
    i_truth = 1e-6 * (-vp * (1e-3 + 2e-3) / 2e-6) + vp / 500.0
    assert vp == pytest.approx(np.asarray(v.v("c"))[j + 1], rel=1e-9)
    assert i_s == pytest.approx(i_truth, rel=1e-6)
    d = _cap_share("dsed")
    vd, _, kd = _limits(d)
    assert np.asarray(vd.i("S"))[kd[0] + 1] == pytest.approx(
        np.asarray(vd.v("a"))[kd[0] + 1] * i_truth / vp, rel=1e-6)


def test_a_gate_edge_on_a_resume_seam_records_its_right_limit():
    """D7: a gate edge exactly at a resume's t_start is an event of the
    resumed run (the snapshot carries the pre-edge mask): its right-limit row
    is the uninterrupted run's."""
    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "vin", "gnd", 10.0)
        b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
        b.add_resistor("R", "a", "c", 1.0)
        b.add_capacitor("C", "c", "gnd", 1e-6, 0.0)
        return b
    def gate(t):
        m = p.SwitchStateMask(1)
        m.set(0, t >= T_E)
        return m
    full = _quiet(p.simulate, build(), t_end=6e-6, dt=DT, switch_fn=gate, **RULE)
    vf, tf, kf = _limits(full)
    (t_e,) = tf[kf]   # the sampled edge: the seam goes there
    first = _quiet(p.simulate, build(), t_end=t_e, dt=DT, switch_fn=gate, **RULE)
    rest = _quiet(p.simulate, build(), t_end=6e-6, dt=DT, switch_fn=gate,
                  resume_from=first.final_snapshot, **RULE)
    vr, tr, kr = _limits(rest)
    assert tr[kr] == pytest.approx([t_e], rel=1e-12)
    for col in ("a", "c"):
        assert np.asarray(vr.v(col))[kr[0] + 1] == pytest.approx(
            np.asarray(vf.v(col))[kf[0] + 1], rel=1e-9, abs=1e-12)


def _half_wave_rl_with(V_f=0.0, gate_at=None):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 50.0)
    b.add_diode("D1", "in", "out", 200.0, 1e-6, 0.0, V_f)
    b.add_resistor("R1", "out", "nl", 10.0)
    b.add_inductor("L1", "nl", "gnd", 1e-3, 0.0)
    fn = None
    if gate_at is not None:   # an ideal switch ties out to 5 V from gate_at on
        b.add_voltage_source("V5", "v5", "gnd", 5.0)
        b.add_ideal_switch("S", "out", "v5", 0.0, 0.0)
        k = b.switch_index_of("S")

        def fn(t):
            m = p.SwitchStateMask(k + 1)
            m.set(k, t >= gate_at)
            return m
    return b, fn


def test_r1_a_located_turn_off_beside_a_gate_edge_reads_the_decided_mode():
    """R1: D1's turn-off is located within the last 1 % of the step and
    landed on b = 10.1 ms, where an ideal switch closes onto 5 V. Forcing the
    crossing diode ON there shorted the sine source against 5 V; the row is
    the mode the step used (S ON, D1 OFF): V(out) = 5 V, i_L continuous. It is
    not a continuity point, so it may be counted, never wrong."""
    b, fn = _half_wave_rl_with(gate_at=10.1e-3 - 1e-9)
    r = _quiet(p.simulate, b, t_end=10.2e-3, dt=20e-6, switch_fn=fn,
               enable_substep_state_correction=True, **RULE)
    v, t, k = _limits(r)
    k = k[np.abs(t[k] - 10.1e-3) < 1e-9]
    assert len(k) == 1
    (j,) = k
    assert np.asarray(v.v("out"))[j + 1] == pytest.approx(5.0, rel=1e-9)
    il = np.asarray(v.i("L1"))
    assert il[j + 1] == pytest.approx(il[j], rel=1e-9, abs=1e-15)


def test_r2_a_v_f_turn_off_is_counted():
    """R2: a V_f > 0 diode's laws do not meet at i = 0 (v_D jumps by V_f),
    so its located turn-off is not a continuity point: the row is solved in
    the decided (OFF) mode, where the locator's residual current goes through
    1/g_off, and it is counted in n_right_limit_fallbacks."""
    r = _quiet(p.simulate, _half_wave_rl_with(V_f=0.7)[0], t_end=12e-3, dt=30e-6,
               enable_substep_state_correction=True, **RULE)
    v, t, k = _limits(r)
    assert len(k[np.abs(t[k] - 10.1e-3) < 0.5e-3]) == 1
    assert r.n_right_limit_fallbacks >= 1


def test_r3_a_mode_faster_than_the_limit_step_is_right_or_counted():
    """R3: 10 V -> ideal S -> 1 ohm -> 1 fF (tau = 1 fs, far below every
    eps the construction uses). The truth at the closing: v_C = 0
    (continuous), i = 10 A. The row is that, or it is counted: an
    unresolved mode never passes silently."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "c", 1.0)
    b.add_capacitor("C", "c", "gnd", 1e-15, 0.0)
    r = _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=_gate(b, "S", False), **RULE)
    v, t, k = _limits(r)
    (j,) = k
    vc, i = np.asarray(v.v("c"))[j + 1], np.asarray(v.i("S"))[j + 1]
    right = abs(vc) <= 1e-6 * 10.0 and i == pytest.approx(10.0, rel=1e-6)
    assert right or r.n_right_limit_fallbacks >= 1, (vc, i, r.n_right_limit_fallbacks)

