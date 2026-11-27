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


def _limits(r, start=False):
    """(view, t, k) of the right-limit rows of ``with_event_limits``: row k is
    the left at an edge, k + 1 its right limit. The start's row (x(0+) of a
    re-initialised inconsistent start, F-Z-10) is not an edge's: left out
    unless ``start``."""
    v = p.with_event_limits(r)
    t = np.asarray(v.times)
    k = np.flatnonzero(t[1:] == np.nextafter(t[:-1], np.inf))
    return v, t, (k if start else k[t[k] != t[0]])


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
    # one row, at the edge: none at t = 0, none at the window's later steps.
    # Without the split the restart rule takes an edge at the nearer grid point
    # (gate read 0.5001 dt before the step end); T_E = 2.25 us is mid-step of
    # (2.0, 2.5] us, a tie, taken at the end: 2.5 us. The split lands it at T_E
    # (an inner event, not a sample)
    v, tv, k = _limits(r)
    assert tv[k] == pytest.approx([T_E if split else 2.5e-6], rel=1e-12)
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
    guard reads it on the fixed engine too, as the cut it is (F-Z-8)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 48.0)
    b.add_inductor("L", "in", "sw", 1e-3)
    b.add_switch("S", "sw", "gnd", 1e3, 1e-9)

    def sf(t):
        m = p.SwitchStateMask(1)
        m.set(0, t < T_E)
        return m
    with pytest.warns(UserWarning, match="'S' cut the .* current of 'L'"):   # F-Z-8
        r = p.simulate(b, t_end=6e-6, dt=DT, switch_fn=sf, **RULE)
    assert len(r.reinit_records) == 1
    v, t, k = _limits(r)
    (j,) = k
    il, vsw = np.asarray(v.i("L")), np.asarray(v.v("sw"))
    # the limit is taken over ε = 1e-9·γh against the 1 ps mode: O((ε/τ)²)
    assert il[j + 1] == pytest.approx(il[j], rel=1e-6)
    assert vsw[j + 1] == pytest.approx(il[j] / 1e-9, rel=1e-6)


@pytest.mark.xfail(strict=True, reason=(
    "release finding F-Z-4: the right-limit rows' V(mid) breaks KCL with their own i(Lload): 99.98794556 V "
    "against 50 + 5e4 i_L = 99.98794785 V (2.30e-6 V, 2.3e-8 of scale, gate 1e-12; i_L itself continuous to "
    "5.9e-15); the same with the split off, device_path / mode_selection legacy"))
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
    vr, tr, kr = _limits(rest, start=True)   # the seam's edge is at the resumed run's t[0]
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



@pytest.mark.parametrize("split", [False, True])
def test_h1_a_state_the_probe_only_moved_is_continuous_at_the_right_limit(split):
    """H1: 10 V -> ideal S -> C1 1 uF (2 V) -> 1 ohm -> C2 1 nF (2 V). At the
    closing C1 jumps to 10 V (the ideal element's jump); C2 (tau = 1 ns) only
    MOVES inside the probe (flag 2): it is continuous, so v_C2+ = 2 V and
    i_R+ = 8 A. The row used to carry every moved state (v_C2+ = 10 V, i = 0)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_capacitor("C1", "a", "gnd", 1e-6, 2.0)
    b.add_resistor("R", "a", "d", 1.0)
    b.add_capacitor("C2", "d", "gnd", 1e-9, 2.0)
    r = _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=_gate(b, "S", False),
               enable_substep_state_correction=split, **RULE)
    v, t, k = _limits(r)
    (j,) = k
    assert np.asarray(v.v("a"))[j + 1] == pytest.approx(10.0, rel=1e-9)
    assert np.asarray(v.v("d"))[j + 1] == pytest.approx(2.0, rel=1e-6)
    assert np.asarray(v.i("R"))[j + 1] == pytest.approx(8.0, rel=1e-6)


@pytest.mark.parametrize("split", [False, True])
def test_m1_a_nonlinear_capacitor_is_continuous_in_charge_at_the_right_limit(split):
    """M1: 10 V -> switch (1 uohm / 1e-12 S) -> 1 ohm -> junction capacitor
    50 nF from 0 V (tau ~ 50 ns << dt). The probe moves the capacitor's charge
    inside the step; the right limit keeps x0's charge (v_C+ = 0) and the
    current the closing drives through it, (10 - 0)/(1 + 1e-6) A (KVL at
    t_e+ with the charge continuous). The row used to read the probe's
    recorded state (10.01 V, -0.011 A): the record reports the collapse that
    follows t_e+, the row does not carry it."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 1e-6, 1e-12)
    b.add_resistor("R", "a", "c", 1.0)
    b.add_nonlinear_capacitor("C", "c", "gnd", 50e-9, 0.7, 0.5)
    r = _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=_gate(b, "S", False),
               enable_substep_state_correction=split, **RULE)
    v, t, k = _limits(r)
    (j,) = k
    vc, i = np.asarray(v.v("c"))[j + 1], np.asarray(v.i("R"))[j + 1]
    assert abs(vc) <= 1e-6 * 10.0, vc
    assert i == pytest.approx(10.0 / (1.0 + 1e-6), rel=1e-6)


def _dump(nl, split):
    """C1 1 uF at 10 V -> S (1 uohm / 1e-12 S) -> R 1 mohm -> C2 50 nF at 0 V,
    no source: tau ~ 50 ps << dt, a fast mode the step cannot resolve."""
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1e-6, 10.0)
    b.add_switch("S", "a", "m", 1e6, 1e-12)
    b.add_resistor("R", "m", "c", 1e-3)
    if nl:
        b.add_nonlinear_capacitor("C2", "c", "gnd", 50e-9, 0.7, 0.5)
    else:
        b.add_capacitor("C2", "c", "gnd", 50e-9, 0.0)
    fn = _gate(b, "S", False)
    r = _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=fn,
               enable_substep_state_correction=split, **RULE)
    return b, fn, r


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("nl", [False, True])
def test_row_waveform_and_ledger_agree_on_an_unresolved_collapse(nl, split):
    """Item 0 of the restart line: ONE definition. x(t_e+) keeps C2's charge
    (continuous: a finite resistor in its path), so the row reads v_C2+ = 0
    and i_R+ = 10 V / (1 uohm + 1 mohm) (KVL at t_e+). The ~50 ps collapse
    after it is the record's (J, dW); the regular part starts after the
    collapse, so record + window integral = the energy the event dissipates.
    Truth (closed form / mpmath, derived here): the charge C1*10 is shared,
    v_inf solves C1*v + Q2(v) = C1*10, E = W(before) - W(after); linear
    E = C1*C2/(C1 + C2)*50 = 2.380952e-6 J. The window integral used to start
    from the row: 0.0167 J (7000x) on the linear circuit."""
    from pulsim.losses import device_loss_summary
    mp = pytest.importorskip("mpmath")
    b, fn, r = _dump(nl, split)
    v, t, k = _limits(r)
    (j,) = k
    assert abs(np.asarray(v.v("c"))[j + 1]) <= 1e-6
    assert np.asarray(v.i("R"))[j + 1] == pytest.approx(10.0 / (1e-6 + 1e-3), rel=1e-6)
    c1, c0, v0, m = mp.mpf("1e-6"), mp.mpf("50e-9"), mp.mpf("0.7"), mp.mpf("0.5")

    def q2(x):
        return c0 * v0 / (1 - m) * ((1 + x / v0) ** (1 - m) - 1) if nl else c0 * x

    def w2(x):
        return mp.quad(lambda u: u * c0 / mp.sqrt(1 + u / v0), [0, x]) if nl else c0 * x * x / 2
    vinf = mp.findroot(lambda x: c1 * x + q2(x) - c1 * 10, 9)
    e_loss = float(c1 * 50 - c1 * vinf ** 2 / 2 - w2(vinf))
    assert np.asarray(r.v("c"))[-1] == pytest.approx(float(vinf), rel=1e-6)
    rows = device_loss_summary(b, r, switch_fn=fn, t_from=0.0, t_to=6e-6)
    regular = sum(e["E_total"] for e in rows if e["name"] in ("S", "R"))
    impulse = sum(x.dW for x in r.reinit_records)
    assert regular + impulse == pytest.approx(e_loss, rel=1e-3)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("rl", [5.0, 0.5, 1e3])
def test_a_charge_sharing_row_shares_the_charge_once(rl, split):
    """C1 1 uF (10 V) -ideal S- C2 50 nF (0 V) || RL. At the closing the
    ideal loop shares the charge: v+ = 10*C1/(C1 + C2) = 9.5238095 V (closed
    form), on both capacitors. The row used to add the probe's jump to the
    states it flagged only (C2) and let the constraint share again:
    9.977 V at RL = 5 ohm, 9.956 V at 0.5 ohm, no fallback counted
    (fresh review of the restart line, item 0)."""
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1e-6, 10.0)
    b.add_ideal_switch("S", "a", "c", 0.0, 0.0)
    b.add_capacitor("C2", "c", "gnd", 50e-9, 0.0)
    b.add_resistor("RL", "c", "gnd", rl)
    r = _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=_gate(b, "S", False),
               enable_substep_state_correction=split, **RULE)
    v, t, k = _limits(r)
    (j,) = k
    vl = np.asarray(v.v("a"))[j]
    want = vl * 1e-6 / (1e-6 + 50e-9)    # C1's charge at the left limit, C2 empty
    assert np.asarray(v.v("a"))[j + 1] == pytest.approx(want, rel=1e-6)
    assert np.asarray(v.v("c"))[j + 1] == pytest.approx(want, rel=1e-6)


# closed (item B): the pin is Anderson-accelerated, exact for a mode faster than eps
def test_a_cut_faster_than_the_row_solve_keeps_the_current():
    """10 V -> R 10 ohm -> L 1 uH (1 A) -> S (1 uohm / 1e-12 S) opens: the
    flux is continuous at t_e+, i_L+ = i_L- (truth: the inductor's state law)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_resistor("R", "vin", "x", 10.0)
    b.add_inductor("L", "x", "s", 1e-6, 1.0)
    b.add_switch("S", "s", "gnd", 1e6, 1e-12)
    r = _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=_gate(b, "S", True), **RULE)
    v, t, k = _limits(r)
    (j,) = k
    il = np.asarray(v.i("L"))
    assert il[j + 1] == pytest.approx(il[j], rel=1e-6)


# closed (item C): stage 2 after an ideal element's jump starts from x0+
@pytest.mark.parametrize("split", [False, True])
def test_the_sample_after_an_ideal_jump_carries_no_jump_current(split):
    """10 V -ideal S- C2 50 nF || RL 10 ohm: after the closing v_c = 10 V,
    i_C = 0, so every stored sample after the edge has |i(V)| = 10/RL = 1 A
    (KCL; the right-limit row itself reads it)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "c", 0.0, 0.0)
    b.add_capacitor("C2", "c", "gnd", 50e-9, 0.0)
    b.add_resistor("RL", "c", "gnd", 10.0)
    r = _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=_gate(b, "S", False),
               enable_substep_state_correction=split, **RULE)
    t, iv = np.asarray(r.times), np.asarray(r.i("V"))
    # without the split the closing acts at the nearer grid point; T_E is
    # mid-step of (2.0, 2.5] us, a tie, taken at the end: the 2.5 us row is
    # still open (its left limit)
    t_on = T_E if split else 2.5e-6
    after = t > t_on + 1e-12
    assert after.sum() >= 6
    assert np.abs(iv[after]) == pytest.approx(np.ones(after.sum()), rel=1e-6)


def _scr(**kw):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 50.0)
    b.add_thyristor("T", "in", "out", R_on=1e-3, R_off=1e9, **kw)
    b.add_resistor("R", "out", "gnd", 2.0)
    b.add_capacitor("C", "out", "gnd", 1e-6, 0.0)
    ns, k = b.graph.num_switches, b.switch_index_of("T")

    def fn(t):
        m = p.SwitchStateMask(ns)
        m.set(k, 1e-3 <= t < 2e-3)
        return m
    return b, fn


@pytest.mark.parametrize("dt", [20e-6, 23.39182767e-6])   # inside a step / landed on b
def test_h2_a_thyristor_turn_off_at_i_h_is_no_continuity_point(dt):
    """H2: a gated latching device turns off at I_H = 1 A (gate off): its ON
    and OFF laws do not meet (i jumps from I_H to the leakage), so the row is
    solved: i_T+ = (v_in - v_out)*g_off (KCL, v_out continuous on C)."""
    b, fn = _scr(I_H=1.0)
    r = _quiet(p.simulate, b, t_end=10e-3, dt=dt, switch_fn=fn,
               enable_substep_state_correction=True, **RULE)
    v, t, k = _limits(r)
    (j,) = [j for j in k if t[j] > 5e-3]
    assert np.asarray(v.i("T"))[j] == pytest.approx(1.0, rel=1e-6)
    vin = 10.0 * np.sin(2 * np.pi * 50.0 * t[j])
    vo = np.asarray(v.v("out"))[j + 1]
    assert np.asarray(v.v("out"))[j] == pytest.approx(vo, rel=1e-9)
    assert abs(np.asarray(v.i("T"))[j + 1]) <= 10 * abs(vin - vo) * 1e-9


def test_h3_a_gate_fired_turn_on_on_a_landed_crossing_is_solved():
    """H3: a V_f = 0 diode's crossing landed on the grid point where a
    thyristor's gate fires (10 V -> T -> 1 ohm): the gate edge is no diode
    flip, the row is solved: i_T+ = 10/(1 + 1e-3) A (KVL)."""
    dt = 1.999800025923e-05      # D2's turn-off at 500*dt (the run's own location)
    tg = 500 * dt
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V1", "in", "gnd", 0.0, 10.0, 50.0)
    b.add_diode("D2", "in", "o2", 1e3, 1e-9)
    b.add_resistor("R2", "o2", "gnd", 10.0)
    b.add_capacitor("C2", "o2", "gnd", 1e-7, 0.0)
    b.add_voltage_source("V3", "s3", "gnd", 10.0)
    b.add_thyristor("T", "s3", "o3", R_on=1e-3, R_off=1e9)
    b.add_resistor("R3", "o3", "gnd", 1.0)
    ns, k = b.graph.num_switches, b.switch_index_of("T")

    def fn(t):
        m = p.SwitchStateMask(ns)
        m.set(k, t >= tg)
        return m
    r = _quiet(p.simulate, b, t_end=11e-3, dt=dt, switch_fn=fn,
               enable_substep_state_correction=True, **RULE)
    v, t, kk = _limits(r)
    (j,) = [j for j in kk if t[j] > 5e-3]
    assert t[j] == pytest.approx(tg, rel=1e-12)
    assert np.asarray(v.i("T"))[j + 1] == pytest.approx(10.0 / (1.0 + 1e-3), rel=1e-6)


# closed (item A, the row): the dependent states from a consistent solve, then every state pinned
def test_an_ideal_jump_beside_a_fast_rc_keeps_the_charge_in_the_row():
    """C1 1 uF (10 V) -ideal S- node c: C2 50 nF || (R3 1 mohm - C3 100 nF at 0 V).
    At t_e+ the ideal loop shares C1, C2: v_a = v_c = 10*C1/(C1 + C2) (closed
    form, from the left value); C3 sits behind R3 and is continuous: v_d = 0."""
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1e-6, 10.0)
    b.add_ideal_switch("S", "a", "c", 0.0, 0.0)
    b.add_capacitor("C2", "c", "gnd", 50e-9, 0.0)
    b.add_resistor("R3", "c", "d", 1e-3)
    b.add_capacitor("C3", "d", "gnd", 100e-9, 0.0)
    r = _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=_gate(b, "S", False), **RULE)
    v, t, k = _limits(r)
    (j,) = k
    want = np.asarray(v.v("a"))[j] * 1e-6 / (1e-6 + 50e-9)
    assert np.asarray(v.v("a"))[j + 1] == pytest.approx(want, rel=1e-6)
    assert abs(np.asarray(v.v("d"))[j + 1]) <= 1e-6


@pytest.mark.parametrize("split", [False, True])
def test_a_row_whose_mode_does_not_hold_at_the_right_limit_books_no_loss(split):
    """10 V -ideal S (1 mOhm)- sw; D gnd->sw; L 10 uH sw->o; R 1 ohm; Co 100 pF at sw. S
    opens at T_E: Co holds v_sw (continuous), i_L carries on and discharges Co; D is
    reverse-biased at t_e+ (i_D = -g_off*v_sw) and turns on ~0.55 ns later at ~0 V. Truth:
    the row's i_D >= -1e-6 A, and any E_on booked for D is at most E_on(10 V, i_L)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "sw", 1e-3, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "o", 10e-6)
    b.add_resistor("R", "o", "gnd", 1.0)
    b.add_capacitor("Co", "sw", "gnd", 100e-12, 0.0)
    tab = p.LossTable(v_axis=[0.0, 100.0], i_axis=[0.0, 100.0], tj_axis=[25.0],
                      energy=np.array([[[0.0], [0.0]], [[0.0], [1e-3]]]))
    b.set_device_losses("D", Tj=25.0, E_on=tab, E_off=tab)
    k, ns = b.switch_index_of("S"), b.graph.num_switches   # S and D: two bits

    def fn(t):
        m = p.SwitchStateMask(ns)
        m.set(k, t < T_E)
        return m
    r = _quiet(p.simulate, b, t_end=3e-6, dt=DT, switch_fn=fn,
               enable_substep_state_correction=split, **RULE)
    v, t, kk = _limits(r)
    for j in kk:
        assert float(np.asarray(v.i("D"))[j + 1]) >= -1e-6
    i_l = float(np.max(np.abs(np.asarray(r.i("L")))))
    bound = 1e-3 * (10.0 / 100.0) * (i_l / 100.0)
    assert all(ev["E"] <= 2 * bound for ev in r.loss_events("D") if ev["kind"] == "on")


@pytest.mark.parametrize("split", [False, True])
def test_a_row_whose_mode_holds_at_the_right_limit_books_the_turn_on(split):
    """Control for the test above (the fix must not pass by booking nothing): the same
    circuit without Co. At t_e+ the inductor current has only D: D's ON mode holds,
    i_D = i_L(t_e) (continuous), and D blocked v_D = -10 V before. Truth: the RL rise
    i_L = 10/1.001·(1 − e^{−t/τ}), τ = L/(R + 1 mOhm), at the row's instant (pwl's
    trapezoid at dt = τ/20: O((dt/τ)²)), and E_on = 1e-3·(10/100)·(i_L/100) (bilinear)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "sw", 1e-3, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "o", 10e-6)
    b.add_resistor("R", "o", "gnd", 1.0)
    tab = p.LossTable(v_axis=[0.0, 100.0], i_axis=[0.0, 100.0], tj_axis=[25.0],
                      energy=np.array([[[0.0], [0.0]], [[0.0], [1e-3]]]))
    b.set_device_losses("D", Tj=25.0, E_on=tab, E_off=tab)
    k, ns = b.switch_index_of("S"), b.graph.num_switches

    def fn(t):
        m = p.SwitchStateMask(ns)
        m.set(k, t < T_E)
        return m
    r = _quiet(p.simulate, b, t_end=3e-6, dt=DT, switch_fn=fn,
               enable_substep_state_correction=split, **RULE)
    v, t, kk = _limits(r)
    (j,) = [j for j in kk if t[j] > 0]
    i_e = 10.0 / 1.001 * (1.0 - np.exp(-t[j] * 1.001 / 10e-6))
    assert float(np.asarray(v.i("D"))[j + 1]) == pytest.approx(i_e, rel=1e-3)
    (ev,) = [ev for ev in r.loss_events("D") if ev["kind"] == "on"]
    assert ev["E"] == pytest.approx(1e-3 * 0.1 * i_e / 100.0, rel=2e-3)


@pytest.mark.parametrize("split", [False, True])
def test_a_gated_device_the_row_re_decides_keeps_its_gate(split):
    """Review of F1 (1): the row's re-decision must not read a gated device's gate from a
    stamped conduction bit. Ca (100 pF, ~1 V from V1 through 1 kOhm) on node a; at T_E S
    (1 mOhm) ties a to -10 V and Q's gate turns on. At t_e+ v_a ~ 1 V (Ca continuous) is
    forward across Q above V_ce0 = 0.5 V with the gate on: Q conducts. Truth (closed form):
    i_Q = (v_a - 0.5)/1 mOhm ~ 500 A in the row (v_a: Ca's DC value)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "p1", "gnd", 1.0)
    b.add_resistor("Ra", "p1", "a", 1e3)
    b.add_capacitor("Ca", "a", "gnd", 100e-12, 1.0)
    b.add_voltage_source("Vn", "neg", "gnd", -10.0)
    b.add_switch("S", "a", "neg", 1e3, 1e-9)
    b.add_igbt("Q", "a", "gnd", 1e-3, 1e9, V_ce0=0.5, reverse_blocking=True)
    ns, ks, kq = b.graph.num_switches, b.switch_index_of("S"), b.switch_index_of("Q")

    def fn(t):
        m = p.SwitchStateMask(ns)
        m.set(ks, t >= T_E)
        m.set(kq, t >= T_E)
        return m
    r = _quiet(p.simulate, b, t_end=3e-6, dt=DT, switch_fn=fn,
               enable_substep_state_correction=split, **RULE)
    v, t, kk = _limits(r)
    (j,) = [j for j in kk if t[j] > 0]
    # Ca's DC value behind Ra against S's and Q's OFF leakage (1e-9 S each; RaCa = 0.1 us)
    va = (1.0 / 1e3 - 10.0 * 1e-9) / (1.0 / 1e3 + 2e-9)
    assert float(np.asarray(v.v("a"))[j + 1]) == pytest.approx(va, rel=1e-6)
    assert float(np.asarray(v.i("Q"))[j + 1]) == pytest.approx((va - 0.5) / 1e-3, rel=1e-6)


@pytest.mark.parametrize("co", [100e-12, 1e-9, 10e-9, 100e-9])
def test_a_diode_riding_its_boundary_at_the_row_keeps_the_row(co):
    """Review of F1 (2): at t = 0 (zero state, S closed) v_sw = 0 (Co continuous) and the
    V_th = 0 diode D sits on its boundary (v_D = 0, i_D = 0): either mode holds, the
    re-decision must not chatter on round-off and drop the row. Truth: a row at t = 0 with
    v_sw = 0 and i_D = 0 (to the leakage)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "sw", 1e-3, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "o", 10e-6)
    b.add_resistor("R", "o", "gnd", 1.0)
    b.add_capacitor("Co", "sw", "gnd", co, 0.0)
    k, ns = b.switch_index_of("S"), b.graph.num_switches

    def fn(t):
        m = p.SwitchStateMask(ns)
        m.set(k, t < T_E)
        return m
    r = _quiet(p.simulate, b, t_end=3e-6, dt=DT, switch_fn=fn, **RULE)
    v, t, kk = _limits(r, start=True)
    (j,) = [j for j in kk if t[j] == 0.0]
    assert abs(float(np.asarray(v.v("sw"))[j + 1])) <= 1e-12
    assert abs(float(np.asarray(v.i("D"))[j + 1])) <= 1e-6


def _jump_into_freewheel(d_kind):
    """C1 1 nF (10 V) -ideal S- c; C2 50 pF at c; D gnd->c freewheels L (10 A) into 1 mOhm;
    S closes at T_E. d_kind: "diode" (1 mOhm conductance diode), "ideal" (branch form,
    R_on = 0), "resistor" (a plain 1 mOhm resistor in D's place)."""
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1e-9, 10.0)
    b.add_ideal_switch("S", "a", "c", 0.0, 0.0)
    b.add_capacitor("C2", "c", "gnd", 50e-12, 0.0)
    if d_kind == "diode":
        b.add_diode("D", "gnd", "c", 1e3, 1e-9)
    elif d_kind == "ideal":
        b.add_diode("D", "gnd", "c", None, 1e-9, form="branch", R_on=0.0)
    else:
        b.add_resistor("D", "gnd", "c", 1e-3)
    b.add_inductor("L", "c", "o", 10e-6, 10.0)
    b.add_resistor("R", "o", "gnd", 1e-3)
    k, ns = b.switch_index_of("S"), b.graph.num_switches

    def fn(t):
        m = p.SwitchStateMask(ns)
        m.set(k, t >= T_E)
        return m
    return b, fn


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("d_kind", ["diode", "ideal"])
def test_an_ideal_jump_whose_row_mode_does_not_hold_re_decides(d_kind, split):
    """Review of F1 (a), the ideal-jump branch (circuit: _jump_into_freewheel). The ideal
    loop shares the charge, v_c+ = (C1 v_a- + C2 v_c-)/(C1 + C2) = 9.52 V (closed form, from
    the left values): D is reverse-biased at t_e+ (i_D = leakage). Control: D conducts the
    inductor's 10 A at the left limit. Before the fix the row kept stage 1's D ON: 1 mOhm D
    i_D = -9522 A, v_c+ 9.5220 V; ideal D (whose post-jump current is forward: only the
    jump's impulse shows the reverse drive) v_c+ = 0 V."""
    c1, c2 = 1e-9, 50e-12
    b, fn = _jump_into_freewheel(d_kind)
    r = _quiet(p.simulate, b, t_end=4e-6, dt=DT, switch_fn=fn,
               enable_substep_state_correction=split, **RULE)
    v, t, kk = _limits(r)
    (j,) = kk
    va, vc, i_d = (np.asarray(x) for x in (v.v("a"), v.v("c"), v.i("D")))
    want = (c1 * va[j] + c2 * vc[j]) / (c1 + c2)
    assert vc[j + 1] == pytest.approx(want, rel=1e-6)
    assert va[j + 1] == pytest.approx(want, rel=1e-6)
    assert -1e-6 <= i_d[j + 1] <= 0.0
    assert i_d[j] == pytest.approx(np.asarray(v.i("L"))[j], rel=1e-3)   # the control


# closed (item A): the ideal jump is impulsive too, its J holds the fast collapse
@pytest.mark.parametrize("split", [False, True])
def test_the_sample_after_an_ideal_jump_into_a_fast_rc_obeys_kcl(split):
    """_jump_into_freewheel with a plain 1 mOhm resistor in D's place: the capacitors settle
    in ~1 ps after the share, so at the next stored sample KCL at c gives i_R = i_L (truth:
    KCL; i_R = -v_c / 1 mOhm). Before: stage 1 started from x0 and left the 9.5 V collapse
    to the SDIRK pair, R(z) ~ (1 - c)/z of it in every later sample: i_R = 10.0917 A (no
    split) / 10.1883 A (split) for i_L = 9.9950 A."""
    b, fn = _jump_into_freewheel("resistor")
    r = _quiet(p.simulate, b, t_end=4e-6, dt=DT, switch_fn=fn,
               enable_substep_state_correction=split, **RULE)
    v, t, kk = _limits(r)
    (j,) = [j for j in kk if t[j] > 1e-6]   # the closing (not a start row)
    i_r = -np.asarray(v.v("c"))[j + 2:] / 1e-3   # every stored sample after the row
    np.testing.assert_allclose(i_r, np.asarray(v.i("L"))[j + 2:], rtol=1e-6)


@pytest.mark.parametrize("split", [False, True])
def test_a_three_way_ideal_share_has_its_row(split):
    """Review (rev_fin2 probe_i): C1 1 nF (10 V) -ideal S- c (C2 50 pF) -ideal Sd (R_on 0)-
    d (C3 1 nF, 88 V) with a 0.04 A current source draining d; S and Sd close at T_E. The
    pin of that ideal three-way share diverged (a = -37.49 V for the 8.78 V share) and was
    published; then no row was stored (counted). Since item 3 the share is read as an
    ideal jump (its loop sums jump): the row is s3, the three-way charge share of the
    left values, on all three nodes, and the samples after it follow the closed form
    v(t) = s3 - 0.04 (t - t_e)/(C1 + C2 + C3) (every state tied by the two ideal switches)."""
    c1, c2, c3, i0 = 1e-9, 50e-12, 1e-9, -0.04
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", c1, 10.0)
    b.add_ideal_switch("S", "a", "c", 0.0, 0.0)
    b.add_capacitor("C2", "c", "gnd", c2, 0.0)
    b.add_ideal_switch("Sd", "c", "d", 0.0, 1e-9)
    b.add_capacitor("C3", "d", "gnd", c3, 88.0)
    b.add_current_source("I", "d", "gnd", i0)   # flows out of d's terminal into d
    k, kd, ns = b.switch_index_of("S"), b.switch_index_of("Sd"), b.graph.num_switches

    def fn(t):
        m = p.SwitchStateMask(ns)
        m.set(k, t >= T_E)
        m.set(kd, t >= T_E)
        return m
    r = _quiet(p.simulate, b, t_end=4e-6, dt=DT, switch_fn=fn,
               enable_substep_state_correction=split, **RULE)
    v, t, kk = _limits(r)
    assert r.n_right_limit_fallbacks == 0
    va, vc, vd = (np.asarray(x) for x in (v.v("a"), v.v("c"), v.v("d")))
    (jr,) = [j for j in kk if t[j] > 1e-6]
    s3r = (c1 * va[jr] + c2 * vc[jr] + c3 * vd[jr]) / (c1 + c2 + c3)
    for y in (va, vc, vd):
        assert y[jr + 1] == pytest.approx(s3r, rel=1e-6)
    after = slice(jr + 1, None)
    want = s3r + i0 * (t[after] - t[jr]) / (c1 + c2 + c3)
    for y in (va, vc, vd):
        np.testing.assert_allclose(y[after], want, rtol=0, atol=1e-6 * 100.0)


def _share_into_fast_mode(c3):
    """C1 1 nF (10 V) -ideal S- c: C2 50 pF, c -1 mOhm- d: C3 (c3; 0: none), 1 MOhm leak."""
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1e-9, 10.0)
    b.add_ideal_switch("S", "a", "c", 0.0, 0.0)
    b.add_capacitor("C2", "c", "gnd", 50e-12, 0.0)
    if c3:
        b.add_resistor("Rf", "c", "d", 1e-3)
        b.add_capacitor("C3", "d", "gnd", c3, 0.0)
        b.add_resistor("Rl", "d", "gnd", 1e6)
    else:
        b.add_resistor("Rl", "c", "gnd", 1e6)
    return b


@pytest.mark.parametrize("split", [False, True])
def test_an_ideal_jump_books_the_fast_collapse_it_excites(split):
    """Review of 4b945143 (finding 1): the share C1-C2 (J_i) excites C1||C2 -1 mOhm- C3
    (tau 0.5 ps), which collapses inside the event (J_f). Closed form of the event's
    loss: W(x0) - W(all at Q/(C1 + C2 + C3)) = 25.61 nJ = the share's 2.381 nJ + the
    collapse's 1/2 C_eq (9.524 V)^2, C_eq = (C1 + C2) C3/(C1 + C2 + C3). 5c2bca28 booked
    the share only (2.381 nJ)."""
    c1, c2, c3 = 1e-9, 50e-12, 1e-9
    b = _share_into_fast_mode(c3)
    r = _quiet(p.simulate, b, t_end=4e-6, dt=DT, switch_fn=_gate(b, "S", False),
               enable_substep_state_correction=split, **RULE)
    recs = [x for x in r.reinit_records if x.t > 1e-6]
    w0, q = 0.5 * c1 * 100.0, c1 * 10.0
    want = w0 - 0.5 * q * q / (c1 + c2 + c3)
    assert sum(x.dW for x in recs) == pytest.approx(want, rel=1e-4)
    share = 0.5 * c1 * c2 / (c1 + c2) * 100.0
    assert [x.dW for x in recs if x.states == ["C1", "C2"]] == [pytest.approx(share, rel=1e-6)]
    assert all(x.cause == "S" for x in recs)


@pytest.mark.parametrize("split", [False, True])
def test_an_ideal_jump_without_a_fast_mode_books_the_share_only(split):
    """Control: C1 -S- C2 with a 1 MOhm leak only (no fast mode): one record, the
    share's closed form 1/2 C1 C2/(C1 + C2) (10 V)^2 = 2.381 nJ."""
    b = _share_into_fast_mode(0.0)
    r = _quiet(p.simulate, b, t_end=4e-6, dt=DT, switch_fn=_gate(b, "S", False),
               enable_substep_state_correction=split, **RULE)
    recs = [x for x in r.reinit_records if x.t > 1e-6]
    assert [x.dW for x in recs] == [pytest.approx(0.5 * 1e-9 * 50e-12 / 1.05e-9 * 100.0, rel=1e-6)]


def _gate_n(b):
    k, ns = b.switch_index_of("S"), b.graph.num_switches

    def fn(t):
        m = p.SwitchStateMask(ns)
        m.set(k, t >= T_E)
        return m
    return fn


def _clamped_dump(v1, il0):
    """C1 1 nF (v1) -ideal S- c: C2 50 pF (0 V), ideal D gnd->c (R_on 0), L 10 uH (il0) -> 1 mOhm."""
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1e-9, v1)
    b.add_ideal_switch("S", "a", "c", 0.0, 0.0)
    b.add_capacitor("C2", "c", "gnd", 50e-12, 0.0)
    b.add_diode("D", "gnd", "c", None, 1e-9, form="branch", R_on=0.0)
    b.add_inductor("L", "c", "o", 10e-6, il0)
    b.add_resistor("R", "o", "gnd", 1e-3)
    return b


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("il0", [10.0, 0.0])
def test_an_ideal_clamp_dump_books_the_whole_dump(il0, split):
    """Review of 4b945143 (finding 2): v1 = -10 V; closing S forward-biases D, which holds c
    at 0, so C1 dumps all of its charge through S + D: closed form 1/2 C1 v1^2 = 50 nJ
    (dsed 50.0 nJ). 2.1.0 and 5c2bca28 booked the free share C1-C2 (2.381 nJ), as if D
    were off. Control of D conducting: v_a = 0 after the event."""
    b = _clamped_dump(-10.0, il0)
    r = _quiet(p.simulate, b, t_end=4e-6, dt=DT, switch_fn=_gate_n(b),
               enable_substep_state_correction=split, **RULE)
    recs = [x for x in r.reinit_records if x.t > 1e-6]
    assert sum(x.dW for x in recs) == pytest.approx(0.5 * 1e-9 * 100.0, rel=1e-4)
    assert abs(np.asarray(r.v("a"))[-1]) < 1e-9


def test_an_ideal_clamp_with_nothing_to_dump_books_nothing():
    """Control: C1 at 0 V (D at its boundary, nothing moves when S closes): no record."""
    b = _clamped_dump(0.0, 10.0)
    r = _quiet(p.simulate, b, t_end=4e-6, dt=DT, switch_fn=_gate_n(b), **RULE)
    assert [x.dW for x in r.reinit_records if x.t > 1e-6] == []


def _share_beside_lc(v2):
    """C1 1 nF (10 V) -ideal S- c: C2 50 pF (v2; None: no C2), L 10 uH (0 A) -1 mOhm- gnd."""
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1e-9, 10.0)
    b.add_ideal_switch("S", "a", "c", 0.0, 0.0)
    if v2 is not None:
        b.add_capacitor("C2", "c", "gnd", 50e-12, v2)
    b.add_inductor("L", "c", "o", 10e-6, 0.0)
    b.add_resistor("R", "o", "gnd", 1e-3)
    return b


@pytest.mark.parametrize("split", [False, True])
def test_an_ideal_share_beside_an_lc_has_its_row(split):
    """Item 3: with the split the closing lands at T_E on a half step, where no
    state's d clears its slow motion (the LC rings at 0.64 us): the share was not
    read as a jump, the row pinned C1 and C2 to x0 (which the loop cannot meet),
    the pin diverged and no row was stored (n_right_limit_fallbacks 1). Truth: the
    charge share v+ = (C1 v_a- + C2 v_c-)/(C1 + C2) on both, i_L continuous."""
    b = _share_beside_lc(0.0)
    r = _quiet(p.simulate, b, t_end=3e-6, dt=DT, switch_fn=_gate_n(b),
               enable_substep_state_correction=split, **RULE)
    assert r.n_right_limit_fallbacks == 0
    v, t, k = _limits(r)
    (j,) = [q for q in k if t[q] > 1e-6]
    va, vc, il = (np.asarray(x) for x in (v.v("a"), v.v("c"), v.i("L")))
    want = (1e-9 * va[j] + 50e-12 * vc[j]) / (1e-9 + 50e-12)
    assert va[j + 1] == pytest.approx(want, rel=1e-6)
    assert vc[j + 1] == pytest.approx(want, rel=1e-6)
    assert il[j + 1] == pytest.approx(il[j], abs=1e-9)


@pytest.mark.parametrize("split", [False, True])
def test_a_closing_beside_an_lc_with_nothing_to_share_is_continuous(split):
    """Control: no C2 (no loop, nothing to share): the row is the left limit's states and
    S conducts (v_a = v_c after; C1 rings down through L)."""
    b = _share_beside_lc(None)
    r = _quiet(p.simulate, b, t_end=3e-6, dt=DT, switch_fn=_gate_n(b),
               enable_substep_state_correction=split, **RULE)
    assert r.n_right_limit_fallbacks == 0
    v, t, k = _limits(r)
    (j,) = [q for q in k if t[q] > 1e-6]
    va, vc, il = (np.asarray(x) for x in (v.v("a"), v.v("c"), v.i("L")))
    assert va[j + 1] == pytest.approx(va[j], rel=1e-6)
    assert il[j + 1] == pytest.approx(il[j], abs=1e-9)
    np.testing.assert_allclose(va[j + 1:], vc[j + 1:], rtol=0, atol=1e-9)
    assert va[-1] < 9.0   # the ring through S moved both


def test_a_due_right_limit_is_stored_or_counted():
    """ll13 (active-clamp forward, pwl + rule): at S_main's turn-offs (1.5,
    5.5, ... us) the record steps' right-limit solve (the ε mode, Newton)
    throws; the row was dropped with n_right_limit_fallbacks = 0. Every
    reinit record's step without a row is now a counted fallback."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path[:0] = [str(root / "tools"), str(root)]
    v1_reader = pytest.importorskip("v1_reader")
    case = v1_reader.load(str(root / "circuits" / "switching" / "isolated" /
                              "ll13_active_clamp_forward.yaml"))
    r = _quiet(case.run, "pwl", discontinuity_restart=True)
    t = np.asarray(r.times)
    rows = {float(t[i]) for i in np.asarray(r.right_limit_index)}
    missing = {float(rec.t) for rec in r.reinit_records} - rows
    assert missing                                   # measured: 19 turn-offs
    assert r.n_right_limit_fallbacks >= len(missing), (len(missing), r.n_right_limit_fallbacks)


def _rl_from_source():
    # 10 V -> R 2 ohm -> node a -> L 1 mH -> gnd
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_resistor("R", "vin", "a", 2.0)
    b.add_inductor("L", "a", "gnd", 1e-3)
    return b


def test_an_inconsistent_start_publishes_its_right_limit_row():
    """F-Z-10: from the zero state with the source on, row 0 is the user's state
    (all zeros, not a solution) and x(0+) is a right-limit row: i_L = 0 is
    continuous, so v_in = v_a = 10 V (no current through R)."""
    r = _quiet(p.simulate, _rl_from_source(), t_end=5e-6, dt=DT, **RULE)
    v, t, kk = _limits(r, start=True)
    assert [float(t[j]) for j in kk] == [0.0]
    assert np.asarray(v.v("a"))[0] == 0.0 and np.asarray(v.v("vin"))[0] == 0.0
    assert np.asarray(v.v("vin"))[1] == pytest.approx(10.0, rel=1e-9)
    assert np.asarray(v.v("a"))[1] == pytest.approx(10.0, rel=1e-9)
    assert abs(float(np.asarray(v.i("L"))[1])) <= 1e-9 * 5.0   # the pin's 1e-9 of scale
    # plain rows are unchanged by the record
    assert np.asarray(r.times)[0] == 0.0 and np.asarray(r.times)[1] == DT


def test_a_consistent_start_publishes_no_start_row():
    """The same cell from its DC operating point (i_L = 5 A, v_a = 0): nothing
    is re-initialised beyond the noise band, no row at t = 0."""
    r = _quiet(p.simulate, _rl_from_source(), t_end=5e-6, dt=DT,
               start_from_dc_op=True, **RULE)
    idx = r.right_limit_index
    assert idx is None or len(idx) == 0


def test_the_loss_summary_books_the_start_jump_once():
    """F-Z-10: the diode rig from zeros (10 A forced at t = 0): the summary missed
    h/2·p(0+) (0.23994 J); with the start row it is the kernel's 12 W · t_end."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_device_losses_kernel import _diode_rig
    b = _diode_rig(r_th=1.0, c_th=0.01)
    b.set_device_losses("D1", junction="tj")
    r = _quiet(p.simulate, b, 0.02, dt=1e-5, engine="pwl", discontinuity_restart=True)
    (row,) = [q for q in p.device_loss_summary(b, r) if q["name"] == "D1"]
    E = r.device_energy("D1")["E_cond_electrical"]
    assert row["E_total"] == pytest.approx(E, rel=1e-12)
    assert E == pytest.approx(12.0 * 0.02, rel=1e-12)


def _kcl_mid_bound(vcc, vm, il, n):
    """|KCL| at the 100k/100k divider node mid (it sees only the divider and the
    inductor): after one refinement step of the right-limit solve the residual is
    componentwise backward stable (Skeel), |r| <= n·eps·(|A|·|x|)_mid, with no
    condition factor (L/ε = 1e7 Ω beside 1e5 Ω rows only enters the forward error).
    The factor 4 is the margin over the n·eps of the bound itself."""
    eps = np.finfo(float).eps
    return 4 * n * eps * (abs(vcc) / 1e5 + abs(vm) * 2e-5 + abs(il))


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("L_h", [1e-3, 1e-6])
def test_a_right_limit_row_satisfies_kcl_of_the_post_edge_network(split, L_h):
    """F-Z-4: a 100 V source with a 100k/100k divider (mid) feeding L -> 12 Ω, and a
    1 µF capacitor switched onto the source at T_E. Before, the right-limit row read
    V(mid) 1.0e-7 V (split 2.0e-6 V) off 50 - 5e4·i_L for 1 mH (1e-9 V for 1 µH,
    the round-off of L/ε): a non-solution of the post-edge network."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vcc", "gnd", 100.0)
    b.add_resistor("R1", "vcc", "mid", 1e5)
    b.add_resistor("R2", "mid", "gnd", 1e5)
    b.add_inductor("L", "mid", "lm", L_h)
    b.add_resistor("RL", "lm", "gnd", 12.0)
    b.add_switch("S", "vcc", "c", 1e3, 1e-9)
    b.add_capacitor("C", "c", "gnd", 1e-6, 0.0)
    k, ns = b.switch_index_of("S"), b.graph.num_switches

    def fn(t):
        m = p.SwitchStateMask(ns)
        m.set(k, t >= T_E)
        return m
    r = _quiet(p.simulate, b, t_end=6e-6, dt=DT, switch_fn=fn, start_from_dc_op=True,
               enable_substep_state_correction=split, **RULE)
    v, t, kk = _limits(r)
    assert len(kk) == 1
    n = np.asarray(r.states).shape[1]
    j = kk[0] + 1
    vcc, vm, il = (float(np.asarray(v.v("vcc"))[j]), float(np.asarray(v.v("mid"))[j]),
                   float(np.asarray(v.i("L"))[j]))
    kcl = (vcc - vm) / 1e5 - vm / 1e5 - il
    assert abs(kcl) <= _kcl_mid_bound(vcc, vm, il, n)


def test_ll09_right_limit_rows_satisfy_kcl_at_mid(monkeypatch):
    """F-Z-4 on ll09 itself (the finding's 4.6e-11 A at mid on the release line)."""
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks" / "tools"))
    pytest.importorskip("yaml")
    import v1_reader

    case = v1_reader.load(ROOT / "benchmarks/circuits/switching/inverter/"
                          "ll09_half_bridge_inverter_leg.yaml")
    r = _quiet(case.run, "pwl", discontinuity_restart=True)
    v, t, k = _limits(r)
    n = np.asarray(r.states).shape[1]
    vcc, vm, il = (np.asarray(v.v("vcc"))[k + 1], np.asarray(v.v("mid"))[k + 1],
                   np.asarray(v.i("Lload"))[k + 1])
    kcl = (vcc - vm) / 1e5 - vm / 1e5 + il   # Lload's current runs load_mid -> mid
    assert len(k) > 0
    assert np.all(np.abs(kcl) <= _kcl_mid_bound(vcc, vm, il, n))
