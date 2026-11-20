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
