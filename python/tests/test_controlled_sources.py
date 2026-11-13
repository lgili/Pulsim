"""VCCS, CCVS, CCCS, gyrator and the sine / pulse / table current sources.

Every expected number is analytic (openspec change
add-controlled-source-set, design.md "Validation plan"). The DC
two-ports and the stamp entries are in
core/tests/layer2/test_controlled_sources.cpp.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p
from pulsim import _pulsim

DT = 1e-6


def rel_err(a, b, scale=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.max(np.abs(b)) if scale is None else scale
    return float(np.max(np.abs(a - b)) / s)


# ---- the gyrator ---------------------------------------------------------

def gyrator_cap(g=0.01, C=1e-6):
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.add_resistor("R1", "in", "p1", 10.0)
    b.add_gyrator("Y1", "p1", "gnd", "p2", "gnd", g)
    b.add_capacitor("C2", "p2", "gnd", C)
    return b


def plain_inductor(L=10e-3):
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.add_resistor("R1", "in", "p1", 10.0)
    b.add_inductor("L1", "p1", "gnd", L)
    return b


def exact_rl(t):
    # C/g² = 1e-6 / 1e-4 = 10 mH; τ = L/R = 1 ms; 1 V through 10 Ω.
    return 0.1 * (1 - np.exp(-np.asarray(t) / 1e-3))


def test_gyrator_turns_a_capacitor_into_an_inductor():
    gy = p.simulate(gyrator_cap(), t_end=1e-3, dt=DT, engine="pwl")
    ind = p.simulate(plain_inductor(), t_end=1e-3, dt=DT, engine="pwl")
    assert rel_err(gy.i("R1"), ind.i("R1")) < 1e-9
    # The fixed engine starts first order (guide trap 9): its error
    # against the exponential is dt/(2τ), the same with the inductor.
    err = rel_err(gy.i("R1"), exact_rl(gy.times), scale=0.1)
    assert err == pytest.approx(0.5 * DT / 1e-3, rel=1e-3)


@pytest.mark.parametrize("engine,tol", [("trbdf2", 1e-6), ("dsed", 1e-12)])
def test_gyrator_matches_the_analytic_current(engine, tol):
    res = p.simulate(gyrator_cap(), t_end=1e-3, engine=engine,
                     rtol=1e-10, atol=1e-13)
    assert rel_err(res.i("R1"), exact_rl(res.times), scale=0.1) < tol
    assert res.i("R1")[-1] == pytest.approx(0.1 * (1 - math.exp(-1)),
                                            rel=1e-6)
    assert round(res.i("R1")[-1] * 1e3, 2) == 63.21


def test_gyrator_port_currents_and_power_balance():
    # An inductor feeds port 1 so i_1 is a solved state, not v/R.
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.add_inductor("L1", "in", "p1", 1e-3)
    b.add_gyrator("Y1", "p1", "gnd", "p2", "gnd", 0.02)
    b.add_resistor("R2", "p2", "gnd", 50.0)
    res = p.simulate(b, t_end=2e-4, dt=DT, engine="pwl")
    v1, v2 = res.v("p1"), res.v("p2")
    i1 = res.i("L1")                 # into p1_pos
    i2 = -res.i("R2")                # into p2_pos
    assert np.max(np.abs(v1 * i1 + v2 * i2)
                  - 1e-12 * np.abs(v1 * i1)) <= 1e-15
    # res.i(port=) agrees with the terminal currents.
    assert rel_err(res.i("Y1", port=1), i1) < 1e-9
    assert rel_err(res.i("Y1", port=2), i2) < 1e-9
    assert rel_err(res.i("Y1"), i2) < 1e-9
    with pytest.raises(ValueError, match="gyrator"):
        res.i("R2", port=1)


# ---- AC sweep --------------------------------------------------------------

def test_gyrator_impedance_in_the_ac_sweep():
    b = p.CircuitBuilder()
    b.add_gyrator("Y1", "p1", "gnd", "p2", "gnd", 0.01)
    b.add_capacitor("C2", "p2", "gnd", 1e-6)
    n1 = b.node_id_of("p1")
    # Unit current injected at p1 -> H = v(p1) = Z_1.
    res = _pulsim.run_mna_sweep_kernel(
        b.graph, b.pool, p.SwitchStateMask(b.graph.num_switches),
        [1e3], n1, n1)
    z = complex(res.H[0])
    assert abs(z) == pytest.approx(2 * math.pi * 1e3 * 10e-3, rel=1e-9)
    assert abs(z) == pytest.approx(62.83, abs=5e-3)
    assert math.degrees(math.atan2(z.imag, z.real)) == pytest.approx(
        90.0, rel=1e-9)


def test_vccs_stage_gain_is_flat_in_the_ac_sweep():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 0.0)
    b.add_resistor("Rin", "in", "gnd", 1e3)
    b.add_vccs("G1", "in", "gnd", "o", "gnd", 2e-3)
    b.add_resistor("RL", "o", "gnd", 5e3)
    b.add_capacitor("Cx", "in", "gnd", 1e-9)  # a dynamic row elsewhere
    src = b.pool.branch_var_id_for_source(b.branch_index_of("Vin"), b.graph)
    freqs = [1.0, 1e3, 1e6]
    res = _pulsim.run_mna_sweep_kernel(
        b.graph, b.pool, p.SwitchStateMask(0), freqs, src,
        b.node_id_of("o"))
    H = np.asarray(res.H, dtype=complex)
    assert np.max(np.abs(H - 10.0)) / 10.0 < 1e-9   # gm·R = 10


# ---- VCCS integrator -------------------------------------------------------

def integrator():
    # dθ/dt = ω: 1 S from a 3 V node into 1 F gives v(t) = 3·t.
    b = p.CircuitBuilder()
    b.add_voltage_source("W", "w", "gnd", 3.0)
    b.add_vccs("G1", "w", "gnd", "th", "gnd", 1.0)
    b.add_capacitor("C1", "th", "gnd", 1.0)
    return b


@pytest.mark.parametrize("engine", ["trbdf2", "dsed"])
def test_vccs_integrates_a_node_voltage(engine):
    # auto_regularize would tie the capacitor-only node with 1 GOhm.
    res = p.simulate(integrator(), t_end=1e-3, engine=engine,
                     rtol=1e-10, atol=1e-14, auto_regularize=False)
    t = np.asarray(res.times)
    assert np.max(np.abs(res.v("th") - 3.0 * t)) <= 1e-9 * 3e-3


def test_vccs_integrator_on_the_fixed_grid():
    dt = 1e-5
    res = p.simulate(integrator(), t_end=1e-3, dt=dt, engine="pwl",
                     auto_regularize=False)
    dv = np.diff(res.v("th"))
    # The first step averages i_C(0) = 0 (trap 9); then exactly 3·dt.
    assert dv[0] == pytest.approx(1.5 * dt, rel=1e-9)
    assert np.max(np.abs(dv[1:] - 3 * dt)) <= 1e-9 * 3 * dt


# ---- engines agree, dsed runs or refuses -----------------------------------

def every_controlled_kind():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.5, 1.0, 2e3)
    b.add_resistor("Ra", "a", "b", 100.0)
    b.add_voltage_source("Vsense", "b", "c", 0.0)
    b.add_capacitor("Cc", "c", "gnd", 1e-6)
    b.add_vccs("G1", "a", "gnd", "g", "gnd", 1e-3)
    b.add_resistor("Rg", "g", "gnd", 1e3)
    b.add_cccs("F1", "Vsense", "f", "gnd", 2.0)
    b.add_resistor("Rf", "f", "gnd", 50.0)
    b.add_ccvs("H1", "Vsense", "h", "gnd", 30.0)
    b.add_resistor("Rh", "h", "gnd", 10.0)
    b.add_gyrator("Y1", "g", "gnd", "y", "gnd", 0.01)
    b.add_capacitor("Cy", "y", "gnd", 1e-7)
    b.add_resistor("Ry", "y", "gnd", 1e3)
    return b


NODES = ("a", "c", "g", "f", "h", "y")


def max_rel(ref, other, nodes, t_from):
    """Max error of `other` against `ref` (interpolated onto other's
    samples from t_from on), relative to each node's peak."""
    t = np.asarray(other.times)
    m = t >= t_from
    return {n: float(np.max(np.abs(np.interp(t[m], ref.times, ref.v(n))
                                   - other.v(n)[m]))
                     / np.max(np.abs(ref.v(n)))) for n in nodes}


def test_trbdf2_and_dsed_agree_on_every_controlled_kind():
    tr = p.simulate(every_controlled_kind(), t_end=1e-3, engine="trbdf2",
                    rtol=1e-10, atol=1e-13)
    ds = p.simulate(every_controlled_kind(), t_end=1e-3, dt=1e-7,
                    engine="dsed", rtol=1e-10, atol=1e-13)
    assert (tr.engine_used, ds.engine_used) == ("trbdf2", "dsed")
    # From 10 us on: sample 0 differs by design (dsed reports the
    # consistent algebraic state, the others 0 V).
    err = max_rel(ds, tr, NODES, 1e-5)
    assert max(err.values()) < 1e-6, err


def test_fixed_engine_converges_to_them_at_first_order():
    tr = p.simulate(every_controlled_kind(), t_end=1e-3, engine="trbdf2",
                    rtol=1e-10, atol=1e-13)
    e1, e2 = (max(max_rel(tr, p.simulate(every_controlled_kind(),
                                         t_end=1e-3, dt=dt, engine="pwl"),
                          NODES, 1e-5).values())
              for dt in (1e-7, 1e-8))
    assert e1 < 1e-3 and e2 < 0.15 * e1, (e1, e2)


# ---- waveform current sources ----------------------------------------------

def test_sine_current_into_a_resistor():
    b = p.CircuitBuilder()
    b.add_sine_current_source("I1", "o", "gnd", 0.0, 1.0, 1e3)
    b.add_resistor("R", "o", "gnd", 10.0)
    res = p.simulate(b, t_end=2e-3, dt=DT, engine="pwl")
    t = np.asarray(res.times)
    i = np.sin(2 * np.pi * 1e3 * t)
    assert np.max(np.abs(res.v("o") - 10 * i)) <= 1e-12 * 10
    assert rel_err(res.i("I1"), i) < 1e-12


def test_pulse_current_matches_the_pulse_voltage_source():
    kw = dict(t_start=1e-4, pulse_width=2e-4, period=5e-4,
              rise_time=5e-5, fall_time=5e-5)
    bi = p.CircuitBuilder()
    bi.add_pulse_current_source("I1", "o", "gnd", 0.2, 1.5, **kw)
    bi.add_resistor("R", "o", "gnd", 1.0)
    bv = p.CircuitBuilder()
    bv.add_pulse_voltage_source("V1", "o", "gnd", 0.2, 1.5, **kw)
    bv.add_resistor("R", "o", "gnd", 1.0)
    ri = p.simulate(bi, t_end=1.5e-3, dt=DT, engine="pwl")
    rv = p.simulate(bv, t_end=1.5e-3, dt=DT, engine="pwl")
    assert np.max(np.abs(ri.v("o") - rv.v("o"))) <= 1e-12 * 1.5
    assert ri.v("o").max() == pytest.approx(1.5)


def triangle(b, t0=0.0, periodic=False):
    b.add_pwl_current_source("I1", "o", "gnd", [t0, t0 + 1e-3, t0 + 2e-3],
                             [0.0, 1.0, 0.0], periodic=periodic)
    b.add_capacitor("C1", "o", "gnd", 1e-3)
    return b


def test_table_source_delivers_its_charge_on_the_grid():
    res = p.simulate(triangle(p.CircuitBuilder()), t_end=3e-3, dt=1e-5,
                     engine="pwl", auto_regularize=False)
    assert res.v("o")[-1] == pytest.approx(1.0, rel=1e-9)


def waveform_rc(kind):
    b = p.CircuitBuilder()
    if kind == "sine":
        b.add_sine_current_source("I1", "o", "gnd", 0.2, 1.0, 1e3, 0.3)
    elif kind == "pulse":
        b.add_pulse_current_source("I1", "o", "gnd", 0.2, 1.5, 1e-4, 2e-4,
                                   period=5e-4, rise_time=5e-5,
                                   fall_time=5e-5)
    else:
        b.add_pwl_current_source("I1", "o", "gnd", [0, 1e-3, 2e-3],
                                 [0, 1, 0], periodic=True)
    b.add_resistor("R", "o", "gnd", 10.0)
    b.add_capacitor("C", "o", "gnd", 1e-5)
    return b


@pytest.mark.parametrize("kind", ["sine", "pulse", "pwl"])
def test_waveform_sources_agree_on_trbdf2_and_dsed(kind):
    tr = p.simulate(waveform_rc(kind), t_end=2e-3, engine="trbdf2",
                    rtol=1e-10, atol=1e-13)
    ds = p.simulate(waveform_rc(kind), t_end=2e-3, dt=1e-7, engine="dsed",
                    rtol=1e-10, atol=1e-13)
    assert ds.engine_used == "dsed"
    assert max_rel(ds, tr, ["o"], 1e-5)["o"] < 1e-6


def test_table_breakpoints_are_not_stepped_over_on_trbdf2():
    # Knots at 0.1234567, 1.1234567, 2.1234567 ms: no natural step.
    t0 = 1.234567e-4
    res = p.simulate(triangle(p.CircuitBuilder(), t0=t0), t_end=3e-3,
                     engine="trbdf2", auto_regularize=False)
    assert res.v("o")[-1] == pytest.approx(1.0, rel=1e-6)
    t = np.asarray(res.times)
    for knot in (t0, t0 + 1e-3, t0 + 2e-3):
        assert np.min(np.abs(t - knot)) < 1e-15, knot


@pytest.mark.parametrize("t,i,needle", [
    ([0.0], [1.0], "two"),
    ([0.0, 2e-3, 1e-3], [0.0, 1.0, 0.0], "increase"),
    ([0.0, 1e-3], [0.0, float("nan")], "finite"),
    ([0.0, 1e-3, 2e-3], [0.0, 1.0], "lengths"),
])
def test_bad_tables_are_refused_naming_the_source(t, i, needle):
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match=needle) as e:
        b.add_pwl_current_source("I_tab", "o", "gnd", t, i)
    assert "I_tab" in str(e.value)


def test_periodic_table_must_close():
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match="I_tab.*periodic"):
        b.add_pwl_current_source("I_tab", "o", "gnd", [0, 1e-3], [0, 1],
                                 periodic=True)


# ---- res.i ------------------------------------------------------------------

def test_res_i_matches_a_series_milliohm():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 1.0, 2.0, 1e3)
    b.add_resistor("Ra", "a", "b", 10.0)
    b.add_voltage_source("Vsense", "b", "gnd", 0.0)
    b.add_vccs("G1", "a", "gnd", "g", "gnd", 1e-2)
    b.add_resistor("Rsg", "g", "g2", 1e-3)
    b.add_resistor("Rg", "g2", "gnd", 100.0)
    b.add_cccs("F1", "Vsense", "f", "gnd", 3.0)
    b.add_resistor("Rsf", "f", "f2", 1e-3)
    b.add_resistor("Rf", "f2", "gnd", 20.0)
    b.add_ccvs("H1", "Vsense", "h", "gnd", 5.0)
    b.add_resistor("Rsh", "h", "h2", 1e-3)
    b.add_resistor("Rh", "h2", "gnd", 20.0)
    b.add_resistor("Rsy", "a", "y1", 1e-3)
    b.add_gyrator("Y1", "y1", "gnd", "y2", "gnd", 0.05)
    b.add_resistor("Ry", "y2", "gnd", 30.0)
    b.add_pulse_current_source("Ip", "q", "gnd", 0.0, 1.0, 1e-4, 2e-4,
                               period=5e-4)
    b.add_resistor("Rsq", "q", "q2", 1e-3)
    b.add_resistor("Rq", "q2", "gnd", 5.0)
    res = p.simulate(b, t_end=1e-3, dt=1e-6, engine="pwl")
    assert rel_err(res.i("G1"), res.i("Rsg")) < 1e-9
    assert rel_err(res.i("F1"), res.i("Rsf")) < 1e-9
    # A voltage source's branch current runs from n_pos THROUGH it to
    # n_neg, so the current it delivers out of out_pos is −i.
    assert rel_err(-res.i("H1"), res.i("Rsh")) < 1e-9
    assert rel_err(res.i("Y1", port=1), res.i("Rsy")) < 1e-9
    assert rel_err(res.i("Ip"), res.i("Rsq")) < 1e-9


# ---- YAML -------------------------------------------------------------------

YAML_ALL = """
circuit:
  devices:
    - {type: voltage_source, name: Vs, from: a, to: gnd, V: 2.0}
    - {type: resistor, name: Ra, from: a, to: b, R: 10.0}
    - {type: voltage_source, name: Vsense, from: b, to: gnd, V: 0.0}
    - {type: vccs, name: G1, in_pos: a, in_neg: gnd, from: g, to: gnd, gm: 0.01}
    - {type: resistor, name: Rg, from: g, to: gnd, R: 100.0}
    - {type: ccvs, name: H1, control: Vsense, out_pos: h, out_neg: gnd, rm: 5.0}
    - {type: resistor, name: Rh, from: h, to: gnd, R: 20.0}
    - {type: cccs, name: F1, control: Vsense, from: f, to: gnd, gain: 3.0}
    - {type: resistor, name: Rf, from: f, to: gnd, R: 20.0}
    - {type: gyrator, name: Y1, p1_pos: g, p1_neg: gnd, p2_pos: y, p2_neg: gnd, g: 0.01}
    - {type: resistor, name: Ry, from: y, to: gnd, R: 30.0}
    - {type: sine_current_source, name: I1, from: s, to: gnd, i_dc: 0.1, i_amplitude: 1.0, frequency: 1000.0, phase: 0.5}
    - {type: resistor, name: Rs, from: s, to: gnd, R: 10.0}
    - {type: pulse_current_source, name: I2, from: q, to: gnd, i_initial: 0.0, i_pulsed: 1.0, t_start: 1.0e-4, pulse_width: 2.0e-4, period: 5.0e-4, rise_time: 1.0e-5, fall_time: 2.0e-5}
    - {type: resistor, name: Rq, from: q, to: gnd, R: 5.0}
    - {type: pwl_current_source, name: I3, from: w, to: gnd, t: [0, 1.0e-3, 2.0e-3], i: [0, 1, 0], periodic: true}
    - {type: resistor, name: Rw, from: w, to: gnd, R: 1.0}
"""


def python_all():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vs", "a", "gnd", 2.0)
    b.add_resistor("Ra", "a", "b", 10.0)
    b.add_voltage_source("Vsense", "b", "gnd", 0.0)
    b.add_vccs("G1", "a", "gnd", "g", "gnd", 0.01)
    b.add_resistor("Rg", "g", "gnd", 100.0)
    b.add_ccvs("H1", "Vsense", "h", "gnd", 5.0)
    b.add_resistor("Rh", "h", "gnd", 20.0)
    b.add_cccs("F1", "Vsense", "f", "gnd", 3.0)
    b.add_resistor("Rf", "f", "gnd", 20.0)
    b.add_gyrator("Y1", "g", "gnd", "y", "gnd", 0.01)
    b.add_resistor("Ry", "y", "gnd", 30.0)
    b.add_sine_current_source("I1", "s", "gnd", 0.1, 1.0, 1000.0, 0.5)
    b.add_resistor("Rs", "s", "gnd", 10.0)
    b.add_pulse_current_source("I2", "q", "gnd", 0.0, 1.0, 1e-4, 2e-4,
                               period=5e-4, rise_time=1e-5, fall_time=2e-5)
    b.add_resistor("Rq", "q", "gnd", 5.0)
    b.add_pwl_current_source("I3", "w", "gnd", [0, 1e-3, 2e-3], [0, 1, 0],
                             periodic=True)
    b.add_resistor("Rw", "w", "gnd", 1.0)
    return b


def test_yaml_and_python_build_identical_components():
    lc = p.load_yaml_string(YAML_ALL)
    assert lc.builder.components() == python_all().components()
    kinds = {d["name"]: d["kind"] for d in lc.builder.components()}
    assert kinds["Y1"] == "gyrator" and kinds["I3"] == "pwl_current_source"


def test_yaml_table_source_value():
    lc = p.load_yaml_string(YAML_ALL)
    bid = lc.builder.branch_index_of("I3")
    assert lc.builder.pool.current_waveform_value(bid, 0.5e-3) == 0.5


def test_yaml_table_from_a_file(tmp_path):
    f = tmp_path / "load.csv"
    f.write_text("# t, i\n0, 0\n1e-3, 1\n2e-3, 0\n")
    lc = p.load_yaml_string(
        "circuit:\n  devices:\n"
        f"    - {{type: pwl_current_source, name: I3, from: w, to: gnd, "
        f"file: '{f}'}}\n"
        "    - {type: resistor, name: Rw, from: w, to: gnd, R: 1.0}\n")
    bid = lc.builder.branch_index_of("I3")
    assert lc.builder.pool.current_waveform_value(bid, 0.5e-3) == 0.5


@pytest.mark.parametrize("yaml_dev,needles", [
    ("{type: gyrator, name: Y9, p1_pos: a, p1_neg: gnd, p2_pos: b, "
     "p2_neg: gnd, g: 0.01, r: 100}", ("Y9", "unknown key", "'r'")),
    ("{type: cccs, name: F9, control: Rx, from: o, to: gnd, gain: 2}",
     ("F9", "Rx")),
    ("{type: pwl_current_source, name: I9, from: o, to: gnd, "
     "t: [0, 1, 2], i: [0, 1]}", ("I9", "lengths")),
    ("{type: vccs, name: G9, in_pos: a, in_neg: gnd, from: o, to: gnd}",
     ("G9", "gm")),
])
def test_yaml_refusals_name_the_device(yaml_dev, needles):
    text = ("circuit:\n  devices:\n"
            "    - {type: resistor, name: Rx, from: a, to: gnd, R: 1.0}\n"
            f"    - {yaml_dev}\n")
    with pytest.raises(Exception) as e:
        p.load_yaml_string(text)
    for n in needles:
        assert n in str(e.value), (n, str(e.value))


def test_current_source_direction_is_out_of_n_pos():
    # The measured convention the two header comments are reconciled to.
    b = p.CircuitBuilder()
    b.add_current_source("I1", "n0", "gnd", 10e-3)
    b.add_resistor("R", "n0", "gnd", 1e3)
    res = p.simulate(b, t_end=1e-5, dt=1e-6, engine="pwl")
    assert res.v("n0")[-1] == pytest.approx(10.0, rel=1e-12)
