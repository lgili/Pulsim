"""add-power-mosfet-datasheet-device — the composite device, the charge laws
on the engines, the double-pulse helper, the VDMOS import and the parts
schema (tasks 1.6, 2.x, 3.x, 4.x)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

VDC, IL = 400.0, 10.0
# Review probe p4_dpt.py, row 3 (hand-built: external linear C_gs = 1 nF,
# C_gd = 100 pF, R_g = 10 ohm), measured 2026-11-12 on pwl at dt = 0.2 ns.
P4_ROW3 = {"Eon_uJ": 136.14, "Eoff_uJ": 158.96}
DEV = dict(K=1.43, V_T=3.0, lambda_=0.0, kappa=15.0, C_gs=1e-9, C_gd=100e-12, C_ds=0.0)


# ---------------------------------------------------------------------------
# the review probe's fixture and windows, verbatim
# ---------------------------------------------------------------------------
def _p4(composite: bool, Rg: float = 10.0, engine: str = "pwl"):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", VDC)
    b.add_inductor("L", "vdc", "sw", 500e-6, i0=IL)
    b.add_shockley_diode("Df", "sw", "vdc")
    b.add_resistor("Rsense", "sw", "d", 1e-3)
    if composite:
        b.add_power_mosfet("M", "d", "gnd", "drv", R_g=Rg, **DEV)
        gate = "M.g"
    else:
        b.add_mosfet_level1("M", "d", "gnd", "g", 1.43, 3.0, 0.0, 15.0, False)
        gate = "g"
    b.add_pulse_voltage_source("Vg", "drv", "gnd", 0.0, 12.0, 100e-9, 700e-9, 0.0, 5e-9, 5e-9)
    if not composite:
        b.add_resistor("Rg", "drv", "g", Rg)
        b.add_capacitor("Cgs", "g", "gnd", 1e-9)
        b.add_capacitor("Cgd", "g", "d", 100e-12)
    kw = dict(dt=2e-10) if engine == "pwl" else {}
    r = p.simulate(b, t_end=1.5e-6, engine=engine, **kw)
    t = np.asarray(r.times)
    vd = np.asarray(r.v("d"))
    vg = np.asarray(r.v(gate))
    i = (np.asarray(r.v("sw")) - vd) / 1e-3
    pw = vd * i
    on = (t > 0.09e-6) & (t < 0.6e-6)
    off = t > 0.75e-6
    m = on & (vg > 5.3) & (vg < 6.0)
    return dict(Eon_uJ=np.trapezoid(pw[on], t[on]) * 1e6,
                Eoff_uJ=np.trapezoid(pw[off], t[off]) * 1e6,
                plateau_ns=np.sum(np.diff(t)[m[:-1]]) * 1e9)


def test_composite_reproduces_the_hand_built_double_pulse():
    """Task 2.4: one add_power_mosfet call == the probe's hand-built circuit."""
    hand = _p4(composite=False)
    comp = _p4(composite=True)
    for k in ("Eon_uJ", "Eoff_uJ"):
        assert comp[k] == pytest.approx(hand[k], rel=1e-9), k
        # the probe's published values (3 significant digits after the point)
        assert comp[k] == pytest.approx(P4_ROW3[k], rel=2e-4), k
    assert comp["plateau_ns"] == pytest.approx(hand["plateau_ns"], rel=1e-6)


# ---------------------------------------------------------------------------
# double_pulse_test
# ---------------------------------------------------------------------------
def _hand_plateau(Rg: float) -> float:
    V_pl = 3.0 + math.sqrt(IL / 1.43)
    I_g = (12.0 - V_pl) / Rg
    return 100e-12 * VDC / I_g          # Q_gd / I_g


@pytest.mark.parametrize("Rg", [10.0, 47.0])
def test_double_pulse_plateau_follows_gate_charge(Rg):
    """Task 2.5: plateau within 10 % of Q_gd / I_g. Measured 2026-11-12:
    64.57 ns vs 62.94 ns (10 ohm), 300.56 ns vs 295.80 ns (47 ohm)."""
    r = p.double_pulse_test(DEV, V_dd=VDC, I_load=IL, R_g_ext=Rg, V_drive=12.0,
                            L_load=500e-6, engine="trbdf2")
    w = r.waveforms
    pre = w["t"] < 100e-9
    # consistent start: from an all-zero state C_gd turned the DUT on
    # before t_on (gate 5.6 V, drain down to 62 V) and halved the 47 ohm plateau
    assert w["v_gs"][pre].max() < 1e-3
    assert w["v_ds"][pre].min() > VDC
    assert r.t_plateau == pytest.approx(_hand_plateau(Rg), rel=0.10)
    assert r.V_plateau == pytest.approx(3.0 + math.sqrt(IL / 1.43), rel=0.02)


@pytest.mark.parametrize("body", [None, {"model": "lauritzen", "tau": 50e-9, "T_M": 10e-9}])
def test_double_pulse_engines_agree(body):
    """Task 2.6: pwl (uniform grid) vs trbdf2 (adaptive). Measured
    2026-11-12: 0.058 % / 0.079 % (E_on / E_off) without a body diode,
    0.058 % / 0.078 % with it; pinned at 0.2 % (2.5x margin)."""
    dev = dict(DEV, body_diode=body)
    a = p.double_pulse_test(dev, V_dd=VDC, I_load=IL, R_g_ext=10.0, V_drive=12.0,
                            L_load=500e-6, engine="pwl")
    b = p.double_pulse_test(dev, V_dd=VDC, I_load=IL, R_g_ext=10.0, V_drive=12.0,
                            L_load=500e-6, engine="trbdf2")
    assert b.n_points < a.n_points        # the grids really differ
    assert b.E_on == pytest.approx(a.E_on, rel=2e-3)
    assert b.E_off == pytest.approx(a.E_off, rel=2e-3)


def test_double_pulse_reverse_recovery_shows_with_a_lauritzen_freewheel():
    r = p.double_pulse_test(DEV, V_dd=VDC, I_load=IL, R_g_ext=10.0, V_drive=12.0,
                            L_load=500e-6, freewheel={"model": "lauritzen", "tau": 50e-9,
                                                      "T_M": 10e-9})
    s = p.double_pulse_test(DEV, V_dd=VDC, I_load=IL, R_g_ext=10.0, V_drive=12.0,
                            L_load=500e-6)
    assert r.I_rr > 5.0 > s.I_rr
    assert r.E_on > s.E_on


# ---------------------------------------------------------------------------
# static behaviour: third quadrant and T_j
# ---------------------------------------------------------------------------
def _dc(dev, V_s, V_gs):
    """Drive the drain from V_s through 1 ohm at a DC gate; returns (v_d, i_d).

    trbdf2 from a seeded gate: a 0 -> V_gs step at t = 0 through C_gs
    stalls TR-BDF2 at h_min, and on pwl the inconsistent start leaves an
    undamped trapezoidal ring on the drain (4.7e-4 V peak-to-peak at
    dt = 10 ns, measured 2026-11-12)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vs", "s", "gnd", V_s)
    b.add_resistor("R", "s", "d", 1.0)
    b.add_voltage_source("Vgs", "g", "gnd", V_gs)
    b.add_power_mosfet("M", "d", "gnd", "g", **dev)
    x0 = np.array(b.initial_state(), dtype=float).ravel()
    names = b.state_var_names()
    x0[names.index("V(g)")], x0[names.index("V(s)")] = V_gs, V_s
    r = p.simulate(b, t_end=2e-6, engine="trbdf2", initial_state=x0)
    v_d = float(np.asarray(r.v("d"))[-1])
    return v_d, V_s - v_d


def _x_body(i_r, V_ov, K, I_S, n=1.0, V_T=0.025852, G=1e-12):
    """Exact Level-1 third-quadrant channel + Shockley body diode split:
    with the drain at -x the drain acts as the source, so the channel
    carries K((V_ov + x)·2x - x²) = K(2 V_ov x + x²)."""
    def f(x):
        return K * (2 * V_ov * x + x * x) + I_S * math.expm1(x / (n * V_T)) + G * x - i_r
    lo, hi = 0.0, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def test_third_quadrant_gate_on_and_off():
    """Task 2.7 (see DECISIONS.md 8 for the exact reference). Measured
    2026-11-12: 3e-16 (gate on) and 2.5e-15 (gate off) relative."""
    dev = dict(DEV, R_d=0.01, body_diode={"model": "shockley", "I_S": 1e-12})
    v_on, i_on = _dc(dev, -10.5, 12.0)
    v_off, i_off = _dc(dev, -10.5, 0.0)
    assert i_on < -9.0 and i_off < -9.0        # third quadrant, ~10 A
    x = _x_body(-i_on, 9.0, 1.43, 1e-12)
    assert v_on == pytest.approx(-(0.01 * -i_on + x), rel=1e-7)
    # small-signal R_DS(on) = R_d + 1/(2K V_ov), within the triode term
    assert v_on == pytest.approx(i_on * (0.01 + 1 / (2 * 1.43 * 9.0)), rel=0.05)
    # gate off: the body diode carries it (≈ 0.77 V for I_S = 1e-12)
    x_d = 0.025852 * math.log1p(-i_off / 1e-12)
    assert v_off == pytest.approx(-(0.01 * -i_off + x_d), rel=1e-7)


def test_on_resistance_follows_the_temperature_factor():
    """Task 2.8: r(125 °C) = 1 + 5e-3·100 + 2e-5·100² = 1.700. Measured
    2026-11-12: 1.70009 (the triode x² term at 0.09 A)."""
    dev = dict(DEV, R_d=0.05, rds_tc1=5e-3, rds_tc2=2e-5)
    v25, i25 = _dc(dict(dev, T_j=25.0), 0.1, 12.0)
    v125, i125 = _dc(dict(dev, T_j=125.0), 0.1, 12.0)
    assert v25 > 0 and i25 > 0
    assert (v25 / i25) == pytest.approx(0.05 + 1 / (2 * 1.43 * 9.0), rel=1e-3)
    assert (v125 / i125) / (v25 / i25) == pytest.approx(1.700, rel=1e-3)


# ---------------------------------------------------------------------------
# builder: introspection and refusals
# ---------------------------------------------------------------------------
def test_components_reports_one_power_mosfet_entry():
    b = p.CircuitBuilder()
    b.add_power_mosfet("Q1", "d", "s", "g", K=1.43, V_T=3.0, C_gs=1e-9,
                       C_gd=p.ChargeLaw.vdmos_cgd(1e-9, 50e-12, 0.5),
                       C_ds=p.ChargeLaw.junction(2e-9, 25.0, 0.5), R_g=3.0, R_d=0.01,
                       L_s=2e-9, body_diode={"model": "lauritzen"}, T_j=100.0, rds_tc1=4e-3)
    comps = b.components()
    pm = [c for c in comps if c["kind"] == "power_mosfet"]
    assert len(pm) == 1
    prm = pm[0]["params"]
    assert prm["R_g"] == 3.0 and prm["C_gs"] == 1e-9 and prm["T_j"] == 100.0
    assert prm["C_gd"]["law"] == "vdmos_cgd" and prm["C_gd"]["cgd_max"] == 1e-9
    assert prm["C_ds"]["law"] == "junction" and prm["body_diode"] == "lauritzen"
    assert prm["R_d_eff"] == pytest.approx(0.01 * 1.3) and prm["K_eff"] == pytest.approx(1.43 / 1.3)
    assert pm[0]["sub_elements"] == ["Q1.rd", "Q1.rg", "Q1.ls", "Q1", "Q1.cgs", "Q1.cgd",
                                     "Q1.cds", "Q1_body"]
    assert {c["name"] for c in comps} >= set(pm[0]["sub_elements"])


@pytest.mark.parametrize("kw,frag", [
    (dict(R_d=-1.0), "R_d"),
    (dict(C_gs=-1e-9), "C_gs"),
    (dict(K=0.0), "K"),
    (dict(T_j=125.0, rds_tc1=-0.02), "temperature factor"),
    (dict(C_gs=0.0, C_gd=0.0, R_g=5.0), "no charge"),
    (dict(C_gd=p.ChargeLaw.vdmos_cgd(1e-12, 1e-9, 0.5)), "CGDMAX"),
    (dict(body_diode={"model": "lauritzen", "BV": 600}), "BV"),
    (dict(body_diode={"model": "zener"}), "zener"),
])
def test_invalid_parameters_are_refused_by_name(kw, frag):
    b = p.CircuitBuilder()
    with pytest.raises((ValueError, TypeError), match=frag) as e:
        b.add_power_mosfet("Q9", "d", "s", "g", **dict(DEV, **kw))
    assert "Q9" in str(e.value)


_YAML = """\
circuit:
  devices:
    - {{type: voltage_source, name: V1, from: d, to: gnd, V: 1.0}}
    - {{type: voltage_source, name: V2, from: g, to: gnd, V: 0.0}}
    - {{type: power_mosfet, name: Q1, drain: d, source: gnd, gate: g, K: 1.43, V_T: 3.0,
       C_gs: 1.0e-9, R_g: 2.0,
       C_gd: {{law: vdmos_cgd, cgd_max: 1.0e-9, cgd_min: 5.0e-11, a: 0.5}},
       C_ds: {{law: table, v: [-1.0, 0.0, 10.0, 100.0], C: [3.0e-9, 2.0e-9, 1.0e-9, 4.0e-10]}},
       body_diode: {{model: shockley, I_S: 1.0e-14}}{extra}}}
simulation: {{t_start: 0.0, t_end: 1.0e-7, dt: 1.0e-9}}
"""


def test_yaml_power_mosfet_type_and_strict_keys():
    lc = p.load_yaml_string(_YAML.format(extra=""))
    pm = [c for c in lc.builder.components() if c["kind"] == "power_mosfet"]
    assert len(pm) == 1
    prm = pm[0]["params"]
    assert prm["C_gd"]["law"] == "vdmos_cgd" and prm["C_ds"]["law"] == "table"
    assert prm["body_diode"] == "shockley" and prm["R_g"] == 2.0
    with pytest.raises(RuntimeError, match="unknown key 'Rg'"):
        p.load_yaml_string(_YAML.format(extra=", Rg: 3.0"))


# ---------------------------------------------------------------------------
# charge laws on the engines (task 1.6)
# ---------------------------------------------------------------------------
def _laws():
    tab = p.ChargeTable.from_capacitance([-10.0, 0.0, 5.0, 20.0, 100.0, 400.0, 500.0],
                                         [3e-9, 2e-9, 1.2e-9, 6e-10, 2.5e-10, 1.2e-10, 1.1e-10])
    return {"junction": p.ChargeLaw.junction(2e-9, 25.0, 0.5),
            "table": p.ChargeLaw.table(tab),
            "vdmos_cgd": p.ChargeLaw.vdmos_cgd(1e-9, 50e-12, 0.5)}


@pytest.mark.parametrize("law", ["junction", "table", "vdmos_cgd"])
@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_charge_is_conserved_over_a_closed_cycle(law, engine):
    """0 → 400 V → 0 through 1 kΩ: the charge the resistor delivered equals
    Q(v_end) − Q(v_0) (≈ 0 on a closed cycle).  Measured 2026-11-12, as a
    fraction of the charge moved: pwl 2.2e-8..8.9e-8 for all three laws (the
    2.0 junction law included, so it is the engine's corner steps, not the
    law); trbdf2 6.6e-5..2.6e-4 (the output-grid trapezoid is not its
    quadrature)."""
    lw = _laws()[law]
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vs", "s", "gnd", 0.0, 400.0, 1e-6, 20e-6, 0.0, 2e-6, 2e-6)
    b.add_resistor("R", "s", "c", 1e3)
    b.add_nonlinear_capacitor("C", "c", "gnd", lw)
    kw = dict(dt=2e-9) if engine == "pwl" else {}
    r = p.simulate(b, t_end=60e-6, engine=engine, **kw)
    t = np.asarray(r.times)
    vs, vc = np.asarray(r.v("s")), np.asarray(r.v("c"))
    q_res = np.trapezoid((vs - vc) / 1e3, t)
    moved = lw.charge(400.0)
    assert vc.max() == pytest.approx(400.0, rel=1e-6)
    assert abs(vc[-1]) < 1e-6 * 400.0
    q_cap = lw.charge(float(vc[-1])) - lw.charge(float(vc[0]))
    tol = 1e-6 if engine == "pwl" else 2e-3
    assert abs(q_res - q_cap) < tol * moved


@pytest.mark.parametrize("law", ["junction", "table", "vdmos_cgd"])
@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_charge_is_conserved_to_1e9_under_the_engines_own_rule(law, engine):
    """Task 1.6 at 1e-9. The law in series with a 10 nF linear capacitor (a
    charge meter): both are stamped in charge form by the same rule and carry
    the same current at every solve, so the charge the law moved over the
    0 → 400 V → 0 cycle equals the meter's C·Δv under the engine's OWN
    quadrature (the output-grid trapezoid of the test above is not trbdf2's).
    Two engine terms enter the balance at the meter node and are accounted
    for, not hidden: no auto-regularize bleeder (a 1 GΩ tie leaks 2.3e-9 of
    the charge), and the 1e-12 S placeholder pwl/assemble.hpp puts on every
    NonlinearCapacitor node (the same 2.3e-9; it is what made the trapezoid
    test above read 2.2e-8 at 400 V). Measured 2026-11-18 (balance / charge
    moved): pwl 7e-14..4e-12, trbdf2 8.7e-11 (table), 1.8e-10 (vdmos_cgd),
    5.6e-10 (junction; 2.2e-9 at rtol 1e-7, not explained). Bound 1e-9 on
    pwl, 2e-9 on trbdf2 (adaptive steps)."""
    lw = _laws()[law]
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vs", "s", "gnd", 0.0, 400.0, 1e-6, 20e-6, 0.0, 2e-6, 2e-6)
    b.add_resistor("R", "s", "c", 1e3)
    b.add_nonlinear_capacitor("C", "c", "m", lw)
    b.add_capacitor("Cm", "m", "gnd", 10e-9)
    kw = dict(dt=2e-9) if engine == "pwl" else {}
    r = p.simulate(b, t_end=60e-6, engine=engine, auto_regularize=False,
                   initial_state=np.zeros(len(b.state_var_names())), **kw)
    t = np.asarray(r.times)
    v = np.asarray(r.v("c")) - np.asarray(r.v("m"))
    vm = np.asarray(r.v("m"))
    assert v.max() > 300.0 and abs(v[-1]) < 1.0
    q_law = lw.charge(float(v[-1])) - lw.charge(float(v[0]))
    q_meter = 10e-9 * float(vm[-1] - vm[0]) + 1e-12 * float(np.trapezoid(vm, t))
    tol = 1e-9 if engine == "pwl" else 2e-9
    assert abs(q_law - q_meter) < tol * lw.charge(float(v.max()))


def test_charge_law_objects():
    lw = _laws()
    assert lw["junction"].kind == "junction" and lw["table"].kind == "table"
    # ngspice 45.2 (probes/cgd.cir, 12 digits): v_gd = +5 V and -5 V
    assert lw["vdmos_cgd"].capacitance(5.0) == pytest.approx(9.95053510609e-10, rel=1e-9)
    assert lw["vdmos_cgd"].capacitance(-5.0) == pytest.approx(1.90610539423e-10, rel=1e-9)
    assert lw["junction"].charge(400.0) == pytest.approx(312.31e-9, rel=1e-4)
    with pytest.raises(ValueError, match="knot"):
        p.ChargeTable([0.0, 1.0, 2.0], [0.0, 1e-9, 2e-9], [1e-9, -1e-9, 1e-9])


# ---------------------------------------------------------------------------
# VDMOS import (task 3)
# ---------------------------------------------------------------------------
_CARD = """* vdmos
.model MV VDMOS(VTO=3 KP=2.86 CGS=1n CGDMAX=1n CGDMIN=50p A=0.5 RG=3{extra})
Vd d 0 DC 1
Vg g 0 DC 0
M1 d g 0 MV
.end
"""


def _import(text, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b = p.spice_to_builder(text, **kw)
    return b, " ".join(str(x.message) for x in w)


def test_vdmos_card_keeps_its_charge():
    b, msg = _import(_CARD.format(extra=""))
    pm = [c for c in b.components() if c["kind"] == "power_mosfet"]
    assert len(pm) == 1
    prm = pm[0]["params"]
    assert prm["C_gs"] == 1e-9 and prm["R_g"] == 3.0 and prm["K"] == pytest.approx(1.43)
    assert prm["C_gd"] == {"law": "vdmos_cgd", "cgd_max": 1e-9, "cgd_min": 50e-12, "a": 0.5}
    assert "kept" in msg and "CGDMAX" in msg


def test_vdmos_charge_drop_restores_the_level1_device():
    b, msg = _import(_CARD.format(extra=""), charge="drop")
    kinds = [c["kind"] for c in b.components()]
    assert "power_mosfet" not in kinds and "mosfet_level1" in kinds
    assert "dropped" in msg and "RG" in msg


def test_vdmos_card_without_charge_imports_as_before():
    card = _CARD.replace(" CGS=1n CGDMAX=1n CGDMIN=50p A=0.5 RG=3", " RD=0.05 RS=0.01")
    keep, _ = _import(card.format(extra=""))
    drop, _ = _import(card.format(extra=""), charge="drop")
    assert keep.components() == drop.components()


def test_vdmos_tt_builds_a_lauritzen_body_diode():
    b, msg = _import(_CARD.format(extra=" TT=100n CJO=2n VJ=2.5"))
    kinds = {c["name"]: c["kind"] for c in b.components()}
    assert kinds["M1_body"] == "lauritzen_diode"
    prm = next(c for c in b.components() if c["kind"] == "power_mosfet")["params"]
    assert prm["C_ds"] == {"law": "junction", "C0": 2e-9, "V0": 2.5, "m": 0.5}


# ---------------------------------------------------------------------------
# datasheet fitters (task 4.2)
# ---------------------------------------------------------------------------
def test_fit_mosfet_channel_round_trip():
    from pulsim import datasheet as ds
    vgs = np.linspace(4.5, 10.0, 23)
    i = 1.43 * (vgs - 3.0) ** 2
    f = ds.fit_mosfet_channel(vgs, i, v_ds=20.0)
    assert f["K"] == pytest.approx(1.43, rel=1e-6) and f["V_T"] == pytest.approx(3.0, rel=1e-6)
    with pytest.raises(ValueError, match="residual"):
        ds.fit_mosfet_channel(vgs, 1.43 * (vgs - 3.0) ** 1.4, v_ds=20.0)


def test_fit_rdson_temperature_round_trip():
    from pulsim import datasheet as ds
    T = np.array([-40.0, 0.0, 25.0, 75.0, 125.0, 150.0])
    r = 1 + 5e-3 * (T - 25) + 2e-5 * (T - 25) ** 2
    f = ds.fit_rdson_temperature(T, r)
    assert abs(f["rds_tc1"] - 5e-3) < 1e-9 and abs(f["rds_tc2"] - 2e-5) < 1e-9


def test_fit_charge_table_two_decades():
    from pulsim import datasheet as ds
    v = np.concatenate(([0.0], np.geomspace(0.5, 400.0, 29)))
    C = 1e-9 / (1 + v / 2.0) ** 0.9 + 2e-12          # ~2.5 decades, 30 points
    f = ds.fit_charge_table(v, C, tol=0.01)
    model = np.array([f.table.capacitance(x) for x in v])
    assert np.max(np.abs(model - C) / C) == pytest.approx(f.residual)
    assert f.residual <= 0.01
    with pytest.raises(ValueError):
        ds.fit_charge_table(v, -C)


def test_fit_body_diode_recovery_and_power_mosfet_from_datasheet():
    from pulsim import datasheet as ds
    rr = ds.fit_body_diode_recovery(Q_rr=1e-6, I_rrm=20.0, I_F=20.0, di_dt=1e9, V_R=400.0)
    assert 0 < rr["T_M"] < rr["tau"] and rr["residual"] < 1e-9
    with pytest.raises(ValueError, match="no charge-control solution"):
        ds.fit_body_diode_recovery(Q_rr=1e-8, I_rrm=20.0, I_F=20.0, di_dt=1e9, V_R=400.0)
    vgs = np.linspace(4.5, 10.0, 12)
    Vds = np.array([0.0, 1.0, 10.0, 100.0, 400.0])
    out = ds.power_mosfet_from_datasheet(
        transfer={"v_gs": vgs, "i_d": 1.43 * (vgs - 3) ** 2, "v_ds": 20.0},
        R_ds_on=0.1, V_gs_spec=10.0,
        C_iss=np.column_stack([Vds, [3e-9, 2.6e-9, 2.2e-9, 2.05e-9, 2.02e-9]]),
        C_oss=np.column_stack([Vds, [2e-9, 1.4e-9, 5e-10, 1.5e-10, 8e-11]]),
        C_rss=np.column_stack([Vds, [1e-9, 6e-10, 2e-10, 5e-11, 2e-11]]), R_g=2.0,
        recovery=dict(Q_rr=1e-6, I_rrm=20.0, I_F=20.0, di_dt=1e9, V_R=400.0))
    kw = out["kwargs"]
    assert kw["R_d"] == pytest.approx(0.1 - 1 / (2 * 1.43 * 7.0), rel=1e-5)
    assert kw["C_gs"] == pytest.approx(2.0e-9)
    b = p.CircuitBuilder()
    b.add_power_mosfet("Q", "d", "s", "g", **kw)
    with pytest.raises(ValueError, match="disagree"):
        ds.power_mosfet_from_datasheet(
            transfer={"v_gs": vgs, "i_d": 1.43 * (vgs - 3) ** 2, "v_ds": 20.0},
            R_ds_on=0.01, V_gs_spec=10.0, C_iss=[[0, 2e-9], [400, 2e-9]],
            C_oss=[[0, 1e-9], [400, 1e-10]], C_rss=[[0, 1e-10], [400, 1e-11]])


# ---------------------------------------------------------------------------
# parts schema (tasks 4.3, 4.4)
# ---------------------------------------------------------------------------
def _part_doc(**pm):
    blk = {"model": dict(K=1.43, V_T=3.0, C_gs=1e-9, C_gd=100e-12, C_ds=0.0),
           "curves": {"transfer": dict(document="SYN-1", revision="0", date="2026-11-12",
                                       figure="Fig. 1", method="digitised")},
           "fitted_on": ["transfer"], "validated_on": ["E_on", "E_off"]}
    blk.update(pm)
    return {"class": "mosfet", "vendor": "Example", "part": "SYN-PM1",
            "provenance": {"source": "synthetic", "source_ref": "test", "retrieved": "2026-11-12",
                           "method": "synthetic"},
            "power_mosfet": blk}


@pytest.mark.parametrize("pm,frag", [
    (dict(fitted_on=["transfer"], validated_on=["transfer"]), "both fitted_on and validated_on"),
    (dict(fitted_on=["Crss"]), "no datasheet reference"),
    (dict(fitted_on=["transfer", "E_on"], validated_on=["E_off"]), "switching energies"),
    (dict(validated=True), "double_pulse"),
    (dict(curves={"transfer": dict(document="X", revision="0", date="d", figure="1",
                                   method="guessed")}), "method"),
])
def test_part_schema_refusals(pm, frag):
    from pulsim.lib.parts import part_from_mapping
    with pytest.raises(ValueError, match=frag):
        part_from_mapping(_part_doc(**pm), "SYN-PM1.yaml")


def test_part_add_to_and_acceptance_gate(tmp_path):
    from pulsim.lib.parts import part_from_mapping
    dp = dict(V_dd=VDC, I_load=IL, R_g_ext=10.0, V_drive=12.0, L_load=500e-6,
              E_on=1.0, E_off=1.0, tolerance=0.2)
    part = part_from_mapping(_part_doc(double_pulse=dp), "SYN-PM1.yaml")
    b = p.CircuitBuilder()
    part.add_to(b, "Q", "d", "gnd", "g", T_j=25.0)
    assert any(c["kind"] == "power_mosfet" for c in b.components())
    ref = p.double_pulse_test(part.power_mosfet_kwargs(), V_dd=VDC, I_load=IL, R_g_ext=10.0,
                              V_drive=12.0, L_load=500e-6)
    good = part_from_mapping(_part_doc(double_pulse=dict(dp, E_on=ref.E_on * 1.1,
                                                         E_off=ref.E_off * 0.9)), "x")
    bad = part_from_mapping(_part_doc(double_pulse=dict(dp, E_on=ref.E_on * 1.5,
                                                        E_off=ref.E_off)), "x")
    assert good.acceptance_check()["passed"]
    assert not bad.acceptance_check()["passed"]
    with pytest.raises(ValueError, match="tolerance"):
        part_from_mapping(_part_doc(double_pulse=dict(dp, tolerance=0.3)), "x")


def test_every_validated_power_mosfet_part_passes_acceptance():
    """The CI gate of the `validated` label (no shipped part carries one yet:
    which real parts ship is an owner decision, DECISIONS.md 1)."""
    from pulsim import lib
    for part in lib.list_parts("mosfet"):
        blk = part.params.get("power_mosfet")
        if blk and blk.get("validated"):
            res = part.acceptance_check()
            assert res["passed"], (part.number, res)


def test_power_mosfet_lookup_by_number(tmp_path):
    yaml = pytest.importorskip("yaml")
    from pulsim import lib
    doc = _part_doc()
    (tmp_path / "SYN-PM1.yaml").write_text(yaml.safe_dump(doc))
    plain = dict(doc, part="SYN-PLAIN")
    del plain["power_mosfet"]
    (tmp_path / "SYN-PLAIN.yaml").write_text(yaml.safe_dump(plain))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = lib.power_mosfet("SYN-PM1", allow_synthetic=True, paths=[tmp_path])
        assert q.power_mosfet_kwargs()["K"] == 1.43
        with pytest.raises(ValueError, match="power_mosfet"):
            lib.power_mosfet("SYN-PLAIN", allow_synthetic=True, paths=[tmp_path])


# ---------------------------------------------------------------------------
# Order across a gate-driven threshold crossing (probes/threshold_order.py)
# ---------------------------------------------------------------------------
def _threshold_case():
    """Gate driven by 6 − 6·cos(2π·2 MHz·t) through R_g = 10 Ω into C_gs = 1 nF
    and a VDMOS C_gd; drain 400 V behind 40 Ω. The channel turns on through its
    own charge dynamics (no source edge, no switch event). Returns the builder,
    its start state and an independent reference: scipy Radau at rtol 1e-12
    on the same two node equations (the Level-1 law and the vdmos_cgd law
    written out here)."""
    import math
    solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp
    K, VT, KAP, CGS, RG, RL, VDD, F = 1.43, 3.0, 15.0, 1e-9, 10.0, 40.0, 400.0, 2e6
    cmax, cmin, a = 1e-9, 50e-12, 0.5
    s_ = (cmax - cmin) / (1 + math.pi / 2)

    def lg(z):
        return 0.5 * (1 + math.tanh(0.5 * z))

    def fwd(vov, vds):
        al, be = lg(KAP * vov), lg(KAP * (vov - vds))
        return al * (be * K * (2 * vov * vds - vds * vds) + (1 - be) * K * vov * vov)

    def rhs(t, y):
        vg, vd = y
        u = vg - vd
        c = cmax - s_ + s_ * (math.tanh(a * u) if u > 0 else math.atan(a * u))
        vov = vg - VT
        i = (-fwd(vov - vd, -vd) + 2 * fwd(vov - 0.5 * vd, 0.0)) if vd < 0 else fwd(vov, vd)
        drv = 6 - 6 * math.cos(2 * math.pi * F * t)
        return np.linalg.solve([[CGS + c, -c], [-c, c]], [(drv - vg) / RG, (VDD - vd) / RL - i])

    ref = solve_ivp(rhs, (0, 250e-9), [0.0, VDD], method="Radau", rtol=1e-12,
                    atol=1e-12, dense_output=True).sol
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdd", "vdd", "gnd", VDD)
    b.add_resistor("RL", "vdd", "d", RL)
    b.add_sine_voltage_source("Vg", "drv", "gnd", 6.0, 6.0, F, -math.pi / 2)
    b.add_power_mosfet("M", "d", "gnd", "drv", K=K, V_T=VT, lambda_=0.0, kappa=KAP,
                       C_gs=CGS, C_gd=p.ChargeLaw.vdmos_cgd(cmax, cmin, a), C_ds=0.0, R_g=RG)
    names = b.state_var_names()
    x0 = np.array(b.initial_state(), dtype=float).ravel()
    x0[names.index("V(d)")] = x0[names.index("V(vdd)")] = VDD
    return b, x0, ref


def _threshold_error(b, x0, ref, **kw):
    r = p.simulate(b, t_end=250e-9, initial_state=x0, **kw)
    t = np.asarray(r.times)
    y = np.column_stack([np.asarray(r.v("M.g")), np.asarray(r.v("d"))])
    return float(np.max(np.abs(y - ref(t).T))), t.size - 1


def test_gate_driven_threshold_crossing_is_second_order():
    """The orchestrator's suspected first-order defect at a gate-driven
    threshold crossing does not reproduce (measured 2026-11-18): pwl 2.00 per
    halving from 250 ps, trbdf2 1.6–2.2 against its step count. Error over the
    engine's own samples against the Radau reference."""
    b, x0, ref = _threshold_case()
    e = [_threshold_error(b, x0, ref, engine="pwl", dt=dt)[0]
         for dt in (2.5e-10, 1.25e-10, 6.25e-11)]
    for e0, e1 in zip(e, e[1:]):
        assert 1.8 < np.log2(e0 / e1) < 2.2, e
    assert e[-1] < 1e-4
    (e6, n6), (e8, n8) = (_threshold_error(b, x0, ref, engine="trbdf2", rtol=rt,
                                           atol=rt * 1e-3, dt=250e-9) for rt in (1e-6, 1e-8))
    assert np.log(e6 / e8) / np.log(n8 / n6) > 1.5, (e6, n6, e8, n8)
    assert e8 < 1e-3


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
@pytest.mark.parametrize("vgs", [3.2, 4.0])
def test_current_through_a_partly_on_channel_crosses_zero(engine, vgs):
    """The Level-1 law jumped by 2·α(1−β)K·V_OV² across V_DS = 0 (5.2 mA at
    V_GS = 3.2 V); a ±10 mA sine through the channel had no solution at its
    zero crossing and both engines aborted (probes/vds_zero_jump.py)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vg", "g", "gnd", vgs)
    b.add_sine_current_source("I", "gnd", "d", 0.0, 10e-3, 1e3)
    b.add_capacitor("C", "d", "gnd", 1e-9)
    b.add_mosfet_level1("M", "d", "gnd", "g", K=1.43, V_T=3.0, lambda_=0.0, kappa=15.0,
                        with_body_diode=False)
    kw = dict(dt=1e-6) if engine == "pwl" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=2e-3, engine=engine, **kw)
    v = np.asarray(r.v("d"))
    assert v.min() < 0 < v.max() and np.max(np.abs(v)) < 0.05


# ---------------------------------------------------------------------------
# group 5: the table channel, the one f/q device, generic outputs
# ---------------------------------------------------------------------------
LAWS = dict(K=1.43, V_T=3.0, C_gs=1e-9, C_gd=None, C_ds=None,
            body_diode={"model": "lauritzen", "tau": 50e-9, "T_M": 10e-9})
FQ = dict(fq_device=True)   # group 5 is opt-in; the composite stays the default


def _pm_laws():
    return dict(LAWS, C_gd=p.ChargeLaw.vdmos_cgd(100e-12, 10e-12, 0.5),
                C_ds=p.ChargeLaw.junction(200e-12, 25.0, 0.5))


def _p4_sense(dev, engine):
    """The probe fixture with a 1 mOhm sense resistor in the drain."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", VDC)
    b.add_inductor("L", "vdc", "sw", 500e-6, i0=IL)
    b.add_shockley_diode("Df", "sw", "vdc")
    b.add_resistor("Rsense", "sw", "d", 1e-3)
    b.add_power_mosfet("M", "d", "gnd", "drv", R_g=10.0, **dev, **FQ)
    b.add_pulse_voltage_source("Vg", "drv", "gnd", 0.0, 12.0, 100e-9, 700e-9, 0.0, 5e-9, 5e-9)
    kw = dict(dt=2e-10) if engine == "pwl" else {}
    return b, p.simulate(b, t_end=1.5e-6, engine=engine, **kw)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
@pytest.mark.parametrize("dev", ["linear", "laws"])
def test_res_i_and_loss_row_against_a_sense_resistor(engine, dev):
    """Task 5.3: res.i(name) is the drain-terminal current (truth: the 1 mOhm
    sense resistor's own current, an independent element), and the loss row
    is the integral of v_DS * i_D on the result's own grid, each factor
    piecewise linear per step (the rule every loss row uses)."""
    d = DEV if dev == "linear" else _pm_laws()
    b, r = _p4_sense(d, engine)
    i_m = np.asarray(r.i("M"))
    i_s = np.asarray(r.i("Rsense"))
    # from sample 1: the start is inconsistent (i0 = 10 A against zero
    # node voltages), so the capacitor currents AT t = 0 are undefined
    # measured 2026-11-26 (laws): pwl <= 1e-9, trbdf2 1.3e-9 of scale. "linear"
    # (constant C_gd at the drain, no R_d) was 2.5 A off on pwl / 0.042 A on
    # trbdf2 while that C_gd was an MNA capacitor read through its ladder
    # measured 2026-11-26 (linear, trbdf2): 3.03e-9 of scale, at the sliver step
    # that ends the gate edge (h ~ 2e-17 s: the Newton charge tolerance over h)
    bound = 5e-9 if (dev, engine) == ("linear", "trbdf2") else 3e-9
    assert np.max(np.abs(i_m[1:] - i_s[1:])) <= bound * np.max(np.abs(i_s))
    t = np.asarray(r.times)
    # the loss row starts at the first solved sample (row 0 is the seed)
    t, v, i_s = t[1:], np.asarray(r.v("d"))[1:], i_s[1:]
    h = np.diff(t)
    E = float(np.sum(h / 6 * (2 * v[:-1] * i_s[:-1] + v[:-1] * i_s[1:]
                              + v[1:] * i_s[:-1] + 2 * v[1:] * i_s[1:])))
    row = [x for x in p.device_loss_summary(b, r) if x["name"] == "M"]
    assert len(row) == 1 and row[0]["kind"] == "power_mosfet"
    assert row[0]["E_total"] == pytest.approx(E, rel=1e-8)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_one_device_matches_the_hand_built_composite_with_laws(engine):
    """Task 5.2: the f/q device against the same circuit built by hand from
    the separate kinds the composite used (Level-1 channel, two nonlinear
    capacitors, a Lauritzen diode). pwl: same steps, same Newton matrix ->
    1e-9 of scale; trbdf2: the error norm sees the device's pin q rows
    instead of one row per capacitor, so the grid differs -> bounded by the
    engine's own tolerance."""
    laws = _pm_laws()

    def run(device):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vdc", "vdc", "gnd", VDC)
        b.add_inductor("L", "vdc", "sw", 500e-6, i0=IL)
        b.add_shockley_diode("Df", "sw", "vdc")
        b.add_resistor("Rsense", "sw", "d", 1e-3)
        b.add_pulse_voltage_source("Vg", "drv", "gnd", 0.0, 12.0, 100e-9, 700e-9, 0.0, 5e-9, 5e-9)
        if device:
            b.add_power_mosfet("M", "d", "gnd", "drv", R_g=10.0, **laws, **FQ)
        else:
            b.add_resistor("M.rg", "drv", "M.g", 10.0)
            b.add_mosfet_level1("M", "d", "gnd", "M.g", 1.43, 3.0, 0.0, 15.0, False)
            b.add_capacitor("M.cgs", "M.g", "gnd", 1e-9)
            b.add_nonlinear_capacitor("M.cgd", "M.g", "d", laws["C_gd"])
            b.add_nonlinear_capacitor("M.cds", "d", "gnd", laws["C_ds"])
            b.add_lauritzen_diode("M_body", "gnd", "d", tau=50e-9, T_M=10e-9)
        kw = dict(dt=2e-10) if engine == "pwl" else {}
        r = p.simulate(b, t_end=1.5e-6, engine=engine, **kw)
        t = np.asarray(r.times)
        return t, np.asarray(r.v("d")), np.asarray(r.v("M.g")), np.asarray(r.i("Rsense"))

    a, c = run(True), run(False)
    if engine == "pwl":
        assert np.array_equal(a[0], c[0])
        for x, y in zip(a[1:], c[1:]):
            # measured 2026-11-26: 1.1e-9 / 1.3e-9 of scale on the currents
            # (the composite's 1e-12 S placeholder on each nonlinear-capacitor
            # node, DECISIONS 20, is not in the device)
            assert np.max(np.abs(x - y)) <= 3e-9 * np.max(np.abs(y))
    else:
        Ea = np.trapezoid(a[1] * a[3], a[0])
        Ec = np.trapezoid(c[1] * c[3], c[0])
        assert Ea == pytest.approx(Ec, rel=2e-3)


def _square_law_table():
    vgs = [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0]
    vds = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 50.0, 100.0, 500.0]
    cur = [[1.43 * (2 * max(g - 3, 0) * min(v, max(g - 3, 0)) - min(v, max(g - 3, 0)) ** 2)
          for v in vds] for g in vgs]
    return p.ChannelTable(vgs, vds, cur)


def test_channel_table_refusals_are_typed_and_name_the_point():
    vgs, vds = [3.0, 4.0, 5.0], [0.0, 1.0, 2.0, 4.0]
    good = [[0, 0, 0, 0], [0, 1.0, 1.5, 1.6], [0, 2.0, 3.0, 3.2]]
    bad = [row[:] for row in good]
    bad[2][3] = 2.9
    with pytest.raises(p.NetlistError, match=r"not monotone in V_DS at V_GS = 5.*\[2\]\[3\]"):
        p.ChannelTable(vgs, vds, bad)
    bad = [row[:] for row in good]
    bad[0][0] = 1e-3
    with pytest.raises(p.NetlistError, match="must be 0"):
        p.ChannelTable(vgs, vds, bad)
    t = p.ChannelTable(vgs, vds, good)
    assert t.current(5.0, 2.0)[0] == 3.0
    i_f, gg_f, gd_f = t.current(4.0, 1e-13)
    i_r, gg_r, gd_r = t.current(4.0, -1e-13)
    assert abs(i_f + i_r) <= 1e-12 * abs(i_f) and gd_f == pytest.approx(gd_r, rel=1e-9)
    with pytest.raises(ValueError, match="channel table carries its own"):
        b = p.CircuitBuilder()
        b.add_power_mosfet("M", "d", "gnd", "g", K=1.0, V_T=3.0, C_gs=1e-9, C_gd=0.0,
                           C_ds=0.0, channel=t, rds_tc1=4e-3, **FQ)
    with pytest.raises(ValueError, match="needs fq_device = true"):
        b = p.CircuitBuilder()
        b.add_power_mosfet("M", "d", "gnd", "g", K=1.0, V_T=3.0, C_gs=1e-9, C_gd=0.0,
                           C_ds=0.0, channel=t)


def test_double_pulse_with_a_table_channel_follows_the_level1_law():
    """A table sampled from the Level-1 square law (its kappa blend aside)
    gives the Level-1 device's switching energies to a few percent; the
    truth is the Level-1 device, an independent law."""
    lvl = p.double_pulse_test(dict(DEV, **FQ), V_dd=VDC, I_load=IL, R_g_ext=10.0, V_drive=12.0,
                              L_load=500e-6, engine="trbdf2")
    tab = p.double_pulse_test(dict(DEV, channel=_square_law_table(), **FQ), V_dd=VDC, I_load=IL,
                              R_g_ext=10.0, V_drive=12.0, L_load=500e-6, engine="trbdf2")
    assert tab.E_on == pytest.approx(lvl.E_on, rel=0.05)
    assert tab.E_off == pytest.approx(lvl.E_off, rel=0.05)
    assert tab.t_plateau == pytest.approx(lvl.t_plateau, rel=0.05)


def test_ac_sweep_of_the_device_against_its_linearised_closed_form():
    """Task 5.3: AC from f and q at a triode operating point. Closed form:
    the deep-triode Level-1 conductance K(2 V_OV - 2 x), the VDMOS C_gd and
    junction C_ds laws (design.md formulas), the Shockley body diode's
    reverse conductance; x from the DC balance by bisection."""
    K, VT0, Vgs, Vb, R = 1e-2, 2.0, 15.0, 1.0, 100.0
    cmax, cmin, a, C0, V0, m = 1e-9, 50e-12, 0.5, 1e-9, 25.0, 0.5
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", Vb)
    b.add_resistor("R", "in", "o", R)
    b.add_voltage_source("Vg", "g", "gnd", Vgs)
    b.add_power_mosfet("Md", "o", "gnd", "g", K=K, V_T=VT0, C_gs=1e-9,
                       C_gd=p.ChargeLaw.vdmos_cgd(cmax, cmin, a),
                       C_ds=p.ChargeLaw.junction(C0, V0, m), body_diode={"model": "shockley"},
                       **FQ)
    vov, IS, Vt, G = Vgs - VT0, 1e-12, 0.025852, 1e-12

    def bal(x):   # current into o: source side minus channel, plus the reverse diode
        return (Vb - x) / R + IS * math.expm1(-x / Vt) - G * x - K * (2 * vov * x - x * x)
    lo, hi = 0.0, Vb
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if bal(mid) > 0 else (lo, mid)
    x = 0.5 * (lo + hi)
    g = K * (2 * vov - 2 * x) + IS / Vt * math.exp(-x / Vt) + G
    s = (cmax - cmin) / (1 + math.pi / 2)
    u = Vgs - x                                   # v_gd > 0
    C = (cmax - s + s * math.tanh(a * u)) + C0 * (1 + x / V0) ** (-m)
    F = np.array([1e5, 1e6, 1e7])
    w = 2 * np.pi * F
    Z = 1 / (g + 1j * w * C)
    exact = Z / (R + Z)
    H = p.run_mna_sweep(b, output_idx=b.node_id_of("o"), freqs=F,
                        v_in_branch_id=b.branch_id_of("V")).H
    np.testing.assert_allclose(H, exact, rtol=1e-9)


def _fq_full(b, **over):
    kw = dict(K=1.43, V_T=3.0, C_gs=1e-9, C_gd=p.ChargeLaw.vdmos_cgd(1e-9, 50e-12, 0.5),
              C_ds=p.ChargeLaw.junction(2e-9, 25.0, 0.5), R_g=3.0, R_d=0.01, L_s=2e-9,
              body_diode={"model": "lauritzen", "tau": 50e-9, "T_M": 10e-9}, T_j=100.0,
              rds_tc1=4e-3, **FQ)
    kw.update(over)
    b.add_voltage_source("Vdc", "vdc", "gnd", 48.0)
    b.add_resistor("Rl", "vdc", "d", 4.8)
    b.add_pulse_voltage_source("Vg", "drv", "gnd", 0.0, 12.0, 50e-9, 300e-9, 0.0, 5e-9, 5e-9)
    b.add_power_mosfet("Q1", "d", "gnd", "drv", **kw)
    return b


@pytest.mark.parametrize("over", [
    {},                                                   # laws, Lauritzen, R_d, L_s
    dict(C_gd=100e-12, C_ds=300e-12, R_d=0.0, L_s=0.0,    # constants at d', Shockley
         body_diode={"model": "shockley"}, rds_tc1=0.0, T_j=25.0),
    dict(channel=None, rds_tc1=0.0, T_j=25.0),            # replaced by a table below
])
def test_fq_device_round_trips_through_yaml(over):
    """The f/q device survives to_yaml -> from_yaml: same hashes, the text is
    a fixed point, same components(), and the loaded circuit runs bit-identically.
    Before: to_yaml refused it ("an external device is not written to YAML") and
    value_hash held the per-process model handle (two builds hashed apart)."""
    if "channel" in over:
        over = dict(over, channel=_square_law_table())
    c = _fq_full(p.CircuitBuilder(), **over).build()
    assert c.value_hash() == _fq_full(p.CircuitBuilder(), **over).build().value_hash()
    y = c.to_yaml()
    c2 = p.Circuit.from_yaml(y)
    assert (c2.value_hash(), c2.structure_hash()) == (c.value_hash(), c.structure_hash())
    assert c2.to_yaml() == y
    assert c2.components() == c.components()
    a = p.simulate(c, t_end=4e-7, dt=1e-9, engine="pwl")
    b = p.simulate(c2, t_end=4e-7, dt=1e-9, engine="pwl")
    np.testing.assert_array_equal(np.asarray(a.states), np.asarray(b.states))
    # a changed device value changes the hash
    other = _fq_full(p.CircuitBuilder(), **dict(over, K=1.5)).build()
    assert other.value_hash() != c.value_hash()


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_fq_device_matches_the_default_composite_on_a_resistive_switching_cell(engine):
    """Independent reference: the same call without fq_device (the composite of
    separately tested kinds) on a 48 V / 4.8 ohm resistive-load switching cell,
    constant C_gd / C_ds, Shockley body diode, R_g. Plus a closed form of the
    on-state: V_DS solves (V - v)/R = K (2 V_OV v - v^2) (deep triode)."""
    def run(fq):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vdc", "vdc", "gnd", 48.0)
        b.add_resistor("Rl", "vdc", "d", 4.8)
        b.add_pulse_voltage_source("Vg", "drv", "gnd", 0.0, 12.0, 50e-9, 600e-9, 0.0,
                                   5e-9, 5e-9)
        b.add_power_mosfet("Q1", "d", "gnd", "drv", K=1.43, V_T=3.0, lambda_=0.0, kappa=15.0,
                           C_gs=1e-9, C_gd=100e-12, C_ds=300e-12, R_g=3.0,
                           body_diode={"model": "shockley"}, fq_device=fq)
        kw = dict(dt=1e-10) if engine == "pwl" else {}
        r = p.simulate(b, t_end=1e-6, engine=engine, **kw)
        return np.asarray(r.times), np.asarray(r.v("d")), np.asarray(r.i("Rl"))
    (ta, va, ia), (tc, vc, ic) = run(True), run(False)
    # closed form at the end of the on-interval (gate at 12 V for >> R_g C_iss)
    K, vov, V, R = 1.43, 9.0, 48.0, 4.8
    v_on = ((2 * K * vov + 1 / R) - math.sqrt((2 * K * vov + 1 / R) ** 2 - 4 * K * V / R)) / (2 * K)
    k_on = np.searchsorted(ta, 600e-9) - 1
    assert va[k_on] == pytest.approx(v_on, rel=REF_ON_REL)
    if engine == "pwl":
        assert np.array_equal(ta, tc)
        err_v = np.max(np.abs(va - vc)) / V
        err_i = np.max(np.abs(ia - ic)) / (V / R)
        assert max(err_v, err_i) <= REF_PWL_REL
    else:
        Ea, Ec = np.trapezoid(va * ia, ta), np.trapezoid(vc * ic, tc)
        assert Ea == pytest.approx(Ec, rel=REF_TRBDF2_E_REL)


# measured 2026-11-26: v_on 1.40e-13 (pwl) / 1.42e-13 (trbdf2) of the closed
# form; pwl device vs composite 1.9e-14 of scale on v_d and i (same grid);
# trbdf2 switching-cell energy 3.7e-6 (grids differ: 1627 vs 1512 steps)
REF_ON_REL = 1e-11
REF_PWL_REL = 1e-12
REF_TRBDF2_E_REL = 1e-5
