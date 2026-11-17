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
