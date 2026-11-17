"""add-power-mosfet-datasheet-device task 7 — Pulsim against ngspice 45.2.

The ngspice outputs of ``benchmarks/spice/ngspice/pm_*.cir`` are pinned in
``power_mosfet_reference.json``, so the Pulsim side runs everywhere. When
ngspice is on PATH the decks are re-run and checked against the pinned
numbers too (the other direction); without it those tests skip.

Every tolerance below is the measured difference (2026-11-12) with a
margin; none of the model parameters was tuned on these numbers.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

DECKS = Path(__file__).resolve().parents[2] / "benchmarks" / "spice" / "ngspice"
REF = json.loads((DECKS / "power_mosfet_reference.json").read_text())
K, V_T, KAPPA = 1.43, 3.0, 15.0


# ---------------------------------------------------------------------------
# Pulsim against the pinned ngspice numbers
# ---------------------------------------------------------------------------
def test_charge_laws_match_ngspice():
    """Task 7.2: vdmos_cgd and the C_ds junction law (reverse bias; ngspice
    linearises the junction beyond FC·VJ forward). Measured 1.1e-12."""
    ref = REF["charge_laws"]
    cgd = p.ChargeLaw.vdmos_cgd(1e-9, 50e-12, 0.5)
    cds = p.ChargeLaw.junction(2e-9, 2.5, 0.5)
    for v, c_gd, c_tot in zip(ref["V_DS"], ref["C_gd"], ref["C_gd_plus_C_ds"]):
        assert cgd.capacitance(-v) == pytest.approx(c_gd, rel=1e-6), v
        if v >= 0:
            assert cds.capacitance(v) == pytest.approx(c_tot - c_gd, rel=1e-6), v


def _static(add, V_a, V_g=None):
    """Current out of the swept source of an algebraic circuit."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Va", "a", "gnd", V_a)
    if V_g is not None:
        b.add_voltage_source("Vg", "g", "gnd", V_g)
    add(b)
    r = p.simulate(b, t_end=2e-9, engine="pwl", dt=1e-9)
    return -float(np.asarray(r.i("Va"))[-1])


def _level1(V_DS, V_GS):
    return _static(lambda b: b.add_mosfet_level1("M", "a", "gnd", "g", K, V_T, 0.01, KAPPA,
                                                 False), V_DS, V_GS)


def test_level1_output_curves_match_ngspice_per_region():
    """Task 7.1: SPICE's hard triode/saturation boundary against the kappa
    sigmoid blend. Measured: 3.8e-8 at |V_OV − V_DS| >= 1 V, 1.6e-5 within
    0.5 V of the boundary (the blend's width is a few 1/kappa = 67 mV)."""
    for vg, curve in REF["level1"]["output"].items():
        V_ov = float(vg) - V_T
        for v, i_ng in zip(curve["V_DS"], curve["I_D"]):
            tol = 1e-6 if abs(V_ov - v) >= 1.0 else 1e-4
            assert _level1(v, float(vg)) == pytest.approx(i_ng, rel=tol), (vg, v)


def test_level1_transfer_curve_matches_ngspice():
    """Measured 3.1e-7 at V_OV = 1 V (the cut-off sigmoid), 3e-12 above 2 V."""
    tr = REF["level1"]["transfer_V_DS_20"]
    for vg, i_ng in zip(tr["V_GS"], tr["I_D"]):
        assert _level1(20.0, vg) == pytest.approx(i_ng, rel=1e-5), vg


def test_shockley_curve_matches_ngspice():
    """Forward: measured 1.1e-5 at 0.8 V (ngspice's k/q constants give a V_T
    3.5e-7 apart). Reverse: the pA leakage plus GMIN, measured 2.5e-15 A."""
    s = REF["shockley"]
    for v, i_ng in zip(s["V"], s["I"]):
        i = _static(lambda b: b.add_shockley_diode("D", "a", "gnd"), v)
        if v > 0:
            assert i == pytest.approx(i_ng, rel=1e-4), v
        else:
            assert abs(i - i_ng) < 1e-14, v


_CARD = (".model MV VDMOS(VTO=3 KP=2.86 LAMBDA=0 CGDMAX=1n CGDMIN=50p A=0.5 CGS=1n CJO=0 "
         "RG=0 RD=0 RS=0 IS=1e-14 N=1 TT=0)\n")


def test_vdmos_card_imports_to_the_mapping_the_double_pulse_uses():
    """The deck's card through spice_to_builder gives exactly _VDMOS_DEV."""
    b = p.spice_to_builder(_CARD + "M1 d g 0 MV\nRx d 0 1meg\nVx g 0 0\n.end\n")
    comps = {c["name"]: c for c in b.components()}
    prm = comps["M1"]["params"]
    assert comps["M1"]["kind"] == "power_mosfet"
    assert (prm["K"], prm["V_T"], prm["lambda_"], prm["C_gs"]) == (1.43, 3.0, 0.0, 1e-9)
    assert prm["C_gd"] == {"law": "vdmos_cgd", "cgd_max": 1e-9, "cgd_min": 5e-11, "a": 0.5}
    assert prm["C_ds"]["C"] == 0.0 and prm["R_g"] == 0.0
    # components() reports no Shockley parameters; the note carries I_S
    assert comps["M1_body"]["kind"] == "shockley_diode"
    with pytest.warns(UserWarning, match=r"Is=1e-14 A"):
        p.spice_to_builder(_CARD + "M1 d g 0 MV\nRx d 0 1meg\nVx g 0 0\n.end\n")


_VDMOS_DEV = dict(K=1.43, V_T=3.0, lambda_=0.0, C_gs=1e-9,
                  C_gd=p.ChargeLaw.vdmos_cgd(1e-9, 50e-12, 0.5), C_ds=0.0,
                  body_diode={"model": "shockley", "I_S": 1e-14})


def test_vdmos_double_pulse_matches_ngspice():
    """Task 7.3, same card and circuit on both sides (trbdf2). Measured:
    E_on +0.15 %, E_off +0.07 %, plateau +0.35 %, V_plateau 5e-8,
    dv/dt_on −0.08 %, dv/dt_off +0.01 %, I_rr −0.01 %; di/dt is a
    two-crossing slope on the adaptive grid (−0.6 % on, +1.2 % off) and
    gets the design's 5 %. pwl at dt = 0.2 ns: E_on +0.03 %, E_off −0.03 %."""
    ng = REF["vdmos_double_pulse"]
    r = p.double_pulse_test(_VDMOS_DEV, V_dd=400.0, I_load=10.0, R_g_ext=10.0, V_drive=12.0,
                            L_load=500e-6, engine="trbdf2")
    tol = dict(E_on=5e-3, E_off=5e-3, t_plateau=1e-2, V_plateau=1e-5, dv_dt_on=5e-3,
               dv_dt_off=5e-3, di_dt_on=5e-2, di_dt_off=5e-2, I_rr=1e-2)
    for k, rel in tol.items():
        assert getattr(r, k) == pytest.approx(ng[k], rel=rel), k


def _first_reverse_lobe(t, i):
    """(I_rrm, Q_rr): peak and charge of the first negative lobe of i."""
    k0 = int(np.argmax(i < 0))
    after = np.nonzero(i[k0:] >= 0)[0]
    k1 = k0 + int(after[0]) if after.size else i.size
    return float(-i.min()), float(-np.trapezoid(np.minimum(i[k0 - 1:k1 + 1], 0.0),
                                               t[k0 - 1:k1 + 1]))


def _recovery(engine):
    from pulsim.spice_import import _VDMOS_TM_OVER_TT
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vs", "s", "gnd", 20.0, -100.0, 5e-6, 100e-6, 200e-6, 1e-9, 1e-9)
    b.add_resistor("R1", "s", "x", 2.0)
    b.add_inductor("L1", "x", "a", 1e-6)
    b.add_lauritzen_diode("D1", "a", "gnd", I_S=1e-12, tau=50e-9,
                          T_M=_VDMOS_TM_OVER_TT * 50e-9)
    b.add_resistor("Rs", "a", "b", 20.0)
    b.add_capacitor("Cs", "b", "gnd", 2e-9)
    kw = dict(dt=0.05e-9) if engine == "pwl" else {}
    r = p.simulate(b, t_end=6e-6, engine=engine, **kw)
    t, i = np.asarray(r.times), np.asarray(r.i("L1"))
    m = t > 4.9e-6
    return _first_reverse_lobe(t[m], i[m])


def test_lauritzen_at_the_importer_ratio_recovers_like_spice_tt():
    """Task 3.4: SPICE TT is the T_M -> 0 limit of the Lauritzen diode.
    Swept T_M/TT in {1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3}: I_rrm error
    +0.01, -0.01, -0.09, -0.45, -2.2, -7.6 %, Q_rr within 0.2 % up to 0.1.
    At the importer's 0.01 (trbdf2): I_rrm -0.09 %, Q_rr +0.11 %."""
    ref = REF["diode_recovery"]
    I_rrm, Q_rr = _recovery("trbdf2")
    assert I_rrm == pytest.approx(ref["I_rrm"], rel=5e-3)
    assert Q_rr == pytest.approx(ref["Q_rr"], rel=5e-3)


# ---------------------------------------------------------------------------
# ngspice against the pinned numbers (needs ngspice on PATH)
# ---------------------------------------------------------------------------
NGSPICE = shutil.which("ngspice")
needs_ngspice = pytest.mark.skipif(NGSPICE is None, reason="ngspice not on PATH")


def _run(deck: str, cwd: Path) -> str:
    shutil.copy(DECKS / deck, cwd / deck)
    out = subprocess.run([NGSPICE, "-b", deck], cwd=cwd, capture_output=True, text=True,
                         timeout=600, check=True)
    return out.stdout


@needs_ngspice
def test_ngspice_reproduces_the_pinned_static_references(tmp_path):
    log = _run("pm_charge_laws.cir", tmp_path)
    cgd = [-float(x) for x in re.findall(r"^cgd = (\S+)", log, re.M)]
    assert np.allclose(cgd, REF["charge_laws"]["C_gd"], rtol=1e-9, atol=0)
    _run("pm_level1_static.cir", tmp_path)
    for vg, curve in REF["level1"]["output"].items():
        a = np.loadtxt(tmp_path / f"pm_level1_out_vg{vg}.out")
        assert np.allclose(a[:, 1], curve["I_D"], rtol=1e-9, atol=0), vg
    _run("pm_shockley_static.cir", tmp_path)
    a = np.loadtxt(tmp_path / "pm_shockley.out")
    assert np.allclose(a[:, 1], REF["shockley"]["I"], rtol=1e-9, atol=1e-20)


@needs_ngspice
def test_ngspice_reproduces_the_pinned_double_pulse(tmp_path):
    _run("pm_vdmos_double_pulse.cir", tmp_path)
    a = np.loadtxt(tmp_path / "pm_vdmos_double_pulse.out")
    t, v_d, v_sw, v_g = a[:, 0], a[:, 1], a[:, 3], a[:, 5]
    m = p.double_pulse_metrics(t, v_d, (v_sw - v_d) / 1e-3, v_g, V_dd=400.0, I_load=10.0,
                               t_on=100e-9, t_off=805e-9)
    for k, v in REF["vdmos_double_pulse"].items():
        assert m[k] == pytest.approx(v, rel=1e-6), k


@needs_ngspice
def test_ngspice_reproduces_the_pinned_recovery(tmp_path):
    _run("pm_diode_recovery.cir", tmp_path)
    a = np.loadtxt(tmp_path / "pm_diode_recovery.out")
    I_rrm, Q_rr = _first_reverse_lobe(a[:, 0], a[:, 1])
    assert I_rrm == pytest.approx(REF["diode_recovery"]["I_rrm"], rel=1e-6)
    assert Q_rr == pytest.approx(REF["diode_recovery"]["Q_rr"], rel=1e-6)
