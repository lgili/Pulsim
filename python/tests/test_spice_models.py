"""SPICE import: ``.MODEL`` applied or refused, ``.SUBCKT`` flattened — audit C.5.

Measured before this: a netlist with ``.MODEL DX D(IS=1e-9 N=1.5
RS=0.02 BV=600)`` and ``D1 a b DX`` imported as a PWL diode with
V_th = 0.7 V, whatever the card said — and that V_th is only the
turn-on threshold; once on, the branch is 1 mΩ. Against the Shockley
law the card describes:

    I = 2.4 A:  PWL V_F = 2.4 mV, P = 5.7 mW  |  Shockley+RS 0.885 V, 2.12 W
    I = 11 A:   PWL V_F = 11 mV,  P = 0.12 W  |  Shockley+RS 1.116 V, 12.3 W

conduction loss under-reported by two orders of magnitude, with no
warning. A LEVEL-1 NMOS card became a 10 mΩ switch with a body diode,
VTO and KP discarded. The textual ``.SUBCKT`` flattener first written
for this prefixed every inner designator with ``X…``, so the
tokeniser then read every inner resistor as a subcircuit instance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p
from pulsim.spice_import import parse_spice_netlist


def _kinds(b):
    return {c["name"]: c["kind"] for c in b.components()}


def _last(res, tr):
    return float(np.asarray(tr)[-1])


# --------------------------------------------------------------------------
# diodes
# --------------------------------------------------------------------------

def test_a_diode_model_becomes_the_shockley_junction_it_describes():
    net = """* diode with a real model
V1 a 0 DC 1.2
D1 a b DX
R1 b 0 0.5
.MODEL DX D(IS=1e-9 N=1.5 RS=0.02 BV=600)
.END
"""
    b = p.spice_to_builder(net)
    k = _kinds(b)
    assert k["D1.rs"] == "resistor"           # RS as an explicit series element
    assert "shockley" in k["D1"].lower()
    res = p.simulate(b, t_end=1e-4, dt=1e-6)
    i = _last(res, res.i("R1"))
    v_d = _last(res, res.v("a")) - _last(res, res.v("b"))
    v_expect = 1.5 * 0.025852 * math.log(i / 1e-9 + 1.0) + i * 0.02
    assert v_d == pytest.approx(v_expect, rel=2e-2), (v_d, v_expect, i)
    assert v_d > 0.8                            # not the 2 mV of the old importer


def test_ltspice_pwl_diode_card_maps_onto_the_pwl_diode():
    net = """V1 a 0 DC 1.2
D1 a b DPWL
R1 b 0 0.5
.MODEL DPWL D(Ron=0.01 Roff=1Meg Vfwd=0.6)
.END
"""
    b = p.spice_to_builder(net)
    assert "shockley" not in _kinds(b)["D1"].lower()
    res = p.simulate(b, t_end=1e-4, dt=1e-6)
    i = _last(res, res.i("R1"))
    v_d = _last(res, res.v("a")) - _last(res, res.v("b"))
    assert v_d == pytest.approx(0.6 + 0.01 * i, rel=5e-2), (v_d, i)


def test_a_referenced_model_that_is_missing_is_refused_not_defaulted():
    net = "V1 a 0 DC 1\nD1 a b DMISSING\nR1 b 0 1\n.END\n"
    with pytest.raises(ValueError, match="DMISSING"):
        p.spice_to_builder(net)


def test_a_diode_without_a_model_name_is_refused_as_spice_would():
    net = "V1 a 0 DC 1\nD1 a b\nR1 b 0 1\n.END\n"
    with pytest.raises(ValueError, match="no model name"):
        p.spice_to_builder(net)


def test_junction_dynamics_in_the_card_are_named_not_dropped_silently():
    net = """V1 a 0 DC 1
D1 a b DRR
R1 b 0 1
.MODEL DRR D(IS=1e-12 N=1.1 CJO=100p TT=50n)
.END
"""
    with pytest.warns(UserWarning, match="reverse recovery"):
        p.spice_to_builder(net)
    with pytest.raises(ValueError, match="TT"):
        p.spice_to_builder(net, strict=True)


@pytest.mark.parametrize("card,frag", [
    (".MODEL DB D(IS=1e-12 N=1 IKF=2)", "IKF"),
    (".MODEL DB D(Ron=0.01 Vfwd=0.6 Vrev=30)", "Vrev"),
    (".MODEL DB AKO:DX D(IS=1e-12)", "AKO"),
    (".MODEL DB NMOS(VTO=1 KP=1e-4)", "not D"),
])
def test_diode_cards_the_mapping_cannot_honour_are_refused(card, frag):
    net = f"V1 a 0 DC 1\nD1 a b DB\nR1 b 0 1\n{card}\n.END\n"
    with pytest.raises(ValueError, match=frag):
        p.spice_to_builder(net)


# --------------------------------------------------------------------------
# MOSFETs
# --------------------------------------------------------------------------

def _level1_current(K, VT, lam, Vg, Vd, R_s, R_d=0.0):
    """Saturation-region drain current with source/drain lead
    resistors, by bisection (the residual is monotone in i)."""
    def resid(i):
        vgs = Vg - i * R_s
        vds = Vd - i * (R_s + R_d)
        return i - K * max(vgs - VT, 0.0) ** 2 * (1 + lam * vds)
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def test_a_level1_nmos_card_maps_onto_shichman_hodges():
    net = """M1 d g s s NMOD W=100u L=10u
.MODEL NMOD NMOS(LEVEL=1 VTO=2.5 KP=50e-6 LAMBDA=0.02)
V1 d 0 DC 12
V2 g 0 DC 5
R1 s 0 1
.END
"""
    b = p.spice_to_builder(net)
    assert "level1" in _kinds(b)["M1"].lower()
    res = p.simulate(b, t_end=1e-4, dt=1e-7)
    i = _last(res, res.i("R1"))
    K = 50e-6 * 100e-6 / (2 * 10e-6)        # KP·W/(2L) = 250 uA/V^2
    assert i == pytest.approx(_level1_current(K, 2.5, 0.02, 5.0, 12.0, 1.0), rel=5e-2)


def test_ld_m_and_rd_rs_are_honoured():
    net = """M1 d g s s NMOD W=100u L=12u M=2
.MODEL NMOD NMOS(LEVEL=1 VTO=2.5 KP=50e-6 LD=1u RD=0.5 RS=0.25)
V1 d 0 DC 12
V2 g 0 DC 5
R1 s 0 1
.END
"""
    b = p.spice_to_builder(net)
    k = _kinds(b)
    assert k["M1.rd"] == "resistor" and k["M1.rs"] == "resistor"
    res = p.simulate(b, t_end=1e-4, dt=1e-7)
    i = _last(res, res.i("R1"))
    # L_eff = 12u − 2·1u = 10u, K doubled by M=2; RS/M = 0.125 sits inside the
    # source lead so the intrinsic V_GS sees it; RD/M = 0.25 is in the drain.
    K = 2 * 50e-6 * 100e-6 / (2 * 10e-6)
    ii = _level1_current(K, 2.5, 0.0, 5.0, 12.0, R_s=1.0 + 0.125, R_d=0.25)
    assert i == pytest.approx(ii, rel=5e-2), (i, ii)


@pytest.mark.parametrize("line,card,frag", [
    ("M1 d g s s NB", ".MODEL NB NMOS(LEVEL=3 VTO=1 KP=1e-4)", "LEVEL 3"),
    ("M1 d g s s NB", ".MODEL NB PMOS(LEVEL=1 VTO=-1 KP=1e-4)", "PMOS"),
    ("M1 d g s s NB", ".MODEL NB NMOS(LEVEL=1 KP=1e-4)", "VTO"),
    ("M1 d g s b NB", ".MODEL NB NMOS(LEVEL=1 VTO=1 KP=1e-4 GAMMA=0.5)", "bulk"),
    ("M1 d g s s NB", ".MODEL NB VDMOS(VTO=-3 KP=10 pchan)", "pchan"),
    ("M1 d g s s NB", ".MODEL NB VDMOS(VTO=3 KP=10 mtriode=0.5)", "mtriode"),
    ("M1 d g s", ".MODEL NB NMOS(LEVEL=1 VTO=1 KP=1e-4)", "D G S"),
    ("M1 d g s s", ".MODEL NB NMOS(LEVEL=1 VTO=1 KP=1e-4)", "model 's'"),
])
def test_mosfet_cards_the_mapping_cannot_honour_are_refused(line, card, frag):
    net = f"{line}\n{card}\nV1 d 0 DC 12\nV2 g 0 DC 5\nR1 s 0 1\nRb b 0 1k\n.END\n"
    with pytest.raises(ValueError, match=frag):
        p.spice_to_builder(net)


def test_level_2_3_parameters_at_level_1_are_ignored_as_spice_does_and_said():
    net = """M1 d g s s NB
.MODEL NB NMOS(LEVEL=1 VTO=1 KP=1e-4 VMAX=1e5)
V1 d 0 DC 12
V2 g 0 DC 5
R1 s 0 1
.END
"""
    with pytest.warns(UserWarning, match="VMAX"):
        p.spice_to_builder(net)
    with pytest.raises(ValueError, match="VMAX"):
        p.spice_to_builder(net, strict=True)


def test_vdmos_card_gives_kp_over_two_and_its_own_body_diode():
    net = """M1 d g s VD
.MODEL VD VDMOS(VTO=3 KP=20 LAMBDA=0 RDS=1e6 RS=0.01 IS=1e-12 N=1.2 RB=0.005)
V1 d 0 DC 12
V2 g 0 DC 5
R1 s 0 1
.END
"""
    with pytest.warns(UserWarning, match="VDMOS"):
        b = p.spice_to_builder(net)
    k = _kinds(b)
    assert k["M1.rs"] == "resistor" and k["M1.rds"] == "resistor"
    assert "shockley" in k["M1_body"].lower() and k["M1_body.rb"] == "resistor"
    res = p.simulate(b, t_end=1e-4, dt=1e-7)
    i = _last(res, res.i("R1"))
    # K = KP/2 = 10 A/V², V_GS = 5 − i·(1 + 0.01) → i = 10 (V_GS − 3)²
    ii = _level1_current(10.0, 3.0, 0.0, 5.0, 12.0, R_s=1.01)
    assert i == pytest.approx(ii, rel=5e-2), (i, ii)


# --------------------------------------------------------------------------
# subcircuits
# --------------------------------------------------------------------------

def test_subcircuits_are_flattened_with_scoped_names():
    net = """* two instances of one subcircuit, nested once
.SUBCKT HALF in out
R1 in mid 10
RL mid out 20
.ENDS
.SUBCKT PAIR a b
X1 a m HALF
X2 m b HALF
.ENDS
V1 top 0 DC 30
XA top out PAIR
RO out 0 60
.END
"""
    b = p.spice_to_builder(net)
    names = set(_kinds(b))
    assert {"XA.X1.R1", "XA.X1.RL", "XA.X2.R1", "XA.X2.RL", "RO", "V1"} <= names
    res = p.simulate(b, t_end=1e-4, dt=1e-6)
    # Series chain 10+20+10+20 = 60 Ω, then 60 Ω to ground: v(out) = 15 V.
    assert _last(res, res.v("out")) == pytest.approx(15.0, rel=1e-6)
    # The first HALF's internal node is scoped, not the outer 'mid'.
    assert _last(res, res.v("xa.x1.mid")) == pytest.approx(30.0 - 10 * 0.25, rel=1e-6)


def test_subcircuit_parameters_are_evaluated_in_the_callers_scope():
    net = """.PARAM rload=2k
.SUBCKT DIV in out PARAMS: r=1k
R1 in out {r}
.ENDS
V1 a 0 DC 1
X1 a b DIV r={rload}
X2 b c DIV
R2 c 0 1k
.END
"""
    b = p.spice_to_builder(net)
    res = p.simulate(b, t_end=1e-4, dt=1e-6)
    # 2k (passed) + 1k (default) + 1k: v(c) = 1/4.
    assert _last(res, res.v("c")) == pytest.approx(0.25, rel=1e-6)


def test_local_models_shadow_global_ones_and_global_nodes_are_not_scoped():
    net = """.GLOBAL vdd
.MODEL DX D(IS=1e-14 N=1)
.SUBCKT S a k
D1 a k DX
.MODEL DX D(IS=1e-9 N=1.5)
R1 a vdd 1k
.ENDS
V1 vdd 0 DC 5
X1 in out S
D2 in out DX
.END
"""
    els, _params, models = parse_spice_netlist(net, return_models=True)
    by = {e.designator: e for e in els}
    assert by["X1.D1"].model.params["IS"] == pytest.approx(1e-9)
    assert by["D2"].model.params["IS"] == pytest.approx(1e-14)
    assert by["X1.R1"].nodes == ["in", "vdd"]        # vdd not prefixed
    assert models["DX"].params["N"] == 1.0


def test_subcircuit_errors_are_named():
    with pytest.raises(ValueError, match="not defined"):
        p.spice_to_builder("V1 a 0 DC 1\nX1 a 0 NOPE\n.END\n")
    with pytest.raises(ValueError, match="has 2 ports"):
        p.spice_to_builder(".SUBCKT S a b\nR1 a b 1\n.ENDS\nV1 a 0 DC 1\nX1 a S\n.END\n")
    with pytest.raises(ValueError, match="instantiates itself"):
        p.spice_to_builder(".SUBCKT S a b\nX1 a b S\n.ENDS\nV1 a 0 DC 1\nX1 a 0 S\n.END\n")
    with pytest.raises(ValueError, match="no .ENDS"):
        p.spice_to_builder(".SUBCKT S a b\nR1 a b 1\nV1 a 0 DC 1\n.END\n")
