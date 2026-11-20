"""Task 12.8 (add-ai-usage-guide): the switching energy of a hard-switched
leg, with and without dead time, against a closed form independent of the
simulator.

One leg: 600 V bus, bidirectional switches HS (dcp -> a) and LS (a -> gnd),
R_on = 1 mOhm, each with an antiparallel PWL diode (V_f = 0.7 V, R_d =
50 mOhm); load R = 10 Ohm + L = 5 mH from ``a`` to a source ``Vm``: Vm = 0
keeps the load current positive (out of ``a``), Vm = 600 V negative, and
Vm = 300 V at duty 0.5 makes it a zero-mean triangle that crosses zero
inside every period. Switching tables E(v, i) = E_ref (v / 600)(i / 50),
bilinear, so exact in a LossTable.

The truth (datasheet-table reading of a hard-switched leg): at each
commutation the device that TAKES the current turns on against the voltage
it was blocking (E_on), the channel that INTERRUPTS current in its forward
direction turns off against the voltage that comes up (E_off), and the
complementary edge of the pair is soft. With a dead time the antiparallel
diode carries the current in between, so the soft channel switches against
the diode's drop V_f + R_d |i| and the hard one against the bus plus that
drop; with none, the soft edge has no voltage across it and costs 0. The
load current at each edge is the periodic solution of the piecewise-linear
R-L circuit (exponentials in closed form, no simulator).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p

V, R, L, F = 600.0, 10.0, 5e-3, 10e3
R_ON, V_F, R_D = 1e-3, 0.7, 0.05
E_ON, E_OFF = 4e-3, 3.5e-3
T_FROM, T_END = 4.975e-3, 5.975e-3          # ten periods after 10 tau


def _tab(e):
    return p.LossTable(v_axis=[0.0, V], i_axis=[0.0, 50.0], tj_axis=[25.0],
                       energy=np.array([[[0.0], [0.0]], [[0.0], [e]]]))


def _e(e_ref, v, i):
    return e_ref * (v / V) * (abs(i) / 50.0)


def truth(vm, td, duty=0.5):
    """Per-device switching power (HS, LS) in W, closed form.

    The schedule of make_dead_time_pwm_pair_fn: HS on [0, DT - td), LS on
    [DT, T - td), both off for td before each turn-on. A segment drives
    the load with v_a - vm = u - r i, so i(t) = u/R' + (i0 - u/R') e^(-t R'/L)
    with R' = R + r; the periodic orbit is the fixed point of the
    composed affine map."""
    T = 1.0 / F
    t_hs, t_ls = duty * T - td, (1.0 - duty) * T - td

    def seg(i0, dur, u, r):
        rr = R + r
        return u / rr + (i0 - u / rr) * math.exp(-dur * rr / L)

    def dead(i0):     # the diode that carries i0: Dl for i > 0, Dh for i < 0
        return ((-V_F - vm, R_D) if i0 > 0 else (V + V_F - vm, R_D))

    def period(i0):
        i1 = seg(i0, t_hs, V - vm, R_ON)            # HS conducts
        i2 = seg(i1, td, *dead(i1)) if td else i1   # dead time
        i3 = seg(i2, t_ls, -vm, R_ON)               # LS conducts
        i4 = seg(i3, td, *dead(i3)) if td else i3
        return i1, i2, i3, i4

    # the map is affine in i0 while the dead-time diodes do not change:
    # Newton on that piece, its slope taken at the current guess
    i0 = 0.0
    for _ in range(6):
        b = period(i0 + 1e-3)[3] - period(i0)[3]
        i0 += (period(i0)[3] - i0) / (1.0 - b / 1e-3)
    i1, i2, i3, i4 = period(i0)
    for x, y in ((i1, i2), (i3, i4)):
        assert td == 0 or x * y > 0, "the current reverses in a dead time"
    hs = ls = 0.0
    # HS turns off carrying i1 (its forward direction is i > 0)
    if i1 > 0:      # HS interrupts forward current: hard; LS takes it softly
        hs += _e(E_OFF, V + (V_F + R_D * i1 if td else 0.0), i1)
        ls += _e(E_ON, V_F + R_D * i2, i2) if td else 0.0
    else:           # HS carried reverse current: Dh takes it, LS turns on hard
        hs += _e(E_OFF, V_F + R_D * abs(i1), i1) if td else 0.0
        ls += _e(E_ON, V + (V_F + R_D * abs(i2) if td else 0.0), i2)
    # LS turns off carrying -i3 (its forward direction is a -> gnd)
    if i3 < 0:      # LS interrupts forward current: hard
        ls += _e(E_OFF, V + (V_F + R_D * abs(i3) if td else 0.0), i3)
        hs += _e(E_ON, V_F + R_D * abs(i4), i4) if td else 0.0
    else:           # LS carried reverse current: Dl takes it, HS hard on
        ls += _e(E_OFF, V_F + R_D * i3, i3) if td else 0.0
        hs += _e(E_ON, V + (V_F + R_D * i4 if td else 0.0), i4)
    return hs * F, ls * F


def _leg(vm, td, port):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", V)
    b.add_voltage_source("Vm", "m", "gnd", vm)
    b.add_switch("HS", "dcp", "a", 1.0 / R_ON, 1e-9)
    b.add_switch("LS", "a", "gnd", 1.0 / R_ON, 1e-9)
    b.add_diode("Dh", "a", "dcp", 1.0 / R_D, 1e-9, V_f=V_F)
    b.add_diode("Dl", "gnd", "a", 1.0 / R_D, 1e-9, V_f=V_F)
    b.add_resistor("R", "a", "x", R)
    b.add_inductor("L", "x", "m", L)
    if port:
        for d in ("HS", "LS"):
            b.set_device_losses(d, Tj=25.0, E_on=_tab(E_ON), E_off=_tab(E_OFF))
    fn = p.make_dead_time_pwm_pair_fn(F, 0.5, b.switch_index_of("HS"),
                                      b.switch_index_of("LS"),
                                      b.graph.num_switches, td)
    return b, fn


def measured(vm, td, engine):
    """(HS, LS) switching power over [T_FROM, T_END): the in-kernel loss
    port on `engine`, or device_loss_summary's table path ('table', on a
    pwl result)."""
    port = engine != "table"
    b, fn = _leg(vm, td, port)
    kw = {"dt": 1e-7} if engine in ("pwl", "table") else {}
    res = p.simulate(b, t_end=T_END, switch_fn=fn,
                     engine="pwl" if engine == "table" else engine, **kw)
    if port:
        out = []
        for d in ("HS", "LS"):
            ev = res.loss_events(d)
            w = ev["t"] >= T_FROM
            out.append(math.fsum(ev["E"][w]) / (T_END - T_FROM))
        return tuple(out)
    spec = {"E_on_ref": E_ON, "E_off_ref": E_OFF, "V_ref": V, "I_ref": 50.0}
    rows = {r["name"]: r for r in p.device_loss_summary(
        b, res, switch_specs={"HS": spec, "LS": spec}, t_from=T_FROM)}
    return rows["HS"]["P_sw_avg"], rows["LS"]["P_sw_avg"]


# pwl and the table path read the samples one 100 ns step either side of
# the edge (first order: at Vm = 300 V the 1.5 A edge current moves
# 300 V / 5 mH * 1e-7 s = 6 mA per step, 4e-3); trbdf2 and dsed read the
# edge's own left / right limits.
_RTOL = {"pwl": 5e-3, "table": 5e-3, "trbdf2": 2e-4, "dsed": 2e-4}
# an absolute floor: one soft edge's diode-drop energy, read a step late
_ATOL = 1e-3


@pytest.mark.parametrize("engine", ["pwl", "table", "trbdf2", "dsed"])
@pytest.mark.parametrize("vm", [0.0, 300.0, 600.0])
def test_zero_dead_time_leg_charges_one_hard_edge_per_commutation(engine, vm):
    hs, ls = measured(vm, 0.0, engine)
    t_hs, t_ls = truth(vm, 0.0)
    assert hs == pytest.approx(t_hs, rel=_RTOL[engine], abs=_ATOL)
    assert ls == pytest.approx(t_ls, rel=_RTOL[engine], abs=_ATOL)


@pytest.mark.parametrize("engine", ["pwl", "table", "trbdf2", "dsed"])
@pytest.mark.parametrize("vm", [0.0, 300.0])
def test_dead_time_leg_matches_the_closed_form(engine, vm):
    hs, ls = measured(vm, 1e-6, engine)
    t_hs, t_ls = truth(vm, 1e-6)
    assert hs == pytest.approx(t_hs, rel=_RTOL[engine], abs=_ATOL)
    assert ls == pytest.approx(t_ls, rel=_RTOL[engine], abs=_ATOL)


def test_leg_loss_is_continuous_as_the_dead_time_vanishes():
    """P_sw(td) for td = 1 us, 100 ns, 10 ns, 0 on the dsed engine (it
    lands every edge): the leg total converges to the zero-dead-time
    value; what remains at td -> 0 is the two soft edges' diode-drop
    energy, (V_f + R_d i) / V of a hard edge."""
    vm = 0.0
    p0 = sum(measured(vm, 0.0, "dsed"))
    t0 = sum(truth(vm, 0.0))
    assert p0 == pytest.approx(t0, rel=2e-4)
    for td in (1e-6, 1e-7, 1e-8):
        ptd = sum(measured(vm, td, "dsed"))
        assert ptd == pytest.approx(sum(truth(vm, td)), rel=2e-4)
        # the diode-drop share bounds the gap at every td
        assert abs(ptd - p0) / p0 < 1e-3 + 4 * td * F
