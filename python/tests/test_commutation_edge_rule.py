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
import warnings

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

    The schedule of make_dead_time_pwm_pair_fn: HS on [0, D T - td), LS on
    [D T, T - td), both off for td before each turn-on. A segment drives
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


@pytest.mark.parametrize("vm", [0.0, 300.0])
def test_leg_loss_is_continuous_as_the_dead_time_vanishes(vm):
    """P_sw(td) for td = 1 us, 100 ns, 10 ns, 0 on the dsed engine (it
    lands every edge). The leg total tends to the zero-dead-time value
    up to the diode's drop: with any dead time the diode conducts in
    between, so the hard edge commutates V + V_f + R_d |i| and the soft
    one V_f + R_d |i|, where with none they see V and nothing. The gap is
    at most 2 (V_f + R_d i_max) / V of the total (0.98 % here at Vm = 0,
    0.70 % measured) plus the dead time's own O(td f) change of the
    current. Before task 12.8 it was 100 % (the soft edges at the bus)."""
    p0 = sum(measured(vm, 0.0, "dsed"))
    assert p0 == pytest.approx(sum(truth(vm, 0.0)), rel=2e-4)
    i_max = V / R
    for td in (1e-6, 1e-7, 1e-8):
        ptd = sum(measured(vm, td, "dsed"))
        assert ptd == pytest.approx(sum(truth(vm, td)), rel=2e-4)
        assert abs(ptd - p0) / p0 < 2 * (V_F + R_D * i_max) / V + 4 * td * F


def test_three_phase_vsi_switch_loss_is_the_half_cycle_average():
    """Guide 3.7's VSI (600 V, 10 kHz SPWM, m = 0.8, 50 Hz, 10 Ohm + 5 mH
    star load) with NO dead time, on dsed.

    Closed form: the load current is I sin(wt - phi) with I = sqrt(2) V_ph
    / |Z|, V_ph = sqrt(3/8) m V / sqrt(3). A switch is the hard one of its
    leg over the half cycle in which the current flows in its forward
    direction; there each carrier period costs (E_on + E_off)(V / 600)
    (|i| / 50) (linear tables). Averaged over a fundamental period each
    switch costs f_sw (E_on + E_off)(I / 50) / pi = 11.32 W, the leg
    22.65 W (45.28 W before task 12.8: every edge at the bus). The leg
    total holds to 0.5 % (the (E_off - E_on) x ripple term is 0.2 %).

    The split between the two switches is not 50/50 (11.61 / 10.99 W):
    the floating neutral bends the ripple at the other legs' edges, so
    the edge samples of i_a are biased differently in the two half
    cycles. Per device the truth is the classification by the load
    current itself: each commutation of the leg (an HSa gate edge) costs
    E_on + E_off of whichever switch carries i_a forward, at |i_a| there.
    """
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", V)
    vsi = p.add_three_phase_vsi(b, "INV", vdc_pos="dcp", vdc_neg="gnd",
                                out_a="a", out_b="b", out_c="c")
    for ph in "abc":
        b.add_resistor(f"R{ph}", ph, f"m{ph}", 10.0)
        b.add_inductor(f"L{ph}", f"m{ph}", "n", 5e-3)
    for d in ("INV__HSa", "INV__LSa"):
        b.set_device_losses(d, Tj=25.0, E_on=_tab(E_ON), E_off=_tab(E_OFF))
    legs = p.ThreePhaseLegIndices(*vsi.switch_indices)
    fn = p.make_three_phase_spwm_fn(F, 50.0, 0.8, legs,
                                    b.graph.num_switches, 0.0)
    res = p.simulate(b, t_end=40e-3, switch_fn=fn, engine="dsed")
    v_ph = math.sqrt(3.0 / 8.0) * 0.8 * V / math.sqrt(3.0)
    i_pk = math.sqrt(2.0) * v_ph / abs(complex(10.0, 2 * math.pi * 50 * 5e-3))
    p_truth = F * (E_ON + E_OFF) * (i_pk / 50.0) / math.pi
    p_sw = {}
    for d in ("INV__HSa", "INV__LSa"):
        ev = res.loss_events(d)
        w = ev["t"] >= 20e-3
        p_sw[d] = math.fsum(ev["E"][w]) / 20e-3
    assert p_sw["INV__HSa"] + p_sw["INV__LSa"] == pytest.approx(
        2 * p_truth, rel=5e-3)
    # per device: the load current's sign at each commutation decides
    t = np.asarray(res.times)
    ia = np.asarray(res.i("La"))
    hs = ls = 0.0
    ev = res.loss_events("INV__HSa")
    for e in ev[ev["t"] >= 20e-3]:
        i = float(np.interp(e["t"], t, ia))
        hs_on = e["kind"] == "on"
        if i > 0:
            hs += (E_ON if hs_on else E_OFF) * i / 50.0
        else:
            ls += (E_OFF if hs_on else E_ON) * -i / 50.0
    assert p_sw["INV__HSa"] == pytest.approx(hs / 20e-3, rel=1e-3)
    assert p_sw["INV__LSa"] == pytest.approx(ls / 20e-3, rel=1e-3)


# ---- task 12.8 follow-up: edges at about zero current, impulsive edges,
# non-finite readings (thermal::commutation_point, one rule, four callers)

def _tab_c(c, e, v_ref=V):
    """E(v, i) = c v / v_ref + e (v / v_ref)(i / 50): an E(v, 0) intercept
    (a datasheet table's output-capacitance energy), bilinear."""
    return p.LossTable(v_axis=[0.0, v_ref], i_axis=[0.0, 50.0], tj_axis=[25.0],
                       energy=np.array([[[0.0], [0.0]], [[c], [c + e]]]))


def test_a_reverse_current_within_the_leakage_has_no_sign():
    """The rule on its own: a turn-on from 600 V whose closed-side current
    is -1e-15 A (rounding) or -1e-7 A (below the 6e-7 A the open switch
    itself leaks, g_off = 1 nS) is hard; -40 A (a conducted current) is
    soft. Same answer from the period path's binding and the table
    path's numpy mirror."""
    from pulsim._pulsim import _commutation_points
    from pulsim.losses import _switch_switching_loss
    law = (True, 1e3, 1e-9)
    tab = _tab_c(1e-3, 4e-3)
    spec = {"E_on_table": tab, "Tj": 25.0}
    for i_after, soft in ((-1e-15, False), (-1e-7, False), (-40.0, True)):
        seq = [(False, 600.0, 6e-7), (True, 1e-3, i_after)]
        (on,), _ = _commutation_points(seq, *law)
        assert on[2] is soft
        assert on[0] == (0.0 if soft else 600.0)
        out = _switch_switching_loss(np.array([False, True]), np.zeros(2),
                                     np.array([600.0, 1e-3]),
                                     np.array([6e-7, i_after]), spec, law=law)
        assert out["E_sw_on_total"] == (0.0 if soft else pytest.approx(
            1e-3 + 4e-3 * abs(i_after) / 50.0, rel=1e-12))


# DCM buck: 24 V, 10 uH, 100 uF, 20 Ohm, 100 kHz, D = 0.3 (K = 2L/(R T) =
# 0.1 < 1 - D); switch and diode 1 mOhm / 1 nS, V_f = 0
VIN, LB, CB, RB, FB, DB = 24.0, 10e-6, 100e-6, 20.0, 100e3, 0.3
C_ON, E_ONB, VT = 1e-6, 2e-4, 50.0


def _dcm_vout():
    """M = 2 / (1 + sqrt(1 + 4K / D^2)) of the ideal DCM buck."""
    k = 2.0 * LB * FB / RB
    return VIN * 2.0 / (1.0 + math.sqrt(1.0 + 4.0 * k / DB ** 2))


@pytest.mark.parametrize("engine", ["pwl", "table", "trbdf2", "dsed"])
def test_a_dcm_turn_on_books_the_table_intercept(engine):
    """Every turn-on of a DCM buck starts from zero inductor current: the
    closed-side reading is the off devices' leakage (or nothing), and the
    edge books E_on(V_in - V_out, ~0) -- the table's intercept -- every
    period. Truth: 50 turn-ons x [c (V_in - V_out)/V_T + the current part
    at the reading the engine makes: 0 at the edge's right limit (trbdf2,
    dsed), (V_in - V_out) dt / L one step later (pwl, the table path on
    its result)]. V_out from the closed-form DCM ratio (1e-3 here: the
    1 mOhm drops and the output ripple). A regression guard: a5af1ade
    booked these too (the leakage reads forward), the rule-level tests
    above are the ones that fail on it."""
    vo = _dcm_vout()
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", LB)
    b.add_capacitor("C", "out", "gnd", CB, vo)
    b.add_resistor("R", "out", "gnd", RB)
    tab = _tab_c(C_ON, E_ONB, VT)
    if engine != "table":
        b.set_device_losses("S", Tj=25.0, E_on=tab)
    fn = p.make_pwm_switch_fn(FB, DB, b.switch_index_of("S"),
                              b.graph.num_switches, 0.0)
    dt = 1e-8
    # the default pwl rings on the switch node's leakage mode in the idle
    # interval (trapezoidal at tau = L g_off << dt); the restart rule does not
    kw = ({"dt": dt, "discontinuity_restart": True}
          if engine in ("pwl", "table") else {})
    t0, t1 = 1.5e-3 - 0.5 / FB, 2e-3 - 0.5 / FB
    res = p.simulate(b, t_end=t1, switch_fn=fn,
                     engine="pwl" if engine == "table" else engine, **kw)
    if engine == "table":
        row = next(r for r in p.device_loss_summary(
            b, res, switch_specs={"S": {"E_on_table": tab, "Tj": 25.0}},
            t_from=t0) if r["name"] == "S")
        n, e_on = row["n_turn_on_events"], row["E_sw_on_total"]
    else:
        ev = res.loss_events("S")
        ev = ev[(ev["t"] >= t0) & (ev["kind"] == "on")]
        n, e_on = len(ev), math.fsum(ev["E"])
    v = VIN - vo
    i = v * dt / LB if engine in ("pwl", "table") else 0.0
    assert n == 50
    assert e_on == pytest.approx(
        n * (C_ON * v / VT + E_ONB * (v / VT) * (i / 50.0)), rel=2e-3)


def _bare_leg(goff):
    """The leg with nothing on its output (a state off the leg, R-C on the
    bus, keeps every engine's state vector non-empty): every commutation is
    at zero current -- the off devices' leakage, or exactly 0."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", V)
    b.add_resistor("Rb", "dcp", "y", 1.0)
    b.add_capacitor("Cb", "y", "gnd", 1e-6)
    b.add_switch("HS", "dcp", "a", 1.0 / R_ON, goff)
    b.add_switch("LS", "a", "gnd", 1.0 / R_ON, goff)
    b.add_diode("Dh", "a", "dcp", 1.0 / R_D, goff, V_f=V_F)
    b.add_diode("Dl", "gnd", "a", 1.0 / R_D, goff, V_f=V_F)
    fn = p.make_dead_time_pwm_pair_fn(F, 0.5, b.switch_index_of("HS"),
                                      b.switch_index_of("LS"),
                                      b.graph.num_switches, 0.0)
    return b, fn


@pytest.mark.parametrize("goff", [1e-9, 0.0])
@pytest.mark.parametrize("engine", ["pwl", "table", "trbdf2", "dsed"])
def test_a_zero_current_commutation_books_the_intercept_once(engine, goff):
    """No dead time, no load current: each of the 20 commutations in
    [T/4, 1 ms + T/4) books E_on(600 V, ~0) = 1 mJ once (the incoming
    switch; E_off has no intercept), on every engine and the table
    path. A regression guard (a5af1ade passes: the leakage of the newly
    blocking devices flows forward through the incoming switch)."""
    b, fn = _bare_leg(goff)
    tab_on, tab_off = _tab_c(1e-3, E_ON), _tab_c(0.0, E_OFF)
    if engine != "table":
        for d in ("HS", "LS"):
            b.set_device_losses(d, Tj=25.0, E_on=tab_on, E_off=tab_off)
    t0, t1 = 0.25 / F, 1e-3 + 0.25 / F
    res = p.simulate(b, t_end=t1, switch_fn=fn,
                     engine="pwl" if engine == "table" else engine,
                     **({"dt": 1e-7} if engine in ("pwl", "table") else {}))
    total = 0.0
    if engine == "table":
        spec = {"E_on_table": tab_on, "E_off_table": tab_off, "Tj": 25.0}
        for r in p.device_loss_summary(b, res, t_from=t0,
                                       switch_specs={"HS": spec, "LS": spec}):
            if r["name"] in ("HS", "LS"):
                total += r["E_sw_total"]
    else:
        for d in ("HS", "LS"):
            ev = res.loss_events(d)
            total += math.fsum(ev["E"][ev["t"] >= t0])
    assert total == pytest.approx(20 * 1e-3, rel=1e-6)


@pytest.mark.parametrize("engine", ["pwl", "table", "trbdf2", "dsed"])
def test_soft_edges_of_amps_stay_free_with_an_intercept(engine):
    """Vm = 0, no dead time: LS turns on and off against the load's ~45 A
    in its reverse direction -- soft, 0 J even where the table has an
    intercept; HS books its hard edges plus one intercept per period.
    A regression guard: the threshold must not turn real amps hard."""
    c = 1e-3
    b, fn = _leg(0.0, 0.0, False)
    tab_on, tab_off = _tab_c(c, E_ON), _tab(E_OFF)
    if engine != "table":
        for d in ("HS", "LS"):
            b.set_device_losses(d, Tj=25.0, E_on=tab_on, E_off=tab_off)
    kw = {"dt": 1e-7} if engine in ("pwl", "table") else {}
    res = p.simulate(b, t_end=T_END, switch_fn=fn,
                     engine="pwl" if engine == "table" else engine, **kw)
    if engine == "table":
        spec = {"E_on_table": tab_on, "E_off_table": tab_off, "Tj": 25.0}
        rows = {r["name"]: r for r in p.device_loss_summary(
            b, res, switch_specs={"HS": spec, "LS": spec}, t_from=T_FROM)}
        hs, ls = rows["HS"]["P_sw_avg"], rows["LS"]["E_sw_total"]
    else:
        e = {}
        for d in ("HS", "LS"):
            ev = res.loss_events(d)
            e[d] = math.fsum(ev["E"][ev["t"] >= T_FROM])
        hs, ls = e["HS"] / (T_END - T_FROM), e["LS"]
    assert ls == 0.0
    assert hs == pytest.approx(truth(0.0, 0.0)[0] + c * F,
                               rel=_RTOL[engine], abs=_ATOL)


def _cap_loop(engine, t_end=5e-6):
    """1 uF at 10 V closed onto 3 uF at 2 V through an ideal switch at
    1 us, opened at 2 us; E(v, i) = 1e-9 (v / 10) + 1e-6 i."""
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "x", "gnd", 1e-6, 10.0)
    b.add_capacitor("C2", "y", "gnd", 3e-6, 2.0)
    b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
    k = b.switch_index_of("S")

    def fn(t, k=k, n=b.graph.num_switches):
        m = p.SwitchStateMask(n)
        m.set(k, 1e-6 <= t < 2e-6)
        return m
    tab = p.LossTable(v_axis=[0.0, 10.0], i_axis=[0.0, 1.0], tj_axis=[25.0],
                      energy=np.array([[[0.0], [1e-6]], [[1e-9], [1e-9 + 1e-6]]]))
    if not engine.startswith("table"):
        b.set_device_losses("S", Tj=25.0, E_on=tab, E_off=tab)
    kw = ({"dt": 1e-8, "discontinuity_restart": True}
          if engine in ("pwl", "table_pwl") else {})
    run_on = {"table": "trbdf2", "table_dsed": "dsed",
              "table_pwl": "pwl"}.get(engine, engine)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=t_end, switch_fn=fn, auto_regularize=False,
                       engine=run_on, **kw)
    return b, r, tab


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed", "table",
                                    "table_dsed"])
def test_an_impulsive_turn_on_reads_the_pre_jump_voltage(engine):
    """Closing the capacitor loop is an impulse (E_impulse = 1/2 C1 C2 /
    (C1 + C2) dV^2 = 24 uJ), not a commutation of load current: the
    closed side's -8397 A on trbdf2 is the jump's own step. The edge is
    hard at the 8 V the switch blocked and the regular current after the
    jump (0: the loop carries none once equal): E_on = 0.8 nJ, never soft
    (it read 0 on trbdf2), never at the jump's current (1 uJ)."""
    b, r, tab = _cap_loop(engine)
    dw = 0.5 * 1e-6 * 3e-6 / 4e-6 * 8.0 ** 2
    if engine.startswith("table"):
        row = next(x for x in p.device_loss_summary(
            b, r, switch_specs={"S": {"E_on_table": tab, "E_off_table": tab,
                                      "Tj": 25.0}}) if x["name"] == "S")
        assert row["n_turn_on_events"] == 1
        assert row["E_sw_on_total"] == pytest.approx(0.8e-9, rel=1e-6)
        from pulsim.losses import impulse_losses
        assert impulse_losses(b, r)["S"]["E_impulse_total"] == pytest.approx(
            dw, rel=1e-8)
        return
    e = r.device_energy("S")
    assert (e["n_on"], e["n_off"]) == (1, 1)
    assert e["E_impulse"] == pytest.approx(dw, rel=1e-8)
    ev = r.loss_events("S")
    on = ev[ev["kind"] == "on"][0]
    assert on["v"] == pytest.approx(8.0, rel=1e-6)
    assert on["E"] == pytest.approx(0.8e-9, rel=1e-6)


def test_a_non_finite_reading_is_refused_on_the_table_and_period_paths():
    """An infinite blocking voltage against a reverse current used to be
    soft (v = 0); it is refused by name. The period path's binding keeps
    it hard (the general path then refuses it)."""
    from pulsim._pulsim import _commutation_points
    from pulsim.losses import _switch_switching_loss
    (on,), _ = _commutation_points([(False, math.inf, 0.0), (True, 0.0, -40.0)],
                                   True, 1e3, 1e-9)
    assert on[2] is False and math.isinf(on[0])
    with pytest.raises(ValueError, match=r"'Q'.*non-finite"):
        _switch_switching_loss(np.array([False, True]), np.zeros(2),
                               np.array([math.inf, 0.0]), np.array([0.0, -40.0]),
                               {"E_on_table": _tab(E_ON), "Tj": 25.0},
                               law=(True, 1e3, 1e-9), name="Q")


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed", "table"])
def test_an_impulsive_edge_on_the_last_step_is_kept(engine):
    """The run ends right after the loop closes (pwl: on the jump's own
    step, 1e-6 -> 1.01e-6): no regular state follows, so the edge is read
    on its pre-edge side (8 V and the open switch's 8 nA), 0.8 nJ, never
    dropped and never at the jump's current. The table path on that pwl
    result books the same."""
    t_end = 1.01e-6 if engine in ("pwl", "table") else 1e-6 + 1e-9
    b, r, tab = _cap_loop("pwl" if engine == "table" else engine, t_end)
    if engine in ("pwl", "table"):
        assert np.asarray(r.times)[-2] == pytest.approx(1e-6)  # the jump's step is the last
    if engine == "table":
        from pulsim.losses import device_loss_summary
        b2, r2, _ = _cap_loop("table_pwl", t_end)
        row = next(x for x in device_loss_summary(
            b2, r2, switch_specs={"S": {"E_on_table": tab, "Tj": 25.0}})
            if x["name"] == "S")
        assert row["E_sw_on_total"] == pytest.approx(0.8e-9, rel=1e-4)
        return
    e = r.device_energy("S")
    assert e["n_on"] == 1
    ev = r.loss_events("S")
    assert len(ev) == 1 and ev["v"][0] == pytest.approx(8.0, rel=1e-6)
    assert e["E_on"] == pytest.approx(0.8e-9, rel=1e-4)
