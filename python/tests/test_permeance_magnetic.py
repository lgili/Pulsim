"""Permeance (gyrator–capacitor) magnetic domain.

openspec change add-permeance-magnetic-domain. Every expected number is a
closed form of the magnetic circuit (Kirchhoff on the reluctance network,
L = N²·P_seen) or a cross-model check against an independent device (the
coupled-inductor `add_transformer`, the T-model, the saturable
transformer); the saturating three-leg case is checked against an
independent fourth-order Runge–Kutta solution of the same equations
(`probes/saturating_leg_reference.py`, numbers pinned below). The
parameters are illustrations, not a real part.

Energy is checked with the discrete identity of the trapezoidal rule: on
a fixed grid, with x̄ the mean of two consecutive samples, the energy a
port takes in over a step, h·v̄·ī, equals the change of P·F²/2 of a
linear permeance and R·ī²·h of a resistor exactly, so the balance
electrical input = stored energy change + loss holds to rounding.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

import pulsim as p

MU0 = 4e-7 * math.pi

# The gapped-core example of the design (ETD29-class, synthetic numbers).
N_G, A_G, L_G, GAP, MUR, BSAT = 25, 76e-6, 72e-3, 0.5e-3, 2000.0, 0.35
P_CORE = MU0 * MUR * A_G / L_G           # 2.653 µH
P_GAP = MU0 * A_G / GAP                  # 191.0 nH
L_GAPPED = N_G ** 2 / (1 / P_CORE + 1 / P_GAP)   # 111.36 µH


def rel(a, b, scale=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.max(np.abs(b)) if scale is None else scale
    return float(np.max(np.abs(a - b)) / s)


def mid(x):
    x = np.asarray(x, float)
    return 0.5 * (x[1:] + x[:-1])


def slope(res, branch, k0=-200, k1=-1, port=None):
    i = res.i(branch) if port is None else res.i(branch, port=port)
    t = np.asarray(res.times)
    return (i[k1] - i[k0]) / (t[k1] - t[k0])


# ---------------------------------------------------------------------------
# Circuits
# ---------------------------------------------------------------------------

def gapped_inductor(V=1.0, *, R=None):
    b = p.CircuitBuilder()
    if R is None:
        b.add_voltage_source("V", "e", "gnd", V)
    else:
        b.add_voltage_source("V", "in", "gnd", V)
        b.add_resistor("R", "in", "e", R)
    b.add_winding("W", "e", "gnd", "m1", "gnd", N_G)
    b.add_permeance("Pc", "m1", "m2", P_CORE)
    b.add_air_gap("G", "m2", "gnd", A_G, GAP)
    return b


def winding_current(res, name="W"):
    return np.asarray(res.i(name, port=1))


# ---------------------------------------------------------------------------
# 1. Linear networks
# ---------------------------------------------------------------------------

def test_air_gap_permeance_is_mu0_area_over_length():
    assert P_GAP == pytest.approx(191.0e-9, rel=1e-3)
    assert P_CORE == pytest.approx(2.653e-6, rel=1e-3)
    assert L_GAPPED == pytest.approx(111.36e-6, rel=1e-4)


@pytest.mark.filterwarnings("ignore:simulate\\(engine='trbdf2'\\)")
@pytest.mark.parametrize("engine,tol", [("pwl", 1e-9), ("trbdf2", 1e-9), ("dsed", 1e-9)])
def test_gapped_core_terminal_inductance(engine, tol):
    # A 1 V step across the winding: di/dt = V/L once the start is past.
    kw = dict(dt=1e-8) if engine == "pwl" else dict(rtol=1e-10, atol=1e-14)
    res = p.simulate(gapped_inductor(), t_end=4e-6, engine=engine, **kw)
    L = 1.0 / slope(res, "W", k0=len(res.times) // 2, port=1)
    assert abs(L - L_GAPPED) / L_GAPPED < tol


def test_gapped_core_energy_splits_between_core_and_gap():
    res = p.simulate(gapped_inductor(), t_end=4e-6, dt=1e-8, engine="pwl")
    i = winding_current(res)
    phi_c, phi_g = res.flux("Pc"), res.flux("G")
    # One flux path: the same flux through core and gap, every sample.
    assert rel(phi_c, phi_g) < 1e-10
    E_core = 0.5 * phi_c ** 2 / P_CORE
    E_gap = 0.5 * phi_g ** 2 / P_GAP
    # Closed form: E_gap / E_core = P_core / P_gap (same flux).
    k = len(i) - 1
    assert E_gap[k] / E_core[k] == pytest.approx(P_CORE / P_GAP, rel=1e-9)
    # Total = L·i²/2, and the source supplied it (discrete identity).
    assert E_core[k] + E_gap[k] == pytest.approx(0.5 * L_GAPPED * i[k] ** 2, rel=1e-9)
    E_in = np.sum(np.diff(res.times) * mid(res.v("e")) * mid(i))
    assert E_in == pytest.approx(E_core[k] + E_gap[k], rel=1e-9)
    # B and H from geometry.
    assert res.b_field("G")[k] == pytest.approx(phi_g[k] / A_G, rel=1e-12)
    F_gap = res.v("m2")[k]
    assert res.h_field("G")[k] == pytest.approx(F_gap / GAP, rel=1e-12)


NP, NS = 20, 10
PC, PSP, PSS = 2e-6, 2e-8, 3e-8


def transformer_permeance(load=10.0):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "p", "gnd", 0.0, 100.0, 10e3)
    b.add_winding("Wp", "p", "gnd", "a", "gnd", NP)
    b.add_permeance("Lp", "a", "gnd", PSP)
    b.add_permeance("Pc", "a", "b", PC)
    # reversed magnetic port: the dots of both windings on e_pos
    b.add_winding("Ws", "s", "gnd", "gnd", "b", NS)
    b.add_permeance("Ls", "b", "gnd", PSS)
    b.add_resistor("RL", "s", "gnd", load)
    return b


def transformer_coupled(load=10.0):
    Lp = NP ** 2 * (PSP + PC)
    Ls = NS ** 2 * (PSS + PC)
    M = NP * NS * PC
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "p", "gnd", 0.0, 100.0, 10e3)
    b.add_transformer("T", "p", "gnd", "s", "gnd", Lp, Ls, M / math.sqrt(Lp * Ls))
    b.add_resistor("RL", "s", "gnd", load)
    return b


def transformer_tmodel(load=10.0):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "p", "gnd", 0.0, 100.0, 10e3)
    b.add_inductor("Llp", "p", "x", NP ** 2 * PSP)
    b.add_inductor("Lm", "x", "gnd", NP ** 2 * PC)
    b.add_ideal_transformer("X", "x", "gnd", "y", "gnd", NS / NP)
    b.add_inductor("Lls", "y", "s", NS ** 2 * PSS)
    b.add_resistor("RL", "s", "gnd", load)
    return b


def test_transformer_from_geometry_matches_the_coupled_inductor_and_t_model():
    kw = dict(t_end=3e-4, dt=2e-7, engine="pwl")
    a = p.simulate(transformer_permeance(), **kw)
    c = p.simulate(transformer_coupled(), **kw)
    t = p.simulate(transformer_tmodel(), **kw)
    is_ = a.i("RL")
    assert rel(is_, c.i("RL")) < 1e-6
    assert rel(is_, t.i("RL")) < 1e-6
    assert rel(a.v("s"), c.v("s")) < 1e-6
    # Primary current: the source branch current of each form.
    assert rel(a.i("V"), c.i("V")) < 1e-6
    assert rel(a.i("V"), t.i("V")) < 1e-6


@pytest.mark.parametrize("engine,tol", [("trbdf2", 1e-6), ("dsed", 1e-9)])
def test_transformer_from_geometry_on_the_other_engines(engine, tol):
    kw = dict(t_end=3e-4, engine=engine, rtol=1e-10, atol=1e-13)
    a = p.simulate(transformer_permeance(), **kw)
    c = p.simulate(transformer_coupled(), **kw)
    tt = np.linspace(1e-5, 3e-4, 60)
    assert rel(np.interp(tt, a.times, a.i("RL")),
               np.interp(tt, c.times, c.i("RL"))) < tol


# ---- three-limb core -------------------------------------------------------

N3, P_LIMB, P_AIR = 100, 5e-6, 50e-9
L1_3 = N3 ** 2 * P_LIMB                                  # 50 mH
L0_3 = N3 ** 2 / (1 / P_LIMB + 3 / P_AIR)                # 166.1 µH


def three_limb(v, *, saturable=False):
    # Limb k: winding k (gnd -> x_k) in series with the limb permeance
    # (x_k -> T); the top yoke T returns to the bottom yoke (gnd) through
    # the air path. Yokes are ideal (zero reluctance).
    b = p.CircuitBuilder()
    for k, vk in enumerate(v):
        b.add_voltage_source(f"V{k}", f"e{k}", "gnd", vk)
        b.add_winding(f"W{k}", f"e{k}", "gnd", f"x{k}", "gnd", N3)
        if saturable:
            b.add_saturable_permeance(f"L{k}", f"x{k}", "T", 4e-4, 0.25,
                                      mu_r=2487.0, B_sat=1.5)
        else:
            b.add_permeance(f"L{k}", f"x{k}", "T", P_LIMB)
    b.add_permeance("Air", "T", "gnd", P_AIR)
    return b


def test_three_limb_positive_and_zero_sequence_inductance():
    assert L1_3 == pytest.approx(50e-3, rel=1e-12)
    assert L0_3 == pytest.approx(166.1e-6, rel=1e-3)
    pos = p.simulate(three_limb((1.0, -0.5, -0.5)), t_end=2e-4, dt=1e-6, engine="pwl")
    for k, vk in enumerate((1.0, -0.5, -0.5)):
        L = vk / slope(pos, f"W{k}", k0=50, port=1)
        assert abs(L - L1_3) / L1_3 < 1e-6
    zero = p.simulate(three_limb((1.0, 1.0, 1.0)), t_end=2e-5, dt=1e-8, engine="pwl")
    for k in range(3):
        L = 1.0 / slope(zero, f"W{k}", k0=50, port=1)
        assert abs(L - L0_3) / L0_3 < 1e-6


def test_three_limb_flux_is_conserved_at_the_yoke_every_step():
    res = p.simulate(three_limb((1.0, 0.3, -0.8)), t_end=2e-4, dt=1e-6, engine="pwl")
    fluxes = [res.flux(f"L{k}") for k in range(3)]
    net = fluxes[0] + fluxes[1] + fluxes[2] - res.flux("Air")
    biggest = max(np.max(np.abs(f)) for f in fluxes)
    # The design asks 1e-12; that is the rounding floor (each step adds a
    # KCL residual of ~eps to the yoke's net flux: 1.1e-12 after 200
    # steps here), so the bound carries a 10x margin.
    assert np.max(np.abs(net)) <= 1e-11 * biggest


# ---- EE integrated coupled inductor ---------------------------------------

def ee_core(drive_second=False, *, P_outer=(2e-6, 2e-6), P_centre=0.5e-6, N=10):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "e1", "gnd", 1.0)
    b.add_winding("W1", "e1", "gnd", "x1", "gnd", N)
    b.add_winding("W2", "e2", "gnd", "x2", "gnd", N)
    b.add_resistor("Rload2", "e2", "gnd", 1e12)   # winding 2 open
    b.add_permeance("Po1", "x1", "T", P_outer[0])
    b.add_permeance("Po2", "x2", "T", P_outer[1])
    b.add_permeance("Pctr", "T", "gnd", P_centre)
    return b


def test_ee_coupled_inductor_self_and_mutual():
    res = p.simulate(ee_core(), t_end=2e-5, dt=1e-8, engine="pwl")
    L = 1.0 / slope(res, "W1", k0=200, port=1)
    assert L == pytest.approx(111.1e-6, rel=1e-3)
    assert abs(L - 100 * 2e-6 * 2.5e-6 / 4.5e-6) / L < 1e-6
    # Open winding 2: v2 = M·di1/dt, so v2/v1 = M/L = k (equal windings).
    k = res.v("e2")[-1] / res.v("e1")[-1]
    assert abs(abs(k) - 0.8) < 1e-6
    # Same-facing magnetic ports in parallel legs: opposite dots.
    assert k < 0


# ---- eddy loss (magnetic resistance) --------------------------------------

def eddy(R_m=0.01, *, electrical=False):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "e", "gnd", 0.0, 100.0, 1e3)
    if electrical:
        b.add_inductor("L", "e", "gnd", 100 * 1e-6)
        b.add_resistor("Req", "e", "gnd", 100 / R_m)
        return b
    b.add_winding("W", "e", "gnd", "m1", "gnd", 10)
    b.add_magnetic_resistance("Rm", "m1", "m2", R_m)
    b.add_permeance("P", "m2", "gnd", 1e-6)
    return b


def test_eddy_resistance_loss_under_sinusoidal_flux():
    kw = dict(t_end=5e-3, dt=5e-6, engine="pwl")
    res = p.simulate(eddy(), **kw)
    ref = p.simulate(eddy(electrical=True), **kw)
    t = np.asarray(res.times)
    F = res.v("m1") - res.v("m2")
    loss = F ** 2 / 0.01
    P = p.time_average(t, loss, t_from=1e-3, t_to=5e-3)
    assert abs(P - 0.500) / 0.5 < 1e-4
    # The source supplies it (the permeance stores, on average, nothing).
    p_in = res.v("e") * res.i("W", port=1)
    assert abs(p.time_average(t, p_in, t_from=1e-3, t_to=5e-3) - P) / P < 1e-4
    # It is a 10 kOhm resistor across the winding.
    assert rel(res.i("W", port=1), -ref.i("V")) < 1e-9
    # Reported as core loss.
    rows = p.device_loss_summary(eddy(), res, t_from=1e-3, t_to=5e-3)
    row = next(r for r in rows if r["name"] == "Rm")
    assert row["kind"] == "magnetic_resistance" and row["category"] == "core_loss"
    # The summary integrates the piecewise-linear F exactly, (wh)^2/6 =
    # 1.6e-4 below the sine's mean square at 200 samples per period.
    assert abs(row["P_avg"] - 0.5) / 0.5 < 3e-4


def test_energy_balance_with_loss_is_exact_on_the_grid():
    res = p.simulate(eddy(), t_end=2e-3, dt=5e-6, engine="pwl")
    h = np.diff(res.times)
    i = res.i("W", port=1)
    E_in = np.cumsum(h * mid(res.v("e")) * mid(i))
    F_R = res.v("m1") - res.v("m2")
    E_loss = np.cumsum(h * mid(F_R) ** 2 / 0.01)
    E_store = 0.5 * 1e-6 * res.v("m2") ** 2
    resid = E_in - E_loss - (E_store[1:] - E_store[0])
    assert np.max(np.abs(resid)) <= 1e-9 * np.max(np.abs(E_in))


# ---- DC operating point and a consistent start -----------------------------

def test_dc_bias_then_transient_starts_without_a_flux_jump():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 2.0)
    b.add_resistor("R", "in", "e", 1.0)
    b.add_winding("W", "e", "gnd", "m1", "gnd", N_G)
    b.add_permeance("Pc", "m1", "m2", P_CORE)
    b.add_air_gap("G", "m2", "gnd", A_G, GAP)
    x = p.compute_dc_op(b)
    F = 2.0 * N_G
    F_gap = x[b.node_id_of("m2")]
    assert abs(F_gap - F * (1 / P_GAP) / (1 / P_CORE + 1 / P_GAP)) / F < 1e-9
    res = p.simulate(b, t_end=1e-5, dt=1e-7, engine="pwl", start_from_dc_op=True)
    assert rel(res.v("m2"), np.full(len(res.times), F_gap)) < 1e-9
    assert rel(res.v("m1"), np.full(len(res.times), F)) < 1e-9


def test_dc_bias_with_an_internal_magnetic_resistance():
    # m2 and m3 are joined only by R_m: at DC it carries no flux rate, so
    # F2 = F3 and the core and gap carry one flux (they share a flux row).
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 2.0)
    b.add_resistor("R", "in", "e", 1.0)
    b.add_winding("W", "e", "gnd", "m1", "gnd", N_G)
    b.add_permeance("Pc", "m1", "m2", P_CORE)
    b.add_magnetic_resistance("Rm", "m2", "m3", 0.5)
    b.add_air_gap("G", "m3", "gnd", A_G, GAP)
    F_gap = 2.0 * N_G * P_CORE / (P_CORE + P_GAP)
    res = p.simulate(b, t_end=1e-5, dt=1e-7, engine="pwl", start_from_dc_op=True)
    for node in ("m2", "m3"):
        assert rel(res.v(node), np.full(len(res.times), F_gap)) < 1e-9
    assert rel(res.flux("Pc"), res.flux("G")) < 1e-9



def test_dc_bias_with_a_floating_winding_port():
    # The winding sits between a and c, neither on gnd; its MMF divides so
    # that the flux through Pa returns through Pc. The transient from the
    # DC point stays there.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 2.0)
    b.add_resistor("R", "in", "e", 1.0)
    b.add_winding("W", "e", "gnd", "a", "c", 10)
    b.add_permeance("Pa", "a", "gnd", 1e-6)
    b.add_permeance("Pc", "c", "gnd", 3e-6)
    res = p.simulate(b, t_end=1e-5, dt=1e-7, engine="pwl", start_from_dc_op=True)
    n = len(res.times)
    assert rel(res.v("a"), np.full(n, 15.0)) < 1e-9
    assert rel(res.v("c"), np.full(n, -5.0), scale=15.0) < 1e-9

# ---- reference rule ----------------------------------------------------------

def test_island_without_reference_is_refused_by_name():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "e", "gnd", 1.0)
    b.add_winding("W", "e", "gnd", "m1", "m0", 10)
    b.add_permeance("P", "m1", "m0", 1e-6)
    with pytest.raises(ValueError, match="'m1'"):
        p.simulate(b, t_end=1e-6, dt=1e-8, engine="pwl")


# ---------------------------------------------------------------------------
# 3. Saturation
# ---------------------------------------------------------------------------

NS_SAT = 10
L_AIR = N_G ** 2 * MU0 * A_G / (L_G + GAP)              # 0.823 µH


def _ramp(b, V):
    # 0 V at t = 0 (consistent with the zero initial state: the fixed
    # engine is first order from an inconsistent start), then a 0.1 µs
    # ramp to V, both corners on the grid.
    b.add_pulse_voltage_source("V", "p", "gnd", 0.0, V, 1e-6, 1.0, 0.0, 1e-7, 1e-7)


def sat_permeance(V=10.0, load=10.0):
    # Both windings in series on one flux path (no leakage): the
    # permeance form of add_saturable_transformer with zero leakage.
    b = p.CircuitBuilder()
    _ramp(b, V)
    b.add_winding("Wp", "p", "gnd", "a", "gnd", N_G)
    b.add_winding("Ws", "s", "gnd", "b", "a", NS_SAT)
    b.add_saturable_permeance("Core", "b", "c", A_G, L_G, mu_r=MUR, B_sat=BSAT)
    b.add_air_gap("Gap", "c", "gnd", A_G, GAP)
    b.add_resistor("RL", "s", "gnd", load)
    return b


def sat_tmodel(V=10.0, load=10.0):
    b = p.CircuitBuilder()
    _ramp(b, V)
    b.add_saturable_transformer("T", "p", "gnd", "s", "gnd", N_G, NS_SAT,
                                A_G, L_G, GAP, MUR, BSAT)
    b.add_resistor("RL", "s", "gnd", load)
    return b


def test_saturating_transformer_matches_the_t_model_to_three_times_the_knee():
    kw = dict(t_end=8e-5, dt=1e-8, engine="pwl")
    a = p.simulate(sat_permeance(), **kw)
    t = p.simulate(sat_tmodel(), **kw)
    i_m = np.asarray(t.i("T.m"))
    knee = 6.2   # GappedCore: H = 2·H0 at ~6.2 A for this core
    assert np.max(np.abs(i_m)) > 3 * knee
    # The primary current carries the magnetising current, so it tests
    # the core law. (The load current does not: with no leakage it is
    # (N_s/N_p)·v_p/R_L whatever the core does.)
    assert rel(a.i("V"), t.i("V")) < 1e-3
    # The flux through core and gap is one flux.
    assert rel(a.flux("Core"), a.flux("Gap")) < 1e-9


def test_saturated_inductance_is_the_air_value():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "e", "gnd", 10.0)
    b.add_winding("W", "e", "gnd", "m1", "gnd", N_G)
    b.add_saturable_permeance("Core", "m1", "m2", A_G, L_G, mu_r=MUR, B_sat=BSAT)
    b.add_air_gap("Gap", "m2", "gnd", A_G, GAP)
    res = p.simulate(b, t_end=8e-5, dt=1e-8, engine="pwl")
    assert L_AIR == pytest.approx(0.823e-6, rel=1e-3)
    assert res.i("W", port=1)[-1] > 3 * 6.2
    L = 10.0 / slope(res, "W", k0=-20, port=1)
    assert abs(L - L_AIR) / L_AIR < 0.01
    # B saturates at B_sat (+ mu0·H) and H = F/l.
    B = res.b_field("Core")[-1]
    H = res.h_field("Core")[-1]
    assert B == pytest.approx(MU0 * H + BSAT * math.tanh(H * (MUR - 1) * MU0 / BSAT), rel=1e-12)


def test_saturating_transformer_pwl_and_trbdf2_agree():
    a = p.simulate(sat_permeance(), t_end=8e-5, dt=1e-8, engine="pwl")
    b = p.simulate(sat_permeance(), t_end=8e-5, engine="trbdf2", rtol=1e-8, atol=1e-10)
    # At trbdf2's own samples (they lie on the pwl grid): interpolating
    # its coarser grid across the saturation knee would measure the
    # interpolation, not the engines. Design tolerance 1e-4; measured 2e-10
    # (the winding voltage is constant, so the flux is exact on both: this
    # checks the F(Phi) inversion across the knee; the integrators meet a
    # saturating core in the three-leg RK4 case below).
    for name in ("V", "RL"):
        assert rel(b.i(name), np.interp(b.times, a.times, a.i(name))) < 1e-4


# The saturating three-leg core against an independent RK4 solution
# (openspec/changes/add-permeance-magnetic-domain/probes/
# saturating_leg_reference.py; RK4 at 1e-7 and 5e-8 s differ by 3e-14 of
# the span). Leg b saturates first; leg c then carries the flux.
SAT3_REF = [  # t, i [A], Phi_b [Wb], Phi_c [Wb]
    (1.000e-03, 9.497051115512021e-01, 1.505494207386107e-05, 3.497721604727587e-05),
    (2.000e-03, 2.956308900405485e+00, 1.518098112058885e-05, 1.152161597921016e-04),
    (3.000e-03, 4.193852617861187e+00, 1.525871386515202e-05, 1.647023619414171e-04),
    (4.000e-03, 3.962145093488119e+00, 1.524415982431176e-05, 1.554369717746624e-04),
    (5.000e-03, 2.265896199972078e+00, 1.513761483515669e-05, 8.760832503185175e-05),
    (6.000e-03, -9.014619719900776e-04, -8.747620599246042e-08, -1.856323768111102e-08),
    (7.000e-03, -2.487858735016971e+00, -1.515155677255726e-05, -9.648403804616737e-05),
    (8.000e-03, -4.021303076473567e+00, -1.524787566264169e-05, -1.578025479264143e-04),
    (9.000e-03, -3.898587971657697e+00, -1.524016766674454e-05, -1.528954853329590e-04),
    (1.000e-02, -2.242485487298605e+00, -1.513614435866150e-05, -8.667219062021190e-05),
]


def sat_three_leg():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 100.0)
    b.add_resistor("R", "in", "e", 2.0)
    b.add_winding("W", "e", "gnd", "x", "gnd", 50)
    b.add_permeance("Pa", "x", "T", 4e-6)
    b.add_saturable_permeance("Pb", "T", "gnd", 50e-6, 40e-3, mu_r=3000.0, B_sat=0.3)
    b.add_permeance("Pc", "T", "gnd", 1e-6)
    return b


@pytest.mark.parametrize("engine,kw,tol", [
    ("pwl", dict(dt=1e-6), 2e-6),
    ("trbdf2", dict(rtol=1e-10, atol=1e-13), 2e-6),
])
def test_saturating_three_leg_core_against_independent_rk4(engine, kw, tol):
    res = p.simulate(sat_three_leg(), t_end=1e-2, engine=engine, **kw)
    t = np.asarray(res.times)
    ref = np.array(SAT3_REF)
    got = np.column_stack([np.interp(ref[:, 0], t, y) for y in
                           (res.i("W", port=1), res.flux("Pb"), res.flux("Pc"))])
    for j in range(3):
        span = np.max(np.abs(ref[:, j + 1]))
        assert np.max(np.abs(got[:, j] - ref[:, j + 1])) <= tol * span, j


# ---- energy balance on the remaining cases --------------------------------

def test_transformer_energy_balance_is_exact_on_the_grid():
    res = p.simulate(transformer_permeance(), t_end=3e-4, dt=2e-7, engine="pwl")
    h = np.diff(res.times)
    E_in = np.cumsum(h * mid(res.v("p")) * mid(res.i("Wp", port=1)))
    E_load = np.cumsum(h * mid(res.v("s")) ** 2 / 10.0)
    E_store = sum(0.5 * P * (res.v(a) - res.v(b)) ** 2
                  for P, a, b in ((PSP, "a", "gnd"), (PC, "a", "b"), (PSS, "b", "gnd")))
    resid = E_in - E_load - (E_store[1:] - E_store[0])
    assert np.max(np.abs(resid)) <= 1e-9 * np.max(np.abs(E_in))


def _tanh_coenergy_integral(law_max, law_min, v0, F):
    # ∫0^F Φ dF' for Φ = Cmin·F + (Cmax − Cmin)·v0·tanh(F/v0)
    x = np.abs(F) / v0
    log_cosh = x + np.log1p(np.exp(-2 * x)) - math.log(2)
    return 0.5 * law_min * F ** 2 + (law_max - law_min) * v0 ** 2 * log_cosh


def test_saturating_three_leg_energy_balance_to_integration_accuracy():
    # Charge-form trapezoid: the stored energy of a NONLINEAR permeance,
    # W(Φ) = ∫F dΦ, is matched to the integration error, not to rounding.
    res = p.simulate(sat_three_leg(), t_end=1e-2, dt=1e-6, engine="pwl")
    h = np.diff(res.times)
    i = res.i("W", port=1)
    E_in = np.cumsum(h * mid(res.v("in")) * mid(-res.i("V")))
    E_R = np.cumsum(h * mid(res.v("in") - res.v("e")) ** 2 / 2.0)
    Fa = res.v("x") - res.v("T")
    FT = res.v("T")
    phi_b = res.flux("Pb")
    p_air = MU0 * 50e-6 / 40e-3
    v0 = 40e-3 * (0.3 / MU0) / 2999.0
    W_b = FT * phi_b - _tanh_coenergy_integral(3000 * p_air, p_air, v0, FT)
    E_store = 0.5 * 4e-6 * Fa ** 2 + 0.5 * 1e-6 * FT ** 2 + W_b
    resid = E_in - E_R - (E_store[1:] - E_store[0])
    assert np.max(np.abs(resid)) <= 1e-6 * np.max(np.abs(E_in))
    assert rel(-res.i("V"), i) < 1e-10


# ---------------------------------------------------------------------------
# 5. Surface: YAML, the IR round trip, outputs
# ---------------------------------------------------------------------------

YAML_MAG = """
circuit:
  devices:
    - {type: voltage_source, name: V, from: e, to: gnd, V: 1.0}
    - {type: winding, name: W, e_pos: e, e_neg: gnd, m_pos: m1, m_neg: gnd, turns: 25}
    - {type: magnetic_resistance, name: Rm, from: m1, to: m2, R_m: 0.5}
    - {type: saturable_permeance, name: Core, from: m2, to: m3, area: 7.6e-5, length: 0.072, mu_r: 2000, B_sat: 0.35}
    - {type: permeance, name: Leak, from: m1, to: gnd, P: 1.0e-9}
    - {type: air_gap, name: G, from: m3, to: gnd, area: 7.6e-5, length: 5.0e-4}
"""


def python_mag():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "e", "gnd", 1.0)
    b.add_winding("W", "e", "gnd", "m1", "gnd", 25)
    b.add_magnetic_resistance("Rm", "m1", "m2", 0.5)
    b.add_saturable_permeance("Core", "m2", "m3", 7.6e-5, 0.072, mu_r=2000.0, B_sat=0.35)
    b.add_permeance("Leak", "m1", "gnd", 1e-9)
    b.add_air_gap("G", "m3", "gnd", 7.6e-5, 5e-4)
    return b


def test_yaml_types_build_the_same_circuit():
    lc = p.load_yaml_string(YAML_MAG)
    assert lc.builder.components() == python_mag().components()
    assert lc.builder.magnetic_nodes() == python_mag().magnetic_nodes() == ["m1", "m2", "m3"]
    assert lc.builder.build().value_hash() == python_mag().build().value_hash()


@pytest.mark.parametrize("bad,needle", [
    ("{type: winding, name: W, e_pos: e, e_neg: gnd, m_pos: m1, m_neg: gnd, turns: 25, N: 3}", "N"),
    ("{type: air_gap, name: G, from: m1, to: gnd, area: 1.0e-4}", "length"),
    ("{type: hysteretic_permeance, name: H, from: m1, to: gnd}", "Ms"),
    ("{type: excess_loss, name: X, from: m1, to: gnd, c: 0.01}", "eps"),
    ("{type: excess_loss, name: X, from: m1, to: gnd, c: 0.01, eps: 0.0}", "eps must be positive"),
])
def test_yaml_magnetic_types_are_strict(bad, needle):
    text = ("circuit:\n  devices:\n    - {type: voltage_source, name: V, from: e, to: gnd, V: 1.0}\n"
            f"    - {bad}\n")
    with pytest.raises(Exception, match=needle):
        p.load_yaml_string(text)


def test_ir_yaml_round_trip_keeps_the_magnetic_nodes_and_geometry():
    c = python_mag().build()
    y = c.to_yaml()
    assert "magnetic_nodes:" in y and "magnetic_geometry:" in y and "law: tanh" in y
    c2 = p.Circuit.from_yaml(y)
    assert c2.value_hash() == c.value_hash()
    assert c2.structure_hash() == c.structure_hash()
    assert c2.to_yaml() == y
    # The round-tripped circuit runs and answers the same.
    a = p.simulate(c, t_end=2e-5, dt=1e-8, engine="pwl")
    b = p.simulate(c2, t_end=2e-5, dt=1e-8, engine="pwl")
    assert np.array_equal(a.i("W", port=1), b.i("W", port=1))
    assert np.array_equal(a.b_field("Core"), b.b_field("Core"))
    # A segment with a non-positive area is refused on read, by name.
    bad = re.sub(r'(device: "G", area: )[^,]+', r"\g<1>-1", y)
    assert bad != y
    with pytest.raises(Exception, match="area and length"):
        p.Circuit.from_yaml(bad)


def test_magnetic_element_descriptions_and_output_refusals():
    b = python_mag()
    assert b.magnetic_element("W")["turns"] == pytest.approx(25.0, rel=1e-15)
    assert b.magnetic_element("Rm")["R_m"] == pytest.approx(0.5, rel=1e-15)
    g = b.magnetic_element("G")
    assert g["kind"] == "permeance" and g["P"] == pytest.approx(MU0 * 7.6e-5 / 5e-4, rel=1e-15)
    assert (g["area"], g["length"]) == (7.6e-5, 5e-4)
    core = b.magnetic_element("Core")
    assert core["kind"] == "saturable_permeance" and core["law"].kind == "tanh"
    assert b.magnetic_element("Leak")["area"] is None
    with pytest.raises(KeyError):
        b.magnetic_element("V")
    res = p.simulate(b, t_end=1e-6, dt=1e-8, engine="pwl")
    with pytest.raises(ValueError, match="no area"):
        res.b_field("Leak")
    with pytest.raises(ValueError, match="winding"):
        res.flux("W")


def test_saturable_permeance_needs_its_parameters():
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match="mu_r and B_sat"):
        b.add_saturable_permeance("S", "m", "gnd", 1e-4, 0.1)
    with pytest.raises(ValueError, match="not both"):
        b.add_saturable_permeance(
            "S", "m", "gnd", 1e-4, 0.1, mu_r=2000.0,
            table=p.ChargeTable.from_capacitance([-1.0, 0.0, 1.0], [1e-9, 2e-9, 1e-9]))
    assert p.ChargeLaw.tanh(2e-9, 1e-12, 1.0).capacitance(0.0) == pytest.approx(2e-9, rel=1e-15)


def test_saturable_permeance_from_a_table_follows_its_law():
    # Knots (F, Phi, P) sampled from the tanh law of a segment: the cubic
    # Hermite table matches the closed form between knots, so the two
    # runs agree to the interpolation error; flux reads through the table.
    area, length = 7.6e-5, 7.2e-2
    def build(table=None):
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "e", "gnd", 0.0, 6.0, 1e3)
        b.add_resistor("R", "e", "w", 1.0)
        b.add_winding("W", "w", "gnd", "m", "gnd", 25)
        if table is None:
            b.add_saturable_permeance("Core", "m", "gnd", area, length,
                                      mu_r=2000.0, B_sat=0.35)
        else:
            b.add_saturable_permeance("Core", "m", "gnd", table=table)
        return b

    law = build().magnetic_element("Core")["law"]
    F = np.linspace(-2000.0, 2000.0, 4001)
    table = p.ChargeTable(F, [law.charge(f) for f in F], [law.capacitance(f) for f in F])
    kw = dict(t_end=1e-3, dt=1e-6, engine="pwl")
    a = p.simulate(build(), **kw)
    t = p.simulate(build(table), **kw)
    h0 = (0.35 / MU0) / 1999.0
    assert np.max(np.abs(a.v("m"))) > 5 * (2 * h0 * length)  # 5x the knee MMF
    assert rel(t.i("R"), a.i("R")) < 1e-4
    assert rel(t.flux("Core"), [table.charge(f) for f in t.v("m")]) < 1e-15
    assert build(table).magnetic_element("Core")["area"] is None


def _buck(magnetic):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_mosfet("Q1", "vin", "sw")
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    if magnetic:
        b.add_winding("W", "sw", "vout", "m1", "gnd", N_G)
        b.add_permeance("Pc", "m1", "m2", P_CORE)
        b.add_air_gap("G", "m2", "gnd", A_G, GAP)
    else:
        b.add_inductor("L", "sw", "vout", L_GAPPED)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    return b


@pytest.mark.filterwarnings("ignore:simulate\\(engine='trbdf2'\\)")
@pytest.mark.parametrize("engine,kw,tol", [
    ("pwl", dict(dt=1e-7), 1e-9),
    ("trbdf2", dict(rtol=1e-8, atol=1e-10), 1e-4),
])
def test_gapped_core_in_a_switching_buck_equals_its_inductor(engine, kw, tol):
    # The magnetic nodes beside a switching stage (scaling risk of the
    # design): same answer as the electrical inductor L = N²/(R_c + R_g).
    # Measured 3.8e-12 on pwl (same grid), 2.8e-6 on trbdf2 (own grids).
    runs = []
    for magnetic in (True, False):
        b = _buck(magnetic)
        pwm = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
        runs.append(p.simulate(b, t_end=1e-3, engine=engine, switch_fn=pwm, **kw))
    tt = np.linspace(1e-4, 1e-3, 500)
    a, c = (np.interp(tt, r.times, r.v("vout")) for r in runs)
    assert rel(a, c) < tol


@pytest.mark.parametrize("case", ["three_limb", "ee"])
def test_energy_balance_of_the_multi_leg_cases_is_exact_on_the_grid(case):
    if case == "three_limb":
        b = three_limb((1.0, 0.3, -0.8))
        ports = [(f"e{k}", f"W{k}") for k in range(3)]
        perms = [(P_LIMB, f"x{k}", "T") for k in range(3)] + [(P_AIR, "T", "gnd")]
    else:
        b = ee_core()
        ports = [("e1", "W1"), ("e2", "W2")]
        perms = [(2e-6, "x1", "T"), (2e-6, "x2", "T"), (0.5e-6, "T", "gnd")]
    res = p.simulate(b, t_end=2e-5, dt=1e-8, engine="pwl")
    h = np.diff(res.times)

    def v(n):
        return np.zeros(len(res.times)) if n == "gnd" else np.asarray(res.v(n))

    # Energy into the magnetic network through every winding port (the
    # EE's "open" winding 2 feeds its 1 TOhm load: a negative term).
    E_in = sum(np.cumsum(h * mid(v(e)) * mid(res.i(w, port=1))) for e, w in ports)
    E_store = sum(0.5 * P * (v(a) - v(c)) ** 2 for P, a, c in perms)
    resid = E_in - (E_store[1:] - E_store[0])
    # Measured 1.0e-10 (three-limb: a per-step residual on the yoke that
    # accumulates over 2000 steps, as in the flux check) and 8.6e-14 (EE).
    assert np.max(np.abs(resid)) <= 1e-8 * np.max(np.abs(E_store))


# ---------------------------------------------------------------------
# 4.1 Hysteretic permeance: the Jiles-Atherton law as a core segment.
# Reference: add_hysteretic_core_inductor with the same core and N turns
# (an independent device: the JA inductor seen from the winding), on the
# 50 Hz drive of test_hysteretic_core_in_loop.py (B_pk 0.32 T, steady
# cycle 60-80 ms, source started at its peak).
# ---------------------------------------------------------------------

N_H, A_H, L_H = 100, 1e-4, 0.05


def _ja_run(perm, engine="pwl", t_end=0.08):
    ja = p.reference_material("ferrite_n87")
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 1.0, 50.0, math.pi / 2)
    b.add_resistor("Rs", "a", "n1", 0.1)
    if perm:
        b.add_winding("W", "n1", "gnd", "m1", "gnd", N_H)
        b.add_hysteretic_permeance("Core", "m1", "gnd", A_H, L_H, ja.Ms,
                                   ja.a, ja.alpha, ja.c, ja.k)
    else:
        b.add_hysteretic_core_inductor("Core", "n1", "gnd", float(N_H), A_H,
                                       L_H, 0.0, ja.Ms, ja.a, ja.alpha, ja.c,
                                       ja.k)
    r = p.simulate(b, t_end=t_end, engine=engine, dt=5e-6)
    i = r.i("W", port=1) if perm else r.i("Core")
    return np.asarray(r.times), np.asarray(i), r


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_hysteretic_permeance_equals_the_ja_inductor(engine):
    t_ref, i_ref, _ = _ja_run(False, engine)
    t, i, _ = _ja_run(True, engine)
    assert np.max(np.abs(i_ref)) > 0.05           # the core is driven
    assert rel(np.interp(t_ref, t, i), i_ref) < 1e-6


def test_hysteretic_permeance_loop_area_is_the_absorbed_energy():
    # ∮F·dΦ of the segment (F = res.i(name), dΦ/dt = v(name.e)) over the
    # steady cycle equals the electrical energy the JA inductor absorbs,
    # ∮v·i dt; the loop is open (hysteresis loss > 0), and the replay of the
    # B-H trajectory (pulsim.compute_bh_loop, an independent integration of
    # the JA ODE) agrees within its own 5 % (test_hysteretic_core_in_loop).
    t_ref, i_ref, r_ref = _ja_run(False)
    t, _, r = _ja_run(True)
    m = (t_ref >= 0.06) & (t_ref <= 0.08)
    W_el = float(np.trapezoid(np.asarray(r_ref.v("n1"))[m] * i_ref[m], t_ref[m]))
    m = (t >= 0.06) & (t <= 0.08)
    F = np.asarray(r.i("Core"))
    W_loop = float(np.trapezoid(F[m] * np.asarray(r.v("Core.e"))[m], t[m]))
    assert W_el > 0.0
    assert abs(W_loop - W_el) < 1e-6 * W_el
    ja = p.reference_material("ferrite_n87")
    loop = p.compute_bh_loop(F / N_H, t, params=ja, N_turns=N_H, l_m=L_H,
                             A_core=A_H, period=0.02)
    W_bh = float(loop.energy_per_cycle_per_m3) * A_H * L_H
    assert abs(W_bh - W_el) < 0.05 * W_el


def test_hysteretic_permeance_surfaces():
    ja = p.reference_material("ferrite_n87")
    b = p.CircuitBuilder()
    b.add_hysteretic_permeance("Core", "m1", "gnd", A_H, L_H, ja.Ms, ja.a,
                               ja.alpha, ja.c, ja.k)
    assert b.magnetic_nodes() == ["m1"]
    names = {d["name"]: d["kind"] for d in b.components()}
    assert names["Core.w"] == "gyrator" and names["Core"] == "saturable_inductor"
    with pytest.raises(KeyError, match="hysteretic permeance"):
        b.magnetic_element("Core")
    with pytest.raises(ValueError, match="add_hysteretic_permeance.*area"):
        p.CircuitBuilder().add_hysteretic_permeance(
            "X", "m1", "gnd", 0.0, L_H, ja.Ms, ja.a, ja.alpha, ja.c, ja.k)
    # the exact engine refuses the JA kind by name
    b.add_winding("W", "e", "gnd", "m1", "gnd", N_H)
    b.add_sine_voltage_source("V", "e", "gnd", 0.0, 1.0, 50.0)
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, t_end=1e-3, engine="dsed", dt=1e-5)
    assert ei.value.device == "Core"
    # YAML builds the same circuit
    y = ("circuit:\n  devices:\n"
         f"    - {{type: hysteretic_permeance, name: Core, from: m1, to: gnd, "
         f"area: {A_H}, length: {L_H}, Ms: {ja.Ms}, a: {ja.a}, "
         f"alpha: {ja.alpha}, c: {ja.c}, k: {ja.k}}}\n")
    lc = p.load_yaml_string(y)
    b2 = p.CircuitBuilder()
    b2.add_hysteretic_permeance("Core", "m1", "gnd", A_H, L_H, ja.Ms, ja.a,
                                ja.alpha, ja.c, ja.k)
    assert lc.builder.components() == b2.components()
    assert lc.builder.build().value_hash() == b2.build().value_hash()


# ---------------------------------------------------------------------
# 4.4 fit_core_loss_elements: round trip on loss data generated from known
# coefficients (a synthetic illustration, not a material), and the fitted
# R_m reproducing its own eddy share in a run (closed form 2π²·k_inst·f²B²).
# ---------------------------------------------------------------------

def _loss_data(k_h=40.0, beta=2.3, k_e=0.6, k_x=1.8):
    f = np.repeat([25e3, 50e3, 100e3, 200e3], 3)
    B = np.tile([0.05, 0.1, 0.2], 4)
    P = k_h * f * B ** beta + k_e * f ** 2 * B ** 2 + k_x * f ** 1.5 * B ** 1.5
    return f, B, P


def test_fit_core_loss_elements_round_trip():
    f, B, P = _loss_data()
    A, seg_l = 76e-6, 72e-3
    fit = p.fit_core_loss_elements(f, B, P, geometry=(A, seg_l))
    assert fit.beta == pytest.approx(2.3, rel=1e-6)
    assert fit.k_h == pytest.approx(40.0, rel=1e-6)
    assert fit.R_m == pytest.approx(0.6 * seg_l / (2 * math.pi ** 2 * A), rel=1e-6)
    # mean|cos|^1.5 = Gamma(5/4)/(sqrt(pi)*Gamma(7/4)) = 0.5564178944 (the
    # closeout's 0.5563852309 was 5.9e-5 off; test_excess_loss.py runs it)
    K15 = math.gamma(1.25) / (math.sqrt(math.pi) * math.gamma(1.75))
    assert K15 == pytest.approx(0.5564178944493822, rel=1e-10)   # math.gamma: libm
    want_c = 1.8 * seg_l / ((2 * math.pi) ** 1.5 * K15 * math.sqrt(A))
    assert fit.c == pytest.approx(want_c, rel=1e-6)
    assert fit.residual < 1e-6
    assert len(fit.fitted_on) == f.size          # the points are reported fitted
    with pytest.raises(ValueError, match="at least four"):
        p.fit_core_loss_elements(f[:3], B[:3], P[:3], geometry=(A, seg_l))
    with pytest.raises(ValueError, match="misses the data"):
        p.fit_core_loss_elements(f, B, P * (1 + 0.5 * np.sin(np.arange(f.size))),
                                 geometry=(A, seg_l))


def test_fitted_magnetic_resistance_dissipates_its_eddy_share():
    # one winding (N = 10) on a segment of the fitted R_m in series with a
    # permeance; a 100 V, 50 kHz sine gives B_pk = V/(ωNA) and the
    # resistance's in-loop loss must equal k_e·f²·B²·volume
    f, B, P = _loss_data()
    A, seg_l = 76e-6, 72e-3
    fit = p.fit_core_loss_elements(f, B, P, geometry=(A, seg_l))
    fr, V, N = 50e3, 100.0, 10
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "e", "gnd", 0.0, V, fr)
    b.add_winding("W", "e", "gnd", "m1", "gnd", N)
    b.add_magnetic_resistance("Rm", "m1", "m2", fit.R_m)
    b.add_permeance("Pc", "m2", "gnd", 2e-6)
    r = p.simulate(b, t_end=4 / fr, dt=1 / (fr * 2000))
    # the solved loss of the element (its MMF from the run), as reported
    rows = p.device_loss_summary(b, r, t_from=2 / fr, t_to=4 / fr)
    P_sim = next(x for x in rows if x["name"] == "Rm")["P_avg"]
    B_pk = V / (2 * math.pi * fr * N * A)
    assert P_sim == pytest.approx(fit.k_e * fr ** 2 * B_pk ** 2 * A * seg_l, rel=1e-4)


def test_fit_core_loss_elements_refuses_by_argument():
    f, B, P = _loss_data()
    geo = (76e-6, 72e-3)
    with pytest.raises(p.OptionsError, match="same length") as e:
        p.fit_core_loss_elements(f[:-1], B, P, geometry=geo)
    assert e.value.option == "P_v"
    with pytest.raises(p.OptionsError, match="B must be positive") as e:
        p.fit_core_loss_elements(f, -np.asarray(B), P, geometry=geo)
    assert e.value.option == "B"
    with pytest.raises(p.OptionsError, match="geometry") as e:
        p.fit_core_loss_elements(f, B, P, geometry=(0.0, 1.0))
    # data that are not a three-term split (random scatter) miss tol
    rng = np.random.default_rng(0)
    with pytest.raises(p.OptionsError, match="misses the data") as e:
        p.fit_core_loss_elements(f, B, np.asarray(P) * rng.uniform(0.5, 2, len(P)),
                                 geometry=geo)
    assert e.value.option == "tol"
