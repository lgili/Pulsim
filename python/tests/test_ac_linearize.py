"""The kernel AC sweep linearises the Newton kinds at the DC operating point
(add-unified-device-interface task 10, ``analysis/linearize.hpp``).

Each kind is checked twice at 100 Hz, 1 kHz and 10 kHz:

* against the circuit's closed form with the device's small-signal
  parameters — from the law's own closed form where it has one (Shockley
  at ~1 A: g = I_S·e^{v/nV_T}/(nV_T) + G_min with v from an mpmath
  bisection; the saturable inductor's L(I); the junction capacitor's C(V);
  the Lauritzen and IGBT-tail charge dynamics), else from a central
  difference of two DC operating points (MOSFET / IGBT g_m, g_ds);
* against a small-signal transient: a 1e-3-relative sine on top of the bias
  on trbdf2 at a tight tolerance, amplitude and phase fitted over whole
  periods with TIME-WEIGHTED integrals (trapezoid; never a plain mean on
  an adaptive grid).
"""

from __future__ import annotations

import warnings

import mpmath as mp
import numpy as np
import pytest

import pulsim as p
from pulsim import _pulsim as k

F = np.array([100.0, 1e3, 1e4])
W = 2 * np.pi * F
VT = 0.025852
SETTLE = 2e-3   # s: start-up transients of every circuit here are < 1e-8 after it
# trbdf2's step ceiling, steps per period: its error control did not refine
# a 1e-3 sine below the default ceiling (span/1000; 1.5e-4 off at 1 µs on
# the junction-capacitor case, 1.5e-6 at 0.1 µs), so the ceiling sets it
HMAX_PER_PERIOD = 400


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mask(b):
    return p.SwitchStateMask(b.graph.num_switches)


def _sweep(b, out, src="V"):
    return p.run_mna_sweep(b, output_idx=b.node_id_of(out), freqs=F,
                           v_in_branch_id=b.branch_id_of(src)).H


def _dc(b, path=k.DevicePath.Legacy):
    return np.asarray(k.compute_dc_operating_point(
        b.graph, b.pool, _mask(b), tol_dx=1e-14, tol_res=1e-14,
        device_path=path).x)


def _node(b, x, name):
    return float(x[b.node_id_of(name)])


def _rel(H, exact):
    return float(np.max(np.abs(np.asarray(H) - exact) / np.abs(exact)))


def _transient_H(make, out, bias, f, *, periods=4, settle=None,
                 path=k.DevicePath.Legacy, amp=None):
    """H(jω) of `out` from a transient driven by bias + 1e-3·|bias|·sin(ωt):
    whole periods after `settle` periods (default: ≥ 2 ms, twenty of the
    slowest time constant of these circuits, 0.1 ms, so the start-up
    transient is below 1e-8), time-weighted Fourier integrals."""
    amp = 1e-3 * abs(bias) if amp is None else amp
    b = make(bias, amp, f)
    T = 1.0 / f
    if settle is None:
        settle = max(2, int(np.ceil(SETTLE * f)))
    t0, t1 = settle * T, (settle + periods) * T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t1, dt=T / HMAX_PER_PERIOD, engine="trbdf2", rtol=1e-10,
                       atol=1e-13, initial_state=_dc(b, path))
    return _fit(np.asarray(r.times, float), np.asarray(r.v(out), float), t0, t1, f) / amp


def _fit(t, y, t0, t1, f):
    """Amplitude and phase of the f component of y over [t0, t1] (whole
    periods): time-weighted Fourier integrals of a cubic spline through the
    accepted points (the trapezoid of the raw samples carries an
    O((ωh)²) error on trbdf2's steps, ~1e-4 here), on 400 points a period."""
    from scipy.interpolate import CubicSpline
    from scipy.integrate import simpson
    keep = np.concatenate([[True], np.diff(t) > 0])
    s = CubicSpline(t[keep], y[keep])
    n = int(round((t1 - t0) * f)) * 400
    tt = np.linspace(t0, t1, n + 1)
    yy = s(tt)
    span = t1 - t0
    yy = yy - simpson(yy, x=tt) / span
    w = 2 * np.pi * f
    a = 2 / span * simpson(yy * np.sin(w * tt), x=tt)
    c = 2 / span * simpson(yy * np.cos(w * tt), x=tt)
    return a + 1j * c


def _check(make, out, bias, exact, tol, ttol=1e-3):
    b = make(bias, 0.0, 1.0)
    H = _sweep(b, out)
    assert _rel(H, exact) < tol, (H, exact, _rel(H, exact))
    for kf, f in enumerate(F):
        Ht = _transient_H(make, out, bias, f)
        assert abs(Ht - H[kf]) / abs(H[kf]) < ttol, (f, Ht, H[kf])
    return H


def _source(b, bias, amp, f):
    b.add_sine_voltage_source("V", "in", "gnd", bias, amp, f)


# ---------------------------------------------------------------------------
# the five analytic cases of the design (models/p3_ac.py)
# ---------------------------------------------------------------------------

def test_shockley_diode_at_one_ampere():
    I_S, n, Gm, R, C = 1e-12, 1.0, 1e-12, 1.0, 1e-3

    def make(bias, amp, f):
        b = p.CircuitBuilder()
        _source(b, bias, amp, f)
        b.add_resistor("R", "in", "a", R)
        b.add_shockley_diode("D", "a", "gnd", I_S=I_S, n=n, V_T=VT, G_min=Gm)
        b.add_capacitor("C", "a", "gnd", C)
        return b

    bias = 1.7
    mp.mp.dps = 40
    g = lambda v: (bias - v) / R - (I_S * (mp.e ** (v / (n * VT)) - 1) + v * Gm)  # noqa: E731
    v0 = float(mp.findroot(g, (mp.mpf("0.5"), mp.mpf("0.9")), solver="bisect", tol=1e-30))
    I0 = (bias - v0) / R
    assert 0.9 < I0 < 1.1
    # DC point (task 10.3) against the bisection
    assert abs(_node(make(bias, 0, 1), _dc(make(bias, 0, 1)), "a") - v0) < 1e-11  # measured <= 2e-14 (Newton tol_dx); 1e-11: CI lesson, nothing <= 1e-12 through libm
    gd = I_S / (n * VT) * float(mp.e ** (v0 / (n * VT))) + Gm
    # the small-signal resistance n·V_T/(I + I_S) (G_min aside)
    assert abs(1 / (gd - Gm) - n * VT / (I0 - v0 * Gm + I_S)) < 1e-12
    exact = 1 / (1 + R * (gd + 1j * W * C))
    _check(make, "a", bias, exact, 1e-9)


def test_branch_form_switch_and_shockley_diode_in_one_sweep():
    # integration 14m: the branch-form stamp (feat/core-branch-form-coverage)
    # and the Newton linearisation at the DC point (batch 14l) together: RC
    # through a CLOSED IDEAL switch (R_on = 0, its own current row) into a
    # Shockley diode whose exponential term carries exactly 1 A at the bias
    I_S, n, Gm, R, C, I = 1e-12, 1.0, 1e-12, 1.0, 1e-3, 1.0
    v0 = n * VT * np.log1p(I / I_S)              # I_S·(e^{v0/nV_T} − 1) = I
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", v0 + R * (I + v0 * Gm))
    b.add_switch("S", "in", "m", form="branch", R_on=0.0, g_off=1e-9)
    b.add_resistor("R", "m", "a", R)
    b.add_shockley_diode("D", "a", "gnd", I_S=I_S, n=n, V_T=VT, G_min=Gm)
    b.add_capacitor("C", "a", "gnd", C)
    m = p.SwitchStateMask(b.graph.num_switches)
    m.set(b.switch_index_of("S"), True)
    H = p.run_mna_sweep(b, output_idx=b.node_id_of("a"), freqs=F,
                        v_in_branch_id=b.branch_id_of("V"), mask=m).H
    g = 1.0 / (n * VT / (I + I_S)) + Gm          # small-signal conductance
    exact = 1 / (1 + R * (g + 1j * W * C))       # 1/(1+jωRC) loaded by g
    assert _rel(H, exact) < 1e-9, (H, exact, _rel(H, exact))
    # the open switch leaves the diode alone at its zero-bias point
    m_off = p.SwitchStateMask(b.graph.num_switches)
    H_off = p.run_mna_sweep(b, output_idx=b.node_id_of("a"), freqs=F,
                            v_in_branch_id=b.branch_id_of("V"), mask=m_off).H
    assert np.max(np.abs(H_off)) < 1e-6


def test_shockley_linearisation_is_smooth_at_one_picoampere():
    # 1 pA of forward current: the sweep sees g = I_S/(nV_T)·e^{v/nV_T} + G_min
    I_S, Gm, R = 1e-12, 1e-12, 1e3

    def make(bias, amp, f):
        b = p.CircuitBuilder()
        _source(b, bias, amp, f)
        b.add_resistor("R", "in", "a", R)
        b.add_shockley_diode("D", "a", "gnd", I_S=I_S, V_T=VT, G_min=Gm)
        return b

    bias = 1e-9 + 1e-12 * R   # ~1 pA through R into the diode
    b = make(bias, 0, 1)
    v0 = _node(b, _dc(b), "a")
    gd = I_S / VT * np.exp(v0 / VT) + Gm
    H = _sweep(b, "a")
    # H ≈ 1 − R·g: compare g itself; 1/H − 1 cancels to eps/(R·g) ≈ 6e-9
    np.testing.assert_allclose(((1 / H) - 1).real / R, gd, rtol=1e-7)
    np.testing.assert_allclose(np.abs(H.imag), 0, atol=1e-15)


def test_saturable_inductor_incremental_inductance():
    L0, Isat, Lr, R, bias = 1e-3, 0.05, 1e-4, 10.0, 1.0

    def make(bias, amp, f):
        b = p.CircuitBuilder()
        _source(b, bias, amp, f)
        b.add_resistor("R", "in", "a", R)
        b.add_saturable_inductor("L", "a", "gnd", L0, Isat, Lr)
        return b

    I0 = bias / R   # DC: the inductor is a short
    b = make(bias, 0, 1)
    x = _dc(b)
    assert abs(x[b.pool.branch_var_id_for_inductor(b.branch_id_of("L"), b.graph)] - I0) < 1e-11  # measured <= 2e-14 (Newton tol_dx); 1e-11: CI lesson, nothing <= 1e-12 through libm
    L = Lr + (L0 - Lr) / (1 + (I0 / Isat) ** 2)
    exact = 1j * W * L / (R + 1j * W * L)
    _check(make, "a", bias, exact, 1e-9)


def test_saturable_inductor_at_one_picoampere():
    L0, Isat, R = 1e-3, 0.05, 10.0
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1e-11)
    b.add_resistor("R", "in", "a", R)
    b.add_saturable_inductor("L", "a", "gnd", L0, Isat, 0.0)
    exact = 1j * W * L0 / (R + 1j * W * L0)
    np.testing.assert_allclose(_sweep(b, "a"), exact, rtol=1e-9)


def test_saturable_inductor_table_law():
    # a straight-line table: L is the slope everywhere (the table path)
    L, R = 2e-4, 10.0
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.3)
    b.add_resistor("R", "in", "a", R)
    b.add_saturable_inductor_table("L", "a", "gnd", [0.0, 0.1, 1.0], [0.0, 0.1 * L, L])
    exact = 1j * W * L / (R + 1j * W * L)
    np.testing.assert_allclose(_sweep(b, "a"), exact, rtol=1e-9)


def test_nonlinear_capacitor_junction_law():
    C0, V0, m, R, bias = 1e-6, 1.0, 0.5, 100.0, 5.0

    def make(bias, amp, f):
        b = p.CircuitBuilder()
        _source(b, bias, amp, f)
        b.add_resistor("R", "in", "a", R)
        b.add_nonlinear_capacitor("C", "a", "gnd", C0, V0, m)
        return b

    b = make(bias, 0, 1)
    # open at DC (the 1e-12 S node floor against R: 5e-10 V)
    assert _node(b, _dc(b), "a") == pytest.approx(bias, abs=1e-9)
    C = C0 * (1 + bias / V0) ** (-m)
    exact = 1 / (1 + 1j * W * R * C)
    _check(make, "a", bias, exact, 1e-9)


def test_nonlinear_capacitor_vdmos_law():
    cmax, cmin, a, R, bias = 1e-6, 1e-7, 0.5, 100.0, 2.0
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", bias)
    b.add_resistor("R", "in", "o", R)
    b.add_nonlinear_capacitor("C", "o", "gnd", p.ChargeLaw.vdmos_cgd(cmax, cmin, a))
    s = (cmax - cmin) / (1 + np.pi / 2)
    C = cmax - s + s * np.tanh(a * bias)
    np.testing.assert_allclose(_sweep(b, "o"), 1 / (1 + 1j * W * R * C), rtol=1e-9)


def test_nonlinear_capacitor_table_law():
    # the table's C(v) at the bias, between knots and at a knot (where it is
    # the datum itself)
    tab = p.ChargeTable.from_capacitance([-1.0, 0.0, 5.0, 10.0, 20.0],
                                         [1.2e-6, 1e-6, 5e-7, 3e-7, 2e-7])
    R = 100.0
    for bias in (3.0, 5.0):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", bias)
        b.add_resistor("R", "in", "o", R)
        b.add_nonlinear_capacitor("C", "o", "gnd", p.ChargeLaw.table(tab))
        v0 = _node(b, _dc(b), "o")
        C = tab.capacitance(v0)
        if bias == 5.0:
            assert C == pytest.approx(5e-7, rel=1e-8)
        np.testing.assert_allclose(_sweep(b, "o"), 1 / (1 + 1j * W * R * C), rtol=1e-9)


def test_coupled_transformer_k099_against_the_transient():
    Rs, RL, Lp, Ls, kc = 10.0, 10.0, 1e-3, 1e-3, 0.99
    M = kc * np.sqrt(Lp * Ls)

    def make(bias, amp, f):
        b = p.CircuitBuilder()
        _source(b, bias, amp, f)
        b.add_resistor("Rs", "in", "a", Rs)
        b.add_transformer("T", "a", "gnd", "s", "gnd", Lp, Ls, kc)
        b.add_resistor("RL", "s", "gnd", RL)
        return b

    exact = 1j * W * M * RL / ((Rs + 1j * W * Lp) * (RL + 1j * W * Ls) + W ** 2 * M ** 2)
    _check(make, "s", 1.0, exact, 1e-9)


def test_rc_through_a_closed_switch_against_the_transient():
    def make(bias, amp, f):
        b = p.CircuitBuilder()
        _source(b, bias, amp, f)
        b.add_resistor("R", "in", "a", 100.0)
        b.add_switch("S", "a", "o", 1e3, 1e-9)
        b.add_capacitor("C", "o", "gnd", 1e-6)
        b.add_resistor("RL", "o", "gnd", 1e6)
        return b

    b = make(1.0, 0.0, 1.0)
    m = _mask(b)
    m.set(b.switch_index_of("S"), True)
    H = p.run_mna_sweep(b, output_idx=b.node_id_of("o"), freqs=F,
                        v_in_branch_id=b.branch_id_of("V"), mask=m).H
    Z = 1.0 / (1.0 / 1e6 + 1j * W * 1e-6)
    exact = Z / (100.0 + 1.0 / 1e3 + Z)
    assert _rel(H, exact) < 1e-9
    n = b.graph.num_switches
    i = b.switch_index_of("S")

    def closed(_t):
        mm = p.SwitchStateMask(n)
        mm.set(i, True)
        return mm

    for kf, f in enumerate(F):
        amp, T = 1e-3, 1.0 / f
        ns = max(2, int(np.ceil(SETTLE * f)))
        bb = make(1.0, amp, f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x0 = k.compute_dc_operating_point(bb.graph, bb.pool, closed(0.0)).x
            r = p.simulate(bb, (ns + 4) * T, dt=T / HMAX_PER_PERIOD, engine="trbdf2",
                           rtol=1e-10, atol=1e-13,
                           initial_state=x0, switch_fn=closed)
        Ht = _fit(np.asarray(r.times, float), np.asarray(r.v("o"), float),
                  ns * T, (ns + 4) * T, f) / amp
        assert abs(Ht - H[kf]) / abs(H[kf]) < 1e-3, (f, Ht, H[kf])


# ---------------------------------------------------------------------------
# the other Newton kinds
# ---------------------------------------------------------------------------

def test_lauritzen_diode_charge_dynamics():
    I_S, n, tau, TM, Gm, R, bias = 1e-12, 1.0, 1e-4, 5e-5, 1e-12, 1.0, 1.7

    def make(bias, amp, f):
        b = p.CircuitBuilder()
        _source(b, bias, amp, f)
        b.add_resistor("R", "in", "a", R)
        b.add_lauritzen_diode("D", "a", "gnd", I_S=I_S, n=n, V_T=VT, tau=tau,
                              T_M=TM, G_min=Gm)
        return b

    # DC (task 10.3, interface path): i = τ/(τ+T_M)·i_S(v) + G_min·v
    mp.mp.dps = 40
    ratio = mp.mpf(tau) / (mp.mpf(tau) + mp.mpf(TM))
    g = lambda v: (bias - v) / R - (ratio * I_S * (mp.e ** (v / (n * VT)) - 1) + v * Gm)  # noqa: E731
    v0 = float(mp.findroot(g, (mp.mpf("0.4"), mp.mpf("0.95")), solver="bisect", tol=1e-30))
    b = make(bias, 0, 1)
    assert abs(_node(b, _dc(b, k.DevicePath.Interface), "a") - v0) < 1e-11  # measured <= 2e-14 (Newton tol_dx); 1e-11: CI lesson, nothing <= 1e-12 through libm
    # q_E = τ·i_S(v); q̇_M = q_E/T_M − A·q_M; i = (q_E − q_M)/T_M + G_min·v
    gs = I_S / (n * VT) * float(mp.e ** (v0 / (n * VT)))
    A = 1 / TM + 1 / tau
    Y = tau * gs / TM * (1 - 1 / (TM * (1j * W + A))) + Gm
    exact = 1 / (1 + R * Y)
    b = make(bias, 0.0, 1.0)
    assert _rel(_sweep(b, "a"), exact) < 1e-10
    # the transient starts from the DC node voltages with the stored charge
    # at 0 (the history's start): it settles within 2 ms (τ = 0.1 ms)
    for kf, f in enumerate(F):
        Ht = _transient_H(make, "a", bias, f, path=k.DevicePath.Interface)
        assert abs(Ht - exact[kf]) / abs(exact[kf]) < 1e-3, (f, Ht, exact[kf])


def _g_central(make, out, bias, R, dv=1e-4):
    """dI/dv of the device between `out` and ground, I = (V − v)/R, from two
    DC operating points (central difference)."""
    vs = []
    for s in (+1, -1):
        b = make(bias + s * dv, 0.0, 1.0)
        vs.append(_node(b, _dc(b), out))
    i = [(bias + s * dv - v) / R for s, v in zip((+1, -1), vs)]
    return (i[0] - i[1]) / (vs[0] - vs[1])


@pytest.mark.parametrize("tail", [False, True])
def test_igbt_static_and_tail(tail):
    R, bias, tau, kt = 10.0, 12.0, 1e-4, 0.5

    def make(bias, amp, f):
        b = p.CircuitBuilder()
        _source(b, bias, amp, f)
        b.add_voltage_source("Vg", "g", "gnd", 15.0)
        b.add_resistor("R", "in", "c", R)
        b.add_igbt_level1("Q", "c", "gnd", "g", tau_tail=tau if tail else 0.0,
                          k_tail=kt if tail else 0.0)
        return b

    gss = _g_central(make, "c", bias, R)
    assert gss > 0
    Y = gss * (1 + 1j * W * tau * (1 - kt)) / (1 + 1j * W * tau) if tail else gss + 0 * W
    exact = 1 / (1 + R * Y)
    _check(make, "c", bias, exact, 1e-6)


def _common_source(bias, amp, f, vdd=12.0):
    b = p.CircuitBuilder()
    _source(b, bias, amp, f)          # the gate drive
    b.add_voltage_source("VDD", "dd", "gnd", vdd)
    b.add_resistor("RD", "dd", "d", 1e3)
    b.add_capacitor("CL", "d", "gnd", 1e-7)
    b.add_mosfet_level1("M", "d", "gnd", "in", K=2e-3, V_T=2.0, with_body_diode=False)
    return b


def test_mosfet_level1_common_source():
    RD, CL, VDD, bias, dv = 1e3, 1e-7, 12.0, 2.6, 1e-4

    def op(vg, vdd):
        b = _common_source(vg, 0.0, 1.0, vdd)
        vd = _node(b, _dc(b), "d")
        return (vdd - vd) / RD, vd

    # g_m and g_ds from central differences of DC points
    (_, v1), (_, v2) = op(bias + dv, VDD), op(bias - dv, VDD)
    (i3, v3), (i4, v4) = op(bias, VDD + dv), op(bias, VDD - dv)
    gds = (i3 - i4) / (v3 - v4)
    # at fixed V_DD: Δi = g_m·Δv_g + g_ds·Δv_d with Δi = −Δv_d/R_D
    gm = (-(v1 - v2) / RD - gds * (v1 - v2)) / (2 * dv)
    assert gm > 0 and gds > 0
    exact = -gm / (1 / RD + gds + 1j * W * CL)
    _check(_common_source, "d", bias, exact, 1e-6)


def test_smooth_ideal_diode():
    R, C, bias = 1.0, 1e-3, 1.5

    def make(bias, amp, f):
        b = p.CircuitBuilder()
        _source(b, bias, amp, f)
        b.add_resistor("R", "in", "a", R)
        b.add_nonlinear_diode("D", "a", "gnd", k.IdealDiodeParams(0.7, 0.01, 1e-9, 20.0),
                              allow_non_passive=True)   # retired, still a kind
        b.add_capacitor("C", "a", "gnd", C)
        return b

    gd = _g_central(make, "a", bias, R)
    exact = 1 / (1 + R * (gd + 1j * W * C))
    _check(make, "a", bias, exact, 1e-6)


# ---- the PMSM: non-salient (L_d = L_q = L_0 = L, so L(θ) = L·I) ----------
PM = dict(R_s=2.0, L=2e-4, psi=0.05, pp=2.0, J=1e-6, B=1e-3, Rx=1.0)  # every τ ≤ 0.1 ms
OFF = np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])


def _pmsm(bias, amp, f, *, theta0=0.6, omega0=0.0, T_load=0.0):
    b = p.CircuitBuilder()
    _source(b, bias, amp, f)
    b.add_resistor("Rx", "in", "a", PM["Rx"])
    b.add_pmsm_mna("M", "a", "gnd", "gnd", "gnd", "w", "th", PM["R_s"], PM["L"], PM["L"],
                   PM["psi"], PM["pp"], PM["J"], B=PM["B"], T_load=T_load,
                   L_0=PM["L"], omega0=omega0, theta0=theta0)
    return b


def test_pmsm_at_rest_motional_impedance():
    # At i = 0, ω = 0 (T = 0 = T_load: an equilibrium) phases b, c shorted:
    #   v̂ = Z·î_a − K_a·ω̂,  0 = Z·î_k − K_k·ω̂,  (jωJ + B)·ω̂ = −Σ K_k·î_k,
    # K_k = pp·ψ·sin(θ_e + off_k), Z = R + jωL: Z_in = Z + K_a²/D,
    # D = jωJ + B + (K_b² + K_c²)/Z.
    theta0 = 0.6
    K = PM["pp"] * PM["psi"] * np.sin(PM["pp"] * theta0 + OFF)
    Z = PM["R_s"] + 1j * W * PM["L"]
    D = 1j * W * PM["J"] + PM["B"] + (K[1] ** 2 + K[2] ** 2) / Z
    Zin = Z + K[0] ** 2 / D
    exact = Zin / (PM["Rx"] + Zin)
    b = _pmsm(0.0, 0.0, 1.0)
    H = _sweep(b, "a")
    assert _rel(H, exact) < 1e-9, (H, exact)
    # the motional term matters: without it H at 100 Hz moves by 17 %
    assert abs(H[0] - Z[0] / (PM["Rx"] + Z[0])) / abs(H[0]) > 1e-2
    for kf, f in enumerate(F):
        Ht = _transient_H(_pmsm, "a", 0.0, f, amp=1e-3)
        assert abs(Ht - H[kf]) / abs(H[kf]) < 1e-3, (f, Ht, H[kf])


def test_pmsm_held_rotor_with_load_stiffness():
    # A DC current i_a0 at rest, balanced by T_load: the torque now varies
    # with θ, a stiffness S = ∂(Σ K_k i_k)/∂θ = pp²·ψ·cos(θ_e)·i_a0 in the
    # shaft row (θ̂ = ω̂/jω): D gains S/(jω). Probes the θ column.
    V, th0 = 1.0, 0.2
    i_a0 = V / (PM["Rx"] + PM["R_s"])
    K = PM["pp"] * PM["psi"] * np.sin(PM["pp"] * th0 + OFF)
    S = PM["pp"] ** 2 * PM["psi"] * np.cos(PM["pp"] * th0) * i_a0
    Z = PM["R_s"] + 1j * W * PM["L"]

    def h(stiff):
        D = 1j * W * PM["J"] + PM["B"] + stiff / (1j * W) + (K[1] ** 2 + K[2] ** 2) / Z
        Zin = Z + K[0] ** 2 / D
        return Zin / (PM["Rx"] + Zin)

    b = _pmsm(V, 0.0, 1.0, theta0=th0, T_load=-K[0] * i_a0)
    H = _sweep(b, "a")
    assert _rel(H, h(S)) < 1e-9, (H, h(S))
    assert abs(H[0] - h(0.0)[0]) / abs(H[0]) > 1e-6   # the stiffness is seen


def test_pmsm_balance_tolerates_a_femto_torque():
    # 1e-12 V bias: a 3e-14 N·m residual is G_min-scale, not an imbalance
    _sweep(_pmsm(1e-12, 0.0, 1.0), "a")


def test_pmsm_spinning_is_refused_by_name():
    with pytest.raises(p.UnsupportedError) as ei:
        _sweep(_pmsm(0.0, 0.0, 1.0, omega0=50.0), "a")
    assert ei.value.engine == "ac_sweep" and ei.value.device.startswith("M")
    assert "omega" in str(ei.value)


def test_pmsm_out_of_torque_balance_is_refused():
    # 1 V on phase a at θ_e ≠ 0: a current, so a torque the held DC point
    # does not balance; the rotor would accelerate from it
    with pytest.raises(p.UnsupportedError) as ei:
        _sweep(_pmsm(1.0, 0.0, 1.0), "a")
    assert "torque balance" in str(ei.value)


def test_pmsm_dc_point_at_held_speed():
    # task 10.3: the DC point holds ω0, θ0; non-salient, so
    # v_k − v_n = R·i_k + e_k, e_k = −ω0·pp·ψ·sin(θ_e + off_k)
    V, th0, w0 = 2.0, 0.6, 40.0
    b = _pmsm(V, 0.0, 1.0, theta0=th0, omega0=w0)
    e = -w0 * PM["pp"] * PM["psi"] * np.sin(PM["pp"] * th0 + OFF)
    i_a = (V - e[0]) / (PM["Rx"] + PM["R_s"])
    i_bc = -e[1:] / PM["R_s"]
    for path in (k.DevicePath.Legacy, k.DevicePath.Interface):
        x = _dc(b, path)
        got = [x[b.pool.branch_var_id_for_inductor(b.branch_id_of(f"M_{c}"), b.graph)]
               for c in "abc"]
        # measured 3e-14 of 1.9 A; rtol 1e-11: CI lesson (np.sin vs std::sin)
        np.testing.assert_allclose(got, [i_a, *i_bc], rtol=1e-11, atol=1e-14)
        assert _node(b, x, "w") == pytest.approx(w0, rel=1e-15)
        assert _node(b, x, "th") == pytest.approx(th0, rel=1e-15)


def test_jiles_atherton_inductor_is_refused_by_name():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_resistor("R", "in", "a", 1.0)
    b.add_hysteretic_core_inductor("Lh", "a", "gnd", 100.0, 1e-4, 0.1, 0.0, 4.0e5,
                                   50.0, 5e-5, 0.2, 30.0)
    with pytest.raises(p.UnsupportedError) as ei:
        _sweep(b, "a")
    assert ei.value.engine == "ac_sweep" and ei.value.device == "Lh"
    assert "hysteretic" in str(ei.value)


def test_induction_machine_stays_refused_by_name():
    caps = p.engine_capabilities()
    assert not caps["InductionMachineMna"]["ac_sweep"]
    for kind in ("NonlinearDiode", "ShockleyDiode", "MosfetLevel1", "IgbtLevel1",
                 "SaturableInductor", "NonlinearCapacitor", "LauritzenDiode",
                 "PmsmMna", "ExternalDevice"):
        assert caps[kind]["ac_sweep"], kind


def test_sweep_ignores_the_transient_integrator():
    # integration 14l: the sweep has no integrator; running the same
    # circuit's transient with integrator='esdirk43' leaves it unchanged.
    def make():
        b = p.CircuitBuilder()
        _source(b, 1.7, 0.0, 1.0)
        b.add_resistor("R", "in", "a", 1.0)
        b.add_shockley_diode("D", "a", "gnd")
        b.add_capacitor("C", "a", "gnd", 1e-3)
        return b
    b = make()
    before = _sweep(b, "a")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.simulate(b, 1e-4, engine="trbdf2", integrator="esdirk43")
    assert np.array_equal(_sweep(b, "a"), before)
    assert np.array_equal(_sweep(make(), "a"), before)
    assert np.all(np.isfinite(before))
