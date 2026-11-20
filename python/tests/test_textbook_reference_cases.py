"""Textbook reference cases (OpenSpec ``add-textbook-reference-cases``).

Each case is a circuit from a power-electronics textbook. Its truth is
computed HERE, independently of pulsim, with mpmath:

* the dc-dc converters (buck, inverting buck-boost, boost with winding
  resistance) by the exact periodic orbit of their piecewise-linear state
  equations, written by hand below with the netlist's own g_on / g_off
  (mp.expm per mode, the DCM diode turn-off by mp.findroot);
* the three-phase diode bridge on an R-L load by the closed-form periodic
  solution of its 60-degree segment ODE;
* the full-bridge bipolar sine-triangle inverter on an R-L load by the
  closed-form periodic solution between its exact (mp.findroot) gate edges.

Each run starts ON the orbit (consistent initial conditions), so the error
an engine shows is its own method's error, not a start-up transient.

Two separate assertions per case:

1. TRUTH vs TEXTBOOK (the published formula, within the 2 % the proposal
   names): proves the circuit is the textbook's and the truth is right.
   The textbook formula idealises (small ripple, ideal switches), so it is
   never held tighter than its idealisation explains.
2. ENGINE vs TRUTH, on every engine that admits the circuit, each through
   explicit options (``ENGINES``), held to a tolerance its method explains:
   - ``pwl`` (trapezoidal, fixed dt) WITHOUT restart is first order after
     each switching event (stale history, AI guide trap 9) and locates a
     diode commutation only on the dt grid: O(dt / T) relative.
   - ``pwl`` WITH ``discontinuity_restart`` is second order at gate edges;
     a diode commutation still lands on the grid: O(dt / T) at DCM / diode
     events, O((dt / T)^2) otherwise.
   - ``trbdf2`` and ``trbdf2 + esdirk43`` are error-controlled at the
     explicit ``rtol`` / ``atol`` passed (events located by the engine).
   - ``dsed`` (exact engine) solves each mode in closed form: limited only
     by event location and float64.

Citations name edition and chapter only: the books were not at hand to
verify section, equation or figure numbers, so none is quoted. Every
formula used is written out, so a reader can check it without the book.

* R. W. Erickson, D. Maksimovic, *Fundamentals of Power Electronics*,
  3rd ed., Springer 2020. Ch. 2 (buck ripple: dI_pp = (Vg - V) D Ts / L,
  dV_pp = dI_pp / (8 f C)); ch. 3 (boost with inductor winding
  resistance: V / Vg = (1 / D') / (1 + R_L / (D'^2 R))); ch. 5 (DCM:
  K = 2 L / (R Ts); buck M = 2 / (1 + sqrt(1 + 4 K / D^2)), buck-boost
  M = -D / sqrt(K), K_crit buck = 1 - D, buck-boost = (1 - D)^2).
* J. G. Kassakian, M. F. Schlecht, G. C. Verghese, *Principles of Power
  Electronics*, Addison-Wesley 1991 (2nd ed. 2023), polyphase diode
  rectifier with a DC inductor: V_d0 = (3 sqrt(3) / pi) V_m (V_m phase
  peak), line current a 120-degree block, I_rms = sqrt(2/3) I_d.
* N. Mohan, T. Undeland, W. Robbins, *Power Electronics*, 3rd ed.,
  Wiley 2003, ch. 8, single-phase full bridge with bipolar sine-triangle
  PWM: fundamental V_o1 (peak) = m_a V_d for m_a <= 1.
"""
from __future__ import annotations

import functools
import math
import os
import subprocess
import sys
import textwrap

import mpmath as mp
import numpy as np
import pytest

import pulsim as p

mp.mp.dps = 40

# Engine -> explicit simulate() options (dt filled per case for pwl).
ENGINES = {
    "pwl": dict(engine="pwl", discontinuity_restart=False,
                enable_substep_state_correction=False),
    "pwl_restart": dict(engine="pwl", discontinuity_restart=True,
                        enable_substep_state_correction=False),
    "trbdf2": dict(engine="trbdf2", rtol=1e-6, atol=1e-9),
    "esdirk43": dict(engine="trbdf2", integrator="esdirk43", rtol=1e-6, atol=1e-9),
    # output_rtol: samples dense enough that a linearly interpolated
    # window mean is good to ~1e-6 (the default grid is the 10 us cap).
    "dsed": dict(engine="dsed", output_rtol=1e-6),
}


def run(build, cfg, t_end, dt, switch_fn=None):
    opts = dict(ENGINES[cfg])
    if cfg.startswith("pwl"):
        opts["dt"] = dt
    return p.simulate(build(), t_end=t_end, switch_fn=switch_fn, **opts)


def at(res, t, y):
    return float(np.interp(t, np.asarray(res.times), np.asarray(y)))


def avg(res, y, t0, t1):
    return p.time_average(np.asarray(res.times), np.asarray(y), t_from=t0, t_to=t1)


# ------------------------------------------------------ dc-dc exact orbit --
# State [iL, vC, 1]; each topology gives A(gs, gd) and the diode current row.

def _buck(Vin, L, C, R):
    def A(gs, gd):
        s = gs + gd
        return mp.matrix([[-1 / (s * L), -1 / L, gs * Vin / (s * L)],
                          [1 / C, -1 / (R * C), 0], [0, 0, 0]])

    def iD(gs, gd):              # anode gnd, cathode sw: gd (0 - v_sw)
        s = gs + gd
        return mp.matrix([[gd / s, 0, -gd * gs * Vin / s]])
    return A, iD


def _buckboost(Vin, L, C, R):
    def A(gs, gd):               # S vin->x, L x->gnd, D out->x, C||R at out
        s = gs + gd
        k = gd * (1 - gd / s)
        return mp.matrix([[-1 / (s * L), gd / (s * L), gs * Vin / (s * L)],
                          [-gd / (s * C), -(k + 1 / R) / C, gd * gs * Vin / (s * C)],
                          [0, 0, 0]])

    def iD(gs, gd):              # gd (vC - vx)
        s = gs + gd
        return mp.matrix([[gd / s, gd * (1 - gd / s), -gd * gs * Vin / s]])
    return A, iD


def _boost(Vin, L, C, R, RL):
    def A(gs, gd):               # L+RL vin->x, S x->gnd, D x->out, C||R
        s = gs + gd
        k = gd * (1 - gd / s)
        return mp.matrix([[-(RL + 1 / s) / L, -gd / (s * L), Vin / L],
                          [gd / (s * C), -(k + 1 / R) / C, 0], [0, 0, 0]])

    def iD(gs, gd):              # gd (vx - vC)
        s = gs + gd
        return mp.matrix([[gd / s, -gd * (1 - gd / s), 0]])
    return A, iD


def dcdc_orbit(topo, gon, goff, f, D):
    """Exact periodic orbit: S on for D T, then the diode until its current
    reaches 0 (or T), then both off. Returns the states at 0, DT, DT + td,
    the period means of iL and vC, and the mode."""
    A, iD = topo
    gon, goff = mp.mpf(gon), mp.mpf(goff)
    T = 1 / mp.mpf(f)
    DT = mp.mpf(D) * T
    A1, A2, A3 = A(gon, goff), A(goff, gon), A(goff, goff)

    def aug(Am):                 # [x; integral of iL; integral of vC]
        B = mp.zeros(5, 5)
        for i in range(3):
            for j in range(3):
                B[i, j] = Am[i, j]
        B[3, 0] = 1
        B[4, 1] = 1
        return B

    def phi(Am, h):
        return mp.expm(aug(Am) * h)

    def P(td):
        return phi(A3, T - DT - td) * phi(A2, td) * phi(A1, DT)

    def x0(td):
        M = P(td)
        x = mp.lu_solve(mp.eye(2) - M[:2, :2], M[:2, 2])
        return mp.matrix([x[0], x[1], 1, 0, 0])

    def resid(td):
        y = phi(A2, td) * phi(A1, DT) * x0(td)
        return (iD(goff, gon) * y[:3, 0])[0]

    if resid(T - DT) > 0:
        td, mode = T - DT, "CCM"
    else:
        td, mode = mp.findroot(resid, (mp.mpf(0), T - DT), solver="anderson"), "DCM"
    xs = x0(td)
    x1 = phi(A1, DT) * xs
    x2 = phi(A2, td) * x1
    xe = phi(A3, T - DT - td) * x2
    assert abs(xe[0] - xs[0]) < 1e-20 and abs(xe[1] - xs[1]) < 1e-20
    # Consistency of the assumed diode states (independent of any engine):
    # conducting mid-way through its interval, reverse-biased while off.
    mid = phi(A2, td / 2) * x1
    assert (iD(goff, gon) * mid[:3, 0])[0] > 0
    if mode == "DCM":
        idle = phi(A3, (T - DT - td) / 2) * x2
        assert (iD(goff, goff) * idle[:3, 0])[0] <= 0
    return dict(T=T, DT=DT, td=td, mode=mode, iL0=xs[0], vC0=xs[1],
                iL_pk=x1[0], vC_DT=x1[1], iL_mean=xe[3] / T, vC_mean=xe[4] / T)


def check_dcdc(truth, build, gate_name, f, D, cfg, dt, n_per, tol):
    """Run n_per periods from the orbit; compare end state, the inductor peak
    in the last period and the last period's mean output voltage."""
    b = build()
    pwm = p.make_pwm_switch_fn(f, D, b.switch_index_of(gate_name), b.graph.num_switches)
    T, DT = float(truth["T"]), float(truth["DT"])
    res = run(build, cfg, n_per * T, dt, pwm)
    iL, vC = np.asarray(res.i("L")), np.asarray(res.v("out"))
    i_sc = abs(float(truth["iL_pk"]))
    v_sc = abs(float(truth["vC_mean"]))
    err = {
        "iL(NT)": abs(at(res, n_per * T, iL) - float(truth["iL0"])) / i_sc,
        "iL_pk": abs(at(res, (n_per - 1) * T + DT, iL) - float(truth["iL_pk"])) / i_sc,
        "vC(NT)": abs(at(res, n_per * T, vC) - float(truth["vC0"])) / v_sc,
        "vC_mean": abs(avg(res, vC, (n_per - 1) * T, n_per * T)
                       - float(truth["vC_mean"])) / v_sc,
    }
    return err, max(err.values()) <= tol


# --------------------------------------- three-phase bridge, R-L load ----

def bridge_truth(Vm, f, R, L, gon):
    """Periodic DC current of a six-diode bridge on R-L, continuous
    conduction, no source inductance: on each 60-degree segment the load
    sees the largest line voltage sqrt(3) Vm cos(w t) (t = 0 its centre)
    through two diodes (R' = R + 2 / g_on):
        L i' + R' i = sqrt(3) Vm cos(w t),  i(-h) = i(h),  h = T / 12.
    The diodes' leakage (g_off = 1e-9 S: 6 * 563 V * 1e-9 = 3.4 uA against
    ~50 A) is below 1e-7 relative and ignored. The two-diode overlap at a
    commutation is NOT negligible for the line current: while
    |v_a - v_b| < i / g_on both top diodes conduct, i_a = i / 2 +
    g_on (v_a - v_b) / 2 ramps over W = 2 i / (g_on sqrt(3) V_m w)
    (0.6 us here) instead of stepping, which lowers i_a^2's integral by
    i^2 W / 6 at each of phase a's four commutations a period (1.4e-5 of
    I_a,rms: the exact engine found it). It is subtracted below; it moves
    the DC side by O(i / g_on * W / T), ~1e-10, ignored."""
    Vm, f, R, L = (mp.mpf(x) for x in (Vm, f, R, L))
    w = 2 * mp.pi * f
    Rp = R + 2 / mp.mpf(gon)
    tau = L / Rp
    Z = mp.sqrt(Rp**2 + (w * L)**2)
    phi = mp.atan2(w * L, Rp)
    h = 1 / (12 * f)

    def ip(t):
        return mp.sqrt(3) * Vm / Z * mp.cos(w * t - phi)
    K = (ip(h) - ip(-h)) / (mp.exp(h / tau) - mp.exp(-h / tau))

    def i(t):                    # any t: fold onto [-h, h)
        t = mp.mpf(t)
        t = t - 2 * h * mp.floor((t + h) / (2 * h))
        return ip(t) + K * mp.exp(-t / tau)
    i_mean = mp.quad(i, [-h, h]) / (2 * h)
    i2_mean = mp.quad(lambda t: i(t)**2, [-h, h]) / (2 * h)
    assert min(i(-h), i(0), i(h)) > 0          # continuous conduction
    W = 2 * i(h) / (mp.mpf(gon) * mp.sqrt(3) * Vm * w)
    ia2 = mp.mpf(2) / 3 * i2_mean - 4 * f * i(h)**2 * W / 6
    return dict(i=i, h=h, Vd_mean=R * i_mean, i_mean=i_mean, Ia_rms=mp.sqrt(ia2))


def bridge_builder(Vm, f, R, L, gon, goff, i0):
    def build():
        b = p.CircuitBuilder()
        for ph, ang in (("a", 0.0), ("b", -2 * math.pi / 3), ("c", 2 * math.pi / 3)):
            # phase a at pi/3 puts the peak of v_ab at t = 0
            b.add_sine_voltage_source(f"V{ph}", ph, "gnd", 0.0, Vm, f, math.pi / 3 + ang)
            b.add_diode(f"D{ph}p", ph, "dp", gon, goff)
            b.add_diode(f"D{ph}n", "dn", ph, gon, goff)
        b.add_inductor("L", "dp", "m", L, i0=i0)
        b.add_resistor("R", "m", "dn", R)
        return b
    return build


# ------------------------ full bridge, bipolar sine-triangle PWM, R-L ----

def spwm_edges(m, f1, mf):
    """Natural-sampling crossings of m sin(w t) with a +-1 triangle that
    starts at -1 (t = 0): one per carrier half-period (|slope| of the
    triangle 4 mf f1 > m w), found by mp.findroot in its bracket."""
    m, f1 = mp.mpf(m), mp.mpf(f1)
    Tc = 1 / (mf * f1)
    w = 2 * mp.pi * f1
    edges = []
    for k in range(2 * mf):
        a, b = k * Tc / 2, (k + 1) * Tc / 2
        sgn = 1 if k % 2 == 0 else -1          # rising, falling
        def g(t, a=a, sgn=sgn):
            tri = -1 + 4 * (t - a) / Tc if sgn > 0 else 1 - 4 * (t - a) / Tc
            return m * mp.sin(w * t) - tri
        edges.append(mp.findroot(g, (a, b), solver="anderson"))
    return edges


def spwm_truth(Vd, m, f1, mf, R, L, gon):
    """Periodic load current of the bipolar full bridge: v_AB = s Vd through
    two closed switches (R' = R + 2 / g_on), s = +1 from 0 to the first
    edge (m sin > triangle), flipping at each edge. Closed forms per
    segment: i = s Vd / R' + (i_k - s Vd / R') exp(-(t - t_k) / tau)."""
    Vd, f1, R, L = (mp.mpf(x) for x in (Vd, f1, R, L))
    Rp = R + 2 / mp.mpf(gon)
    tau = L / Rp
    w = 2 * mp.pi * f1
    T1 = 1 / f1
    edges = spwm_edges(m, f1, mf)
    knots = [mp.mpf(0)] + edges + [T1]
    # i(T1) = a i0 + c, linear in i0
    a, c, s = mp.mpf(1), mp.mpf(0), 1
    for t0, t1 in zip(knots[:-1], knots[1:]):
        e = mp.exp(-(t1 - t0) / tau)
        a, c = a * e, c * e + s * Vd / Rp * (1 - e)
        s = -s
    i0 = c / (1 - a)
    c1, isq, vc1, ik, s = mp.mpc(0), mp.mpf(0), mp.mpc(0), i0, 1
    for t0, t1 in zip(knots[:-1], knots[1:]):
        A = s * Vd / Rp
        B = ik - A                               # i = A + B exp(-(t - t0)/tau)
        hh = t1 - t0
        lam = 1 / tau + 1j * w
        c1 += A * (mp.exp(-1j * w * t1) - mp.exp(-1j * w * t0)) / (-1j * w)
        c1 += B * mp.exp(-1j * w * t0) * (1 - mp.exp(-lam * hh)) / lam
        vc1 += s * Vd * (mp.exp(-1j * w * t1) - mp.exp(-1j * w * t0)) / (-1j * w)
        isq += (A**2 * hh + 2 * A * B * tau * (1 - mp.exp(-hh / tau))
                + B**2 * tau / 2 * (1 - mp.exp(-2 * hh / tau)))
        ik = A + B * mp.exp(-hh / tau)
        s = -s
    assert abs(ik - i0) < 1e-25
    return dict(edges=edges, T1=T1, i0=i0, I1=abs(2 * c1 / T1), V1=abs(2 * vc1 / T1),
                I_rms=mp.sqrt(isq / T1), Rp=Rp, w=w)


class BipolarGate:
    """Gate schedule from exact edges: S1, S4 on while s = +1, S2, S3 while
    s = -1; ``next_edge_after`` so variable-step engines land on edges."""

    def __init__(self, edges, T1, plus, minus, n):
        self.e = np.array([float(x) for x in edges])
        self.T1, self.plus, self.minus, self.n = float(T1), plus, minus, n

    def __call__(self, t):
        k, r = divmod(t, self.T1)
        on = self.plus if np.searchsorted(self.e, r, side="right") % 2 == 0 else self.minus
        mask = p.SwitchStateMask(self.n)
        for i in on:
            mask.set(i, True)
        return mask

    def next_edge_after(self, t):
        k, r = divmod(t, self.T1)
        j = int(np.searchsorted(self.e, r, side="right"))
        while True:              # strictly after t, whatever divmod rounded
            e = k * self.T1 + self.e[j % len(self.e)] + (j // len(self.e)) * self.T1
            if e > t:
                return e
            j += 1


def spwm_builder(Vd, R, L, gon, goff, i0):
    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vd", "dc", "gnd", Vd)
        b.add_switch("S1", "dc", "A", gon, goff)
        b.add_switch("S2", "A", "gnd", gon, goff)
        b.add_switch("S3", "dc", "B", gon, goff)
        b.add_switch("S4", "B", "gnd", gon, goff)
        b.add_resistor("R", "A", "m", R)
        b.add_inductor("L", "m", "B", L, i0=i0)
        return b
    return build


# ------------------------------------------------------------ dc-dc cases --
# name: (topology factory args, g_on, g_off, f, D, dt). The buck is the
# repo's benchmarks/circuits/switching/buck/textbook_erickson_buck_ccm.yaml
# (Vin 24 V, D 0.5, 50 kHz, 47 uH, 100 uF, g 200 / 1e-6 S) at R = 10 ohm
# (the file's load) and at R = 5 ohm.

DCDC = {
    "buck_R10": ("buck", (24.0, 47e-6, 100e-6, 10.0), 200.0, 1e-6, 50e3, 0.5),
    "buck_R5": ("buck", (24.0, 47e-6, 100e-6, 5.0), 200.0, 1e-6, 50e3, 0.5),
    "buckboost_ccm": ("buckboost", (12.0, 100e-6, 100e-6, 10.0), 1e3, 1e-9, 100e3, 0.6),
    "buckboost_dcm": ("buckboost", (12.0, 100e-6, 100e-6, 200.0), 1e3, 1e-9, 100e3, 0.6),
    "boost_RL": ("boost", (12.0, 47e-6, 47e-6, 10.0, 0.1), 1e3, 1e-9, 50e3, 0.4),
}
DT_DCDC = 1e-8
N_PER = 4
TOPO = {"buck": _buck, "buckboost": _buckboost, "boost": _boost}


@functools.lru_cache(maxsize=None)
def dcdc_truth(name):
    kind, args, gon, goff, f, D = DCDC[name]
    return dcdc_orbit(TOPO[kind](*args), gon, goff, f, D)


def dcdc_builder(name):
    kind, args, gon, goff, f, D = DCDC[name]
    tr = dcdc_truth(name)
    i0, c0 = float(tr["iL0"]), float(tr["vC0"])

    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", args[0])
        if kind == "buck":
            b.add_switch("S", "vin", "sw", gon, goff)
            b.add_diode("D", "gnd", "sw", gon, goff)
            b.add_inductor("L", "sw", "out", args[1], i0=i0)
        elif kind == "buckboost":
            b.add_switch("S", "vin", "x", gon, goff)
            b.add_diode("D", "out", "x", gon, goff)
            b.add_inductor("L", "x", "gnd", args[1], i0=i0)
        else:
            b.add_resistor("RLw", "vin", "m", args[4])
            b.add_inductor("L", "m", "x", args[1], i0=i0)
            b.add_switch("S", "x", "gnd", gon, goff)
            b.add_diode("D", "x", "out", gon, goff)
        b.add_capacitor("C", "out", "gnd", args[2], c0=c0)
        b.add_resistor("R", "out", "gnd", args[3])
        return b
    return build


def engine_tol(cfg, r, grid_events=True):
    """Relative tolerance of an engine at step ratio r = dt / T (module
    docstring): first order 10 r, second order 10 r^2, error-controlled
    50 rtol (local control, accumulated over the run's steps), exact engine
    10 output_rtol (interpolated window means). ``grid_events=False``: the
    circuit has events off the dt grid (diode commutations at current,
    sine-triangle edges), which the fixed engine places on the grid with
    or without restart: first order for both."""
    restart = 10 * r * r if grid_events else 10 * r
    return {"pwl": 10 * r, "pwl_restart": restart, "trbdf2": 50e-6,
            "esdirk43": 50e-6, "dsed": 1e-5}[cfg]


def mean_sq(res, y, t0, t1):
    """Exact mean of y^2 over [t0, t1] for y linear between samples (the
    window clipped by interpolation): sum h (a^2 + a b + b^2) / 3. A
    trapezoid of y^2 overstates it by sum h (dy)^2 / 6, which on a sparse
    adaptive grid is the metric's error, not the engine's."""
    t, y = np.asarray(res.times), np.asarray(y)
    k = (t > t0) & (t < t1)
    tt = np.concatenate(([t0], t[k], [t1]))
    yy = np.concatenate(([np.interp(t0, t, y)], y[k], [np.interp(t1, t, y)]))
    a, b, h = yy[:-1], yy[1:], np.diff(tt)
    return float(np.sum(h * (a * a + a * b + b * b)) / 3 / (t1 - t0))


def _ripple_vc(tr, name, n=400):
    """Peak-to-peak output ripple of the orbit (sampled, n per mode)."""
    kind, args, gon, goff, f, D = DCDC[name]
    A, _ = TOPO[kind](*args)
    gon, goff = mp.mpf(gon), mp.mpf(goff)
    x = mp.matrix([tr["iL0"], tr["vC0"], 1])
    vs = []
    for Am, h in ((A(gon, goff), tr["DT"]), (A(goff, gon), tr["td"]),
                  (A(goff, goff), tr["T"] - tr["DT"] - tr["td"])):
        E = mp.expm(Am * h / n)
        for _ in range(n):
            x = E * x
            vs.append(x[1])
    return max(vs) - min(vs)


def test_truth_buck_ccm_matches_erickson():
    """Erickson ch. 2: V = D Vg, dI_pp = (Vg - V) D Ts / L, dV_pp =
    dI_pp / (8 f C). R = 5 ohm: K = 2 L / (R Ts) = 0.94 > K_crit = 0.5."""
    tr = dcdc_truth("buck_R5")
    assert tr["mode"] == "CCM"
    V, dI, dV = 12.0, (24 - 12) * 0.5 / 50e3 / 47e-6, (24 - 12) * 0.5 / 50e3 / 47e-6 / (8 * 50e3 * 100e-6)
    assert dI == pytest.approx(2.5532, rel=1e-4) and dV == pytest.approx(0.063830, rel=1e-4)
    assert float(tr["vC_mean"]) == pytest.approx(V, rel=0.02)
    assert float(tr["iL_pk"] - tr["iL0"]) == pytest.approx(dI, rel=0.02)
    assert float(_ripple_vc(tr, "buck_R5")) == pytest.approx(dV, rel=0.02)


def test_truth_buck_yaml_point_is_dcm():
    """FINDING F-T-1 (benchmark, not engine): the repo's "CCM" buck
    (textbook_erickson_buck_ccm.yaml, R = 10 ohm) is in DCM:
    K = 2 * 47e-6 * 50e3 / 10 = 0.47 < K_crit = 1 - D = 0.5. Its steady
    state follows Erickson ch. 5, M = 2 / (1 + sqrt(1 + 4 K / D^2)) =
    0.51035 (12.248 V), not D Vg = 12 V (2.1 % off, outside the proposal's
    +-2 %), and the inductor current is zero for 0.42 us of each period."""
    tr = dcdc_truth("buck_R10")
    assert tr["mode"] == "DCM"
    K, D = 2 * 47e-6 * 50e3 / 10.0, 0.5
    M = 2 / (1 + math.sqrt(1 + 4 * K / D**2))
    assert float(tr["vC_mean"]) == pytest.approx(M * 24, rel=0.02)
    assert abs(float(tr["vC_mean"]) / 12.0 - 1) > 0.02


def test_truth_buckboost_matches_erickson():
    """Erickson ch. 5: CCM M = -D / (1 - D) for K > (1 - D)^2 = 0.16;
    DCM M = -D / sqrt(K)."""
    ccm, dcm = dcdc_truth("buckboost_ccm"), dcdc_truth("buckboost_dcm")
    assert (ccm["mode"], dcm["mode"]) == ("CCM", "DCM")
    assert float(ccm["vC_mean"]) == pytest.approx(-12 * 0.6 / 0.4, rel=0.02)
    K = 2 * 100e-6 * 100e3 / 200.0
    assert float(dcm["vC_mean"]) == pytest.approx(-12 * 0.6 / math.sqrt(K), rel=0.02)


def test_truth_boost_matches_erickson():
    """Erickson ch. 3: V / Vg = (1 / D') / (1 + R_L / (D'^2 R))."""
    tr = dcdc_truth("boost_RL")
    assert tr["mode"] == "CCM"
    Dp = 0.6
    assert float(tr["vC_mean"]) == pytest.approx(12 / Dp / (1 + 0.1 / (Dp**2 * 10)), rel=0.02)


@pytest.mark.parametrize("cfg", list(ENGINES))
@pytest.mark.parametrize("name", list(DCDC))
def test_dcdc_engine_matches_orbit(name, cfg):
    kind, args, gon, goff, f, D = DCDC[name]
    tr = dcdc_truth(name)
    tol = engine_tol(cfg, DT_DCDC * f)
    err, ok = check_dcdc(tr, dcdc_builder(name), "S", f, D, cfg, DT_DCDC, N_PER, tol)
    assert ok, f"{name} on {cfg}: relative errors {err} > {tol:.1e}"


# ------------------------------------------- three-phase bridge (KSV) ----

BRIDGE = dict(Vm=325.0, f=50.0, R=10.0, L=5e-3, gon=1e3)
BR_GOFF, BR_DT, BR_NPER = 1e-9, 1e-6, 2


@functools.lru_cache(maxsize=None)
def bridge():
    return bridge_truth(**BRIDGE)


def test_truth_bridge_matches_kassakian():
    """V_d0 = (3 sqrt(3) / pi) V_m = 537.55 V less the two diode drops
    (2 I_d / g_on = 0.107 V); I_a,rms = sqrt(2/3) I_d with a finite L."""
    tr = bridge()
    Vd0 = 3 * math.sqrt(3) / math.pi * BRIDGE["Vm"]
    assert Vd0 == pytest.approx(537.5457, rel=1e-6)
    assert float(tr["Vd_mean"]) == pytest.approx(Vd0, rel=0.02)
    assert float(tr["Ia_rms"]) == pytest.approx(math.sqrt(2 / 3) * float(tr["i_mean"]), rel=0.02)


@pytest.mark.parametrize("cfg", list(ENGINES))
def test_bridge_engine_matches_truth(cfg):
    tr = bridge()
    T = 1 / BRIDGE["f"]
    h = float(tr["h"])
    build = bridge_builder(BRIDGE["Vm"], BRIDGE["f"], BRIDGE["R"], BRIDGE["L"],
                           BRIDGE["gon"], BR_GOFF, float(tr["i"](0)))
    res = run(build, cfg, BR_NPER * T, BR_DT)
    tN, t0 = BR_NPER * T, (BR_NPER - 1) * T
    iL = np.asarray(res.i("L"))
    vd = np.asarray(res.v("dp")) - np.asarray(res.v("dn"))
    ia = np.asarray(res.i("Va"))
    sc = float(tr["i_mean"])
    err = {
        "i(NT)": abs(at(res, tN, iL) - float(tr["i"](0))) / sc,
        "i(NT-h)": abs(at(res, tN - h, iL) - float(tr["i"](h))) / sc,
        "Vd_mean": abs(avg(res, vd, t0, tN) / float(tr["Vd_mean"]) - 1),
        "Ia_rms": abs(math.sqrt(mean_sq(res, ia, t0, tN)) / float(tr["Ia_rms"]) - 1),
    }
    tol = engine_tol(cfg, BR_DT / (2 * h), grid_events=False)
    assert max(err.values()) <= tol, f"bridge on {cfg}: {err} > {tol:.1e}"


# ------------------------------- bipolar full-bridge SPWM (Mohan ch. 8) ----

SPWM = dict(Vd=300.0, m=0.8, f1=50.0, mf=21, R=10.0, L=20e-3, gon=1e3)
SP_GOFF, SP_DT = 1e-9, 1e-6


@functools.lru_cache(maxsize=None)
def spwm():
    return spwm_truth(**SPWM)


def test_truth_spwm_matches_mohan():
    """Bipolar sine-triangle, m_a = 0.8, m_f = 21: V_o1 = m_a V_d = 240 V
    (the exact Fourier coefficient of the edges); the load's fundamental
    current is V_o1 / |R' + j w L|."""
    tr = spwm()
    S = SPWM
    assert float(tr["V1"]) == pytest.approx(S["m"] * S["Vd"], rel=1e-6)
    I1 = S["m"] * S["Vd"] / abs(float(tr["Rp"]) + 1j * float(tr["w"]) * S["L"])
    assert float(tr["I1"]) == pytest.approx(I1, rel=1e-4)


@pytest.mark.parametrize("cfg", list(ENGINES))
def test_spwm_engine_matches_truth(cfg):
    tr = spwm()
    S = SPWM
    T1 = float(tr["T1"])
    build = spwm_builder(S["Vd"], S["R"], S["L"], S["gon"], SP_GOFF, float(tr["i0"]))
    b = build()
    gate = BipolarGate(tr["edges"], T1, [b.switch_index_of("S1"), b.switch_index_of("S4")],
                       [b.switch_index_of("S2"), b.switch_index_of("S3")], b.graph.num_switches)
    res = run(build, cfg, 2 * T1, SP_DT, gate)
    t = np.asarray(res.times)
    iL = np.asarray(res.i("L"))
    w = 2 * math.pi * S["f1"]
    c1 = 2 * (avg(res, iL * np.cos(w * t), T1, 2 * T1)
              - 1j * avg(res, iL * np.sin(w * t), T1, 2 * T1))
    sc = float(tr["I1"])
    err = {
        "i(2T)": abs(at(res, 2 * T1, iL) - float(tr["i0"])) / sc,
        "I1": abs(abs(c1) / sc - 1),
        "I_rms": abs(math.sqrt(mean_sq(res, iL, T1, 2 * T1)) / float(tr["I_rms"]) - 1),
    }
    tol = engine_tol(cfg, SP_DT * S["mf"] * S["f1"], grid_events=False)
    assert max(err.values()) <= tol, f"spwm on {cfg}: {err} > {tol:.1e}"


# ------------------------------------- resonant charging (Mohan ch. 9) ----
# Vin -> L -> diode -> C: the diode conducts one half-cycle of the series
# resonance and turns off at zero current with v_C ~= 2 Vin.

RES = dict(Vin=100.0, L=10e-6, C=1e-6, gon=1e3)
RES_GOFF, RES_DT = 1e-9, 1e-8


def resonant_truth(Vin, L, C, gon):
    """Underdamped series R'LC from rest (R' = 1 / g_on, the diode):
    i = Vin / (wd L) e^(-a t) sin(wd t), off at t_off = pi / wd, where
    v_C = Vin (1 + e^(-a pi / wd)); a = R' / (2 L), wd = sqrt(1/(LC) - a^2).
    After t_off the diode blocks Vin (leak g_off Vin = 1e-7 A: 1e-7 V)."""
    Vin, L, C = mp.mpf(Vin), mp.mpf(L), mp.mpf(C)
    a = 1 / mp.mpf(gon) / (2 * L)
    wd = mp.sqrt(1 / (L * C) - a**2)
    t_off = mp.pi / wd
    return dict(t_off=t_off, vC=Vin * (1 + mp.exp(-a * t_off)),
                i=lambda t: Vin / (wd * L) * mp.exp(-a * t) * mp.sin(wd * t))


def test_truth_resonant_charging_matches_mohan():
    """Mohan ch. 9 (basic resonant circuit): v_C,peak = 2 V_d, the half
    period pi sqrt(L C) = 9.935 us, peak current V_d / Z0 = 31.6 A."""
    tr = resonant_truth(**RES)
    Z0 = math.sqrt(RES["L"] / RES["C"])
    assert float(tr["vC"]) == pytest.approx(2 * RES["Vin"], rel=0.02)
    assert float(tr["t_off"]) == pytest.approx(math.pi * math.sqrt(RES["L"] * RES["C"]), rel=0.02)
    assert float(tr["i"](tr["t_off"] / 2)) == pytest.approx(RES["Vin"] / Z0, rel=0.02)


@pytest.mark.parametrize("cfg", list(ENGINES))
def test_resonant_charging_engine_matches_truth(cfg):
    tr = resonant_truth(**RES)
    t_off = float(tr["t_off"])

    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "in", "gnd", RES["Vin"])
        b.add_inductor("L", "in", "a", RES["L"], i0=0.0)
        b.add_diode("D", "a", "c", RES["gon"], RES_GOFF)
        b.add_capacitor("C", "c", "gnd", RES["C"], c0=0.0)
        return b
    res = run(build, cfg, 2.5 * t_off, RES_DT)
    i_pk = float(tr["i"](tr["t_off"] / 2))
    err = {
        "i(t_off/2)": abs(at(res, t_off / 2, res.i("L")) - i_pk) / i_pk,
        "vC(end)": abs(at(res, 2.5 * t_off, res.v("c")) / float(tr["vC"]) - 1),
        "i(end)": abs(at(res, 2.5 * t_off, res.i("L"))) / i_pk,
    }
    tol = engine_tol(cfg, RES_DT / t_off, grid_events=False)
    assert max(err.values()) <= tol, f"resonant on {cfg}: {err} > {tol:.1e}"


# ------------------------------------------------------------- findings ----

_HANG = textwrap.dedent("""
    import pulsim as p

    class Gate:  # next_edge_after returns t itself: not strictly after t
        def __init__(self, n, i):
            self.n, self.i = n, i
        def __call__(self, t):
            m = p.SwitchStateMask(self.n)
            m.set(self.i, (t % 1e-5) < 5e-6)
            return m
        def next_edge_after(self, t):
            return t

    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_switch("S", "a", "o", 1e3, 1e-9)
    b.add_resistor("R", "o", "gnd", 10.0)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    try:
        p.simulate(b, t_end=1e-4, engine="dsed",
                   switch_fn=Gate(b.graph.num_switches, b.switch_index_of("S")))
        print("RETURNED")
    except p.PulsimError as e:
        print("REFUSED", e)
""")


@pytest.mark.xfail(strict=True, reason=(
    "finding F-T-2: engine='dsed' never returns when a switch_fn's "
    "next_edge_after(t) returns t (trbdf2 returns, periodic_steady_state "
    "refuses it by name)"))
def test_dsed_finishes_on_a_non_advancing_next_edge_after():
    """Found while writing BipolarGate: a float round-off made its
    next_edge_after return t at an edge, and the exact engine spun at
    100 % CPU (killed after minutes; 30 s here, the run itself takes
    milliseconds). Either a refusal by name or a result passes."""
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(p.__file__)))
    try:
        out = subprocess.run([sys.executable, "-c", _HANG], capture_output=True,
                             text=True, encoding="utf-8", timeout=30, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("simulate(engine='dsed') did not finish within 30 s")
    assert out.stdout.startswith(("RETURNED", "REFUSED")), out.stderr[-500:]
