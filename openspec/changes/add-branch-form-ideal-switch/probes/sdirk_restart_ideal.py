"""Ideal switches in branch form on the implicit fixed-step route, with the restart rule as the
maintainer decided it on 2026-11-13: THREE consecutive Alexander SDIRK(2,2) steps after every
discontinuity (gamma = 1 - 1/sqrt2; one factor per mode, the trapezoidal matrix at dt_eq = 2 gamma h),
then the trapezoidal rule. Compared with: trapezoid only, and the backward-Euler pair it replaced.

Run: nice -n 15 python3 sdirk_restart_ideal.py     (numpy, scipy; < 1 min)
Needs ../../add-complementarity-mode-selection/probes/lcp_core.py (netlist and the LCP mode selection).

Companion form of the engine: MNA DAE  E y' = J y + b(t), y = [node v | V-source i | inductor i |
switch i | diode i]; algebraic rows are enforced at the new time, the derivative history is the
vector E y'(t_n) (stale across an event -- the cause of the ringing).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../add-complementarity-mode-selection/probes"))
from lcp_core import Circuit, select_modes   # noqa: E402

G = 1.0 - 1.0 / np.sqrt(2.0)


def dae(c, sw_on, d_on):
    """E, J and a source function b(t) for one mode (branch form, R_on = 0 and g_off = 0 legal)."""
    N, nV, nL, nS, nD = len(c.nodes), len(c.V), len(c.L), len(c.S), len(c.D)
    iV, iL, iS, iD = N, N + nV, N + nV + nL, N + nV + nL + nS
    n = iD + nD
    E, J = np.zeros((n, n)), np.zeros((n, n))
    bc = np.zeros((n, c.ns))                    # b(t) = bc @ s(t)

    def kcl(M, a, b, col, v=1.0):               # branch current a -> b: KCL rows are "sum leaving = 0" -> J gets -1
        if a >= 0: M[a, col] -= v
        if b >= 0: M[b, col] += v

    def vrow(M, row, a, b, v=1.0):
        if a >= 0: M[row, a] += v
        if b >= 0: M[row, b] -= v

    for a, b, R in c.R:
        for p_, q_, sg in ((a, a, -1), (b, b, -1), (a, b, 1), (b, a, 1)):
            if p_ >= 0 and q_ >= 0:
                J[p_, q_] += sg / R
    for _, a, b, C, _ in c.C:
        for p_, q_, sg in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            if p_ >= 0 and q_ >= 0:
                E[p_, q_] += sg * C
    for j, (_, a, b, spec) in enumerate(c.V):   # 0 = v_a - v_b - u  ->  J row, b = -u
        kcl(J, a, b, iV + j); vrow(J, iV + j, a, b); bc[iV + j] = -c.src_row(spec)
    for j, (_, a, b, L, _) in enumerate(c.L):   # L i' = v_a - v_b
        kcl(J, a, b, iL + j); vrow(J, iL + j, a, b); E[iL + j, iL + j] = L
    for _, a, b, I in c.I:
        if a >= 0: bc[a, 0] -= I
        if b >= 0: bc[b, 0] += I
    for j, (_, a, b, R_on) in enumerate(c.S):
        kcl(J, a, b, iS + j)
        if sw_on[j]:
            vrow(J, iS + j, a, b); J[iS + j, iS + j] = -R_on
        else:
            J[iS + j, iS + j] = 1.0
    for j, (_, a, b, V_f, R_on) in enumerate(c.D):
        kcl(J, a, b, iD + j)
        if d_on[j]:
            vrow(J, iD + j, a, b); J[iD + j, iD + j] = -R_on; bc[iD + j, 0] = -V_f
        else:
            J[iD + j, iD + j] = 1.0
    return E, J, bc, dict(iV=iV, iL=iL, iS=iS, iD=iD)


def states(c, y):
    """(v_C..., i_L...) from the MNA vector, for the LCP (states held at the event)."""
    N, nV = len(c.nodes), len(c.V)
    v = lambda k: 0.0 if k < 0 else y[k]
    return np.array([v(a) - v(b) for _, a, b, _, _ in c.C] + [y[N + nV + j] for j in range(len(c.L))])


class Stepper:
    def __init__(self, c, h, restart="sdirk3"):
        self.c, self.h, self.restart = c, h, restart
        self.cache = {}
        self.n_factor = 0

    def mats(self, key, coef):
        k = (key, coef)
        if k not in self.cache:
            E, J, bc, o = dae(self.c, *key)
            self.cache[k] = (np.linalg.inv(E * coef - J), E, J, bc, o)   # dense stand-in for one LU
            self.n_factor += 1
        return self.cache[k]

    def trap(self, key, y0, Ey0, t0, h):
        A, E, J, bc, _ = self.mats(key, 2.0 / h)
        y1 = A @ (bc @ self.c.s_at(t0 + h) + (2.0 / h) * (E @ y0) + Ey0)
        return y1, (2.0 / h) * (E @ (y1 - y0)) - Ey0

    def be(self, key, y0, t0, h):
        A, E, J, bc, _ = self.mats(key, 1.0 / h)
        y1 = A @ (bc @ self.c.s_at(t0 + h) + (1.0 / h) * (E @ y0))
        return y1, (E @ (y1 - y0)) / h

    def sdirk(self, key, y0, t0, h):
        g = G * h
        A, E, J, bc, _ = self.mats(key, 1.0 / g)          # = trapezoid matrix at dt_eq = 2 gamma h
        Y1 = A @ (bc @ self.c.s_at(t0 + g) + (E @ y0) / g)
        Ek1 = (E @ (Y1 - y0)) / g
        Y2 = A @ (bc @ self.c.s_at(t0 + h) + (E @ y0) / g + ((1 - G) / G) * Ek1)
        Ek2 = (E @ (Y2 - y0) - (1 - G) * h * Ek1) / g
        return Y2, Ek2, (Y1, Ek1)

    def step(self, key, y0, Ey0, t0, h, k_restart):
        """k_restart: index of this step inside the restart sequence, or None."""
        if k_restart is None:
            return self.trap(key, y0, Ey0, t0, h)
        if self.restart == "sdirk3":
            y, Ey, _ = self.sdirk(key, y0, t0, h)
            return y, Ey
        if self.restart == "be_pair":                     # two backward-Euler half-steps (the old rule)
            y, _ = self.be(key, y0, t0, h / 2)
            return self.be(key, y, t0 + h / 2, h / 2)
        return self.trap(key, y0, Ey0, t0, h)             # 'none'


def restart_len(rule):
    return {"sdirk3": 3, "be_pair": 1, "none": 0}[rule]


np.set_printoptions(precision=6, linewidth=150)
print("=" * 100)
print("(1) inductor cut-set: 10 V - ideal S - 1 ohm - 1 mH; S opens at 5 ms with no other path; h = tau/20")
c = Circuit()
c.add_Vdc("V", "vin", "0", 10.0); c.add_S("S", "vin", "a"); c.add_R("a", "b", 1.0); c.add_L("L", "b", "0", 1e-3)
h = 1e-3 / 20
for rule in ("none", "be_pair", "sdirk3"):
    st = Stepper(c, h, rule)
    key = ((True,), ())
    E, J, bc, o = dae(c, *key)
    y = np.zeros(E.shape[0]); Ey = np.zeros_like(y)
    out = []
    left = None
    for k in range(106):
        if k == 100:
            key = ((False,), ()); left = restart_len(rule) or None
            iL_minus = y[o["iL"]]
        kr = (restart_len(rule) - left) if left else None
        y_new, Ey = st.step(key, y, Ey, k * h, h, kr)
        if k == 100 and rule == "sdirk3":
            _, _, (Y1, _) = st.sdirk(key, y, k * h, h)
            area = h * ((1 - G) * Y1[1] + G * y_new[1])      # quadrature of the SDIRK step on v_a
        if left:
            left -= 1
        y = y_new
        out.append((y[1], y[o["iL"]]))
    out = np.array(out)
    print(f"   restart={rule:8s} i_L steps 100..105: {out[100:106, 1]}")
    print(f"                     v_a steps 100..105: {out[100:106, 0]}")
    if rule == "sdirk3":
        print(f"                     flux removed L*(i+ - i-) = {1e-3 * (out[100, 1] - iL_minus):+.8e} V s"
              f" (L i(t-) = {1e-3 * iL_minus:.8e}); SDIRK quadrature of v_a over the step = {area:+.8e} V s;"
              f" energy record 1/2 L i^2 = {0.5e-3 * iL_minus ** 2 * 1e3:.4f} mJ")

print("=" * 100)
print("(2) capacitor loop: 1 uF at 10 V joined to 3 uF at 2 V by an ideal switch, h = 1e-7 s")
c = Circuit()
c.add_C("C1", "x", "0", 1e-6, 10.0); c.add_C("C2", "y", "0", 3e-6, 2.0); c.add_S("S", "x", "y")
for rule in ("none", "be_pair", "sdirk3"):
    st = Stepper(c, 1e-7, rule)
    key = ((True,), ())
    E, J, bc, o = dae(c, *key)
    y = np.zeros(E.shape[0]); y[0], y[1] = 10.0, 2.0     # held states before the closing (left limit)
    Ey = np.zeros_like(y)
    rows = []
    left = restart_len(rule)
    for k in range(5):
        kr = (restart_len(rule) - left) if left else None
        y, Ey = st.step(key, y, Ey, k * 1e-7, 1e-7, kr)
        left = max(0, left - 1)
        rows.append((y[0], y[1], y[o["iS"]]))
    rows = np.array(rows)
    q = 1e-6 * (10.0 - rows[0, 0])
    print(f"   restart={rule:8s} v1 {rows[:, 0]}  v2 {rows[:, 1]}  i_S {rows[:, 2]}  charge moved {q:.6e} C"
          f"  energy lost {0.5e-6 * 100 + 0.5 * 3e-6 * 4 - 0.5 * 4e-6 * rows[-1, 0] ** 2:.6e} J (exact 2.4e-05)")

print("=" * 100)
print("(3) ideal DCM buck (24 V, S, D, 100 uH, 47 uF, 50 ohm, D = 0.3, 100 kHz), fixed step T/N, diode")
print("    turn-off located by linear interpolation of i_D and the step split there; modes by the LCP;")
print("    restart after every gate edge and diode event. Start on the 40-digit orbit, 300 periods.")
VC_REF = 8.9963970815477
T = 1e-5
c = Circuit()
c.add_Vdc("Vin", "vin", "0", 24.0); c.add_S("S", "vin", "sw"); c.add_D("D1", "0", "sw")
c.add_L("L", "sw", "vout", 100e-6); c.add_C("C", "vout", "0", 47e-6); c.add_R("vout", "0", 50.0)


def consistent_start(c, key, vC, iL, st):
    """state and derivative history of the orbit start (gate rising edge, i_L = 0)."""
    E, J, bc, o = dae(c, *key)
    n = E.shape[0]
    y = np.zeros(n)
    # solve the algebraic system with v_C and i_L pinned: replace the two differential rows
    M = J.copy(); r = -bc @ c.s_at(0.0)
    vout = c.nodes["vout"]
    M[vout] = 0; M[vout, vout] = 1; r[vout] = vC
    M[o["iL"]] = 0; M[o["iL"], o["iL"]] = 1; r[o["iL"]] = iL
    y = np.linalg.solve(M, r)
    Ey = J @ y + bc @ c.s_at(0.0)
    return y, Ey


def run_buck(N, rule, periods=300):
    h = T / N
    st = Stepper(c, h, rule)
    key = ((True,), (False,))
    y, Ey = consistent_start(c, key, VC_REF, 0.0, st)
    o = dae(c, *key)[3]
    t = 0.0
    left = 0
    nev = 0
    on_steps = int(round(0.3 * N))
    for k in range(periods * N):
        t0 = k * h
        ph = k % N
        if ph == 0 or ph == on_steps:                  # gate edge on the grid: LCP, then restart
            sw = (ph == 0,)
            d, _ = select_modes(c, np.concatenate([states(c, y), c.s_at(t0)]), t0, sw, key[1], 1e-6 * T)
            key = (sw, d); left = restart_len(rule); nev += 1
        kr = (restart_len(rule) - left) if left else None
        y1, Ey1 = st.step(key, y, Ey, t0, h, kr)
        iD = y1[o["iD"]]
        if key[1][0] and iD < 0:                       # diode current crosses zero inside the step
            frac = y[o["iD"]] / (y[o["iD"]] - iD)
            hs = frac * h
            ys, Eys = st.step(key, y, Ey, t0, hs, kr)  # sub-step to the located zero
            d, _ = select_modes(c, np.concatenate([states(c, ys), c.s_at(t0 + hs)]), t0 + hs, key[0], key[1], 1e-6 * T)
            key = (key[0], d); nev += 1
            left = restart_len(rule)
            kr = 0 if left else None
            y1, Ey1 = st.step(key, ys, Eys, t0 + hs, h - hs, kr)
        if left:
            left -= 1
        y, Ey = y1, Ey1
    return y[c.nodes["vout"]], y[o["iL"]], st.n_factor, nev


print("    rule      N     v_C(300 T) - orbit      i_L(300 T)    factorisations")
prev = {}
for rule in ("none", "be_pair", "sdirk3"):
    for N in (50, 100, 200, 400):
        vC, iL, nf, nev = run_buck(N, rule)
        err = vC - VC_REF
        order = "" if rule not in prev else f"  order {np.log2(abs(prev[rule]) / abs(err)):.2f}"
        prev[rule] = err
        print(f"    {rule:8s} {N:4d}   {err:+.3e} V            {iL:+.1e} A     {nf}{order}")
