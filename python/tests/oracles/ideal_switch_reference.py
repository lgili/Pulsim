# ruff: noqa: E741, E731  -- port of the prototype probes; kept diffable against them
"""Test-only dense references for add-branch-form-ideal-switch (task 1.2).

Ports of ``openspec/changes/add-branch-form-ideal-switch/probes/branch_form.py`` (``rl_switch``)
and ``probes/sdirk_restart_ideal.py`` (``dae``, ``Stepper``, ``cutset``, ``capacitor_loop``,
``run_buck``). numpy only; the mode selection of the DCM buck is the LCP oracle of
``lcp_reference.select_modes``.

Branch form (design D1): unknowns are node voltages and one current per switch / diode;
ON: ``v_a - v_b - R_on i = V_f``; OFF: ``-g_off (v_a - v_b) + i = 0``.
"""
import numpy as np

from .lcp_reference import Circuit, select_modes

G = 1.0 - 1.0 / np.sqrt(2.0)


# ---------------------------------------------------------------------------------------------
# branch_form.py (1): V(10) - S - R(1) - L(1 mH) to ground, trapezoid (backward Euler where asked)
# ---------------------------------------------------------------------------------------------
def rl_switch(branch, R_on, g_off, closed, h, steps, be_first=0, i0=0.0, on_steps=None):
    """Returns (rows of (v_a, v_b, i_L) per step, max cond(J))."""
    V, R, L = 10.0, 1.0, 1e-3
    n = 6 if branch else 5
    iL, vL = i0, 0.0
    out, conds = [], []
    for k in range(steps):
        cl = closed if on_steps is None else (k < on_steps)
        be = k < be_first or (on_steps is not None and on_steps <= k < on_steps + be_first)
        J = np.zeros((n, n)); b = np.zeros(n)
        J[0, 3] = 1; J[3, 0] = 1; b[3] = V
        J[1, 1] += 1 / R; J[1, 2] -= 1 / R; J[2, 1] -= 1 / R; J[2, 2] += 1 / R
        J[2, 4] += 1; J[4, 2] = 1
        if be:
            J[4, 4] = -L / h; b[4] = -L / h * iL
        else:
            J[4, 4] = -2 * L / h; b[4] = -2 * L / h * iL - vL
        if branch:
            J[0, 5] += 1; J[1, 5] -= 1
            if cl:
                J[5, 0] = 1; J[5, 1] = -1; J[5, 5] = -R_on
            else:
                J[5, 0] = -g_off; J[5, 1] = g_off; J[5, 5] = 1
        else:
            g = (1 / R_on) if cl else g_off
            J[0, 0] += g; J[0, 1] -= g; J[1, 0] -= g; J[1, 1] += g
        x = np.linalg.solve(J, b)
        conds.append(np.linalg.cond(J))
        iL, vL = x[4], x[2]
        out.append((x[1], x[2], x[4]))
    return np.array(out), max(conds)


# ---------------------------------------------------------------------------------------------
# sdirk_restart_ideal.py: the companion form of the engine on the MNA DAE  E y' = J y + b(t)
# ---------------------------------------------------------------------------------------------
def dae(c, sw_on, d_on):
    """E, J and b(t) = bc @ s(t) for one mode (R_on = 0 and g_off = 0 legal)."""
    N, nV, nL, nS, nD = len(c.nodes), len(c.V), len(c.L), len(c.S), len(c.D)
    iV, iL, iS, iD = N, N + nV, N + nV + nL, N + nV + nL + nS
    n = iD + nD
    E, J = np.zeros((n, n)), np.zeros((n, n))
    bc = np.zeros((n, c.ns))

    def kcl(M, a, b, col, v=1.0):
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
    for j, (_, a, b, spec) in enumerate(c.V):
        kcl(J, a, b, iV + j); vrow(J, iV + j, a, b); bc[iV + j] = -c.src_row(spec)
    for j, (_, a, b, L, _) in enumerate(c.L):
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
            self.cache[k] = (np.linalg.inv(E * coef - J), E, J, bc, o)
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
        A, E, J, bc, _ = self.mats(key, 1.0 / g)
        Y1 = A @ (bc @ self.c.s_at(t0 + g) + (E @ y0) / g)
        Ek1 = (E @ (Y1 - y0)) / g
        Y2 = A @ (bc @ self.c.s_at(t0 + h) + (E @ y0) / g + ((1 - G) / G) * Ek1)
        Ek2 = (E @ (Y2 - y0) - (1 - G) * h * Ek1) / g
        return Y2, Ek2, (Y1, Ek1)

    def step(self, key, y0, Ey0, t0, h, k_restart):
        if k_restart is None:
            return self.trap(key, y0, Ey0, t0, h)
        if self.restart == "sdirk3":
            y, Ey, _ = self.sdirk(key, y0, t0, h)
            return y, Ey
        if self.restart == "be_pair":
            y, _ = self.be(key, y0, t0, h / 2)
            return self.be(key, y, t0 + h / 2, h / 2)
        return self.trap(key, y0, Ey0, t0, h)


def restart_len(rule):
    return {"sdirk3": 3, "be_pair": 1, "none": 0}[rule]


def cutset(rule="sdirk3", N=20, steps=106, open_at=100):
    """(1): 10 V - ideal S - 1 ohm - 1 mH; S opens at step `open_at`, h = tau/N.
    Returns dict(v_a, i_L per step, i_L(t-), area = SDIRK quadrature of v_a over the event step)."""
    c = Circuit()
    c.add_Vdc("V", "vin", "0", 10.0); c.add_S("S", "vin", "a"); c.add_R("a", "b", 1.0)
    c.add_L("L", "b", "0", 1e-3)
    h = 1e-3 / N
    st = Stepper(c, h, rule)
    key = ((True,), ())
    E, J, bc, o = dae(c, *key)
    y = np.zeros(E.shape[0]); Ey = np.zeros_like(y)
    out, left, area, iL_minus = [], None, None, None
    for k in range(steps):
        if k == open_at:
            key = ((False,), ()); left = restart_len(rule) or None
            iL_minus = y[o["iL"]]
        kr = (restart_len(rule) - left) if left else None
        y_new, Ey = st.step(key, y, Ey, k * h, h, kr)
        if k == open_at and rule == "sdirk3":
            _, _, (Y1, _) = st.sdirk(key, y, k * h, h)
            area = h * ((1 - G) * Y1[1] + G * y_new[1])
        if left:
            left -= 1
        y = y_new
        out.append((y[1], y[o["iL"]]))
    out = np.array(out)
    return dict(v_a=out[:, 0], i_L=out[:, 1], iL_minus=iL_minus, area=area, h=h)


def capacitor_loop(rule="sdirk3", h=1e-7, steps=5):
    """(2): 1 uF at 10 V joined to 3 uF at 2 V by an ideal switch. Rows (v1, v2, i_S) per step."""
    c = Circuit()
    c.add_C("C1", "x", "0", 1e-6, 10.0); c.add_C("C2", "y", "0", 3e-6, 2.0); c.add_S("S", "x", "y")
    st = Stepper(c, h, rule)
    key = ((True,), ())
    E, J, bc, o = dae(c, *key)
    y = np.zeros(E.shape[0]); y[0], y[1] = 10.0, 2.0
    Ey = np.zeros_like(y)
    rows = []
    left = restart_len(rule)
    for k in range(steps):
        kr = (restart_len(rule) - left) if left else None
        y, Ey = st.step(key, y, Ey, k * h, h, kr)
        left = max(0, left - 1)
        rows.append((y[0], y[1], y[o["iS"]]))
    return np.array(rows)


VC_REF = 8.9963970815477      # 40-digit orbit of probes/ideal_dcm_buck.py
T_BUCK = 1e-5


def buck_circuit():
    c = Circuit()
    c.add_Vdc("Vin", "vin", "0", 24.0); c.add_S("S", "vin", "sw"); c.add_D("D1", "0", "sw")
    c.add_L("L", "sw", "vout", 100e-6); c.add_C("C", "vout", "0", 47e-6); c.add_R("vout", "0", 50.0)
    return c


def consistent_start(c, key, vC, iL):
    E, J, bc, o = dae(c, *key)
    M = J.copy(); r = -bc @ c.s_at(0.0)
    vout = c.nodes["vout"]
    M[vout] = 0; M[vout, vout] = 1; r[vout] = vC
    M[o["iL"]] = 0; M[o["iL"], o["iL"]] = 1; r[o["iL"]] = iL
    y = np.linalg.solve(M, r)
    return y, J @ y + bc @ c.s_at(0.0)


def run_buck(N, rule="sdirk3", periods=300):
    """(3): ideal DCM buck, fixed step T/N, diode turn-off located by linear interpolation of i_D
    and the step split there, modes by the LCP, restart after every gate edge and diode event.
    Returns (v_C(periods T), i_L(periods T), factorisations, events)."""
    c = buck_circuit()
    h = T_BUCK / N
    st = Stepper(c, h, rule)
    key = ((True,), (False,))
    y, Ey = consistent_start(c, key, VC_REF, 0.0)
    o = dae(c, *key)[3]
    left = 0
    nev = 0
    on_steps = int(round(0.3 * N))
    for k in range(periods * N):
        t0 = k * h
        ph = k % N
        if ph == 0 or ph == on_steps:
            sw = (ph == 0,)
            d, _ = select_modes(c, np.concatenate([states(c, y), c.s_at(t0)]), t0, sw, key[1], 1e-6 * T_BUCK)
            key = (sw, d); left = restart_len(rule); nev += 1
        kr = (restart_len(rule) - left) if left else None
        y1, Ey1 = st.step(key, y, Ey, t0, h, kr)
        iD = y1[o["iD"]]
        if key[1][0] and iD < 0:
            frac = y[o["iD"]] / (y[o["iD"]] - iD)
            hs = frac * h
            ys, Eys = st.step(key, y, Ey, t0, hs, kr)
            d, _ = select_modes(c, np.concatenate([states(c, ys), c.s_at(t0 + hs)]), t0 + hs, key[0],
                                key[1], 1e-6 * T_BUCK)
            key = (key[0], d); nev += 1
            left = restart_len(rule)
            kr = 0 if left else None
            y1, Ey1 = st.step(key, ys, Eys, t0 + hs, h - hs, kr)
        if left:
            left -= 1
        y, Ey = y1, Ey1
    return y[c.nodes["vout"]], y[o["iL"]], st.n_factor, nev
