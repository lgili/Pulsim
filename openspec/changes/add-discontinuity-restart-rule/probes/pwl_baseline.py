"""Engine probes for add-discontinuity-restart-rule.

    python pwl_baseline.py save  OUT.npz            # record (any build)
    python pwl_baseline.py check OUT.npz [--off]    # bit-identity vs OUT.npz
    python pwl_baseline.py table [--on|--off]       # error-vs-exact table

`save` and `table` without a flag never pass `discontinuity_restart`,
so they run on the build that predates the keyword. `check --off`
re-runs every fixed-engine circuit with `discontinuity_restart=False`
AND with the keyword absent, and the trbdf2 / dsed circuits as they
are, and demands np.array_equal on states and switch bits.

Circuits are the core-review probes (p1_edge, p2_buck, p3_dcm,
p8_caploop, p13, test_first_step_order, test_nonlinear_capacitor) run
in the ENGINE, plus a saturable inductor across a gate edge.
"""
from __future__ import annotations

import math
import sys
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq

import pulsim as p

warnings.simplefilter("ignore")
FLAG: dict = {}          # extra simulate() kwargs (the restart flag)
GON, GOFF = 1e3, 1e-9


def sim(b, **kw):
    if kw.get("engine", "pwl") == "pwl":
        kw = {**kw, **FLAG}
    return p.simulate(b, **kw)


def mask_fn(b, name, on):
    n, k = b.graph.num_switches, b.switch_index_of(name)

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, bool(on(t)))
        return m
    return fn


def arrays(res):
    return np.asarray(res.times), np.asarray(res.states), np.asarray(res.switch_states)


def order(e):
    return " ".join(f"{math.log2(abs(a / b)):.2f}" for a, b in zip(e, e[1:]))


# ---- A. RL behind a switch, edge on the grid (p1_edge.py) -------------------
V_RL, R_RL, L_RL = 10.0, 1.0, 1e-3
TAU_RL = L_RL / (R_RL + 1 / GON)


def rl_edge(dt):
    te = round(1e-3 / dt) * dt
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V_RL)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_resistor("R", "a", "m", R_RL)
    b.add_inductor("L", "m", "gnd", L_RL)
    res = sim(b, t_end=te + 0.6e-3 + 1e-12, dt=dt,
              switch_fn=mask_fn(b, "S", lambda t: t >= te - 1e-15))
    t, i = np.asarray(res.times), np.asarray(res.i("L"))
    j = int(np.argmin(abs(t - (te + 0.5e-3))))
    ex = V_RL / (R_RL + 1 / GON) * (1 - math.exp(-(t[j] - te) / TAU_RL))
    slope = V_RL / (R_RL + 1 / GON) / TAU_RL * math.exp(-(t[j] - te) / TAU_RL)
    return res, i[j] - ex, (i[j] - ex) / slope / dt


# ---- B/C. buck CCM / DCM against the exact piecewise-LTI solution (p2_buck.py)
VIN, LB, CB, FSW = 24.0, 100e-6, 47e-6, 100e3
TB = 1 / FSW
T0 = 2e-6


def _prop(A, B, x, dt):
    M = np.zeros((3, 3))
    M[:2, :2] = A
    M[:2, 2] = B
    E = expm(M * dt)
    return E[:2, :2] @ x + E[:2, 2]


def buck_exact(R, D, nper):
    def AB(gs, gd):
        k = 1 / (gs + gd)
        return (np.array([[-k / LB, -1 / LB], [1 / CB, -1 / (R * CB)]]),
                np.array([gs * VIN * k / LB, 0.0]))
    x = np.zeros(2)

    def seg(x, rem, gs):
        d_on = -((gs * VIN - x[0]) / (gs + GOFF)) >= 0
        A, B = AB(gs, GON if d_on else GOFF)
        f = lambda h: -((gs * VIN - _prop(A, B, x, h)[0]) / (gs + (GON if d_on else GOFF)))  # noqa: E731
        if f(1e-15) * f(rem) < 0:
            h = brentq(f, 1e-15, rem, xtol=1e-18, rtol=1e-15)
            x = _prop(A, B, x, h)
            A2, B2 = AB(gs, GOFF if d_on else GON)
            return _prop(A2, B2, x, rem - h)
        return _prop(A, B, x, rem)
    for _ in range(nper):
        x = seg(x, D * TB, GON)
        x = seg(x, (1 - D) * TB, GOFF)
    return x


def buck(R, D, N, nper=40):
    dt = TB / N
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF)
    b.add_inductor("L", "sw", "vout", LB)
    b.add_capacitor("C", "vout", "gnd", CB)
    b.add_resistor("RL", "vout", "gnd", R)

    def on(t):
        ph = (t - T0) / TB
        return t >= T0 - 1e-13 and ph - math.floor(ph + 1e-9) < D - 1e-9
    return sim(b, t_end=T0 + nper * TB + 1e-12, dt=dt, switch_fn=mask_fn(b, "S", on))


def buck_err(R, D, N, nper=40):
    res = buck(R, D, N, nper)
    xe = buck_exact(R, round(D * N) / N, nper)
    t = np.asarray(res.times)
    j = int(np.argmin(abs(t - (T0 + nper * TB))))
    return res, np.asarray(res.i("L"))[j] - xe[0], np.asarray(res.v("vout"))[j] - xe[1]


def dcm_idle(res):
    t = np.asarray(res.times)
    vsw, vo = np.asarray(res.v("sw")), np.asarray(res.v("vout"))
    m = t >= t[-1] - TB + 1e-12
    tt = (t[m] - t[m][0]) / TB
    idle = (tt > 0.75) & (tt < 0.98)
    return vsw[m][idle].min(), vsw[m][idle].max(), vo[m][idle].mean()


# ---- D. hard-switched capacitor (p8_caploop.py) -----------------------------
def caploop(dt, nper=20):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 100.0)
    b.add_resistor("R", "in", "sw", 10.0)
    b.add_switch("S", "sw", "gnd", GON, GOFF)
    b.add_capacitor("C", "sw", "gnd", 1e-9)

    def on(t):
        ph = t / TB
        return ph - math.floor(ph + 1e-9) >= 0.5 - 1e-9
    res = sim(b, t_end=nper * TB + 1e-12, dt=dt, switch_fn=mask_fn(b, "S", on))
    t = np.asarray(res.times)
    i_s = np.asarray(res.v("sw")) * GON
    ph = (t / TB) % 1.0
    sel = (ph > 0.52) & (ph < 0.98) & (t > nper // 2 * TB)
    P = np.trapezoid(np.where(sel, i_s ** 2 / GON, 0.0), t) / (t[-1] - nper // 2 * TB)
    return res, i_s[sel].min(), i_s[sel].max(), P


# ---- E. pulse source step on the grid into an RC (p13.py / spec) ------------
def rc_pulse(dt, te=2e-6, tau=1e-6):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "in", "gnd", 0.0, 10.0, te, 50e-6, 0.0, 0.0, 0.0)
    b.add_resistor("R", "in", "a", 1e3)
    b.add_capacitor("C", "a", "gnd", tau / 1e3)
    res = sim(b, t_end=te + 4 * tau + 1e-12, dt=dt)
    t, v = np.asarray(res.times), np.asarray(res.v("a"))
    ex = np.where(t > te, 10 * (1 - np.exp(-(t - te) / tau)), 0.0)
    return res, float(np.abs(v - ex).max())


# ---- F. first-step order (test_first_step_order.py) -------------------------
def first_cases():
    def rl_ic(dt):
        b = p.CircuitBuilder()
        b.add_resistor("R", "a", "gnd", 1.0)
        b.add_inductor("L", "a", "gnd", 1e-3, i0=5.0)
        r = sim(b, t_end=5e-3, dt=dt)
        return r, np.asarray(r.i("L"))[-1]

    def rc_ic(dt):
        b = p.CircuitBuilder()
        b.add_resistor("R", "a", "gnd", 1000.0)
        b.add_capacitor("C", "a", "gnd", 1e-6, c0=5.0)
        r = sim(b, t_end=5e-3, dt=dt)
        return r, np.asarray(r.v("a"))[-1]

    def rl_step(dt):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "a", "gnd", 1.0)
        b.add_resistor("R", "a", "m", 1.0)
        b.add_inductor("L", "m", "gnd", 1e-3)
        r = sim(b, t_end=5e-3, dt=dt)
        return r, np.asarray(r.i("L"))[-1]

    def rc_step(dt):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "s", "gnd", 1.0)
        b.add_resistor("R", "s", "a", 1000.0)
        b.add_capacitor("C", "a", "gnd", 1e-6)
        r = sim(b, t_end=5e-3, dt=dt)
        return r, np.asarray(r.v("a"))[-1]

    def rl_sine(dt):
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0, v_amplitude=1.0,
                                  frequency=200.0, phase=0.0)
        b.add_resistor("R", "a", "m", 1.0)
        b.add_inductor("L", "m", "gnd", 1e-3)
        r = sim(b, t_end=5e-3, dt=dt)
        return r, np.asarray(r.i("L"))[-1]
    w = 2 * math.pi * 200.0
    Z2 = 1 + (w * 1e-3) ** 2
    iss = lambda t: (math.sin(w * t) - w * 1e-3 * math.cos(w * t)) / Z2  # noqa: E731
    return [("RL i0", rl_ic, 5 * math.exp(-5)), ("RC c0", rc_ic, 5 * math.exp(-5)),
            ("RL step", rl_step, 1 - math.exp(-5)), ("RC step", rc_step, 1 - math.exp(-5)),
            ("RL sine", rl_sine, iss(5e-3) - iss(0) * math.exp(-5))]


# ---- G. charge-based Coss charged from rest (test_nonlinear_capacitor.py) ---
C0, V0C, MC = 2000e-12, 25.0, 0.5


def coss(dt=2e-11):
    b = p.CircuitBuilder()
    b.add_current_source("Isrc", "n", "gnd", 6.0)
    b.add_nonlinear_capacitor("Coss", "n", "gnd", C0, V0C, MC)
    b.add_resistor("Rleak", "n", "gnd", 1e12)
    res = sim(b, t_end=80e-9, dt=dt)
    t, v = np.asarray(res.times), np.asarray(res.v("n"))
    q = C0 * V0C / (1 - MC) * ((1 + v / V0C) ** (1 - MC) - 1)
    return res, [q[k] / (6.0 * t[k]) for k in (1, 2, 11, 40)]


# ---- H. saturable inductor behind a switch closing on the grid --------------
L0S, ISAT, LRES = 1e-3, 2.0, 5e-5


def sat_edge(dt, te=1e-4):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_saturable_inductor("Ls", "m", "gnd", L0S, ISAT, LRES)
    res = sim(b, t_end=te + 0.5e-3 + 1e-12, dt=dt,
              switch_fn=mask_fn(b, "S", lambda t: t >= te - 1e-15))
    t, i = np.asarray(res.times), np.asarray(res.i("Ls"))
    j = int(np.argmin(abs(t - (te + 0.5e-3))))
    return res, i[j] - sat_ref()


_SAT_REF = []


def sat_ref():
    if not _SAT_REF:
        Lf = lambda i: LRES + (L0S - LRES) / (1 + (i / ISAT) ** 2)  # noqa: E731
        R = 1.0 + 1 / GON
        sol = solve_ivp(lambda t, y: [(10.0 - R * y[0]) / Lf(y[0])], (0, 0.5e-3), [0.0],
                        rtol=1e-12, atol=1e-14, method="DOP853")
        _SAT_REF.append(sol.y[0, -1])
    return _SAT_REF[0]


# ---- the record / table ----------------------------------------------------
def runs():
    """name -> SimulationResult, the set used for bit-identity."""
    out = {}
    out["rl_edge_2us"] = rl_edge(2e-6)[0]
    out["buck_ccm_T100"] = buck(1.0, 0.5, 100)
    out["buck_dcm_T100"] = buck(50.0, 0.3, 100)
    out["caploop_1e-8"] = caploop(1e-8)[0]
    out["rc_pulse_tau10"] = rc_pulse(1e-7)[0]
    for name, fn, _ in first_cases():
        out["first_" + name.replace(" ", "_")] = fn(2e-6)[0]
    out["coss"] = coss()[0]
    out["sat_edge_2us"] = sat_edge(2e-6)[0]
    # the other two engines: must not move at all
    for eng in ("trbdf2", "dsed"):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", VIN)
        b.add_switch("S", "vin", "sw", GON, GOFF)
        b.add_diode("D1", "gnd", "sw", GON, GOFF)
        b.add_inductor("L", "sw", "vout", LB)
        b.add_capacitor("C", "vout", "gnd", CB)
        b.add_resistor("RL", "vout", "gnd", 50.0)
        pwm = mask_fn(b, "S", lambda t: (t / TB) % 1.0 < 0.3)
        out[f"{eng}_buck_dcm"] = p.simulate(b, t_end=20 * TB, engine=eng, switch_fn=pwm)
        b = p.CircuitBuilder()
        b.add_pulse_voltage_source("V", "in", "gnd", 0.0, 10.0, 2e-6, 50e-6, 0.0, 0.0, 0.0)
        b.add_resistor("R", "in", "a", 1e3)
        b.add_capacitor("C", "a", "gnd", 1e-9)
        out[f"{eng}_rc_pulse"] = p.simulate(b, t_end=8e-6, engine=eng)
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "s", "gnd", 5.0)
        b.add_resistor("R", "s", "a", 1.0)
        b.add_inductor("L", "a", "m", 1e-3)
        b.add_capacitor("C", "m", "gnd", 1e-6)
        out[f"{eng}_rlc_step"] = p.simulate(b, t_end=5e-3, engine=eng)
    return out


def save(path):
    data = {}
    for name, res in runs().items():
        t, x, sw = arrays(res)
        data[name + "/t"], data[name + "/x"], data[name + "/sw"] = t, x, sw
    np.savez(path, **data)
    print(f"saved {len(data) // 3} runs to {path}")


def check(path, off):
    ref = np.load(path)
    variants = [{}]
    if off:
        variants.append({"discontinuity_restart": False})
    bad = 0
    for kw in variants:
        FLAG.clear()
        FLAG.update(kw)
        for name, res in runs().items():
            for part, arr in zip(("t", "x", "sw"), arrays(res)):
                same = np.array_equal(ref[f"{name}/{part}"], arr)
                bad += not same
                if not same:
                    print(f"DIFFERS {kw} {name}/{part}")
        print(f"checked {kw or 'keyword absent'}")
    print("BIT-IDENTICAL" if bad == 0 else f"{bad} ARRAYS DIFFER")
    return bad


def table():
    print("A. RL edge on the grid, err at te + 0.5 ms [A] (shift in dt)")
    e = []
    for dt in (4e-6, 2e-6, 1e-6):
        _, err, sh = rl_edge(dt)
        e.append(err)
        print(f"   dt={dt:.0e}: {err:+.3e} ({sh:+.3f} dt)")
    print("   order", order(e))
    for lab, R, D in (("B. buck CCM", 1.0, 0.5), ("B'. buck DCM", 50.0, 0.3)):
        ei, ev = [], []
        for N in (50, 100, 200, 400, 800):
            _, a, v = buck_err(R, D, N)
            ei.append(a)
            ev.append(v)
        print(f"{lab} err iL at period 40: " + " ".join(f"{v:+.2e}" for v in ei) + "  order " + order(ei))
        print(f"{' ' * len(lab)} err vC            : " + " ".join(f"{v:+.2e}" for v in ev) + "  order " + order(ev))
    for N in (100, 400):
        lo, hi, vo = dcm_idle(buck(50.0, 0.3, N))
        print(f"C. DCM idle T/{N}: v(sw) {lo:.3f} .. {hi:.3f} V, v_out {vo:.3f} V, worst {max(abs(lo - vo), abs(hi - vo)) / vo * 100:.2f} %")
    for dt in (1e-8, 1e-9):
        _, lo, hi, P = caploop(dt)
        print(f"D. cap loop dt={dt:.0e}: on-state i_S {lo:+.3f} .. {hi:+.3f} A, i2R {P:.4f} W")
    e = [rc_pulse(1e-6 / n)[1] for n in (10, 20, 40)]
    print("E. RC pulse on the grid, max err tau/10, /20, /40: " + " ".join(f"{v:.3e}" for v in e) + "  order " + order(e))
    for name, fn, ex in first_cases():
        e = [abs(fn(dt)[1] / ex - 1) for dt in (4e-6, 2e-6, 1e-6, 5e-7)]
        print(f"F. {name:8s} rel err: " + " ".join(f"{v:.3e}" for v in e) + "  order " + order(e))
    print("G. Coss Q/(I t) at samples 1, 2, 11, 40: " + " ".join(f"{v:.6f}" for v in coss()[1]))
    e = [sat_edge(dt)[1] for dt in (4e-6, 2e-6, 1e-6, 5e-7)]
    print(f"H. saturable edge err (ref {sat_ref():.9f} A): " + " ".join(f"{v:+.3e}" for v in e) + "  order " + order(e))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if "--on" in sys.argv:
        FLAG["discontinuity_restart"] = True
    if "--off" in sys.argv and cmd == "table":
        FLAG["discontinuity_restart"] = False
    if cmd == "save":
        save(sys.argv[2])
    elif cmd == "check":
        sys.exit(1 if check(sys.argv[2], "--off" in sys.argv) else 0)
    else:
        table()
