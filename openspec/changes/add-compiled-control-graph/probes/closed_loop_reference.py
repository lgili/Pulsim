"""Independent reference for the closed-loop benchmark circuits (G23).

Outside pulsim: between the exact gate edges, each PWL mode's flow by
scipy.linalg.expm (diode crossings by bisection on that flow, in the
leaving direction), a Shockley diode by Radau (rtol 1e-12) with the switch
node solved by brentq; the PI law re-typed from pulsim.PIController; ticks
at kT on x(kT) (state time), the duty held from kT, gate ON for
(t - kT)/T < d. A v1 file without `uic` starts from the engines' DC
operating point, solved here (fsolve on the ON mode; the engines' start is
printed as a cross-check). Compares v(out)
and the duty at every tick for each engine configuration.
`OPAQUE(before)` hides the gate's edges (the engines before G23).

  PYTHONPATH=<repo>/python:<repo>/benchmarks/tools \\
      python closed_loop_reference.py ll14_buck_closed_loop

Measured 2026-11-18 (macOS arm64), max |v(kT) - ref| / span:
  circuit                 trbdf2 1e-10  1e-8    default  pwl     pwl+split (before)  dt/16 split
  cl_buck_pi              1.7e-8        3.6e-7  4.2e-5   3.6e-2  1.4e-3 (4.5e-2)     8.5e-5
  cl_buck_aggressive_pi   3.0e-8        6.4e-7  6.9e-5   1.4e-2  1.9e-2 (2.9e-2)     1.2e-3
  cl_boost_pi             1.1e-7        9.8e-7  5.7e-5   2.0e-1  4.2e-4 (not run)    2.4e-5
  ll14_buck_closed_loop   5.0e-8        1.6e-7  1.1e-5   4.1e-2  3.0e-4 (4.0e-2)     1.8e-5
  ll15_boost_closed_loop  2.9e-8        5.0e-7  3.9e-5   3.0e-2  9.9e-6 (3.0e-2)     6.8e-7
pwl converges at first order (label time: the helper samples x(kT - dt)).
cl_boost_pi before: trbdf2 hit max_steps; OPAQUE pwl+split is skipped (a
Python gate x 1e7 steps). Only v(out) and the duty are compared here; the
V(sw) / I(L) figures in G23 are engine vs trbdf2 rtol 1e-10.
"""
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq


VT = 0.025852


class PI:
    def __init__(s, kp, ki, lo, hi):
        s.kp, s.ki, s.lo, s.hi = kp, ki, lo, hi
        s.i = s.e = 0.0

    def update(s, sp, m, dt):
        e = sp - m
        ni = s.i + s.ki * dt * 0.5 * (e + s.e)
        u = s.kp * e + ni
        if u > s.hi and e > 0:
            ni = s.i; u = s.kp * e + ni
        elif u < s.lo and e < 0:
            ni = s.i; u = s.kp * e + ni
        u = max(s.lo, min(s.hi, u))
        s.i, s.e = ni, e
        return u


def root(f, a, b):
    """Root of a decreasing f; the bracket grows on the side that misses it
    (a floating node: v = vin - iL/g_off can sit far outside the rails)."""
    w = 1.0
    while f(b) > 0:
        b += w; w *= 4
    w = 1.0
    while f(a) < 0:
        a -= w; w *= 4
    return brentq(f, a, b, xtol=1e-14, rtol=8.9e-16, maxiter=500)


def shockley(p, v):
    return p["Is"] * math.expm1(min(v / (p["n"] * VT), 700.0)) + v * 1e-12


def node_v(p, on, iL, dstate):
    """Switch-node voltage. buck: switch vin-sw, diode 0->sw, iL leaves sw.
    boost: switch sw-0, diode sw->out, iL enters sw (needs vC)."""
    Gs = p["gon"] if on else p["goff"]
    if p["topo"] == "buck":
        if p["diode"] == "pwl":
            Gd = p["gd_on"] if dstate else p["gd_off"]
            return (Gs * p["vin"] - iL) / (Gs + Gd)
        f = lambda v: Gs * (p["vin"] - v) + shockley(p, -v) - iL
        return root(f, -3.0, p["vin"] + 3.0)
    raise NotImplementedError


def rhs(p, on, dstate):
    def f(t, x):
        iL, vC = x
        if p["topo"] == "buck":
            v = node_v(p, on, iL, dstate)
            return [(v - iL * p["rL"] - vC) / p["L"], (iL - vC / p["R"]) / p["C"]]
        # boost: L from vin to sw (series rL), switch sw-0, diode sw->out
        Gs = p["gon"] if on else p["goff"]
        if p["diode"] == "pwl":
            Gd = p["gd_on"] if dstate else p["gd_off"]
            vsw = (iL + Gd * vC) / (Gs + Gd)
            idio = Gd * (vsw - vC)
        else:
            g = lambda v: iL - Gs * v - shockley(p, v - vC)
            vsw = root(g, -3.0, vC + 3.0)
            idio = shockley(p, vsw - vC)
        return [(p["vin"] - iL * p["rL"] - vsw) / p["L"], (idio - vC / p["R"]) / p["C"]]
    return f


def dthr(p, on, x):
    """PWL diode: (signal whose sign is the diode state)."""
    Gs = p["gon"] if on else p["goff"]
    if p["topo"] == "buck":
        return x[0] - Gs * p["vin"]          # ON iff iL > Gs vin
    return x[0] - Gs * x[1]                  # boost: ON iff iL > Gs vC


def affine(p, on, ds):
    f = rhs(p, on, ds)
    c = np.asarray(f(0, np.zeros(2)))
    A = np.column_stack([np.asarray(f(0, e)) - c for e in np.eye(2)])
    M = np.zeros((3, 3)); M[:2, :2] = A; M[:2, 2] = c
    return M


def pwl_segment(p, on, ds, x, a, b, nsub=32):
    """Exact (expm) flow of the linear mode from a toward b; stops at the
    first crossing of the diode signal in the leaving direction."""
    M = affine(p, on, ds)
    sgn = 1.0 if ds else -1.0          # ON: leave when signal < 0
    hs = (b - a) / nsub
    E = expm(M * hs)
    z = np.append(x, 1.0)
    for i in range(nsub):
        zn = E @ z
        if sgn * dthr(p, on, zn[:2]) < 0:
            lo, hi = 0.0, hs
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if not (lo < mid < hi):
                    break
                if sgn * dthr(p, on, (expm(M * mid) @ z)[:2]) < 0:
                    hi = mid
                else:
                    lo = mid
            zt = expm(M * hi) @ z
            return zt[:2], a + i * hs + hi, True
        z = zn
    return z[:2], b, False


def run(p, t_end, dense=False):
    T = p["T"]
    pi = PI(p["kp"], p["ki"], p["lo"], p["hi"])
    x = np.array([p.get("iL0", 0.0), p.get("vC0", 0.0)])
    ticks, tt, xx = [], [], []
    k = 0
    while k * T < t_end - 1e-15:
        tk = k * T
        d = pi.update(p["sp"], x[1], T)
        ticks.append((tk, d, x[1], x[0]))
        segs = [(tk, tk + d * T, True), (tk + d * T, (k + 1) * T, False)] if d > 0 else [(tk, (k + 1) * T, False)]
        for a, b, on in segs:
            b = min(b, t_end)
            if b <= a:
                continue
            if p["diode"] == "shockley":
                sol = solve_ivp(rhs(p, on, False), (a, b), x, method="Radau",
                                rtol=1e-12, atol=1e-14, dense_output=dense)
                x = sol.y[:, -1].copy()
                if dense: tt.append(sol.t); xx.append(sol.y)
                continue
            ds = dthr(p, on, x) > 0
            while a < b:
                x, a, flipped = pwl_segment(p, on, ds, x, a, b)
                if flipped:
                    ds = not ds
        k += 1
    out = np.array(ticks)
    if dense:
        return out, np.concatenate(tt), np.concatenate(xx, axis=1)
    return out


P = {
    "cl_buck_pi": dict(topo="buck", diode="pwl", vin=10., gon=100., goff=1e-6, gd_on=100., gd_off=1e-6,
                       rL=0.25, L=10e-6, C=100e-6, R=10., T=1e-5, kp=0.05, ki=200., lo=0., hi=1., sp=5.),
    "cl_buck_aggressive_pi": dict(topo="buck", diode="pwl", vin=12., gon=100., goff=1e-6, gd_on=100.,
                                  gd_off=1e-6, rL=0.25, L=10e-6, C=47e-6, R=5., T=1e-5, kp=0.15, ki=800.,
                                  lo=0.05, hi=0.95, sp=7.),
    "ll14_buck_closed_loop": dict(topo="buck", diode="shockley", Is=1e-12, n=1.2, vin=24., gon=20.,
                                  goff=1e-7, rL=0., L=100e-6, C=220e-6, R=10., T=1e-5, kp=0.02, ki=40.,
                                  lo=0.1, hi=0.9, sp=12.),
    "ll15_boost_closed_loop": dict(topo="boost", diode="shockley", Is=1e-12, n=1.2, vin=12., gon=50.,
                                   goff=1e-7, rL=0., L=50e-6, C=100e-6, R=20., T=5e-6, kp=0.005, ki=10.,
                                   lo=0.05, hi=0.85, sp=24.),
    "cl_boost_pi": dict(topo="boost", diode="pwl", vin=5., gon=100., goff=1e-6, gd_on=100., gd_off=1e-6,
                        rL=0.1, L=47e-6, C=470e-6, R=40., T=1e-5, kp=0.005, ki=50., lo=0., hi=0.9, sp=10.,
                        vC0=10.),
}


def main():
    import pulsim
    import v1_reader
    warnings.simplefilter("ignore")
    name = sys.argv[1] if len(sys.argv) > 1 else "cl_buck_pi"
    p = dict(P[name])
    root = Path(__file__).resolve().parents[4] / "benchmarks" / "circuits" / "closed_loop"
    case = v1_reader.load(root / f"{name}.yaml")
    meas = v1_reader._node([c for c in case._components if c["type"] == "pi_controller"][0]["nodes"][1])
    if not case.uic:   # the v1 file starts from the DC operating point
        # solved here (switch ON, the mask at t = 0; Shockley diode), and
        # only cross-checked against the engines' start
        from scipy.optimize import fsolve
        x0 = fsolve(lambda x: rhs(p, True, False)(0.0, x), [p["vin"] / p["R"], p["vin"]],
                    xtol=1e-14)
        r0 = case.run("pwl")
        ind = [c["name"] for c in case._components if c["type"] == "inductor"][0]
        eng = (float(r0.i(ind)[0]), float(r0.v(meas)[0]))
        print(f"DC start: iL {x0[0]:.9g} vC {x0[1]:.9g} (engines {eng[0]:.9g} {eng[1]:.9g})")
        p["iL0"], p["vC0"] = float(x0[0]), float(x0[1])
    t0 = time.time()
    ref = run(p, case.t_end)
    print(f"reference: {len(ref)} ticks in {time.time() - t0:.1f} s")
    tk, dk, vk = ref[:, 0], ref[:, 1], ref[:, 2]
    span = np.ptp(vk)

    loops = []
    bind = pulsim.bind_pi_to_switch

    def spy(*a, **k):
        loops.append(bind(*a, **k))
        return loops[-1]
    pulsim.bind_pi_to_switch = spy
    combine = pulsim.make_combined_switch_fn

    def opaque(n, parts):
        f = combine(n, parts)
        return lambda t: f(t)

    split = dict(discontinuity_restart=True, enable_substep_state_correction=True)
    cfgs = [("trbdf2 1e-10", "trbdf2", dict(rtol=1e-10, atol=1e-14)),
            ("trbdf2 1e-8", "trbdf2", dict(rtol=1e-8, atol=1e-12)),
            ("trbdf2 default", "trbdf2", {}),
            ("pwl", "pwl", {}), ("pwl+restart", "pwl", dict(discontinuity_restart=True)),
            ("pwl+split", "pwl", split),
            ("pwl dt/4 split", "pwl", dict(dt=case.dt / 4, **split)),
            ("pwl dt/16 split", "pwl", dict(dt=case.dt / 16, **split)),
            ("pwl dt/4 restart", "pwl", dict(dt=case.dt / 4, discontinuity_restart=True)),
            ("pwl dt/16 restart", "pwl", dict(dt=case.dt / 16, discontinuity_restart=True))]
    if name != "cl_boost_pi":   # its opaque trbdf2 run grinds to max_steps
        cfgs = [("trbdf2 1e-8 OPAQUE(before)", "trbdf2", dict(rtol=1e-8, atol=1e-12)),
                ("pwl+split OPAQUE(before)", "pwl", split)] + cfgs
    print(f"{'config':28s} {'max|dv|/span':>13s} {'rms dv/span':>12s} {'max|dd|':>10s}")
    for label, eng, kw in cfgs:
        pulsim.make_combined_switch_fn = opaque if "OPAQUE" in label else combine
        try:
            res = case.run(eng, **kw)
        except Exception as ex:  # noqa: BLE001 - reported
            print(f"{label:28s} {type(ex).__name__}: {str(ex)[:100]}")
            continue
        t, v = np.asarray(res.times), np.asarray(res.v(meas))
        ve = v[np.searchsorted(t, tk * (1 + 1e-12), side="right") - 1]
        d = np.asarray(list(loops[-1].duty_history))[:, 1]
        n = min(len(d), len(dk))
        dv = np.abs(ve - vk) / span
        print(f"{label:28s} {dv.max():13.3e} {np.sqrt(np.mean(dv ** 2)):12.3e} "
              f"{np.abs(d[:n] - dk[:n]).max():10.3e}", flush=True)


if __name__ == "__main__":
    main()
