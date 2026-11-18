"""Exact ground truth for the floating-bus diode bridge (DECISIONS #19 of
add-branch-form-ideal-switch; KNOWN_LIMITATIONS "TR-BDF2 on a bridge whose
DC side floats on a tie resistor").

Circuit (python/tests/test_trbdf2_rounding_fuzz.py::_bridge, zero initial
state, t_end 40 ms):

    Vac s-gnd 325 sin(2*pi*50 t); Rs s-a 0.5; Ls a-ac1 1 mH;
    D1 ac1->dp, D2 gnd->dp, D3 dn->ac1, D4 dn->gnd (g_on 1e3, g_off 1e-9,
    V_th = V_f = 0); C dp-dn 470 uF; RL dp-dn 100; Rtie dn-gnd 1 MOhm.

Diode model (core/include/pulsim/models/switched_diode.hpp, V_th = V_f = 0):
ON stamps g_on, OFF stamps g_off; ON -> OFF iff g_on*v <= 0, OFF -> ON iff
v >= 0.  A mode is consistent iff every ON diode has v > 0 and every OFF
diode v < 0.  With both thresholds at v = 0 the i(v) law is continuous and
strictly monotone, so the network is a Lipschitz ODE in (iL, vC).

Method (mpmath, no simulator dependency): per mode the state z = [iL, vC,
sin wt, cos wt] obeys z' = F_m z exactly; z(t) = V e^{Lambda t} V^-1 z0.
Every commutation is bracketed on a scan and bisected to 10^-(dps-12) s; at
every event all 16 modes are enumerated and the forward-consistent ones
(leading Taylor sign of each diode voltage matches its state) are reported.
Window integrals (bus average, losses, energy) are exact sums of
exponentials.

    uv run --with mpmath --with numpy --with scipy python \
        benchmarks/tools/bridge_ground_truth.py [--dps 50] [--all]
"""
from __future__ import annotations

import argparse
import itertools
import random
import time

import mpmath as mp

NAMES = ("D1", "D2", "D3", "D4")
T_END = "0.04"
WIN = ("0.03", "0.04")


def params(rtie=1e6, scale=None):
    # The engine's values are doubles: take their exact binary value.
    p = dict(Vm=325.0, f=50.0, Rs=0.5, L=1e-3, C=470e-6, RL=100.0,
             Rt=rtie, gon=1e3, goff=1e-9)
    p = {k: mp.mpf(v) for k, v in p.items()}
    for k, s in (scale or {}).items():
        p[k] *= 1 + s
    return p


class Bridge:
    def __init__(self, p):
        self.p = p
        self.w = 2 * mp.pi * p["f"]
        self.cache = {}

    def mode(self, m):
        """m = 4-tuple of bools (D1..D4 ON). Returns F, diode/node rows, eig."""
        if m in self.cache:
            return self.cache[m]
        p = self.p
        g1, g2, g3, g4 = (p["gon"] if b else p["goff"] for b in m)
        G = g1 + g2 + g3 + g4 + 1 / p["Rt"]
        # KCL at ac1 and over the {dp, dn} supernode, unknowns (v_ac1, v_dn):
        K = mp.matrix([[g1 + g3, -(g1 + g3)], [g1 + g3, -G]])
        B = mp.matrix([[1, g1], [0, g1 + g2]])
        N = mp.lu_solve(K, B[:, 0]), mp.lu_solve(K, B[:, 1])
        vac = [N[0][0], N[1][0]]            # v_ac1 = vac . (iL, vC)
        vdn = [N[0][1], N[1][1]]
        vdp = [vdn[0], vdn[1] + 1]

        def row(r2):
            return mp.matrix([[r2[0], r2[1], 0, 0]])

        def sub(a, b):
            return [a[0] - b[0], a[1] - b[1]]

        vd = [sub(vac, vdp), [-vdp[0], -vdp[1]], sub(vdn, vac), vdn]
        F = mp.zeros(4, 4)
        F[0, 0] = (-p["Rs"] - vac[0]) / p["L"]
        F[0, 1] = -vac[1] / p["L"]
        F[0, 2] = p["Vm"] / p["L"]
        ic = [g1 * vd[0][j] + g2 * vd[1][j] for j in range(2)]
        ic[1] -= 1 / p["RL"]
        F[1, 0], F[1, 1] = ic[0] / p["C"], ic[1] / p["C"]
        F[2, 3], F[3, 2] = self.w, -self.w
        lam, V = mp.eig(F)
        gaps = [abs(a - b) for a, b in itertools.combinations(lam, 2)]
        assert min(gaps) > mp.mpf(10) ** (-mp.mp.dps // 3) * max(abs(x) for x in lam)
        rows = dict(D=[row(r) for r in vd], g=[g1, g2, g3, g4],
                    vdn=row(vdn), vdp=row(vdp))
        out = (F, rows, lam, V, mp.inverse(V))
        self.cache[m] = out
        return out


def expo(mu, h):
    """(e^{mu h} - 1)/mu, exact limit h at mu = 0."""
    if abs(mu * h) < mp.mpf(10) ** (-mp.mp.dps // 2):
        return h * (1 + mu * h / 2)
    return mp.expm1(mu * h) / mu


def forward_sign(F, row, z, ztol):
    """Sign of y(t) = row.z(t) for t -> 0+ in the mode with matrix F."""
    v = z
    for _ in range(8):
        y = (row * v)[0]
        scale = sum(abs(row[0, j] * v[j]) for j in range(4))
        if abs(y) > ztol * scale and abs(y) > 0:
            return 1 if y > 0 else -1
        v = F * v
    return 0


def classify(br, z, ztol):
    """All 16 modes: (mode, static-consistent, forward-consistent, v)."""
    res = []
    for m in itertools.product((False, True), repeat=4):
        F, rows, *_ = br.mode(m)
        stat, fwd, vs = True, True, []
        for d in range(4):
            y = (rows["D"][d] * z)[0]
            scale = sum(abs(rows["D"][d][0, j] * z[j]) for j in range(4))
            vs.append(y)
            want = 1 if m[d] else -1
            if abs(y) > ztol * scale and (1 if y > 0 else -1) != want:
                stat = False
            if forward_sign(F, rows["D"][d], z, ztol) != want:
                fwd = False
        res.append((m, stat, stat and fwd, vs))
    return res


class Seg:
    """Exact propagation inside one mode from (t0, z0)."""

    def __init__(self, br, m, t0, z0):
        self.F, self.rows, self.lam, V, Vi = br.mode(m)
        self.t0, self.m, self.V = t0, m, V
        self.c = Vi * z0

    def coef(self, row):
        return [(row * self.V[:, k])[0] * self.c[k] for k in range(4)]

    def z(self, tau):
        e = [mp.exp(lk * tau) * self.c[k] for k, lk in enumerate(self.lam)]
        return mp.matrix([mp.re(sum(self.V[i, k] * e[k] for k in range(4)))
                          for i in range(4)])

    def integral(self, a, h):
        return mp.re(sum(a[k] * expo(self.lam[k], h) for k in range(4)))

    def integral2(self, a, b, h):
        return mp.re(sum(a[k] * b[j] * expo(self.lam[k] + self.lam[j], h)
                         for k in range(4) for j in range(4)))


def run(p, dps=50, forced=None, verbose=False, scan=mp.mpf("2e-6")):
    """Exact trajectory 0..40 ms. forced: {event index: mode} for branches."""
    mp.mp.dps = dps
    p = {k: +v for k, v in p.items()}
    br = Bridge(p)
    forced = forced or {}
    ztol = mp.mpf(10) ** (-(dps - 12))
    rtol_t = mp.mpf(10) ** (-(dps - 12))
    t, z = mp.mpf(0), mp.matrix([0, 0, 0, 1])
    t_end, w0, w1 = mp.mpf(T_END), mp.mpf(WIN[0]), mp.mpf(WIN[1])
    events, ambiguous = [], []
    acc = dict(bus=0, cm=0, Psrc=0, PRs=0, PRL=0, Ptie=0,
               PD=[0, 0, 0, 0], PDon=[0, 0, 0, 0], ton=[0, 0, 0, 0])
    E0 = None

    def pick(k, t, z):
        cl = classify(br, z, ztol)
        stat = [c for c in cl if c[1]]
        fwd = [c for c in cl if c[2]]
        if len(fwd) != 1:
            ambiguous.append((k, t, [c[0] for c in fwd]))
        if not fwd:
            raise RuntimeError(f"no forward-consistent mode at t={t}")
        m = forced.get(k, fwd[0][0])
        return m, len(stat), len(fwd), [c[0] for c in fwd]

    m, ns, nf, cands = pick(0, t, z)
    events.append(dict(t=t, mode=m, prev=None, n_static=ns, n_fwd=nf, cands=cands))
    while t < t_end:
        seg = Seg(br, m, t, z)
        stop = min(x for x in (w0, w1, t_end) if x > t)
        sgn = [1 if b else -1 for b in m]
        A = [[sgn[d] * a for a in seg.coef(seg.rows["D"][d])] for d in range(4)]
        lam = seg.lam

        AL = [[A[d][k] * lam[k] for k in range(4)] for d in range(4)]

        def g(d, tau):
            return mp.re(sum(A[d][k] * mp.exp(lam[k] * tau) for k in range(4)))

        def dg(d, tau):
            return mp.re(sum(AL[d][k] * mp.exp(lam[k] * tau) for k in range(4)))

        def both(tau):
            e = [mp.exp(x * tau) for x in lam]
            return [(mp.re(sum(A[d][k] * e[k] for k in range(4))),
                     mp.re(sum(AL[d][k] * e[k] for k in range(4)))) for d in range(4)]

        H = stop - t
        grid = [mp.mpf(0)]
        x = mp.mpf("1e-24")
        while x < min(scan, H):
            grid.append(x)
            x *= 4
        x = scan
        while x < H:
            grid.append(x)
            x += scan
        grid.append(H)
        hit = None
        prev = both(grid[0])
        for a, b in zip(grid, grid[1:]):
            best = None
            cur = both(b)
            for d in range(4):
                gb, db = cur[d]
                da = prev[d][1]
                lo = None
                if gb <= 0:
                    lo = (a, b)
                elif da < 0 < db:      # interior minimum: check it is not a hidden double root
                    x0, x1 = a, b
                    while x1 - x0 > rtol_t:
                        xm = (x0 + x1) / 2
                        if dg(d, xm) < 0:
                            x0 = xm
                        else:
                            x1 = xm
                    if g(d, x1) <= 0:
                        lo = (a, x1)
                if lo is not None:
                    x0, x1 = lo
                    while x1 - x0 > rtol_t:
                        xm = (x0 + x1) / 2
                        if g(d, xm) > 0:
                            x0 = xm
                        else:
                            x1 = xm
                    if best is None or x1 < best[0]:
                        best = (x1, d)
            if best is not None:
                hit = best
                break
            prev = cur
        tau = hit[0] if hit else H
        # window integrals over [t, t + tau] intersected with [w0, w1]
        if t >= w0 and t + tau <= w1 + rtol_t:
            r = seg.rows
            cIL = seg.coef(mp.matrix([[1, 0, 0, 0]]))
            cVC = seg.coef(mp.matrix([[0, 1, 0, 0]]))
            cVS = seg.coef(mp.matrix([[0, 0, p["Vm"], 0]]))
            cDN = seg.coef(r["vdn"])
            acc["bus"] += seg.integral(cVC, tau)
            acc["cm"] += seg.integral(cDN, tau) + seg.integral(cVC, tau) / 2
            acc["Psrc"] += seg.integral2(cVS, cIL, tau)
            acc["PRs"] += p["Rs"] * seg.integral2(cIL, cIL, tau)
            acc["PRL"] += seg.integral2(cVC, cVC, tau) / p["RL"]
            acc["Ptie"] += seg.integral2(cDN, cDN, tau) / p["Rt"]
            for d in range(4):
                cD = seg.coef(r["D"][d])
                e = r["g"][d] * seg.integral2(cD, cD, tau)
                acc["PD"][d] += e
                if m[d]:
                    acc["PDon"][d] += e
                    acc["ton"][d] += tau
        if E0 is None and t + tau >= w0 and t >= w0:
            E0 = (p["L"] * z[0] ** 2 + p["C"] * z[1] ** 2) / 2
        z = seg.z(tau)
        t = t + tau
        if hit:
            k = len(events)
            new, ns, nf, cands = pick(k, t, z)
            events.append(dict(t=t, mode=new, prev=m, n_static=ns, n_fwd=nf,
                               cands=cands))
            if verbose:
                print(f"  event {k:2d} t={mp.nstr(t, 25):>28s} "
                      f"{_flips(m, new):24s} static={ns} forward={nf}")
            m = new
    E1 = (p["L"] * z[0] ** 2 + p["C"] * z[1] ** 2) / 2
    T = w1 - w0
    out = dict(events=events, ambiguous=ambiguous,
               bus=acc["bus"] / T, cm=acc["cm"] / T,
               PD=[x / T for x in acc["PD"]], PDon=[x / T for x in acc["PDon"]],
               ton=acc["ton"], Psrc=acc["Psrc"] / T)
    diss = acc["PRs"] + acc["PRL"] + acc["Ptie"] + sum(acc["PD"])
    out["energy_rel"] = (acc["Psrc"] - diss - (E1 - E0)) / acc["Psrc"]
    out["zend"] = z
    return out


def _flips(a, b):
    return " ".join(f"{n}{'+' if y else '-'}" for n, x, y in zip(NAMES, a, b) if x != y) or "-"


def _ncomm(r):
    """Events that changed the mode (a root hit can re-select the same mode)."""
    return sum(e["prev"] is not None and e["prev"] != e["mode"] for e in r["events"])


def scipy_check(rtie):
    """Independent float64 check: full 3-node MNA (ac1, dp, dn) with the
    capacitor as a voltage source, the PWL network solved by trying the 16
    modes, integrated by scipy Radau (the vector field is Lipschitz)."""
    import numpy as np
    from scipy.integrate import solve_ivp

    Vm, w, Rs, L, C, RL = 325.0, 2 * np.pi * 50, 0.5, 1e-3, 470e-6, 100.0
    gon, goff = 1e3, 1e-9
    D = [(0, 1), (-1, 1), (2, 0), (2, -1)]   # (anode, cathode); -1 = ground
    modes = list(itertools.product((False, True), repeat=4))

    def net(iL, vC):
        for m in modes:
            Y = np.zeros((4, 4))
            rhs = np.zeros(4)
            for (a, c), on in zip(D, m):
                gg = gon if on else goff
                for i, j, s in ((a, a, 1), (c, c, 1), (a, c, -1), (c, a, -1)):
                    if i >= 0 and j >= 0:
                        Y[i, j] += s * gg
            Y[2, 2] += 1 / rtie
            Y[1, 1] += 1 / RL
            Y[1, 2] -= 1 / RL
            Y[2, 2] += 1 / RL
            Y[2, 1] -= 1 / RL
            Y[1, 3], Y[2, 3], Y[3, 1], Y[3, 2] = 1, -1, 1, -1
            rhs[0], rhs[3] = iL, vC
            x = np.linalg.solve(Y, rhs)
            v = [(x[a] if a >= 0 else 0) - (x[c] if c >= 0 else 0) for a, c in D]
            if all((vi >= 0) == on or abs(vi) < 1e-12 for vi, on in zip(v, m)):
                return x, -x[3]
        raise RuntimeError("no consistent mode")

    def f(t, y):
        x, ic = net(y[0], y[1])
        return [(Vm * np.sin(w * t) - Rs * y[0] - x[0]) / L, ic / C, y[1]]

    s = solve_ivp(f, (0, 0.04), [0, 0, 0], method="Radau", rtol=1e-11,
                  atol=[1e-12, 1e-10, 1e-14], max_step=2e-6, t_eval=[0.03, 0.04])
    return (s.y[2, 1] - s.y[2, 0]) / 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dps", type=int, default=50)
    ap.add_argument("--all", action="store_true", help="precision, grounded, sensitivity, scipy")
    a = ap.parse_args()
    t0 = time.time()
    r = run(params(), a.dps)
    print(f"[floating, dps {a.dps}, {time.time() - t0:.0f} s]")
    print("events:")
    for k, e in enumerate(r["events"]):
        print(f"  {k:2d} t={mp.nstr(e['t'], 30):>34s}  {_flips(e['prev'] or (False,) * 4, e['mode']):18s}"
              f" static-consistent={e['n_static']:2d} forward-consistent={e['n_fwd']}")
    print("ambiguous events:", r["ambiguous"] or "none")
    print("bus avg 30-40 ms (dp-dn):", mp.nstr(r["bus"], 30))
    print("common mode avg (dp+dn)/2:", mp.nstr(r["cm"], 20))
    for d in range(4):
        print(f"  {NAMES[d]} avg loss {mp.nstr(r['PD'][d], 20)} W "
              f"(ON part {mp.nstr(r['PDon'][d], 20)} W, ON time {mp.nstr(r['ton'][d], 12)} s)")
    print("energy balance rel:", mp.nstr(r["energy_rel"], 5))
    if not a.all:
        return
    for d2 in (30, 60):
        t0 = time.time()
        r2 = run(params(), d2)
        print(f"dps {d2}: bus {mp.nstr(r2['bus'], 30)}  diff {mp.nstr(r2['bus'] - r['bus'], 3)}"
              f"  commutations {_ncomm(r2)} (root hits {len(r2['events']) - 1})  [{time.time() - t0:.0f} s]")
    mp.mp.dps = a.dps
    rg = run(params(rtie=1e-3), a.dps)
    print("grounded (Rtie 1 mOhm): bus", mp.nstr(rg["bus"], 20),
          "commutations", _ncomm(rg), "energy rel", mp.nstr(rg["energy_rel"], 3))
    rnd = random.Random(20261118)
    keys = ("Vm", "Rs", "L", "C", "RL", "Rt")
    for eps in (1e-15, 1e-9):
        for _ in range(4 if eps == 1e-15 else 2):
            sc = {k: mp.mpf(eps) * rnd.choice((-1, 1)) for k in keys}
            rs = run(params(scale=sc), a.dps)
            print(f"perturb {eps:g} {''.join('+' if v > 0 else '-' for v in sc.values())}: "
                  f"bus {mp.nstr(rs['bus'], 20)} d={mp.nstr(rs['bus'] - r['bus'], 3)} "
                  f"commutations {_ncomm(rs)} ambiguous {len(rs['ambiguous'])}")
    for rt in (1e6, 1e-3):
        print(f"scipy Radau (float64, independent MNA) Rtie={rt:g}: bus {scipy_check(rt):.9f}")


if __name__ == "__main__":
    main()
