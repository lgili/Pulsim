"""Truths for the nonlinear-charge tests of core/tests/step/test_esdirk43.cpp
(no Pulsim code involved).

Saturable inductor, flux linkage (smooth, monotone, cancellation-free):
    lam(i)  = Ls*i + (L0 - Ls)*Is*tanh(i/Is),   Ls = L0/k
    lam'(i) = Ls + (L0 - Ls)/cosh(i/Is)^2      (falls by k in saturation)
Junction-type Coss (m = 1/2), charge written without cancellation:
    qc(v) = 2*C0*v/(sqrt(1 + v/phi) + 1),   qc'(v) = C0/sqrt(1 + v/phi)

A. "sat_series": Vs = 10 sin(2 pi 50 t) - R 1 - saturable L (L0 10 mH,
   Is 1 A) to ground, from rest; k = 10 and k = 1000 (inrush: the flux runs
   to about 6x its knee, the current to ~9 A, lam' collapses by k).
       di/dt = (Vs - R i)/lam'(i)
B. "nl_tank": Vdc 10 - Rs 0.1 - node m - saturable L (L0 100 uH, Is 1 A,
   k 100) - node v - Coss (C0 10 uF, phi 0.7) || Rp 100 to ground, from
   rest; a current limiter Ilim*tanh(vm/V0) (2 A, 5 V) from m to ground
   makes the algebraic vm a NONLINEAR function of iL:
       (vm - Vdc)/Rs + i + Ilim tanh(vm/V0) = 0   (solved by brentq)
       di/dt = (vm - v)/lam'(i),  dv/dt = (i - v/Rp)/qc'(v)
References: scipy Radau and LSODA at rtol 1e-13 (their spread is printed:
it is the reference's own error); mpmath odefun (Taylor, 25 digits) on A
with k = 10 as a third, independent integrator.
Run: python3 esdirk43_nonlinear_q.py   (mpmath, numpy, scipy)
"""
import math

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def dlam(i, L0, Is, k):
    Ls = L0 / k
    return Ls + (L0 - Ls) / math.cosh(i / Is) ** 2


def solve(rhs, y0, ts, scale):
    out = []
    for method in ("Radau", "LSODA"):
        s = solve_ivp(rhs, (0, ts[-1]), y0, method=method, rtol=1e-13,
                      atol=1e-16 * np.asarray(scale), t_eval=ts)
        assert s.success, s.message
        out.append(s.y)
    return out


W, V, R, L0A, IsA = 2 * math.pi * 50, 10.0, 1.0, 10e-3, 1.0
tsA = [5e-3, 10e-3, 15e-3, 20e-3]
for k in (10, 1000):
    def rhsA(t, y, k=k):
        return [(V * math.sin(W * t) - R * y[0]) / dlam(y[0], L0A, IsA, k)]
    ra, rl = solve(rhsA, [0.0], tsA, [10.0])
    print(f"A k={k}: spread Radau-LSODA {np.max(np.abs(ra - rl)):.2e} A")
    for t, i in zip(tsA, ra[0]):
        print(f"  t={t:g}: iL={i!r}  u=i/Is={i / IsA:.2f}")
    dense = solve_ivp(rhsA, (0, 0.02), [0.0], method="Radau", rtol=1e-10, atol=1e-14,
                      max_step=1e-5, dense_output=True)
    tt = np.linspace(0, 0.02, 4001)
    print(f"  max |iL| {np.max(np.abs(dense.sol(tt)[0])):.3f} A over one period")

# independent third integrator (Taylor series, 25 digits) on A, k = 10
mp.mp.dps = 25
k = 10
f = mp.odefun(lambda t, i: (V * mp.sin(W * t) - R * i)
              / (L0A / k + (L0A - L0A / k) / mp.cosh(i / IsA) ** 2), 0, mp.mpf(0))
for t in tsA[:2]:
    print(f"A k=10 mpmath odefun t={t:g}: iL={mp.nstr(f(mp.mpf(t)), 18)}")

Vdc, Rs, L0B, IsB, kB, C0, phi, Rp = 10.0, 0.1, 100e-6, 1.0, 100, 10e-6, 0.7, 100.0
Ilim, V0 = 2.0, 5.0
tsB = [50e-6, 100e-6, 150e-6, 200e-6]


def vm_of(i):
    return brentq(lambda u: (u - Vdc) / Rs + i + Ilim * math.tanh(u / V0), -50.0, 50.0, xtol=1e-15)


def rhsB(t, y):
    i, v = y
    return [(vm_of(i) - v) / dlam(i, L0B, IsB, kB), (i - v / Rp) / (C0 / math.sqrt(1 + v / phi))]


rb, lb = solve(rhsB, [0.0, 0.0], tsB, [10.0, 10.0])
print(f"B: spread Radau-LSODA iL {np.max(np.abs(rb[0] - lb[0])):.2e} A, v {np.max(np.abs(rb[1] - lb[1])):.2e} V")
for t, i, v in zip(tsB, rb[0], rb[1]):
    print(f"  t={t:g}: iL={i!r} v={v!r} vm={vm_of(i)!r}")
dense = solve_ivp(rhsB, (0, 2e-4), [0.0, 0.0], method="Radau", rtol=1e-10, atol=1e-14,
                  max_step=1e-7, dense_output=True)
tt = np.linspace(0, 2e-4, 4001)
y = dense.sol(tt)
print(f"  max iL {np.max(y[0]):.3f} A (u = {np.max(y[0]) / IsB:.2f}), v in [{np.min(y[1]):.3f}, {np.max(y[1]):.3f}] V")

# ---- C. "sat_square": circuit A (k = 1000) driven by a +-10 V square wave,
# edges every 5 ms (the source flips while the core is deep in saturation).
# Piecewise-smooth: integrated segment by segment between the edges.
def square_run(k, t_end, ts):
    y, out = [0.0], {}
    for n in range(int(round(t_end / 5e-3))):
        a, b = n * 5e-3, (n + 1) * 5e-3
        lvl = 10.0 if n % 2 == 0 else -10.0
        te = [t for t in ts if a < t <= b]
        res = []
        for method in ("Radau", "LSODA"):
            s = solve_ivp(lambda t, z: [(lvl - R * z[0]) / dlam(z[0], L0A, IsA, k)], (a, b), y,
                          method=method, rtol=1e-13, atol=1e-15, t_eval=sorted(set(te + [b])))
            assert s.success, s.message
            res.append(s)
        for t, i0, i1 in zip(res[0].t, res[0].y[0], res[1].y[0]):
            out[t] = (i0, abs(i0 - i1))
        y = [res[0].y[0][-1]]
    return out


tsC = [5.2e-3, 5.5e-3, 6e-3, 6.5e-3, 7e-3, 10.5e-3, 11e-3, 20e-3]
o = square_run(1000, 20e-3, tsC)
print("C sat_square k=1000 (Radau; |Radau-LSODA|):")
for t in tsC:
    print(f"  t={t:g}: iL={o[min(o, key=lambda u: abs(u - t))][0]!r}  spread {o[min(o, key=lambda u: abs(u - t))][1]:.1e}")

# ---- D. Synchronous buck at 100 kHz, D = 0.4, ideal switch node (a PWL
# source Vin / 0), L 10 uH + RL 10 mOhm, C 10 uF || R 1 Ohm, from rest.
# Linear between edges: exact by mpmath expm, 30 digits.
mp.mp.dps = 30
Vin, Lb, RLb, Cb, Rb, Tsw, Dty = 12, mp.mpf("1e-5"), mp.mpf("0.01"), mp.mpf("1e-5"), 1, mp.mpf("1e-5"), mp.mpf("0.4")
Ab = mp.matrix([[-RLb / Lb, -1 / Lb], [1 / Cb, -1 / (Rb * Cb)]])


def seg(z, lvl, dt):
    bvec = mp.matrix([lvl / Lb, 0])
    zss = -mp.lu_solve(Ab, bvec)
    return zss + mp.expm(Ab * dt) * (z - zss)


z = mp.matrix([0, 0])
print("D buck (iL, vo) at the end of periods 5, 10, 15, 20:")
for n in range(1, 21):
    z = seg(z, Vin, Dty * Tsw)
    z = seg(z, 0, (1 - Dty) * Tsw)
    if n % 5 == 0:
        print(f"  t={n * 10}us: iL={mp.nstr(z[0], 20)} vo={mp.nstr(z[1], 20)}")

# ---- E. sat_series reference states for one-step dense-output errors:
# iL at t0 = 2.6 ms and at t0 + h/2 (Radau rtol 1e-13; LSODA spread).
t0 = 2.6e-3
for k in (10, 1000):
    def rhsA(t, y, k=k):
        return [(V * math.sin(W * t) - R * y[0]) / dlam(y[0], L0A, IsA, k)]
    te = [t0] + [t0 + 0.5 * h for h in (4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5)]
    ra, rl = solve(rhsA, [0.0], sorted(te), [10.0])
    print(f"E k={k}: spread {np.max(np.abs(ra - rl)):.1e}; t, iL:")
    for t, i in zip(sorted(te), ra[0]):
        print(f"  {{{t!r}, {i!r}}},")
