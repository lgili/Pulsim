# numpy/scipy only. Run from this directory. `python3 shooting_dcm.py pade` repeats it with scipy's expm.
# Prototype for add-periodic-steady-state-shooting and add-parameter-sensitivities.
# The audit's DCM buck (p2_buck.py): Vin=24, L=100u, C=47u, g_on=1e3, g_off=1e-9, 100 kHz, R=50, D=0.3.
# Exact per-mode propagators (expm), Newton shooting on the true state x=[iL, vC], saltation at the
# diode turn-off, forward parameter sensitivities with the event jump correction.
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
np.set_printoptions(precision=10, linewidth=160)
Vin, C, gon, goff, T = 24.0, 47e-6, 1e3, 1e-9, 1e-5
P0 = dict(L=100e-6, R=50.0, D=0.3)

def AB(p, gs, gd):
    k = 1/(gs+gd); L, R = p['L'], p['R']
    return np.array([[-k/L, -1/L], [1/C, -1/(R*C)]]), np.array([gs*Vin*k/L, 0.0])
def dAB(p, gs, gd, name):
    k = 1/(gs+gd); L, R = p['L'], p['R']
    if name == 'L': return np.array([[k/L**2, 1/L**2], [0, 0]]), np.array([-gs*Vin*k/L**2, 0.0])
    if name == 'R': return np.array([[0, 0], [0, 1/(R**2*C)]]), np.zeros(2)
    return np.zeros((2, 2)), np.zeros(2)
USE_EIG = True
def _expm_aug(M, h, dM=None):
    """exp(M h) (and its derivative along dM at fixed h) through the eigendecomposition (the dsed formula)."""
    lam, V = np.linalg.eig(M); Vi = np.linalg.inv(V); e = np.exp(lam*h)
    E = (V*e)@Vi
    if dM is None: return E.real, None
    G = Vi@dM@V; n = len(lam); F = np.empty((n, n), complex)
    for i in range(n):
        for j in range(n):
            d = lam[i]-lam[j]
            F[i, j] = h*e[i] if abs(d*h) < 1e-12 else (e[i]-e[j])/d      # Daleckii-Krein divided differences
    return E.real, (V@(F*G)@Vi).real
def seg(p, gs, gd, x, h, s=None, name=None):
    """advance x by h; if s given also advance s = dx/dp at fixed h. returns x1, Phi, s1"""
    A, b = AB(p, gs, gd); M = np.zeros((3, 3)); M[:2, :2] = A; M[:2, 2] = b
    if s is None:
        E = _expm_aug(M, h)[0] if USE_EIG else expm(M*h)
        return E[:2, :2]@x + E[:2, 2], E[:2, :2], None
    dA, db = dAB(p, gs, gd, name); dM = np.zeros((3, 3)); dM[:2, :2] = dA; dM[:2, 2] = db
    if USE_EIG:
        E, D = _expm_aug(M, h, dM)
    else:
        Z = np.zeros((6, 6)); Z[:3, :3] = M; Z[3:, 3:] = M; Z[:3, 3:] = dM
        E6 = expm(Z*h); E = E6[:3, :3]; D = E6[:3, 3:]
    return E[:2, :2]@x + E[:2, 2], E[:2, :2], E[:2, :2]@s + D[:2, :2]@x + D[:2, 2]
def f(p, gs, gd, x):
    A, b = AB(p, gs, gd); return A@x + b
gx = np.array([1.0, 0.0]); i_thr = goff*Vin          # diode current zero  <=>  iL = g_off*Vin

def period(p, x, s=None, name=None, salt=True):
    """one period from the rising gate edge. returns x(T), monodromy, s(T), tau/T"""
    D = p['D']; Mono = np.eye(2)
    # 1) S on, D off, duration D*T (time event; its instant depends on p only when name == 'D')
    f_minus = None
    x, Ph, s = seg(p, gon, goff, x, D*T, s, name); Mono = Ph@Mono
    if s is not None and name == 'D':                # time-event jump: dtau/dD = T
        s = s + (f(p, gon, goff, x) - f(p, goff, gon, x))*T
    # 2) S off, D on until iL = i_thr (state event) or the period ends (CCM)
    rem = (1-D)*T
    g = lambda h: seg(p, goff, gon, x, h)[0][0] - i_thr
    if x[0] > i_thr and g(rem) < 0:
        h2 = brentq(g, 0.0, rem, xtol=1e-22, rtol=1e-15)
        x, Ph, s = seg(p, goff, gon, x, h2, s, name); Mono = Ph@Mono
        fm, fp = f(p, goff, gon, x), f(p, goff, goff, x)
        if salt:
            S = np.eye(2) + np.outer(fp - fm, gx)/(gx@fm); Mono = S@Mono
            if s is not None:
                dtau = -(gx@s)/(gx@fm)
                s = s + (fm - fp)*dtau
        # 3) S off, D off for the rest
        x, Ph, s = seg(p, goff, goff, x, rem-h2, s, name); Mono = Ph@Mono
        tau = D + h2/T
    else:
        x, Ph, s = seg(p, goff, gon, x, rem, s, name); Mono = Ph@Mono; tau = None
    return x, Mono, s, tau

def shoot(p, x0=None, salt=True, tol=1e-12, itmax=50, verbose=False):
    x = np.zeros(2) if x0 is None else x0.copy(); hist = []
    for k in range(itmax):
        x1, Mono, _, tau = period(p, x, salt=salt)
        r = x1 - x; hist.append(np.abs(r).max())
        if verbose: print(f"   it {k}: |P(x)-x|={hist[-1]:.3e} x={x} tau/T={tau}")
        if hist[-1] < tol: break
        x = x + np.linalg.solve(np.eye(2) - Mono, r)
    return x, Mono, hist

import sys
if len(sys.argv) > 1 and sys.argv[1] == "pade": USE_EIG = False
print("propagator:", "eigendecomposition" if USE_EIG else "scipy expm (Pade scaling-and-squaring)")
print("== DCM buck R=50 D=0.3: Newton shooting with saltation, start from x=0")
xs, Mono, hist = shoot(P0, verbose=True)
print("   x* =", xs, " residual history", ["%.1e" % h for h in hist])
# brute force
x = np.zeros(2)
for n in range(4000): x = period(P0, x)[0]
print("   brute force 4000 periods:", x, " diff", np.abs(x-xs).max())
# monodromy check by finite difference of the exact map
eps = 1e-7; Mfd = np.zeros((2, 2))
for j in range(2):
    e = np.zeros(2); e[j] = eps
    Mfd[:, j] = (period(P0, xs+e)[0] - period(P0, xs-e)[0])/(2*eps)
print("   monodromy (saltation):\n", Mono, "\n   FD of the exact map:\n", Mfd)
print("   Floquet multipliers with saltation:", np.linalg.eigvals(Mono), " FD:", np.linalg.eigvals(Mfd))
print("   Floquet multipliers WITHOUT saltation:", np.linalg.eigvals(period(P0, xs, salt=False)[1]))
xs2, _, hist2 = shoot(P0, salt=False, itmax=200)
print(f"   Newton without saltation: {len(hist2)} iterations, final residual {hist2[-1]:.1e}, x={xs2}")
print("== CCM buck R=1 D=0.5")
Pc = dict(L=100e-6, R=1.0, D=0.5); xc, Mc, hc = shoot(Pc)
print("   x* =", xc, "iterations", len(hc), "multipliers", np.linalg.eigvals(Mc), "|.|", abs(np.linalg.eigvals(Mc)))

print("== steady-state gradients dx*/dp = (I - M)^-1 dP/dp  vs central finite difference of the PSS solve")
for name in ['L', 'R', 'D']:
    _, M_, sT, _ = period(P0, xs, np.zeros(2), name)
    grad = np.linalg.solve(np.eye(2) - M_, sT)
    h = 1e-5*P0[name]; pp = dict(P0); pm = dict(P0); pp[name] += h; pm[name] -= h
    fd = (shoot(pp, xs)[0] - shoot(pm, xs)[0])/(2*h)
    print(f"   d x*/d{name}: analytic {grad}  FD {fd}  rel diff {np.abs(grad-fd).max()/np.abs(fd).max():.2e}")

print("== transient sensitivity d x(40T)/dp from x(0)=0, with and without the event jump correction")
for name in ['L', 'R', 'D']:
    def run(p, sens, salt=True):
        x = np.zeros(2); s = np.zeros(2) if sens else None
        for n in range(40): x, _, s, _ = period(p, x, s, name if sens else None, salt=salt)
        return x, s
    _, s_ok = run(P0, True); _, s_no = run(P0, True, salt=False)
    h = 1e-5*P0[name]; pp = dict(P0); pm = dict(P0); pp[name] += h; pm[name] -= h
    fd = (run(pp, False)[0] - run(pm, False)[0])/(2*h)
    print(f"   d/d{name}: with jump {s_ok}  no jump {s_no}  FD {fd}  rel err with={np.abs(s_ok-fd).max()/np.abs(fd).max():.2e} without={np.abs(s_no-fd).max()/np.abs(fd).max():.2e}")
