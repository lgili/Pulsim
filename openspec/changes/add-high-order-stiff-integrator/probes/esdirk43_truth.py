"""Truths for core/tests/step/test_esdirk43.cpp (no Pulsim code involved).

1. Stiff ladder (time-constant spread 1e9): exact solution by mpmath expm
   at 40 digits, after a 10 V step of the source at t = 0 from rest.
       s-a Rsrc 0.1; a-gnd Ca 1 pF (tau 1e-13 s); a-b L 100 uH;
       b-gnd C 100 nF; b-gnd Rload 50; b-c Rsn 1; c-gnd Csn 10 fF (tau 1e-14 s)
   differential state z = [va, vb, vc, iL]; vs = 10, iV = -(vs - va)/Rsrc.
2. Index-1 RC network: the closed form used by the C++ test, checked
   against mpmath odefun (Taylor, 30 digits).
3. Diode clamp (Shockley, smooth): Vs = 5 sin(2 pi 1e3 t) - R 10 - out;
   out: C 10 uF, RL 100, diode Is 1e-12, n*Vt 0.0388 to ground. Reference by
   scipy Radau and DOP853 at rtol 1e-13 (their spread is printed: it is
   the reference's own error).
Run: python3 esdirk43_truth.py   (mpmath, numpy, scipy)
"""
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

mp.mp.dps = 40
# ---- 1. ladder ----
Rs, Ca, L, C, RL, Rsn, Csn, V = map(mp.mpf, ["0.1", "1e-12", "1e-4", "1e-7", "50", "1", "1e-14", "10"])
A = mp.matrix([[-1/(Rs*Ca), 0, 0, -1/Ca],
               [0, -(1/Rsn + 1/RL)/C, 1/(Rsn*C), 1/C],
               [0, 1/(Rsn*Csn), -1/(Rsn*Csn), 0],
               [1/L, -1/L, 0, 0]])
bb = mp.matrix([V/(Rs*Ca), 0, 0, 0])
zss = -mp.lu_solve(A, bb)
print("ladder eigenvalues:", [mp.nstr(e, 6) for e in mp.eig(A)[0]])
for t in ["1e-7", "1e-6", "1e-5", "2e-5", "5e-5"]:
    z = zss + mp.expm(A*mp.mpf(t))*(-zss)
    print(f"ladder t={t}: va={mp.nstr(z[0], 20)} vb={mp.nstr(z[1], 20)} vc={mp.nstr(z[2], 20)} iL={mp.nstr(z[3], 20)}")

# ---- 2. RC network closed form vs odefun ----
mp.mp.dps = 30
R1, R2, R3, Cn, w = 1, 2, 3, 1, mp.mpf(3)
S = mp.mpf(1)/R1 + mp.mpf(1)/R2 + mp.mpf(1)/R3
a = (1 - 1/(R2*S))/(R2*Cn); beta = 1/(R1*S*R2*Cn)
def vp(t):
    return beta*(a*mp.sin(w*t) - w*mp.cos(w*t))/(a**2 + w**2)
v30 = mp.mpf("0.5")
def closed(t):
    return vp(t) + (v30 - vp(0))*mp.e**(-a*t)
sol = mp.odefun(lambda t, v: -a*v + beta*mp.sin(w*t), 0, v30)
for t in [mp.mpf("0.7"), mp.mpf(2)]:
    print(f"RC network v3({t}): closed - odefun = {mp.nstr(closed(t) - sol(t), 3)}")

# ---- 3. diode clamp ----
Is, nVt, R, Cd, RLd = 1e-12, 0.0388, 10.0, 1e-5, 100.0
def f(t, v):
    return [((5*np.sin(2e3*np.pi*t) - v[0])/R - Is*np.expm1(v[0]/nVt) - v[0]/RLd)/Cd]
def jac(t, v):
    return [[(-1/R - Is/nVt*np.exp(v[0]/nVt) - 1/RLd)/Cd]]
tt = [2.5e-4, 5e-4, 1e-3, 1.5e-3, 2e-3]
r1 = solve_ivp(f, (0, 2e-3), [0.0], method="Radau", jac=jac, rtol=1e-13, atol=1e-16, t_eval=tt)
r2 = solve_ivp(f, (0, 2e-3), [0.0], method="DOP853", rtol=1e-13, atol=1e-16, t_eval=tt)
for t, x, y in zip(tt, r1.y[0], r2.y[0]):
    print(f"diode t={t:g}: v={x!r} (DOP853 - Radau = {y - x:.1e})")
