# numpy/scipy only. Run from this directory (it reads shooting_dcm.py).
# Saltation matters when the vector field jumps at a STATE event: peak-current-mode buck.
# Clock turns S on at kT; S turns off when iL reaches I_ref (state event, g = iL - I_ref).
# Analytic sampled-data multiplier for a stiff output: -m2/m1 = -D/(1-D).
import numpy as np
from scipy.optimize import brentq
exec(open("shooting_dcm.py").read().split("def period(")[0])
Lh, R = 100e-6, 2.0
def period_pcm(p, x, Iref, salt=True):
    g = lambda h: seg(p, gon, goff, x, h)[0][0] - Iref
    h1 = brentq(g, 0.0, T, xtol=1e-22, rtol=1e-15)
    x, P1, _ = seg(p, gon, goff, x, h1)
    fm, fp = f(p, gon, goff, x), f(p, goff, gon, x)
    S = np.eye(2) + np.outer(fp - fm, gx)/(gx@fm) if salt else np.eye(2)
    x, P2, _ = seg(p, goff, gon, x, T-h1)
    return x, P2@S@P1, h1/T
for Cv, Iref in [(47e-6, 7.0), (47e-6, 9.0), (4700e-6, 7.0), (4700e-6, 9.0)]:
    C = Cv; p = dict(L=Lh, R=R, D=0.0)
    # find the fixed point by Newton with saltation (it may be unstable; Newton does not care)
    x = np.array([Iref-1.0, 12.0])
    for k in range(60):
        x1, M, d = period_pcm(p, x, Iref)
        r = x1 - x
        if np.abs(r).max() < 1e-12: break
        x = x + np.linalg.solve(np.eye(2) - M, r)
    eps = 1e-7; Mfd = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = eps
        Mfd[:, j] = (period_pcm(p, x+e, Iref)[0] - period_pcm(p, x-e, Iref)[0])/(2*eps)
    Mno = period_pcm(p, x, Iref, salt=False)[1]
    ev = lambda M: np.sort_complex(np.linalg.eigvals(M))
    print(f"C={Cv:g} Iref={Iref}: Newton its={k} D={d:.4f} x*={x}  -D/(1-D)={-d/(1-d):+.4f}")
    print("   multipliers saltation:", ev(M), " FD of map:", ev(Mfd), " no saltation:", ev(Mno))
