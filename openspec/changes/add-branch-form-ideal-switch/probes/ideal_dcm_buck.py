# Ideal-switch DCM buck (R_on = 0, g_off = 0): periodic orbit and monodromy with the projection saltation.
# 24 V, 100 uH, 47 uF, 50 ohm, D = 0.3, 100 kHz. In the idle interval i_L is algebraic (= 0).
import numpy as np, mpmath as mp
from scipy.optimize import brentq
mp.mp.dps = 40
Vin, L, C, R, D, T = map(mp.mpf, ["24", "100e-6", "47e-6", "50", "0.3", "1e-5"])
A = mp.matrix([[0, -1/L], [1/C, -1/(R*C)]])
def flow(x, b, h):
    M = mp.matrix(3, 3); M[0, 0] = A[0, 0]; M[0, 1] = A[0, 1]; M[1, 0] = A[1, 0]; M[1, 1] = A[1, 1]; M[0, 2] = b
    E = mp.expm(M*h); return mp.matrix([E[0, 0]*x[0] + E[0, 1]*x[1] + E[0, 2], E[1, 0]*x[0] + E[1, 1]*x[1] + E[1, 2]]), E
def period(v0):
    x, E1 = flow(mp.matrix([0, v0]), Vin/L, D*T)
    g = lambda h: flow(x, 0, h)[0][0]
    h2 = mp.findroot(g, (mp.mpf("4e-6"), mp.mpf("6e-6")), solver='anderson', tol=mp.mpf(10)**(-30))
    x2, E2 = flow(x, 0, h2)
    v_end = x2[1]*mp.e**(-((1-D)*T - h2)/(R*C))
    return v_end, h2, x, x2, E1, E2
v = mp.findroot(lambda v0: period(v0)[0] - v0, (mp.mpf("8.9"), mp.mpf("9.1")), solver='anderson', tol=mp.mpf(10)**(-30))
v_end, h2, x1, x2, E1, E2 = period(v)
print("ideal DCM orbit: v_C(0) =", mp.nstr(v, 15), " i_peak =", mp.nstr(x1[0], 12), " diode off at tau/T =", mp.nstr(D + h2/T, 15))
# averaged-model check: M = 2D / (D + sqrt(D^2 + 8L/(R T)))
K = 2*L/(R*T); Mconv = 2/(1 + mp.sqrt(1 + 4*K/D**2)); print("textbook DCM ratio * Vin =", mp.nstr(Mconv*Vin, 10))
# monodromy on (i, v): Phi3 * S * Phi2 * Phi1 with S the projection saltation at i = 0
f_minus = mp.matrix([-x2[1]/L, (x2[0] - x2[1]/R)/C]); f_plus = mp.matrix([0, -x2[1]/(R*C)])
gx = mp.matrix([[1, 0]])
S = mp.eye(2) + (f_plus - f_minus)*gx/(gx*f_minus)[0]
P1 = mp.matrix([[E1[0, 0], E1[0, 1]], [E1[1, 0], E1[1, 1]]]); P2 = mp.matrix([[E2[0, 0], E2[0, 1]], [E2[1, 0], E2[1, 1]]])
P3 = mp.matrix([[0, 0], [0, mp.e**(-((1-D)*T - h2)/(R*C))]])
Mono = P3*S*P2*P1
print("S =", S); print("multipliers:", [mp.nstr(e, 10) for e in mp.eig(Mono)[0]])
eps = mp.mpf("1e-12"); print("FD d v(T)/d v(0) =", mp.nstr((period(v+eps)[0] - period(v-eps)[0])/(2*eps), 10), " Mono[1,1] =", mp.nstr(Mono[1, 1], 10))
print("without S: Mono[1,1] =", mp.nstr((P3*P2*P1)[1, 1], 10))
