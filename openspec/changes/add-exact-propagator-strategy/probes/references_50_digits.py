# NOTE: 50-digit mpmath; takes about two minutes. Prints the reference values quoted in the spec.
import mpmath as mp, numpy as np
mp.mp.dps = 50
Vin, L, C, gon, goff, T = map(mp.mpf, ["24", "100e-6", "47e-6", "1e3", "1e-9", "1e-5"])
def M(gs, gd, R):
    k = 1/(gs+gd); return mp.matrix([[-k/L, -1/L, gs*Vin*k/L], [1/C, -1/(R*C), 0], [0, 0, 0]])
# CCM reference
R, D = mp.mpf(1), mp.mpf("0.5")
Eon = mp.expm(M(gon, goff, R)*D*T); Eoff = mp.expm(M(goff, gon, R)*(1-D)*T)
x = mp.matrix([0, 0, 1])
for n in range(40): x = Eoff*(Eon*x)
print("CCM x(40T): iL =", mp.nstr(x[0], 15), " vC =", mp.nstr(x[1], 15))
# DCM fixed point in 50 digits: fixed-point iteration from the double-precision answer, event by mp.findroot
R, D = mp.mpf(50), mp.mpf("0.3"); ithr = goff*Vin
def period(x):
    x = mp.expm(M(gon, goff, R)*D*T)*x
    Mo = M(goff, gon, R)
    g = lambda h: (mp.expm(Mo*h)*x)[0] - ithr
    h2 = mp.findroot(g, (mp.mpf("4.9e-6"), mp.mpf("5.1e-6")), solver='anderson', tol=mp.mpf(10)**(-40))
    x = mp.expm(Mo*h2)*x
    return mp.expm(M(goff, goff, R)*((1-D)*T-h2))*x, D + h2/T
x = mp.matrix(["6.0075058240e-09", "8.9962470872", 1])
for n in range(3000):
    x, tau = period(x)
x2, tau = period(x)
print("DCM fixed point: iL(0) =", mp.nstr(x2[0], 15), " vC(0) =", mp.nstr(x2[1], 15), " tau/T =", mp.nstr(tau, 15), " drift over the last period =", mp.nstr(abs(x2[1]-x[1]), 5))
