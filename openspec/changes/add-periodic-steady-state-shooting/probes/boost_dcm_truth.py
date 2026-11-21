"""mpmath periodic orbit of the boost of test_periodic_orbit_kernel (DCM, R=500).

x = (i_L, v_C) at the rising gate edge. Per period: switch on D*T
(gs=GON, gd=GOFF); switch off, diode on (gd=GON) until its current
gd*(v_sw - v_C) reaches zero, i.e. i_L = GOFF*v_C; then both off.
Exact propagators: expm of the augmented 3x3 matrix.
"""
import mpmath as mp

mp.mp.dps = 50
VIN, L, C, R = mp.mpf(12), mp.mpf("100e-6"), mp.mpf("47e-6"), mp.mpf(500)
GON, GOFF, T, D = mp.mpf(1000), mp.mpf("1e-9"), mp.mpf("1e-5"), mp.mpf("0.5")


def M(gs, gd):
    k = 1 / (gs + gd)  # v_sw = k (i_L + gd v_C)
    A = mp.matrix(3, 3)
    A[0, 0], A[0, 1], A[0, 2] = -k / L, -k * gd / L, VIN / L
    A[1, 0] = gd * k / C
    A[1, 1] = (gd * k * gd - gd - 1 / R) / C
    return A


def seg(gs, gd, x, h):
    E = mp.expm(M(gs, gd) * h)
    v = E * mp.matrix([x[0], x[1], 1])
    return [v[0], v[1]]


def period(x):
    x = seg(GON, GOFF, x, D * T)
    rem = (1 - D) * T
    g = lambda h: seg(GOFF, GON, x, h)[0] - GOFF * seg(GOFF, GON, x, h)[1]
    if g(rem) > 0:  # CCM
        return seg(GOFF, GON, x, rem), None
    a, b = mp.mpf(0), rem
    for _ in range(200):  # bisection: g(a) > 0 > g(b)
        m = (a + b) / 2
        if g(m) > 0:
            a = m
        else:
            b = m
    h2 = (a + b) / 2
    x = seg(GOFF, GON, x, h2)
    return seg(GOFF, GOFF, x, rem - h2), (D * T + h2) / T


x = [mp.mpf(0), mp.mpf(36.6)]
for it in range(30):  # Newton with a central-difference Jacobian
    y, tau = period(x)
    r = [y[0] - x[0], y[1] - x[1]]
    if max(abs(r[0]), abs(r[1])) < mp.mpf("1e-40"):
        break
    J = mp.matrix(2, 2)
    for j in range(2):
        e = mp.mpf("1e-25") * max(1, abs(x[j]))
        xp, xm = list(x), list(x)
        xp[j] += e
        xm[j] -= e
        yp, ym = period(xp)[0], period(xm)[0]
        for i in range(2):
            J[i, j] = (yp[i] - ym[i]) / (2 * e) - (1 if i == j else 0)
    dx = mp.lu_solve(J, mp.matrix([-r[0], -r[1]]))
    x = [x[0] + dx[0], x[1] + dx[1]]
print("iters", it, "residual", mp.nstr(max(abs(r[0]), abs(r[1])), 3))
print("i_L*", mp.nstr(x[0], 20), "v_C*", mp.nstr(x[1], 20),
      "diode off at t/T", mp.nstr(tau, 15))
