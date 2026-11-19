"""mpmath orbit GRADIENTS of the boost DCM orbit of boost_dcm_truth.py.

Same circuit and segments (12 V, 100 uH, 47 uF, 500 ohm, D = 0.5,
g_on = 1e3, g_off = 1e-9, 100 kHz). The diode-off instant is found by
Newton on h (g'(h) from the segment's own derivative) instead of
bisection, so one orbit costs seconds. dv_C*/dp for p in (L, C, R, D) is
a central difference of the 50-digit orbit with relative step 1e-15:
truncation ~1e-30; the orbit's 1e-44 residual over 1/(1 - mu) and 2e
leaves ~1e-28 relative: exact to every digit printed. The orbit itself
is the one of boost_dcm_truth.py (checked).
"""
import mpmath as mp

mp.mp.dps = 50
VIN, GON, GOFF, T = mp.mpf(12), mp.mpf(1000), mp.mpf("1e-9"), mp.mpf("1e-5")
BASE = dict(L=mp.mpf("100e-6"), C=mp.mpf("47e-6"), R=mp.mpf(500),
            D=mp.mpf("0.5"))


def A(gs, gd, p):
    k = 1 / (gs + gd)
    a = mp.matrix(3, 3)
    a[0, 0], a[0, 1], a[0, 2] = -k / p["L"], -k * gd / p["L"], VIN / p["L"]
    a[1, 0] = gd * k / p["C"]
    a[1, 1] = (gd * k * gd - gd - 1 / p["R"]) / p["C"]
    return a


def seg(a, x, h):
    v = mp.expm(a * h) * mp.matrix([x[0], x[1], 1])
    return [v[0], v[1]]


def period(x, p):
    x = seg(A(GON, GOFF, p), x, p["D"] * T)
    rem = (1 - p["D"]) * T
    a = A(GOFF, GON, p)

    def g(h):
        y = seg(a, x, h)
        return y[0] - GOFF * y[1], y

    def dg(y):
        f = a * mp.matrix([y[0], y[1], 1])
        return f[0] - GOFF * f[1]

    if g(rem)[0] > 0:
        raise RuntimeError("CCM: not this probe's orbit")
    h = rem * mp.mpf("0.49")          # the off instant is near 0.244 T
    for _ in range(60):
        gv, y = g(h)
        dh = gv / dg(y)
        h -= dh
        if abs(dh) < mp.mpf("1e-45") * T:
            break
    x = seg(a, x, h)
    return seg(A(GOFF, GOFF, p), x, rem - h), (p["D"] * T + h) / T


def orbit(p, x=None):
    x = x or [mp.mpf("0"), mp.mpf("36.5944")]
    for _ in range(40):
        y, tau = period(x, p)
        r = [y[0] - x[0], y[1] - x[1]]
        if max(abs(r[0]), abs(r[1])) < mp.mpf("1e-44"):
            return x, tau
        J = mp.matrix(2, 2)
        for j in range(2):
            e = mp.mpf("1e-22") * max(1, abs(x[j]))
            xp, xm = list(x), list(x)
            xp[j] += e
            xm[j] -= e
            yp, ym = period(xp, p)[0], period(xm, p)[0]
            for i in range(2):
                J[i, j] = (yp[i] - ym[i]) / (2 * e) - (1 if i == j else 0)
        dx = mp.lu_solve(J, mp.matrix([-r[0], -r[1]]))
        x = [x[0] + dx[0], x[1] + dx[1]]
    raise RuntimeError("no convergence")


if __name__ == "__main__":
    x0, tau = orbit(BASE)
    print("v_C*", mp.nstr(x0[1], 20), "i_L*", mp.nstr(x0[0], 20),
          "off at t/T", mp.nstr(tau, 15))
    for name in ("L", "C", "R", "D"):
        e = BASE[name] * mp.mpf("1e-15")
        hi = orbit({**BASE, name: BASE[name] + e}, x0)[0][1]
        lo = orbit({**BASE, name: BASE[name] - e}, x0)[0][1]
        print(f"dv_C*/d{name}", mp.nstr((hi - lo) / (2 * e), 15))
