"""Numpy-only references for the parameter-sensitivity tests.

Ports of ``openspec/changes/add-parameter-sensitivities/probes/`` and of
the sensitivity half of
``add-periodic-steady-state-shooting/probes/shooting_dcm.py`` (scipy's
``brentq`` replaced by bisection, the modal propagator kept), plus the
finite-difference harness the tests use on any engine.

The constants are the change's ``tasks.md`` table; the reference
functions reproduce them, so a test that compares the kernel with a
constant compares it with a computation that lives here.
"""

import numpy as np

from . import _pss_reference as pss

# ---------------------------------------------------------------------
# The tasks.md table
# ---------------------------------------------------------------------

#: RC step, 10 V into 1 kOhm and 1 uF, at t = RC: dv/dR, dv/dC.
RC = dict(V=10.0, R=1e3, C=1e-6)
RC_DV_DR = -10.0 / (1e3 * np.e)          # -3.678794e-3 V/Ohm
RC_DV_DC = -10.0 / (1e-6 * np.e)         # -3.678794e6 V/F
#: DCM buck from zero, v_C(40 T).
DCM_TRANSIENT = dict(L=1758.1254, R=2.5551883e-2, D=43.093591)
#: DCM buck orbit, d v_C* / d p.
DCM_ORBIT = dict(L=-3.4575154e4, R=6.92255e-2, D=23.0895697)
#: Shockley rectifier: worst relative difference forward vs central FD.
SHOCKLEY_TOL = dict(R=3.4e-7, C=1e-8, Rl=1e-8, Is=1.1e-8, nVt=8e-9)
SHOCKLEY_P0 = dict(R=1.0, C=100e-6, Rl=50.0, Is=1e-9, nVt=0.05)

T, VIN, GON, GOFF = pss.T, pss.VIN, pss.GON, pss.GOFF
DCM_P0 = dict(L=100e-6, C=47e-6, R=50.0, D=0.3)


# ---------------------------------------------------------------------
# Finite-difference harness (task 1.2)
# ---------------------------------------------------------------------

def central_difference(fn, p0, *, rel_steps=(1e-2, 1e-3, 1e-4, 1e-5),
                       scale=None):
    """Central difference of ``fn`` at ``p0`` with step selection.

    For each step ``h`` of the ladder it forms ``D(h)``, ``D(h/2)`` and
    the Richardson value ``R = (4 D(h/2) - D(h)) / 3``; the error
    estimate is ``|D(h/2) - R|`` (max-norm for vector values). The step
    with the smallest estimate wins: large steps lose to truncation,
    small ones to rounding. Returns ``(R, error_estimate, h)``. Works on
    any engine: ``fn`` is a whole run.
    """
    s = abs(p0) if scale is None else scale
    best = None
    for rel in rel_steps:
        h = rel * s
        d1 = (np.asarray(fn(p0 + h)) - np.asarray(fn(p0 - h))) / (2 * h)
        d2 = (np.asarray(fn(p0 + h / 2)) - np.asarray(fn(p0 - h / 2))) / h
        r = (4 * d2 - d1) / 3
        err = float(np.max(np.abs(d2 - r)))
        if best is None or err < best[1]:
            best = (r, err, h)
    return best


def rel_err(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# ---------------------------------------------------------------------
# Implicit path: Shockley rectifier (probes/forward_sensitivity_mna.py)
# ---------------------------------------------------------------------

SHOCKLEY_NAMES = list(SHOCKLEY_P0)


def shockley_run(prm, sens=True, N=4000, t_end=40e-3):
    """Trapezoidal + Newton; sensitivities on the converged Jacobian.
    Returns ``(x(t_end), S)``, ``S`` columns in ``SHOCKLEY_NAMES``."""
    h = t_end / N
    R, C, Rl, Is, nVt = (prm[k] for k in SHOCKLEY_NAMES)

    def vs(t):
        return 20 * np.sin(2 * np.pi * 50 * t)

    def idio(v):
        return Is * (np.exp(v / nVt) - 1)

    def gdio(v):
        return Is / nVt * np.exp(v / nVt)

    def f(x, t):
        iD = idio(x[0] - x[1])
        return np.array([(x[0] - vs(t)) / R + iD, -iD + x[1] / Rl])

    def dfdx(x):
        g = gdio(x[0] - x[1])
        return np.array([[1 / R + g, -g], [-g, g + 1 / Rl]])

    def dfdp(x, t):
        vd = x[0] - x[1]
        e = np.exp(vd / nVt)
        dR = np.array([-(x[0] - vs(t)) / R ** 2, 0])
        dC = np.zeros(2)
        dRl = np.array([0, -x[1] / Rl ** 2])
        dIs = np.array([e - 1, -(e - 1)])
        dn = np.array([-Is * e * vd / nVt ** 2, Is * e * vd / nVt ** 2])
        return np.stack([dR, dC, dRl, dIs, dn], axis=1)

    Cm = np.array([[0, 0], [0, C]])
    dCm = np.array([[0, 0], [0, 1.0]])
    x = np.zeros(2)
    S = np.zeros((2, 5))
    f0 = f(x, 0.0)
    t = 0.0
    for _ in range(N):
        t1 = t + h
        x1 = x.copy()
        for _it in range(50):
            F = Cm @ (x1 - x) / h + 0.5 * (f(x1, t1) + f0)
            J = Cm / h + 0.5 * dfdx(x1)
            dx = np.linalg.solve(J, -F)
            x1 = x1 + dx
            if np.abs(dx).max() < 1e-13:
                break
        if sens:
            J = Cm / h + 0.5 * dfdx(x1)
            rhs = (Cm / h - 0.5 * dfdx(x)) @ S \
                - 0.5 * (dfdp(x1, t1) + dfdp(x, t))
            rhs[:, 1] -= dCm @ (x1 - x) / h
            S = np.linalg.solve(J, rhs)
        x = x1
        f0 = f(x, t1)
        t = t1
    return x, S


# ---------------------------------------------------------------------
# Exact path: DCM buck (shooting_dcm.py), x = (i_L, v_C)
# ---------------------------------------------------------------------

def _dAB(p, gs, gd, name):
    k = 1 / (gs + gd)
    L, R, C = p["L"], p["R"], p["C"]
    if name == "L":
        return (np.array([[k / L ** 2, 1 / L ** 2], [0, 0]]),
                np.array([-gs * VIN * k / L ** 2, 0.0]))
    if name == "R":
        return np.array([[0, 0], [0, 1 / (R ** 2 * C)]]), np.zeros(2)
    return np.zeros((2, 2)), np.zeros(2)


def _expm_dk(M, h, dM):
    """``exp(M h)`` and its derivative along ``dM`` (Daleckii-Krein,
    the prototype's plain divided differences)."""
    lam, V = np.linalg.eig(M)
    Vi = np.linalg.inv(V)
    e = np.exp(lam * h)
    G = Vi @ dM @ V
    n = len(lam)
    F = np.empty((n, n), complex)
    for i in range(n):
        for j in range(n):
            d = lam[i] - lam[j]
            F[i, j] = h * e[i] if abs(d * h) < 1e-12 else (e[i] - e[j]) / d
    return ((V * e) @ Vi).real, (V @ (F * G) @ Vi).real


def _seg_s(p, gs, gd, x, h, s, name):
    A, b = pss._AB(p, gs, gd)
    M = np.zeros((3, 3))
    M[:2, :2], M[:2, 2] = A, b
    dA, db = _dAB(p, gs, gd, name)
    dM = np.zeros((3, 3))
    dM[:2, :2], dM[:2, 2] = dA, db
    E, D = _expm_dk(M, h, dM)
    return (E[:2, :2] @ x + E[:2, 2], E[:2, :2],
            E[:2, :2] @ s + D[:2, :2] @ x + D[:2, 2])


def dcm_period_s(p, x, s, name, jump=True):
    """One period with the sensitivity ``s = dx/dp`` (p in L, R, D).
    Returns ``(x(T), monodromy, s(T))``. ``jump=False`` drops the event
    terms (the regression guard of task 2.4)."""
    D = p["D"]
    x, Ph, s = _seg_s(p, GON, GOFF, x, D * T, s, name)
    mono = Ph
    if name == "D" and jump:                 # gate edge: dtau/dD = T
        s = s + (pss._f(p, GON, GOFF, x) - pss._f(p, GOFF, GON, x)) * T
    rem = (1 - D) * T

    def g(h):
        return pss._seg(p, GOFF, GON, x, h)[0][0] - pss.I_THR

    if x[0] > pss.I_THR and g(rem) < 0:
        h2 = pss._root(g, 0.0, rem)
        x, Ph, s = _seg_s(p, GOFF, GON, x, h2, s, name)
        mono = Ph @ mono
        fm, fp = pss._f(p, GOFF, GON, x), pss._f(p, GOFF, GOFF, x)
        mono = (np.eye(2) + np.outer(fp - fm, pss.GX)
                / (pss.GX @ fm)) @ mono
        if jump:
            dtau = -(pss.GX @ s) / (pss.GX @ fm)
            s = s + (fm - fp) * dtau
        x, Ph, s = _seg_s(p, GOFF, GOFF, x, rem - h2, s, name)
        return x, Ph @ mono, s
    x, Ph, s = _seg_s(p, GOFF, GON, x, rem, s, name)
    return x, Ph @ mono, s


def dcm_transient(p, name, n_periods=40, jump=True):
    """``(x(n T), dx(n T)/dp)`` from zero."""
    x, s = np.zeros(2), np.zeros(2)
    for _ in range(n_periods):
        x, _M, s = dcm_period_s(p, x, s, name, jump)
    return x, s


def dcm_orbit_gradient(p, name):
    """``(x*, dx*/dp = (I - M)^-1 dP/dp)``."""
    xs, _M, _h = pss.shoot(pss.buck_period, p, np.zeros(2))
    _x, M, sT = dcm_period_s(p, xs, np.zeros(2), name)
    return xs, np.linalg.solve(np.eye(2) - M, sT)
