"""Numpy-only reference for the periodic-steady-state tests.

Port of ``openspec/changes/add-periodic-steady-state-shooting/probes/``
(``shooting_dcm.py`` and ``peak_current_mode.py``), with scipy's
``brentq`` replaced by bisection to float resolution. The buck is the
audit's ``numerics/p2_buck.py``: Vin = 24 V, L = 100 uH, g_on = 1e3 S,
g_off = 1e-9 S, 100 kHz; state ``x = (i_L, v_C)`` at the rising gate
edge. Exact per-mode propagators (modal form), events located in time,
Newton on the true state with the saltation-corrected monodromy.
"""

import numpy as np

VIN, GON, GOFF, T = 24.0, 1e3, 1e-9, 1e-5
GX = np.array([1.0, 0.0])
I_THR = GOFF * VIN          # diode current zero  <=>  i_L = g_off * Vin

# The table of the change's tasks.md (50-digit references for DCM).
DCM = dict(L=100e-6, C=47e-6, R=50.0, D=0.3)
DCM_ORBIT = (6.00750582388986e-9, 8.99624708728943)
DCM_TURN_OFF = 0.799830698444468          # diode off, fraction of T
DCM_MULTIPLIERS = (0.0, 0.988977)
DCM_DVC_DD = 23.0895697                    # d v_C* / d D
CCM = dict(L=100e-6, C=47e-6, R=1.0, D=0.5)
CCM_ORBIT = (11.6878790893, 11.9877289409)
CCM_MULTIPLIERS = (0.894557 + 0.089619j, 0.894557 - 0.089619j)
PCM = dict(L=100e-6, C=4700e-6, R=2.0, I_ref=7.0)
PCM_ORBIT = (6.4083405593, 13.4083540775)
PCM_MULTIPLIERS = (-1.267293, 0.998949)
PCM_NO_SALTATION = (0.9993 + 0.0146j, 0.9993 - 0.0146j)


def _AB(p, gs, gd):
    k = 1 / (gs + gd)
    L, C, R = p["L"], p["C"], p["R"]
    return (np.array([[-k / L, -1 / L], [1 / C, -1 / (R * C)]]),
            np.array([gs * VIN * k / L, 0.0]))


def _f(p, gs, gd, x):
    A, b = _AB(p, gs, gd)
    return A @ x + b


def _seg(p, gs, gd, x, h):
    """Advance by h; return (x(h), exp(A h))."""
    A, b = _AB(p, gs, gd)
    M = np.zeros((3, 3))
    M[:2, :2] = A
    M[:2, 2] = b
    lam, V = np.linalg.eig(M)
    E = ((V * np.exp(lam * h)) @ np.linalg.inv(V)).real
    return E[:2, :2] @ x + E[:2, 2], E[:2, :2]


def _root(g, a, b):
    """Bisection to float resolution; g(a) and g(b) differ in sign."""
    ga = g(a)
    for _ in range(200):
        m = 0.5 * (a + b)
        if m in (a, b):
            break
        gm = g(m)
        if (gm < 0) == (ga < 0):
            a, ga = m, gm
        else:
            b = m
    return 0.5 * (a + b)


def buck_period(p, x, salt=True):
    """One period from the rising edge: (x(T), monodromy, tau/T)."""
    D = p["D"]
    x, Ph = _seg(p, GON, GOFF, x, D * T)
    mono = Ph
    rem = (1 - D) * T

    def g(h):
        return _seg(p, GOFF, GON, x, h)[0][0] - I_THR

    if x[0] > I_THR and g(rem) < 0:
        h2 = _root(g, 0.0, rem)
        x, Ph = _seg(p, GOFF, GON, x, h2)
        mono = Ph @ mono
        fm, fp = _f(p, GOFF, GON, x), _f(p, GOFF, GOFF, x)
        if salt:
            mono = (np.eye(2) + np.outer(fp - fm, GX) / (GX @ fm)) @ mono
        x, Ph = _seg(p, GOFF, GOFF, x, rem - h2)
        return x, Ph @ mono, D + h2 / T
    x, Ph = _seg(p, GOFF, GON, x, rem)
    return x, Ph @ mono, None


def pcm_period(p, x, salt=True):
    """Peak-current-mode: on at the clock, off when i_L = I_ref."""
    Iref = p["I_ref"]
    q = dict(p, D=0.0)

    def g(h):
        return _seg(q, GON, GOFF, x, h)[0][0] - Iref

    h1 = _root(g, 0.0, T)
    x1, P1 = _seg(q, GON, GOFF, x, h1)
    fm, fp = _f(q, GON, GOFF, x1), _f(q, GOFF, GON, x1)
    S = np.eye(2) + np.outer(fp - fm, GX) / (GX @ fm) if salt \
        else np.eye(2)
    x2, P2 = _seg(q, GOFF, GON, x1, T - h1)
    return x2, P2 @ S @ P1, h1 / T


def shoot(period_fn, p, x0, salt=True, tol=1e-12, itmax=60):
    """Newton shooting; returns (x*, monodromy, residual history)."""
    x = np.asarray(x0, dtype=float).copy()
    hist = []
    for _ in range(itmax):
        x1, M, _tau = period_fn(p, x, salt)
        r = x1 - x
        hist.append(float(np.abs(r).max()))
        if hist[-1] < tol:
            break
        x = x + np.linalg.solve(np.eye(2) - M, r)
    return x, M, hist
