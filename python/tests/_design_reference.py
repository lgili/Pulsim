"""Numpy reference for the design-evaluation oracle (task 1.3).

CCM buck with resistor-pair switches and an inductor resistance:
state ``x = (i_L, v_C)``, two modes (switch on / diode off for
``D.T``, then switch off / diode on). Per mode ``x' = A x + b`` with
the switch node eliminated,
``v_sw = (g_s V_in - i_L) / (g_s + g_d)``. The orbit is the fixed
point of the affine period map (CCM: the pattern is fixed), each
segment propagated by the matrix exponential of the augmented
``[[A, b], [0, 0]]``; the period integrals are Gauss-Legendre on the
exact solution (64 nodes per segment: an entire function on an
interval much shorter than its time constants, so exact to
rounding); the output-voltage extrema are the zeros of
``i_C = i_L - v_C / R`` located by bisection.
"""

import numpy as np

VIN, GON, GOFF = 24.0, 1e3, 1e-9


def _mode(p, on):
    gs, gd = (GON, GOFF) if on else (GOFF, GON)
    k = 1.0 / (gs + gd)
    L, C, R, Rs = p["L"], p["C"], p["R"], p["R_L"]
    A = np.array([[-(k + Rs) / L, -1.0 / L], [1.0 / C, -1.0 / (R * C)]])
    b = np.array([gs * VIN * k / L, 0.0])
    return A, b, gs, gd, k


def _expm_aug(A, b, h):
    M = np.zeros((3, 3))
    M[:2, :2] = A
    M[:2, 2] = b
    lam, V = np.linalg.eig(M * h)
    if np.linalg.cond(V) < 1e8:
        return ((V * np.exp(lam)) @ np.linalg.inv(V)).real
    # scaling and squaring of a Taylor series (the eigenbasis of the
    # zero eigenvalue of the augmented block can be ill-conditioned)
    X = M * h
    s = max(0, int(np.ceil(np.log2(max(np.abs(X).sum(1).max(), 1e-300)))) + 4)
    X = X / 2.0 ** s
    E, term = np.eye(3), np.eye(3)
    for j in range(1, 25):
        term = term @ X / j
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


def segments(p):
    T, D = p["T"], p["D"]
    return [(True, 0.0, D * T), (False, D * T, T)]


def orbit(p):
    """x0 at the rising gate edge."""
    Phi = np.eye(3)
    for on, t0, t1 in segments(p):
        A, b, *_ = _mode(p, on)
        Phi = _expm_aug(A, b, t1 - t0) @ Phi
    return np.linalg.solve(np.eye(2) - Phi[:2, :2], Phi[:2, 2])


def _x_at(p, on, x0, tau):
    A, b, *_ = _mode(p, on)
    E = _expm_aug(A, b, tau)
    return E[:2, :2] @ x0 + E[:2, 2]


def kpis(p):
    T = p["T"]
    gx, gw = np.polynomial.legendre.leggauss(64)
    x = orbit(p)
    acc = dict(p_in=0.0, p_s=0.0, p_d=0.0, p_rl=0.0, p_out=0.0)
    vmin, vmax = np.inf, -np.inf
    for on, t0, t1 in segments(p):
        A, b, gs, gd, k = _mode(p, on)
        h = t1 - t0
        taus = 0.5 * h * (gx + 1.0)
        X = np.array([_x_at(p, on, x, tau) for tau in taus])
        i, v = X[:, 0], X[:, 1]
        vsw = (gs * VIN - i) * k
        w = 0.5 * h * gw / T
        acc["p_in"] += w @ (VIN * gs * (VIN - vsw))
        acc["p_s"] += w @ (gs * (VIN - vsw) ** 2)
        acc["p_d"] += w @ (gd * vsw ** 2)
        acc["p_rl"] += w @ (p["R_L"] * i ** 2)
        acc["p_out"] += w @ (v ** 2 / p["R"])
        # extrema of v_C: segment ends and zeros of i_C
        def ic(tau):
            y = _x_at(p, on, x, tau)
            return y[0] - y[1] / p["R"]
        grid = np.linspace(0.0, h, 65)
        vals = [ic(g) for g in grid]
        cand = [x[1], _x_at(p, on, x, h)[1]]
        for a_, b_, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if fa == 0.0 or fa * fb < 0:
                lo, hi = a_, b_
                for _ in range(200):
                    m = 0.5 * (lo + hi)
                    if ic(lo) * ic(m) <= 0:
                        hi = m
                    else:
                        lo = m
                cand.append(_x_at(p, on, x, 0.5 * (lo + hi))[1])
        vmin, vmax = min(vmin, *cand), max(vmax, *cand)
        x = _x_at(p, on, x, h)
    acc["efficiency"] = acc["p_out"] / acc["p_in"]
    acc["p_loss"] = acc["p_s"] + acc["p_d"] + acc["p_rl"]
    acc["ripple_v"] = vmax - vmin
    acc["x0"] = orbit(p)
    return acc
