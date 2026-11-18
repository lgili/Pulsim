"""Rounding of the diode port problem on a floating island (DECISIONS #30).

The single-phase bridge with every diode OFF: the DC bus (p, n; C || R) is tied to the rest only
through the four OFF diodes (g_off = 1e-9 S) and an optional tie R_tie from n to ground. The
backward-Euler companion J = G + C/h of one port solve is posed at a consistent all-OFF state
and solved four ways in double, against the same system solved in mpmath (60 digits):

  hist   J s = (C/h) x - b             (history form, the port model before #22)
  incr   J d = -(G x + b), s = x + d    (increment form, #22)
  hist+I, incr+I                        the same, then the island's KCL imposed exactly:
         sigma = sum_j g_S(j) s_j + sum_S b_i must be 0, s -= w sigma / (g_S . w),
         w = J^-1 e_r (one more solve, r = an island node); g_S(j) = the boundary branches'
         conductances (the C/h stamps and R are internal and cancel analytically)
  incr+Isum                             the same with g_S taken as column sums of the assembled
         G over the island (the internal 1/R rounds the leaks: ~eps*G_int/g_off relative)

Printed: the worst port-voltage error |u - u_exact| (V) and the worst relative error of the
island-impedance entries of Z, over h = H_E_REL * 0.02 s and R_tie.
    python3 island_rounding.py
"""
import mpmath as mp
import numpy as np
from scipy.linalg import lu_factor, lu_solve

mp.mp.dps = 60
GOFF, C, R, W = 1e-9, 1e-3, 100.0, 2 * np.pi * 50
PORTS = [(0, 1), (-1, 1), (2, 0), (2, -1)]      # D1 a->p, D3 0->p, D2 n->a, D4 n->0 (anode, cathode)


def system(h, rtie, t, ftype=float):
    """G, Cm, b(t + h) of the bridge, every diode OFF; unknowns a, p, n, i_Vs."""
    f = ftype
    G = [[f(0)] * 4 for _ in range(4)]
    Cm = [[f(0)] * 4 for _ in range(4)]
    b = [f(0)] * 4

    def cond(i, j, g):
        for a, c, s in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            if a >= 0 and c >= 0:
                G[a][c] += s * g
    for a, c in PORTS:
        cond(a, c, f(GOFF))
    cond(1, 2, 1 / f(R))
    if rtie:
        cond(2, -1, 1 / f(rtie))
    for a, c, s in ((1, 1, 1), (2, 2, 1), (1, 2, -1), (2, 1, -1)):
        Cm[a][c] += s * f(C)
    G[0][3] += 1
    G[3][0] += 1
    b[3] = -f(100) * (mp.sin(f(W) * (t + h)) if ftype is not float else np.sin(W * (t + h)))
    return G, Cm, b


def consistent_state(t, vc, rtie):
    """Exact all-OFF state at t (island KCL of the leaks), rounded to double."""
    va = 100 * mp.sin(mp.mpf(W) * t)
    gt = mp.mpf(0) if not rtie else 1 / mp.mpf(rtie)
    g = mp.mpf(GOFF)
    # g(va - vp) + g(0 - vp) + g(va - vn) + g(0 - vn) - gt vn = 0, vp = vn + vc
    vn = (2 * g * va - 2 * g * vc) / (4 * g + gt)
    vp = vn + vc
    ivs = -(g * (vp - va) + g * (vn - va))       # Vs branch current closing node a's KCL
    return [float(va), float(vp), float(vn), float(ivs)]


def gaps(s):
    return np.array([(s[a] if a >= 0 else 0.0) - (s[c] if c >= 0 else 0.0) for a, c in PORTS])


def solve_exact(h, rtie, t, x):
    G, Cm, b = system(mp.mpf(h), rtie, mp.mpf(t), mp.mpf)
    J = mp.matrix(4, 4)
    for i in range(4):
        for j in range(4):
            J[i, j] = G[i][j] + Cm[i][j] / mp.mpf(h)
    rhs = mp.matrix([sum(Cm[i][j] * mp.mpf(x[j]) for j in range(4)) / mp.mpf(h) - b[i] for i in range(4)])
    s = mp.lu_solve(J, rhs)
    u = [float((s[a] if a >= 0 else 0) - (s[c] if c >= 0 else 0)) for a, c in PORTS]
    zc = mp.lu_solve(J, mp.matrix([0, 1, -1, 0][::1]))       # unit current p -> n (inside)
    zi = mp.lu_solve(J, mp.matrix([0, 0, -1, 0]))            # unit current out of n: common mode
    return np.array(u), float(zi[2]), float(zc[1] - zc[2])


def solve_double(h, rtie, t, x, form, island):
    G, Cm, b = (np.array(m, float) for m in system(h, rtie, t))
    J = G + Cm / h
    lu = lu_factor(J)
    x = np.array(x)
    if form == "hist":
        s = lu_solve(lu, Cm @ x / h - b)
    else:
        s = x + lu_solve(lu, -(G @ x + b))
    zi = lu_solve(lu, np.array([0.0, 0.0, -1.0, 0.0]))
    if island:
        S = [1, 2]
        if island == 1:                          # column sums of the assembled G
            gS = G[S, :].sum(axis=0)             # (C/h is internal: its column sums are 0)
        else:                                    # from the boundary branches themselves
            gS = np.zeros(4)
            bnd = [(a, c, GOFF) for a, c in PORTS] + ([(2, -1, 1 / rtie)] if rtie else [])
            for a, c, g in bnd:
                for i, j in ((a, c), (c, a)):
                    if i in S and j not in S:
                        gS[i] += g
                        if j >= 0:
                            gS[j] -= g
        w = lu_solve(lu, np.eye(4)[S[0]])
        kap = gS @ w
        s = s - w * (gS @ s + b[S].sum()) / kap
        zi = zi - w * (gS @ zi + 1.0) / kap     # J zi = rhs: the island sums rhs = -1 A
    return gaps(s), zi[2]


def main():
    t = 0.0049                                   # rising toward the peak, bus held at 99.9 V
    vc = mp.mpf("99.9")
    print("h_e/T_ref  R_tie   |  max |u - u_exact| (V)                                 | rel err Z_island")
    print("                   |  hist      incr      incr+Isum hist+I    incr+I      |  incr      incr+I")
    for rtie in (None, 1e6, 1e9, 1e12):
        x = consistent_state(mp.mpf(t), vc, rtie)
        for rel in (1e-12, 1e-10, 1e-9, 1e-7, 1e-5, 1e-4):
            h = rel * 0.02
            ue, zie, _ = solve_exact(h, rtie, t, x)
            row = []
            for form, isl in (("hist", 0), ("incr", 0), ("incr", 1), ("hist", 2), ("incr", 2)):
                u, zi = solve_double(h, rtie, t, x, form, isl)
                row.append(np.abs(u - ue).max())
                if form == "incr" and not isl:
                    zp = abs(zi - zie) / abs(zie)
                if form == "incr" and isl == 2:
                    zq = abs(zi - zie) / abs(zie)
            print(f"{rel:8.0e}  {rtie or 0:7.0e} |  " + "  ".join(f"{v:8.1e}" for v in row)
                  + f"      |  {zp:8.1e}  {zq:8.1e}")


if __name__ == "__main__":
    main()
