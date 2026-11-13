"""Prototype core for add-complementarity-mode-selection (numpy / scipy only).

Ideal-switch circuits, every switch and diode in branch form (R_on >= 0, g_off = 0):

* per-mode reduced dynamics on y = [x; s]  (x: capacitor voltages and inductor currents,
  s: source states -- s[0] = 1 carries constants, one (cos, sin) oscillator pair per frequency),
  dependent states handled through the left / right null spaces of the state-held MNA matrix;
* mode selection at an event by ONE linear complementarity problem on the diode ports, posed on
  the backward-Euler companion network of step h_e (the rational complementarity problem at
  sigma = 1/h_e), written in the hybrid coordinates of a nonsingular complementary basis built
  from the previous mode, solved by Lemke's method with the lexicographic ratio test;
* exact propagation y(t + tau) = expm(A_m tau) y, event location on the exact trajectory with a
  post-crossing bracket end, consistent re-initialisation by the mass-norm projection with an
  impulse record (transferred flux / charge, energy removed).

Nothing here is tuned per circuit: no band, no iteration limit on the mode selection (Lemke is
finite with the lexicographic rule; the 10 n pivot cap below is a bug trap and never triggers).
"""
import numpy as np
from scipy.linalg import expm, null_space, svd

EPS = np.finfo(float).eps


# ----------------------------------------------------------------------------------------------
# Netlist
# ----------------------------------------------------------------------------------------------
class Circuit:
    def __init__(self):
        self.nodes = {}
        self.R, self.C, self.L, self.V, self.I, self.S, self.D = [], [], [], [], [], [], []
        self.omegas = []

    def n(self, name):
        if name in ("0", "gnd"):
            return -1
        return self.nodes.setdefault(name, len(self.nodes))

    def add_R(self, a, b, R):
        self.R.append((self.n(a), self.n(b), float(R)))

    def add_C(self, name, a, b, C, v0=0.0):
        self.C.append((name, self.n(a), self.n(b), float(C), float(v0)))

    def add_L(self, name, a, b, L, i0=0.0):
        self.L.append((name, self.n(a), self.n(b), float(L), float(i0)))

    def _osc(self, w):
        for k, w0 in enumerate(self.omegas):
            if abs(w0 - w) <= 1e-12 * w:
                return k
        self.omegas.append(w)
        return len(self.omegas) - 1

    def add_Vdc(self, name, a, b, V):
        self.V.append((name, self.n(a), self.n(b), ("dc", float(V))))

    def add_Vsin(self, name, a, b, amp, f, phase=0.0, offset=0.0):
        k = self._osc(2 * np.pi * f)
        self.V.append((name, self.n(a), self.n(b), ("sin", float(amp), k, float(phase), float(offset))))

    def add_Idc(self, name, a, b, I):          # current I flows a -> b through the source
        self.I.append((name, self.n(a), self.n(b), float(I)))

    def add_S(self, name, a, b, R_on=0.0):
        self.S.append((name, self.n(a), self.n(b), float(R_on)))

    def add_D(self, name, a, b, V_f=0.0, R_on=0.0):   # anode a, cathode b
        self.D.append((name, self.n(a), self.n(b), float(V_f), float(R_on)))

    # sizes ------------------------------------------------------------------------------------
    @property
    def nx(self):
        return len(self.C) + len(self.L)

    @property
    def ns(self):
        return 1 + 2 * len(self.omegas)

    @property
    def ny(self):
        return self.nx + self.ns

    def mass(self):
        return np.array([c[3] for c in self.C] + [l[3] for l in self.L])

    def x0(self):
        return np.array([c[4] for c in self.C] + [l[4] for l in self.L])

    def s_at(self, t):
        s = np.zeros(self.ns)
        s[0] = 1.0
        for k, w in enumerate(self.omegas):
            s[1 + 2 * k], s[2 + 2 * k] = np.cos(w * t), np.sin(w * t)
        return s

    def Omega(self):
        O = np.zeros((self.ns, self.ns))
        for k, w in enumerate(self.omegas):
            c, s = 1 + 2 * k, 2 + 2 * k
            O[c, s], O[s, c] = -w, w
        return O

    def src_row(self, spec):
        """Row u = r . s of a voltage source."""
        r = np.zeros(self.ns)
        if spec[0] == "dc":
            r[0] = spec[1]
        else:
            _, amp, k, ph, off = spec
            r[0] = off
            r[1 + 2 * k] += amp * np.sin(ph)      # amp sin(wt + ph) = amp (sin wt cos ph + cos wt sin ph)
            r[2 + 2 * k] += amp * np.cos(ph)
        return r


# ----------------------------------------------------------------------------------------------
# State-held MNA of one mode:  K z = F y,  x' = Sx z
#   z = [node voltages | V-source currents | capacitor currents | switch currents | diode currents]
# ----------------------------------------------------------------------------------------------
def layout(c):
    N, nV, nC, nS, nD = len(c.nodes), len(c.V), len(c.C), len(c.S), len(c.D)
    o = dict(v=0, iV=N, iC=N + nV, iS=N + nV + nC, iD=N + nV + nC + nS)
    o["nz"] = N + nV + nC + nS + nD
    return o


def _kcl(M, a, b, col, val=1.0):
    """branch current (column col) flows a -> b: leaves a (+), enters b (-)."""
    if a >= 0:
        M[a, col] += val
    if b >= 0:
        M[b, col] -= val


def _vrow(M, row, a, b, val=1.0):
    if a >= 0:
        M[row, a] += val
    if b >= 0:
        M[row, b] -= val


def mna_held(c, sw_on, d_on):
    o = layout(c)
    nz, nx = o["nz"], c.nx
    K = np.zeros((nz, nz))
    F = np.zeros((nz, c.ny))
    for a, b, R in c.R:
        for p, q, sgn in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            if p >= 0 and q >= 0:
                K[p, q] += sgn / R
    for j, (_, a, b, spec) in enumerate(c.V):
        col = o["iV"] + j
        _kcl(K, a, b, col)
        _vrow(K, col, a, b)
        F[col, nx:] = c.src_row(spec)
    for j, (_, a, b, C, _) in enumerate(c.C):
        col = o["iC"] + j
        _kcl(K, a, b, col)
        _vrow(K, col, a, b)
        F[col, j] = 1.0
    for j, (_, a, b, L, _) in enumerate(c.L):    # inductor current (a state) leaves a, enters b
        xj = len(c.C) + j
        if a >= 0:
            F[a, xj] -= 1.0
        if b >= 0:
            F[b, xj] += 1.0
    for _, a, b, I in c.I:
        if a >= 0:
            F[a, nx] -= I
        if b >= 0:
            F[b, nx] += I
    for j, (_, a, b, R_on) in enumerate(c.S):
        col = o["iS"] + j
        _kcl(K, a, b, col)
        if sw_on[j]:
            _vrow(K, col, a, b)
            K[col, col] = -R_on
        else:
            K[col, col] = 1.0
    for j, (_, a, b, V_f, R_on) in enumerate(c.D):
        col = o["iD"] + j
        _kcl(K, a, b, col)
        if d_on[j]:
            _vrow(K, col, a, b)
            K[col, col] = -R_on
            F[col, nx] = V_f
        else:
            K[col, col] = 1.0
    Sx = np.zeros((nx, nz))
    for j, (_, a, b, C, _) in enumerate(c.C):
        Sx[j, o["iC"] + j] = 1.0 / C
    for j, (_, a, b, L, _) in enumerate(c.L):
        _vrow(Sx, len(c.C) + j, a, b, 1.0 / L)
    return K, F, Sx, o


class ModeError(RuntimeError):
    pass


class Mode:
    """Reduced dynamics of one mode. y' = A y on the constraint manifold Gc y = 0."""

    def __init__(self, c, sw_on, d_on):
        self.key = (tuple(sw_on), tuple(d_on))
        K, F, Sx, o = mna_held(c, sw_on, d_on)
        self.o = o
        nx, ns = c.nx, c.ns
        U, sv, Vt = svd(K)
        tol = 1e-11 * sv[0]
        r = int(np.sum(sv > tol))
        if r == K.shape[0]:
            Z = np.linalg.solve(K, F)
            self.Gc = np.zeros((0, c.ny))
            self.Rg = np.zeros((K.shape[0], 0))
        else:
            Nl, Rn = U[:, r:], Vt[r:].T
            Kp = (Vt[:r].T / sv[:r]) @ U[:, :r].T
            Gc = Nl.T @ F                                   # constraints on y (Nl orthonormal)
            Gc = np.where(np.abs(Gc) < 1e-12 * np.abs(F).max(), 0.0, Gc)
            Gx, Gs = Gc[:, :nx], Gc[:, nx:]
            keep = np.linalg.norm(Gx, axis=1) > 0
            if np.any(~keep & (np.linalg.norm(Gs, axis=1) > 0)):
                raise ModeError(f"mode {self.key}: ideal loop of voltage sources / closed switches")
            Gc, Gx, Gs = Gc[keep], Gx[keep], Gs[keep]
            W = Gx @ Sx @ Rn                                # d/dt of the constraints vs the free directions
            rhs = -Gx @ Sx @ Kp @ F
            rhs[:, nx:] -= Gs @ c.Omega()
            # pseudo-inverse and null space with an ABSOLUTE cut (a rounding-level W is zero)
            wscale = np.abs(Sx).max() * max(1.0, np.abs(Gx).max(initial=0.0))
            if W.size:
                Uw, sw_, Vwt = np.linalg.svd(W)
                rw = int(np.sum(sw_ > 1e-10 * wscale))
                Wp = (Vwt[:rw].T / sw_[:rw]) @ Uw[:, :rw].T
                nW = Vwt[rw:].T
            else:
                Wp = np.zeros((Rn.shape[1], 0))
                nW = np.eye(Rn.shape[1])
            resid = W @ Wp @ rhs - rhs
            if resid.size and np.abs(resid).max() > 1e-8 * max(1.0, np.abs(rhs).max()):
                raise ModeError(f"mode {self.key}: constraint not maintainable (ideal source loop / cut-set)")
            Z = Kp @ F + Rn @ Wp @ rhs
            Rg = Rn @ nW                                    # gauge directions: free, must not move any state
            if Rg.shape[1] and np.abs(Sx @ Rg).max() > 1e-9 * max(1.0, np.abs(Sx).max()):
                raise ModeError(f"mode {self.key}: a state derivative is undetermined (floating state)")
            keep = np.ones(Gc.shape[0], bool)
            self.Gc = Gc[keep]
            self.Rg = Rg
        # drop rounding-level entries left by the SVD (a dependent state must have an exactly zero row)
        Z = np.where(np.abs(Z) < 1e-13 * np.abs(Z).max(axis=1, keepdims=True), 0.0, Z)
        if self.Gc.shape[0]:
            G = self.Gc / np.abs(self.Gc).max(axis=1, keepdims=True)
            self.Gc = np.where(np.abs(G) < 1e-12, 0.0, G)
        self.Z = Z
        A = np.zeros((c.ny, c.ny))
        A[:nx] = Sx @ Z
        A[nx:, nx:] = c.Omega()
        self.A = A
        self.c = c
        self._eig()
        self._predicates()

    def _eig(self):
        lam, V = np.linalg.eig(self.A)
        self.wmax = float(np.max(np.abs(lam.imag))) if lam.size else 0.0
        self.rho = float(np.max(np.abs(lam))) if lam.size else 0.0
        kap = np.linalg.cond(V)
        self.modal = kap < 1e6
        if self.modal:
            self.lam, self.V, self.Vi = lam, V, np.linalg.inv(V)

    def flow(self, y, tau):
        if self.modal:
            return (self.V @ (np.exp(self.lam * tau) * (self.Vi @ y))).real
        return expm(self.A * tau) @ y

    def _predicates(self):
        """Rows P (on y) whose sign must stay >= 0 in this mode.
        ON diode: current >= 0.  OFF diode: gap V_f - v >= 0; a gap that depends on a gauge
        direction (floating island) is combined pairwise into gauge-free sums."""
        c, o, Z, Rg = self.c, self.o, self.Z, self.Rg
        rows, labels, gauge_rows = [], [], []
        _, d_on = self.key
        for j, (name, a, b, V_f, R_on) in enumerate(c.D):
            if d_on[j]:
                rows.append(Z[o["iD"] + j]); labels.append((name, "i"))
            else:
                r = np.zeros(c.ny); r[c.nx] = V_f
                g = np.zeros(Rg.shape[1])
                if a >= 0:
                    r -= Z[a]; g -= Rg[a]
                if b >= 0:
                    r += Z[b]; g += Rg[b]
                if np.abs(g).max(initial=0.0) < 1e-12:
                    rows.append(r); labels.append((name, "w"))
                else:
                    gauge_rows.append((name, r, g))
        # eliminate the gauge: for one island g is a multiple of one vector; pair opposite signs
        for i1 in range(len(gauge_rows)):
            for i2 in range(i1 + 1, len(gauge_rows)):
                n1, r1, g1 = gauge_rows[i1]
                n2, r2, g2 = gauge_rows[i2]
                # find a, b > 0 with a g1 + b g2 = 0
                k = np.argmax(np.abs(g1))
                if abs(g2[k]) < 1e-12 or np.sign(g1[k]) == np.sign(g2[k]):
                    continue
                al, be = abs(g2[k]), abs(g1[k])
                if np.abs(al * g1 + be * g2).max() > 1e-9 * (al + be):
                    continue
                rows.append((al * r1 + be * r2) / (al + be)); labels.append(((n1, n2), "w+w"))
        self.P = np.array(rows) if rows else np.zeros((0, c.ny))
        self.plabels = labels
        self.n_gauge = Rg.shape[1]


# ----------------------------------------------------------------------------------------------
# Lemke's method, lexicographic ratio test (finite for every LCP; Cottle-Pang-Stone 4.4)
# ----------------------------------------------------------------------------------------------
def lemke(M, q, max_pivots=None):
    """Solve w = M z + q, z, w >= 0, z.w = 0. Returns (z, w, pivots, status)."""
    n = len(q)
    if np.all(q >= 0):
        return np.zeros(n), q.copy(), 0, "trivial"
    max_pivots = max_pivots or 10 * (n + 1) ** 2      # bug trap only (lexicographic Lemke is finite)
    # tableau columns: w (0..n-1), z (n..2n-1), z0 (2n); rows: basic variables
    T = np.hstack([np.eye(n), -M, -np.ones((n, 1))])
    b = q.astype(float).copy()
    basis = list(range(n))
    scale = max(1.0, np.abs(M).max(), np.abs(q).max())
    tol = 1e-12 * scale

    def pivot(r, col):
        p = T[r, col]
        T[r] /= p; b[r] /= p
        for i in range(n):
            if i != r and T[i, col] != 0.0:
                f = T[i, col]
                T[i] -= f * T[r]; b[i] -= f * b[r]
        T[:, col] = 0.0; T[r, col] = 1.0

    def lexmin(cands, col):
        # lexicographic minimum of (b_i, B^-1 row_i) / d_i ; B^-1 = columns of the initial identity
        best = None
        for i in cands:
            key = np.concatenate(([b[i]], T[i, :n])) / T[i, col]
            if best is None:
                best, bi = key, i
                continue
            d = key - best
            nz = np.nonzero(np.abs(d) > 1e-13 * (1 + np.abs(best)))[0]
            if nz.size and d[nz[0]] < 0:
                best, bi = key, i
        return bi

    # first pivot: z0 enters; the row with the lexicographically smallest (b_i, B^-1_i) leaves
    col = 2 * n
    m = b.min()
    cand = [i for i in range(n) if b[i] <= m + 1e-14 * scale]
    r = cand[0]
    for i in cand[1:]:
        d = np.concatenate(([b[i]], T[i, :n])) - np.concatenate(([b[r]], T[r, :n]))
        nz = np.nonzero(np.abs(d) > 1e-14)[0]
        if nz.size and d[nz[0]] < 0:
            r = i
    leaving = basis[r]
    pivot(r, col)
    basis[r] = col
    pivots = 1
    entering = leaving + n if leaving < n else leaving - n
    while True:
        if pivots > max_pivots:
            return None, None, pivots, "pivot cap (bug)"
        d = T[:, entering]
        cands = [i for i in range(n) if d[i] > 1e-12 * max(1.0, np.abs(d).max())]
        if not cands:
            return None, None, pivots, "secondary ray"
        ratios = np.array([b[i] / d[i] for i in cands])
        mr = ratios.min()
        ties = [cands[k] for k in range(len(cands)) if ratios[k] <= mr + 1e-13 * (1 + abs(mr))]
        z0row = [i for i in ties if basis[i] == 2 * n]
        r = z0row[0] if z0row else (ties[0] if len(ties) == 1 else lexmin(ties, entering))
        leaving = basis[r]
        pivot(r, entering)
        basis[r] = entering
        pivots += 1
        if leaving == 2 * n:
            break
        entering = leaving + n if leaving < n else leaving - n
    z, w = np.zeros(n), np.zeros(n)
    for i, v in enumerate(basis):
        if v < n:
            w[v] = b[i]
        elif v < 2 * n:
            z[v - n] = b[i]
    return z, w, pivots, "solution"


# ----------------------------------------------------------------------------------------------
# Mode selection: LCP on the diode ports of the backward-Euler companion network of step h
# ----------------------------------------------------------------------------------------------
def port_behaviour(c, y, t, sw_on, h):
    """Affine set {(i, w)} of the diode ports (i: anode->cathode current, w = V_f + R_on i - v >= 0)
    of the BE companion network over [t, t+h] from the held state y(t). Returns (p0, Q): (i, w) = p0 + Q a."""
    o = layout(c)
    N, nV, nL, nS, k = len(c.nodes), len(c.V), len(c.L), len(c.S), len(c.D)
    iV, iL, iS = N, N + nV, N + nV + nL
    iD, iW = iS + nS, iS + nS + k
    nu, ne = iW + k, iS + nS + k
    A = np.zeros((ne, nu)); r = np.zeros(ne)
    x, s = y[:c.nx], y[c.nx:]
    s1 = c.s_at(t + h) if c.omegas else s
    sig = 1.0 / h
    for a, b, R in c.R:
        for p, q, sg in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            if p >= 0 and q >= 0:
                A[p, q] += sg / R
    for j, (_, a, b, C, _) in enumerate(c.C):     # i_C = sig C (v_a - v_b - v_C(t))
        for p, q, sg in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            if p >= 0 and q >= 0:
                A[p, q] += sg * sig * C
        if a >= 0: r[a] += sig * C * x[j]
        if b >= 0: r[b] -= sig * C * x[j]
    for j, (_, a, b, spec) in enumerate(c.V):
        _kcl(A, a, b, iV + j); _vrow(A, iV + j, a, b); r[iV + j] = c.src_row(spec) @ s1
    for j, (_, a, b, L, _) in enumerate(c.L):     # v_a - v_b - sig L i = - sig L i(t)
        col = iL + j
        _kcl(A, a, b, col); _vrow(A, col, a, b); A[col, col] = -sig * L
        r[col] = -sig * L * x[len(c.C) + j]
    for _, a, b, I in c.I:
        if a >= 0: r[a] -= I
        if b >= 0: r[b] += I
    for j, (_, a, b, R_on) in enumerate(c.S):
        col = iS + j
        _kcl(A, a, b, col)
        if sw_on[j]:
            _vrow(A, col, a, b); A[col, col] = -R_on
        else:
            A[col, col] = 1.0
    for j, (_, a, b, V_f, R_on) in enumerate(c.D):   # v - R_on i + w = V_f
        row = iS + nS + j
        _kcl(A, a, b, iD + j)
        _vrow(A, row, a, b); A[row, iD + j] = -R_on; A[row, iW + j] = 1.0
        r[row] = V_f
    # equilibrate rows and columns (sig*C and sig*L span many decades), then solve in scaled form
    dr = 1.0 / np.maximum(np.abs(A).max(axis=1), 1e-300)
    As = A * dr[:, None]
    dc = 1.0 / np.maximum(np.abs(As).max(axis=0), 1e-300)
    As = As * dc[None, :]
    rs = r * dr
    sol_s, *_ = np.linalg.lstsq(As, rs, rcond=1e-13)
    if np.linalg.norm(As @ sol_s - rs) > 1e-8 * max(1.0, np.linalg.norm(rs)):
        raise ModeError("inconsistent port equations (ideal source loop)")
    sol = sol_s * dc
    Nb = null_space(As, rcond=1e-12) * dc[:, None]
    P = Nb[iD:iD + 2 * k]                            # port part of the solution set (gauge drops out)
    U, sv, _ = svd(P, full_matrices=False)
    Q = U[:, sv > 1e-10 * max(sv.max(initial=0), 1e-300)]
    p0 = sol[iD:iD + 2 * k]
    p0 = p0 - Q @ (Q.T @ p0)                         # the point of the port set nearest the origin
    return p0, Q


def _lcp_at(c, y, t, sw_on, prev_on, h):
    """One LCP on the BE companion of step h. Returns (status per diode: 1 ON, 0 OFF, -1 tie, pivots)."""
    k = len(c.D)
    if k == 0:
        return [], 0
    p0, Q = port_behaviour(c, y, t, sw_on, h)
    if Q.shape[1] != k:
        raise ModeError(f"port behaviour has dimension {Q.shape[1]} for {k} diodes")
    # scale currents to volts with one impedance scale so that the tableau is balanced
    # impedance scale from the circuit (volts per ampere), never from the port data: a port that is
    # a voltage source has no current row and would send a data-derived scale to zero
    vs = max([1.0] + [abs(d[3]) for d in c.D] + [np.abs(c.src_row(v[3])).max() for v in c.V] + list(np.abs(y[:len(c.C)])))
    is_ = max([1e-12] + [abs(I[3]) for I in c.I] + list(np.abs(y[len(c.C):c.nx])) + [vs / R for (_, _, R) in c.R])
    zs = vs / is_
    D = np.concatenate([np.full(k, zs), np.ones(k)])
    p0s, Qs = p0 * D, Q * D[:, None]
    # complementary basis: independent coordinate per diode (i_j if OFF before, w_j if ON before)
    # greedy pivoted choice, preferring the previous mode's basis unless the other coordinate of the
    # pair is 10x better conditioned (a near-singular basis turns the LCP data into rounding noise)
    if True:
        sel = []
        Bs = np.zeros((0, Qs.shape[1]))
        for j in range(k):
            best = None
            for cand in ((k + j, j) if prev_on[j] else (j, k + j)):
                row = Qs[cand]
                res = row - (Bs.T @ np.linalg.lstsq(Bs.T, row, rcond=None)[0] if Bs.shape[0] else 0)
                nr = np.linalg.norm(res)
                if best is None or nr > 10 * best[1]:
                    best = (cand, nr)
            sel.append(best[0]); Bs = Qs[sel]
        if np.linalg.cond(Qs[sel]) > 1e12:
            raise ModeError("no well-conditioned complementary basis")
    oth = [s + k if s < k else s - k for s in sel]
    Hs = Qs[oth] @ np.linalg.inv(Qs[sel])
    qv = p0s[oth] - Hs @ p0s[sel]
    z, w, piv, status = lemke(Hs, qv)
    if z is None:
        raise ModeError(f"LCP: {status} after {piv} pivots (network not passive at the diode ports?)")
    sc = max(1.0, np.abs(qv).max())
    if min(z.min(), w.min()) < -1e-9 * sc or np.abs(Hs @ z + qv - w).max() > 1e-8 * sc * max(1.0, np.abs(Hs).max()):
        raise ModeError("LCP solution failed its residual check")
    coords = np.empty(2 * k)
    coords[sel], coords[oth] = z, w
    i_ph, w_v = coords[:k] / zs, coords[k:]           # physical current (A), gap (V)
    # absolute tolerances on the circuit's own scales (never relative to the solution: at a
    # series-pair turn-off every current is at rounding level and must read as zero)
    vscale = max([1.0] + [abs(d[3]) for d in c.D] + [np.abs(c.src_row(v[3])).max() for v in c.V]
                 + list(np.abs(y[:len(c.C)])))
    iscale = max([1e-12] + [abs(I[3]) for I in c.I] + list(np.abs(y[len(c.C):c.nx]))
                 + [vscale / R for (_, _, R) in c.R])
    tol_w = 1e-11 * vscale
    # current resolution of the companion network: a gap rounding of tol_w through the stiffest
    # capacitive port (sigma C) is a current of tol_w sigma C
    tol_i = 1e-11 * iscale + tol_w * max([0.0] + [C[3] / h for C in c.C])
    st = []
    for j in range(k):
        if w_v[j] > tol_w:
            st.append(0)
        elif i_ph[j] > tol_i:
            st.append(1)
        else:
            st.append(-1)
    return st, piv


def select_modes(c, y, t, sw_on, prev_on, h, widen=(100.0, 1e4)):
    """One LCP per event at step h. A diode left at i = w = 0 within the resolution (a tie) is
    decided by the same LCP at a larger step (the next term of the rational complementarity problem
    becomes visible); a diode still tied is OFF (both modes consistent). Returns (d_on, info)."""
    k = len(c.D)
    if k == 0:
        return (), dict(pivots=0, pivots_total=0, status="no diodes", ties=0, widen=0)
    st, piv = _lcp_at(c, y, t, sw_on, prev_on, h)
    first, pivots, nw = piv, piv, 0
    for f in widen:
        if -1 not in st:
            break
        st2, p2 = _lcp_at(c, y, t, sw_on, prev_on, h * f)
        pivots += p2; nw += 1
        st = [s2 if s == -1 else s for s, s2 in zip(st, st2)]
    ties = sum(1 for s in st if s == -1)
    return tuple(s == 1 for s in st), dict(pivots=first, pivots_total=pivots, status="solution", ties=ties, widen=nw)


# ----------------------------------------------------------------------------------------------
# Projection at mode entry
# ----------------------------------------------------------------------------------------------
def project(c, mode, y):
    if mode.Gc.shape[0] == 0:
        return y, None
    nx = c.nx
    Gx, Gs = mode.Gc[:, :nx], mode.Gc[:, nx:]
    viol = Gx @ y[:nx] + Gs @ y[nx:]
    m = c.mass()
    Mi = 1.0 / m
    S = (Gx * Mi) @ Gx.T
    lam = np.linalg.lstsq(S, viol, rcond=1e-12)[0]
    dx = -(Mi * (Gx.T @ lam))
    y2 = y.copy(); y2[:nx] += dx                    # always land on the manifold
    dW = 0.5 * np.sum(m * dx ** 2)
    W = 0.5 * np.sum(m * y[:nx] ** 2)
    W2 = 0.5 * np.sum(m * y2[:nx] ** 2)
    if dW <= 1e-12 * max(W, W2, 1e-30):                  # rounding-level: consistent commutation, no record
        return y2, None
    return y2, dict(dx=dx, dW=dW)


# ----------------------------------------------------------------------------------------------
# Event-driven exact simulation
# ----------------------------------------------------------------------------------------------
class Result:
    pass


def simulate(c, t_end, gate=None, edges=(), h_e=1e-9, sample_times=(), y0=None, t0=0.0,
             confirm=True, record_events=True, max_events=10 ** 7):
    """gate(t) -> tuple of switch states on (t, next edge]; edges: sorted gate-edge times."""
    ns_ = len(c.S)
    gate = gate or (lambda t: (False,) * ns_)
    y = np.concatenate([c.x0(), c.s_at(t0)]) if y0 is None else y0.copy()
    t = t0
    cache = {}

    def get_mode(sw, d):
        key = (tuple(sw), tuple(d))
        if key not in cache:
            cache[key] = Mode(c, sw, d)
        return cache[key]

    res = Result()
    res.events, res.impulses = [], []
    res.pivots, res.ties, res.wrong_mode, res.rcp_disagree, res.immediate = [], 0, 0, 0, 0
    res.widen = 0
    res.pivots_total = []
    samples = sorted(sample_times)
    res.samples = []
    si = 0
    edges = sorted(e for e in edges if e > t0)
    ei = 0
    sw = gate(t)
    d_prev = (False,) * len(c.D)

    def do_select(t, y, sw, d_prev):
        d, info = select_modes(c, y, t, sw, d_prev, h_e)
        if confirm and len(c.D):
            d2, _ = select_modes(c, y, t, sw, d_prev, h_e / 10)
            if d2 != d:
                res.rcp_disagree += 1
                d, info = select_modes(c, y, t, sw, d_prev, h_e / 1000)
        res.pivots.append(info["pivots"]); res.ties += info["ties"]; res.widen += info["widen"]
        res.pivots_total.append(info["pivots_total"])
        return d

    d = do_select(t, y, sw, d_prev)
    mode = get_mode(sw, d)
    y, imp = project(c, mode, y)
    if imp:
        res.impulses.append((t, imp["dW"]))
    nev = 0
    zero_adv = 0
    while t < t_end - 1e-15 * max(1.0, abs(t_end)):
        t_sched = min(edges[ei] if ei < len(edges) else np.inf, t_end)
        # scan for the first predicate crossing (>= 0 to < 0) in (t, t_sched]
        P = mode.P
        span = t_sched - t
        scan = span
        if mode.wmax > 0:
            scan = min(scan, np.pi / (4 * mode.wmax))
        scan = min(scan, span / 1.0)
        nsteps = max(1, int(np.ceil(span / scan)))
        g0 = P @ y if P.shape[0] else np.zeros(0)
        gscale = 1e-9 * max(1.0, np.abs(P).max() * np.abs(y).max()) if P.shape[0] else 0.0
        hit = None
        ta, ya, ga = 0.0, y, g0
        for kstep in range(1, nsteps + 1):
            tb = span * kstep / nsteps
            yb = mode.flow(y, tb)
            gb = P @ yb if P.shape[0] else np.zeros(0)
            # crossing into violation: ga >= -tol and gb < -tol
            bad = np.nonzero(gb < -gscale)[0]
            if bad.size:
                hit = (ta, tb, ga, gb, bad)
                break
            ta, ya, ga = tb, yb, gb
        # sample outputs inside (t, t_e]
        if hit is None:
            t_new = t_sched
        else:
            a, b, _, _, bad = hit
            # locate the earliest root among the violated predicates by bisection on a bracket
            best = None
            for j in bad:
                f = lambda tau: P[j] @ mode.flow(y, tau)
                lo, hi = a, b
                if f(lo) < -gscale:          # violated at the bracket start: immediate violation
                    if lo == 0.0:
                        res.immediate += 1
                    best = (lo, j) if best is None or lo < best[0] else best
                    continue
                for _ in range(200):
                    if hi - lo <= max(4 * EPS * abs(t + hi), 1e-13 * span):
                        break
                    m_ = 0.5 * (lo + hi)
                    if f(m_) < 0:
                        hi = m_
                    else:
                        lo = m_
                if best is None or hi < best[0]:
                    best = (hi, j)
            t_new = t + best[0]
        while si < len(samples) and samples[si] <= t_new:
            res.samples.append((samples[si], mode.flow(y, samples[si] - t)))
            si += 1
        y = mode.flow(y, t_new - t)
        zero_adv = zero_adv + 1 if t_new == t else 0
        if zero_adv > 4 * (len(c.D) + 1):
            raise ModeError(f"zero-time event loop at t = {t:.9e}")
        t = t_new
        if hit is None and ei < len(edges) and t == edges[ei]:
            ei += 1
            kind = "gate"
        elif hit is None:
            kind = "end"
        else:
            kind = "diode"
        if kind == "end":
            break
        sw_new = gate(t) if kind == "gate" else sw
        d_new = do_select(t, y, sw_new, mode.key[1])
        if kind == "diode" and d_new == mode.key[1] and sw_new == sw:
            res.wrong_mode += 1                       # the LCP kept a mode whose predicate is violated
        new = get_mode(sw_new, d_new)
        y, imp = project(c, new, y)
        if imp:
            res.impulses.append((t, imp["dW"]))
        if record_events:
            res.events.append((t, kind, sw_new, d_new))
        mode, sw = new, sw_new
        nev += 1
        if nev > max_events:
            raise RuntimeError("event cap")
    while si < len(samples):
        res.samples.append((samples[si], mode.flow(y, samples[si] - t)))
        si += 1
    res.t, res.y, res.mode, res.cache = t, y, mode, cache
    return res


def pwm(T, D, t0=0.0, idx=0, nsw=1, comp=None):
    """gate function and edge list for one switch (optionally a complementary one)."""
    def gate(t):
        ph = ((t - t0) / T) % 1.0
        if ph > 1.0 - 1e-9:
            ph = 0.0
        on = (t >= t0) and (ph < D - 1e-9)   # edge times are exact; take the right limit there
        st = [False] * nsw
        st[idx] = on
        if comp is not None:
            st[comp] = not on
        return tuple(st)
    return gate


def pwm_edges(T, D, t_end, t0=0.0):
    e = []
    k = 0
    while t0 + k * T <= t_end:
        e += [t0 + k * T, t0 + (k + D) * T]
        k += 1
    return [x for x in e if x <= t_end]
