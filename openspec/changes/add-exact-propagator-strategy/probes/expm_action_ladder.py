"""Re-centred scope of add-exact-propagator-strategy: when is a dense propagator right, when the
action of the exponential (Al-Mohy & Higham 2011, scipy.sparse.linalg.expm_multiply), and when must
the strategy hand over to the implicit stepper?  Plus: a sine source propagated exactly through the
augmented system, against a source held constant over the step.

Run: VECLIB_MAXIMUM_THREADS=1 nice -n 15 python3 expm_action_ladder.py   (numpy, scipy; ~1-2 min)

Circuit: RLC ladder driven by 10 V DC, m sections of (series L = 10 uH with 10 mOhm, shunt C = 1 uF),
10 Ohm load; n = 2 m states. Step h = 1 us (T/10 at 100 kHz). The stiff variant adds a 1 Ohm / 1 nF
snubber across every shunt capacitor (tau = 1 ns, ||A h|| ~ 1e3).
Timings are single-thread wall clock on a shared machine: read the RATIOS and the matvec counts.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm, solve
from scipy.sparse.linalg import expm_multiply, LinearOperator, splu


def ladder(m, snub=False, L=10e-6, C=1e-6, Rs=10e-3, Rload=10.0, Cs=1e-9, Rsn=1.0):
    """States: x = [i_1, v_1, i_2, v_2, ...] (+ snubber capacitor voltages); returns A (csr), b."""
    n = 2 * m + (m if snub else 0)
    A = sp.lil_matrix((n, n)); b = np.zeros(n)
    for k in range(m):
        i, v = 2 * k, 2 * k + 1
        vin = None if k == 0 else 2 * k - 1
        A[i, i] = -Rs / L
        A[i, v] = -1 / L
        if vin is None:
            b[i] = 10.0 / L
        else:
            A[i, vin] = 1 / L
        A[v, i] = 1 / C
        if k + 1 < m:
            A[v, 2 * (k + 1)] = -1 / C
        else:
            A[v, v] += -1 / (Rload * C)
        if snub:
            s = 2 * m + k                       # snubber: Rsn in series with Cs across C
            # current into snubber = (v - vs)/Rsn
            A[v, v] += -1 / (Rsn * C); A[v, s] += 1 / (Rsn * C)
            A[s, v] = 1 / (Rsn * Cs); A[s, s] = -1 / (Rsn * Cs)
    return A.tocsr(), b


def aug(A, b):
    """[[A, b], [0, 0]] so that the constant source is a state (exact affine propagation)."""
    n = A.shape[0]
    return sp.bmat([[A, sp.csr_matrix(b.reshape(-1, 1))], [sp.csr_matrix((1, n)), sp.csr_matrix((1, 1))]]).tocsr()


class Counting(LinearOperator):
    def __init__(self, M):
        super().__init__(M.dtype, M.shape)
        self.M, self.n_mv = M, 0

    def _matvec(self, x):
        self.n_mv += 1
        return self.M @ x

    def _matmat(self, X):
        self.n_mv += X.shape[1]
        return self.M @ X

    def _adjoint(self):
        return Counting(self.M.T.conj())


h = 1e-6
steps = 20
print("dense  : Phi = expm(A_aug h) once (Pade 13, scaling and squaring), then Phi x per step (n^2)")
print("action : expm_multiply(A_aug h, x) per step (truncated Taylor, ~||A h||_1 matvecs, sparse)")
print("implicit: one sparse LU of (I - h/2 A) once, trapezoid step = 1 solve (2nd order, NOT exact) -- the hand-over")
print(f"{'case':16s} {'n':>5s} {'||Ah||_1':>9s} {'dense setup':>12s} {'dense/step':>11s} {'action/step':>12s} {'mv/step':>8s} "
      f"{'|dense-action|':>15s} {'LU setup':>9s} {'trap/step':>10s}")
rows = []
for snub in (False, True):
    for m in ((100, 250, 500, 1000) if not snub else (100, 250, 500)):
        A, b = ladder(m, snub)
        Aa = aug(A, b)
        n = A.shape[0]
        x0 = np.zeros(n + 1); x0[-1] = 1.0
        nrm = abs(Aa * h).sum(axis=0).max()
        # dense
        t1 = time.perf_counter(); Phi = expm(Aa.toarray() * h); t_set = time.perf_counter() - t1
        x = x0.copy(); t1 = time.perf_counter()
        for _ in range(steps):
            x = Phi @ x
        t_dense = (time.perf_counter() - t1) / steps
        xd = x
        # action
        op = Counting(Aa * h)
        x = x0.copy(); t1 = time.perf_counter()
        for _ in range(steps):
            x = expm_multiply(op, x, traceA=float((Aa * h).diagonal().sum()))
        t_act = (time.perf_counter() - t1) / steps
        xa = x
        diff = np.abs(xd - xa).max() / max(1.0, np.abs(xd).max())
        # implicit trapezoid hand-over (sparse LU once)
        I = sp.identity(n, format="csc")
        t1 = time.perf_counter(); lu = splu((I - 0.5 * h * A).tocsc()); Mr = (I + 0.5 * h * A).tocsr(); t_lu = time.perf_counter() - t1
        x = np.zeros(n); t1 = time.perf_counter()
        for _ in range(steps):
            x = lu.solve(Mr @ x + h * b)
        t_tr = (time.perf_counter() - t1) / steps
        tr_err = np.abs(x - xd[:n]).max() / max(1.0, np.abs(xd[:n]).max())
        name = "ladder+snubber" if snub else "ladder"
        print(f"{name:16s} {n:5d} {nrm:9.2e} {t_set * 1e3:10.1f}ms {t_dense * 1e6:9.1f}us {t_act * 1e6:10.1f}us "
              f"{op.n_mv / steps:8.1f} {diff:15.1e} {t_lu * 1e3:7.2f}ms {t_tr * 1e6:8.1f}us  (trap vs exact {tr_err:.1e})")
        rows.append((name, n, nrm, t_set, t_dense, t_act, op.n_mv / steps, Aa.nnz))

print()
print("flop model (the scipy action call carries ~300 us of Python overhead per step, so wall time above")
print("overstates it): dense step 2 n^2, action step 2 nnz mv, dense setup ~ 2 n^3 (6 + log2 ||Ah||) for Pade 13")
for name, n, nrm, t_set, t_dense, t_act, mv, nnz in rows:
    fd, fa = 2.0 * (n + 1) ** 2, 2.0 * nnz * mv
    fs = 2.0 * (n + 1) ** 3 * (6 + max(0.0, np.log2(nrm)))
    be = fs / (fa - fd) if fa > fd else float("inf")
    print(f"   {name:16s} n={n:5d} nnz={nnz:6d}: dense {fd:9.2e} flop/step, action {fa:9.2e} flop/step"
          f" (action/dense {fa / fd:7.2f}); dense setup {fs:8.1e} flop = {be:9.0f} action steps")

print()
print("Sine source: R = 1 Ohm, L = 1 mH driven by 100 sin(2 pi 50 t), exact vs closed form at h = T/20 over 5 periods")
R, L, w = 1.0, 1e-3, 2 * np.pi * 50
# augmented state z = [i, cos wt, sin wt]
Az = np.array([[-R / L, 0.0, 100.0 / L], [0.0, 0.0, -w], [0.0, w, 0.0]])
Th = 2 * np.pi / w
hh = Th / 20
Phi = expm(Az * hh)
z = np.array([0.0, 1.0, 0.0])
# source held constant over the step (zero-order hold of u, the 'b constant per step' shortcut)
Ar = np.array([[-R / L, 1 / L], [0.0, 0.0]]); Phr = expm(Ar * hh)
iz = 0.0
err_aug, err_zoh = 0.0, 0.0
for k in range(1, 101):
    t = k * hh
    z = Phi @ z
    u0 = 100 * np.sin(w * (t - hh))
    iz = (Phr @ np.array([iz, u0]))[0]
    tau = L / R
    Z = np.hypot(R, w * L); ph = np.arctan2(w * L, R)
    exact = 100 / Z * (np.sin(w * t - ph) + np.sin(ph) * np.exp(-t / tau))
    err_aug = max(err_aug, abs(z[0] - exact)); err_zoh = max(err_zoh, abs(iz - exact))
print(f"   augmented (oscillator states, one expm per step size): max |i - exact| = {err_aug:.1e} A")
print(f"   source held at its step-start value:                   max |i - exact| = {err_zoh:.1e} A  (first order in h)")
print(f"   peak current {100 / np.hypot(R, w * L):.3f} A")
