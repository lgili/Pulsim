# Prototype for add-high-order-stiff-integrator.
# Kennedy & Carpenter (2003) ARK4(3)6L[2]SA implicit part = ESDIRK4(3)6L[2]SA, gamma = 1/4.
# (1) verify the tableau (order conditions, stage order 2, stiff accuracy, A/L-stability);
# (2) ideal LC over 1000 periods, same controller for TR-BDF2 and ESDIRK4(3): steps, solves, energy, phase.
import numpy as np
from fractions import Fraction as F
g = F(1, 4)
A = [[0]*6 for _ in range(6)]
A[1][0] = F(1, 4); A[1][1] = g
A[2][0] = F(8611, 62500); A[2][1] = F(-1743, 31250); A[2][2] = g
A[3][0] = F(5012029, 34652500); A[3][1] = F(-654441, 2922500); A[3][2] = F(174375, 388108); A[3][3] = g
A[4][0] = F(15267082809, 155376265600); A[4][1] = F(-71443401, 120774400); A[4][2] = F(730878875, 902184768); A[4][3] = F(2285395, 8070912); A[4][4] = g
A[5] = [F(82889, 524892), F(0), F(15625, 83664), F(69875, 102672), F(-2260, 8211), g]
b = A[5][:]
bh = [F(4586570599, 29645900160), F(0), F(178811875, 945068544), F(814220225, 1159782912), F(-3700637, 11593932), F(61727, 225920)]
c = [sum(r) for r in A]
print("c =", [str(x) for x in c])
dot = lambda u, v: sum(x*y for x, y in zip(u, v))
Ac = [dot(r, c) for r in A]; Ac2 = [dot(r, [x*x for x in c]) for r in A]; AAc = [dot(r, Ac) for r in A]
c2 = [x*x for x in c]; c3 = [x**3 for x in c]
def conds(w):
    return {"1": dot(w, [1]*6)-1, "2": dot(w, c)-F(1, 2), "3a": dot(w, c2)-F(1, 3), "3b": dot(w, Ac)-F(1, 6),
            "4a": dot(w, c3)-F(1, 4), "4b": dot(w, [x*y for x, y in zip(c, Ac)])-F(1, 8), "4c": dot(w, Ac2)-F(1, 12), "4d": dot(w, AAc)-F(1, 24)}
print("b  order-condition residuals:", {k: float(v) for k, v in conds(b).items()})
print("bh order-condition residuals:", {k: float(v) for k, v in conds(bh).items()})
print("stage order 2 residual (A c - c^2/2), stages 2..6:", [float(Ac[i]-c2[i]/2) for i in range(1, 6)])
An = np.array([[float(x) for x in r] for r in A]); bn = np.array([float(x) for x in b]); bhn = np.array([float(x) for x in bh])
def R(z, w=bn):
    return 1 + z*w@np.linalg.solve(np.eye(6) - z*An, np.ones(6))
ys = np.logspace(-3, 6, 4000)
print("max |R(iy)| on the imaginary axis:", max(abs(R(1j*y)) for y in ys), " |R(-1e12)| =", abs(R(-1e12)), " |Rhat(-1e12)| =", abs(R(-1e12, bhn)))
# phase/amplitude error per step for an oscillator at theta = omega*h
for th in [2*np.pi/20, 2*np.pi/10, 2*np.pi/5]:
    r = R(1j*th); print(f"   omega*h=2pi/{2*np.pi/th:.0f}: |R|-1={abs(r)-1:+.2e}  phase err={np.angle(r)-th:+.2e} rad/step")

# ---- generic stiffly-accurate ESDIRK integrator on M y' = J y (linear), one LU per h ----
def tableau_trbdf2():
    gm = 2-np.sqrt(2); d = gm/2; w = np.sqrt(2)/4
    A = np.array([[0, 0, 0], [d, d, 0], [w, w, d]]); bhat = np.array([(1-w)/3, (3*w+1)/3, d/3])
    return A, A[-1].copy(), bhat, 3      # error estimate O(h^3) -> exponent 1/3
def tableau_kc43():
    return An, bn, bhn, 4                # embedded order 3 -> local error O(h^4) -> exponent 1/4
def integrate(tab, Jm, y0, t_end, rtol, atol, h0):
    A, b, bhat, q = tab; s = len(b); d = A[1, 1]
    t = 0.0; y = y0.copy(); h = h0; nacc = nrej = nsolve = nfac = 0; hlast = None
    I = np.eye(len(y0)); out_t = [0.0]; out_y = [y0.copy()]
    while t < t_end*(1-1e-14):
        h = min(h, t_end-t)
        if h != hlast: Minv = np.linalg.inv(I - h*d*Jm); nfac += 1; hlast = h
        K = np.zeros((s, len(y0))); K[0] = Jm@y
        for i in range(1, s):
            rhs = y + h*(A[i, :i]@K[:i])
            Y = Minv@rhs; nsolve += 1
            K[i] = Jm@Y
        ynew = Y                                         # stiffly accurate
        err = h*((b-bhat)@K)
        sc = atol + rtol*np.maximum(abs(y), abs(ynew))
        e = np.sqrt(np.mean((err/sc)**2))
        if e <= 1.0:
            t += h; y = ynew; nacc += 1; out_t.append(t); out_y.append(y.copy())
        else:
            nrej += 1
        h = h*min(5.0, max(0.2, 0.9*max(e, 1e-10)**(-1.0/q)))
    return np.array(out_t), np.array(out_y), nacc, nrej, nsolve, nfac
L, C, V0 = 100e-6, 100e-9, 10.0; w0 = 1/np.sqrt(L*C); T0 = 2*np.pi/w0; NP = 1000
Jm = np.array([[0, -1/C], [1/L, 0]]); y0 = np.array([V0, 0.0])      # y = [v, i]: C v' = -i, L i' = v
E = lambda y: 0.5*C*y[:, 0]**2 + 0.5*L*y[:, 1]**2
print("ideal LC, 1000 periods (same RMS norm and I-controller for both methods)")
for rtol, atol in [(1e-4, 1e-8), (1e-5, 1e-8), (1e-6, 1e-9), (1e-8, 1e-11)]:
    for name, tab in [("TR-BDF2", tableau_trbdf2()), ("ESDIRK4(3)6L[2]SA", tableau_kc43())]:
        t, y, na, nr, ns, nf = integrate(tab, Jm, y0, NP*T0, rtol, atol, T0/200)
        ex = V0*np.cos(w0*t)
        print(f"  rtol={rtol:.0e} {name:18s}: {na/NP:7.1f} steps/period, {ns/NP:7.1f} solves/period, rejects {nr}, factorizations {nf}, E_end/E0-1={E(y)[-1]/E(y)[0]-1:+.3e}, max|v-exact|={abs(y[:, 0]-ex).max():.3e} V")
