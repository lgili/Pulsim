# Prototype for add-parameter-sensitivities on the implicit (MNA) path:
# trapezoidal + Newton on a sine source -> R -> Shockley diode -> (C || Rl). Unknowns x = [v1 (after R), v2 (cap)].
# Discretise-then-differentiate: the sensitivity system reuses the CONVERGED Newton Jacobian of each step.
#   F(x1, x0, p) = 0  =>  J s1 = -(dF/dx0) s0 - dF/dp        (one back-substitution per parameter per step)
import numpy as np
P0 = dict(R=1.0, C=100e-6, Rl=50.0, Is=1e-9, nVt=0.05)
names = list(P0)
def run(p, sens=True, N=4000, t_end=40e-3):
    h = t_end/N; R, C, Rl, Is, nVt = (p[k] for k in names)
    vs = lambda t: 20*np.sin(2*np.pi*50*t)
    idio = lambda v: Is*(np.exp(v/nVt)-1); gdio = lambda v: Is/nVt*np.exp(v/nVt)
    x = np.zeros(2); S = np.zeros((2, 5)); it_tot = 0
    # f(x,t,p): node currents (static part); q(x,p): charge on node 2
    def f(x, t):
        iD = idio(x[0]-x[1]); return np.array([(x[0]-vs(t))/R + iD, -iD + x[1]/Rl])
    def dfdx(x):
        g = gdio(x[0]-x[1]); return np.array([[1/R+g, -g], [-g, g+1/Rl]])
    def dfdp(x, t):
        vd = x[0]-x[1]; iD = idio(vd); e = np.exp(vd/nVt)
        dR = np.array([-(x[0]-vs(t))/R**2, 0]); dC = np.zeros(2); dRl = np.array([0, -x[1]/Rl**2])
        dIs = np.array([e-1, -(e-1)]); dn = np.array([-Is*e*vd/nVt**2, Is*e*vd/nVt**2])
        return np.stack([dR, dC, dRl, dIs, dn], axis=1)
    Cm = np.array([[0, 0], [0, C]]); dCm = np.array([[0, 0], [0, 1.0]])
    f0 = f(x, 0.0); t = 0.0
    for k in range(N):
        t1 = t + h; x1 = x.copy()
        for it in range(50):                      # trap: Cm (x1-x)/h + (f(x1)+f(x))/2 = 0
            F = Cm@(x1-x)/h + 0.5*(f(x1, t1)+f0)
            J = Cm/h + 0.5*dfdx(x1)
            dx = np.linalg.solve(J, -F); x1 = x1 + dx; it_tot += 1
            if np.abs(dx).max() < 1e-13: break
        if sens:
            J = Cm/h + 0.5*dfdx(x1)
            rhs = (Cm/h - 0.5*dfdx(x))@S - 0.5*(dfdp(x1, t1)+dfdp(x, t))
            rhs[:, 1] -= dCm@(x1-x)/h
            S = np.linalg.solve(J, rhs)          # same matrix, 5 right-hand sides
        x = x1; f0 = f(x, t1); t = t1
    return x, S, it_tot
x, S, its = run(P0)
print("x(40 ms) =", x, " Newton iterations/step =", its/4000)
for j, n in enumerate(names):
    d = 1e-6*P0[n]; pp = dict(P0); pm = dict(P0); pp[n] += d; pm[n] -= d
    fd = (run(pp, False)[0] - run(pm, False)[0])/(2*d)
    print(f"  d x/d{n:3s}: forward {S[:, j]}  central FD {fd}  rel diff {np.abs(S[:, j]-fd).max()/np.abs(fd).max():.2e}")
