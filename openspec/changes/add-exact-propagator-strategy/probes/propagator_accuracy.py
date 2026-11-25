# Prototype for add-exact-propagator-strategy: (a) phi1 cancellation, (b) eigenbasis conditioning vs expm,
# (c) Phi/Gamma discrete update on the audit's buck (CCM + DCM steady-state v_out).
import numpy as np
from scipy.linalg import expm
np.set_printoptions(precision=12)
# (a) phi1: direct (exp(s)-1)/s vs expm1(s)/s around the series switch at |s|=1e-6
def phi1_code(s):
    if abs(s) < 1e-6: return 1 + s*(0.5 + s*(1/6 + s/24))
    return (np.exp(s) - 1)/s
for s in [1e-7, 0.99e-6, 1.01e-6, 3e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
    ref = np.expm1(s)/s
    print(f"phi1 s={s:.2e} rel err of code branch = {abs(phi1_code(s)-ref)/ref:.2e}")
# (b) near-defective eigenbasis: series RLC near critical damping, x=[i,v]
L, C = 100e-6, 100e-9
Rc = 2*np.sqrt(L/C)
def modal(A, b, x, h):
    lam, V = np.linalg.eig(A); Vi = np.linalg.inv(V)
    z = Vi@x; w = Vi@b; s = lam*h
    ph = np.where(abs(s) < 1e-6, 1 + s/2, np.expm1(s)/np.where(s == 0, 1, s))
    return (V@(np.exp(s)*z + h*ph*w)).real, np.linalg.cond(V)
def aug(A, b, x, h):
    n = len(x); M = np.zeros((n+1, n+1)); M[:n, :n] = A; M[:n, n] = b
    E = expm(M*h); return E[:n, :n]@x + E[:n, n]
print("near-critical RLC, h = 2*pi*sqrt(LC):")
for eps in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13, 0.0]:
    R = Rc*(1+eps)
    A = np.array([[-R/L, -1/L], [1/C, 0.0]]); b = np.array([10.0/L, 0.0]); x = np.array([0.3, 2.0])
    h = 2*np.pi*np.sqrt(L*C)
    xm, cv = modal(A, b, x, h); xa = aug(A, b, x, h)
    print(f"  R/Rc-1={eps:.0e} cond(V)={cv:.2e} modal-vs-expm rel err={np.linalg.norm(xm-xa)/np.linalg.norm(xa):.2e}")
