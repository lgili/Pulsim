# Prototype for the per-mask discrete update x <- Phi x + Gamma (fixed step), audit's CCM buck (R=1, D=0.5, t0=0).
# Reference: 50-digit mpmath propagation of the same piecewise-LTI system.
import numpy as np, mpmath as mp
from scipy.linalg import expm
mp.mp.dps = 50
Vin, L, C, gon, goff, T, R, D = 24.0, 100e-6, 47e-6, 1e3, 1e-9, 1e-5, 1.0, 0.5
def AB(gs, gd):
    k = 1/(gs+gd); return np.array([[-k/L, -1/L], [1/C, -1/(R*C)]]), np.array([gs*Vin*k/L, 0.0])
def aug(A, b):
    M = np.zeros((3, 3)); M[:2, :2] = A; M[:2, 2] = b; return M
def PhiGam_eig(A, b, h):
    lam, V = np.linalg.eig(A); Vi = np.linalg.inv(V); s = lam*h
    Phi = ((V*np.exp(s))@Vi).real
    Gam = ((V*(h*np.expm1(s)/s))@Vi@b).real
    return Phi, Gam
def PhiGam_pade(A, b, h):
    E = expm(aug(A, b)*h); return E[:2, :2], E[:2, 2]
nper = 40
# reference
Mon = mp.matrix(aug(*AB(gon, goff)).tolist()); Moff = mp.matrix(aug(*AB(goff, gon)).tolist())
Eon = mp.expm(Mon*mp.mpf(D)*mp.mpf(T)); Eoff = mp.expm(Moff*(1-mp.mpf(D))*mp.mpf(T))
x = mp.matrix([0, 0, 1])
for n in range(nper): x = Eoff*(Eon*x)
ref = np.array([float(x[0]), float(x[1])]); print("reference x(40T) =", ref)
for N in [10, 50, 100]:
    h = T/N
    for name, fn in [("eig", PhiGam_eig), ("pade", PhiGam_pade)]:
        Pon, Gon = fn(*AB(gon, goff), h); Poff, Goff = fn(*AB(goff, gon), h)
        x = np.zeros(2)
        for n in range(nper):
            for k in range(N):
                x = (Pon@x + Gon) if k < round(D*N) else (Poff@x + Goff)
        print(f"  Phi/Gamma ({name:4s}) fixed dt=T/{N}: err iL={x[0]-ref[0]:+.2e} A  vC={x[1]-ref[1]:+.2e} V")
print("flops per step for n=2 states: Phi x + Gamma = n*n mul + n*n add =", 2*2*2)
# DCM idle mode (S off, D off): the stiff one. Pade vs eig vs 50-digit reference for Phi(h), h = 2e-6
R = 50.0
A, b = AB(goff, goff); h = 2e-6
Er = mp.expm(mp.matrix(aug(A, b).tolist())*mp.mpf(h))
Pe, Ge = PhiGam_eig(A, b, h); Pp, Gp = PhiGam_pade(A, b, h)
print("idle mode ||A h|| =", np.linalg.norm(A*h), " Phi[1,1] ref", mp.nstr(Er[1, 1], 17), " eig err", float(Pe[1, 1]-Er[1, 1]), " pade err", float(Pp[1, 1]-Er[1, 1]))
print("  Gamma[1] ref", mp.nstr(Er[1, 2], 17), " eig err", float(Ge[1]-Er[1, 2]), " pade err", float(Gp[1]-Er[1, 2]))
