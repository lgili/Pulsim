# Prototype for add-branch-form-ideal-switch. Dense MNA, trapezoidal companion (BE where stated).
# Unknowns: node voltages, then branch currents (V source, inductor, switch).
# Switch row (branch form):  ON:  v_a - v_b - R_on*i = 0     OFF:  -g_off*(v_a - v_b) + i = 0
import numpy as np
np.set_printoptions(precision=6, linewidth=150)

def rl_switch(branch, R_on, g_off, closed, h, steps, be_first=0, i0=0.0, on_steps=None):
    """V(10) - switch - R(1) - L(1 mH) to gnd. nodes: 1=vin 2=a(after switch) 3=b(after R). x=[v1,v2,v3,iV,iL,(iS)]"""
    V, R, L = 10.0, 1.0, 1e-3
    n = 6 if branch else 5
    iL, vL = i0, 0.0; out = []; conds = []
    for k in range(steps):
        cl = closed if on_steps is None else (k < on_steps)
        be = k < be_first or (on_steps is not None and on_steps <= k < on_steps + be_first)
        G = np.zeros((n, n)); b = np.zeros(n)
        G[0, 3] = 1; G[3, 0] = 1; b[3] = V                     # V source
        G[1, 1] += 1/R; G[1, 2] -= 1/R; G[2, 1] -= 1/R; G[2, 2] += 1/R
        # inductor branch row: trap: v3 - (2L/h) iL = -(2L/h) iL_prev - vL_prev ; BE: v3 - (L/h) iL = -(L/h) iL_prev
        G[2, 4] += 1; G[4, 2] = 1
        if be: G[4, 4] = -L/h; b[4] = -L/h*iL
        else:  G[4, 4] = -2*L/h; b[4] = -2*L/h*iL - vL
        if branch:
            G[0, 5] += 1; G[1, 5] -= 1                         # KCL: i leaves node 1, enters node 2
            if cl: G[5, 0] = 1; G[5, 1] = -1; G[5, 5] = -R_on
            else:  G[5, 0] = -g_off; G[5, 1] = g_off; G[5, 5] = 1
        else:
            g = (1/R_on) if cl else g_off
            G[0, 0] += g; G[0, 1] -= g; G[1, 0] -= g; G[1, 1] += g
        x = np.linalg.solve(G, b); conds.append(np.linalg.cond(G))
        iL, vL = x[4], x[2]; out.append((x[1], x[2], x[4]))
    return np.array(out), max(conds)

print("(1) RL charge through a closed switch, 200 steps of h = tau/20, exact i = 10(1-exp(-t/tau))")
h = 1e-3/20; t = h*np.arange(1, 201); ex = 10*(1-np.exp(-t/1e-3))
for name, kw in [("conductance g_on=1e3", dict(branch=False, R_on=1e-3, g_off=1e-9)),
                 ("branch R_on=1e-3", dict(branch=True, R_on=1e-3, g_off=1e-9)),
                 ("branch R_on=0", dict(branch=True, R_on=0.0, g_off=0.0))]:
    o, c = rl_switch(closed=True, h=h, steps=200, **kw)
    ref = ex if "R_on=0" in name else 10/1.001*(1-np.exp(-t*1.001/1e-3))
    print(f"   {name:22s}: max|i-exact|={abs(o[:, 2]-ref).max():.3e} A (trap h=tau/20), cond(J)={c:.2e}")
print("(2) inductor cut-set: switch opens at step 100 with i_L = 9.93 A, no freewheel path")
for name, kw in [("conductance g_off=1e-9, trap", dict(branch=False, R_on=1e-3, g_off=1e-9, be_first=0)),
                 ("branch g_off=0, trap only", dict(branch=True, R_on=0.0, g_off=0.0, be_first=0)),
                 ("branch g_off=0, 2 BE steps after the event", dict(branch=True, R_on=0.0, g_off=0.0, be_first=2))]:
    o, c = rl_switch(closed=True, h=h, steps=106, on_steps=100, **kw)
    print(f"   {name}:")
    print("      i_L  steps 100..105:", o[100:106, 2])
    print("      v_b  steps 100..105:", o[100:106, 1], f" volt-seconds of the first step h*v_b = {h*o[100, 1]:+.6e} (L*i_L(t-) = {1e-3*o[99, 2]:.6e})")
print("(3) capacitor loop: C1=1u at 10 V paralleled with C2=3u at 2 V through an ideal switch (R_on=0); charge-conserving v = 4.0 V")
def caps(h, be_first, steps=4):
    C1, C2 = 1e-6, 3e-6; v = np.array([10.0, 2.0]); ic = np.zeros(2); out = []
    for k in range(steps):
        be = k < be_first
        G = np.zeros((3, 3)); b = np.zeros(3)          # x = [v1, v2, iS]
        for j, Cj in enumerate((C1, C2)):
            if be: G[j, j] += Cj/h; b[j] += Cj/h*v[j]
            else:  G[j, j] += 2*Cj/h; b[j] += 2*Cj/h*v[j] + ic[j]
        G[0, 2] += 1; G[1, 2] -= 1; G[2, 0] = 1; G[2, 1] = -1      # v1 - v2 = 0
        x = np.linalg.solve(G, b)
        for j, Cj in enumerate((C1, C2)):
            ic[j] = Cj/h*(x[j]-v[j]) if be else 2*Cj/h*(x[j]-v[j]) - ic[j]
        v = x[:2].copy(); out.append((x[0], x[1], x[2]))
    return np.array(out)
for bf in [0, 2]:
    o = caps(1e-7, bf)
    print(f"   {'trap only' if bf == 0 else '2 BE steps then trap'}: v1 per step {o[:, 0]}  i_S per step {o[:, 2]}  charge moved by step 1 = {1e-7*o[0, 2]:+.6e} C (exact 6.0e-06)")
print("(4) eigenvalues of the reduced state matrix, buck idle mode (S off, D off), L=100u, C=47u, R=50")
L, C, R = 100e-6, 47e-6, 50.0
for goff in [1e-9, 1e-12]:
    k = 1/(2*goff); A = np.array([[-k/L, -1/L], [1/C, -1/(R*C)]]); print(f"   resistor pair g_off={goff:g}: lambda = {np.linalg.eigvals(A)}")
print("   branch form g_off=0: i_L is algebraic (i_L = 0), one state v_C: lambda =", -1/(R*C))
