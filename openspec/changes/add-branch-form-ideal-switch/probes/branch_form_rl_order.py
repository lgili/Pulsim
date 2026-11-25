# numpy only. Run from this directory (it reads branch_form.py).
import numpy as np
exec(open("branch_form.py").read().split('print("(1)')[0])
h = 1e-3/20; t = h*np.arange(1, 201); ex = 10*(1-np.exp(-t/1e-3))
for N in [20, 40, 80]:
    h = 1e-3/N; t = h*np.arange(1, 10*N+1); ex = 10*(1-np.exp(-t/1e-3))
    o, c = rl_switch(branch=True, R_on=0.0, g_off=0.0, closed=True, h=h, steps=10*N, be_first=2)
    o2, c2 = rl_switch(branch=False, R_on=1e-3, g_off=1e-9, closed=True, h=h, steps=10*N, be_first=2)
    ex2 = 10/1.001*(1-np.exp(-t*1.001/1e-3))
    print(f"h=tau/{N}: branch R_on=0 max|i-exact|={abs(o[:,2]-ex).max():.3e} A ; conductance 1e3 vs its own exact {abs(o2[:,2]-ex2).max():.3e} A ; conductance 1e3 vs the IDEAL-switch exact {abs(o2[:,2]-ex).max():.3e} A")
