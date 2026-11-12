# Probe of the CURRENT kernel (needs PYTHONPATH=<worktree>/python): dsed, V_th = 0, every switch open, state at rest.
import numpy as np, pulsim as p, warnings
warnings.simplefilter("ignore")
Vin,L,C=24.0,100e-6,47e-6; gon,goff=1e3,1e-9; T=1e-5
def run(vth, x0=None, t_end=1e-6, goff_s=goff):
    b=p.CircuitBuilder()
    b.add_voltage_source("Vin","vin","gnd",Vin); b.add_switch("S","vin","sw",gon,goff_s); b.add_diode("D1","gnd","sw",gon,goff,vth)
    b.add_inductor("L","sw","vout",L); b.add_capacitor("C","vout","gnd",C); b.add_resistor("RL","vout","gnd",50.0)
    n=b.graph.num_switches
    def fn(t): return p.SwitchStateMask(n)     # everything open
    kw = {} if x0 is None else dict(initial_state=x0)
    return p.simulate(b,t_end=t_end,engine='dsed',switch_fn=fn,**kw)
for lab, kw in [("x0=0", {}), ("x0=[vC=1,iL=0]", dict(x0=[1.0,0.0])), ("x0=[0,1e-3]", dict(x0=[0.0,1e-3])), ("x0=[0,-1e-3]", dict(x0=[0.0,-1e-3])), ("S g_off=2e-9", dict(goff_s=2e-9)), ("S g_off=0.5e-9", dict(goff_s=0.5e-9))]:
    try:
        r=run(0.0, **kw); print(f"{lab}: ok  v(sw) first/last = {np.asarray(r.v('sw'))[0]:.6g} / {np.asarray(r.v('sw'))[-1]:.6g}  iL last={np.asarray(r.i('L'))[-1]:.3e}")
    except Exception as e: print(f"{lab}: {type(e).__name__}: {str(e)[:120]}")
