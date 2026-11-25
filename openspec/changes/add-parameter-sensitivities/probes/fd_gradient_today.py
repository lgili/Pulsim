# Probe of the CURRENT kernel (needs PYTHONPATH=<worktree>/python): how good is a finite-difference gradient today?
# DCM buck of numerics/p2_buck.py from zero, d v_C(40 T)/dL. Reference from the exact-propagator prototype
# (add-periodic-steady-state-shooting/probes/shooting_dcm.py): 1758.1254 V/H.
import numpy as np, pulsim as p, warnings
warnings.simplefilter("ignore")
Vin, C, gon, goff, T = 24.0, 47e-6, 1e3, 1e-9, 1e-5
REF = 1758.1254
def vend(L, engine, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", Vin); b.add_switch("S", "vin", "sw", gon, goff); b.add_diode("D1", "gnd", "sw", gon, goff, 1e-6)
    b.add_inductor("L", "sw", "vout", L); b.add_capacitor("C", "vout", "gnd", C); b.add_resistor("RL", "vout", "gnd", 50.0)
    fn = p.make_pwm_switch_fn(1/T, 0.3, b.switch_index_of("S"), b.graph.num_switches)
    r = p.simulate(b, t_end=40*T, engine=engine, switch_fn=fn, **kw)
    return float(np.asarray(r.v("vout"))[-1])
L0 = 100e-6
for eng, kw in [("pwl", dict(dt=T/100)), ("pwl", dict(dt=T/400)), ("trbdf2", {}), ("trbdf2", dict(rtol=1e-8, atol=1e-11)), ("dsed", {})]:
    row = []
    for rel in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        d = rel*L0
        try:
            fd = (vend(L0+d, eng, **kw) - vend(L0-d, eng, **kw))/(2*d); row.append(f"{rel:.0e}: {fd/REF-1:+.2e}")
        except Exception as e:
            row.append(f"{rel:.0e}: {type(e).__name__} ({str(e)[:60]})")
    print(f"{eng:7s} {str(kw):32s} relative error of central FD by step -> " + "  ".join(row))
