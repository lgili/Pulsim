"""D9: buck whose freewheel path is a gate-off MOSFET body diode.
today  = add_mosfet(V_F=0.7)                        (V_F is V_th, no drop)
future = add_switch + add_diode(V_th=0, V_f=0.7)    (what the proposal builds)"""
import warnings, numpy as np, pulsim as p
warnings.simplefilter("ignore")
T, D = 1e-5, 0.5
def mk(future):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("QH", "in", "sw", 1e3, 1e-9)
    if future:
        b.add_switch("QL", "sw", "gnd", 1e3, 1e-9)
        b.add_diode("QL_body", "gnd", "sw", 1e3, 1e-9, 0.0, 0.7)
    else:
        b.add_mosfet("QL", "sw", "gnd", 1e-3, 1e9, True, 0.7)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 100e-6)
    b.add_resistor("R", "out", "gnd", 1.2)
    return b
for eng, kw in (("pwl", dict(dt=T/200)), ("trbdf2", {}), ("dsed", {})):
    for future in (False, True):
        b = mk(future); n = b.graph.num_switches; k = b.switch_index_of("QH")
        fn = p.pwm_switch_fn(n, k, 1/T, D) if hasattr(p, "pwm_switch_fn") else None
        if fn is None:
            def fn(t, n=n, k=k):
                m = p.SwitchStateMask(n); m.set(k, (t/T) % 1.0 < D); return m
        try:
            r = p.simulate(b, 400*T, engine=eng, switch_fn=fn, **kw)
            t = np.asarray(r.times); v = np.asarray(r.v("out"))
            vo = p.time_average(t, v, t_from=300*T)
            rows = {e["name"]: e.get("P_avg", 0.0) for e in p.device_loss_summary(b, r)}
            print(f"{eng:6s} {'future' if future else 'today ':6s} V_out={vo:.4f} V  P(QL_body)={rows.get('QL_body', float('nan')):.4f} W")
        except Exception as e:
            print(f"{eng:6s} {'future' if future else 'today ':6s} {type(e).__name__}: {str(e)[:150]}")
