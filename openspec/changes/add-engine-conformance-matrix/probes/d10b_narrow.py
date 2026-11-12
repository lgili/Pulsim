"""D10b: a narrow window in a NativeMultiMaskPwm on trbdf2 — are its edges all seen?"""
import warnings, numpy as np, pulsim as p
T = 1e-3
def mk():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S1", "in", "sw", 1e3, 1e-9)
    b.add_switch("S2", "sw", "gnd", 1e3, 1e-9)
    b.add_inductor("L", "sw", "o", 1.0)       # tau = 0.1 s: steps want to be large
    b.add_resistor("R", "o", "gnd", 10.0)
    return b
b = mk(); n = b.graph.num_switches
hi = p.SwitchStateMask(n); hi.set(0, True)
lo = p.SwitchStateMask(n); lo.set(1, True)
for width in (1e-2, 1e-3, 1e-4):
    fn = p.NativeMultiMaskPwm(T, [0.5, 0.5 + width, 1.0], [hi, lo, hi])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = p.simulate(mk(), 20*T, engine="trbdf2", switch_fn=fn)
    st = getattr(r, "_trbdf2_stats", {})
    print(f"window={width*T:.1e} s  gate_events={st.get('n_gate_events')} (analytic 40)  steps={len(r.times)} warnings={len(w)}")
    for x in w[:1]: print("   warn:", str(x.message)[:200])
