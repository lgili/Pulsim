"""D10: NativeMultiMaskPwm vs NativePwm2Switch on engine='trbdf2' (edge landing)."""
import warnings, numpy as np, pulsim as p
T, D = 1e-4, 0.37
def mk():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S1", "in", "sw", 1e3, 1e-9)
    b.add_switch("S2", "sw", "gnd", 1e3, 1e-9)
    b.add_inductor("L", "sw", "o", 1e-3)
    b.add_resistor("R", "o", "gnd", 10.0)
    return b
b = mk(); n = b.graph.num_switches
m_hi = p.SwitchStateMask(n); m_hi.set(0, True)
m_lo = p.SwitchStateMask(n); m_lo.set(1, True)
fns = {"NativePwm2Switch": p.NativePwm2Switch(T, D, n),
       "NativeMultiMaskPwm": p.NativeMultiMaskPwm(T, [D, 1.0], [m_hi, m_lo])}
edges = np.array([k*T + D*T for k in range(10)] + [k*T for k in range(1, 10)])
for name, fn in fns.items():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = p.simulate(mk(), 10*T, engine="trbdf2", switch_fn=fn)
    t = np.asarray(r.times)
    miss = np.array([np.abs(t - e).min() for e in edges])
    st = getattr(r, "_trbdf2_stats", {})
    print(f"{name:20s} steps={len(t):5d} worst |t_grid - edge|={miss.max():.3e} s  n_solves={st.get('n_solves')} "
          f"gate_events={st.get('n_gate_events')} gate_pulse_min={st.get('gate_pulse_min')} warnings={len(w)}")
    for x in w[:2]: print("   warn:", str(x.message)[:160])
