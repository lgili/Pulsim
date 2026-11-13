"""Task 3.4: capacitor-current roundoff of the refresh. Buck with a 10 mF
output capacitor (Vin 800 V, D = 0.5), 40 periods from zero (v_out
reaches ~150 V), rtol 1e-6 and 1e-8: accepted steps, the
smallest first step after a gate edge, and v_out against pwl at T/4000."""
import numpy as np, pulsim as p, warnings
warnings.simplefilter("ignore")
T = 1e-5


def run(engine, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 800.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 10e-3)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    n = b.graph.num_switches; k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n); m.set(k, bool(t / T - np.floor(t / T + 1e-9) < 0.5 - 1e-9)); return m
    return p.simulate(b, t_end=40 * T, engine=engine, switch_fn=fn, **kw)


ref = run("pwl", dt=T / 4000)
vr = np.asarray(ref.v("vout"))[-1]
print(f"v_out(pwl) = {vr:.4f} V")
for rtol, atol in ((1e-6, 1e-9), (1e-8, 1e-11)):
    r = run("trbdf2", rtol=rtol, atol=atol); st = r._trbdf2_stats
    print(f"rtol {rtol:.0e}: accepted {st['n_accept']}, rejects {st['n_reject']}, refreshes {st.get('n_derivative_refreshes')}, "
          f"h_post_event_min {st.get('h_post_event_min', 0):.2e}, v_out err vs pwl T/4000 {np.asarray(r.v('vout'))[-1] - vr:+.2e} V")
