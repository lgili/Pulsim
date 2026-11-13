"""Reproducer of the documented TR-BDF2 h_min stall
(test_saturable_transformer.py::test_engines_agree_below_saturation).
Prints whether the run completes; if it aborts, the abort message and the
last 20 accepted steps of the partial trace."""
import numpy as np, pulsim as p, warnings
warnings.simplefilter("ignore")
CORE = dict(N_p=25, N_s=13, Ae=76e-6, le=72e-3, lg=0.5e-3, B_sat=0.35)


def build():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_switch("Q1", "sw", "gnd", 1e3, 1e-9)
    b.add_resistor("Rsn", "sw", "sn1", 47.0)
    b.add_capacitor("Csn", "sn1", "gnd", 2.2e-9)
    b.add_saturable_transformer("T1", "vin", "sw", "sec_neg", "sec_anode", L_leak_p=5e-6, **CORE)
    b.add_diode("D1", "sec_anode", "vout", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cout", "vout", "sec_neg", 100e-6)
    b.add_resistor("R_L", "vout", "sec_neg", 5.0)
    b.add_resistor("Rgnd", "sec_neg", "gnd", 1e-6)
    return b


T = 10e-6
b = build(); q = b.switch_index_of("Q1"); n = b.graph.num_switches


def sf(t):
    m = p.SwitchStateMask(n); m.set(q, (t % T) / T < 0.4); return m


try:
    var = p.simulate(build(), t_end=1e-3, switch_fn=sf, engine="trbdf2", rtol=1e-6, atol=1e-9)
    t = np.asarray(var.times); v = np.asarray(var.v("vout")); m = t > 0.8e-3
    dc = np.trapezoid(v[m], t[m]) / (t[m][-1] - t[m][0])
    print(f"COMPLETED: V_out(0.8-1 ms, time-weighted) = {dc:.4f} V (pwl dt=1e-8: 15.31 V); "
          f"min step {np.diff(t).min():.2e}; stats {var._trbdf2_stats}")
except p.SimulationAborted as e:
    t = np.asarray(e.partial.times)
    print("ABORTED:", str(e)[:900])
    print("last 20 steps (s):", " ".join(f"{x:.2e}" for x in np.diff(t)[-20:]))
