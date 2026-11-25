"""Interleaved A/B wall time for update-fixed-step-loop-bookkeeping.

    python perf_ab.py A_DIR B_DIR [reps]

A_DIR and B_DIR each hold a `pulsim` package (e.g. a copy of the
package built before the change, and python/). Each measurement runs
in a fresh subprocess, alternating A and B, best of `reps`. Report
only: wall time is never asserted.
"""
import json
import os
import subprocess
import sys

CASES = r'''
import json, sys, time, numpy as np, pulsim as p, warnings
warnings.simplefilter("ignore")
def best(fn, reps=3):
    w = 1e99
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); w = min(w, time.perf_counter() - t0)
    return w, out
def buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    return b, p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
def vsi():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", 600.0)
    v = p.add_three_phase_vsi(b, "INV", vdc_pos="dcp", vdc_neg="gnd", out_a="a", out_b="b", out_c="c")
    for ph in "abc":
        b.add_resistor(f"R{ph}", ph, f"m{ph}", 10.0)
        b.add_inductor(f"L{ph}", f"m{ph}", "n", 5e-3)
    legs = p.ThreePhaseLegIndices(*v.switch_indices)
    return b, p.make_three_phase_spwm_fn(10e3, 50.0, 0.8, legs, b.graph.num_switches, 1e-6)
def rlc(N):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "n0", "gnd", 0.0, 10.0, 1e3)
    for k in range(N):
        b.add_inductor(f"L{k}", f"n{k}", f"m{k}", 1e-5)
        b.add_resistor(f"R{k}", f"m{k}", f"n{k+1}", 0.1)
        b.add_capacitor(f"C{k}", f"n{k+1}", "gnd", 1e-7)
    b.add_resistor("Rload", f"n{N}", "gnd", 50.0)
    return b
out = {}
S = 200000; dt = 1e-7
b, fn = buck()
for tag, kw in (("buck", {}), ("buck_rule", {"discontinuity_restart": True}),
                ("buck_split", {"discontinuity_restart": True, "enable_substep_state_correction": True})):
    w, r = best(lambda: p.simulate(b, t_end=S * dt, dt=dt, switch_fn=fn, engine="pwl", **kw))
    out[tag] = w / S * 1e9
b, fn = vsi()
w, _ = best(lambda: p.simulate(b, t_end=S * dt, dt=dt, switch_fn=fn, engine="pwl"))
out["vsi"] = w / S * 1e9
bb = rlc(1)
w, _ = best(lambda: p.simulate(bb, t_end=S * dt, dt=dt, engine="pwl"))
out["rlc5_default"] = w / S * 1e9
c0 = p.make_combined_switch_fn(0, [])
w, _ = best(lambda: p.simulate(bb, t_end=S * dt, dt=dt, engine="pwl", switch_fn=c0))
out["rlc5_const"] = w / S * 1e9
bb = rlc(750)
w, _ = best(lambda: p.simulate(bb, t_end=2000 * dt, dt=dt, engine="pwl"), 2)
out["rlc1502"] = w / 2000 * 1e9
b, fn = buck()
r = p.simulate(b, t_end=1e5 * dt, dt=dt, switch_fn=fn, engine="pwl")
t0 = time.perf_counter(); np.asarray(r.times); out["times_ms"] = (time.perf_counter() - t0) * 1e3
print(json.dumps(out))
'''


def run(pkg):
    env = dict(os.environ, PYTHONPATH=os.path.abspath(pkg))
    r = subprocess.run(["nice", "-n", "5", sys.executable, "-c", CASES], env=env,
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    best = {"A": {}, "B": {}}
    for _ in range(reps):
        for side, d in (("A", a_dir), ("B", b_dir)):
            for k, v in run(d).items():
                best[side][k] = min(best[side].get(k, 1e99), v)
    for k in best["A"]:
        a, b = best["A"][k], best["B"][k]
        unit = "ms" if k.endswith("_ms") else "ns/step"
        print(f"{k:14s} A {a:9.2f}  B {b:9.2f} {unit}  B/A {b / a:5.2f}")
