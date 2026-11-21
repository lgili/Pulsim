"""ns/step with resolution_diagnostics on vs off, interleaved in one
process, best of `reps` (report only; wall time is never asserted).

    python perf_onoff.py [reps]
"""
import sys
import time
import warnings

import pulsim as p

warnings.simplefilter("ignore")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
S, DT = 200000, 1e-7


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


def rlc(n):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "n0", "gnd", 0.0, 10.0, 1e3)
    for k in range(n):
        b.add_inductor(f"L{k}", f"n{k}", f"m{k}", 1e-5)
        b.add_resistor(f"R{k}", f"m{k}", f"n{k+1}", 0.1)
        b.add_capacitor(f"C{k}", f"n{k+1}", "gnd", 1e-7)
    b.add_resistor("Rload", f"n{n}", "gnd", 50.0)
    return b, None


CASES = [("buck", buck, {}, S), ("buck_rule", buck, {"discontinuity_restart": True}, S),
         ("vsi", vsi, {}, S), ("rlc1502", lambda: rlc(750), {}, 2000)]
for name, mk, kw, steps in CASES:
    b, fn = mk()
    best = {False: 1e99, True: 1e99}
    for _ in range(REPS):
        for on in (False, True):
            t0 = time.perf_counter()
            p.simulate(b, t_end=steps * DT, dt=DT, switch_fn=fn, engine="pwl",
                       resolution_diagnostics=on, **kw)
            best[on] = min(best[on], time.perf_counter() - t0)
    off, on = (best[x] / steps * 1e9 for x in (False, True))
    print(f"{name:10s} off {off:10.2f}  on {on:10.2f} ns/step  on/off {on / off:.3f}")
