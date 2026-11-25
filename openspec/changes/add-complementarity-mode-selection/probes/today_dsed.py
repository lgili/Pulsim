"""TODAY's kernel on the circuits of lcp_cases.py (needs PYTHONPATH=<worktree>/python of a built tree).

Resistor-pair diodes (g_on = 1e3, g_off = 1e-9), value law, 64-round settle and chatter guard.
Prints ok / the error text per engine and V_th. Measured 2026-11-13 on feat/core-integration-wip
8b26db32 (the p1-sweep-sources worktree build); see the proposal for the table.
"""
import time
import warnings
import numpy as np
import pulsim as p

warnings.simplefilter("ignore")
GON, GOFF, T = 1e3, 1e-9, 1e-5


def run(label, f):
    t1 = time.perf_counter()
    try:
        out = f()
        print(f"{label}: ok {out} [{(time.perf_counter() - t1) * 1e3:.0f} ms]")
    except Exception as e:
        print(f"{label}: FAILS {type(e).__name__}: {str(e)[:150]} [{(time.perf_counter() - t1) * 1e3:.0f} ms]")


def buck(vth):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0); b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF, vth)
    b.add_inductor("L", "sw", "vout", 100e-6); b.add_capacitor("C", "vout", "gnd", 47e-6); b.add_resistor("R", "vout", "gnd", 50.0)
    return b, p.make_pwm_switch_fn(1 / T, 0.3, b.switch_index_of("S"), b.graph.num_switches, 0.0)


def boost(vth, R):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0); b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("S", "sw", "gnd", GON, GOFF); b.add_diode("D1", "sw", "out", GON, GOFF, vth)
    b.add_capacitor("C", "out", "gnd", 47e-6); b.add_resistor("R", "out", "gnd", R)
    return b, p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"), b.graph.num_switches, 0.0)


def bridge1(vth):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 100.0, 50.0)
    for n, a, k in (("D1", "a", "p"), ("D3", "gnd", "p"), ("D2", "n", "a"), ("D4", "n", "gnd")):
        b.add_diode(n, a, k, GON, GOFF, vth)
    b.add_capacitor("C", "p", "n", 1e-3); b.add_resistor("R", "p", "n", 100.0)
    return b


def bridge3(vth):
    b = p.CircuitBuilder()
    for ph, ang in (("a", 0.0), ("b", -2 * np.pi / 3), ("c", 2 * np.pi / 3)):
        b.add_sine_voltage_source("V" + ph, ph, "gnd", 0.0, 100.0, 50.0, ang)
        b.add_diode("Dt" + ph, ph, "p", GON, GOFF, vth); b.add_diode("Db" + ph, "n", ph, GON, GOFF, vth)
    b.add_capacitor("C", "p", "n", 1e-3); b.add_resistor("R", "p", "n", 20.0)
    return b


def last(r, node):
    return f"v({node}) end = {np.asarray(r.v(node))[-1]:.6f}"


print("pulsim from", p.__file__)
for vth in (0.0, 1e-6):
    run(f"(i)   buck at rest, gate off, dsed, V_th={vth:g}", lambda: last(p.simulate(
        buck(vth)[0], t_end=1e-6, engine="dsed", switch_fn=lambda t: p.SwitchStateMask(2)), "sw"))
for vth in (0.0, 1e-6):
    for eng in ("dsed",):
        run(f"(iii) DCM buck 4000 T, {eng}, V_th={vth:g}", lambda: last(p.simulate(
            buck(vth)[0], t_end=4000 * T, engine=eng, switch_fn=buck(vth)[1]), "vout"))
for R in (50.0, 500.0):
    for vth in (0.0, 1e-3, 1e-2):
        b, fn = boost(vth, R)
        run(f"(iv)  boost R={R:g} 400 T, dsed, V_th={vth:g}", lambda: last(p.simulate(
            b, t_end=400 * T, engine="dsed", switch_fn=fn), "out"))
for vth in (0.0, 1e-6, 1e-2):
    for eng in ("dsed", "trbdf2", "pwl"):
        kw = {"dt": 1e-6} if eng == "pwl" else {}
        run(f"(ii)  1-ph bridge 0.2 s, {eng}, V_th={vth:g}", lambda: last(p.simulate(
            bridge1(vth), t_end=0.2, engine=eng, **kw), "p"))
for vth in (0.0, 1e-6, 1e-2):
    for eng in ("dsed", "trbdf2", "pwl"):
        kw = {"dt": 1e-6} if eng == "pwl" else {}
        run(f"(v-b) 3-ph bridge 0.1 s, {eng}, V_th={vth:g}", lambda: last(p.simulate(
            bridge3(vth), t_end=0.1, engine=eng, **kw), "p"))
