"""Event splitting x the devices that arrived on the integration branch.

Each case: a small circuit with an OFF-GRID edge (D = 0.4567 at 100 kHz,
or a source edge at 1.37 us), run with the restart rule + split at
T/50 .. T/800; the order is read from successive differences of one
state at t_end (no reference needed). A typed pulsim.UnsupportedError is
reported as such.
Run: PYTHONPATH=python python3 split_devices.py
"""
import math
import warnings

import numpy as np

import pulsim as p

warnings.simplefilter("ignore")
F = 100e3
T = 1 / F
D = 0.4567
SPLIT = dict(discontinuity_restart=True, enable_substep_state_correction=True)
G = (1e3, 1e-9)


def _sw(b, name="S"):
    return p.make_pwm_switch_fn(F, D, b.switch_index_of(name), b.graph.num_switches, 0.0)


def power_mosfet():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdd", "vdd", "gnd", 12.0)
    b.add_resistor("Rl", "vdd", "l", 10.0)
    b.add_inductor("Ll", "l", "d", 10e-6)
    b.add_pwm_voltage_source("Vg", "g", "gnd", 10.0, 0.0, F, D, 0.0)
    b.add_power_mosfet("M", "d", "gnd", "g", K=10.0, V_T=2.0, C_gs=1e-9,
                       C_gd=0.2e-9, C_ds=0.5e-9, R_g=200.0)
    return b, None, ("i", "Ll"), {"enable_nonlinear_refresh": True}


def mlcc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", *G)
    b.add_resistor("R", "a", "c", 2.0)
    law = p.ChargeLaw.table(p.ChargeTable.from_capacitance(
        [0.0, 5.0, 10.0, 20.0], [1e-6, 0.8e-6, 0.5e-6, 0.3e-6]))
    b.add_nonlinear_capacitor("C", "c", "gnd", law)
    b.add_resistor("Rd", "c", "gnd", 20.0)
    return b, _sw(b), ("v", "c"), {}


def dc_machine():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 24.0)
    b.add_switch("S", "in", "a", *G)
    b.add_diode("Df", "gnd", "a", *G)
    b.add_dc_machine("M", "a", "gnd", "w", R_a=1.0, L_a=50e-6, K=0.05,
                     J=1e-5, B=1e-5)
    return b, _sw(b), ("v", "w"), {}


def controlled_sources():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_switch("S", "in", "a", *G)
    b.add_resistor("R", "a", "c", 100.0)
    b.add_capacitor("C", "c", "gnd", 100e-9)
    b.add_vcvs("E", "c", "gnd", "o", "gnd", 2.0)
    b.add_resistor("Ro", "o", "m", 10.0)
    b.add_inductor("Lo", "m", "gnd", 10e-6)
    b.add_vccs("Gm", "c", "gnd", "m", "gnd", 0.01)
    return b, _sw(b), ("i", "Lo"), {}


def _isrc(add):
    b = p.CircuitBuilder()
    add(b)
    b.add_resistor("R", "a", "gnd", 100.0)
    b.add_capacitor("C", "a", "gnd", 10e-9)
    b.add_switch("S", "a", "r", *G)
    b.add_resistor("R2", "r", "gnd", 50.0)
    return b, _sw(b), ("v", "a"), {}


def pulse_current():
    return _isrc(lambda b: b.add_pulse_current_source(
        "I", "gnd", "a", 0.0, 0.1, 1.37e-6, 2.0e-6, T, 0.0, 0.0))


def sine_current():
    return _isrc(lambda b: b.add_sine_current_source("I", "gnd", "a", 0.0, 0.1, 37e3))


def table_current():
    return _isrc(lambda b: b.add_pwl_current_source(
        "I", "gnd", "a", [0.0, 3e-6, 7e-6, 10e-6], [0.0, 0.1, -0.05, 0.0],
        periodic=True))


def diode_with_drop():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 24.0)
    b.add_switch("S", "in", "sw", *G)
    b.add_diode("D", "gnd", "sw", *G, V_th=0.8, V_f=0.7)
    b.add_inductor("L", "sw", "o", 50e-6)
    b.add_capacitor("C", "o", "gnd", 20e-6)
    b.add_resistor("RL", "o", "gnd", 2.0)
    return b, _sw(b), ("i", "L"), {}


CASES = [power_mosfet, mlcc, dc_machine, controlled_sources, pulse_current,
         sine_current, table_current, diode_with_drop]


def run(case, N=(50, 100, 200, 400, 800), nper=8.3):
    x = []
    for n in N:
        b, fn, (kind, name), kw = case()
        kw = {**kw, **SPLIT}
        if fn is not None:
            kw["switch_fn"] = fn
        r = p.simulate(b, t_end=nper * T, dt=T / n, engine="pwl", **kw)
        x.append(float(np.asarray(getattr(r, kind)(name))[-1]))
        splits = r.n_split_steps
    d = [a - c for a, c in zip(x, x[1:])]
    return d, [math.log2(abs(d[i] / d[i + 1])) if d[i + 1] else math.inf
               for i in range(len(d) - 1)], splits


if __name__ == "__main__":
    for case in CASES:
        try:
            d, o, s = run(case)
            print(f"{case.__name__:20s} diffs " + " ".join(f"{v:+.2e}" for v in d)
                  + "  order " + " ".join(f"{v:.2f}" for v in o) + f"  splits(T/800) {s}")
        except p.UnsupportedError as e:
            print(f"{case.__name__:20s} UnsupportedError({e.engine}, {e.feature}, {e.device}, {e.kind})")
        except Exception as e:  # noqa: BLE001 - a report
            print(f"{case.__name__:20s} {type(e).__name__}: {str(e)[:160]}")
