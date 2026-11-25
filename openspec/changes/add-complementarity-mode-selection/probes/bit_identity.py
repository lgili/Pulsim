"""Bit-identity of pwl / trbdf2 / dsed trajectories before and after this change (invariant).

    python3 bit_identity.py save  BASE.npz          # on the parent build (adf7689a)
    python3 bit_identity.py check BASE.npz          # default mode_selection: np.array_equal
    python3 bit_identity.py check BASE.npz shadow   # mode_selection='shadow' where an engine takes it

Six circuits x three engines; every recorded time and every full state/sample array is compared
with np.array_equal (no tolerance). Engines that refuse mode_selection='shadow' by name are
reported as refused and compared at the default instead.
"""
import sys
import warnings

import numpy as np

import pulsim as p

warnings.simplefilter("ignore")
T = 1e-5
GON, GOFF = 1e3, 1e-9


def buck(vth, R, D, n):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0); b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF, vth)
    b.add_inductor("L", "sw", "vout", 100e-6); b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("R", "vout", "gnd", R)
    fn = p.make_pwm_switch_fn(1 / T, D, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    return b, fn, n * T


def boost(vth, R, n):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0); b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("S", "sw", "gnd", GON, GOFF); b.add_diode("D1", "sw", "out", GON, GOFF, vth)
    b.add_capacitor("C", "out", "gnd", 47e-6); b.add_resistor("R", "out", "gnd", R)
    fn = p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    return b, fn, n * T


def bridge1(vth, vf):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 100.0, 50.0)
    for nm, a, k in (("D1", "a", "p"), ("D3", "gnd", "p"), ("D2", "n", "a"), ("D4", "n", "gnd")):
        b.add_diode(nm, a, k, GON, GOFF, vth, vf)
    b.add_capacitor("C", "p", "n", 1e-3); b.add_resistor("R", "p", "n", 100.0)
    return b, None, 0.04


def bridge3(vth, vf):
    b = p.CircuitBuilder()
    for ph, ang in (("a", 0.0), ("b", -2 * np.pi / 3), ("c", 2 * np.pi / 3)):
        b.add_sine_voltage_source("V" + ph, ph, "gnd", 0.0, 100.0, 50.0, ang)
        b.add_diode("Dt" + ph, ph, "p", GON, GOFF, vth, vf)
        b.add_diode("Db" + ph, "n", ph, GON, GOFF, vth, vf)
    b.add_capacitor("C", "p", "n", 1e-3); b.add_resistor("R", "p", "n", 20.0)
    return b, None, 0.04


CIRCUITS = {
    "buck_ccm": lambda: buck(1e-2, 5.0, 0.5, 200),
    "buck_dcm": lambda: buck(0.0, 50.0, 0.3, 200),
    "boost": lambda: boost(0.0, 500.0, 200),
    "bridge1_vf07": lambda: bridge1(0.0, 0.7),
    "bridge3": lambda: bridge3(0.0, 0.0),
    "bridge3_vf07": lambda: bridge3(1e-2, 0.7),
}
ENGINES = {"pwl": {"dt": 1e-7}, "trbdf2": {}, "dsed": {}}


def run_all(mode):
    out, notes = {}, []
    for cname, make in CIRCUITS.items():
        for eng, kw in ENGINES.items():
            b, fn, t_end = make()
            kws = dict(kw)
            if fn is not None:
                kws["switch_fn"] = fn
            if mode is not None:
                kws["mode_selection"] = mode
            try:
                r = p.simulate(b, t_end=t_end, engine=eng, **kws)
            except p.UnsupportedError as e:
                notes.append(f"{cname}/{eng}: mode_selection={mode!r} refused by name ({e.feature})")
                kws.pop("mode_selection")
                b, fn, t_end = make()
                r = p.simulate(b, t_end=t_end, engine=eng, **kws)
            out[f"{cname}/{eng}/t"] = np.asarray(r.times, dtype=float)
            out[f"{cname}/{eng}/x"] = np.asarray(r.states, dtype=float)
    return out, notes


def main():
    cmd, path = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else None
    print("pulsim from", p.__file__)
    data, notes = run_all(mode)
    if cmd == "save":
        np.savez(path, **data)
        print(f"saved {len(data)} arrays")
        return
    base = np.load(path)
    bad = [k for k in base.files if k not in data or not np.array_equal(base[k], data[k])]
    for n in notes:
        print(n)
    for k in sorted(base.files):
        print(f"{k:28s} shape={base[k].shape} {'IDENTICAL' if k not in bad else 'DIFFERENT'}")
    print(f"bit-identity ({'default' if mode is None else mode}): "
          f"{len(base.files) - len(bad)}/{len(base.files)} arrays identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
