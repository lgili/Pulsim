"""Task 4: mode_selection='shadow' over a broad set of diode circuits on dsed -> the disagreement table.

    PYTHONPATH=<worktree>/python python3 shadow_survey.py > shadow_survey.out

Every circuit runs with today's settle deciding (numbers unchanged, checked bit-identical against a
default run here too) while the LCP is posed on the same held state at every event. For each
disagreement the kernel's judge propagates BOTH candidate modes exactly from the held state
(100 h_e and 1e4 h_e) and checks them: every ON diode keeps i >= 0, every OFF diode v <= V_on.
"right" = consistent there, a reference independent of both deciders.
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
import pulsim as p  # noqa: E402
from tests.oracles import lcp_pulsim_bridge as B  # noqa: E402

T = B.T
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))


def pwm_fn(b, names, D=0.3, f=1e5):
    if len(names) == 1:
        return p.make_pwm_switch_fn(f, D, b.switch_index_of(names[0]), b.graph.num_switches, 0.0)
    raise ValueError


def fam_case(name, V_th, t_end):
    fam = B.FAMILIES[name]

    def make():
        b = fam.builder(V_th)
        sws = [a[0] for k, *a in fam.parts if k == "S"]
        D = 0.5 if name.startswith("boost") else 0.3
        return b, (pwm_fn(b, sws, D) if sws else None), t_end
    return make


def yaml_case(fname, t_end, gate):
    def make():
        lc = p.load_yaml_file(os.path.join(ROOT, "examples", fname))
        b = lc.builder
        n = b.graph.num_switches
        fn = None
        if gate == "pwm" and n:
            fn = p.make_pwm_switch_fn(1e5, 0.4, 0, n, 0.0)
        elif gate == "off":
            off = p.SwitchStateMask(n)
            fn = lambda t: off  # noqa: E731
        return lc, fn, t_end
    return make


def series_diode_lc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_diode("D", "in", "x", 1e3, 1e-9, 0.0)
    b.add_inductor("L", "x", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    b.add_resistor("R", "out", "gnd", 1e4)
    return b, None, 2e-3


def buck_vf(V_f, V_th=0.0):
    def make():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", 24.0)
        b.add_switch("S", "vin", "sw", 1e3, 1e-9)
        b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th, V_f)
        b.add_inductor("L", "sw", "vout", 100e-6)
        b.add_capacitor("C", "vout", "gnd", 47e-6)
        b.add_resistor("R", "vout", "gnd", 50.0)
        return b, pwm_fn(b, ["S"]), 400 * T
    return make


CASES = [
    ("buck at rest, gate off (i)", lambda: (lambda b: (b, (lambda t, n=b.graph.num_switches: p.SwitchStateMask(n)),
                                                       1e-6))(B.FAMILIES["buck_rest"].builder(0.0))),
    ("DCM buck V_th=0, 400 T (iii)", fam_case("dcm_buck", 0.0, 400 * T)),
    ("DCM buck V_th=1e-2, 400 T", fam_case("dcm_buck", 1e-2, 400 * T)),
    ("DCM buck V_f=0.7, 400 T", buck_vf(0.7)),
    ("boost R=50 V_th=0, 400 T (iv)", fam_case("boost_50", 0.0, 400 * T)),
    ("boost R=50 V_th=1e-2, 400 T", fam_case("boost_50", 1e-2, 400 * T)),
    ("boost R=500 V_th=0, 400 T (iv)", fam_case("boost_500", 0.0, 400 * T)),
    ("boost R=500 V_th=1e-3, 400 T", fam_case("boost_500", 1e-3, 400 * T)),
    ("1-ph bridge V_f=0.7, 0.2 s (ii)", fam_case("bridge1_vf07", 0.0, 0.2)),
    ("1-ph bridge V_f=0, 0.2 s (ii)", fam_case("bridge1_vf0", 0.0, 0.2)),
    ("1-ph bridge V_f=0, V_th=1e-2", fam_case("bridge1_vf0", 1e-2, 0.2)),
    ("3-ph bridge C filter V_f=0.7 (v-b)", fam_case("bridge3_cfilter", 0.0, 0.1)),
    ("3-ph bridge C filter, V_th=1e-2", fam_case("bridge3_cfilter", 1e-2, 0.1)),
    ("series diode L-C from rest", series_diode_lc),
    ("examples/buck.yaml (PWM 100k, 0.4)", yaml_case("buck.yaml", 2e-3, "pwm")),
    ("examples/buck_converter.yaml (PWM)", yaml_case("buck_converter.yaml", 2e-3, "pwm")),
    ("examples/stress_buck_switching.yaml", yaml_case("stress_buck_switching.yaml", 2e-3, "pwm")),
    ("examples/three_phase_diode_rectifier", yaml_case("three_phase_diode_rectifier.yaml", 0.06, None)),
    ("examples/three_phase_inverter (off)", yaml_case("three_phase_inverter.yaml", 0.02, "off")),
]


def verdict(d):
    s, l_ = d["settle_consistent"], d["lcp_consistent"]
    if s is None or l_ is None:
        return "unknown"
    return {(True, False): "settle", (False, True): "LCP", (True, True): "both",
            (False, False): "neither"}[(s, l_)]


def run(make, mode):
    b, fn, t_end = make()
    kw = {} if fn is None else {"switch_fn": fn}
    if mode:
        kw["mode_selection"] = mode
    t0 = time.perf_counter()
    r = p.simulate(b, t_end=t_end, engine="dsed", **kw)
    return r, time.perf_counter() - t0


def main():
    print("pulsim from", p.__file__)
    hdr = (f"{'circuit':38s} {'events':>6s} {'disagr':>6s} {'LCP right':>9s} {'settle right':>12s} "
           f"{'both':>4s} {'neither':>7s} {'unk':>3s} {'maxpiv':>6s} {'ties':>4s} {'wid':>4s} "
           f"{'rounds/ev':>9s} {'bit-id':>6s} {'t_def':>6s} {'t_shd':>6s}")
    print(hdr)
    for label, make in CASES:
        try:
            r0, t_def = run(make, None)
            r1, t_shd = run(make, "shadow")
        except Exception as e:
            print(f"{label:38s} FAILS {type(e).__name__}: {str(e)[:120]}")
            continue
        ident = (np.array_equal(np.asarray(r0.times), np.asarray(r1.times))
                 and np.array_equal(np.asarray(r0.states), np.asarray(r1.states)))
        ms = r1.diagnostics["mode_shadow"]
        dis = ms["disagreements"]
        v = [verdict(d) for d in dis]
        rounds = ms["settle_rounds_total"] / max(1, ms["events"])
        print(f"{label:38s} {ms['events']:6d} {len(dis):6d} {v.count('LCP'):9d} {v.count('settle'):12d} "
              f"{v.count('both'):4d} {v.count('neither'):7d} {v.count('unknown'):3d} "
              f"{ms['max_pivots_per_event']:6d} {ms['ties']:4d} {ms['widenings']:4d} {rounds:9.2f} "
              f"{'yes' if ident else 'NO':>6s} {t_def:6.2f} {t_shd:6.2f}")
        if ms["errors"]:
            print(f"    errors: {ms['errors']} last: {ms['last_error'][:150]}")
        for d in dis[:2]:
            print(f"    t={d['t']:.9e} settle={d['settle_mode']} lcp={d['lcp_mode']} prev={d['previous_mode']} "
                  f"rounds={d['settle_rounds']} piv={d['pivots']} status={d['status']} "
                  f"viol settle/lcp={d['settle_violation']:.2e}/{d['lcp_violation']:.2e} -> {verdict(d)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
