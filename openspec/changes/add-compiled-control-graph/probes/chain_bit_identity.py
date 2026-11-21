"""Bit-identity of block chains across the control-graph rewrite.

    python chain_bit_identity.py record  [--dir D]   # on the tree before
    python chain_bit_identity.py compare [--dir D]   # on the tree after

Three chains already in topological order, run on `pwl`:

A  buck PI: Subtract -> PIController -> PwmGenerator, gate through
   make_pwm_switch_fn (the circuit closes the loop);
B  every transform and filter kind of the C++ chain on a three-phase
   source (Clarke, PLL, Park, LPF, inverse Park / Clarke, SVM, integrator,
   PID, Sum, Subtract, Gain);
C  a Gain reading a node and a Limiter: before the rewrite both made
   the C++ build fail and the chain ran on the Python evaluator.

The simulation result (times, states) and the recorded channels are
compared byte for byte.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pulsim as p

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DIR = ROOT / ".scratch" / "chain_golden"


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def chain_a():
    b = _buck()
    ch = p.MixedDomainBlockChain()
    ch.add("err", p.Subtract(), inputs=dict(a=12.0, b="out"), output="e")
    ch.add("pi", p.PIController(Kp=0.02, Ki=20.0, output_min=0.05,
                                output_max=0.95),
           inputs=dict(setpoint=12.0, measured="out", dt="dt"),
           output="duty")
    ch.add("pwm", p.PwmGenerator(frequency=20e3),
           inputs=dict(duty="channel:duty", t="time"), output="gate")
    return b, ch, ["e", "duty", "gate"], 1


def chain_b():
    b = p.CircuitBuilder()
    for k, ph in enumerate(("a", "b", "c")):
        b.add_sine_voltage_source(f"V{ph}", ph, "gnd", 10.0, 50.0,
                                  -2.0 * np.pi * k / 3.0)
        b.add_resistor(f"R{ph}", ph, "gnd", 10.0)
    ch = p.MixedDomainBlockChain()
    ch.add("clarke", p.ClarkeTransform(), inputs=dict(a="a", b="b", c="c"),
           output=("al", "be", "ze"))
    ch.add("pll", p.PLL(f_nominal=50.0, Kp=50.0, Ki=500.0),
           inputs=dict(v_alpha="channel:al", v_beta="channel:be", dt="dt"),
           output=("th", "om", "fr"))
    ch.add("park", p.ParkTransform(),
           inputs=dict(alpha="channel:al", beta="channel:be",
                       theta="channel:th"), output=("d", "q"))
    ch.add("lpf", p.FirstOrderLowPass(tau=1e-3),
           inputs=dict(input_value="channel:d", dt="dt"), output="df")
    ch.add("ipark", p.InverseParkTransform(),
           inputs=dict(d="channel:d", q="channel:q", theta="channel:th"),
           output=("al2", "be2"))
    ch.add("iclarke", p.InverseClarkeTransform(),
           inputs=dict(alpha="channel:al2", beta="channel:be2"),
           output=("a2", "b2", "c2"))
    ch.add("svm", p.SpaceVectorModulator(v_dc=40.0),
           inputs=dict(v_alpha="channel:al2", v_beta="channel:be2"),
           output=("da", "db", "dc"))
    ch.add("integ", p.Integrator(gain=3.0, output_min=-1.0, output_max=1.0),
           inputs=dict(x="channel:q", dt="dt"), output="qi")
    ch.add("pid", p.PIDController(Kp=0.1, Ki=10.0, Kd=1e-4,
                                  output_min=-5.0, output_max=5.0),  # tau_d 0
           inputs=dict(setpoint=1.0, measured="channel:df", dt="dt"),
           output="u")
    ch.add("sum", p.Sum(), inputs=dict(a="channel:u", b="channel:qi"),
           output="s")
    ch.add("sub", p.Subtract(),
           inputs=dict(a="channel:s", b="channel:q"), output="m")
    ch.add("gain", p.Gain(k=0.25), inputs=dict(x="channel:m"), output="g")
    rec = ["al", "be", "th", "om", "fr", "d", "q", "df", "a2", "da", "dc",
           "qi", "u", "s", "m", "g"]
    return b, ch, rec, None


def chain_c():
    b = _buck()
    ch = p.MixedDomainBlockChain()
    ch.add("g", p.Gain(k=0.04), inputs=dict(x="out"), output="raw")
    ch.add("lim", p.Limiter(output_min=0.1, output_max=0.6),
           inputs=dict(x="channel:raw"), output="duty")
    ch.add("pwm", p.PwmGenerator(frequency=20e3),
           inputs=dict(duty="channel:duty", t="time"), output="gate")
    return b, ch, [], 1


CASES = {"A": chain_a, "B": chain_b, "C": chain_c}
DT, T_END = 1e-6, 2e-3


def run(case):
    b, ch, rec, n_sw = CASES[case]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs = ch.make_step_observer(b, dt=DT)
    cxx = getattr(ch, "_cxx_chain", None)
    if cxx is not None:
        for name in rec:
            cxx.record_channel(name)
    kw = {}
    if n_sw:
        kw["switch_fn"] = ch.make_pwm_switch_fn(
            "gate", num_switches=b.graph.num_switches)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, T_END, dt=DT, engine="pwl", step_observer=obs, **kw)
    out = {"times": np.asarray(r.times),
           "states": np.asarray(r.states)}
    if cxx is not None:
        for name in rec:
            out["ch_" + name] = np.asarray(cxx.get_channel_history(name))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("record", "compare"))
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    a = ap.parse_args(argv)
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    bad = 0
    for case in CASES:
        out = run(case)
        f = d / f"{case}.npz"
        if a.mode == "record":
            np.savez(f, **out)
            print(f"{case}: recorded {len(out)} arrays")
            continue
        ref = np.load(f)
        diffs = [k for k in ref.files
                 if k not in out or ref[k].tobytes() != out[k].tobytes()]
        print(f"{case}: {'IDENTICAL' if not diffs else 'DIFFERS ' + str(diffs)}"
              f" ({len(ref.files)} arrays)")
        bad += bool(diffs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
