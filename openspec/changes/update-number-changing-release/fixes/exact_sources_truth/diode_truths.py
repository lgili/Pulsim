"""mpmath truths for python/tests/test_exact_integrals_sources.py (diode
circuits on time-varying sources). Never imports pulsim.

Needs benchmarks/tools/exact_truth.py at fa4b5c71 (origin/feat/core-soft-
switching-validation: the conditioning fix) next to this file or on the path:

    git show fa4b5c71:benchmarks/tools/exact_truth.py > exact_truth.py
    python3 diode_truths.py > diode_truths.json

Per circuit: the window [A, B]'s mean / rms of each column (V(..), I(L..))
and the exact energy of every element over the window (resistors and diodes
absorbed, sources delivered: 'src:<name>'), the stored-energy change, and
the energies over the whole run for the ledger.
"""
import json
import sys

import mpmath as mp

sys.path.insert(0, ".")
import exact_truth as et  # noqa: E402

DPS = 40
G = dict(g_on=1e3, g_off=1e-9)


def sine(n, a, b, amp, f, ph=0.0, off=0.0):
    return dict(type="voltage_source", name=n, nodes=[a, b],
                waveform=dict(type="sine", amplitude=amp, frequency=f, phase=ph, offset=off))


def R(n, a, b, v):
    return dict(type="resistor", name=n, nodes=[a, b], value=v)


def Lc(n, a, b, v):
    return dict(type="inductor", name=n, nodes=[a, b], value=v, ic=0.0)


def Cc(n, a, b, v):
    return dict(type="capacitor", name=n, nodes=[a, b], value=v, ic=0.0)


def D(n, a, k):
    return dict(type="diode", name=n, nodes=[a, k], **G)


CIRCUITS = {
    # half-wave: sine -> D -> L -> C || R (an RLC across commutations)
    "halfwave_rlc": dict(comps=[sine("V", "s", 0, 20.0, 1000.0, 0.4), D("D1", "s", "a"),
                                Lc("L", "a", "o", 2e-3), Cc("C", "o", 0, 4.7e-6), R("RL", "o", 0, 50.0)],
                         t_end=6e-3, A=0.7123e-3, B=5.3321e-3, cols=["I(L)", "V(o)"]),
    # single-phase bridge with line inductance, floating bus (1 MOhm tie)
    "bridge_1ph": dict(comps=[sine("Vac", "s", 0, 325.0, 50.0), R("Rs", "s", "a", 0.5),
                              Lc("Ls", "a", "ac1", 1e-3), D("D1", "ac1", "dp"), D("D2", 0, "dp"),
                              D("D3", "dn", "ac1"), D("D4", "dn", 0), Cc("C", "dp", "dn", 470e-6),
                              R("RL", "dp", "dn", 100.0), R("Rtie", "dn", 0, 1e6)],
                       t_end=30e-3, A=7.31e-3, B=27.77e-3, cols=["I(Ls)", "V(dp,dn)"]),
    # three-phase bridge, line inductance per phase, star point grounded
    "bridge_3ph": dict(comps=[sine("Va", "sa", 0, 325.0, 50.0, 0.0),
                              sine("Vb", "sb", 0, 325.0, 50.0, -2.0943951023931953),
                              sine("Vc", "sc", 0, 325.0, 50.0, 2.0943951023931953),
                              Lc("La", "sa", "a", 0.5e-3), Lc("Lb", "sb", "b", 0.5e-3),
                              Lc("Lc", "sc", "c", 0.5e-3),
                              D("D1", "a", "dp"), D("D3", "b", "dp"), D("D5", "c", "dp"),
                              D("D4", "dn", "a"), D("D6", "dn", "b"), D("D2", "dn", "c"),
                              Cc("C", "dp", "dn", 220e-6), R("RL", "dp", "dn", 40.0),
                              R("Rtie", "dn", 0, 1e6)],
                       t_end=12e-3, A=3.137e-3, B=11.21e-3, cols=["I(La)", "I(Lb)", "V(dp,dn)"]),
    # a pulse with finite edges (ramps: the engine's level + ramp) into an RLC (no diode:
    # with a diode at the ramp's start the engine refuses, DiodeModeInconsistentError,
    # a tie the scheduler's crossing scan misses; recorded in the notes)
    "pulse_ramps": dict(comps=[dict(type="voltage_source", name="Vp", nodes=["in", 0],
                                    waveform=dict(type="pulse", v_initial=0.0, v_pulse=10.0,
                                                  t_delay=5e-6, t_rise=20e-6, t_width=50e-6,
                                                  t_fall=30e-6, period=200e-6)),
                               R("R", "in", "a", 100.0), Lc("L", "a", "o", 1e-3),
                               Cc("C", "o", 0, 1e-7), R("RL", "o", 0, 1e3)],
                        t_end=600e-6, A=37.3e-6, B=512.9e-6, cols=["I(L)", "V(o)"]),
}
if len(sys.argv) > 1:
    CIRCUITS = {k: CIRCUITS[k] for k in sys.argv[1:]}


def f(x):
    return float(x)


out = {}
for name, c in CIRCUITS.items():
    ck = et.Circuit(et._net(name, c["comps"], c["t_end"], 1e-6, c["cols"]))
    sol = et.Solver(ck, DPS).run()
    A, B = mp.mpf(c["A"]), mp.mpf(c["B"])
    w = sol.window(c["cols"], A, B)
    ew = sol.energy(A, B)
    ea = sol.energy()
    out[name] = dict(
        events=[f(e["t"]) for e in sol.events[1:]],
        window={k: dict(mean=f(v["mean"]), rms=f(v["rms"])) for k, v in w.items()},
        energy_window={k: f(v) for k, v in ew["per"].items()},
        dE_window=f(ew["dE"]), balance_window=f(ew["balance_rel"]),
        energy_run={k: f(v) for k, v in ea["per"].items()},
        dE_run=f(ea["dE"]), balance_run=f(ea["balance_rel"]))
    print(name, "events", len(out[name]["events"]), "balance", out[name]["balance_window"],
          file=sys.stderr)
json.dump(out, sys.stdout, indent=1)
