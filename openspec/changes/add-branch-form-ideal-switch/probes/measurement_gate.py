"""Group 7 measurement gate: branch form against conductance form (tasks 7.1-7.3).

    PYTHONPATH=<worktree>/python:<worktree>/python/tests python3 measurement_gate.py > measurement_gate.out

The circuits are test_branch_form_switch.py's: the three-phase inverter (6 switches + 6 antiparallel
diodes = 12 switching elements, 400 V, 10 kHz carrier, dt = 1 us, 4 ms) and the buck (switch +
diode, T = 10 us, dt = T/100, 2 ms), each built in both forms, on every engine in ENGINES (7.1 pwl, 7.2 trbdf2;
an engine that refuses a form by name prints the refusal instead of a ratio).
ns/step is the median of REPS interleaved wall-clock runs divided by the stored steps; counters are from one run. 7.3: the flip
is allowed only if the inverter's branch-form ns/step is at most 1.5x the conductance form's.
"""
import statistics
import time
import warnings

import numpy as np

import pulsim as p

import test_branch_form_switch as T

REPS = 7
ENGINES = ("pwl", "trbdf2")


def run(make, branch, engine):
    b, kw, _ = make(branch)
    kw = dict(kw, engine=engine)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        r = p.simulate(b, **kw)
        wall = time.perf_counter() - t0
    return r, wall


def counters(r):
    """Unknowns, steps, event-iteration rounds and the engine's own counters (simulate builds its
    cache in C++: factorizations are not exposed; `segment_searches` counts the mask lookups that
    miss the per-step memo, i.e. at most one factor or cache hit each)."""
    out = {"unknowns": int(np.asarray(r.states).shape[1]), "steps": len(r.times) - 1,
           "event_rounds": int(np.sum(np.asarray(r.event_iteration_count)))}
    out.update({k: int(v) for k, v in dict(r.counters).items() if v})
    return out


def main():
    print("pulsim from", p.__file__)
    ratios = {}
    for name, make in (("inverter", T._inverter), ("buck", T._buck)):
        for engine in ENGINES:
            walls = {False: [], True: []}
            res = {}
            try:
                for _ in range(REPS):
                    for branch in (False, True):      # interleaved: same load on both
                        r, w = run(make, branch, engine)
                        walls[branch].append(w)
                        res[branch] = r
            except p.UnsupportedError as e:
                print(f"{name:9s} {engine:7s} refused: {e.feature} ({e.device})")
                continue
            ns = {}
            for branch in (False, True):
                c = counters(res[branch])
                ns[branch] = statistics.median(walls[branch]) / max(c["steps"], 1) * 1e9
                form = "branch" if branch else "conductance"
                print(f"{name:9s} {engine:7s} {form:12s} ns/step {ns[branch]:9.0f}  "
                      f"(median of {REPS}, spread {min(walls[branch]):.4f}-{max(walls[branch]):.4f} s)  {c}")
            ratios[(name, engine)] = ns[True] / ns[False]
            print(f"{name:9s} {engine:7s} branch / conductance = {ratios[(name, engine)]:.3f}")
    r = ratios[("inverter", "pwl")]
    if ("inverter", "trbdf2") in ratios:
        r2 = ratios[("inverter", "trbdf2")]
        print(f"7.2 (12-switch inverter, trbdf2): {r2:.3f}")
    print(f"7.3 gate (12-switch inverter, pwl): {r:.3f} {'<=' if r <= 1.5 else '>'} 1.5 -> "
          f"{'flip allowed' if r <= 1.5 else 'flip held'}")


if __name__ == "__main__":
    main()
