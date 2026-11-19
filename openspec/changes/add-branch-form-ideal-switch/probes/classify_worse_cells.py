"""Classify the 'worse' ideal-twin cells: reference / engine / circuit.
For each (circuit, config): under ideal_form
  R1 = trbdf2 rtol=1e-8 (the harness reference), R2 = pwl dt/16 (plain),
  e_ref = err(R2 vs R1)    -> reference trust (both methods agree?)
  e1 = err(cfg @ dt vs R1), e4 = err(cfg @ dt/4 vs R1)
Rules: e_ref > tol/2 -> 'reference'; else e4 <= max(e1/2.5, tol/4) -> 'circuit'
(converges with dt: discretisation of a sharper circuit); else 'engine'.
Also the conductance twin's e1 for context.  Prints one JSON line per cell."""
import json
import sys
import time
from pathlib import Path
sys.path[:0] = ["benchmarks/tools", "benchmarks"]
import differential_runner as dr  # noqa: E402
import ideal_switch_flip_evidence as ev  # noqa: E402
import reference_status as rs  # noqa: E402
import v1_reader  # noqa: E402

ROOT = Path("benchmarks")
cells = [tuple(a.split("@")) for a in sys.argv[1:]]
for rel, cfg in cells:
    t0 = time.perf_counter()
    case = v1_reader.load(ROOT / rel)
    cols = dr.observables_of(case)
    eng, kw = rs.CONFIGS[cfg]
    out = dict(circuit=rel, config=cfg)
    try:
        with ev.ideal_form():
            R1 = dr.traces(case.run("trbdf2", rtol=1e-8, atol=1e-12), cols)
            with rs.extra_kwargs({"discontinuity_restart": True}):
                R2 = dr.traces(case.run("pwl", dt=case.dt / 16), cols)
            out["e_ref"] = dr.compare(R2, R1, cols, 1e-3)[0]
            with rs.extra_kwargs(kw):
                if eng == "pwl":
                    out["e1"] = dr.compare(dr.traces(case.run(eng), cols), R1, cols, 1e-3)[0]
                    out["e4"] = dr.compare(dr.traces(case.run(eng, dt=case.dt / 4), cols),
                                           R1, cols, 1e-3)[0]
                    out["e1_vs_R2"] = dr.compare(dr.traces(case.run(eng), cols), R2, cols, 1e-3)[0]
                else:
                    out["e1"] = dr.compare(dr.traces(case.run(eng), cols), R1, cols, 1e-3)[0]
                    out["e1_vs_R2"] = dr.compare(dr.traces(case.run(eng), cols), R2, cols, 1e-3)[0]
                    if eng == "trbdf2":
                        out["e_tight"] = dr.compare(dr.traces(case.run(eng, rtol=1e-6, atol=1e-9),
                                                              cols), R1, cols, 1e-3)[0]
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {ex}"[:200]
    out["wall"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(out), flush=True)
