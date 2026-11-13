"""Task 4.5: which shipped v2 YAML examples warn, run as the example
test runs them (200 fixed steps at the file's dt, all gates OFF), and
over their full window with the same drive.

    python examples_warn.py [full]
"""
import sys
import warnings
from pathlib import Path

import pulsim as p

EX = Path(__file__).resolve().parents[4] / "examples"
full = len(sys.argv) > 1
for path in sorted(EX.glob("*.yaml")):
    try:
        lc = p.load_yaml_file(str(path))
    except RuntimeError:
        continue  # v1 netlist
    t_end = lc.t_end if full else min(lc.t_end, lc.t_start + 200 * lc.dt)
    off = p.SwitchStateMask(lc.builder.graph.num_switches)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = p.simulate(lc, t_end=t_end, engine="pwl", switch_fn=lambda t: off,
                             resolution_diagnostics=True)
        except Exception as e:  # noqa: BLE001 — a report, not a check
            print(f"{path.name:40s} ERROR {type(e).__name__}")
            continue
    rd = res.resolution
    rw = [x for x in w if issubclass(x.category, p.ResolutionWarning)]
    names = lc.builder.state_var_names()
    ring = max(rd.ringing, key=lambda r: r.length, default=None)
    br = lc.builder.graph.branch_name(rd.lte_branch) if rd.lte_branch >= 0 else "-"
    print(f"{path.name:40s} warns={len(rw)} lte_max={rd.lte_max:.2e} ({br})"
          + (f" ring={names[ring.var]}:{ring.length}" if ring else ""))
