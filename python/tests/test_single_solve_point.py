"""refactor-unified-transient-stepper task 2.3: neither MNA driver calls the
cache's solves or the Newton solver except through its solve point.

Every direct call line carries a role tag: `solve-point` (inside the one
dispatch of the engine), `factor-reuse` (a diagnostic solve on an existing
factor: noise floor, mode judge, LTE filter; it produces no state) or
`solve-exempt: <why>` (a path outside the step loop, DECISIONS 22). A new
untagged call fails here, and so does a new solve point.
"""

import re
from pathlib import Path

SOLVER = Path(__file__).resolve().parents[2] / "core" / "include" / "pulsim" / "solver"
CALL = re.compile(r"cache\.solve(_at|_segment)?\(|solve_with_newton_b_extra\(")
TAG = re.compile(r"//\s*(solve-point|factor-reuse|solve-exempt: .+)$")
# the solve-point lines of each engine (fixed: mode_solve 3, event iteration
# 1 Newton + 3 linear, written out on purpose: DECISIONS 22a)
EXPECTED_SOLVE_POINT = {"run_transient.hpp": 7, "trbdf2_transient.hpp": 5}


def test_every_direct_solve_is_tagged():
    for name, n_point in EXPECTED_SOLVE_POINT.items():
        untagged, points = [], 0
        for k, line in enumerate((SOLVER / name).read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("//", 1)[0]
            if not CALL.search(code):
                continue
            m = TAG.search(line)
            if m is None:
                untagged.append(f"{name}:{k}: {line.strip()}")
            elif m.group(1) == "solve-point":
                points += 1
        assert not untagged, "direct solve outside the solve point:\n" + "\n".join(untagged)
        assert points == n_point, f"{name}: {points} solve-point lines, expected {n_point}"
