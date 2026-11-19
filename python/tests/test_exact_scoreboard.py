"""A CI-safe subset of benchmarks/EXACT_SCOREBOARD.md: engine runs graded
with the ``exact`` measure against the exact truth, as bounds with margin.

Values on macOS at the scoreboard's tip (bounds >= 30x above / 3x below):
diode_rectifier pwl 0.0097, trbdf2 2.4e-7; flyback_converter pwl 0.95 (a
real engine error, not the interp floor), dsed 2.1e-6. An engine fix that
moves a cell past its bound regenerates the scoreboard and moves the bound.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mpmath")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import differential_runner as dr  # noqa: E402
import exact_scoreboard as sb  # noqa: E402
import exact_truth as et  # noqa: E402
import v1_reader  # noqa: E402

CELLS = {
    "circuits/diodes/diode_rectifier.yaml": {"pwl": (3e-3, 0.03), "trbdf2": (0.0, 1e-5)},
    "circuits/switching/isolated/flyback_converter.yaml": {"pwl": (0.3, 3.0), "dsed": (0.0, 1e-4)},
}


@pytest.mark.parametrize("rel", list(CELLS))
def test_scoreboard_subset(rel):
    sol = et.Solver(et.Circuit(ROOT / rel), 30).run()
    case = v1_reader.load(ROOT / rel)
    cols = dr.observables_of(case)
    g, _, _ = sb.graded_columns(sol, cols)
    sp = sb.spans(sol, g)
    for eng, (lo, hi) in CELLS[rel].items():
        r = sb.grade(sol, dr.traces(case.run(eng), cols), g, sp, case.dt, 1e-3)
        assert lo <= r["exact"] <= hi, (eng, r["exact"])
        assert r["self"] < 1e-11, (eng, r["self"])     # the truth on this engine's own grid
