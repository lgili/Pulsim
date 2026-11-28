"""A CI-safe subset of benchmarks/EXACT_SCOREBOARD.md: engine runs graded
with the ``exact`` measure (three parts: smooth, instants, limits; rule (1)
for a reported projection) against the exact truth.

Each bound is a ceiling with margin, explained by the part that sets the
cell. A cell that is a known open finding is pinned AS the finding (a floor
too, with its id): fixing the engine fails the floor, and the fix then
tightens the bound. An engine change that moves a cell past its bound
regenerates the scoreboard and moves the bound.
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

SPLIT = dict(discontinuity_restart=True, enable_substep_state_correction=True)

# circuit -> label -> (engine, simulate() keywords, floor, ceiling, part that sets it)
CELLS = {
    "circuits/diodes/diode_rectifier.yaml": {
        # plain pwl's error on the conduction intervals: its left values at D1's events
        # (limits 0.020) and the same error in RMS (smooth 0.010)
        "pwl": ("pwl", {}, 0.0, 0.06, "limits"),
        "trbdf2": ("trbdf2", {}, 0.0, 1e-5, "limits"),      # 2.6e-7
    },
    "circuits/switching/isolated/flyback_converter.yaml": {
        # a real engine error of plain pwl (smooth 0.95 alone), not the measure
        "pwl": ("pwl", {}, 0.3, 10.0, "limits"),
        "dsed": ("dsed", {}, 0.0, 1e-4, "limits"),         # 2.2e-6
    },
    "circuits/switching/basic/vcswitch_resistor.yaml": {
        # finding F-SB-2 (the split rule applied the gate edge from the start of its step,
        # delta = -0.005 h: instants 0.0032) is closed on the restart line: the edge is
        # located, instants 2.0e-9
        "pwl+split F-SB-2": ("pwl", SPLIT, 0.0, 1e-6, "instants"),
    },
    "circuits/diodes/diode_half_wave_rl_load.yaml": {
        # finding F-SB-3 / F-R-3 (closed): the right limit of V(out) at D1's turn-off is the
        # projected value (-3.1e-6 V; the truth -0.314 V). Without a projection record it read
        # limits 0.030; the exact engine now records every event projection, so rule (1)
        # prices it as a 1 ns collapsed transient (instants 4.8e-6)
        "dsed F-SB-3": ("dsed", {}, 0.0, 1e-4, "instants"),
    },
}


@pytest.mark.parametrize("rel", list(CELLS))
def test_scoreboard_subset(rel):
    sol = et.Solver(et.Circuit(ROOT / rel), 30).run()
    case = v1_reader.load(ROOT / rel)
    cols = dr.observables_of(case)
    g, _, _ = sb.graded_columns(sol, cols)
    sp = sb.spans(sol, g)
    for label, (eng, kw, lo, hi, part) in CELLS[rel].items():
        r = sb.grade(sol, dr.traces(case.run(eng, **kw), cols), g, sp, case.dt, 1e-3,
                     parasitic=sb.PARASITIC.get(eng, 0.0) * case.dt)
        assert lo <= r["exact"] <= hi, (label, r["exact"])
        assert r["exact_part"] == part, (label, r["exact_part"])
        assert r["self"] < 1e-11, (label, r["self"])     # the truth on this engine's own grid
