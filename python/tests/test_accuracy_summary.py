"""benchmarks/ACCURACY_SUMMARY.md is generated from the stored evidence files
(exact_scoreboard.json, NEXT_DEFAULTS_REHEARSAL.md, IDEAL_SWITCH_FLIP_EVIDENCE.md, EXACT_TRUTH.md):
it must equal what benchmarks/tools/accuracy_summary.py builds from them. Nothing is
simulated; a fresh measurement is not compared."""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "benchmarks" / "tools"
sys.path.insert(0, str(TOOLS))
import accuracy_summary  # noqa: E402


def test_accuracy_summary_is_current():
    page = accuracy_summary.OUT.read_text(encoding="utf-8")
    assert page == accuracy_summary.build(), (
        "benchmarks/ACCURACY_SUMMARY.md is stale: run benchmarks/tools/accuracy_summary.py")
