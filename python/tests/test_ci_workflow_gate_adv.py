"""Workflow contract tests for Gate ADV benchmark decision pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_ci_workflow_includes_gate_adv_prototype_and_decision_steps() -> None:
    workflow = _load_yaml(ROOT / ".github/workflows/ci.yml")
    benchmark_steps = workflow["jobs"]["benchmark"]["steps"]

    prototype_step = next(
        (
            step
            for step in benchmark_steps
            if step.get("name") == "Run advanced solver isolated prototype (Gate ADV evidence)"
        ),
        None,
    )
    assert prototype_step is not None
    prototype_run = prototype_step.get("run", "")
    assert "benchmarks/advanced_solver_prototype_runner.py" in prototype_run
    assert "--candidate sundials_ida_direct" in prototype_run
    assert "--output-dir benchmarks/out_ci/advanced_solver" in prototype_run

    decision_step = next(
        (
            step
            for step in benchmark_steps
            if step.get("name") == "Finalize Gate ADV decision artifact"
        ),
        None,
    )
    assert decision_step is not None
    decision_run = decision_step.get("run", "")
    assert "benchmarks/gate_adv_decision.py" in decision_run
    assert (
        "--report benchmarks/out_ci/advanced_solver/advanced_solver_prototype_report.json"
        in decision_run
    )
    assert "--adr docs/advanced-solver-adr.md" in decision_run
    assert "--decision-out benchmarks/out_ci/advanced_solver/gate_adv_decision.json" in decision_run
