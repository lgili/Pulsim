"""Tests for Gate ADV decision artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import gate_adv_decision as gate


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prototype_report_payload(*, hard_constraints_passed: bool) -> Dict[str, Any]:
    return {
        "schema": "pulsim-advanced-solver-prototype-report-v1",
        "version": 1,
        "candidate": {
            "id": "sundials_ida_direct",
            "backend": "sundials",
            "solver_family": "ida",
        },
        "selection_policy": {
            "require_all_hard_constraints": True,
        },
        "summary": {
            "total_cases": 8,
            "comparable_runtime_cases": 6,
            "baseline_success_rate": 0.75,
            "prototype_success_rate": 0.80,
            "success_rate_gain_abs": 0.05,
            "runtime_regression_rel": 0.08,
            "hard_constraints_passed": hard_constraints_passed,
            "hard_constraints": {
                "min_success_rate_gain_abs": hard_constraints_passed,
            },
            "weighted_total_score": 0.81,
        },
    }


def test_build_gate_adv_decision_marks_candidate_ready_when_constraints_pass(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    adr_path = tmp_path / "adr.md"
    _write_json(report_path, _prototype_report_payload(hard_constraints_passed=True))
    adr_path.write_text("# ADR\n", encoding="utf-8")

    report = gate._load_json(report_path)
    decision = gate.build_gate_adv_decision(
        report=report,
        report_path=report_path,
        adr_path=adr_path,
    )

    assert decision["status"] == "approved"
    assert decision["decision"] == "candidate_ready_for_integration_review"
    assert decision["evidence"]["hard_constraints_passed"] is True
    assert len(decision["inputs"]["prototype_report_sha256"]) == 64
    assert len(decision["inputs"]["adr_sha256"]) == 64


def test_build_gate_adv_decision_defers_when_constraints_fail(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    adr_path = tmp_path / "adr.md"
    _write_json(report_path, _prototype_report_payload(hard_constraints_passed=False))
    adr_path.write_text("# ADR\n", encoding="utf-8")

    report = gate._load_json(report_path)
    decision = gate.build_gate_adv_decision(
        report=report,
        report_path=report_path,
        adr_path=adr_path,
    )

    assert decision["status"] == "approved"
    assert decision["decision"] == "defer_adoption_keep_isolated"
    assert decision["evidence"]["hard_constraints_passed"] is False


def test_gate_adv_decision_main_honors_fail_on_hard_constraints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "report.json"
    adr_path = tmp_path / "adr.md"
    out_path = tmp_path / "decision.json"

    _write_json(report_path, _prototype_report_payload(hard_constraints_passed=False))
    adr_path.write_text("# ADR\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "gate_adv_decision.py",
            "--report",
            str(report_path),
            "--adr",
            str(adr_path),
            "--decision-out",
            str(out_path),
            "--fail-on-hard-constraints",
        ],
    )
    assert gate.main() == 2
