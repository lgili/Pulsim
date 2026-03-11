#!/usr/bin/env python3
"""Finalize Gate ADV decision from prototype report with reproducible evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return bool(value)


def build_gate_adv_decision(
    *,
    report: Dict[str, Any],
    report_path: Path,
    adr_path: Path,
) -> Dict[str, Any]:
    if report.get("schema") != "pulsim-advanced-solver-prototype-report-v1":
        raise RuntimeError("Unexpected report schema for Gate ADV decision input")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("Prototype report must include summary object")

    candidate = report.get("candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError("Prototype report must include candidate object")

    candidate_id = str(candidate.get("id", "")).strip()
    if not candidate_id:
        raise RuntimeError("Prototype report candidate.id must be non-empty")

    hard_constraints_passed = _as_bool(summary.get("hard_constraints_passed"), default=False)
    require_all_hard = _as_bool(
        report.get("selection_policy", {}).get("require_all_hard_constraints"), default=True
    )

    if hard_constraints_passed:
        decision = "candidate_ready_for_integration_review"
        rationale = (
            "Gate ADV hard constraints passed in isolated prototype evaluation; "
            "candidate can proceed to controlled integration review."
        )
    elif require_all_hard:
        decision = "defer_adoption_keep_isolated"
        rationale = (
            "Gate ADV hard constraints not met under strict policy; keep candidate "
            "isolated and continue evidence collection."
        )
    else:
        decision = "defer_adoption_policy_optional"
        rationale = (
            "Gate ADV hard constraints are optional for this policy profile; "
            "adoption deferred until stricter evidence is available."
        )

    decision_payload = {
        "schema": "pulsim-gate-adv-decision-v1",
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "approved",
        "decision": decision,
        "rationale": rationale,
        "candidate_id": candidate_id,
        "candidate_backend": str(candidate.get("backend", "")),
        "candidate_solver_family": str(candidate.get("solver_family", "")),
        "inputs": {
            "prototype_report_path": str(report_path.resolve()),
            "prototype_report_sha256": _sha256(report_path),
            "adr_path": str(adr_path.resolve()),
            "adr_sha256": _sha256(adr_path),
        },
        "evidence": {
            "total_cases": int(summary.get("total_cases", 0) or 0),
            "comparable_runtime_cases": int(summary.get("comparable_runtime_cases", 0) or 0),
            "baseline_success_rate": float(summary.get("baseline_success_rate", 0.0) or 0.0),
            "prototype_success_rate": float(summary.get("prototype_success_rate", 0.0) or 0.0),
            "success_rate_gain_abs": float(summary.get("success_rate_gain_abs", 0.0) or 0.0),
            "runtime_regression_rel": float(summary.get("runtime_regression_rel", 0.0) or 0.0),
            "hard_constraints_passed": hard_constraints_passed,
            "hard_constraints": summary.get("hard_constraints", {}),
            "weighted_total_score": float(summary.get("weighted_total_score", 0.0) or 0.0),
        },
    }
    return decision_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Produce formal Gate ADV decision from prototype report",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmarks/out_advanced_solver/advanced_solver_prototype_report.json"),
    )
    parser.add_argument(
        "--adr",
        type=Path,
        default=Path("docs/advanced-solver-adr.md"),
    )
    parser.add_argument(
        "--decision-out",
        type=Path,
        default=Path("benchmarks/out_advanced_solver/gate_adv_decision.json"),
    )
    parser.add_argument(
        "--fail-on-hard-constraints",
        action="store_true",
        help="Return non-zero when hard constraints are not passed",
    )
    args = parser.parse_args()

    report = _load_json(args.report.resolve())
    decision = build_gate_adv_decision(
        report=report,
        report_path=args.report.resolve(),
        adr_path=args.adr.resolve(),
    )

    args.decision_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.decision_out, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=False)
        handle.write("\n")

    hard_pass = bool(decision["evidence"]["hard_constraints_passed"])
    print(f"Gate ADV decision written to: {args.decision_out}")
    print(f"Decision: {decision['decision']} (status={decision['status']})")
    print(f"Hard constraints passed: {hard_pass}")

    if args.fail_on_hard_constraints and not hard_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
