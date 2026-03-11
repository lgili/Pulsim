#!/usr/bin/env python3
"""Validate advanced-solver adoption decision benchmark matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml


ALLOWED_SOLVER_FAMILIES = {"ida", "cvode", "arkode", "kinsol", "snes"}
REQUIRED_DECISION_KEYS = (
    "min_success_rate_gain_abs",
    "max_runtime_regression_rel",
    "max_memory_regression_rel",
    "min_portability_score",
    "max_maintenance_cost_score",
)
REQUIRED_WEIGHT_KEYS = ("robustness", "runtime", "portability", "maintenance")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return payload


def _as_non_empty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _as_non_negative_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0.0:
        return None
    return numeric


def validate_decision_matrix(path: Path) -> List[str]:
    payload = _load_yaml(path)
    errors: List[str] = []

    if payload.get("schema") != "pulsim-advanced-solver-decision-v1":
        errors.append("schema must be 'pulsim-advanced-solver-decision-v1'")

    version = payload.get("version")
    if not isinstance(version, int) or version < 1:
        errors.append("version must be an integer >= 1")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("candidates must be a non-empty list")
    else:
        seen_ids: set[str] = set()
        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                errors.append(f"candidate #{idx} must be an object")
                continue
            candidate_id = _as_non_empty_str(candidate.get("id"))
            if candidate_id is None:
                errors.append(f"candidate #{idx} missing non-empty id")
            elif candidate_id in seen_ids:
                errors.append(f"candidate id '{candidate_id}' duplicated")
            else:
                seen_ids.add(candidate_id)

            backend = _as_non_empty_str(candidate.get("backend"))
            if backend is None:
                errors.append(f"candidate '{candidate.get('id', idx)}' missing backend")

            family = _as_non_empty_str(candidate.get("solver_family"))
            if family is None:
                errors.append(f"candidate '{candidate.get('id', idx)}' missing solver_family")
            elif family.lower() not in ALLOWED_SOLVER_FAMILIES:
                errors.append(
                    f"candidate '{candidate.get('id', idx)}' has unsupported solver_family '{family}'"
                )

    benchmark_set = payload.get("benchmark_set")
    if not isinstance(benchmark_set, list) or not benchmark_set:
        errors.append("benchmark_set must be a non-empty list")
    else:
        for idx, case in enumerate(benchmark_set):
            if not isinstance(case, dict):
                errors.append(f"benchmark_set entry #{idx} must be an object")
                continue
            benchmark_id = _as_non_empty_str(case.get("benchmark_id"))
            if benchmark_id is None:
                errors.append(f"benchmark_set entry #{idx} missing benchmark_id")

            scenarios = case.get("scenarios")
            if not isinstance(scenarios, list) or not scenarios:
                errors.append(f"benchmark_set entry #{idx} must define non-empty scenarios")
            else:
                for scenario in scenarios:
                    if _as_non_empty_str(scenario) is None:
                        errors.append(
                            f"benchmark_set entry #{idx} contains invalid scenario value"
                        )

    decision_criteria = payload.get("decision_criteria")
    if not isinstance(decision_criteria, dict):
        errors.append("decision_criteria must be an object")
    else:
        for key in REQUIRED_DECISION_KEYS:
            if key not in decision_criteria:
                errors.append(f"decision_criteria missing '{key}'")
                continue
            if _as_non_negative_float(decision_criteria.get(key)) is None:
                errors.append(f"decision_criteria '{key}' must be a non-negative number")

        portability = _as_non_negative_float(decision_criteria.get("min_portability_score"))
        if portability is not None and portability > 1.0:
            errors.append("decision_criteria 'min_portability_score' must be <= 1.0")

    weights = payload.get("scoring_weights")
    if not isinstance(weights, dict):
        errors.append("scoring_weights must be an object")
    else:
        weight_sum = 0.0
        for key in REQUIRED_WEIGHT_KEYS:
            if key not in weights:
                errors.append(f"scoring_weights missing '{key}'")
                continue
            numeric = _as_non_negative_float(weights.get(key))
            if numeric is None:
                errors.append(f"scoring_weights '{key}' must be a non-negative number")
                continue
            weight_sum += numeric
        if abs(weight_sum - 1.0) > 1e-9:
            errors.append("scoring_weights must sum to 1.0")

    policy = payload.get("selection_policy")
    if not isinstance(policy, dict):
        errors.append("selection_policy must be an object")
    else:
        if not isinstance(policy.get("require_all_hard_constraints"), bool):
            errors.append("selection_policy.require_all_hard_constraints must be boolean")
        tie_breaker_order = policy.get("tie_breaker_order")
        if not isinstance(tie_breaker_order, list) or not tie_breaker_order:
            errors.append("selection_policy.tie_breaker_order must be a non-empty list")
        else:
            for index, key in enumerate(tie_breaker_order):
                if _as_non_empty_str(key) is None:
                    errors.append(
                        f"selection_policy.tie_breaker_order entry #{index} must be a non-empty string"
                    )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate advanced solver decision benchmark matrix",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("benchmarks/advanced_solver_decision_matrix.yaml"),
    )
    args = parser.parse_args()

    errors = validate_decision_matrix(args.matrix.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Advanced solver decision matrix is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
