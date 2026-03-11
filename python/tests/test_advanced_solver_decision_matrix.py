"""Tests for advanced solver decision matrix validation."""

from __future__ import annotations

from pathlib import Path

import yaml

import validate_advanced_solver_decision_matrix as validator


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _valid_payload() -> dict:
    return {
        "schema": "pulsim-advanced-solver-decision-v1",
        "version": 1,
        "description": "Objective benchmark contract",
        "candidates": [
            {
                "id": "sundials_ida_direct",
                "backend": "sundials",
                "solver_family": "ida",
                "formulation": "direct",
                "maturity": "production_candidate",
                "enabled_by_default": False,
            },
            {
                "id": "petsc_snes_ksp",
                "backend": "petsc",
                "solver_family": "snes",
                "formulation": "direct",
                "maturity": "experimental",
                "enabled_by_default": False,
            },
        ],
        "benchmark_set": [
            {
                "benchmark_id": "buck_switching",
                "scenarios": ["direct_trap", "trbdf2"],
                "class": "switch_heavy",
            },
            {
                "benchmark_id": "magnetic_core_saturation",
                "scenarios": ["direct_trap"],
                "class": "magnetic_nonlinear",
            },
        ],
        "decision_criteria": {
            "min_success_rate_gain_abs": 0.02,
            "max_runtime_regression_rel": 0.15,
            "max_memory_regression_rel": 0.20,
            "min_portability_score": 0.70,
            "max_maintenance_cost_score": 6.0,
        },
        "scoring_weights": {
            "robustness": 0.45,
            "runtime": 0.25,
            "portability": 0.20,
            "maintenance": 0.10,
        },
        "selection_policy": {
            "require_all_hard_constraints": True,
            "tie_breaker_order": [
                "robustness_gain",
                "runtime_regression",
                "portability_score",
                "maintenance_cost",
            ],
        },
    }


def test_validate_advanced_solver_decision_matrix_passes_for_valid_payload(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "advanced_solver_decision_matrix.yaml"
    _write_yaml(matrix_path, _valid_payload())

    errors = validator.validate_decision_matrix(matrix_path)
    assert errors == []


def test_validate_advanced_solver_decision_matrix_reports_schema_and_weight_errors(
    tmp_path: Path,
) -> None:
    payload = _valid_payload()
    payload["schema"] = "wrong-schema"
    payload["scoring_weights"]["maintenance"] = 0.20
    matrix_path = tmp_path / "advanced_solver_decision_matrix.yaml"
    _write_yaml(matrix_path, payload)

    errors = validator.validate_decision_matrix(matrix_path)
    assert any("schema must be" in error for error in errors)
    assert any("scoring_weights must sum to 1.0" in error for error in errors)


def test_validate_advanced_solver_decision_matrix_reports_candidate_validation_errors(
    tmp_path: Path,
) -> None:
    payload = _valid_payload()
    payload["candidates"][1]["id"] = "sundials_ida_direct"
    payload["candidates"][1]["solver_family"] = "unsupported_family"
    matrix_path = tmp_path / "advanced_solver_decision_matrix.yaml"
    _write_yaml(matrix_path, payload)

    errors = validator.validate_decision_matrix(matrix_path)
    assert any("duplicated" in error for error in errors)
    assert any("unsupported solver_family" in error for error in errors)
