"""Tests for advanced solver prototype runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from pulsim_python_backend import BackendRunResult
import advanced_solver_prototype_runner as runner


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _matrix_payload() -> Dict[str, Any]:
    return {
        "schema": "pulsim-advanced-solver-decision-v1",
        "version": 1,
        "candidates": [
            {
                "id": "sundials_ida_direct",
                "backend": "sundials",
                "solver_family": "ida",
                "formulation": "direct",
                "maturity": "production_candidate",
                "enabled_by_default": False,
                "prototype_portability_score": 0.85,
                "prototype_maintenance_cost_score": 4.0,
            }
        ],
        "benchmark_set": [
            {
                "benchmark_id": "bench_case",
                "scenarios": ["direct_trap"],
                "class": "switch_heavy",
            }
        ],
        "decision_criteria": {
            "min_success_rate_gain_abs": 0.0,
            "max_runtime_regression_rel": 0.50,
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


def _manifest_payload() -> Dict[str, Any]:
    return {
        "benchmarks": [
            {
                "path": "circuits/bench_case.yaml",
                "scenarios": ["direct_trap"],
            }
        ],
        "scenarios": {
            "direct_trap": {
                "simulation": {
                    "integrator": "trapezoidal",
                }
            }
        },
    }


def _circuit_payload() -> Dict[str, Any]:
    return {
        "schema": "pulsim-v1",
        "version": 1,
        "benchmark": {"id": "bench_case", "validation": {"type": "none"}},
        "simulation": {"tstop": 1e-4, "dt": 1e-6},
        "components": [],
    }


def test_prototype_runner_applies_transient_override_and_generates_report(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "advanced_solver_decision_matrix.yaml"
    manifest_path = tmp_path / "benchmarks.yaml"
    circuits_dir = tmp_path / "circuits"
    circuits_dir.mkdir(parents=True, exist_ok=True)
    circuit_path = circuits_dir / "bench_case.yaml"
    output_dir = tmp_path / "out"

    _write_yaml(matrix_path, _matrix_payload())
    _write_yaml(manifest_path, _manifest_payload())
    _write_yaml(circuit_path, _circuit_payload())

    seen_payloads: List[Dict[str, Any]] = []

    def fake_run(
        netlist_path: Path,
        output_path: Path,
        preferred_mode: str | None = None,
        use_initial_conditions: bool = False,
    ) -> BackendRunResult:
        payload = yaml.safe_load(netlist_path.read_text(encoding="utf-8"))
        seen_payloads.append(payload)
        sim = payload.get("simulation", {})
        is_prototype = sim.get("integrator") == "trbdf2"
        output_path.write_text("time\n0\n", encoding="utf-8")
        return BackendRunResult(
            runtime_s=1.2 if is_prototype else 1.0,
            steps=200 if is_prototype else 220,
            mode=preferred_mode or "transient",
            telemetry={},
        )

    report = runner.run_advanced_solver_prototype(
        matrix_path=matrix_path,
        manifest_path=manifest_path,
        candidate_id="sundials_ida_direct",
        output_dir=output_dir,
        enforce_hard_constraints=False,
        max_cases=None,
        run_fn=fake_run,
    )

    assert report["schema"] == "pulsim-advanced-solver-prototype-report-v1"
    assert report["summary"]["total_cases"] == 1
    assert report["cases"][0]["mode"] == "transient"
    assert report["cases"][0]["runtime_regression_rel"] is not None
    assert (output_dir / "advanced_solver_prototype_report.json").exists()
    assert (output_dir / "advanced_solver_prototype_results.csv").exists()
    assert len(seen_payloads) == 2

    baseline_sim = seen_payloads[0]["simulation"]
    prototype_sim = seen_payloads[1]["simulation"]
    assert "fallback" not in baseline_sim
    assert "fallback_policy" not in baseline_sim
    assert "fallback" not in prototype_sim
    assert "fallback_policy" not in prototype_sim


def test_prototype_runner_does_not_apply_transient_override_on_shooting_mode(
    tmp_path: Path,
) -> None:
    matrix = _matrix_payload()
    matrix["benchmark_set"][0]["scenarios"] = ["shooting_default"]

    manifest = _manifest_payload()
    manifest["benchmarks"][0]["scenarios"] = ["shooting_default"]
    manifest["scenarios"] = {
        "shooting_default": {"simulation": {"shooting": {"period": 1e-3}}},
    }

    matrix_path = tmp_path / "advanced_solver_decision_matrix.yaml"
    manifest_path = tmp_path / "benchmarks.yaml"
    circuits_dir = tmp_path / "circuits"
    circuits_dir.mkdir(parents=True, exist_ok=True)
    circuit_path = circuits_dir / "bench_case.yaml"
    output_dir = tmp_path / "out"

    _write_yaml(matrix_path, matrix)
    _write_yaml(manifest_path, manifest)
    _write_yaml(circuit_path, _circuit_payload())

    seen_integrators: List[Any] = []

    def fake_run(
        netlist_path: Path,
        output_path: Path,
        preferred_mode: str | None = None,
        use_initial_conditions: bool = False,
    ) -> BackendRunResult:
        payload = yaml.safe_load(netlist_path.read_text(encoding="utf-8"))
        sim = payload.get("simulation", {})
        seen_integrators.append(sim.get("integrator"))
        output_path.write_text("time\n0\n", encoding="utf-8")
        return BackendRunResult(
            runtime_s=0.5,
            steps=10,
            mode=preferred_mode or "shooting",
            telemetry={},
        )

    report = runner.run_advanced_solver_prototype(
        matrix_path=matrix_path,
        manifest_path=manifest_path,
        candidate_id="sundials_ida_direct",
        output_dir=output_dir,
        enforce_hard_constraints=False,
        max_cases=None,
        run_fn=fake_run,
    )

    assert report["cases"][0]["mode"] == "shooting"
    assert seen_integrators == [None, None]


def test_prototype_runner_enforces_hard_constraints_when_requested(
    tmp_path: Path,
) -> None:
    matrix = _matrix_payload()
    matrix["decision_criteria"]["max_runtime_regression_rel"] = 0.01

    matrix_path = tmp_path / "advanced_solver_decision_matrix.yaml"
    manifest_path = tmp_path / "benchmarks.yaml"
    circuits_dir = tmp_path / "circuits"
    circuits_dir.mkdir(parents=True, exist_ok=True)
    circuit_path = circuits_dir / "bench_case.yaml"
    output_dir = tmp_path / "out"

    _write_yaml(matrix_path, matrix)
    _write_yaml(manifest_path, _manifest_payload())
    _write_yaml(circuit_path, _circuit_payload())

    def fake_run(
        netlist_path: Path,
        output_path: Path,
        preferred_mode: str | None = None,
        use_initial_conditions: bool = False,
    ) -> BackendRunResult:
        payload = yaml.safe_load(netlist_path.read_text(encoding="utf-8"))
        sim = payload.get("simulation", {})
        is_prototype = sim.get("integrator") == "trbdf2"
        output_path.write_text("time\n0\n", encoding="utf-8")
        return BackendRunResult(
            runtime_s=2.0 if is_prototype else 1.0,
            steps=20,
            mode=preferred_mode or "transient",
            telemetry={},
        )

    try:
        runner.run_advanced_solver_prototype(
            matrix_path=matrix_path,
            manifest_path=manifest_path,
            candidate_id="sundials_ida_direct",
            output_dir=output_dir,
            enforce_hard_constraints=True,
            max_cases=None,
            run_fn=fake_run,
        )
        assert False, "Expected hard constraint failure"
    except RuntimeError as exc:
        assert "failed hard constraints" in str(exc).lower()
