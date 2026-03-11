"""Tests for benchmark KPI regression gate."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import pytest
import yaml

import kpi_gate


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_baseline_with_manifest(
    baseline_path: Path,
    baseline_id: str,
    source_bench_results: Path,
    metrics: dict,
    source_artifacts_root: Path | None = None,
    machine_class: str | None = None,
) -> None:
    effective_machine_class = machine_class or (platform.machine() or "unknown")
    captured_at = "2026-02-25T00:00:00Z"
    baseline = {
        "schema_version": "pulsim-kpi-baseline-v1",
        "baseline_id": baseline_id,
        "captured_at_utc": captured_at,
        "source_bench_results": str(source_bench_results.resolve()),
        "metrics": metrics,
        "environment": {
            "os": "test-os",
            "python": "Python 3.13",
            "machine_class": effective_machine_class,
            "compiler": "clang test",
            "cc": "clang test",
            "cmake": "cmake test",
            "cxx_flags": "-O3",
        },
    }
    if source_artifacts_root is not None:
        baseline["source_artifacts_root"] = str(source_artifacts_root.resolve())
    _write_json(baseline_path, baseline)

    manifest = {
        "schema_version": "pulsim-kpi-baseline-manifest-v1",
        "baseline_id": baseline_id,
        "captured_at_utc": captured_at,
        "files": [
            {
                "path": str(source_bench_results.resolve()),
                "sha256": _sha256_file(source_bench_results.resolve()),
                "size_bytes": source_bench_results.resolve().stat().st_size,
            }
        ],
    }
    _write_json(baseline_path.with_name("artifact_manifest.json"), manifest)


def test_kpi_gate_passes_with_non_regressive_metrics(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {"status": "passed", "runtime_s": 0.12},
            {"status": "passed", "runtime_s": 0.18},
            {"status": "passed", "runtime_s": 0.21},
        ]
    }
    baseline_metrics = {
        "convergence_success_rate": 1.0,
        "runtime_p95": 0.25,
        "stress_tier_failure_count": 0.0,
    }
    thresholds = {
        "metrics": {
            "convergence_success_rate": {
                "direction": "higher_is_better",
                "max_regression_abs": 0.01,
                "required": True,
            },
            "runtime_p95": {
                "direction": "lower_is_better",
                "max_regression_rel": 0.10,
                "required": True,
            },
            "stress_tier_failure_count": {
                "direction": "lower_is_better",
                "max_regression_abs": 0.0,
                "required": True,
            },
        }
    }
    stress_summary = {"tiers_failed": 0}

    bench_path = tmp_path / "bench.json"
    baseline_path = tmp_path / "kpi_baseline.json"
    thresholds_path = tmp_path / "thresholds.yaml"
    stress_path = tmp_path / "stress_summary.json"

    _write_json(bench_path, bench_results)
    _write_baseline_with_manifest(
        baseline_path=baseline_path,
        baseline_id="phase0",
        source_bench_results=bench_path,
        metrics=baseline_metrics,
    )
    _write_yaml(thresholds_path, thresholds)
    _write_json(stress_path, stress_summary)

    report = kpi_gate.run_gate(
        baseline_path=baseline_path,
        thresholds_path=thresholds_path,
        bench_results_path=bench_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        stress_summary_path=stress_path,
    )

    assert report["overall_status"] == "passed"
    assert report["failed_required_metrics"] == 0


def test_kpi_gate_fails_on_runtime_regression(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {"status": "passed", "runtime_s": 0.10},
            {"status": "passed", "runtime_s": 0.50},
            {"status": "passed", "runtime_s": 0.80},
        ]
    }
    baseline_metrics = {
        "convergence_success_rate": 1.0,
        "runtime_p95": 0.20,
    }
    thresholds = {
        "metrics": {
            "convergence_success_rate": {
                "direction": "higher_is_better",
                "max_regression_abs": 0.01,
                "required": True,
            },
            "runtime_p95": {
                "direction": "lower_is_better",
                "max_regression_rel": 0.05,
                "required": True,
            },
            "parity_rms_error_ltspice": {
                "direction": "lower_is_better",
                "max_regression_rel": 0.10,
                "required": False,
            },
        }
    }

    bench_path = tmp_path / "bench.json"
    baseline_path = tmp_path / "kpi_baseline.json"
    thresholds_path = tmp_path / "thresholds.yaml"

    _write_json(bench_path, bench_results)
    _write_baseline_with_manifest(
        baseline_path=baseline_path,
        baseline_id="phase0",
        source_bench_results=bench_path,
        metrics=baseline_metrics,
    )
    _write_yaml(thresholds_path, thresholds)

    report = kpi_gate.run_gate(
        baseline_path=baseline_path,
        thresholds_path=thresholds_path,
        bench_results_path=bench_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        stress_summary_path=None,
    )

    assert report["overall_status"] == "failed"
    assert report["failed_required_metrics"] == 1
    assert report["comparisons"]["parity_rms_error_ltspice"]["status"] == "skipped"


def test_kpi_gate_fails_on_component_consistency_regression(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {
                "benchmark_id": "buck_electrothermal",
                "scenario": "fixed_mode",
                "status": "passed",
                "runtime_s": 0.12,
                "telemetry": {
                    "component_coverage_rate": 0.75,
                    "component_coverage_gap": 1.0,
                    "component_loss_summary_consistency_error": 2.0e-2,
                    "component_thermal_summary_consistency_error": 1.0e-2,
                },
            }
        ]
    }
    baseline_metrics = {
        "component_coverage_rate": 1.0,
        "component_coverage_gap": 0.0,
        "component_loss_summary_consistency_error": 0.0,
        "component_thermal_summary_consistency_error": 0.0,
    }
    thresholds = {
        "metrics": {
            "component_coverage_rate": {
                "direction": "higher_is_better",
                "max_regression_abs": 0.01,
                "required": True,
            },
            "component_coverage_gap": {
                "direction": "lower_is_better",
                "max_regression_abs": 0.0,
                "required": True,
            },
            "component_loss_summary_consistency_error": {
                "direction": "lower_is_better",
                "max_regression_abs": 1e-6,
                "required": True,
            },
            "component_thermal_summary_consistency_error": {
                "direction": "lower_is_better",
                "max_regression_abs": 1e-6,
                "required": True,
            },
        }
    }

    bench_path = tmp_path / "bench.json"
    baseline_path = tmp_path / "kpi_baseline.json"
    thresholds_path = tmp_path / "thresholds.yaml"

    _write_json(bench_path, bench_results)
    _write_baseline_with_manifest(
        baseline_path=baseline_path,
        baseline_id="phase0",
        source_bench_results=bench_path,
        metrics=baseline_metrics,
    )
    _write_yaml(thresholds_path, thresholds)

    report = kpi_gate.run_gate(
        baseline_path=baseline_path,
        thresholds_path=thresholds_path,
        bench_results_path=bench_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        stress_summary_path=None,
    )

    assert report["overall_status"] == "failed"
    assert report["failed_required_metrics"] == 4
    assert report["comparisons"]["component_coverage_rate"]["status"] == "failed"
    assert report["comparisons"]["component_loss_summary_consistency_error"]["status"] == "failed"


def test_compute_metrics_includes_state_space_primary_ratio(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {
                "status": "passed",
                "runtime_s": 0.10,
                "telemetry": {
                    "state_space_primary_steps": 8.0,
                    "dae_fallback_steps": 2.0,
                },
            },
            {
                "status": "passed",
                "runtime_s": 0.12,
                "telemetry": {
                    "state_space_primary_steps": 6.0,
                    "dae_fallback_steps": 4.0,
                },
            },
            {
                "status": "failed",
                "runtime_s": 0.14,
                "telemetry": {
                    "state_space_primary_steps": 0.0,
                    "dae_fallback_steps": 5.0,
                },
            },
        ]
    }
    bench_path = tmp_path / "bench.json"
    _write_json(bench_path, bench_results)

    metrics = kpi_gate.compute_metrics(bench_results_path=bench_path)
    assert metrics["state_space_primary_ratio"] == 14.0 / 25.0


def test_compute_metrics_includes_electrothermal_metrics(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {
                "status": "passed",
                "runtime_s": 0.10,
                "telemetry": {
                    "state_space_primary_steps": 8.0,
                    "dae_fallback_steps": 2.0,
                    "loss_energy_balance_error": 0.012,
                    "thermal_peak_temperature_delta": 18.0,
                    "runtime_module_order_crc32": 101.0,
                    "runtime_module_count_match": 1.0,
                    "output_reallocation_total": 0.0,
                    "output_reallocation_free": 1.0,
                    "component_coverage_rate": 1.0,
                    "component_coverage_gap": 0.0,
                    "component_loss_summary_consistency_error": 0.0,
                    "component_thermal_summary_consistency_error": 0.0,
                },
            },
            {
                "status": "passed",
                "runtime_s": 0.12,
                "telemetry": {
                    "state_space_primary_steps": 6.0,
                    "dae_fallback_steps": 4.0,
                    "loss_energy_balance_error": 0.008,
                    "thermal_peak_temperature_delta": 22.0,
                    "runtime_module_order_crc32": 101.0,
                    "runtime_module_count_match": 1.0,
                    "output_reallocation_total": 0.0,
                    "output_reallocation_free": 1.0,
                    "component_coverage_rate": 0.9,
                    "component_coverage_gap": 0.1,
                    "component_loss_summary_consistency_error": 1.0e-4,
                    "component_thermal_summary_consistency_error": 5.0e-5,
                },
            },
        ]
    }
    bench_path = tmp_path / "bench.json"
    _write_json(bench_path, bench_results)

    metrics = kpi_gate.compute_metrics(bench_results_path=bench_path)
    assert metrics["state_space_primary_ratio"] == 14.0 / 20.0
    assert metrics["dae_fallback_ratio"] == 6.0 / 20.0
    assert metrics["module_order_mismatch_rate"] == 0.0
    assert metrics["module_order_count_match_rate"] == 1.0
    assert metrics["module_output_reallocation_p95"] == 0.0
    assert metrics["module_output_reallocation_free_rate"] == 1.0
    assert metrics["loss_energy_balance_error"] == 0.01
    assert metrics["thermal_peak_temperature_delta"] == 20.0
    assert metrics["component_coverage_rate"] == 0.95
    assert metrics["component_coverage_gap"] == 0.05
    assert metrics["component_loss_summary_consistency_error"] == 5.0e-5
    assert metrics["component_thermal_summary_consistency_error"] == 2.5e-5


def test_compute_metrics_detects_module_order_mismatch_rate(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {
                "status": "passed",
                "runtime_s": 0.10,
                "telemetry": {"runtime_module_order_crc32": 55.0, "output_reallocation_total": 0.0},
            },
            {
                "status": "passed",
                "runtime_s": 0.11,
                "telemetry": {"runtime_module_order_crc32": 55.0, "output_reallocation_total": 0.0},
            },
            {
                "status": "passed",
                "runtime_s": 0.12,
                "telemetry": {"runtime_module_order_crc32": 99.0, "output_reallocation_total": 1.0},
            },
        ]
    }
    bench_path = tmp_path / "bench.json"
    _write_json(bench_path, bench_results)

    metrics = kpi_gate.compute_metrics(bench_results_path=bench_path)
    assert metrics["module_order_mismatch_rate"] == 1.0 / 3.0
    assert math.isclose(float(metrics["module_output_reallocation_p95"]), 0.9, rel_tol=0.0, abs_tol=1e-12)


def test_compute_metrics_includes_convergence_policy_kpis(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {
                "status": "passed",
                "runtime_s": 0.10,
                "telemetry": {
                    "classified_fallback_events": 4.0,
                    "policy_dry_run_events": 4.0,
                    "policy_recommendation_matches": 3.0,
                    "policy_recommendation_mismatches": 1.0,
                    "anti_overfit_violations": 1.0,
                    "anti_overfit_budget_exceeded": 1.0,
                },
            },
            {
                "status": "passed",
                "runtime_s": 0.12,
                "telemetry": {
                    "classified_fallback_events": 2.0,
                    "policy_dry_run_events": 2.0,
                    "policy_recommendation_matches": 2.0,
                    "policy_recommendation_mismatches": 0.0,
                    "anti_overfit_violations": 0.0,
                    "anti_overfit_budget_exceeded": 0.0,
                },
            },
        ]
    }
    bench_path = tmp_path / "bench.json"
    _write_json(bench_path, bench_results)

    metrics = kpi_gate.compute_metrics(bench_results_path=bench_path)
    assert metrics["classified_fallback_events_p95"] is not None
    assert math.isclose(
        float(metrics["classified_fallback_events_p95"]),
        3.9,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert metrics["convergence_policy_match_rate"] is not None
    assert math.isclose(
        float(metrics["convergence_policy_match_rate"]),
        5.0 / 6.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert metrics["convergence_policy_mismatch_rate"] is not None
    assert math.isclose(
        float(metrics["convergence_policy_mismatch_rate"]),
        1.0 / 6.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert metrics["anti_overfit_violation_rate"] is not None
    assert math.isclose(
        float(metrics["anti_overfit_violation_rate"]),
        1.0 / 6.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert metrics["anti_overfit_budget_exceeded_rate"] == 0.5


def test_compute_metrics_includes_convergence_class_matrix_kpis(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {
                "benchmark_id": "diode_rectifier",
                "scenario": "direct_trap",
                "status": "passed",
                "runtime_s": 0.10,
                "telemetry": {
                    "newton_iterations": 10.0,
                    "timestep_rejections": 2.0,
                    "classified_fallback_events": 1.0,
                    "policy_dry_run_events": 1.0,
                    "policy_recommendation_matches": 1.0,
                    "policy_recommendation_mismatches": 0.0,
                    "anti_overfit_violations": 0.0,
                    "anti_overfit_budget_exceeded": 0.0,
                },
            },
            {
                "benchmark_id": "buck_switching",
                "scenario": "direct_trap",
                "status": "failed",
                "runtime_s": 0.20,
                "telemetry": {
                    "newton_iterations": 20.0,
                    "timestep_rejections": 5.0,
                    "classified_fallback_events": 2.0,
                    "policy_dry_run_events": 2.0,
                    "policy_recommendation_matches": 1.0,
                    "policy_recommendation_mismatches": 1.0,
                    "anti_overfit_violations": 0.0,
                    "anti_overfit_budget_exceeded": 0.0,
                },
            },
            {
                "benchmark_id": "buck_switching",
                "scenario": "trbdf2",
                "status": "passed",
                "runtime_s": 0.25,
                "telemetry": {
                    "newton_iterations": 30.0,
                    "timestep_rejections": 8.0,
                },
            },
            {
                "benchmark_id": "magnetic_core_saturation",
                "scenario": "direct_trap",
                "status": "passed",
                "runtime_s": 0.05,
                "telemetry": {
                    "newton_iterations": 5.0,
                    "timestep_rejections": 1.0,
                    "classified_fallback_events": 0.0,
                    "policy_dry_run_events": 0.0,
                    "policy_recommendation_matches": 0.0,
                    "policy_recommendation_mismatches": 0.0,
                    "anti_overfit_violations": 0.0,
                    "anti_overfit_budget_exceeded": 0.0,
                },
            },
        ]
    }
    class_matrix = {
        "schema": "pulsim-convergence-class-matrix-v1",
        "version": 1,
        "classes": {
            "diode_heavy": {
                "cases": [
                    {"benchmark_id": "diode_rectifier", "scenarios": ["direct_trap"]},
                ]
            },
            "switch_heavy": {
                "cases": [
                    {"benchmark_id": "buck_switching", "scenarios": ["direct_trap", "trbdf2"]},
                ]
            },
            "closed_loop_control": {
                "cases": [
                    {"benchmark_id": "periodic_rc_pwm", "scenarios": ["shooting_default"]},
                ]
            },
        },
    }
    bench_path = tmp_path / "bench.json"
    class_matrix_path = tmp_path / "class_matrix.yaml"
    _write_json(bench_path, bench_results)
    _write_yaml(class_matrix_path, class_matrix)

    metrics = kpi_gate.compute_metrics(
        bench_results_path=bench_path,
        class_matrix_path=class_matrix_path,
    )

    assert math.isclose(
        float(metrics["typed_convergence_schema_coverage_rate"]),
        0.75,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert metrics["class_diode_heavy_case_count"] == 1.0
    assert metrics["class_diode_heavy_coverage_rate"] == 1.0
    assert metrics["class_diode_heavy_pass_rate"] == 1.0
    assert metrics["class_diode_heavy_typed_schema_coverage_rate"] == 1.0

    assert metrics["class_switch_heavy_case_count"] == 2.0
    assert metrics["class_switch_heavy_coverage_rate"] == 1.0
    assert metrics["class_switch_heavy_pass_rate"] == 0.5
    assert metrics["class_switch_heavy_typed_schema_coverage_rate"] == 0.5
    assert math.isclose(
        float(metrics["class_switch_heavy_runtime_p95"]),
        0.2475,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        float(metrics["class_switch_heavy_newton_iterations_p95"]),
        29.5,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert metrics["class_closed_loop_control_case_count"] == 1.0
    assert metrics["class_closed_loop_control_coverage_rate"] == 0.0
    assert metrics["class_closed_loop_control_pass_rate"] is None
    assert metrics["policy_target_pass_rate"] == 0.5
    assert metrics["policy_target_match_rate"] == 0.5
    assert metrics["policy_target_mismatch_rate"] == 0.5
    assert metrics["policy_stable_pass_rate"] == 1.0
    assert metrics["policy_stable_mismatch_rate"] == 0.0
    assert metrics["policy_stable_anti_overfit_violation_rate"] == 0.0


def test_compute_policy_guard_rates_default_to_zero_without_policy_events(
    tmp_path: Path,
) -> None:
    bench_results = {
        "results": [
            {
                "benchmark_id": "buck_switching",
                "scenario": "direct_trap",
                "status": "passed",
                "runtime_s": 0.20,
                "telemetry": {
                    "policy_dry_run_events": 0.0,
                    "policy_recommendation_matches": 0.0,
                    "policy_recommendation_mismatches": 0.0,
                    "anti_overfit_violations": 0.0,
                    "anti_overfit_budget_exceeded": 0.0,
                    "classified_fallback_events": 0.0,
                },
            },
            {
                "benchmark_id": "diode_rectifier",
                "scenario": "direct_trap",
                "status": "passed",
                "runtime_s": 0.10,
                "telemetry": {
                    "policy_dry_run_events": 0.0,
                    "policy_recommendation_matches": 0.0,
                    "policy_recommendation_mismatches": 0.0,
                    "anti_overfit_violations": 0.0,
                    "anti_overfit_budget_exceeded": 0.0,
                    "classified_fallback_events": 0.0,
                },
            },
        ]
    }
    class_matrix = {
        "schema": "pulsim-convergence-class-matrix-v1",
        "version": 1,
        "classes": {
            "switch_heavy": {
                "cases": [
                    {"benchmark_id": "buck_switching", "scenarios": ["direct_trap"]},
                ]
            },
            "diode_heavy": {
                "cases": [
                    {"benchmark_id": "diode_rectifier", "scenarios": ["direct_trap"]},
                ]
            },
        },
    }
    bench_path = tmp_path / "bench.json"
    class_matrix_path = tmp_path / "class_matrix.yaml"
    _write_json(bench_path, bench_results)
    _write_yaml(class_matrix_path, class_matrix)

    metrics = kpi_gate.compute_metrics(
        bench_results_path=bench_path,
        class_matrix_path=class_matrix_path,
    )

    assert metrics["policy_target_pass_rate"] == 1.0
    assert metrics["policy_target_match_rate"] is None
    assert metrics["policy_target_mismatch_rate"] == 0.0
    assert metrics["policy_stable_pass_rate"] == 1.0
    assert metrics["policy_stable_mismatch_rate"] == 0.0
    assert metrics["policy_stable_anti_overfit_violation_rate"] == 0.0


def test_compute_metrics_includes_ac_sweep_accuracy_and_runtime_metrics(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {
                "benchmark_id": "ac_rc_lowpass",
                "scenario": "direct_trap",
                "mode": "frequency_analysis",
                "status": "passed",
                "runtime_s": 0.020,
                "telemetry": {
                    "ac_sweep_case": 1.0,
                    "ac_sweep_mag_error": 0.002,
                    "ac_sweep_phase_error": 0.12,
                },
            },
            {
                "benchmark_id": "ac_rc_lowpass",
                "scenario": "fast_grid",
                "mode": "frequency_analysis",
                "status": "passed",
                "runtime_s": 0.060,
                "telemetry": {
                    "ac_sweep_case": 1.0,
                    "ac_sweep_mag_error": 0.004,
                    "ac_sweep_phase_error": 0.18,
                },
            },
            {
                "benchmark_id": "rc_step",
                "scenario": "direct_trap",
                "mode": "transient",
                "status": "passed",
                "runtime_s": 0.300,
                "telemetry": {},
            },
        ]
    }
    bench_path = tmp_path / "bench.json"
    _write_json(bench_path, bench_results)

    metrics = kpi_gate.compute_metrics(bench_results_path=bench_path)
    assert metrics["ac_sweep_mag_error"] == 0.003
    assert metrics["ac_sweep_phase_error"] == 0.15
    assert metrics["ac_runtime_p95"] is not None
    assert math.isclose(float(metrics["ac_runtime_p95"]), 0.058, rel_tol=0.0, abs_tol=1e-12)
    assert metrics["runtime_p95"] is not None
    assert float(metrics["runtime_p95"]) > float(metrics["ac_runtime_p95"])


def test_compute_metrics_includes_averaged_pair_metrics(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {
                "benchmark_id": "buck_switching",
                "scenario": "direct_trap",
                "mode": "transient",
                "status": "passed",
                "runtime_s": 0.30,
                "telemetry": {
                    "averaged_pair_case": 1.0,
                    "averaged_pair_group_crc32": 101.0,
                    "averaged_pair_role_switching": 1.0,
                    "averaged_pair_role_averaged": 0.0,
                },
            },
            {
                "benchmark_id": "buck_averaged_mvp",
                "scenario": "direct_trap",
                "mode": "transient",
                "status": "passed",
                "runtime_s": 0.10,
                "max_error": 0.25,
                "telemetry": {
                    "averaged_pair_case": 1.0,
                    "averaged_pair_group_crc32": 101.0,
                    "averaged_pair_role_switching": 0.0,
                    "averaged_pair_role_averaged": 1.0,
                },
            },
            {
                "benchmark_id": "buck_switching_fast",
                "scenario": "direct_trap",
                "mode": "transient",
                "status": "passed",
                "runtime_s": 0.50,
                "telemetry": {
                    "averaged_pair_case": 1.0,
                    "averaged_pair_group_crc32": 202.0,
                    "averaged_pair_role_switching": 1.0,
                    "averaged_pair_role_averaged": 0.0,
                },
            },
            {
                "benchmark_id": "buck_averaged_fast",
                "scenario": "direct_trap",
                "mode": "transient",
                "status": "passed",
                "runtime_s": 0.25,
                "max_error": 0.15,
                "telemetry": {
                    "averaged_pair_case": 1.0,
                    "averaged_pair_group_crc32": 202.0,
                    "averaged_pair_role_switching": 0.0,
                    "averaged_pair_role_averaged": 1.0,
                },
            },
        ]
    }
    bench_path = tmp_path / "bench.json"
    _write_json(bench_path, bench_results)

    metrics = kpi_gate.compute_metrics(bench_results_path=bench_path)
    assert metrics["averaged_pair_case_count"] == 4.0
    assert metrics["averaged_pair_fidelity_error"] is not None
    assert math.isclose(
        float(metrics["averaged_pair_fidelity_error"]),
        0.20,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert metrics["averaged_pair_runtime_speedup_min"] is not None
    assert math.isclose(
        float(metrics["averaged_pair_runtime_speedup_min"]),
        2.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert metrics["averaged_pair_runtime_speedup_mean"] is not None
    assert math.isclose(
        float(metrics["averaged_pair_runtime_speedup_mean"]),
        2.5,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_compute_metrics_includes_magnetic_core_metrics(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {
                "benchmark_id": "magnetic_core_saturation",
                "scenario": "direct_trap",
                "mode": "transient",
                "status": "passed",
                "runtime_s": 0.04,
                "telemetry": {
                    "magnetic_fixture_case": 1.0,
                    "magnetic_fixture_saturation": 1.0,
                    "magnetic_sat_error": 2.0e-7,
                    "output_reallocation_total": 0.0,
                },
            },
            {
                "benchmark_id": "magnetic_core_hysteresis",
                "scenario": "direct_trap",
                "mode": "transient",
                "status": "passed",
                "runtime_s": 0.06,
                "telemetry": {
                    "magnetic_fixture_case": 1.0,
                    "magnetic_fixture_hysteresis": 1.0,
                    "magnetic_hysteresis_cycle_energy_error": 4.0e-7,
                    "output_reallocation_total": 0.0,
                },
            },
            {
                "benchmark_id": "magnetic_core_frequency_trend_low",
                "scenario": "direct_trap",
                "mode": "transient",
                "status": "passed",
                "runtime_s": 0.05,
                "telemetry": {
                    "magnetic_fixture_case": 1.0,
                    "magnetic_fixture_frequency_trend": 1.0,
                    "magnetic_trend_group_crc32": 101.0,
                    "magnetic_trend_role_low": 1.0,
                    "magnetic_avg_core_loss": 0.5,
                    "output_reallocation_total": 0.0,
                },
            },
            {
                "benchmark_id": "magnetic_core_frequency_trend_high",
                "scenario": "direct_trap",
                "mode": "transient",
                "status": "passed",
                "runtime_s": 0.05,
                "telemetry": {
                    "magnetic_fixture_case": 1.0,
                    "magnetic_fixture_frequency_trend": 1.0,
                    "magnetic_trend_group_crc32": 101.0,
                    "magnetic_trend_role_high": 1.0,
                    "magnetic_avg_core_loss": 1.2,
                    "output_reallocation_total": 0.0,
                },
            },
            {
                "benchmark_id": "magnetic_core_determinism_cmp",
                "scenario": "direct_trap",
                "mode": "transient",
                "status": "passed",
                "runtime_s": 0.05,
                "max_error": 1.0e-12,
                "telemetry": {
                    "magnetic_fixture_case": 1.0,
                    "magnetic_determinism_case": 1.0,
                    "validation_max_error": 1.0e-12,
                    "output_reallocation_total": 0.0,
                },
            },
        ]
    }
    bench_path = tmp_path / "bench.json"
    _write_json(bench_path, bench_results)

    metrics = kpi_gate.compute_metrics(bench_results_path=bench_path)
    assert metrics["magnetic_sat_error"] == 2.0e-7
    assert metrics["magnetic_hysteresis_cycle_energy_error"] == 4.0e-7
    assert metrics["magnetic_core_loss_trend_error"] == 0.0
    assert metrics["magnetic_determinism_drift"] == 1.0e-12
    assert math.isclose(
        float(metrics["magnetic_runtime_p95"]),
        0.058,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert metrics["magnetic_allocation_regression"] == 0.0


def test_compute_metrics_runtime_quantiles_can_use_case_filter(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {"benchmark_id": "a", "scenario": "s0", "status": "passed", "runtime_s": 0.10},
            {"benchmark_id": "b", "scenario": "s0", "status": "passed", "runtime_s": 0.20},
            {"benchmark_id": "c", "scenario": "s0", "status": "passed", "runtime_s": 10.0},
        ]
    }
    bench_path = tmp_path / "bench.json"
    _write_json(bench_path, bench_results)

    metrics_all = kpi_gate.compute_metrics(bench_results_path=bench_path)
    metrics_filtered = kpi_gate.compute_metrics(
        bench_results_path=bench_path,
        runtime_case_filter={("a", "s0"), ("b", "s0")},
    )

    assert metrics_all["runtime_p95"] is not None
    assert metrics_filtered["runtime_p95"] is not None
    assert metrics_all["runtime_p95"] > 1.0
    assert metrics_filtered["runtime_p95"] < 0.25


def test_run_gate_uses_baseline_case_intersection_for_runtime(tmp_path: Path) -> None:
    source_root = tmp_path / "phase_artifacts"
    source_bench_results = source_root / "benchmarks" / "results.json"
    source_bench_results.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        source_bench_results,
        {
            "results": [
                {"benchmark_id": "a", "scenario": "s0", "status": "passed", "runtime_s": 0.11},
                {"benchmark_id": "b", "scenario": "s0", "status": "passed", "runtime_s": 0.18},
            ]
        },
    )

    bench_results = {
        "results": [
            {"benchmark_id": "a", "scenario": "s0", "status": "passed", "runtime_s": 0.12},
            {"benchmark_id": "b", "scenario": "s0", "status": "passed", "runtime_s": 0.19},
            {"benchmark_id": "x", "scenario": "new", "status": "passed", "runtime_s": 9.00},
        ]
    }
    baseline = {
        "runtime_p95": 0.25,
    }
    thresholds = {
        "metrics": {
            "runtime_p95": {
                "direction": "lower_is_better",
                "max_regression_rel": 0.10,
                "required": True,
            },
        }
    }

    bench_path = tmp_path / "bench.json"
    baseline_path = tmp_path / "kpi_baseline.json"
    thresholds_path = tmp_path / "thresholds.yaml"

    _write_json(bench_path, bench_results)
    _write_baseline_with_manifest(
        baseline_path=baseline_path,
        baseline_id="phase0",
        source_bench_results=source_bench_results,
        source_artifacts_root=source_root,
        metrics=baseline,
    )
    _write_yaml(thresholds_path, thresholds)

    report = kpi_gate.run_gate(
        baseline_path=baseline_path,
        thresholds_path=thresholds_path,
        bench_results_path=bench_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        stress_summary_path=None,
    )

    assert report["overall_status"] == "passed"
    assert report["runtime_scope"]["mode"] == "baseline_intersection"
    assert report["runtime_scope"]["comparable_total"] == 2


def test_kpi_gate_blocks_on_manifest_hash_mismatch(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {"status": "passed", "runtime_s": 0.10},
            {"status": "passed", "runtime_s": 0.12},
        ]
    }
    thresholds = {
        "metrics": {
            "runtime_p95": {
                "direction": "lower_is_better",
                "max_regression_rel": 0.10,
                "required": True,
            },
        }
    }

    bench_path = tmp_path / "bench.json"
    baseline_path = tmp_path / "kpi_baseline.json"
    thresholds_path = tmp_path / "thresholds.yaml"
    _write_json(bench_path, bench_results)
    _write_baseline_with_manifest(
        baseline_path=baseline_path,
        baseline_id="phase0",
        source_bench_results=bench_path,
        metrics={"runtime_p95": 0.20},
    )
    _write_yaml(thresholds_path, thresholds)

    manifest_path = baseline_path.with_name("artifact_manifest.json")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["files"][0]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest_payload)

    report = kpi_gate.run_gate(
        baseline_path=baseline_path,
        thresholds_path=thresholds_path,
        bench_results_path=bench_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        stress_summary_path=None,
    )

    assert report["overall_status"] == "failed"
    assert report["blocked_by_provenance"] is True
    assert report["comparisons"] == {}
    assert report["provenance"]["baseline"]["status"] == "failed"


def test_kpi_gate_can_continue_when_strict_provenance_is_disabled(tmp_path: Path) -> None:
    bench_results = {"results": [{"status": "passed", "runtime_s": 0.10}]}
    thresholds = {
        "metrics": {
            "runtime_p95": {
                "direction": "lower_is_better",
                "max_regression_rel": 0.10,
                "required": True,
            },
        }
    }
    bench_path = tmp_path / "bench.json"
    baseline_path = tmp_path / "kpi_baseline.json"
    thresholds_path = tmp_path / "thresholds.yaml"

    _write_json(bench_path, bench_results)
    _write_baseline_with_manifest(
        baseline_path=baseline_path,
        baseline_id="phase0",
        source_bench_results=bench_path,
        metrics={"runtime_p95": 0.20},
    )
    _write_yaml(thresholds_path, thresholds)

    # Corrupt manifest and run in compatibility mode.
    manifest_path = baseline_path.with_name("artifact_manifest.json")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["files"][0]["sha256"] = "f" * 64
    _write_json(manifest_path, manifest_payload)

    report = kpi_gate.run_gate(
        baseline_path=baseline_path,
        thresholds_path=thresholds_path,
        bench_results_path=bench_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        stress_summary_path=None,
        strict_provenance=False,
    )

    assert report["blocked_by_provenance"] is False
    assert report["comparisons"]["runtime_p95"]["status"] == "passed"


def test_kpi_gate_accepts_manifest_text_artifact_with_crlf_conversion(
    tmp_path: Path,
) -> None:
    bench_results = {"results": [{"status": "passed", "runtime_s": 0.10}]}
    thresholds = {
        "metrics": {
            "runtime_p95": {
                "direction": "lower_is_better",
                "max_regression_rel": 0.10,
                "required": True,
            },
        }
    }
    bench_path = tmp_path / "bench.json"
    baseline_path = tmp_path / "kpi_baseline.json"
    thresholds_path = tmp_path / "thresholds.yaml"

    _write_json(bench_path, bench_results)
    # Force LF for baseline digest capture so the later CRLF rewrite is deterministic.
    lf_text = bench_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    with open(bench_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(lf_text)

    _write_baseline_with_manifest(
        baseline_path=baseline_path,
        baseline_id="phase0",
        source_bench_results=bench_path,
        metrics={"runtime_p95": 0.20},
    )
    _write_yaml(thresholds_path, thresholds)

    # Simulate Windows checkout line-ending conversion without changing content.
    with open(bench_path, "w", encoding="utf-8", newline="\r\n") as handle:
        handle.write(lf_text)

    report = kpi_gate.run_gate(
        baseline_path=baseline_path,
        thresholds_path=thresholds_path,
        bench_results_path=bench_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        stress_summary_path=None,
    )

    assert report["overall_status"] == "passed"
    assert report["blocked_by_provenance"] is False
    assert report["provenance"]["baseline"]["status"] == "passed"
    assert any(
        "newline normalization matched" in warning
        for warning in report["provenance"]["baseline"]["warnings"]
    )
    assert report["provenance"]["baseline"]["checks"]["eol_normalization_matches"] == 1


def test_kpi_gate_skips_runtime_metrics_on_machine_class_mismatch(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {"benchmark_id": "a", "scenario": "s0", "status": "passed", "runtime_s": 0.80},
            {"benchmark_id": "b", "scenario": "s0", "status": "passed", "runtime_s": 0.90},
        ]
    }
    thresholds = {
        "metrics": {
            "runtime_p95": {
                "direction": "lower_is_better",
                "max_regression_rel": 0.05,
                "required": True,
            },
        }
    }
    bench_path = tmp_path / "bench.json"
    baseline_path = tmp_path / "kpi_baseline.json"
    thresholds_path = tmp_path / "thresholds.yaml"

    _write_json(bench_path, bench_results)
    _write_baseline_with_manifest(
        baseline_path=baseline_path,
        baseline_id="phase0",
        source_bench_results=bench_path,
        metrics={"runtime_p95": 0.10},
        machine_class="mismatch-machine-class",
    )
    _write_yaml(thresholds_path, thresholds)

    report = kpi_gate.run_gate(
        baseline_path=baseline_path,
        thresholds_path=thresholds_path,
        bench_results_path=bench_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        stress_summary_path=None,
    )

    assert report["overall_status"] == "passed"
    assert report["comparisons"]["runtime_p95"]["status"] == "skipped"
    assert "machine_class mismatch" in report["comparisons"]["runtime_p95"]["message"]


def test_run_gate_merges_phase_budget_metrics(tmp_path: Path) -> None:
    bench_results = {"results": [{"status": "passed", "runtime_s": 0.30}]}
    baseline_metrics = {"runtime_p95": 0.20}
    thresholds = {
        "metrics": {
            "runtime_p95": {
                "direction": "lower_is_better",
                "max_regression_rel": 1.0,
                "required": False,
            },
        }
    }
    phase_budget = {
        "schema": "pulsim-convergence-phase-budgets-v1",
        "version": 1,
        "phases": {
            "gate_b": {
                "metrics": {
                    "runtime_p95": {
                        "direction": "lower_is_better",
                        "max_regression_rel": 0.05,
                        "required": True,
                    },
                }
            }
        },
    }

    bench_path = tmp_path / "bench.json"
    baseline_path = tmp_path / "kpi_baseline.json"
    thresholds_path = tmp_path / "thresholds.yaml"
    phase_budget_path = tmp_path / "phase_budget.yaml"
    _write_json(bench_path, bench_results)
    _write_baseline_with_manifest(
        baseline_path=baseline_path,
        baseline_id="phase_budget_test",
        source_bench_results=bench_path,
        metrics=baseline_metrics,
    )
    _write_yaml(thresholds_path, thresholds)
    _write_yaml(phase_budget_path, phase_budget)

    report_without_budget = kpi_gate.run_gate(
        baseline_path=baseline_path,
        thresholds_path=thresholds_path,
        bench_results_path=bench_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        stress_summary_path=None,
    )
    assert report_without_budget["overall_status"] == "passed"

    report_with_budget = kpi_gate.run_gate(
        baseline_path=baseline_path,
        thresholds_path=thresholds_path,
        bench_results_path=bench_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        stress_summary_path=None,
        phase_budget_path=phase_budget_path,
        phase_budget_key="gate_b",
    )
    assert report_with_budget["overall_status"] == "failed"
    assert report_with_budget["phase_budget_key"] == "gate_b"
    assert report_with_budget["phase_budget_path"] == str(phase_budget_path)
    assert report_with_budget["comparisons"]["runtime_p95"]["required"] is True


def test_run_gate_phase_budget_requires_path_and_key(tmp_path: Path) -> None:
    bench_results = {"results": [{"status": "passed", "runtime_s": 0.10}]}
    baseline_metrics = {"runtime_p95": 0.20}
    thresholds = {"metrics": {}}

    bench_path = tmp_path / "bench.json"
    baseline_path = tmp_path / "kpi_baseline.json"
    thresholds_path = tmp_path / "thresholds.yaml"
    _write_json(bench_path, bench_results)
    _write_baseline_with_manifest(
        baseline_path=baseline_path,
        baseline_id="phase_budget_missing_key",
        source_bench_results=bench_path,
        metrics=baseline_metrics,
    )
    _write_yaml(thresholds_path, thresholds)

    with pytest.raises(RuntimeError, match="phase budget requires both"):
        kpi_gate.run_gate(
            baseline_path=baseline_path,
            thresholds_path=thresholds_path,
            bench_results_path=bench_path,
            parity_ltspice_results_path=None,
            parity_ngspice_results_path=None,
            stress_summary_path=None,
            phase_budget_path=tmp_path / "missing.yaml",
            phase_budget_key=None,
        )


def test_gate_b_phase_budget_blocks_cross_class_regression(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {
                "benchmark_id": "buck_switching",
                "scenario": "direct_trap",
                "status": "passed",
                "runtime_s": 0.20,
                "telemetry": {
                    "policy_dry_run_events": 4.0,
                    "policy_recommendation_matches": 1.0,
                    "policy_recommendation_mismatches": 3.0,
                    "anti_overfit_violations": 0.0,
                    "anti_overfit_budget_exceeded": 0.0,
                    "classified_fallback_events": 2.0,
                },
            },
            {
                "benchmark_id": "diode_rectifier",
                "scenario": "direct_trap",
                "status": "passed",
                "runtime_s": 0.10,
                "telemetry": {
                    "policy_dry_run_events": 2.0,
                    "policy_recommendation_matches": 0.0,
                    "policy_recommendation_mismatches": 2.0,
                    "anti_overfit_violations": 1.0,
                    "anti_overfit_budget_exceeded": 1.0,
                    "classified_fallback_events": 1.0,
                },
            },
        ]
    }
    class_matrix = {
        "schema": "pulsim-convergence-class-matrix-v1",
        "version": 1,
        "classes": {
            "switch_heavy": {
                "cases": [
                    {"benchmark_id": "buck_switching", "scenarios": ["direct_trap"]},
                ]
            },
            "diode_heavy": {
                "cases": [
                    {"benchmark_id": "diode_rectifier", "scenarios": ["direct_trap"]},
                ]
            },
        },
    }
    baseline_metrics = {
        "policy_target_pass_rate": 1.0,
        "policy_target_match_rate": 0.9,
        "policy_target_mismatch_rate": 0.1,
        "policy_stable_pass_rate": 1.0,
        "policy_stable_mismatch_rate": 0.0,
        "policy_stable_anti_overfit_violation_rate": 0.0,
    }
    thresholds = {"metrics": {}}
    phase_budget = {
        "schema": "pulsim-convergence-phase-budgets-v1",
        "version": 1,
        "phases": {
            "gate_b": {
                "metrics": {
                    "policy_target_pass_rate": {
                        "direction": "higher_is_better",
                        "max_regression_abs": 0.0,
                        "required": True,
                    },
                    "policy_target_match_rate": {
                        "direction": "higher_is_better",
                        "max_regression_abs": 0.0,
                        "required": True,
                    },
                    "policy_target_mismatch_rate": {
                        "direction": "lower_is_better",
                        "max_regression_abs": 0.0,
                        "required": True,
                    },
                    "policy_stable_pass_rate": {
                        "direction": "higher_is_better",
                        "max_regression_abs": 0.0,
                        "required": True,
                    },
                    "policy_stable_mismatch_rate": {
                        "direction": "lower_is_better",
                        "max_regression_abs": 0.0,
                        "required": True,
                    },
                    "policy_stable_anti_overfit_violation_rate": {
                        "direction": "lower_is_better",
                        "max_regression_abs": 0.0,
                        "required": True,
                    },
                }
            }
        },
    }

    bench_path = tmp_path / "bench.json"
    baseline_path = tmp_path / "kpi_baseline.json"
    thresholds_path = tmp_path / "thresholds.yaml"
    class_matrix_path = tmp_path / "class_matrix.yaml"
    phase_budget_path = tmp_path / "phase_budget.yaml"
    _write_json(bench_path, bench_results)
    _write_baseline_with_manifest(
        baseline_path=baseline_path,
        baseline_id="phase_gate_b_regression",
        source_bench_results=bench_path,
        metrics=baseline_metrics,
    )
    _write_yaml(thresholds_path, thresholds)
    _write_yaml(class_matrix_path, class_matrix)
    _write_yaml(phase_budget_path, phase_budget)

    report = kpi_gate.run_gate(
        baseline_path=baseline_path,
        thresholds_path=thresholds_path,
        bench_results_path=bench_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        stress_summary_path=None,
        class_matrix_path=class_matrix_path,
        phase_budget_path=phase_budget_path,
        phase_budget_key="gate_b",
    )

    assert report["overall_status"] == "failed"
    assert report["failed_required_metrics"] >= 1
    assert report["comparisons"]["policy_stable_mismatch_rate"]["status"] == "failed"
    assert report["comparisons"]["policy_stable_anti_overfit_violation_rate"]["status"] == "failed"


def test_compare_metric_uses_epsilon_guard_for_near_zero_baseline() -> None:
    cmp = kpi_gate.compare_metric(
        name="loss_energy_balance_error",
        current_value=3.4399697850963737e-17,
        baseline_value=3.271133600931126e-17,
        rules={
            "direction": "lower_is_better",
            "max_regression_rel": 0.05,
            "required": True,
        },
    )

    assert cmp.status == "passed"
    assert cmp.regression_abs is not None
    assert cmp.regression_rel == 0.0
