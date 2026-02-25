"""Tests for benchmark KPI regression gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
) -> None:
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
            "machine_class": "ci",
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
                },
            },
        ]
    }
    bench_path = tmp_path / "bench.json"
    _write_json(bench_path, bench_results)

    metrics = kpi_gate.compute_metrics(bench_results_path=bench_path)
    assert metrics["state_space_primary_ratio"] == 14.0 / 20.0
    assert metrics["dae_fallback_ratio"] == 6.0 / 20.0
    assert metrics["loss_energy_balance_error"] == 0.01
    assert metrics["thermal_peak_temperature_delta"] == 20.0


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
