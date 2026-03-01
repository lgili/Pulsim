"""Tests for KPI baseline freezing utility."""

from __future__ import annotations

import json
from pathlib import Path

import freeze_kpi_baseline


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_freeze_baseline_writes_baseline_and_manifest(tmp_path: Path) -> None:
    bench_results = {
        "results": [
            {"status": "passed", "runtime_s": 0.10},
            {"status": "passed", "runtime_s": 0.15},
            {"status": "failed", "runtime_s": 0.20},
        ]
    }
    stress_summary = {"tiers_failed": 0}

    bench_path = tmp_path / "results.json"
    stress_path = tmp_path / "stress_summary.json"
    out_dir = tmp_path / "baseline_out"
    _write_json(bench_path, bench_results)
    _write_json(stress_path, stress_summary)

    baseline_path, manifest_path = freeze_kpi_baseline.freeze_baseline(
        baseline_id="phaseX_test",
        output_dir=out_dir,
        bench_results_path=bench_path,
        stress_summary_path=stress_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        source_artifacts_root=tmp_path,
        machine_class_override="ci-test",
        cxx_flags_override="-O3",
        overwrite=True,
    )

    assert baseline_path.exists()
    assert manifest_path.exists()

    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert baseline_payload["schema_version"] == "pulsim-kpi-baseline-v1"
    assert baseline_payload["baseline_id"] == "phaseX_test"
    assert baseline_payload["environment"]["machine_class"] == "ci-test"
    assert baseline_payload["environment"]["cxx_flags"] == "-O3"
    assert baseline_payload["metrics"]["convergence_success_rate"] == 2.0 / 3.0
    assert baseline_payload["metrics"]["runtime_p95"] is not None
    assert baseline_payload["metrics"]["stress_tier_failure_count"] == 0.0

    assert manifest_payload["schema_version"] == "pulsim-kpi-baseline-manifest-v1"
    assert manifest_payload["baseline_id"] == "phaseX_test"
    assert len(manifest_payload["files"]) == 2
