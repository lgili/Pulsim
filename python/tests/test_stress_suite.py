"""Tests for tiered stress suite orchestration and criteria evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

import benchmark_runner as br
import stress_suite as ss


def _result(
    benchmark_id: str,
    scenario: str,
    status: str = "passed",
    runtime_s: float = 0.2,
    max_error: float | None = 1e-4,
    timestep_rejections: float = 0.0,
    missing_telemetry: bool = False,
) -> br.ScenarioResult:
    telemetry = {
        "runtime_s": runtime_s,
        "steps": 10.0,
        "newton_iterations": 20.0,
        "timestep_rejections": timestep_rejections,
        "linear_fallbacks": 0.0,
    }
    if missing_telemetry:
        telemetry.pop("newton_iterations")

    return br.ScenarioResult(
        benchmark_id=benchmark_id,
        scenario=scenario,
        status=status,
        runtime_s=runtime_s,
        steps=10,
        max_error=max_error,
        rms_error=max_error,
        message="",
        telemetry=telemetry,
    )


def test_evaluate_tier_results_passes_when_all_criteria_hold() -> None:
    criteria = ss.TierCriteria(
        min_pass_rate=1.0,
        max_runtime_s=1.0,
        max_max_error=1e-2,
        max_timestep_rejections=5.0,
        required_telemetry=("runtime_s", "steps", "newton_iterations", "timestep_rejections"),
    )
    results = [
        _result("rc_step", "direct_trap"),
        _result("rl_step", "gmres_trbdf2", runtime_s=0.4, max_error=5e-4),
    ]

    evaluation = ss.evaluate_tier_results("tier_a", results, criteria)

    assert evaluation.status == "passed"
    assert evaluation.pass_rate == 1.0
    assert evaluation.missing_telemetry_rows == 0


def test_evaluate_tier_results_fails_on_missing_telemetry_and_rejections() -> None:
    criteria = ss.TierCriteria(
        min_pass_rate=1.0,
        max_runtime_s=1.0,
        max_max_error=1e-2,
        max_timestep_rejections=1.0,
        required_telemetry=("runtime_s", "newton_iterations"),
    )
    results = [
        _result("buck_switching", "direct_trap", timestep_rejections=3.0),
        _result("periodic_rc_pwm", "harmonic_balance", missing_telemetry=True),
    ]

    evaluation = ss.evaluate_tier_results("tier_b", results, criteria)

    assert evaluation.status == "failed"
    assert evaluation.max_timestep_rejections_observed == 3.0
    assert evaluation.missing_telemetry_rows == 1


def test_run_stress_suite_uses_tier_cases_and_generates_evaluations(
    tmp_path: Path, monkeypatch
) -> None:
    circuit = {
        "schema": "pulsim-v1",
        "version": 1,
        "benchmark": {"id": "rc_step", "validation": {"type": "none"}},
        "simulation": {"tstop": 1e-5, "dt": 1e-6},
        "components": [
            {
                "type": "voltage_source",
                "name": "V1",
                "nodes": ["in", "0"],
                "waveform": {"type": "dc", "value": 1.0},
            },
            {"type": "resistor", "name": "R1", "nodes": ["in", "out"], "value": "1k"},
            {"type": "capacitor", "name": "C1", "nodes": ["out", "0"], "value": "1u"},
        ],
    }
    (tmp_path / "circuit.yaml").write_text(yaml.safe_dump(circuit, sort_keys=False), encoding="utf-8")

    manifest = {
        "benchmarks": [{"path": "circuit.yaml", "scenarios": ["direct_trap"]}],
        "scenarios": {"direct_trap": {"simulation": {"integrator": "trapezoidal"}}},
    }
    manifest_path = tmp_path / "benchmarks.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    catalog = {
        "schema": "pulsim-stress-catalog-v1",
        "version": 1,
        "tiers": {
            "tier_a": {
                "description": "analytical",
                "criteria": {
                    "min_pass_rate": 1.0,
                    "required_telemetry": ["runtime_s", "steps", "newton_iterations", "timestep_rejections"],
                },
                "cases": [{"benchmark_id": "rc_step", "scenarios": ["direct_trap"]}],
            }
        },
    }
    catalog_path = tmp_path / "stress_catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

    def fake_run_benchmarks(
        benchmarks_path: Path,
        output_dir: Path,
        selected=None,
        matrix: bool = False,
        generate_baselines: bool = False,
    ):
        return [
            _result(
                benchmark_id="rc_step",
                scenario="direct_trap",
                status="passed",
                runtime_s=0.1,
                max_error=1e-4,
            )
        ]

    monkeypatch.setattr(ss, "run_benchmarks", fake_run_benchmarks)

    tier_runs = ss.run_stress_suite(
        benchmarks_manifest_path=manifest_path,
        stress_catalog_path=catalog_path,
        output_dir=tmp_path / "out",
        selected_tiers=None,
    )

    assert len(tier_runs) == 1
    assert tier_runs[0].tier == "tier_a"
    assert tier_runs[0].evaluation.status == "passed"
