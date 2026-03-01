"""Tests for local limit suite orchestration and output checks."""

from __future__ import annotations

from pathlib import Path

import yaml

import benchmark_runner as br
import local_limit_suite as lls


def test_materialize_manifest_scales_tstop_and_tracks_expectations(tmp_path: Path) -> None:
    circuit = {
        "schema": "pulsim-v1",
        "version": 1,
        "benchmark": {"id": "ll_smoke", "validation": {"type": "none"}},
        "simulation": {"tstart": 0.0, "tstop": 1e-3, "dt": 1e-6},
        "components": [
            {
                "type": "voltage_source",
                "name": "V1",
                "nodes": ["in", "0"],
                "waveform": {"type": "dc", "value": 5.0},
            },
            {"type": "resistor", "name": "R1", "nodes": ["in", "out"], "value": "1k"},
            {"type": "capacitor", "name": "C1", "nodes": ["out", "0"], "value": "1u"},
        ],
    }
    (tmp_path / "c1.yaml").write_text(yaml.safe_dump(circuit, sort_keys=False), encoding="utf-8")

    manifest = {
        "benchmarks": [{"path": "c1.yaml", "difficulty": "1-basic", "scenarios": ["fixed_long"]}],
        "scenarios": {"fixed_long": {"simulation": {"step_mode": "fixed", "adaptive_timestep": False}}},
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    materialized_path, expected_tstop, difficulty = lls._materialize_manifest(
        manifest_path,
        tmp_path / "work",
        duration_scale=3.0,
    )

    loaded_manifest = br.load_yaml(materialized_path)
    rel_path = loaded_manifest["benchmarks"][0]["path"]
    scaled_circuit = br.load_yaml((materialized_path.parent / rel_path).resolve())

    assert abs(float(scaled_circuit["simulation"]["tstop"]) - 3e-3) < 1e-15
    assert abs(float(expected_tstop[("ll_smoke", "fixed_long")]) - 3e-3) < 1e-15
    assert difficulty["ll_smoke"] == "1-basic"


def test_inspect_output_csv_rejects_non_finite_values(tmp_path: Path) -> None:
    output_csv = tmp_path / "out.csv"
    output_csv.write_text(
        "time,V(out)\n"
        "0.0,0.0\n"
        "1.0,inf\n",
        encoding="utf-8",
    )

    finite, samples, final_time, message = lls._inspect_output_csv(output_csv)

    assert not finite
    assert samples == 1
    assert final_time == 0.0
    assert "non-finite value" in message


def test_evaluate_results_applies_runtime_budget_failure(tmp_path: Path) -> None:
    output_csv = tmp_path / "outputs" / "ll01" / "fixed_long" / "pulsim.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text(
        "time,V(out)\n"
        "0.0,0.0\n"
        "1.0,1.0\n"
        "2.0,2.0\n",
        encoding="utf-8",
    )

    base_results = [
        br.ScenarioResult(
            benchmark_id="ll01",
            scenario="fixed_long",
            status="passed",
            runtime_s=2.0,
            steps=2,
            max_error=None,
            rms_error=None,
            message="",
            telemetry={"runtime_s": 2.0},
        )
    ]

    evaluated = lls._evaluate_results(
        base_results=base_results,
        output_dir=tmp_path,
        expected_tstop={("ll01", "fixed_long"): 2.0},
        difficulty_by_id={"ll01": "1-basic"},
        min_samples=2,
        min_completion=0.9,
        max_runtime_s=1.0,
    )

    assert len(evaluated) == 1
    assert evaluated[0].status == "failed"
    assert "exceeds limit" in evaluated[0].message
