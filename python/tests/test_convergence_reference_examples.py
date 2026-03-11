"""Tests for convergence reference examples catalog validation."""

from __future__ import annotations

from pathlib import Path

import yaml

import validate_reference_examples as vre


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_validate_reference_examples_passes_for_valid_catalog(tmp_path: Path) -> None:
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
    (tmp_path / "rc_step.yaml").write_text(
        yaml.safe_dump(circuit, sort_keys=False),
        encoding="utf-8",
    )

    manifest = {
        "benchmarks": [{"path": "rc_step.yaml", "scenarios": ["direct_trap"]}],
        "scenarios": {"direct_trap": {}},
    }
    manifest_path = tmp_path / "benchmarks.yaml"
    _write_yaml(manifest_path, manifest)

    examples = {
        "schema": "pulsim-convergence-reference-examples-v1",
        "version": 1,
        "classes": [
            {
                "failure_class": "event_burst_zero_cross",
                "examples": [
                    {
                        "benchmark_id": "rc_step",
                        "scenarios": ["direct_trap"],
                        "expected_kpi": {"status": "passed"},
                    }
                ],
            }
        ],
    }
    examples_path = tmp_path / "examples.yaml"
    _write_yaml(examples_path, examples)

    errors = vre.validate_examples(
        manifest_path=manifest_path,
        examples_path=examples_path,
    )
    assert errors == []


def test_validate_reference_examples_reports_unknown_benchmark_and_scenario(
    tmp_path: Path,
) -> None:
    circuit = {
        "schema": "pulsim-v1",
        "version": 1,
        "benchmark": {"id": "rc_step", "validation": {"type": "none"}},
        "simulation": {"tstop": 1e-5, "dt": 1e-6},
        "components": [],
    }
    (tmp_path / "rc_step.yaml").write_text(
        yaml.safe_dump(circuit, sort_keys=False),
        encoding="utf-8",
    )

    manifest = {
        "benchmarks": [{"path": "rc_step.yaml", "scenarios": ["direct_trap"]}],
        "scenarios": {"direct_trap": {}},
    }
    manifest_path = tmp_path / "benchmarks.yaml"
    _write_yaml(manifest_path, manifest)

    examples = {
        "schema": "pulsim-convergence-reference-examples-v1",
        "version": 1,
        "classes": [
            {
                "failure_class": "switch_chattering",
                "examples": [
                    {
                        "benchmark_id": "unknown_benchmark",
                        "scenarios": ["direct_trap"],
                        "expected_kpi": {"status": "passed"},
                    },
                    {
                        "benchmark_id": "rc_step",
                        "scenarios": ["missing_scenario"],
                        "expected_kpi": {"status": "passed"},
                    },
                ],
            }
        ],
    }
    examples_path = tmp_path / "examples.yaml"
    _write_yaml(examples_path, examples)

    errors = vre.validate_examples(
        manifest_path=manifest_path,
        examples_path=examples_path,
    )
    assert any("unknown benchmark_id" in item for item in errors)
    assert any("undefined scenario" in item for item in errors)
