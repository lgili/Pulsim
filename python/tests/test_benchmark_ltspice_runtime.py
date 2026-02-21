"""Tests for LTspice parity backend and unified comparator artifacts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

import benchmark_ngspice as ng


def _write_minimal_manifest(tmp_path: Path) -> Path:
    netlist = {
        "schema": "pulsim-v1",
        "version": 1,
        "benchmark": {"id": "lt_missing_path", "validation": {"type": "ltspice"}},
        "simulation": {"tstop": 1e-4, "dt": 1e-6},
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
    (tmp_path / "circuit.yaml").write_text(yaml.safe_dump(netlist, sort_keys=False), encoding="utf-8")

    manifest = {
        "benchmarks": [
            {
                "path": "circuit.yaml",
                "ltspice_netlist": "ltspice/rc_step.cir",
                "ltspice_observables": [{"column": "V(out)", "ltspice_vector": "V(out)"}],
                "scenarios": ["default"],
            }
        ],
        "scenarios": {"default": {}},
    }
    manifest_path = tmp_path / "benchmarks.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def test_ltspice_requires_explicit_path_and_records_failure(tmp_path: Path) -> None:
    manifest_path = _write_minimal_manifest(tmp_path)

    results = ng.run_manifest(
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        only=None,
        matrix=False,
        force_scenario=None,
        cli_observables=None,
        backend="ltspice",
        ltspice_executable=None,
    )

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].failure_reason == "configuration_error"
    assert "executable path" in results[0].message.lower()

    ng.write_results(tmp_path / "out", results, backend="ltspice", executable=None)
    summary = json.loads((tmp_path / "out" / "parity_summary.json").read_text(encoding="utf-8"))
    assert summary["failed"] == 1
    assert summary["failure_reasons"]["configuration_error"] == 1


def test_resolve_observables_accepts_ltspice_vector_mapping() -> None:
    specs = ng.resolve_observables(
        benchmark_meta={},
        validation_meta={},
        entry_observables=[{"column": "V(out)", "ltspice_vector": "V(out)"}],
        cli_observables=None,
        backend="ltspice",
    )
    assert len(specs) == 1
    assert specs[0].column == "V(out)"
    assert specs[0].spice_vector == "V(out)"


def test_unified_metrics_compute_phase_and_steady_state() -> None:
    freq = 10_000.0
    period = 1.0 / freq
    phase_shift_deg = 30.0
    phase_shift_rad = math.radians(phase_shift_deg)

    times = [idx * (period / 200.0) for idx in range(2000)]
    ref_values = [math.sin(2.0 * math.pi * freq * t) for t in times]
    pulsim_values = [math.sin(2.0 * math.pi * freq * t + phase_shift_rad) for t in times]

    phase_error = ng.compute_phase_error_deg(times, pulsim_values, ref_values, period_hint=period)
    steady_max, steady_rms = ng.compute_steady_state_errors(times, pulsim_values, ref_values, period_hint=period)

    assert phase_error is not None
    assert abs(phase_error - phase_shift_deg) < 7.0
    assert steady_max is not None and steady_max > 0.0
    assert steady_rms is not None and steady_rms > 0.0


def test_write_results_emits_machine_readable_parity_artifacts(tmp_path: Path) -> None:
    result = ng.BenchmarkResult(
        benchmark_id="rc_step",
        scenario="default",
        status="passed",
        message="",
        pulsim_runtime_s=0.01,
        ngspice_runtime_s=0.03,
        pulsim_steps=100,
        ngspice_steps=95,
        speedup=3.0,
        max_error=1e-3,
        rms_error=5e-4,
        observables=[
            ng.ObservableMetrics(
                column="V(out)",
                spice_vector="V(out)",
                samples=100,
                max_error=1e-3,
                rms_error=5e-4,
                phase_error_deg=2.0,
                steady_state_max_error=8e-4,
                steady_state_rms_error=4e-4,
            )
        ],
        backend="ltspice",
        phase_error_deg=2.0,
        steady_state_max_error=8e-4,
        steady_state_rms_error=4e-4,
        reference_runtime_s=0.03,
        reference_steps=95,
    )

    ng.write_results(
        output_dir=tmp_path / "out",
        results=[result],
        backend="ltspice",
        executable=Path("/tmp/fake-ltspice"),
    )

    parity_json = tmp_path / "out" / "parity_results.json"
    parity_csv = tmp_path / "out" / "parity_results.csv"
    parity_summary = tmp_path / "out" / "parity_summary.json"
    assert parity_json.exists()
    assert parity_csv.exists()
    assert parity_summary.exists()

    payload = json.loads(parity_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "pulsim-parity-v1"
    assert payload["backend"] == "ltspice"
    assert payload["results"][0]["phase_error_deg"] == 2.0
