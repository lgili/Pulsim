"""Tests for ngspice comparator integration with Python runtime runner."""

from __future__ import annotations

from pathlib import Path

import yaml

import benchmark_ngspice as ng
from benchmark_runner import PulsimRunResult


def test_run_case_uses_python_runtime_path(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_run_pulsim(
        netlist_path: Path,
        output_path: Path,
        preferred_mode: str | None = None,
        use_initial_conditions: bool = False,
    ) -> PulsimRunResult:
        calls["netlist_path"] = netlist_path
        calls["preferred_mode"] = preferred_mode
        calls["use_initial_conditions"] = use_initial_conditions
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("time,V(out)\n0.0,0.0\n1e-6,1.0\n", encoding="utf-8")
        return PulsimRunResult(
            runtime_s=0.01,
            steps=1,
            mode=preferred_mode or "transient",
            telemetry={"python_backend": 1.0},
        )

    def fake_compare(
        pulsim_csv_path: Path,
        spice_netlist_path: Path,
        observable_specs: list[ng.ObservableSpec],
    ) -> tuple[float, int, list[ng.ObservableMetrics], float, float]:
        calls["pulsim_csv_path"] = pulsim_csv_path
        calls["spice_netlist_path"] = spice_netlist_path
        calls["observable_specs"] = observable_specs
        return (
            0.02,
            2,
            [
                ng.ObservableMetrics(
                    column="V(out)",
                    spice_vector="v(out)",
                    samples=2,
                    max_error=1e-4,
                    rms_error=5e-5,
                )
            ],
            1e-4,
            5e-5,
        )

    monkeypatch.setattr(ng, "run_pulsim", fake_run_pulsim)
    monkeypatch.setattr(ng, "compare_pulsim_vs_ngspice", fake_compare)

    scenario_netlist = {
        "schema": "pulsim-v1",
        "version": 1,
        "simulation": {"tstop": 1e-3, "dt": 1e-6},
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

    spice_path = tmp_path / "dummy.cir"
    spice_path.write_text("* dummy spice", encoding="utf-8")

    result = ng.run_case(
        scenario_netlist=scenario_netlist,
        benchmark_id="ng_runtime_smoke",
        scenario_name="harmonic_balance",
        spice_netlist_path=spice_path,
        observable_specs=[ng.ObservableSpec(column="V(out)", spice_vector="v(out)")],
        max_error_threshold=1e-3,
        output_dir=tmp_path / "out",
        preferred_mode="harmonic_balance",
    )

    assert calls["preferred_mode"] == "harmonic_balance"
    assert isinstance(calls["pulsim_csv_path"], Path)
    assert result.status == "passed"
    assert result.pulsim_steps == 1
    assert result.ngspice_steps == 2


def test_run_manifest_selects_preferred_periodic_mode(tmp_path: Path, monkeypatch) -> None:
    captured_modes: list[str | None] = []

    def fake_run_case(
        scenario_netlist: dict,
        benchmark_id: str,
        scenario_name: str,
        spice_netlist_path: Path,
        observable_specs: list[ng.ObservableSpec],
        max_error_threshold: float | None,
        output_dir: Path,
        preferred_mode: str | None,
        backend_config: ng.BackendConfig | None = None,
        period_hint: float | None = None,
        rms_error_threshold: float | None = None,
        phase_error_threshold: float | None = None,
        steady_state_max_threshold: float | None = None,
        steady_state_rms_threshold: float | None = None,
    ) -> ng.BenchmarkResult:
        captured_modes.append(preferred_mode)
        return ng.BenchmarkResult(
            benchmark_id=benchmark_id,
            scenario=scenario_name,
            status="passed",
            message="",
            pulsim_runtime_s=0.01,
            ngspice_runtime_s=0.02,
            pulsim_steps=1,
            ngspice_steps=2,
            speedup=2.0,
            max_error=1e-4,
            rms_error=5e-5,
            observables=[],
        )

    monkeypatch.setattr(ng, "run_case", fake_run_case)

    periodic_netlist = {
        "schema": "pulsim-v1",
        "version": 1,
        "benchmark": {
            "id": "periodic_case",
            "validation": {
                "type": "reference",
                "observable": "V(out)",
                "spice_netlist": "dummy.cir",
            },
        },
        "simulation": {
            "shooting": {"period": 1e-3},
            "harmonic_balance": {"period": 1e-3, "num_samples": 16},
        },
        "components": [
            {"type": "resistor", "name": "R1", "nodes": ["out", "0"], "value": "1k"},
            {
                "type": "voltage_source",
                "name": "V1",
                "nodes": ["out", "0"],
                "waveform": {"type": "dc", "value": 1.0},
            },
        ],
    }

    manifest = {
        "benchmarks": [
            {
                "path": "periodic.yaml",
                "ngspice_netlist": "dummy.cir",
                "scenarios": ["shooting_default", "harmonic_balance"],
            }
        ],
        "scenarios": {
            "shooting_default": {"simulation": {"shooting": {"period": 1e-3}}},
            "harmonic_balance": {"simulation": {"harmonic_balance": {"period": 1e-3, "num_samples": 16}}},
        },
    }

    manifest_path = tmp_path / "benchmarks.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    (tmp_path / "periodic.yaml").write_text(yaml.safe_dump(periodic_netlist, sort_keys=False), encoding="utf-8")
    (tmp_path / "dummy.cir").write_text("* dummy", encoding="utf-8")

    results = ng.run_manifest(
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        only=None,
        matrix=False,
        force_scenario=None,
        cli_observables=["V(out)"],
    )

    assert len(results) == 2
    assert captured_modes == ["shooting", "harmonic_balance"]
