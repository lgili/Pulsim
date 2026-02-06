"""Tests for Python-first benchmark execution (no CLI dependency)."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

import benchmark_runner as br
import pulsim_python_backend as backend


def _write_manifest(tmp_path: Path, netlist_name: str) -> Path:
    manifest = {
        "benchmarks": [{"path": netlist_name, "scenarios": ["default"]}],
        "scenarios": {"default": {}},
    }
    manifest_path = tmp_path / "benchmarks.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def _write_rc_netlist(tmp_path: Path, benchmark_block: dict, filename: str = "circuit.yaml") -> Path:
    netlist = {
        "schema": "pulsim-v1",
        "version": 1,
        "benchmark": benchmark_block,
        "simulation": {
            "tstart": 0.0,
            "tstop": 5e-4,
            "dt": 1e-6,
        },
        "components": [
            {
                "type": "voltage_source",
                "name": "V1",
                "nodes": ["in", "0"],
                "waveform": {"type": "dc", "value": 5.0},
            },
            {"type": "resistor", "name": "R1", "nodes": ["in", "out"], "value": "1k"},
            {"type": "capacitor", "name": "C1", "nodes": ["out", "0"], "value": "1u", "ic": 0.0},
        ],
    }
    netlist_path = tmp_path / filename
    netlist_path.write_text(yaml.safe_dump(netlist, sort_keys=False), encoding="utf-8")
    return netlist_path


def _write_periodic_netlist(tmp_path: Path) -> Path:
    netlist = {
        "schema": "pulsim-v1",
        "version": 1,
        "simulation": {
            "tstart": 0.0,
            "tstop": 1e-3,
            "dt": 1e-5,
            "adaptive_timestep": False,
            "shooting": {
                "period": 1e-3,
                "max_iterations": 8,
                "tolerance": 1e-6,
                "relaxation": 0.5,
                "store_last_transient": True,
            },
            "harmonic_balance": {
                "period": 1e-3,
                "num_samples": 32,
                "max_iterations": 10,
                "tolerance": 1e-6,
                "relaxation": 0.8,
                "initialize_from_transient": True,
            },
        },
        "components": [
            {
                "type": "voltage_source",
                "name": "V1",
                "nodes": ["in", "0"],
                "waveform": {"type": "dc", "value": 5.0},
            },
            {"type": "resistor", "name": "R1", "nodes": ["in", "out"], "value": "1k"},
            {"type": "capacitor", "name": "C1", "nodes": ["out", "0"], "value": "1u", "ic": 0.0},
        ],
    }
    netlist_path = tmp_path / "periodic.yaml"
    netlist_path.write_text(yaml.safe_dump(netlist, sort_keys=False), encoding="utf-8")
    return netlist_path


def test_run_benchmarks_uses_python_backend_without_cli(tmp_path: Path) -> None:
    _write_rc_netlist(
        tmp_path,
        benchmark_block={
            "id": "smoke_python_backend",
            "validation": {"type": "none"},
        },
    )
    manifest_path = _write_manifest(tmp_path, "circuit.yaml")

    results = br.run_benchmarks(manifest_path, tmp_path / "out")

    assert len(results) == 1
    result = results[0]
    assert result.status == "passed"
    assert result.steps > 0
    assert result.telemetry.get("python_backend") == 1.0


def test_reference_validation_missing_baseline_is_failed_not_skipped(tmp_path: Path) -> None:
    _write_rc_netlist(
        tmp_path,
        benchmark_block={
            "id": "missing_baseline",
            "validation": {
                "type": "reference",
                "observable": "V(out)",
                "baseline": "baselines/does_not_exist.csv",
            },
        },
    )
    manifest_path = _write_manifest(tmp_path, "circuit.yaml")

    results = br.run_benchmarks(manifest_path, tmp_path / "out")

    assert len(results) == 1
    result = results[0]
    assert result.status == "failed"
    assert "Baseline missing" in result.message


def test_python_backend_runs_shooting_and_harmonic_balance(tmp_path: Path) -> None:
    netlist_path = _write_periodic_netlist(tmp_path)

    shooting_out = tmp_path / "shooting.csv"
    hb_out = tmp_path / "hb.csv"

    shooting = backend.run_from_yaml(netlist_path, shooting_out, preferred_mode="shooting")
    harmonic_balance = backend.run_from_yaml(netlist_path, hb_out, preferred_mode="harmonic_balance")

    assert shooting.mode == "shooting"
    assert shooting.steps > 0
    assert shooting_out.exists()
    assert shooting.telemetry.get("periodic_iterations", 0.0) >= 1.0

    assert harmonic_balance.mode == "harmonic_balance"
    assert harmonic_balance.steps > 0
    assert hb_out.exists()
    assert harmonic_balance.telemetry.get("harmonic_balance_iterations", 0.0) >= 1.0


def test_runner_applies_default_fixed_timestep_for_unset_adaptive(tmp_path: Path) -> None:
    netlist = {
        "schema": "pulsim-v1",
        "version": 1,
        "benchmark": {
            "id": "pulse_default_fixed_step",
            "validation": {"type": "none"},
        },
        "simulation": {
            "tstart": 0.0,
            "tstop": 2e-4,
            "dt": 1e-6,
            # intentionally omit adaptive_timestep to test runtime default
        },
        "components": [
            {
                "type": "voltage_source",
                "name": "Vpulse",
                "nodes": ["in", "0"],
                "waveform": {
                    "type": "pulse",
                    "v_initial": 0.0,
                    "v_pulse": 5.0,
                    "t_delay": 0.0,
                    "t_rise": 1e-9,
                    "t_fall": 1e-9,
                    "t_width": 1e-3,
                    "period": 2e-3,
                },
            },
            {"type": "resistor", "name": "R1", "nodes": ["in", "out"], "value": "1k"},
            {"type": "capacitor", "name": "C1", "nodes": ["out", "0"], "value": "1u", "ic": 0.0},
        ],
    }
    (tmp_path / "pulse.yaml").write_text(yaml.safe_dump(netlist, sort_keys=False), encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, "pulse.yaml")

    results = br.run_benchmarks(manifest_path, tmp_path / "out")

    assert len(results) == 1
    assert results[0].status == "passed"


def test_runner_retries_trbdf2_with_rosenbrock_fallback(tmp_path: Path) -> None:
    netlist = {
        "schema": "pulsim-v1",
        "version": 1,
        "benchmark": {
            "id": "trbdf2_fallback",
            "validation": {"type": "none"},
        },
        "simulation": {
            "tstart": 0.0,
            "tstop": 5e-4,
            "dt": 1e-6,
            "uic": True,
            "integrator": "trbdf2",
            "solver": {"order": ["gmres"]},
        },
        "components": [
            {
                "type": "voltage_source",
                "name": "Vpulse",
                "nodes": ["in", "0"],
                "waveform": {
                    "type": "pulse",
                    "v_initial": 0.0,
                    "v_pulse": 5.0,
                    "t_delay": 0.0,
                    "t_rise": 1e-9,
                    "t_fall": 1e-9,
                    "t_width": 1e-3,
                    "period": 2e-3,
                },
            },
            {"type": "resistor", "name": "R1", "nodes": ["in", "out"], "value": "1k"},
            {"type": "capacitor", "name": "C1", "nodes": ["out", "0"], "value": "1u", "ic": 0.0},
        ],
    }
    (tmp_path / "trbdf2.yaml").write_text(yaml.safe_dump(netlist, sort_keys=False), encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, "trbdf2.yaml")

    results = br.run_benchmarks(manifest_path, tmp_path / "out")

    assert len(results) == 1
    assert results[0].status == "passed"
    assert results[0].telemetry.get("integrator_mapped_to_rosenbrockw") == 1.0


def test_shooting_uses_warm_start_retry_for_pwm_case(tmp_path: Path) -> None:
    netlist = {
        "schema": "pulsim-v1",
        "version": 1,
        "simulation": {
            "tstart": 0.0,
            "tstop": 200e-6,
            "dt": 0.5e-6,
            "adaptive_timestep": False,
            "shooting": {
                "period": 20e-6,
                "max_iterations": 20,
                "tolerance": 1e-6,
                "relaxation": 0.5,
                "store_last_transient": True,
            },
        },
        "components": [
            {
                "type": "voltage_source",
                "name": "Vpwm",
                "nodes": ["in", "0"],
                "waveform": {
                    "type": "pwm",
                    "v_high": 5.0,
                    "v_low": 0.0,
                    "frequency": 50000,
                    "duty": 0.5,
                    "dead_time": 200e-9,
                },
            },
            {"type": "resistor", "name": "R1", "nodes": ["in", "out"], "value": 100},
            {"type": "capacitor", "name": "C1", "nodes": ["out", "0"], "value": "1u"},
        ],
    }
    netlist_path = tmp_path / "shooting_pwm.yaml"
    netlist_path.write_text(yaml.safe_dump(netlist, sort_keys=False), encoding="utf-8")

    result = backend.run_from_yaml(netlist_path, tmp_path / "shooting.csv", preferred_mode="shooting")

    assert result.mode == "shooting"
    assert result.steps > 0
    assert result.telemetry.get("shooting_warm_start_retry", 0.0) >= 1.0
