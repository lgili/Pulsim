"""Tests for Python-first benchmark execution (no CLI dependency)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

import benchmark_runner as br
import kpi_gate
import pulsim_python_backend as backend


def _write_manifest(tmp_path: Path, netlist_name: str) -> Path:
    manifest = {
        "benchmarks": [{"path": netlist_name, "scenarios": ["default"]}],
        "scenarios": {"default": {}},
    }
    manifest_path = tmp_path / "benchmarks.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return manifest_path


def _write_rc_netlist(
    tmp_path: Path, benchmark_block: dict, filename: str = "circuit.yaml"
) -> Path:
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
            {
                "type": "capacitor",
                "name": "C1",
                "nodes": ["out", "0"],
                "value": "1u",
                "ic": 0.0,
            },
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
            {
                "type": "capacitor",
                "name": "C1",
                "nodes": ["out", "0"],
                "value": "1u",
                "ic": 0.0,
            },
        ],
    }
    netlist_path = tmp_path / "periodic.yaml"
    netlist_path.write_text(yaml.safe_dump(netlist, sort_keys=False), encoding="utf-8")
    return netlist_path


def _write_ac_rc_netlist(
    tmp_path: Path, benchmark_block: dict, filename: str = "ac_rc.yaml"
) -> Path:
    netlist = {
        "schema": "pulsim-v1",
        "version": 1,
        "benchmark": benchmark_block,
        "simulation": {
            "tstart": 0.0,
            "tstop": 1e-3,
            "dt": 1e-6,
            "frequency_analysis": {
                "enabled": True,
                "mode": "open_loop_transfer",
                "anchor": "dc",
                "sweep": {
                    "scale": "log",
                    "f_start_hz": 10.0,
                    "f_stop_hz": 100000.0,
                    "points": 80,
                },
                "injection_current_amplitude": 1.0,
                "perturbation": {"positive": "in", "negative": "0"},
                "output": {"positive": "out", "negative": "0"},
            },
        },
        "components": [
            {"type": "resistor", "name": "R1", "nodes": ["in", "out"], "value": "1k"},
            {
                "type": "capacitor",
                "name": "C1",
                "nodes": ["out", "0"],
                "value": "1u",
                "ic": 0.0,
            },
        ],
    }
    netlist_path = tmp_path / filename
    netlist_path.write_text(yaml.safe_dump(netlist, sort_keys=False), encoding="utf-8")
    return netlist_path


def _write_ac_control_expected_failure_netlist(
    tmp_path: Path,
    benchmark_block: dict,
    filename: str = "ac_control_fail.yaml",
) -> Path:
    netlist = {
        "schema": "pulsim-v1",
        "version": 1,
        "benchmark": benchmark_block,
        "simulation": {
            "tstart": 0.0,
            "tstop": 1e-3,
            "dt": 1e-6,
            "frequency_analysis": {
                "enabled": True,
                "mode": "open_loop_transfer",
                "anchor": "dc",
                "sweep": {
                    "scale": "log",
                    "f_start_hz": 10.0,
                    "f_stop_hz": 10000.0,
                    "points": 40,
                },
                "injection_current_amplitude": 1.0,
                "perturbation": {"positive": "in", "negative": "0"},
                "output": {"positive": "out", "negative": "0"},
            },
        },
        "components": [
            {
                "type": "voltage_source",
                "name": "Vin",
                "nodes": ["in", "0"],
                "waveform": {"type": "dc", "value": 12.0},
            },
            {"type": "resistor", "name": "R1", "nodes": ["in", "out"], "value": "1k"},
            {
                "type": "capacitor",
                "name": "C1",
                "nodes": ["out", "0"],
                "value": "1u",
                "ic": 0.0,
            },
            {
                "type": "voltage_source",
                "name": "Vref",
                "nodes": ["vref", "0"],
                "waveform": {"type": "dc", "value": 5.0},
            },
            {
                "type": "pi_controller",
                "name": "PI1",
                "nodes": ["vref", "out", "ctrl"],
                "kp": 0.08,
                "ki": 100.0,
                "output_min": 0.0,
                "output_max": 0.95,
                "anti_windup": 1.0,
            },
        ],
    }
    netlist_path = tmp_path / filename
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
    assert result.telemetry.get("component_declared_count") == 3.0
    assert result.telemetry.get("component_reported_count") == 3.0
    assert result.telemetry.get("component_coverage_rate") == 1.0
    assert result.telemetry.get("runtime_module_count", 0.0) >= 5.0
    assert result.telemetry.get("runtime_module_count_match") == 1.0
    assert result.telemetry.get("output_reallocation_total", 0.0) >= 0.0


def test_transient_telemetry_exports_convergence_policy_schema_fields() -> None:
    result = SimpleNamespace(
        newton_iterations_total=9,
        timestep_rejections=1,
        total_time_seconds=2.5e-4,
        total_steps=12,
        time=[0.0, 1.0e-6],
        backend_telemetry=SimpleNamespace(
            classified_fallback_events=7,
            policy_dry_run_events=5,
            policy_recommendation_matches=4,
            policy_recommendation_mismatches=1,
            anti_overfit_violations=2,
            anti_overfit_budget_exceeded=True,
        ),
    )

    telemetry = backend._transient_telemetry(result, runtime_s=3.0e-4)

    assert telemetry["classified_fallback_events"] == 7.0
    assert telemetry["policy_dry_run_events"] == 5.0
    assert telemetry["policy_recommendation_matches"] == 4.0
    assert telemetry["policy_recommendation_mismatches"] == 1.0
    assert telemetry["anti_overfit_violations"] == 2.0
    assert telemetry["anti_overfit_budget_exceeded"] == 1.0


def test_run_benchmarks_backfills_convergence_policy_fields_for_legacy_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_rc_netlist(
        tmp_path,
        benchmark_block={
            "id": "legacy_telemetry_backfill",
            "validation": {"type": "none"},
        },
    )
    manifest_path = _write_manifest(tmp_path, "circuit.yaml")

    def fake_run_pulsim(
        netlist_path: Path,
        output_path: Path,
        preferred_mode: str | None = None,
        use_initial_conditions: bool = False,
    ) -> br.PulsimRunResult:
        del netlist_path, preferred_mode, use_initial_conditions
        output_path.write_text("time,V(in),V(out)\n0.0,5.0,0.0\n1.0e-6,5.0,0.1\n", encoding="utf-8")
        return br.PulsimRunResult(
            runtime_s=2.0e-4,
            steps=2,
            mode="transient",
            telemetry={"newton_iterations": 3.0},
        )

    monkeypatch.setattr(br, "run_pulsim", fake_run_pulsim)

    results = br.run_benchmarks(manifest_path, tmp_path / "out")

    assert len(results) == 1
    row = results[0]
    assert row.status == "passed"
    for key in (
        "classified_fallback_events",
        "policy_dry_run_events",
        "policy_recommendation_matches",
        "policy_recommendation_mismatches",
        "anti_overfit_violations",
        "anti_overfit_budget_exceeded",
    ):
        assert row.telemetry.get(key) == 0.0


def test_reference_validation_missing_baseline_is_failed_not_skipped(
    tmp_path: Path,
) -> None:
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

    shooting = backend.run_from_yaml(
        netlist_path, shooting_out, preferred_mode="shooting"
    )
    harmonic_balance = backend.run_from_yaml(
        netlist_path, hb_out, preferred_mode="harmonic_balance"
    )

    assert shooting.mode == "shooting"
    assert shooting.steps > 0
    assert shooting_out.exists()
    assert shooting.telemetry.get("periodic_iterations", 0.0) >= 1.0

    assert harmonic_balance.mode == "harmonic_balance"
    assert harmonic_balance.steps > 0
    assert hb_out.exists()
    assert harmonic_balance.telemetry.get("harmonic_balance_iterations", 0.0) >= 1.0


def test_runner_applies_default_fixed_timestep_for_unset_adaptive(
    tmp_path: Path,
) -> None:
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
            {
                "type": "capacitor",
                "name": "C1",
                "nodes": ["out", "0"],
                "value": "1u",
                "ic": 0.0,
            },
        ],
    }
    (tmp_path / "pulse.yaml").write_text(
        yaml.safe_dump(netlist, sort_keys=False), encoding="utf-8"
    )
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
            {
                "type": "capacitor",
                "name": "C1",
                "nodes": ["out", "0"],
                "value": "1u",
                "ic": 0.0,
            },
        ],
    }
    (tmp_path / "trbdf2.yaml").write_text(
        yaml.safe_dump(netlist, sort_keys=False), encoding="utf-8"
    )
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

    result = backend.run_from_yaml(
        netlist_path, tmp_path / "shooting.csv", preferred_mode="shooting"
    )

    assert result.mode == "shooting"
    assert result.steps > 0
    assert result.telemetry.get("shooting_warm_start_retry", 0.0) >= 1.0


def test_periodic_benchmark_shooting_default_keeps_reference_error_bounded(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "benchmarks" / "benchmarks.yaml"

    results = br.run_benchmarks(
        manifest_path,
        tmp_path / "out",
        selected=["periodic_rc_pwm"],
        scenario_filter=["shooting_default"],
    )

    assert len(results) == 1
    result = results[0]
    assert result.status == "passed"
    assert result.max_error is not None
    assert result.max_error <= 0.11


def test_averaged_benchmark_pair_case_emits_pair_telemetry(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "benchmarks" / "benchmarks.yaml"

    results = br.run_benchmarks(
        manifest_path,
        tmp_path / "out",
        selected=["buck_switching_paired", "buck_averaged_mvp"],
        scenario_filter=["direct_trap"],
    )

    assert len(results) == 2
    result = next(item for item in results if item.benchmark_id == "buck_averaged_mvp")
    assert result.status == "passed"
    assert result.mode == "transient"
    assert result.max_error is not None
    assert result.max_error <= 10.0
    assert result.telemetry.get("averaged_pair_case") == 1.0
    assert result.telemetry.get("averaged_pair_role_averaged") == 1.0
    assert result.telemetry.get("averaged_pair_role_switching") == 0.0
    assert result.telemetry.get("averaged_pair_group_crc32") is not None
    assert result.telemetry.get("paired_reference_case") == 1.0


def test_averaged_benchmark_repeat_runs_are_deterministic(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "benchmarks" / "benchmarks.yaml"

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    br.run_benchmarks(
        manifest_path,
        out_a,
        selected=["buck_switching_paired", "buck_averaged_mvp"],
        scenario_filter=["direct_trap"],
    )
    br.run_benchmarks(
        manifest_path,
        out_b,
        selected=["buck_switching_paired", "buck_averaged_mvp"],
        scenario_filter=["direct_trap"],
    )

    csv_a = out_a / "outputs" / "buck_averaged_mvp" / "direct_trap" / "pulsim.csv"
    csv_b = out_b / "outputs" / "buck_averaged_mvp" / "direct_trap" / "pulsim.csv"
    assert csv_a.read_text(encoding="utf-8") == csv_b.read_text(encoding="utf-8")


def test_run_benchmarks_accepts_expected_failure_for_averaged_invalid_mapping(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "benchmarks" / "benchmarks.yaml"

    results = br.run_benchmarks(
        manifest_path,
        tmp_path / "out",
        selected=["buck_averaged_expected_failure"],
        scenario_filter=["direct_trap"],
    )

    assert len(results) == 1
    result = results[0]
    assert result.status == "passed"
    assert result.mode == "transient"
    assert "Expected failure matched" in result.message
    assert result.telemetry.get("expected_failure_matched") == 1.0


def test_averaged_benchmark_kpi_gate_passes_with_frozen_baseline(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "benchmarks" / "benchmarks.yaml"
    out_dir = tmp_path / "averaged_gate_out"

    results = br.run_benchmarks(
        manifest_path,
        out_dir,
        selected=[
            "buck_switching_paired",
            "buck_averaged_mvp",
            "buck_averaged_expected_failure",
        ],
        scenario_filter=["direct_trap"],
    )
    br.write_results(out_dir, results)
    assert len(results) == 3
    assert all(item.status == "passed" for item in results)

    baseline_path = (
        repo_root
        / "benchmarks"
        / "kpi_baselines"
        / "averaged_converter_phase14_2026-03-07"
        / "kpi_baseline.json"
    )
    thresholds_path = repo_root / "benchmarks" / "kpi_thresholds_averaged.yaml"
    bench_results_path = out_dir / "results.json"

    report = kpi_gate.run_gate(
        baseline_path=baseline_path,
        thresholds_path=thresholds_path,
        bench_results_path=bench_results_path,
        parity_ltspice_results_path=None,
        parity_ngspice_results_path=None,
        stress_summary_path=None,
    )

    assert report["overall_status"] == "passed"
    assert report["failed_required_metrics"] == 0
    assert report["comparisons"]["averaged_pair_case_count"]["status"] == "passed"
    assert report["comparisons"]["averaged_pair_fidelity_error"]["status"] == "passed"
    assert report["comparisons"]["averaged_pair_runtime_speedup_min"]["status"] in {
        "passed",
        "skipped",
    }


def test_python_backend_runs_frequency_analysis_mode_and_writes_frequency_csv(
    tmp_path: Path,
) -> None:
    netlist_path = _write_ac_rc_netlist(
        tmp_path,
        benchmark_block={
            "id": "ac_backend_mode",
            "validation": {"type": "none"},
        },
    )
    output_path = tmp_path / "ac_frequency.csv"
    run = backend.run_from_yaml(
        netlist_path, output_path, preferred_mode="frequency_analysis"
    )

    assert run.mode == "frequency_analysis"
    assert run.steps > 0
    assert run.telemetry.get("ac_sweep_case") == 1.0
    header = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert (
        header
        == "frequency_hz,response_real,response_imag,magnitude,magnitude_db,phase_deg"
    )


def test_frequency_analysis_repeat_runs_are_deterministic(tmp_path: Path) -> None:
    netlist_path = _write_ac_rc_netlist(
        tmp_path,
        benchmark_block={
            "id": "ac_repeatability",
            "validation": {"type": "none"},
        },
    )
    out_a = tmp_path / "ac_run_a.csv"
    out_b = tmp_path / "ac_run_b.csv"
    backend.run_from_yaml(netlist_path, out_a, preferred_mode="frequency_analysis")
    backend.run_from_yaml(netlist_path, out_b, preferred_mode="frequency_analysis")

    assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")


def test_run_benchmarks_validates_ac_analytical_case(tmp_path: Path) -> None:
    _write_ac_rc_netlist(
        tmp_path,
        benchmark_block={
            "id": "ac_analytical_smoke",
            "validation": {
                "type": "ac_analytical",
                "model": "rc_lowpass",
                "params": {"r": "1k", "c": "1u"},
            },
            "expectations": {
                "metrics": {
                    "max_error": 1e-4,
                    "phase_error_deg": 1e-3,
                }
            },
        },
    )
    manifest_path = _write_manifest(tmp_path, "ac_rc.yaml")

    results = br.run_benchmarks(manifest_path, tmp_path / "out")

    assert len(results) == 1
    result = results[0]
    assert result.status == "passed"
    assert result.mode == "frequency_analysis"
    assert result.max_error is not None
    assert result.phase_error_deg is not None
    assert result.telemetry.get("ac_sweep_mag_error") is not None
    assert result.telemetry.get("ac_sweep_phase_error") is not None


def test_run_benchmarks_accepts_expected_frequency_failure_for_control_workflow(
    tmp_path: Path,
) -> None:
    _write_ac_control_expected_failure_netlist(
        tmp_path,
        benchmark_block={
            "id": "ac_control_expected_failure",
            "validation": {"type": "none"},
            "expectations": {
                "expected_failure": {
                    "mode": "frequency_analysis",
                    "diagnostic": "FrequencyUnsupportedConfiguration",
                    "message_contains": "probe/scope virtual components only",
                }
            },
        },
    )
    manifest_path = _write_manifest(tmp_path, "ac_control_fail.yaml")

    results = br.run_benchmarks(manifest_path, tmp_path / "out")

    assert len(results) == 1
    result = results[0]
    assert result.status == "passed"
    assert result.mode == "frequency_analysis"
    assert "Expected failure matched" in result.message
    assert result.telemetry.get("expected_failure_matched") == 1.0


def test_run_benchmarks_validates_magnetic_core_saturation_and_hysteresis(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "benchmarks" / "magnetic_core_benchmarks.yaml"

    results = br.run_benchmarks(
        manifest_path,
        tmp_path / "out",
        selected=["magnetic_core_saturation", "magnetic_core_hysteresis"],
    )

    assert len(results) == 2
    rows = {row.benchmark_id: row for row in results}

    sat = rows["magnetic_core_saturation"]
    assert sat.status == "passed"
    assert sat.max_error is not None
    assert sat.telemetry.get("magnetic_fixture_saturation") == 1.0
    assert sat.telemetry.get("magnetic_sat_error") is not None

    hyst = rows["magnetic_core_hysteresis"]
    assert hyst.status == "passed"
    assert hyst.max_error is not None
    assert hyst.telemetry.get("magnetic_fixture_hysteresis") == 1.0
    assert hyst.telemetry.get("magnetic_hysteresis_cycle_energy_error") is not None


def test_run_benchmarks_emits_magnetic_trend_and_determinism_telemetry(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "benchmarks" / "magnetic_core_benchmarks.yaml"

    selected_ids = [
        "magnetic_core_frequency_trend_low",
        "magnetic_core_frequency_trend_high",
        "magnetic_core_determinism_ref",
        "magnetic_core_determinism_cmp",
    ]
    results = br.run_benchmarks(
        manifest_path,
        tmp_path / "out",
        selected=selected_ids,
    )

    assert len(results) == len(selected_ids)
    rows = {row.benchmark_id: row for row in results}
    for bench_id in selected_ids:
        assert rows[bench_id].status == "passed"

    low = rows["magnetic_core_frequency_trend_low"]
    high = rows["magnetic_core_frequency_trend_high"]
    assert low.telemetry.get("magnetic_fixture_frequency_trend") == 1.0
    assert high.telemetry.get("magnetic_fixture_frequency_trend") == 1.0
    assert low.telemetry.get("magnetic_trend_role_low") == 1.0
    assert high.telemetry.get("magnetic_trend_role_high") == 1.0
    assert low.telemetry.get("magnetic_avg_core_loss") is not None
    assert high.telemetry.get("magnetic_avg_core_loss") is not None
    assert float(high.telemetry["magnetic_avg_core_loss"]) > float(low.telemetry["magnetic_avg_core_loss"])

    det = rows["magnetic_core_determinism_cmp"]
    assert det.telemetry.get("magnetic_determinism_case") == 1.0
    assert det.max_error is not None
    assert det.max_error <= 1e-12
