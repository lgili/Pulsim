#!/usr/bin/env python3
"""Benchmark and validation runner for Pulsim YAML netlists."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

try:
    from pulsim_python_backend import is_available as python_backend_available
    from pulsim_python_backend import run_from_yaml as run_pulsim_python
except ImportError:  # pragma: no cover - local import fallback
    python_backend_available = None
    run_pulsim_python = None


@dataclass
class PulsimRunResult:
    runtime_s: float
    steps: int
    mode: str
    telemetry: Dict[str, Optional[float]]


@dataclass
class ScenarioResult:
    benchmark_id: str
    scenario: str
    status: str
    runtime_s: float
    steps: int
    max_error: Optional[float]
    rms_error: Optional[float]
    message: str
    telemetry: Dict[str, Optional[float]]


def can_use_pulsim_python_backend() -> bool:
    if python_backend_available is None:
        return False
    try:
        return bool(python_backend_available())
    except Exception:
        return False


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        raise ValueError("Missing value")
    raw = str(value).strip().lower()
    suffixes = {
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "µ": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "meg": 1e6,
        "g": 1e9,
        "t": 1e12,
    }
    if raw.endswith("meg"):
        return float(raw[:-3]) * suffixes["meg"]
    for suffix, multiplier in suffixes.items():
        if suffix != "meg" and raw.endswith(suffix):
            return float(raw[: -len(suffix)]) * multiplier
    return float(raw)


def coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return parse_value(value)


def infer_preferred_mode(scenario_name: str, scenario_override: Dict[str, Any]) -> Optional[str]:
    sim = scenario_override.get("simulation") if isinstance(scenario_override, dict) else None
    if isinstance(sim, dict):
        has_shooting = "shooting" in sim
        has_hb = "harmonic_balance" in sim or "hb" in sim
        if has_shooting and not has_hb:
            return "shooting"
        if has_hb and not has_shooting:
            return "harmonic_balance"

    lowered = scenario_name.lower()
    if "shooting" in lowered:
        return "shooting"
    if "harmonic" in lowered or lowered == "hb":
        return "harmonic_balance"
    return None


def normalize_periodic_mode(netlist: Dict[str, Any], preferred_mode: Optional[str]) -> None:
    if preferred_mode is None:
        return
    simulation = netlist.get("simulation")
    if not isinstance(simulation, dict):
        return

    if preferred_mode == "shooting":
        simulation.pop("harmonic_balance", None)
        simulation.pop("hb", None)
    elif preferred_mode == "harmonic_balance":
        simulation.pop("shooting", None)


def apply_runtime_defaults(netlist: Dict[str, Any]) -> None:
    """Apply benchmark runtime defaults without overriding explicit YAML choices."""
    simulation = netlist.get("simulation")
    if not isinstance(simulation, dict):
        return

    # Benchmark suite targets deterministic comparisons by default.
    if "adaptive_timestep" not in simulation:
        simulation["adaptive_timestep"] = False


def run_pulsim(
    netlist_path: Path,
    output_path: Path,
    preferred_mode: Optional[str] = None,
    use_initial_conditions: bool = False,
) -> PulsimRunResult:
    if run_pulsim_python is None or not can_use_pulsim_python_backend():
        raise RuntimeError(
            "Pulsim Python runtime backend unavailable. "
            "Build Python bindings and expose them via build/python or install pulsim package."
        )

    raw_result = run_pulsim_python(
        netlist_path,
        output_path,
        preferred_mode=preferred_mode,
        use_initial_conditions=use_initial_conditions,
    )

    if not hasattr(raw_result, "runtime_s") or not hasattr(raw_result, "telemetry"):
        raise RuntimeError("Unexpected backend response: structured telemetry result expected")

    return PulsimRunResult(
        runtime_s=float(getattr(raw_result, "runtime_s")),
        steps=int(getattr(raw_result, "steps")),
        mode=str(getattr(raw_result, "mode")),
        telemetry=dict(getattr(raw_result, "telemetry")),
    )


def load_csv_series(path: Path) -> Tuple[List[float], Dict[str, List[float]]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        times: List[float] = []
        series: Dict[str, List[float]] = {}
        for row in reader:
            times.append(float(row["time"]))
            for key, value in row.items():
                if key == "time":
                    continue
                series.setdefault(key, []).append(float(value))
    return times, series


def analytical_rc_step(times: List[float], v0: float, r: float, c: float) -> List[float]:
    tau = r * c
    return [v0 * (1.0 - math.exp(-t / tau)) for t in times]


def analytical_rl_step(times: List[float], v0: float, r: float, l: float) -> List[float]:
    tau = l / r
    return [v0 * math.exp(-t / tau) for t in times]


def analytical_rlc_step(times: List[float], v0: float, r: float, l: float, c: float) -> List[float]:
    alpha = r / (2.0 * l)
    omega0 = 1.0 / math.sqrt(l * c)
    result = []
    if alpha < omega0:
        omega_d = math.sqrt(omega0 * omega0 - alpha * alpha)
        for t in times:
            value = 1.0 - math.exp(-alpha * t) * (
                math.cos(omega_d * t) + (alpha / omega_d) * math.sin(omega_d * t)
            )
            result.append(v0 * value)
    else:
        s1 = -alpha + math.sqrt(alpha * alpha - omega0 * omega0)
        s2 = -alpha - math.sqrt(alpha * alpha - omega0 * omega0)
        a = v0 * s2 / (s2 - s1)
        b = -v0 * s1 / (s2 - s1)
        for t in times:
            result.append(v0 - a * math.exp(s1 * t) - b * math.exp(s2 * t))
    return result


def compute_errors(values: List[float], reference: List[float]) -> Tuple[float, float]:
    if len(values) != len(reference):
        raise ValueError("Length mismatch between values and reference")
    errors = [abs(a - b) for a, b in zip(values, reference)]
    max_error = max(errors) if errors else 0.0
    rms_error = math.sqrt(sum(err * err for err in errors) / len(errors)) if errors else 0.0
    return max_error, rms_error


def validate_analytical(times: List[float], values: List[float], model: str, params: Dict[str, Any]) -> Tuple[float, float]:
    v0 = parse_value(params["v0"])
    if model == "rc_step":
        r = parse_value(params["r"])
        c = parse_value(params["c"])
        reference = analytical_rc_step(times, v0, r, c)
    elif model == "rl_step":
        r = parse_value(params["r"])
        l = parse_value(params["l"])
        reference = analytical_rl_step(times, v0, r, l)
    elif model == "rlc_step":
        r = parse_value(params["r"])
        l = parse_value(params["l"])
        c = parse_value(params["c"])
        reference = analytical_rlc_step(times, v0, r, l, c)
    else:
        raise ValueError(f"Unknown analytical model: {model}")
    return compute_errors(values, reference)


def validate_reference(times: List[float], values: List[float], baseline_path: Path, column: str) -> Tuple[Optional[float], Optional[float], str]:
    if not baseline_path.exists():
        return None, None, f"Baseline missing: {baseline_path}"
    ref_times, ref_series = load_csv_series(baseline_path)
    if column not in ref_series:
        return None, None, f"Baseline missing column: {column}"
    ref_values = ref_series[column]
    if len(ref_times) < 2 or len(times) < 2:
        return None, None, "Not enough samples for baseline comparison"

    overlap_start = max(times[0], ref_times[0])
    overlap_end = min(times[-1], ref_times[-1])
    if overlap_start >= overlap_end:
        return None, None, "No overlapping time range with baseline"

    eval_times: List[float] = []
    eval_values: List[float] = []
    for t, v in zip(times, values):
        if overlap_start <= t <= overlap_end:
            eval_times.append(t)
            eval_values.append(v)

    if len(eval_times) < 2:
        return None, None, "Not enough overlapping samples with baseline"

    interp_ref: List[float] = []
    idx = 0
    for t in eval_times:
        while idx + 1 < len(ref_times) and ref_times[idx + 1] < t:
            idx += 1
        if idx + 1 >= len(ref_times):
            interp_ref.append(ref_values[-1])
            continue
        t0, t1 = ref_times[idx], ref_times[idx + 1]
        v0, v1 = ref_values[idx], ref_values[idx + 1]
        if t1 == t0:
            interp_ref.append(v0)
            continue
        alpha = (t - t0) / (t1 - t0)
        interp_ref.append(v0 + alpha * (v1 - v0))

    return (*compute_errors(eval_values, interp_ref), "")


def apply_validation_window(
    times: List[float],
    values: List[float],
    validation: Dict[str, Any],
) -> Tuple[List[float], List[float]]:
    if len(times) != len(values):
        raise ValueError("Time/value length mismatch")

    ignore_initial_samples = int(validation.get("ignore_initial_samples", 0) or 0)
    start_time = validation.get("start_time")
    start_time_value = parse_value(start_time) if start_time is not None else None

    filtered_times: List[float] = []
    filtered_values: List[float] = []
    for idx, (t, v) in enumerate(zip(times, values)):
        if idx < ignore_initial_samples:
            continue
        if start_time_value is not None and t < start_time_value:
            continue
        filtered_times.append(t)
        filtered_values.append(v)

    if len(filtered_times) < 2:
        raise ValueError("Validation window has fewer than 2 samples")

    return filtered_times, filtered_values


def run_benchmarks(
    benchmarks_path: Path,
    output_dir: Path,
    selected: Optional[List[str]] = None,
    matrix: bool = False,
    generate_baselines: bool = False,
) -> List[ScenarioResult]:
    manifest = load_yaml(benchmarks_path)
    scenarios = manifest.get("scenarios", {})
    results: List[ScenarioResult] = []

    for entry in manifest.get("benchmarks", []):
        circuit_path = (benchmarks_path.parent / entry["path"]).resolve()
        netlist = load_yaml(circuit_path)
        bench_meta = netlist.get("benchmark", {})
        benchmark_id = bench_meta.get("id", circuit_path.stem)
        if selected and benchmark_id not in selected:
            continue

        scenario_names = list(scenarios.keys()) if matrix else entry.get("scenarios", ["default"])
        if "default" in scenario_names and "default" not in scenarios:
            scenarios["default"] = {}

        for scenario_name in scenario_names:
            scenario_override = scenarios.get(scenario_name, {})
            scenario_netlist = deep_merge(netlist, scenario_override)
            preferred_mode = infer_preferred_mode(scenario_name, scenario_override)
            normalize_periodic_mode(scenario_netlist, preferred_mode)
            apply_runtime_defaults(scenario_netlist)
            simulation_cfg = scenario_netlist.get("simulation", {})
            use_initial_conditions = bool(
                simulation_cfg.get("uic", False) if isinstance(simulation_cfg, dict) else False
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                scenario_file = tmpdir_path / f"{benchmark_id}_{scenario_name}.yaml"
                output_path = output_dir / "outputs" / benchmark_id / scenario_name / "pulsim.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if yaml is None:
                    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
                with open(scenario_file, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(scenario_netlist, handle, sort_keys=False)

                try:
                    run_result = run_pulsim(
                        scenario_file,
                        output_path,
                        preferred_mode=preferred_mode,
                        use_initial_conditions=use_initial_conditions,
                    )
                except Exception as exc:
                    results.append(
                        ScenarioResult(
                            benchmark_id=benchmark_id,
                            scenario=scenario_name,
                            status="failed",
                            runtime_s=0.0,
                            steps=0,
                            max_error=None,
                            rms_error=None,
                            message=str(exc),
                            telemetry={},
                        )
                    )
                    continue

                status = "passed"
                max_error = None
                rms_error = None
                message = ""
                runtime_s = run_result.runtime_s
                steps = run_result.steps
                telemetry = dict(run_result.telemetry)
                telemetry["steps"] = float(steps)
                telemetry["runtime_s"] = float(runtime_s)
                telemetry["python_backend"] = 1.0
                for key in ("newton_iterations", "timestep_rejections", "linear_fallbacks"):
                    if telemetry.get(key) is None:
                        telemetry[key] = 0.0

                validation = bench_meta.get("validation", {})
                validation_type = validation.get("type", "none")
                observable = validation.get("observable")
                expectations = bench_meta.get("expectations", {})
                max_threshold = coerce_optional_float(expectations.get("metrics", {}).get("max_error"))

                if validation_type != "none" and not observable:
                    status = "failed"
                    message = "Missing validation observable"
                elif validation_type != "none":
                    times, series = load_csv_series(output_path)
                    if observable not in series:
                        status = "failed"
                        message = f"Missing observable column: {observable}"
                    else:
                        values = series[observable]
                        try:
                            times_eval, values_eval = apply_validation_window(times, values, validation)
                        except Exception as exc:
                            status = "failed"
                            message = str(exc)
                        else:
                            if validation_type == "analytical":
                                try:
                                    max_error, rms_error = validate_analytical(
                                        times_eval,
                                        values_eval,
                                        validation.get("model", ""),
                                        validation.get("params", {}),
                                    )
                                    if max_threshold is not None and max_error is not None:
                                        if max_error <= max_threshold:
                                            status = "passed"
                                        else:
                                            status = "failed"
                                            message = (
                                                f"max_error {max_error:.6e} > threshold {max_threshold:.6e}"
                                            )
                                except Exception as exc:
                                    status = "failed"
                                    message = str(exc)
                            elif validation_type == "reference":
                                baseline_rel = validation.get("baseline")
                                if not baseline_rel:
                                    status = "failed"
                                    message = "Missing baseline path"
                                else:
                                    baseline_path = (benchmarks_path.parent / baseline_rel).resolve()
                                    if not baseline_path.exists() and generate_baselines:
                                        baseline_path.parent.mkdir(parents=True, exist_ok=True)
                                        with open(output_path, "r", encoding="utf-8") as src, open(
                                            baseline_path, "w", encoding="utf-8"
                                        ) as dst:
                                            dst.write(src.read())
                                        status = "baseline"
                                        message = "Baseline generated"
                                    else:
                                        max_error, rms_error, message = validate_reference(
                                            times_eval, values_eval, baseline_path, observable
                                        )
                                        if max_error is None:
                                            status = "failed"
                                        elif max_threshold is not None:
                                            if max_error <= max_threshold:
                                                status = "passed"
                                            else:
                                                status = "failed"
                                                message = (
                                                    f"max_error {max_error:.6e} > threshold {max_threshold:.6e}"
                                                )
                            else:
                                status = "failed"
                                message = f"Unsupported validation type: {validation_type}"

                results.append(
                    ScenarioResult(
                        benchmark_id=benchmark_id,
                        scenario=scenario_name,
                        status=status,
                        runtime_s=runtime_s,
                        steps=steps,
                        max_error=max_error,
                        rms_error=rms_error,
                        message=message,
                        telemetry=telemetry,
                    )
                )

    return results


def write_results(output_dir: Path, results: List[ScenarioResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "results.csv"
    results_json = output_dir / "results.json"
    summary_json = output_dir / "summary.json"

    with open(results_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "benchmark_id",
            "scenario",
            "status",
            "runtime_s",
            "steps",
            "max_error",
            "rms_error",
            "message",
        ])
        for item in results:
            writer.writerow([
                item.benchmark_id,
                item.scenario,
                item.status,
                f"{item.runtime_s:.6f}",
                item.steps,
                "" if item.max_error is None else f"{item.max_error:.6e}",
                "" if item.rms_error is None else f"{item.rms_error:.6e}",
                item.message,
            ])

    payload = {
        "results": [item.__dict__ for item in results],
    }
    with open(results_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    summary = {
        "passed": sum(1 for item in results if item.status == "passed"),
        "failed": sum(1 for item in results if item.status == "failed"),
        "skipped": sum(1 for item in results if item.status == "skipped"),
        "baseline": sum(1 for item in results if item.status == "baseline"),
        "total": len(results),
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pulsim benchmark suite")
    parser.add_argument("--benchmarks", type=Path, default=Path(__file__).with_name("benchmarks.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/out"))
    parser.add_argument("--only", nargs="*", help="Benchmark ids to run")
    parser.add_argument("--matrix", action="store_true", help="Run full validation matrix")
    parser.add_argument("--generate-baselines", action="store_true", help="Generate missing reference baselines")
    args = parser.parse_args()

    if yaml is None:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml")

    if not can_use_pulsim_python_backend():
        raise SystemExit(
            "Pulsim Python runtime backend unavailable. "
            "Build Python bindings and expose build/python on PYTHONPATH or install pulsim package."
        )

    results = run_benchmarks(
        args.benchmarks,
        args.output_dir,
        selected=args.only,
        matrix=args.matrix,
        generate_baselines=args.generate_baselines,
    )
    write_results(args.output_dir, results)

    summary = {
        "passed": sum(1 for item in results if item.status == "passed"),
        "failed": sum(1 for item in results if item.status == "failed"),
        "skipped": sum(1 for item in results if item.status == "skipped"),
        "baseline": sum(1 for item in results if item.status == "baseline"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
