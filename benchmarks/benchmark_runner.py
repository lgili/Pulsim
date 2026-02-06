#!/usr/bin/env python3
"""Benchmark and validation runner for Pulsim YAML netlists."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    from pulsim_python_backend import is_available as python_backend_available
    from pulsim_python_backend import run_from_yaml as run_pulsim_python
except ImportError:  # pragma: no cover - local import fallback
    python_backend_available = None
    run_pulsim_python = None


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


def find_pulsim_cli() -> Optional[Path]:
    script_dir = Path(__file__).resolve().parent.parent
    candidates = [
        script_dir / "build" / "cli" / "pulsim",
        script_dir / "build" / "Release" / "cli" / "pulsim",
        script_dir / "build" / "Debug" / "cli" / "pulsim",
        Path("pulsim"),
    ]
    for path in candidates:
        if path.exists():
            return path
        if path.name == "pulsim":
            if subprocess.run(["which", "pulsim"], capture_output=True).returncode == 0:
                return path
    return None


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


def run_pulsim(cli_path: Optional[Path], netlist_path: Path, output_path: Path) -> Tuple[float, int, str]:
    if cli_path is None:
        if run_pulsim_python is None or not can_use_pulsim_python_backend():
            raise RuntimeError(
                "Pulsim CLI not found and Python backend unavailable. "
                "Install pulsim package or provide --pulsim-cli."
            )
        return run_pulsim_python(netlist_path, output_path)

    start = time.perf_counter()
    try:
        result = subprocess.run(
            [str(cli_path), "run", str(netlist_path), "-o", str(output_path)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        if run_pulsim_python is not None and can_use_pulsim_python_backend():
            return run_pulsim_python(netlist_path, output_path)
        raise RuntimeError(f"Pulsim CLI not found: {cli_path}") from exc

    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Pulsim run failed")
    steps = 0
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as handle:
            steps = max(0, sum(1 for _ in handle) - 1)
    return elapsed, steps, result.stdout


def parse_telemetry(output: str) -> Dict[str, Optional[float]]:
    telemetry: Dict[str, Optional[float]] = {
        "newton_iterations": None,
        "linear_iterations": None,
        "residual_norm": None,
        "timestep_rejections": None,
    }

    patterns = {
        "newton_iterations": r"Newton iterations\s*[:=]\s*(\d+)",
        "linear_iterations": r"Linear iterations\s*[:=]\s*(\d+)",
        "residual_norm": r"Residual norm\s*[:=]\s*([0-9eE+\-.]+)",
        "timestep_rejections": r"Timestep rejections\s*[:=]\s*(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1)
            telemetry[key] = float(value) if "." in value or "e" in value.lower() else float(int(value))

    return telemetry


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
    if len(ref_values) != len(values):
        return None, None, "Baseline length mismatch"
    return (*compute_errors(values, ref_values), "")


def run_benchmarks(
    benchmarks_path: Path,
    output_dir: Path,
    cli_path: Optional[Path],
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
                    runtime_s, steps, stdout = run_pulsim(cli_path, scenario_file, output_path)
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
                telemetry = parse_telemetry(stdout)
                telemetry["steps"] = float(steps)
                telemetry["runtime_s"] = float(runtime_s)
                using_python_backend = "Backend: python-api" in stdout
                telemetry["python_backend"] = 1.0 if using_python_backend else 0.0

                validation = bench_meta.get("validation", {})
                validation_type = validation.get("type", "none")
                observable = validation.get("observable")
                expectations = bench_meta.get("expectations", {})
                max_threshold = coerce_optional_float(expectations.get("metrics", {}).get("max_error"))

                if using_python_backend and validation_type != "none":
                    status = "skipped"
                    message = (
                        "Validation skipped on Python backend fallback "
                        "(CLI-only solver/integrator configuration not available)."
                    )
                elif validation_type != "none" and observable:
                    times, series = load_csv_series(output_path)
                    if observable not in series:
                        status = "failed"
                        message = f"Missing observable column: {observable}"
                    else:
                        values = series[observable]
                        if validation_type == "analytical":
                            try:
                                max_error, rms_error = validate_analytical(
                                    times,
                                    values,
                                    validation.get("model", ""),
                                    validation.get("params", {}),
                                )
                                if max_threshold is not None and max_error is not None:
                                    status = "passed" if max_error <= max_threshold else "failed"
                            except Exception as exc:
                                status = "failed"
                                message = str(exc)
                        elif validation_type == "reference":
                            baseline_rel = validation.get("baseline")
                            if baseline_rel:
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
                                        times, values, baseline_path, observable
                                    )
                                    if max_error is None:
                                        status = "skipped"
                                    elif max_threshold is not None:
                                        status = "passed" if max_error <= max_threshold else "failed"
                            else:
                                status = "skipped"
                                message = "Missing baseline path"
                        else:
                            status = "skipped"
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
    parser.add_argument("--pulsim-cli", type=Path, default=None)
    parser.add_argument("--only", nargs="*", help="Benchmark ids to run")
    parser.add_argument("--matrix", action="store_true", help="Run full validation matrix")
    parser.add_argument("--generate-baselines", action="store_true", help="Generate missing reference baselines")
    args = parser.parse_args()

    if yaml is None:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml")

    cli_path = args.pulsim_cli or find_pulsim_cli()
    if cli_path is None and not can_use_pulsim_python_backend():
        raise SystemExit(
            "Pulsim CLI not found and Python backend unavailable. "
            "Build the project, pass --pulsim-cli, or install the pulsim Python package."
        )
    if cli_path is None:
        print("Pulsim CLI not found. Using Python API backend.")

    results = run_benchmarks(
        args.benchmarks,
        args.output_dir,
        cli_path,
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
