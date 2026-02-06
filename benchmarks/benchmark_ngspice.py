#!/usr/bin/env python3
"""Benchmark Pulsim against ngspice using equivalent circuit files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmark_runner import (
    can_use_pulsim_python_backend,
    deep_merge,
    find_pulsim_cli,
    load_csv_series,
    load_yaml,
    run_pulsim,
    yaml,
)


@dataclass
class ObservableSpec:
    column: str
    spice_vector: str


@dataclass
class ObservableMetrics:
    column: str
    spice_vector: str
    samples: int
    max_error: float
    rms_error: float


@dataclass
class BenchmarkResult:
    benchmark_id: str
    scenario: str
    status: str
    message: str
    pulsim_runtime_s: float
    ngspice_runtime_s: float
    pulsim_steps: int
    ngspice_steps: int
    speedup: Optional[float]
    max_error: Optional[float]
    rms_error: Optional[float]
    observables: List[ObservableMetrics]


def coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def check_ngspice() -> bool:
    try:
        result = subprocess.run(["ngspice", "--version"], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return result.returncode == 0


def strip_control_blocks_and_end(text: str) -> str:
    output: List[str] = []
    in_control = False
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith(".control"):
            in_control = True
            continue
        if in_control:
            if stripped.startswith(".endc"):
                in_control = False
            continue
        if stripped == ".end":
            continue
        output.append(line)
    return "\n".join(output).rstrip()


def write_batch_ngspice_netlist(
    source_netlist: Path,
    vectors: Sequence[str],
    output_files: Sequence[Path],
    destination: Path,
) -> None:
    if len(vectors) != len(output_files):
        raise ValueError("Each ngspice vector must have one output file")

    source_text = source_netlist.read_text(encoding="utf-8")
    base_text = strip_control_blocks_and_end(source_text)
    lines: List[str] = [base_text, "", ".control", "set filetype=ascii", "run"]
    for vector, out_file in zip(vectors, output_files):
        lines.append(f"wrdata {out_file.as_posix()} {vector}")
    lines.extend(["quit", ".endc", ".end", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")


def parse_ngspice_wrdata(path: Path) -> Tuple[List[float], List[float]]:
    times: List[float] = []
    values: List[float] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                numeric = [float(token) for token in parts]
            except ValueError:
                continue
            # ngspice may prepend an index column.
            if len(numeric) >= 3 and len(numeric) % 2 == 1:
                numeric = numeric[1:]
            if len(numeric) < 2:
                continue
            times.append(numeric[0])
            values.append(numeric[1])

    if not times:
        raise RuntimeError(f"ngspice output file has no numeric data: {path}")

    return times, values


def run_ngspice_vectors(
    netlist_path: Path,
    vectors: Sequence[str],
) -> Tuple[float, int, Dict[str, Tuple[List[float], List[float]]]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        batch_netlist = tmp / "benchmark_job.cir"
        output_files = [tmp / f"vector_{idx}.dat" for idx in range(len(vectors))]
        write_batch_ngspice_netlist(netlist_path, vectors, output_files, batch_netlist)

        start = time.perf_counter()
        result = subprocess.run(
            ["ngspice", "-b", str(batch_netlist)],
            capture_output=True,
            text=True,
            cwd=netlist_path.parent,
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "ngspice failed"
            raise RuntimeError(detail)

        series: Dict[str, Tuple[List[float], List[float]]] = {}
        max_steps = 0
        for vector, out_file in zip(vectors, output_files):
            if not out_file.exists():
                raise RuntimeError(f"ngspice did not generate vector output: {out_file}")
            times, values = parse_ngspice_wrdata(out_file)
            series[vector] = (times, values)
            max_steps = max(max_steps, max(0, len(times) - 1))

    return elapsed, max_steps, series


def interpolate_linear(target_times: Sequence[float], ref_times: Sequence[float], ref_values: Sequence[float]) -> List[float]:
    if len(ref_times) < 2:
        raise ValueError("Reference time series requires at least 2 points")
    output: List[float] = []
    idx = 0

    for t in target_times:
        while idx + 1 < len(ref_times) and ref_times[idx + 1] < t:
            idx += 1
        if idx + 1 >= len(ref_times):
            output.append(ref_values[-1])
            continue
        t0, t1 = ref_times[idx], ref_times[idx + 1]
        v0, v1 = ref_values[idx], ref_values[idx + 1]
        if t1 == t0:
            output.append(v0)
            continue
        alpha = (t - t0) / (t1 - t0)
        output.append(v0 + alpha * (v1 - v0))

    return output


def compare_series(
    pulsim_times: Sequence[float],
    pulsim_values: Sequence[float],
    spice_times: Sequence[float],
    spice_values: Sequence[float],
) -> Tuple[List[float], int]:
    if len(pulsim_times) != len(pulsim_values):
        raise ValueError("Pulsim series length mismatch")
    if len(spice_times) != len(spice_values):
        raise ValueError("ngspice series length mismatch")
    if not pulsim_times or not spice_times:
        raise ValueError("Empty series")

    overlap_start = max(pulsim_times[0], spice_times[0])
    overlap_end = min(pulsim_times[-1], spice_times[-1])
    if overlap_start >= overlap_end:
        raise ValueError("No overlapping time range between Pulsim and ngspice")

    filtered_times: List[float] = []
    filtered_values: List[float] = []
    for t, v in zip(pulsim_times, pulsim_values):
        if overlap_start <= t <= overlap_end:
            filtered_times.append(t)
            filtered_values.append(v)

    if len(filtered_times) < 2:
        raise ValueError("Not enough overlapping samples for comparison")

    ref_interp = interpolate_linear(filtered_times, spice_times, spice_values)
    errors = [abs(a - b) for a, b in zip(filtered_values, ref_interp)]
    return errors, len(filtered_times)


def normalize_observable(entry: Any) -> Optional[ObservableSpec]:
    if isinstance(entry, str):
        col = entry.strip()
        if not col:
            return None
        return ObservableSpec(column=col, spice_vector=col.lower())
    if isinstance(entry, dict):
        col = str(entry.get("column", "")).strip()
        if not col:
            return None
        spice_vector = str(entry.get("spice_vector") or entry.get("spice") or col.lower()).strip()
        return ObservableSpec(column=col, spice_vector=spice_vector)
    return None


def resolve_observables(
    benchmark_meta: Dict[str, Any],
    validation_meta: Dict[str, Any],
    entry_observables: Optional[List[Any]],
    cli_observables: Optional[List[str]],
) -> List[ObservableSpec]:
    raw_entries: List[Any] = []
    if cli_observables:
        raw_entries = cli_observables
    elif entry_observables:
        raw_entries = entry_observables
    elif isinstance(validation_meta.get("observables"), list):
        raw_entries = validation_meta["observables"]
    elif isinstance(benchmark_meta.get("observables"), list):
        raw_entries = benchmark_meta["observables"]
    elif validation_meta.get("observable"):
        raw_entries = [validation_meta["observable"]]

    specs: List[ObservableSpec] = []
    seen: set[str] = set()
    for raw in raw_entries:
        spec = normalize_observable(raw)
        if spec is None or spec.column in seen:
            continue
        seen.add(spec.column)
        specs.append(spec)
    return specs


def compare_pulsim_vs_ngspice(
    pulsim_csv_path: Path,
    spice_netlist_path: Path,
    observable_specs: List[ObservableSpec],
) -> Tuple[float, int, List[ObservableMetrics], Optional[float], Optional[float]]:
    if not observable_specs:
        return 0.0, 0, [], None, None

    pulsim_times, pulsim_series = load_csv_series(pulsim_csv_path)
    vectors = [spec.spice_vector for spec in observable_specs]
    ng_runtime, ng_steps, ng_series = run_ngspice_vectors(spice_netlist_path, vectors)

    metrics: List[ObservableMetrics] = []
    all_errors: List[float] = []

    for spec in observable_specs:
        if spec.column not in pulsim_series:
            raise RuntimeError(f"Pulsim output missing observable column: {spec.column}")
        if spec.spice_vector not in ng_series:
            raise RuntimeError(f"ngspice output missing vector: {spec.spice_vector}")

        spice_times, spice_values = ng_series[spec.spice_vector]
        errors, samples = compare_series(
            pulsim_times,
            pulsim_series[spec.column],
            spice_times,
            spice_values,
        )
        all_errors.extend(errors)
        rms = math.sqrt(sum(e * e for e in errors) / len(errors))
        metrics.append(
            ObservableMetrics(
                column=spec.column,
                spice_vector=spec.spice_vector,
                samples=samples,
                max_error=max(errors),
                rms_error=rms,
            )
        )

    global_max = max(all_errors) if all_errors else None
    global_rms = math.sqrt(sum(e * e for e in all_errors) / len(all_errors)) if all_errors else None
    return ng_runtime, ng_steps, metrics, global_max, global_rms


def run_case(
    pulsim_cli: Optional[Path],
    scenario_netlist: Dict[str, Any],
    benchmark_id: str,
    scenario_name: str,
    spice_netlist_path: Path,
    observable_specs: List[ObservableSpec],
    max_error_threshold: Optional[float],
    output_dir: Path,
) -> BenchmarkResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        scenario_file = tmpdir_path / f"{benchmark_id}_{scenario_name}.yaml"
        pulsim_output = output_dir / "outputs" / benchmark_id / scenario_name / "pulsim.csv"
        pulsim_output.parent.mkdir(parents=True, exist_ok=True)

        if yaml is None:
            raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
        with open(scenario_file, "w", encoding="utf-8") as handle:
            yaml.safe_dump(scenario_netlist, handle, sort_keys=False)

        pulsim_runtime, pulsim_steps, _ = run_pulsim(pulsim_cli, scenario_file, pulsim_output)

    ng_runtime, ng_steps, observables, max_error, rms_error = compare_pulsim_vs_ngspice(
        pulsim_output,
        spice_netlist_path,
        observable_specs,
    )

    speedup = (ng_runtime / pulsim_runtime) if pulsim_runtime > 0.0 else None
    status = "passed"
    message = ""
    if max_error_threshold is not None and max_error is not None and max_error > max_error_threshold:
        status = "failed"
        message = f"max_error {max_error:.6e} > threshold {max_error_threshold:.6e}"

    return BenchmarkResult(
        benchmark_id=benchmark_id,
        scenario=scenario_name,
        status=status,
        message=message,
        pulsim_runtime_s=pulsim_runtime,
        ngspice_runtime_s=ng_runtime,
        pulsim_steps=pulsim_steps,
        ngspice_steps=ng_steps,
        speedup=speedup,
        max_error=max_error,
        rms_error=rms_error,
        observables=observables,
    )


def run_single_pair(
    pulsim_cli: Optional[Path],
    pulsim_netlist: Path,
    spice_netlist: Path,
    output_dir: Path,
    cli_observables: Optional[List[str]],
) -> List[BenchmarkResult]:
    netlist = load_yaml(pulsim_netlist)
    benchmark = netlist.get("benchmark", {})
    validation = benchmark.get("validation", {})
    benchmark_id = benchmark.get("id", pulsim_netlist.stem)
    observable_specs = resolve_observables(benchmark, validation, None, cli_observables)
    max_threshold = coerce_optional_float(benchmark.get("expectations", {}).get("metrics", {}).get("max_error"))

    if not observable_specs:
        observable_specs = [ObservableSpec(column="V(out)", spice_vector="v(out)")]

    result = run_case(
        pulsim_cli=pulsim_cli,
        scenario_netlist=netlist,
        benchmark_id=benchmark_id,
        scenario_name="default",
        spice_netlist_path=spice_netlist.resolve(),
        observable_specs=observable_specs,
        max_error_threshold=max_threshold,
        output_dir=output_dir,
    )
    return [result]


def run_manifest(
    pulsim_cli: Optional[Path],
    manifest_path: Path,
    output_dir: Path,
    only: Optional[List[str]],
    matrix: bool,
    force_scenario: Optional[str],
    cli_observables: Optional[List[str]],
) -> List[BenchmarkResult]:
    manifest = load_yaml(manifest_path)
    scenarios = manifest.get("scenarios", {})
    results: List[BenchmarkResult] = []

    for entry in manifest.get("benchmarks", []):
        circuit_path = (manifest_path.parent / entry["path"]).resolve()
        netlist = load_yaml(circuit_path)
        bench_meta = netlist.get("benchmark", {})
        benchmark_id = bench_meta.get("id", circuit_path.stem)
        if only and benchmark_id not in only:
            continue

        validation = bench_meta.get("validation", {})
        entry_observables = entry.get("ngspice_observables") if isinstance(entry.get("ngspice_observables"), list) else None
        observable_specs = resolve_observables(bench_meta, validation, entry_observables, cli_observables)
        max_threshold = coerce_optional_float(bench_meta.get("expectations", {}).get("metrics", {}).get("max_error"))

        spice_rel = entry.get("ngspice_netlist") or validation.get("spice_netlist")
        if force_scenario:
            scenario_names = [force_scenario]
        elif matrix:
            scenario_names = list(scenarios.keys())
        else:
            scenario_names = entry.get("scenarios", ["default"])
        if "default" in scenario_names and "default" not in scenarios:
            scenarios["default"] = {}

        if not spice_rel:
            for scenario_name in scenario_names:
                results.append(
                    BenchmarkResult(
                        benchmark_id=benchmark_id,
                        scenario=scenario_name,
                        status="skipped",
                        message="Missing ngspice netlist (set entry.ngspice_netlist or benchmark.validation.spice_netlist)",
                        pulsim_runtime_s=0.0,
                        ngspice_runtime_s=0.0,
                        pulsim_steps=0,
                        ngspice_steps=0,
                        speedup=None,
                        max_error=None,
                        rms_error=None,
                        observables=[],
                    )
                )
            continue

        spice_path = (manifest_path.parent / spice_rel).resolve()
        if not spice_path.exists():
            for scenario_name in scenario_names:
                results.append(
                    BenchmarkResult(
                        benchmark_id=benchmark_id,
                        scenario=scenario_name,
                        status="failed",
                        message=f"ngspice netlist not found: {spice_path}",
                        pulsim_runtime_s=0.0,
                        ngspice_runtime_s=0.0,
                        pulsim_steps=0,
                        ngspice_steps=0,
                        speedup=None,
                        max_error=None,
                        rms_error=None,
                        observables=[],
                    )
                )
            continue

        if not observable_specs:
            for scenario_name in scenario_names:
                results.append(
                    BenchmarkResult(
                        benchmark_id=benchmark_id,
                        scenario=scenario_name,
                        status="skipped",
                        message="No observables configured for ngspice comparison",
                        pulsim_runtime_s=0.0,
                        ngspice_runtime_s=0.0,
                        pulsim_steps=0,
                        ngspice_steps=0,
                        speedup=None,
                        max_error=None,
                        rms_error=None,
                        observables=[],
                    )
                )
            continue

        for scenario_name in scenario_names:
            scenario_override = scenarios.get(scenario_name, {})
            scenario_netlist = deep_merge(netlist, scenario_override)
            try:
                result = run_case(
                    pulsim_cli=pulsim_cli,
                    scenario_netlist=scenario_netlist,
                    benchmark_id=benchmark_id,
                    scenario_name=scenario_name,
                    spice_netlist_path=spice_path,
                    observable_specs=observable_specs,
                    max_error_threshold=max_threshold,
                    output_dir=output_dir,
                )
            except Exception as exc:  # pragma: no cover - runtime dependent
                result = BenchmarkResult(
                    benchmark_id=benchmark_id,
                    scenario=scenario_name,
                    status="failed",
                    message=str(exc),
                    pulsim_runtime_s=0.0,
                    ngspice_runtime_s=0.0,
                    pulsim_steps=0,
                    ngspice_steps=0,
                    speedup=None,
                    max_error=None,
                    rms_error=None,
                    observables=[],
                )
            results.append(result)

    return results


def write_results(output_dir: Path, results: List[BenchmarkResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ngspice_results.csv"
    json_path = output_dir / "ngspice_results.json"
    summary_path = output_dir / "ngspice_summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "benchmark_id",
                "scenario",
                "status",
                "pulsim_runtime_s",
                "ngspice_runtime_s",
                "speedup",
                "pulsim_steps",
                "ngspice_steps",
                "max_error",
                "rms_error",
                "message",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.benchmark_id,
                    item.scenario,
                    item.status,
                    f"{item.pulsim_runtime_s:.6f}",
                    f"{item.ngspice_runtime_s:.6f}",
                    "" if item.speedup is None else f"{item.speedup:.6f}",
                    item.pulsim_steps,
                    item.ngspice_steps,
                    "" if item.max_error is None else f"{item.max_error:.6e}",
                    "" if item.rms_error is None else f"{item.rms_error:.6e}",
                    item.message,
                ]
            )

    payload = {
        "results": [
            {
                **asdict(item),
                "observables": [asdict(obs) for obs in item.observables],
            }
            for item in results
        ]
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    summary = {
        "passed": sum(1 for item in results if item.status == "passed"),
        "failed": sum(1 for item in results if item.status == "failed"),
        "skipped": sum(1 for item in results if item.status == "skipped"),
        "total": len(results),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Pulsim and ngspice using equivalent circuits")
    parser.add_argument("--benchmarks", type=Path, default=Path(__file__).with_name("benchmarks.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/ngspice_out"))
    parser.add_argument("--pulsim-cli", type=Path, default=None)
    parser.add_argument("--only", nargs="*", help="Benchmark ids to run (manifest mode)")
    parser.add_argument("--matrix", action="store_true", help="Run all scenarios from manifest")
    parser.add_argument("--scenario", type=str, default=None, help="Run only one scenario name")
    parser.add_argument("--observable", action="append", default=None, help="Observable column to compare (repeatable)")
    parser.add_argument("--pulsim-netlist", type=Path, default=None, help="Single mode: Pulsim YAML netlist")
    parser.add_argument("--spice-netlist", type=Path, default=None, help="Single mode: SPICE netlist")
    args = parser.parse_args()

    if yaml is None:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml")
    if not check_ngspice():
        raise SystemExit("ngspice not found. Install it before running this comparison.")

    pulsim_cli = args.pulsim_cli or find_pulsim_cli()
    if pulsim_cli is None and not can_use_pulsim_python_backend():
        raise SystemExit(
            "Pulsim CLI not found and Python backend unavailable. "
            "Build the project, pass --pulsim-cli, or install the pulsim Python package."
        )
    if pulsim_cli is None:
        print("Pulsim CLI not found. Using Python API backend.")

    if args.pulsim_netlist is not None or args.spice_netlist is not None:
        if args.pulsim_netlist is None or args.spice_netlist is None:
            raise SystemExit("Single mode requires both --pulsim-netlist and --spice-netlist")
        results = run_single_pair(
            pulsim_cli=pulsim_cli,
            pulsim_netlist=args.pulsim_netlist.resolve(),
            spice_netlist=args.spice_netlist.resolve(),
            output_dir=args.output_dir,
            cli_observables=args.observable,
        )
    else:
        results = run_manifest(
            pulsim_cli=pulsim_cli,
            manifest_path=args.benchmarks.resolve(),
            output_dir=args.output_dir,
            only=args.only,
            matrix=args.matrix,
            force_scenario=args.scenario,
            cli_observables=args.observable,
        )

    write_results(args.output_dir, results)
    summary = {
        "passed": sum(1 for item in results if item.status == "passed"),
        "failed": sum(1 for item in results if item.status == "failed"),
        "skipped": sum(1 for item in results if item.status == "skipped"),
        "total": len(results),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
