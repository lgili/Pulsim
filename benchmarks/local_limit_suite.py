#!/usr/bin/env python3
"""Local limit suite runner (fixed + variable) for stress discovery on one machine."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import benchmark_runner as br

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class LocalLimitResult:
    benchmark_id: str
    difficulty: str
    scenario: str
    status: str
    runtime_s: float
    steps: int
    final_time_s: Optional[float]
    tstop_target_s: Optional[float]
    completion_ratio: Optional[float]
    finite_samples: bool
    message: str
    telemetry: Dict[str, Optional[float]]


def _parse_optional_real(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return br.parse_value(value)
    except Exception:
        return None


def _scenario_filter_for_mode(mode: str) -> List[str]:
    if mode == "fixed":
        return ["fixed_long"]
    if mode == "variable":
        return ["variable_long"]
    return ["fixed_long", "variable_long"]


def _materialize_manifest(
    manifest_path: Path,
    work_root: Path,
    duration_scale: float,
) -> Tuple[Path, Dict[Tuple[str, str], Optional[float]], Dict[str, str]]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")

    manifest = br.load_yaml(manifest_path)
    scenarios = manifest.get("scenarios", {})
    entries = list(manifest.get("benchmarks", []))

    out_manifest: Dict[str, Any] = {
        "benchmarks": [],
        "scenarios": scenarios,
    }

    expected_tstop: Dict[Tuple[str, str], Optional[float]] = {}
    difficulty_by_id: Dict[str, str] = {}

    circuits_dir = work_root / "circuits"
    circuits_dir.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        src_rel = entry.get("path")
        if not src_rel:
            continue
        src_path = (manifest_path.parent / src_rel).resolve()
        netlist = br.load_yaml(src_path)

        benchmark_meta = netlist.get("benchmark", {}) if isinstance(netlist.get("benchmark"), dict) else {}
        benchmark_id = benchmark_meta.get("id", src_path.stem)

        simulation = netlist.get("simulation")
        if not isinstance(simulation, dict):
            simulation = {}
            netlist["simulation"] = simulation

        if duration_scale <= 0.0:
            raise ValueError("duration-scale must be > 0")
        if simulation.get("tstop") is not None:
            tstop_value = br.parse_value(simulation.get("tstop"))
            simulation["tstop"] = float(tstop_value * duration_scale)

        dst_rel = Path("circuits") / src_path.name
        dst_path = work_root / dst_rel
        with open(dst_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(netlist, handle, sort_keys=False)

        out_entry = dict(entry)
        out_entry["path"] = dst_rel.as_posix()
        out_manifest["benchmarks"].append(out_entry)

        difficulty = str(entry.get("difficulty", "unclassified"))
        difficulty_by_id[str(benchmark_id)] = difficulty

        scenario_names = out_entry.get("scenarios", ["default"])
        if "default" in scenario_names and "default" not in scenarios:
            scenarios["default"] = {}

        for scenario_name in scenario_names:
            scenario_override = scenarios.get(scenario_name, {})
            scenario_netlist = br.deep_merge(netlist, scenario_override)
            br.apply_runtime_defaults(scenario_netlist)
            scenario_sim = scenario_netlist.get("simulation", {})
            tstop = _parse_optional_real(scenario_sim.get("tstop") if isinstance(scenario_sim, dict) else None)
            expected_tstop[(str(benchmark_id), str(scenario_name))] = tstop

    materialized_manifest_path = work_root / "benchmarks_local_limit_materialized.yaml"
    with open(materialized_manifest_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(out_manifest, handle, sort_keys=False)

    return materialized_manifest_path, expected_tstop, difficulty_by_id


def _inspect_output_csv(path: Path) -> Tuple[bool, int, Optional[float], str]:
    if not path.exists():
        return False, 0, None, f"missing output CSV: {path}"

    sample_count = 0
    final_time: Optional[float] = None
    prev_time: Optional[float] = None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "time" not in reader.fieldnames:
                return False, 0, None, "output CSV missing time column"

            for row_idx, row in enumerate(reader, start=2):
                raw_time = row.get("time")
                if raw_time is None:
                    return False, sample_count, final_time, f"row {row_idx}: missing time value"
                time_value = float(raw_time)
                if not math.isfinite(time_value):
                    return False, sample_count, final_time, f"row {row_idx}: non-finite time value"
                if prev_time is not None and time_value < prev_time:
                    return False, sample_count, final_time, f"row {row_idx}: non-monotonic time sequence"

                for key, raw in row.items():
                    if key == "time" or raw is None or raw == "":
                        continue
                    value = float(raw)
                    if not math.isfinite(value):
                        return False, sample_count, final_time, f"row {row_idx}: non-finite value at column '{key}'"

                prev_time = time_value
                final_time = time_value
                sample_count += 1

    except Exception as exc:
        return False, sample_count, final_time, f"failed to parse output CSV: {exc}"

    return True, sample_count, final_time, ""


def _evaluate_results(
    base_results: List[br.ScenarioResult],
    output_dir: Path,
    expected_tstop: Dict[Tuple[str, str], Optional[float]],
    difficulty_by_id: Dict[str, str],
    min_samples: int,
    min_completion: float,
    max_runtime_s: Optional[float],
) -> List[LocalLimitResult]:
    evaluated: List[LocalLimitResult] = []

    for item in base_results:
        benchmark_id = str(item.benchmark_id)
        scenario = str(item.scenario)
        difficulty = difficulty_by_id.get(benchmark_id, "unclassified")
        csv_path = output_dir / "outputs" / benchmark_id / scenario / "pulsim.csv"
        tstop_target = expected_tstop.get((benchmark_id, scenario))

        status = item.status
        message = item.message
        finite_samples = False
        final_time_s: Optional[float] = None
        completion_ratio: Optional[float] = None

        if status == "passed":
            finite_samples, sample_count, final_time_s, inspect_msg = _inspect_output_csv(csv_path)
            if not finite_samples:
                status = "failed"
                message = inspect_msg
            elif sample_count < min_samples:
                status = "failed"
                message = f"too few samples: {sample_count} < {min_samples}"

            if status == "passed" and tstop_target is not None and tstop_target > 0.0 and final_time_s is not None:
                completion_ratio = final_time_s / tstop_target
                if completion_ratio < min_completion:
                    status = "failed"
                    message = (
                        f"insufficient time coverage: ratio {completion_ratio:.4f} < {min_completion:.4f}"
                    )

            if status == "passed" and max_runtime_s is not None and item.runtime_s > max_runtime_s:
                status = "failed"
                message = f"runtime {item.runtime_s:.6f}s exceeds limit {max_runtime_s:.6f}s"

        evaluated.append(
            LocalLimitResult(
                benchmark_id=benchmark_id,
                difficulty=difficulty,
                scenario=scenario,
                status=status,
                runtime_s=float(item.runtime_s),
                steps=int(item.steps),
                final_time_s=final_time_s,
                tstop_target_s=tstop_target,
                completion_ratio=completion_ratio,
                finite_samples=finite_samples,
                message=message,
                telemetry=dict(item.telemetry),
            )
        )

    return evaluated


def _build_summary(results: List[LocalLimitResult]) -> Dict[str, Any]:
    by_status = Counter(item.status for item in results)
    by_scenario = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0, "avg_runtime_s": 0.0})
    by_difficulty = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0})
    failure_reasons = Counter()

    for item in results:
        s = by_scenario[item.scenario]
        s["total"] += 1
        s[item.status] = s.get(item.status, 0) + 1
        s["avg_runtime_s"] += item.runtime_s

        d = by_difficulty[item.difficulty]
        d["total"] += 1
        d[item.status] = d.get(item.status, 0) + 1

        if item.status != "passed":
            failure_reasons[item.message or "unspecified failure"] += 1

    for value in by_scenario.values():
        total = int(value.get("total", 0))
        if total > 0:
            value["avg_runtime_s"] = float(value["avg_runtime_s"]) / float(total)

    return {
        "passed": by_status.get("passed", 0),
        "failed": by_status.get("failed", 0),
        "total": len(results),
        "per_scenario": dict(sorted(by_scenario.items())),
        "per_difficulty": dict(sorted(by_difficulty.items())),
        "failure_reasons": failure_reasons.most_common(),
    }


def _write_outputs(output_dir: Path, results: List[LocalLimitResult], summary: Dict[str, Any], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_csv = output_dir / "results.csv"
    rows_json = output_dir / "results.json"
    summary_json = output_dir / "summary.json"

    with open(rows_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "benchmark_id",
                "difficulty",
                "scenario",
                "status",
                "runtime_s",
                "steps",
                "final_time_s",
                "tstop_target_s",
                "completion_ratio",
                "finite_samples",
                "message",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.benchmark_id,
                    item.difficulty,
                    item.scenario,
                    item.status,
                    f"{item.runtime_s:.9f}",
                    item.steps,
                    "" if item.final_time_s is None else f"{item.final_time_s:.9e}",
                    "" if item.tstop_target_s is None else f"{item.tstop_target_s:.9e}",
                    "" if item.completion_ratio is None else f"{item.completion_ratio:.9f}",
                    int(item.finite_samples),
                    item.message,
                ]
            )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "duration_scale": args.duration_scale,
        "min_samples": args.min_samples,
        "min_completion": args.min_completion,
        "max_runtime_s": args.max_runtime_s,
        "results": [asdict(item) for item in results],
    }

    with open(rows_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a local fixed+variable limit suite over progressive circuits"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent / "local_limit" / "benchmarks_local_limit.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/out_local_limit"))
    parser.add_argument("--only", nargs="*", help="Benchmark ids to run")
    parser.add_argument("--mode", choices=["fixed", "variable", "both"], default="both")
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Multiply each circuit simulation.tstop by this factor",
    )
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--min-completion", type=float, default=0.97)
    parser.add_argument("--max-runtime-s", type=float, default=None)
    parser.add_argument("--list-circuits", action="store_true", help="List benchmark ids and exit")
    args = parser.parse_args()

    if yaml is None:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml")

    if not br.can_use_pulsim_python_backend():
        raise SystemExit(
            "Pulsim Python runtime backend unavailable. Build bindings and expose build/python on PYTHONPATH."
        )

    scenario_filter = _scenario_filter_for_mode(args.mode)

    with tempfile.TemporaryDirectory(prefix="pulsim_local_limit_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        manifest_path, expected_tstop, difficulty_by_id = _materialize_manifest(
            args.manifest,
            tmp_root,
            args.duration_scale,
        )

        if args.list_circuits:
            listed = sorted(difficulty_by_id.items(), key=lambda item: item[0])
            for benchmark_id, difficulty in listed:
                print(f"{benchmark_id}: {difficulty}")
            return 0

        base_results = br.run_benchmarks(
            manifest_path,
            args.output_dir,
            selected=args.only,
            scenario_filter=scenario_filter,
        )

    evaluated = _evaluate_results(
        base_results,
        args.output_dir,
        expected_tstop,
        difficulty_by_id,
        min_samples=max(2, args.min_samples),
        min_completion=args.min_completion,
        max_runtime_s=args.max_runtime_s,
    )

    summary = _build_summary(evaluated)
    _write_outputs(args.output_dir, evaluated, summary, args)

    print(json.dumps(summary, indent=2))

    if summary.get("failed", 0) > 0:
        print("\nTop failure reasons:")
        for reason, count in summary.get("failure_reasons", [])[:10]:
            print(f"- ({count}) {reason}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
