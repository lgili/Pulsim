#!/usr/bin/env python3
"""Tiered stress validation runner for Pulsim benchmark suites."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from benchmark_runner import ScenarioResult, load_yaml, run_benchmarks, yaml

STRESS_SCHEMA_VERSION = "pulsim-stress-v1"


@dataclass
class TierCriteria:
    min_pass_rate: float = 1.0
    max_runtime_s: Optional[float] = None
    max_max_error: Optional[float] = None
    max_timestep_rejections: Optional[float] = None
    required_telemetry: Tuple[str, ...] = ()


@dataclass
class TierEvaluation:
    tier: str
    status: str
    message: str
    total: int
    passed: int
    failed: int
    skipped: int
    pass_rate: float
    max_runtime_s_observed: Optional[float]
    max_max_error_observed: Optional[float]
    max_timestep_rejections_observed: Optional[float]
    missing_telemetry_rows: int


@dataclass
class TierRunResult:
    tier: str
    description: str
    criteria: TierCriteria
    evaluation: TierEvaluation
    results: List[ScenarioResult]


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def parse_tier_criteria(raw: Dict[str, Any]) -> TierCriteria:
    required_raw = raw.get("required_telemetry", [])
    required: Tuple[str, ...]
    if isinstance(required_raw, list):
        required = tuple(str(item).strip() for item in required_raw if str(item).strip())
    else:
        required = ()

    return TierCriteria(
        min_pass_rate=float(raw.get("min_pass_rate", 1.0)),
        max_runtime_s=_coerce_optional_float(raw.get("max_runtime_s")),
        max_max_error=_coerce_optional_float(raw.get("max_max_error")),
        max_timestep_rejections=_coerce_optional_float(raw.get("max_timestep_rejections")),
        required_telemetry=required,
    )


def build_benchmark_index(benchmarks_manifest_path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    manifest = load_yaml(benchmarks_manifest_path)
    index: Dict[str, Dict[str, Any]] = {}

    for entry in manifest.get("benchmarks", []):
        circuit_rel = Path(entry["path"])
        circuit_path = (benchmarks_manifest_path.parent / circuit_rel).resolve()
        netlist = load_yaml(circuit_path)
        benchmark_meta = netlist.get("benchmark", {})
        benchmark_id = benchmark_meta.get("id", circuit_path.stem)
        index[benchmark_id] = {
            "entry": entry,
            "circuit_path": circuit_path,
            "benchmark_id": benchmark_id,
        }

    return manifest, index


def build_tier_manifest(
    base_manifest: Dict[str, Any],
    benchmark_index: Dict[str, Dict[str, Any]],
    tier_cases: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    selected_entries: List[Dict[str, Any]] = []
    required_scenarios: Dict[str, Dict[str, Any]] = {}
    base_scenarios = base_manifest.get("scenarios", {})

    for case in tier_cases:
        benchmark_id = str(case.get("benchmark_id", "")).strip()
        if not benchmark_id:
            raise ValueError("Tier case missing benchmark_id")
        if benchmark_id not in benchmark_index:
            raise ValueError(f"Tier case references unknown benchmark_id: {benchmark_id}")

        base_entry = dict(benchmark_index[benchmark_id]["entry"])
        base_entry["path"] = str(benchmark_index[benchmark_id]["circuit_path"])
        scenarios = case.get("scenarios")
        if isinstance(scenarios, list) and scenarios:
            case_scenarios = [str(item).strip() for item in scenarios if str(item).strip()]
            base_entry["scenarios"] = case_scenarios
        else:
            case_scenarios = list(base_entry.get("scenarios", ["default"]))
        selected_entries.append(base_entry)

        for scenario_name in case_scenarios:
            if scenario_name == "default":
                continue
            if scenario_name not in base_scenarios:
                raise ValueError(
                    f"Tier case benchmark '{benchmark_id}' references undefined scenario '{scenario_name}'"
                )
            required_scenarios[scenario_name] = base_scenarios[scenario_name]

    if any("default" in entry.get("scenarios", []) for entry in selected_entries):
        required_scenarios.setdefault("default", {})

    return {
        "benchmarks": selected_entries,
        "scenarios": required_scenarios,
    }


def evaluate_tier_results(
    tier: str,
    results: Sequence[ScenarioResult],
    criteria: TierCriteria,
) -> TierEvaluation:
    total = len(results)
    passed = sum(1 for item in results if item.status == "passed")
    failed = sum(1 for item in results if item.status == "failed")
    skipped = sum(1 for item in results if item.status == "skipped")
    pass_rate = (float(passed) / float(total)) if total else 0.0

    max_runtime = max((item.runtime_s for item in results), default=None)
    max_error_values = [item.max_error for item in results if item.max_error is not None]
    max_error = max(max_error_values) if max_error_values else None

    rejection_values: List[float] = []
    missing_telemetry_rows = 0
    for item in results:
        telemetry = item.telemetry or {}
        for key in criteria.required_telemetry:
            if telemetry.get(key) is None:
                missing_telemetry_rows += 1
                break
        value = telemetry.get("timestep_rejections")
        if value is not None:
            rejection_values.append(float(value))
    max_rejections = max(rejection_values) if rejection_values else None

    failures: List[str] = []
    if total == 0:
        failures.append("tier has zero executed scenarios")
    if pass_rate < criteria.min_pass_rate:
        failures.append(f"pass_rate={pass_rate:.3f} < min_pass_rate={criteria.min_pass_rate:.3f}")
    if criteria.max_runtime_s is not None and max_runtime is not None and max_runtime > criteria.max_runtime_s:
        failures.append(f"max_runtime_s={max_runtime:.6f} > {criteria.max_runtime_s:.6f}")
    if criteria.max_max_error is not None and max_error is not None and max_error > criteria.max_max_error:
        failures.append(f"max_error={max_error:.6e} > {criteria.max_max_error:.6e}")
    if (
        criteria.max_timestep_rejections is not None
        and max_rejections is not None
        and max_rejections > criteria.max_timestep_rejections
    ):
        failures.append(
            f"max_timestep_rejections={max_rejections:.2f} > {criteria.max_timestep_rejections:.2f}"
        )
    if missing_telemetry_rows > 0:
        failures.append(f"missing required telemetry in {missing_telemetry_rows} result rows")

    status = "passed" if not failures else "failed"
    message = "; ".join(failures) if failures else "tier criteria satisfied"
    return TierEvaluation(
        tier=tier,
        status=status,
        message=message,
        total=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        pass_rate=pass_rate,
        max_runtime_s_observed=max_runtime,
        max_max_error_observed=max_error,
        max_timestep_rejections_observed=max_rejections,
        missing_telemetry_rows=missing_telemetry_rows,
    )


def run_stress_suite(
    benchmarks_manifest_path: Path,
    stress_catalog_path: Path,
    output_dir: Path,
    selected_tiers: Optional[Sequence[str]] = None,
) -> List[TierRunResult]:
    catalog = load_yaml(stress_catalog_path)
    tiers = catalog.get("tiers", {})
    if not isinstance(tiers, dict) or not tiers:
        raise RuntimeError("Stress catalog has no tiers")

    base_manifest, benchmark_index = build_benchmark_index(benchmarks_manifest_path)
    requested = set(selected_tiers) if selected_tiers else None

    tier_results: List[TierRunResult] = []
    for tier_name, tier_data in tiers.items():
        if requested is not None and tier_name not in requested:
            continue
        if not isinstance(tier_data, dict):
            raise RuntimeError(f"Tier '{tier_name}' must be a mapping")

        description = str(tier_data.get("description", "")).strip()
        criteria = parse_tier_criteria(tier_data.get("criteria", {}))
        cases = tier_data.get("cases", [])
        if not isinstance(cases, list) or not cases:
            raise RuntimeError(f"Tier '{tier_name}' has no cases")

        tier_manifest = build_tier_manifest(base_manifest, benchmark_index, cases)
        if yaml is None:
            raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")

        # Keep the temporary manifest under the original benchmarks directory so
        # benchmark-relative assets (for example, baselines/*.csv) still resolve.
        tmp_manifest_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=f"_{tier_name}.yaml",
                prefix="stress_",
                dir=str(benchmarks_manifest_path.parent),
                delete=False,
                encoding="utf-8",
            ) as handle:
                yaml.safe_dump(tier_manifest, handle, sort_keys=False)
                tmp_manifest_path = Path(handle.name)

            results = run_benchmarks(
                benchmarks_path=tmp_manifest_path,
                output_dir=output_dir / "tiers" / tier_name,
                selected=None,
                matrix=False,
                generate_baselines=False,
            )
        finally:
            if tmp_manifest_path is not None:
                tmp_manifest_path.unlink(missing_ok=True)

        evaluation = evaluate_tier_results(tier=tier_name, results=results, criteria=criteria)
        tier_results.append(
            TierRunResult(
                tier=tier_name,
                description=description,
                criteria=criteria,
                evaluation=evaluation,
                results=results,
            )
        )

    return tier_results


def write_stress_artifacts(output_dir: Path, tier_runs: Sequence[TierRunResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stress_results.csv"
    json_path = output_dir / "stress_results.json"
    summary_path = output_dir / "stress_summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tier",
                "benchmark_id",
                "scenario",
                "status",
                "runtime_s",
                "steps",
                "max_error",
                "rms_error",
                "newton_iterations",
                "timestep_rejections",
                "linear_fallbacks",
                "message",
            ]
        )
        for tier_run in tier_runs:
            for result in tier_run.results:
                telemetry = result.telemetry or {}
                writer.writerow(
                    [
                        tier_run.tier,
                        result.benchmark_id,
                        result.scenario,
                        result.status,
                        f"{result.runtime_s:.6f}",
                        result.steps,
                        "" if result.max_error is None else f"{result.max_error:.6e}",
                        "" if result.rms_error is None else f"{result.rms_error:.6e}",
                        "" if telemetry.get("newton_iterations") is None else f"{telemetry['newton_iterations']:.6f}",
                        ""
                        if telemetry.get("timestep_rejections") is None
                        else f"{telemetry['timestep_rejections']:.6f}",
                        "" if telemetry.get("linear_fallbacks") is None else f"{telemetry['linear_fallbacks']:.6f}",
                        result.message,
                    ]
                )

    payload = {
        "schema_version": STRESS_SCHEMA_VERSION,
        "tiers": [],
    }
    for tier_run in tier_runs:
        payload["tiers"].append(
            {
                "tier": tier_run.tier,
                "description": tier_run.description,
                "criteria": asdict(tier_run.criteria),
                "evaluation": asdict(tier_run.evaluation),
                "results": [asdict(item) for item in tier_run.results],
            }
        )

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    summary = {
        "schema_version": STRESS_SCHEMA_VERSION,
        "tiers_total": len(tier_runs),
        "tiers_passed": sum(1 for item in tier_runs if item.evaluation.status == "passed"),
        "tiers_failed": sum(1 for item in tier_runs if item.evaluation.status == "failed"),
        "overall_status": "passed"
        if all(item.evaluation.status == "passed" for item in tier_runs)
        else "failed",
        "tier_evaluations": {item.tier: asdict(item.evaluation) for item in tier_runs},
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pulsim tiered stress validation suite")
    parser.add_argument("--benchmarks", type=Path, default=Path(__file__).with_name("benchmarks.yaml"))
    parser.add_argument("--catalog", type=Path, default=Path(__file__).with_name("stress_catalog.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/stress_out"))
    parser.add_argument("--tier", action="append", default=None, help="Run only selected tier (repeatable)")
    args = parser.parse_args()

    tier_runs = run_stress_suite(
        benchmarks_manifest_path=args.benchmarks.resolve(),
        stress_catalog_path=args.catalog.resolve(),
        output_dir=args.output_dir,
        selected_tiers=args.tier,
    )
    write_stress_artifacts(args.output_dir, tier_runs)

    summary = {
        "schema_version": STRESS_SCHEMA_VERSION,
        "tiers_total": len(tier_runs),
        "tiers_passed": sum(1 for item in tier_runs if item.evaluation.status == "passed"),
        "tiers_failed": sum(1 for item in tier_runs if item.evaluation.status == "failed"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
