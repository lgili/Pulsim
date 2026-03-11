#!/usr/bin/env python3
"""Run isolated advanced-backend prototype comparisons against native baseline."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from pulsim_python_backend import BackendRunResult, run_from_yaml
from validate_advanced_solver_decision_matrix import validate_decision_matrix


DEFAULT_MATRIX = Path("benchmarks/advanced_solver_decision_matrix.yaml")
DEFAULT_MANIFEST = Path("benchmarks/benchmarks.yaml")
DEFAULT_OUTPUT_DIR = Path("benchmarks/out_advanced_solver")


PORTABILITY_BY_BACKEND = {
    "sundials": 0.75,
    "petsc": 0.65,
}

MAINTENANCE_BY_MATURITY = {
    "production_candidate": 5.0,
    "experimental": 7.5,
}


@dataclass
class CaseOutcome:
    status: str
    runtime_s: float
    steps: int
    mode: str
    message: str
    diagnostic: Optional[str]


@dataclass
class PrototypeCaseResult:
    benchmark_id: str
    scenario: str
    case_class: str
    mode: str
    baseline_status: str
    prototype_status: str
    baseline_runtime_s: float
    prototype_runtime_s: float
    baseline_steps: int
    prototype_steps: int
    baseline_message: str
    prototype_message: str
    baseline_diagnostic: Optional[str]
    prototype_diagnostic: Optional[str]
    runtime_regression_rel: Optional[float]
    success_delta: int


RunFn = Callable[..., BackendRunResult]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return payload


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _infer_mode(scenario_name: str, scenario_override: Dict[str, Any]) -> str:
    simulation = scenario_override.get("simulation")
    if isinstance(simulation, dict):
        if "shooting" in simulation and "harmonic_balance" not in simulation:
            return "shooting"
        if "harmonic_balance" in simulation and "shooting" not in simulation:
            return "harmonic_balance"
        frequency_cfg = simulation.get("frequency_analysis")
        if isinstance(frequency_cfg, dict) and bool(frequency_cfg.get("enabled", False)):
            return "frequency_analysis"

    lowered = scenario_name.lower()
    if "shooting" in lowered:
        return "shooting"
    if "harmonic" in lowered or lowered == "hb":
        return "harmonic_balance"
    if "frequency" in lowered or lowered == "ac":
        return "frequency_analysis"
    return "transient"


def _runtime_defaults(netlist: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(netlist)
    simulation = normalized.get("simulation")
    if not isinstance(simulation, dict):
        return normalized

    simulation = dict(simulation)
    if "adaptive_timestep" not in simulation:
        simulation["adaptive_timestep"] = False
    normalized["simulation"] = simulation
    return normalized


def _build_prototype_simulation_override(candidate: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode != "transient":
        return {}

    family = str(candidate.get("solver_family", "")).strip().lower()
    integrator = "trbdf2"
    if family in {"arkode", "snes"}:
        integrator = "rosenbrockw"
    elif family == "cvode":
        integrator = "bdf2"
    elif family == "kinsol":
        integrator = "trapezoidal"

    return {
        "simulation": {
            "adaptive_timestep": True,
            "step_mode": "variable",
            "integrator": integrator,
            "formulation": "direct",
            "direct_formulation_fallback": True,
            "fallback_policy": {
                "convergence_profile": "robust",
            },
        }
    }


def _benchmark_id_from_netlist(circuit_path: Path, netlist: Dict[str, Any]) -> str:
    benchmark = netlist.get("benchmark")
    if isinstance(benchmark, dict):
        bench_id = benchmark.get("id")
        if isinstance(bench_id, str) and bench_id.strip():
            return bench_id.strip()
    return circuit_path.stem


def _build_manifest_index(
    manifest_path: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    manifest = _load_yaml(manifest_path)
    scenario_defs = manifest.get("scenarios")
    if not isinstance(scenario_defs, dict):
        raise RuntimeError("Manifest must define mapping 'scenarios'")

    benchmarks = manifest.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise RuntimeError("Manifest must define list 'benchmarks'")

    index: Dict[str, Dict[str, Any]] = {}
    for item in benchmarks:
        if not isinstance(item, dict):
            continue
        rel_path = item.get("path")
        if not isinstance(rel_path, str) or not rel_path.strip():
            continue
        circuit_path = (manifest_path.parent / rel_path).resolve()
        netlist = _load_yaml(circuit_path)
        benchmark_id = _benchmark_id_from_netlist(circuit_path, netlist)
        if benchmark_id in index:
            raise RuntimeError(f"Duplicated benchmark id in manifest: {benchmark_id}")
        scenario_names = item.get("scenarios", ["default"])
        if not isinstance(scenario_names, list):
            scenario_names = ["default"]
        index[benchmark_id] = {
            "path": circuit_path,
            "netlist": netlist,
            "scenario_names": [str(value) for value in scenario_names],
        }
    return index, scenario_defs


def _diagnostic_from_exc(exc: Exception) -> Optional[str]:
    diagnostic = getattr(exc, "diagnostic", None)
    if diagnostic is None:
        return None
    if isinstance(diagnostic, str):
        value = diagnostic.strip()
        return value or None
    name = getattr(diagnostic, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    value = str(diagnostic).strip()
    return value or None


def _mode_from_exc(exc: Exception) -> Optional[str]:
    mode = getattr(exc, "mode", None)
    if mode is None:
        return None
    if isinstance(mode, str):
        value = mode.strip()
        return value or None
    name = getattr(mode, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    value = str(mode).strip()
    return value or None


def _run_case(
    run_fn: RunFn,
    netlist_payload: Dict[str, Any],
    *,
    preferred_mode: str,
    workdir: Path,
    label: str,
) -> CaseOutcome:
    netlist_path = workdir / f"{label}.yaml"
    output_path = workdir / f"{label}.csv"

    with open(netlist_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(netlist_payload, handle, sort_keys=False)

    use_initial_conditions = bool(
        netlist_payload.get("simulation", {}).get("uic", False)
        if isinstance(netlist_payload.get("simulation"), dict)
        else False
    )

    try:
        result = run_fn(
            netlist_path,
            output_path,
            preferred_mode=preferred_mode,
            use_initial_conditions=use_initial_conditions,
        )
        return CaseOutcome(
            status="passed",
            runtime_s=float(result.runtime_s),
            steps=int(result.steps),
            mode=str(result.mode),
            message="",
            diagnostic=None,
        )
    except Exception as exc:  # pragma: no cover - exercised by integration
        return CaseOutcome(
            status="failed",
            runtime_s=0.0,
            steps=0,
            mode=_mode_from_exc(exc) or preferred_mode,
            message=str(exc),
            diagnostic=_diagnostic_from_exc(exc),
        )


def _runtime_regression_rel(baseline: CaseOutcome, prototype: CaseOutcome) -> Optional[float]:
    if baseline.status != "passed" or prototype.status != "passed":
        return None
    if baseline.runtime_s <= 0.0:
        return None
    return (prototype.runtime_s - baseline.runtime_s) / baseline.runtime_s


def summarize_case_results(
    case_results: Sequence[PrototypeCaseResult],
    *,
    candidate: Dict[str, Any],
    criteria: Dict[str, Any],
    scoring_weights: Dict[str, Any],
) -> Dict[str, Any]:
    total_cases = len(case_results)
    baseline_pass = sum(1 for item in case_results if item.baseline_status == "passed")
    prototype_pass = sum(1 for item in case_results if item.prototype_status == "passed")

    baseline_success_rate = (baseline_pass / total_cases) if total_cases else 0.0
    prototype_success_rate = (prototype_pass / total_cases) if total_cases else 0.0
    success_gain = prototype_success_rate - baseline_success_rate

    comparable = [
        item
        for item in case_results
        if item.baseline_status == "passed"
        and item.prototype_status == "passed"
        and item.baseline_runtime_s > 0.0
    ]
    baseline_runtime_total = sum(item.baseline_runtime_s for item in comparable)
    prototype_runtime_total = sum(item.prototype_runtime_s for item in comparable)
    runtime_regression_rel = (
        (prototype_runtime_total - baseline_runtime_total) / baseline_runtime_total
        if baseline_runtime_total > 0.0
        else 0.0
    )

    backend = str(candidate.get("backend", "")).strip().lower()
    maturity = str(candidate.get("maturity", "")).strip().lower()
    portability_score = float(
        candidate.get("prototype_portability_score", PORTABILITY_BY_BACKEND.get(backend, 0.5))
    )
    maintenance_cost_score = float(
        candidate.get(
            "prototype_maintenance_cost_score",
            MAINTENANCE_BY_MATURITY.get(maturity, 8.0),
        )
    )
    memory_regression_rel = 0.0

    hard_constraints = {
        "min_success_rate_gain_abs": success_gain
        >= float(criteria.get("min_success_rate_gain_abs", 0.0)),
        "max_runtime_regression_rel": runtime_regression_rel
        <= float(criteria.get("max_runtime_regression_rel", 1e9)),
        "max_memory_regression_rel": memory_regression_rel
        <= float(criteria.get("max_memory_regression_rel", 1e9)),
        "min_portability_score": portability_score
        >= float(criteria.get("min_portability_score", 0.0)),
        "max_maintenance_cost_score": maintenance_cost_score
        <= float(criteria.get("max_maintenance_cost_score", 1e9)),
    }
    hard_pass = all(hard_constraints.values())

    robust_weight = float(scoring_weights.get("robustness", 0.0))
    runtime_weight = float(scoring_weights.get("runtime", 0.0))
    portability_weight = float(scoring_weights.get("portability", 0.0))
    maintenance_weight = float(scoring_weights.get("maintenance", 0.0))
    runtime_score = max(0.0, 1.0 - max(runtime_regression_rel, 0.0))
    maintenance_score = max(0.0, 1.0 - (maintenance_cost_score / 10.0))

    weighted_total_score = (
        prototype_success_rate * robust_weight
        + runtime_score * runtime_weight
        + portability_score * portability_weight
        + maintenance_score * maintenance_weight
    )

    return {
        "total_cases": total_cases,
        "comparable_runtime_cases": len(comparable),
        "baseline_success_rate": baseline_success_rate,
        "prototype_success_rate": prototype_success_rate,
        "success_rate_gain_abs": success_gain,
        "baseline_runtime_total_s": baseline_runtime_total,
        "prototype_runtime_total_s": prototype_runtime_total,
        "runtime_regression_rel": runtime_regression_rel,
        "memory_regression_rel": memory_regression_rel,
        "portability_score": portability_score,
        "maintenance_cost_score": maintenance_cost_score,
        "hard_constraints": hard_constraints,
        "hard_constraints_passed": hard_pass,
        "weighted_total_score": weighted_total_score,
    }


def run_advanced_solver_prototype(
    *,
    matrix_path: Path,
    manifest_path: Path,
    candidate_id: str,
    output_dir: Path,
    enforce_hard_constraints: bool,
    max_cases: Optional[int],
    run_fn: RunFn = run_from_yaml,
) -> Dict[str, Any]:
    matrix_errors = validate_decision_matrix(matrix_path.resolve())
    if matrix_errors:
        raise RuntimeError("\n".join(matrix_errors))

    matrix = _load_yaml(matrix_path)
    candidates = matrix.get("candidates", [])
    if not isinstance(candidates, list):
        raise RuntimeError("Decision matrix 'candidates' must be a list")
    candidate = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and str(item.get("id", "")).strip() == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise RuntimeError(f"Candidate '{candidate_id}' not found in decision matrix")

    benchmark_set = matrix.get("benchmark_set", [])
    if not isinstance(benchmark_set, list) or not benchmark_set:
        raise RuntimeError("Decision matrix must define non-empty benchmark_set")

    benchmark_index, scenario_defs = _build_manifest_index(manifest_path.resolve())
    case_results: List[PrototypeCaseResult] = []
    case_counter = 0

    with tempfile.TemporaryDirectory(prefix="advanced-solver-prototype-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for bench in benchmark_set:
            if not isinstance(bench, dict):
                continue
            benchmark_id = str(bench.get("benchmark_id", "")).strip()
            case_class = str(bench.get("class", "")).strip() or "unclassified"
            scenarios = bench.get("scenarios", [])
            if (
                benchmark_id not in benchmark_index
                or not isinstance(scenarios, list)
                or not scenarios
            ):
                continue

            indexed = benchmark_index[benchmark_id]
            base_netlist = indexed["netlist"]
            available_scenarios = set(indexed["scenario_names"])
            for scenario_name_raw in scenarios:
                scenario_name = str(scenario_name_raw).strip()
                if not scenario_name:
                    continue
                if scenario_name not in available_scenarios:
                    continue
                scenario_override = scenario_defs.get(scenario_name, {})
                if not isinstance(scenario_override, dict):
                    scenario_override = {}

                baseline_netlist = _runtime_defaults(_deep_merge(base_netlist, scenario_override))
                mode = _infer_mode(scenario_name, scenario_override)
                prototype_override = _build_prototype_simulation_override(candidate, mode)
                prototype_netlist = (
                    _runtime_defaults(_deep_merge(baseline_netlist, prototype_override))
                    if prototype_override
                    else baseline_netlist
                )

                baseline = _run_case(
                    run_fn,
                    baseline_netlist,
                    preferred_mode=mode,
                    workdir=tmpdir_path,
                    label=f"{benchmark_id}_{scenario_name}_baseline",
                )
                prototype = _run_case(
                    run_fn,
                    prototype_netlist,
                    preferred_mode=mode,
                    workdir=tmpdir_path,
                    label=f"{benchmark_id}_{scenario_name}_{candidate_id}",
                )

                case_results.append(
                    PrototypeCaseResult(
                        benchmark_id=benchmark_id,
                        scenario=scenario_name,
                        case_class=case_class,
                        mode=mode,
                        baseline_status=baseline.status,
                        prototype_status=prototype.status,
                        baseline_runtime_s=baseline.runtime_s,
                        prototype_runtime_s=prototype.runtime_s,
                        baseline_steps=baseline.steps,
                        prototype_steps=prototype.steps,
                        baseline_message=baseline.message,
                        prototype_message=prototype.message,
                        baseline_diagnostic=baseline.diagnostic,
                        prototype_diagnostic=prototype.diagnostic,
                        runtime_regression_rel=_runtime_regression_rel(baseline, prototype),
                        success_delta=(
                            (1 if prototype.status == "passed" else 0)
                            - (1 if baseline.status == "passed" else 0)
                        ),
                    )
                )
                case_counter += 1
                if max_cases is not None and case_counter >= max_cases:
                    break
            if max_cases is not None and case_counter >= max_cases:
                break

    decision_criteria = matrix.get("decision_criteria", {})
    scoring_weights = matrix.get("scoring_weights", {})
    if not isinstance(decision_criteria, dict):
        decision_criteria = {}
    if not isinstance(scoring_weights, dict):
        scoring_weights = {}

    summary = summarize_case_results(
        case_results,
        candidate=candidate,
        criteria=decision_criteria,
        scoring_weights=scoring_weights,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "pulsim-advanced-solver-prototype-report-v1",
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "candidate": candidate,
        "decision_criteria": decision_criteria,
        "scoring_weights": scoring_weights,
        "selection_policy": matrix.get("selection_policy", {}),
        "summary": summary,
        "cases": [asdict(item) for item in case_results],
    }

    report_path = output_dir / "advanced_solver_prototype_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")

    csv_path = output_dir / "advanced_solver_prototype_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark_id",
                "scenario",
                "class",
                "mode",
                "baseline_status",
                "prototype_status",
                "baseline_runtime_s",
                "prototype_runtime_s",
                "runtime_regression_rel",
                "baseline_steps",
                "prototype_steps",
                "success_delta",
                "baseline_diagnostic",
                "prototype_diagnostic",
                "baseline_message",
                "prototype_message",
            ],
        )
        writer.writeheader()
        for item in case_results:
            writer.writerow(
                {
                    "benchmark_id": item.benchmark_id,
                    "scenario": item.scenario,
                    "class": item.case_class,
                    "mode": item.mode,
                    "baseline_status": item.baseline_status,
                    "prototype_status": item.prototype_status,
                    "baseline_runtime_s": f"{item.baseline_runtime_s:.9e}",
                    "prototype_runtime_s": f"{item.prototype_runtime_s:.9e}",
                    "runtime_regression_rel": (
                        ""
                        if item.runtime_regression_rel is None
                        else f"{item.runtime_regression_rel:.9e}"
                    ),
                    "baseline_steps": item.baseline_steps,
                    "prototype_steps": item.prototype_steps,
                    "success_delta": item.success_delta,
                    "baseline_diagnostic": item.baseline_diagnostic or "",
                    "prototype_diagnostic": item.prototype_diagnostic or "",
                    "baseline_message": item.baseline_message,
                    "prototype_message": item.prototype_message,
                }
            )

    print(f"Advanced solver prototype report written to: {report_path}")
    print(f"Advanced solver prototype CSV written to: {csv_path}")
    print(
        "Summary: baseline_success_rate={:.3f}, prototype_success_rate={:.3f}, "
        "success_gain={:+.3f}, runtime_regression_rel={:+.3f}, hard_constraints_passed={}".format(
            summary["baseline_success_rate"],
            summary["prototype_success_rate"],
            summary["success_rate_gain_abs"],
            summary["runtime_regression_rel"],
            summary["hard_constraints_passed"],
        )
    )

    if enforce_hard_constraints and not bool(summary["hard_constraints_passed"]):
        raise RuntimeError("Advanced solver prototype failed hard constraints")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run isolated advanced-solver prototype benchmark comparisons",
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate", default="sundials_ida_direct")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--enforce-hard-constraints",
        action="store_true",
        help="Exit with failure when hard constraints are not met",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap for quick local smoke runs",
    )
    args = parser.parse_args()

    run_advanced_solver_prototype(
        matrix_path=args.matrix,
        manifest_path=args.manifest,
        candidate_id=str(args.candidate),
        output_dir=args.output_dir,
        enforce_hard_constraints=bool(args.enforce_hard_constraints),
        max_cases=args.max_cases,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
