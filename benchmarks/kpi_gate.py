#!/usr/bin/env python3
"""KPI regression gate for solver refactor phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency for CLI usage
    yaml = None


def _quantile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    frac = index - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_thresholds(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    if yaml is None:
        raise RuntimeError("PyYAML is required to load non-JSON thresholds")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc_timestamp(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    normalized = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def validate_threshold_policy(payload: Dict[str, Any]) -> Dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: Dict[str, Any] = {}

    schema = payload.get("schema")
    if schema is not None and schema != "pulsim-kpi-gates-v1":
        errors.append("threshold policy has unsupported schema value")
    checks["schema"] = schema

    version = payload.get("version")
    if version is not None and not isinstance(version, int):
        errors.append("threshold policy version must be an integer when provided")
    checks["version"] = version

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append("threshold policy must define a non-empty metrics mapping")
        checks["metrics_count"] = 0
    else:
        checks["metrics_count"] = len(metrics)
        for metric_name, rules in metrics.items():
            if not isinstance(metric_name, str) or not metric_name:
                errors.append("threshold metric names must be non-empty strings")
                continue
            if not isinstance(rules, dict):
                errors.append(f"threshold rules for metric '{metric_name}' must be an object")
                continue
            direction = rules.get("direction", "lower_is_better")
            if direction not in ("lower_is_better", "higher_is_better"):
                errors.append(
                    f"threshold metric '{metric_name}' has invalid direction '{direction}'"
                )
            for key in ("max_regression_abs", "max_regression_rel"):
                value = rules.get(key)
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    errors.append(
                        f"threshold metric '{metric_name}' field '{key}' must be numeric"
                    )
                    continue
                if numeric < 0.0:
                    errors.append(
                        f"threshold metric '{metric_name}' field '{key}' must be >= 0"
                    )
            if "max_regression_abs" not in rules and "max_regression_rel" not in rules:
                warnings.append(
                    f"threshold metric '{metric_name}' has no regression bound and will never fail"
                )

    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def validate_baseline_provenance(
    baseline_payload: Dict[str, Any],
    baseline_path: Path,
) -> Dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: Dict[str, Any] = {
        "baseline_path": str(baseline_path),
    }

    schema_version = baseline_payload.get("schema_version")
    if schema_version != "pulsim-kpi-baseline-v1":
        errors.append("baseline schema_version must be 'pulsim-kpi-baseline-v1'")
    checks["schema_version"] = schema_version

    baseline_id = baseline_payload.get("baseline_id")
    if not isinstance(baseline_id, str) or not baseline_id.strip():
        errors.append("baseline_id must be a non-empty string")
    checks["baseline_id"] = baseline_id

    captured_at = baseline_payload.get("captured_at_utc")
    captured_at_ts = _parse_utc_timestamp(captured_at)
    if captured_at_ts is None:
        errors.append("captured_at_utc must be a valid ISO-8601 timestamp")
    checks["captured_at_utc"] = captured_at

    metrics = baseline_payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append("baseline metrics must be a non-empty object")
        checks["metrics_count"] = 0
    else:
        checks["metrics_count"] = len(metrics)

    environment = baseline_payload.get("environment")
    if not isinstance(environment, dict):
        errors.append("baseline environment fingerprint is required")
        checks["environment_keys"] = []
    else:
        env_keys = sorted(str(key) for key in environment.keys())
        checks["environment_keys"] = env_keys
        for required in ("os", "python"):
            value = environment.get(required)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"baseline environment field '{required}' is required")
        recommended_keys = ("machine_class", "compiler", "cmake", "cxx_flags")
        for recommended in recommended_keys:
            value = environment.get(recommended)
            if not isinstance(value, str) or not value.strip():
                warnings.append(
                    f"baseline environment field '{recommended}' is recommended for stronger provenance"
                )

    manifest_path = baseline_path.with_name("artifact_manifest.json")
    checks["manifest_path"] = str(manifest_path)
    if not manifest_path.is_file():
        errors.append("artifact_manifest.json is required beside kpi_baseline.json")
        return {
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
        }

    manifest_payload = _load_json(manifest_path)
    manifest_schema = manifest_payload.get("schema_version")
    if manifest_schema != "pulsim-kpi-baseline-manifest-v1":
        errors.append("manifest schema_version must be 'pulsim-kpi-baseline-manifest-v1'")
    checks["manifest_schema_version"] = manifest_schema

    manifest_baseline_id = manifest_payload.get("baseline_id")
    if manifest_baseline_id != baseline_id:
        errors.append("baseline_id mismatch between baseline and manifest")
    checks["manifest_baseline_id"] = manifest_baseline_id

    manifest_captured = manifest_payload.get("captured_at_utc")
    manifest_captured_ts = _parse_utc_timestamp(manifest_captured)
    if manifest_captured_ts is None:
        errors.append("manifest captured_at_utc must be a valid ISO-8601 timestamp")
    elif captured_at_ts is not None and manifest_captured_ts != captured_at_ts:
        errors.append("captured_at_utc mismatch between baseline and manifest")
    checks["manifest_captured_at_utc"] = manifest_captured

    files = manifest_payload.get("files")
    if not isinstance(files, list) or not files:
        errors.append("manifest files must be a non-empty array")
        checks["manifest_file_count"] = 0
        return {
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
        }
    checks["manifest_file_count"] = len(files)

    source_bench_results = baseline_payload.get("source_bench_results")
    source_bench_results_resolved: Optional[Path] = None
    if isinstance(source_bench_results, str) and source_bench_results.strip():
        source_bench_results_resolved = _resolve_existing_path(
            source_bench_results.strip(),
            baseline_path,
        )
    checks["source_bench_results"] = source_bench_results
    checks["source_bench_results_resolved"] = (
        str(source_bench_results_resolved)
        if source_bench_results_resolved is not None
        else None
    )

    verified_files = 0
    missing_files = 0
    digest_mismatches = 0
    size_mismatches = 0
    bench_results_declared = 0

    for idx, entry in enumerate(files):
        if not isinstance(entry, dict):
            errors.append(f"manifest file entry {idx} must be an object")
            continue
        path_value = entry.get("path")
        sha_value = entry.get("sha256")
        size_value = entry.get("size_bytes")

        if not isinstance(path_value, str) or not path_value.strip():
            errors.append(f"manifest file entry {idx} path must be a non-empty string")
            continue
        if path_value.endswith("benchmarks/results.json"):
            bench_results_declared += 1

        if not isinstance(sha_value, str) or not re.fullmatch(r"[0-9a-f]{64}", sha_value):
            errors.append(f"manifest file entry {idx} has invalid sha256 value")
            continue

        if not isinstance(size_value, int) or size_value < 0:
            errors.append(f"manifest file entry {idx} size_bytes must be a non-negative integer")
            continue

        resolved = _resolve_existing_path(path_value, baseline_path)
        if resolved is None or not resolved.is_file():
            errors.append(f"manifest file entry {idx} does not resolve to an existing file")
            missing_files += 1
            continue

        actual_size = resolved.stat().st_size
        if actual_size != size_value:
            errors.append(
                f"manifest file entry {idx} size mismatch for '{path_value}' "
                f"(expected={size_value}, actual={actual_size})"
            )
            size_mismatches += 1
            continue

        actual_sha = _sha256_file(resolved)
        if actual_sha != sha_value:
            errors.append(
                f"manifest file entry {idx} sha256 mismatch for '{path_value}'"
            )
            digest_mismatches += 1
            continue

        verified_files += 1

    checks["verified_files"] = verified_files
    checks["missing_files"] = missing_files
    checks["digest_mismatches"] = digest_mismatches
    checks["size_mismatches"] = size_mismatches
    checks["bench_results_declared"] = bench_results_declared

    if bench_results_declared == 0:
        if source_bench_results_resolved is None:
            errors.append(
                "baseline must provide a resolvable source_bench_results when "
                "manifest lacks benchmarks/results.json"
            )
        else:
            warnings.append(
                "manifest does not include a benchmarks/results.json artifact entry; "
                "runtime scope uses source_bench_results"
            )

    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def _case_key(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    benchmark_id = row.get("benchmark_id")
    scenario = row.get("scenario")
    if not isinstance(benchmark_id, str) or not benchmark_id:
        return None
    if not isinstance(scenario, str) or not scenario:
        return None
    return (benchmark_id, scenario)


def _collect_case_keys(rows: list[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    for row in rows:
        key = _case_key(row)
        if key is not None:
            keys.add(key)
    return keys


def _resolve_existing_path(raw_path: str, baseline_path: Path) -> Optional[Path]:
    candidate = Path(raw_path)
    candidates: list[Path] = []

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        # Try common roots so CLI works from repo root or nested directories.
        candidates.append(Path.cwd() / candidate)
        candidates.append(baseline_path.parent / candidate)
        candidates.extend(parent / candidate for parent in baseline_path.parents)

    for path in candidates:
        resolved = path.resolve()
        if resolved.exists():
            return resolved
    return None


def _resolve_baseline_bench_results_path(
    baseline_payload: Dict[str, Any],
    baseline_path: Path,
) -> Optional[Path]:
    direct = baseline_payload.get("source_bench_results")
    if isinstance(direct, str) and direct.strip():
        resolved = _resolve_existing_path(direct.strip(), baseline_path)
        if resolved is not None and resolved.is_file():
            return resolved

    source_root = baseline_payload.get("source_artifacts_root")
    if isinstance(source_root, str) and source_root.strip():
        resolved_root = _resolve_existing_path(source_root.strip(), baseline_path)
        if resolved_root is not None:
            root_path = resolved_root if resolved_root.is_dir() else resolved_root.parent
            for suffix in ("benchmarks/results.json", "results.json"):
                candidate = (root_path / suffix).resolve()
                if candidate.is_file():
                    return candidate

    manifest_path = baseline_path.with_name("artifact_manifest.json")
    if manifest_path.is_file():
        manifest_payload = _load_json(manifest_path)
        for entry in manifest_payload.get("files", []):
            if not isinstance(entry, dict):
                continue
            path_value = entry.get("path")
            if not isinstance(path_value, str):
                continue
            if not path_value.endswith("benchmarks/results.json"):
                continue
            resolved = _resolve_existing_path(path_value, baseline_path)
            if resolved is not None and resolved.is_file():
                return resolved

    return None


def resolve_runtime_scope(
    baseline_payload: Dict[str, Any],
    baseline_path: Path,
    bench_results_path: Path,
) -> Tuple[Optional[Set[Tuple[str, str]]], Dict[str, Any]]:
    bench_payload = _load_json(bench_results_path)
    bench_results = bench_payload.get("results", [])
    executed = [
        row
        for row in bench_results
        if row.get("status") in ("passed", "failed") and isinstance(row, dict)
    ]
    executed_keys = _collect_case_keys(executed)

    scope_info: Dict[str, Any] = {
        "mode": "all_executed",
        "reason": "baseline_scope_unavailable",
        "executed_total": len(executed),
        "executed_with_ids": len(executed_keys),
        "baseline_scope_total": 0,
        "comparable_total": len(executed),
        "baseline_results_path": None,
    }

    baseline_results_path = _resolve_baseline_bench_results_path(baseline_payload, baseline_path)
    if baseline_results_path is None:
        return None, scope_info

    baseline_payload_results = _load_json(baseline_results_path)
    baseline_rows = baseline_payload_results.get("results", [])
    baseline_scope = [
        row
        for row in baseline_rows
        if row.get("status") in ("passed", "failed") and isinstance(row, dict)
    ]
    baseline_scope_keys = _collect_case_keys(baseline_scope)

    scope_info["baseline_results_path"] = str(baseline_results_path)
    scope_info["baseline_scope_total"] = len(baseline_scope_keys)

    if not baseline_scope_keys:
        scope_info["reason"] = "baseline_scope_has_no_case_ids"
        return None, scope_info

    comparable_keys = executed_keys.intersection(baseline_scope_keys)
    if comparable_keys:
        scope_info["mode"] = "baseline_intersection"
        scope_info["reason"] = "using_baseline_case_intersection"
        scope_info["comparable_total"] = len(comparable_keys)
        return comparable_keys, scope_info

    scope_info["reason"] = "no_intersection_between_current_and_baseline_cases"
    scope_info["comparable_total"] = len(executed)
    return None, scope_info


def compute_metrics(
    bench_results_path: Path,
    parity_ltspice_results_path: Optional[Path] = None,
    parity_ngspice_results_path: Optional[Path] = None,
    stress_summary_path: Optional[Path] = None,
    runtime_case_filter: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}

    bench_payload = _load_json(bench_results_path)
    bench_results = bench_payload.get("results", [])
    executed = [
        row
        for row in bench_results
        if isinstance(row, dict) and row.get("status") in ("passed", "failed")
    ]

    passed = sum(1 for row in executed if row.get("status") == "passed")
    failed = sum(1 for row in executed if row.get("status") == "failed")
    denom = passed + failed
    metrics["convergence_success_rate"] = (float(passed) / float(denom)) if denom > 0 else None

    runtime_rows = executed
    if runtime_case_filter is not None:
        runtime_rows = [
            row for row in executed
            if _case_key(row) in runtime_case_filter
        ]

    runtimes = [
        float(row["runtime_s"])
        for row in runtime_rows
        if row.get("runtime_s") is not None
    ]
    metrics["runtime_p50"] = _quantile(runtimes, 0.50)
    metrics["runtime_p95"] = _quantile(runtimes, 0.95)

    state_space_primary_sum = 0.0
    dae_fallback_sum = 0.0
    state_space_total_sum = 0.0
    for row in executed:
        telemetry = row.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        primary = telemetry.get("state_space_primary_steps")
        fallback = telemetry.get("dae_fallback_steps")
        if primary is None or fallback is None:
            continue
        primary_f = float(primary)
        fallback_f = float(fallback)
        if not (primary_f >= 0.0 and fallback_f >= 0.0):
            continue
        state_space_primary_sum += primary_f
        dae_fallback_sum += fallback_f
        state_space_total_sum += primary_f + fallback_f
    metrics["state_space_primary_ratio"] = (
        state_space_primary_sum / state_space_total_sum
        if state_space_total_sum > 0.0
        else None
    )
    metrics["dae_fallback_ratio"] = (
        dae_fallback_sum / state_space_total_sum
        if state_space_total_sum > 0.0
        else None
    )

    def telemetry_mean(metric_name: str) -> Optional[float]:
        values: list[float] = []
        for row in executed:
            telemetry = row.get("telemetry")
            if not isinstance(telemetry, dict):
                continue
            value = telemetry.get(metric_name)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            values.append(numeric)
        if not values:
            return None
        return float(sum(values) / len(values))

    metrics["loss_energy_balance_error"] = telemetry_mean("loss_energy_balance_error")
    metrics["thermal_peak_temperature_delta"] = telemetry_mean("thermal_peak_temperature_delta")

    def parity_mean_rms(path: Optional[Path]) -> Optional[float]:
        if path is None:
            return None
        payload = _load_json(path)
        values = [
            float(row["rms_error"])
            for row in payload.get("results", [])
            if row.get("status") == "passed" and row.get("rms_error") is not None
        ]
        if not values:
            return None
        return float(sum(values) / len(values))

    metrics["parity_rms_error_ltspice"] = parity_mean_rms(parity_ltspice_results_path)
    metrics["parity_rms_error_ngspice"] = parity_mean_rms(parity_ngspice_results_path)

    if stress_summary_path is not None:
        stress_summary = _load_json(stress_summary_path)
        tiers_failed = stress_summary.get("tiers_failed")
        metrics["stress_tier_failure_count"] = float(tiers_failed) if tiers_failed is not None else None
    else:
        metrics["stress_tier_failure_count"] = None

    return metrics


@dataclass
class MetricComparison:
    name: str
    status: str
    message: str
    current: Optional[float]
    baseline: Optional[float]
    direction: str
    regression_abs: Optional[float]
    regression_rel: Optional[float]
    max_regression_abs: Optional[float]
    max_regression_rel: Optional[float]
    required: bool


def compare_metric(
    name: str,
    current_value: Optional[float],
    baseline_value: Optional[float],
    rules: Dict[str, Any],
) -> MetricComparison:
    direction = str(rules.get("direction", "lower_is_better")).strip()
    required = bool(rules.get("required", True))
    max_reg_abs = rules.get("max_regression_abs")
    max_reg_rel = rules.get("max_regression_rel")

    max_reg_abs = float(max_reg_abs) if max_reg_abs is not None else None
    max_reg_rel = float(max_reg_rel) if max_reg_rel is not None else None

    if current_value is None:
        status = "failed" if required else "skipped"
        message = "current metric unavailable"
        return MetricComparison(
            name=name,
            status=status,
            message=message,
            current=current_value,
            baseline=baseline_value,
            direction=direction,
            regression_abs=None,
            regression_rel=None,
            max_regression_abs=max_reg_abs,
            max_regression_rel=max_reg_rel,
            required=required,
        )

    if baseline_value is None:
        status = "failed" if required else "skipped"
        message = "baseline metric unavailable"
        return MetricComparison(
            name=name,
            status=status,
            message=message,
            current=current_value,
            baseline=baseline_value,
            direction=direction,
            regression_abs=None,
            regression_rel=None,
            max_regression_abs=max_reg_abs,
            max_regression_rel=max_reg_rel,
            required=required,
        )

    if direction == "higher_is_better":
        regression_abs = max(0.0, baseline_value - current_value)
    elif direction == "lower_is_better":
        regression_abs = max(0.0, current_value - baseline_value)
    else:
        return MetricComparison(
            name=name,
            status="failed",
            message=f"unknown direction '{direction}'",
            current=current_value,
            baseline=baseline_value,
            direction=direction,
            regression_abs=None,
            regression_rel=None,
            max_regression_abs=max_reg_abs,
            max_regression_rel=max_reg_rel,
            required=required,
        )

    if baseline_value == 0.0:
        regression_rel = 0.0 if regression_abs == 0.0 else float("inf")
    else:
        regression_rel = regression_abs / abs(baseline_value)

    failures: list[str] = []
    if max_reg_abs is not None and regression_abs > max_reg_abs:
        failures.append(
            f"regression_abs={regression_abs:.6g} > max_regression_abs={max_reg_abs:.6g}"
        )
    if max_reg_rel is not None and regression_rel > max_reg_rel:
        failures.append(
            f"regression_rel={regression_rel:.6g} > max_regression_rel={max_reg_rel:.6g}"
        )

    if failures:
        status = "failed"
        message = "; ".join(failures)
    else:
        status = "passed"
        message = "within threshold"

    return MetricComparison(
        name=name,
        status=status,
        message=message,
        current=current_value,
        baseline=baseline_value,
        direction=direction,
        regression_abs=regression_abs,
        regression_rel=regression_rel,
        max_regression_abs=max_reg_abs,
        max_regression_rel=max_reg_rel,
        required=required,
    )


def run_gate(
    baseline_path: Path,
    thresholds_path: Path,
    bench_results_path: Path,
    parity_ltspice_results_path: Optional[Path],
    parity_ngspice_results_path: Optional[Path],
    stress_summary_path: Optional[Path],
    strict_provenance: bool = True,
) -> Dict[str, Any]:
    baseline_payload = _load_json(baseline_path)
    threshold_payload = _load_thresholds(thresholds_path)
    baseline_provenance = validate_baseline_provenance(
        baseline_payload=baseline_payload,
        baseline_path=baseline_path,
    )
    threshold_provenance = validate_threshold_policy(threshold_payload)
    blocked_by_provenance = (
        baseline_provenance["status"] == "failed" or threshold_provenance["status"] == "failed"
    )

    runtime_case_filter: Optional[Set[Tuple[str, str]]] = None
    runtime_scope: Dict[str, Any] = {
        "mode": "all_executed",
        "reason": "provenance_gate_blocked",
        "executed_total": 0,
        "executed_with_ids": 0,
        "baseline_scope_total": 0,
        "comparable_total": 0,
        "baseline_results_path": None,
    }
    if not blocked_by_provenance or not strict_provenance:
        runtime_case_filter, runtime_scope = resolve_runtime_scope(
            baseline_payload=baseline_payload,
            baseline_path=baseline_path,
            bench_results_path=bench_results_path,
        )

    baseline_metrics = baseline_payload.get("metrics", {})
    if not isinstance(baseline_metrics, dict):
        baseline_metrics = {}
    current_metrics = compute_metrics(
        bench_results_path=bench_results_path,
        parity_ltspice_results_path=parity_ltspice_results_path,
        parity_ngspice_results_path=parity_ngspice_results_path,
        stress_summary_path=stress_summary_path,
        runtime_case_filter=runtime_case_filter,
    )

    metric_rules = threshold_payload.get("metrics", {})
    if not isinstance(metric_rules, dict):
        metric_rules = {}
    comparisons: Dict[str, Dict[str, Any]] = {}

    failed_required = 0
    skipped_optional = 0

    if not (blocked_by_provenance and strict_provenance):
        for metric_name, rules in metric_rules.items():
            cmp = compare_metric(
                name=metric_name,
                current_value=current_metrics.get(metric_name),
                baseline_value=baseline_metrics.get(metric_name),
                rules=rules or {},
            )
            comparisons[metric_name] = {
                "status": cmp.status,
                "message": cmp.message,
                "current": cmp.current,
                "baseline": cmp.baseline,
                "direction": cmp.direction,
                "regression_abs": cmp.regression_abs,
                "regression_rel": cmp.regression_rel,
                "max_regression_abs": cmp.max_regression_abs,
                "max_regression_rel": cmp.max_regression_rel,
                "required": cmp.required,
            }
            if cmp.status == "failed" and cmp.required:
                failed_required += 1
            if cmp.status == "skipped" and not cmp.required:
                skipped_optional += 1

    overall_status = "failed" if failed_required > 0 else "passed"
    if blocked_by_provenance and strict_provenance:
        overall_status = "failed"

    report = {
        "schema_version": "pulsim-kpi-gate-report-v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_id": baseline_payload.get("baseline_id"),
        "overall_status": overall_status,
        "failed_required_metrics": failed_required,
        "skipped_optional_metrics": skipped_optional,
        "current_metrics": current_metrics,
        "runtime_scope": runtime_scope,
        "comparisons": comparisons,
        "blocked_by_provenance": blocked_by_provenance and strict_provenance,
        "provenance": {
            "strict_mode": strict_provenance,
            "baseline": baseline_provenance,
            "thresholds": threshold_provenance,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce KPI regression thresholds")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/kpi_baselines/phase0_2026-02-23/kpi_baseline.json"),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("benchmarks/kpi_thresholds.yaml"),
    )
    parser.add_argument("--bench-results", type=Path, required=True)
    parser.add_argument("--parity-ltspice-results", type=Path)
    parser.add_argument("--parity-ngspice-results", type=Path)
    parser.add_argument("--stress-summary", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--print-report", action="store_true")
    parser.add_argument(
        "--no-strict-provenance",
        action="store_true",
        help="Allow metric comparison even when baseline/threshold provenance checks fail",
    )
    args = parser.parse_args()

    report = run_gate(
        baseline_path=args.baseline,
        thresholds_path=args.thresholds,
        bench_results_path=args.bench_results,
        parity_ltspice_results_path=args.parity_ltspice_results,
        parity_ngspice_results_path=args.parity_ngspice_results,
        stress_summary_path=args.stress_summary,
        strict_provenance=not args.no_strict_provenance,
    )

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if args.print_report:
        print(json.dumps(report, indent=2, sort_keys=True))

    return 1 if report["overall_status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
