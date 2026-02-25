#!/usr/bin/env python3
"""Freeze KPI baseline snapshots with provenance metadata and manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import kpi_gate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_repo_relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _run_version(*cmd: str) -> str:
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unavailable"
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return "unavailable"
    return lines[0]


def collect_environment_fingerprint(
    machine_class_override: Optional[str] = None,
    cxx_flags_override: Optional[str] = None,
) -> Dict[str, str]:
    cc_cmd = os.environ.get("CC", "cc")
    cxx_cmd = os.environ.get("CXX", "c++")
    machine_class = machine_class_override or platform.machine() or "unknown"
    cxx_flags = (
        cxx_flags_override
        or os.environ.get("CXXFLAGS")
        or os.environ.get("CMAKE_CXX_FLAGS")
        or "unset"
    )

    return {
        "os": f"{platform.system()}-{platform.release()}-{platform.machine()}",
        "python": f"Python {platform.python_version()}",
        "machine_class": machine_class,
        "compiler": _run_version(cxx_cmd, "--version"),
        "cc": _run_version(cc_cmd, "--version"),
        "cmake": _run_version("cmake", "--version"),
        "cxx_flags": cxx_flags,
    }


def _manifest_entry(path: Path, repo_root: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"artifact file does not exist: {resolved}")
    return {
        "path": _as_repo_relative(resolved, repo_root),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _unique_existing_paths(paths: Sequence[Optional[Path]]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def freeze_baseline(
    baseline_id: str,
    output_dir: Path,
    bench_results_path: Path,
    stress_summary_path: Optional[Path],
    parity_ltspice_results_path: Optional[Path],
    parity_ngspice_results_path: Optional[Path],
    source_artifacts_root: Optional[Path],
    machine_class_override: Optional[str],
    cxx_flags_override: Optional[str],
    overwrite: bool,
) -> Tuple[Path, Path]:
    repo_root = Path.cwd().resolve()
    bench_results = bench_results_path.resolve()
    stress_summary = stress_summary_path.resolve() if stress_summary_path is not None else None
    parity_ltspice = (
        parity_ltspice_results_path.resolve()
        if parity_ltspice_results_path is not None
        else None
    )
    parity_ngspice = (
        parity_ngspice_results_path.resolve()
        if parity_ngspice_results_path is not None
        else None
    )

    if not bench_results.is_file():
        raise FileNotFoundError(f"bench results not found: {bench_results}")
    if stress_summary is not None and not stress_summary.is_file():
        raise FileNotFoundError(f"stress summary not found: {stress_summary}")
    if parity_ltspice is not None and not parity_ltspice.is_file():
        raise FileNotFoundError(f"LTspice parity results not found: {parity_ltspice}")
    if parity_ngspice is not None and not parity_ngspice.is_file():
        raise FileNotFoundError(f"ngspice parity results not found: {parity_ngspice}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "kpi_baseline.json"
    manifest_path = output_dir / "artifact_manifest.json"
    if not overwrite and (baseline_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            "baseline output files already exist; use --overwrite to replace them"
        )

    metrics = kpi_gate.compute_metrics(
        bench_results_path=bench_results,
        parity_ltspice_results_path=parity_ltspice,
        parity_ngspice_results_path=parity_ngspice,
        stress_summary_path=stress_summary,
    )

    captured_at = datetime.now(timezone.utc).isoformat()
    baseline_payload: Dict[str, Any] = {
        "schema_version": "pulsim-kpi-baseline-v1",
        "baseline_id": baseline_id,
        "captured_at_utc": captured_at,
        "source_bench_results": _as_repo_relative(bench_results, repo_root),
        "metrics": metrics,
        "environment": collect_environment_fingerprint(
            machine_class_override=machine_class_override,
            cxx_flags_override=cxx_flags_override,
        ),
    }
    if source_artifacts_root is not None:
        baseline_payload["source_artifacts_root"] = _as_repo_relative(
            source_artifacts_root.resolve(),
            repo_root,
        )

    artifacts_for_manifest = _unique_existing_paths(
        (
            bench_results,
            stress_summary,
            parity_ltspice,
            parity_ngspice,
        )
    )
    manifest_payload = {
        "schema_version": "pulsim-kpi-baseline-manifest-v1",
        "baseline_id": baseline_id,
        "captured_at_utc": captured_at,
        "files": [
            _manifest_entry(path=artifact, repo_root=repo_root)
            for artifact in artifacts_for_manifest
        ],
    }

    with open(baseline_path, "w", encoding="utf-8") as handle:
        json.dump(baseline_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return baseline_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a KPI baseline snapshot with provenance metadata",
    )
    parser.add_argument("--baseline-id", required=True, help="Unique baseline snapshot id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for kpi_baseline.json and artifact_manifest.json",
    )
    parser.add_argument("--bench-results", type=Path, required=True)
    parser.add_argument("--stress-summary", type=Path)
    parser.add_argument("--parity-ltspice-results", type=Path)
    parser.add_argument("--parity-ngspice-results", type=Path)
    parser.add_argument(
        "--source-artifacts-root",
        type=Path,
        help="Optional root directory containing benchmark artifacts",
    )
    parser.add_argument("--machine-class", help="Optional machine class fingerprint override")
    parser.add_argument("--cxx-flags", help="Optional compiler flags fingerprint override")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing baseline files in output directory",
    )
    args = parser.parse_args()

    baseline_id = args.baseline_id.strip()
    if not baseline_id:
        raise SystemExit("baseline id must not be empty")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("benchmarks/kpi_baselines") / baseline_id

    baseline_path, manifest_path = freeze_baseline(
        baseline_id=baseline_id,
        output_dir=output_dir,
        bench_results_path=args.bench_results,
        stress_summary_path=args.stress_summary,
        parity_ltspice_results_path=args.parity_ltspice_results,
        parity_ngspice_results_path=args.parity_ngspice_results,
        source_artifacts_root=args.source_artifacts_root,
        machine_class_override=args.machine_class,
        cxx_flags_override=args.cxx_flags,
        overwrite=args.overwrite,
    )
    print(f"Wrote baseline: {baseline_path}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
