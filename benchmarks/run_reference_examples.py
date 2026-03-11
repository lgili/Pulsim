#!/usr/bin/env python3
"""Run convergence reference examples by class using benchmark_runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

import benchmark_runner as br


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return payload


def collect_selected_benchmarks(
    examples_path: Path,
    selected_classes: Set[str],
) -> List[str]:
    payload = load_yaml(examples_path)
    classes = payload.get("classes", [])
    if not isinstance(classes, list):
        return []

    selected: List[str] = []
    seen: Set[str] = set()
    for entry in classes:
        if not isinstance(entry, dict):
            continue
        failure_class_raw = entry.get("failure_class")
        if not isinstance(failure_class_raw, str):
            continue
        failure_class = failure_class_raw.strip()
        if selected_classes and failure_class not in selected_classes:
            continue
        refs = entry.get("examples", [])
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            bench_id_raw = ref.get("benchmark_id")
            if not isinstance(bench_id_raw, str) or not bench_id_raw.strip():
                continue
            bench_id = bench_id_raw.strip()
            if bench_id not in seen:
                seen.add(bench_id)
                selected.append(bench_id)

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run convergence reference examples for selected classes",
    )
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=Path("benchmarks/benchmarks.yaml"),
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("benchmarks/convergence_reference_examples.yaml"),
    )
    parser.add_argument(
        "--class",
        dest="selected_classes",
        action="append",
        default=None,
        help="Failure class to execute (repeatable). Defaults to all classes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/out_reference_examples"),
    )
    args = parser.parse_args()

    selected_classes: Set[str] = set(args.selected_classes or [])
    selected_benchmarks = collect_selected_benchmarks(
        examples_path=args.examples.resolve(),
        selected_classes=selected_classes,
    )
    if not selected_benchmarks:
        raise SystemExit("No benchmarks selected from reference examples")

    results = br.run_benchmarks(
        benchmarks_path=args.benchmarks.resolve(),
        output_dir=args.output_dir.resolve(),
        selected=selected_benchmarks,
    )
    br.write_results(args.output_dir.resolve(), results)

    failed = sum(1 for row in results if row.status == "failed")
    skipped = sum(1 for row in results if row.status == "skipped")
    passed = sum(1 for row in results if row.status == "passed")
    print(
        {
            "selected_benchmarks": selected_benchmarks,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "output_dir": str(args.output_dir.resolve()),
        }
    )
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
