#!/usr/bin/env python3
"""Validate convergence reference examples catalog against benchmark manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return payload


def build_benchmark_index(manifest_path: Path) -> Dict[str, Set[str]]:
    manifest = load_yaml(manifest_path)
    benchmarks = manifest.get("benchmarks", [])
    if not isinstance(benchmarks, list):
        raise RuntimeError("benchmarks manifest 'benchmarks' must be a list")

    index: Dict[str, Set[str]] = {}
    for entry in benchmarks:
        if not isinstance(entry, dict):
            continue
        rel_path = entry.get("path")
        if not isinstance(rel_path, str) or not rel_path.strip():
            continue

        circuit_path = (manifest_path.parent / rel_path).resolve()
        netlist = load_yaml(circuit_path)
        bench_meta = netlist.get("benchmark", {})
        benchmark_id = circuit_path.stem
        if isinstance(bench_meta, dict):
            bench_id_raw = bench_meta.get("id")
            if isinstance(bench_id_raw, str) and bench_id_raw.strip():
                benchmark_id = bench_id_raw.strip()

        scenarios_raw = entry.get("scenarios", ["default"])
        scenarios: Set[str] = set()
        if isinstance(scenarios_raw, list):
            for item in scenarios_raw:
                if isinstance(item, str) and item.strip():
                    scenarios.add(item.strip())
        if not scenarios:
            scenarios.add("default")

        index[benchmark_id] = scenarios

    return index


def validate_examples(manifest_path: Path, examples_path: Path) -> List[str]:
    benchmark_index = build_benchmark_index(manifest_path)
    examples = load_yaml(examples_path)

    errors: List[str] = []
    if examples.get("schema") != "pulsim-convergence-reference-examples-v1":
        errors.append("examples schema must be 'pulsim-convergence-reference-examples-v1'")

    classes = examples.get("classes")
    if not isinstance(classes, list) or not classes:
        errors.append("examples file must define a non-empty 'classes' list")
        return errors

    for class_idx, class_entry in enumerate(classes):
        if not isinstance(class_entry, dict):
            errors.append(f"class entry #{class_idx} must be an object")
            continue
        failure_class = class_entry.get("failure_class")
        if not isinstance(failure_class, str) or not failure_class.strip():
            errors.append(f"class entry #{class_idx} missing non-empty failure_class")
            continue

        references = class_entry.get("examples")
        if not isinstance(references, list) or not references:
            errors.append(f"class '{failure_class}' must define non-empty examples list")
            continue

        for ref_idx, ref in enumerate(references):
            if not isinstance(ref, dict):
                errors.append(f"class '{failure_class}' example #{ref_idx} must be an object")
                continue
            bench_id = ref.get("benchmark_id")
            if not isinstance(bench_id, str) or not bench_id.strip():
                errors.append(f"class '{failure_class}' example #{ref_idx} missing benchmark_id")
                continue
            bench_id = bench_id.strip()
            if bench_id not in benchmark_index:
                errors.append(
                    f"class '{failure_class}' references unknown benchmark_id '{bench_id}'"
                )
                continue

            scenarios_raw = ref.get("scenarios", [])
            if not isinstance(scenarios_raw, list) or not scenarios_raw:
                errors.append(
                    f"class '{failure_class}' benchmark '{bench_id}' must define at least one scenario"
                )
            else:
                for scenario in scenarios_raw:
                    if not isinstance(scenario, str) or not scenario.strip():
                        errors.append(
                            f"class '{failure_class}' benchmark '{bench_id}' has invalid scenario"
                        )
                        continue
                    if scenario.strip() not in benchmark_index[bench_id]:
                        errors.append(
                            f"class '{failure_class}' benchmark '{bench_id}' references undefined scenario '{scenario}'"
                        )

            expected_kpi = ref.get("expected_kpi")
            if not isinstance(expected_kpi, dict) or not expected_kpi:
                errors.append(
                    f"class '{failure_class}' benchmark '{bench_id}' must define expected_kpi mapping"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate convergence reference examples catalog",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/benchmarks.yaml"),
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("benchmarks/convergence_reference_examples.yaml"),
    )
    args = parser.parse_args()

    errors = validate_examples(
        manifest_path=args.manifest.resolve(),
        examples_path=args.examples.resolve(),
    )
    if errors:
        for item in errors:
            print(f"ERROR: {item}")
        return 1

    print("Reference examples catalog is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
