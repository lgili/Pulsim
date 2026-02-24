#!/usr/bin/env python3
"""Run variable-step benchmark matrix (adaptive timestep forced on)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_runner import can_use_pulsim_python_backend, run_benchmarks, write_results

DEFAULT_VARIABLE_CASES = [
    "stiff_rlc",
]
DEFAULT_VARIABLE_SCENARIOS = ["direct_trap", "trbdf2", "rosenbrockw"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run variable-step benchmark matrix")
    parser.add_argument("--benchmarks", type=Path, default=Path(__file__).with_name("benchmarks.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/out_variable_matrix"))
    parser.add_argument(
        "--only",
        nargs="*",
        default=DEFAULT_VARIABLE_CASES,
        help="Benchmark ids to run (defaults to stable variable-mode matrix set)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=DEFAULT_VARIABLE_SCENARIOS,
        help="Scenarios to include in variable matrix",
    )
    parser.add_argument(
        "--dt-max-factor",
        type=float,
        default=4.0,
        help="Clamp adaptive dt_max to at most factor*dt for variable-mode accuracy stability",
    )
    args = parser.parse_args()

    if not can_use_pulsim_python_backend():
        raise SystemExit(
            "Pulsim Python runtime backend unavailable. "
            "Build Python bindings and expose build/python on PYTHONPATH or install pulsim package."
        )

    results = run_benchmarks(
        args.benchmarks,
        args.output_dir,
        selected=args.only,
        matrix=False,
        simulation_overrides={"adaptive_timestep": True},
        scenario_filter=args.scenarios,
        adaptive_dt_max_factor=args.dt_max_factor,
    )
    write_results(args.output_dir, results)

    summary = {
        "passed": sum(1 for item in results if item.status == "passed"),
        "failed": sum(1 for item in results if item.status == "failed"),
        "skipped": sum(1 for item in results if item.status == "skipped"),
        "baseline": sum(1 for item in results if item.status == "baseline"),
        "total": len(results),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
