#!/usr/bin/env python3
"""Run full solver/integrator validation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_runner import can_use_pulsim_python_backend, find_pulsim_cli, run_benchmarks, write_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pulsim validation matrix")
    parser.add_argument("--benchmarks", type=Path, default=Path(__file__).with_name("benchmarks.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/matrix"))
    parser.add_argument("--pulsim-cli", type=Path, default=None)
    args = parser.parse_args()

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
        selected=None,
        matrix=True,
    )
    write_results(args.output_dir, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
