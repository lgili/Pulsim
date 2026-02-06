#!/usr/bin/env python3
"""Run full solver/integrator validation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_runner import can_use_pulsim_python_backend, run_benchmarks, write_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pulsim validation matrix")
    parser.add_argument("--benchmarks", type=Path, default=Path(__file__).with_name("benchmarks.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/matrix"))
    parser.add_argument(
        "--pulsim-cli",
        type=Path,
        default=None,
        help="Deprecated: ignored. Validation matrix uses Python runtime bindings only.",
    )
    args = parser.parse_args()

    if args.pulsim_cli is not None:
        print("Warning: --pulsim-cli is deprecated and ignored. Using Python runtime backend.")

    if not can_use_pulsim_python_backend():
        raise SystemExit(
            "Pulsim Python runtime backend unavailable. "
            "Build Python bindings and expose build/python on PYTHONPATH or install pulsim package."
        )

    results = run_benchmarks(
        args.benchmarks,
        args.output_dir,
        selected=None,
        matrix=True,
    )
    write_results(args.output_dir, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
