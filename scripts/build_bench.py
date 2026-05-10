#!/usr/bin/env python3
"""Build-wallclock benchmark for Pulsim.

`refactor-modular-build-split` Phase 1: measure clean + incremental
rebuild times so the modular-split work can prove a wall-clock win
without fingers crossed.

Usage::

    python3 scripts/build_bench.py --build-dir build --target pulsim_tests
    python3 scripts/build_bench.py --build-dir build --target pulsim_tests --json out.json

The script:
  1. Runs `cmake --build <dir> --clean-first` and times it (baseline
     clean build).
  2. Touches a single header (defaults to a small, infrequently-edited
     one) and times the incremental rebuild.
  3. Reports both wallclock numbers + the incremental / clean ratio
     (Phase 2 target: ≤ 10 %).
  4. Optionally emits a JSON artifact for CI ratcheting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _run_build(build_dir: Path, target: str, clean: bool) -> float:
    cmd = ["cmake", "--build", str(build_dir), "--target", target]
    if clean:
        cmd.append("--clean-first")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.exit(1)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build",
                        help="CMake build directory (default: build)")
    parser.add_argument("--target", default="pulsim_tests",
                        help="CMake target to build (default: pulsim_tests)")
    parser.add_argument("--touch-file",
                        default="core/include/pulsim/v1/numeric_types.hpp",
                        help="File to touch between clean and incremental "
                             "rebuilds (default: numeric_types.hpp — small "
                             "leaf header)")
    parser.add_argument("--json", default=None,
                        help="Optional JSON artifact path")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    if not build_dir.exists():
        sys.exit(f"build dir not found: {build_dir}")

    print(f"Clean build target={args.target} ...")
    clean_seconds = _run_build(build_dir, args.target, clean=True)
    print(f"  → {clean_seconds:.2f} s")

    touch_path = Path(args.touch_file)
    if not touch_path.exists():
        sys.exit(f"touch file not found: {touch_path}")
    print(f"Touching {touch_path} ...")
    os.utime(touch_path, None)

    print(f"Incremental rebuild target={args.target} ...")
    incremental_seconds = _run_build(build_dir, args.target, clean=False)
    print(f"  → {incremental_seconds:.2f} s")

    ratio = incremental_seconds / max(clean_seconds, 1e-9)
    print(f"\nClean:        {clean_seconds:.2f} s")
    print(f"Incremental:  {incremental_seconds:.2f} s")
    print(f"Ratio:        {ratio*100:.1f} %  "
          f"(Phase 2 target ≤ 10 %, current baseline often 30–50 %)")

    if args.json:
        out = {
            "target": args.target,
            "touch_file": str(touch_path),
            "clean_seconds": clean_seconds,
            "incremental_seconds": incremental_seconds,
            "incremental_ratio": ratio,
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nWrote artifact: {args.json}")


if __name__ == "__main__":
    main()
