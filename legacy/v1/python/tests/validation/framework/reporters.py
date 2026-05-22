"""Reporting utilities for validation results.

These helpers emit machine-readable artifacts to track regressions
in accuracy, performance, and memory across runs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from .base import ValidationResult


def _to_dict(result: ValidationResult) -> dict:
    """Convert ValidationResult to a JSON-serializable dict."""
    data = asdict(result)
    # Convert numpy arrays to lists for serialization
    for key in (
        "pulsim_times",
        "pulsim_values",
        "reference_times",
        "reference_values",
    ):
        arr = data.get(key)
        if hasattr(arr, "tolist"):
            data[key] = arr.tolist()
    return data


def write_json_report(results: Iterable[ValidationResult], path: str | Path) -> Path:
    """Write validation results to a JSON file.

    Args:
        results: Iterable of ValidationResult
        path: Output JSON path
    Returns:
        Path to the written file
    """
    path = Path(path)
    payload: List[dict] = [_to_dict(r) for r in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_csv_report(results: Iterable[ValidationResult], path: str | Path) -> Path:
    """Write a compact CSV summary of validation results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "test_name",
        "passed",
        "max_error",
        "rms_error",
        "max_relative_error",
        "tolerance",
        "execution_time_ms",
        "notes",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "test_name": r.test_name,
                "passed": r.passed,
                "max_error": r.max_error,
                "rms_error": r.rms_error,
                "max_relative_error": r.max_relative_error,
                "tolerance": r.tolerance,
                "execution_time_ms": r.execution_time_ms,
                "notes": r.notes,
            })
    return path


def write_markdown_report(results: Iterable[ValidationResult], path: str | Path) -> Path:
    """Write a simple Markdown table for human-friendly review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "| Test | Status | Max Error | RMS Error | Max Rel Error (%) | Tol (%) | Time (ms) | Notes |",
        "|------|--------|-----------|-----------|-------------------|---------|----------|-------|",
    ]

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| {r.test_name} | {status} | {r.max_error:.3e} | {r.rms_error:.3e} | "
            f"{r.max_relative_error*100:.4f} | {r.tolerance*100:.2f} | {r.execution_time_ms:.2f} | {r.notes} |"
        )

    path.write_text("\n".join(lines))
    return path


__all__ = [
    "write_json_report",
    "write_csv_report",
    "write_markdown_report",
]
