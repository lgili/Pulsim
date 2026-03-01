"""Tests for v1 layered dependency boundary checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_v1_layer_boundary_checker_passes_on_mapped_core() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checker = repo_root / "scripts" / "check_v1_layer_boundaries.py"
    layer_map = repo_root / "core" / "v1_layer_map.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(checker),
            "--project-root",
            str(repo_root),
            "--map",
            str(layer_map),
            "--strict",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "passed"
    assert report["violations"] == []
    assert report["missing_files"] == []
    assert report["checked_file_count"] > 0
