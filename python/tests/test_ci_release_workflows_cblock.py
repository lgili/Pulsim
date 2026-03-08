"""Workflow contract tests for C-Block CI and release smoke checks."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_ci_workflow_has_dedicated_cblock_job() -> None:
    workflow = _load_yaml(ROOT / ".github/workflows/ci.yml")
    jobs = workflow["jobs"]

    assert "test_cblock" in jobs
    matrix = jobs["test_cblock"]["strategy"]["matrix"]
    assert "ubuntu-24.04" in matrix["os"]
    assert "macos-14" in matrix["os"]

    run_snippets = [
        step.get("run", "")
        for step in jobs["test_cblock"]["steps"]
        if isinstance(step, dict)
    ]
    assert any("pytest python/tests/test_cblock.py -v" in text for text in run_snippets)


def test_ci_workflow_enforces_format_check_for_cblock_paths() -> None:
    workflow = _load_yaml(ROOT / ".github/workflows/ci.yml")
    lint_steps = workflow["jobs"]["lint"]["steps"]
    format_step = next(
        (
            step
            for step in lint_steps
            if step.get("name") == "Run ruff format check (C-Block paths)"
        ),
        None,
    )
    assert format_step is not None
    run_script = format_step["run"]
    assert "ruff format --check" in run_script
    assert "python/pulsim/cblock.py" in run_script


def test_publish_workflow_smoke_test_exercises_cblock_api() -> None:
    workflow = _load_yaml(ROOT / ".github/workflows/publish.yml")
    build_steps = workflow["jobs"]["build-wheels"]["steps"]
    build_wheels_step = next(
        (step for step in build_steps if step.get("name") == "Build wheels"),
        None,
    )
    assert build_wheels_step is not None
    test_command = build_wheels_step["env"]["CIBW_TEST_COMMAND"]
    assert "PythonCBlock" in test_command
    assert "compile_cblock" in test_command
    assert "CBlockLibrary" in test_command
