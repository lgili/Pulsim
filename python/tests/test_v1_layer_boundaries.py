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


def test_runtime_loop_uses_module_orchestrator_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    simulation_cpp = repo_root / "core" / "src" / "v1" / "simulation.cpp"
    source = simulation_cpp.read_text(encoding="utf-8")

    assert "RuntimeModuleOrchestrator runtime_modules(" in source
    assert "runtime_modules.on_run_initialize(x);" in source
    assert "runtime_modules.on_sample_emit(" in source
    assert "runtime_modules.on_step_accepted(" in source
    assert "runtime_modules.on_hold_step_accepted(" in source
    assert "runtime_modules.on_finalize();" in source

    forbidden_direct_bindings = (
        "ControlMixedDomainModule control_mixed_module(",
        "EventTopologyModule event_topology_module(",
        "LossAccountingModule loss_module(",
        "ThermalCouplingModule thermal_module(",
        "ElectrothermalTelemetryModule electrothermal_module(",
    )
    for marker in forbidden_direct_bindings:
        assert marker not in source
