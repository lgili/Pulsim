"""Deterministic replay checks for modular transient runtime execution."""

from __future__ import annotations

import math
from pathlib import Path

import pulsim as ps


def _load_closed_loop_example() -> tuple[ps.Circuit, ps.SimulationOptions]:
    root = Path(__file__).resolve().parents[2]
    example = root / "examples" / "09_buck_closed_loop_loss_thermal_validation_backend.yaml"
    parser = ps.YamlParser()
    return parser.load_string(example.read_text(encoding="utf-8"))


def _run_example_once() -> ps.SimulationResult:
    circuit, options = _load_closed_loop_example()
    result = ps.Simulator(circuit, options).run_transient()
    assert result.success, result.message
    return result


def _assert_series_close(lhs: list[float], rhs: list[float], *, rel: float = 1e-10, abs_: float = 1e-12) -> None:
    assert len(lhs) == len(rhs)
    for left, right in zip(lhs, rhs):
        assert math.isclose(left, right, rel_tol=rel, abs_tol=abs_), (
            f"left={left} right={right} rel={rel} abs={abs_}"
        )


def test_modular_runtime_replay_is_deterministic_for_closed_loop_electrothermal_case() -> None:
    first = _run_example_once()
    second = _run_example_once()

    assert first.backend_telemetry.runtime_module_count >= 5
    assert first.backend_telemetry.runtime_module_count == second.backend_telemetry.runtime_module_count
    assert first.backend_telemetry.runtime_module_order == second.backend_telemetry.runtime_module_order

    _assert_series_close(first.time, second.time)

    assert set(first.virtual_channels.keys()) == set(second.virtual_channels.keys())
    for channel, series in first.virtual_channels.items():
        assert len(series) == len(first.time), f"{channel} does not match time base"
        assert len(second.virtual_channels[channel]) == len(second.time)

    for channel in ("PWM1.duty", "Ploss(M1)", "T(M1)"):
        assert channel in first.virtual_channels
        _assert_series_close(first.virtual_channels[channel], second.virtual_channels[channel])

    assert len(first.events) == len(second.events)
    for left, right in zip(first.events, second.events):
        assert left.type == right.type
        assert left.component == right.component
        assert math.isclose(left.time, right.time, rel_tol=1e-10, abs_tol=1e-12)
        assert math.isclose(left.value1, right.value1, rel_tol=1e-10, abs_tol=1e-12)
        assert math.isclose(left.value2, right.value2, rel_tol=1e-10, abs_tol=1e-12)
