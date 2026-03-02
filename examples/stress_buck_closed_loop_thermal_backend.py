#!/usr/bin/env python3
"""Stress sweep for closed-loop electrothermal buck backend netlist.

Runs the same netlist repeatedly with increasing tstop (10ms, 20ms, 30ms, ...)
until a failure criterion is reached, then writes machine-readable reports.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import multiprocessing as mp
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required: {exc}") from exc


def import_pulsim() -> Any:
    """Import pulsim, preferring local build/python from this repository."""
    repo_root = Path(__file__).resolve().parents[1]
    build_python = repo_root / "build" / "python"
    if build_python.exists():
        build_python_str = str(build_python)
        if build_python_str not in sys.path:
            # Always prioritize the local bindings for reproducible backend diagnostics.
            sys.path.insert(0, build_python_str)

    try:
        import pulsim as ps  # type: ignore

        return ps
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Unable to import pulsim from local build/python or environment. "
            "Build bindings or set PYTHONPATH=build/python."
        ) from exc


@dataclass
class SweepCase:
    tstop_ms: int
    status: str
    reason: str
    runtime_s: float
    total_steps: int
    timestep_rejections: int
    rejection_ratio: Optional[float]
    newton_iterations: int
    newton_per_step: Optional[float]
    state_space_primary_ratio: Optional[float]
    linear_factor_cache_miss_ratio: Optional[float]
    final_time_s: Optional[float]
    completion_ratio: Optional[float]
    thermal_peak_temperature_c: Optional[float]
    thermal_final_temperature_c: Optional[float]
    loss_total_power_w: Optional[float]
    loss_total_energy_j: Optional[float]
    final_vout_v: Optional[float]
    final_sw_v: Optional[float]
    final_pi_output: Optional[float]
    final_pwm_duty: Optional[float]
    fragilities: List[str]
    components: List[Dict[str, Any]]


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML root in {path}")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    except Exception:
        return None


def collect_component_rows(result: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in list(getattr(result, "component_electrothermal", [])):
        rows.append(
            {
                "component_name": str(getattr(item, "component_name", "")),
                "thermal_enabled": bool(getattr(item, "thermal_enabled", False)),
                "conduction_w": as_float(getattr(item, "conduction", None)),
                "turn_on_w": as_float(getattr(item, "turn_on", None)),
                "turn_off_w": as_float(getattr(item, "turn_off", None)),
                "reverse_recovery_w": as_float(getattr(item, "reverse_recovery", None)),
                "total_loss_w": as_float(getattr(item, "total_loss", None)),
                "total_energy_j": as_float(getattr(item, "total_energy", None)),
                "average_power_w": as_float(getattr(item, "average_power", None)),
                "peak_power_w": as_float(getattr(item, "peak_power", None)),
                "final_temperature_c": as_float(getattr(item, "final_temperature", None)),
                "peak_temperature_c": as_float(getattr(item, "peak_temperature", None)),
                "average_temperature_c": as_float(getattr(item, "average_temperature", None)),
            }
        )
    if rows:
        return rows

    # Fallback path for builds where unified component telemetry is empty:
    # reconstruct per-device rows from loss + thermal summaries.
    loss_by_name: Dict[str, Any] = {}
    loss_summary = getattr(result, "loss_summary", None)
    for item in list(getattr(loss_summary, "device_losses", [])):
        loss_by_name[str(getattr(item, "device_name", ""))] = item

    thermal_by_name: Dict[str, Any] = {}
    thermal_summary = getattr(result, "thermal_summary", None)
    for item in list(getattr(thermal_summary, "device_temperatures", [])):
        thermal_by_name[str(getattr(item, "device_name", ""))] = item

    names = sorted(set(loss_by_name.keys()) | set(thermal_by_name.keys()))
    for name in names:
        loss_item = loss_by_name.get(name)
        thermal_item = thermal_by_name.get(name)
        breakdown = getattr(loss_item, "breakdown", None) if loss_item is not None else None
        total_loss_w = None
        if breakdown is not None:
            total_attr = getattr(breakdown, "total", None)
            if callable(total_attr):
                total_loss_w = as_float(total_attr())
            else:
                total_loss_w = as_float(total_attr)
        rows.append(
            {
                "component_name": name,
                "thermal_enabled": bool(getattr(thermal_item, "enabled", False))
                if thermal_item is not None
                else False,
                "conduction_w": as_float(getattr(breakdown, "conduction", None))
                if breakdown is not None
                else None,
                "turn_on_w": as_float(getattr(breakdown, "turn_on", None))
                if breakdown is not None
                else None,
                "turn_off_w": as_float(getattr(breakdown, "turn_off", None))
                if breakdown is not None
                else None,
                "reverse_recovery_w": as_float(getattr(breakdown, "reverse_recovery", None))
                if breakdown is not None
                else None,
                "total_loss_w": total_loss_w,
                "total_energy_j": as_float(getattr(loss_item, "total_energy", None))
                if loss_item is not None
                else None,
                "average_power_w": as_float(getattr(loss_item, "average_power", None))
                if loss_item is not None
                else None,
                "peak_power_w": as_float(getattr(loss_item, "peak_power", None))
                if loss_item is not None
                else None,
                "final_temperature_c": as_float(getattr(thermal_item, "final_temperature", None))
                if thermal_item is not None
                else None,
                "peak_temperature_c": as_float(getattr(thermal_item, "peak_temperature", None))
                if thermal_item is not None
                else None,
                "average_temperature_c": as_float(getattr(thermal_item, "average_temperature", None))
                if thermal_item is not None
                else None,
            }
        )
    return rows


def detect_fragilities(
    rejection_ratio: Optional[float],
    newton_per_step: Optional[float],
    state_space_primary_ratio: Optional[float],
    linear_factor_cache_miss_ratio: Optional[float],
    thermal_peak_temperature_c: Optional[float],
) -> List[str]:
    flags: List[str] = []
    if rejection_ratio is not None and rejection_ratio > 0.20:
        flags.append("high_timestep_rejection_ratio")
    if newton_per_step is not None and newton_per_step > 12.0:
        flags.append("high_newton_iterations_per_step")
    if state_space_primary_ratio is not None and state_space_primary_ratio > 0.98:
        flags.append("nearly_all_steps_state_space_path")
    if linear_factor_cache_miss_ratio is not None and linear_factor_cache_miss_ratio > 0.70:
        flags.append("high_linear_factor_cache_miss_ratio")
    if thermal_peak_temperature_c is not None and thermal_peak_temperature_c > 150.0:
        flags.append("mosfet_like_peak_temperature_over_150c")
    return flags


def run_case(
    template: Dict[str, Any],
    tstop_ms: int,
    runtime_limit_s: float,
) -> SweepCase:
    ps = import_pulsim()
    netlist = copy.deepcopy(template)
    netlist.setdefault("simulation", {})
    sim = netlist["simulation"]
    if not isinstance(sim, dict):
        raise RuntimeError("simulation block must be a map")
    sim["tstop"] = float(tstop_ms) * 1e-3

    with tempfile.TemporaryDirectory(prefix="pulsim_buck_stress_") as tmpdir:
        tmp_yaml = Path(tmpdir) / f"buck_thermal_{tstop_ms:04d}ms.yaml"
        write_yaml(tmp_yaml, netlist)

        parser_options = ps.YamlParserOptions()
        parser_options.strict = False
        parser = ps.YamlParser(parser_options)
        circuit, options = parser.load(str(tmp_yaml))

        parser_errors = list(getattr(parser, "errors", []))
        if parser_errors:
            return SweepCase(
                tstop_ms=tstop_ms,
                status="failed",
                reason="parser_error: " + "; ".join(str(e) for e in parser_errors),
                runtime_s=0.0,
                total_steps=0,
                timestep_rejections=0,
                rejection_ratio=None,
                newton_iterations=0,
                newton_per_step=None,
                state_space_primary_ratio=None,
                linear_factor_cache_miss_ratio=None,
                final_time_s=None,
                completion_ratio=None,
                thermal_peak_temperature_c=None,
                thermal_final_temperature_c=None,
                loss_total_power_w=None,
                loss_total_energy_j=None,
                final_vout_v=None,
                final_sw_v=None,
                final_pi_output=None,
                final_pwm_duty=None,
                fragilities=[],
                components=[],
            )

        options.newton_options.num_nodes = int(circuit.num_nodes())
        options.newton_options.num_branches = int(circuit.num_branches())

        simulator = ps.Simulator(circuit, options)
        x0 = circuit.initial_state()

        t0 = time.perf_counter()
        result = simulator.run_transient(x0)
        elapsed = time.perf_counter() - t0

        total_steps = int(getattr(result, "total_steps", 0))
        timestep_rejections = int(getattr(result, "timestep_rejections", 0))
        newton_iterations = int(getattr(result, "newton_iterations_total", 0))
        attempts = total_steps + timestep_rejections
        rejection_ratio = (
            float(timestep_rejections) / float(attempts)
            if attempts > 0
            else None
        )
        newton_per_step = (
            float(newton_iterations) / float(total_steps)
            if total_steps > 0
            else None
        )

        backend = getattr(result, "backend_telemetry", None)
        state_space_primary_steps = as_float(
            getattr(backend, "state_space_primary_steps", None)
        )
        dae_fallback_steps = as_float(getattr(backend, "dae_fallback_steps", None))
        state_space_primary_ratio = None
        if state_space_primary_steps is not None and dae_fallback_steps is not None:
            denom = state_space_primary_steps + dae_fallback_steps
            if denom > 0.0:
                state_space_primary_ratio = state_space_primary_steps / denom

        linear = getattr(result, "linear_solver_telemetry", None)
        linear_factorize_calls = as_float(
            getattr(linear, "total_factorize_calls", None)
        )
        linear_factor_cache_misses = as_float(
            getattr(backend, "linear_factor_cache_misses", None)
        )
        linear_factor_cache_miss_ratio = None
        if (
            linear_factorize_calls is not None
            and linear_factorize_calls > 0.0
            and linear_factor_cache_misses is not None
        ):
            linear_factor_cache_miss_ratio = (
                linear_factor_cache_misses / linear_factorize_calls
            )

        times = list(getattr(result, "time", []))
        final_time_s = as_float(times[-1]) if times else None
        target_tstop_s = float(tstop_ms) * 1e-3
        completion_ratio = (
            final_time_s / target_tstop_s
            if final_time_s is not None and target_tstop_s > 0.0
            else None
        )

        thermal_summary = getattr(result, "thermal_summary", None)
        thermal_peak_temperature_c = as_float(
            getattr(thermal_summary, "max_temperature", None)
            if thermal_summary is not None
            else None
        )
        thermal_final_temperature_c = None
        if thermal_summary is not None:
            device_temps = list(getattr(thermal_summary, "device_temperatures", []))
            if device_temps:
                thermal_final_temperature_c = max(
                    as_float(getattr(item, "final_temperature", None)) or 25.0
                    for item in device_temps
                )
            else:
                thermal_final_temperature_c = thermal_peak_temperature_c

        loss_summary = getattr(result, "loss_summary", None)
        loss_total_power_w = as_float(
            getattr(loss_summary, "total_loss", None) if loss_summary is not None else None
        )
        loss_total_energy_j = None
        if loss_summary is not None:
            rows = list(getattr(loss_summary, "device_losses", []))
            loss_total_energy_j = sum(
                as_float(getattr(item, "total_energy", None)) or 0.0 for item in rows
            )

        component_rows = collect_component_rows(result)
        virtual_channels = getattr(result, "virtual_channels", {})
        final_vout_v = None
        final_sw_v = None
        final_pi_output = None
        final_pwm_duty = None
        try:
            xout = list(virtual_channels.get("Xout", []))
            xsw = list(virtual_channels.get("Xsw", []))
            pi = list(virtual_channels.get("PI1", []))
            duty = list(virtual_channels.get("PWM1.duty", []))
            final_vout_v = as_float(xout[-1]) if xout else None
            final_sw_v = as_float(xsw[-1]) if xsw else None
            final_pi_output = as_float(pi[-1]) if pi else None
            final_pwm_duty = as_float(duty[-1]) if duty else None
        except Exception:
            final_vout_v = None
            final_sw_v = None
            final_pi_output = None
            final_pwm_duty = None
        fragilities = detect_fragilities(
            rejection_ratio,
            newton_per_step,
            state_space_primary_ratio,
            linear_factor_cache_miss_ratio,
            thermal_peak_temperature_c,
        )

        status = "passed"
        reason = ""
        if not bool(getattr(result, "success", False)):
            status = "failed"
            reason = str(getattr(result, "message", "solver_failure"))
        elif completion_ratio is not None and completion_ratio < 0.99:
            status = "failed"
            reason = f"incomplete_time_coverage: {completion_ratio:.6f}"
        elif elapsed > runtime_limit_s:
            status = "failed"
            reason = f"runtime_limit_exceeded: {elapsed:.6f}s > {runtime_limit_s:.6f}s"
        elif total_steps <= 0:
            status = "failed"
            reason = "zero_steps"

        return SweepCase(
            tstop_ms=tstop_ms,
            status=status,
            reason=reason,
            runtime_s=float(elapsed),
            total_steps=total_steps,
            timestep_rejections=timestep_rejections,
            rejection_ratio=rejection_ratio,
            newton_iterations=newton_iterations,
            newton_per_step=newton_per_step,
            state_space_primary_ratio=state_space_primary_ratio,
            linear_factor_cache_miss_ratio=linear_factor_cache_miss_ratio,
            final_time_s=final_time_s,
            completion_ratio=completion_ratio,
            thermal_peak_temperature_c=thermal_peak_temperature_c,
            thermal_final_temperature_c=thermal_final_temperature_c,
            loss_total_power_w=loss_total_power_w,
            loss_total_energy_j=loss_total_energy_j,
            final_vout_v=final_vout_v,
            final_sw_v=final_sw_v,
            final_pi_output=final_pi_output,
            final_pwm_duty=final_pwm_duty,
            fragilities=fragilities,
            components=component_rows,
        )


def _run_case_worker(
    template: Dict[str, Any],
    tstop_ms: int,
    runtime_limit_s: float,
    queue: Any,
) -> None:
    try:
        case = run_case(template, tstop_ms, runtime_limit_s)
        queue.put(asdict(case))
    except Exception as exc:  # pragma: no cover
        queue.put(
            {
                "tstop_ms": tstop_ms,
                "status": "failed",
                "reason": f"worker_exception: {exc.__class__.__name__}: {exc}",
                "runtime_s": 0.0,
                "total_steps": 0,
                "timestep_rejections": 0,
                "rejection_ratio": None,
                "newton_iterations": 0,
                "newton_per_step": None,
                "state_space_primary_ratio": None,
                "linear_factor_cache_miss_ratio": None,
                "final_time_s": None,
                "completion_ratio": None,
                "thermal_peak_temperature_c": None,
                "thermal_final_temperature_c": None,
                "loss_total_power_w": None,
                "loss_total_energy_j": None,
                "final_vout_v": None,
                "final_sw_v": None,
                "final_pi_output": None,
                "final_pwm_duty": None,
                "fragilities": [],
                "components": [],
            }
        )


def write_reports(output_dir: Path, cases: List[SweepCase], netlist: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stress_summary.json"
    csv_path = output_dir / "stress_summary.csv"

    first_failure = next((case.tstop_ms for case in cases if case.status != "passed"), None)

    payload = {
        "netlist": str(netlist),
        "cases": [asdict(case) for case in cases],
        "first_failure_tstop_ms": first_failure,
        "max_pass_tstop_ms": max((case.tstop_ms for case in cases if case.status == "passed"), default=None),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tstop_ms",
                "status",
                "reason",
                "runtime_s",
                "total_steps",
                "timestep_rejections",
                "rejection_ratio",
                "newton_iterations",
                "newton_per_step",
                "state_space_primary_ratio",
                "linear_factor_cache_miss_ratio",
                "completion_ratio",
                "thermal_peak_temperature_c",
                "thermal_final_temperature_c",
                "loss_total_power_w",
                "loss_total_energy_j",
                "final_vout_v",
                "final_sw_v",
                "final_pi_output",
                "final_pwm_duty",
                "fragilities",
            ]
        )
        for case in cases:
            writer.writerow(
                [
                    case.tstop_ms,
                    case.status,
                    case.reason,
                    f"{case.runtime_s:.6f}",
                    case.total_steps,
                    case.timestep_rejections,
                    "" if case.rejection_ratio is None else f"{case.rejection_ratio:.6f}",
                    case.newton_iterations,
                    "" if case.newton_per_step is None else f"{case.newton_per_step:.6f}",
                    ""
                    if case.state_space_primary_ratio is None
                    else f"{case.state_space_primary_ratio:.6f}",
                    ""
                    if case.linear_factor_cache_miss_ratio is None
                    else f"{case.linear_factor_cache_miss_ratio:.6f}",
                    "" if case.completion_ratio is None else f"{case.completion_ratio:.6f}",
                    ""
                    if case.thermal_peak_temperature_c is None
                    else f"{case.thermal_peak_temperature_c:.6f}",
                    ""
                    if case.thermal_final_temperature_c is None
                    else f"{case.thermal_final_temperature_c:.6f}",
                    "" if case.loss_total_power_w is None else f"{case.loss_total_power_w:.9e}",
                    "" if case.loss_total_energy_j is None else f"{case.loss_total_energy_j:.9e}",
                    "" if case.final_vout_v is None else f"{case.final_vout_v:.9e}",
                    "" if case.final_sw_v is None else f"{case.final_sw_v:.9e}",
                    "" if case.final_pi_output is None else f"{case.final_pi_output:.9e}",
                    "" if case.final_pwm_duty is None else f"{case.final_pwm_duty:.9e}",
                    "|".join(case.fragilities),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stress-sweep closed-loop electrothermal buck netlist"
    )
    parser.add_argument(
        "--netlist",
        type=Path,
        default=Path("examples/09_buck_closed_loop_loss_thermal_validation_backend.yaml"),
    )
    parser.add_argument("--start-ms", type=int, default=10)
    parser.add_argument("--step-ms", type=int, default=10)
    parser.add_argument("--max-ms", type=int, default=300)
    parser.add_argument("--runtime-limit-s", type=float, default=60.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/out_buck_closed_loop_thermal_stress"),
    )
    args = parser.parse_args()

    if args.start_ms <= 0 or args.step_ms <= 0 or args.max_ms < args.start_ms:
        raise SystemExit("Invalid sweep range")

    template = load_yaml(args.netlist.resolve())
    ctx = mp.get_context("spawn")

    cases: List[SweepCase] = []
    tstop_ms = args.start_ms
    while tstop_ms <= args.max_ms:
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_run_case_worker,
            args=(template, tstop_ms, args.runtime_limit_s, queue),
        )
        proc.start()
        proc.join(args.runtime_limit_s + 5.0)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            case = SweepCase(
                tstop_ms=tstop_ms,
                status="failed",
                reason=f"hard_timeout: exceeded {args.runtime_limit_s:.2f}s",
                runtime_s=args.runtime_limit_s,
                total_steps=0,
                timestep_rejections=0,
                rejection_ratio=None,
                newton_iterations=0,
                newton_per_step=None,
                state_space_primary_ratio=None,
                linear_factor_cache_miss_ratio=None,
                final_time_s=None,
                completion_ratio=None,
                thermal_peak_temperature_c=None,
                thermal_final_temperature_c=None,
                loss_total_power_w=None,
                loss_total_energy_j=None,
                final_vout_v=None,
                final_sw_v=None,
                final_pi_output=None,
                final_pwm_duty=None,
                fragilities=["hard_timeout"],
                components=[],
            )
        else:
            if queue.empty():
                case = SweepCase(
                    tstop_ms=tstop_ms,
                    status="failed",
                    reason="worker_no_result",
                    runtime_s=0.0,
                    total_steps=0,
                    timestep_rejections=0,
                    rejection_ratio=None,
                    newton_iterations=0,
                    newton_per_step=None,
                    state_space_primary_ratio=None,
                    linear_factor_cache_miss_ratio=None,
                    final_time_s=None,
                    completion_ratio=None,
                    thermal_peak_temperature_c=None,
                    thermal_final_temperature_c=None,
                    loss_total_power_w=None,
                    loss_total_energy_j=None,
                    final_vout_v=None,
                    final_sw_v=None,
                    final_pi_output=None,
                    final_pwm_duty=None,
                    fragilities=["worker_no_result"],
                    components=[],
                )
            else:
                payload = queue.get()
                case = SweepCase(**payload)
        cases.append(case)
        print(
            f"tstop={case.tstop_ms:4d}ms status={case.status:6s} "
            f"runtime={case.runtime_s:8.3f}s steps={case.total_steps:8d} "
            f"rejections={case.timestep_rejections:8d} reason={case.reason}"
        )
        if case.status != "passed":
            break
        tstop_ms += args.step_ms

    write_reports(args.output_dir.resolve(), cases, args.netlist.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
