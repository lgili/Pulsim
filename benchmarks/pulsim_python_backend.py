#!/usr/bin/env python3
"""Python runtime backend for benchmark runners."""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class BackendRunResult:
    runtime_s: float
    steps: int
    mode: str
    telemetry: Dict[str, Optional[float]]


def _import_pulsim():
    try:
        import pulsim as ps

        return ps
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parent.parent
    build_python = repo_root / "build" / "python"
    if build_python.exists():
        build_python_str = str(build_python)
        if build_python_str not in sys.path:
            sys.path.insert(0, build_python_str)
        try:
            import pulsim as ps

            return ps
        except Exception:
            return None

    return None


def _has_runtime_api(ps: object) -> bool:
    required = [
        "YamlParser",
        "Simulator",
        "SimulationOptions",
        "SimulationResult",
        "PeriodicSteadyStateOptions",
        "HarmonicBalanceOptions",
    ]
    return all(hasattr(ps, name) for name in required)


def _raise_if_parser_errors(parser: object) -> None:
    errors = list(getattr(parser, "errors", []))
    if errors:
        raise RuntimeError("; ".join(str(item) for item in errors))


def _write_state_csv(
    output_path: Path,
    times: Sequence[float],
    states: Sequence[Sequence[float]],
    signal_names: Sequence[str],
) -> int:
    if len(times) != len(states):
        raise RuntimeError("Simulation result time/state length mismatch")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", *signal_names])
        for t, state in zip(times, states):
            if len(state) != len(signal_names):
                raise RuntimeError("Simulation state size mismatch with circuit signals")
            row = [f"{float(t):.9e}"]
            row.extend(f"{float(value):.9e}" for value in state)
            writer.writerow(row)

    return max(0, len(times) - 1)


def _reshape_hb_solution(solution: Sequence[float], state_size: int) -> List[List[float]]:
    if state_size <= 0:
        raise RuntimeError("Invalid state size for harmonic balance output")
    if len(solution) % state_size != 0:
        raise RuntimeError("Harmonic balance solution size is not divisible by state size")

    samples = len(solution) // state_size
    out: List[List[float]] = []
    for idx in range(samples):
        start = idx * state_size
        end = start + state_size
        out.append([float(value) for value in solution[start:end]])
    return out


def _transient_telemetry(result: object, runtime_s: float) -> Dict[str, Optional[float]]:
    linear = getattr(result, "linear_solver_telemetry", None)
    backend = getattr(result, "backend_telemetry", None)

    state_space_primary_steps = (
        float(getattr(backend, "state_space_primary_steps", 0.0)) if backend is not None else None
    )
    dae_fallback_steps = (
        float(getattr(backend, "dae_fallback_steps", 0.0)) if backend is not None else None
    )
    segment_non_admissible_steps = (
        float(getattr(backend, "segment_non_admissible_steps", 0.0)) if backend is not None else None
    )
    segment_model_cache_hits = (
        float(getattr(backend, "segment_model_cache_hits", 0.0)) if backend is not None else None
    )
    segment_model_cache_misses = (
        float(getattr(backend, "segment_model_cache_misses", 0.0)) if backend is not None else None
    )

    state_space_primary_ratio: Optional[float]
    if state_space_primary_steps is None or dae_fallback_steps is None:
        state_space_primary_ratio = None
    else:
        total_path_steps = state_space_primary_steps + dae_fallback_steps
        state_space_primary_ratio = (
            state_space_primary_steps / total_path_steps if total_path_steps > 0.0 else None
        )

    return {
        "newton_iterations": float(getattr(result, "newton_iterations_total", 0)),
        "linear_iterations": float(getattr(linear, "total_iterations", 0)) if linear is not None else None,
        "linear_solve_calls": float(getattr(linear, "total_solve_calls", 0)) if linear is not None else None,
        "linear_fallbacks": float(getattr(linear, "total_fallbacks", 0)) if linear is not None else None,
        "linear_analyze_calls": float(getattr(linear, "total_analyze_calls", 0)) if linear is not None else None,
        "linear_factorize_calls": float(getattr(linear, "total_factorize_calls", 0)) if linear is not None else None,
        "linear_analyze_time_s": float(getattr(linear, "total_analyze_time_seconds", 0.0))
        if linear is not None
        else None,
        "linear_factorize_time_s": float(getattr(linear, "total_factorize_time_seconds", 0.0))
        if linear is not None
        else None,
        "linear_solve_time_s": float(getattr(linear, "total_solve_time_seconds", 0.0))
        if linear is not None
        else None,
        "residual_norm": None,
        "timestep_rejections": float(getattr(result, "timestep_rejections", 0)),
        "runtime_kernel_s": float(getattr(result, "total_time_seconds", 0.0)),
        "runtime_s": float(runtime_s),
        "steps": float(getattr(result, "total_steps", 0)),
        "state_space_primary_steps": state_space_primary_steps,
        "dae_fallback_steps": dae_fallback_steps,
        "segment_non_admissible_steps": segment_non_admissible_steps,
        "segment_model_cache_hits": segment_model_cache_hits,
        "segment_model_cache_misses": segment_model_cache_misses,
        "state_space_primary_ratio": state_space_primary_ratio,
        "equation_assemble_system_calls": float(getattr(backend, "equation_assemble_system_calls", 0))
        if backend is not None
        else None,
        "equation_assemble_residual_calls": float(getattr(backend, "equation_assemble_residual_calls", 0))
        if backend is not None
        else None,
        "equation_assemble_system_time_s": float(
            getattr(backend, "equation_assemble_system_time_seconds", 0.0)
        )
        if backend is not None
        else None,
        "equation_assemble_residual_time_s": float(
            getattr(backend, "equation_assemble_residual_time_seconds", 0.0)
        )
        if backend is not None
        else None,
        "python_backend": 1.0,
    }


def _hb_telemetry(result: object, runtime_s: float, steps: int) -> Dict[str, Optional[float]]:
    return {
        "newton_iterations": float(getattr(result, "iterations", 0)),
        "linear_iterations": None,
        "linear_solve_calls": None,
        "linear_fallbacks": None,
        "residual_norm": float(getattr(result, "residual_norm", 0.0)),
        "timestep_rejections": None,
        "runtime_kernel_s": None,
        "runtime_s": float(runtime_s),
        "steps": float(steps),
        "python_backend": 1.0,
        "harmonic_balance_iterations": float(getattr(result, "iterations", 0)),
        "harmonic_balance_residual_norm": float(getattr(result, "residual_norm", 0.0)),
    }


def _run_transient_with_optional_x0(simulator: object, x0: Optional[object]) -> object:
    if x0 is not None:
        return simulator.run_transient(x0)
    return simulator.run_transient()


def _clone_shooting_options(ps: object, src: object) -> object:
    dst = ps.PeriodicSteadyStateOptions()
    dst.period = src.period
    dst.max_iterations = src.max_iterations
    dst.tolerance = src.tolerance
    dst.relaxation = src.relaxation
    dst.store_last_transient = src.store_last_transient
    return dst


def _select_mode(options: object, preferred_mode: Optional[str]) -> str:
    valid_modes = {"transient", "shooting", "harmonic_balance"}
    if preferred_mode is not None and preferred_mode not in valid_modes:
        raise RuntimeError(f"Invalid preferred mode: {preferred_mode}")

    enable_shooting = bool(getattr(options, "enable_periodic_shooting", False))
    enable_hb = bool(getattr(options, "enable_harmonic_balance", False))

    if preferred_mode == "transient":
        return "transient"
    if preferred_mode == "shooting":
        if not enable_shooting:
            raise RuntimeError("Scenario requested shooting mode but simulation.shooting is not configured")
        return "shooting"
    if preferred_mode == "harmonic_balance":
        if not enable_hb:
            raise RuntimeError("Scenario requested harmonic_balance mode but simulation.harmonic_balance is not configured")
        return "harmonic_balance"

    if enable_shooting and not enable_hb:
        return "shooting"
    if enable_hb and not enable_shooting:
        return "harmonic_balance"
    if enable_shooting and enable_hb:
        return "shooting"
    return "transient"


def is_available() -> bool:
    ps = _import_pulsim()
    return bool(ps and _has_runtime_api(ps))


def run_from_yaml(
    netlist_path: Path,
    output_path: Path,
    preferred_mode: Optional[str] = None,
    use_initial_conditions: bool = False,
) -> BackendRunResult:
    """Run benchmark netlist using Pulsim Python runtime and write output CSV."""

    ps = _import_pulsim()
    if ps is None:
        raise RuntimeError("Python package 'pulsim' is not available")
    if not _has_runtime_api(ps):
        raise RuntimeError("Installed pulsim package does not expose required runtime APIs")

    parser_options = ps.YamlParserOptions()
    parser_options.strict = False
    parser = ps.YamlParser(parser_options)
    circuit, options = parser.load(str(netlist_path))
    _raise_if_parser_errors(parser)

    # Some solver internals require dimensions to be set before Simulator construction.
    options.newton_options.num_nodes = int(circuit.num_nodes())
    options.newton_options.num_branches = int(circuit.num_branches())

    signal_names = list(circuit.signal_names())
    mode = _select_mode(options, preferred_mode)
    x0 = circuit.initial_state() if use_initial_conditions else None
    integrator_mapped_to_rosen = False

    if (
        mode == "transient"
        and hasattr(ps, "Integrator")
        and getattr(options, "integrator", None) == ps.Integrator.TRBDF2
    ):
        # Temporary migration mapping: TRBDF2 path is not yet parity-stable in benchmark runtime.
        options.integrator = ps.Integrator.RosenbrockW
        integrator_mapped_to_rosen = True

    simulator = ps.Simulator(circuit, options)

    if mode == "shooting":
        shooting_opts = options.periodic_options
        shooting_opts.store_last_transient = True
        used_warm_start = False

        start = time.perf_counter()
        if x0 is not None:
            shooting_result = simulator.run_periodic_shooting(x0, shooting_opts)
        else:
            shooting_result = simulator.run_periodic_shooting(shooting_opts)
        elapsed = time.perf_counter() - start

        if not shooting_result.success:
            # Warm-start shooting with a long transient trajectory to improve periodic convergence.
            warm_sim = ps.Simulator(circuit, options)
            warm_state = x0 if x0 is not None else circuit.initial_state()
            warm_ok = False
            for _ in range(5):
                warm = warm_sim.run_transient(warm_state)
                if not warm.success or not warm.states:
                    warm_ok = False
                    break
                warm_ok = True
                warm_state = warm.states[-1]

            if warm_ok:
                boosted = _clone_shooting_options(ps, shooting_opts)
                boosted.max_iterations = max(boosted.max_iterations * 6, 120)
                boosted.store_last_transient = True
                retry_sim = ps.Simulator(circuit, options)
                retry = retry_sim.run_periodic_shooting(warm_state, boosted)
                if retry.success:
                    shooting_result = retry
                    elapsed = time.perf_counter() - start
                    used_warm_start = True

        if not shooting_result.success:
            raise RuntimeError(shooting_result.message or "Periodic shooting failed")

        cycle = shooting_result.last_cycle
        steps = _write_state_csv(output_path, cycle.time, cycle.states, signal_names)
        telemetry = _transient_telemetry(cycle, elapsed)
        telemetry["periodic_iterations"] = float(shooting_result.iterations)
        telemetry["periodic_residual_norm"] = float(shooting_result.residual_norm)
        telemetry["steps"] = float(steps)
        telemetry["shooting_warm_start_retry"] = 1.0 if used_warm_start else 0.0

        return BackendRunResult(
            runtime_s=elapsed,
            steps=steps,
            mode="shooting",
            telemetry=telemetry,
        )

    if mode == "harmonic_balance":
        start = time.perf_counter()
        if x0 is not None:
            hb_result = simulator.run_harmonic_balance(x0, options.harmonic_balance)
        else:
            hb_result = simulator.run_harmonic_balance(options.harmonic_balance)
        elapsed = time.perf_counter() - start

        if not hb_result.success:
            raise RuntimeError(hb_result.message or "Harmonic balance failed")

        times = list(hb_result.sample_times)
        states = _reshape_hb_solution(hb_result.solution, len(signal_names))
        if len(times) != len(states):
            raise RuntimeError("Harmonic balance output time/state length mismatch")

        steps = _write_state_csv(output_path, times, states, signal_names)
        telemetry = _hb_telemetry(hb_result, elapsed, steps)

        return BackendRunResult(
            runtime_s=elapsed,
            steps=steps,
            mode="harmonic_balance",
            telemetry=telemetry,
        )

    start = time.perf_counter()
    transient_result = _run_transient_with_optional_x0(simulator, x0)
    used_integrator_fallback = False

    if (
        not transient_result.success
        and hasattr(ps, "Integrator")
        and getattr(options, "integrator", None) == ps.Integrator.TRBDF2
    ):
        # TRBDF2 can stall on some benchmark combinations; retry with RosenbrockW.
        retry_options = options
        retry_options.integrator = ps.Integrator.RosenbrockW
        retry_options.newton_options.num_nodes = int(circuit.num_nodes())
        retry_options.newton_options.num_branches = int(circuit.num_branches())
        retry_sim = ps.Simulator(circuit, retry_options)
        retry_result = _run_transient_with_optional_x0(retry_sim, x0)
        if retry_result.success:
            transient_result = retry_result
            used_integrator_fallback = True
    elapsed = time.perf_counter() - start

    if not transient_result.success:
        raise RuntimeError(transient_result.message or "Transient simulation failed")

    steps = _write_state_csv(output_path, transient_result.time, transient_result.states, signal_names)
    telemetry = _transient_telemetry(transient_result, elapsed)
    telemetry["steps"] = float(steps)
    telemetry["integrator_fallback_to_rosenbrockw"] = 1.0 if used_integrator_fallback else 0.0
    telemetry["integrator_mapped_to_rosenbrockw"] = 1.0 if integrator_mapped_to_rosen else 0.0

    return BackendRunResult(
        runtime_s=elapsed,
        steps=steps,
        mode="transient",
        telemetry=telemetry,
    )
