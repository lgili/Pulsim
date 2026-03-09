#!/usr/bin/env python3
"""Benchmark and validation runner for Pulsim YAML netlists."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

try:
    from pulsim_python_backend import availability_error as python_backend_error
    from pulsim_python_backend import is_available as python_backend_available
    from pulsim_python_backend import run_from_yaml as run_pulsim_python
except ImportError:  # pragma: no cover - local import fallback
    python_backend_error = None
    python_backend_available = None
    run_pulsim_python = None


@dataclass
class PulsimRunResult:
    runtime_s: float
    steps: int
    mode: str
    telemetry: Dict[str, Optional[float]]


@dataclass
class ScenarioResult:
    benchmark_id: str
    scenario: str
    status: str
    runtime_s: float
    steps: int
    max_error: Optional[float]
    rms_error: Optional[float]
    message: str
    telemetry: Dict[str, Optional[float]]
    mode: str = "transient"
    phase_error_deg: Optional[float] = None


def can_use_pulsim_python_backend() -> bool:
    if python_backend_available is None:
        return False
    try:
        return bool(python_backend_available())
    except Exception:
        return False


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        raise ValueError("Missing value")
    raw = str(value).strip().lower()
    suffixes = {
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "µ": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "meg": 1e6,
        "g": 1e9,
        "t": 1e12,
    }
    if raw.endswith("meg"):
        return float(raw[:-3]) * suffixes["meg"]
    for suffix, multiplier in suffixes.items():
        if suffix != "meg" and raw.endswith(suffix):
            return float(raw[: -len(suffix)]) * multiplier
    return float(raw)


def coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return parse_value(value)


def infer_preferred_mode(scenario_name: str, scenario_override: Dict[str, Any]) -> Optional[str]:
    sim = scenario_override.get("simulation") if isinstance(scenario_override, dict) else None
    if isinstance(sim, dict):
        has_shooting = "shooting" in sim
        has_hb = "harmonic_balance" in sim or "hb" in sim
        has_frequency = False
        frequency_cfg = sim.get("frequency_analysis")
        if isinstance(frequency_cfg, dict):
            has_frequency = bool(frequency_cfg.get("enabled", False))
        elif frequency_cfg is True:
            has_frequency = True

        if has_frequency and not has_shooting and not has_hb:
            return "frequency_analysis"
        if has_shooting and not has_hb:
            return "shooting"
        if has_hb and not has_shooting:
            return "harmonic_balance"

    lowered = scenario_name.lower()
    if "frequency" in lowered or "ac" in lowered:
        return "frequency_analysis"
    if "shooting" in lowered:
        return "shooting"
    if "harmonic" in lowered or lowered == "hb":
        return "harmonic_balance"
    return None


def normalize_periodic_mode(netlist: Dict[str, Any], preferred_mode: Optional[str]) -> None:
    if preferred_mode is None:
        return
    simulation = netlist.get("simulation")
    if not isinstance(simulation, dict):
        return

    if preferred_mode == "shooting":
        simulation.pop("harmonic_balance", None)
        simulation.pop("hb", None)
    elif preferred_mode == "harmonic_balance":
        simulation.pop("shooting", None)


def apply_runtime_defaults(netlist: Dict[str, Any]) -> None:
    """Apply benchmark runtime defaults without overriding explicit YAML choices."""
    simulation = netlist.get("simulation")
    if not isinstance(simulation, dict):
        return

    # Benchmark suite targets deterministic comparisons by default.
    if "adaptive_timestep" not in simulation:
        simulation["adaptive_timestep"] = False


def extract_averaged_pair_telemetry(benchmark_meta: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract deterministic averaged-pair telemetry tags from benchmark metadata."""
    if not isinstance(benchmark_meta, dict):
        return {}

    averaged_pair = benchmark_meta.get("averaged_pair")
    if not isinstance(averaged_pair, dict):
        return {}

    pair_id = averaged_pair.get("id")
    if not isinstance(pair_id, str) or not pair_id.strip():
        return {}

    role_raw = averaged_pair.get("role")
    role = str(role_raw).strip().lower() if role_raw is not None else ""
    if role not in {"switching", "averaged"}:
        return {}

    pair_crc32 = float(zlib.crc32(pair_id.strip().encode("utf-8")) & 0xFFFFFFFF)
    return {
        "averaged_pair_case": 1.0,
        "averaged_pair_group_crc32": pair_crc32,
        "averaged_pair_role_switching": 1.0 if role == "switching" else 0.0,
        "averaged_pair_role_averaged": 1.0 if role == "averaged" else 0.0,
    }


def extract_magnetic_kpi_telemetry(benchmark_meta: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract optional magnetic KPI tags from benchmark metadata."""
    if not isinstance(benchmark_meta, dict):
        return {}

    magnetic_kpi = benchmark_meta.get("magnetic_kpi")
    if not isinstance(magnetic_kpi, dict):
        return {}

    telemetry: Dict[str, Optional[float]] = {"magnetic_fixture_case": 1.0}

    category_raw = magnetic_kpi.get("category", "")
    category = str(category_raw).strip().lower()
    category_flags = {
        "saturation": "magnetic_fixture_saturation",
        "hysteresis": "magnetic_fixture_hysteresis",
        "frequency_trend": "magnetic_fixture_frequency_trend",
        "determinism": "magnetic_determinism_case",
    }
    if category in category_flags:
        telemetry[category_flags[category]] = 1.0

    trend_group = magnetic_kpi.get("trend_group")
    if isinstance(trend_group, str) and trend_group.strip():
        telemetry["magnetic_trend_group_crc32"] = float(
            zlib.crc32(trend_group.strip().encode("utf-8")) & 0xFFFFFFFF
        )

    trend_role_raw = magnetic_kpi.get("trend_role")
    trend_role = str(trend_role_raw).strip().lower() if trend_role_raw is not None else ""
    if trend_role in {"low", "high"}:
        telemetry["magnetic_trend_role_low"] = 1.0 if trend_role == "low" else 0.0
        telemetry["magnetic_trend_role_high"] = 1.0 if trend_role == "high" else 0.0

    return telemetry


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _load_series_values(series: Dict[str, List[float]], observable: str) -> List[float]:
    values = series.get(observable)
    if values is None:
        raise ValueError(f"Missing observable column: {observable}")
    if len(values) < 2:
        raise ValueError(f"Observable '{observable}' has fewer than 2 samples")
    return values


def validate_magnetic_saturation(
    series: Dict[str, List[float]],
    params: Dict[str, Any],
) -> Tuple[float, float]:
    i_observable = str(params.get("i_observable", "Lsat.i_est"))
    l_eff_observable = str(params.get("l_eff_observable", "Lsat.l_eff"))
    i_values = _load_series_values(series, i_observable)
    l_eff_values = _load_series_values(series, l_eff_observable)

    if len(i_values) != len(l_eff_values):
        raise ValueError("Magnetic saturation observables must have matching lengths")

    l_unsat = max(abs(parse_value(params["inductance"])), 1e-12)
    i_sat = max(abs(parse_value(params["saturation_current"])), 1e-12)
    l_sat_raw = abs(parse_value(params.get("saturation_inductance", l_unsat * 0.2)))
    l_sat = _clamp(l_sat_raw, 1e-12, l_unsat)
    exponent = _clamp(parse_value(params.get("saturation_exponent", 2.0)), 1.0, 8.0)

    reference: List[float] = []
    for i_value in i_values:
        ratio = math.pow(abs(float(i_value)) / i_sat, exponent)
        l_eff_ref = l_sat + (l_unsat - l_sat) / (1.0 + ratio)
        reference.append(max(l_eff_ref, 1e-12))

    return compute_errors(l_eff_values, reference)


def _expected_magnetic_core_loss_series(
    times: List[float],
    i_signed: List[float],
    h_state: List[float],
    params: Dict[str, Any],
) -> List[float]:
    core_loss_k = max(parse_value(params.get("core_loss_k", 0.0)), 0.0)
    if core_loss_k <= 0.0:
        return [0.0 for _ in times]

    core_loss_alpha = _clamp(parse_value(params.get("core_loss_alpha", 2.0)), 0.0, 8.0)
    core_loss_freq_coeff = max(parse_value(params.get("core_loss_freq_coeff", 0.0)), 0.0)
    hysteresis_loss_coeff = _clamp(
        abs(parse_value(params.get("hysteresis_loss_coeff", 0.2))),
        0.0,
        50.0,
    )
    band = max(parse_value(params.get("hysteresis_band", 0.0)), 0.0)
    magnetic_i_equiv_init = max(parse_value(params.get("magnetic_i_equiv_init", 0.0)), 0.0)

    if not (len(times) == len(i_signed) == len(h_state)):
        raise ValueError("Magnetic hysteresis observables must have matching lengths")

    out: List[float] = []
    prev_i: Optional[float] = None
    prev_t: Optional[float] = None
    for t, i_val, state in zip(times, i_signed, h_state):
        i_equiv = abs(float(i_val))
        freq_multiplier = 1.0
        if core_loss_freq_coeff > 0.0:
            if prev_i is None or prev_t is None:
                prev_i = magnetic_i_equiv_init if magnetic_i_equiv_init > 0.0 else i_equiv
                prev_t = float(t)
            else:
                dt = float(t) - prev_t
                if dt > 0.0 and math.isfinite(dt):
                    di_dt = (i_equiv - prev_i) / dt
                    additive = core_loss_freq_coeff * abs(di_dt)
                    if math.isfinite(additive) and additive > 0.0:
                        freq_multiplier += additive
                prev_i = i_equiv
                prev_t = float(t)

        direction = 1.0 if i_val > band else -1.0 if i_val < -band else 0.0
        mismatch = 0.5 if direction == 0.0 else 0.5 * (1.0 - float(state) * direction)
        hysteresis_multiplier = 1.0 + hysteresis_loss_coeff * mismatch

        core_loss = (
            core_loss_k
            * math.pow(i_equiv, core_loss_alpha)
            * freq_multiplier
            * hysteresis_multiplier
        )
        out.append(max(core_loss, 0.0))

    return out


def validate_magnetic_hysteresis(
    times: List[float],
    series: Dict[str, List[float]],
    params: Dict[str, Any],
) -> Tuple[float, float, float]:
    i_observable = str(params.get("i_observable", "Lsat.i_est"))
    h_state_observable = str(params.get("h_state_observable", "Lsat.h_state"))
    core_loss_observable = str(params.get("core_loss_observable", "Lsat.core_loss"))

    i_values = _load_series_values(series, i_observable)
    h_state_values = _load_series_values(series, h_state_observable)
    core_loss_values = _load_series_values(series, core_loss_observable)

    reference = _expected_magnetic_core_loss_series(
        times=times,
        i_signed=i_values,
        h_state=h_state_values,
        params=params,
    )
    max_error, rms_error = compute_errors(core_loss_values, reference)

    energy_actual = 0.0
    energy_reference = 0.0
    for idx in range(1, len(times)):
        dt = float(times[idx] - times[idx - 1])
        if dt <= 0.0 or not math.isfinite(dt):
            continue
        energy_actual += float(core_loss_values[idx]) * dt
        energy_reference += float(reference[idx]) * dt

    denom = max(abs(energy_actual), abs(energy_reference), 1e-12)
    cycle_energy_error = abs(energy_actual - energy_reference) / denom
    return max_error, rms_error, cycle_energy_error


def compute_magnetic_core_loss_tail_mean(
    output_path: Path,
    observable: str,
) -> Optional[float]:
    try:
        _, series = load_csv_series(output_path)
    except Exception:
        return None

    values = series.get(observable)
    if not values:
        return None
    tail = values[len(values) // 2 :]
    if not tail:
        return None
    return float(sum(tail) / len(tail))


def run_pulsim(
    netlist_path: Path,
    output_path: Path,
    preferred_mode: Optional[str] = None,
    use_initial_conditions: bool = False,
) -> PulsimRunResult:
    if run_pulsim_python is None:
        reason = "benchmark backend module import failed"
        if python_backend_error is not None:
            try:
                backend_reason = python_backend_error()
                if backend_reason:
                    reason = backend_reason
            except Exception:
                pass
        raise RuntimeError(
            "Pulsim Python runtime backend unavailable. "
            "Build Python bindings and expose them via build/python or install pulsim package. "
            f"Reason: {reason}"
        )

    raw_result = run_pulsim_python(
        netlist_path,
        output_path,
        preferred_mode=preferred_mode,
        use_initial_conditions=use_initial_conditions,
    )

    if not hasattr(raw_result, "runtime_s") or not hasattr(raw_result, "telemetry"):
        raise RuntimeError("Unexpected backend response: structured telemetry result expected")

    return PulsimRunResult(
        runtime_s=float(getattr(raw_result, "runtime_s")),
        steps=int(getattr(raw_result, "steps")),
        mode=str(getattr(raw_result, "mode")),
        telemetry=dict(getattr(raw_result, "telemetry")),
    )


def load_csv_series(path: Path) -> Tuple[List[float], Dict[str, List[float]]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        times: List[float] = []
        series: Dict[str, List[float]] = {}
        for row in reader:
            times.append(float(row["time"]))
            for key, value in row.items():
                if key == "time":
                    continue
                series.setdefault(key, []).append(float(value))
    return times, series


def load_frequency_csv_series(path: Path) -> Tuple[List[float], Dict[str, List[float]]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        frequencies: List[float] = []
        series: Dict[str, List[float]] = {}
        for row in reader:
            frequencies.append(float(row["frequency_hz"]))
            for key, value in row.items():
                if key == "frequency_hz":
                    continue
                series.setdefault(key, []).append(float(value))
    return frequencies, series


def analytical_rc_step(times: List[float], v0: float, r: float, c: float) -> List[float]:
    tau = r * c
    return [v0 * (1.0 - math.exp(-t / tau)) for t in times]


def analytical_rl_step(times: List[float], v0: float, r: float, l: float) -> List[float]:
    tau = l / r
    return [v0 * math.exp(-t / tau) for t in times]


def analytical_rlc_step(times: List[float], v0: float, r: float, l: float, c: float) -> List[float]:
    alpha = r / (2.0 * l)
    omega0 = 1.0 / math.sqrt(l * c)
    result = []
    if alpha < omega0:
        omega_d = math.sqrt(omega0 * omega0 - alpha * alpha)
        for t in times:
            value = 1.0 - math.exp(-alpha * t) * (
                math.cos(omega_d * t) + (alpha / omega_d) * math.sin(omega_d * t)
            )
            result.append(v0 * value)
    else:
        s1 = -alpha + math.sqrt(alpha * alpha - omega0 * omega0)
        s2 = -alpha - math.sqrt(alpha * alpha - omega0 * omega0)
        a = v0 * s2 / (s2 - s1)
        b = -v0 * s1 / (s2 - s1)
        for t in times:
            result.append(v0 - a * math.exp(s1 * t) - b * math.exp(s2 * t))
    return result


def analytical_rc_lowpass_frequency(
    frequencies_hz: List[float],
    r: float,
    c: float,
) -> Tuple[List[float], List[float], List[float]]:
    magnitude: List[float] = []
    magnitude_db: List[float] = []
    phase_deg: List[float] = []
    for f_hz in frequencies_hz:
        omega_rc = 2.0 * math.pi * f_hz * r * c
        mag = 1.0 / math.sqrt(1.0 + omega_rc * omega_rc)
        phase = -math.degrees(math.atan(omega_rc))
        magnitude.append(mag)
        magnitude_db.append(20.0 * math.log10(mag))
        phase_deg.append(phase)
    return magnitude, magnitude_db, phase_deg


def compute_errors(values: List[float], reference: List[float]) -> Tuple[float, float]:
    if len(values) != len(reference):
        raise ValueError("Length mismatch between values and reference")
    errors = [abs(a - b) for a, b in zip(values, reference)]
    max_error = max(errors) if errors else 0.0
    rms_error = math.sqrt(sum(err * err for err in errors) / len(errors)) if errors else 0.0
    return max_error, rms_error


def validate_ac_analytical(
    frequencies_hz: List[float],
    series: Dict[str, List[float]],
    model: str,
    params: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    if model != "rc_lowpass":
        raise ValueError(f"Unknown AC analytical model: {model}")

    if "magnitude_db" not in series:
        raise ValueError("AC result is missing 'magnitude_db' column")
    if "phase_deg" not in series:
        raise ValueError("AC result is missing 'phase_deg' column")

    r = parse_value(params["r"])
    c = parse_value(params["c"])
    _, ref_mag_db, ref_phase_deg = analytical_rc_lowpass_frequency(frequencies_hz, r, c)

    mag_max, mag_rms = compute_errors(series["magnitude_db"], ref_mag_db)
    phase_max, phase_rms = compute_errors(series["phase_deg"], ref_phase_deg)
    return mag_max, mag_rms, phase_max, phase_rms


def validate_analytical(times: List[float], values: List[float], model: str, params: Dict[str, Any]) -> Tuple[float, float]:
    v0 = parse_value(params["v0"])
    if model == "rc_step":
        r = parse_value(params["r"])
        c = parse_value(params["c"])
        reference = analytical_rc_step(times, v0, r, c)
    elif model == "rl_step":
        r = parse_value(params["r"])
        l = parse_value(params["l"])
        reference = analytical_rl_step(times, v0, r, l)
    elif model == "rlc_step":
        r = parse_value(params["r"])
        l = parse_value(params["l"])
        c = parse_value(params["c"])
        reference = analytical_rlc_step(times, v0, r, l, c)
    else:
        raise ValueError(f"Unknown analytical model: {model}")
    return compute_errors(values, reference)


def validate_reference(times: List[float], values: List[float], baseline_path: Path, column: str) -> Tuple[Optional[float], Optional[float], str]:
    if not baseline_path.exists():
        return None, None, f"Baseline missing: {baseline_path}"
    ref_times, ref_series = load_csv_series(baseline_path)
    if column not in ref_series:
        return None, None, f"Baseline missing column: {column}"
    ref_values = ref_series[column]
    if len(ref_times) < 2 or len(times) < 2:
        return None, None, "Not enough samples for baseline comparison"

    overlap_start = max(times[0], ref_times[0])
    overlap_end = min(times[-1], ref_times[-1])
    if overlap_start >= overlap_end:
        return None, None, "No overlapping time range with baseline"

    eval_times: List[float] = []
    eval_values: List[float] = []
    for t, v in zip(times, values):
        if overlap_start <= t <= overlap_end:
            eval_times.append(t)
            eval_values.append(v)

    if len(eval_times) < 2:
        return None, None, "Not enough overlapping samples with baseline"

    interp_ref: List[float] = []
    idx = 0
    for t in eval_times:
        while idx + 1 < len(ref_times) and ref_times[idx + 1] < t:
            idx += 1
        if idx + 1 >= len(ref_times):
            interp_ref.append(ref_values[-1])
            continue
        t0, t1 = ref_times[idx], ref_times[idx + 1]
        v0, v1 = ref_values[idx], ref_values[idx + 1]
        if t1 == t0:
            interp_ref.append(v0)
            continue
        alpha = (t - t0) / (t1 - t0)
        interp_ref.append(v0 + alpha * (v1 - v0))

    return (*compute_errors(eval_values, interp_ref), "")


def apply_validation_window(
    times: List[float],
    values: List[float],
    validation: Dict[str, Any],
) -> Tuple[List[float], List[float]]:
    if len(times) != len(values):
        raise ValueError("Time/value length mismatch")

    ignore_initial_samples = int(validation.get("ignore_initial_samples", 0) or 0)
    start_time = validation.get("start_time")
    start_time_value = parse_value(start_time) if start_time is not None else None

    filtered_times: List[float] = []
    filtered_values: List[float] = []
    for idx, (t, v) in enumerate(zip(times, values)):
        if idx < ignore_initial_samples:
            continue
        if start_time_value is not None and t < start_time_value:
            continue
        filtered_times.append(t)
        filtered_values.append(v)

    if len(filtered_times) < 2:
        raise ValueError("Validation window has fewer than 2 samples")

    return filtered_times, filtered_values


def _error_diagnostic_name(exc: Exception) -> Optional[str]:
    diagnostic = getattr(exc, "diagnostic", None)
    if diagnostic is None:
        return None
    if isinstance(diagnostic, str):
        value = diagnostic.strip()
        return value or None
    name = getattr(diagnostic, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    value = str(diagnostic).strip()
    return value or None


def _error_mode_name(exc: Exception) -> Optional[str]:
    mode = getattr(exc, "mode", None)
    if mode is None:
        return None
    if isinstance(mode, str):
        value = mode.strip()
        return value or None
    name = getattr(mode, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    value = str(mode).strip()
    return value or None


def _matches_expected_failure(
    expected_failure: Any,
    *,
    diagnostic: Optional[str],
    mode: Optional[str],
    message: str,
) -> bool:
    if not isinstance(expected_failure, dict):
        return False

    expected_diagnostic = expected_failure.get("diagnostic")
    if expected_diagnostic:
        if not diagnostic or str(expected_diagnostic).strip() != diagnostic:
            return False

    expected_mode = expected_failure.get("mode")
    if expected_mode:
        if not mode or str(expected_mode).strip() != mode:
            return False

    contains = expected_failure.get("message_contains")
    if contains:
        text = str(message).lower()
        if str(contains).lower() not in text:
            return False

    return True


def run_benchmarks(
    benchmarks_path: Path,
    output_dir: Path,
    selected: Optional[List[str]] = None,
    matrix: bool = False,
    generate_baselines: bool = False,
    simulation_overrides: Optional[Dict[str, Any]] = None,
    scenario_filter: Optional[List[str]] = None,
    adaptive_dt_max_factor: Optional[float] = None,
) -> List[ScenarioResult]:
    manifest = load_yaml(benchmarks_path)
    scenarios = manifest.get("scenarios", {})
    results: List[ScenarioResult] = []
    output_index: Dict[Tuple[str, str], Path] = {}

    for entry in manifest.get("benchmarks", []):
        circuit_path = (benchmarks_path.parent / entry["path"]).resolve()
        netlist = load_yaml(circuit_path)
        base_bench_meta = netlist.get("benchmark", {})
        if not isinstance(base_bench_meta, dict):
            base_bench_meta = {}
        benchmark_id = base_bench_meta.get("id", circuit_path.stem)
        if selected and benchmark_id not in selected:
            continue

        scenario_names = list(scenarios.keys()) if matrix else entry.get("scenarios", ["default"])
        if scenario_filter:
            allow = set(scenario_filter)
            scenario_names = [name for name in scenario_names if name in allow]
            if not scenario_names:
                continue
        if "default" in scenario_names and "default" not in scenarios:
            scenarios["default"] = {}

        for scenario_name in scenario_names:
            scenario_override = scenarios.get(scenario_name, {})
            scenario_netlist = deep_merge(netlist, scenario_override)
            scenario_bench_meta = scenario_netlist.get("benchmark", base_bench_meta)
            if not isinstance(scenario_bench_meta, dict):
                scenario_bench_meta = base_bench_meta
            preferred_mode = infer_preferred_mode(scenario_name, scenario_override)
            normalize_periodic_mode(scenario_netlist, preferred_mode)
            apply_runtime_defaults(scenario_netlist)
            if simulation_overrides:
                sim_cfg = scenario_netlist.get("simulation", {})
                if not isinstance(sim_cfg, dict):
                    sim_cfg = {}
                scenario_netlist["simulation"] = deep_merge(sim_cfg, simulation_overrides)
            simulation_cfg = scenario_netlist.get("simulation", {})
            if (
                adaptive_dt_max_factor is not None
                and isinstance(simulation_cfg, dict)
                and bool(simulation_cfg.get("adaptive_timestep", False))
                and simulation_cfg.get("dt") is not None
            ):
                try:
                    dt_value = parse_value(simulation_cfg.get("dt"))
                    dt_limit = max(dt_value * float(adaptive_dt_max_factor), dt_value)
                    current_dt_max = (
                        parse_value(simulation_cfg.get("dt_max"))
                        if simulation_cfg.get("dt_max") is not None
                        else None
                    )
                    if current_dt_max is None or current_dt_max > dt_limit:
                        simulation_cfg["dt_max"] = dt_limit
                except Exception:
                    pass
            use_initial_conditions = bool(
                simulation_cfg.get("uic", False) if isinstance(simulation_cfg, dict) else False
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                scenario_file = tmpdir_path / f"{benchmark_id}_{scenario_name}.yaml"
                output_path = output_dir / "outputs" / benchmark_id / scenario_name / "pulsim.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if yaml is None:
                    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
                runtime_netlist = dict(scenario_netlist)
                runtime_netlist.pop("benchmark", None)
                with open(scenario_file, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(runtime_netlist, handle, sort_keys=False)

                expectations = scenario_bench_meta.get("expectations", {})
                if not isinstance(expectations, dict):
                    expectations = {}
                expected_failure = expectations.get("expected_failure")
                averaged_pair_telemetry = extract_averaged_pair_telemetry(scenario_bench_meta)
                magnetic_kpi_telemetry = extract_magnetic_kpi_telemetry(scenario_bench_meta)

                try:
                    run_result = run_pulsim(
                        scenario_file,
                        output_path,
                        preferred_mode=preferred_mode,
                        use_initial_conditions=use_initial_conditions,
                    )
                except Exception as exc:
                    diagnostic = _error_diagnostic_name(exc)
                    mode = _error_mode_name(exc) or preferred_mode or "unknown"
                    if _matches_expected_failure(
                        expected_failure,
                        diagnostic=diagnostic,
                        mode=mode,
                        message=str(exc),
                    ):
                        telemetry: Dict[str, Optional[float]] = {
                            "python_backend": 1.0,
                            "expected_failure_matched": 1.0,
                            "expected_failure_case": 1.0,
                        }
                        telemetry.update(averaged_pair_telemetry)
                        telemetry.update(magnetic_kpi_telemetry)
                        results.append(
                            ScenarioResult(
                                benchmark_id=benchmark_id,
                                scenario=scenario_name,
                                mode=mode,
                                status="passed",
                                runtime_s=0.0,
                                steps=0,
                                max_error=None,
                                rms_error=None,
                                phase_error_deg=None,
                                message=f"Expected failure matched: {exc}",
                                telemetry=telemetry,
                            )
                        )
                        continue

                    results.append(
                        ScenarioResult(
                            benchmark_id=benchmark_id,
                            scenario=scenario_name,
                            mode=_error_mode_name(exc) or preferred_mode or "unknown",
                            status="failed",
                            runtime_s=0.0,
                            steps=0,
                            max_error=None,
                            rms_error=None,
                            phase_error_deg=None,
                            message=str(exc),
                            telemetry={},
                        )
                    )
                    continue

                status = "passed"
                max_error = None
                rms_error = None
                phase_error_deg = None
                message = ""
                mode = run_result.mode
                runtime_s = run_result.runtime_s
                steps = run_result.steps
                telemetry = dict(run_result.telemetry)
                telemetry["steps"] = float(steps)
                telemetry["runtime_s"] = float(runtime_s)
                telemetry["python_backend"] = 1.0
                telemetry.update(averaged_pair_telemetry)
                telemetry.update(magnetic_kpi_telemetry)
                output_index[(benchmark_id, scenario_name)] = output_path
                for key in ("newton_iterations", "timestep_rejections", "linear_fallbacks"):
                    if telemetry.get(key) is None:
                        telemetry[key] = 0.0
                components_block = scenario_netlist.get("components", [])
                if isinstance(components_block, list):
                    declared_components = sum(1 for item in components_block if isinstance(item, dict))
                    telemetry["component_declared_count"] = float(declared_components)
                    reported_value = telemetry.get("component_reported_count")
                    if reported_value is not None:
                        try:
                            reported_components = float(reported_value)
                            if math.isfinite(reported_components):
                                covered = min(reported_components, float(declared_components))
                                telemetry["component_coverage_rate"] = (
                                    (covered / float(declared_components))
                                    if declared_components > 0
                                    else 1.0
                                )
                                telemetry["component_coverage_gap"] = max(
                                    0.0, float(declared_components) - reported_components
                                )
                        except (TypeError, ValueError):
                            pass

                validation = scenario_bench_meta.get("validation", {})
                if not isinstance(validation, dict):
                    validation = {}
                validation_type = validation.get("type", "none")
                observable = validation.get("observable")
                max_threshold = coerce_optional_float(expectations.get("metrics", {}).get("max_error"))
                phase_threshold = coerce_optional_float(
                    expectations.get("metrics", {}).get("phase_error_deg")
                )
                cycle_energy_threshold = coerce_optional_float(
                    expectations.get("metrics", {}).get("cycle_energy_error")
                )

                if validation_type == "ac_analytical":
                    try:
                        frequencies, freq_series = load_frequency_csv_series(output_path)
                        max_error, rms_error, phase_error_deg, phase_rms_error = validate_ac_analytical(
                            frequencies,
                            freq_series,
                            validation.get("model", ""),
                            validation.get("params", {}),
                        )
                        telemetry["ac_sweep_case"] = 1.0
                        telemetry["ac_sweep_mag_error"] = float(max_error)
                        telemetry["ac_sweep_mag_rms_error"] = float(rms_error)
                        telemetry["ac_sweep_phase_error"] = float(phase_error_deg)
                        telemetry["ac_sweep_phase_rms_error"] = float(phase_rms_error)

                        if max_threshold is not None and max_error is not None and max_error > max_threshold:
                            status = "failed"
                            message = f"max_error {max_error:.6e} > threshold {max_threshold:.6e}"
                        if (
                            status == "passed"
                            and phase_threshold is not None
                            and phase_error_deg is not None
                            and phase_error_deg > phase_threshold
                        ):
                            status = "failed"
                            message = (
                                f"phase_error_deg {phase_error_deg:.6e} > "
                                f"threshold {phase_threshold:.6e}"
                            )
                    except Exception as exc:
                        status = "failed"
                        message = str(exc)
                elif validation_type == "magnetic_saturation":
                    try:
                        _, series = load_csv_series(output_path)
                        max_error, rms_error = validate_magnetic_saturation(
                            series,
                            validation.get("params", {}),
                        )
                        telemetry["magnetic_sat_error"] = float(max_error)
                        if max_threshold is not None and max_error is not None and max_error > max_threshold:
                            status = "failed"
                            message = f"max_error {max_error:.6e} > threshold {max_threshold:.6e}"
                    except Exception as exc:
                        status = "failed"
                        message = str(exc)
                elif validation_type == "magnetic_hysteresis":
                    try:
                        times, series = load_csv_series(output_path)
                        max_error, rms_error, cycle_energy_error = validate_magnetic_hysteresis(
                            times,
                            series,
                            validation.get("params", {}),
                        )
                        telemetry["magnetic_hysteresis_waveform_error"] = float(max_error)
                        telemetry["magnetic_hysteresis_cycle_energy_error"] = float(cycle_energy_error)
                        if max_threshold is not None and max_error is not None and max_error > max_threshold:
                            status = "failed"
                            message = f"max_error {max_error:.6e} > threshold {max_threshold:.6e}"
                        if (
                            status == "passed"
                            and cycle_energy_threshold is not None
                            and cycle_energy_error > cycle_energy_threshold
                        ):
                            status = "failed"
                            message = (
                                "cycle_energy_error "
                                f"{cycle_energy_error:.6e} > threshold {cycle_energy_threshold:.6e}"
                            )
                    except Exception as exc:
                        status = "failed"
                        message = str(exc)
                elif validation_type != "none" and not observable:
                    status = "failed"
                    message = "Missing validation observable"
                elif validation_type != "none":
                    times, series = load_csv_series(output_path)
                    if observable not in series:
                        status = "failed"
                        message = f"Missing observable column: {observable}"
                    else:
                        values = series[observable]
                        try:
                            times_eval, values_eval = apply_validation_window(times, values, validation)
                        except Exception as exc:
                            status = "failed"
                            message = str(exc)
                        else:
                            if validation_type == "analytical":
                                try:
                                    max_error, rms_error = validate_analytical(
                                        times_eval,
                                        values_eval,
                                        validation.get("model", ""),
                                        validation.get("params", {}),
                                    )
                                    if max_threshold is not None and max_error is not None:
                                        if max_error <= max_threshold:
                                            status = "passed"
                                        else:
                                            status = "failed"
                                            message = (
                                                f"max_error {max_error:.6e} > threshold {max_threshold:.6e}"
                                            )
                                except Exception as exc:
                                    status = "failed"
                                    message = str(exc)
                            elif validation_type == "reference":
                                baseline_rel = validation.get("baseline")
                                if not baseline_rel:
                                    status = "failed"
                                    message = "Missing baseline path"
                                else:
                                    baseline_path = (benchmarks_path.parent / baseline_rel).resolve()
                                    if not baseline_path.exists() and generate_baselines:
                                        baseline_path.parent.mkdir(parents=True, exist_ok=True)
                                        with open(output_path, "r", encoding="utf-8") as src, open(
                                            baseline_path, "w", encoding="utf-8"
                                        ) as dst:
                                            dst.write(src.read())
                                        status = "baseline"
                                        message = "Baseline generated"
                                    else:
                                        max_error, rms_error, message = validate_reference(
                                            times_eval, values_eval, baseline_path, observable
                                        )
                                        if max_error is None:
                                            status = "failed"
                                        elif max_threshold is not None:
                                            if max_error <= max_threshold:
                                                status = "passed"
                                            else:
                                                status = "failed"
                                                message = (
                                                    f"max_error {max_error:.6e} > threshold {max_threshold:.6e}"
                                                )
                            elif validation_type == "paired_reference":
                                pair_benchmark_id = validation.get("pair_benchmark_id")
                                pair_scenario = validation.get("pair_scenario", scenario_name)
                                pair_observable = validation.get("pair_observable", observable)
                                if (
                                    not isinstance(pair_benchmark_id, str)
                                    or not pair_benchmark_id.strip()
                                ):
                                    status = "failed"
                                    message = "Missing validation pair_benchmark_id"
                                elif not isinstance(pair_scenario, str) or not pair_scenario.strip():
                                    status = "failed"
                                    message = "Invalid validation pair_scenario"
                                elif not isinstance(pair_observable, str) or not pair_observable.strip():
                                    status = "failed"
                                    message = "Invalid validation pair_observable"
                                else:
                                    pair_key = (pair_benchmark_id.strip(), pair_scenario.strip())
                                    pair_output_path = output_index.get(pair_key)
                                    if pair_output_path is None or not pair_output_path.exists():
                                        status = "failed"
                                        message = (
                                            "Missing paired reference output for "
                                            f"{pair_key[0]}/{pair_key[1]}"
                                        )
                                    else:
                                        max_error, rms_error, message = validate_reference(
                                            times_eval,
                                            values_eval,
                                            pair_output_path,
                                            pair_observable.strip(),
                                        )
                                        telemetry["paired_reference_case"] = 1.0
                                        telemetry["paired_reference_group_crc32"] = float(
                                            zlib.crc32(
                                                f"{pair_key[0]}::{pair_key[1]}".encode("utf-8")
                                            )
                                            & 0xFFFFFFFF
                                        )
                                        if max_error is None:
                                            status = "failed"
                                        elif max_threshold is not None:
                                            if max_error <= max_threshold:
                                                status = "passed"
                                            else:
                                                status = "failed"
                                                message = (
                                                    f"max_error {max_error:.6e} > threshold {max_threshold:.6e}"
                                                )
                            else:
                                status = "failed"
                                message = f"Unsupported validation type: {validation_type}"

                magnetic_core_loss_observable = scenario_bench_meta.get(
                    "magnetic_core_loss_observable",
                    "Lsat.core_loss",
                )
                if (
                    isinstance(magnetic_core_loss_observable, str)
                    and magnetic_core_loss_observable.strip()
                    and telemetry.get("magnetic_fixture_frequency_trend", 0.0) > 0.0
                ):
                    core_loss_mean = compute_magnetic_core_loss_tail_mean(
                        output_path=output_path,
                        observable=magnetic_core_loss_observable.strip(),
                    )
                    if core_loss_mean is not None:
                        telemetry["magnetic_avg_core_loss"] = float(core_loss_mean)

                if max_error is not None:
                    telemetry["validation_max_error"] = float(max_error)
                if rms_error is not None:
                    telemetry["validation_rms_error"] = float(rms_error)
                if phase_error_deg is not None:
                    telemetry["validation_phase_error_deg"] = float(phase_error_deg)

                results.append(
                    ScenarioResult(
                        benchmark_id=benchmark_id,
                        scenario=scenario_name,
                        mode=mode,
                        status=status,
                        runtime_s=runtime_s,
                        steps=steps,
                        max_error=max_error,
                        rms_error=rms_error,
                        phase_error_deg=phase_error_deg,
                        message=message,
                        telemetry=telemetry,
                    )
                )

    return results


def write_results(output_dir: Path, results: List[ScenarioResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "results.csv"
    results_json = output_dir / "results.json"
    summary_json = output_dir / "summary.json"

    with open(results_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "benchmark_id",
            "scenario",
            "mode",
            "status",
            "runtime_s",
            "steps",
            "max_error",
            "rms_error",
            "phase_error_deg",
            "message",
        ])
        for item in results:
            writer.writerow([
                item.benchmark_id,
                item.scenario,
                item.mode,
                item.status,
                f"{item.runtime_s:.6f}",
                item.steps,
                "" if item.max_error is None else f"{item.max_error:.6e}",
                "" if item.rms_error is None else f"{item.rms_error:.6e}",
                "" if item.phase_error_deg is None else f"{item.phase_error_deg:.6e}",
                item.message,
            ])

    payload = {
        "results": [item.__dict__ for item in results],
    }
    with open(results_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    summary = {
        "passed": sum(1 for item in results if item.status == "passed"),
        "failed": sum(1 for item in results if item.status == "failed"),
        "skipped": sum(1 for item in results if item.status == "skipped"),
        "baseline": sum(1 for item in results if item.status == "baseline"),
        "total": len(results),
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pulsim benchmark suite")
    parser.add_argument("--benchmarks", type=Path, default=Path(__file__).with_name("benchmarks.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/out"))
    parser.add_argument("--only", nargs="*", help="Benchmark ids to run")
    parser.add_argument("--matrix", action="store_true", help="Run full validation matrix")
    parser.add_argument("--generate-baselines", action="store_true", help="Generate missing reference baselines")
    parser.add_argument("--force-adaptive", action="store_true", help="Force simulation.adaptive_timestep=true")
    parser.add_argument("--scenario-filter", nargs="*", help="Run only selected scenarios")
    args = parser.parse_args()

    if yaml is None:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml")

    backend_reason = None
    if python_backend_error is not None:
        try:
            backend_reason = python_backend_error()
        except Exception as exc:
            backend_reason = f"{exc.__class__.__name__}: {exc}"

    if backend_reason is not None:
        raise SystemExit(
            "Pulsim Python runtime backend unavailable. "
            "Build Python bindings and expose build/python on PYTHONPATH or install pulsim package. "
            f"Reason: {backend_reason}"
        )

    results = run_benchmarks(
        args.benchmarks,
        args.output_dir,
        selected=args.only,
        matrix=args.matrix,
        generate_baselines=args.generate_baselines,
        simulation_overrides={"adaptive_timestep": True} if args.force_adaptive else None,
        scenario_filter=args.scenario_filter,
    )
    write_results(args.output_dir, results)

    summary = {
        "passed": sum(1 for item in results if item.status == "passed"),
        "failed": sum(1 for item in results if item.status == "failed"),
        "skipped": sum(1 for item in results if item.status == "skipped"),
        "baseline": sum(1 for item in results if item.status == "baseline"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
