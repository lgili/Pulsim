#!/usr/bin/env python3
"""Pulsim vs external SPICE parity runner (ngspice and LTspice)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmark_runner import (
    apply_runtime_defaults,
    deep_merge,
    infer_preferred_mode,
    load_csv_series,
    load_yaml,
    normalize_periodic_mode,
    parse_value,
    run_pulsim,
    yaml,
)

PARITY_SCHEMA_VERSION = "pulsim-parity-v1"
SUPPORTED_BACKENDS = {"ngspice", "ltspice"}
DEFAULT_LTSPICE_ARGS = ["-ascii", "-b"]


@dataclass
class ObservableSpec:
    column: str
    spice_vector: str


@dataclass
class ObservableMetrics:
    column: str
    spice_vector: str
    samples: int
    max_error: float
    rms_error: float
    phase_error_deg: Optional[float] = None
    steady_state_max_error: Optional[float] = None
    steady_state_rms_error: Optional[float] = None


@dataclass
class BackendConfig:
    backend: str
    executable: Optional[Path]
    args: List[str]
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    benchmark_id: str
    scenario: str
    status: str
    message: str
    pulsim_runtime_s: float
    ngspice_runtime_s: float
    pulsim_steps: int
    ngspice_steps: int
    speedup: Optional[float]
    max_error: Optional[float]
    rms_error: Optional[float]
    observables: List[ObservableMetrics]
    backend: str = "ngspice"
    phase_error_deg: Optional[float] = None
    steady_state_max_error: Optional[float] = None
    steady_state_rms_error: Optional[float] = None
    failure_reason: Optional[str] = None
    reference_runtime_s: Optional[float] = None
    reference_steps: Optional[int] = None


def coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def _result_for_error(
    benchmark_id: str,
    scenario: str,
    backend: str,
    message: str,
    failure_reason: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_id=benchmark_id,
        scenario=scenario,
        status="failed",
        message=message,
        pulsim_runtime_s=0.0,
        ngspice_runtime_s=0.0,
        pulsim_steps=0,
        ngspice_steps=0,
        speedup=None,
        max_error=None,
        rms_error=None,
        observables=[],
        backend=backend,
        failure_reason=failure_reason,
        reference_runtime_s=0.0,
        reference_steps=0,
    )


def _resolve_executable_path(executable: Optional[Path], fallback_name: str) -> Tuple[Optional[Path], Optional[str]]:
    if executable is not None:
        candidate = executable.expanduser().resolve()
        if not candidate.exists():
            return None, f"Executable not found: {candidate}"
        if not os.access(candidate, os.X_OK):
            return None, f"Executable is not runnable: {candidate}"
        return candidate, None

    resolved = shutil.which(fallback_name)
    if resolved is None:
        return None, f"{fallback_name} executable not found on PATH"
    return Path(resolved), None


def resolve_backend_config(
    backend: str,
    manifest: Optional[Dict[str, Any]],
    ngspice_executable: Optional[Path],
    ltspice_executable: Optional[Path],
    backend_args: Optional[List[str]],
) -> BackendConfig:
    backend_name = backend.lower().strip()
    if backend_name not in SUPPORTED_BACKENDS:
        return BackendConfig(
            backend=backend_name,
            executable=None,
            args=[],
            error=f"Unsupported backend: {backend_name}",
        )

    simulators = manifest.get("simulators", {}) if isinstance(manifest, dict) else {}
    sim_cfg = simulators.get(backend_name, {}) if isinstance(simulators, dict) else {}

    cli_executable: Optional[Path]
    env_var: Optional[str]
    if backend_name == "ltspice":
        cli_executable = ltspice_executable
        env_var = os.environ.get("PULSIM_LTSPICE_EXE")
        if cli_executable is None and env_var:
            cli_executable = Path(env_var)
        if cli_executable is None and isinstance(sim_cfg, dict) and sim_cfg.get("executable"):
            cli_executable = Path(str(sim_cfg["executable"]))
        if cli_executable is None:
            return BackendConfig(
                backend=backend_name,
                executable=None,
                args=[],
                error=(
                    "LTspice backend requires explicit executable path. "
                    "Use --ltspice-exe, PULSIM_LTSPICE_EXE, or simulators.ltspice.executable in manifest."
                ),
            )
        executable, error = _resolve_executable_path(cli_executable, "ltspice")
    else:
        cli_executable = ngspice_executable
        env_var = os.environ.get("PULSIM_NGSPICE_EXE")
        if cli_executable is None and env_var:
            cli_executable = Path(env_var)
        if cli_executable is None and isinstance(sim_cfg, dict) and sim_cfg.get("executable"):
            cli_executable = Path(str(sim_cfg["executable"]))
        executable, error = _resolve_executable_path(cli_executable, "ngspice")

    configured_args = sim_cfg.get("args") if isinstance(sim_cfg, dict) else None
    args: List[str] = []
    if isinstance(configured_args, list):
        args.extend(str(item) for item in configured_args)
    if backend_args:
        args.extend(backend_args)
    if backend_name == "ltspice" and not args:
        args = list(DEFAULT_LTSPICE_ARGS)

    return BackendConfig(backend=backend_name, executable=executable, args=args, error=error)


def check_ngspice(executable: Optional[Path] = None) -> bool:
    command = str(executable) if executable is not None else "ngspice"
    try:
        result = subprocess.run([command, "--version"], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return result.returncode == 0


def strip_control_blocks_and_end(text: str) -> str:
    output: List[str] = []
    in_control = False
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith(".control"):
            in_control = True
            continue
        if in_control:
            if stripped.startswith(".endc"):
                in_control = False
            continue
        if stripped == ".end":
            continue
        output.append(line)
    return "\n".join(output).rstrip()


def write_batch_ngspice_netlist(
    source_netlist: Path,
    vectors: Sequence[str],
    output_files: Sequence[Path],
    destination: Path,
) -> None:
    if len(vectors) != len(output_files):
        raise ValueError("Each ngspice vector must have one output file")

    source_text = source_netlist.read_text(encoding="utf-8")
    base_text = strip_control_blocks_and_end(source_text)
    lines: List[str] = [base_text, "", ".control", "set filetype=ascii", "run"]
    for vector, out_file in zip(vectors, output_files):
        lines.append(f"wrdata {out_file.as_posix()} {vector}")
    lines.extend(["quit", ".endc", ".end", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")


def parse_ngspice_wrdata(path: Path) -> Tuple[List[float], List[float]]:
    times: List[float] = []
    values: List[float] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                numeric = [float(token) for token in parts]
            except ValueError:
                continue
            if len(numeric) >= 3 and len(numeric) % 2 == 1:
                numeric = numeric[1:]
            if len(numeric) < 2:
                continue
            times.append(numeric[0])
            values.append(numeric[1])

    if not times:
        raise RuntimeError(f"ngspice output file has no numeric data: {path}")

    return times, values


def run_ngspice_vectors(
    netlist_path: Path,
    vectors: Sequence[str],
    executable: Path,
) -> Tuple[float, int, Dict[str, Tuple[List[float], List[float]]]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        batch_netlist = tmp / "benchmark_job.cir"
        output_files = [tmp / f"vector_{idx}.dat" for idx in range(len(vectors))]
        write_batch_ngspice_netlist(netlist_path, vectors, output_files, batch_netlist)

        start = time.perf_counter()
        result = subprocess.run(
            [str(executable), "-b", str(batch_netlist)],
            capture_output=True,
            text=True,
            cwd=netlist_path.parent,
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "ngspice failed"
            raise RuntimeError(detail)

        series: Dict[str, Tuple[List[float], List[float]]] = {}
        max_steps = 0
        for vector, out_file in zip(vectors, output_files):
            if not out_file.exists():
                raise RuntimeError(f"ngspice did not generate vector output: {out_file}")
            times, values = parse_ngspice_wrdata(out_file)
            series[vector] = (times, values)
            max_steps = max(max_steps, max(0, len(times) - 1))

    return elapsed, max_steps, series


def _parse_ltspice_numeric(token: str) -> float:
    clean = token.strip()
    if clean.startswith("(") and clean.endswith(")"):
        clean = clean[1:-1]
    if "," in clean:
        clean = clean.split(",", 1)[0]
    return float(clean)


def parse_ltspice_ascii_raw(path: Path) -> Tuple[List[float], Dict[str, List[float]]]:
    raw_bytes = path.read_bytes()
    if b"\x00" in raw_bytes[:128]:
        text = raw_bytes.decode("utf-16-le", errors="replace")
    else:
        text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    variable_names: List[str] = []
    series: Dict[str, List[float]] = {}
    values_start = -1

    for idx, line in enumerate(lines):
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered == "variables:":
            cursor = idx + 1
            while cursor < len(lines):
                item = lines[cursor].strip()
                if not item:
                    cursor += 1
                    continue
                if item.lower() == "values:":
                    values_start = cursor + 1
                    break
                parts = item.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    name = parts[1]
                    if name not in series:
                        variable_names.append(name)
                        series[name] = []
                cursor += 1
            break

    if not variable_names or values_start < 0:
        raise RuntimeError(f"LTspice raw parser could not find Variables/Values sections: {path}")

    cursor = values_start
    while cursor < len(lines):
        head = lines[cursor].strip()
        if not head:
            cursor += 1
            continue
        parts = head.split()
        if not parts or not parts[0].isdigit():
            cursor += 1
            continue

        if len(parts) < 2:
            raise RuntimeError(f"Malformed LTspice raw values row in {path} at line {cursor + 1}")

        row_values = [_parse_ltspice_numeric(parts[1])]
        cursor += 1

        while len(row_values) < len(variable_names) and cursor < len(lines):
            value_line = lines[cursor].strip()
            if not value_line:
                cursor += 1
                continue
            value_parts = value_line.split()
            token = value_parts[0]
            if value_parts and value_parts[0].isdigit() and len(value_parts) > 1:
                token = value_parts[1]
            row_values.append(_parse_ltspice_numeric(token))
            cursor += 1

        if len(row_values) != len(variable_names):
            raise RuntimeError(
                f"Malformed LTspice raw row: expected {len(variable_names)} values, got {len(row_values)}"
            )

        for name, value in zip(variable_names, row_values):
            series[name].append(value)

    if not series:
        raise RuntimeError(f"LTspice raw parser found no samples in {path}")

    time_key = next((name for name in variable_names if name.lower() == "time"), variable_names[0])
    times = list(series.get(time_key, []))
    if not times:
        raise RuntimeError(f"LTspice raw parser could not resolve time axis in {path}")

    return times, series


def _build_ltspice_command(executable: Path, args: Sequence[str], netlist: Path) -> List[str]:
    command = [str(executable)]
    args_list = [str(arg) for arg in args] if args else list(DEFAULT_LTSPICE_ARGS)
    if not args_list:
        args_list = list(DEFAULT_LTSPICE_ARGS)

    # Keep "-ascii" in front of "-b" on macOS LTspice CLI to avoid input parsing errors.
    if "-ascii" in args_list and "-b" in args_list:
        ascii_idx = args_list.index("-ascii")
        batch_idx = args_list.index("-b")
        if ascii_idx > batch_idx:
            args_list.pop(ascii_idx)
            args_list.insert(batch_idx, "-ascii")

    has_netlist_placeholder = False
    for arg in args_list:
        if arg == "{netlist}":
            command.append(str(netlist))
            has_netlist_placeholder = True
        else:
            command.append(arg)
    if not has_netlist_placeholder:
        command.append(str(netlist))
    return command


def _resolve_vector_key(series: Dict[str, List[float]], requested: str) -> Optional[str]:
    if requested in series:
        return requested
    lowered = {key.lower(): key for key in series}
    return lowered.get(requested.lower())


def run_ltspice_vectors(
    netlist_path: Path,
    vectors: Sequence[str],
    executable: Path,
    args: Sequence[str],
) -> Tuple[float, int, Dict[str, Tuple[List[float], List[float]]]]:
    raw_path = netlist_path.with_suffix(".raw")
    if raw_path.exists():
        raw_path.unlink()

    command = _build_ltspice_command(executable, args, netlist_path)
    start = time.perf_counter()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=netlist_path.parent,
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "LTspice failed"
        raise RuntimeError(detail)
    if not raw_path.exists():
        raise RuntimeError(
            f"LTspice run completed but raw output was not found: {raw_path}. "
            "Ensure backend args include '-ascii' and netlist has a transient analysis."
        )

    times, series = parse_ltspice_ascii_raw(raw_path)
    max_steps = max(0, len(times) - 1)
    selected: Dict[str, Tuple[List[float], List[float]]] = {}
    for vector in vectors:
        key = _resolve_vector_key(series, vector)
        if key is None:
            available = ", ".join(sorted(series.keys())[:10])
            raise RuntimeError(f"LTspice output missing vector '{vector}'. Available: {available}")
        selected[vector] = (times, list(series[key]))

    return elapsed, max_steps, selected


def interpolate_linear(
    target_times: Sequence[float],
    ref_times: Sequence[float],
    ref_values: Sequence[float],
) -> List[float]:
    if len(ref_times) < 2:
        raise ValueError("Reference time series requires at least 2 points")
    output: List[float] = []
    idx = 0

    for t in target_times:
        while idx + 1 < len(ref_times) and ref_times[idx + 1] < t:
            idx += 1
        if idx + 1 >= len(ref_times):
            output.append(ref_values[-1])
            continue
        t0, t1 = ref_times[idx], ref_times[idx + 1]
        v0, v1 = ref_values[idx], ref_values[idx + 1]
        if t1 == t0:
            output.append(v0)
            continue
        alpha = (t - t0) / (t1 - t0)
        output.append(v0 + alpha * (v1 - v0))

    return output


def compare_series(
    pulsim_times: Sequence[float],
    pulsim_values: Sequence[float],
    spice_times: Sequence[float],
    spice_values: Sequence[float],
) -> Tuple[List[float], int, List[float], List[float], List[float]]:
    if len(pulsim_times) != len(pulsim_values):
        raise ValueError("Pulsim series length mismatch")
    if len(spice_times) != len(spice_values):
        raise ValueError("SPICE series length mismatch")
    if not pulsim_times or not spice_times:
        raise ValueError("Empty series")

    overlap_start = max(pulsim_times[0], spice_times[0])
    overlap_end = min(pulsim_times[-1], spice_times[-1])
    if overlap_start >= overlap_end:
        raise ValueError("No overlapping time range between Pulsim and SPICE")

    filtered_times: List[float] = []
    filtered_values: List[float] = []
    for t, v in zip(pulsim_times, pulsim_values):
        if overlap_start <= t <= overlap_end:
            filtered_times.append(t)
            filtered_values.append(v)

    if len(filtered_times) < 2:
        raise ValueError("Not enough overlapping samples for comparison")

    ref_interp = interpolate_linear(filtered_times, spice_times, spice_values)
    errors = [abs(a - b) for a, b in zip(filtered_values, ref_interp)]
    return errors, len(filtered_times), filtered_times, filtered_values, ref_interp


def _select_window(
    times: Sequence[float],
    lhs: Sequence[float],
    rhs: Sequence[float],
    period_hint: Optional[float],
) -> Tuple[List[float], List[float], List[float]]:
    if len(times) < 2:
        return [], [], []
    if period_hint is not None and period_hint > 0.0:
        start = times[-1] - period_hint
    else:
        span = times[-1] - times[0]
        start = times[0] + 0.8 * span

    out_t: List[float] = []
    out_lhs: List[float] = []
    out_rhs: List[float] = []
    for t, a, b in zip(times, lhs, rhs):
        if t >= start:
            out_t.append(t)
            out_lhs.append(a)
            out_rhs.append(b)

    if len(out_t) < 2:
        fallback = max(2, len(times) // 5)
        out_t = list(times[-fallback:])
        out_lhs = list(lhs[-fallback:])
        out_rhs = list(rhs[-fallback:])

    return out_t, out_lhs, out_rhs


def compute_steady_state_errors(
    times: Sequence[float],
    pulsim_values: Sequence[float],
    ref_values: Sequence[float],
    period_hint: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    window_t, window_p, window_r = _select_window(times, pulsim_values, ref_values, period_hint)
    if len(window_t) < 2:
        return None, None

    errors = [abs(a - b) for a, b in zip(window_p, window_r)]
    max_error = max(errors)
    rms_error = math.sqrt(sum(err * err for err in errors) / len(errors))
    return max_error, rms_error


def compute_phase_error_deg(
    times: Sequence[float],
    pulsim_values: Sequence[float],
    ref_values: Sequence[float],
    period_hint: Optional[float],
) -> Optional[float]:
    if period_hint is None or period_hint <= 0.0:
        return None

    window_t, window_p, window_r = _select_window(times, pulsim_values, ref_values, period_hint)
    if len(window_t) < 8:
        return None

    max_lag = min(len(window_t) // 4, 512)
    if max_lag < 1:
        return None

    best_lag = 0
    best_score = -1.0e30
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            p_seg = window_p[-lag:]
            r_seg = window_r[: len(p_seg)]
        elif lag > 0:
            p_seg = window_p[:-lag]
            r_seg = window_r[lag:]
        else:
            p_seg = window_p
            r_seg = window_r

        if len(p_seg) < 4 or len(r_seg) < 4:
            continue

        p_mean = sum(p_seg) / len(p_seg)
        r_mean = sum(r_seg) / len(r_seg)
        p_centered = [value - p_mean for value in p_seg]
        r_centered = [value - r_mean for value in r_seg]

        numerator = sum(a * b for a, b in zip(p_centered, r_centered))
        denom_l = math.sqrt(sum(a * a for a in p_centered))
        denom_r = math.sqrt(sum(b * b for b in r_centered))
        denom = denom_l * denom_r
        if denom <= 1e-20:
            continue

        score = numerator / denom
        if score > best_score:
            best_score = score
            best_lag = lag

    if best_score <= -1.0e20:
        return None

    dt = (window_t[-1] - window_t[0]) / max(1, len(window_t) - 1)
    if dt <= 0.0:
        return None

    phase = (best_lag * dt / period_hint) * 360.0
    while phase > 180.0:
        phase -= 360.0
    while phase < -180.0:
        phase += 360.0
    return abs(phase)


def normalize_observable(entry: Any, backend: str = "ngspice") -> Optional[ObservableSpec]:
    if isinstance(entry, str):
        col = entry.strip()
        if not col:
            return None
        default_vector = col if backend == "ltspice" else col.lower()
        return ObservableSpec(column=col, spice_vector=default_vector)

    if isinstance(entry, dict):
        col = str(entry.get("column", "")).strip()
        if not col:
            return None

        vector_keys = [
            f"{backend}_vector",
            "spice_vector",
            "vector",
            "spice",
            "ngspice_vector",
            "ltspice_vector",
        ]
        selected: Optional[str] = None
        for key in vector_keys:
            raw = entry.get(key)
            if raw is not None and str(raw).strip():
                selected = str(raw).strip()
                break

        if selected is None:
            selected = col if backend == "ltspice" else col.lower()

        return ObservableSpec(column=col, spice_vector=selected)

    return None


def resolve_observables(
    benchmark_meta: Dict[str, Any],
    validation_meta: Dict[str, Any],
    entry_observables: Optional[List[Any]],
    cli_observables: Optional[List[str]],
    backend: str = "ngspice",
) -> List[ObservableSpec]:
    raw_entries: List[Any] = []
    backend_key = f"{backend}_observables"

    if cli_observables:
        raw_entries = cli_observables
    elif entry_observables:
        raw_entries = entry_observables
    elif isinstance(validation_meta.get(backend_key), list):
        raw_entries = validation_meta[backend_key]
    elif isinstance(validation_meta.get("observables"), list):
        raw_entries = validation_meta["observables"]
    elif isinstance(benchmark_meta.get(backend_key), list):
        raw_entries = benchmark_meta[backend_key]
    elif isinstance(benchmark_meta.get("observables"), list):
        raw_entries = benchmark_meta["observables"]
    elif validation_meta.get("observable"):
        raw_entries = [validation_meta["observable"]]

    specs: List[ObservableSpec] = []
    seen: set[str] = set()
    for raw in raw_entries:
        spec = normalize_observable(raw, backend=backend)
        if spec is None or spec.column in seen:
            continue
        seen.add(spec.column)
        specs.append(spec)
    return specs


def _resolve_period_hint(validation_meta: Dict[str, Any], scenario_netlist: Dict[str, Any]) -> Optional[float]:
    try:
        period_raw = validation_meta.get("period")
        if period_raw is not None:
            return parse_value(period_raw)

        simulation = scenario_netlist.get("simulation", {})
        if not isinstance(simulation, dict):
            return None

        shooting = simulation.get("shooting", {})
        if isinstance(shooting, dict) and shooting.get("period") is not None:
            return parse_value(shooting["period"])

        hb = simulation.get("harmonic_balance", {})
        if isinstance(hb, dict) and hb.get("period") is not None:
            return parse_value(hb["period"])

        hb_alias = simulation.get("hb", {})
        if isinstance(hb_alias, dict) and hb_alias.get("period") is not None:
            return parse_value(hb_alias["period"])
    except Exception:
        return None

    return None


def compare_pulsim_vs_spice(
    pulsim_csv_path: Path,
    spice_netlist_path: Path,
    observable_specs: List[ObservableSpec],
    backend_config: BackendConfig,
    period_hint: Optional[float],
) -> Tuple[
    float,
    int,
    List[ObservableMetrics],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    if not observable_specs:
        return 0.0, 0, [], None, None, None, None, None
    if backend_config.executable is None:
        raise RuntimeError(f"Missing executable for backend '{backend_config.backend}'")

    pulsim_times, pulsim_series = load_csv_series(pulsim_csv_path)
    vectors = [spec.spice_vector for spec in observable_specs]

    if backend_config.backend == "ltspice":
        spice_runtime, spice_steps, spice_series = run_ltspice_vectors(
            spice_netlist_path,
            vectors,
            executable=backend_config.executable,
            args=backend_config.args,
        )
    else:
        spice_runtime, spice_steps, spice_series = run_ngspice_vectors(
            spice_netlist_path,
            vectors,
            executable=backend_config.executable,
        )

    metrics: List[ObservableMetrics] = []
    all_errors: List[float] = []
    all_steady_errors: List[float] = []
    phase_errors: List[float] = []

    for spec in observable_specs:
        if spec.column not in pulsim_series:
            raise RuntimeError(f"Pulsim output missing observable column: {spec.column}")
        if spec.spice_vector not in spice_series:
            raise RuntimeError(f"{backend_config.backend} output missing vector: {spec.spice_vector}")

        spice_times, spice_values = spice_series[spec.spice_vector]
        errors, samples, times_eval, values_eval, ref_interp = compare_series(
            pulsim_times,
            pulsim_series[spec.column],
            spice_times,
            spice_values,
        )

        all_errors.extend(errors)
        rms = math.sqrt(sum(e * e for e in errors) / len(errors))

        phase_error = compute_phase_error_deg(times_eval, values_eval, ref_interp, period_hint)
        if phase_error is not None:
            phase_errors.append(phase_error)

        steady_max, steady_rms = compute_steady_state_errors(times_eval, values_eval, ref_interp, period_hint)
        if steady_max is not None:
            window_t, window_p, window_r = _select_window(times_eval, values_eval, ref_interp, period_hint)
            all_steady_errors.extend(abs(a - b) for a, b in zip(window_p, window_r))

        metrics.append(
            ObservableMetrics(
                column=spec.column,
                spice_vector=spec.spice_vector,
                samples=samples,
                max_error=max(errors),
                rms_error=rms,
                phase_error_deg=phase_error,
                steady_state_max_error=steady_max,
                steady_state_rms_error=steady_rms,
            )
        )

    global_max = max(all_errors) if all_errors else None
    global_rms = math.sqrt(sum(e * e for e in all_errors) / len(all_errors)) if all_errors else None
    global_phase = max(phase_errors) if phase_errors else None
    steady_max = max(all_steady_errors) if all_steady_errors else None
    steady_rms = (
        math.sqrt(sum(err * err for err in all_steady_errors) / len(all_steady_errors))
        if all_steady_errors
        else None
    )
    return spice_runtime, spice_steps, metrics, global_max, global_rms, global_phase, steady_max, steady_rms


def compare_pulsim_vs_ngspice(
    pulsim_csv_path: Path,
    spice_netlist_path: Path,
    observable_specs: List[ObservableSpec],
) -> Tuple[float, int, List[ObservableMetrics], Optional[float], Optional[float]]:
    executable = shutil.which("ngspice")
    if executable is None:
        raise RuntimeError("ngspice executable not found on PATH")
    backend_config = BackendConfig(backend="ngspice", executable=Path(executable), args=[])
    runtime, steps, observables, max_error, rms_error, _, _, _ = compare_pulsim_vs_spice(
        pulsim_csv_path=pulsim_csv_path,
        spice_netlist_path=spice_netlist_path,
        observable_specs=observable_specs,
        backend_config=backend_config,
        period_hint=None,
    )
    return runtime, steps, observables, max_error, rms_error


def run_case(
    scenario_netlist: Dict[str, Any],
    benchmark_id: str,
    scenario_name: str,
    spice_netlist_path: Path,
    observable_specs: List[ObservableSpec],
    max_error_threshold: Optional[float],
    output_dir: Path,
    preferred_mode: Optional[str],
    backend_config: Optional[BackendConfig] = None,
    period_hint: Optional[float] = None,
    rms_error_threshold: Optional[float] = None,
    phase_error_threshold: Optional[float] = None,
    steady_state_max_threshold: Optional[float] = None,
    steady_state_rms_threshold: Optional[float] = None,
) -> BenchmarkResult:
    backend_name = backend_config.backend if backend_config is not None else "ngspice"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        scenario_file = tmpdir_path / f"{benchmark_id}_{scenario_name}.yaml"
        pulsim_output = output_dir / "outputs" / benchmark_id / scenario_name / "pulsim.csv"
        pulsim_output.parent.mkdir(parents=True, exist_ok=True)

        if yaml is None:
            raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
        with open(scenario_file, "w", encoding="utf-8") as handle:
            yaml.safe_dump(scenario_netlist, handle, sort_keys=False)

        simulation_cfg = scenario_netlist.get("simulation", {})
        use_initial_conditions = bool(
            simulation_cfg.get("uic", False) if isinstance(simulation_cfg, dict) else False
        )
        run_result = run_pulsim(
            scenario_file,
            pulsim_output,
            preferred_mode=preferred_mode,
            use_initial_conditions=use_initial_conditions,
        )
        pulsim_runtime = run_result.runtime_s
        pulsim_steps = run_result.steps

    phase_error: Optional[float] = None
    steady_max_error: Optional[float] = None
    steady_rms_error: Optional[float] = None

    if backend_config is None or backend_config.backend == "ngspice":
        ng_runtime, ng_steps, observables, max_error, rms_error = compare_pulsim_vs_ngspice(
            pulsim_output,
            spice_netlist_path,
            observable_specs,
        )
    else:
        ng_runtime, ng_steps, observables, max_error, rms_error, phase_error, steady_max_error, steady_rms_error = (
            compare_pulsim_vs_spice(
                pulsim_csv_path=pulsim_output,
                spice_netlist_path=spice_netlist_path,
                observable_specs=observable_specs,
                backend_config=backend_config,
                period_hint=period_hint,
            )
        )

    if phase_error is None and observables:
        phase_candidates = [obs.phase_error_deg for obs in observables if obs.phase_error_deg is not None]
        if phase_candidates:
            phase_error = max(phase_candidates)
    if steady_max_error is None and observables:
        steady_max_candidates = [
            obs.steady_state_max_error for obs in observables if obs.steady_state_max_error is not None
        ]
        if steady_max_candidates:
            steady_max_error = max(steady_max_candidates)
    if steady_rms_error is None and observables:
        all_terms = [obs.steady_state_rms_error for obs in observables if obs.steady_state_rms_error is not None]
        if all_terms:
            steady_rms_error = max(all_terms)

    speedup = (ng_runtime / pulsim_runtime) if pulsim_runtime > 0.0 else None
    status = "passed"
    message = ""
    failure_reason: Optional[str] = None

    if max_error_threshold is not None and max_error is not None and max_error > max_error_threshold:
        status = "failed"
        failure_reason = "threshold_exceeded"
        message = f"max_error {max_error:.6e} > threshold {max_error_threshold:.6e}"
    elif rms_error_threshold is not None and rms_error is not None and rms_error > rms_error_threshold:
        status = "failed"
        failure_reason = "threshold_exceeded"
        message = f"rms_error {rms_error:.6e} > threshold {rms_error_threshold:.6e}"
    elif (
        phase_error_threshold is not None
        and phase_error is not None
        and phase_error > phase_error_threshold
    ):
        status = "failed"
        failure_reason = "threshold_exceeded"
        message = f"phase_error_deg {phase_error:.6e} > threshold {phase_error_threshold:.6e}"
    elif (
        steady_state_max_threshold is not None
        and steady_max_error is not None
        and steady_max_error > steady_state_max_threshold
    ):
        status = "failed"
        failure_reason = "threshold_exceeded"
        message = (
            f"steady_state_max_error {steady_max_error:.6e} > "
            f"threshold {steady_state_max_threshold:.6e}"
        )
    elif (
        steady_state_rms_threshold is not None
        and steady_rms_error is not None
        and steady_rms_error > steady_state_rms_threshold
    ):
        status = "failed"
        failure_reason = "threshold_exceeded"
        message = (
            f"steady_state_rms_error {steady_rms_error:.6e} > "
            f"threshold {steady_state_rms_threshold:.6e}"
        )

    return BenchmarkResult(
        benchmark_id=benchmark_id,
        scenario=scenario_name,
        status=status,
        message=message,
        pulsim_runtime_s=pulsim_runtime,
        ngspice_runtime_s=ng_runtime,
        pulsim_steps=pulsim_steps,
        ngspice_steps=ng_steps,
        speedup=speedup,
        max_error=max_error,
        rms_error=rms_error,
        observables=observables,
        backend=backend_name,
        phase_error_deg=phase_error,
        steady_state_max_error=steady_max_error,
        steady_state_rms_error=steady_rms_error,
        failure_reason=failure_reason,
        reference_runtime_s=ng_runtime,
        reference_steps=ng_steps,
    )


def run_single_pair(
    pulsim_netlist: Path,
    spice_netlist: Path,
    output_dir: Path,
    cli_observables: Optional[List[str]],
    backend: str = "ngspice",
    backend_config: Optional[BackendConfig] = None,
) -> List[BenchmarkResult]:
    netlist = load_yaml(pulsim_netlist)
    apply_runtime_defaults(netlist)
    benchmark = netlist.get("benchmark", {})
    validation = benchmark.get("validation", {})
    benchmark_id = benchmark.get("id", pulsim_netlist.stem)

    observable_specs = resolve_observables(
        benchmark,
        validation,
        None,
        cli_observables,
        backend=backend,
    )
    thresholds = benchmark.get("expectations", {}).get("metrics", {})
    max_threshold = coerce_optional_float(thresholds.get("max_error"))
    rms_threshold = coerce_optional_float(thresholds.get("rms_error"))
    phase_threshold = coerce_optional_float(thresholds.get("phase_error_deg"))
    steady_max_threshold = coerce_optional_float(thresholds.get("steady_state_max_error"))
    steady_rms_threshold = coerce_optional_float(thresholds.get("steady_state_rms_error"))
    period_hint = _resolve_period_hint(validation, netlist)

    if not observable_specs:
        default_vector = "V(out)" if backend == "ltspice" else "v(out)"
        observable_specs = [ObservableSpec(column="V(out)", spice_vector=default_vector)]

    if backend == "ltspice" and backend_config is not None and backend_config.error:
        return [
            _result_for_error(
                benchmark_id=benchmark_id,
                scenario="default",
                backend=backend,
                message=backend_config.error,
                failure_reason="configuration_error",
            )
        ]

    effective_backend_config = backend_config
    if backend == "ngspice" and backend_config is not None and backend_config.executable is None:
        effective_backend_config = None

    try:
        result = run_case(
            scenario_netlist=netlist,
            benchmark_id=benchmark_id,
            scenario_name="default",
            spice_netlist_path=spice_netlist.resolve(),
            observable_specs=observable_specs,
            max_error_threshold=max_threshold,
            output_dir=output_dir,
            preferred_mode=None,
            backend_config=effective_backend_config,
            period_hint=period_hint,
            rms_error_threshold=rms_threshold,
            phase_error_threshold=phase_threshold,
            steady_state_max_threshold=steady_max_threshold,
            steady_state_rms_threshold=steady_rms_threshold,
        )
    except Exception as exc:  # pragma: no cover - runtime dependent
        result = _result_for_error(
            benchmark_id=benchmark_id,
            scenario="default",
            backend=backend,
            message=str(exc),
            failure_reason="runtime_error",
        )

    return [result]


def _resolve_spice_netlist(
    entry: Dict[str, Any],
    validation_meta: Dict[str, Any],
    backend: str,
) -> Optional[str]:
    keys = [f"{backend}_netlist", "spice_netlist"]
    if backend == "ngspice":
        keys.insert(1, "ngspice_netlist")
    if backend == "ltspice":
        keys.insert(1, "ltspice_netlist")

    for key in keys:
        raw = entry.get(key)
        if raw:
            return str(raw)
    for key in keys:
        raw = validation_meta.get(key)
        if raw:
            return str(raw)
    return None


def run_manifest(
    manifest_path: Path,
    output_dir: Path,
    only: Optional[List[str]],
    matrix: bool,
    force_scenario: Optional[str],
    cli_observables: Optional[List[str]],
    backend: str = "ngspice",
    ngspice_executable: Optional[Path] = None,
    ltspice_executable: Optional[Path] = None,
    backend_args: Optional[List[str]] = None,
) -> List[BenchmarkResult]:
    manifest = load_yaml(manifest_path)
    scenarios = manifest.get("scenarios", {})
    results: List[BenchmarkResult] = []

    backend_config = resolve_backend_config(
        backend=backend,
        manifest=manifest,
        ngspice_executable=ngspice_executable,
        ltspice_executable=ltspice_executable,
        backend_args=backend_args,
    )
    hard_config_error = backend == "ltspice" and backend_config.error is not None

    for entry in manifest.get("benchmarks", []):
        circuit_path = (manifest_path.parent / entry["path"]).resolve()
        netlist = load_yaml(circuit_path)
        bench_meta = netlist.get("benchmark", {})
        benchmark_id = bench_meta.get("id", circuit_path.stem)
        if only and benchmark_id not in only:
            continue

        validation = bench_meta.get("validation", {})
        entry_obs_key = f"{backend}_observables"
        entry_observables = entry.get(entry_obs_key) if isinstance(entry.get(entry_obs_key), list) else None
        observable_specs = resolve_observables(
            bench_meta,
            validation,
            entry_observables=entry_observables,
            cli_observables=cli_observables,
            backend=backend,
        )

        expectations = bench_meta.get("expectations", {}).get("metrics", {})
        max_threshold = coerce_optional_float(expectations.get("max_error"))
        rms_threshold = coerce_optional_float(expectations.get("rms_error"))
        phase_threshold = coerce_optional_float(expectations.get("phase_error_deg"))
        steady_max_threshold = coerce_optional_float(expectations.get("steady_state_max_error"))
        steady_rms_threshold = coerce_optional_float(expectations.get("steady_state_rms_error"))

        spice_rel = _resolve_spice_netlist(entry, validation, backend=backend)

        if force_scenario:
            scenario_names = [force_scenario]
        elif matrix:
            scenario_names = list(scenarios.keys())
        else:
            scenario_names = entry.get("scenarios", ["default"])
        if "default" in scenario_names and "default" not in scenarios:
            scenarios["default"] = {}

        if hard_config_error:
            for scenario_name in scenario_names:
                results.append(
                    _result_for_error(
                        benchmark_id=benchmark_id,
                        scenario=scenario_name,
                        backend=backend,
                        message=backend_config.error,
                        failure_reason="configuration_error",
                    )
                )
            continue

        if not spice_rel:
            status = "skipped"
            reason = None
            for scenario_name in scenario_names:
                entry_result = _result_for_error(
                    benchmark_id=benchmark_id,
                    scenario=scenario_name,
                    backend=backend,
                    message=(
                        f"Missing {backend} netlist mapping "
                        f"(set entry.{backend}_netlist or benchmark.validation.{backend}_netlist)"
                    ),
                    failure_reason=reason or "mapping_error",
                )
                entry_result.status = status
                if status == "skipped":
                    entry_result.failure_reason = None
                results.append(entry_result)
            continue

        spice_path = (manifest_path.parent / spice_rel).resolve()
        if not spice_path.exists():
            for scenario_name in scenario_names:
                results.append(
                    _result_for_error(
                        benchmark_id=benchmark_id,
                        scenario=scenario_name,
                        backend=backend,
                        message=f"{backend} netlist not found: {spice_path}",
                        failure_reason="mapping_error",
                    )
                )
            continue

        if not observable_specs:
            status = "skipped"
            reason = None
            for scenario_name in scenario_names:
                entry_result = _result_for_error(
                    benchmark_id=benchmark_id,
                    scenario=scenario_name,
                    backend=backend,
                    message=f"No observables configured for {backend} comparison",
                    failure_reason=reason or "mapping_error",
                )
                entry_result.status = status
                if status == "skipped":
                    entry_result.failure_reason = None
                results.append(entry_result)
            continue

        for scenario_name in scenario_names:
            scenario_override = scenarios.get(scenario_name, {})
            scenario_netlist = deep_merge(netlist, scenario_override)
            preferred_mode = infer_preferred_mode(scenario_name, scenario_override)
            normalize_periodic_mode(scenario_netlist, preferred_mode)
            apply_runtime_defaults(scenario_netlist)
            period_hint = _resolve_period_hint(validation, scenario_netlist)

            effective_backend_config: Optional[BackendConfig] = backend_config
            if backend == "ngspice" and backend_config.executable is None:
                effective_backend_config = None

            try:
                result = run_case(
                    scenario_netlist=scenario_netlist,
                    benchmark_id=benchmark_id,
                    scenario_name=scenario_name,
                    spice_netlist_path=spice_path,
                    observable_specs=observable_specs,
                    max_error_threshold=max_threshold,
                    output_dir=output_dir,
                    preferred_mode=preferred_mode,
                    backend_config=effective_backend_config,
                    period_hint=period_hint,
                    rms_error_threshold=rms_threshold,
                    phase_error_threshold=phase_threshold,
                    steady_state_max_threshold=steady_max_threshold,
                    steady_state_rms_threshold=steady_rms_threshold,
                )
            except Exception as exc:  # pragma: no cover - runtime dependent
                result = _result_for_error(
                    benchmark_id=benchmark_id,
                    scenario=scenario_name,
                    backend=backend,
                    message=str(exc),
                    failure_reason="runtime_error",
                )
            results.append(result)

    return results


def write_results(
    output_dir: Path,
    results: List[BenchmarkResult],
    backend: str = "ngspice",
    executable: Optional[Path] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "parity_results.csv"
    json_path = output_dir / "parity_results.json"
    summary_path = output_dir / "parity_summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "benchmark_id",
                "scenario",
                "backend",
                "status",
                "pulsim_runtime_s",
                "reference_runtime_s",
                "speedup",
                "pulsim_steps",
                "reference_steps",
                "max_error",
                "rms_error",
                "phase_error_deg",
                "steady_state_max_error",
                "steady_state_rms_error",
                "failure_reason",
                "message",
            ]
        )
        for item in results:
            reference_runtime = (
                item.reference_runtime_s if item.reference_runtime_s is not None else item.ngspice_runtime_s
            )
            reference_steps = item.reference_steps if item.reference_steps is not None else item.ngspice_steps
            writer.writerow(
                [
                    item.benchmark_id,
                    item.scenario,
                    item.backend,
                    item.status,
                    f"{item.pulsim_runtime_s:.6f}",
                    f"{reference_runtime:.6f}",
                    "" if item.speedup is None else f"{item.speedup:.6f}",
                    item.pulsim_steps,
                    reference_steps,
                    "" if item.max_error is None else f"{item.max_error:.6e}",
                    "" if item.rms_error is None else f"{item.rms_error:.6e}",
                    "" if item.phase_error_deg is None else f"{item.phase_error_deg:.6e}",
                    "" if item.steady_state_max_error is None else f"{item.steady_state_max_error:.6e}",
                    "" if item.steady_state_rms_error is None else f"{item.steady_state_rms_error:.6e}",
                    item.failure_reason or "",
                    item.message,
                ]
            )

    payload = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "backend": backend,
        "executable": str(executable) if executable is not None else None,
        "results": [
            {
                **asdict(item),
                "observables": [asdict(obs) for obs in item.observables],
            }
            for item in results
        ],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    summary = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "backend": backend,
        "passed": sum(1 for item in results if item.status == "passed"),
        "failed": sum(1 for item in results if item.status == "failed"),
        "skipped": sum(1 for item in results if item.status == "skipped"),
        "total": len(results),
        "failure_reasons": {},
    }
    for item in results:
        reason = item.failure_reason
        if reason:
            summary["failure_reasons"][reason] = summary["failure_reasons"].get(reason, 0) + 1

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if backend == "ngspice":
        (output_dir / "ngspice_results.csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
        (output_dir / "ngspice_results.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
        (output_dir / "ngspice_summary.json").write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Pulsim and external SPICE backends")
    parser.add_argument("--benchmarks", type=Path, default=Path(__file__).with_name("benchmarks.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/parity_out"))
    parser.add_argument("--backend", choices=["ngspice", "ltspice"], default="ngspice")
    parser.add_argument("--ngspice-exe", type=Path, default=None, help="Path to ngspice executable")
    parser.add_argument("--ltspice-exe", type=Path, default=None, help="Path to LTspice executable")
    parser.add_argument(
        "--backend-arg",
        action="append",
        default=None,
        help="Extra argument passed to backend executable (repeatable).",
    )
    parser.add_argument("--only", nargs="*", help="Benchmark ids to run (manifest mode)")
    parser.add_argument("--matrix", action="store_true", help="Run all scenarios from manifest")
    parser.add_argument("--scenario", type=str, default=None, help="Run only one scenario name")
    parser.add_argument(
        "--observable",
        action="append",
        default=None,
        help="Observable column to compare (repeatable)",
    )
    parser.add_argument("--pulsim-netlist", type=Path, default=None, help="Single mode: Pulsim YAML netlist")
    parser.add_argument("--spice-netlist", type=Path, default=None, help="Single mode: SPICE netlist")
    args = parser.parse_args()

    if yaml is None:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml")

    manifest: Optional[Dict[str, Any]] = None
    if args.pulsim_netlist is None and args.spice_netlist is None:
        manifest = load_yaml(args.benchmarks.resolve())

    backend_config = resolve_backend_config(
        backend=args.backend,
        manifest=manifest,
        ngspice_executable=args.ngspice_exe,
        ltspice_executable=args.ltspice_exe,
        backend_args=args.backend_arg,
    )

    if args.pulsim_netlist is not None or args.spice_netlist is not None:
        if args.pulsim_netlist is None or args.spice_netlist is None:
            raise SystemExit("Single mode requires both --pulsim-netlist and --spice-netlist")
        results = run_single_pair(
            pulsim_netlist=args.pulsim_netlist.resolve(),
            spice_netlist=args.spice_netlist.resolve(),
            output_dir=args.output_dir,
            cli_observables=args.observable,
            backend=args.backend,
            backend_config=backend_config,
        )
    else:
        results = run_manifest(
            manifest_path=args.benchmarks.resolve(),
            output_dir=args.output_dir,
            only=args.only,
            matrix=args.matrix,
            force_scenario=args.scenario,
            cli_observables=args.observable,
            backend=args.backend,
            ngspice_executable=args.ngspice_exe,
            ltspice_executable=args.ltspice_exe,
            backend_args=args.backend_arg,
        )

    write_results(
        output_dir=args.output_dir,
        results=results,
        backend=args.backend,
        executable=backend_config.executable,
    )
    summary = {
        "passed": sum(1 for item in results if item.status == "passed"),
        "failed": sum(1 for item in results if item.status == "failed"),
        "skipped": sum(1 for item in results if item.status == "skipped"),
        "total": len(results),
        "backend": args.backend,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
