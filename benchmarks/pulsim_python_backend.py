#!/usr/bin/env python3
"""Python API fallback backend for benchmark runners."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def parse_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        raise ValueError("Missing numeric value")
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


def _import_pulsim():
    try:
        import pulsim as ps
    except Exception:  # pragma: no cover - runtime import path dependent
        return None
    return ps


def is_available() -> bool:
    ps = _import_pulsim()
    return bool(ps and hasattr(ps, "Circuit") and hasattr(ps, "run_transient"))


def _nodes_from_component(component: Dict[str, Any]) -> List[str]:
    if isinstance(component.get("nodes"), list):
        return [str(x) for x in component["nodes"]]
    if component.get("n1") is not None and component.get("n2") is not None:
        return [str(component["n1"]), str(component["n2"])]
    if component.get("npos") is not None and component.get("nneg") is not None:
        return [str(component["npos"]), str(component["nneg"])]
    return []


def _resolve_node(ps: Any, node_map: Dict[str, int], name: str) -> int:
    lowered = name.strip().lower()
    if lowered in {"0", "gnd", "ground"}:
        return int(ps.Circuit.ground())
    if name not in node_map:
        raise RuntimeError(f"Node '{name}' was not created before device stamping")
    return node_map[name]


def _coerce_g(value: Any, reciprocal_from_resistance: Any, default: float) -> float:
    if value is not None:
        return parse_value(value)
    if reciprocal_from_resistance is not None:
        resistance = parse_value(reciprocal_from_resistance)
        return (1.0 / resistance) if resistance > 0.0 else 0.0
    return default


def _add_voltage_source(
    ps: Any,
    circuit: Any,
    name: str,
    nodes: Sequence[int],
    component: Dict[str, Any],
) -> None:
    waveform = component.get("waveform", {})
    if not isinstance(waveform, dict) or not waveform:
        value = parse_value(component.get("value", 0.0))
        circuit.add_voltage_source(name, nodes[0], nodes[1], value)
        return

    wtype = str(waveform.get("type", "dc")).strip().lower()
    if wtype == "dc":
        value = parse_value(waveform.get("value", component.get("value", 0.0)))
        circuit.add_voltage_source(name, nodes[0], nodes[1], value)
        return
    if wtype == "pulse":
        params = ps.PulseParams()
        if waveform.get("v_initial") is not None:
            params.v_initial = parse_value(waveform["v_initial"])
        if waveform.get("v_pulse") is not None:
            params.v_pulse = parse_value(waveform["v_pulse"])
        if waveform.get("t_delay") is not None:
            params.t_delay = parse_value(waveform["t_delay"])
        if waveform.get("t_rise") is not None:
            params.t_rise = parse_value(waveform["t_rise"])
        if waveform.get("t_fall") is not None:
            params.t_fall = parse_value(waveform["t_fall"])
        if waveform.get("t_width") is not None:
            params.t_width = parse_value(waveform["t_width"])
        if waveform.get("period") is not None:
            params.period = parse_value(waveform["period"])
        circuit.add_pulse_voltage_source(name, nodes[0], nodes[1], params)
        return
    if wtype == "sine":
        params = ps.SineParams()
        if waveform.get("amplitude") is not None:
            params.amplitude = parse_value(waveform["amplitude"])
        if waveform.get("frequency") is not None:
            params.frequency = parse_value(waveform["frequency"])
        if waveform.get("offset") is not None:
            params.offset = parse_value(waveform["offset"])
        if waveform.get("phase") is not None:
            params.phase = parse_value(waveform["phase"])
        circuit.add_sine_voltage_source(name, nodes[0], nodes[1], params)
        return
    if wtype == "pwm":
        params = ps.PWMParams()
        if waveform.get("v_high") is not None:
            params.v_high = parse_value(waveform["v_high"])
        if waveform.get("v_low") is not None:
            params.v_low = parse_value(waveform["v_low"])
        if waveform.get("frequency") is not None:
            params.frequency = parse_value(waveform["frequency"])
        if waveform.get("duty") is not None:
            params.duty = parse_value(waveform["duty"])
        if waveform.get("dead_time") is not None:
            params.dead_time = parse_value(waveform["dead_time"])
        if waveform.get("phase") is not None:
            params.phase = parse_value(waveform["phase"])
        if waveform.get("t_rise") is not None and hasattr(params, "rise_time"):
            params.rise_time = parse_value(waveform["t_rise"])
        if waveform.get("t_fall") is not None and hasattr(params, "fall_time"):
            params.fall_time = parse_value(waveform["t_fall"])
        circuit.add_pwm_voltage_source(name, nodes[0], nodes[1], params)
        return

    raise RuntimeError(f"Unsupported voltage source waveform in Python backend: {wtype}")


def _build_circuit_from_yaml(ps: Any, netlist: Dict[str, Any]) -> Tuple[Any, Dict[str, Any], List[str]]:
    circuit = ps.Circuit()
    node_map: Dict[str, int] = {}
    branch_names: List[str] = []

    components = [c for c in netlist.get("components", []) if isinstance(c, dict)]

    # Circuit expects nodes to exist before devices are added.
    for component in components:
        for node_name in _nodes_from_component(component):
            lowered = node_name.strip().lower()
            if lowered in {"0", "gnd", "ground"}:
                continue
            if node_name not in node_map:
                node_map[node_name] = int(circuit.add_node(node_name))

    for index, component in enumerate(components):
        ctype = str(component.get("type", "")).strip().lower()
        name = str(component.get("name", f"X{index + 1}"))
        node_names = _nodes_from_component(component)
        if len(node_names) < 2:
            raise RuntimeError(f"Component '{name}' has invalid or missing nodes")
        nodes = [_resolve_node(ps, node_map, n) for n in node_names]

        if ctype in {"resistor", "r"}:
            value = parse_value(component.get("value", component.get("resistance")))
            circuit.add_resistor(name, nodes[0], nodes[1], value)
        elif ctype in {"capacitor", "c"}:
            value = parse_value(component.get("value", component.get("capacitance")))
            ic = parse_value(component.get("ic", 0.0))
            circuit.add_capacitor(name, nodes[0], nodes[1], value, ic)
        elif ctype in {"inductor", "l"}:
            value = parse_value(component.get("value", component.get("inductance")))
            ic = parse_value(component.get("ic", 0.0))
            circuit.add_inductor(name, nodes[0], nodes[1], value, ic)
            branch_names.append(f"I({name})")
        elif ctype in {"voltage_source", "v"}:
            _add_voltage_source(ps, circuit, name, nodes, component)
            branch_names.append(f"I({name})")
        elif ctype in {"current_source", "i"}:
            value = parse_value(component.get("value", 0.0))
            circuit.add_current_source(name, nodes[0], nodes[1], value)
        elif ctype in {"diode", "d"}:
            g_on = _coerce_g(component.get("g_on"), component.get("ron"), 1e3)
            g_off = _coerce_g(component.get("g_off"), component.get("roff"), 1e-9)
            circuit.add_diode(name, nodes[0], nodes[1], g_on, g_off)
        elif ctype == "vcswitch":
            if len(nodes) < 3:
                raise RuntimeError(f"Component '{name}' (vcswitch) requires 3 nodes")
            v_threshold = parse_value(component.get("v_threshold", 2.5))
            g_on = _coerce_g(component.get("g_on"), component.get("ron"), 1e3)
            g_off = _coerce_g(component.get("g_off"), component.get("roff"), 1e-9)
            circuit.add_vcswitch(name, nodes[0], nodes[1], nodes[2], v_threshold, g_on, g_off)
        else:
            raise RuntimeError(f"Unsupported component type in Python backend: {ctype}")

    return circuit, netlist.get("simulation", {}), branch_names


def _make_newton_options(ps: Any, simulation: Dict[str, Any], circuit: Any) -> Any:
    opts = ps.NewtonOptions()
    newton = simulation.get("newton", {}) if isinstance(simulation.get("newton"), dict) else {}

    for field in (
        "max_iterations",
        "initial_damping",
        "min_damping",
        "auto_damping",
        "track_history",
        "check_per_variable",
        "enable_limiting",
        "max_voltage_step",
        "max_current_step",
    ):
        if field in newton and hasattr(opts, field):
            value = newton[field]
            if isinstance(getattr(opts, field), bool):
                setattr(opts, field, bool(value))
            elif isinstance(getattr(opts, field), int):
                setattr(opts, field, int(value))
            else:
                setattr(opts, field, parse_value(value))

    opts.num_nodes = int(circuit.num_nodes())
    opts.num_branches = int(circuit.num_branches())
    return opts


def _write_transient_csv(
    output_path: Path,
    times: Sequence[float],
    states: Sequence[Sequence[float]],
    node_names: Sequence[str],
    branch_names: Sequence[str],
    num_branches: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    effective_branch_names = list(branch_names)
    while len(effective_branch_names) < num_branches:
        effective_branch_names.append(f"I(B{len(effective_branch_names) + 1})")

    header = ["time"] + [f"V({name})" for name in node_names] + effective_branch_names[:num_branches]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for t, state in zip(times, states):
            row = [f"{float(t):.9e}"]
            row.extend(f"{float(v):.9e}" for v in state)
            writer.writerow(row)

    return max(0, len(times) - 1)


def run_from_yaml(netlist_path: Path, output_path: Path) -> Tuple[float, int, str]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for Python backend (pip install pyyaml)")

    ps = _import_pulsim()
    if ps is None:
        raise RuntimeError("Python package 'pulsim' is not available")

    with open(netlist_path, "r", encoding="utf-8") as handle:
        netlist = yaml.safe_load(handle)
    if not isinstance(netlist, dict):
        raise RuntimeError("Invalid YAML netlist")

    circuit, simulation, branch_names = _build_circuit_from_yaml(ps, netlist)
    if not isinstance(simulation, dict):
        simulation = {}

    # The public Python API available in this environment exposes only transient run.
    if "shooting" in simulation or "harmonic_balance" in simulation or "hb" in simulation:
        raise RuntimeError("Python backend does not support shooting/harmonic balance scenarios")

    t_start = parse_value(simulation.get("tstart", 0.0))
    t_stop = parse_value(simulation.get("tstop", 1e-3))
    dt = parse_value(simulation.get("dt", 1e-6))
    if dt <= 0.0:
        raise RuntimeError("Simulation dt must be > 0")
    if t_stop <= t_start:
        raise RuntimeError("Simulation tstop must be greater than tstart")

    newton_opts = _make_newton_options(ps, simulation, circuit)

    start = time.perf_counter()
    times, states, success, message = ps.run_transient(circuit, t_start, t_stop, dt, newton_opts)
    elapsed = time.perf_counter() - start

    if not success:
        raise RuntimeError(message or "Transient simulation failed in Python backend")

    node_names = list(circuit.node_names())
    steps = _write_transient_csv(
        output_path=output_path,
        times=times,
        states=states,
        node_names=node_names,
        branch_names=branch_names,
        num_branches=int(circuit.num_branches()),
    )
    stdout = "\n".join(
        [
            "Backend: python-api",
            f"steps: {steps}",
            f"runtime_s: {elapsed:.6f}",
        ]
    )
    return elapsed, steps, stdout
