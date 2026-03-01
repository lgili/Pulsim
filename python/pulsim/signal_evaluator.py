"""Signal-flow evaluator for closed-loop control blocks.

This module builds a Directed Acyclic Graph (DAG) from the signal-domain
components in a circuit description and evaluates them in topological order
each simulation step, enabling closed-loop control (PI → PWM) without the GUI.

It can be used standalone with the ``pulsim`` library, without any GUI dependency::

    import pulsim as ps
    from pulsim.signal_evaluator import SignalEvaluator

    # circuit_data is a plain dict – no GUI types required.
    # Keys: "components" (list), "wires" (list), "node_map" (dict), "node_aliases" (dict)
    #
    # Each component dict must have:
    #   id        – unique string identifier
    #   name      – human-readable label (used as PWM backend name)
    #   type      – string matching one of the SIGNAL_TYPES below
    #   parameters – dict of block parameters (kp, ki, gain, value, etc.)
    #   pins       – list of {index, name, x, y} dicts
    #
    # Each wire dict must have:
    #   start_connection – {component_id, pin_index}
    #   end_connection   – {component_id, pin_index}

    circuit_data = {
        "components": [
            {"id": "ref", "name": "REF",  "type": "CONSTANT",
             "parameters": {"value": 12.0},
             "pins": [{"index": 0, "name": "OUT", "x": 0, "y": 0}]},
            {"id": "pi1", "name": "PI1",  "type": "PI_CONTROLLER",
             "parameters": {"kp": 0.5, "ki": 50.0, "output_min": 0.0, "output_max": 1.0},
             "pins": [{"index": 0, "name": "IN", "x": 0, "y": 0},
                      {"index": 1, "name": "OUT", "x": 0, "y": 0}]},
            {"id": "pwm", "name": "PWM1", "type": "PWM_GENERATOR",
             "parameters": {"frequency": 10000, "duty_cycle": 0.5},
             "pins": [{"index": 0, "name": "OUT", "x": 0, "y": 0},
                      {"index": 1, "name": "DUTY_IN", "x": 0, "y": 0}]},
        ],
        "wires": [
            {"start_connection": {"component_id": "ref",  "pin_index": 0},
             "end_connection":   {"component_id": "pi1",  "pin_index": 0}},
            {"start_connection": {"component_id": "pi1",  "pin_index": 1},
             "end_connection":   {"component_id": "pwm",  "pin_index": 1}},
        ],
        "node_map": {}, "node_aliases": {},
    }

    ev = SignalEvaluator(circuit_data)
    ev.build()

    # Register duty callback on the C++ circuit object
    circuit = ps.Circuit()
    # ... build your circuit ...
    for comp_id, pwm_name in ev.pwm_components().items():
        circuit.set_pwm_duty_callback(
            pwm_name,
            lambda t, cid=comp_id, evaluator=ev: evaluator.step(t)[cid],
        )

    # Each simulation step, inject probe measurements:
    # ev.update_probes({"vprobe_id": measured_voltage})
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy native C++ controller imports – do not crash if binding not installed
# ---------------------------------------------------------------------------

def _try_import_native() -> dict[str, Any]:
    """Try to import C++ control classes from the pulsim binding."""
    native: dict[str, Any] = {}
    try:
        from pulsim._pulsim import (  # type: ignore[import]
            PIController,
            PIDController,
            RateLimiter,
            HysteresisController,
            SampleHold,
        )
        native["PIController"] = PIController
        native["PIDController"] = PIDController
        native["RateLimiter"] = RateLimiter
        native["HysteresisController"] = HysteresisController
        native["SampleHold"] = SampleHold
    except Exception as exc:  # pragma: no cover
        log.debug("Native C++ controller classes not available: %s", exc)
    return native


_NATIVE = _try_import_native()


# ---------------------------------------------------------------------------
# Signal-domain component type registry
# ---------------------------------------------------------------------------

# String names matching the component type field in circuit_data dicts.
SIGNAL_TYPES: frozenset[str] = frozenset({
    "CONSTANT",
    "GAIN",
    "SUM",
    "SUBTRACTOR",
    "LIMITER",
    "RATE_LIMITER",
    "PI_CONTROLLER",
    "PID_CONTROLLER",
    "PWM_GENERATOR",
    "VOLTAGE_PROBE",
    "CURRENT_PROBE",
    "POWER_PROBE",
    "INTEGRATOR",
    "DIFFERENTIATOR",
    "HYSTERESIS",
    "SAMPLE_HOLD",
    "MATH_BLOCK",
    "SIGNAL_MUX",
    "SIGNAL_DEMUX",
})

# Types that are pure sources (no signal input pins).
_SOURCE_TYPES: frozenset[str] = frozenset({
    "CONSTANT",
    "VOLTAGE_PROBE",
    "CURRENT_PROBE",
    "POWER_PROBE",
})

# Output pin names per type (first matching name is treated as output).
_OUTPUT_PIN_NAMES: dict[str, list[str]] = {
    "CONSTANT":        ["OUT"],
    "GAIN":            ["OUT"],
    "SUM":             ["OUT"],
    "SUBTRACTOR":      ["OUT"],
    "LIMITER":         ["OUT"],
    "RATE_LIMITER":    ["OUT"],
    "PI_CONTROLLER":   ["OUT"],
    "PID_CONTROLLER":  ["OUT"],
    "INTEGRATOR":      ["OUT"],
    "DIFFERENTIATOR":  ["OUT"],
    "HYSTERESIS":      ["OUT"],
    "SIGNAL_MUX":      ["OUT"],
    "SIGNAL_DEMUX":    ["OUT1", "OUT2", "OUT3", "OUT4", "OUT5", "OUT6", "OUT7", "OUT8"],
    "VOLTAGE_PROBE":   ["OUT"],
    "CURRENT_PROBE":   ["MEAS"],
    "POWER_PROBE":     ["OUT"],
    "PWM_GENERATOR":   ["OUT"],
    "MATH_BLOCK":      ["OUT"],
    "SAMPLE_HOLD":     ["OUT"],
}


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------

class AlgebraicLoopError(RuntimeError):
    """Raised when a cycle is detected in the signal-flow graph.

    Attributes
    ----------
    cycle_ids : list[str]
        Component names (or IDs) that form the cycle.
    """

    def __init__(self, cycle_ids: list[str]) -> None:
        self.cycle_ids = cycle_ids
        names = ", ".join(cycle_ids)
        super().__init__(
            f"Algebraic loop detected in signal network. "
            f"Blocks involved: [{names}]. "
            "Break the loop (e.g. add a unit-delay or restructure the control path)."
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SignalEvaluator:
    """Evaluates a signal-flow DAG from a circuit description dict each step.

    The evaluator is deliberately independent of any GUI framework.  It
    operates on plain Python dicts (the ``circuit_data`` format) and uses the
    native C++ control objects from ``pulsim._pulsim`` when available,
    falling back to pure-Python equivalents otherwise.

    Parameters
    ----------
    circuit_data : dict
        Circuit description with keys ``"components"``, ``"wires"``,
        ``"node_map"``, and ``"node_aliases"``.  The format is identical to
        the dict produced by the GUI's ``SimulationService.convert_gui_circuit()``,
        but it can also be constructed programmatically (no GUI needed).

    Examples
    --------
    See module docstring for a complete usage example.
    """

    def __init__(self, circuit_data: dict) -> None:
        self._circuit_data = circuit_data

        # Component registry: {comp_id: comp_dict}
        self._comps: dict[str, dict] = {}
        # Signal DAG adjacency: src_comp_id → [(dst_comp_id, dst_pin_name)]
        self._adj: dict[str, list[tuple[str, str]]] = {}
        # Current output value per component
        self._state: dict[str, float] = {}
        # Stateful controller objects (C++ or Python dict) keyed by comp_id
        self._controllers: dict[str, Any] = {}
        # Evaluation order (topologically sorted comp_ids)
        self._order: list[str] = []
        # PWM generators that have a DUTY_IN wire connected: {comp_id: name}
        self._pwm_names: dict[str, str] = {}
        # Probe node names for diagnostics
        self._probe_nodes: dict[str, str] = {}

    # =========================================================================
    # Public API
    # =========================================================================

    def build(self) -> None:
        """Parse the circuit dict, build the signal DAG, and initialise state.

        Must be called once before :meth:`step`.  Call again to rebuild after
        the circuit description changes.

        Raises
        ------
        AlgebraicLoopError
            When a directed cycle exists among the signal-domain blocks.
        """
        self._comps.clear()
        self._adj.clear()
        self._state.clear()
        self._controllers.clear()
        self._order.clear()
        self._pwm_names.clear()

        self._collect_signal_components()
        self._build_graph()
        self._order = self._topological_sort()
        self._init_controllers()

        # Only expose PWM components that have at least one incoming signal wire.
        connected_ids: set[str] = {
            dst_id
            for edges in self._adj.values()
            for dst_id, _ in edges
        }
        self._pwm_names = {
            cid: name
            for cid, name in self._pwm_names.items()
            if cid in connected_ids
        }

        log.debug(
            "SignalEvaluator built: %d blocks, order=%s",
            len(self._order),
            self._order,
        )

    def has_signal_blocks(self) -> bool:
        """Return True if there are any evaluable signal blocks."""
        return bool(self._order)

    def pwm_components(self) -> dict[str, str]:
        """Return ``{comp_id: backend_name}`` for PWM generators with DUTY_IN connected."""
        return dict(self._pwm_names)

    def update_probes(self, probe_values: dict[str, float]) -> None:
        """Inject probe measurements into the evaluator state.

        Call this after each electrical simulation step with the latest
        measured values so that feedback blocks (VOLTAGE_PROBE, CURRENT_PROBE)
        reflect the true circuit state.

        Parameters
        ----------
        probe_values : dict[str, float]
            ``{component_id: measured_value}`` mapping.  Unrecognised IDs
            are silently ignored.
        """
        for comp_id, value in probe_values.items():
            if comp_id in self._comps:
                self._state[comp_id] = float(value)

    def step(self, t: float) -> dict[str, float]:
        """Evaluate all signal blocks in topological order at time *t*.

        Parameters
        ----------
        t : float
            Current simulation time in seconds.

        Returns
        -------
        dict[str, float]
            ``{comp_id: output_value}`` for every signal block.
            PWM duty values are clamped to ``[0, 1]``.
        """
        for comp_id in self._order:
            comp = self._comps[comp_id]
            ctype = comp.get("type", "")
            params = comp.get("parameters") or {}

            # Source blocks ─────────────────────────────────────────────────
            if ctype in _SOURCE_TYPES:
                if ctype == "CONSTANT":
                    self._state[comp_id] = float(params.get("value", 0.0))
                # Probe types: value injected externally via update_probes()
                continue

            inputs = self._collect_inputs(comp_id, comp)

            # Processing blocks ──────────────────────────────────────────────
            if ctype == "GAIN":
                k = float(params.get("gain", 1.0))
                self._state[comp_id] = k * (inputs[0] if inputs else 0.0)

            elif ctype in ("SUM", "MATH_BLOCK"):
                signs = list(params.get("signs") or ["+"] * len(inputs))
                total = 0.0
                for i, v in enumerate(inputs):
                    s = signs[i] if i < len(signs) else "+"
                    total += v if s == "+" else -v
                self._state[comp_id] = total

            elif ctype == "SUBTRACTOR":
                a = inputs[0] if len(inputs) > 0 else 0.0
                b = inputs[1] if len(inputs) > 1 else 0.0
                self._state[comp_id] = a - b

            elif ctype == "LIMITER":
                lo = float(params.get("lower_limit", -1e9))
                hi = float(params.get("upper_limit", 1e9))
                val = inputs[0] if inputs else 0.0
                self._state[comp_id] = max(lo, min(hi, val))

            elif ctype == "RATE_LIMITER":
                ctl = self._controllers.get(comp_id)
                if ctl is not None:
                    self._state[comp_id] = ctl.update(inputs[0] if inputs else 0.0, t)
                else:
                    self._state[comp_id] = inputs[0] if inputs else 0.0

            elif ctype == "INTEGRATOR":
                ctl = self._controllers.get(comp_id)
                if ctl is not None:
                    t_prev = ctl["t_prev"]
                    dt = (t - t_prev) if t_prev >= 0 else 0.0
                    ctl["t_prev"] = t
                    k = float(params.get("gain", 1.0))
                    ctl["integral"] = ctl.get("integral", 0.0) + k * (inputs[0] if inputs else 0.0) * dt
                    lo = float(params.get("output_min", -1e6))
                    hi = float(params.get("output_max", 1e6))
                    self._state[comp_id] = max(lo, min(hi, ctl["integral"]))

            elif ctype == "PI_CONTROLLER":
                ctl = self._controllers.get(comp_id)
                error = inputs[0] if inputs else 0.0
                if isinstance(ctl, dict):
                    # Pure-Python fallback
                    t_prev = ctl["t_prev"]
                    dt = (t - t_prev) if t_prev >= 0.0 else 0.0
                    ctl["t_prev"] = t
                    kp = float(params.get("kp", 1.0))
                    ki = float(params.get("ki", 0.0))
                    ctl["integral"] += error * dt
                    raw = kp * error + ki * ctl["integral"]
                    lo = float(params.get("output_min", -1e9))
                    hi = float(params.get("output_max", 1e9))
                    self._state[comp_id] = max(lo, min(hi, raw))
                elif ctl is not None:
                    # Native C++ PIController
                    self._state[comp_id] = ctl.update(error, t)
                else:
                    self._state[comp_id] = 0.0

            elif ctype == "PID_CONTROLLER":
                ctl = self._controllers.get(comp_id)
                error = inputs[0] if inputs else 0.0
                if ctl is not None and not isinstance(ctl, dict):
                    self._state[comp_id] = ctl.update(error, t)
                else:
                    self._state[comp_id] = error

            elif ctype == "HYSTERESIS":
                ctl = self._controllers.get(comp_id)
                if ctl is not None and not isinstance(ctl, dict):
                    self._state[comp_id] = ctl.update(inputs[0] if inputs else 0.0)
                else:
                    self._state[comp_id] = inputs[0] if inputs else 0.0

            elif ctype == "SAMPLE_HOLD":
                ctl = self._controllers.get(comp_id)
                if ctl is not None and not isinstance(ctl, dict):
                    self._state[comp_id] = ctl.update(inputs[0] if inputs else 0.0, t)
                else:
                    self._state[comp_id] = inputs[0] if inputs else 0.0

            elif ctype in ("SIGNAL_MUX", "SIGNAL_DEMUX"):
                self._state[comp_id] = inputs[0] if inputs else 0.0

            elif ctype == "PWM_GENERATOR":
                duty = inputs[0] if inputs else float(params.get("duty_cycle", 0.5))
                self._state[comp_id] = max(0.0, min(1.0, duty))

            else:
                self._state[comp_id] = inputs[0] if inputs else 0.0

        return dict(self._state)

    def get_pwm_duty(self, comp_id: str) -> float:
        """Return current computed duty cycle [0, 1] for a PWM component."""
        return float(self._state.get(comp_id, 0.5))

    def reset(self) -> None:
        """Reset all stateful controller objects (integrators, PI state, etc.)."""
        for ctl in self._controllers.values():
            if hasattr(ctl, "reset"):
                ctl.reset()
            elif isinstance(ctl, dict):
                ctl.update({"integral": 0.0, "t_prev": -1.0})
        log.debug("SignalEvaluator: all controller states reset")

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _collect_signal_components(self) -> None:
        for comp in self._circuit_data.get("components", []):
            ctype = comp.get("type", "")
            if ctype in SIGNAL_TYPES:
                comp_id = str(comp["id"])
                self._comps[comp_id] = comp
                self._adj[comp_id] = []
                self._state[comp_id] = 0.0
                if ctype == "PWM_GENERATOR":
                    self._pwm_names[comp_id] = comp.get("name", comp_id)
                if ctype in ("VOLTAGE_PROBE", "CURRENT_PROBE"):
                    nodes = comp.get("pin_nodes") or []
                    self._probe_nodes[comp_id] = nodes[0] if nodes else ""

    def _build_graph(self) -> None:
        comp_pin_set: set[tuple[str, int]] = {
            (cid, pin["index"])
            for cid, comp in self._comps.items()
            for pin in comp.get("pins") or []
        }

        for wire in self._circuit_data.get("wires", []):
            sc = wire.get("start_connection") or {}
            ec = wire.get("end_connection") or {}
            if not sc or not ec:
                continue

            src_id = str(sc.get("component_id", ""))
            src_pin = int(sc.get("pin_index", -1))
            dst_id = str(ec.get("component_id", ""))
            dst_pin = int(ec.get("pin_index", -1))

            if src_id not in self._comps or dst_id not in self._comps:
                continue
            if (src_id, src_pin) not in comp_pin_set or (dst_id, dst_pin) not in comp_pin_set:
                continue

            src_comp = self._comps[src_id]
            src_type = src_comp.get("type", "")
            output_names = _OUTPUT_PIN_NAMES.get(src_type, ["OUT"])
            src_pin_name = self._pin_name(src_comp, src_pin)

            if src_pin_name not in output_names:
                # Wire might be stored in reverse order – try swapping
                dst_comp = self._comps[dst_id]
                dst_type = dst_comp.get("type", "")
                dst_output_names = _OUTPUT_PIN_NAMES.get(dst_type, ["OUT"])
                dst_pin_name = self._pin_name(dst_comp, dst_pin)
                if dst_pin_name in dst_output_names:
                    src_id, dst_id = dst_id, src_id
                    src_pin, dst_pin = dst_pin, src_pin
                    src_comp, dst_comp = dst_comp, src_comp
                    src_type, dst_type = dst_type, src_type
                    src_pin_name, dst_pin_name = dst_pin_name, src_pin_name
                else:
                    continue

            dst_pin_name = self._pin_name(self._comps[dst_id], dst_pin)
            self._adj[src_id].append((dst_id, dst_pin_name))

    def _topological_sort(self) -> list[str]:
        in_degree: dict[str, int] = {cid: 0 for cid in self._comps}
        for src_id, edges in self._adj.items():
            for dst_id, _ in edges:
                in_degree[dst_id] = in_degree.get(dst_id, 0) + 1

        queue: deque[str] = deque(
            cid for cid, deg in in_degree.items() if deg == 0
        )
        order: list[str] = []
        while queue:
            cid = queue.popleft()
            order.append(cid)
            for dst_id, _ in self._adj.get(cid, []):
                in_degree[dst_id] -= 1
                if in_degree[dst_id] == 0:
                    queue.append(dst_id)

        remaining = [cid for cid, deg in in_degree.items() if deg > 0]
        if remaining:
            names = [self._comps[cid].get("name") or cid for cid in remaining]
            raise AlgebraicLoopError(names)

        return order

    def _init_controllers(self) -> None:
        """Instantiate native C++ control objects (or Python dicts as fallback)."""
        for comp_id in self._order:
            comp = self._comps[comp_id]
            ctype = comp.get("type", "")
            params = comp.get("parameters") or {}

            if ctype == "PI_CONTROLLER":
                pi_cls = _NATIVE.get("PIController")
                if pi_cls is not None:
                    try:
                        self._controllers[comp_id] = pi_cls(
                            float(params.get("kp", 1.0)),
                            float(params.get("ki", 0.0)),
                            float(params.get("output_min", -1e9)),
                            float(params.get("output_max", 1e9)),
                        )
                        continue
                    except Exception as exc:
                        log.debug("PIController init failed (%s); Python fallback", exc)
                self._controllers[comp_id] = {"integral": 0.0, "t_prev": -1.0}

            elif ctype == "PID_CONTROLLER":
                pid_cls = _NATIVE.get("PIDController")
                if pid_cls is not None:
                    try:
                        self._controllers[comp_id] = pid_cls(
                            float(params.get("kp", 1.0)),
                            float(params.get("ki", 0.0)),
                            float(params.get("kd", 0.01)),
                            float(params.get("output_min", -1e9)),
                            float(params.get("output_max", 1e9)),
                        )
                        continue
                    except Exception:
                        pass

            elif ctype == "RATE_LIMITER":
                rl_cls = _NATIVE.get("RateLimiter")
                if rl_cls is not None:
                    try:
                        self._controllers[comp_id] = rl_cls(
                            float(params.get("rising_rate", 1e6)),
                            float(params.get("falling_rate", -1e6)),
                        )
                        continue
                    except Exception:
                        pass

            elif ctype == "HYSTERESIS":
                hyst_cls = _NATIVE.get("HysteresisController")
                if hyst_cls is not None:
                    try:
                        self._controllers[comp_id] = hyst_cls(
                            float(params.get("upper_threshold", 0.5)),
                            float(params.get("upper_threshold", 0.5))
                            - float(params.get("lower_threshold", -0.5)),
                            float(params.get("output_high", 1.0)),
                            float(params.get("output_low", 0.0)),
                        )
                        continue
                    except Exception:
                        pass

            elif ctype == "SAMPLE_HOLD":
                sh_cls = _NATIVE.get("SampleHold")
                if sh_cls is not None:
                    try:
                        self._controllers[comp_id] = sh_cls(
                            float(params.get("sample_time", 1e-4))
                        )
                        continue
                    except Exception:
                        pass

            elif ctype == "INTEGRATOR":
                self._controllers[comp_id] = {"integral": 0.0, "t_prev": -1.0}

    def _collect_inputs(self, comp_id: str, comp: dict) -> list[float]:
        pin_values: dict[int, float] = {}
        for src_id, edges in self._adj.items():
            for dst_id, dst_pin_name in edges:
                if dst_id != comp_id:
                    continue
                for pin in comp.get("pins") or []:
                    if pin.get("name") == dst_pin_name:
                        pin_values[int(pin["index"])] = self._state.get(src_id, 0.0)
        if not pin_values:
            return []
        return [v for _, v in sorted(pin_values.items())]

    @staticmethod
    def _pin_name(comp: dict, pin_index: int) -> str:
        for pin in comp.get("pins") or []:
            if int(pin.get("index", -1)) == pin_index:
                return pin.get("name", "")
        return ""


__all__ = ["SignalEvaluator", "AlgebraicLoopError", "SIGNAL_TYPES"]
