"""03_pi_controller_closed_loop.py — Custom C PI controller in a closed loop.

Demonstrates:
- A C-block replacing the built-in PI_CONTROLLER block
- Closing a voltage regulation loop:
    CONSTANT(setpoint) → SUBTRACTOR(error) → C_BLOCK(PI) → PWM_GENERATOR
- Comparison with the built-in PI controller to verify numeric equivalence
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "build" / "python"))

from pulsim.cblock import compile_cblock, detect_compiler
from pulsim.signal_evaluator import SignalEvaluator

HERE = Path(__file__).parent
C_SOURCE = HERE / "pi_controller.c"
INCLUDE_DIR = HERE.parents[2] / "core" / "include"

KP, KI = 0.5, 50.0
OUTPUT_MIN, OUTPUT_MAX = 0.0, 1.0
SETPOINT = 0.6


def _pin(idx: int, name: str) -> dict:
    return {"index": idx, "name": name, "x": 0, "y": 0}


def build_c_circuit(lib_path: Path) -> dict:
    """CONSTANT → SUBTRACTOR → C_BLOCK(PI) → PWM."""
    return {
        "components": [
            {"id": "ref", "name": "REF", "type": "CONSTANT",
             "parameters": {"value": SETPOINT},
             "pins": [_pin(0, "OUT")]},
            {"id": "fbk", "name": "FBK", "type": "CONSTANT",
             "parameters": {"value": 0.0},
             "pins": [_pin(0, "OUT")]},
            {"id": "sub", "name": "SUB", "type": "SUBTRACTOR",
             "parameters": {},
             "pins": [_pin(0, "IN1"), _pin(1, "IN2"), _pin(2, "OUT")]},
            {"id": "pi",  "name": "PI",  "type": "C_BLOCK",
             "parameters": {
                 "n_inputs": 1, "n_outputs": 1,
                 "lib_path": str(lib_path),
             },
             "pins": [_pin(0, "IN0"), _pin(1, "OUT")]},
            {"id": "pwm", "name": "PWM", "type": "PWM_GENERATOR",
             "parameters": {"frequency": 10000, "duty_cycle": 0.5},
             "pins": [_pin(0, "OUT"), _pin(1, "DUTY_IN")]},
        ],
        "wires": [
            {"start_connection": {"component_id": "ref", "pin_index": 0},
             "end_connection":   {"component_id": "sub", "pin_index": 0}},
            {"start_connection": {"component_id": "fbk", "pin_index": 0},
             "end_connection":   {"component_id": "sub", "pin_index": 1}},
            {"start_connection": {"component_id": "sub", "pin_index": 2},
             "end_connection":   {"component_id": "pi",  "pin_index": 0}},
            {"start_connection": {"component_id": "pi",  "pin_index": 1},
             "end_connection":   {"component_id": "pwm", "pin_index": 1}},
        ],
        "node_map": {}, "node_aliases": {},
    }


def build_py_circuit() -> dict:
    """Same topology using the built-in PI_CONTROLLER."""
    return {
        "components": [
            {"id": "ref", "name": "REF", "type": "CONSTANT",
             "parameters": {"value": SETPOINT},
             "pins": [_pin(0, "OUT")]},
            {"id": "fbk", "name": "FBK", "type": "CONSTANT",
             "parameters": {"value": 0.0},
             "pins": [_pin(0, "OUT")]},
            {"id": "sub", "name": "SUB", "type": "SUBTRACTOR",
             "parameters": {},
             "pins": [_pin(0, "IN1"), _pin(1, "IN2"), _pin(2, "OUT")]},
            {"id": "pi",  "name": "PI",  "type": "PI_CONTROLLER",
             "parameters": {
                 "kp": KP, "ki": KI,
                 "output_min": OUTPUT_MIN, "output_max": OUTPUT_MAX,
             },
             "pins": [_pin(0, "IN"), _pin(1, "OUT")]},
            {"id": "pwm", "name": "PWM", "type": "PWM_GENERATOR",
             "parameters": {"frequency": 10000, "duty_cycle": 0.5},
             "pins": [_pin(0, "OUT"), _pin(1, "DUTY_IN")]},
        ],
        "wires": [
            {"start_connection": {"component_id": "ref", "pin_index": 0},
             "end_connection":   {"component_id": "sub", "pin_index": 0}},
            {"start_connection": {"component_id": "fbk", "pin_index": 0},
             "end_connection":   {"component_id": "sub", "pin_index": 1}},
            {"start_connection": {"component_id": "sub", "pin_index": 2},
             "end_connection":   {"component_id": "pi",  "pin_index": 0}},
            {"start_connection": {"component_id": "pi",  "pin_index": 1},
             "end_connection":   {"component_id": "pwm", "pin_index": 1}},
        ],
        "node_map": {}, "node_aliases": {},
    }


def run(ev: SignalEvaluator, n: int = 20, dt: float = 1e-4) -> list[float]:
    ev.build()
    duties = []
    for i in range(n):
        state = ev.step(i * dt)
        duties.append(state["pwm"])
    return duties


def main() -> None:
    cc = detect_compiler()
    if cc is None:
        print("No C compiler found. Set PULSIM_CC or install gcc/clang.")
        sys.exit(1)

    print(f"Compiling {C_SOURCE.name} ...")
    lib_path = compile_cblock(
        C_SOURCE,
        name="pi_controller",
        extra_cflags=[f"-I{INCLUDE_DIR}"],
        compiler=cc,
    )

    c_ev  = SignalEvaluator(build_c_circuit(lib_path))
    py_ev = SignalEvaluator(build_py_circuit())

    c_duties  = run(c_ev)
    py_duties = run(py_ev)

    print(f"\nComparison (setpoint={SETPOINT}, kp={KP}, ki={KI}):")
    print(f"{'Step':>5}  {'C PI duty':>12}  {'Built-in PI':>12}  {'Diff':>10}")
    max_diff = 0.0
    for i, (cd, pd) in enumerate(zip(c_duties, py_duties)):
        diff = abs(cd - pd)
        max_diff = max(max_diff, diff)
        if i % 4 == 0:
            print(f"{i:>5}  {cd:>12.6f}  {pd:>12.6f}  {diff:>10.2e}")

    print(f"\nMax discrepancy: {max_diff:.2e}")
    # Duties must be in [0, 1] and PI outputs should start rising from setpoint error
    assert all(0.0 <= d <= 1.0 for d in c_duties), "C PI duty out of [0,1]"
    assert max_diff < 0.01, f"C and built-in PI differ by {max_diff:.4f} > 0.01"
    print("Numeric equivalence verified ✓")


if __name__ == "__main__":
    main()
