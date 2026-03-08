"""Example 03 — C_BLOCK inside a SignalEvaluator pipeline (Python fn).

Shows how to embed a PythonCBlock inside a SignalEvaluator DAG so that the
block participates in closed-loop control without needing a C compiler.

Pipeline::

    CONSTANT(5.0) ──► C_BLOCK(gain=3) ──► LIMITER(0, 10) ──► output

Run::

    PYTHONPATH=python python examples/cblock/03_signal_evaluator_python_fn.py
"""

from pulsim.signal_evaluator import SignalEvaluator


def gain3(ctx, t, dt, inputs):
    return [3.0 * inputs[0]]


circuit_data = {
    "components": [
        {
            "id": "src",
            "name": "SRC",
            "type": "CONSTANT",
            "parameters": {"value": 5.0},
            "pins": [{"index": 0, "name": "OUT", "x": 0, "y": 0}],
        },
        {
            "id": "cb",
            "name": "CB",
            "type": "C_BLOCK",
            "parameters": {"n_inputs": 1, "n_outputs": 1, "python_fn": gain3},
            "pins": [
                {"index": 0, "name": "IN",  "x": 0, "y": 0},
                {"index": 1, "name": "OUT", "x": 0, "y": 0},
            ],
        },
        {
            "id": "lim",
            "name": "LIM",
            "type": "LIMITER",
            "parameters": {"lower_limit": 0.0, "upper_limit": 10.0},
            "pins": [
                {"index": 0, "name": "IN",  "x": 0, "y": 0},
                {"index": 1, "name": "OUT", "x": 0, "y": 0},
            ],
        },
    ],
    "wires": [
        {
            "start_connection": {"component_id": "src", "pin_index": 0},
            "end_connection":   {"component_id": "cb",  "pin_index": 0},
        },
        {
            "start_connection": {"component_id": "cb",  "pin_index": 1},
            "end_connection":   {"component_id": "lim", "pin_index": 0},
        },
    ],
    "node_map": {},
    "node_aliases": {},
}

ev = SignalEvaluator(circuit_data)
ev.build()

state = ev.step(0.0)
print(f"src  = {state['src']:.1f}")   # 5.0
print(f"cb   = {state['cb']:.1f}")    # 15.0 (5 * 3)
print(f"lim  = {state['lim']:.1f}")   # 10.0 (clamped)
assert state["cb"] == 15.0
assert state["lim"] == 10.0
print("OK")
