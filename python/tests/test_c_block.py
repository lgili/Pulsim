"""Phase 1 tests for the custom-code block (``add_c_block``): Python step
function on the PWL engine — input wires, output injection (controlled
sources), sample-time / ZOH, multi-IO, and persistent state."""
from __future__ import annotations

import numpy as np
import pulsim as p


def _read(res, node):
    return np.asarray(res.v(node), dtype=float)


def test_voltage_output_drives_node():
    """A block reads V(in)=2 and imposes V(out)=3·V(in)=6 via a
    controlled voltage source."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 2.0)
    b.add_resistor("Rin", "in", "gnd", 1e3)
    b.add_resistor("Rout", "out", "gnd", 1e3)

    def step(t, dt, inp, out, state):
        out[0] = 3.0 * inp[0]

    h = p.add_c_block(b, inputs=[("v", "in")], outputs=[("v", "out", "gnd")],
                      dt=1e-4, fn=step)
    res = p.simulate(b, t_end=2e-3, dt=2e-6)
    v_out = _read(res, "out")[-1]
    assert abs(v_out - 6.0) < 1e-6, f"V(out)={v_out} (expected 6.0)"
    assert h.n_fires > 0


def test_current_output_injects_current():
    """A current-source output forces I into a resistor: V = I·R."""
    b = p.CircuitBuilder()
    b.add_resistor("R", "n", "gnd", 100.0)

    def step(t, dt, inp, out, state):
        out[0] = 0.05            # 50 mA into the node

    p.add_c_block(b, inputs=[], outputs=[("i", "n", "gnd")],
                  dt=1e-4, fn=step)
    res = p.simulate(b, t_end=2e-3, dt=2e-6)
    v_n = _read(res, "n")[-1]
    # |V| = |I|·R = 0.05·100 = 5 V (sign depends on source orientation).
    assert abs(abs(v_n) - 5.0) < 1e-6, f"V(n)={v_n} (expected ±5 V)"


def test_sample_time_zero_order_hold():
    """With dt_block = 25·dt the injected output is piecewise-constant,
    updating only at block boundaries."""
    b = p.CircuitBuilder()
    b.add_resistor("Rout", "out", "gnd", 1e3)
    dt = 2e-6
    dt_block = 50e-6           # = 25·dt

    def step(t, dt_, inp, out, state):
        out[0] = float(t)       # ramp → easy to see the staircase

    p.add_c_block(b, inputs=[], outputs=[("v", "out", "gnd")],
                  dt=dt_block, fn=step)
    res = p.simulate(b, t_end=1e-3, dt=dt)
    v = _read(res, "out")
    # Count distinct held levels; a ramp sampled at 25·dt over 500 steps
    # should produce ~20 plateaus, far fewer than 500 unique values.
    n_changes = int(np.count_nonzero(np.abs(np.diff(v)) > 1e-9))
    assert n_changes < 40, f"expected a coarse staircase, got {n_changes} steps"
    assert n_changes >= 5, f"expected several updates, got {n_changes}"


def test_multi_io_and_state_integrator():
    """2-in / 1-out block with persistent state: integrate (in0 - in1)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Va", "a", "gnd", 1.0)
    b.add_voltage_source("Vb", "b", "gnd", 0.25)
    b.add_resistor("Ra", "a", "gnd", 1e3)
    b.add_resistor("Rb", "b", "gnd", 1e3)
    b.add_resistor("Rout", "out", "gnd", 1e3)
    dt_block = 1e-4

    def step(t, dt, inp, out, state):
        state["acc"] = state.get("acc", 0.0) + (inp[0] - inp[1]) * dt
        out[0] = state["acc"]

    h = p.add_c_block(
        b, inputs=[("v", "a"), ("v", "b")], outputs=[("v", "out", "gnd")],
        dt=dt_block, fn=step)
    t_end = 5e-3
    res = p.simulate(b, t_end=t_end, dt=2e-6)
    v_out = _read(res, "out")[-1]
    # acc ≈ ∫(1 - 0.25) dt = 0.75 · t_end = 0.75 · 5e-3 = 3.75e-3
    assert abs(v_out - 0.75 * t_end) < 0.75 * dt_block * 2, v_out
    assert h.state["acc"] > 0.0


def test_sub_dt_block_warns_and_clamps():
    b = p.CircuitBuilder()
    b.add_resistor("Rout", "out", "gnd", 1e3)

    def step(t, dt, inp, out, state):
        out[0] = 1.0

    import pytest
    with pytest.warns(UserWarning, match="clamping"):
        p.add_c_block(b, inputs=[], outputs=[("v", "out", "gnd")],
                      dt=1e-7, fn=step, sim_dt=2e-6)
