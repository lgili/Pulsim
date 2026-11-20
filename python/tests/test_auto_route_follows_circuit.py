"""update-number-changing-release group H: the 'auto' route without dt is
decided on every call from the circuit as it is NOW (no cached route): a
builder mutated between two calls moves the next call off the exact engine.
"""

import warnings

import pytest

import pulsim as p


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    b.add_resistor("R", "a", "b", 1e3)
    b.add_capacitor("C", "b", "gnd", 1e-6)
    return b


def test_a_newton_device_added_between_calls_leaves_the_exact_route():
    b = _rc()
    assert p.simulate(b, 1e-4).engine_used == "dsed"
    b.add_saturable_inductor("Ls", "b", "gnd", 1e-3, 2.0, 1e-5)
    assert p.simulate(b, 1e-4).engine_used == "trbdf2"


def test_a_c_block_added_between_calls_leaves_the_exact_route():
    b = _rc()
    assert p.simulate(b, 1e-4).engine_used == "dsed"

    def step(t, dt, inp, out, state):
        out[0] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        p.add_c_block(b, inputs=[("v", "b")], outputs=[("v", "a", "gnd")],
                      dt=1e-5, fn=step)
    # 'auto' routes a C block to the fixed step, which needs dt
    with pytest.raises(ValueError, match="variable-step engine"):
        p.simulate(b, 1e-4)
