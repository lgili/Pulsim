"""Tests for `pulsim.fast_block` — the PSIM/PLECS-style JIT control
block helper.

Numba is an optional dependency (``pulsim[fast]``); every test that
requires the JIT path is gated on import availability. The
no-numba-installed case is itself a tested code path
(:func:`is_available` returns ``False``, importing the decorator at
runtime raises an actionable ``ImportError``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import sys

import pulsim as p
# Top-level `pulsim.fast_block` is the decorator (re-exported);
# the submodule object lives at sys.modules['pulsim.fast_block']
# after `import pulsim.fast_block` triggered by the top-level
# re-export. Grab it for `is_available` + the `_NUMBA` cache
# `monkeypatch` needs to flip.
_fb = sys.modules["pulsim.fast_block"]


pytestmark = pytest.mark.skipif(
    not _fb.is_available(),
    reason="numba not installed — install via `pip install pulsim[fast]`")


def test_fast_block_decorator_returns_FastBlock() -> None:
    """Bare-decorator form must yield a `FastBlock` instance whose
    callable forwards to the JIT-compiled function."""
    @p.fast_block
    def gain(x, state):
        return 2.0 * x

    assert isinstance(gain, p.FastBlock)
    assert gain(3.0, np.zeros(1)) == pytest.approx(6.0)
    # py_func gives the un-JIT'd original — handy for debugging.
    assert gain.py_func(3.0, np.zeros(1)) == pytest.approx(6.0)
    # Wrapper preserves the original name.
    assert gain.__name__ == "gain"


def test_fast_block_parametrised_decorator() -> None:
    """``@fast_block(n_states=N)`` sizes ``make_state()`` correctly."""
    @p.fast_block(n_states=3)
    def f(x, state):
        return state[0] + state[1] + state[2] + x

    s = f.make_state()
    assert s.shape == (3,)
    assert s.dtype == np.float64
    assert np.all(s == 0.0)
    assert f(1.0, s) == pytest.approx(1.0)


def test_fast_block_pi_controller_matches_pure_python() -> None:
    """A canonical PI controller compiled with `fast_block` must
    return bit-identical results to the same control law written
    in pure Python (modulo Numba's float-mode normalisation, which
    keeps doubles exact for this expression)."""
    KP, KI = 0.5, 100.0
    DT = 1e-4

    @p.fast_block
    def pi_jit(err, dt, Kp, Ki, state):
        state[0] += Ki * dt * err
        return Kp * err + state[0]

    def pi_py(err, dt, Kp, Ki, state):
        state[0] += Ki * dt * err
        return Kp * err + state[0]

    s_jit = pi_jit.make_state()
    s_py = np.zeros(1)
    errs = [1.0, 0.5, -0.2, 0.0, -1.0, 0.3]
    for e in errs:
        u_jit = pi_jit(e, DT, KP, KI, s_jit)
        u_py = pi_py(e, DT, KP, KI, s_py)
        assert math.isclose(u_jit, u_py, rel_tol=1e-12), (u_jit, u_py)
    assert math.isclose(float(s_jit[0]), float(s_py[0]), rel_tol=1e-12)


def test_fast_block_warm_up_smoke() -> None:
    """``warm_up()`` must trigger JIT specialisation without
    blowing up — the second timed call after warm-up should be
    measurably faster than a cold first call, but we don't pin
    timing in CI; just verify the method completes."""
    @p.fast_block
    def f(x, scale, state):
        state[0] += 0.1 * x
        return scale * state[0]

    f.warm_up()  # heuristic dispatch — no args supplied
    s = f.make_state()
    out = f(1.5, 2.0, s)
    assert math.isfinite(out)


def test_fast_block_warm_up_with_explicit_args() -> None:
    @p.fast_block
    def f(a, b, state):
        return a + b + state[0]

    f.warm_up(0.0, 0.0, np.zeros(1))  # explicit signature
    s = f.make_state()
    assert f(1.0, 2.0, s) == pytest.approx(3.0)


def test_fast_block_inside_step_observer() -> None:
    """End-to-end smoke: use a JIT-compiled PI inside a real
    `simulate()` step observer. Verifies the Python → Numba boundary
    works under the kernel callback path."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 5.0)
    b.add_resistor("R1", "vin", "mid", 1.0)
    b.add_resistor("R_load", "mid", "gnd", 4.0)

    @p.fast_block
    def pi(err, dt, Kp, Ki, state):
        state[0] += Ki * dt * err
        return Kp * err + state[0]

    state = pi.make_state()
    log: list[float] = []
    DT = 1e-5
    SETPOINT = 4.0

    def observer(t, x):
        # Read v_mid via the state vector.
        v_mid = x[b.node_id_of("mid")]
        u = pi(SETPOINT - v_mid, DT, 0.5, 100.0, state)
        log.append(float(u))

    res = p.simulate(b, t_end=2e-3, dt=DT, step_observer=observer)
    assert res.num_steps() > 0
    # State has accumulated, not just zero.
    assert state[0] != 0.0
    # Observer was called every step.
    assert len(log) == res.num_steps()


def test_fast_block_import_error_when_numba_missing(monkeypatch) -> None:
    """When numba is unavailable, calling the decorator must raise
    ``ImportError`` with an actionable message (pip install hint).
    Simulates a numba-less environment by patching the cached probe."""
    monkeypatch.setattr(_fb, "_NUMBA", False, raising=False)

    with pytest.raises(ImportError, match="pip install pulsim"):
        @p.fast_block
        def _f(x, state):
            return x

    # Reset for any subsequent tests in the same run.
    monkeypatch.setattr(_fb, "_NUMBA", None, raising=False)


def test_is_available_returns_bool() -> None:
    """`is_available` must return a plain bool either way — never
    a sentinel."""
    val = _fb.is_available()
    assert isinstance(val, bool)
