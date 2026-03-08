"""Tests for SignalEvaluator – signal-flow closed-loop control block.

These tests live in PulsimCore because SignalEvaluator is part of the pulsim
library and must work without any GUI dependency.
"""

from __future__ import annotations

import pytest

from pulsim.signal_evaluator import AlgebraicLoopError, SignalEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pin(index: int, name: str) -> dict:
    return {"index": index, "name": name, "x": 0.0, "y": 0.0}


def _wire(src_id: str, src_pin: int, dst_id: str, dst_pin: int) -> dict:
    return {
        "start_connection": {"component_id": src_id, "pin_index": src_pin},
        "end_connection": {"component_id": dst_id, "pin_index": dst_pin},
    }


def _circuit(*components, wires=()) -> dict:
    return {
        "components": list(components),
        "wires": list(wires),
        "node_map": {},
        "node_aliases": {},
    }


def _constant(cid: str, value: float) -> dict:
    return {
        "id": cid,
        "name": cid.upper(),
        "type": "CONSTANT",
        "parameters": {"value": value},
        "pins": [_pin(0, "OUT")],
    }


def _gain(cid: str, k: float) -> dict:
    return {
        "id": cid,
        "name": cid.upper(),
        "type": "GAIN",
        "parameters": {"gain": k},
        "pins": [_pin(0, "IN"), _pin(1, "OUT")],
    }


def _limiter(cid: str, lo: float, hi: float) -> dict:
    return {
        "id": cid,
        "name": cid.upper(),
        "type": "LIMITER",
        "parameters": {"lower_limit": lo, "upper_limit": hi},
        "pins": [_pin(0, "IN"), _pin(1, "OUT")],
    }


def _subtractor(cid: str) -> dict:
    return {
        "id": cid,
        "name": cid.upper(),
        "type": "SUBTRACTOR",
        "parameters": {"input_count": 2},
        "pins": [_pin(0, "IN1"), _pin(1, "IN2"), _pin(2, "OUT")],
    }


def _summer(cid: str) -> dict:
    return {
        "id": cid,
        "name": cid.upper(),
        "type": "SUM",
        "parameters": {"input_count": 2, "signs": ["+", "+"]},
        "pins": [_pin(0, "IN1"), _pin(1, "IN2"), _pin(2, "OUT")],
    }


def _pwm(cid: str) -> dict:
    return {
        "id": cid,
        "name": cid.upper(),
        "type": "PWM_GENERATOR",
        "parameters": {"duty_cycle": 0.5, "frequency": 10000},
        "pins": [_pin(0, "OUT"), _pin(1, "DUTY_IN")],
    }


def _probe(cid: str) -> dict:
    return {
        "id": cid,
        "name": cid.upper(),
        "type": "VOLTAGE_PROBE",
        "parameters": {"display_name": "V"},
        "pins": [_pin(0, "OUT")],
    }


def _pi(
    cid: str,
    kp: float = 1.0,
    ki: float = 10.0,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> dict:
    return {
        "id": cid,
        "name": cid.upper(),
        "type": "PI_CONTROLLER",
        "parameters": {
            "kp": kp,
            "ki": ki,
            "output_min": out_min,
            "output_max": out_max,
        },
        "pins": [_pin(0, "IN"), _pin(1, "OUT")],
    }


# ---------------------------------------------------------------------------
# Basic block evaluation
# ---------------------------------------------------------------------------


class TestConstant:
    def test_constant_output(self) -> None:
        cd = _circuit(_constant("c", 3.14))
        ev = SignalEvaluator(cd)
        ev.build()
        assert abs(ev.step(0.0)["c"] - 3.14) < 1e-12

    def test_constant_zero_default(self) -> None:
        comp = {
            "id": "c0",
            "name": "C0",
            "type": "CONSTANT",
            "parameters": {},
            "pins": [_pin(0, "OUT")],
        }
        ev = SignalEvaluator(_circuit(comp))
        ev.build()
        assert ev.step(0.0)["c0"] == 0.0


class TestGain:
    def test_gain_multiplication(self) -> None:
        cd = _circuit(
            _constant("c", 2.0),
            _gain("g", 3.0),
            wires=[_wire("c", 0, "g", 0)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        assert abs(ev.step(0.0)["g"] - 6.0) < 1e-12

    def test_gain_no_input_yields_zero(self) -> None:
        ev = SignalEvaluator(_circuit(_gain("g", 5.0)))
        ev.build()
        assert ev.step(0.0)["g"] == 0.0


class TestLimiter:
    def test_clamp_upper(self) -> None:
        cd = _circuit(
            _constant("c", 2.5),
            _limiter("l", 0.0, 1.0),
            wires=[_wire("c", 0, "l", 0)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        assert abs(ev.step(0.0)["l"] - 1.0) < 1e-12

    def test_clamp_lower(self) -> None:
        cd = _circuit(
            _constant("c", -5.0),
            _limiter("l", -1.0, 1.0),
            wires=[_wire("c", 0, "l", 0)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        assert abs(ev.step(0.0)["l"] - (-1.0)) < 1e-12

    def test_passthrough_within_limits(self) -> None:
        cd = _circuit(
            _constant("c", 0.4),
            _limiter("l", 0.0, 1.0),
            wires=[_wire("c", 0, "l", 0)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        assert abs(ev.step(0.0)["l"] - 0.4) < 1e-12


class TestSubtractor:
    def test_difference(self) -> None:
        cd = _circuit(
            _constant("a", 10.0),
            _constant("b", 4.0),
            _subtractor("sub"),
            wires=[_wire("a", 0, "sub", 0), _wire("b", 0, "sub", 1)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        assert abs(ev.step(0.0)["sub"] - 6.0) < 1e-12


class TestSum:
    def test_addition(self) -> None:
        cd = _circuit(
            _constant("a", 3.0),
            _constant("b", 1.5),
            _summer("s"),
            wires=[_wire("a", 0, "s", 0), _wire("b", 0, "s", 1)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        assert abs(ev.step(0.0)["s"] - 4.5) < 1e-12


# ---------------------------------------------------------------------------
# PWM duty chain
# ---------------------------------------------------------------------------


class TestPWMDuty:
    def test_constant_to_pwm(self) -> None:
        cd = _circuit(
            _constant("c", 0.7),
            _pwm("pwm"),
            wires=[_wire("c", 0, "pwm", 1)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        assert abs(ev.step(0.0)["pwm"] - 0.7) < 1e-12

    def test_pwm_duty_clamped(self) -> None:
        cd = _circuit(
            _constant("c", 1.5),
            _pwm("pwm"),
            wires=[_wire("c", 0, "pwm", 1)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        assert abs(ev.step(0.0)["pwm"] - 1.0) < 1e-12

    def test_chain_constant_gain_limiter_pwm(self) -> None:
        """CONSTANT(0.4) → GAIN(1.5) → LIMITER(0,1) → PWM: duty = 0.6"""
        cd = _circuit(
            _constant("c", 0.4),
            _gain("g", 1.5),
            _limiter("l", 0.0, 1.0),
            _pwm("pwm"),
            wires=[
                _wire("c", 0, "g", 0),
                _wire("g", 1, "l", 0),
                _wire("l", 1, "pwm", 1),
            ],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        assert abs(ev.step(0.001)["pwm"] - 0.6) < 1e-9

    def test_pwm_in_pwm_components(self) -> None:
        cd = _circuit(
            _constant("c", 0.5),
            _pwm("pwm"),
            wires=[_wire("c", 0, "pwm", 1)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        assert "pwm" in ev.pwm_components()

    def test_pwm_without_duty_in_not_in_components(self) -> None:
        """PWM without a DUTY_IN wire must NOT appear in pwm_components."""
        ev = SignalEvaluator(_circuit(_pwm("pwm")))
        ev.build()
        assert "pwm" not in ev.pwm_components()


# ---------------------------------------------------------------------------
# Probe feedback
# ---------------------------------------------------------------------------


class TestProbe:
    def test_update_probe_value(self) -> None:
        cd = _circuit(
            _probe("vp"),
            _gain("g", 2.0),
            wires=[_wire("vp", 0, "g", 0)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        ev.update_probes({"vp": 3.0})
        assert abs(ev.step(0.0)["g"] - 6.0) < 1e-12


# ---------------------------------------------------------------------------
# PI controller (Python fallback – no native binding required for these tests)
# ---------------------------------------------------------------------------


class TestPI:
    def test_pi_proportional_only(self) -> None:
        cd = _circuit(
            _constant("ref", 0.5),
            _pi("pi", kp=0.1, ki=0.0, out_min=0.0, out_max=1.0),
            wires=[_wire("ref", 0, "pi", 0)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        assert abs(ev.step(0.001)["pi"] - 0.05) < 1e-9

    def test_pi_clamps_output(self) -> None:
        cd = _circuit(
            _constant("ref", 100.0),
            _pi("pi", kp=10.0, ki=0.0, out_min=0.0, out_max=1.0),
            wires=[_wire("ref", 0, "pi", 0)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        assert abs(ev.step(0.001)["pi"] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Algebraic loop detection
# ---------------------------------------------------------------------------


class TestAlgebraicLoop:
    def test_direct_cycle_raises(self) -> None:
        cd = _circuit(
            _gain("g1", 1.0),
            _gain("g2", 1.0),
            wires=[_wire("g1", 1, "g2", 0), _wire("g2", 1, "g1", 0)],
        )
        with pytest.raises(AlgebraicLoopError) as exc_info:
            SignalEvaluator(cd).build()
        assert "G1" in str(exc_info.value) or "G2" in str(exc_info.value)

    def test_loop_error_contains_block_names(self) -> None:
        err = AlgebraicLoopError(["GA", "GB"])
        assert "GA" in str(err)
        assert "GB" in str(err)


# ---------------------------------------------------------------------------
# has_signal_blocks
# ---------------------------------------------------------------------------


class TestHasSignalBlocks:
    def test_empty_circuit(self) -> None:
        ev = SignalEvaluator(_circuit())
        ev.build()
        assert not ev.has_signal_blocks()

    def test_with_constant(self) -> None:
        ev = SignalEvaluator(_circuit(_constant("c", 1.0)))
        ev.build()
        assert ev.has_signal_blocks()


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_pi_integral(self) -> None:
        cd = _circuit(
            _constant("ref", 1.0),
            _pi("pi", kp=0.0, ki=1.0, out_min=-100.0, out_max=100.0),
            wires=[_wire("ref", 0, "pi", 0)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        ev.step(0.0)
        ev.step(0.1)
        assert ev.step(0.2)["pi"] != 0.0
        ev.reset()
        # After reset: integral=0, kp=0 → output=0
        assert abs(ev.step(0.0)["pi"]) < 1e-9


# ---------------------------------------------------------------------------
# Standalone usage (no GUI) – verifies the module-level docstring example works
# ---------------------------------------------------------------------------


class TestStandaloneUsage:
    def test_build_circuit_data_programmatically(self) -> None:
        """SignalEvaluator must work from pure Python dicts, no GUI needed."""
        circuit_data = {
            "components": [
                {
                    "id": "ref",
                    "name": "REF",
                    "type": "CONSTANT",
                    "parameters": {"value": 12.0},
                    "pins": [{"index": 0, "name": "OUT", "x": 0, "y": 0}],
                },
                {
                    "id": "lim",
                    "name": "LIM",
                    "type": "LIMITER",
                    "parameters": {"lower_limit": 0.0, "upper_limit": 1.0},
                    "pins": [
                        {"index": 0, "name": "IN", "x": 0, "y": 0},
                        {"index": 1, "name": "OUT", "x": 0, "y": 0},
                    ],
                },
                {
                    "id": "pwm",
                    "name": "PWM1",
                    "type": "PWM_GENERATOR",
                    "parameters": {"frequency": 10000, "duty_cycle": 0.5},
                    "pins": [
                        {"index": 0, "name": "OUT", "x": 0, "y": 0},
                        {"index": 1, "name": "DUTY_IN", "x": 0, "y": 0},
                    ],
                },
            ],
            "wires": [
                {
                    "start_connection": {"component_id": "ref", "pin_index": 0},
                    "end_connection": {"component_id": "lim", "pin_index": 0},
                },
                {
                    "start_connection": {"component_id": "lim", "pin_index": 1},
                    "end_connection": {"component_id": "pwm", "pin_index": 1},
                },
            ],
            "node_map": {},
            "node_aliases": {},
        }
        ev = SignalEvaluator(circuit_data)
        ev.build()
        state = ev.step(0.0)
        # CONSTANT(12.0) → LIMITER(0,1) → clamped to 1.0 → PWM duty = 1.0
        assert abs(state["pwm"] - 1.0) < 1e-12
        assert ev.pwm_components() == {"pwm": "PWM1"}


# ---------------------------------------------------------------------------
# C_BLOCK integration (PythonCBlock path – no compiler required)
# ---------------------------------------------------------------------------


def _cblock(cid: str, fn, n_inputs: int = 1, n_outputs: int = 1) -> dict:
    return {
        "id": cid,
        "name": cid.upper(),
        "type": "C_BLOCK",
        "parameters": {
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "python_fn": fn,
        },
        "pins": [_pin(i, f"IN{i}") for i in range(n_inputs)] + [_pin(n_inputs, "OUT")],
    }


class TestCBlockSignalEvaluator:
    def test_python_fnblock_gain(self) -> None:
        """CONSTANT → C_BLOCK(gain=3) produces correct output."""

        def gain3(ctx, t, dt, inputs):
            return [3.0 * inputs[0]]

        cd = _circuit(
            _constant("c", 4.0),
            _cblock("blk", gain3),
            wires=[_wire("c", 0, "blk", 0)],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        state = ev.step(0.0)
        assert state["blk"] == pytest.approx(12.0)

    def test_python_fnblock_dt_passed(self) -> None:
        """C_BLOCK receives correct dt = t2 - t1."""
        received_dt: list[float] = []

        def capture(ctx, t, dt, inputs):
            received_dt.append(dt)
            return [0.0]

        cd = _circuit(_cblock("blk", capture))
        ev = SignalEvaluator(cd)
        ev.build()
        ev.step(0.0)
        ev.step(1e-4)
        ev.step(3e-4)
        assert received_dt[0] == pytest.approx(0.0)  # first step: dt=0
        assert received_dt[1] == pytest.approx(1e-4)
        assert received_dt[2] == pytest.approx(2e-4)

    def test_python_fnblock_reset_clears_context(self) -> None:
        """reset() clears PythonCBlock context dict."""

        def accumulator(ctx, t, dt, inputs):
            ctx["n"] = ctx.get("n", 0) + 1
            return [float(ctx["n"])]

        cd = _circuit(_cblock("blk", accumulator))
        ev = SignalEvaluator(cd)
        ev.build()
        ev.step(0.0)
        ev.step(1e-4)
        ev.reset()
        state = ev.step(0.0)
        assert state["blk"] == pytest.approx(1.0)

    def test_c_block_in_signal_types(self) -> None:
        from pulsim.signal_evaluator import SIGNAL_TYPES

        assert "C_BLOCK" in SIGNAL_TYPES

    def test_c_block_no_lib_path_outputs_zero(self) -> None:
        """C_BLOCK without implementation raises a configuration error."""
        cd = _circuit(
            {
                "id": "blk",
                "name": "BLK",
                "type": "C_BLOCK",
                "parameters": {"n_inputs": 1, "n_outputs": 1},
                "pins": [_pin(0, "IN"), _pin(1, "OUT")],
            }
        )
        ev = SignalEvaluator(cd)
        with pytest.raises(ValueError):
            ev.build()

    def test_c_block_chained_with_gain(self) -> None:
        """C_BLOCK output → GAIN block chain works end-to-end."""

        def triple(ctx, t, dt, inputs):
            return [3.0 * inputs[0]]

        cd = _circuit(
            _constant("c", 2.0),
            _cblock("cb", triple),
            _gain("g", 2.0),
            wires=[
                _wire("c", 0, "cb", 0),
                _wire("cb", 1, "g", 0),
            ],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        state = ev.step(0.0)
        assert state["g"] == pytest.approx(12.0)  # 2 * 3 * 2 = 12

    # ------------------------------------------------------------------
    # 6.2.2  Algebraic loop containing a C_BLOCK must raise AlgebraicLoopError
    # ------------------------------------------------------------------

    def test_signal_evaluator_cblock_algebraic_loop(self) -> None:
        """C_BLOCK wired to its own input raises AlgebraicLoopError."""
        from pulsim.signal_evaluator import AlgebraicLoopError

        def identity(ctx, t, dt, inputs):
            return inputs

        # Single C_BLOCK whose OUT pin feeds back into its own IN pin
        blk = {
            "id": "loop",
            "name": "LOOP",
            "type": "C_BLOCK",
            "parameters": {"n_inputs": 1, "n_outputs": 1, "python_fn": identity},
            "pins": [_pin(0, "IN0"), _pin(1, "OUT")],
        }
        cd = _circuit(blk, wires=[_wire("loop", 1, "loop", 0)])

        ev = SignalEvaluator(cd)
        with pytest.raises(AlgebraicLoopError):
            ev.build()

    # ------------------------------------------------------------------
    # 6.2.3  Multi-output C_BLOCK feeding a SIGNAL_DEMUX
    # ------------------------------------------------------------------

    def test_signal_evaluator_cblock_multi_output_demux(self) -> None:
        """C_BLOCK with 3 outputs routed through SIGNAL_DEMUX is correct."""

        def split3(ctx, t, dt, inputs):
            x = inputs[0] if inputs else 0.0
            return [x, x * 2.0, x * 3.0]

        # C_BLOCK: 1 input, 3 outputs
        source_blk = _constant("c", 5.0)
        cblock_blk = {
            "id": "cb",
            "name": "CB",
            "type": "C_BLOCK",
            "parameters": {"n_inputs": 1, "n_outputs": 3, "python_fn": split3},
            "pins": [_pin(0, "IN0"), _pin(1, "OUT0"), _pin(2, "OUT1"), _pin(3, "OUT2")],
        }
        demux = {
            "id": "dmx",
            "name": "DMX",
            "type": "SIGNAL_DEMUX",
            "parameters": {},
            "pins": [_pin(0, "IN"), _pin(1, "OUT1"), _pin(2, "OUT2"), _pin(3, "OUT3")],
        }
        g1 = {
            "id": "g1",
            "name": "G1",
            "type": "GAIN",
            "parameters": {"gain": 1.0},
            "pins": [_pin(0, "IN"), _pin(1, "OUT")],
        }
        g2 = {
            "id": "g2",
            "name": "G2",
            "type": "GAIN",
            "parameters": {"gain": 1.0},
            "pins": [_pin(0, "IN"), _pin(1, "OUT")],
        }
        g3 = {
            "id": "g3",
            "name": "G3",
            "type": "GAIN",
            "parameters": {"gain": 1.0},
            "pins": [_pin(0, "IN"), _pin(1, "OUT")],
        }
        cd = _circuit(
            source_blk,
            cblock_blk,
            demux,
            g1,
            g2,
            g3,
            wires=[
                _wire("c", 0, "cb", 0),
                _wire("cb", 1, "dmx", 0),
                _wire("dmx", 1, "g1", 0),
                _wire("dmx", 2, "g2", 0),
                _wire("dmx", 3, "g3", 0),
            ],
        )
        ev = SignalEvaluator(cd)
        ev.build()
        state = ev.step(0.0)
        assert state["cb"] == pytest.approx([5.0, 10.0, 15.0])
        assert state["dmx"] == pytest.approx([5.0, 10.0, 15.0])
        assert state["g1"] == pytest.approx(5.0)
        assert state["g2"] == pytest.approx(10.0)
        assert state["g3"] == pytest.approx(15.0)

    # ------------------------------------------------------------------
    # 6.2.5  Compiled shared-library C_BLOCK (skipped if no compiler)
    # ------------------------------------------------------------------

    def test_signal_evaluator_cblock_compiled_lib(self, tmp_path) -> None:
        """Compile a simple gain block and wire it into SignalEvaluator."""
        from pulsim.cblock import detect_compiler, compile_cblock

        cc = detect_compiler()
        if cc is None:
            pytest.skip("No C compiler available")

        c_src = tmp_path / "gain2.c"
        c_src.write_text(
            '#include "pulsim/v1/cblock_abi.h"\n'
            "PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;\n"
            "PULSIM_CBLOCK_EXPORT int"
            " pulsim_cblock_step(PulsimCBlockCtx* ctx,"
            " double t, double dt,"
            " const double* in, double* out) {\n"
            "    (void)ctx; (void)t; (void)dt;\n"
            "    out[0] = 2.0 * in[0];\n"
            "    return 0;\n"
            "}\n"
        )

        lib_path = compile_cblock(c_src, output_dir=tmp_path, name="gain2", compiler=cc)

        # Construct circuit with lib_path
        blk = {
            "id": "cb",
            "name": "CB",
            "type": "C_BLOCK",
            "parameters": {"n_inputs": 1, "n_outputs": 1, "lib_path": str(lib_path)},
            "pins": [_pin(0, "IN0"), _pin(1, "OUT")],
        }
        cd = _circuit(_constant("src", 7.0), blk, wires=[_wire("src", 0, "cb", 0)])
        ev = SignalEvaluator(cd)
        ev.build()
        state = ev.step(0.0)
        assert state["cb"] == pytest.approx(14.0)

    # ------------------------------------------------------------------
    # 6.2.6  Full closed-loop: CONSTANT → SUBTRACTOR → PI → C_BLOCK → PWM
    # ------------------------------------------------------------------

    def test_signal_evaluator_cblock_in_closed_loop(self) -> None:
        """Full PI + C_BLOCK chain produces a PWM duty in [0, 1]."""

        def unity(ctx, t, dt, inputs):
            return [max(0.0, min(1.0, inputs[0]))]

        setpoint = _constant("ref", 1.0)
        feedback = _constant("fbk", 0.0)
        sub = _subtractor("sub")
        pi = {
            "id": "pi",
            "name": "PI",
            "type": "PI_CONTROLLER",
            "parameters": {"kp": 0.5, "ki": 10.0, "output_min": 0.0, "output_max": 1.0},
            "pins": [_pin(0, "IN"), _pin(1, "OUT")],
        }
        cb = _cblock("cb", unity, n_inputs=1, n_outputs=1)
        pwm = _pwm("pwm")

        cd = _circuit(
            setpoint,
            feedback,
            sub,
            pi,
            cb,
            pwm,
            wires=[
                _wire("ref", 0, "sub", 0),
                _wire("fbk", 0, "sub", 1),
                _wire("sub", 2, "pi", 0),
                _wire("pi", 1, "cb", 0),
                _wire("cb", 1, "pwm", 1),
            ],
        )
        ev = SignalEvaluator(cd)
        ev.build()

        dt = 1e-4
        for i in range(10):
            state = ev.step(i * dt)

        duty = state["pwm"]
        assert 0.0 <= duty <= 1.0
