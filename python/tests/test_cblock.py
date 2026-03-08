"""Tests for pulsim.cblock – CBlock, PythonCBlock, compile_cblock utilities.

These tests are deterministic, require no network access, and run headless.
CBlockLibrary tests that require compilation are skipped when no C compiler
is available (POSIX CI) and when ctypes cannot load the resulting library.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from pulsim.cblock import (
    CBlockABIError,
    CBlockCompileError,
    CBlockRuntimeError,
    CBlockLibrary,
    PythonCBlock,
    compile_cblock,
    detect_compiler,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _gain_source(gain: float = 2.0) -> str:
    """Minimal CBlock C source: single-input gain block."""
    return textwrap.dedent(f"""
        #include <stdlib.h>

        int pulsim_cblock_abi_version = 1;

        int pulsim_cblock_step(
            void* ctx, double t, double dt,
            const double* in, double* out)
        {{
            (void)ctx; (void)t; (void)dt;
            out[0] = {gain} * in[0];
            return 0;
        }}
    """)


def _error_step_source() -> str:
    """CBlock that always returns -1 from step."""
    return textwrap.dedent("""
        int pulsim_cblock_abi_version = 1;

        int pulsim_cblock_step(
            void* ctx, double t, double dt,
            const double* in, double* out)
        {
            (void)ctx; (void)t; (void)dt; (void)in; (void)out;
            return -1;
        }
    """)


def _bad_abi_source() -> str:
    """CBlock that declares wrong ABI version."""
    return textwrap.dedent("""
        int pulsim_cblock_abi_version = 99;

        int pulsim_cblock_step(
            void* ctx, double t, double dt,
            const double* in, double* out)
        {
            (void)ctx; (void)t; (void)dt; (void)in; (void)out;
            return 0;
        }
    """)


def _no_step_source() -> str:
    """CBlock with version symbol but no step function."""
    return textwrap.dedent("""
        int pulsim_cblock_abi_version = 1;
    """)


@pytest.fixture()
def cc() -> str:
    """Skip tests that need a C compiler when none is available."""
    found = detect_compiler()
    if found is None:
        pytest.skip("No C compiler available on this machine")
    return found


@pytest.fixture()
def gain_lib(tmp_path: Path, cc: str) -> Path:
    """Compile and return the path to the gain-2 shared library."""
    return compile_cblock(
        _gain_source(2.0), output_dir=tmp_path, name="gain2", compiler=cc
    )


# ---------------------------------------------------------------------------
# detect_compiler
# ---------------------------------------------------------------------------


class TestDetectCompiler:
    def test_returns_string_or_none(self) -> None:
        result = detect_compiler()
        assert result is None or isinstance(result, str)

    def test_pulsim_cc_env_wins(self, tmp_path: Path) -> None:
        # Create a dummy executable
        fake_cc = tmp_path / "mycc"
        fake_cc.touch()
        fake_cc.chmod(0o755)
        with patch.dict(os.environ, {"PULSIM_CC": str(fake_cc)}):
            found = detect_compiler()
        assert found == str(fake_cc)

    def test_empty_pulsim_cc_falls_through(self) -> None:
        with patch.dict(os.environ, {"PULSIM_CC": ""}):
            # Should not crash; may return a system compiler or None
            result = detect_compiler()
            assert result is None or isinstance(result, str)

    def test_nonexistent_pulsim_cc_returns_none(self) -> None:
        with patch.dict(os.environ, {"PULSIM_CC": "/nonexistent/compiler_xyz"}):
            result = detect_compiler()
        # /nonexistent/compiler_xyz is not a real file → falls through
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# compile_cblock
# ---------------------------------------------------------------------------


class TestCompileCBlock:
    def test_compiles_string_source(self, tmp_path: Path, cc: str) -> None:
        out = compile_cblock(
            _gain_source(), output_dir=tmp_path, name="gain_str", compiler=cc
        )
        assert out.is_file()
        assert out.suffix in (".so", ".dylib", ".dll")

    def test_compiles_path_source(self, tmp_path: Path, cc: str) -> None:
        src = tmp_path / "mygain.c"
        src.write_text(_gain_source(), encoding="utf-8")
        out = compile_cblock(src, output_dir=tmp_path, name="gain_path", compiler=cc)
        assert out.is_file()

    def test_output_dir_created(self, tmp_path: Path, cc: str) -> None:
        out_dir = tmp_path / "nested" / "output"
        out = compile_cblock(
            _gain_source(), output_dir=out_dir, name="gain_dir", compiler=cc
        )
        assert out.is_file()
        assert out.parent == out_dir

    def test_default_output_dir_is_temp(self, cc: str) -> None:
        out = compile_cblock(_gain_source(), name="gain_tmp", compiler=cc)
        assert out.is_file()
        # Should be in a temp-like directory
        assert out.parent != Path.cwd()

    def test_bad_source_raises_compile_error(self, tmp_path: Path, cc: str) -> None:
        bad_source = "this is not valid C code {{{{ ??? ;"
        with pytest.raises(CBlockCompileError) as exc_info:
            compile_cblock(bad_source, output_dir=tmp_path, name="bad", compiler=cc)
        err = exc_info.value
        assert err.compiler_path == cc
        assert len(err.stderr_output) > 0

    def test_missing_source_file_raises_compile_error(
        self, tmp_path: Path, cc: str
    ) -> None:
        with pytest.raises(CBlockCompileError):
            compile_cblock(
                tmp_path / "does_not_exist.c", output_dir=tmp_path, compiler=cc
            )

    def test_bad_compiler_name_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CBlockCompileError) as exc_info:
            compile_cblock(
                _gain_source(),
                output_dir=tmp_path,
                compiler="/nonexistent/cc_xyz_12345",
            )
        assert exc_info.value.compiler_path is not None

    def test_extra_cflags_forwarded(self, tmp_path: Path, cc: str) -> None:
        # Adding -DPULSIM_TEST_FLAG should not break compilation
        out = compile_cblock(
            _gain_source(),
            output_dir=tmp_path,
            name="gain_flags",
            extra_cflags=["-DPULSIM_TEST_FLAG=1"],
            compiler=cc,
        )
        assert out.is_file()

    def test_no_compiler_raises_when_none_available(self, tmp_path: Path) -> None:
        with patch("pulsim.cblock.detect_compiler", return_value=None):
            with pytest.raises(CBlockCompileError) as exc_info:
                compile_cblock(_gain_source(), output_dir=tmp_path)
        assert exc_info.value.compiler_path is None

    def test_compile_error_has_source_attribute(self, tmp_path: Path, cc: str) -> None:
        src_code = "not C code"
        with pytest.raises(CBlockCompileError) as exc_info:
            compile_cblock(src_code, output_dir=tmp_path, name="bad2", compiler=cc)
        assert exc_info.value.source == src_code


# ---------------------------------------------------------------------------
# CBlockLibrary
# ---------------------------------------------------------------------------


class TestCBlockLibrary:
    def test_step_basic(self, gain_lib: Path) -> None:
        with CBlockLibrary(gain_lib, n_inputs=1, n_outputs=1) as blk:
            result = blk.step(0.0, 1e-6, [3.0])
        assert result == pytest.approx([6.0])

    def test_step_multiple_calls(self, gain_lib: Path) -> None:
        with CBlockLibrary(gain_lib, n_inputs=1, n_outputs=1) as blk:
            for i in range(10):
                out = blk.step(i * 1e-6, 1e-6, [float(i)])
                assert out == pytest.approx([2.0 * i])

    def test_context_manager(self, gain_lib: Path) -> None:
        with CBlockLibrary(gain_lib, n_inputs=1, n_outputs=1) as blk:
            assert blk.n_inputs == 1
            assert blk.n_outputs == 1
        with pytest.raises(CBlockABIError):
            blk.step(0.0, 1e-6, [1.0])

    def test_properties(self, gain_lib: Path) -> None:
        blk = CBlockLibrary(gain_lib, n_inputs=1, n_outputs=1, name="myblk")
        assert blk.n_inputs == 1
        assert blk.n_outputs == 1
        assert blk.name == "myblk"
        blk._cleanup()

    def test_reset(self, gain_lib: Path) -> None:
        blk = CBlockLibrary(gain_lib, n_inputs=1, n_outputs=1)
        blk.step(0.0, 1e-6, [1.0])
        blk.reset()
        out = blk.step(0.0, 0.0, [5.0])
        assert out == pytest.approx([10.0])
        blk._cleanup()

    def test_bad_abi_version_raises(self, tmp_path: Path, cc: str) -> None:
        lib = compile_cblock(
            _bad_abi_source(), output_dir=tmp_path, name="badabi", compiler=cc
        )
        with pytest.raises(CBlockABIError) as exc_info:
            CBlockLibrary(lib)
        err = exc_info.value
        assert err.expected_version == 1
        assert err.found_version == 99

    def test_missing_step_sym_raises(self, tmp_path: Path, cc: str) -> None:
        lib = compile_cblock(
            _no_step_source(), output_dir=tmp_path, name="nostep", compiler=cc
        )
        with pytest.raises(CBlockABIError):
            CBlockLibrary(lib)

    def test_step_error_raises_runtime_error(self, tmp_path: Path, cc: str) -> None:
        lib = compile_cblock(
            _error_step_source(), output_dir=tmp_path, name="errblk", compiler=cc
        )
        with CBlockLibrary(lib, n_inputs=1, n_outputs=1) as blk:
            with pytest.raises(CBlockRuntimeError) as exc_info:
                blk.step(0.0, 0.0, [1.0])
        assert exc_info.value.return_code == -1
        assert exc_info.value.t == pytest.approx(0.0)

    def test_step_wrong_input_count_raises_value_error(self, gain_lib: Path) -> None:
        with CBlockLibrary(gain_lib, n_inputs=1, n_outputs=1) as blk:
            with pytest.raises(ValueError):
                blk.step(0.0, 0.0, [])

    def test_nonexistent_lib_raises_abi_error(self) -> None:
        with pytest.raises(CBlockABIError):
            CBlockLibrary("/nonexistent/path/libfoo.so")


# ---------------------------------------------------------------------------
# PythonCBlock
# ---------------------------------------------------------------------------


class TestPythonCBlock:
    def test_basic_step(self) -> None:
        def gain(ctx, t, dt, inputs):
            return [2.0 * inputs[0]]

        blk = PythonCBlock(gain, n_inputs=1, n_outputs=1)
        assert blk.step(0.0, 1e-6, [3.0]) == pytest.approx([6.0])

    def test_context_persists_across_steps(self) -> None:
        def accumulator(ctx, t, dt, inputs):
            ctx["total"] = ctx.get("total", 0.0) + inputs[0]
            return [ctx["total"]]

        blk = PythonCBlock(accumulator, n_inputs=1, n_outputs=1)
        blk.step(0.0, 1e-6, [1.0])
        blk.step(1e-6, 1e-6, [2.0])
        out = blk.step(2e-6, 1e-6, [3.0])
        assert out == pytest.approx([6.0])

    def test_reset_clears_context(self) -> None:
        def accumulator(ctx, t, dt, inputs):
            ctx["total"] = ctx.get("total", 0.0) + inputs[0]
            return [ctx["total"]]

        blk = PythonCBlock(accumulator, n_inputs=1, n_outputs=1)
        blk.step(0.0, 1e-6, [5.0])
        blk.reset()
        out = blk.step(0.0, 0.0, [1.0])
        assert out == pytest.approx([1.0])

    def test_properties(self) -> None:
        blk = PythonCBlock(
            lambda ctx, t, dt, inp: [0.0], n_inputs=3, n_outputs=2, name="testblk"
        )
        assert blk.n_inputs == 3
        assert blk.n_outputs == 2
        assert blk.name == "testblk"

    def test_dt_passed_correctly(self) -> None:
        received: list[float] = []

        def capture_dt(ctx, t, dt, inputs):
            received.append(dt)
            return [0.0]

        blk = PythonCBlock(capture_dt, n_inputs=1, n_outputs=1)
        blk.step(0.0, 0.0, [0.0])
        blk.step(1e-4, 1e-4, [0.0])
        assert received[0] == pytest.approx(0.0)
        assert received[1] == pytest.approx(1e-4)

    def test_multi_output(self) -> None:
        def split(ctx, t, dt, inputs):
            v = inputs[0]
            return [v, -v]

        blk = PythonCBlock(split, n_inputs=1, n_outputs=2)
        out = blk.step(0.0, 0.0, [3.0])
        assert out == pytest.approx([3.0, -3.0])

    def test_step_index_increments(self) -> None:
        blk = PythonCBlock(lambda ctx, t, dt, inp: [0.0])
        assert blk._step_index == 0
        blk.step(0.0, 0.0, [0.0])
        assert blk._step_index == 1
        blk.step(1e-4, 1e-4, [0.0])
        assert blk._step_index == 2
        blk.reset()
        assert blk._step_index == 0

    def test_wrong_input_count_raises(self) -> None:
        blk = PythonCBlock(lambda ctx, t, dt, inp: [0.0], n_inputs=2, n_outputs=1)
        with pytest.raises(ValueError):
            blk.step(0.0, 0.0, [1.0])

    def test_wrong_output_count_raises(self) -> None:
        blk = PythonCBlock(lambda ctx, t, dt, inp: [1.0, 2.0], n_inputs=1, n_outputs=1)
        with pytest.raises(ValueError):
            blk.step(0.0, 0.0, [1.0])


# ---------------------------------------------------------------------------
# Exception attributes
# ---------------------------------------------------------------------------


class TestExceptionAttributes:
    def test_compile_error_attributes(self) -> None:
        err = CBlockCompileError(
            "msg",
            source="int x;",
            stderr_output="error: ...",
            compiler_path="/usr/bin/gcc",
        )
        assert err.source == "int x;"
        assert err.stderr_output == "error: ..."
        assert err.compiler_path == "/usr/bin/gcc"
        assert str(err) == "msg"

    def test_abi_error_attributes(self) -> None:
        err = CBlockABIError("abi mismatch", expected_version=1, found_version=99)
        assert err.expected_version == 1
        assert err.found_version == 99

    def test_runtime_error_attributes(self) -> None:
        err = CBlockRuntimeError("rt error", return_code=-2, t=1.5e-3, step_index=42)
        assert err.return_code == -2
        assert err.t == pytest.approx(1.5e-3)
        assert err.step_index == 42
