"""CBlock — Custom C and Python computation blocks for Pulsim simulations.

This module provides two concrete block implementations:

* :class:`CBlockLibrary` — loads a compiled shared library (``.so`` / ``.dylib``
  / ``.dll``) that exports the CBlock ABI symbols defined in
  ``core/include/pulsim/v1/cblock_abi.h``.

* :class:`PythonCBlock` — wraps a plain Python callable so it can be used in a
  :class:`~pulsim.signal_evaluator.SignalEvaluator` without compilation.

Helper utilities:

* :func:`detect_compiler` — finds a suitable C compiler on the current machine.
* :func:`compile_cblock` — compiles a ``.c`` source file into a shared library.

Basic usage::

    from pulsim.cblock import compile_cblock, CBlockLibrary

    source = r'''
    #include "pulsim/v1/cblock_abi.h"

    PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;

    PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
        PulsimCBlockCtx* ctx, double t, double dt,
        const double* in, double* out)
    {
        (void)ctx; (void)t; (void)dt;
        out[0] = 2.0 * in[0];
        return 0;
    }
    '''

    lib_path = compile_cblock(source, name="gain2")
    with CBlockLibrary(lib_path, n_inputs=1, n_outputs=1) as blk:
        result = blk.step(0.0, 1e-6, [3.0])
        assert result == [6.0]
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

__all__ = [
    # Exceptions
    "CBlockCompileError",
    "CBlockABIError",
    "CBlockRuntimeError",
    # Utilities
    "detect_compiler",
    "compile_cblock",
    # Block types
    "CBlockLibrary",
    "PythonCBlock",
]

# ---------------------------------------------------------------------------
# ABI constants (must match cblock_abi.h)
# ---------------------------------------------------------------------------

_ABI_VERSION_EXPECTED: int = 1
_SYM_VERSION: str = "pulsim_cblock_abi_version"
_SYM_INIT: str = "pulsim_cblock_init"
_SYM_STEP: str = "pulsim_cblock_step"
_SYM_DESTROY: str = "pulsim_cblock_destroy"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CBlockCompileError(RuntimeError):
    """Raised when compilation of a C source file fails.

    Attributes
    ----------
    source : str
        The C source code that was compiled (or the path, if a Path was given).
    stderr_output : str
        Captured stderr from the compiler invocation.
    compiler_path : str or None
        Absolute path to the compiler binary that was used, or ``None`` if the
        compiler could not be located before the attempt.
    """

    def __init__(
        self,
        message: str,
        *,
        source: str = "",
        stderr_output: str = "",
        compiler_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.source: str = source
        self.stderr_output: str = stderr_output
        self.compiler_path: str | None = compiler_path


class CBlockABIError(RuntimeError):
    """Raised when the loaded library exports an incompatible ABI version.

    Attributes
    ----------
    expected_version : int
        The ABI version that Pulsim expects (``PULSIM_CBLOCK_ABI_VERSION``).
    found_version : int or None
        The version value read from the library, or ``None`` if the version
        symbol was absent.
    """

    def __init__(
        self,
        message: str,
        *,
        expected_version: int = _ABI_VERSION_EXPECTED,
        found_version: int | None = None,
    ) -> None:
        super().__init__(message)
        self.expected_version: int = expected_version
        self.found_version: int | None = found_version


class CBlockRuntimeError(RuntimeError):
    """Raised when a CBlock's ``step`` function returns a nonzero code.

    Attributes
    ----------
    return_code : int
        The nonzero value returned by the C step function.
    t : float
        Simulation time [s] at which the error occurred.
    step_index : int
        Zero-based index of the simulation step.
    """

    def __init__(
        self,
        message: str,
        *,
        return_code: int = -1,
        t: float = 0.0,
        step_index: int = 0,
    ) -> None:
        super().__init__(message)
        self.return_code: int = return_code
        self.t: float = t
        self.step_index: int = step_index


# ---------------------------------------------------------------------------
# Compiler detection
# ---------------------------------------------------------------------------


def detect_compiler() -> str | None:
    """Attempt to locate a suitable C compiler on the current machine.

    Resolution order:

    1. ``PULSIM_CC`` environment variable (any platform).
    2. POSIX: ``cc`` → ``gcc`` → ``clang``.
    3. Windows: ``cl`` → ``gcc`` → ``clang``.

    Returns
    -------
    str or None
        Absolute path to a compiler executable, or ``None`` if none was found.
    """
    env_cc = os.environ.get("PULSIM_CC", "").strip()
    if env_cc:
        resolved = shutil.which(env_cc) or (env_cc if os.path.isfile(env_cc) else None)
        if resolved:
            return resolved

    if sys.platform == "win32":
        candidates = ["cl", "gcc", "clang"]
    else:
        candidates = ["cc", "gcc", "clang"]

    for name in candidates:
        found = shutil.which(name)
        if found:
            return found

    return None


# ---------------------------------------------------------------------------
# Compilation helper
# ---------------------------------------------------------------------------


def compile_cblock(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    name: str = "cblock",
    extra_cflags: list[str] | None = None,
    compiler: str | None = None,
) -> Path:
    """Compile a C source file into a CBlock shared library.

    Parameters
    ----------
    source : str or Path
        Either the C source **code** as a string, or a :class:`~pathlib.Path`
        pointing to an existing ``.c`` file.
    output_dir : str or Path or None
        Directory for the output ``.so`` / ``.dylib`` / ``.dll``.
        Defaults to a temporary directory (cleaned up by the OS on reboot;
        the caller owns the lifetime of the file).
    name : str
        Base name for the output library (no extension).
    extra_cflags : list[str] or None
        Additional flags forwarded verbatim to the compiler.
    compiler : str or None
        Explicit path (or name) of the C compiler to use.
        If ``None``, :func:`detect_compiler` is called automatically.

    Returns
    -------
    pathlib.Path
        Absolute path to the compiled shared library.

    Raises
    ------
    CBlockCompileError
        When compilation fails or no compiler can be found.
    """
    if extra_cflags is None:
        extra_cflags = []

    # --- Resolve compiler ---------------------------------------------------
    if compiler is not None:
        cc = shutil.which(compiler) or (compiler if os.path.isfile(compiler) else None)
        if cc is None:
            raise CBlockCompileError(
                f"Specified compiler not found: {compiler!r}",
                source=str(source),
                compiler_path=compiler,
            )
    else:
        cc = detect_compiler()
        if cc is None:
            raise CBlockCompileError(
                "No C compiler found. Install gcc/clang or set PULSIM_CC.",
                source=str(source),
                compiler_path=None,
            )

    # --- Resolve include path for cblock_abi.h ------------------------------
    # We only add include directories that actually contain
    # pulsim/v1/cblock_abi.h. This keeps behavior deterministic across source
    # tree, build tree, and installed package layouts.
    abi_rel = Path("pulsim") / "v1" / "cblock_abi.h"
    include_dirs: list[Path] = []
    seen_dirs: set[str] = set()

    def _add_include_dir(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen_dirs:
            return
        if (resolved / abi_rel).is_file():
            include_dirs.append(resolved)
            seen_dirs.add(key)

    env_include = os.environ.get("PULSIM_CBLOCK_INCLUDE", "").strip()
    if env_include:
        for item in env_include.split(os.pathsep):
            candidate = Path(item).expanduser()
            if candidate.exists():
                _add_include_dir(candidate)

    module_path = Path(__file__).resolve()
    for parent in [module_path.parent, *module_path.parents]:
        _add_include_dir(parent / "core" / "include")
        _add_include_dir(parent / "include")

    sys_include = sysconfig.get_paths().get("include")
    if sys_include:
        _add_include_dir(Path(sys_include))
    _add_include_dir(Path(sys.prefix) / "include")
    _add_include_dir(Path(sys.base_prefix) / "include")
    _add_include_dir(Path(sys.prefix) / "include" / "pulsim")
    _add_include_dir(Path(sys.base_prefix) / "include" / "pulsim")

    include_flags: list[str]
    if Path(cc).name.lower() in ("cl", "cl.exe"):
        include_flags = [f"/I{inc}" for inc in include_dirs]
    else:
        include_flags = [f"-I{inc}" for inc in include_dirs]

    # --- Prepare source file ------------------------------------------------
    _tmp_dir: tempfile.TemporaryDirectory[str] | None = None

    if isinstance(source, str):
        _tmp_dir = tempfile.TemporaryDirectory(prefix="pulsim_cblock_")
        src_path = Path(_tmp_dir.name) / f"{name}.c"
        src_path.write_text(source, encoding="utf-8")
    else:
        src_path = Path(source).resolve()
        if not src_path.is_file():
            raise CBlockCompileError(
                f"Source file not found: {src_path}",
                source=str(source),
                compiler_path=cc,
            )

    # --- Prepare output path ------------------------------------------------
    if output_dir is None:
        # Always use mkdtemp for output so the directory survives the function
        # call (TemporaryDirectory would be GC'd and deleted on return).
        out_dir = Path(tempfile.mkdtemp(prefix="pulsim_cblock_out_"))
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        lib_name = f"{name}.dll"
    elif sys.platform == "darwin":
        lib_name = f"lib{name}.dylib"
    else:
        lib_name = f"lib{name}.so"

    out_path = out_dir / lib_name

    # --- Build compiler command ---------------------------------------------
    is_msvc = Path(cc).name.lower() in ("cl", "cl.exe")

    if is_msvc:
        cmd = [
            cc,
            "/nologo",
            "/LD",  # build DLL
            "/O2",
            "/std:c11",
            "/DPULSIM_BUILDING_CBLOCK",
            *include_flags,
            *extra_cflags,
            str(src_path),
            f"/Fe{out_path}",  # output name
        ]
    else:
        if sys.platform == "darwin":
            shared_flag = "-dynamiclib"
        else:
            shared_flag = "-shared"

        cmd = [
            cc,
            "-O2",
            shared_flag,
            "-fPIC",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-DPULSIM_BUILDING_CBLOCK",
            *include_flags,
            *extra_cflags,
            str(src_path),
            "-o",
            str(out_path),
        ]

    # --- Run compiler -------------------------------------------------------
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise CBlockCompileError(
            f"Compiler executable not found: {cc}",
            source=str(source),
            stderr_output="",
            compiler_path=cc,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CBlockCompileError(
            "Compiler timed out after 120 s",
            source=str(source),
            stderr_output="",
            compiler_path=cc,
        ) from exc
    finally:
        if _tmp_dir is not None and output_dir is not None:
            # Keep temp source dir alive until after we check the output
            pass

    if proc.returncode != 0:
        raise CBlockCompileError(
            f"Compilation failed (exit {proc.returncode}): {cc}",
            source=str(source),
            stderr_output=proc.stderr or proc.stdout,
            compiler_path=cc,
        )

    if not out_path.is_file():
        raise CBlockCompileError(
            f"Compilation succeeded but output not found: {out_path}",
            source=str(source),
            stderr_output=proc.stderr,
            compiler_path=cc,
        )

    return out_path.resolve()


# ---------------------------------------------------------------------------
# CBlock ABI ctypes function prototypes
# ---------------------------------------------------------------------------

# void* ctx_ptr (opaque)
_CtxPtr = ctypes.c_void_p


# PulsimCBlockInfo struct
class _CBlockInfo(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("n_inputs", ctypes.c_int),
        ("n_outputs", ctypes.c_int),
        ("name", ctypes.c_char_p),
    ]


# ---------------------------------------------------------------------------
# CBlockLibrary
# ---------------------------------------------------------------------------


class CBlockLibrary:
    """Loaded CBlock shared library.

    Wraps a ``.so`` / ``.dylib`` / ``.dll`` that exports the Pulsim CBlock
    ABI.  Use as a context manager to guarantee cleanup::

        with CBlockLibrary(lib_path, n_inputs=1, n_outputs=1) as blk:
            out = blk.step(0.0, 1e-6, [1.0])

    Parameters
    ----------
    lib_path : str or Path
        Path to the shared library.
    n_inputs : int
        Number of scalar input channels expected by the library.
    n_outputs : int
        Number of scalar output channels produced by the library.
    name : str
        Optional block name forwarded to the library's ``init`` function.

    Raises
    ------
    CBlockABIError
        When the library's ABI version does not match ``PULSIM_CBLOCK_ABI_VERSION``
        or the required ``pulsim_cblock_step`` symbol is missing.
    """

    def __init__(
        self,
        lib_path: str | Path,
        n_inputs: int = 1,
        n_outputs: int = 1,
        name: str = "",
    ) -> None:
        self._lib_path = Path(lib_path).resolve()
        self._n_inputs = int(n_inputs)
        self._n_outputs = int(n_outputs)
        self._name = name
        self._ctx_ptr: ctypes.c_void_p | None = None
        self._step_index: int = 0
        self._lib: ctypes.CDLL | None = None

        self._fn_init: Any = None
        self._fn_step: Any = None
        self._fn_destroy: Any = None

        self._load()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, t: float, dt: float, inputs: Sequence[float]) -> list[float]:
        """Evaluate one simulation step.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        dt : float
            Elapsed time since the previous step [s]; 0 at the first step.
        inputs : list[float]
            Input signal values; length must equal :attr:`n_inputs`.

        Returns
        -------
        list[float]
            Output signal values; length equals :attr:`n_outputs`.

        Raises
        ------
        CBlockRuntimeError
            When the C step function returns nonzero.
        """
        if self._lib is None or self._fn_step is None:
            raise CBlockABIError(
                "CBlockLibrary.step called but library is not loaded",
                expected_version=_ABI_VERSION_EXPECTED,
            )

        if len(inputs) != self._n_inputs:
            raise ValueError(
                f"CBlock expected {self._n_inputs} inputs, got {len(inputs)}"
            )

        in_arr = (ctypes.c_double * self._n_inputs)(*list(inputs))
        out_arr = (ctypes.c_double * self._n_outputs)()

        rc = self._fn_step(
            self._ctx_ptr, ctypes.c_double(t), ctypes.c_double(dt), in_arr, out_arr
        )
        if rc != 0:
            raise CBlockRuntimeError(
                f"CBlock step returned {rc} at t={t:.6g}",
                return_code=rc,
                t=t,
                step_index=self._step_index,
            )

        self._step_index += 1
        return list(out_arr)

    def reset(self) -> None:
        """Re-initialise the block by calling destroy + init."""
        self._call_destroy()
        self._call_init()
        self._step_index = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CBlockLibrary":
        return self

    def __exit__(self, *_args: Any) -> None:
        self._cleanup()

    def __del__(self) -> None:
        try:
            self._cleanup()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            lib = ctypes.CDLL(str(self._lib_path))
        except OSError as exc:
            raise CBlockABIError(
                f"Cannot load shared library: {self._lib_path}\n{exc}",
                expected_version=_ABI_VERSION_EXPECTED,
                found_version=None,
            ) from exc

        self._lib = lib

        # --- Validate ABI version -------------------------------------------
        try:
            ver_obj = ctypes.c_int.in_dll(lib, _SYM_VERSION)
            found_version: int | None = int(ver_obj.value)
        except (AttributeError, ValueError):
            found_version = None

        if found_version != _ABI_VERSION_EXPECTED:
            raise CBlockABIError(
                f"ABI version mismatch: expected {_ABI_VERSION_EXPECTED}, "
                f"got {found_version!r} in {self._lib_path.name}",
                expected_version=_ABI_VERSION_EXPECTED,
                found_version=found_version,
            )

        # --- Resolve step (required) ----------------------------------------
        try:
            fn_step = lib[_SYM_STEP]
        except AttributeError as exc:
            raise CBlockABIError(
                f"Required symbol '{_SYM_STEP}' not found in {self._lib_path.name}",
                expected_version=_ABI_VERSION_EXPECTED,
                found_version=found_version,
            ) from exc

        fn_step.restype = ctypes.c_int
        fn_step.argtypes = [
            _CtxPtr,  # ctx
            ctypes.c_double,  # t
            ctypes.c_double,  # dt
            ctypes.POINTER(ctypes.c_double),  # in
            ctypes.POINTER(ctypes.c_double),  # out
        ]
        self._fn_step = fn_step

        # --- Resolve init (optional) ----------------------------------------
        try:
            fn_init = lib[_SYM_INIT]
            fn_init.restype = ctypes.c_int
            fn_init.argtypes = [
                ctypes.POINTER(_CtxPtr),
                ctypes.POINTER(_CBlockInfo),
            ]
            self._fn_init = fn_init
        except AttributeError:
            self._fn_init = None

        # --- Resolve destroy (optional) -------------------------------------
        try:
            fn_destroy = lib[_SYM_DESTROY]
            fn_destroy.restype = None
            fn_destroy.argtypes = [_CtxPtr]
            self._fn_destroy = fn_destroy
        except AttributeError:
            self._fn_destroy = None

        # --- Call init ------------------------------------------------------
        self._call_init()

    def _call_init(self) -> None:
        if self._fn_init is None:
            self._ctx_ptr = None
            return

        info = _CBlockInfo(
            abi_version=_ABI_VERSION_EXPECTED,
            n_inputs=self._n_inputs,
            n_outputs=self._n_outputs,
            name=self._name.encode("utf-8") if self._name else None,
        )
        ctx_holder = _CtxPtr(None)
        rc = self._fn_init(ctypes.byref(ctx_holder), ctypes.byref(info))
        if rc != 0:
            raise CBlockRuntimeError(
                f"CBlock init returned {rc}",
                return_code=rc,
                t=0.0,
                step_index=0,
            )
        self._ctx_ptr = ctx_holder

    def _call_destroy(self) -> None:
        if self._fn_destroy is not None and self._ctx_ptr is not None:
            self._fn_destroy(self._ctx_ptr)
        self._ctx_ptr = None

    def _cleanup(self) -> None:
        self._call_destroy()
        self._fn_init = None
        self._fn_step = None
        self._fn_destroy = None
        self._lib = None


# ---------------------------------------------------------------------------
# PythonCBlock
# ---------------------------------------------------------------------------


class PythonCBlock:
    """Python callable wrapped as a CBlock-compatible block.

    Useful for prototyping control algorithms without a C compiler, or for
    testing the signal evaluator pipeline.

    The callable receives ``(ctx, t, dt, inputs)`` where:

    * ``ctx`` – a mutable ``dict`` (persists across steps; reset by
      :meth:`reset`).
    * ``t`` – simulation time [s].
    * ``dt`` – elapsed time since the previous step [s]; 0 at the first step.
    * ``inputs`` – ``list[float]`` of length :attr:`n_inputs`.

    It must return a ``list[float]`` (or any iterable) of length
    :attr:`n_outputs`.

    Example::

        def my_gain(ctx, t, dt, inputs):
            return [2.0 * inputs[0]]

        blk = PythonCBlock(my_gain, n_inputs=1, n_outputs=1, name="gain2")
        out = blk.step(0.0, 1e-6, [3.0])
        assert out == [6.0]
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        n_inputs: int = 1,
        n_outputs: int = 1,
        name: str = "",
    ) -> None:
        self._fn = fn
        self._n_inputs = int(n_inputs)
        self._n_outputs = int(n_outputs)
        self._name = name
        self._ctx: dict[str, Any] = {}
        self._step_index: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, t: float, dt: float, inputs: Sequence[float]) -> list[float]:
        """Evaluate one simulation step via the Python callable.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        dt : float
            Elapsed time since the previous step [s].
        inputs : list[float]
            Input signal values.

        Returns
        -------
        list[float]
            Output signal values; length equals :attr:`n_outputs`.
        """
        if len(inputs) != self._n_inputs:
            raise ValueError(
                f"PythonCBlock expected {self._n_inputs} inputs, got {len(inputs)}"
            )

        result = self._fn(self._ctx, float(t), float(dt), list(inputs))
        if isinstance(result, (float, int)):
            output = [float(result)]
        else:
            output = [float(v) for v in list(result)]
        if len(output) != self._n_outputs:
            raise ValueError(
                f"PythonCBlock expected {self._n_outputs} outputs, got {len(output)}"
            )
        self._step_index += 1
        return output

    def reset(self) -> None:
        """Clear the context dict and reset the step counter."""
        self._ctx = {}
        self._step_index = 0
