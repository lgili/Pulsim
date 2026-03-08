## ADDED Requirements

### Requirement: `compile_cblock` Function
The Python package SHALL expose a top-level
`compile_cblock(source, *, output_dir, name, extra_cflags, compiler)` function
that compiles a C source file or string to a shared library suitable for use with
`CBlockLibrary`.  The function MUST:
- Accept `source` as either a `str` (inline C code) or a `pathlib.Path` (path to
  a `.c` file).
- Accept an optional `compiler: str | None = None` keyword argument.  When
  provided, the given executable path SHALL be used directly and `detect_compiler()`
  MUST NOT be called.  When `None`, the function MUST call `detect_compiler()` and
  raise `CBlockCompileError` if no compiler is found.
- Apply default flags: `-O2 -shared -fPIC -std=c11 -Wall -Wextra` on POSIX;
  `/LD /O2 /std:c11` on Windows MSVC.
- Return a `pathlib.Path` pointing to the compiled shared library.
- Raise `CBlockCompileError` with compiler stderr accessible via `.stderr_output`
  if compilation fails.

#### Scenario: Successful inline C compilation
- **GIVEN** a valid C source string containing `pulsim_cblock_step` and a detected system compiler
- **WHEN** `compile_cblock(source_str, name="my_block")` is called
- **THEN** a `.so` / `.dylib` / `.dll` file is created in a temporary directory
- **AND** the returned `Path` points to a file that exists and is loadable by `ctypes.CDLL`

#### Scenario: Successful file-based compilation
- **GIVEN** a path to a valid `.c` file
- **WHEN** `compile_cblock(Path("my_block.c"))` is called
- **THEN** compilation succeeds and the library path is returned

#### Scenario: Compilation error surfaces stderr
- **GIVEN** a C source string with a syntax error
- **WHEN** `compile_cblock()` is called
- **THEN** `CBlockCompileError` is raised
- **AND** `exc.stderr_output` contains the compiler error message
- **AND** `exc.source` reproduces the failing source for debugging

#### Scenario: No compiler available raises clear error
- **GIVEN** no C compiler is available and `PULSIM_CC` is not set
- **WHEN** `compile_cblock()` is called without `compiler=`
- **THEN** `CBlockCompileError` is raised with an actionable message explaining
  how to install a compiler, set `PULSIM_CC`, or pass `compiler=` explicitly

#### Scenario: Explicit compiler path bypasses auto-detection
- **GIVEN** a valid C source and the user passes `compiler="/opt/homebrew/bin/gcc-14"`
- **WHEN** `compile_cblock(source, compiler="/opt/homebrew/bin/gcc-14")` is called
- **THEN** the subprocess is invoked with that exact compiler executable
- **AND** `detect_compiler()` is NOT called
- **AND** compilation succeeds

#### Scenario: Invalid explicit compiler path raises clear error
- **GIVEN** the user passes `compiler="/nonexistent/gcc"`
- **WHEN** `compile_cblock()` is called
- **THEN** `CBlockCompileError` is raised with a message naming the path that was not found

#### Scenario: Extra compiler flags forwarded correctly
- **GIVEN** valid C source and `extra_cflags=["-DUSE_FEATURE=1", "-march=native"]`
- **WHEN** `compile_cblock()` is called
- **THEN** the compiler subprocess receives the extra flags appended after defaults
- **AND** compilation succeeds (flags are valid)

### Requirement: `detect_compiler` Function
The Python package SHALL expose `detect_compiler() -> str | None` that returns the
path to the best available C compiler on the current system, or `None` if none is
found.  Detection priority MUST be: (1) `PULSIM_CC` environment variable,
(2) `cc`, `gcc`, `clang` on POSIX, (3) `cl.exe` / `gcc.exe` on Windows.

#### Scenario: PULSIM_CC override respected
- **GIVEN** `PULSIM_CC=/usr/local/bin/cc` is set in the environment
- **WHEN** `detect_compiler()` is called
- **THEN** it returns `/usr/local/bin/cc` without probing the system

#### Scenario: Returns None gracefully when no compiler found
- **GIVEN** no compiler is on PATH and `PULSIM_CC` is not set (mocked in test)
- **WHEN** `detect_compiler()` is called
- **THEN** it returns `None` without raising an exception

### Requirement: `CBlockLibrary` Class
The Python package SHALL expose a `CBlockLibrary` class that loads a compiled
C-Block shared library via `ctypes`, resolves the required and optional symbols,
calls `init` if present, and provides a `step(t, dt, inputs) -> list[float]`
method for evaluating the block each timestep.  The class MUST:
- Raise `CBlockABIError` if `pulsim_cblock_step` symbol is not found.
- Raise `CBlockABIError` if the exported ABI version does not match
  `PULSIM_CBLOCK_ABI_VERSION`.
- Support use as a context manager (`with CBlockLibrary(path) as blk:`).
- Expose `n_inputs: int`, `n_outputs: int`, `name: str` properties.
- Provide `reset()` that reinitialises C state via `destroy` + `init`.

#### Scenario: Load and step a simple compiled library
- **GIVEN** a compiled `.so` for a gain block (`out[0] = in[0] * 3.0`)
- **WHEN** `CBlockLibrary(path)` is constructed and `step(t=0.0, dt=1e-6, inputs=[2.0])` is called
- **THEN** the result is `[6.0]`

#### Scenario: Context manager releases library
- **GIVEN** a `CBlockLibrary` used as a context manager
- **WHEN** the `with` block exits normally
- **THEN** `destroy` is called (if present) and the library is unloaded
- **AND** calling `step` after context exit raises `RuntimeError`

#### Scenario: `reset()` reinitialises stateful block
- **GIVEN** an IIR filter C-Block that has processed 50 steps (non-zero state)
- **WHEN** `reset()` is called
- **THEN** the next `step` call produces the same output as the very first step
  with the same input (state is back to initial conditions)

#### Scenario: Wrong input length raises ValueError
- **GIVEN** a `CBlockLibrary` with `n_inputs=2`
- **WHEN** `step(inputs=[1.0])` is called (only 1 input instead of 2)
- **THEN** `ValueError` is raised before calling the C function

### Requirement: `PythonCBlock` Class
The Python package SHALL expose a `PythonCBlock` class that accepts any Python
callable as a signal block, providing the same interface as `CBlockLibrary`.
The callable SHALL receive `(ctx: dict, t: float, dt: float, inputs: list[float])`
and return `list[float]`.  The `ctx` dict MUST persist across calls and be reset
by `reset()`.  `PythonCBlock` MUST be usable anywhere `CBlockLibrary` is accepted.

#### Scenario: Callable receives correct arguments on each step
- **GIVEN** a `PythonCBlock` wrapping a function that records all calls to a list
- **WHEN** `step` is called with `t=1e-3`, `dt=1e-6`, `inputs=[0.5]`
- **THEN** the recorded call has the exact values `t=1e-3`, `dt=1e-6`, `inputs=[0.5]`

#### Scenario: `PythonCBlock` interchangeable with `CBlockLibrary` in evaluator
- **GIVEN** a `SignalEvaluator` built with a `C_BLOCK` component carrying a `PythonCBlock` controller
- **WHEN** `evaluator.step(t)` is called
- **THEN** the block output propagates downstream identically to a compiled `CBlockLibrary`

### Requirement: `CBlockCompileError`, `CBlockABIError`, `CBlockRuntimeError`
The Python package SHALL expose three exception types:
- `CBlockCompileError(RuntimeError)` with attributes `.source: str`,
  `.stderr_output: str`, and `.compiler_path: str | None` (the compiler executable
  that was used or attempted, `None` if no compiler was found at all) — for
  compile-time failures.
- `CBlockABIError(RuntimeError)` with attributes `.expected_version: int` and
  `.found_version: int | None` — for ABI contract violations at load time.
- `CBlockRuntimeError(RuntimeError)` with attributes `.return_code: int`,
  `.t: float`, `.step_index: int` — for non-zero return codes from `step`.
All three SHALL be importable directly from `pulsim`.

#### Scenario: All three exception classes are importable from top-level package
- **GIVEN** `import pulsim as ps`
- **WHEN** `ps.CBlockCompileError`, `ps.CBlockABIError`, `ps.CBlockRuntimeError` are accessed
- **THEN** they are the correct exception classes (not `AttributeError`)
