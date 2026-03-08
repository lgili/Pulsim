## 1. ABI Header (C, no compiler changes required)

- [x] 1.1 Create `core/include/pulsim/v1/cblock_abi.h` with:
  - `PULSIM_CBLOCK_ABI_VERSION` macro (int, currently 1)
  - `PulsimCBlockInfo` struct (`abi_version`, `n_inputs`, `n_outputs`, `name`)
  - `PulsimCBlockCtx` opaque forward declaration
  - `pulsim_cblock_init_fn` typedef (optional; `void**`, `const PulsimCBlockInfo*` → int)
  - `pulsim_cblock_step_fn` typedef (required; `ctx`, `t`, `dt`, `in`, `out` → int)
  - `pulsim_cblock_destroy_fn` typedef (optional; `ctx` → void)
  - Documentation comment explaining each field and return-code contract
- [x] 1.2 Add `cblock_abi.h` to the `pulsim_core` interface target install set in
  `core/CMakeLists.txt` so it ships to downstream consumers

## 2. Python `cblock.py` Module

- [x] 2.1 Create `python/pulsim/cblock.py` with class `CBlockCompileError(RuntimeError)`
- [x] 2.2 Implement `detect_compiler() -> str | None`:
  - Checks `PULSIM_CC` env var first
  - POSIX: tries `cc`, `gcc`, `clang` in order via `shutil.which`
  - Windows: tries `cl.exe`, then `gcc`
  - Returns the compiler executable path or `None`
- [x] 2.3 Implement `compile_cblock(source, *, output_dir, name, extra_cflags, compiler) -> Path`:
  - Accepts `source` as `str` (C source code) or `Path` (path to `.c` file)
  - Accepts `compiler: str | None = None`; when given, used directly (skips
    `detect_compiler()`); when `None`, falls back to env var then auto-detect
  - Compiles to shared library in `output_dir` (defaults to `tempfile.mkdtemp()`)
  - Applies flags: `-O2 -shared -fPIC -std=c11 -Wall -Wextra` on POSIX;
    `/LD /O2` on Windows MSVC
  - Raises `CBlockCompileError` (with `.compiler_path` attribute) if the given
    compiler executable does not exist or process exits non-zero
  - Returns `Path` to the compiled `.so`/`.dylib`/`.dll`
- [x] 2.4 Implement class `CBlockLibrary`:
  - `__init__(lib_path: Path | str)` — loads via `ctypes.CDLL`, resolves symbols,
    reads `n_inputs` / `n_outputs` from `pulsim_cblock_info()` if exported;
    validates ABI version; calls `pulsim_cblock_init` if present
  - Properties: `n_inputs: int`, `n_outputs: int`, `name: str`
  - `step(t: float, dt: float, inputs: Sequence[float]) -> list[float]` —
    calls C `step` function via ctypes, returns output list
  - `reset()` — calls `pulsim_cblock_destroy` + re-calls `pulsim_cblock_init`
    to reset persistent state
  - `__del__` / context manager — safely unloads library
  - Raises `CBlockABIError` if ABI version mismatch or required symbol missing
- [x] 2.5 Implement class `PythonCBlock` (no-compiler fallback):
  - `__init__(fn: Callable, n_inputs: int, n_outputs: int, name: str = "")`
  - `fn` signature: `(ctx: dict, t: float, dt: float, inputs: list[float]) -> list[float]`
  - `step(t, dt, inputs) -> list[float]` — calls `fn(self._ctx, t, dt, inputs)`
    where `self._ctx` is a plain dict for user state
  - `reset()` — resets `self._ctx` to `{}`
  - Properties: `n_inputs`, `n_outputs`, `name`
- [x] 2.6 Add `CBlockCompileError`, `CBlockABIError` to `__all__` in `cblock.py`

## 3. `SignalEvaluator` Integration

- [x] 3.1 Add `"C_BLOCK"` to `SIGNAL_TYPES` frozenset in `signal_evaluator.py`
- [x] 3.2 Add `"C_BLOCK": ["OUT"]` to `_OUTPUT_PIN_NAMES` dict
  - Note: for `n_outputs > 1`, downstream `SIGNAL_DEMUX` is the contract;
    `OUT` carries the first output for single-output blocks
- [x] 3.3 Add `"C_BLOCK"` to `_SOURCE_TYPES`? No — it has inputs; leave out of
  `_SOURCE_TYPES`
- [x] 3.4 In `build()` → controller initialisation section: for `C_BLOCK` components:
  - If `parameters.lib_path` is set: instantiate `CBlockLibrary(lib_path)`
  - If `parameters.source` is set: call `compile_cblock(source)` then wrap in
    `CBlockLibrary`
  - If `parameters.python_fn` callable is set (internal/test use): wrap in
    `PythonCBlock`
  - Store instance in `self._controllers[comp_id]`
  - Pin output count: store `ctl.n_outputs` as block metadata
- [x] 3.5 In `step()` evaluation loop: add `elif ctype == "C_BLOCK":` branch:
  - Retrieve `ctl = self._controllers[comp_id]`
  - Compute `dt` from last accepted step time (store `_prev_t` on evaluator)
  - Call `outputs = ctl.step(t, dt, inputs)`
  - Store full vector: `self._state[comp_id] = outputs`  (list[float])
  - Single-output shortcut: if `len(outputs) == 1`, store `outputs[0]` as float
    for backward compat with downstream wires expecting scalar
- [x] 3.6 In `reset()`: call `ctl.reset()` for all `C_BLOCK` controllers
- [x] 3.7 Update `_collect_inputs()` to handle `C_BLOCK` source components
  correctly (they have explicit input pin connections, not just one `IN` pin)

## 4. YAML Netlist Schema and Parser

- [x] 4.1 Add `C_BLOCK` to the allowed component types in `core/src/v1/yaml_parser.cpp`
- [x] 4.2 Validate required parameters:
  - `n_inputs` (int ≥ 1) and `n_outputs` (int ≥ 1) are required
  - Exactly one of `lib_path` or `source` must be present (or neither, for
    Python-only usage); reject circuits that specify both
  - `lib_path`: string path; warn if file does not exist at parse time
  - `source`: string path to `.c` file; warn if file does not exist at parse time
  - `extra_cflags`: optional list of strings
- [x] 4.3 Produce structured `ParseDiagnostic` entries for:
  - Missing `n_inputs` / `n_outputs`
  - Both `lib_path` and `source` specified
  - Neither specified (valid for Python-only context, emit warning-level diagnostic, not ERROR)
  - `lib_path` points to non-existent file
- [x] 4.4 Add YAML round-trip test: parse → re-serialise → compare

## 5. Python Public API

- [x] 5.1 Import and re-export from `python/pulsim/__init__.py`:
  ```python
  from pulsim.cblock import (
      CBlockLibrary,
      CBlockCompileError,
      CBlockABIError,
      PythonCBlock,
      compile_cblock,
      detect_compiler,
  )
  ```
- [x] 5.2 Add all six names to `__all__` in `__init__.py`
- [x] 5.3 Add `cblock` to the module-level `__version_info__` extras metadata
  (or a `pulsim.capabilities` dict with `"c_block": True`)

## 6. Tests

### 6.1 Unit tests — `python/tests/test_cblock.py`

- [x] 6.1.1 `test_python_cblock_passthrough` — `PythonCBlock` with identity fn, 1 in / 1 out
- [x] 6.1.2 `test_python_cblock_stateful` — `PythonCBlock` that accumulates state
  (running sum), verifies value at step 3
- [x] 6.1.3 `test_python_cblock_multi_output` — 2 inputs / 3 outputs, verify output shapes
- [x] 6.1.4 `test_python_cblock_reset` — call step, reset, verify state cleared
- [x] 6.1.5 `test_compile_cblock_no_compiler_skip` — skip if `detect_compiler()` is None
- [x] 6.1.5b `test_compile_cblock_explicit_compiler` — pass `compiler=detect_compiler()`
  explicitly, assert same result as auto-detection (skip if no compiler)
- [x] 6.1.5c `test_compile_cblock_invalid_compiler_path` — pass
  `compiler="/nonexistent/gcc"`, assert `CBlockCompileError` raised with
  `.compiler_path == "/nonexistent/gcc"`
- [x] 6.1.6 `test_compile_cblock_simple` — compile the minimal passthrough `.c` template,
  load with `CBlockLibrary`, call `step`, assert output == input
  (requires compiler; skip otherwise)
- [x] 6.1.7 `test_compile_cblock_with_state` — compile integrator `.c`, run 10 steps with
  fixed `dt=1e-6`, assert accumulated value matches analytic result
- [x] 6.1.8 `test_compile_cblock_error_propagation` — C `step` returns non-zero, verify
  `CBlockLibrary.step()` raises `CBlockRuntimeError`
- [x] 6.1.9 `test_cblock_abi_version_check` — library with wrong ABI version emits
  `CBlockABIError` at `CBlockLibrary.__init__`
- [x] 6.1.10 `test_compile_cblock_syntax_error` — invalid C source raises
  `CBlockCompileError` with compiler stderr in message
- [x] 6.1.11 `test_cblock_library_context_manager` — `with CBlockLibrary(path) as blk:`
  does not segfault after block exits

### 6.2 `SignalEvaluator` integration tests — `python/tests/test_signal_evaluator.py`

- [x] 6.2.1 `test_signal_evaluator_cblock_python_fn` — build a circuit dict with one
  `C_BLOCK` component (using `python_fn`), call `evaluator.step()` 5 times,
  assert outputs correct
- [x] 6.2.2 `test_signal_evaluator_cblock_algebraic_loop` — circuit with C_BLOCK feeding
  its own input raises `AlgebraicLoopError`
- [x] 6.2.3 `test_signal_evaluator_cblock_multi_output_demux` — C_BLOCK with 3 outputs
  connected to SIGNAL_DEMUX, verify all three downstream values are correct
- [x] 6.2.4 `test_signal_evaluator_cblock_reset` — `evaluator.reset()` re-initialises
  C_BLOCK state
- [x] 6.2.5 `test_signal_evaluator_cblock_compiled_lib` — (skip if no compiler)
  compile the gain block, wire into evaluator, run 3 steps, assert correct
- [x] 6.2.6 `test_signal_evaluator_cblock_in_closed_loop` — full PI + C_BLOCK chain:
  `CONSTANT → SUBTRACTOR → PI_CONTROLLER → C_BLOCK(gain) → PWM_GENERATOR`,
  run 10 steps, assert PWM duty in [0, 1]

### 6.3 YAML round-trip tests — `python/tests/test_netlist_parser.py`

- [x] 6.3.1 `test_yaml_cblock_valid_lib_path` — parse valid C_BLOCK with `lib_path`,
  no diagnostics at ERROR level
- [x] 6.3.2 `test_yaml_cblock_valid_source` — parse valid C_BLOCK with `source` path
- [x] 6.3.3 `test_yaml_cblock_both_specified_error` — parse with both `lib_path` and
  `source` → expect parse error diagnostic
- [x] 6.3.4 `test_yaml_cblock_missing_n_inputs_error` — missing `n_inputs` → error
- [x] 6.3.5 `test_yaml_cblock_neither_path_nor_source_info` — neither specified →
  no error (valid for Python-only), INFO level diagnostic emitted

## 7. Examples

Each example lives under `examples/cblock/` and is a fully self-contained
runnable Python script + `.c` source file.

- [x] 7.1 **`01_passthrough_gain`** — simplest possible C-Block (multiply input by gain).
  Files: `gain_block.c`, `01_passthrough_gain.py`.
  Demonstrates: compile, load, wire into circuit, run simulation, plot output.

- [x] 7.2 **`02_first_order_filter`** — first-order IIR low-pass filter in C with
  persistent state (one state variable in `ctx`).
  Files: `iir_filter.c`, `02_first_order_filter.py`.
  Demonstrates: `pulsim_cblock_init` / `pulsim_cblock_destroy` lifecycle,
  state persistence across steps, filter frequency response verification.

- [x] 7.3 **`03_pi_controller_closed_loop`** — custom PI controller in C replacing
  the built-in `PI_CONTROLLER` block, closing a voltage regulation loop.
  Files: `pi_controller.c`, `03_pi_controller_closed_loop.py`.
  Demonstrates: 2-input block (error + integral state via ctx), PWM duty output,
  comparison with built-in PI to verify numeric equivalence.

- [x] 7.4 **`04_lookup_table_efficiency`** — MOSFET switching loss lookup table
  (2D: Vds × Id) loaded from CSV in `pulsim_cblock_init`, queried per step.
  Files: `efficiency_map.c`, `efficiency_map.csv`, `04_lookup_table_efficiency.py`.
  Demonstrates: file I/O in init, complex state allocation, multi-output (loss,
  efficiency), integration with loss-tracking simulation result.

- [x] 7.5 **`05_python_callable_no_compiler`** — same gain block as example 1 but
  using `PythonCBlock` with a Python lambda.  No C compiler needed.
  Files: `05_python_callable_no_compiler.py` (single file, no `.c`).
  Demonstrates: prototyping path, identical API to `CBlockLibrary`.

## 8. Documentation

- [x] 8.1 Create `docs/cblock/index.md` — overview page:
  - What is a C-Block, why use it
  - Quick-start (5-line example)
  - Comparison table: C-Block vs Python callable vs built-in blocks
  - Link to user guide, API reference, examples

- [x] 8.2 Create `docs/cblock/user-guide.md`:
  - Section: "Writing your first C-Block" (step-by-step with the ABI struct)
  - Section: "Persistent state with `pulsim_cblock_init`"
  - Section: "Multi-output blocks and SIGNAL_DEMUX"
  - Section: "Compiling and loading" (`compile_cblock` + `CBlockLibrary`)
  - Section: "Python callable fallback (`PythonCBlock`)"
  - Section: "YAML netlist integration"
  - Section: "Debugging C-Block code" (print via `fprintf(stderr,...)`,
    diagnostic codes, GDB attach tips)
  - Section: "Security and trust model" (explicit warning about untrusted libs)

- [x] 8.3 Create `docs/cblock/abi-reference.md`:
  - Full `cblock_abi.h` listing with inline explanation of every field
  - ABI version table (currently just version 1)
  - Error code contract: what values of `step` return code mean
  - Null-terminator and memory ownership rules
  - Platform notes (symbol visibility, `-fvisibility=default`, `.def` files on MSVC)

- [x] 8.4 Create `docs/cblock/api-reference.md`:
  - `compile_cblock()` — full signature including `compiler=`, parameters, return
    value, exceptions; note on priority order (explicit > env var > auto-detect)
  - `detect_compiler()` — behavior, env var override
  - `CBlockLibrary` — constructor, all methods, exceptions raised
  - `PythonCBlock` — constructor, all methods
  - `CBlockCompileError` — attributes (`.source`, `.stderr_output`)
  - `CBlockABIError` — attributes (`.expected_version`, `.found_version`)
  - `CBlockRuntimeError` — attributes (`.return_code`, `.t`, `.step_index`)

- [x] 8.5 Create `docs/cblock/examples.md` — annotated walkthrough of all 5 examples
  with explanations of design decisions

- [x] 8.6 Add C-Block section to `mkdocs.yml` nav under a "Signal Domain" group

- [x] 8.7 Add `docs/cblock/` `requirements.txt` entry if any new Sphinx/MkDocs
  plugin is needed (expected: none)

## 9. Quality Gates

- [x] 9.1 All new Python code passes `ruff check` and `ruff format --check`
- [x] 9.2 All new Python code passes `mypy --strict python/pulsim/cblock.py`
- [x] 9.3 `cblock_abi.h` compiles clean with `-Wall -Wextra -Wpedantic -std=c11`
  and `-std=c++20` (as C includes it with `extern "C"`)
- [x] 9.4 `pytest python/tests/test_cblock.py -v` passes (compiler tests auto-skip
  if compiler not available)
- [x] 9.5 `pytest python/tests/test_signal_evaluator.py -v` passes (full suite
  including new C_BLOCK tests)
- [x] 9.6 `pytest python/tests/test_netlist_parser.py -v` passes
- [x] 9.7 All 5 examples run end-to-end without errors:
  `python examples/cblock/01_passthrough_gain.py` (skip if no compiler)
- [x] 9.8 `openspec validate add-custom-c-script-and-dll-blocks --strict` passes
- [x] 9.9 CI: add `test_cblock` job to `.github/workflows/` that:
  - Runs on ubuntu-latest and macos-latest
  - Installs `gcc` (ubuntu) / XCode CLI (macos)
  - Sets `PULSIM_CC=$(which gcc)` / `$(which clang)`
  - Runs `pytest python/tests/test_cblock.py -v`
