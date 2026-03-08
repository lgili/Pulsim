## Why

PSIM and PLECS both offer a **C-Block** (also called DLL Block or C-Script Block): a
signal-domain component where the user writes arbitrary C code compiled to a shared
library, called once per accepted simulation step with input signals and expected to
produce output signals that feed back into the circuit.  This is one of the most
demanded professional features in power-electronics simulation — it lets engineers
embed custom control algorithms, real-time DSP code, lookup tables, state machines,
and legacy embedded-C firmware directly into the simulation loop without recompiling
the simulator.

Pulsim does not have this capability today.  The `SignalEvaluator` DAG supports
built-in blocks (PI, PID, PWM, etc.) but has no mechanism for user-injected
computation.  This gap makes Pulsim unsuitable for workflows where control code
cannot be expressed with the available built-in blocks.

This change introduces the **Pulsim C-Block** feature, which closes that gap with:
- A versioned, stable C ABI (`pulsim_cblock_abi.h`) that user code compiles against.
- A Python helper (`pulsim.cblock`) that compiles `.c` source → `.so`/`.dll` at
  runtime and wraps it in a safe `CBlockLibrary` object.
- A **Python-callable fallback** (`fn=` parameter) for rapid prototyping without
  a C compiler.
- First-class `SignalEvaluator` integration: `"C_BLOCK"` becomes a native signal
  type, evaluated in topological order with full algebraic-loop detection.
- YAML netlist support: `type: C_BLOCK` component with `lib_path`, `source`,
  `n_inputs`, `n_outputs` parameters.
- Comprehensive examples, tutorial, and API reference.

### Reference Implementations
- PSIM C DLL Block: embedded C function `simuser(t, dt, in, out)` signature
- PLECS C-Script Block: `Output(…)`, `State(…)` function hooks per block
- Pulsim departs from both by using a **context pointer** (`void*`) for persistent
  state, avoiding global state and enabling multiple independent instances.

## What Changes

- **New public header** `core/include/pulsim/v1/cblock_abi.h` — versioned C ABI
  (`PULSIM_CBLOCK_ABI_VERSION = 1`)
- **New Python module** `python/pulsim/cblock.py` — `compile_cblock()`,
  `CBlockLibrary`, `PythonCBlock` (callable fallback)
- **`SignalEvaluator` extended** (`python/pulsim/signal_evaluator.py`) — adds
  `"C_BLOCK"` to `SIGNAL_TYPES`, evaluates via `CBlockLibrary.step()` or
  `PythonCBlock.step()`
- **YAML parser extended** (`core/src/v1/yaml_parser.cpp`) — `type: C_BLOCK`
  component schema with validation
- **Python public API** (`python/pulsim/__init__.py`) — exports `compile_cblock`,
  `CBlockLibrary`, `PythonCBlock`
- **New examples** under `examples/cblock/` — 5 worked examples (C and Python fallback)
- **New documentation** under `docs/cblock/` — user guide, API reference, tutorial
- **New tests** — unit (Python), integration (round-trip), cross-platform matrix

### Breaking Changes
None.  All additions are backward-compatible.  Existing circuits without `C_BLOCK`
components are completely unaffected.

## Impact

- Affected specs: `device-models`, `kernel-v1-core`, `netlist-yaml`, `python-bindings`
- Affected code:
  - `python/pulsim/signal_evaluator.py` — extend `SIGNAL_TYPES` + evaluation
  - `python/pulsim/cblock.py` — new module
  - `python/pulsim/__init__.py` — re-export public API
  - `core/include/pulsim/v1/cblock_abi.h` — new public header
  - `core/src/v1/yaml_parser.cpp` — C_BLOCK schema + validation
  - `python/tests/test_cblock.py` — new test module
  - `python/tests/test_signal_evaluator.py` — extend for C_BLOCK type
  - `examples/cblock/` — new directory with 5 examples
  - `docs/cblock/` — new documentation section
- Dependencies: none new (uses `ctypes` stdlib, `subprocess` for compiler)
