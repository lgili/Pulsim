## Why

Pulsim has no way to host **user-defined block code** that reads circuit
signals and drives signals back at a chosen sample rate. Power-electronics
users routinely need this: digital control laws, lookup tables, protection
logic, state machines, or custom device models that don't map to a fixed
component. PSIM ("C block" / "Simplified C block" / "DLL block"), PLECS
("C-Script"), and Simulink (S-function) all provide it. Today a Pulsim
user must hand-wire `step_observer` + `b_extra_fn` + controlled sources
and manage sample-rate throttling themselves — error-prone and not
reusable, and impossible in C/C++.

## What Changes

- New **`c-block`** capability: a sampled subsystem with a user-chosen
  number of **inputs** and **outputs**, **wired** to the circuit, running
  user code at a **user-chosen timestep** (its own sample rate, ≥ sim dt),
  with zero-order hold between block steps.
- The block's `step` code may be written in **Python** (a callable),
  **C**, or **C++**:
  - C/C++ as a **compiled shared library** (`.so`/`.dll`) exporting a
    fixed C ABI, loaded at runtime via `ctypes` (no kernel rebuild — the
    "DLL block" model); C++ via `extern "C"`.
  - C/C++ as **inline source** auto-compiled to a temporary shared
    library (content-hash cached) using the system compiler, then loaded
    via the same ctypes path.
- **Inputs** are wires that read node voltages or branch currents.
  **Outputs** are wires that drive controlled voltage/current sources
  injected into the circuit (held ZOH between block updates).
- Surfaces: **Python API** (`add_c_block`), **netlist YAML** (a
  `c_block` node), and a **schematic/GUI** node (PulsimGUI, which emits
  the same YAML/Python representation).
- Outputs use the **PWL** engine (`b_extra` residual injection).
  Inputs-only blocks (logging/observers) also work under DSED.

## Impact

- Affected specs: **new `c-block`**; **`netlist-yaml`** (new `c_block`
  node type).
- Affected code: `python/pulsim/` (new `c_block.py` module +
  `simulate()` composition of its observer/b_extra), the YAML loader
  (`python/pulsim/yaml_chain.py` / loader), docs (`docs/`), tests
  (`python/tests/`). No C++ kernel change is required for v1 (the block
  rides the existing PWL callback path). The GUI node lives in PulsimGUI
  (separate repo) and emits this representation — tracked but out of this
  change's spec scope.
- Security: compiling/loading user C/C++ executes arbitrary native code
  with the same trust level as running the user's Python — documented as
  a trust boundary; no sandbox is provided.
