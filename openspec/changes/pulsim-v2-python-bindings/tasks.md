## Phase 1 — pybind11 wrapper (~0.5 days)

- [x] 1.1 New `python/bindings_v2_kernel.cpp` exposing
      CircuitBuilder, Graph, DevicePool, SwitchStateMask,
      PwlStateSpaceCache, SimulationOptions,
      SimulationResult, CommutationEvent, run_transient,
      IdealDiodeParams.
- [x] 1.2 Modify `python/bindings.cpp` PYBIND11_MODULE
      to call `init_module(m.def_submodule("v2_kernel"))`.
- [x] 1.3 Modify `python/CMakeLists.txt` to compile the
      new file into `_pulsim`.

## Phase 2 — Python wrapper (~0.1 days)

- [x] 2.1 New `python/pulsim/v2.py` that re-exports all
      v2_kernel symbols under `pulsim.v2`.
- [x] 2.2 Verify `_pulsim_top_files` glob in
      `python/CMakeLists.txt` picks up `v2.py` automatically
      (no extra wiring needed).

## Phase 3 — Tests (~0.4 days)

- [x] 3.1 Smoke test: `import pulsim.v2 as p` works.
- [x] 3.2 Public-symbol coverage: every documented binding
      is reachable.
- [x] 3.3 `gnd` alias maps to ground.
- [x] 3.4 `node_id_of` throws for unknown names.
- [x] 3.5 SimulationOptions constructor + flag mutability.
- [x] 3.6 V_dc + R DC solve roundtrip.
- [x] 3.7 Half-wave rectifier built and run from Python
      with > 90 % half-wave tracking.
- [x] 3.8 IdealDiodeParams defaults + keyword args.
- [x] 3.9 add_nonlinear_diode binding builds successfully.
- [x] 3.10 Graph num_nodes / num_branches accessors.

## Phase 4 — Regression + docs (~0.1 days)

- [x] 4.1 All C++ v2 tests stay green (regression sweep
      via `cmake --build` and per-binary `./build/...`).
- [x] 4.2 `openspec validate pulsim-v2-python-bindings
      --strict` passes.
- [x] 4.3 `docs/pulsim-v2/layer7-python-bindings.md`.
