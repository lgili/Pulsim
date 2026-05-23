## Phase 1 — Builder header (~0.4 days)

- [x] 1.1 New header `pulsim/v2/builder/circuit_builder.hpp`.
- [x] 1.2 Class `CircuitBuilder` with `node()` lookup
      helper and a private `resolve_node_(name)` that
      handles the `gnd` alias + auto-creation.
- [x] 1.3 Device methods for V, R, C, L, switched diode,
      nonlinear diode, switch. User-friendly units
      (ohms / farads / henries), method chaining via
      `return *this`.
- [x] 1.4 Accessors `.graph()`, `.pool()`,
      `.num_branches()`, `.node_id_of(name)`.

## Phase 2 — Tests (~0.4 days)

- [x] 2.1 `"gnd"` alias maps to ground.
- [x] 2.2 Node lookup creates / reuses correctly.
- [x] 2.3 `node_id_of` throws for unknown name.
- [x] 2.4 Implicit node creation via device methods.
- [x] 2.5 Resistor unit conversion (R → G = 1/R).
- [x] 2.6 Capacitor / Inductor pass-through.
- [x] 2.7 Method chaining works.
- [x] 2.8 Round-trip V_dc + R: cache.solve gives V_dc on
      the node.
- [x] 2.9 Half-wave rectifier: builder ≡ manual setup
      sample-by-sample within 1 µV through `run_transient`.
- [x] 2.10 Nonlinear diode DC load-line via the builder.

## Phase 3 — Wire CMake (~0.1 days)

- [x] 3.1 `pulsim_v2_builder_tests` target added to
      `core/CMakeLists.txt`.
- [x] 3.2 Full v2 regression sweep green (14 binaries,
      4871 assertions in 278 cases).

## Phase 4 — OpenSpec + docs (~0.1 days)

- [x] 4.1 `openspec validate pulsim-v2-builder-api --strict`
      passes.
- [x] 4.2 `docs/pulsim-v2/layer6-builder-api.md`.
