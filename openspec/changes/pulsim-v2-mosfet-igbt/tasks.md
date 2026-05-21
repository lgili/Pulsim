## Phase 1 — Builder helpers (~0.3 days)

- [x] 1.1 `CircuitBuilder::add_mosfet` (1 branch).
- [x] 1.2 `CircuitBuilder::add_mosfet_with_body_diode`
      (2 branches: switch + anti-parallel diode).
- [x] 1.3 `CircuitBuilder::add_igbt` (1 branch).
- [x] 1.4 SMPS-realistic defaults documented in the
      header.

## Phase 2 — C++ tests (~0.3 days)

- [x] 2.1 `add_mosfet` smoke — branch + conductance.
- [x] 2.2 `add_mosfet` defaults check.
- [x] 2.3 `add_mosfet_with_body_diode` smoke — 2 branches,
      diode antiparallel direction verified.
- [x] 2.4 `add_mosfet_with_body_diode` body-diode V_F
      default check.
- [x] 2.5 `add_igbt` smoke — 1 branch with IGBT defaults.
- [x] 2.6 Buck-topology smoke: source + MOSFET-with-body-
      diode + freewheeling diode + resistor = 5 branches.

## Phase 3 — Python bindings + tests (~0.2 days)

- [x] 3.1 Expose the 3 builder methods in
      `bindings_v2_kernel.cpp`.
- [x] 3.2 Python smoke tests for each method (5 cases).
- [x] 3.3 Buck topology via the Python MOSFET helper +
      cache.build() smoke.

## Phase 4 — Regression + docs (~0.1 days)

- [x] 4.1 All 14 v2 C++ binaries pass (4894 assertions
      in 284 cases). All 16 Python tests pass.
- [x] 4.2 `openspec validate pulsim-v2-mosfet-igbt
      --strict` passes.
- [x] 4.3 `docs/pulsim-v2/layer2-v1-mosfet-igbt.md`.
