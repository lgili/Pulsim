## Phase 1 — C++ showcase test (~0.4 days)

- [x] 1.1 New target `pulsim_v2_showcase_tests` in
      `core/CMakeLists.txt`.
- [x] 1.2 `test_buck_open_loop.cpp`: load buck.yaml, run
      100 kHz / 50 % PWM, verify steady-state V_out and
      ripple.

## Phase 2 — Python showcase script (~0.2 days)

- [x] 2.1 `examples/v2/scripts/run_buck.py`: load YAML,
      drive PWM, print steady-state stats.

## Phase 3 — Docs (~0.1 days)

- [x] 3.1 `docs/pulsim-v2/layer9-smps-showcase.md`.

## Phase 4 — Regression (~0.1 days)

- [x] 4.1 All previous tests stay green.
- [x] 4.2 `openspec validate pulsim-v2-smps-showcase
      --strict` passes.
