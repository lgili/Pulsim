## ADDED Requirements

### Requirement: Python Codegen API
Python bindings SHALL expose `pulsim.codegen.generate(circuit, target, step, out_dir, **opts)` and `pulsim.codegen.run_pil_bench(out_dir)`.

#### Scenario: Programmatic codegen
- **GIVEN** a `Circuit` object built via templates or YAML
- **WHEN** Python calls `pulsim.codegen.generate(circuit, "c99", 1e-6, "gen/")`
- **THEN** the output directory contains `pulsim_model.{c,h}`, `pulsim_topologies.c`, `Makefile`
- **AND** the function returns a `CodegenSummary` with budgets and warnings

#### Scenario: PIL bench programmatic
- **WHEN** Python calls `pulsim.codegen.run_pil_bench("gen/")`
- **THEN** the bench compiles the generated C, runs against native Pulsim, and returns `PilBenchResult` with parity verdict
- **AND** divergence above tolerance raises `PilParityError`
