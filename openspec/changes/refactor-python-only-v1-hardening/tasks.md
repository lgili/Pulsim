## Gates & Definition of Done

- [ ] G1 Python-only product surface is enforced in docs, examples, and packaging.
- [ ] G2 Legacy code is removed from default build/runtime paths after successful migration.
- [ ] G3 Declared converter component matrix is fully supported in v1 (YAML + Python + runtime).
- [ ] G4 Electro-thermal converter workflows pass correctness and stability gates.
- [ ] G5 Stress tiers (A/B/C) converge with deterministic solver fallback traces.
- [ ] G6 LTspice parity suite runs for mapped benchmarks with configured thresholds and artifacts.
- [ ] G7 Required test suites have no planned-API placeholder skips in the supported surface.

## Phase 0: Baseline and Scope Lock

- [x] 0.1 Produce legacy inventory matrix (`feature`, `current_path`, `target_v1_path`, `status`).
- [x] 0.2 Define supported user-facing runtime surface as Python-only in docs and spec references.
- [x] 0.3 Define converter support matrix and parity catalog for acceptance gates.

## Phase 1: Build and Packaging Cleanup

- [x] 1.1 Remove legacy include dependencies from Python build (`python/CMakeLists.txt` and related targets).
- [x] 1.2 Remove obsolete build options and stale references that imply unsupported CLI/grpc user flows.
- [x] 1.3 Remove duplicated/unused binding sources once parity with active binding target is confirmed.

## Phase 2: Legacy Migration and Removal

- [x] 2.1 Port required legacy-only behaviors to v1 with regression tests before deletion.
- [x] 2.2 Remove migrated legacy sources and headers from repository paths.
- [x] 2.3 Remove JSON-era dependencies and fixtures that are no longer part of supported runtime loading.

## Phase 3: Converter and Thermal Capability Completion

- [ ] 3.1 Implement missing converter-critical component behaviors in v1 runtime.
- [ ] 3.2 Ensure YAML schema covers required electrical, loss, and thermal parameters.
- [ ] 3.3 Expose all required component and solver options through Python bindings.
- [ ] 3.4 Add electro-thermal coupled simulation mode and result telemetry where declared.

## Phase 4: Solver Robustness Hardening

- [ ] 4.1 Expand convergence fallback policies for stiff converter transients.
- [ ] 4.2 Add deterministic fallback reason codes and structured telemetry for every failed/retried step.
- [ ] 4.3 Add large-scale stress regressions to prevent hangs and non-terminating convergence loops.

## Phase 5: LTspice Parity Infrastructure

- [ ] 5.1 Add LTspice runner backend with explicit executable-path configuration.
- [ ] 5.2 Add benchmark mapping metadata for Pulsim signals to LTspice vectors.
- [ ] 5.3 Implement unified comparator metrics (max/rms/phase/steady-state where applicable).
- [ ] 5.4 Publish parity artifacts and summaries in stable machine-readable formats.

## Phase 6: Validation and Stress Suite Expansion

- [ ] 6.1 Add tiered circuit suite (analytical, nonlinear switching, large stiff converters).
- [ ] 6.2 Add per-tier pass criteria for accuracy, convergence, and runtime telemetry.
- [ ] 6.3 Eliminate required-suite placeholder skips by either implementing or removing stale tests.

## Phase 7: Documentation and Migration Guides

- [ ] 7.1 Rewrite user docs to reflect Python-only supported usage.
- [ ] 7.2 Remove stale CLI/grpc/JSON instructions from README and guides.
- [ ] 7.3 Add migration notes for removed APIs and a versioned deprecation timeline.

## Phase 8: Final Verification

- [ ] 8.1 Run full test/benchmark/parity pipeline and store artifacts.
- [ ] 8.2 Validate OpenSpec change with `openspec validate refactor-python-only-v1-hardening --strict`.
- [ ] 8.3 Confirm all gates are met before archive.
