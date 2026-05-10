## Why

Pulsim's existing test suite is good — 71 validation tests against analytical solutions, parity vs LTspice/NgSpice, KPI gates. But it tests **specific scenarios**. Power-electronics simulation has **invariants** that should hold across *every* simulation regardless of topology:

- **Kirchhoff's Current Law**: at every node, sum of currents = 0 (at every step, within numerical tolerance).
- **Kirchhoff's Voltage Law**: around every loop, sum of voltages = 0.
- **Tellegen's theorem**: for any network and any compatible voltage/current vectors, `Σ v_k * i_k = 0`.
- **Energy conservation**: in lossless circuits, total stored energy + dissipated energy = energy delivered by sources, within numerical noise.
- **Passivity**: passive elements never deliver energy; sign of `v · i` consistent with element type.
- **Periodicity**: in steady-state of periodic systems, `x(t + T) = x(t)` within tolerance.
- **Reciprocity**: reciprocal networks have symmetric admittance matrices.

These are **physical laws**. Property-based testing (Hypothesis in Python, RapidCheck in C++) generates random circuits that satisfy structural constraints and then asserts these invariants hold. When a property fails, the framework **shrinks** the failing example to a minimal repro automatically — invaluable for debugging.

This change adds a property-testing harness covering the above invariants, generating circuits with bounded complexity (1–20 components, 1–10 nodes), and integrates into CI.

## What Changes

### Property-Based Test Harness
- New `python/tests/properties/` directory using Hypothesis.
- Strategy: random valid circuits with bounded size, compositional generation (start with primitive, add components, ensure connectivity).
- Per-step invariants checked at every accepted simulation step (not just final state).

### Property Library
- **KCL** invariant: `assert max(abs(K @ x)) < tol` at every step where K is incidence matrix.
- **KVL** invariant: similarly for branch voltages.
- **Tellegen** invariant: `assert abs(sum(v * i for branches)) < tol`.
- **Energy** invariant: `assert abs(stored + dissipated - input) < tol` over lossless test runs.
- **Passivity** invariant: `assert sign(v_R * i_R) >= 0` for resistors; analogous for capacitors, inductors over a cycle.
- **Periodicity** invariant: in periodic steady-state, `assert max(abs(x(t+T) - x(t))) < tol`.
- **Reciprocity** invariant: in linear circuits, AC sweep matrix is symmetric.

### Circuit Generation Strategies
- `gen_passive_circuit(min_n=2, max_n=20)` — random passive RLC.
- `gen_switching_circuit(...)` — random circuit with at least one switching device.
- `gen_converter_topology(family="buck|boost|...")` — randomly parameterized known-topology.
- All strategies produce circuits that are guaranteed connected, have at least one source, and have a unique DC solution.

### Determinism and Seed Management
- Hypothesis seeds logged in CI; failure cases automatically reduced to minimal repro and added to a regression corpus under `python/tests/properties/regressions/`.
- Cross-platform / cross-compiler property runs in nightly CI.

### Performance Budget
- Property runs use small circuits (≤20 components) and short simulations (≤1 ms simulated time). Per-property test budget ≤30 s in default suite.
- Extended nightly run with larger circuits (≤200 components) and longer trajectories.

### C++ Property Tests
- Add RapidCheck integration to `core/tests/` for invariants checkable purely in C++ (Tellegen, KCL/KVL on assembled MNA).
- Useful for catching bugs at MNA-stamping level without round-trip through Python.

### Failure Diagnostics
- On property failure, output:
  - Hypothesis-shrunken circuit YAML.
  - Per-step invariant violation (which step, which invariant, magnitude).
  - Topology trace.
  - Suggested manual investigation command.

## Impact

- **No new capability spec** — these are testing improvements that strengthen `benchmark-suite`.
- **Affected specs**: `benchmark-suite` (new property-based requirements).
- **Affected code**: new `python/tests/properties/`, additions to `core/tests/` for RapidCheck, CI workflow updates.
- **Performance**: default suite runtime increase ≤30 s; nightly extended +10 min.

## Success Criteria

1. **Coverage**: 7 invariants implemented (KCL, KVL, Tellegen, energy, passivity, periodicity, reciprocity).
2. **Bug discovery**: at least 3 latent bugs identified during initial run (failures in shrunken examples documented).
3. **Determinism**: identical seed reproduces identical failure trace.
4. **Regression corpus**: failing examples added to `regressions/` and replayed in every CI run.
5. **CI integration**: property suite runs in default CI within 30 s budget; nightly extended runs separately.
