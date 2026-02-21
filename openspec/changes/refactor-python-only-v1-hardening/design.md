## Context

The repository already converged on a unified v1 kernel, but still carries parallel legacy artifacts and stale workflow narratives:

- `core/legacy/**` still exists and can reintroduce confusion about supported runtime paths.
- `python/CMakeLists.txt` still includes `core/legacy/include`.
- Duplicate or stale surfaces exist (`python/bindings_v2.cpp`, skipped planned-API tests, old docs mentioning CLI/grpc/JSON user flows).
- Benchmarks are Python-first but external parity is centered on ngspice, while product validation priority now requires LTspice parity coverage.

The goal is not only cleanup; it is operational confidence: robust convergence, declared component support, and reproducible parity evidence.

## Goals / Non-Goals

### Goals

- Enforce Python-only user-facing contract with v1 kernel backend.
- Retire legacy code safely via port-then-delete policy.
- Define and satisfy a converter-focused component matrix with thermal and loss coverage.
- Introduce LTspice-first parity benchmarking with deterministic artifacts.
- Expand stress testing to cover light, medium, and heavy simulation workloads.

### Non-Goals

- Full physical parity with every LTspice device model in one release.
- New GUI or grpc productization work.
- Rewriting stable core algorithms without measured benefit.

## Decisions

### 1) Python-Only Product Surface

Python package APIs are the only supported user interface.
C++ remains an implementation detail for bindings and internal tooling.
Documentation and examples must not advertise unsupported direct-user C++ or CLI paths.

### 2) Legacy Retirement Policy

Legacy functionality follows a strict sequence:
1) Inventory and classify (drop, migrate, keep temporary).
2) Port required behavior to v1 (with tests and parity checks).
3) Remove legacy source/build/docs references.

No legacy deletion happens before equivalent v1 behavior is validated.

### 3) Component Support Matrix

Define a declared support matrix for converter simulation:
- Passive and controlled sources
- Power switches (diode, switch, MOSFET, IGBT)
- Magnetics and converter structures (inductor, transformer)
- Loss and thermal modeling paths

Every declared item needs YAML schema support, Python exposure, and validation coverage.

### 4) Electro-Thermal Coupling

Thermal simulation must be usable in converter workflows, not only as an isolated post-process utility.
The design will support coupled runs where electrical loss results feed thermal states and thermal states feed temperature-sensitive parameters when enabled.

### 5) LTspice Parity Framework

Benchmark framework adds an LTspice backend contract:
- explicit LTspice executable path and run mode
- waveform/vector mapping per benchmark
- unified comparator metrics and artifacts
- deterministic pass/fail semantics (no silent skips for mapped required cases)

ngspice can remain optional, but LTspice parity is the primary acceptance gate.

### 6) Stress and Determinism Gating

Stress suites are organized into tiers:
- Tier A: analytical sanity (RC/RL/RLC)
- Tier B: nonlinear and switching
- Tier C: large sparse and stiff converter cases

Each tier tracks convergence status, fallback path, iteration budgets, and runtime metrics.
Deterministic fields (status, steps, error metrics) must be reproducible for fixed configs on a given hardware class.

## Risks / Trade-offs

- LTspice automation varies by OS and installation path; adapter code must be explicit and testable.
- Aggressive legacy removal can break hidden dependencies; migration matrix and staged gates are mandatory.
- Electro-thermal coupling increases solver stiffness; robust fallback and timestep policies are required.

## Migration Plan

1. Build audit matrix (legacy features, owners, migration target, delete criteria).
2. Freeze Python-only API contract and mark deprecated surfaces.
3. Port missing converter/thermal capabilities to v1 and expose in YAML + Python.
4. Introduce LTspice parity harness and mapped benchmark catalog.
5. Remove legacy sources/build references after parity and stress gates pass.
6. Clean docs, examples, and CI so supported paths are unambiguous.

## Open Questions

- Minimum LTspice parity catalog required for first acceptance gate (which converter topologies).
- Platform policy for LTspice execution in CI vs local-only validation.
- Deprecation window duration for procedural Python aliases before hard removal.
