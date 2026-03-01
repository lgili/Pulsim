## Context

Pulsim is a new circuit simulator focused on power electronics. It aims to provide:
- Modern API-first architecture (unlike legacy tools like ngspice)
- First-class Python integration for automation and analysis
- Focus on power electronics features (switching, losses, thermal)
- Open-source alternative to commercial tools

**Stakeholders:**
- Power electronics engineers needing simulation tools
- Researchers automating circuit analysis
- Students learning circuit simulation

**Constraints:**
- Must be cross-platform (Linux, macOS, Windows)
- No GPL dependencies in core (licensing compatibility)
- Performance must be competitive with ngspice

## Goals / Non-Goals

### Goals
- High-performance transient simulation for power electronics
- Clean separation between kernel, API, and bindings
- Streaming results for real-time visualization
- Extensible device and solver architecture
- Python-first user experience

### Non-Goals (for now)
- RF/microwave simulation (different domain)
- PCB layout integration
- Full SPICE compatibility from day one
- GPU acceleration (evaluate later)
- Commercial support/licensing

## Decisions

### D1: C++20 for Kernel
**Decision:** Use C++20 as the kernel language.

**Rationale:**
- Mature numeric ecosystem (Eigen, SUNDIALS, SuiteSparse)
- pybind11 provides excellent Python bindings
- Concepts and ranges improve code quality
- Industry-standard for scientific computing

**Alternatives considered:**
- Rust: Better safety, but smaller numeric ecosystem
- Julia: Good for prototyping, harder to deploy as library

### D2: Eigen for Linear Algebra
**Decision:** Use Eigen 3.4+ as the primary linear algebra library.

**Rationale:**
- Header-only, easy to integrate
- Excellent sparse matrix support
- Expression templates minimize temporaries
- MPL2 license is permissive

**Alternatives considered:**
- Armadillo: Similar features, different API style
- Native BLAS/LAPACK: Lower-level, more boilerplate

### D3: gRPC for API
**Decision:** Use gRPC with Protobuf as the primary API protocol.

**Rationale:**
- Streaming support for waveforms
- Type-safe, auto-generated clients
- Cross-language support (Python, Go, web via gRPC-web)
- Efficient binary protocol

**Alternatives considered:**
- REST only: No native streaming
- WebSocket: Less structured, manual serialization

### D4: pybind11 for Python Bindings
**Decision:** Use pybind11 for native Python bindings.

**Rationale:**
- Seamless numpy integration
- Automatic reference counting
- Supports modern C++ features
- Well-documented and maintained

**Alternatives considered:**
- SWIG: More languages, but complex
- Cython: Good but requires separate codebase
- nanobind: Newer, less proven

### D5: JSON as Primary Netlist Format
**Decision:** Use JSON as the primary netlist format, with YAML as secondary.

**Rationale:**
- Unambiguous parsing
- Schema validation available
- Easy to generate programmatically
- Python/JavaScript native support

**Alternatives considered:**
- SPICE format: Complex grammar, ambiguous
- YAML only: Ordering issues with complex structures
- Custom DSL: Higher learning curve

### D6: Event-Driven Hybrid Simulation
**Decision:** Implement hybrid continuous-discrete simulation with event queue.

**Rationale:**
- Essential for power electronics (switches, PWM)
- Improves accuracy at switching transitions
- Reduces unnecessary timesteps during constant states
- Industry standard approach (e.g., PLECS, PSIM)

### D7: Backward Euler as Default Integrator
**Decision:** Use Backward Euler as the default time integration method.

**Rationale:**
- L-stable (handles stiff systems well)
- Simple to implement initially
- Good for power electronics (fast transients)
- Can add higher-order methods later

**Alternatives considered:**
- Trapezoidal: A-stable but can ring
- Gear/BDF: More accurate but complex

### D8: Sparse Direct Solver
**Decision:** Use direct (LU) sparse solvers, not iterative.

**Rationale:**
- Circuit matrices are typically small-medium size
- Direct methods are robust and reliable
- LU factors can be reused across timesteps
- Iterative methods need preconditioning tuning

## Risks / Trade-offs

### R1: C++ Complexity
**Risk:** C++ has steep learning curve and memory safety issues.
**Mitigation:**
- Use smart pointers and RAII everywhere
- AddressSanitizer and UBSanitizer in CI
- Consider Rust for isolated new modules

### R2: SPICE Compatibility
**Risk:** Users expect SPICE netlist compatibility.
**Mitigation:**
- Plan SPICE parser for MVP-3+
- Document format differences clearly
- Provide conversion tools

### R3: Performance vs Features
**Risk:** Adding features may degrade performance.
**Mitigation:**
- Continuous benchmarking in CI
- Profile before optimizing
- Keep hot paths minimal

### R4: API Stability
**Risk:** Early API changes break users.
**Mitigation:**
- Version the Protobuf API
- Semantic versioning for Python package
- Deprecation policy before removal

## Migration Plan

Not applicable - this is a new project.

## Open Questions

1. **Model format details:** Final JSON schema for netlists needs community input.
2. **Device parameter fitting:** How to best extract model parameters from datasheets?
3. **Thermal network topology:** Should thermal networks be user-defined or auto-generated?
4. **FMU standard version:** FMI 2.0 or 3.0?
5. **GUI framework:** If/when we add GUI, React vs other frameworks?

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Pulsim                                  │
├─────────────────────────────────────────────────────────────────────┤
│  CLI (pulsim)        Python (pulsim)       Web UI (future)     │
│  ─────────────         ─────────────────       ────────────────    │
│  run, serve, sweep     Native or gRPC client   React + gRPC-web    │
└───────────┬────────────────────┬───────────────────────┬────────────┘
            │                    │                       │
            ▼                    ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         gRPC API Layer                              │
│  ────────────────────────────────────────────────────────────────   │
│  CreateSession, StartSimulation, StreamWaveforms, GetResult         │
│  REST Gateway (optional), Authentication, Quotas                    │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Kernel (C++20)                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Parser  │  │   MNA    │  │  Solver  │  │ Devices  │            │
│  │ (JSON/   │──│ Assembly │──│ (Newton, │──│ (R,L,C,  │            │
│  │  YAML)   │  │ (Sparse) │  │  LU)     │  │ MOSFET)  │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│                                    │                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                          │
│  │  Events  │  │ Thermal  │  │  Losses  │                          │
│  │ (Switch, │──│ (Foster  │──│ (Cond,   │                          │
│  │  PWM)    │  │  RC net) │  │  Sw)     │                          │
│  └──────────┘  └──────────┘  └──────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     External Libraries                              │
│  ────────────────────────────────────────────────────────────────   │
│  Eigen (Linear Algebra), SUNDIALS (ODE/DAE), SuiteSparse (LU)      │
│  nlohmann/json (Parser), pybind11 (Python), gRPC (API)             │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Netlist (JSON) ──► Parser ──► IR (Internal Representation)
                                       │
                                       ▼
                              ┌────────────────┐
                              │   Simulation   │◄────── Time Loop
                              │     Engine     │
                              └────────┬───────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           ▼                           ▼                           ▼
    ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
    │ MNA Matrix  │            │   Newton    │            │   Event     │
    │  Assembly   │───────────►│   Solver    │◄───────────│   Queue     │
    └─────────────┘            └──────┬──────┘            └─────────────┘
                                      │
                                      ▼
                              ┌─────────────┐
                              │   Results   │──► CSV, HDF5, Stream
                              └─────────────┘
```
