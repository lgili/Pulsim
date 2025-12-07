# Project Context

## Purpose
Pulsim is a high-performance circuit simulator focused on power electronics. It provides a standalone kernel with APIs for integration, Python bindings for automation, and eventually a modern web/desktop UI.

**Core Goals:**
- High-performance transient simulation with hybrid event-driven capabilities
- Power electronics focus: switching events, losses, thermal coupling
- Modern API-first architecture (gRPC + REST gateway)
- Python-friendly for automation and Jupyter integration
- Extensible via plugins for new devices and solvers

## Tech Stack

### Kernel (core/)
- **Language**: C++20
- **Build**: CMake 3.20+
- **Linear Algebra**: Eigen 3.4+, SuiteSparse (KLU/UMFPACK)
- **ODE/DAE Solvers**: SUNDIALS (CVODE/IDA/ARKODE)
- **Testing**: Catch2 or Google Test

### API Layer (api/)
- **Protocol**: gRPC with Protobuf
- **Gateway**: gRPC-gateway for REST
- **Streaming**: gRPC streaming for waveforms

### Python Bindings (python/)
- **Native**: pybind11 for direct kernel access
- **Client**: gRPC client library
- **Package**: pip-installable `pulsim`

### Front-end (future)
- **Web**: React + Vite + Tailwind
- **Desktop**: Tauri wrapper
- **Protocol**: gRPC-web via Envoy proxy

## Project Conventions

### Code Style
- C++: Follow Google C++ Style Guide with modern C++20 features
- Use `clang-format` with project config
- Python: PEP 8, formatted with `black`
- Naming: `snake_case` for functions/variables, `PascalCase` for classes

### Architecture Patterns
- **Separation of concerns**: Kernel knows nothing about I/O or API
- **Dependency injection**: Solvers, devices, exporters are pluggable
- **Immutable simulation state**: Each run produces immutable results
- **Streaming-first**: Results can be streamed during simulation

### Testing Strategy
- Unit tests for all kernel components (solver, devices, parser)
- Integration tests with reference circuits
- Benchmark suite comparing against known solutions
- Regression tests for numerical accuracy

### Git Workflow
- Main branch: `main`
- Feature branches: `feature/<name>`
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`

## Domain Context

### Circuit Simulation Fundamentals
- **MNA (Modified Nodal Analysis)**: Matrix formulation for circuit equations
- **DAE (Differential-Algebraic Equations)**: Time integration of circuit dynamics
- **Newton-Raphson**: Nonlinear solver for each timestep
- **Event-driven**: Discrete switching events in power electronics

### Power Electronics Specifics
- **Switching devices**: MOSFETs, IGBTs, diodes with on/off states
- **Loss calculation**: Conduction losses (I²R), switching losses (Eon/Eoff)
- **Thermal modeling**: Lumped RC networks for junction temperature
- **Converters**: Buck, boost, full-bridge, three-phase topologies

## Important Constraints
- Numerical stability is critical: prefer proven algorithms
- Memory efficiency: circuits can have thousands of nodes
- Cross-platform: Linux, macOS, Windows support required
- No GPL dependencies in core (LGPL or permissive only)

## External Dependencies
- **SUNDIALS**: BSD-3-Clause, Lawrence Livermore National Lab
- **Eigen**: MPL2, Benoît Jacob et al.
- **SuiteSparse**: Various (mostly LGPL), Tim Davis
- **pybind11**: BSD-3-Clause
- **gRPC**: Apache 2.0, Google
