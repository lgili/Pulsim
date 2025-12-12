## Why

Pulsim aims to be a modern, high-performance circuit simulator focused on power electronics applications. Existing tools like ngspice and LTspice lack modern API interfaces, streaming capabilities, and Python-first integration. Commercial tools are expensive and closed-source. There's a need for an open-source simulator with:
- High-performance C++ kernel with hybrid event-driven simulation
- Modern gRPC/REST API for integration
- First-class Python bindings for automation and Jupyter
- Focus on power electronics: switching losses, thermal modeling

## What Changes

This proposal defines the complete architecture for Pulsim v1.0:

### Kernel Components
- **kernel-parser**: Netlist/model parsing (JSON/YAML format, future SPICE compatibility)
- **kernel-mna**: Modified Nodal Analysis matrix assembly
- **kernel-solver**: Time integration and nonlinear solving (Newton-Raphson, SUNDIALS)
- **kernel-devices**: Component library (R, L, C, switches, MOSFETs, diodes, sources)
- **kernel-events**: Event-driven simulation for switching
- **kernel-thermal**: Lumped thermal RC networks
- **kernel-losses**: Switching and conduction loss calculation

### API Layer
- **api-grpc**: gRPC service with streaming waveforms

### Integration
- **python-bindings**: Native (pybind11) and gRPC client
- **cli**: Command-line interface for batch simulation

## Impact

- Affected specs: All new (no existing specs)
- Affected code: Entire codebase (new project)

### MVP Phases

**MVP-0**: Minimal kernel
- Parser (JSON netlist)
- MNA for linear components (R, L, C)
- Backward-Euler integration
- Newton solver
- CLI: `pulsim run circuit.json -o result.csv`

**MVP-1**: Power electronics basics
- Ideal switch, diode models
- Event manager for switching
- Loss engine basics
- Python bindings (native)

**MVP-2**: Full features
- MOSFET/IGBT models with parameters
- Thermal coupling
- gRPC API with streaming
- Python gRPC client

**MVP-3**: Performance & scale
- SUNDIALS integration
- Solver reuse/caching
- Multi-threading
- Job queue for batch runs
