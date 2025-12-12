## Phase 0: Project Setup

- [x] 0.1 Initialize CMake project structure with C++20 configuration
- [x] 0.2 Configure Conan or vcpkg for dependency management
- [x] 0.3 Add Eigen, nlohmann/json as initial dependencies
- [x] 0.4 Set up clang-format and clang-tidy configuration
- [x] 0.5 Configure CI pipeline (GitHub Actions) for Linux/macOS/Windows
- [x] 0.6 Create basic README with build instructions

## Phase 1: MVP-0 - Minimal Kernel

### 1.1 Parser (kernel-parser)
- [x] 1.1.1 Define JSON netlist schema
- [x] 1.1.2 Implement JSON parser with nlohmann/json
- [x] 1.1.3 Create IR (Internal Representation) data structures
- [x] 1.1.4 Implement node name to index mapping
- [x] 1.1.5 Add validation for component parameters
- [x] 1.1.6 Add unit tests for parser

### 1.2 MNA Assembly (kernel-mna)
- [x] 1.2.1 Implement sparse matrix class (CSR format) using Eigen
- [x] 1.2.2 Implement resistor stamp
- [x] 1.2.3 Implement capacitor companion model (Backward Euler)
- [x] 1.2.4 Implement inductor companion model (Backward Euler)
- [x] 1.2.5 Implement voltage source stamp
- [x] 1.2.6 Implement current source stamp
- [x] 1.2.7 Add ground node elimination
- [x] 1.2.8 Add unit tests for MNA assembly


### 1.3 Linear Solver (kernel-solver)
- [x] 1.3.1 Integrate Eigen SparseLU solver
- [x] 1.3.2 Implement factorization caching
- [x] 1.3.3 Add singular matrix detection and error reporting
- [x] 1.3.4 Add unit tests for linear solver


### 1.4 Nonlinear Solver (kernel-solver)
- [x] 1.4.1 Implement Newton-Raphson iteration loop
- [x] 1.4.2 Add convergence checking (abstol, reltol)
- [x] 1.4.3 Implement damping for divergent steps
- [x] 1.4.4 Add iteration limit and failure reporting
- [x] 1.4.5 Add unit tests for Newton solver


### 1.5 Time Integration (kernel-solver)
- [x] 1.5.1 Implement Backward Euler time stepping
- [x] 1.5.2 Implement fixed timestep simulation loop
- [x] 1.5.3 Add DC operating point analysis
- [x] 1.5.4 Add result storage (in-memory)
- [x] 1.5.5 Add unit tests for transient simulation


### 1.6 Basic CLI (cli)
- [x] 1.6.1 Create CLI application with argparse or CLI11
- [x] 1.6.2 Implement `run` command with basic options
- [x] 1.6.3 Implement CSV output
- [x] 1.6.4 Add `validate` command
- [x] 1.6.5 Add progress reporting to stderr
- [x] 1.6.6 Add integration tests for CLI


### 1.7 MVP-0 Validation
- [x] 1.7.1 Create test circuits: RC, RL, RLC
- [x] 1.7.2 Verify against analytical solutions
- [x] 1.7.3 Create benchmark comparison with ngspice (see benchmarks/BENCHMARK_REPORT.md)

## Phase 2: MVP-1 - Power Electronics Basics

### 2.1 Devices (kernel-devices)
- [x] 2.1.1 Implement ideal switch model (voltage-controlled)
- [x] 2.1.2 Implement ideal diode model
- [x] 2.1.3 Implement pulse voltage source
- [x] 2.1.4 Implement PWL voltage source
- [x] 2.1.5 Add unit tests for devices

### 2.2 Event Manager (kernel-events)
- [x] 2.2.1 Implement event queue (priority queue by time)
- [x] 2.2.2 Implement zero-crossing detection
- [x] 2.2.3 Implement event-triggered timestep adjustment
- [x] 2.2.4 Handle switch state changes
- [x] 2.2.5 Add integration restart after events
- [x] 2.2.6 Add unit tests for event handling

### 2.3 PWM Support (kernel-events)
- [x] 2.3.1 Implement fixed-frequency PWM event generator
- [x] 2.3.2 Support variable duty cycle
- [x] 2.3.3 Add dead-time handling

### 2.4 Loss Engine Basics (kernel-losses)
- [x] 2.4.1 Implement conduction loss calculation (I²R)
- [x] 2.4.2 Add loss accumulation over time
- [x] 2.4.3 Output loss summary per device
- [x] 2.4.4 Add unit tests for loss calculation

### 2.5 Native Python Bindings (python-bindings)
- [x] 2.5.1 Set up pybind11 in CMake
- [x] 2.5.2 Create Python module `pulsim`
- [x] 2.5.3 Expose `simulate()` function
- [x] 2.5.4 Return results as numpy arrays
- [x] 2.5.5 Add Python tests with pytest
- [x] 2.5.6 Create pip-installable package

### 2.6 MVP-1 Validation
- [x] 2.6.1 Simulate buck converter with ideal components
- [x] 2.6.2 Verify switching waveforms
- [x] 2.6.3 Verify loss calculations
- [x] 2.6.4 Create example Jupyter notebook

## Phase 3: MVP-2 - Full Features

### 3.1 Advanced Devices (kernel-devices)
- [x] 3.1.1 Implement Shockley diode model
- [x] 3.1.2 Implement diode with junction capacitance
- [x] 3.1.3 Implement Level 1 MOSFET model
- [x] 3.1.4 Implement MOSFET with body diode
- [x] 3.1.5 Implement MOSFET capacitances (Cgs, Cgd, Cds)
- [x] 3.1.6 Implement IGBT simplified model
- [x] 3.1.7 Add parameter library for common devices
- [x] 3.1.8 Add unit tests for all models

### 3.2 Transformer (kernel-devices)
- [x] 3.2.1 Implement ideal transformer
- [x] 3.2.2 Add magnetizing inductance
- [x] 3.2.3 Add leakage inductances
- [x] 3.2.4 Add unit tests

### 3.3 Switching Losses (kernel-losses)
- [x] 3.3.1 Implement turn-on energy (Eon) calculation
- [x] 3.3.2 Implement turn-off energy (Eoff) calculation
- [x] 3.3.3 Implement diode reverse recovery loss (Err)
- [x] 3.3.4 Support lookup table interpolation
- [x] 3.3.5 Add loss breakdown output
- [x] 3.3.6 Add efficiency calculation

### 3.4 Thermal Modeling (kernel-thermal)
- [x] 3.4.1 Implement thermal node and network
- [x] 3.4.2 Implement Foster network from parameters
- [x] 3.4.3 Couple power loss to thermal network
- [x] 3.4.4 Implement temperature-dependent Rds_on
- [x] 3.4.5 Add junction temperature output
- [x] 3.4.6 Add thermal limit warnings
- [x] 3.4.7 Add unit tests

### 3.5 gRPC API (api-grpc)
- [x] 3.5.1 Define protobuf messages and service
- [x] 3.5.2 Implement gRPC server skeleton
- [x] 3.5.3 Implement CreateSession/StartSimulation
- [x] 3.5.4 Implement StreamWaveforms with gRPC streaming
- [x] 3.5.5 Implement GetResult with format options
- [x] 3.5.6 Add session management and cleanup
- [x] 3.5.7 Add integration tests for API

### 3.6 Python gRPC Client (python-bindings)
- [x] 3.6.1 Generate Python gRPC stubs
- [x] 3.6.2 Implement Client class with connection management
- [x] 3.6.3 Implement streaming result handling
- [x] 3.6.4 Add DataFrame/xarray conversion
- [x] 3.6.5 Add async iterator interface
- [x] 3.6.6 Add Jupyter widgets for streaming plots
- [x] 3.6.7 Add Python client tests

### 3.7 CLI Enhancements (cli)
- [x] 3.7.1 Add `serve` command for API server
- [x] 3.7.2 Add `sweep` command with parallel execution
- [x] 3.7.3 Add HDF5 and Parquet output support
- [x] 3.7.4 Add configuration file support
- [x] 3.7.5 Add `info` command for device documentation

### 3.8 MVP-2 Validation
- [x] 3.8.1 Simulate full-bridge inverter
- [x] 3.8.2 Validate MOSFET switching waveforms against datasheet
- [x] 3.8.3 Validate thermal response with step power
- [x] 3.8.4 Verify efficiency calculation against manual computation

## Phase 4: MVP-3 - Performance and Scale

### 4.1 Advanced Solvers (kernel-solver)
- [x] 4.1.1 Integrate SUNDIALS (IDA for DAE)
- [x] 4.1.2 Implement adaptive timestep with error control
- [x] 4.1.3 Integrate SuiteSparse KLU for faster LU
- [x] 4.1.4 Implement factorization reuse across timesteps
- [x] 4.1.5 Add Trapezoidal integration (GEAR-2) for O(dt²) accuracy (current Backward Euler is O(dt))
- [x] 4.1.6 Benchmark against ngspice

### 4.2 Convergence Aids (kernel-solver)
- [x] 4.2.1 Implement Gmin stepping
- [x] 4.2.2 Implement source stepping
- [x] 4.2.3 Implement pseudo-transient continuation

### 4.3 AC Analysis (kernel-solver)
- [x] 4.3.1 Implement linearization at operating point
- [x] 4.3.2 Implement complex impedance matrix solve
- [x] 4.3.3 Add frequency sweep
- [x] 4.3.4 Output magnitude/phase (Bode data)

### 4.4 Parallelization
- [x] 4.4.1 Multi-thread matrix assembly
- [x] 4.4.2 SIMD optimization for device evaluation
- [x] 4.4.3 Parallel parameter sweeps
- [x] 4.4.4 Job queue for batch runs

### 4.5 Scale (api-grpc)
- [x] 4.5.1 Implement job queue with Redis or similar
- [x] 4.5.2 Add worker pool for concurrent simulations
- [x] 4.5.3 Add per-user quotas and resource limits
- [x] 4.5.4 Add Prometheus metrics
- [x] 4.5.5 Create Docker image
- [x] 4.5.6 Create Kubernetes deployment manifests

### 4.6 Documentation
- [x] 4.6.1 Write user guide with examples
- [x] 4.6.2 Document netlist format and device models
- [x] 4.6.3 Create API reference (doxygen for C++, sphinx for Python)
- [x] 4.6.4 Create tutorial Jupyter notebooks

## Phase 5: Maturation (Future)

### 5.1 Format Support
- [x] 5.1.1 Implement SPICE netlist parser (.cir/.sp,asc)
- [x] 5.1.2 Implement YAML netlist parser
- [x] 5.1.3 Implement subcircuit support (.subckt)
- [x] 5.1.4 Implement FMU export (Model Exchange)
- [x] 5.1.5 Implement FMU co-simulation support

### 5.2 Advanced Models
- [x] 5.2.1 Implement higher-level MOSFET models
- [x] 5.2.2 Implement magnetic core models (with saturation)
- [x] 5.2.3 Implement control blocks (PI, PID, comparator)

### 5.3 Front-end (future scope)
- [ ] 5.3.1 Create React web UI skeleton
- [ ] 5.3.2 Implement schematic editor
- [ ] 5.3.3 Implement waveform viewer
- [ ] 5.3.4 Create Tauri desktop app wrapper
