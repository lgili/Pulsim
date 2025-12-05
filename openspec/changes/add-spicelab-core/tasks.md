## Phase 0: Project Setup

- [ ] 0.1 Initialize CMake project structure with C++20 configuration
- [ ] 0.2 Configure Conan or vcpkg for dependency management
- [ ] 0.3 Add Eigen, nlohmann/json as initial dependencies
- [ ] 0.4 Set up clang-format and clang-tidy configuration
- [ ] 0.5 Configure CI pipeline (GitHub Actions) for Linux/macOS/Windows
- [ ] 0.6 Create basic README with build instructions

## Phase 1: MVP-0 - Minimal Kernel

### 1.1 Parser (kernel-parser)
- [ ] 1.1.1 Define JSON netlist schema
- [ ] 1.1.2 Implement JSON parser with nlohmann/json
- [ ] 1.1.3 Create IR (Internal Representation) data structures
- [ ] 1.1.4 Implement node name to index mapping
- [ ] 1.1.5 Add validation for component parameters
- [ ] 1.1.6 Add unit tests for parser

### 1.2 MNA Assembly (kernel-mna)
- [ ] 1.2.1 Implement sparse matrix class (CSR format) using Eigen
- [ ] 1.2.2 Implement resistor stamp
- [ ] 1.2.3 Implement capacitor companion model (Backward Euler)
- [ ] 1.2.4 Implement inductor companion model (Backward Euler)
- [ ] 1.2.5 Implement voltage source stamp
- [ ] 1.2.6 Implement current source stamp
- [ ] 1.2.7 Add ground node elimination
- [ ] 1.2.8 Add unit tests for MNA assembly

### 1.3 Linear Solver (kernel-solver)
- [ ] 1.3.1 Integrate Eigen SparseLU solver
- [ ] 1.3.2 Implement factorization caching
- [ ] 1.3.3 Add singular matrix detection and error reporting
- [ ] 1.3.4 Add unit tests for linear solver

### 1.4 Nonlinear Solver (kernel-solver)
- [ ] 1.4.1 Implement Newton-Raphson iteration loop
- [ ] 1.4.2 Add convergence checking (abstol, reltol)
- [ ] 1.4.3 Implement damping for divergent steps
- [ ] 1.4.4 Add iteration limit and failure reporting
- [ ] 1.4.5 Add unit tests for Newton solver

### 1.5 Time Integration (kernel-solver)
- [ ] 1.5.1 Implement Backward Euler time stepping
- [ ] 1.5.2 Implement fixed timestep simulation loop
- [ ] 1.5.3 Add DC operating point analysis
- [ ] 1.5.4 Add result storage (in-memory)
- [ ] 1.5.5 Add unit tests for transient simulation

### 1.6 Basic CLI (cli)
- [ ] 1.6.1 Create CLI application with argparse or CLI11
- [ ] 1.6.2 Implement `run` command with basic options
- [ ] 1.6.3 Implement CSV output
- [ ] 1.6.4 Add `validate` command
- [ ] 1.6.5 Add progress reporting to stderr
- [ ] 1.6.6 Add integration tests for CLI

### 1.7 MVP-0 Validation
- [ ] 1.7.1 Create test circuits: RC, RL, RLC
- [ ] 1.7.2 Verify against analytical solutions
- [ ] 1.7.3 Create benchmark comparison with ngspice

## Phase 2: MVP-1 - Power Electronics Basics

### 2.1 Devices (kernel-devices)
- [ ] 2.1.1 Implement ideal switch model (voltage-controlled)
- [ ] 2.1.2 Implement ideal diode model
- [ ] 2.1.3 Implement pulse voltage source
- [ ] 2.1.4 Implement PWL voltage source
- [ ] 2.1.5 Add unit tests for devices

### 2.2 Event Manager (kernel-events)
- [ ] 2.2.1 Implement event queue (priority queue by time)
- [ ] 2.2.2 Implement zero-crossing detection
- [ ] 2.2.3 Implement event-triggered timestep adjustment
- [ ] 2.2.4 Handle switch state changes
- [ ] 2.2.5 Add integration restart after events
- [ ] 2.2.6 Add unit tests for event handling

### 2.3 PWM Support (kernel-events)
- [ ] 2.3.1 Implement fixed-frequency PWM event generator
- [ ] 2.3.2 Support variable duty cycle
- [ ] 2.3.3 Add dead-time handling

### 2.4 Loss Engine Basics (kernel-losses)
- [ ] 2.4.1 Implement conduction loss calculation (I²R)
- [ ] 2.4.2 Add loss accumulation over time
- [ ] 2.4.3 Output loss summary per device
- [ ] 2.4.4 Add unit tests for loss calculation

### 2.5 Native Python Bindings (python-bindings)
- [ ] 2.5.1 Set up pybind11 in CMake
- [ ] 2.5.2 Create Python module `spicelab`
- [ ] 2.5.3 Expose `simulate()` function
- [ ] 2.5.4 Return results as numpy arrays
- [ ] 2.5.5 Add Python tests with pytest
- [ ] 2.5.6 Create pip-installable package

### 2.6 MVP-1 Validation
- [ ] 2.6.1 Simulate buck converter with ideal components
- [ ] 2.6.2 Verify switching waveforms
- [ ] 2.6.3 Verify loss calculations
- [ ] 2.6.4 Create example Jupyter notebook

## Phase 3: MVP-2 - Full Features

### 3.1 Advanced Devices (kernel-devices)
- [ ] 3.1.1 Implement Shockley diode model
- [ ] 3.1.2 Implement diode with junction capacitance
- [ ] 3.1.3 Implement Level 1 MOSFET model
- [ ] 3.1.4 Implement MOSFET with body diode
- [ ] 3.1.5 Implement MOSFET capacitances (Cgs, Cgd, Cds)
- [ ] 3.1.6 Implement IGBT simplified model
- [ ] 3.1.7 Add parameter library for common devices
- [ ] 3.1.8 Add unit tests for all models

### 3.2 Transformer (kernel-devices)
- [ ] 3.2.1 Implement ideal transformer
- [ ] 3.2.2 Add magnetizing inductance
- [ ] 3.2.3 Add leakage inductances
- [ ] 3.2.4 Add unit tests

### 3.3 Switching Losses (kernel-losses)
- [ ] 3.3.1 Implement turn-on energy (Eon) calculation
- [ ] 3.3.2 Implement turn-off energy (Eoff) calculation
- [ ] 3.3.3 Implement diode reverse recovery loss (Err)
- [ ] 3.3.4 Support lookup table interpolation
- [ ] 3.3.5 Add loss breakdown output
- [ ] 3.3.6 Add efficiency calculation

### 3.4 Thermal Modeling (kernel-thermal)
- [ ] 3.4.1 Implement thermal node and network
- [ ] 3.4.2 Implement Foster network from parameters
- [ ] 3.4.3 Couple power loss to thermal network
- [ ] 3.4.4 Implement temperature-dependent Rds_on
- [ ] 3.4.5 Add junction temperature output
- [ ] 3.4.6 Add thermal limit warnings
- [ ] 3.4.7 Add unit tests

### 3.5 gRPC API (api-grpc)
- [ ] 3.5.1 Define protobuf messages and service
- [ ] 3.5.2 Implement gRPC server skeleton
- [ ] 3.5.3 Implement CreateSession/StartSimulation
- [ ] 3.5.4 Implement StreamWaveforms with gRPC streaming
- [ ] 3.5.5 Implement GetResult with format options
- [ ] 3.5.6 Add session management and cleanup
- [ ] 3.5.7 Add integration tests for API

### 3.6 Python gRPC Client (python-bindings)
- [ ] 3.6.1 Generate Python gRPC stubs
- [ ] 3.6.2 Implement Client class with connection management
- [ ] 3.6.3 Implement streaming result handling
- [ ] 3.6.4 Add DataFrame/xarray conversion
- [ ] 3.6.5 Add async iterator interface
- [ ] 3.6.6 Add Jupyter widgets for streaming plots
- [ ] 3.6.7 Add Python client tests

### 3.7 CLI Enhancements (cli)
- [ ] 3.7.1 Add `serve` command for API server
- [ ] 3.7.2 Add `sweep` command with parallel execution
- [ ] 3.7.3 Add HDF5 and Parquet output support
- [ ] 3.7.4 Add configuration file support
- [ ] 3.7.5 Add `info` command for device documentation

### 3.8 MVP-2 Validation
- [ ] 3.8.1 Simulate full-bridge inverter
- [ ] 3.8.2 Validate MOSFET switching waveforms against datasheet
- [ ] 3.8.3 Validate thermal response with step power
- [ ] 3.8.4 Verify efficiency calculation against manual computation

## Phase 4: MVP-3 - Performance and Scale

### 4.1 Advanced Solvers (kernel-solver)
- [ ] 4.1.1 Integrate SUNDIALS (IDA for DAE)
- [ ] 4.1.2 Implement adaptive timestep with error control
- [ ] 4.1.3 Integrate SuiteSparse KLU for faster LU
- [ ] 4.1.4 Implement factorization reuse across timesteps
- [ ] 4.1.5 Add Trapezoidal and Gear integration methods
- [ ] 4.1.6 Benchmark against ngspice

### 4.2 Convergence Aids (kernel-solver)
- [ ] 4.2.1 Implement Gmin stepping
- [ ] 4.2.2 Implement source stepping
- [ ] 4.2.3 Implement pseudo-transient continuation

### 4.3 AC Analysis (kernel-solver)
- [ ] 4.3.1 Implement linearization at operating point
- [ ] 4.3.2 Implement complex impedance matrix solve
- [ ] 4.3.3 Add frequency sweep
- [ ] 4.3.4 Output magnitude/phase (Bode data)

### 4.4 Parallelization
- [ ] 4.4.1 Multi-thread matrix assembly
- [ ] 4.4.2 SIMD optimization for device evaluation
- [ ] 4.4.3 Parallel parameter sweeps
- [ ] 4.4.4 Job queue for batch runs

### 4.5 Scale (api-grpc)
- [ ] 4.5.1 Implement job queue with Redis or similar
- [ ] 4.5.2 Add worker pool for concurrent simulations
- [ ] 4.5.3 Add per-user quotas and resource limits
- [ ] 4.5.4 Add Prometheus metrics
- [ ] 4.5.5 Create Docker image
- [ ] 4.5.6 Create Kubernetes deployment manifests

### 4.6 Documentation
- [ ] 4.6.1 Write user guide with examples
- [ ] 4.6.2 Document netlist format and device models
- [ ] 4.6.3 Create API reference (doxygen for C++, sphinx for Python)
- [ ] 4.6.4 Create tutorial Jupyter notebooks

## Phase 5: Maturation (Future)

### 5.1 Format Support
- [ ] 5.1.1 Implement SPICE netlist parser (.cir/.sp)
- [ ] 5.1.2 Implement YAML netlist parser
- [ ] 5.1.3 Implement subcircuit support (.subckt)
- [ ] 5.1.4 Implement FMU export (Model Exchange)
- [ ] 5.1.5 Implement FMU co-simulation support

### 5.2 Advanced Models
- [ ] 5.2.1 Implement higher-level MOSFET models
- [ ] 5.2.2 Implement magnetic core models (with saturation)
- [ ] 5.2.3 Implement control blocks (PI, PID, comparator)

### 5.3 Front-end (future scope)
- [ ] 5.3.1 Create React web UI skeleton
- [ ] 5.3.2 Implement schematic editor
- [ ] 5.3.3 Implement waveform viewer
- [ ] 5.3.4 Create Tauri desktop app wrapper
