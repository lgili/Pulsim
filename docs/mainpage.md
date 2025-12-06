# SpiceLab C++ API Reference {#mainpage}

Welcome to the SpiceLab C++ API documentation. SpiceLab is a high-performance circuit simulator optimized for power electronics applications.

## Overview

SpiceLab provides a complete simulation engine for transient, DC, and AC circuit analysis with:

- **Fast sparse matrix solvers** using Eigen and KLU
- **Accurate device models** for MOSFETs, IGBTs, diodes, and transformers
- **Thermal modeling** with Foster networks and temperature-dependent parameters
- **Loss calculation** for efficiency analysis
- **Event-driven simulation** for switching circuits

## Architecture

The SpiceLab library is organized into the following main namespaces:

### Core Library (`spicelab`)

- @ref spicelab::Circuit - Circuit representation and component management
- @ref spicelab::Simulator - Main simulation engine
- @ref spicelab::MNA - Modified Nodal Analysis matrix assembly
- @ref spicelab::Solver - Linear and nonlinear solvers

### Device Models (`spicelab::devices`)

- @ref spicelab::devices::Resistor
- @ref spicelab::devices::Capacitor
- @ref spicelab::devices::Inductor
- @ref spicelab::devices::VoltageSource
- @ref spicelab::devices::Diode
- @ref spicelab::devices::MOSFET
- @ref spicelab::devices::IGBT
- @ref spicelab::devices::Transformer

### gRPC API (`spicelab::api::grpc`)

- @ref spicelab::api::grpc::SimulatorServer - gRPC server implementation
- @ref spicelab::api::grpc::SessionManager - Session management
- @ref spicelab::api::grpc::JobQueue - Job queue for async simulation
- @ref spicelab::api::grpc::MetricsServer - Prometheus metrics

## Quick Start

### Basic Simulation

```cpp
#include <spicelab/circuit.hpp>
#include <spicelab/simulation.hpp>

using namespace spicelab;

int main() {
    // Create circuit
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", 12.0);
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_capacitor("C1", "out", "0", 1e-6);

    // Configure simulation
    SimulationOptions options;
    options.stop_time = 0.01;
    options.timestep = 1e-6;

    // Run simulation
    Simulator sim(circuit, options);
    SimulationResult result = sim.run_transient();

    // Access results
    for (size_t i = 0; i < result.time.size(); ++i) {
        std::cout << result.time[i] << ", "
                  << result.voltages["out"][i] << "\n";
    }

    return 0;
}
```

### Loading from JSON

```cpp
#include <spicelab/parser.hpp>
#include <spicelab/simulation.hpp>

int main() {
    // Parse netlist
    auto [circuit, options] = spicelab::parse_netlist("circuit.json");

    // Run simulation
    Simulator sim(circuit, options);
    auto result = sim.run_transient();

    return 0;
}
```

### Using the gRPC Server

```cpp
#include <spicelab/api/grpc/server.hpp>

int main() {
    spicelab::api::grpc::ServerConfig config;
    config.listen_address = "0.0.0.0:50051";
    config.max_sessions = 64;
    config.num_workers = 8;

    spicelab::api::grpc::SimulatorServer server(config);
    server.start();
    server.wait();

    return 0;
}
```

## Building

### Requirements

- C++20 compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.20+
- Eigen 3.4+

### Build Commands

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `SPICELAB_BUILD_TESTS` | ON | Build unit tests |
| `SPICELAB_BUILD_PYTHON` | OFF | Build Python bindings |
| `SPICELAB_BUILD_GRPC` | OFF | Build gRPC API |
| `SPICELAB_BUILD_EXAMPLES` | OFF | Build examples |

## Module Documentation

- [Core Types](@ref types.hpp) - Basic types and data structures
- [Circuit](@ref circuit.hpp) - Circuit representation
- [MNA](@ref mna.hpp) - Matrix assembly
- [Simulation](@ref simulation.hpp) - Simulation engine
- [Devices](@ref devices/) - Device models
- [Thermal](@ref thermal.hpp) - Thermal modeling
- [Losses](@ref losses.hpp) - Loss calculation
- [gRPC API](@ref api/grpc/) - Remote API

## See Also

- [User Guide](user-guide.md)
- [Netlist Format](netlist-format.md)
- [Device Models](device-models.md)
