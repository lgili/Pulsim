# Pulsim C++ API Reference {#mainpage}

Welcome to the Pulsim C++ API documentation. Pulsim is a high-performance circuit simulator optimized for power electronics applications.

## Overview

Pulsim provides a complete simulation engine for transient, DC, and AC circuit analysis with:

- **Fast sparse matrix solvers** using Eigen and KLU
- **Accurate device models** for MOSFETs, IGBTs, diodes, and transformers
- **Thermal modeling** with Foster networks and temperature-dependent parameters
- **Loss calculation** for efficiency analysis
- **Event-driven simulation** for switching circuits

## Architecture

The Pulsim library is organized into the following main namespaces:

### Core Library (`pulsim`)

- @ref pulsim::Circuit - Circuit representation and component management
- @ref pulsim::Simulator - Main simulation engine
- @ref pulsim::MNA - Modified Nodal Analysis matrix assembly
- @ref pulsim::Solver - Linear and nonlinear solvers

### Device Models (`pulsim::devices`)

- @ref pulsim::devices::Resistor
- @ref pulsim::devices::Capacitor
- @ref pulsim::devices::Inductor
- @ref pulsim::devices::VoltageSource
- @ref pulsim::devices::Diode
- @ref pulsim::devices::MOSFET
- @ref pulsim::devices::IGBT
- @ref pulsim::devices::Transformer

### gRPC API (`pulsim::api::grpc`)

- @ref pulsim::api::grpc::SimulatorServer - gRPC server implementation
- @ref pulsim::api::grpc::SessionManager - Session management
- @ref pulsim::api::grpc::JobQueue - Job queue for async simulation
- @ref pulsim::api::grpc::MetricsServer - Prometheus metrics

## Quick Start

### Basic Simulation

```cpp
#include <pulsim/circuit.hpp>
#include <pulsim/simulation.hpp>

using namespace pulsim;

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
#include <pulsim/parser.hpp>
#include <pulsim/simulation.hpp>

int main() {
    // Parse netlist
    auto [circuit, options] = pulsim::parse_netlist("circuit.json");

    // Run simulation
    Simulator sim(circuit, options);
    auto result = sim.run_transient();

    return 0;
}
```

### Using the gRPC Server

```cpp
#include <pulsim/api/grpc/server.hpp>

int main() {
    pulsim::api::grpc::ServerConfig config;
    config.listen_address = "0.0.0.0:50051";
    config.max_sessions = 64;
    config.num_workers = 8;

    pulsim::api::grpc::SimulatorServer server(config);
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
| `PULSIM_BUILD_TESTS` | ON | Build unit tests |
| `PULSIM_BUILD_PYTHON` | OFF | Build Python bindings |
| `PULSIM_BUILD_GRPC` | OFF | Build gRPC API |
| `PULSIM_BUILD_EXAMPLES` | OFF | Build examples |

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
