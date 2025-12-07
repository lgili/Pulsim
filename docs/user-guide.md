# Pulsim User Guide

Pulsim is a high-performance circuit simulator optimized for power electronics applications. It provides fast transient simulation with accurate switching device models, thermal modeling, and loss calculation.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [CLI Usage](#cli-usage)
5. [Python Interface](#python-interface)
6. [gRPC API](#grpc-api)
7. [Examples](#examples)

---

## Getting Started

### System Requirements

- **Operating Systems**: Linux, macOS, Windows
- **Compiler**: C++20 compatible (GCC 10+, Clang 12+, MSVC 2019+)
- **CMake**: 3.20 or later
- **Python**: 3.8+ (for Python bindings)

### Key Features

- **Fast Transient Simulation**: Optimized sparse matrix solvers with adaptive timestep
- **Power Electronics Focus**: Ideal switches, diodes, MOSFETs, IGBTs with accurate models
- **Thermal Modeling**: Foster network thermal models with temperature-dependent parameters
- **Loss Calculation**: Conduction and switching losses with efficiency reporting
- **Multiple Interfaces**: CLI, Python bindings, gRPC API
- **Parallel Execution**: Multi-threaded assembly and parameter sweeps

---

## Installation

### Building from Source

```bash
# Clone the repository
git clone https://github.com/your-org/pulsim-core.git
cd pulsim-core

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DPULSIM_BUILD_PYTHON=ON \
         -DPULSIM_BUILD_GRPC=ON

# Build
cmake --build . --parallel

# Install (optional)
sudo cmake --install .
```

### Python Package

```bash
# Install from source
pip install ./python

# Or install pre-built wheel (when available)
pip install pulsim
```

### Docker

```bash
# Pull image
docker pull pulsim:latest

# Run gRPC server
docker run -p 50051:50051 -p 9090:9090 pulsim:latest
```

---

## Quick Start

### 1. Create a Circuit (JSON Netlist)

Create a file `rc_circuit.json`:

```json
{
  "name": "RC Low-Pass Filter",
  "components": [
    {"type": "V", "name": "V1", "nodes": ["in", "0"], "value": 1.0},
    {"type": "R", "name": "R1", "nodes": ["in", "out"], "value": 1000},
    {"type": "C", "name": "C1", "nodes": ["out", "0"], "value": 1e-6}
  ],
  "simulation": {
    "type": "transient",
    "stop_time": 0.01,
    "timestep": 1e-6
  }
}
```

### 2. Run Simulation

**CLI:**
```bash
pulsim run rc_circuit.json -o results.csv
```

**Python:**
```python
import pulsim

result = pulsim.simulate("rc_circuit.json")
print(result.time)
print(result.voltages["out"])
```

### 3. View Results

The output CSV contains time-series data:
```
time,V(in),V(out),I(V1),I(R1)
0.000000e+00,1.000000e+00,0.000000e+00,-1.000000e-03,1.000000e-03
1.000000e-06,1.000000e+00,9.990005e-04,-9.990005e-04,9.990005e-04
...
```

---

## CLI Usage

### Commands

| Command | Description |
|---------|-------------|
| `run` | Run a simulation |
| `validate` | Validate a netlist without running |
| `sweep` | Parameter sweep with parallel execution |
| `serve` | Start gRPC API server |
| `info` | Display device model documentation |

### Run Command

```bash
pulsim run <netlist> [options]

Options:
  -o, --output <file>     Output file (default: stdout)
  -f, --format <format>   Output format: csv, json, hdf5, parquet
  --timestep <value>      Override simulation timestep
  --stop-time <value>     Override stop time
  --abstol <value>        Absolute tolerance (default: 1e-12)
  --reltol <value>        Relative tolerance (default: 1e-3)
  -v, --verbose           Enable verbose output
  -q, --quiet             Suppress progress output
```

**Examples:**

```bash
# Basic simulation with CSV output
pulsim run buck_converter.json -o results.csv

# JSON output with custom tolerances
pulsim run circuit.json -f json --abstol 1e-14 --reltol 1e-4

# HDF5 output for large simulations
pulsim run large_circuit.json -f hdf5 -o results.h5
```

### Sweep Command

```bash
pulsim sweep <netlist> --param <name>=<start>:<stop>:<steps> [options]

Options:
  -j, --jobs <n>          Parallel jobs (default: CPU count)
  -o, --output <dir>      Output directory
  --format <format>       Output format per run
```

**Example:**

```bash
# Sweep load resistance from 10 to 100 ohms in 10 steps
pulsim sweep buck.json --param R_load=10:100:10 -j 8 -o sweep_results/
```

### Serve Command

```bash
pulsim serve [options]

Options:
  --listen <addr>         Listen address (default: 0.0.0.0:50051)
  --metrics-port <port>   Prometheus metrics port (default: 9090)
  --workers <n>           Worker threads (default: auto)
  --max-sessions <n>      Maximum concurrent sessions (default: 64)
```

### Info Command

```bash
# List all available device models
pulsim info --list

# Show details for a specific model
pulsim info MOSFET

# Show netlist format documentation
pulsim info --format
```

---

## Python Interface

### Basic Usage

```python
import pulsim
import numpy as np

# Load and simulate
result = pulsim.simulate("circuit.json")

# Access results
time = result.time                    # numpy array
v_out = result.voltages["out"]        # voltage at node "out"
i_r1 = result.currents["R1"]          # current through R1

# Get simulation statistics
print(f"Total steps: {result.total_steps}")
print(f"Newton iterations: {result.newton_iterations_total}")
```

### Building Circuits Programmatically

```python
import pulsim

# Create circuit
circuit = pulsim.Circuit("Buck Converter")

# Add components
circuit.add_voltage_source("Vin", "in", "0", 12.0)
circuit.add_switch("S1", "in", "sw", control="PWM1")
circuit.add_diode("D1", "0", "sw")
circuit.add_inductor("L1", "sw", "out", 100e-6)
circuit.add_capacitor("C1", "out", "0", 100e-6)
circuit.add_resistor("R_load", "out", "0", 10.0)

# Add PWM source
circuit.add_pwm("PWM1", frequency=100e3, duty_cycle=0.5)

# Configure simulation
options = pulsim.SimulationOptions(
    stop_time=1e-3,
    timestep=1e-8,
    abstol=1e-12,
    reltol=1e-3
)

# Run simulation
result = pulsim.simulate(circuit, options)
```

### Plotting Results

```python
import matplotlib.pyplot as plt

result = pulsim.simulate("buck.json")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Output voltage
axes[0].plot(result.time * 1e3, result.voltages["out"])
axes[0].set_ylabel("Vout (V)")
axes[0].grid(True)

# Inductor current
axes[1].plot(result.time * 1e3, result.currents["L1"])
axes[1].set_ylabel("IL (A)")
axes[1].grid(True)

# Switch voltage
axes[2].plot(result.time * 1e3, result.voltages["sw"])
axes[2].set_ylabel("Vsw (V)")
axes[2].set_xlabel("Time (ms)")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("buck_waveforms.png")
```

### Loss Analysis

```python
result = pulsim.simulate("inverter.json")

# Get loss breakdown
losses = result.losses

print("Loss Summary:")
print(f"  Total: {losses.total_energy * 1e3:.2f} mJ")
print(f"  Conduction: {losses.conduction_energy * 1e3:.2f} mJ")
print(f"  Switching: {losses.switching_energy * 1e3:.2f} mJ")

# Per-device losses
for device, loss in losses.by_device.items():
    print(f"  {device}: {loss * 1e3:.2f} mJ")

# Calculate efficiency
p_in = np.mean(result.voltages["in"] * result.currents["Vin"])
p_out = np.mean(result.voltages["out"] ** 2 / 10)  # R_load = 10
efficiency = p_out / p_in * 100
print(f"Efficiency: {efficiency:.1f}%")
```

### Thermal Simulation

```python
# Circuit with thermal model
circuit = pulsim.Circuit("MOSFET with Thermal")

# MOSFET with thermal coupling
circuit.add_mosfet("M1", "drain", "gate", "source", "0",
    model="IRF540N",
    thermal_node="Tj_M1"
)

# Thermal network (Foster model)
circuit.add_thermal_network("Tj_M1", "Tc",
    rth=[0.1, 0.2, 0.3],  # Thermal resistances (K/W)
    tau=[1e-3, 10e-3, 100e-3]  # Time constants (s)
)

# Heatsink
circuit.add_thermal_resistor("Rth_cs", "Tc", "Ta", 0.5)  # Case to sink
circuit.add_thermal_resistor("Rth_sa", "Ta", "0", 1.0)   # Sink to ambient

result = pulsim.simulate(circuit, options)

# Plot junction temperature
plt.plot(result.time, result.temperatures["Tj_M1"])
plt.xlabel("Time (s)")
plt.ylabel("Junction Temperature (°C)")
```

### Parameter Sweeps

```python
import pulsim
from concurrent.futures import ProcessPoolExecutor

def run_with_duty(duty):
    circuit = pulsim.load("buck.json")
    circuit.set_parameter("PWM1.duty_cycle", duty)
    result = pulsim.simulate(circuit)
    v_out_avg = np.mean(result.voltages["out"][-1000:])
    return duty, v_out_avg

# Parallel sweep
duties = np.linspace(0.1, 0.9, 17)
with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(run_with_duty, duties))

# Plot transfer characteristic
duties, v_outs = zip(*results)
plt.plot(duties, v_outs, 'o-')
plt.xlabel("Duty Cycle")
plt.ylabel("Average Output Voltage (V)")
```

---

## gRPC API

### Connecting to Server

**Python:**
```python
from pulsim.client import PulsimClient

# Connect to server
client = PulsimClient("localhost:50051")

# Create session
session = client.create_session()

# Run simulation
result = session.simulate("circuit.json")

# Stream results (for long simulations)
for chunk in session.stream_simulation("large_circuit.json"):
    print(f"Progress: {chunk.progress}%")
    process_partial_results(chunk.data)

# Clean up
session.close()
```

**Using grpcurl:**
```bash
# Create session
grpcurl -d '{}' localhost:50051 pulsim.Simulator/CreateSession

# Run simulation
grpcurl -d '{
  "session_id": "abc123",
  "netlist_json": "{...}"
}' localhost:50051 pulsim.Simulator/StartSimulation

# Get results
grpcurl -d '{"session_id": "abc123"}' localhost:50051 pulsim.Simulator/GetResult
```

### Streaming Waveforms

```python
# For long simulations, stream results as they're computed
async for waveform in session.stream_waveforms("simulation.json"):
    # Update live plot
    update_plot(waveform.time, waveform.data)
```

---

## Examples

### Example 1: RC Low-Pass Filter

```json
{
  "name": "RC Low-Pass Filter",
  "components": [
    {"type": "V", "name": "Vin", "nodes": ["in", "0"],
     "waveform": {"type": "pulse", "v1": 0, "v2": 1, "period": 1e-3}},
    {"type": "R", "name": "R1", "nodes": ["in", "out"], "value": 1000},
    {"type": "C", "name": "C1", "nodes": ["out", "0"], "value": 1e-6}
  ],
  "simulation": {
    "type": "transient",
    "stop_time": 5e-3,
    "timestep": 1e-7
  }
}
```

### Example 2: Buck Converter

```json
{
  "name": "Synchronous Buck Converter",
  "components": [
    {"type": "V", "name": "Vin", "nodes": ["in", "0"], "value": 12},
    {"type": "SWITCH", "name": "S_high", "nodes": ["in", "sw"],
     "control": "PWM", "ron": 0.01},
    {"type": "SWITCH", "name": "S_low", "nodes": ["sw", "0"],
     "control": "PWM_inv", "ron": 0.01},
    {"type": "L", "name": "L1", "nodes": ["sw", "out"], "value": 100e-6},
    {"type": "C", "name": "C1", "nodes": ["out", "0"], "value": 100e-6},
    {"type": "R", "name": "R_load", "nodes": ["out", "0"], "value": 5}
  ],
  "sources": [
    {"type": "PWM", "name": "PWM", "frequency": 100e3, "duty": 0.5},
    {"type": "PWM", "name": "PWM_inv", "frequency": 100e3, "duty": 0.5,
     "inverted": true, "dead_time": 100e-9}
  ],
  "simulation": {
    "type": "transient",
    "stop_time": 1e-3,
    "timestep": 1e-9
  }
}
```

### Example 3: Full-Bridge Inverter

```json
{
  "name": "H-Bridge Inverter",
  "components": [
    {"type": "V", "name": "Vdc", "nodes": ["vdc", "0"], "value": 400},
    {"type": "MOSFET", "name": "Q1", "nodes": ["vdc", "g1", "a", "0"],
     "model": "IRF540N"},
    {"type": "MOSFET", "name": "Q2", "nodes": ["a", "g2", "0", "0"],
     "model": "IRF540N"},
    {"type": "MOSFET", "name": "Q3", "nodes": ["vdc", "g3", "b", "0"],
     "model": "IRF540N"},
    {"type": "MOSFET", "name": "Q4", "nodes": ["b", "g4", "0", "0"],
     "model": "IRF540N"},
    {"type": "L", "name": "L_load", "nodes": ["a", "mid"], "value": 1e-3},
    {"type": "R", "name": "R_load", "nodes": ["mid", "b"], "value": 10}
  ],
  "sources": [
    {"type": "PWM", "name": "g1", "frequency": 10e3, "duty": 0.5},
    {"type": "PWM", "name": "g2", "frequency": 10e3, "duty": 0.5,
     "inverted": true, "dead_time": 500e-9},
    {"type": "PWM", "name": "g3", "frequency": 10e3, "duty": 0.5,
     "phase": 180},
    {"type": "PWM", "name": "g4", "frequency": 10e3, "duty": 0.5,
     "phase": 180, "inverted": true, "dead_time": 500e-9}
  ],
  "simulation": {
    "type": "transient",
    "stop_time": 10e-3,
    "timestep": 10e-9
  }
}
```

### Example 4: MOSFET with Thermal Model

```json
{
  "name": "MOSFET Thermal Test",
  "components": [
    {"type": "V", "name": "Vds", "nodes": ["drain", "0"], "value": 50},
    {"type": "V", "name": "Vgs", "nodes": ["gate", "0"], "value": 10},
    {"type": "MOSFET", "name": "M1", "nodes": ["drain", "gate", "source", "0"],
     "model": "IRF540N",
     "thermal": {
       "node": "Tj",
       "rth_jc": 0.5,
       "foster": {
         "r": [0.1, 0.2, 0.15],
         "tau": [1e-3, 10e-3, 100e-3]
       }
     }
    },
    {"type": "R", "name": "R_source", "nodes": ["source", "0"], "value": 0.1}
  ],
  "thermal": {
    "ambient": 25,
    "heatsink_rth": 1.5
  },
  "simulation": {
    "type": "transient",
    "stop_time": 1.0,
    "timestep": 1e-6
  }
}
```

---

## Troubleshooting

### Common Issues

**1. Simulation doesn't converge**
- Try reducing timestep
- Enable Gmin stepping: `"convergence": {"gmin_stepping": true}`
- Check for floating nodes

**2. Slow simulation**
- Increase timestep if accuracy permits
- Use adaptive timestep: `"timestep": "auto"`
- Enable parallel assembly for large circuits

**3. Memory issues with large simulations**
- Use HDF5 output format
- Enable result streaming
- Reduce output points with `"output_interval"`

### Getting Help

- Documentation: https://pulsim.dev/docs
- GitHub Issues: https://github.com/your-org/pulsim-core/issues
- Discord: https://discord.gg/pulsim

---

## Next Steps

- [Netlist Format Reference](netlist-format.md)
- [Device Model Reference](device-models.md)
- [API Reference](api-reference.md)
- [Tutorial Notebooks](../examples/notebooks/)
