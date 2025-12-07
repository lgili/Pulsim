# Pulsim

**Power electronics simulation, simplified.**

High-performance circuit simulator focused on power electronics.

## Features

- **High-performance C++20 kernel** with sparse matrix solvers
- **Modified Nodal Analysis (MNA)** for circuit formulation
- **Multiple integration methods** (Backward Euler, Trapezoidal, BDF2, GEAR2)
- **Newton-Raphson** nonlinear solver with damping and adaptive timestep
- **JSON netlist format** with schematic position storage
- **CLI tool** for batch simulation
- **Python bindings** for scripting and GUI integration
- **GUI integration API** with pause/resume/stop, progress callbacks, and validation

## Quick Start

### Build

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Run tests
ctest --test-dir build --output-on-failure
```

### Usage

```bash
# Run a simulation
./build/cli/pulsim run examples/rc_circuit.json -o result.csv

# Validate a netlist
./build/cli/pulsim validate examples/voltage_divider.json

# Get circuit info
./build/cli/pulsim info examples/rlc_circuit.json
```

## Netlist Format

Pulsim uses a JSON-based netlist format:

```json
{
    "components": [
        {"type": "voltage_source", "name": "V1", "npos": "in", "nneg": "0", "waveform": 5.0},
        {"type": "resistor", "name": "R1", "n1": "in", "n2": "out", "value": "1k"},
        {"type": "capacitor", "name": "C1", "n1": "out", "n2": "0", "value": "1u"}
    ]
}
```

### Supported Components

| Component | Type | Parameters |
|-----------|------|------------|
| Resistor | `resistor`, `R` | `value` (ohms) |
| Capacitor | `capacitor`, `C` | `value` (F), `ic` (initial voltage) |
| Inductor | `inductor`, `L` | `value` (H), `ic` (initial current) |
| Voltage Source | `voltage_source`, `V` | `waveform` |
| Current Source | `current_source`, `I` | `waveform` |
| Diode | `diode`, `D` | `is`, `n`, `ideal` |
| Switch | `switch`, `S` | `ron`, `roff`, `vth`, `ctrl_pos`, `ctrl_neg` |
| MOSFET | `mosfet`, `nmos`, `pmos`, `M` | `vth`, `kp`, `lambda`, `w`, `l`, `rds_on`, `body_diode` |
| Transformer | `transformer`, `T` | `turns_ratio`, `lm` (magnetizing inductance) |

### Waveform Types

- **DC**: `5.0` or `{"type": "dc", "value": 5.0}`
- **Pulse**: `{"type": "pulse", "v1": 0, "v2": 5, "period": 1e-3, ...}`
- **Sine**: `{"type": "sin", "amplitude": 2.5, "frequency": 1000, ...}`
- **PWL**: `{"type": "pwl", "points": [[0, 0], [1e-3, 5], [2e-3, 0]]}`

### SI Prefixes

Values support SI prefixes: `f`, `p`, `n`, `u`, `m`, `k`, `meg`, `g`, `t`

Examples: `"1k"` = 1000, `"100n"` = 100e-9, `"4.7u"` = 4.7e-6

## CLI Options

```
pulsim run <netlist> [options]
  -o, --output    Output file (CSV)
  --tstop         Stop time (default: 1e-3)
  --dt            Time step (default: 1e-6)
  --tstart        Start time (default: 0)
  --abstol        Absolute tolerance (default: 1e-12)
  --reltol        Relative tolerance (default: 1e-3)
  -v, --verbose   Verbose output
  -q, --quiet     Quiet mode

pulsim validate <netlist>
  Validates netlist syntax and circuit topology

pulsim info <netlist>
  Shows circuit information
```

## Project Structure

```
pulsim-core/
├── core/               # C++ kernel library
│   ├── include/        # Public headers
│   ├── src/            # Implementation
│   └── tests/          # Unit tests
├── cli/                # Command-line interface
├── examples/           # Example circuits
└── openspec/           # Specifications
```

## Python Bindings

Pulsim includes Python bindings for scripting and GUI integration.

### Installation

```bash
# Build with Python bindings
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Install (or add build/python to PYTHONPATH)
pip install build/python
```

### Basic Usage

```python
import pulsim

# Create a circuit
circuit = pulsim.Circuit()
circuit.add_voltage_source("V1", "in", "0", 5.0)
circuit.add_resistor("R1", "in", "out", 1000.0)
circuit.add_capacitor("C1", "out", "0", 1e-6)

# Validate the circuit
result = pulsim.validate_circuit(circuit)
if not result.is_valid:
    for error in result.errors():
        print(f"Error: {error.message}")

# Run simulation
opts = pulsim.SimulationOptions()
opts.tstop = 0.01  # 10ms
opts.dt = 1e-7     # 100ns timestep

sim = pulsim.Simulator(circuit, opts)
result = sim.run_transient()

print(f"Simulated {result.num_points()} points in {result.total_time_seconds:.2f}s")
```

### GUI Integration

Pulsim provides a complete API for GUI integration:

```python
import pulsim
import threading

# Create a simulation controller for pause/resume/stop
controller = pulsim.SimulationController()

# Progress callback for progress bars
def update_progress(progress):
    print(f"Progress: {progress.progress_percent:.1f}%")
    if progress.convergence_warning:
        print("Warning: Convergence issues detected")

# Run simulation with progress tracking
sim = pulsim.Simulator(circuit, opts)
result = sim.run_transient_with_progress(
    control=controller,
    progress_callback=update_progress,
    min_interval_ms=100  # Update every 100ms
)

# From another thread (GUI button handlers):
controller.request_pause()   # Pause simulation
controller.request_resume()  # Resume simulation
controller.request_stop()    # Stop simulation
```

### Component Metadata for Palettes

```python
# Build component palette from metadata
registry = pulsim.ComponentRegistry.instance()

for category in registry.all_categories():
    print(f"[{category}]")
    for comp_type in registry.types_in_category(category):
        meta = registry.get(comp_type)
        print(f"  - {meta.display_name}")
        for param in meta.parameters:
            print(f"      {param.display_name} [{param.unit}]")
```

### Schematic Position Storage

```python
# Store component positions for layout persistence
circuit.set_position("R1", pulsim.SchematicPosition(x=100, y=50, orientation=90))

# Export circuit with positions
json_str = pulsim.circuit_to_json(circuit, include_positions=True)

# Import circuit (positions preserved)
loaded = pulsim.parse_netlist_string(json_str)
pos = loaded.get_position("R1")  # Returns the saved position
```

See `examples/gui_integration_example.py` for more examples.

## Roadmap

- [x] MVP-0: Basic kernel (R, L, C, sources, transient)
- [x] MVP-1: Power electronics (switches, events, losses)
- [x] MVP-2: Advanced devices (MOSFETs, transformers)
- [x] MVP-2b: GUI integration API (validation, progress, metadata)
- [ ] MVP-3: Performance (SUNDIALS, parallel)

## License

MIT License
