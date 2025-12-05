# SpiceLab

High-performance circuit simulator focused on power electronics.

## Features

- **High-performance C++20 kernel** with sparse matrix solvers
- **Modified Nodal Analysis (MNA)** for circuit formulation
- **Backward Euler** time integration (more methods coming)
- **Newton-Raphson** nonlinear solver with damping
- **JSON netlist format** (SPICE compatibility planned)
- **CLI tool** for batch simulation

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
./build/cli/spicelab run examples/rc_circuit.json -o result.csv

# Validate a netlist
./build/cli/spicelab validate examples/voltage_divider.json

# Get circuit info
./build/cli/spicelab info examples/rlc_circuit.json
```

## Netlist Format

SpiceLab uses a JSON-based netlist format:

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
spicelab run <netlist> [options]
  -o, --output    Output file (CSV)
  --tstop         Stop time (default: 1e-3)
  --dt            Time step (default: 1e-6)
  --tstart        Start time (default: 0)
  --abstol        Absolute tolerance (default: 1e-12)
  --reltol        Relative tolerance (default: 1e-3)
  -v, --verbose   Verbose output
  -q, --quiet     Quiet mode

spicelab validate <netlist>
  Validates netlist syntax and circuit topology

spicelab info <netlist>
  Shows circuit information
```

## Project Structure

```
spicelab-core/
├── core/               # C++ kernel library
│   ├── include/        # Public headers
│   ├── src/            # Implementation
│   └── tests/          # Unit tests
├── cli/                # Command-line interface
├── examples/           # Example circuits
└── openspec/           # Specifications
```

## Roadmap

- [x] MVP-0: Basic kernel (R, L, C, sources, transient)
- [x] MVP-1: Power electronics (switches, events, losses)
- [x] MVP-2: Advanced devices (MOSFETs, transformers)
- [ ] MVP-2b: Full features (thermal, gRPC API, Python bindings)
- [ ] MVP-3: Performance (SUNDIALS, parallel)

## License

MIT License
