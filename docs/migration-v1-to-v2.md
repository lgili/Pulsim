# Migration Guide: PulsimCore v1 to v2

This guide helps you migrate from PulsimCore v1 API to the new high-performance v2 API.

## Overview

The v2 API provides:
- **2x+ performance improvement** through SIMD optimization and cache-friendly layouts
- **Better convergence** with advanced aids (Gmin stepping, source stepping, pseudo-transient)
- **Improved accuracy** with proper BDF methods and adaptive timestep control
- **C++23 features** including concepts, std::expected, and compile-time optimization

## Accessing v2

### Python

```python
# v1 API (unchanged)
import pulsim
circuit = pulsim.Circuit()

# v2 API (new)
import pulsim.v2 as v2
# or
from pulsim import v2
```

### C++

```cpp
// v1 API
#include <pulsim/circuit.hpp>
using namespace pulsim;

// v2 API
#include <pulsim/v2/core.hpp>
using namespace pulsim::v2;
// or use the alias
namespace pv2 = pulsim::v2;
```

## Key Differences

### 1. Device Creation

**v1 (Python):**
```python
circuit = pulsim.Circuit()
circuit.add_resistor("R1", "node1", "node2", 1000.0)
circuit.add_capacitor("C1", "node2", "0", 1e-6, ic=0.0)
```

**v2 (Python):**
```python
# Standalone device objects
r = v2.Resistor(1000.0, "R1")
c = v2.Capacitor(1e-6, 0.0, "C1")

print(f"R1 resistance: {r.resistance()} Ohms")
print(f"C1 capacitance: {c.capacitance()} F")
```

### 2. Solver Configuration

**v1:**
```python
opts = pulsim.SimulationOptions()
opts.abstol = 1e-9
opts.reltol = 1e-3
opts.dt = 1e-6
```

**v2:**
```python
# Tolerance configuration
tols = v2.Tolerances.defaults()
tols.voltage_abstol = 1e-9
tols.current_abstol = 1e-12

# Newton solver options
opts = v2.NewtonOptions()
opts.max_iterations = 50
opts.initial_damping = 1.0
opts.tolerances = tols

# Timestep configuration
ts_config = v2.TimestepConfig.defaults()
ts_config.dt_min = 1e-12
ts_config.dt_max = 1e-6
ts_config.error_tolerance = 1e-4
```

### 3. DC Convergence Aids

**v2 introduces advanced convergence strategies:**

```python
# Configure DC convergence
dc_config = v2.DCConvergenceConfig()
dc_config.strategy = v2.DCStrategy.Auto  # Auto-select best strategy

# Or choose specific strategy
dc_config.strategy = v2.DCStrategy.GminStepping
dc_config.strategy = v2.DCStrategy.SourceStepping
dc_config.strategy = v2.DCStrategy.PseudoTransient

# Configure Gmin stepping
dc_config.gmin_config.initial_gmin = 1e-3
dc_config.gmin_config.final_gmin = 1e-12
dc_config.gmin_config.reduction_factor = 10.0

# Configure source stepping
dc_config.source_config.initial_scale = 0.0
dc_config.source_config.final_scale = 1.0
```

### 4. Integration Methods

**v2 provides BDF order control:**

```python
# BDF order configuration
bdf_config = v2.BDFOrderConfig()
bdf_config.min_order = 1
bdf_config.max_order = 2
bdf_config.enable_auto_order = True

# PI timestep controller
ts_config = v2.TimestepConfig()
ts_config.k_p = 0.075  # Proportional gain
ts_config.k_i = 0.175  # Integral gain
ts_config.safety_factor = 0.9
```

### 5. Validation and Benchmarking

**v2 includes built-in analytical solutions:**

```python
# RC circuit analytical solution
rc = v2.RCAnalytical(R=1000, C=1e-6, V_initial=0.0, V_final=5.0)
print(f"Time constant: {rc.tau()} s")
print(f"Voltage at t=tau: {rc.voltage(rc.tau())} V")

# RL circuit analytical solution
rl = v2.RLAnalytical(R=1000, L=1e-3, V_source=10.0, I_initial=0.0)
print(f"Final current: {rl.I_final()} A")

# RLC circuit with damping detection
rlc = v2.RLCAnalytical(R=10, L=1e-3, C=1e-6, V_source=10.0, V_initial=0.0, I_initial=0.0)
print(f"Damping type: {rlc.damping_type()}")  # Underdamped, Critical, Overdamped

# Compare simulation vs analytical
sim_data = [(0.0, 0.0), (0.001, 3.16), (0.002, 4.32)]  # (time, value) pairs
ana_data = [(0.0, 0.0), (0.001, 3.16), (0.002, 4.32)]
result = v2.compare_waveforms("RC_test", sim_data, ana_data, threshold=0.001)
print(f"Validation passed: {result.passed}, Max error: {result.max_error}")
```

### 6. High-Performance Features

**v2 exposes SIMD and solver configuration:**

```python
# Detect SIMD capability
level = v2.detect_simd_level()
print(f"SIMD level: {level}")  # SSE2, AVX, AVX2, AVX512, NEON

width = v2.simd_vector_width()
print(f"Vector width: {width} doubles")

# Linear solver configuration
ls_config = v2.LinearSolverConfig()
ls_config.pivot_tolerance = 1e-13
ls_config.reuse_symbolic = True
ls_config.deterministic_pivoting = True
```

## Enums Reference

### DeviceType
```python
v2.DeviceType.Resistor
v2.DeviceType.Capacitor
v2.DeviceType.Inductor
v2.DeviceType.VoltageSource
v2.DeviceType.CurrentSource
v2.DeviceType.Diode
v2.DeviceType.Switch
v2.DeviceType.MOSFET
v2.DeviceType.IGBT
v2.DeviceType.Transformer
```

### SolverStatus
```python
v2.SolverStatus.Success
v2.SolverStatus.MaxIterationsReached
v2.SolverStatus.SingularMatrix
v2.SolverStatus.NumericalError
v2.SolverStatus.ConvergenceStall
v2.SolverStatus.Diverging
```

### DCStrategy
```python
v2.DCStrategy.Direct           # No aids, direct Newton
v2.DCStrategy.GminStepping     # Add conductance to ground
v2.DCStrategy.SourceStepping   # Scale sources 0->1
v2.DCStrategy.PseudoTransient  # Pseudo-timestep for DC
v2.DCStrategy.Auto             # Automatic strategy selection
```

## Backward Compatibility

The v1 API remains fully functional. You can use both APIs in the same application:

```python
import pulsim

# v1 simulation
circuit = pulsim.Circuit()
circuit.add_resistor("R1", "in", "out", 1000.0)
result = pulsim.simulate(circuit, opts)

# v2 analytical validation
import pulsim.v2 as v2
rc = v2.RCAnalytical(1000, 1e-6, 0.0, 5.0)
```

## Performance Tips

1. **Use v2.DCStrategy.Auto** for automatic convergence strategy selection
2. **Enable symbolic reuse** in LinearSolverConfig for repeated solves
3. **Use BDF2** for most transient simulations (good stability/accuracy balance)
4. **Configure TimestepConfig.aggressive()** for faster but less accurate simulations
5. **Set deterministic_pivoting=True** for reproducible results

## Next Steps

- See [Performance Tuning Guide](performance-tuning.md) for optimization tips
- See [Determinism Guide](determinism.md) for reproducibility settings
