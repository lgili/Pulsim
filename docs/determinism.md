# Determinism and Reproducibility Guide

This guide covers how to ensure reproducible simulation results in PulsimCore v1.

## Overview

Circuit simulation can produce non-deterministic results due to:
- Floating-point ordering in parallel operations
- LU decomposition pivot selection
- Random number generation in convergence aids
- Platform-specific math library implementations

PulsimCore v2 provides controls to eliminate these sources of non-determinism.

## Deterministic Pivoting

### Enable Deterministic Mode

```python
import pulsim.v2 as v2

cfg = v1.LinearSolverConfig()
cfg.deterministic_pivoting = True
```

When enabled:
- Pivot selection uses a consistent tie-breaking rule
- Results are identical across runs on the same platform
- Slight performance penalty (~5-10%)

### How It Works

Standard LU decomposition may choose different pivots when multiple elements have similar magnitudes. Deterministic pivoting uses the element's matrix position as a secondary sort key:

```
if abs(a[i]) == abs(a[j]):
    choose min(i, j)  # Deterministic tie-break
```

## Reproducible Initialization

### Fixed Seed for Random Restarts

```python
init_cfg = v1.InitializationConfig()
init_cfg.random_seed = 42  # Fixed seed
init_cfg.max_random_restarts = 5
init_cfg.random_voltage_range = 1.0
```

With a fixed seed:
- Random initial guesses are reproducible
- DC convergence attempts follow the same sequence
- Debugging is easier since failures are repeatable

### Disable Randomness

```python
init_cfg = v1.InitializationConfig()
init_cfg.max_random_restarts = 0  # No random restarts
init_cfg.use_zero_init = True     # Start from zero
```

## DC Convergence Reproducibility

### Deterministic Strategy

```python
dc_cfg = v1.DCConvergenceConfig()
dc_cfg.strategy = v1.DCStrategy.GminStepping  # Deterministic
dc_cfg.enable_random_restart = False
```

Strategy determinism:
- **Direct**: Fully deterministic
- **GminStepping**: Fully deterministic (fixed reduction sequence)
- **SourceStepping**: Fully deterministic (fixed scaling sequence)
- **PseudoTransient**: Fully deterministic
- **Auto**: May try different strategies on failure

### Logging for Debugging

```python
gmin = v1.GminConfig()
gmin.enable_logging = True  # Log each Gmin step

src = v1.SourceSteppingConfig()
src.enable_logging = True  # Log each source scale step

ptran = v1.PseudoTransientConfig()
ptran.enable_logging = True  # Log each pseudo-timestep
```

## Transient Simulation Reproducibility

### Fixed Timestep Mode

For maximum reproducibility, disable adaptive timestep:

```python
cfg = v1.TimestepConfig()
cfg.dt_min = 1e-6
cfg.dt_max = 1e-6  # Same as dt_min = fixed step
cfg.dt_initial = 1e-6
```

### Deterministic Adaptive Timestep

If adaptive timestep is needed, ensure deterministic error estimation:

```python
cfg = v1.TimestepConfig()
cfg.safety_factor = 0.9
cfg.growth_factor = 2.0
cfg.shrink_factor = 0.5
cfg.max_rejections = 10
```

The PI controller is fully deterministic given the same input sequence.

## BDF Order Control

### Fixed Order

```python
bdf = v1.BDFOrderConfig()
bdf.min_order = 2
bdf.max_order = 2  # Fixed at BDF2
bdf.enable_auto_order = False
```

### Deterministic Auto Order

```python
bdf = v1.BDFOrderConfig()
bdf.enable_auto_order = True
bdf.order_increase_threshold = 0.5
bdf.order_decrease_threshold = 1.5
bdf.steps_before_increase = 3
```

Auto order is deterministic given the same error estimates.

## Cross-Platform Reproducibility

### Challenges

Different platforms may produce slightly different results due to:
- FMA (fused multiply-add) availability
- Math library implementations (sin, cos, exp, log)
- Compiler optimizations

### Mitigation

1. **Use consistent compiler flags**:
   ```cmake
   # Strict IEEE 754 compliance
   target_compile_options(target PRIVATE -ffp-contract=off)
   ```

2. **Avoid platform-specific SIMD**:
   ```python
   # Force baseline SIMD level (not recommended for performance)
   # Compile with -DPULSIM_SIMD_LEVEL=SSE2
   ```

3. **Round-trip validation**:
   ```python
   # Export results with full precision
   result1 = simulate_on_platform_a()
   result2 = simulate_on_platform_b()

   # Compare with tolerance
   diff = max(abs(r1 - r2) for r1, r2 in zip(result1, result2))
   assert diff < 1e-10, "Cross-platform difference detected"
   ```

## Newton Solver Logging

### Enable History Tracking

```python
opts = v1.NewtonOptions()
opts.track_history = True  # Track iteration history
```

Access iteration data:

```python
result = v1.NewtonResult()
print(f"Iterations: {result.iterations}")
print(f"Final residual: {result.final_residual:.2e}")
print(f"Final weighted error: {result.final_weighted_error:.2e}")
```

### Status Codes

```python
status = result.status
msg = v1.solver_status_to_string(status)
print(f"Status: {msg}")

if status == v1.SolverStatus.MaxIterationsReached:
    print("Increase max_iterations or improve initial guess")
elif status == v1.SolverStatus.SingularMatrix:
    print("Check circuit topology for floating nodes")
elif status == v1.SolverStatus.Diverging:
    print("Enable damping or try convergence aids")
```

## Validation Export

### Export Results for Comparison

```python
# Run simulation
sim_data = [(t, v) for t, v in zip(times, voltages)]
ana_data = [(t, analytical(t)) for t in times]

result = v1.compare_waveforms("test", sim_data, ana_data)

# CSV for spreadsheet comparison
csv = v1.export_validation_csv([result])
with open("validation.csv", "w") as f:
    f.write(csv)

# JSON for automated comparison
json_str = v1.export_validation_json([result])
with open("validation.json", "w") as f:
    f.write(json_str)
```

## Complete Reproducibility Checklist

```python
import pulsim.v2 as v2

# 1. Linear solver
ls_cfg = v1.LinearSolverConfig()
ls_cfg.deterministic_pivoting = True

# 2. Initialization
init_cfg = v1.InitializationConfig()
init_cfg.random_seed = 42
init_cfg.max_random_restarts = 0  # Or fixed count with seed

# 3. DC convergence
dc_cfg = v1.DCConvergenceConfig()
dc_cfg.strategy = v1.DCStrategy.GminStepping  # Not Auto
dc_cfg.enable_random_restart = False

# 4. Newton solver
newton_opts = v1.NewtonOptions()
newton_opts.track_history = True

# 5. Timestep (optional: fixed for maximum reproducibility)
ts_cfg = v1.TimestepConfig()
ts_cfg.dt_min = 1e-6
ts_cfg.dt_max = 1e-6

# 6. BDF order (optional: fixed)
bdf_cfg = v1.BDFOrderConfig()
bdf_cfg.min_order = 2
bdf_cfg.max_order = 2
bdf_cfg.enable_auto_order = False
```

## Debugging Non-Reproducibility

1. **Identify the source**:
   - Run twice, compare results
   - Binary search to find divergence point

2. **Enable logging**:
   ```python
   gmin.enable_logging = True
   src.enable_logging = True
   ptran.enable_logging = True
   ```

3. **Check random seeds**:
   ```python
   print(f"Init seed: {init_cfg.random_seed}")
   ```

4. **Verify deterministic flags**:
   ```python
   assert ls_cfg.deterministic_pivoting == True
   assert init_cfg.random_seed != 0
   ```

5. **Compare platform details**:
   ```python
   print(f"SIMD level: {v1.detect_simd_level()}")
   print(f"Vector width: {v1.simd_vector_width()}")
   ```
