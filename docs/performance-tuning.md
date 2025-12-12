# Performance Tuning Guide

This guide covers optimization techniques for PulsimCore v2 simulations.

## SIMD Optimization

### Runtime Detection

PulsimCore automatically detects and uses the best available SIMD instruction set:

```python
import pulsim.v2 as v2

level = v2.detect_simd_level()
width = v2.simd_vector_width()

print(f"SIMD: {level}, Vector width: {width} doubles")
```

Supported levels (in order of performance):
- **AVX512**: 8 doubles per operation (Intel Skylake-X+, AMD Zen4+)
- **AVX2**: 4 doubles per operation (Intel Haswell+, AMD Excavator+)
- **AVX**: 4 doubles per operation (Intel Sandy Bridge+)
- **SSE4**: 2 doubles per operation
- **SSE2**: 2 doubles per operation (baseline x86-64)
- **NEON**: 2 doubles per operation (ARM64)

### Memory Alignment

For optimal SIMD performance, ensure data is aligned:

```cpp
// C++ - automatic alignment in v2 containers
alignas(64) double matrix_data[N];  // 64-byte alignment for AVX512
```

## Linear Solver Configuration

### Symbolic Factorization Reuse

Enable symbolic reuse for repeated solves with the same matrix pattern:

```python
cfg = v2.LinearSolverConfig()
cfg.reuse_symbolic = True  # Reuse symbolic factorization
cfg.detect_pattern_change = True  # Auto-detect pattern changes
```

This provides 2-3x speedup for transient simulations where the matrix structure is constant.

### Pivot Tolerance

Adjust pivot tolerance based on circuit characteristics:

```python
cfg = v2.LinearSolverConfig()

# For well-conditioned circuits (resistive)
cfg.pivot_tolerance = 1e-13

# For ill-conditioned circuits (power electronics)
cfg.pivot_tolerance = 1e-10
```

Lower tolerance = more accurate but slower. Higher tolerance = faster but may fail on ill-conditioned matrices.

## Newton Solver Tuning

### Iteration Limits

```python
opts = v2.NewtonOptions()
opts.max_iterations = 50  # Default: 50

# For simple circuits
opts.max_iterations = 20

# For highly nonlinear circuits (power electronics)
opts.max_iterations = 100
```

### Damping Strategy

```python
opts = v2.NewtonOptions()
opts.initial_damping = 1.0  # Full Newton step
opts.min_damping = 0.1      # Minimum damping factor
opts.auto_damping = True    # Enable automatic damping adjustment
```

- **initial_damping=1.0**: Full Newton step, fastest for well-behaved circuits
- **initial_damping=0.5**: Start with half steps for difficult convergence
- **auto_damping=True**: Automatically reduce damping on non-convergence

### Per-Variable Convergence

```python
opts = v2.NewtonOptions()
opts.check_per_variable = True  # Check each variable separately
```

Useful for mixed-domain simulations where voltage and current scales differ significantly.

## Tolerance Settings

### Default Tolerances

```python
tols = v2.Tolerances.defaults()
# voltage_abstol = 1e-6 V
# current_abstol = 1e-12 A
# residual_tol = 1e-9
```

### Accuracy vs Speed Trade-off

```python
# High accuracy (slower)
tols = v2.Tolerances()
tols.voltage_abstol = 1e-9
tols.current_abstol = 1e-15
tols.residual_tol = 1e-12

# Fast simulation (less accurate)
tols = v2.Tolerances()
tols.voltage_abstol = 1e-3
tols.current_abstol = 1e-9
tols.residual_tol = 1e-6
```

## Timestep Control

### PI Controller Tuning

The adaptive timestep uses a PI controller:

```python
cfg = v2.TimestepConfig()
cfg.k_p = 0.075  # Proportional gain
cfg.k_i = 0.175  # Integral gain
cfg.safety_factor = 0.9
```

- **Higher k_p**: More aggressive timestep changes
- **Higher k_i**: Smoother timestep adaptation
- **Lower safety_factor**: More conservative timestep selection

### Preset Configurations

```python
# Default: balanced accuracy and speed
cfg = v2.TimestepConfig.defaults()

# Conservative: smaller steps, better accuracy
cfg = v2.TimestepConfig.conservative()

# Aggressive: larger steps, faster simulation
cfg = v2.TimestepConfig.aggressive()
```

### Manual Limits

```python
cfg = v2.TimestepConfig()
cfg.dt_min = 1e-15  # Minimum timestep
cfg.dt_max = 1e-6   # Maximum timestep
cfg.dt_initial = 1e-9  # Starting timestep
```

## BDF Order Control

```python
bdf = v2.BDFOrderConfig()
bdf.min_order = 1  # BDF1 (backward Euler)
bdf.max_order = 2  # BDF2
bdf.initial_order = 1
bdf.enable_auto_order = True
```

- **BDF1**: A-stable, more damping, good for stiff circuits
- **BDF2**: More accurate, good for oscillatory circuits

## DC Convergence Strategies

### Strategy Selection

```python
dc_cfg = v2.DCConvergenceConfig()

# Let solver choose best strategy
dc_cfg.strategy = v2.DCStrategy.Auto

# Or specify explicitly
dc_cfg.strategy = v2.DCStrategy.GminStepping  # Add shunt conductance
dc_cfg.strategy = v2.DCStrategy.SourceStepping  # Scale sources 0->1
dc_cfg.strategy = v2.DCStrategy.PseudoTransient  # Fake timestep
dc_cfg.strategy = v2.DCStrategy.Direct  # No aids (fastest if it works)
```

### Gmin Stepping Tuning

```python
gmin = v2.GminConfig()
gmin.initial_gmin = 1e-3  # Start with large conductance
gmin.final_gmin = 1e-12   # Target conductance
gmin.reduction_factor = 10.0  # Reduce by 10x each step
```

Required steps = log10(initial/final) / log10(reduction_factor)

### Source Stepping Tuning

```python
src = v2.SourceSteppingConfig()
src.initial_scale = 0.0  # Start with zero sources
src.final_scale = 1.0    # End with full sources
src.initial_step = 0.1   # First step size
src.min_step = 0.01      # Minimum step size
```

## Performance Benchmarking

### Measuring Performance

```python
import time

timing = v2.BenchmarkTiming()
timing.name = "my_circuit"

start = time.perf_counter()
# ... run simulation ...
elapsed = time.perf_counter() - start

timing.iterations = num_timesteps
print(f"Time per step: {elapsed/num_timesteps*1000:.3f} ms")
```

### Export Results

```python
results = [result1, result2, result3]

# CSV for spreadsheets
csv = v2.export_benchmark_csv(results)

# JSON for automation
json_str = v2.export_benchmark_json(results)
```

## Best Practices Summary

1. **Enable symbolic reuse** for repeated matrix solves
2. **Use deterministic pivoting** for reproducible results
3. **Start with Auto DC strategy**, tune if needed
4. **Use BDF2** for most transient simulations
5. **Set appropriate tolerances** based on accuracy needs
6. **Use aggressive timestep config** for parameter sweeps
7. **Profile before optimizing** - identify actual bottlenecks

## Typical Speedup Factors

| Optimization | Speedup |
|--------------|---------|
| Symbolic reuse | 2-3x |
| AVX2 vs SSE2 | 1.5-2x |
| AVX512 vs AVX2 | 1.2-1.5x |
| Aggressive timestep | 2-5x |
| Lower tolerances | 1.5-3x |
| BDF1 vs BDF2 | 1.1-1.3x |
