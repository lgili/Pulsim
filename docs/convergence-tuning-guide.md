# Convergence Tuning Guide

This guide provides practical advice for achieving reliable convergence in circuit simulations, with a focus on power electronics applications.

## Table of Contents

1. [Understanding Convergence](#understanding-convergence)
2. [Common Convergence Problems](#common-convergence-problems)
3. [Strategy Selection](#strategy-selection)
4. [Tuning Newton Options](#tuning-newton-options)
5. [Tuning GMIN Stepping](#tuning-gmin-stepping)
6. [Tuning Source Stepping](#tuning-source-stepping)
7. [Timestep Control](#timestep-control)
8. [Circuit-Specific Tips](#circuit-specific-tips)
9. [Troubleshooting Checklist](#troubleshooting-checklist)

---

## Understanding Convergence

### What is Convergence?

Circuit simulation solves nonlinear equations using Newton-Raphson iteration. Convergence means the iteration finds a solution where:

1. **Residual is small**: The circuit equations are satisfied (KCL/KVL)
2. **Solution is stable**: Further iterations don't change the answer

### Why Do Circuits Fail to Converge?

1. **Floating nodes**: Nodes without DC path to ground
2. **Stiff equations**: Widely varying time constants (ns to ms)
3. **Discontinuous models**: Ideal switches, sharp nonlinearities
4. **Poor initial guess**: Starting far from solution
5. **Numerical issues**: Ill-conditioned matrices, round-off errors

---

## Common Convergence Problems

### Problem: DC Operating Point Fails

**Symptoms:**
- "Failed to find DC operating point"
- Newton iterations reach maximum without converging
- Solution oscillates or diverges

**Common Causes:**
1. Floating gate/base nodes in MOSFETs/BJTs
2. Forward-biased diodes with no current limit
3. Missing bias resistors
4. Unstable feedback loops

**Solutions:**
- Add high-value resistors (1-10 MΩ) to floating nodes
- Use GMIN stepping (enabled by default)
- Reduce initial voltage step limit

### Problem: Transient Simulation Stalls

**Symptoms:**
- Timestep becomes extremely small
- Simulation progress slows dramatically
- Many rejected timesteps

**Common Causes:**
1. Switch events creating discontinuities
2. Diode reverse recovery
3. Oscillatory numerical behavior
4. LTE tolerance too tight

**Solutions:**
- Increase LTE tolerance (1e-4 instead of 1e-6)
- Use power electronics timestep preset
- Add snubbers to switches
- Check for numerical oscillations

### Problem: Wrong DC Solution

**Symptoms:**
- Simulation runs but results are incorrect
- Voltages at unexpected values
- Multiple valid DC operating points

**Common Causes:**
1. Bistable circuits (flip-flops, latches)
2. Circuits with feedback
3. Incorrect initial conditions

**Solutions:**
- Use source stepping to guide to correct state
- Set explicit initial conditions
- Start from known good state

---

## Strategy Selection

### When to Use Each Strategy

| Strategy | Best For | Avoid When |
|----------|----------|------------|
| **Newton** | Well-posed circuits, quick solves | Floating nodes, stiff circuits |
| **GMIN Stepping** | Floating nodes, FET circuits | Already converging with Newton |
| **Source Stepping** | Bistable circuits, strong nonlinearity | Simple linear circuits |
| **Pseudo-Transient** | Very stiff circuits, last resort | Simple circuits (too slow) |
| **Auto** | General use (recommended) | Known circuit characteristics |

### Recommended Strategy Order

For most power electronics circuits:

```cpp
DCConvergenceConfig config;
config.strategy = DCStrategy::Auto;
config.strategy_order = {
    DCStrategy::Newton,          // Try fast method first
    DCStrategy::GminStepping,    // Handle floating nodes
    DCStrategy::SourceStepping,  // Handle strong nonlinearity
    DCStrategy::PseudoTransient  // Last resort
};
```

For MOSFET/IGBT circuits with floating gates:

```cpp
config.strategy_order = {
    DCStrategy::GminStepping,    // Start with GMIN
    DCStrategy::Newton,          // Then try Newton
    DCStrategy::SourceStepping
};
```

---

## Tuning Newton Options

### Voltage Limiting

Voltage limiting prevents Newton from taking steps that would put devices in unrealistic regions.

```cpp
NewtonOptions newton;
newton.enable_limiting = true;
newton.max_voltage_step = 0.5;   // Conservative default
newton.max_current_step = 1e-3;
```

**Tuning Tips:**
- Reduce `max_voltage_step` if Newton oscillates (try 0.3V or 0.1V)
- Increase if convergence is too slow (try 1.0V)
- For high-voltage circuits (>100V), scale accordingly

### Damping

Newton damping reduces step size to improve stability:

```cpp
newton.enable_damping = true;
newton.damping_factor = 0.7;    // Take 70% of Newton step
newton.min_damping = 0.1;       // Never less than 10%
```

**When to Use Damping:**
- Newton oscillates between two states
- Residual decreases then increases
- Near-singular Jacobian warnings

**When to Avoid:**
- Slows convergence on well-behaved circuits
- Can prevent reaching tight tolerances

### Tolerances

```cpp
newton.abs_tol = 1e-12;   // Absolute residual tolerance
newton.rel_tol = 1e-6;    // Relative change tolerance
newton.v_tol = 1e-6;      // 1 uV voltage tolerance
newton.i_tol = 1e-12;     // 1 pA current tolerance
```

**Tolerance Guidelines:**
- Tighter tolerances = more accuracy but harder convergence
- For power electronics: `v_tol = 1e-3` to `1e-6` is usually sufficient
- For precision analog: may need `v_tol = 1e-9`

---

## Tuning GMIN Stepping

### GMIN Configuration

```cpp
GminConfig gmin;
gmin.initial_gmin = 1e-3;      // Starting conductance
gmin.final_gmin = 1e-12;       // Floor conductance
gmin.reduction_factor = 10.0;  // Reduce by 10x each step
gmin.max_steps = 10;           // Maximum steps
```

### Tuning Guidelines

**If GMIN stepping is too slow:**
- Increase `initial_gmin` to 1e-2 (start closer to solution)
- Increase `reduction_factor` to 100 (fewer steps)

**If GMIN stepping fails:**
- Decrease `initial_gmin` to 1e-4 (more gradual approach)
- Decrease `reduction_factor` to 3 (smaller steps)
- Increase `max_steps` to 20

**For circuits with very high impedance:**
- Decrease `final_gmin` to 1e-15
- Use smaller reduction factor

### Example: MOSFET Gate Circuits

MOSFETs have nearly infinite gate impedance. Without GMIN:

```
Gate node has no DC path to ground
→ Matrix becomes singular
→ Newton fails
```

With GMIN:

```
GMIN adds tiny conductance to all nodes
→ Gate has path (1e-12 S)
→ Matrix is well-conditioned
→ Solution found
→ GMIN reduced to negligible level
```

---

## Tuning Source Stepping

### Source Stepping Configuration

```cpp
SourceSteppingConfig source;
source.initial_scale = 0.0;    // Start at zero
source.final_scale = 1.0;      // End at full value
source.step_size = 0.1;        // 10% increments
source.min_step = 0.01;        // Minimum 1% step
source.adaptive = true;        // Adjust based on convergence
```

### Tuning Guidelines

**If source stepping is too slow:**
- Increase `step_size` to 0.2 or 0.25
- Only for circuits known to converge well

**If source stepping fails midway:**
- Decrease `step_size` to 0.05
- Enable adaptive mode
- Check for bistability at intermediate scales

**For strongly nonlinear circuits:**
- Start with `step_size = 0.01`
- Use adaptive mode
- Accept longer solve time for reliability

### When Source Stepping Helps

1. **Transistor amplifiers**: Need correct bias point
2. **Oscillators**: Can settle to wrong state
3. **Power supplies with feedback**: Loop gain changes with level
4. **Circuits with protection diodes**: Clamp behavior changes

---

## Timestep Control

### Configuration Presets

```cpp
// For switching power supplies
auto config = AdvancedTimestepConfig::switching_preset();
// dt_min = 1e-12, dt_max = 1e-6, lte_target = 1e-4

// For power electronics (default)
auto config = AdvancedTimestepConfig::power_electronics_preset();
// dt_min = 1e-15, dt_max = 1e-3, lte_target = 1e-5
```

### Manual Tuning

```cpp
AdvancedTimestepConfig ts;
ts.lte_target = 1e-5;          // Local truncation error target
ts.lte_safety = 0.9;           // Safety factor
ts.target_newton_iters = 5;    // Target Newton iterations
ts.newton_weight = 0.3;        // Weight Newton vs LTE control
ts.dt_min = 1e-15;             // Minimum timestep
ts.dt_max = 1e-3;              // Maximum timestep
ts.max_growth_rate = 2.0;      // Max dt increase per step
ts.max_shrink_rate = 0.5;      // Max dt decrease per step
```

### Tuning for Specific Behaviors

**Simulation too slow (many small timesteps):**
- Increase `lte_target` (1e-4 or 1e-3)
- Increase `dt_max`
- Reduce `target_newton_iters`

**Simulation inaccurate:**
- Decrease `lte_target` (1e-6 or 1e-7)
- Reduce `dt_max`
- Increase `target_newton_iters`

**Timestep oscillates (grows then shrinks repeatedly):**
- Reduce `max_growth_rate` to 1.5
- Increase `lte_safety` to 0.95
- Check for numerical oscillation in solution

---

## Circuit-Specific Tips

### Buck Converter

```cpp
// Use switching preset
auto config = AdvancedTimestepConfig::switching_preset();

// Ensure timestep hits PWM edges
// (handled automatically by event detection)

// Common issues:
// - Inductor current discontinuity: Add small ESR
// - Output ripple: Ensure sufficient timesteps per period
```

### Boost Converter

```cpp
// Similar to Buck, but watch for:
// - Very high voltage transients when switch opens
// - Diode reverse recovery effects

newton.max_voltage_step = 1.0;  // Allow larger voltage swings
```

### Flyback Converter

```cpp
// Transformer adds coupling complexity
// - Ensure magnetizing inductance is specified
// - Watch for transformer saturation

gmin.initial_gmin = 1e-4;  // May need more GMIN
```

### H-Bridge

```cpp
// Cross-conduction can cause issues
// - Add dead-time between switches
// - Use voltage-controlled switch model

source.step_size = 0.05;  // Smaller steps for four-switch topology
```

### LLC Resonant Converter

```cpp
// Resonant circuits are sensitive to timestep
ts.lte_target = 1e-6;     // Tighter tolerance
ts.dt_max = 1e-7;         // Limit to fraction of resonant period

// May need many cycles to reach steady state
```

---

## Troubleshooting Checklist

### DC Convergence Failure

- [ ] Check for floating nodes (add bias resistors)
- [ ] Enable GMIN stepping
- [ ] Reduce Newton voltage step limit
- [ ] Try source stepping
- [ ] Check for model parameter errors
- [ ] Verify power supply polarities
- [ ] Look for missing ground connections

### Transient Convergence Failure

- [ ] Check switch model parameters
- [ ] Add snubber circuits
- [ ] Increase LTE tolerance
- [ ] Reduce maximum timestep
- [ ] Check for numerical oscillation
- [ ] Verify initial conditions
- [ ] Look for modeling discontinuities

### Slow Simulation

- [ ] Increase LTE tolerance
- [ ] Increase maximum timestep
- [ ] Use KLU solver instead of Eigen
- [ ] Enable symbolic factorization reuse
- [ ] Check for unnecessarily small component values
- [ ] Consider simplified device models

### Inaccurate Results

- [ ] Decrease LTE tolerance
- [ ] Decrease maximum timestep
- [ ] Verify device model parameters
- [ ] Check for truncation in output sampling
- [ ] Compare with simpler test cases
- [ ] Validate against analytical solutions

---

## Example: Tuning a Difficult Circuit

Consider a 100kHz buck converter with synchronous rectification:

```cpp
// Start with defaults
DCConvergenceConfig dc;
AdvancedTimestepConfig ts;

// Circuit fails to find DC point
// → Enable GMIN stepping (often needed for FETs)
dc.gmin.initial_gmin = 1e-3;

// DC converges, but transient fails at switch events
// → Use switching preset
ts = AdvancedTimestepConfig::switching_preset();

// Simulation is slow (10x real time)
// → Relax LTE tolerance
ts.lte_target = 1e-4;
ts.dt_max = 100e-9;  // 1% of switching period

// Results look good, but ripple seems wrong
// → Tighten tolerance for accuracy
ts.lte_target = 1e-5;

// Now runs at 5x real time with good accuracy
```

---

## See Also

- [Convergence Algorithms API](convergence-algorithms.md) - Full API reference
- [Device Models](device-models.md) - Device parameter reference
- [Performance Tuning](performance-tuning.md) - Speed optimization
