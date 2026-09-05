# Python API reference

This page is a one-stop reference for the `pulsim` Python module. Everything here is also exposed in C++ under `pulsim::*` — the Python surface mirrors the C++ API.

## Module overview

```python
import pulsim as p

# High-level entry point.
p.simulate(builder, t_end, dt, **kwargs) -> SimulationResult

# Builder.
p.CircuitBuilder()
  .add_voltage_source(...)
  .add_current_source(...)
  .add_pwm_voltage_source(...)
  .add_sine_voltage_source(...)
  .add_pulse_voltage_source(...)
  .add_resistor(...)
  .add_capacitor(...)
  .add_inductor(...)
  .add_transformer(...)            # linear coupled-inductor pair, no core
  .add_inductor_coupling(...)
  .add_ideal_transformer(...)      # v_s = n v_p, i_p = -n i_s
  .add_saturable_transformer(...)  # leakage + ideal + gapped-core magnetising branch
  .add_saturable_inductor(...)
  .add_gapped_core_inductor(...)
  .add_saturable_inductor_table(...)
  .add_hysteretic_core_inductor(...)   # Jiles-Atherton, solved in the Newton loop
  .add_multi_winding_transformer(...)   # C++ only (not bound in Python)
  .add_diode(...)
  .add_nonlinear_diode(...)
  .add_switch(...)
  .add_mosfet(...)
  .add_mosfet_with_body_diode(...)
  .add_mosfet_level1(...)
  .add_igbt(...)
  .add_igbt_level1(...)
  .add_vcvs(...)

# Topology / device handles.
p.Graph, p.DevicePool, p.SwitchStateMask

# Cache + simulation.
p.PwlStateSpaceCache, p.SimulationOptions, p.SimulationResult,
p.CommutationEvent, p.run_transient

# Nonlinear diode params.
p.IdealDiodeParams

# YAML loader.
p.load_yaml_string, p.load_yaml_file, p.LoadedCircuit

# Source-helper factories.
p.make_pwm_switch_fn
p.make_dead_time_pwm_pair_fn
p.make_spwm_pair_fn
p.ThreePhaseLegIndices, p.make_three_phase_spwm_fn
p.make_phase_shift_full_bridge_fn
p.make_combined_switch_fn
```

## `simulate(builder, t_end, dt, **kwargs)`

The one-call high-level API (proposal #3.3). Builds the PWL cache, derives a default `switch_fn`, constructs `SimulationOptions`, runs the transient.

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `builder` | `CircuitBuilder` | required | The populated builder. |
| `t_end` | `float` | required | End time, seconds. |
| `dt` | `float` | required | Fixed time step, seconds. |
| `t_start` | `float` | `0.0` | Start time. |
| `switch_fn` | `Callable[[float], SwitchStateMask]` | all closed | Switch driver. |
| `b_extra_fn` | `Callable[[float], list[float]]` | `None` | Per-step constant residual extras. |
| `start_from_dc_op` | `bool` | `False` | Seed initial state from DC OP. |
| `enable_nonlinear_refresh` | `bool` or `None` | auto-detect | Force-enable Newton refresh. `None` = auto via `pool.has_nonlinear_devices()`. |
| `max_newton_iterations` | `int` | `0` (engine default) | Newton iteration cap. |
| `max_event_iterations` | `int` | `0` (engine default) | Event-iteration cap. |

Returns a `SimulationResult` with parallel arrays:
- `result.times` — `list[float]`, length `floor((t_end − t_start) / dt) + 1`, or that divided by `store_every` (rounded up) when decimation is on.
- `result.states` — read-only `numpy.ndarray` of shape
  `(num_steps, pool.state_size(graph))`. A **zero-copy view** over the
  kernel's contiguous sample buffer (v2.0): indexing, slicing,
  iteration and `np.asarray` behave as before, but access is O(1)
  instead of rebuilding a list of per-sample arrays. Call `.copy()`
  for a writable array.
- `result.num_steps()` — convenience accessor.
- `result.states_bytes` — bytes held by the sample buffer.

Pass `simulate(..., store_every=m)` — or set it on a
`SolverOptions` bundle, on a raw `SimulationOptions`, or as
`simulation.store_every` in YAML — to record only every m-th
step. The solver still integrates at `dt`; the recorded grid stays
strictly uniform at `m · dt` (so FFT / harmonic analysis on the
result remains valid), and memory drops by the same factor.
`opts.expected_sample_count()` reports what a run will store.

## `CircuitBuilder`

Top-level fluent builder. Stores a `Graph` (topology) + `DevicePool` (per-device parameters) + an internal node-name → node-id map.

### Node accessors

| Method | Returns | Notes |
|---|---|---|
| `b.node(name)` | `int` | Get-or-create. `"gnd"`, `"GND"`, `"0"` all map to `Graph.ground()`. |
| `b.node_id_of(name)` | `int` | Throws if name doesn't exist. |
| `b.graph` | `Graph` | The underlying topology. |
| `b.pool` | `DevicePool` | The underlying parameter pool. |
| `b.num_branches` | `int` | Branch count. |

### Source devices

| Method | Signature |
|---|---|
| `add_voltage_source` | `(name, from, to, V)` |
| `add_current_source` | `(name, from, to, I)` |
| `add_pwm_voltage_source` | `(name, from, to, *, v_high, v_low, frequency, duty, phase=0)` |
| `add_sine_voltage_source` | `(name, from, to, *, v_dc=0, v_amplitude, frequency, phase=0)` |
| `add_pulse_voltage_source` | `(name, from, to, *, v_initial, v_pulsed, t_start, pulse_width, period=0, rise_time=0, fall_time=0)` |
| `add_vcvs` | `(name, *, in_pos, in_neg, out_pos, out_neg, gain)` |

### Passive devices

| Method | Signature |
|---|---|
| `add_resistor` | `(name, from, to, R_ohms)` |
| `add_capacitor` | `(name, from, to, C_farads)` |
| `add_inductor` | `(name, from, to, L_henries)` |
| `add_transformer` | `(name, primary_pos, primary_neg, secondary_pos, secondary_neg, L_p, L_s, k)` — a LINEAR coupled-inductor pair: the magnetising inductance is folded into `L_p` and `M`, there is no core, and nothing in it can saturate |
| `add_inductor_coupling` | `(name_a, name_b, k)` — couple two existing linear inductors |
| `add_ideal_transformer` | `(name, p_from, p_to, s_from, s_to, n)` — `v_s = n·v_p`, `i_p = −n·i_s`, `n = N_s/N_p`; transforms DC (put a magnetising inductance across the primary if the real device has one). `i(name)` is the secondary current |
| `add_saturable_transformer` | `(name, p_from, p_to, s_from, s_to, N_p, N_s, Ae, le, lg, mu_r0=2000, B_sat=0.35, L_leak_p=0, L_leak_s=0)` — T-model: per-winding leakage, ideal `n`, and ONE magnetising branch whose `λ(i)` comes from the gapped core (`N_p` turns, area `Ae` [m²], path `le` [m], TOTAL gap `lg` [m], initial `μ_r`, `B_sat ≡ μ₀M_s` [T]). `i(name + ".m")` is the magnetising current referred to the primary. Below the knee it equals `add_transformer` with `L_p = L_leak_p + L_m`, `M = n·L_m`, `L_s = n²·L_m + L_leak_s` |
| `add_gapped_core_inductor` | `(name, from, to, N, Ae, le, lg, mu_r0=2000, B_sat=0.35, knots=128)` — saturable inductor from core geometry; `λ(i)` generated by sweeping the core field through `B = μ₀(H + M_s·tanh(H/H₀))`, `H₀ = M_s/(μ_r0−1)`, with the exact slope at every knot |
| `add_saturable_inductor` | `(name, from, to, L_0, I_sat, L_residual=0)` — the analytic law `L(i) = L_res + (L_0−L_res)/(1+(i/I_sat)²)`, `λ(i) = L_res·i + (L_0−L_res)·I_sat·atan(i/I_sat)` |
| `add_hysteretic_core_inductor` | `(name, from, to, N, Ae, le, lg, ja, substeps_min=8, M0=0)` (C++; Python: `pulsim.add_hysteretic_inductor(builder, name=…, from_node=…, to_node=…, params=JilesAthertonParams, N_turns=…, l_m=…, A_core=…, l_gap=0, M0=0)`) — Jiles-Atherton hysteresis INSIDE the Newton loop: `M` integrated from the committed `(H_n, M_n)` to the `H` that Ampère with the gap fixes for the trial current, exact `L = dλ/di` in the Jacobian, branch direction held per step. Replaces the observer (`make_hysteretic_inductor_observer`, now refusing by name), whose EMF was sign-inverted and unstable above `q = L_M/(dt(R+2L_0/dt)) ≈ 0.5`. `HystereticInductor.bh_loop(res)` replays the kernel's integrator on the current trace |
| `add_saturable_inductor_table` | `(name, n_pos, n_neg, i_knots, lambda_knots)` — measured `(i, λ)` pairs for `i ≥ 0` from the origin, strictly increasing; odd-extended, monotone-cubic between knots, linear beyond the last |
| `add_multi_winding_transformer` | `(name, windings, k_matrix)` — **C++ only**; not bound in Python. For a multi-output saturable design compose `add_saturable_transformer` primaries or several `add_ideal_transformer` secondaries on one magnetising node |

### Switches + diodes

| Method | Signature |
|---|---|
| `add_switch` | `(name, from, to, g_on, g_off)` |
| `add_diode` | `(name, from, to, g_on, g_off, V_th=0)` |
| `add_nonlinear_diode` | `(name, from, to, params: IdealDiodeParams)` |

### Nonlinear actives

| Method | Signature |
|---|---|
| `add_mosfet` | `(name, drain, source, R_on=1e-3, R_off=1e9)` (linear switch model) |
| `add_mosfet_with_body_diode` | `(name, drain, source, R_on=1e-3, R_off=1e9, V_F=0.7, g_on_diode=1e3, g_off_diode=1e-9)` |
| `add_mosfet_level1` | `(name, drain, source, gate, K, V_T, lambda_=0.02, kappa=15.0, with_body_diode=False)` |
| `add_igbt` | `(name, collector, emitter, R_on=10e-3, R_off=1e9)` (linear) |
| `add_igbt_level1` | `(name, collector, emitter, gate, V_CE_sat=2.0, R_CE_sat=0.05, V_T=5.0, kappa=10.0)` |

## `DevicePool`

Opaque parameter store. Most methods are not exposed to Python — you go through `CircuitBuilder`. Exceptions:

| Method | Returns | Use case |
|---|---|---|
| `has_nonlinear_devices()` | `bool` | Auto-detect by `simulate()`. |

## Source-helper factories

All return a `Callable[[float], SwitchStateMask]` ready for `switch_fn=`.

### `make_pwm_switch_fn(switch_index, frequency, duty, phase=0)`

Single-bit PWM. Sets bit `switch_index` ON for the first `duty · T` of each period.

### `make_dead_time_pwm_pair_fn(high_switch_index, low_switch_index, frequency, duty, dead_time, phase=0)`

Complementary half-bridge with anti-shoot-through dead-time.

### `make_spwm_pair_fn(high_switch_index, low_switch_index, carrier_frequency, modulating_frequency, modulation_index, phase=0)`

Sinusoidal modulation of the duty.

### `make_three_phase_spwm_fn(leg_indices, carrier_frequency, modulating_frequency, modulation_index, phase=0)`

6-switch inverter. `leg_indices = ThreePhaseLegIndices(a_high, a_low, b_high, b_low, c_high, c_low)`.

### `make_phase_shift_full_bridge_fn(diag1_switch_index, diag2_switch_index, frequency, phase_shift)`

4-switch full-bridge ZVS phase-shifted modulation.

### `make_combined_switch_fn(*switch_fns)`

Bit-wise OR of multiple switch_fn callables. Useful when you have an SPWM scheduler + an auxiliary low-frequency relay on the same mask.

## `SimulationOptions` / `SimulationResult`

```python
opts = p.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5)
opts.valid()                  # bool
opts.max_newton_iterations    # int (engine default if 0)
opts.max_event_iterations     # int
opts.newton_tol_dx            # float
opts.event_tol_v              # float
opts.event_tol_i              # float
```

`SimulationResult`:
- `result.times` — list[float]
- `result.states` — read-only 2-D `numpy.ndarray`, shape
  `(num_steps, state_size)` (zero-copy view)
- `result.commutations` — list[CommutationEvent] (event-detection details)
- `result.num_steps()` — int

## YAML loader

```python
loaded = p.load_yaml_file("examples/buck.yaml")
# or
loaded = p.load_yaml_string("""
nodes: [n0]
components: ...
""")
loaded.builder              # CircuitBuilder
loaded.switch_fn            # combined Callable
loaded.t_start, loaded.t_end, loaded.dt
```

Pass the result straight to `simulate(loaded.builder, loaded.t_end, loaded.dt, switch_fn=loaded.switch_fn)`.

## C++ namespace mapping

| Python | C++ |
|---|---|
| `pulsim.CircuitBuilder` | `pulsim::builder::CircuitBuilder` |
| `pulsim.PwlStateSpaceCache` | `pulsim::pwl::PwlStateSpaceCache` |
| `pulsim.run_transient` | `pulsim::solver::run_transient` |
| `pulsim.make_pwm_switch_fn` | `pulsim::sources::make_pwm_switch_fn` |

C++ code lives header-only under `core/include/pulsim/`. Includes you'll commonly need:

```cpp
#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/pwl_state_space_cache.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/make_pwm_switch_fn.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"   // for combined refresh
```
