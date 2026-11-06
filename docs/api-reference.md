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
  .add_multi_winding_transformer(...)   # N windings, no ceiling
  .add_saturable_transformer_n(...)     # N secondaries on one gapped core
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
| `dt` | `float` or `None` | `None` | Fixed time step, seconds. Passing it selects the fixed-step `pwl` engine; omit it to let `engine='auto'` pick the engine and step. |
| `t_start` | `float` | `0.0` | Start time. |
| `switch_fn` | `Callable[[float], SwitchStateMask]` | all closed | Switch driver. |
| `b_extra_fn` | `Callable[[float], list[float]]` | `None` | Per-step constant residual extras. |
| `start_from_dc_op` | `bool` | `False` | Seed initial state from DC OP. |
| `enable_nonlinear_refresh` | `bool` or `None` | auto-detect | Force-enable Newton refresh. `None` = auto via `pool.has_nonlinear_devices()`. |
| `max_newton_iterations` | `int` | `0` (engine default) | Newton iteration cap. |
| `max_event_iterations` | `int` or `None` | `None` (engine default) | Event-iteration cap. |

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
| `add_voltage_source` | `(name, n_pos, n_neg, V)` |
| `add_current_source` | `(name, n_pos, n_neg, I)` |
| `add_pwm_voltage_source` | `(name, n_pos, n_neg, v_high, v_low, frequency, duty, phase=0)` |
| `add_sine_voltage_source` | `(name, n_pos, n_neg, v_dc, v_amplitude, frequency, phase=0)` |
| `add_pulse_voltage_source` | `(name, n_pos, n_neg, v_initial, v_pulsed, t_start, pulse_width, period=0, rise_time=0, fall_time=0)` |
| `add_vcvs` | `(name, in_pos, in_neg, out_pos, out_neg, gain)` |

### Passive devices

| Method | Signature |
|---|---|
| `add_resistor` | `(name, n_pos, n_neg, R_ohms)` |
| `add_capacitor` | `(name, n_pos, n_neg, C_farads, c0=None)` |
| `add_inductor` | `(name, n_pos, n_neg, L_henries, i0=None)` |
| `add_transformer` | `(name, p_from, p_to, s_from, s_to, L_p, L_s, k=1.0)` — a LINEAR coupled-inductor pair: the magnetising inductance is folded into `L_p` and `M`, there is no core, and nothing in it can saturate |
| `add_inductor_coupling` | `(name_a, name_b, k)` — couple two existing linear inductors |
| `add_ideal_transformer` | `(name, p_from, p_to, s_from, s_to, n)` — `v_s = n·v_p`, `i_p = −n·i_s`, `n = N_s/N_p`; transforms DC (put a magnetising inductance across the primary if the real device has one). `i(name)` is the secondary current |
| `add_saturable_transformer` | `(name, p_from, p_to, s_from, s_to, N_p, N_s, Ae, le, lg, mu_r0=2000, B_sat=0.35, L_leak_p=0, L_leak_s=0)` — T-model: per-winding leakage, ideal `n`, and ONE magnetising branch whose `λ(i)` comes from the gapped core (`N_p` turns, area `Ae` [m²], path `le` [m], TOTAL gap `lg` [m], initial `μ_r`, `B_sat ≡ μ₀M_s` [T]). `i(name + ".m")` is the magnetising current referred to the primary. Below the knee it equals `add_transformer` with `L_p = L_leak_p + L_m`, `M = n·L_m`, `L_s = n²·L_m + L_leak_s` |
| `add_gapped_core_inductor` | `(name, n_pos, n_neg, N, Ae, le, lg, mu_r0=2000, B_sat=0.35, knots=128)` — saturable inductor from core geometry; `λ(i)` generated by sweeping the core field through `B = μ₀(H + M_s·tanh(H/H₀))`, `H₀ = M_s/(μ_r0−1)`, with the exact slope at every knot |
| `add_saturable_inductor` | `(name, n_pos, n_neg, L_0, I_sat, L_residual=0)` — the analytic law `L(i) = L_res + (L_0−L_res)/(1+(i/I_sat)²)`, `λ(i) = L_res·i + (L_0−L_res)·I_sat·atan(i/I_sat)` |
| `add_hysteretic_core_inductor` | `(name, from_node, to_node, N, Ae, le, lg, Ms, a, alpha, c, k, substeps_min=8, M0=0)` — the Jiles-Atherton parameters `Ms, a, alpha, c, k` passed flat (C++ takes them as one `ja` struct; Python: `pulsim.add_hysteretic_inductor(builder, name=…, from_node=…, to_node=…, params=JilesAthertonParams, N_turns=…, l_m=…, A_core=…, l_gap=0, M0=0)`) — Jiles-Atherton hysteresis INSIDE the Newton loop: `M` integrated from the committed `(H_n, M_n)` to the `H` that Ampère with the gap fixes for the trial current, exact `L = dλ/di` in the Jacobian, branch direction held per step. Replaces the observer (`make_hysteretic_inductor_observer`, now refusing by name), whose EMF was sign-inverted and unstable above `q = L_M/(dt(R+2L_0/dt)) ≈ 0.5`. `HystereticInductor.bh_loop(res)` replays the kernel's integrator on the current trace |
| `add_saturable_inductor_table` | `(name, n_pos, n_neg, i_knots, lambda_knots)` — measured `(i, λ)` pairs for `i ≥ 0` from the origin, strictly increasing; odd-extended, monotone-cubic between knots, linear beyond the last |
| `add_multi_winding_transformer` | `(name, windings=[(from, to, L), …], k_matrix=[[…]])` — N-winding linear coupled-inductor transformer, **no ceiling** (the old `[2, 6]` was an argument check with nothing behind it). Couplings that are not realisable — an inductance matrix that is not positive definite, e.g. `k12 = k13 = 1` with `k23 < 1` — are refused by name |
| `add_saturable_transformer_n` | `(name, p_from, p_to, N_p, L_leak_p, secondaries=[(s_from, s_to, N_s, L_leak_s), …], Ae, le, lg, mu_r0=2000, B_sat=0.35)` — the saturable transformer with any number of secondaries: one gapped-core magnetising branch `name.m` on the primary and one ideal transformer `name.s<k>` per secondary, `n_k = N_k/N_p` |

### Switches + diodes

| Method | Signature |
|---|---|
| `add_switch` | `(name, n_pos, n_neg, g_on, g_off)` |
| `add_diode` | `(name, anode, cathode, g_on, g_off, V_th=0)` — binary switched diode; `V_th` is the commutation threshold, not a forward drop (the ON state is the bare conductance `g_on`) |
| `add_nonlinear_diode` | `(name, anode, cathode, params: IdealDiodeParams)` |

### Nonlinear actives

| Method | Signature |
|---|---|
| `add_mosfet` | `(name, drain, source, R_on=1e-3, R_off=1e9, body_diode=True, V_F=0.7)` — linear switch model PLUS an anti-parallel body diode `<name>_body`, ON by default. Both are switching branches, so each `add_mosfet` takes **two** bits of the `SwitchStateMask`; use `builder.switch_index_of(name)` |
| `add_mosfet_with_body_diode` | `(name, drain, source, R_on=1e-3, R_off=1e9, V_F=0.7, g_on_diode=1e3, g_off_diode=1e-9)` |
| `add_mosfet_level1` | `(name, drain, source, gate, K, V_T, lambda_=0.02, kappa=15.0, with_body_diode=True)` |
| `add_igbt` | `(name, collector, emitter, R_on=1e-2, R_off=1e9)` (linear) |
| `add_igbt_level1` | `(name, collector, emitter, gate, V_CE_sat=1.5, R_CE_sat=0.05, V_T=5.0, kappa=10.0, v_knee=0.01, with_fwd=False, tau_tail=0.0, k_tail=0.0)` |

## `DevicePool`

Opaque parameter store. Most methods are not exposed to Python — you go through `CircuitBuilder`. Exceptions:

| Method | Returns | Use case |
|---|---|---|
| `has_nonlinear_devices()` | `bool` | Auto-detect by `simulate()`. |

## Source-helper factories

All return a `Callable[[float], SwitchStateMask]` ready for `switch_fn=`.
Every factory takes `num_switches` — the mask size — and switch *bit
indices*. Get both from the builder rather than counting by hand:
`builder.graph.num_switches` and `builder.switch_index_of("Q1")`. Bits
follow the builder-call order of every switching branch, body diodes
included, so a hard-coded index silently moves when a device is added
or reordered. All non-driven bits stay OFF; diode bits are set by event
detection.

### `make_pwm_switch_fn(frequency, duty, switch_idx, num_switches, phase=0)`

Single-bit PWM. Sets bit `switch_idx` ON for the first `duty · T` of each period.

```python
pwm = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
```

### `make_dead_time_pwm_pair_fn(frequency, duty, hs_switch_idx, ls_switch_idx, num_switches, dead_time, phase=0)`

Complementary half-bridge with anti-shoot-through dead-time.

### `make_spwm_pair_fn(carrier_frequency, modulation_frequency, modulation_index, hs_switch_idx, ls_switch_idx, num_switches, dead_time, carrier_phase=0, modulation_phase=0)`

Sinusoidal modulation of the duty.

### `make_three_phase_spwm_fn(carrier_frequency, modulation_frequency, modulation_index, legs, num_switches, dead_time, modulation_phase=0, carrier_phase=0)`

Three-leg 2-level inverter. `legs = ThreePhaseLegIndices(hs_a, ls_a, hs_b, ls_b, hs_c, ls_c)` (positional; bit indices from `switch_index_of`).

### `make_phase_shift_full_bridge_fn(switching_frequency, phase_shift, leg_a_hs_idx, leg_a_ls_idx, leg_b_hs_idx, leg_b_ls_idx, num_switches, dead_time, carrier_phase=0)`

4-switch full-bridge ZVS phase-shifted modulation.

### `make_combined_switch_fn(num_switches, fns)`

Bit-wise OR of a list of switch_fn callables. Useful when you have an SPWM scheduler + an auxiliary low-frequency relay on the same mask.

## `SimulationOptions` / `SimulationResult`

```python
opts = p.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5)
opts.valid()                  # bool
opts.max_newton_iterations    # int (engine default if 0)
opts.max_event_iterations     # int
opts.tol_newton_dx            # float
opts.tol_newton_res           # float
opts.store_every              # int
```

`SimulationResult`:
- `result.times` — list[float]
- `result.states` — read-only 2-D `numpy.ndarray`, shape
  `(num_steps, state_size)` (zero-copy view)
- `result.commutation_events` — list[CommutationEvent] (event-detection details)
- `result.num_steps()` — int

## YAML loader

```python
loaded = p.load_yaml_file("examples/buck.yaml")
# or
loaded = p.load_yaml_string("""
circuit:
  devices:
    - {type: voltage_source, name: V1, from: in,  to: gnd, V: 5.0}
    - {type: resistor,       name: R1, from: in,  to: out, R: 1000.0}
    - {type: capacitor,      name: C1, from: out, to: gnd, C: 1.0e-6}
simulation:
  t_end: 5.0e-3
  dt: 1.0e-6
""")
loaded.builder              # CircuitBuilder
loaded.options              # SimulationOptions (t_start, t_end, dt, ...)
```

Pass it to `simulate(loaded.builder, loaded.options.t_end, loaded.options.dt)`.
The YAML describes the circuit and run settings only — build any gate
drive yourself (e.g. `make_pwm_switch_fn`) and pass it as `switch_fn=`.

## Parts library — `pulsim.lib`

```python
q = p.lib.igbt("AK40N120", paths=["./parts"])   # or PULSIM_PARTS_PATH
spec  = q.switch_spec(Tj=125.0)                   # E_on_table / E_off_table + Tj
cond  = q.conduction_spec(Tj=125.0)               # V_CE_sat / R_CE_sat fitted from the on-state table
chain = q.thermal().foster_stages()               # for add_foster_network
part  = p.lib.import_plecs_xml("~/Downloads/IKW40N120T2.xml")
p.lib.save_part(part, "parts/IKW40N120T2.yaml")
```

* Every part file carries a `provenance:` block (`source`, `source_ref`,
  `retrieved`, `method` ∈ transcribed | imported_plecs_xml | measured |
  vendor_model | synthetic); a file without one is refused by name.
* The parts shipped with Pulsim are **synthetic illustrations** (their
  switching-energy tables are straight lines through the origin). They
  are listed by `list_parts()` but `part()` refuses to resolve them by
  number unless `allow_synthetic=True`.
* Search path: directories passed as `paths=`, then `PULSIM_PARTS_PATH`,
  then the shipped data. The working directory is not searched; a user
  file shadowing a shipped one is announced once.
* A table taken at ONE junction temperature refuses `switch_spec(Tj=)`
  at another temperature unless `allow_tj_mismatch=True` (audit C.1's
  −35 % error). A diode's recovery table becomes `E_rr_curve` +
  `V_R_ref` at the stated `V_ref` and `Tj`.
* `import_plecs_xml` honours the PLECS sign conventions: a `Diode`
  file's switching losses sit in the −V/+I quadrant, a `… with Diode`
  file's diode in the +V/−I quadrant (split out as `Erec`, `v_on_diode`,
  `diode_spec()`). Formulas, class/type mismatches, package
  descriptions, gate-dependent conduction (without `conduction_index`)
  and thermal elements that do not match their branch are refused by
  name; `Variables` Min/Max become `Part.limits`.

## Resuming a run — `SolverSnapshot` / `simulate(resume_from=...)`

```python
first  = p.simulate(b, t_end=1e-3, dt=2e-7)          # engine='pwl'
second = p.simulate(b, t_end=5e-3, dt=2e-7, resume_from=first.final_snapshot)
```

`result.final_snapshot` carries the whole state at the end of a run:
the MNA vector `x`, the linear trapezoidal companion `history`, the
solver-owned `diode_on` bits, and one named field per stateful device
kind — `saturable_history` (7 reals per saturable, gapped-core or
hysteretic-core inductor, and per saturable-transformer magnetising
branch: λ, i, v, then the Jiles-Atherton H, M, branch direction and
sub-step count), `coss_history` (3), `lauritzen_history` (2),
`igbt_tail_history` (2), `pmsm_history` (9 per machine). Snapshots
pickle.

* `engine='pwl'` resumes **exactly**: a two-segment run reproduces the
  continuous one to machine precision.
* `engine='trbdf2'` / `'auto'` fills a snapshot and resumes from one to
  the run's own tolerance — the step controller's state (last accepted
  `h`, the LTE history) is not carried, so the resumed segment restarts
  from `h_init`. A TR-BDF2 snapshot resumes exactly on the fixed engine.
* `engine='dsed'` has no snapshot of its own and refuses `resume_from=`
  by name.
* An invalid snapshot, or one whose device histories do not match the
  circuit, is refused by name rather than resumed from zero. Builder
  initial conditions (`i0=`, `c0=`) are not applied when resuming: the
  snapshot is the state.

`steady_state` refuses a circuit containing any of these devices: its
one-period map is affine only while every dynamic device is a linear
capacitor or inductor. Settle with a transient run and use its
`final_snapshot` instead.

## SPICE import — `spice_to_builder(text_or_path, *, strict=False)`

`.MODEL` cards are applied: `D(IS N RS BV IBV)` → `add_shockley_diode`
with RS as a series resistor (BV mapped to the kernel's knee at IBV),
LTspice `D(Ron Roff Vfwd)` → `add_nonlinear_diode`, LEVEL-1 `NMOS`
→ `add_mosfet_level1(K = KP·W_eff/(2·L_eff)·M, V_T = VTO, λ = LAMBDA)`
with RD/RS series resistors and an explicit body diode when the card
gives IS/JS, LTspice `VDMOS` → the same with K = KP/2, Rds, and the
body diode from Is/N/Rb/BV. `.SUBCKT` instances are flattened with
scoped names (`XA.X1.R1`, node `xa.x1.mid`); `PARAMS:` defaults and
`k=v` on the X line are evaluated in the caller's scope; local
`.MODEL` cards shadow outer ones; `.GLOBAL` nodes and `0`/`gnd` are not
scoped. Refused by name: a `D`/`M` without a model, an undefined model
(`.INC`/`.LIB` are not followed), a non-LEVEL-1 MOSFET, PMOS/`pchan`,
a bulk away from the source with GAMMA, IKF, PWL reverse breakdown,
`AKO:`. Dropped parameters (TT/CJO, gate capacitances, RG, level-2/3
knobs) are listed in one warning; `strict=True` makes them refusals.

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
#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/pwm_switch_fn.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"   // for combined refresh
```
