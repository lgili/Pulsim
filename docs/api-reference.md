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
  .add_shockley_diode(...)
  .add_switch(...)
  .add_mosfet(...)
  .add_mosfet_with_body_diode(...)
  .add_mosfet_level1(...)
  .add_igbt(...)
  .add_igbt_level1(...)
  .add_vcvs(...)
  .add_vccs(...) / .add_cccs(...) / .add_ccvs(...) / .add_gyrator(...)
  .add_sine_current_source(...) / .add_pulse_current_source(...)
  .add_pwl_current_source(...)        # table i(t)

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
| `builder` | `CircuitBuilder` or `LoadedCircuit` | required | The populated builder, or a `LoadedCircuit` from the YAML loader (then `t_end`, `dt`, `t_start` and `store_every` default to its `simulation:` block; see *YAML loader*). |
| `t_end` | `float` | required with a builder | End time, seconds; must be greater than `t_start` (a `ValueError` otherwise). |
| `dt` | `float` or `None` | `None` | Fixed time step, seconds. Passing it selects the fixed-step `pwl` engine; omit it to let `engine='auto'` pick the engine and step. |
| `t_start` | `float` | `0.0` | Start time. |
| `switch_fn` | `Callable[[float], SwitchStateMask]` | all closed | Switch driver. |
| `b_extra_fn` | `Callable[[float], list[float]]` | `None` | Per-step constant residual extras. |
| `start_from_dc_op` | `bool` | `False` | Seed initial state from DC OP. |
| `enable_nonlinear_refresh` | `bool` or `None` | `None` | `None`: the kernel stamps the Newton devices when the circuit has any. `True` forces the static chain. `False` on a circuit with a smooth / Shockley diode, MOSFET or IGBT raises `UnsupportedError` (it used to simulate them as open circuits). |
| `max_newton_iterations` | `int` | `0` (engine default) | Newton iteration cap. |
| `max_event_iterations` | `int` or `None` | `None` (engine default) | Event-iteration cap. |
| `engine` | `str` | `'auto'` | `'auto'`, `'pwl'`, `'trbdf2'` or `'exact'`. `'exact'` is the canonical name of the event-driven exact engine and `'dsed'` its alias: one engine, bit-identical runs; `result.engine_used` and every refusal's `engine` field read `'dsed'` (the capability-table column) until the default flip. |

`pulsim.exact_route(builder, t_end, switch_fn=None, **kw)` runs the
circuit on `engine='exact'` and classifies every mask it visits as
exact-eligible or not (add-exact-propagator-strategy D11, report only —
`'auto'` does not read it): `{"exact": bool, "reason": ..., "segments":
[{"mask", "exact", "reason", "device", "kind", "detail", "tier",
"n_segments"}, ...]}`. Reasons: `'device'` (a device with no linear / PWL
law, named), `'source'` (a PWM / pulse source or a `b_extra_fn`),
`'extraction'` (coupled inductors, a capacitor loop, an inductor
cut-set), `'cost'` (the mask's propagator outside its accuracy tiers).
The same rows ride on every exact run as `result.exact["segments"]`.

Returns a `SimulationResult` with parallel arrays:
- `result.times` — `list[float]`. With a fixed `dt`: length `floor((t_end − t_start) / dt) + 1`, or that divided by `store_every` (rounded up) when decimation is on. Without one the grid is **variable-step** and the spacing is not uniform — average with [the time-weighted helpers](#statistics-over-a-result-time-weighted), not `.mean()`.
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
| `add_vccs` | `(name, in_pos, in_neg, n_pos, n_neg, gm)` — `gm·(V(in_pos) − V(in_neg))` leaves at `n_pos` |
| `add_cccs` | `(name, control, n_pos, n_neg, gain)` — `gain·i(control)` leaves at `n_pos` |
| `add_ccvs` | `(name, control, out_pos, out_neg, rm)` — `V(out_pos) − V(out_neg) = rm·i(control)` |
| `add_gyrator` | `(name, p1_pos, p1_neg, p2_pos, p2_neg, g)` — `i1 = g·v2`, `i2 = −g·v1`, both into the `_pos` terminals; a `C` on one port is `C/g²` of inductance on the other |
| `add_sine_current_source` | `(name, n_pos, n_neg, i_dc, i_amplitude, frequency, phase=0)` |
| `add_pulse_current_source` | `(name, n_pos, n_neg, i_initial, i_pulsed, t_start, pulse_width, period=0, rise_time=0, fall_time=0)` |
| `add_pwl_current_source` | `(name, n_pos, n_neg, t, i, periodic=False)` — linear between knots, `i[0]` before `t[0]`, `i[-1]` after (or repeated with period `t[-1] − t[0]`) |

Every current source (DC, sine, pulse, table, `vccs`, `cccs`) drives a
positive value OUT of `n_pos` into the circuit: 10 mA into 1 kΩ from
`n_pos` to ground reads +10 V, and `res.i` of the source reads +10 mA.
`control` names a branch added earlier that carries a current unknown:
a voltage source of any kind (a 0 V one is the usual sense element), an
inductor, a VCVS, a CCVS or an ideal-transformer secondary; the current
is positive from that branch's first node to its second. `res.i` covers
the new kinds; `res.i(gyrator, port=1)` gives port 1 (port 2 by default).
A `ccvs` is a voltage source: `res.i` is its branch current, positive
from `out_pos` through it to `out_neg`.

All seven run on `pwl`, `trbdf2`, `dsed` and the DC operating point (a
waveform source at its `t = 0` value); the AC sweep takes the four
controlled ones, and the waveform current sources, being independent,
add no AC excitation. `trbdf2` lands on table knots and pulse corners.

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
| `add_diode` | `(name, anode, cathode, g_on, g_off, V_th=0, V_f=0)` — binary switched diode; `V_f` is the forward drop while ON (PSIM's diode voltage drop: `v = V_f + i/g_on`), `V_th` the turn-on threshold (effectively `max(V_th, V_f)`), not a drop |
| `add_nonlinear_diode` | **retired** — raises `RetiredDeviceError`: its law sources power for `0 < v < V_F0` (−1.39 A out of the anode at +0.636 V). Use `add_diode(..., V_f=)` or `add_shockley_diode`. `allow_non_passive=True` builds the legacy device (test fixtures only); YAML `type: nonlinear_diode` needs `allow_non_passive: true` |

### Nonlinear actives

| Method | Signature |
|---|---|
| `add_mosfet` | `(name, drain, source, R_on=1e-3, R_off=1e9, body_diode=True, V_F=0.7)` — linear switch model PLUS an anti-parallel body diode `<name>_body`, ON by default; `V_F` is its conduction drop (`v_SD = V_F + i/1e3 S`). Both are switching branches, so each `add_mosfet` takes **two** bits of the `SwitchStateMask`; use `builder.switch_index_of(name)` |
| `add_mosfet_with_body_diode` | `(name, drain, source, R_on=1e-3, R_off=1e9, V_F=0.7, g_on_diode=1e3, g_off_diode=1e-9)` — `V_F` is the body diode's conduction drop (`v_SD = V_F + i/g_on_diode`) |
| `add_mosfet_level1` | `(name, drain, source, gate, K, V_T, lambda_=0.02, kappa=15.0, with_body_diode=True, V_F=0.7)` — `V_F` is the body diode's conduction drop |
| `add_igbt` | `(name, collector, emitter, R_on=1e-2, R_off=1e9)` (linear) |
| `add_igbt_level1` | `(name, collector, emitter, gate, V_CE_sat=1.5, R_CE_sat=0.05, V_T=5.0, kappa=10.0, v_knee=0.01, with_fwd=False, tau_tail=0.0, k_tail=0.0, V_F=0.7)` — `V_F` is the freewheeling diode's conduction drop |

## `DevicePool`

Opaque parameter store. Most methods are not exposed to Python — you go through `CircuitBuilder`. Exceptions:

| Method | Returns | Use case |
|---|---|---|
| `has_nonlinear_devices()` | `bool` | Alias of `requires_newton()`: any kind the capability table (`pulsim.engine_capabilities()`) marks NewtonStatic / NewtonStateful. |
| `requires_static_refresh()` | `bool` | Any smooth / Shockley diode, MOSFET or IGBT (the kernel then builds the static Newton chain). |

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
- `result.step_loop_python_callbacks` — tuple naming the Python callables
  `simulate()` put in the step loop (`"switch_fn"`, `"b_extra_fn"`,
  `"step_observer"`, `"should_continue"`, `"closed_loops"`); `()` when the
  loop ran fully native (e.g. a `make_pwm_switch_fn` schedule)
- `result.control_tick_counts` / `result.control_tick_jitter_max` — ticks
  delivered per controller source, and the largest lateness on the `dt`
  grid
- `result.n_restart_steps` / `result.n_restart_factorizations` — SDIRK
  restart steps taken and the factorisations they caused, with
  `discontinuity_restart=True` (0 otherwise)
- `result.newton` — `NewtonCounters` of the run's Newton workspace
  (`pwl` / `trbdf2`): `solves`, `iterations`, `analyses`,
  `factorizations`, `partial_refactors`, `lm_attempts` and
  `iterations_per_solve` (the iteration count of every Newton call)
- `result.newton_roundoff_accepts` / `result.newton_roundoff_omega_max` /
  `result.newton_roundoff_row` — transient Newton solves accepted at row
  roundoff (row-wise backward error ≤ 8 eps) instead of aborting, the
  worst backward error and its MNA row (DC never accepts)

## Statistics over a result (time-weighted)

`result.times` is uniform only when you passed `dt`. Without one
(`engine='auto'`, the default, picks TR-BDF2) and on `engine='trbdf2'` /
`'dsed'`, the step controller packs samples around switching edges and
diode events — down to femtosecond steps after a turn-off — and
stretches them across the quiet stretches in between. Plain sample
statistics then weight each *sample* equally instead of each *second*.

Measured on an L–D–RC rectifier (20 V, 1 kHz sine; 111 µH, 2 Ω ‖ 10 µF),
average output over 2–3 ms:

| engine | samples | `v[t >= 2e-3].mean()` | `p.time_average(t, v, 2e-3)` |
|---|---:|---:|---:|
| `pwl`, `dt=2e-7` | 15001 | 6.3295 V | 6.3308 V |
| `trbdf2`, `dt=4e-6` (ceiling) | 1270 | **3.9677 V** | 6.3298 V |
| `auto`, no `dt` | 1432 | **4.5746 V** | 6.3300 V |

The engines agree; the sample mean does not. The same applies to
`np.sqrt((v**2).mean())`, and to slicing by index
(`v[int(0.7 * len(v)):]` is not "the last 30 % of the time").

```python
import pulsim as p

res = p.simulate(b, t_end=3e-3)                  # the rectifier above; variable-step

s = p.steady_state_stats(res, "o", t_from=2e-3)  # node, branch, or array
s.mean, s.rms, s.min, s.max, s.ripple_pp         # TimeStats

s = p.steady_state_stats(res, "Lc", last=1e-3)  # branch current, last period
p.time_average(res.times, res.v("o"), 2e-3, 3e-3)
p.time_rms(res.times, res.i("Lc"), t_from=2e-3)
t_w, v_w = p.time_window(res.times, res.v("o"), 2e-3, 3e-3)
```

| Function | Returns |
|---|---|
| `time_average(t, y, t_from=None, t_to=None)` | `(1/T) ∫ y dt` |
| `time_rms(t, y, t_from=None, t_to=None)` | `sqrt((1/T) ∫ y² dt)` |
| `time_window(t, y, t_from=None, t_to=None)` | samples inside the window, end points interpolated onto its edges |
| `steady_state_stats(result, what, *, t_from=, last=, t_to=None, kind=None)` | `TimeStats(mean, rms, min, max, ripple_pp, t_from, t_to)` |

What they do:

- **Integrate the piecewise-linear waveform through the samples
  exactly** — the curve a plot draws. Ramps and triangles come out
  exact; a sine converges as the square of the step.
- **Interpolate window edges** that fall between samples, so a window
  covers exactly `[t_from, t_to]`.
- **Repeated timestamps** (an event recorded on both sides of a jump)
  get zero weight, and the jump is kept: a window that *starts* on the
  event takes the post-jump value.
- **Refuse** (`ValueError`) an empty or reversed window, one that
  reaches outside the record, decreasing `t`, mismatched lengths, or
  NaN/inf inside the window. `steady_state_stats` also requires
  exactly one of `t_from=` / `last=`: averaging from `t = 0` would
  fold the start-up transient into a "steady-state" figure.
- `steady_state_stats(result, "name")` looks the name up as a node
  (`result.v`) and as a branch (`result.i`); a name that is both raises
  and asks for `kind='v'` / `kind='i'`. An array must be sampled on
  `result.times`; for a signal with its own time base (observer logs,
  `result.signal(...)`), call `time_average(t_sig, y_sig, ...)`.

Choose a window that spans a whole number of switching (or line)
periods; a fractional period biases the mean by up to half the ripple.

`min` / `max` need no weighting, but on a variable grid they include
every event-landing sample, sub-nanosecond excursions too. With an
explicit `dt` the grid is uniform and `.mean()` agrees with
`time_average` to about four significant digits (the difference is the
half weight the trapezoid gives the two end samples) — the helpers are
still the form that keeps working when the `dt` is dropped.

## Parameter sweeps — `sweep` / `monte_carlo`

```python
out = p.sweep(builder_factory, params={"R": [5, 10, 20]}, kpi_fn=kpi,
              parallel=True, n_workers=4, backend="auto",
              t_end=1e-3, dt=1e-7, switch_fn=pwm)
out.kpis["v_out"]; out.failed; out.backend   # "serial" | "thread" | "process"
```

`backend` applies when `parallel=True`:

- `"auto"` (default) runs the first job in the calling thread and reads
  its `res.step_loop_python_callbacks`. Empty (native schedule) → the
  other jobs run on a thread pool; the kernel releases the GIL, so they
  run in parallel. Any Python callable → a `multiprocessing.Pool`, since
  threads would serialise on the GIL.
- `"thread"` forces threads. `builder_factory` and `kpi_fn` then run
  concurrently in one interpreter: they must not share mutable state.
  Closures work (nothing is pickled).
- `"process"` forces processes (the model before 2.1): every argument
  must pickle, which the native schedules from `make_pwm_switch_fn` and
  friends do not.

Results are in job order and equal the serial run on every backend. A
job that raises is recorded in `out.failed` as `(index, message)` on the
serial and thread backends; the process backend raises. A run without a
`switch_fn` still installs a Python default mask today, so it reports
`("switch_fn",)` and `"auto"` keeps processes for it.

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
loaded.source               # the path, or '<string>'
loaded.has_simulation       # True: the YAML has a top-level simulation:
loaded.t_start, loaded.t_end, loaded.dt     # read-only, from options
loaded.node_names           # ['in', 'out'] (ground not listed)
loaded.switch_names         # switching branches in SwitchStateMask bit order
loaded.warnings             # non-fatal loader findings (also UserWarnings)

res = p.simulate(loaded)    # t_end / dt from the file; keywords override
```

`simulate(loaded)` takes `t_end`, `dt` (when the file sets one),
`t_start` and `store_every` from the `simulation:` block; any keyword
you pass wins. The other `simulation:` fields (Newton tolerances,
`integrator`, ...) stay on `loaded.options` for you to forward. A
`LoadedCircuit` with no top-level `simulation:` block (or no `t_end` in
it) is refused with a `ValueError` naming the file, unless you pass
`t_end=`.

Controlled and waveform-current-source types: `vccs` (`in_pos`,
`in_neg`, `from`, `to`, `gm`), `ccvs` (`control`, `out_pos`, `out_neg`,
`rm`), `cccs` (`control`, `from`, `to`, `gain`), `gyrator` (`p1_pos`,
`p1_neg`, `p2_pos`, `p2_neg`, `g`), `sine_current_source` (`from`, `to`,
`i_dc`, `i_amplitude`, `frequency`, `phase`), `pulse_current_source`
(`from`, `to`, `i_initial`, `i_pulsed`, `t_start`, `pulse_width`,
`period`, `rise_time`, `fall_time`), `pwl_current_source` (`from`, `to`,
`t` and `i` lists, or `file:` a two-column comma- or space-separated
table relative to the working directory, `#` comments allowed;
`periodic`). These types are strict: an unknown key is an error naming
the device.

The YAML describes the circuit and run settings only — build any gate
drive yourself (e.g. `make_pwm_switch_fn`) and pass it as `switch_fn=`.

Key checks. The loader accepts only `circuit:` and `simulation:` at the
top level and only `nodes:` / `devices:` under `circuit:`; anything else
is ignored with a `UserWarning` naming the key (also listed in
`loaded.warnings`). A `simulation:` block indented
under `circuit:` is refused with the fix spelled out (until 2.0.x it was
ignored and the file loaded with `t_end = dt = 0`). A
`schema: pulsim-v1` netlist is refused as such. Unknown keys inside
`simulation:` (e.g. the v1 spelling `tstop`) are not fatal: each is a
`UserWarning` and an entry of `loaded.warnings`.

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

## Periodic steady state — `periodic_steady_state(b, *, period, switch_fn=None, start_from=None)`

Newton shooting on the **true state** `x0` (capacitor voltages and
inductor currents, the dsed reduced state, labels in
`r.state_labels`) at the section `t = 0` — the rising edge of the
zero-phase built-in PWM helpers. Each Newton step is one period of
the dsed exact stepper with every event located in time, and the
Jacobian is the monodromy `M = Φ_K·S_K·…·S_1·Φ_0` (flow Jacobians
`e^{A h}` and saltation matrices), so the commutation pattern may
change between iterations and DCM works. The audit's DCM buck (24 V,
100 µH, 47 µF, 50 Ω, D = 0.3) converges from zero in 5 map
evaluations plus one verification period to `v_C = 8.99624708729 V`
(50-digit reference 8.99624708728943).

```python
sf = p.make_pwm_switch_fn(100e3, 0.3, b.switch_index_of("S"), b.graph.num_switches)
r = p.periodic_steady_state(b, period=1e-5, switch_fn=sf)
r.x0, r.state_labels, r.orbit        # state at t=0; one period (simulate result)
r.multipliers, r.stable              # Floquet multipliers; max|mu| < 1
r.pattern                            # [(event, instant/T)] at the orbit
r.n_period_runs, r.residual_history  # the cost meter, Newton residuals
sm = r.sampled_model(inputs=[("duty", "S")], outputs=["vout"])
sm.response([0.0])                   # H(z=1): d v_out / d D = 23.09 V
```

* **Other engines.** `engine="pwl", dt=T/100` (a gate edge at
  `t = 0`, `T/dt` an integer) or `engine="trbdf2"` (step ceiling `dt`,
  default `T/100`) shoot with that engine's own period runs — the fixed
  engine with its restart rule — and `M` by finite differences with
  Broyden updates: the orbit is the one that engine's transient
  reaches (DCM buck: 12 period runs on `pwl`, 1.2e-11 of the state
  scale from a 4000-period transient). `r.full_state` resumes a
  transient; `sensitivities=` and nonlinear devices are refused.
  `run_periodic_shooting` / `run_harmonic_balance` are deprecated
  delegates of it.
* **Multipliers.** `stable = max|μ| < 1`. An orbit with a multiplier
  outside the unit circle is returned with `stable=False`, and
  `r.unstable_multipliers` names each one: −1 side `flip
  (period-doubling)`, +1 side `fold`, complex `Neimark-Sacker`. A
  multiplier on the unit circle (a lossless LC loop, a floating
  capacitor) is refused: the orbit is not isolated, and the error
  names the states involved.
* **Resume** with `simulate(..., engine='dsed', initial_state=r.x0)`.
* **Small-signal model.** `sampled_model(inputs, outputs)` returns
  `x[k+1] = A x[k] + B u[k]`, `y = C x + D u` with `A = M` and `B =
  ∂P/∂u` from the same propagators: `("duty", switch)` moves the
  switch's turn-off edges by `T` per unit, `("source", name)` is a DC
  source's value. Outputs are state labels, inductor names or node
  names sampled at `t = 0`. `response(f)` is `H(e^{j2πfT})`, valid
  below `1/(2T)`.
* **Covered** (slice 1): linear R, L, C, controlled switches, PWL
  diodes, DC sources, sine sources whose period divides `period`,
  gate schedules that name their edges (`make_pwm_switch_fn` & co.,
  `NativePwm2Switch`, `NativeMultiMaskPwm`, a callable with
  `next_edge_after`). **Refused by name**: an opaque Python
  `switch_fn`, a schedule or source not periodic in `period`, PWM or
  pulse voltage sources, nonlinear and stateful devices, a grazing
  event, a marginal multiplier, an exhausted budget
  (`max_period_runs=60`, with the residual history, last pattern and
  last multipliers).
* `steady_state(...)` and `frequency_response(...)` **without `dt`**
  run on this solver (`frequency_response` takes `B` as a central
  difference of the true-state map in `u`); with `dt` they keep the
  fixed-engine companion-history solve, which raises "no periodic
  orbit found" in DCM.

## SPICE import — `spice_to_builder(text_or_path, *, strict=False)`

`.MODEL` cards are applied: `D(IS N RS BV IBV)` → `add_shockley_diode`
with RS as a series resistor (BV mapped to the kernel's knee at IBV),
LTspice `D(Ron Roff Vfwd)` → `add_diode(g_on = area/Ron, g_off = area/Roff, V_th = 0, V_f = Vfwd)`, LEVEL-1 `NMOS`
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
