# Decisions taken while implementing the first slice

Slice = tasks 1–6 (groups 7 and the later changes are not in it). The
owner was away; the design's open questions were delegated. Each entry:
question → option → evidence → how to reverse. Numbers are measured on
the CCM buck of task 1.3 (24 V, 100 µH, 47 µF, 1 Ω, 10 mΩ inductor
resistance, `g_on = 1e3`, `g_off = 1e-9`, 100 kHz, D = 0.5) unless
stated.

## 1. How the KPIs integrate one period

→ Piecewise-linear, time-weighted, on the run's own grid, with the
right limit of every event inserted as a repeated instant
(`pulsim.time_average` / `time_window`, `device_loss_summary`). On the
`pss` route the KPI period is a separate `dsed` run from the orbit's
`x0` with the step capped at `T / samples_per_period`; default
`samples_per_period = 2000`.

Evidence (relative error against the matrix-exponential reference,
`python/tests/_design_reference.py`):

| samples / period | efficiency | total loss | ripple (sample max) | ripple (vertex) | evaluate |
|---|---|---|---|---|---|
| 100 | 8.5e-10 | −7.6e-11 | −1.8e-4 | 3.4e-7 | 5 ms |
| 500 | 3.4e-11 | −3.0e-12 | −7.5e-6 | −2.8e-9 | 4 ms |
| 2000 | 1.9e-12 | −8.2e-15 | −5.4e-7 | −6.0e-11 | 4–7 ms |
| 10000 | −1.8e-12 | −3.9e-13 | −5.2e-9 | −6.5e-10 | 9–12 ms |

Without the right limits a trapezoid over the gate edge smears the
switch current's jump over one step: the input power was 1 % high at
100 samples per period and 0.1 % at 1000 (142.58 W against 142.43 W),
first order in the step.

→ Reverse: change the default; the KPIs stay time-weighted.

## 2. Ripple extrema

→ A peak-to-peak is the difference of two extrema; an extremum that is a
smooth interior maximum of the samples (three distinct instants around
it, no event) is replaced by the vertex of the parabola through those
three samples; an extremum at an event (a kink: the inductor current at
a gate edge) is the sample itself. The sampled maximum is off by
O(h²) (−5.4e-7 of the ripple at 2000 samples, table above), the vertex
by O(h³) (−6.0e-11). → Reverse: `_p2p` in `design.py`.

## 3. Loss categories

→ A switch's or diode's resistor-pair dissipation (`P_avg` of
`device_loss_summary`) is `conduction`; when `loss_options` carries
`conduction_specs`, the datasheet model `P_cond_model_avg` replaces it
(it is "a separate estimate, not an addition", `losses.py`). A
switch's `P_sw_avg` is `switching`, a diode's is `recovery`; an
inductor's `P_core_avg` is `core`. A resistor that is not a load is
`resistive` — not `copper` or `esr` as design D1 lists: the evaluator
cannot know whether a resistor is a winding, an ESR or a snubber.
→ Reverse: a user map `{device: category}` in `Evaluator`.

## 4. Efficiency

→ `P_out` = the dissipation of the devices named in `loads`;
`P_in` = `−Σ V · mean(i)` over the DC voltage sources (the MNA branch
current flows + → − inside the source), measured on the source, not
inferred from the balance. On the oracle `P_in − P_out − P_loss` is
2e-12 relative. The efficiency KPI is `P_out / (P_out + P_loss)`
(decision 11 says why). A source of another kind (sine, PWM, pulse) is
**refused** (`UnsupportedError(engine='design.evaluate',
feature='sources', device=name)`), never guessed; name the DC sources
in `sources=`. → Reverse: integrate `v(t)·i(t)` for waveform sources.

## 5. What is `failed`

→ (a) A `RuntimeError` raised by an engine call — the kernel's
non-convergence (`did not settle`, chattering, step floor,
`SimulationAborted`) is a `RuntimeError`; a refusal is an
`UnsupportedError`, a `ValueError`, so it is never caught here;
cancellation (`pulsim.Cancelled`) is re-raised. Only the engine call is
wrapped, never `build` or the KPI code, so a bug there propagates.
(b) A `ValueError` of `periodic_steady_state` whose message is one of
its non-convergence diagnostics (no orbit within the budget, no
progress, singular `I − M`, a non-isolated orbit, a grazing event, a
failed period run, an irreproducible map). Its other `ValueError`s are
configuration errors (a schedule that is not periodic in `period`) and
propagate. (c) A non-finite KPI (`P_in = 0`). (d) `not_settled`.

(b) is now typed: `periodic_steady_state` raises
`pulsim.ConvergenceError` (still a `ValueError`) at its seven
non-convergence sites and the evaluator catches that class; the message
list is gone (see "Missing from the kernel", items 1 and 2).
→ Reverse: none.

Merged with integration 1aef1a52 (slice 3): one mapping. A singular
`I − M` surfacing as `numpy.linalg.LinAlgError` is caught with
`ConvergenceError` (failed). An unextractable state space is
`UnsupportedError(feature="state_space")` raised by `_Solver` itself,
so the explicit and the auto route refuse it without matching a
message; the capability probe runs inside `_engine`. Accepted-with-
error runs stay `ok` with a counter in the record: trbdf2
`forced_accepts` (error estimate above tolerance at the step floor) and
`newton_roundoff_accepts` (Newton stopped at roundoff) and
`resolution_warnings` (the fixed-step `ResolutionWarning`), next to
integration's `event_iteration_breaches` and `chatter_breaks`. Neither
is failed: a forced accept is the sliver between two events and its
error is bounded by the step floor times the slope, and a roundoff
accept has every row's residual within 8 eps of its terms. What would
make them failed: a forced accept whose step is more than 1e-3 of the
period (its error is then no longer a sliver) or a roundoff accept whose
`newton_roundoff_omega_max` exceeds 8 eps; neither is reported per step
today, so the threshold cannot be applied yet.
→ Reverse: move the two counters into `_raise_if_suspect`.

Batch 6c (2026-11-13): the four warnings that still landed `ok` with no
trace in the record, decided one by one. A warning that says the
RESULT may be wrong is `failed`; one that says the run was more careful
than asked is a counter.

| warning (`simulate`) | record | evidence |
|---|---|---|
| a step re-taken at `dt/2^k` (`res.dt_retries`) | `ok`, `counters["dt_retries"]` | the interval is integrated more finely than asked; the grid is unchanged |
| gate edge taken at the next grid point under sub-step correction (`res.counters["unlocated_gate_steps"]`) | `failed`, `unlocated_gate_edges` | CCM buck, pwl, `dt = 4e-7`, opaque gate: was `ok` with `v_out` 4.4674 V; the oracle'd schedule gives 4.3033 V (3.8 %) |
| `dsed` polling a schedule without `next_edge_after` | `ok`, `counters["polled_switch_fn_runs"]` | each polled change is bisected to its instant; the same CCM buck's KPIs equal the oracle'd schedule's to 1e-6 (test); only a pulse narrower than `dt_max/10` is missed, and the `pss` route refuses such a schedule outright |
| nodes reached only through switches/diodes (trbdf2 `probe_floor_used > 0`) | `failed`, `floating_subnet` | the common mode is set by `g_off`: E3 bridge variants read 323.94 V vs 324.51 V and a 76.7 kV sample on `dn` (comment at the warning) |

Found on the way: on `pwl`/`trbdf2` the transient route's KPI window
was `[(n−1)T, nT]` while a `dt` grid summed to `nT` can end an ulp
short (`4e-7 × 200 = 7.999999999999999e-05`), so `evaluate` raised
`ValueError` ("reaches outside the simulated span") and stopped the
batch. The window now ends at the run's last sample when it is within
`1e-9·T` of `nT`.
→ Reverse: drop the two `_Failed` branches in `_raise_if_suspect`.

With the TR-BDF2 track merged (batch 6c, job 3): its
`n_roundoff_floor_accepts` (a step accepted within the error
estimate's own roundoff floor, a `g_off`-held node) is
`counters["roundoff_floor_accepts"]`, `ok`; its run budget (more than
130 consecutive Newton roundoff accepts, `NewtonRoundoffBudgetExceeded`)
reaches Python as `SimulationAborted`, a `ConvergenceError`, so the
record is `failed` like any engine non-convergence.

## 6. `build` and `period`

→ `build(design)` returns a builder or `(builder, switch_fn)`: the
steady-state solver and every engine need the schedule beside the
circuit, and the schedule is a design variable (duty, frequency).
`period` is a float or `period(design) -> T` (the worked example sweeps
`f_sw`). → Reverse: a `Circuit` object carrying both once
`add-immutable-circuit-ir` lands (D7).

## 7. Adaptors

→ `pulsim.design.as_objectives(record, names, *, maximise=(),
penalty=1e30)` and `as_constraints(record, names, *, penalty=1e30)`,
module functions (D3 says `pulsim.design.adapters`; a namespace of two
functions earns nothing). `maximise` negates, so every component is
minimised. `as_constraints` raises `KeyError` on a name the record has
no violation for — a misspelt constraint must not read as feasible.
→ Reverse: re-export them under an `adapters` namespace.

## 8. User KPIs

→ The default KPIs are always computed; `kpis=` adds
`Kpi(name, fn(period))` where `period` is a `SteadyPeriod` (result,
builder, design, window, seed) with time-weighted `average` / `rms`
helpers. → Reverse: let `kpis=` replace the defaults.

## 9. Settle criterion of `method='transient'` (open question of the design)

→ The run doubles from `min_periods = 16` (re-run from `t = 0`: at most
2× the cost of the final run, and deterministic) until the estimated
distance to the engine's own orbit is below `tol`, relative to each
state's magnitude over the last period (inductor currents, capacitor
voltages and every column of the engine's state vector, so a
stateful device's internal state counts; a Python `switch_fn`'s own
memory is invisible and is the user's to settle, review H2); `max_periods = 4096`, then `failed`, `not_settled`.
Default `tol = 1e-6`. The estimate is the larger of two geometric
extrapolations per state: over the last period (`d1·ρ/(1 − ρ)`) and
over the last half (`d_a·r/(1 − r)`, quarters). The state is read at
`kT` when the engine landed every period start, else as the mean over
the period ending there (interpolating across an edge that falls
between samples read 1e-4 of the state).

Evidence, DCM buck from rest (multiplier 0.989), error of `v_C` at
the section against the 50-digit orbit, relative to 9 V:

| engine | tol | periods | estimate | true error |
|---|---|---|---|---|
| dsed | 1e-4 | 1024 | 1.16e-5 | 1.15e-5 |
| dsed | 1e-6 | 2048 | 1.4e-10 | 1.4e-10 |
| trbdf2 | 1e-4 | 1024 | 1.15e-5 | 3.6e-5 |
| trbdf2 | 1e-6 | 2048 | 1.4e-10 | 2.2e-5 |

The trbdf2 rows show the second limit: `tol` bounds the distance to
the ENGINE's orbit, and trbdf2's own orbit at default tolerances is
2.2e-5 from the exact one. A first version took only the halves'
estimate and stopped at 32 periods with 6e-3 for a state 51 % off
(start-up overshoot: the halves contract fast, each period still moves
1.1 %); the test keeps that case. A first version also took the max
over states of each difference separately, mixing the inductor
current's start-up drop into the capacitor's ratio. A PWM voltage
source on trbdf2 (its edges not landed) never settles to 1e-6: the
period-to-period noise is 2.7e-4 of the state — reported as
`not_settled`, which is that engine's accuracy on that circuit.
→ Reverse: `SteadySpec(tol=..., min_periods=..., max_periods=...)`.

## 10. `auto`: the capability check

→ `SteadySpec(method='auto')` constructs the steady-state solver's
applicability checks (`periodic_steady_state._Solver`: circuit and
schedule) before running anything; an `UnsupportedError` there routes
to the transient, recorded in `counters['steady_method']` and
`counters['steady_route_reason']`. A failure AFTER the check (no orbit
within the budget) is `failed`; it never falls back. An evaluator with
`engine='pwl'` or `'trbdf2'` routes `auto` to the transient (the
steady-state solver runs on `dsed` only); `method='pss'` with those
engines is rejected in `__init__`. → Reverse: a public capability
query on `periodic_steady_state` (kernel list, item 2).

## 11. Losses the evaluator cannot account for

→ When `loads` are named, a device whose power is in neither
`device_loss_summary` nor the measured sources (anything but R,
switches, PWL and Shockley diodes, L, C and the DC voltage sources) is
refused (`UnsupportedError('design.evaluate', 'losses', name,
kind)`). Evidence: a buck with a Lauritzen diode read efficiency
0.99989 with the diode's 1.36 W of the 28.4 W input missing. Without
`loads` the other KPIs are returned, no `p_loss`, and
`counters['losses_not_covered']` names the devices. Efficiency is
`P_out / (P_out + P_loss)` (decision 4 kept `P_in` as a measured KPI):
with a datasheet conduction model or a loss table the losses include
energy the resistor-pair circuit does not dissipate, so `P_in` would
not contain it; without them the two agree (2e-12 on the oracle).
→ Reverse: extend `device_loss_summary` to the device (in-kernel
losses).

## 12. Batch scaling (open question of the design)

→ `evaluate_many` defaults to the thread backend, `backend='process'`
accepted; the first three designs run serially (the stop rule), the
rest on the pool. Measured (64 DCM-buck designs, `pss` route, shared
16 GB machine, observation, not asserted): 1 worker 0.50 s, 2 → 0.45 s,
4 → 0.43 s, 8 → 0.47 s. A design costs about 7 ms, of which the
kernel is 13 % (cProfile: `run_ped_native_builder` 27 ms of 210 ms for
32 designs); the Newton shooting and the KPI post-processing are
Python and hold the GIL, so threads stop scaling at 2 workers. 256
designs on processes: serial 1.83 s, 2 → 1.83 s, 4 → 1.43 s, 8 →
1.52 s (spawn cost against 7 ms jobs). Equal records on every backend
(tested). → Reverse: nothing to reverse; the lever is in the kernel
(item 3).

## 13. The junction-temperature fixed point

→ `ThermalSpec(r_th={device: K/W}, t_amb, t_limit=150, tol=0.1,
max_iter=20, loss_at=None, temperature_dependent=False)`. Per
iteration: the loss summary at `T_j` (a switch's 3-D `E_on_table` /
`E_off_table` read at the device's `T_j`), then `loss_at(device, P,
T_j)` if given (it scales the device's categories), then
`T̂ = T_amb + R_th·P`, `T_j += ω(T̂ − T_j)`, `ω` halved when the
increment grows. It stops when the estimated DISTANCE to the fixed
point, `inc·q/(1 − q)` with `q` the ratio of successive increments,
is below `tol` (at least 2 iterations); stopping on the increment
alone ended 0.9 K short at `q = 0.9` (review H3). Increments that do
not shrink with `T_j > t_limit`, or with `ω < 1/64`, or no stop in
`max_iter`, are `infeasible`, `thermal_runaway`; a fixed point above
`t_limit` is `infeasible`, `thermal_limit`; a non-finite loss is
`failed`. All carry NO KPIs (a number at an unsettled `T_j` is not a number); the adaptors
give the penalty for a record without KPIs. `temperature_dependent`
means `build` reads `design['_T_j']` (device → °C) and the steady
state is re-run per iterate; otherwise one simulation serves every
iterate. `T_j` per device is in `record.t_j` and the KPI `t_j.<dev>`.
`R_th` is a single junction-to-ambient resistance per device (no
shared sink): the steady thermal network of D4 reduced to its
simplest form; `device_thermal_summary`'s Foster chains are
transient, so their steady value is their `R` sum.

Evidence: `P = 10·(1 + 0.004·(T − 25))` W, `R_th = 2`, `T_amb = 40`:
`T_j` = 63.031680 °C against the closed form 63.043478 (0.0118 K, the
estimated distance 0.0117983, exact for a linear map) in 3
iterations, one simulation as without the loop. `R_th·P₀·α = 0.9`:
2424.90 °C against 2425 in 96 iterations (estimate 0.097 K).
`R_th·P₀·α = 1.2`: ratio 1.2 above the limit at iteration 2,
`thermal_runaway`. A switch with `R_on(T) = 50 mΩ·(1 + 0.006(T −
25))`, `R_th = 10`: 84.79 °C in 4 iterations, 12 simulations (16
before the kernel period map: each iterate's sampled KPI period run
is gone; re-measured in batch 6c).
→ Reverse: `mode='kernel'` when `add-in-kernel-event-losses` lands
(task 7.1).

## 14. The worked example (design D8)

→ `examples/optimization/buck_pareto.py` + notebook. Deviations from
D8, each stated in the example's docstring:

* **Parts.** D8 names "two synthetic MOSFET entries of `lib/data`";
  this tree ships none (its catalog holds one 1200 V IGBT and one SiC
  diode, both synthetic, neither sensible in a 48 V buck). The example
  builds two MOSFETs ("SYN-A": 4 mΩ, 24 + 12 µJ at 48 V / 10 A;
  "SYN-B": 10 mΩ, 8 + 4 µJ) through `pulsim.lib.parts.part_from_mapping`
  with `provenance.method: synthetic` (so `Part.is_synthetic`), 3-D
  straight-line energy tables `E·(v/48)(i/10)(1 + 0.003(T_j − 25))`.
  The real-part decision stays the owner's.
* **Thermal.** The heatsink is SIZED so the hotter switch sits at
  125 °C (`R_th,sa = (85 K − R_jc·P_max)/P_total`), so `R_on` and the
  tables are read at 125 °C and no fixed point is needed; the `T_j`
  constraint becomes `R_th,sa ≥ 0.05 K/W`. The fixed point is tested
  on its own oracle (tasks 3.2).
* **Circuit.** Open loop, D = 0.25, no dead time, no body diode; the
  low-side switch's switching loss is neglected (it commutates at
  near-zero voltage in a synchronous buck); inductor DCR 3 mΩ.
* **User KPIs see the default KPIs.** `SteadyPeriod.kpis` / `.losses`
  (read-only) so the volume law can size the sink from the losses.
* Volume constants `K_L = 3e-4 m³/J^0.75`, `K_C = 7e-5 m³/(F·V²)`,
  `K_HS = 3e-4 m³·K/W`: assumptions (2 cm³ for 10 µH at 12/10 A, 1 cm³
  for 100 µF at 12 V, 300 cm³ for a 1 K/W natural-convection sink).

Observation (not asserted), 512 Latin-hypercube designs, seed 0, this
machine: 376 ok, 136 infeasible, 0 refused, 0 failed; 4 designs on the
front, 97.73 % at 13.3 cm³ to 97.90 % at 16.5 cm³, all SYN-B, between
50 and 65 kHz; 1.3 s wall (single worker, kernel route). Re-measured
after the section-edge fix of §16 (batch 6c, 2026-11-13): before it the
front had 7 designs, 97.91 % at 11.3 cm³ to 98.50 % at 13.2 cm³,
between 50 and 91 kHz, 3.3 s — the turn-on energy was missing from
every record, which flattered the higher switching frequencies. The CI test (32 designs) asserts
the counts, a non-empty front and the constraints of every member. The
NSGA-II path (`pymoo`, optional extra `pulsim[opt]`) is not exercised
here (pymoo is not installed on this machine); its test is an
`importorskip`.

## 15. Fresh-context review (opus, read-only) and what was done

* H1 an unstable `pss` orbit came back `ok` → `failed`,
  `unstable_orbit` (test by a monkeypatched result).
* H2 the settle check saw only L/C states → all state columns (§9).
* H3 thermal stop on the increment → on the estimated distance (§13).
* M1 `NotImplementedError` / `RecursionError` became `failed` → they
  propagate (and `_UnsupportedNotImplemented` reaches the refusal).
* M2 a non-finite thermal loss, a first-iterate overshoot read as a
  runaway → §13.
* M3 a callable in a design hashed by its name only → name, code,
  defaults and captured values; arrays as lists; objects by type and
  `vars()`.
* M4 the example's volume was `inf` at `R_th,sa ≤ 0` → floored at the
  constraint's bound, so the constraint flags it.
* L1 `auto` raised on "cannot extract this circuit's state space" →
  routed to the transient with that reason (no circuit in the tests
  triggers it; the branch is untested).
* L3 a user KPI named like a default one, `min_periods < 8`, list
  values in a design → rejected / frozen to tuples.
* Not done: an energy-balance guard on `p_in − p_out − p_loss`;
  cancelling queued futures when a batch raises.

## 16. The kernel period map (slice 3, kernel item 3)

→ `dsed_period_map` (`core/include/pulsim/dsed/period_integrals.hpp`):
one call runs a period of the exact stepper and post-processes it
(x(T), events, `M`, per-observable `∫y`, `∫y²`, `∫y_a·y_b`, extrema)
with the GIL released for both. `periodic_steady_state` runs every
Newton and verification period on it (DC sources; a sine `b(t)` keeps
the Python path); the evaluator reads its KPIs and loss rows from the
verification period (`_period_kpis.py`), so an evaluation without a
user KPI simulates no waveform. Options the integrals do not cover
(conduction, diode and core specs, a non-scalar `Tj`, other device
kinds) and a declined period (near-defective basis, a numeric step)
take the sampled route with the reason in `counters["kpi_path"]`.

Evidence (buck Pareto SYN-A, 64 designs, interleaved best of 5, shared
Mac, observation, not asserted):

| route | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| kernel | 2465 evals/s (0.41 ms) | 2050 | 1984 |
| old (Python map + sampled KPIs) | 368 evals/s (2.72 ms) | 297 | 274 |

Three-phase VSI (SPWM 2 kHz over a 20 ms fundamental, RL load, 32
designs, kernel 82 % of an evaluation by cProfile): kernel 93.7
evals/s on 1 thread, 180.4 on 4 (1.93x), 177.7 on 8 (1.90x, the Mac
shared with other builds); old route 18.0 evals/s on 1 thread
(5.2x slower; 4 designs: 1.11x on 8 threads).

Buck: 6.7x per evaluation on one thread; threads still do not scale. The
kernel is now ~11 % of an evaluation (cProfile, 16 designs: 48
`dsed_period_map` calls, 1 ms of 9 ms): the Newton loop
(eigvals/svd/solve on 2x2 matrices), the record and the loss rows are
Python, and at 0.4 ms per evaluation the thread pool's own cost
dominates. The counters prove what moved (asserted:
`python_callbacks == 0`, `nogil_period_runs == pss_period_runs`), not
a speed-up. The process backend cannot run the example (its
evaluator's KPIs are lambdas, which do not pickle).
→ What would make threads scale: the Newton loop in C++ (one call per
design, `periodic_steady_state` group 6.2 shares the variational
recurrence). Reverse: delete `dsed_period_map` from the module and
`_Solver` falls back to the Python map and the sampled KPIs.

KPI equality: the kernel's integrals and extrema equal
`pulsim.sensitivity`'s closed forms on the same orbit within 3e-15 of
the signal scale (CCM and DCM buck, asserted 1e-12). Against the
present sampled KPIs (2000 samples per period, trapezoid): within
1e-6, the difference being the trapezoid's O(h²) (CCM ripple_v moved
from 6.0e-11 to 7.8e-13 off the matrix-exponential reference). `M`:
equal to the Python segment product to 6e-16 in CCM; in DCM 2.9e-11
from a central difference where the Python product is 5.0e-10 off
(orbit gradients moved from 4.8e-8 to 9e-9 of the prototype).

The verification period is now the same projection-off map Newton
converged on, so its residual check is a reproducibility check of one
code path; the independent check (`simulate`, projection on) is
`result.orbit`, asserted to return to `x0` within 1e-8 of the state
scale in `test_period_kernel_kpis.py`.

Bug found and fixed: a one-period sampled KPI run starting at the
section (a PWM turn-on at t = 0) never saw that edge, so every pss
record of the buck example missed `E_on` (P_sw 4.42 W from `E_off`
alone on the first design). The sampled route measures the second of
two periods; the kernel reads the edge from `x(T⁻) → x(0⁺)`.

## 17. Slice 3 order and review

Integration was merged before the period map was written (not after,
as the slice listed it): it rewrote `builder_adapter.hpp`,
`dsed_bindings.cpp` and `scheduler.hpp`, the three files the map hooks.
→ Reverse: none (history).

Fresh-context review (opus, read-only) of the map: thread safety and
the closed forms found clean (every Python object converted before the
GIL is released; no statics or thread_locals in dsed/pwl/events; each
evaluation builds its own builder, cache and adapter). Acted on:
(1) the saltation picked the diode by the first differing mask bit,
but the logged new mask is the one after the zero-time settle, so it
could take the wrong diode or throw when the settle reverted the flip
(`test_boost_with_a_zero_band_diode_settles` failed that way): the
diode is now found by the event's name, and an unchanged mask is S = I;
(2) user KPIs read a one-period run (the lost section edge again): now
the second of two periods; (3) the lazy result was a proxy: now
`SteadyPeriod.result` is a property that simulates on first access;
(4) the verification no longer independent: recorded in 16 and
covered by the `orbit` test; (5) events left unmatched now raise.
Not acted on (recorded): extrema of a mode faster than h/16 between
two sign changes of y' in one search cell can be missed (a log-spaced
grid near s = 0 would fix it); an event within 1e-12 T of the section
is neither a section edge nor an inside edge in `Period.edges`.

## 18. Integration batch 6c: fresh-context review of slices 2 and 3

Reviewer (opus, read-only) on 1aef1a52..3ea998c6; each fix is its own
commit with a test proved non-vacuous against the previous code.

Acted on:
* (d3) design ids collided: `math.sin` vs `math.cos` lambdas, two
  `functools.partial`s, two bound methods, keyword-only defaults, a
  read global; a nested function hashed differently per process → the
  id now covers names, nested code, kw-defaults, globals, partials and
  methods (§15 M3 extended).
* (a1, d1) a zero-valued parameter gave NaN sensitivities (relative
  steps 0/0; `dev > lim` is False for NaN) → refused by name; a
  non-finite evaluator gradient goes to `grads_refused`.
* (a2) near-miss events were a once-per-call-site warning →
  `counters["grad_near_misses"]`, the record stays `ok` (the gradient
  is exact for this event pattern).
* (b1) the extrema search used 16 uniform cells per step: a fast mode's
  peak and the following valley in one cell were both missed (1.707 at
  92 ns of a 10 µs step reported as 0.5) → the grid adds `h/2^m` down
  to `1/(2 max|Re μ|)`, the same in C++ (`detail::extrema_grid`) and
  Python (`_extrema_grid`). Still a grid: two stationary points closer
  than one geometric cell can be missed; no root isolation.
* (d2) an inconsistent kernel edge log raised outside the engine
  wrapper and stopped the batch → `failed` for that design.
* (e1) the Python `LossTable` returned NaN / inf / 0 for non-finite
  queries where the kernel table refuses → both refuse.

Recorded, not changed:
* (c1) the pss verification period is the Newton map itself, so its
  residual test cannot fire; periodicity is certified by the Newton
  stop and, independently, by the `orbit` test (§16). An independent
  per-evaluation check costs one `simulate` period.
* (a7) two separately logged events at the same instant on one sample
  can raise `KeyError` in the Python sensitivity jump (a crash, not a
  wrong number); no circuit in the tests produces it.
* (b4) the kernel attaches an event to the first sample at or after its
  time; Python raises in that case. (a3) the grazing cosine is
  dimensionally sound only when each state label names one device;
  the kernel's 1e-12 test is on the raw cosine.

## 19. The Newton loop in the kernel (slice 4)

Question: where does the 89 % of a buck evaluation that is not the
kernel go, and what can leave the GIL? → Profile first (64 SYN-A
designs of the Pareto example, per-phase wall time, best of 5; shared
Mac, observation): 3ea998c6 spent 0.96 ms per evaluation, of which the
kernel period map 14 %, the Python Newton loop around it 42 % (numpy
`eigvals`/`svd`/`solve` on 2x2 matrices, the per-run dict conversion),
the loss rows 26 % (`LossTable.__call__`'s numpy trilinear read, 86 µs
per point), the KPIs 7 %, the record 10 % (`isinstance(v, typing.Mapping)`
per value, provenance re-frozen per record). 1744 Python function
calls per evaluation (cProfile).

→ (1) `dsed_periodic_orbit` (`core/include/pulsim/dsed/periodic_orbit.hpp`):
the loop of `periodic_steady_state`, statement for statement (the same
scaled norm with the scale folded in first, the Armijo test against the
OLD iterate's norm under the new scale, `polished`, three relaxation
rounds, the budget before every run but the first, the singular test
`σ_min ≤ 1e-10·max(1, σ_max)`, the unit-circle test, the verification
period with the caller's observables), one call with the GIL released.
Each period builds a fresh `PEDSimulator` (its diode latch and burst
counters must not carry over); the Python schedule object is held by a
non-owning `SwitchFnRef`, so no reference count is touched without the
GIL. Outcomes are statuses; Python formats the same `ConvergenceError`
messages. Two deliberate differences: a NaN anywhere fails every test
(numpy's `max` propagated it already; C++'s `std::max` would not), and
the verification check is `!(residual <= bound)`, so a NaN residual is
irreproducible instead of accepted (the Python loop now has the same
test). An exception raised by a Python schedule inside a period now
propagates as itself; the Python loop turned any `RuntimeError` there
into "a period run failed" (only the engine's `std::runtime_error`
becomes `run_failed` now). (2) A loss table's energies are read by its
kernel copy (`LossTable.total`, cached per table by weak reference,
bit for bit `np.sum(table(v, i, tj))`, asserted on 500 random queries
inside and outside the grid). (3) Provenance frozen once per evaluator;
`_fz` checks scalars by type before any `abc` test.

Evidence. Kernel loop against the Python loop (kept as
`periodic_steady_state(..., _python_loop=True)`;
`test_periodic_orbit_kernel.py`):

| case | period runs | iterations | steps | max|Δx0|/scale | max|ΔM| |
|---|---|---|---|---|---|
| buck CCM (R 1 Ω, D 0.5) | 3 / 3 | 2 / 2 | 6 / 6 | 0 | 0 |
| buck DCM (R 50 Ω, D 0.3) | 6 / 6 | 5 / 5 | 17 / 17 | 1.8e-14 | 3.0e-15 |
| boost CCM 20 Ω, V_th 0 | 17 / 17 | 3 / 3 | 34 / 34 | 1.9e-17 | 0 |
| boost DCM 500 Ω, V_th 0 | 21 / 21 | 7 / 7 | 51 / 51 | 0 | 0 |
| boost CCM, V_th 1 mV | 3 / 3 | 2 / 2 | 6 / 6 | 1.5e-16 | 0 |
| boost DCM, V_th 1 mV | 7 / 7 | 6 / 6 | 20 / 20 | 6.8e-14 | 5.3e-15 |
| negative-conductance RC (unstable, μ = e^{2000T}) | 3 / 3 | 2 / 2 | 30 / 30 | 0 | 0 |

Boost orbits: 24.05362948 V (CCM) and 36.59440084 V (DCM) on both
loops (the slice brief quoted 36.5943997644 for the DCM case, 1.07e-6 V
lower; neither loop nor the pre-existing test, 36.594401 ± 1e-5,
gives that). Budget, singular `I − M` (lossless LC), non-isolated
orbit, the unstable orbit (returned, `stable = False`) and the
refusals before the loop keep their types. C++ tests
(`test_periodic_orbit.cpp`): every status on affine maps with a known
fixed point, a NaN period never accepted (52 runs, `no_progress`), and
the DCM buck orbit (6 runs, 5 iterations, v_C within 5.2e-11 of the
prototype, μ = 0.988977 + 1.7e-7); Eigen 3.4.0 gives the same counts,
v_C 5.19e-11 off and the same μ to 1e-16.

Throughput (64 designs, `evaluate_many`, one process per tree,
interleaved best of 5, load average 10–13, observation, not asserted):

| tree | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| 3ea998c6 | 1516 evals/s | 965 | 1011 |
| this slice | 2445 evals/s | 1848 | 1740 |

Same statuses; KPIs within 1.8e-12 relative (the DCM orbit's rounding).
Per evaluation now: 0.44 ms, kernel 21 % (was 14 %), 707 Python
function calls (was 1744).

Threads still do not scale, and this is why: about 80 % of an
evaluation still holds the GIL — `build` (the user's builder calls,
~25 µs), the solver's set-up (applicability checks, the schedule's
periodicity check over two periods, the state-space cache and adapter,
the state labels: ~50 µs), `spec_for` (the observables, ~38 µs), the
loss rows (`_switch_switching_loss`'s numpy plumbing on two-element
arrays, ~56 µs, plus the rows, ~50 µs), the KPIs and the record
(~60 µs). With 20 % released the ceiling is 1.25x, and the GIL hand-off
between threads costs more than that on this machine. What would make
threads scale: the loss rows and the observables spec in the kernel
(the period's edges are already there), or the process backend for
batches (item 2 of this slice).
→ Reverse: delete `dsed_periodic_orbit` from the module (`_Solver._orbit`
is then `None` and the Python loop runs); `LossTable.total` falls back by
deleting its body to `np.sum(self(...))`.

## 20. The circuit form (slice 5, task 7.2) and what still holds the GIL

Question: how does a design reach the immutable circuit IR? → (1)
`Evaluator(circuit=base, fields={variable: target})`, the target a
named parameter (`with_params`) or `"<component>.<field>"`
(`with_value`, the sensitivities' naming); the gate schedule fixed
(`switch_fn=`) or per design (`schedule(design)`); design keys outside
`fields` (a frequency, a part) are read by `period` / `schedule` / the
KPIs only. `build=` keeps working unchanged; exactly one of the two.
The circuit form refuses `ThermalSpec(temperature_dependent=True)`
(no `_T_j` input) and `backend='process'` (a `Circuit` does not
pickle) by name. (2) `periodic_steady_state` and the pwl/trbdf2
shooting take a `Circuit` like `simulate` does: the period runs share
its symbolic plan. (3) What is resolved once instead of per record:
the kernel observables (`spec_for`), on the circuit form, when no
variable moves a value they read (whitelist: inductor, capacitor,
resistor; any other kind is re-resolved per design, tested with
`Vin.V`); the state labels of the exact stepper per (structure hash,
section mask) for any `Circuit` (they were one full state-space
extraction per evaluation, 10.7 us); the options digest per evaluator
(the base circuit by `value_hash()`, the design by its variables:
flat designs are canonicalised inline, same ids as before, asserted on
51 designs). (4) The switching-table losses of the kernel route are
read in scalars (`_period_kpis._table_switching_loss`): the same edge
limits, each point by the kernel table, summed as numpy sums them
(a plain loop below 8 terms, `np.sum` above) -- bit for bit equal to
`_switch_switching_loss` (random sequences of 0..17 edges, and the
example's records with the path disabled); any other spec takes the
general path. (5) Hot-path function-level imports replaced by module
attributes read at call time (tests still replace them).
→ Reverse: drop `circuit=` from `Evaluator.__init__` (the factory form
is untouched); `_LABELS` and the spec cache are plain dicts to delete.

Counters (asserted, `test_design_circuit_form.py`): records of the
circuit form equal the factory form's in every field but the id
(dsed pss, exact transient, pwl pss, pwl transient, trbdf2 transient);
a pwl batch of 6 designs on 3 threads makes 1 symbolic analysis on the
circuit form and >= 6 on the factory form; the exact stepper makes no
symbolic analysis at all (dense per-mask extraction), and there the
labels and the observables are computed once per batch (6 per 6
designs before); 8 threads give the 1-thread records and leave the
base's `value_hash` unchanged; < 500 Python calls per evaluation.

Evidence (buck Pareto, `examples/optimization/buck_pareto.py`, 512
designs: 376 ok / 136 infeasible, every record bit for bit equal to
7b195243's on both forms):

| | Python calls / evaluation (cProfile) | wall / evaluation | kernel share |
|---|---|---|---|
| 7b195243 (factory) | 751 | ~440 us | 20 % |
| factory form now | 521 | | 24 % |
| circuit form now | 378 | 164 us | 36 % (59 us) |

Per evaluation, circuit form (wall, nested phases, best of 5): kernel
orbit 59 us; solver set-up 24 (schedule periodicity check 6, circuit
check 2, `to_builder` 1, cache + adapter 1.3, census, the rest glue);
loss rows 15; KPIs 15; build 8 (two `with_value` 5); design id 8;
implausible-voltage check 5; result object 5; the record ~15.

Throughput (`evaluate_many`, 128 SYN-A designs, interleaved best of 5,
one process per tree; load average 9-12 from other agents' builds on
10 cores, observation, not asserted):

| | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| 7b195243 (factory) | 3239 evals/s | 2856 | 3150 |
| factory form now | 3927 | 3802 | 4191 |
| circuit form now | 5025 | 4865 | 4879 |

THREADS STILL DO NOT SCALE: 8 threads are not faster than 1. What holds
the GIL is everything but the one `dsed_periodic_orbit` call: about
105 us of Python per evaluation against 59 us of kernel (Amdahl ceiling
1.6x before any GIL hand-off cost), spread over the phases above with
no single item above 25 us. What would make the thread backend scale:
a batch kernel call (all designs' Newton loops on a C++ pool in one
GIL-free call) lifts the ceiling to ~1.4x only, because the Python
set-up and post-processing stay serial; the process backend has no
shared GIL and is item 2 of this slice (the declarative KPIs make the
example picklable; the circuit form needs a picklable `Circuit`).

Needed from the IR (recorded in its DECISIONS): a picklable `Circuit`
(group 4.3) for the process backend; the Python side state of a builder
(`_graph_driven`) is lost by `build()`, so a control graph's driven
source cannot be refused on the circuit form (group 3).

### 20a. A tied extremum is reported at its earliest instant

CI (Linux, 7b195243) failed `test_kernel_loop_equals_the_python_loop
[boost CCM]`: one extremum time differed by exactly T. On a periodic
orbit x(0) = x(T), so an extremum attained at an orbit end is attained
at both ends, and the kernel and the Python closed forms picked
different ends after a last-bit difference in the value. The rule, now
identical in `dsed/period_integrals.hpp` (`detail::extremum_candidate`,
`kExtremumTie`) and `sensitivity._extrema` (`_EXTREMUM_TIE`): two
candidates whose values agree within 1e-12 relative are one extremum;
the larger (smaller) value is kept and the EARLIEST time is reported.
The test's one-period exception is removed; a new test pins a both-ends
extremum (the CCM buck's inductor valley, a constant node) at t = 0.

An extremum tied at the two orbit ends is NOT flagged `non_unique`
(the coordinator suggested it should be): the two ends are one point of
the orbit, and their parameter derivatives are equal there, because
dx(T)/dp = M dx0/dp + dP/dp = dx0/dp at the fixed point. The gradient is
therefore well defined; `non_unique` stays reserved for two distinct
interior instants. Other sites checked: `design._extreme` returns values
only, records carry no peak times, the near-miss scan reports no times.

Review items a7 and b4 (§18), fixed with it: the sensitivity context
now builds a mode for every mask an event log names, so two events at
one sample go through the mask between them (a test splits a gate edge
in two and gets the unsplit sensitivities bit for bit; KeyError
before). Both sides attach an event to the first sample at or after it
(within 4 ulp) and refuse it, "has no sample of its own", when that
sample is more than 1e-12·max(|t|, t_end) later
(`sensitivity._attach_events`, `detail::kEventSampleTol`): the jump is
applied to that sample's left limit, so a later sample would be a wrong
number. The engine stores a sample at every event, so no run in the
suite reaches the refusal; the tests move a logged event by hand.

## 21. Declarative KPIs and the process backend (slice 5, item 2)

* `Kpi.efficiency()`, `Kpi.ripple(node)`, `Kpi.mean(node)`,
  `Kpi.i_peak(L)`, `Kpi.i_rms(L)`, `Kpi.ripple_i(L)` are frozen
  dataclasses (`_KpiSpec(kind, target)`): they pickle, hash and compare
  by value, and the options digest (hence `design_id`) covers them by
  value, the same in every process. `Constraint.max/min/between` are
  the bounds as data. Callables are still accepted (threads).
* What they compute: a name the evaluator already computes (the
  default KPI) is that default, bit for bit; a node's ripple / mean is
  the kernel's closed-form extrema / integral on the kernel route and
  `_p2p` / `_mean` on the right-limit samples otherwise -- on the
  sampled route bit for bit what the equivalent callable on
  `SteadyPeriod` gives, on the kernel route the closed form (the
  callable reads samples: ripple within 1e-5, mean within 1e-9,
  O(h^2) of the trapezoid; tested). A node they name is resolved with
  the kernel observables, so an unknown node is a `KeyError` naming it.
* `backend='process'`: before the first design runs, every part is
  pickled in turn and the first that fails is refused,
  `UnsupportedError(device=<part>)` (`circuit`, `build`, `schedule`,
  `kpis['name']`, `designs[i]`, ...), with no worker started. Jobs are
  chunks (4 per worker), the evaluator pickled once per chunk. The
  circuit form pickles through a factory: `circuit=` a zero-argument
  module-level callable (`functools.partial(build_circuit, part)`);
  each worker calls it once (cached by value hash) and refuses a
  factory whose circuit hashes differently from the parent's ("must be
  deterministic"). A built `Circuit` does not pickle (IR needs, §20).
* The buck example uses only module-level functions and `Constraint`s;
  `--process` runs its Latin hypercube on processes, records bit for
  bit those of the thread backend and of 7b195243 (512 designs, 376 ok
  / 136 infeasible, the same 4-design front).

## 22. Independent periodicity check and energy balance (slice 5, items 3, 4)

Periodicity (review c1). On the kernel route the verification period
is the shooting map itself, so it cannot catch a defect of that map.
The evaluator now runs one plain `simulate(engine='dsed', dt=T)` period
from `x*` -- another entry point (dispatch, adapter construction,
event projection on) -- and requires it to return within 1e-8 relative
(an order above the shooting's own 1e-9 stop). Measured: 2.2e-16 (CCM
buck), 6.9e-17 (DCM buck), 2.0e-16 (boost), 4.0e-16 (buck Pareto, both
parts). Cost: 200-390 us per check against 90-310 us per whole
evaluation (load 10, observation), i.e. one to two evaluations, so it
runs ONCE per (topology, orbit event-name pattern) per evaluator and
process, logged in `Evaluator.periodicity_checks`, never in a record:
record counters must not depend on batch order or backend (they are
compared bit for bit across 1/8 threads and processes). A mismatch
raises `pulsim.design.PeriodicityError` out of `evaluate` /
`evaluate_many` (review §24, item 1: failing the records made them
depend on which design was checked first). The deviation is relative
per state, its denominator floored at 1e-12 of the largest state. The
Python-loop route (no kernel map) already verifies with `simulate`; the
`pwl` / `trbdf2` routes' map IS that engine's transient, so the check
does not apply there. The first check found a crash: a schedule
narrower than the circuit's switch count (`make_pwm_switch_fn(.., 0,
1)` on a switch + diode buck) made `simulate(engine='dsed')` raise
`IndexError` in `check_diode_modes`, where shooting accepted it; the
containment check now skips a diode the record has no column for (its
"no record, nothing to check" rule).

Energy balance. On every record with KPIs and `loads=`: `res = (P_in -
sum P_avg - dE/T) / P_in` from the same integrals as the KPIs (sources'
delivered power, every device's dissipated `P_avg`, loads included;
`dE/T` of L and C between the window ends; on a kernel period from the
period's own `y_start` / `y_end`, review §24 item 4). The
record carries `counters['energy_residual']` and `counters['p_table']`
(switching, recovery and core losses from tables / datasheets: outside
the circuit's balance, so never in `res`). Not formed when a source of
power is not measured (a voltage source outside `sources=`, a current
or controlled source). Measured |res|:

| circuit | pss/dsed | transient/dsed | pss/pwl T/200 | transient/pwl | transient/trbdf2 |
|---|---|---|---|---|---|
| buck CCM | 2.1e-13 | 3.7e-12 | 2.5e-4 | 5.0e-10 | 5.5e-7 |
| buck DCM | 3.9e-12 (1.0e-11 without dE/T) | 1.3e-9 (4.8e-6 without dE/T) | 9.9e-3 | 1.0e-7 | failed (implausible V) |
| boost CCM | 6.5e-13 | 1.0e-10 | 2.1e-9 | 4.5e-9 | failed |
| boost DCM | 8.4e-14 | not settled | 1.7e-4 | not settled | failed |
| 3-phase VSI (SPWM, dt 1 us) | refused | refused | 1.6e-3 | 6.5e-7 | 9.5e-5 |
| buck Pareto, 512 designs | <= 1.2e-13 | | | | |
| pwl T/25, sub-step correction (test) | | | | 3.9e-2 | |

Tolerance by route, an order above the worst measured: 1e-8 on the
kernel period (`exact`); 1e-8 (2000/n)^2 on a sampled `dsed` run with
n samples per period (`sampled`: the rows integrate products of
per-step lines, O(1/n^2); DCM buck 1.3e-9 / 1.3e-7 / 2.1e-6 at n =
2000 / 200 / 50); 1e-1 on the time-stepping engines (`stepped`). A violation is `failed`,
`'energy_balance: ...'` with both powers and the residual; a test
scales one load's `P_avg` by 1 + 1e-6 and the record fails.

What the stepped tolerance lets through: the `pss/pwl` route's `P_in`
is off. DCM buck at T/200: `P_out` 7.31746 W equals the exact orbit's
7.31746 W, but `P_in` reads 7.394 W against the exact 7.321 W (+1 %,
res 9.9e-3; at T/1000 2.0e-3), while the long `pwl` transient at T/200
balances to 1e-7 (its own `P_out` is 0.9 % low, the engine's
discretisation). So the restart run of the `pss/pwl` window measures
the source current wrongly by ~1 %; `efficiency` does not read `P_in`
(P_out / (P_out + losses)) and is unaffected. Recorded, not fixed here:
the window run is `TransientMap.simulate` from the full state, the
fixed engine's first-order restart is the first suspect.

## 23. Saltation for comparator-driven gates: not done, the hook it needs

`add-periodic-steady-state-shooting` 6.1/6.3 (peak-current-mode buck,
multipliers -1.267293 and 0.998949) needs a gate that turns off when a
state crosses a threshold (`i_L = I_ref`) and on at the clock. The
period runs (`period_run` in `python/dsed_bindings.cpp`) arm only the
diode predicates, and the scheduler tells a schedule about a state
event only through `register_zcd_transition(Real t)` (concept
`HasZcdRegister`), called for every `CurrentZC`, `DiodeOn` and
`DiodeOff` firing alike. A comparator gate latched there would also
latch on the freewheel diode's own events, so it cannot be written
without knowing WHICH predicate fired. Hook needed in
`core/include/pulsim/dsed/scheduler.hpp` (not editable in this slice),
`PEDSimulator::fire_event_`: call, when the schedule offers it,
`switch_fn_.register_event(Real t, const std::string& name,
PredicateType type, int pred_index)` before re-sampling the mask (a new
concept `HasEventRegister`, the ZCD call kept). With it, the kernel
side is additive: a native `ComparatorGate` schedule (clock edges in
`next_edge_after`, latched off by its own predicate), its predicate
`g = w.x - I_ref` added next to the diode predicates in a new
`period_run` overload, and in `period_map_from_run` the saltation `S =
I + (f+ - f-) w^T / (w . f-)` for that event (today only diode events
get one; no other state event reaches the period runs, so nothing is
silently missing now). The probe `probes/peak_current_mode.py` is the
oracle. Not started: this slice ran out of time after items 1-4.

## 24. Fresh-context review of slice 5 (items 1-4) and what changed

Reviewer (opus, read-only) on 18c62e86, 00c7df92 and the tie-rule diff.
Acted on (one commit, tests shown failing before where a test exists):

* (1, bug) The periodicity result was cached per (topology, pattern)
  and decided the records of later designs, so records depended on
  batch order and backend (each process chunk starts with an empty
  cache). A mismatch now raises `pulsim.design.PeriodicityError` (a
  plain `Exception`, so no engine wrapper turns it into `failed`) out of
  `evaluate` / `evaluate_many`: a defect of the map stops the batch, and
  no record depends on which design was checked. A passing check is a
  sample of the map, not a per-design guarantee (documented).
* (2, bug) A native `SwitchSchedule` canonicalised to its type name, so
  `Evaluator(circuit=c, switch_fn=make_pwm_switch_fn(f, 0.4, ..))` and
  the same at 0.5 shared every `design_id`. An object with
  `next_edge_after` is now digested by its behaviour: the mask at 0 and
  after each of its first 64 edges (`_schedule_fingerprint`; shortcut:
  schedules equal on those edges and different later share it).
  The buck example's ids are unchanged (its schedule is a function).
* (3, risk) On a sampled `dsed` window the rows integrate products of
  per-step lines, which is O(1/n^2) away from the exact stored energy:
  DCM buck 1.3e-9 (n = 2000 samples per period), 1.3e-7 (200), 2.1e-6
  (50); at 200 a correct record failed. Route `sampled` now has
  tolerance 1e-8 (2000/n)^2; the kernel period keeps 1e-8.
* (4, risk) The kernel route took dE/T = 0, but the shooting stops at a
  scaled residual: dE/T now comes from the period's own start and end
  values (inductor currents, node voltages; DCM buck residual 1.0e-11
  -> 3.9e-12).
* (5, risk) The periodicity deviation had no floor: a state sitting at
  zero (a DCM inductor current at the section) divided rounding noise
  by ~0. The denominator is floored at 1e-12 of the largest.

Recorded, not changed:
* (6) The tie rule is relative only (1e-12): an extremum of 0 against
  1e-20 noise is not a tie and may still flip with the last bit, and a
  chain of three values each within 1e-12 of the next is resolved by
  the kernel's running rule and Python's best-then-earliest rule
  differently. The CI case (equal to rounding at a nonzero value) is
  covered; an absolute floor needs the observable's span, which the
  kernel's streaming search does not have before the end.
* (7, nit) A non-deterministic circuit factory is refused in the worker
  (at unpickling), not before the workers start: one process cannot
  see it.

Process backend, buck example SYN-A, interleaved best of 5, load 6-9
from other builds (observation, not asserted): 4096 designs, circuit
form 6484 / 9801 / 9692 evaluations/s on 1 / 4 / 8 workers (factory
form 5177 / 8648 / 8868); about 16k designs (32768-point hypercube,
SYN-A half) with a ctest build beside it, load 8-9: processes 2588 /
3153 / 2798, threads 1371 / 1546 / 1615 on the same machine state.
Processes give about 2x the threads under load; worker start-up (spawn, import pulsim,
unpickle, build the base) is about 0.3 s per worker, so a batch needs
thousands of designs before processes pay.

## 25. Second fresh-context review of slice 5 (00c7df92, 02b76db6, 33d3eafb)

Reviewer (opus, medium, read-only). Acted on:

* (b1, bug) The periodicity check's own `simulate` ran inside `_engine`,
  so when it raised, the first design of its pattern became a `failed`
  record (with a reason that was the check's, not the design's) and
  nothing was cached; the next design checked again and, if that
  passed, every later design was `ok`. Status depended on batch order.
  A check that the engine cannot run now raises `PeriodicityError` (the
  map ran that period from x*, the engine does not): raised for every
  design of the pattern, cached by nobody. Test shown failing before.
* (b2, risk) A cache hit skips the check, and the key (topology + event
  names) does not see event times. A design with an event within
  1e-9 T of the section (0 or T), where the map's attach rule (first
  sample at or after, 4 ulp) and the engine's can differ, is now always
  checked and never cached. Option kept rejected: event times binned
  into the key (one check per design: 1-2 evaluations each, DECISIONS
  22). Reverse: drop `at_section` in `_check_periodic`.
* (b6) Python's tie rule now IS the kernel's running rule
  (`sensitivity._running_extremum`, candidates in time order): a chain
  1, 1+8e-13, 1+1.6e-12 is one extremum reported at the first instant
  on both (it was the second in Python). §24 item 6's chain case is
  closed; the absolute floor near zero is still not there (unchanged
  reason: the streaming kernel search has no span).

Recorded, not changed:
* (b3, b4) The stepped tolerance 1e-1 cannot see a dropped loss row on a
  95 % efficient design (losses are 5 % of P_in), and the sampled
  1e-8 (2000/n)^2 rests on one circuit. Both are re-measured with the
  `p_in` quadrature work (slice 6 item 4, §27).
* (b5) `_schedule_fingerprint` reads 64 edges (32 periods of one PWM):
  a soft-start differing later shares a `design_id`. Nothing in pulsim
  caches records by it; a longer horizon costs native calls on every
  `design_id`. Accepted, documented in the function.
* (b7, b8, b9, nits) The kernel-route `Kpi.ripple` / `Kpi.mean` are the
  closed forms (a sampled callable differs by up to 1e-5 on ripple);
  `Kpi.i_peak` is a magnitude; the process backend checks each pattern
  once per chunk (its cache is emptied on pickling). Documented
  behaviour, unchanged.

## 26. The kernel batch evaluator (slice 6, item 2)

Question: threads did not scale (1371 / 1546 / 1615 evaluations/s on
1 / 4 / 8 threads) because the per-design glue was Python. How does a
whole design run without Python?

Option taken: `evaluate_many` (default `backend=None`) hands the kernel
(`_pulsim._design_batch`, `core/include/pulsim/dsed/design_batch.hpp`)
the parameter table, one period and one native schedule per design and
a thread count. Per design, on `std::thread`s with the GIL released:
`with_params` / `with_value` of the shared base `Circuit`, builder,
cache (sharing the base's symbolic plan), adapter, the schedule's
periodicity test, `periodic_orbit_newton`, the stability test, the
implausible-voltage scan, the switching energies from the loss tables
at the period's edge limits, and a REGISTER PROGRAM for the rows,
losses, KPIs, energy balance and violations. The program is written by
`pulsim/_design_kernel.Plan` once per evaluator, operation for
operation in the order of the Python path (`design._kpis`,
`_period_kpis.loss_rows`, `_energy_balance`), so records are EQUAL to
`evaluate()`'s -- every field but `wall_s`, counters included.

* Fallback, not a second implementation of the outcomes: a design the
  kernel cannot finish (orbit status other than converged, an unstable
  multiplier, no closed-form integrals, an implausible voltage, an
  energy balance or a non-finite KPI, any C++ exception -- a bound
  check of `with_value` included) is returned with its reason and
  re-run on the Python path, which owns the wording of every `failed`
  and `refused` record; the record carries `counters['batch'] =
  'python: kernel: <reason>'`. A design whose schedule is not native
  (an opaque Python callable), a missing variable, a period or schedule
  factory that raises: the Python path, named likewise. Statuses
  therefore cannot differ between the paths; `ok` / `infeasible` and
  every number are the program's.
* What the plan cannot express sends the whole batch to the Python
  path with the reason in every record's `counters['batch']` and in
  `ev.last_batch['python_reasons']`: the factory form, a callable KPI,
  a thermal loop (until item 3), `gradients=`, a design variable that
  moves a value the observables read, loss options other than table
  switch specs, a transient or stepped route. `backend='kernel'`
  refuses instead (ValueError naming it); `'thread'` / `'process'` are
  the Python path as before (their records carry no `batch` key).
* `Kpi.formula(name, expr)`: arithmetic over KPIs, design variables and
  `loss.<device>`, compiled once (`_design_kernel.Formula`) and run by
  the same stack interpreter on both paths (IEEE division by zero; a
  non-finite value is `failed`). The buck example's two callables are
  now formulas; their records are the callables' bit for bit (512 of
  512 per part checked).
* Sums: CPython >= 3.12 `sum()` of floats is Neumaier-compensated,
  3.10 / 3.11 a plain loop, so `sum()` in the record arithmetic made a
  record depend on the Python version (and the kernel could not mirror
  it). The record arithmetic now uses `design._psum` (CPython 3.12's
  algorithm on every version, `SumN` in the kernel): records on 3.12+
  unchanged, on 3.10 / 3.11 they may move by an ulp (and now equal
  3.12+'s). `np.sum` of the switching energies (8 or more edges) is
  numpy's pairwise sum, mirrored (`np_pairwise`, checked against numpy
  2.4 on 5840 random vectors).
* The periodicity check stays in Python, once per event pattern of the
  batch (the first design of each pattern in input order: what the
  sequential Python path checks) and for every design with an event at
  the section; a mismatch raises out of `evaluate_many`.
* Threads inside the kernel vs one GIL-released call per design from a
  Python thread pool, measured on the buck Pareto designs (4096, both
  parts, interleaved best of 5, evaluator reused, load average 15-19 on
  10 cores from other agents' builds): kernel threads 10022 / 15502 /
  21621 / 25369 evaluations/s on 1 / 2 / 4 / 8; per-design calls 6037 /
  7991 / 7768 / 7631 (the per-call Python dominates); the Python thread
  path 3597 / 2391 / 2286 / 2405. Kept: threads inside the kernel.
  The kernel call alone runs 10246 -> 49721 designs/s on 1 -> 8 threads
  (8192-design table); the remaining serial part is the record
  materialisation in Python (about 18-23 us per design: `design_id`'s
  digest 8 us, the record and its mappings), which is what limits the
  8-thread total.
* Not unified with circuit IR task 5.2 (`_run_pwl_points`,
  feat/core-p3-circuit-ir a8ee6eb7, not merged here): both derive per
  row and run on `std::thread`s with an atomic index; that one runs
  the pwl transient for `sweep`, this one the exact orbit for the
  evaluator. Unify the row derivation and the thread loop when both are
  in one tree.

Reverse: `evaluate_many(..., backend='thread')` is the old default;
removing `_design_kernel.run` from `evaluate_many` restores it.

## 27. PWM and pulse voltage sources on the steady-state route (slice 6, item 0)

Question: does the kernel period map handle a PWM source's corners, or
must the route refuse them?

Evidence (`.scratch` probe, not shipped): a 24 V / 0 V, 100 kHz,
D = 0.4 PWM source into 100 uH and 47 uF || 5 ohm. With the refusal
bypassed, `periodic_steady_state` returns the orbit
`(v_C, i_L) = (9.5979041, 1.63187764)`; the matrix exponential's fixed
point `(I - Phi)^-1 z` agrees to 4.1e-16 and 3.7e-16 relative. The
stepper lands the source's corners. What is missing is elsewhere:
`builder.components()` exposes NO parameters for `pwm_voltage_source`
and `pulse_voltage_source` (`{}`), so `_not_t_periodic` cannot check
that the source's period divides `T`; a 30 kHz source under a 100 kHz
section would converge to a map that is not a steady state. The kernel
period map behind the design KPIs (`dsed_period_map`,
`design_batch.hpp`) also takes DC sources only.

Option taken: keep the refusal by name on every engine, with the
message corrected (it said the edges were not events of the exact
stepper, which the probe refutes). The test
`test_a_pwm_voltage_source_is_refused_by_name_on_every_engine` checks
the device and the reason.

To reverse: expose the source's `frequency`, `duty`, `phase` (and the
pulse timing) in `components()`, add them to `_not_t_periodic`, drop
the two kinds from the refusal, and add them to the period map's
dynamic-source support.

## 28. Event losses and the junction fixed point on the orbit (slice 6, item 3)

Question: on the exact route the electrical orbit does not depend on
T_j (no `temperature_dependent`); only the loss tables do. How is
`T_j = T_amb + R_th P(T_j)` solved, and what is read at the events?

Option taken. (a) The reading rule of `add-in-kernel-event-losses`
(turn-on: `v` before, `i` after; turn-off the reverse) on the period
map's left and right limits, already used for switches since slice 5,
now also for a diode's `E_off_table` in `switch_specs` (its recovery).
The sampled `device_loss_summary` used to ignore a diode named there;
it now refuses it by name, so the two routes never disagree silently.
(b) `ThermalSpec(mode="kernel")`, the default on the exact route
without `loss_at` / `temperature_dependent`: per device the commutation
points are collected once and `pulsim/dsed/period_thermal.hpp` solves
`g(T) = T - T_amb - R_th P(T) = 0` for the smallest root above `T_amb`
(the point the physical iteration climbs to). `P` is linear in `T_j`
between the tables' `T_j` knots, so the solver brackets segment by
segment and regula falsi (Illinois) lands in one step where `g` is
linear; above the last knot it steps to twice the linear root.
Stopping rule: `|g| <= 1e-9 K`, or a bracket narrower than 1e-13
relative. Runaway: no bracket because `g` does not rise above the last
knot (`R_th dP/dT_j >= 1`) or the search passes 1e6 C; the record is
`infeasible`, `thermal_runaway`, the violation `R_th dP/dT_j - 1`.
(c) A `ConductionTable` linear in `i` (one or two current knots) gives
`(a(T) int|i| + b(T) int i^2) / T` over a new on-state-current
observable (`g_off = 0`); it heats the junction only (`p_cond_table.*`)
so the electrical energy balance still closes.

Evidence: closed-form oracle `(T_amb + R(Pc + Ps(1 - 25a))) / (1 -
R Ps a)` on a buck with linear tables, R = 2 and 20 K/W: 1e-13 K; the
damped loop: 7.7e-4 K and 0.017 K away (its 0.1 K stopping rule).
Runaway at `R Ps a = 1.2`: slope reported 1.2000. Buck example's best
of 32 designs (SYN-B, 60.5 kHz, r_th = 3 K/W): kernel 43.271746 C,
fixed point 43.271621 C. Three-phase SPWM VSI, 10 kHz, 1 us dead time,
600 V, T_j = 125 C: kernel E_on / E_off per switch against
`device_loss_summary` on a settled `dsed` period, 1.8e-13 relative;
diode recovery counts 199 / 201 = turn-off counts.

A finding on the way: with the star load's neutral returned through
1 MOhm the same comparison gave 9.3e-7, every turn-on's `i` 1.3e-4 A
apart. Cause: the neutral resistor makes a common-mode mode of
`(L/3)/R_n` = 1.7 ns, and the sampled summary reads the turn-on current
at the first sample after the edge (0.21 ns later in that run), where
the current has already moved; the kernel reads the exact right limit.
With 1 uF || 1 kOhm the mode is slow and the two agree to 1.8e-13. The
sampled reading is right to O(h) of the step after the edge; it is not
an engine error.

What the kernel mode does not do: feed `T_j` back into the electrical
solution (that is `temperature_dependent=True`, the fixed point), draw
the event energy from the circuit, or run inside the kernel batch (a
thermal evaluator takes the Python path, `counters["batch"] ==
"python: a thermal loop"`; the solve itself is the kernel's).

To reverse: `ThermalSpec(mode="fixed_point")` restores the loop per
evaluator; dropping the `"auto"` resolution in `_evaluate` restores it
everywhere; the diode refusal in `device_loss_summary` is one block.

## Missing from the kernel (recorded, worked around in Python)

Items 1 and 2 are done (add-parameter-sensitivities slice, 2026-11):

* `pulsim.ConvergenceError(RuntimeError)` is created by the extension
  (`python/bindings.cpp`) and is the base of `SimulationAborted`, so
  `except RuntimeError` keeps working. `periodic_steady_state` raises a
  private subclass that is also a `ValueError` (what it raised before)
  at its seven non-convergence sites; its configuration errors stay
  plain `ValueError`s, and "the exact stepper cannot extract this
  circuit's state space" became `UnsupportedError(feature=
  "state_space")`. The `dsed` stepper's own non-convergences ("did not
  settle after 64 rounds", "chattering", "no meaningful progress",
  "exceeded max-step cap", the RK45 NaN) are still `std::runtime_error`
  in `core/include/pulsim/dsed/scheduler.hpp`, a file this slice may
  not edit. Hook: throw `solver::SimulationAborted` (or a new
  `ConvergenceFailure : std::runtime_error` translated to
  `ConvergenceError` in `bindings.cpp`) at those five sites; the
  evaluator needs no change, it already maps every `RuntimeError` of an
  engine call to `failed`.
* `pulsim.periodic_steady_state_supported(builder, period, switch_fn)
  -> (bool, reason)` runs the solver's applicability checks (the
  `_pss_refusal` that the evaluator's `auto` route also calls) and no
  period.

1. **A typed non-convergence.** What: a `RuntimeError` subclass, e.g.
   `pulsim.ConvergenceError(where, reason, t, counters)`, raised by
   the transient engines (Newton, event-round cap, step floor —
   `SimulationAborted` covers part of it) and by
   `periodic_steady_state` (today a plain `ValueError`). Why: the
   evaluator classifies `periodic_steady_state` failures by message
   (decision 5b). Smallest API: the class, raised at the existing
   raise sites; `periodic_steady_state` would raise it with the
   residual history it already formats into the message.
2. **A capability query for the steady-state solver.** What:
   `periodic_steady_state.check(builder, period, switch_fn)` raising
   the same `UnsupportedError` without building the solver. Why: the
   `auto` route reaches into the private `_Solver` constructor
   (decision 10). Smallest API: the two checks it already runs
   (`_check_circuit`, `_check_schedule`) behind one public name.
3. **The period map and the period integrals without the GIL** (done,
   decision 16; the Newton loop itself is still Python). What:
   the shooting Newton (monodromy products, saltation) and the
   one-period integrals of `v·i`, `i²` with right limits, in C++ behind
   one call. Why: 87 % of an evaluation is Python, so the thread
   backend does not scale (decision 12). Smallest API: a native
   `period_integrals(result, pairs)` first (the KPI half), the map
   second (`add-periodic-steady-state-shooting` group 6.2 shares it).
4. **Losses of the stateful and nonlinear devices.** What: per-device
   dissipated power for the Lauritzen diode, the level-1 MOSFET and
   IGBT, and the controlled sources' delivered power. Why: they are
   refused for efficiency today (decision 11). Smallest API: their
   `P_avg` in `device_loss_summary` (the in-kernel event losses change
   brings it for the devices it covers).
