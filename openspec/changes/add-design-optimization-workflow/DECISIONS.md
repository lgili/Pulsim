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
