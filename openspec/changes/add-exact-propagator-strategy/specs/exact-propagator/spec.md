## ADDED Requirements

### Requirement: Exact Segment Propagation

Between two events the kernel SHALL advance a circuit that is linear
time-invariant per switch mask by the closed form
`x(t+h) = e^{Ah}·x(t) + h·φ₁(Ah)·b` on its reduced state (capacitor
voltages and inductor currents), for any `h`, with no step-size
controller and no truncation error. φ₁ and φ₂ SHALL be evaluated with a
relative error of at most 1e-14 for every argument.

#### Scenario: CCM buck against a 50-digit reference
- **GIVEN** the buck of `numerics/p2_buck.py` (24 V, 100 µH, 47 µF,
  `g_on = 1e3`, `g_off = 1e-9`, 100 kHz) with R = 1 Ω, D = 0.5, edges
  from t = 0
- **WHEN** it is run for 40 periods with the exact strategy and 10
  samples per period
- **THEN** `i_L(40T) = 11.8015131102418 A` and
  `v_C(40T) = 12.2288664641139 V` within 1e-10

#### Scenario: DCM periodic orbit
- **GIVEN** the same buck with R = 50 Ω, D = 0.3
- **WHEN** it is run for 4000 periods
- **THEN** `v_C` at the rising gate edge is `8.99624708728943 V` within
  1e-9 and the diode turns off at `0.799830698444468·T` within 1e-12·T

#### Scenario: φ₁ beside the old series switch
- **WHEN** φ₁ is evaluated at `s = 1.01e-6`
- **THEN** its relative error is at most 1e-14 (6.5e-11 before this
  change)

### Requirement: Propagator Tier With a Stated Accuracy

For each switch mask the kernel SHALL choose the propagator by a bound
it can state: the eigendecomposition in energy coordinates when
`u·κ₂(V) ≤ 1e-10`; otherwise a Padé scaling-and-squaring exponential
when `u·‖A·h‖₁ ≤ 1e-10`; otherwise the implicit integrator for the steps
taken under that mask (`u` = 2.2e-16). The result SHALL list, per mask,
the tier used, κ₂(V), and the reason for any tier other than the first.
A basis MUST NOT be accepted on invertibility alone.

#### Scenario: Critically damped RLC
- **GIVEN** a series RLC (100 µH, 100 nF) with `R = 2·√(L/C)` exactly
- **WHEN** it is advanced by one natural period
- **THEN** the state is within 1e-10 (relative) of a 50-digit reference
- **AND** the result reports a tier other than the eigendecomposition
  for that mask (the unguarded modal formula is 7.9e-9 off)

#### Scenario: Stiff idle mode stays on the first tier
- **GIVEN** the DCM buck with switch and diode both open
  (`‖A·h‖₁ = 1e7` at `h = 2 µs`)
- **WHEN** the propagator is built
- **THEN** it uses the eigendecomposition and
  `Φ₂₂ = 0.99914929813728288` within 1e-14 (Padé is 8.2e-11 off there)

#### Scenario: Strict mode refuses the implicit tier
- **GIVEN** a mask that satisfies neither bound and the strict option
- **WHEN** the run reaches that mask
- **THEN** it raises `UnsupportedError` naming the mask, κ₂ and `‖A·h‖₁`

### Requirement: Event Location on the Analytic Trajectory

State events SHALL be located on the closed-form trajectory, with the
predicate evaluated in O(n) per iteration. The search SHALL stop on the
time bracket, `|t_b − t_a| ≤ max(4u·|t|, 1e-12·h_seg)`, and MUST NOT
stop on an absolute value of the predicate. A segment SHALL be scanned
for sign changes at a step no larger than an eighth of the period of
the fastest oscillatory mode visible to the predicate.

#### Scenario: Tolerance scales with the period
- **GIVEN** a 10 MHz buck (period 1e-7 s) in DCM
- **WHEN** it is run with the exact strategy
- **THEN** the realised duty differs from the commanded one by less than
  1e-9 (the absolute 1e-10 s tolerance allowed 1e-3)

#### Scenario: Double crossing inside one segment
- **GIVEN** an under-damped LC whose diode voltage crosses zero twice
  between two gate edges
- **WHEN** the segment is propagated
- **THEN** both commutations are in the event log

### Requirement: Mode Selection in a Zero Band

The kernel SHALL choose the state of a diode whose crossing signal is
within the event tolerance of zero by the sign of the first non-zero
time derivative of that signal in the candidate mode, and SHALL settle
several such diodes sequentially. A diode away from zero SHALL keep the
shared `SwitchedDiode::decide` law. When no candidate state is
consistent the error SHALL name the diodes and print the time in
scientific notation.

#### Scenario: Diode at rest with V_th = 0
- **GIVEN** the buck with `V_th = 0`, every switch open and every state
  zero at t = 0
- **WHEN** it is run with the exact strategy
- **THEN** it does not raise "did not settle after 64 rounds"
- **AND** its states equal the `V_th = 1e-9` run within 1e-12

#### Scenario: DCM idle mode does not chatter
- **GIVEN** the DCM buck with `V_th = 1e-6` and L = 100 µH·(1 ± δ),
  δ ∈ {1e-5, 1e-4, 1e-2}
- **WHEN** each is run for 40 periods
- **THEN** all six complete (runs at each δ abort with "is chattering"
  before this change)
- **AND** the central difference of `v_C(40T)` in L at δ = 1e-5 is
  1758.1254 V/H within 1e-6 relative

#### Scenario: Away from zero the three engines still agree
- **GIVEN** the resistive half-wave rectifier of the `transient-timestep`
  spec (`V_f = 0.7`, 10 V peak, 50 Hz)
- **WHEN** it is run with the exact strategy
- **THEN** conduction starts near 223 µs, ends near 9.777 ms and the
  output averages 2.8268 V, as on `pwl` and `trbdf2`

### Requirement: Initial Conditions on the Reduced State

The exact strategy SHALL honour builder initial conditions by taking
each capacitor's branch voltage and each inductor's current as the
reduced state. When a capacitor loop makes the declared values
inconsistent it SHALL start from the charge-conserving projection and
report the adjustment. An initial condition on a node that is not a
state SHALL be refused by name.

#### Scenario: Inductor current
- **GIVEN** a 1 mH inductor with `i0 = 100 A` across 1 Ω
- **WHEN** it is run with the exact strategy
- **THEN** `i(t) = 100·e^{−t/1 ms}` within 1e-10 at every sample

#### Scenario: Floating capacitor
- **GIVEN** a capacitor between two non-ground nodes with `c0 = 5 V`
- **WHEN** the run starts
- **THEN** the voltage across it at t = 0 is 5 V

#### Scenario: Inconsistent capacitor loop
- **GIVEN** 1 µF at 10 V in parallel with 3 µF at 2 V
- **WHEN** the run starts
- **THEN** both start at 4.0 V and the diagnostics report the adjustment

### Requirement: Coupled Inductors

The exact strategy SHALL propagate circuits with mutual inductance
through the inductance matrix of the coupled windings, and SHALL refuse
a coupling with `|k| = 1` with an error naming the pair.

#### Scenario: Two coupled windings
- **GIVEN** two 1 mH windings with k = 0.9, each in series with 1 Ω, a
  10 V step on the first and the second loop closed
- **WHEN** it is run with the exact strategy
- **THEN** `i₁,₂ = 5·[(1 − e^{−t/1.9 ms}) ± (1 − e^{−t/0.1 ms})]` within
  1e-10

### Requirement: Source Classes

The exact strategy SHALL treat DC sources as constant per mask,
piecewise-constant sources as constant between their breakpoints (each
breakpoint a time event), sine sources through oscillator augmentation
and piecewise-linear sources through the φ₂ term. For any other
time-varying source it MUST NOT claim exactness: it SHALL use the
implicit integrator and record the reason.

#### Scenario: Pulse source
- **GIVEN** a 10 V pulse source driving an RC
- **WHEN** it is run with the exact strategy
- **THEN** the capacitor voltage equals the closed form within 1e-10 at
  every sample and every source edge is in the event log

#### Scenario: Arbitrary source function
- **GIVEN** a circuit with a user `b_extra_fn`
- **WHEN** the exact strategy is requested
- **THEN** the run uses the implicit integrator and the result's route
  reason names `b_extra_fn`

### Requirement: Fixed-Step Discrete Update

With a fixed step and gate edges on the grid the exact strategy SHALL
advance by `x ← Φ_m·x + Γ_m` with `(Φ_m, Γ_m)` cached per
`(mask, dt)`, and SHALL take a state event that falls between two grid
points by exact sub-steps. The result MUST NOT depend on `dt` beyond
rounding.

#### Scenario: Step-size independence
- **GIVEN** the CCM buck of the first requirement
- **WHEN** it is run at dt = T/10, T/50 and T/100
- **THEN** each run is within 1e-10 of the 50-digit reference at 40 T
  (the trapezoidal engine is 1.18e-2 A off at T/50)

#### Scenario: Event between grid points
- **GIVEN** the DCM buck at dt = T/100
- **WHEN** the diode turns off between two grid points
- **THEN** the orbit is within 1e-9 of `8.99624708728943 V`

### Requirement: Per-Step Cost Budget

After warm-up the exact strategy SHALL take a step without heap
allocation, SHALL build at most one eigendecomposition per visited mask
and SHALL evaluate a predicate in O(n). The result SHALL expose the
counters (`n_eig`, `n_pade`, `n_phi_cache_hits`, `n_implicit_steps`,
`n_predicate_evals`, `n_alloc_steady`).

#### Scenario: Synchronous buck, 2000 periods
- **GIVEN** the synchronous buck of `perf/p9_dsed.py`
- **WHEN** it is run for 2000 periods
- **THEN** it takes 4000 ± 1 steps, `n_eig` equals the number of masks
  visited and `n_alloc_steady = 0`

### Requirement: Hybrid Operation With Newton Devices

A circuit that contains Newton devices SHALL run under the exact
strategy only through this rule: a step is propagated exactly when
every Newton device reports an affine region at the current state, the
region being part of the mode key and its boundary a state event; any
other step SHALL be an implicit step. A Newton device MUST NOT be
omitted, frozen or linearised outside its affine region.

#### Scenario: Flux-table inductor below the knee
- **GIVEN** an inductor with a piecewise-linear flux table operated
  below its first breakpoint
- **WHEN** it is run with the exact strategy
- **THEN** the result equals the linear inductor within 1e-12 and the
  crossing of the breakpoint appears as an event

#### Scenario: Smooth device
- **GIVEN** a rectifier with a Shockley diode
- **WHEN** the exact strategy is requested
- **THEN** every step is implicit (`n_exact_steps = 0`) and the result
  matches `trbdf2` within its tolerance

### Requirement: Applicability Limit

The exact strategy SHALL be selected automatically only when the
reduced state count and the memory of the visited masks are within the
documented limits, and an explicit request beyond them SHALL be refused
with `UnsupportedError` naming the limit and the circuit's value.

#### Scenario: Large ladder
- **GIVEN** an RLC ladder with 1502 unknowns
- **WHEN** it is run with `engine='auto'`
- **THEN** the exact strategy is not selected and the route reason
  states the state count against the limit
- **AND** an explicit request raises `UnsupportedError` with both
  numbers

### Requirement: Exact Propagation As The Default Route

With `engine='auto'` the kernel SHALL propagate a segment exactly
whenever, at the event that starts it, every device is linear, a
switch, a PWL diode or on a declared affine region, every source is in
an exact source class, the mode's extraction succeeds and the scaling
rule finds an exact kernel; otherwise it SHALL take the segment with
the implicit strategy and record the reason. With an explicit
`engine='exact'` (alias `'dsed'`) a non-eligible segment SHALL raise
`pulsim.UnsupportedError` naming the failed rule in `feature` unless
`allow_implicit_segments=True`.

#### Scenario: Converters routed exact
- **GIVEN** the DCM buck, the boost and a three-phase diode bridge with
  PWL devices and DC or sine sources
- **WHEN** each is simulated with `engine='auto'`
- **THEN** every segment is recorded as exact and the states are
  bit-identical to `engine='exact'`

#### Scenario: Smooth device routed implicit
- **GIVEN** a rectifier with Shockley diodes
- **WHEN** it is simulated with `engine='auto'`
- **THEN** every segment is implicit with reason `'device'` naming a
  diode, and with `engine='exact'` the run raises
  `pulsim.UnsupportedError`

### Requirement: Hybrid Segment Rule

The kernel SHALL classify segments as exact or implicit at events
only, SHALL keep an implicit segment until the next event, SHALL start
every implicit segment with its strategy's post-event restart (the
restart rule of three consecutive Alexander SDIRK(2,2) steps on the
fixed trapezoid, and after any event with dependent states), SHALL apply the re-initialisation service
on entry to an exact segment whose mode has dependent states, and
SHALL report `n_exact_segments`, `n_implicit_segments`,
`t_exact_fraction` and every hand-over with its time, device and
reason.

#### Scenario: MOSFET in cutoff
- **GIVEN** a buck whose switch is a MOSFET Level 1 and whose diode is
  a PWL diode
- **WHEN** it is simulated with `engine='auto'`
- **THEN** the MOSFET's cutoff intervals are exact segments, its
  conducting intervals implicit, and `v_C` agrees with `trbdf2`
  (rtol 1e-9) within 1e-6

### Requirement: Propagator Scaling Decision

For each mode the kernel SHALL choose between the dense propagator,
the action of the exponential on the sparse system and the hand-over
to the implicit strategy by a cost rule on the mode's state count,
non-zero count, `‖A·h‖₁` and expected steps, and SHALL report the
chosen kernel, the reason and those inputs per mode in
`result.exact['routes']`.

#### Scenario: Large non-stiff ladder
- **GIVEN** the RLC ladder of `probes/expm_action_ladder.py` with
  n = 2000 states at `h = 1 µs` (`‖A·h‖₁ = 2`)
- **WHEN** it is simulated with `engine='auto'`
- **THEN** the mode is propagated by the action of the exponential with
  20 ± 2 matrix-vector products per step and no dense setup

#### Scenario: Stiff ladder with short mode visits
- **GIVEN** the ladder with 1 ns snubbers (`‖A·h‖₁ = 1e3`), n = 300, in
  a mode visited for 50 steps at a time
- **WHEN** it is simulated with `engine='auto'`
- **THEN** the mode is handed to the implicit strategy with reason
  `'cost'`

### Requirement: Exact Source Augmentation

The exact route SHALL propagate constant, polynomial-in-time,
exponential and sinusoidal sources exactly through augmented source
states, SHALL treat the corners of PWL sources as time events, and
SHALL take any segment driven by an arbitrary source function with the
implicit strategy, recording reason `'source'`.

#### Scenario: Sine source
- **GIVEN** 100 V, 50 Hz into 1 Ω and 1 mH, from rest
- **WHEN** it is propagated at `h = T/20` for 5 periods
- **THEN** the current is within 1e-10 A of the closed form (a source
  held per step is 17 A off)

### Requirement: Output Samples And Controller Ticks As Events

The exact route SHALL produce recorded samples by evaluating the
exact flow from the last event without starting a new segment, and
SHALL treat controller ticks as time events at which the controller
outputs update with a zero-order hold.

#### Scenario: Samples do not add segments
- **GIVEN** the DCM buck over 100 periods
- **WHEN** it is simulated with `dt_out = T/7` and with no samples
- **THEN** the number of events is the same and every sample is within
  1e-12 of the exact flow at its time

### Requirement: Ideal Switches On The Exact Route

The exact route SHALL accept branch-form ideal switches and diodes,
SHALL propagate the reduced state of each mode, SHALL apply the
explicit flux- and charge-conserving projection at mode entry, and
SHALL take the mode at every event from the complementarity mode
selection when it is in effect.

#### Scenario: Ideal DCM buck from rest
- **GIVEN** the DCM buck (24 V, 100 µH, 47 µF, 50 Ω, D = 0.3,
  100 kHz) with an ideal switch and an ideal diode
- **WHEN** it is simulated for 4000 periods from rest on the exact
  route
- **THEN** `v_C` at the rising edge of the last period is within 1e-9 V
  of 8.9963970815477 V and the diode turn-off within 1e-11·T of
  `0.79983739563545·T`
